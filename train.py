"""
BraTS 4-modality training pipeline (4 separate identical 3D UNets)
- One model per modality: t1, t1ce, t2, flair
- Input patch: 128x128x128 (tumor-aware random crop)
- Output: logits [B,4,128,128,128] for classes {0,1,2,3} where 3 == original label 4
- Loss: Dice + CE (MONAI DiceCELoss)
- Optimizer: AdamW
- Train/val split: 70/15/15 of valid patients
- Train 30 epochs and save best checkpoint per modality by mean tumor Dice on validation
- After all trainings:
    * compute val Dice per class for each modality model
    * build class-wise weights W (softmax over modalities per class)
    * save W + dice table to disk for reuse in testing/inference

Requirements:
pip install nibabel monai torch tqdm
"""

from __future__ import annotations

import os
import json
import random
from typing import Dict, List, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from tqdm import tqdm
import time

from monai.networks.nets import UNet
from dataset import BraTSModalDataset, build_loaders_for_modality
from utils import seed_everything, CFG
from time import perf_counter

MODALITIES = ["t1"]


import matplotlib.pyplot as plt

class LivePlotter:
    """
    Live-updates 3 figures:
      1) train/val loss
      2) train/val mean tumor dice
      3) per-class dice (train+val => 8 curves)
    Works in .py scripts via plt.pause().
    """
    def __init__(self, num_classes: int, save_dir: str | None = None):
        self.C = num_classes - 1
        self.save_dir = save_dir

        plt.ion()  # interactive on

        self.fig_loss, self.ax_loss = plt.subplots()
        self.fig_dice, self.ax_dice = plt.subplots()
        self.fig_pc, self.ax_pc = plt.subplots()

        self.ax_loss.set_title("Loss")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")

        self.ax_dice.set_title("Mean Tumor Dice (classes 1..3)")
        self.ax_dice.set_xlabel("Epoch")
        self.ax_dice.set_ylabel("Dice")

        self.ax_pc.set_title("Dice per class (train + val)")
        self.ax_pc.set_xlabel("Epoch")
        self.ax_pc.set_ylabel("Dice")

    def update(
        self,
        epochs_done: int,
        train_loss: list[float],
        val_loss: list[float],
        train_dice: list[float],
        val_dice: list[float],
        train_pc: list[list[float]],  # shape [E, C]
        val_pc: list[list[float]],    # shape [E, C]
    ):
        x = list(range(1, epochs_done + 1))

        # ---- 1) loss ----
        self.ax_loss.cla()
        self.ax_loss.set_title("Loss")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.plot(x, train_loss, label="train loss")
        self.ax_loss.plot(x, val_loss, label="val loss")
        self.ax_loss.legend()
        self.ax_loss.grid(True)

        # ---- 2) mean tumor dice ----
        self.ax_dice.cla()
        self.ax_dice.set_title("Mean Tumor Dice (classes 1..3)")
        self.ax_dice.set_xlabel("Epoch")
        self.ax_dice.set_ylabel("Dice")
        self.ax_dice.plot(x, train_dice, label="train dice")
        self.ax_dice.plot(x, val_dice, label="val dice")
        self.ax_dice.legend()
        self.ax_dice.grid(True)

        # ---- 3) per-class dice (8 curves) ----
        self.ax_pc.cla()
        self.ax_pc.set_title("Dice per class (train + val)")
        self.ax_pc.set_xlabel("Epoch")
        self.ax_pc.set_ylabel("Dice")

       # train curves (classes 1..3)
        for i in range(self.C):   # i = 0,1,2
            y = [row[i] for row in train_pc]
            self.ax_pc.plot(x, y, label=f"train c{i+1}")

    # val curves (classes 1..3)
        for i in range(self.C):
            y = [row[i] for row in val_pc]
            self.ax_pc.plot(x, y, label=f"val c{i+1}", linestyle="--")
        self.ax_pc.legend(ncols=2, fontsize=8)
        self.ax_pc.grid(True)

        # draw + allow UI refresh
        self.fig_loss.canvas.draw(); self.fig_dice.canvas.draw(); self.fig_pc.canvas.draw()
        plt.pause(0.001)

        # optional saving every epoch
        if self.save_dir:
            import os
            os.makedirs(self.save_dir, exist_ok=True)
            self.fig_loss.savefig(os.path.join(self.save_dir, "loss.png"), dpi=150)
            self.fig_dice.savefig(os.path.join(self.save_dir, "mean_dice.png"), dpi=150)
            self.fig_pc.savefig(os.path.join(self.save_dir, "per_class_dice.png"), dpi=150)
# -------------------------
# Model
# -------------------------
def build_unet_3d(num_classes: int = 4) -> nn.Module:
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        channels=(32, 64, 128, 256, 320),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
    )


# -------------------------
# Metrics: Dice per class on loader (for val + weight building)
# -------------------------

# -------------------------
# Train / Validate
# -------------------------
# @torch.no_grad()
# def validate_scalar_dice(
#     model: nn.Module,
#     val_loader: DataLoader,
#     device: torch.device,
#     num_classes: int,
#     include_bg: bool,
# ) -> float:
#     dice_vec = dice_per_class_from_loader(model, val_loader, device, num_classes=num_classes, include_bg=include_bg)
#     return torch.nanmean(dice_vec[1:]).item()


def train_one_epoch(
    cfg : CFG,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer,
    scaler : torch.amp.GradScaler
):
    model.train()
    running_loss = 0.0
    dice_sum = 0.0
    dice_count = 0


    num_plot_classes = cfg.num_classes - 1  # classes 1..3
    per_class_sum = torch.zeros(num_plot_classes, dtype=torch.float64)
    per_class_count = torch.zeros(num_plot_classes, dtype=torch.float64)
    eps = 1e-6

    # pbar = tqdm(train_loader, desc=f"[{modality}] epoch {epoch}/{epochs}")
    start = perf_counter()
    for idx, (img, seg) in enumerate(train_loader):
        if idx % 50 == 0 and idx > 0: 
            print(f"reached {idx} index")
        if idx == len(train_loader) - 1:
            end = perf_counter()
            elapsed_ms = (end - start)
            print(f"Time taken: {elapsed_ms:.3f}s")
    
        img, seg = img.to(cfg.device), seg.to(cfg.device)

        optimizer.zero_grad()
        logits = model(img)
        loss = cfg.loss_fn(logits, seg.unsqueeze(1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item())
        pred = torch.argmax(logits, dim=1)

        # pbar.set_postfix(loss=float(loss.item()))
        for b in range(pred.shape[0]):
            dice_per_class = torch.full((cfg.num_classes,), float("nan"), device=cfg.device)

            for c in range(cfg.num_classes):
                if (not cfg.include_bg_in_metric) and c == 0:
                    continue
                p = (pred[b] == c)
                g = (seg[b] == c)

                inter = (p & g).sum().float()
                denom = p.sum().float() + g.sum().float()

                # If class absent in BOTH pred and gt, many people ignore it.
                # We'll ignore by keeping NaN so it doesn't inflate mean.
                if denom < 1e-6:
                    continue

                dice_per_class[c] = (2.0 * inter + eps) / (denom + eps)

            mean_tumor = torch.nanmean(dice_per_class[1:]).item()
            dice_sum += float(mean_tumor)
            dice_count += 1
            d_cpu = dice_per_class.detach().cpu().double()

            # take only classes 1..3 into a compact vector of length 3
            d_compact = d_cpu[1:]  # [c1, c2, c3]
            valid = ~torch.isnan(d_compact)
            per_class_sum[valid] += d_compact[valid]
            per_class_count[valid] += 1.0

    train_loss = running_loss / max(1, len(train_loader))
    train_mean_tumor_dice = dice_sum / max(1, dice_count)

    train_dice_per_class_mean = per_class_sum / torch.clamp(per_class_count, min=1.0)

    print(f"train_loss={train_loss:.4f} \
          train_mean_tumor_dice={train_mean_tumor_dice:.4f} \
            train_dice_per_class_mean={train_dice_per_class_mean.tolist()}")
    return train_loss, train_mean_tumor_dice, train_dice_per_class_mean

@torch.no_grad()
def validate_one_epoch(
    cfg: CFG,
    model: nn.Module,
    val_loader: DataLoader,
):
    model.eval()

    running_loss = 0.0
    dice_sum = 0.0
    dice_count = 0

   
    num_plot_classes = cfg.num_classes - 1  # classes 1..3
    per_class_sum = torch.zeros(num_plot_classes, dtype=torch.float64)
    per_class_count = torch.zeros(num_plot_classes, dtype=torch.float64)
    eps = 1e-6

    start = perf_counter()

    for idx, (img, seg) in enumerate(val_loader):
        print(f"element {idx}")
        if idx == len(val_loader) - 1:
            end = perf_counter()
            elapsed_ms = (end - start)
            print(f"Time taken: {elapsed_ms:.3f}s")
        img, seg = img.to(cfg.device), seg.to(cfg.device)

        logits = model(img)
        loss = cfg.loss_fn(logits, seg.unsqueeze(1))
        running_loss += float(loss.item())

        pred = torch.argmax(logits, dim=1)  # [B,H,W,D]

        # Dice per image in batch
        for b in range(pred.shape[0]):
            dice_per_class = torch.full((cfg.num_classes,), float("nan"), device=cfg.device)

            for c in range(cfg.num_classes):
                if (not cfg.include_bg_in_metric) and c == 0:
                    continue

                p = (pred[b] == c)
                g = (seg[b] == c)

                inter = (p & g).sum().float()
                denom = p.sum().float() + g.sum().float()

                # ignore class if absent in both pred & gt
                if denom < 1e-6:
                    continue

                dice_per_class[c] = (2.0 * inter + eps) / (denom + eps)

            mean_tumor = torch.nanmean(dice_per_class[1:]).item()
            dice_sum += float(mean_tumor)
            dice_count += 1

            d_cpu = dice_per_class.detach().cpu().double()

            # take only classes 1..3 into a compact vector of length 3
            d_compact = d_cpu[1:]  # [c1, c2, c3]
            valid = ~torch.isnan(d_compact)
            per_class_sum[valid] += d_compact[valid]
            per_class_count[valid] += 1.0


    elapsed_s = perf_counter() - start

    val_loss = running_loss / max(1, len(val_loader))
    val_mean_tumor_dice = dice_sum / max(1, dice_count)
    val_dice_per_class_mean = per_class_sum / torch.clamp(per_class_count, min=1.0)

    print(
        f"val_loss={val_loss:.4f} "
        f"val_mean_tumor_dice={val_mean_tumor_dice:.4f} "
        f"val_dice_per_class_mean={val_dice_per_class_mean.tolist()} "
        f"time={elapsed_s:.3f}s"
    )

    return val_loss, val_mean_tumor_dice, val_dice_per_class_mean

# def validate_modality(
#     model,
#     cfg,
#     val_loader,
#     device: torch.device,
# ):

#     dice_vec = dice_per_class_from_loader(
#         model,
#         val_loader,
#         device,
#         num_classes=cfg.num_classes,
#         include_bg=cfg.include_bg_in_metric,
#     )
#     mean_tumor = torch.nanmean(dice_vec[1:]).item()  # classes 1..3
#     return dice_vec, mean_tumor


def load_best_model(ckpt_path: str, num_classes: int, device: torch.device) -> nn.Module:
    model = build_unet_3d(num_classes=num_classes)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model

# def map_back_to_brats_labels(pred_0123: torch.Tensor) -> torch.Tensor:
#     pred = pred_0123.clone()
#     pred[pred == 3] = 4
#     return pred

# def save_single_modality_results(
#     cfg,
#     modality: str,
#     dice_vec: torch.Tensor,
#     best_ckpt_path: str,
# ):
#     """
#     For a single modality, W will be [1,C] all ones (softmax of a single row).
#     Still saved in the same format so you can reuse the code later.
#     """
#     dice_by_model: Dict[str, torch.Tensor] = {modality: dice_vec}
#     names, W = compute_classwise_weights(dice_by_model, temp=cfg.ensemble_temp)

#     os.makedirs("results", exist_ok=True)

#     weights_obj = {
#         "modalities_order": names,                    # [modality]
#         "weights_MxC": W.cpu().tolist(),              # [[...]] shape [1,C]
#         "dice_per_class_internal_0_1_2_3": {modality: dice_vec.tolist()},
#         "best_ckpt_path": best_ckpt_path,
#         "note": "Class 3 corresponds to original BraTS label 4 (enhancing).",
#         "ensemble_temp": cfg.ensemble_temp,
#     }

#     json_path = os.path.join("results", f"{modality}_ensemble_classwise_weights.json")
#     pt_path = os.path.join("results", f"{modality}_ensemble_classwise_weights.pt")

#     with open(json_path, "w") as f:
#         json.dump(weights_obj, f, indent=2)

#     torch.save(
#         {"modalities_order": names, "W": W.cpu(), "dice_by_model": dice_by_model, "best_ckpt_path": best_ckpt_path},
#         pt_path,
#     )

#     print("Saved:")
#     print(" -", json_path)
#     print(" -", pt_path)

#     print("\nClass-wise weights W (rows modalities, cols classes 0..3):")
#     for i, m in enumerate(names):
#         print(m, W[i].cpu().numpy())


# -------------------------
# Main pipeline
# -------------------------
def main(cfg : CFG):
    seed_everything(cfg.seed)
    print("Device:", cfg.device)
    patient_names = sorted([
        os.path.join(cfg.root, d) for d in os.listdir(cfg.root)
        if os.path.isdir(os.path.join(cfg.root, d))
    ])
    train_loader, val_loader = build_loaders_for_modality(
        cfg=cfg, patient_names=patient_names)
    
    model = build_unet_3d(num_classes=cfg.num_classes)
    model = model.to(cfg.device)
    if cfg.optimizer_name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
    )
    elif cfg.optimizer_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
    )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer_name}")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",        # because we monitor val_loss
        factor=0.5,        # new_lr = lr * 0.5
        patience=3,        # wait 3 epochs without improvement
        threshold=1e-4,    # minimum change to count as improvement
        min_lr=1e-6      # do not reduce below this
    )
    scaler = torch.amp.GradScaler('cuda')

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
        "train_pc": [],  # list of list len C
        "val_pc": [],
    }

    plotter = LivePlotter(num_classes=cfg.num_classes, save_dir="live_plots")

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_dice, tr_pc = train_one_epoch(
            cfg=cfg, model=model, train_loader=train_loader, optimizer=optimizer, scaler=scaler
        )
        va_loss, va_dice, va_pc = validate_one_epoch(cfg=cfg, model=model, val_loader=val_loader)
        scheduler.step(va_loss)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_dice"].append(tr_dice)
        history["val_dice"].append(va_dice)

        # tensors -> python lists
        history["train_pc"].append(tr_pc.detach().cpu().tolist())
        history["val_pc"].append(va_pc.detach().cpu().tolist())

        plotter.update(
            epochs_done=epoch,
            train_loss=history["train_loss"],
            val_loss=history["val_loss"],
            train_dice=history["train_dice"],
            val_dice=history["val_dice"],
            train_pc=history["train_pc"],
            val_pc=history["val_pc"],
        )

        print(f"[epoch {epoch}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
              f"train_dice={tr_dice:.4f} val_dice={va_dice:.4f}")


    # save_single_modality_results(
    #     cfg=cfg,
    #     modality=modality,
    #     dice_vec=dice_vec,
    #     best_ckpt_path=best_ckpt_path,
    # )

    # return {
    #     "modality": modality,
    #     "best_val_mean_tumor_dice": best_val,
    #     "val_dice_vec": dice_vec,
    #     "best_ckpt_path": best_ckpt_path,
    # }

if __name__ == "__main__":
    # cfg = CFG(
    #     root="t1_out",  
    #     batch_size=1,
    #     num_workers=1,
    #     epochs=30,
    #     lr=1e-3,
    #     weight_decay=1e-4,
    #     seed=42,
    #     include_bg_in_metric=False,   # typical: exclude background in Dice reporting
    #     ensemble_temp=1.0,
    # )
    cfg = CFG("config.yaml")
    main(cfg)
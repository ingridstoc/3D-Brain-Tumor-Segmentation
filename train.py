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
@torch.no_grad()
def dice_per_class_from_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 4,
    include_bg: bool = False,
):
    model.eval()
    eps = 1e-6
    inter = torch.zeros(num_classes, device=device)
    denom = torch.zeros(num_classes, device=device)

    for x, y in loader:
        x = x.to(device)  # [B,1,H,W,D]
        y = y.to(device)  # [B,H,W,D]
        logits = model(x)
        pred = torch.argmax(logits, dim=1)  # [B,H,W,D]

        for c in range(num_classes):
            if (not include_bg) and c == 0:
                continue
            p = pred == c
            g = y == c
            inter[c] += (p & g).sum()
            denom[c] += p.sum() + g.sum()

    dice = (2.0 * inter + eps) / (denom + eps)  # [C]
    if not include_bg:
        dice[0] = torch.nan
    return dice.detach().cpu()


def compute_classwise_weights(dice_by_model: Dict[str, torch.Tensor], temp: float = 1.0, eps: float = 1e-6):
    """
    dice_by_model: dict modality -> Tensor[C]
    returns:
      names: list[str]
      W: Tensor[M,C] where sum over M is 1 for each class (softmax over modalities)
    """
    names = list(dice_by_model.keys())
    d = torch.stack([dice_by_model[n] for n in names], dim=0)  # [M,C]
    d = torch.nan_to_num(d, nan=0.0)
    scores = torch.log(d + eps) / temp
    W = torch.softmax(scores, dim=0)
    return names, W


# -------------------------
# Train / Validate
# -------------------------
@torch.no_grad()
def validate_scalar_dice(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    include_bg: bool,
) -> float:
    dice_vec = dice_per_class_from_loader(model, val_loader, device, num_classes=num_classes, include_bg=include_bg)
    return torch.nanmean(dice_vec[1:]).item()


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

    per_class_sum = torch.zeros(cfg.num_classes, dtype=torch.float64)
    per_class_count = torch.zeros(cfg.num_classes, dtype=torch.float64)
    eps = 1e-6

    # pbar = tqdm(train_loader, desc=f"[{modality}] epoch {epoch}/{epochs}")
    start = perf_counter()
    for idx, (img, seg) in enumerate(train_loader):
        print(f"element {idx}")
        if idx == 10:
            end = perf_counter()
            elapsed_ms = (end - start)
            print(f"Time taken: {elapsed_ms:.3f}s")
            break
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
            d_cpu = dice_per_class.detach().cpu()
            valid = ~torch.isnan(d_cpu)
            per_class_sum += torch.nan_to_num(d_cpu, nan=0.0).double()
            per_class_count += valid.double()

    train_loss = running_loss / max(1, len(train_loader))
    train_mean_tumor_dice = dice_sum / max(1, dice_count)

    train_dice_per_class_mean = per_class_sum / torch.clamp(per_class_count, min=1.0)

    print(f"train_loss={train_loss:.4f} \
          train_mean_tumor_dice={train_mean_tumor_dice:.4f} \
            train_dice_per_class_mean={train_dice_per_class_mean.tolist()}")
    return train_loss, train_mean_tumor_dice, train_dice_per_class_mean

def validate_modality(
    cfg,
    val_loader,
    best_ckpt_path: str,
    device: torch.device,
):
    model = load_best_model(best_ckpt_path, cfg.num_classes, device)

    dice_vec = dice_per_class_from_loader(
        model,
        val_loader,
        device,
        num_classes=cfg.num_classes,
        include_bg=cfg.include_bg_in_metric,
    )
    mean_tumor = torch.nanmean(dice_vec[1:]).item()  # classes 1..3
    return dice_vec, mean_tumor


def load_best_model(ckpt_path: str, num_classes: int, device: torch.device) -> nn.Module:
    model = build_unet_3d(num_classes=num_classes)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


# -------------------------
# Ensemble prediction (class-wise weights)
# -------------------------
@torch.no_grad()
def ensemble_predict_classwise(models_by_name, W, names, x_by_name, device, num_classes=4):
    """
    models_by_name: dict modality->model
    W: Tensor[M,C] class-wise weights, sum over modalities = 1 for each class
    names: list modality names matching W rows
    x_by_name: dict modality-> input tensor [B,1,H,W,D]
    returns pred [B,H,W,D] in internal labels 0..3 (map 3->4 if needed)
    """
    p_final = None
    for mi, name in enumerate(names):
        model = models_by_name[name]
        x = x_by_name[name].to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)  # [B,C,H,W,D]
        w_mc = W[mi].to(device).view(1, num_classes, 1, 1, 1)
        contrib = probs * w_mc
        p_final = contrib if p_final is None else (p_final + contrib)

    pred = torch.argmax(p_final, dim=1)
    return pred


# def map_back_to_brats_labels(pred_0123: torch.Tensor) -> torch.Tensor:
#     pred = pred_0123.clone()
#     pred[pred == 3] = 4
#     return pred

def save_single_modality_results(
    cfg,
    modality: str,
    dice_vec: torch.Tensor,
    best_ckpt_path: str,
):
    """
    For a single modality, W will be [1,C] all ones (softmax of a single row).
    Still saved in the same format so you can reuse the code later.
    """
    dice_by_model: Dict[str, torch.Tensor] = {modality: dice_vec}
    names, W = compute_classwise_weights(dice_by_model, temp=cfg.ensemble_temp)

    os.makedirs("results", exist_ok=True)

    weights_obj = {
        "modalities_order": names,                    # [modality]
        "weights_MxC": W.cpu().tolist(),              # [[...]] shape [1,C]
        "dice_per_class_internal_0_1_2_3": {modality: dice_vec.tolist()},
        "best_ckpt_path": best_ckpt_path,
        "note": "Class 3 corresponds to original BraTS label 4 (enhancing).",
        "ensemble_temp": cfg.ensemble_temp,
    }

    json_path = os.path.join("results", f"{modality}_ensemble_classwise_weights.json")
    pt_path = os.path.join("results", f"{modality}_ensemble_classwise_weights.pt")

    with open(json_path, "w") as f:
        json.dump(weights_obj, f, indent=2)

    torch.save(
        {"modalities_order": names, "W": W.cpu(), "dice_by_model": dice_by_model, "best_ckpt_path": best_ckpt_path},
        pt_path,
    )

    print("Saved:")
    print(" -", json_path)
    print(" -", pt_path)

    print("\nClass-wise weights W (rows modalities, cols classes 0..3):")
    for i, m in enumerate(names):
        print(m, W[i].cpu().numpy())


# -------------------------
# Main pipeline
# -------------------------
def main(cfg : CFG):
    seed_everything(cfg.seed)
    print("Device:", cfg.device)

    print("Started sorting the file locations....")
    patient_names = sorted([
        os.path.join(cfg.root, d) for d in os.listdir(cfg.root)
        if os.path.isdir(os.path.join(cfg.root, d))
    ])
    print(patient_names[:5])
    print("Initializing train, validation dataset...")
    train_loader, val_loader = build_loaders_for_modality(
        cfg=cfg, patient_names=patient_names)
    
    model = build_unet_3d(num_classes=cfg.num_classes)
    model = model.to(cfg.device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda')
    print("Initialized model and optimizer")

    train_one_epoch(cfg=cfg, model = model, 
                    train_loader=train_loader,
                    optimizer=optimizer,
                    scaler=scaler)
    print("a ajuns aici in sfarsit...")
    exit(1)

    dice_vec, mean_tumor = validate_modality(
        cfg=cfg,
        val_loader=val_loader,
        best_ckpt_path=best_ckpt_path
    )

    print(f"Final validation dice per class for {modality} (0..3): {dice_vec}")
    print(f"Final validation mean tumor dice (classes 1..3): {mean_tumor:.4f}")

    save_single_modality_results(
        cfg=cfg,
        modality=modality,
        dice_vec=dice_vec,
        best_ckpt_path=best_ckpt_path,
    )

    return {
        "modality": modality,
        "best_val_mean_tumor_dice": best_val,
        "val_dice_vec": dice_vec,
        "best_ckpt_path": best_ckpt_path,
    }

if __name__ == "__main__":
    cfg = CFG(
        root="t1_out",  
        batch_size=1,
        num_workers=1,
        epochs=1,
        lr=1e-3,
        weight_decay=1e-4,
        seed=42,
        include_bg_in_metric=False,   # typical: exclude background in Dice reporting
        ensemble_temp=1.0,
    )
    main(cfg)
from __future__ import annotations
from monai.networks.nets import UNet, SegResNet
import os
from typing import Dict, List, Tuple
from monai.metrics import HausdorffDistanceMetric, MeanIoU
from monai.networks.nets import UNet, SegResNet, UNETR, DynUNet, SwinUNETR
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import json
import time

from monai.networks.nets import UNet
from dataset import build_loaders
from utils import (
    CFG,
    seed_everything,
    build_optimizer,
    build_scheduler,
)
from time import perf_counter
# MODALITIES = ["t1"]
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-2, mode: str = "min"):
        """
        mode='min' -> lower is better (e.g. val_loss)
        mode='max' -> higher is better (e.g. val_dice)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_value = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, current_value: float) -> bool:
        if self.best_value is None:
            self.best_value = current_value
            return False

        if self.mode == "min":
            improved = current_value < (self.best_value - self.min_delta)
        elif self.mode == "max":
            improved = current_value > (self.best_value + self.min_delta)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        if improved:
            self.best_value = current_value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.should_stop = True

        return self.should_stop


class LivePlotter:
    def __init__(self, num_classes: int, save_dir: str | None = None):
        self.C = num_classes - 1
        self.save_dir = save_dir

        plt.ion()

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
        train_epochs: list[int],
        val_epochs: list[int],
        train_loss: list[float],
        val_loss: list[float],
        train_dice: list[float],
        val_dice: list[float],
        train_pc: list[list[float]],
        val_pc: list[list[float]],
    ):
    # ---- 1) loss ----
        self.ax_loss.cla()
        self.ax_loss.set_title("Loss")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")

        if train_loss:
            self.ax_loss.plot(train_epochs, train_loss, label="train loss")
        if val_loss:
            self.ax_loss.plot(val_epochs, val_loss, label="val loss")

        self.ax_loss.legend()
        self.ax_loss.grid(True)

        # ---- 2) mean tumor dice ----
        self.ax_dice.cla()
        self.ax_dice.set_title("Mean Tumor Dice (classes 1..3)")
        self.ax_dice.set_xlabel("Epoch")
        self.ax_dice.set_ylabel("Dice")

        if train_dice:
            self.ax_dice.plot(train_epochs, train_dice, label="train dice")
        if val_dice:
            self.ax_dice.plot(val_epochs, val_dice, label="val dice")

        self.ax_dice.legend()
        self.ax_dice.grid(True)

        # ---- 3) per-class dice ----
        self.ax_pc.cla()
        self.ax_pc.set_title("Dice per class (train + val)")
        self.ax_pc.set_xlabel("Epoch")
        self.ax_pc.set_ylabel("Dice")

        for i in range(self.C):
            if train_pc:
                y_train = [row[i] for row in train_pc]
                self.ax_pc.plot(train_epochs, y_train, label=f"train c{i+1}")

        for i in range(self.C):
            if val_pc:
                y_val = [row[i] for row in val_pc]
                self.ax_pc.plot(val_epochs, y_val, label=f"val c{i+1}", linestyle="--")

        self.ax_pc.legend(ncols=2, fontsize=8)
        self.ax_pc.grid(True)

        self.fig_loss.canvas.draw()
        self.fig_dice.canvas.draw()
        self.fig_pc.canvas.draw()
        plt.pause(0.001)

        if self.save_dir:
            import os
            os.makedirs(self.save_dir, exist_ok=True)
            self.fig_loss.savefig(os.path.join(self.save_dir, "loss.png"), dpi=150)
            self.fig_dice.savefig(os.path.join(self.save_dir, "mean_dice.png"), dpi=150)
            self.fig_pc.savefig(os.path.join(self.save_dir, "per_class_dice.png"), dpi=150)

def build_unet_3d(cfg: CFG) -> nn.Module:
    p = cfg.model_params
    return UNet(
        spatial_dims=3,
        in_channels=p.get("in_channels", 4),
        out_channels=cfg.num_classes,
        channels=tuple(p.get("channels", [16, 32, 64, 128, 256])),
        strides=tuple(p.get("strides", [2, 2, 2, 2])),
        num_res_units=p.get("num_res_units", 2),
        norm=p.get("norm", "INSTANCE"),
    )


def build_segresnet_3d(cfg: CFG) -> nn.Module:
    p = cfg.model_params
    return SegResNet(
        spatial_dims=3,
        init_filters=p.get("init_filters", 16),
        in_channels=p.get("in_channels", 4),
        out_channels=cfg.num_classes,
        dropout_prob=p.get("dropout_prob", 0.2),
        blocks_down=tuple(p.get("blocks_down", [1, 2, 2, 4])),
        blocks_up=tuple(p.get("blocks_up", [1, 1, 1])),
    )
def build_swinunetr_3d(cfg: CFG) -> nn.Module:
    p = cfg.model_params
    return SwinUNETR(
        img_size=tuple(p.get("img_size", [192, 192, 160])),
        in_channels=p.get("in_channels", 4),
        out_channels=cfg.num_classes,
        feature_size=p.get("feature_size", 24),
        use_checkpoint=p.get("use_checkpoint", False),
        drop_rate=p.get("drop_rate", 0.0),
        attn_drop_rate=p.get("attn_drop_rate", 0.0),
        dropout_path_rate=p.get("dropout_path_rate", 0.0),
        normalize=p.get("normalize", True),
    )

def build_unetr_3d(cfg: CFG) -> nn.Module:
    p = cfg.model_params
    return UNETR(
        in_channels=p.get("in_channels", 4),
        out_channels=cfg.num_classes,
        img_size=tuple(p.get("img_size", [240, 240, 160])),
        feature_size=p.get("feature_size", 16),
        hidden_size=p.get("hidden_size", 768),
        mlp_dim=p.get("mlp_dim", 3072),
        num_heads=p.get("num_heads", 12),
        proj_type=p.get("proj_type", "conv"),
        norm_name=p.get("norm_name", "instance"),
        res_block=p.get("res_block", True),
        dropout_rate=p.get("dropout_rate", 0.0),
    )


def build_dynunet_3d(cfg: CFG) -> nn.Module:
    p = cfg.model_params
    return DynUNet(
        spatial_dims=3,
        in_channels=p.get("in_channels", 4),
        out_channels=cfg.num_classes,
        kernel_size=[tuple(x) for x in p.get("kernel_size", [[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]])],
        strides=[tuple(x) for x in p.get("strides", [[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2]])],
        upsample_kernel_size=[tuple(x) for x in p.get("upsample_kernel_size", [[2,2,2],[2,2,2],[2,2,2],[2,2,2]])],
        filters=p.get("filters", [32, 64, 128, 256, 320]),
        dropout=p.get("dropout", 0.0),
        deep_supervision=p.get("deep_supervision", False),
        deep_supr_num=p.get("deep_supr_num", 1),
        res_block=p.get("res_block", True),
    )


def build_model(cfg: CFG) -> nn.Module:
    model_name = cfg.model_name.lower()

    if model_name == "unet":
        return build_unet_3d(cfg)

    if model_name == "segresnet":
        return build_segresnet_3d(cfg)

    if model_name == "unetr":
        return build_unetr_3d(cfg)

    if model_name == "dynunet":
        return build_dynunet_3d(cfg)

    if model_name == "swinunetr":
        return build_swinunetr_3d(cfg)

    raise ValueError(f"Unknown model name: {cfg.model_name}")

import torch.nn.functional as F

def should_use_amp(cfg: CFG) -> bool:
    no_amp_models = {"unetr", "swinunetr"}
    return (cfg.device.type == "cuda") and (cfg.model_name.lower() not in no_amp_models)

def dice_from_logits(
    logits: torch.Tensor,
    seg: torch.Tensor,
    num_classes: int,
    include_bg: bool = True,
    eps: float = 1e-6,
):
    """
    Compute Dice score from logits.

    Parameters
    ----------
    logits : torch.Tensor
        Shape [B, C, H, W, D]
    seg : torch.Tensor
        Shape [B, 1, H, W, D] or [B, H, W, D]
    num_classes : int
        Number of segmentation classes
    include_bg : bool
        Whether to include class 0 in the Dice
    eps : float
        Numerical stability constant

    Returns
    -------
    dice : torch.Tensor
        Dice per batch per class → shape [B, C]
    mean_dice : torch.Tensor
        Mean Dice per batch item
    """
    pred = torch.argmax(logits, dim=1)   # [B,H,W,D]

    # ground truth labels
    if seg.ndim == 5:
        gt = seg.squeeze(1)              # [B,H,W,D]
    else:
        gt = seg

    # convert to one-hot
    pred_1h = F.one_hot(pred, num_classes=num_classes)
    gt_1h   = F.one_hot(gt,   num_classes=num_classes)

    pred_1h = pred_1h.permute(0, 4, 1, 2, 3).float()   # [B,C,H,W,D]
    gt_1h   = gt_1h.permute(0, 4, 1, 2, 3).float()

    # intersection and denominator
    inter = (pred_1h * gt_1h).sum(dim=(2, 3, 4))        # [B,C]
    denom = pred_1h.sum(dim=(2, 3, 4)) + gt_1h.sum(dim=(2, 3, 4))

    dice = (2.0 * inter + eps) / (denom + eps)

    # ignore classes absent in both pred and gt
    dice[denom < eps] = torch.nan

    # remove background if requested
    if not include_bg:
        dice_for_mean = dice[:, 1:]
    else:
        dice_for_mean = dice

    mean_dice = torch.nanmean(dice_for_mean, dim=1)

    return dice, mean_dice
def logits_to_onehot(
    logits: torch.Tensor,
    seg: torch.Tensor,
    num_classes: int,
):
    pred = torch.argmax(logits, dim=1)  # [B,H,W,D]

    if seg.ndim == 5 and seg.shape[1] == 1:
        gt = seg[:, 0]
    else:
        gt = seg

    pred_1h = F.one_hot(pred.long(), num_classes=num_classes)
    gt_1h = F.one_hot(gt.long(), num_classes=num_classes)

    pred_1h = pred_1h.permute(0, 4, 1, 2, 3).float()
    gt_1h = gt_1h.permute(0, 4, 1, 2, 3).float()

    return pred_1h, gt_1h
def compute_sensitivity_specificity_from_onehot(
    pred_1h: torch.Tensor,
    gt_1h: torch.Tensor,
    include_bg: bool,
    eps: float = 1e-6,
):
    """
    pred_1h, gt_1h: [B, C, H, W, D]

    returns:
      sensitivity: [B, C_eval]
      specificity: [B, C_eval]
    """
    pred = pred_1h.bool()
    gt = gt_1h.bool()

    dims = (2, 3, 4)

    tp = (pred & gt).sum(dim=dims).float()
    fp = (pred & (~gt)).sum(dim=dims).float()
    tn = ((~pred) & (~gt)).sum(dim=dims).float()
    fn = ((~pred) & gt).sum(dim=dims).float()

    sensitivity = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    # classes absent in gt can make sensitivity not meaningful
    gt_pos = gt.sum(dim=dims)
    sensitivity[gt_pos == 0] = torch.nan

    # classes absent in gt negatives can make specificity not meaningful
    gt_neg = (~gt).sum(dim=dims)
    specificity[gt_neg == 0] = torch.nan

    if not include_bg:
        sensitivity = sensitivity[:, 1:]
        specificity = specificity[:, 1:]

    return sensitivity, specificity

def sanitize_metric_tensor(x: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    x[torch.isinf(x)] = torch.nan
    return x


@torch.no_grad()
def compute_extra_metrics(
    logits: torch.Tensor,
    seg: torch.Tensor,
    num_classes: int,
    include_bg: bool,
    hd95_percentile: float = 95,
):
    pred_1h, gt_1h = logits_to_onehot(logits, seg, num_classes)

    iou_metric = MeanIoU(
        include_background=True,
        reduction="none",
        ignore_empty=True,
    )

    hd95_metric = HausdorffDistanceMetric(
        include_background=True,
        percentile=hd95_percentile,
        reduction="none",
    )

    iou = sanitize_metric_tensor(iou_metric(pred_1h, gt_1h))
    hd95 = sanitize_metric_tensor(hd95_metric(pred_1h, gt_1h))

    sensitivity, specificity = compute_sensitivity_specificity_from_onehot(
        pred_1h=pred_1h,
        gt_1h=gt_1h,
        include_bg=include_bg,
    )

    sensitivity = sanitize_metric_tensor(sensitivity)
    specificity = sanitize_metric_tensor(specificity)

    if not include_bg:
        iou = iou[:, 1:]
        hd95 = hd95[:, 1:]

    return {
        "iou": iou,
        "hd95": hd95,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }

def init_metric_accumulators(num_eval_classes: int):
    return {
        "iou_sum": torch.zeros(num_eval_classes, dtype=torch.float64),
        "iou_count": torch.zeros(num_eval_classes, dtype=torch.float64),
        "hd95_sum": torch.zeros(num_eval_classes, dtype=torch.float64),
        "hd95_count": torch.zeros(num_eval_classes, dtype=torch.float64),
        "sensitivity_sum": torch.zeros(num_eval_classes, dtype=torch.float64),
        "sensitivity_count": torch.zeros(num_eval_classes, dtype=torch.float64),
        "specificity_sum": torch.zeros(num_eval_classes, dtype=torch.float64),
        "specificity_count": torch.zeros(num_eval_classes, dtype=torch.float64),
    }

def update_metric_accumulators(acc, metric_name: str, values: torch.Tensor):
    """
    values expected shape: [B, C]
    """
    values = values.detach().cpu().double()

    if values.ndim == 1:
        values = values.unsqueeze(0)

    if values.ndim != 2:
        raise ValueError(f"{metric_name} values must have shape [B,C], got {tuple(values.shape)}")

    valid = ~torch.isnan(values)

    acc[f"{metric_name}_sum"] += torch.where(
        valid, values, torch.zeros_like(values)
    ).sum(dim=0)

    acc[f"{metric_name}_count"] += valid.sum(dim=0).double()
def finalize_metric(acc, metric_name: str):
    return acc[f"{metric_name}_sum"] / torch.clamp(acc[f"{metric_name}_count"], min=1.0)

def train_one_epoch(
    cfg: CFG,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer,
    scaler: torch.amp.GradScaler
):
    model.train()
    running_loss = 0.0
    dice_sum = 0.0
    dice_count = 0

    num_plot_classes = cfg.num_classes - 1  # classes 1..3
    per_class_sum = torch.zeros(num_plot_classes, dtype=torch.float64)
    per_class_count = torch.zeros(num_plot_classes, dtype=torch.float64)

    start = perf_counter()

    for idx, (img, seg) in enumerate(train_loader):
        if idx % 200 == 0 and idx > 0:
            print(f"reached {idx} index")

        if idx == len(train_loader) - 1:
            end = perf_counter()
            elapsed_s = (end - start)
            # print(f"Time taken: {elapsed_s:.3f}s")

        img = img.to(cfg.device, non_blocking=True)
        seg = seg.to(cfg.device, non_blocking=True)

        # checks before forward
        if not torch.isfinite(img).all():
            print(f"[batch {idx}] Non-finite image detected before forward")
            print("img min/max:", img.min().item(), img.max().item())
            raise RuntimeError("Image contains NaN or inf")

        if not torch.isfinite(seg).all():
            print(f"[batch {idx}] Non-finite seg detected before forward")
            raise RuntimeError("Seg contains NaN or inf")

        # MONAI DiceCELoss with to_onehot_y=True expects [B,1,H,W,D]
        seg_for_loss = seg.unsqueeze(1) if seg.ndim == 4 else seg

        optimizer.zero_grad(set_to_none=True)

        # unetr modif cele 2 linii le scot
        # with torch.autocast(device_type="cuda", enabled=(cfg.device.type == "cuda")):
      
        amp_enabled = should_use_amp(cfg)
        with torch.autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(img)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            elif logits.ndim == 6:
                logits = logits[:, 0]
            loss = cfg.loss_fn(logits, seg_for_loss)

        # checks after forward
        if not torch.isfinite(logits).all():
            print(f"[batch {idx}] Non-finite logits detected")
            print("img min/max:", img.min().item(), img.max().item())
            print("seg unique:", torch.unique(seg))
            raise RuntimeError("Logits contain NaN or inf")

        if not torch.isfinite(loss):
            print(f"[batch {idx}] Non-finite loss detected")
            print("img min/max:", img.min().item(), img.max().item())
            print("seg unique:", torch.unique(seg))
            raise RuntimeError("Loss is NaN or inf")

        # backward + gradient clipping
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += float(loss.item())

        dice_per_class, mean_dice = dice_from_logits(
            logits,
            seg,
            cfg.num_classes,
            cfg.include_bg_in_metric
        )

        dice_sum += torch.nansum(mean_dice).item()
        dice_count += (~torch.isnan(mean_dice)).sum().item()

        d_compact = dice_per_class[:, 1:].detach().cpu().double()
        valid = ~torch.isnan(d_compact)

        per_class_sum += torch.where(valid, d_compact, torch.zeros_like(d_compact)).sum(dim=0)
        per_class_count += valid.sum(dim=0).double()

    train_loss = running_loss / max(1, len(train_loader))
    train_mean_tumor_dice = dice_sum / max(1, dice_count)
    train_dice_per_class_mean = per_class_sum / torch.clamp(per_class_count, min=1.0)

    return train_loss, train_mean_tumor_dice, train_dice_per_class_mean

@torch.no_grad()
def evaluate_one_epoch(
    cfg: CFG,
    model: nn.Module,
    loader: DataLoader,
    split_name: str = "val",
    compute_iou: bool = True,
    compute_hd95: bool = True,
    compute_sensitivity: bool = True,
    compute_specificity: bool = True,
):
    model.eval()

    running_loss = 0.0
    dice_sum = 0.0
    dice_count = 0

    num_eval_classes = cfg.num_classes if cfg.include_bg_in_metric else (cfg.num_classes - 1)

    per_class_sum = torch.zeros(num_eval_classes, dtype=torch.float64)
    per_class_count = torch.zeros(num_eval_classes, dtype=torch.float64)

    extra_acc = init_metric_accumulators(num_eval_classes)

    start = perf_counter()

    for img, seg in loader:
        img = img.to(cfg.device, non_blocking=True)
        seg = seg.to(cfg.device, non_blocking=True)

        seg_for_loss = seg.unsqueeze(1) if seg.ndim == 4 else seg

        logits = model(img)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        elif logits.ndim == 6:
            logits = logits[:, 0]

        loss = cfg.loss_fn(logits, seg_for_loss)
        running_loss += float(loss.item())

        dice_per_class, mean_dice = dice_from_logits(
            logits,
            seg,
            cfg.num_classes,
            cfg.include_bg_in_metric,
        )

        dice_sum += torch.nansum(mean_dice).item()
        dice_count += (~torch.isnan(mean_dice)).sum().item()

        d_compact = dice_per_class if cfg.include_bg_in_metric else dice_per_class[:, 1:]
        d_compact = d_compact.detach().cpu().double()
        valid = ~torch.isnan(d_compact)

        per_class_sum += torch.where(valid, d_compact, torch.zeros_like(d_compact)).sum(dim=0)
        per_class_count += valid.sum(dim=0).double()

        # Only compute extra metrics if at least one of them is requested
        if compute_iou or compute_hd95 or compute_sensitivity or compute_specificity:
            extra = compute_extra_metrics(
                logits=logits,
                seg=seg,
                num_classes=cfg.num_classes,
                include_bg=cfg.include_bg_in_metric,
                hd95_percentile=cfg.hd95_percentile,
            )

            if compute_iou:
                update_metric_accumulators(extra_acc, "iou", extra["iou"])

            if compute_hd95:
                update_metric_accumulators(extra_acc, "hd95", extra["hd95"])

            if compute_sensitivity:
                update_metric_accumulators(extra_acc, "sensitivity", extra["sensitivity"])

            if compute_specificity:
                update_metric_accumulators(extra_acc, "specificity", extra["specificity"])

    elapsed_s = perf_counter() - start

    mean_loss = running_loss / max(1, len(loader))
    mean_tumor_dice = dice_sum / max(1, dice_count)
    dice_pc_mean = per_class_sum / torch.clamp(per_class_count, min=1.0)

    result = {
        f"{split_name}_loss": mean_loss,
        f"{split_name}_mean_tumor_dice": mean_tumor_dice,
        f"{split_name}_dice_per_class_mean": dice_pc_mean,
        f"{split_name}_iou_per_class_mean": finalize_metric(extra_acc, "iou") if compute_iou else None,
        f"{split_name}_hd95_per_class_mean": finalize_metric(extra_acc, "hd95") if compute_hd95 else None,
        f"{split_name}_sensitivity_per_class_mean": finalize_metric(extra_acc, "sensitivity") if compute_sensitivity else None,
        f"{split_name}_specificity_per_class_mean": finalize_metric(extra_acc, "specificity") if compute_specificity else None,
        f"{split_name}_elapsed_s": elapsed_s,
    }

    return result

import json
import os
from typing import Dict, List
def main(cfg: CFG):
    print("Training modality: multimodal_4ch")
    seed_everything(cfg.seed)

    patient_names = sorted([
        os.path.join(cfg.root, d) for d in os.listdir(cfg.root)
        if os.path.isdir(os.path.join(cfg.root, d))
    ])

    train_loader, val_loader, test_loader = build_loaders(
        cfg=cfg, patient_names=patient_names
    )

    model = build_model(cfg).to(cfg.device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    # scaler = torch.amp.GradScaler("cuda", enabled=(cfg.device.type == "cuda"))
    # unetr scot cele 2 linii
    amp_enabled = should_use_amp(cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    run_name = getattr(cfg, "run_name", cfg.modality)

    history: Dict[str, List] = {
        "train_epochs": [],
        "val_epochs": [],
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": [],
        "train_pc": [],
        "val_pc": [],
    }

    plotter = LivePlotter(
        num_classes=cfg.num_classes,
        save_dir=os.path.join("live_plots", run_name),
    )

    best_val_dice = -1.0
    best_epoch = -1
    best_val_pc = None
    best_val_loss = None
    best_val_iou = None
    best_val_hd95 = None
    best_val_sensitivity = None
    best_val_specificity = None

    early_stopper = EarlyStopping(
        patience=5,
        min_delta=1e-3,
        mode="max",
    )

    import gc

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_dice, tr_pc = train_one_epoch(
            cfg=cfg,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
        )

        history["train_epochs"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_dice"].append(tr_dice)
        history["train_pc"].append(tr_pc.detach().cpu().tolist())

        do_validation = (epoch == 1) or (epoch % 2 == 0)

        if do_validation:
            val_metrics = evaluate_one_epoch(
                cfg=cfg,
                model=model,
                loader=val_loader,
                split_name="val",
                compute_iou=False,
                compute_hd95=False,
                compute_sensitivity=False,
                compute_specificity=False,
            )

            torch.cuda.empty_cache()
            gc.collect()

            va_loss = val_metrics["val_loss"]
            va_dice = val_metrics["val_mean_tumor_dice"]
            va_pc = val_metrics["val_dice_per_class_mean"]

            if va_dice > best_val_dice:
                best_val_dice = va_dice
                best_epoch = epoch
                best_val_pc = va_pc.detach().cpu().tolist()
                best_val_loss = va_loss

                best_val_iou = (
                    val_metrics["val_iou_per_class_mean"].detach().cpu().tolist()
                    if val_metrics["val_iou_per_class_mean"] is not None else None
                )
                best_val_hd95 = (
                    val_metrics["val_hd95_per_class_mean"].detach().cpu().tolist()
                    if val_metrics["val_hd95_per_class_mean"] is not None else None
                )
                best_val_sensitivity = (
                    val_metrics["val_sensitivity_per_class_mean"].detach().cpu().tolist()
                    if val_metrics["val_sensitivity_per_class_mean"] is not None else None
                )
                best_val_specificity = (
                    val_metrics["val_specificity_per_class_mean"].detach().cpu().tolist()
                    if val_metrics["val_specificity_per_class_mean"] is not None else None
                )

                os.makedirs("checkpoints", exist_ok=True)
                ckpt_path = os.path.join("checkpoints", f"best_model_{run_name}.pth")

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": best_val_dice,
                    "val_pc": best_val_pc,
                    "val_loss": best_val_loss,
                    "val_iou": best_val_iou,
                    "val_hd95": best_val_hd95,
                    "val_sensitivity": best_val_sensitivity,
                    "val_specificity": best_val_specificity,
                    "model_name": cfg.model_name,
                    "model_params": cfg.model_params,
                }, ckpt_path)

                print(f"[checkpoint] saved new best model at epoch {epoch} with val_dice={va_dice:.4f}")

            if scheduler is not None:
                if cfg.scheduler_name == "reduce_on_plateau":
                    scheduler.step(va_loss)
                else:
                    scheduler.step()

            history["val_epochs"].append(epoch)
            history["val_loss"].append(va_loss)
            history["val_dice"].append(va_dice)
            history["val_pc"].append(va_pc.detach().cpu().tolist())

            plotter.update(
                train_epochs=history["train_epochs"],
                val_epochs=history["val_epochs"],
                train_loss=history["train_loss"],
                val_loss=history["val_loss"],
                train_dice=history["train_dice"],
                val_dice=history["val_dice"],
                train_pc=history["train_pc"],
                val_pc=history["val_pc"],
            )

            print(
                f"[epoch {epoch}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
                f"train_dice={tr_dice:.4f} val_dice={va_dice:.4f}"
            )

            if early_stopper.step(va_dice):
                print(f"Early stopping triggered at epoch {epoch}")
                break

        else:
            plotter.update(
                train_epochs=history["train_epochs"],
                val_epochs=history["val_epochs"],
                train_loss=history["train_loss"],
                val_loss=history["val_loss"],
                train_dice=history["train_dice"],
                val_dice=history["val_dice"],
                train_pc=history["train_pc"],
                val_pc=history["val_pc"],
            )

            print(
                f"[epoch {epoch}] train_loss={tr_loss:.4f} "
                f"train_dice={tr_dice:.4f} (no validation this epoch)"
            )

    ckpt_path = os.path.join("checkpoints", f"best_model_{run_name}.pth")
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate_one_epoch(
        cfg=cfg,
        model=model,
        loader=test_loader,
        split_name="test",
        compute_iou=True,
        compute_hd95=True,
        compute_sensitivity=True,
        compute_specificity=True,
    )

    torch.cuda.empty_cache()
    gc.collect()

    print("\n=== BEST MODEL ===")
    print(f"Model: {cfg.model_name}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val Dice: {best_val_dice:.4f}")
    print(f"Best per-class Dice: {best_val_pc}")
    print(f"Best per-class IoU: {best_val_iou}")
    print(f"Best per-class HD95: {best_val_hd95}")
    print(f"Best per-class Sensitivity: {best_val_sensitivity}")
    print(f"Best per-class Specificity: {best_val_specificity}")

    print("\n=== TEST METRICS (best checkpoint) ===")
    print(f"Test Dice: {test_metrics['test_mean_tumor_dice']:.4f}")
    print(f"Test per-class Dice: {test_metrics['test_dice_per_class_mean'].detach().cpu().tolist()}")
    print(f"Test per-class IoU: {test_metrics['test_iou_per_class_mean'].detach().cpu().tolist() if test_metrics['test_iou_per_class_mean'] is not None else None}")
    print(f"Test per-class HD95: {test_metrics['test_hd95_per_class_mean'].detach().cpu().tolist() if test_metrics['test_hd95_per_class_mean'] is not None else None}")
    print(f"Test per-class Sensitivity: {test_metrics['test_sensitivity_per_class_mean'].detach().cpu().tolist() if test_metrics['test_sensitivity_per_class_mean'] is not None else None}")
    print(f"Test per-class Specificity: {test_metrics['test_specificity_per_class_mean'].detach().cpu().tolist() if test_metrics['test_specificity_per_class_mean'] is not None else None}")

    os.makedirs("results", exist_ok=True)
    summary_path = os.path.join("results", f"{run_name}_best_metrics.json")

    with open(summary_path, "w") as f:
        json.dump({
            "run_name": run_name,
            "modality": "multimodal_4ch",
            "model_name": cfg.model_name,
            "model_params": cfg.model_params,
            "best_epoch": best_epoch,
            "best_val_dice": best_val_dice,
            "best_val_loss": best_val_loss,
            "best_val_pc": best_val_pc,
            "best_val_iou": best_val_iou,
            "best_val_hd95": best_val_hd95,
            "best_val_sensitivity": best_val_sensitivity,
            "best_val_specificity": best_val_specificity,
            "test_loss": test_metrics["test_loss"],
            "test_dice": test_metrics["test_mean_tumor_dice"],
            "test_dice_per_class": test_metrics["test_dice_per_class_mean"].detach().cpu().tolist(),
            "test_iou_per_class": test_metrics["test_iou_per_class_mean"].detach().cpu().tolist() if test_metrics["test_iou_per_class_mean"] is not None else None,
            "test_hd95_per_class": test_metrics["test_hd95_per_class_mean"].detach().cpu().tolist() if test_metrics["test_hd95_per_class_mean"] is not None else None,
            "test_sensitivity_per_class": test_metrics["test_sensitivity_per_class_mean"].detach().cpu().tolist() if test_metrics["test_sensitivity_per_class_mean"] is not None else None,
            "test_specificity_per_class": test_metrics["test_specificity_per_class_mean"].detach().cpu().tolist() if test_metrics["test_specificity_per_class_mean"] is not None else None,
            "history": history,
        }, f, indent=2)

    print(f"Saved summary to {summary_path}")

    return {
        "run_name": run_name,
        "model_name": cfg.model_name,
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "best_val_loss": best_val_loss,
        "best_val_pc": best_val_pc,
        "best_val_iou": best_val_iou,
        "best_val_hd95": best_val_hd95,
        "best_val_sensitivity": best_val_sensitivity,
        "best_val_specificity": best_val_specificity,
        "test_dice": test_metrics["test_mean_tumor_dice"],
        "test_dice_per_class": test_metrics["test_dice_per_class_mean"].detach().cpu().tolist(),
        "test_iou_per_class": test_metrics["test_iou_per_class_mean"].detach().cpu().tolist() if test_metrics["test_iou_per_class_mean"] is not None else None,
        "test_hd95_per_class": test_metrics["test_hd95_per_class_mean"].detach().cpu().tolist() if test_metrics["test_hd95_per_class_mean"] is not None else None,
        "test_sensitivity_per_class": test_metrics["test_sensitivity_per_class_mean"].detach().cpu().tolist() if test_metrics["test_sensitivity_per_class_mean"] is not None else None,
        "test_specificity_per_class": test_metrics["test_specificity_per_class_mean"].detach().cpu().tolist() if test_metrics["test_specificity_per_class_mean"] is not None else None,
    }

if __name__ == "__main__":
    cfg = CFG("config.yaml")
    main(cfg)
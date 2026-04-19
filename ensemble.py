from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from monai.metrics import HausdorffDistanceMetric, MeanIoU

from dataset import BraTSMultiModalDataset, make_patient_splits
from utils import CFG, seed_everything
from train import build_model


# =========================================================
# Configurable model list
# =========================================================

MODEL_SPECS = [
    {
        "name": "unet",
        "config": "configs/unet.yaml",
        "checkpoint": "checkpoints/best_model_unet_4ch.pth",
        "val_eval_json": "eval_outputs/unet_4ch/val/case_metrics.json",
    },
    {
        "name": "segresnet",
        "config": "configs/segresnet.yaml",
        "checkpoint": "checkpoints/best_model_segresnet_4ch.pth",
        "val_eval_json": "eval_outputs/segresnet_4ch/val/case_metrics.json",
    },
    {
        "name": "dynunet",
        "config": "configs/dynunet.yaml",
        "checkpoint": "checkpoints/best_model_dynunet_4ch.pth",
        "val_eval_json": "eval_outputs/dynunet_4ch/val/case_metrics.json",
    },
    {
        "name": "unetr",
        "config": "configs/unetr.yaml",
        "checkpoint": "checkpoints/best_model_unetr_4ch.pth",
        "val_eval_json": "eval_outputs/unetr_4ch/val/case_metrics.json",
    },
    {
        "name": "swinunetr",
        "config": "configs/swinunetr.yaml",
        "checkpoint": "checkpoints/best_model_swinunetr_4ch.pth",
        "val_eval_json": "eval_outputs/swinunetr_4ch/val/case_metrics.json",
    },
]


# =========================================================
# Loading
# =========================================================

def load_model_from_spec(spec: Dict, device: torch.device):
    cfg = CFG(spec["config"])
    model = build_model(cfg).to(device)

    ckpt = torch.load(spec["checkpoint"], map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model, cfg


# =========================================================
# Metrics
# =========================================================

def sanitize_metric_tensor(x: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    x[torch.isinf(x)] = torch.nan
    return x


def labels_to_onehot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    # labels: [B,H,W,D]
    onehot = F.one_hot(labels.long(), num_classes=num_classes)
    onehot = onehot.permute(0, 4, 1, 2, 3).float()
    return onehot


def dice_from_onehot(pred_1h: torch.Tensor, gt_1h: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (pred_1h * gt_1h).sum(dim=(2, 3, 4))
    denom = pred_1h.sum(dim=(2, 3, 4)) + gt_1h.sum(dim=(2, 3, 4))
    dice = (2.0 * inter + eps) / (denom + eps)
    dice[denom < eps] = torch.nan
    return dice


def build_brats_region_masks_from_labels(labels: torch.Tensor) -> torch.Tensor:
    """
    labels: [B,H,W,D]
    returns [B,3,H,W,D] in order WT, TC, ET
    """
    wt = (labels > 0)
    tc = (labels == 1) | (labels == 3)
    et = (labels == 3)

    return torch.stack([wt, tc, et], dim=1).float()


def compute_sensitivity_specificity_from_onehot(pred_1h: torch.Tensor, gt_1h: torch.Tensor, eps: float = 1e-6):
    pred = pred_1h.bool()
    gt = gt_1h.bool()

    dims = (2, 3, 4)

    tp = (pred & gt).sum(dim=dims).float()
    fp = (pred & (~gt)).sum(dim=dims).float()
    tn = ((~pred) & (~gt)).sum(dim=dims).float()
    fn = ((~pred) & gt).sum(dim=dims).float()

    sensitivity = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    gt_pos = gt.sum(dim=dims)
    sensitivity[gt_pos == 0] = torch.nan

    gt_neg = (~gt).sum(dim=dims)
    specificity[gt_neg == 0] = torch.nan

    return sensitivity, specificity


def compute_multiclass_metrics(pred_labels: torch.Tensor, true_labels: torch.Tensor, num_classes: int, include_bg: bool, hd95_percentile: float = 95.0):
    pred_1h = labels_to_onehot(pred_labels, num_classes)
    true_1h = labels_to_onehot(true_labels, num_classes)

    dice = dice_from_onehot(pred_1h, true_1h)

    iou_metric = MeanIoU(include_background=True, reduction="none", ignore_empty=True)
    iou = sanitize_metric_tensor(iou_metric(pred_1h, true_1h))

    hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=hd95_percentile, reduction="none")
    hd95 = sanitize_metric_tensor(hd95_metric(pred_1h, true_1h))

    sensitivity, specificity = compute_sensitivity_specificity_from_onehot(pred_1h, true_1h)
    sensitivity = sanitize_metric_tensor(sensitivity)
    specificity = sanitize_metric_tensor(specificity)

    if not include_bg:
        dice = dice[:, 1:]
        iou = iou[:, 1:]
        hd95 = hd95[:, 1:]
        sensitivity = sensitivity[:, 1:]
        specificity = specificity[:, 1:]

    return {
        "dice": dice,
        "iou": iou,
        "hd95": hd95,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def compute_region_metrics(pred_labels: torch.Tensor, true_labels: torch.Tensor, hd95_percentile: float = 95.0):
    pred_regions = build_brats_region_masks_from_labels(pred_labels)
    true_regions = build_brats_region_masks_from_labels(true_labels)

    dice = dice_from_onehot(pred_regions, true_regions)

    iou_metric = MeanIoU(include_background=True, reduction="none", ignore_empty=True)
    iou = sanitize_metric_tensor(iou_metric(pred_regions, true_regions))

    hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=hd95_percentile, reduction="none")
    hd95 = sanitize_metric_tensor(hd95_metric(pred_regions, true_regions))

    sensitivity, specificity = compute_sensitivity_specificity_from_onehot(pred_regions, true_regions)
    sensitivity = sanitize_metric_tensor(sensitivity)
    specificity = sanitize_metric_tensor(specificity)

    return {
        "dice": dice,
        "iou": iou,
        "hd95": hd95,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def init_metric_sums(num_channels: int):
    return {
        "dice_sum": np.zeros(num_channels, dtype=np.float64),
        "dice_count": np.zeros(num_channels, dtype=np.float64),
        "iou_sum": np.zeros(num_channels, dtype=np.float64),
        "iou_count": np.zeros(num_channels, dtype=np.float64),
        "hd95_sum": np.zeros(num_channels, dtype=np.float64),
        "hd95_count": np.zeros(num_channels, dtype=np.float64),
        "sensitivity_sum": np.zeros(num_channels, dtype=np.float64),
        "sensitivity_count": np.zeros(num_channels, dtype=np.float64),
        "specificity_sum": np.zeros(num_channels, dtype=np.float64),
        "specificity_count": np.zeros(num_channels, dtype=np.float64),
    }


def update_metric_sums(acc: Dict, metric_dict: Dict[str, torch.Tensor]):
    for metric_name, values in metric_dict.items():
        values = values.detach().cpu().numpy()
        valid = ~np.isnan(values)
        acc[f"{metric_name}_sum"] += np.where(valid, values, 0.0).sum(axis=0)
        acc[f"{metric_name}_count"] += valid.sum(axis=0)


def finalize_metric_sums(acc: Dict):
    out = {}
    for metric_name in ["dice", "iou", "hd95", "sensitivity", "specificity"]:
        out[metric_name] = (
            acc[f"{metric_name}_sum"] / np.clip(acc[f"{metric_name}_count"], 1.0, None)
        ).tolist()
    return out


# =========================================================
# Weight builders
# =========================================================

def normalize_vector(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, 0.0, None)
    s = x.sum()
    if s < eps:
        return np.full_like(x, 1.0 / len(x))
    return x / s


def build_equal_weights(num_models: int, num_classes: int) -> torch.Tensor:
    """
    Returns [M, C]
    """
    w = np.full((num_models, num_classes), 1.0 / num_models, dtype=np.float32)
    return torch.tensor(w, dtype=torch.float32)


def build_global_region_dice_weights(model_specs: List[Dict], num_classes: int) -> torch.Tensor:
    """
    One scalar weight per model, replicated across classes.
    Uses mean validation Dice across WT/TC/ET.
    """
    scores = []

    for spec in model_specs:
        with open(spec["val_eval_json"], "r") as f:
            data = json.load(f)

        region_dice = data["summary_mean"]["regions"]["dice"]  # [WT,TC,ET]
        score = float(np.nanmean(region_dice))
        scores.append(score)

    weights_1d = normalize_vector(np.array(scores, dtype=np.float64))
    weights = np.repeat(weights_1d[:, None], num_classes, axis=1)

    # keep background uniform to avoid weird bias
    weights[:, 0] = 1.0 / len(model_specs)
    weights[:, 1:] = weights[:, 1:] / np.clip(weights[:, 1:].sum(axis=0, keepdims=True), 1e-8, None)

    return torch.tensor(weights, dtype=torch.float32)


def build_classwise_dice_weights(model_specs: List[Dict], num_classes: int) -> torch.Tensor:
    """
    Builds [M, C] using validation per-class Dice.
    Background is uniform.
    Foreground classes use per-class validation Dice.
    """
    num_models = len(model_specs)
    weights = np.zeros((num_models, num_classes), dtype=np.float64)

    # background uniform
    weights[:, 0] = 1.0 / num_models

    # classes 1..3 from per-class dice
    class_scores = []
    for spec in model_specs:
        with open(spec["val_eval_json"], "r") as f:
            data = json.load(f)

        dice_pc = data["summary_mean"]["classwise"]["dice"]

        # if include_bg_in_metric was false during eval, this should already be [c1,c2,c3]
        if len(dice_pc) == num_classes - 1:
            class_scores.append(dice_pc)
        elif len(dice_pc) == num_classes:
            class_scores.append(dice_pc[1:])
        else:
            raise ValueError(f"Unexpected classwise dice length in {spec['val_eval_json']}: {len(dice_pc)}")

    class_scores = np.asarray(class_scores, dtype=np.float64)  # [M,3]

    for c in range(1, num_classes):
        weights[:, c] = normalize_vector(class_scores[:, c - 1])

    return torch.tensor(weights, dtype=torch.float32)


# =========================================================
# Fusion
# =========================================================

def ensemble_probs_from_logits(logits_list: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    """
    logits_list: list of M tensors, each [B,C,H,W,D]
    weights: [M,C]
    """
    probs_list = [F.softmax(logits, dim=1) for logits in logits_list]
    probs = torch.stack(probs_list, dim=0)  # [M,B,C,H,W,D]

    w = weights.to(probs.device).unsqueeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [M,1,C,1,1,1]
    ensemble_probs = (w * probs).sum(dim=0)  # [B,C,H,W,D]

    return ensemble_probs


# =========================================================
# Evaluation
# =========================================================

@torch.no_grad()
def evaluate_single_model(loader: DataLoader, model: nn.Module, cfg: CFG):
    class_acc = init_metric_sums(cfg.num_classes - 1 if not cfg.include_bg_in_metric else cfg.num_classes)
    region_acc = init_metric_sums(3)

    for image, seg in loader:
        image = image.to(cfg.device, non_blocking=True)
        seg = seg.to(cfg.device, non_blocking=True)

        logits = model(image)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        elif logits.ndim == 6:
            logits = logits[:, 0]

        pred = torch.argmax(logits, dim=1)

        class_metrics = compute_multiclass_metrics(
            pred_labels=pred,
            true_labels=seg,
            num_classes=cfg.num_classes,
            include_bg=cfg.include_bg_in_metric,
            hd95_percentile=cfg.hd95_percentile,
        )
        region_metrics = compute_region_metrics(
            pred_labels=pred,
            true_labels=seg,
            hd95_percentile=cfg.hd95_percentile,
        )

        update_metric_sums(class_acc, class_metrics)
        update_metric_sums(region_acc, region_metrics)

    return {
        "classwise": finalize_metric_sums(class_acc),
        "regions": finalize_metric_sums(region_acc),
    }


@torch.no_grad()
def evaluate_ensemble(loader: DataLoader, models: List[nn.Module], cfg: CFG, weights: torch.Tensor):
    class_acc = init_metric_sums(cfg.num_classes - 1 if not cfg.include_bg_in_metric else cfg.num_classes)
    region_acc = init_metric_sums(3)

    for image, seg in loader:
        image = image.to(cfg.device, non_blocking=True)
        seg = seg.to(cfg.device, non_blocking=True)

        logits_list = []
        for model in models:
            logits = model(image)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            elif logits.ndim == 6:
                logits = logits[:, 0]
            logits_list.append(logits)

        probs = ensemble_probs_from_logits(logits_list, weights)
        pred = torch.argmax(probs, dim=1)

        class_metrics = compute_multiclass_metrics(
            pred_labels=pred,
            true_labels=seg,
            num_classes=cfg.num_classes,
            include_bg=cfg.include_bg_in_metric,
            hd95_percentile=cfg.hd95_percentile,
        )
        region_metrics = compute_region_metrics(
            pred_labels=pred,
            true_labels=seg,
            hd95_percentile=cfg.hd95_percentile,
        )

        update_metric_sums(class_acc, class_metrics)
        update_metric_sums(region_acc, region_metrics)

    return {
        "classwise": finalize_metric_sums(class_acc),
        "regions": finalize_metric_sums(region_acc),
    }


def print_result(name: str, result: Dict):
    print(f"\n=== {name} ===")

    print("BraTS regions:")
    print(f"  Dice WT/TC/ET: {result['regions']['dice']}")
    print(f"  IoU  WT/TC/ET: {result['regions']['iou']}")
    print(f"  HD95 WT/TC/ET: {result['regions']['hd95']}")

    print("Class-wise debug:")
    print(f"  Dice: {result['classwise']['dice']}")
    print(f"  IoU : {result['classwise']['iou']}")
    print(f"  HD95: {result['classwise']['hd95']}")


# =========================================================
# Main
# =========================================================

def main():
    base_cfg = CFG(MODEL_SPECS[0]["config"])
    seed_everything(base_cfg.seed)

    device = base_cfg.device
    print("Device:", device)

    patient_names = sorted([
        os.path.join(base_cfg.root, d)
        for d in os.listdir(base_cfg.root)
        if os.path.isdir(os.path.join(base_cfg.root, d))
    ])

    _, _, test_patients = make_patient_splits(patient_names, seed=base_cfg.seed)

    test_ds = BraTSMultiModalDataset(
        patient_folders=test_patients,
        root=base_cfg.root,
        transformation=None,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=base_cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(base_cfg.num_workers > 0),
    )

    models = []
    model_cfgs = []

    for spec in MODEL_SPECS:
        model, cfg = load_model_from_spec(spec, device)
        models.append(model)
        model_cfgs.append(cfg)

    results = {
        "single_models": {},
        "ensembles": {},
    }

    # -------------------------
    # Single models
    # -------------------------
    for spec, model, cfg in zip(MODEL_SPECS, models, model_cfgs):
        res = evaluate_single_model(test_loader, model, cfg)
        print_result(spec["name"], res)
        results["single_models"][spec["name"]] = res

    # -------------------------
    # Ensemble 1: Equal
    # -------------------------
    W_equal = build_equal_weights(
        num_models=len(MODEL_SPECS),
        num_classes=base_cfg.num_classes,
    )
    res_equal = evaluate_ensemble(test_loader, models, base_cfg, W_equal)
    print_result("Ensemble - Equal weights", res_equal)
    results["ensembles"]["equal"] = {
        "weights": W_equal.tolist(),
        "metrics": res_equal,
    }

    # -------------------------
    # Ensemble 2: Global region-based model weights
    # -------------------------
    W_global = build_global_region_dice_weights(
        model_specs=MODEL_SPECS,
        num_classes=base_cfg.num_classes,
    )
    res_global = evaluate_ensemble(test_loader, models, base_cfg, W_global)
    print_result("Ensemble - Global weights from val WT/TC/ET Dice", res_global)
    results["ensembles"]["global_region_weighted"] = {
        "weights": W_global.tolist(),
        "metrics": res_global,
    }

    # -------------------------
    # Ensemble 3: Class-wise weights
    # -------------------------
    W_classwise = build_classwise_dice_weights(
        model_specs=MODEL_SPECS,
        num_classes=base_cfg.num_classes,
    )
    res_classwise = evaluate_ensemble(test_loader, models, base_cfg, W_classwise)
    print_result("Ensemble - Class-wise weights from val class Dice", res_classwise)
    results["ensembles"]["classwise_weighted"] = {
        "weights": W_classwise.tolist(),
        "metrics": res_classwise,
    }

    os.makedirs("results", exist_ok=True)
    out_path = "results/ensemble_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved ensemble results to {out_path}")


if __name__ == "__main__":
    main()
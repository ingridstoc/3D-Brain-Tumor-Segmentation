from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from monai.metrics import HausdorffDistanceMetric, MeanIoU
from monai.networks.nets import UNet, SegResNet, UNETR, DynUNet, SwinUNETR

from dataset import BraTSMultiModalDataset, make_patient_splits
from utils import CFG, seed_everything


# =========================================================
# Model builders
# =========================================================

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


def build_unetr_3d(cfg: CFG) -> nn.Module:
    p = cfg.model_params
    return UNETR(
        in_channels=p.get("in_channels", 4),
        out_channels=cfg.num_classes,
        img_size=tuple(p.get("img_size", [192, 192, 160])),
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


def build_model(cfg: CFG) -> nn.Module:
    name = cfg.model_name.lower()
    if name == "unet":
        return build_unet_3d(cfg)
    if name == "segresnet":
        return build_segresnet_3d(cfg)
    if name == "unetr":
        return build_unetr_3d(cfg)
    if name == "dynunet":
        return build_dynunet_3d(cfg)
    if name == "swinunetr":
        return build_swinunetr_3d(cfg)
    raise ValueError(f"Unknown model name: {cfg.model_name}")


# =========================================================
# Helpers
# =========================================================

def sanitize_metric_tensor(x: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    x[torch.isinf(x)] = torch.nan
    return x


def logits_to_pred_labels(logits: torch.Tensor) -> torch.Tensor:
    pred = torch.argmax(logits, dim=1)  # [B,H,W,D]
    return pred


def labels_to_onehot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    # labels: [B,H,W,D]
    onehot = F.one_hot(labels.long(), num_classes=num_classes)  # [B,H,W,D,C]
    onehot = onehot.permute(0, 4, 1, 2, 3).float()              # [B,C,H,W,D]
    return onehot


def dice_from_label_masks(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    pred, true: [B,C,H,W,D] binary/onehot tensors
    returns [B,C]
    """
    inter = (pred * true).sum(dim=(2, 3, 4))
    denom = pred.sum(dim=(2, 3, 4)) + true.sum(dim=(2, 3, 4))
    dice = (2.0 * inter + eps) / (denom + eps)
    dice[denom < eps] = torch.nan
    return dice


def compute_sensitivity_specificity_from_onehot(
    pred_1h: torch.Tensor,
    gt_1h: torch.Tensor,
    eps: float = 1e-6,
):
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


def build_brats_region_masks_from_labels(labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    labels: [B,H,W,D] with classes 0,1,2,3
    WT = labels > 0
    TC = (labels == 1) | (labels == 3)
    ET = labels == 3
    """
    wt = (labels > 0)
    tc = (labels == 1) | (labels == 3)
    et = (labels == 3)

    out = torch.stack([wt, tc, et], dim=1).float()  # [B,3,H,W,D]
    return {
        "WT": out[:, 0:1],
        "TC": out[:, 1:2],
        "ET": out[:, 2:3],
        "ALL": out,
    }


def compute_binary_region_metrics(
    pred_regions: torch.Tensor,
    true_regions: torch.Tensor,
    hd95_percentile: float = 95.0,
) -> Dict[str, torch.Tensor]:
    """
    pred_regions, true_regions: [B,3,H,W,D]
    """
    dice = dice_from_label_masks(pred_regions, true_regions)

    iou_metric = MeanIoU(
        include_background=True,
        reduction="none",
        ignore_empty=True,
    )
    iou = sanitize_metric_tensor(iou_metric(pred_regions, true_regions))

    hd95_metric = HausdorffDistanceMetric(
        include_background=True,
        percentile=hd95_percentile,
        reduction="none",
    )
    hd95 = sanitize_metric_tensor(hd95_metric(pred_regions, true_regions))

    sensitivity, specificity = compute_sensitivity_specificity_from_onehot(
        pred_regions, true_regions
    )
    sensitivity = sanitize_metric_tensor(sensitivity)
    specificity = sanitize_metric_tensor(specificity)

    return {
        "dice": dice,
        "iou": iou,
        "hd95": hd95,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def compute_multiclass_metrics(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    num_classes: int,
    include_bg: bool,
    hd95_percentile: float = 95.0,
) -> Dict[str, torch.Tensor]:
    pred_1h = labels_to_onehot(pred_labels, num_classes=num_classes)
    true_1h = labels_to_onehot(true_labels, num_classes=num_classes)

    dice = dice_from_label_masks(pred_1h, true_1h)

    iou_metric = MeanIoU(
        include_background=True,
        reduction="none",
        ignore_empty=True,
    )
    iou = sanitize_metric_tensor(iou_metric(pred_1h, true_1h))

    hd95_metric = HausdorffDistanceMetric(
        include_background=True,
        percentile=hd95_percentile,
        reduction="none",
    )
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


def update_accumulator(acc: Dict[str, Dict[str, np.ndarray]], group: str, metrics: Dict[str, torch.Tensor]):
    for metric_name, values in metrics.items():
        values = values.detach().cpu().double().numpy()  # [B,C]
        if values.ndim == 1:
            values = values[None, :]

        valid = ~np.isnan(values)
        val_sum = np.where(valid, values, 0.0).sum(axis=0)
        val_cnt = valid.sum(axis=0)

        if metric_name not in acc[group]:
            acc[group][metric_name] = {
                "sum": val_sum.astype(np.float64),
                "count": val_cnt.astype(np.float64),
            }
        else:
            acc[group][metric_name]["sum"] += val_sum
            acc[group][metric_name]["count"] += val_cnt


def finalize_accumulator(acc: Dict[str, Dict[str, Dict[str, np.ndarray]]]) -> Dict:
    out = {}
    for group_name, group_metrics in acc.items():
        out[group_name] = {}
        for metric_name, d in group_metrics.items():
            mean = d["sum"] / np.clip(d["count"], 1.0, None)
            out[group_name][metric_name] = mean.tolist()
    return out


def find_best_slice(seg: np.ndarray) -> int:
    # choose axial slice with most tumor voxels
    tumor_per_slice = (seg > 0).sum(axis=(0, 1))
    if tumor_per_slice.max() == 0:
        return seg.shape[2] // 2
    return int(np.argmax(tumor_per_slice))


def colorize_segmentation(seg: np.ndarray) -> np.ndarray:
    """
    seg: [H,W] values in {0,1,2,3}
    simple RGB map
    """
    rgb = np.zeros(seg.shape + (3,), dtype=np.float32)

    rgb[seg == 1] = [1.0, 0.0, 0.0]   # red
    rgb[seg == 2] = [0.0, 1.0, 0.0]   # green
    rgb[seg == 3] = [0.0, 0.0, 1.0]   # blue

    return rgb


def save_case_visualization(
    image_4ch: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
):
    """
    image_4ch: [4,H,W,D]
    y_true, y_pred: [H,W,D]
    """
    flair = image_4ch[3]  # usually most illustrative
    z = find_best_slice(y_true)

    img2d = flair[:, :, z]
    gt2d = y_true[:, :, z]
    pr2d = y_pred[:, :, z]

    gt_rgb = colorize_segmentation(gt2d)
    pr_rgb = colorize_segmentation(pr2d)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(img2d, cmap="gray")
    axes[0, 0].set_title("FLAIR")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gt2d, cmap="gray")
    axes[0, 1].set_title("Ground Truth mask")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(pr2d, cmap="gray")
    axes[0, 2].set_title("Prediction mask")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(img2d, cmap="gray")
    axes[1, 0].imshow(gt_rgb, alpha=0.45)
    axes[1, 0].set_title("GT overlay")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(img2d, cmap="gray")
    axes[1, 1].imshow(pr_rgb, alpha=0.45)
    axes[1, 1].set_title("Prediction overlay")
    axes[1, 1].axis("off")

    diff = np.zeros(gt2d.shape + (3,), dtype=np.float32)
    diff[(gt2d > 0) & (pr2d == 0)] = [1.0, 0.0, 0.0]   # FN red
    diff[(gt2d == 0) & (pr2d > 0)] = [1.0, 1.0, 0.0]   # FP yellow
    diff[(gt2d > 0) & (pr2d > 0)] = [0.0, 1.0, 0.0]    # TP green

    axes[1, 2].imshow(img2d, cmap="gray")
    axes[1, 2].imshow(diff, alpha=0.45)
    axes[1, 2].set_title("Error map")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def get_patient_name_from_entry(entry: Dict[str, str]) -> str:
    folder = entry["patient_folder"]
    return os.path.basename(folder)


# =========================================================
# Main evaluation
# =========================================================

@torch.no_grad()
def evaluate_dataset(
    cfg: CFG,
    model: nn.Module,
    dataset: BraTSMultiModalDataset,
    split_name: str,
    save_predictions: bool = True,
    save_visualizations: bool = True,
    max_visualizations: int = 20,
):
    model.eval()

    out_root = Path("eval_outputs") / cfg.run_name / split_name
    pred_dir = out_root / "predictions_npy"
    vis_dir = out_root / "visualizations"
    case_metrics_path = out_root / "case_metrics.json"

    pred_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    accumulators = {
        "classwise": {},
        "regions": {},
    }

    per_case_results: List[Dict] = []

    for idx in range(len(dataset)):
        image, label = dataset[idx]
        meta = dataset.index[idx]
        patient_name = get_patient_name_from_entry(meta)

        x = image.unsqueeze(0).to(cfg.device, non_blocking=True)   # [1,4,H,W,D]
        y_true = label.unsqueeze(0).to(cfg.device, non_blocking=True)  # [1,H,W,D]

        logits = model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        elif logits.ndim == 6:
            logits = logits[:, 0]

        pred_labels = logits_to_pred_labels(logits)

        # class-wise metrics
        class_metrics = compute_multiclass_metrics(
            pred_labels=pred_labels,
            true_labels=y_true,
            num_classes=cfg.num_classes,
            include_bg=cfg.include_bg_in_metric,
            hd95_percentile=cfg.hd95_percentile,
        )
        update_accumulator(accumulators, "classwise", class_metrics)

        # BraTS WT/TC/ET region metrics
        pred_regions = build_brats_region_masks_from_labels(pred_labels)["ALL"]
        true_regions = build_brats_region_masks_from_labels(y_true)["ALL"]

        region_metrics = compute_binary_region_metrics(
            pred_regions=pred_regions,
            true_regions=true_regions,
            hd95_percentile=cfg.hd95_percentile,
        )
        update_accumulator(accumulators, "regions", region_metrics)

        pred_np = pred_labels[0].detach().cpu().numpy().astype(np.uint8)
        true_np = y_true[0].detach().cpu().numpy().astype(np.uint8)
        image_np = image.detach().cpu().numpy().astype(np.float32)

        if save_predictions:
            np.save(pred_dir / f"{patient_name}_pred.npy", pred_np)

        if save_visualizations and idx < max_visualizations:
            save_case_visualization(
                image_4ch=image_np,
                y_true=true_np,
                y_pred=pred_np,
                out_path=str(vis_dir / f"{patient_name}.png"),
            )

        case_result = {
            "patient": patient_name,
            "classwise": {
                "dice": class_metrics["dice"][0].detach().cpu().tolist(),
                "iou": class_metrics["iou"][0].detach().cpu().tolist(),
                "hd95": class_metrics["hd95"][0].detach().cpu().tolist(),
                "sensitivity": class_metrics["sensitivity"][0].detach().cpu().tolist(),
                "specificity": class_metrics["specificity"][0].detach().cpu().tolist(),
            },
            "regions": {
                "names": ["WT", "TC", "ET"],
                "dice": region_metrics["dice"][0].detach().cpu().tolist(),
                "iou": region_metrics["iou"][0].detach().cpu().tolist(),
                "hd95": region_metrics["hd95"][0].detach().cpu().tolist(),
                "sensitivity": region_metrics["sensitivity"][0].detach().cpu().tolist(),
                "specificity": region_metrics["specificity"][0].detach().cpu().tolist(),
            },
        }
        per_case_results.append(case_result)

        print(f"[{idx+1}/{len(dataset)}] {patient_name}")

    summary = finalize_accumulator(accumulators)

    final_output = {
        "run_name": cfg.run_name,
        "model_name": cfg.model_name,
        "split": split_name,
        "classwise_metric_order": (
            ["bg", "class1", "class2", "class3"]
            if cfg.include_bg_in_metric else
            ["class1", "class2", "class3"]
        ),
        "region_metric_order": ["WT", "TC", "ET"],
        "summary_mean": summary,
        "per_case": per_case_results,
    }

    with open(case_metrics_path, "w") as f:
        json.dump(final_output, f, indent=2)

    return final_output


def main(cfg: CFG, checkpoint_path: str, split: str = "test"):
    seed_everything(cfg.seed)

    patient_names = sorted([
        os.path.join(cfg.root, d) for d in os.listdir(cfg.root)
        if os.path.isdir(os.path.join(cfg.root, d))
    ])

    train_patients, val_patients, test_patients = make_patient_splits(
        patient_names,
        seed=cfg.seed
    )

    if split == "train":
        ds = BraTSMultiModalDataset(train_patients, cfg.root, transformation=None)
    elif split == "val":
        ds = BraTSMultiModalDataset(val_patients, cfg.root, transformation=None)
    elif split == "test":
        ds = BraTSMultiModalDataset(test_patients, cfg.root, transformation=None)
    else:
        raise ValueError("split must be one of: train, val, test")

    model = build_model(cfg).to(cfg.device)

    ckpt = torch.load(checkpoint_path, map_location=cfg.device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Model: {cfg.model_name}")
    print(f"Split: {split}")
    print(f"Cases: {len(ds)}")

    results = evaluate_dataset(
        cfg=cfg,
        model=model,
        dataset=ds,
        split_name=split,
        save_predictions=True,
        save_visualizations=True,
        max_visualizations=30,
    )

    print("\n=== REGION METRICS (BraTS standard) ===")
    print("Dice WT/TC/ET:", results["summary_mean"]["regions"]["dice"])
    print("IoU  WT/TC/ET:", results["summary_mean"]["regions"]["iou"])
    print("HD95 WT/TC/ET:", results["summary_mean"]["regions"]["hd95"])
    print("Sens WT/TC/ET:", results["summary_mean"]["regions"]["sensitivity"])
    print("Spec WT/TC/ET:", results["summary_mean"]["regions"]["specificity"])

    print("\n=== CLASS-WISE DEBUG METRICS ===")
    print("Dice:", results["summary_mean"]["classwise"]["dice"])
    print("IoU :", results["summary_mean"]["classwise"]["iou"])
    print("HD95:", results["summary_mean"]["classwise"]["hd95"])
    print("Sens:", results["summary_mean"]["classwise"]["sensitivity"])
    print("Spec:", results["summary_mean"]["classwise"]["specificity"])


if __name__ == "__main__":
    cfg = CFG("config.yaml")

    checkpoint_path = os.path.join("checkpoints", f"best_model_{cfg.run_name}.pth")
    main(cfg=cfg, checkpoint_path=checkpoint_path, split="test")
from __future__ import annotations
import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from monai.networks.nets import UNet

from dataset import make_patient_splits
from utils import CFG, seed_everything


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


def load_model(checkpoint_path: str, device: torch.device, num_classes: int = 4) -> nn.Module:
    model = build_unet_3d(num_classes=num_classes).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# -------------------------
# Ensemble dataset
# -------------------------
class BraTSEnsembleDataset(Dataset):
    def __init__(self, patient_folders: List[str], include_random_crops: bool = False):
        self.samples = []

        crop_names = ["crop"] if not include_random_crops else ["crop", "rand1", "rand2", "rand3", "rand4"]

        for patient_folder in patient_folders:
            case_id = os.path.basename(patient_folder).replace("patient_", "")

            for crop in crop_names:
                paths = {
                    "t1": os.path.join(patient_folder, f"{case_id}_T1_{crop}.npy"),
                    "t1ce": os.path.join(patient_folder, f"{case_id}_T1ce_{crop}.npy"),
                    "t2": os.path.join(patient_folder, f"{case_id}_T2_{crop}.npy"),
                    "flair": os.path.join(patient_folder, f"{case_id}_FLAIR_{crop}.npy"),
                    "seg": os.path.join(patient_folder, f"{case_id}_seg_{crop}.npy"),
                }

                if all(os.path.exists(p) for p in paths.values()):
                    self.samples.append(paths)

        if not self.samples:
            raise RuntimeError("No ensemble samples found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        t1 = np.load(s["t1"]).astype(np.float32)[None, ...]
        t1ce = np.load(s["t1ce"]).astype(np.float32)[None, ...]
        t2 = np.load(s["t2"]).astype(np.float32)[None, ...]
        flair = np.load(s["flair"]).astype(np.float32)[None, ...]
        seg = np.load(s["seg"]).astype(np.int64)

        return (
            torch.from_numpy(t1),
            torch.from_numpy(t1ce),
            torch.from_numpy(t2),
            torch.from_numpy(flair),
            torch.from_numpy(seg),
        )


# -------------------------
# Metrics
# -------------------------
def dice_from_probs(
    probs: torch.Tensor,
    seg: torch.Tensor,
    num_classes: int,
    include_bg: bool = True,
    eps: float = 1e-6,
):
    pred = torch.argmax(probs, dim=1)  # [B,H,W,D]

    gt = seg if seg.ndim == 4 else seg.squeeze(1)

    pred_1h = F.one_hot(pred, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    gt_1h = F.one_hot(gt, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    inter = (pred_1h * gt_1h).sum(dim=(2, 3, 4))
    denom = pred_1h.sum(dim=(2, 3, 4)) + gt_1h.sum(dim=(2, 3, 4))

    dice = (2.0 * inter + eps) / (denom + eps)
    dice[denom < eps] = torch.nan

    dice_for_mean = dice if include_bg else dice[:, 1:]
    mean_dice = torch.nanmean(dice_for_mean, dim=1)

    return dice, mean_dice


# -------------------------
# Weights
# -------------------------
def build_classwise_weights() -> torch.Tensor:
    """
    rows = [t1, t1ce, t2, flair]
    cols = [bg, c1, c2, c3]
    """
    return torch.tensor([
        [0.25, 0.2367, 0.2288, 0.2277],  # t1
        [0.25, 0.2968, 0.2474, 0.3099],  # t1ce
        [0.25, 0.2391, 0.2580, 0.2323],  # t2
        [0.25, 0.2274, 0.2658, 0.2300],  # flair
    ], dtype=torch.float32)


def build_equal_weights() -> torch.Tensor:
    return torch.full((4, 4), 0.25, dtype=torch.float32)


def ensemble_probs_from_logits(
    logits_list: List[torch.Tensor],
    weights: torch.Tensor,
) -> torch.Tensor:
    probs_list = [F.softmax(logits, dim=1) for logits in logits_list]

    # [M, B, C, H, W, D]
    probs = torch.stack(probs_list, dim=0)

    # weights [M, C] -> [M, 1, C, 1, 1, 1]
    w = weights.to(probs.device).unsqueeze(1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # sum over modalities
    ensemble_probs = (w * probs).sum(dim=0)
    return ensemble_probs


# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate_ensemble(
    loader: DataLoader,
    models: Dict[str, nn.Module],
    weights: torch.Tensor,
    device: torch.device,
    num_classes: int = 4,
):
    dice_sum = 0.0
    dice_count = 0

    per_class_sum = torch.zeros(num_classes - 1, dtype=torch.float64)
    per_class_count = torch.zeros(num_classes - 1, dtype=torch.float64)

    for t1, t1ce, t2, flair, seg in loader:
        t1 = t1.to(device)
        t1ce = t1ce.to(device)
        t2 = t2.to(device)
        flair = flair.to(device)
        seg = seg.to(device)

        logits_t1 = models["t1"](t1)
        logits_t1ce = models["t1ce"](t1ce)
        logits_t2 = models["t2"](t2)
        logits_flair = models["flair"](flair)

        probs = ensemble_probs_from_logits(
            [logits_t1, logits_t1ce, logits_t2, logits_flair],
            weights,
        )

        dice_per_class, mean_dice = dice_from_probs(
            probs,
            seg,
            num_classes=num_classes,
            include_bg=False,
        )

        dice_sum += torch.nansum(mean_dice).item()
        dice_count += (~torch.isnan(mean_dice)).sum().item()

        d_compact = dice_per_class[:, 1:].detach().cpu().double()
        valid = ~torch.isnan(d_compact)

        per_class_sum += torch.where(valid, d_compact, torch.zeros_like(d_compact)).sum(dim=0)
        per_class_count += valid.sum(dim=0).double()

    mean_val_dice = dice_sum / max(1, dice_count)
    mean_pc = per_class_sum / torch.clamp(per_class_count, min=1.0)

    return mean_val_dice, mean_pc
@torch.no_grad()
def evaluate_single_model(
    loader: DataLoader,
    model: nn.Module,
    modality_index: int,
    device: torch.device,
    num_classes: int = 4,
):
    dice_sum = 0.0
    dice_count = 0

    per_class_sum = torch.zeros(num_classes - 1, dtype=torch.float64)
    per_class_count = torch.zeros(num_classes - 1, dtype=torch.float64)

    for t1, t1ce, t2, flair, seg in loader:
        inputs = [t1, t1ce, t2, flair]

        img = inputs[modality_index].to(device)
        seg = seg.to(device)

        logits = model(img)
        probs = F.softmax(logits, dim=1)

        dice_per_class, mean_dice = dice_from_probs(
            probs,
            seg,
            num_classes=num_classes,
            include_bg=False,
        )

        dice_sum += torch.nansum(mean_dice).item()
        dice_count += (~torch.isnan(mean_dice)).sum().item()

        d_compact = dice_per_class[:, 1:].detach().cpu().double()
        valid = ~torch.isnan(d_compact)

        per_class_sum += torch.where(valid, d_compact, torch.zeros_like(d_compact)).sum(dim=0)
        per_class_count += valid.sum(dim=0).double()

    mean_val_dice = dice_sum / max(1, dice_count)
    mean_pc = per_class_sum / torch.clamp(per_class_count, min=1.0)

    return mean_val_dice, mean_pc

def print_result(name: str, mean_dice: float, mean_pc: torch.Tensor):
    print(f"\n=== {name} ===")
    print(f"Mean test Dice: {mean_dice:.4f}")
    print(f"Per-class Dice: {mean_pc.tolist()}")

# -------------------------
# Main
# -------------------------
def main():
    cfg = CFG("config.yaml")
    seed_everything(cfg.seed)

    device = cfg.device
    print("Device:", device)

    patient_names = sorted([
        os.path.join(cfg.root, d) for d in os.listdir(cfg.root)
        if os.path.isdir(os.path.join(cfg.root, d))
    ])

    _, _, test_patients = make_patient_splits(patient_names, seed=cfg.seed)

    test_ds = BraTSEnsembleDataset(
        test_patients,
        include_random_crops=False,   # start with center crop only
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.type == "cuda"),
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
    )

    models = {
        "t1": load_model("checkpoints/best_model_t1.pth", device),
        "t1ce": load_model("checkpoints/best_model_t1ce.pth", device),
        "t2": load_model("checkpoints/best_model_t2.pth", device),
        "flair": load_model("checkpoints/best_model_flair.pth", device),
    }
    
    results = {}

    # -------------------------
    # Single-modality
    # -------------------------
    t1_dice, t1_pc = evaluate_single_model(test_loader, models["t1"], 0, device, cfg.num_classes)
    t1ce_dice, t1ce_pc = evaluate_single_model(test_loader, models["t1ce"], 1, device, cfg.num_classes)
    t2_dice, t2_pc = evaluate_single_model(test_loader, models["t2"], 2, device, cfg.num_classes)
    flair_dice, flair_pc = evaluate_single_model(test_loader, models["flair"], 3, device, cfg.num_classes)

    print_result("T1", t1_dice, t1_pc)
    print_result("T1ce", t1ce_dice, t1ce_pc)
    print_result("T2", t2_dice, t2_pc)
    print_result("FLAIR", flair_dice, flair_pc)

    results["t1"] = {
        "mean_dice": float(t1_dice),
        "per_class": t1_pc.tolist(),
    }
    results["t1ce"] = {
        "mean_dice": float(t1ce_dice),
        "per_class": t1ce_pc.tolist(),
    }
    results["t2"] = {
        "mean_dice": float(t2_dice),
        "per_class": t2_pc.tolist(),
    }
    results["flair"] = {
        "mean_dice": float(flair_dice),
        "per_class": flair_pc.tolist(),
    }

    # -------------------------
    # Equal ensemble
    # -------------------------
    W_equal = build_equal_weights()
    eq_dice, eq_pc = evaluate_ensemble(test_loader, models, W_equal, device, cfg.num_classes)

    print_result("Equal-weight ensemble", eq_dice, eq_pc)

    results["ensemble_equal"] = {
        "mean_dice": float(eq_dice),
        "per_class": eq_pc.tolist(),
    }

    # -------------------------
    # Class-wise weighted ensemble
    # -------------------------
    W_classwise = build_classwise_weights()
    cw_dice, cw_pc = evaluate_ensemble(test_loader, models, W_classwise, device, cfg.num_classes)

    print_result("Class-wise weighted ensemble", cw_dice, cw_pc)

    results["ensemble_weighted"] = {
        "mean_dice": float(cw_dice),
        "per_class": cw_pc.tolist(),
    }

    # -------------------------
    # Save JSON
    # -------------------------
    out_path = "ensemble_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
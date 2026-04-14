from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Dict, Tuple
import os
import numpy as np
import torch
from utils import CFG

from monai.transforms import (
    Compose,
    OneOf,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandBiasFieldd,
)

def _weak_pipeline():
    return Compose([
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["image", "label"], prob=0.25, max_k=3),
        RandScaleIntensityd(keys=["image"], prob=0.20, factors=0.08),
        RandShiftIntensityd(keys=["image"], prob=0.20, offsets=0.08),
    ])


def _medium_pipeline_a():
    return Compose([
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandAffined(
            keys=["image", "label"],
            prob=0.35,
            rotate_range=(0.08, 0.08, 0.08),
            scale_range=(0.08, 0.08, 0.08),
            translate_range=(6, 6, 6),
            mode=("bilinear", "nearest"),
            padding_mode="border",
        ),
        RandScaleIntensityd(keys=["image"], prob=0.20, factors=0.10),
        RandShiftIntensityd(keys=["image"], prob=0.20, offsets=0.10),
    ])


def _medium_pipeline_b():
    return Compose([
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["image", "label"], prob=0.35, max_k=3),
        RandScaleIntensityd(keys=["image"], prob=0.35, factors=0.12),
        RandShiftIntensityd(keys=["image"], prob=0.35, offsets=0.12),
        RandGaussianNoised(keys=["image"], prob=0.20, mean=0.0, std=0.03),
        RandBiasFieldd(keys=["image"], prob=0.20, coeff_range=(0.0, 0.05)),
    ])


def _strong_pipeline():
    return Compose([
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandAffined(
            keys=["image", "label"],
            prob=0.45,
            rotate_range=(0.12, 0.12, 0.12),
            scale_range=(0.12, 0.12, 0.12),
            translate_range=(8, 8, 8),
            mode=("bilinear", "nearest"),
            padding_mode="border",
        ),
        RandScaleIntensityd(keys=["image"], prob=0.35, factors=0.15),
        RandShiftIntensityd(keys=["image"], prob=0.35, offsets=0.15),
        RandGaussianNoised(keys=["image"], prob=0.25, mean=0.0, std=0.04),
        RandBiasFieldd(keys=["image"], prob=0.25, coeff_range=(0.0, 0.08)),
    ])


def build_train_augmentations(level: str):
    level = level.lower()

    if level == "none":
        return None

    if level == "weak":
        return _weak_pipeline()

    if level == "medium":
        return _medium_pipeline_a()

    if level == "mixed4":
        return OneOf(
            transforms=[
                _weak_pipeline(),
                _medium_pipeline_a(),
                _medium_pipeline_b(),
                _strong_pipeline(),
            ],
            weights=[0.35, 0.25, 0.25, 0.15],
        )

    raise ValueError(f"Unknown augmentation preset: {level}")


class BraTSMultiModalDataset(Dataset):
    """
    Full-volume multimodal dataset: one sample per patient.

    Expected files per patient:
        case_T1_full.npy
        case_T1ce_full.npy
        case_T2_full.npy
        case_FLAIR_full.npy
        case_seg_full.npy

    Returns:
      image: FloatTensor [4, H, W, D]
      label: LongTensor  [H, W, D]
    """

    def __init__(
        self,
        patient_folders: List[str],
        root: str,
        transformation=None,
    ):
        super().__init__()
        self.root = root
        self.patient_folders = patient_folders
        self.index: List[Dict[str, str]] = []
        self.transformation = transformation

        for patient_folder in patient_folders:
            if not os.path.isdir(patient_folder):
                continue

            files = os.listdir(patient_folder)

            t1 = next((f for f in files if f.endswith("_T1_full.npy")), None)
            t1ce = next((f for f in files if f.endswith("_T1ce_full.npy")), None)
            t2 = next((f for f in files if f.endswith("_T2_full.npy")), None)
            flair = next((f for f in files if f.endswith("_FLAIR_full.npy")), None)
            seg = next((f for f in files if f.endswith("_seg_full.npy")), None)

            if not all([t1, t1ce, t2, flair, seg]):
                continue

            self.index.append({
                "t1": os.path.join(patient_folder, t1),
                "t1ce": os.path.join(patient_folder, t1ce),
                "t2": os.path.join(patient_folder, t2),
                "flair": os.path.join(patient_folder, flair),
                "seg": os.path.join(patient_folder, seg),
                "patient_folder": patient_folder,
            })

        if not self.index:
            raise RuntimeError("No multimodal full-volume pairs found. Check folder structure and filenames.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        entry = self.index[idx]

        t1 = np.load(entry["t1"]).astype(np.float32)
        t1ce = np.load(entry["t1ce"]).astype(np.float32)
        t2 = np.load(entry["t2"]).astype(np.float32)
        flair = np.load(entry["flair"]).astype(np.float32)
        seg = np.load(entry["seg"]).astype(np.int64)

        image = np.stack([t1, t1ce, t2, flair], axis=0)   # [4,H,W,D]

        sample = {
            "image": image,
            "label": seg[None, ...],   # [1,H,W,D] for MONAI spatial transforms
        }

        if self.transformation:
            sample = self.transformation(sample)

        image = torch.as_tensor(sample["image"]).float()
        label = torch.as_tensor(sample["label"]).long()

        if label.ndim == 4 and label.shape[0] == 1:
            label = label.squeeze(0)

        return image, label


def make_patient_splits(
    patient_folders: List[str],
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    rng = np.random.default_rng(seed)
    patient_folders = list(patient_folders)
    rng.shuffle(patient_folders)

    n = len(patient_folders)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    train_patients = patient_folders[:n_train]
    val_patients = patient_folders[n_train:n_train + n_val]
    test_patients = patient_folders[n_train + n_val:]

    return train_patients, val_patients, test_patients


def build_loaders(cfg: CFG, patient_names: List[str]):
    train_patients, val_patients, test_patients = make_patient_splits(
        patient_names,
        seed=cfg.seed
    )

    train_ds = BraTSMultiModalDataset(
        train_patients,
        cfg.root,
        transformation=build_train_augmentations(cfg.augmentation_name),
    )

    val_ds = BraTSMultiModalDataset(
        val_patients,
        cfg.root,
        transformation=None,
    )

    test_ds = BraTSMultiModalDataset(
        test_patients,
        cfg.root,
        transformation=None,
    )

    # smaller fixed validation subset for faster training-time validation
    val_subset_size = getattr(cfg, "val_subset_size", 50)
    val_subset_size = min(val_subset_size, len(val_ds))

    rng = np.random.default_rng(cfg.seed)
    val_indices = rng.choice(len(val_ds), size=val_subset_size, replace=False)
    val_indices = sorted(val_indices.tolist())
    val_ds_small = Subset(val_ds, val_indices)

    print("Modality: multimodal_4ch")
    print(f"Patients: total={len(patient_names)}")
    print(
        f"Patient split sizes: "
        f"train={len(train_patients)} val={len(val_patients)} test={len(test_patients)}"
    )
    print(
        f"Sample counts: "
        f"train={len(train_ds)} val_full={len(val_ds)} val_subset={len(val_ds_small)} test={len(test_ds)}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.type == "cuda"),
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds_small,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.type == "cuda"),
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
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

    print(f"train batches = {len(train_loader)}")
    print(f"val batches   = {len(val_loader)}  (subset)")
    print(f"test batches  = {len(test_loader)}")

    return train_loader, val_loader, test_loader
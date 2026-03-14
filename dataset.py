from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import os
import numpy as np
import torch
from utils import CFG

from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandBiasFieldd,
)

def build_train_augmentations():
    """
    Moderate 3D augmentations for BraTS crops.
    Spatial transforms are applied to both image and label.
    Intensity transforms are applied only to image.
    """
    return Compose([
        # --- spatial ---
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),

        RandRotate90d(
            keys=["image", "label"],
            prob=0.3,
            max_k=3,
        ),

        RandAffined(
            keys=["image", "label"],
            prob=0.2,
            rotate_range=(0.1, 0.1, 0.1),     # radians, small rotations
            scale_range=(0.0, 0.0, 0.0),
            translate_range=(6, 6, 6),        # a few voxels
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
    ])


class BraTSModalDataset(Dataset):
    """
    One dataset for one modality (here T1).

    Files per patient may contain:
        case_T1_crop.npy
        case_seg_crop.npy
        case_T1_rand1.npy
        case_seg_rand1.npy
        ...
        case_T1_rand4.npy
        case_seg_rand4.npy

    Train can use all crops.
    Val can use only tumor crop.

    Returns:
      image: FloatTensor [1, H, W, D]
      label: LongTensor  [H, W, D]
    """

    def __init__(
        self,
        patient_folders: List[str],
        root: str,
        include_random_crops: bool = True,
        transformation = None
    ):
        super().__init__()
        self.root = root
        self.patient_folders = patient_folders
        self.include_random_crops = include_random_crops
        self.index: List[Dict[str, str]] = []
        self.transformation = transformation

        for patient_folder in patient_folders:
            if not os.path.isdir(patient_folder):
                continue

            files = os.listdir(patient_folder)

            img_files = sorted(
                f for f in files
                if f.endswith(".npy") and "T1_" in f
            )

            for img_f in img_files:
                is_random_crop = "_rand" in img_f

                # for val: keep only tumor crop
                if not self.include_random_crops and is_random_crop:
                    continue

                seg_f = img_f.replace("T1_", "seg_")

                img_path = os.path.join(patient_folder, img_f)
                seg_path = os.path.join(patient_folder, seg_f)

                if os.path.exists(seg_path):
                    self.index.append({
                        "img": img_path,
                        "seg": seg_path,
                        "patient_folder": patient_folder,
                    })

        if not self.index:
            raise RuntimeError("No crop pairs found. Check folder structure and filenames.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        entry = self.index[idx]
        image = np.load(entry["img"], mmap_mode="r")
        label = np.load(entry["seg"], mmap_mode="r")
        sample = {
            "image": np.asarray(image, dtype=np.float32)[None, ...],
            "label": np.asarray(label, dtype=np.int64)[None, ...],
        }

        if self.transformation:
            sample = self.transformation(sample)

        return (
            torch.as_tensor(sample["image"]).float(),
            torch.as_tensor(sample["label"]).long()
        )

# -------------------------
# Split (70/15/15) once, reuse across modalities
# -------------------------
def make_patient_splits(
    patient_folders: List[str],
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split patient folders into train/val/test at patient level.
    All crops from one patient stay in the same split.
    """
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


def build_loaders_for_modality(cfg: CFG, patient_names: List[str]):

    train_patients, val_patients, test_patients = make_patient_splits(
        patient_names,
        seed=cfg.seed
    )

    # train = tumor crop + random crops
    # val = only tumor crop
    train_ds = BraTSModalDataset(
        train_patients,
        cfg.root,
        include_random_crops=True,
        transformation=build_train_augmentations())
    
    val_ds = BraTSModalDataset(
        val_patients,
        cfg.root,
        include_random_crops=False)

    print(f"Patients: total={len(patient_names)}")
    print(
        f"Patient split sizes: "
        f"train={len(train_patients)} val={len(val_patients)} test={len(test_patients)}"
    )
    print(
        f"Crop/sample counts: "
        f"train={len(train_ds)} val={len(val_ds)}"
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
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.type == "cuda"),
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
    )

    print(len(train_loader))
    return train_loader, val_loader
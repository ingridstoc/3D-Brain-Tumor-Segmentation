# from torch.utils.data import Dataset, DataLoader
# from typing import List, Dict, Tuple
# import os
# import numpy as np
# import torch
# from utils import CFG

# from monai.transforms import (
#     Compose,
#     RandFlipd,
#     RandRotate90d,
#     RandAffined,
#     RandScaleIntensityd,
#     RandShiftIntensityd,
#     RandGaussianNoised,
#     RandBiasFieldd,
# )


# def build_train_augmentations():
#     return Compose([
#         RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
#         RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
#         RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),

#         RandRotate90d(
#             keys=["image", "label"],
#             prob=0.3,
#             max_k=3,
#         ),
#         RandShiftIntensityd(keys=["image"], prob=0.1, offsets=0.05),
#         #RandGaussianNoised(keys=["image"], prob=0.1, std=0.01),
#     ])
# class BraTSModalDataset(Dataset):
#     """
#     One dataset for one modality (here T1).

#     Expected files per patient:
#         case_T1_crop.npy
#         case_seg_crop.npy
#         case_T1_rand1.npy
#         case_seg_rand1.npy
#         ...
#         case_T1_rand4.npy
#         case_seg_rand4.npy

#     Train uses all crops.
#     Val uses only tumor-centered crop ("crop").

#     Returns:
#       image: FloatTensor [1, H, W, D]
#       label: LongTensor  [H, W, D]
#     """

#     def __init__(
#         self,
#         patient_folders: List[str],
#         root: str,
#         include_random_crops: bool = True,
#         transformation=None,
#     ):
#         super().__init__()
#         self.root = root
#         self.patient_folders = patient_folders
#         self.include_random_crops = include_random_crops
#         self.index: List[Dict[str, str]] = []
#         self.transformation = transformation

#         for patient_folder in patient_folders:
#             if not os.path.isdir(patient_folder):
#                 continue

#             files = os.listdir(patient_folder)

#             img_files = sorted(
#                 f for f in files
#                 if f.endswith(".npy") and "_T1_" in f
#             )

#             for img_f in img_files:
#                 crop_name = os.path.splitext(img_f)[0].split("_")[-1]

#                 # validation should keep only tumor-centered crop
#                 if not self.include_random_crops and crop_name != "crop":
#                     continue

#                 seg_f = img_f.replace("_T1_", "_seg_")

#                 img_path = os.path.join(patient_folder, img_f)
#                 seg_path = os.path.join(patient_folder, seg_f)

#                 if os.path.exists(seg_path):
#                     self.index.append({
#                         "img": img_path,
#                         "seg": seg_path,
#                         "patient_folder": patient_folder,
#                     })

#         if not self.index:
#             raise RuntimeError("No crop pairs found. Check folder structure and filenames.")

#     def __len__(self) -> int:
#         return len(self.index)

#     def __getitem__(self, idx: int):
#         entry = self.index[idx]

#         sample = {
#             "image": np.load(entry["img"]).astype(np.float32)[None, ...],  # [1,H,W,D]
#             "label": np.load(entry["seg"]).astype(np.int64)[None, ...],    # [1,H,W,D]
#         }

#         if self.transformation:
#             sample = self.transformation(sample)

#         image = torch.as_tensor(sample["image"]).float()
#         label = torch.as_tensor(sample["label"]).long()

#         # MONAI may return label with channel dim after transforms
#         if label.ndim == 4 and label.shape[0] == 1:
#             label = label.squeeze(0)

#         return image, label


# def make_patient_splits(
#     patient_folders: List[str],
#     seed: int = 42
# ) -> Tuple[List[str], List[str], List[str]]:
#     """
#     Split patient folders into train/val/test at patient level.
#     All crops from one patient stay in the same split.
#     """
#     rng = np.random.default_rng(seed)
#     patient_folders = list(patient_folders)
#     rng.shuffle(patient_folders)

#     n = len(patient_folders)
#     n_train = int(0.70 * n)
#     n_val = int(0.15 * n)

#     train_patients = patient_folders[:n_train]
#     val_patients = patient_folders[n_train:n_train + n_val]
#     test_patients = patient_folders[n_train + n_val:]

#     return train_patients, val_patients, test_patients


# def build_loaders_for_modality(cfg: CFG, patient_names: List[str]):
#     train_patients, val_patients, test_patients = make_patient_splits(
#         patient_names,
#         seed=cfg.seed
#     )

#     train_ds = BraTSModalDataset(
#         train_patients,
#         cfg.root,
#         include_random_crops=True,
#         transformation=build_train_augmentations(),
#     )

#     val_ds = BraTSModalDataset(
#         val_patients,
#         cfg.root,
#         include_random_crops=False,
#         transformation=None,
#     )

#     print(f"Patients: total={len(patient_names)}")
#     print(
#         f"Patient split sizes: "
#         f"train={len(train_patients)} val={len(val_patients)} test={len(test_patients)}"
#     )
#     print(
#         f"Crop/sample counts: "
#         f"train={len(train_ds)} val={len(val_ds)}"
#     )

#     train_loader = DataLoader(
#         train_ds,
#         batch_size=cfg.batch_size,
#         shuffle=True,
#         num_workers=cfg.num_workers,
#         pin_memory=(cfg.device.type == "cuda"),
#         drop_last=False,
#         persistent_workers=(cfg.num_workers > 0),
#     )

#     val_loader = DataLoader(
#         val_ds,
#         batch_size=1,
#         shuffle=False,
#         num_workers=cfg.num_workers,
#         pin_memory=(cfg.device.type == "cuda"),
#         drop_last=False,
#         persistent_workers=(cfg.num_workers > 0),
#     )

#     print(f"train batches = {len(train_loader)}")
#     print(f"val batches   = {len(val_loader)}")

#     return train_loader, val_loader

# sus cod pt care am rulat t1 si a mers
# jos cod nou pt t1ce


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
    #RandShiftIntensityd,
)


# def build_train_augmentations():
#     return Compose([
#         RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
#         RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
#         RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
#         RandRotate90d(
#             keys=["image", "label"],
#             prob=0.3,
#             max_k=3,
#         ),
#         #RandShiftIntensityd(keys=["image"], prob=0.1, offsets=0.05),
#     ])


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

class BraTSModalDataset(Dataset):
    """
    One dataset for one modality.

    Expected files per patient:
        case_T1_crop.npy / case_seg_crop.npy
        case_T1ce_crop.npy / case_seg_crop.npy
        case_T2_crop.npy / case_seg_crop.npy
        case_FLAIR_crop.npy / case_seg_crop.npy
        etc.

    Train uses all crops.
    Val uses only tumor-centered crop ("crop").

    Returns:
      image: FloatTensor [1, H, W, D]
      label: LongTensor  [H, W, D]
    """

    def __init__(
        self,
        patient_folders: List[str],
        root: str,
        modality: str,
        include_random_crops: bool = True,
        transformation=None,
    ):
        super().__init__()
        self.root = root
        self.modality = modality.lower()
        self.patient_folders = patient_folders
        self.include_random_crops = include_random_crops
        self.index: List[Dict[str, str]] = []
        self.transformation = transformation

        modality_to_token = {
            "t1": "T1",
            "t1ce": "T1ce",
            "t2": "T2",
            "flair": "FLAIR",
        }

        if self.modality not in modality_to_token:
            raise ValueError(f"Unknown modality: {self.modality}")

        img_token = f"_{modality_to_token[self.modality]}_"

        for patient_folder in patient_folders:
            if not os.path.isdir(patient_folder):
                continue

            files = os.listdir(patient_folder)

            img_files = sorted(
                f for f in files
                if f.endswith(".npy") and img_token in f
            )

            for img_f in img_files:
                crop_name = os.path.splitext(img_f)[0].split("_")[-1]

                if not self.include_random_crops and crop_name != "crop":
                    continue

                seg_f = img_f.replace(img_token, "_seg_")

                img_path = os.path.join(patient_folder, img_f)
                seg_path = os.path.join(patient_folder, seg_f)

                if os.path.exists(seg_path):
                    self.index.append({
                        "img": img_path,
                        "seg": seg_path,
                        "patient_folder": patient_folder,
                    })

        if not self.index:
            raise RuntimeError(
                f"No crop pairs found for modality={self.modality}. "
                f"Check folder structure and filenames."
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        entry = self.index[idx]

        sample = {
            "image": np.load(entry["img"]).astype(np.float32)[None, ...],
            "label": np.load(entry["seg"]).astype(np.int64)[None, ...],
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


def build_loaders_for_modality(cfg: CFG, patient_names: List[str]):
    train_patients, val_patients, test_patients = make_patient_splits(
        patient_names,
        seed=cfg.seed
    )

    train_ds = BraTSModalDataset(
        train_patients,
        cfg.root,
        modality=cfg.modality,
        include_random_crops=False,
        transformation=build_train_augmentations(cfg.augmentation_name),
    )

    val_ds = BraTSModalDataset(
        val_patients,
        cfg.root,
        modality=cfg.modality,
        include_random_crops=False,
        transformation=None,
    )

    print(f"Modality: {cfg.modality}")
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

    print(f"train batches = {len(train_loader)}")
    print(f"val batches   = {len(val_loader)}")

    return train_loader, val_loader

# from torch.utils.data import Dataset, DataLoader
# from typing import List, Dict, Tuple
# import os
# import numpy as np
# import torch
# from utils import CFG


# class BraTSModalDataset(Dataset):
#     """
#     One dataset for one modality (here T1).
#     Each patient folder may contain:
#         case_T1_crop.npy
#         case_seg_crop.npy
#         case_T1_rand1.npy
#         case_seg_rand1.npy
#         ...
#         case_T1_rand4.npy
#         case_seg_rand4.npy

#     Each crop pair becomes one dataset sample.

#     Returns:
#       image: FloatTensor [1, H, W, D]
#       label: LongTensor  [H, W, D]
#     """

#     def __init__(self, patient_folders: List[str], root: str):
#         super().__init__()
#         self.root = root
#         self.patient_folders = patient_folders
#         self.index: List[Dict[str, str]] = []

#         for patient_folder in patient_folders:
#             if not os.path.isdir(patient_folder):
#                 continue

#             files = os.listdir(patient_folder)

#             # all T1 crop files for this patient
#             img_files = sorted(
#                 f for f in files
#                 if f.endswith(".npy") and "T1_" in f
#             )

#             for img_f in img_files:
#                 seg_f = img_f.replace("T1_", "seg_")

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

#         img = np.load(entry["img"]).astype(np.float32)
#         seg = np.load(entry["seg"]).astype(np.int64)

#         # BraTS remap: 4 -> 3
#         seg[seg == 4] = 3

#         if img.shape != seg.shape:
#             raise ValueError(
#                 f"Shape mismatch at idx={idx}: "
#                 f"img {img.shape} vs seg {seg.shape} "
#                 f"(img={entry['img']}, seg={entry['seg']})"
#             )

#         img_t = torch.from_numpy(img[None, ...]).float()  # [1,H,W,D]
#         seg_t = torch.from_numpy(seg).long()              # [H,W,D]

#         return img_t, seg_t
   
# # -------------------------
# # Split (70/15/15) once, reuse across modalities
# # -------------------------
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

#     return train_patients, val_patients


# def build_loaders_for_modality(
#     cfg: CFG,
#     patient_names: List[str],
# ):
#     train_patients, val_patients= make_patient_splits(
#         patient_names,
#         seed=cfg.seed
#     )

#     train_ds = BraTSModalDataset(train_patients, cfg.root)
#     val_ds = BraTSModalDataset(val_patients, cfg.root)
#     test_ds = BraTSModalDataset(test_patients, cfg.root)

#     print(f"Patients: total={len(patient_names)}")
#     print(
#         f"Patient split sizes: "
#         f"train={len(train_patients)} val={len(val_patients)}"
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
#     )

#     val_loader = DataLoader(
#         val_ds,
#         batch_size=1,
#         shuffle=False,
#         num_workers=cfg.num_workers,
#         pin_memory=(cfg.device.type == "cuda"),
#         drop_last=False,
#     )

#     return train_loader, val_loader


from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import os
import numpy as np
import torch
from utils import CFG


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
    ):
        super().__init__()
        self.root = root
        self.patient_folders = patient_folders
        self.include_random_crops = include_random_crops
        self.index: List[Dict[str, str]] = []

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

        img = np.load(entry["img"]).astype(np.float32)
        seg = np.load(entry["seg"]).astype(np.int64)

        # BraTS remap: 4 -> 3
        seg[seg == 4] = 3

        if img.shape != seg.shape:
            raise ValueError(
                f"Shape mismatch at idx={idx}: "
                f"img {img.shape} vs seg {seg.shape} "
                f"(img={entry['img']}, seg={entry['seg']})"
            )

        img_t = torch.from_numpy(img[None, ...]).float()  # [1,H,W,D]
        seg_t = torch.from_numpy(seg).long()              # [H,W,D]

        return img_t, seg_t


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


def build_loaders_for_modality(
    cfg: CFG,
    patient_names: List[str],
):
    train_patients, val_patients, test_patients = make_patient_splits(
        patient_names,
        seed=cfg.seed
    )

    # train = tumor crop + random crops
    train_ds = BraTSModalDataset(
        train_patients,
        cfg.root,
        include_random_crops=True,
    )

    # val = only tumor crop
    val_ds = BraTSModalDataset(
        val_patients,
        cfg.root,
        include_random_crops=False,
    )

    # test kept for later, but not used now
    # test_ds = BraTSModalDataset(
    #     test_patients,
    #     cfg.root,
    #     include_random_crops=False,
    # )

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
        batch_size=4,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.type == "cuda"),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.type == "cuda"),
        drop_last=False,
    )

    print(len(train_loader))

    return train_loader, val_loader
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Dict
import os
import numpy as np
import torch
import nibabel as nib
from utils import CFG

# -------------------------
# Dataset: one modality per dataset instance
# -------------------------
class BraTSModalDataset(Dataset):
    """
    Assumes each patient folder contains:
      *_t1.nii.gz, *_t1ce.nii.gz, *_t2.nii.gz, *_flair.nii.gz, *_seg.nii.gz

    Returns:
      image: FloatTensor [1,H,W,D]
      label: LongTensor  [H,W,D] values in {0,1,2,3} (original 4 remapped to 3)

    Cropping:
      tumor-aware random crop to patch_size (default 128^3).
    """

    def __init__(self, patient_folders: str, root: str):
        super().__init__()
        self.root = root
        self.folder = patient_folders

        self.index: List[Dict[str, str]] = []
        for patient_folder in patient_folders:
            entry = {"img": None, "seg": None} #
            for f in os.listdir(patient_folder):
                fp = os.path.join(patient_folder, f)
                if "T1_crop.npy" in f:
                    entry["img"] = fp
                # elif "t1ce.nii" in f:
                #     entry["t1ce"] = fp
                # elif "t2.nii" in f:
                #     entry["t2"] = fp
                # elif "flair.nii" in f:
                #     entry["flair"] = fp
                elif "seg_crop.npy" in f:
                    entry["seg"] = fp

            if all(entry[k] for k in ("img", "seg")): #
                self.index.append(entry)

        if not self.index:
            raise RuntimeError("No complete patient entries found. Check folder structure and filenames.")

    def __len__(self) -> int:
        return len(self.index)

    # @staticmethod
    # def _normalize_modality(x: np.ndarray) -> np.ndarray:
    #     # z-score over non-zero voxels
    #     x = x.astype(np.float32)
    #     mask = x != 0
    #     if np.any(mask):
    #         m = x[mask].mean()
    #         s = x[mask].std()
    #         if s < 1e-6:
    #             s = 1.0
    #         x[mask] = (x[mask] - m) / (s + 1e-8)
    #     return x

    # def _select_crop_hw(self, seg: np.ndarray) -> Tuple[int, int]:
    #     H, W, _ = seg.shape
    #     coords = np.where(seg > 0)

    #     if coords[0].size == 0:
    #         h0 = np.random.randint(0, max(1, H - self.patch_h + 1))
    #         w0 = np.random.randint(0, max(1, W - self.patch_w + 1))
    #         return h0, w0

    #     h_min, h_max = coords[0].min(), coords[0].max()
    #     w_min, w_max = coords[1].min(), coords[1].max()

    #     h_start_min = max(0, h_max - self.patch_h + 1)
    #     h_start_max = min(h_min, H - self.patch_h)
    #     w_start_min = max(0, w_max - self.patch_w + 1)
    #     w_start_max = min(w_min, W - self.patch_w)

    #     if h_start_max >= h_start_min:
    #         h0 = np.random.randint(h_start_min, h_start_max + 1)
    #     else:
    #         h0 = int(np.clip(h_min, 0, max(0, H - self.patch_h)))

    #     if w_start_max >= w_start_min:
    #         w0 = np.random.randint(w_start_min, w_start_max + 1)
    #     else:
    #         w0 = int(np.clip(w_min, 0, max(0, W - self.patch_w)))

    #     return h0, w0

    # def _select_crop_d(self, seg: np.ndarray) -> int:
    #     D = seg.shape[2]
    #     coords = np.where(seg > 0)

    #     if coords[2].size == 0:
    #         return max(0, (D - self.patch_d) // 2)

    #     z_min, z_max = coords[2].min(), coords[2].max()
    #     z_center = (z_min + z_max) // 2
    #     return int(np.clip(z_center - self.patch_d // 2, 0, max(0, D - self.patch_d)))

    # @staticmethod
    # def _pad_to_min_shape(vol: np.ndarray, min_shape: Tuple[int, int, int]) -> np.ndarray:
    #     """Pads with zeros if any dimension is smaller than min_shape."""
    #     H, W, D = vol.shape
    #     ph, pw, pd = min_shape

    #     pad_h = max(0, ph - H)
    #     pad_w = max(0, pw - W)
    #     pad_d = max(0, pd - D)

    #     if pad_h == 0 and pad_w == 0 and pad_d == 0:
    #         return vol

    #     pad_before = (pad_h // 2, pad_w // 2, pad_d // 2)
    #     pad_after = (pad_h - pad_before[0], pad_w - pad_before[1], pad_d - pad_before[2])

    #     return np.pad(
    #         vol,
    #         pad_width=((pad_before[0], pad_after[0]), (pad_before[1], pad_after[1]), (pad_before[2], pad_after[2])),
    #         mode="constant",
    #         constant_values=0,
    #     )

    # @staticmethod
    # def _safe_crop(vol: np.ndarray, h0: int, w0: int, d0: int, ph: int, pw: int, pd: int) -> np.ndarray:
    #     H, W, D = vol.shape
    #     h0 = int(np.clip(h0, 0, max(0, H - ph)))
    #     w0 = int(np.clip(w0, 0, max(0, W - pw)))
    #     d0 = int(np.clip(d0, 0, max(0, D - pd)))
    #     return vol[h0 : h0 + ph, w0 : w0 + pw, d0 : d0 + pd]
    def __getitem__(self, idx: int):
        entry = self.index[idx]

        def load_vol(path: str) -> np.ndarray:
            p = path.lower()
            if p.endswith(".npy"):
                return np.load(path)
            raise ValueError(f"Unsupported file type: {path}")

        # Load segmentation (npy or nii)
        seg = load_vol(entry["seg"]).astype(np.int64)

        # Remap BraTS label 4 -> 3 (so classes are 0,1,2,3)
        seg[seg == 4] = 3

        # Load image modality (npy or nii)
        img = load_vol(entry['img']).astype(np.float32)

        # Optional: sanity check that shapes match (very useful while debugging)
        if img.shape != seg.shape:
            raise ValueError(
                f"Shape mismatch for idx={idx}: img {img.shape} vs seg {seg.shape} "
                f"(img path={entry['img']}, seg path={entry['seg']})"
            )

        # Convert to torch tensors
        img_t = torch.from_numpy(img[None, ...]).float()  # [1,H,W,D]
        seg_t = torch.from_numpy(seg).long()              # [H,W,D]
       
        return img_t, seg_t

# -------------------------
# Split (70/15/15) once, reuse across modalities
# -------------------------
def make_splits(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return train_idx, val_idx, test_idx


def build_loaders_for_modality(
    cfg : CFG,
    patient_names: List[str]
):
    ds = BraTSModalDataset(patient_names, cfg.root)
    n = len(ds)
    train_idx, val_idx, test_idx = make_splits(n, cfg.seed)
    print(f"Valid patients: {n}")
    print(f"Split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.type == "cuda"),
        drop_last=False,
    )

    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.type == "cuda"),
        drop_last=False,
    )

    return train_loader, val_loader
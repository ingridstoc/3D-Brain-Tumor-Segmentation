# #!/usr/bin/env python3
# from __future__ import annotations

# import os
# import sys
# import shutil
# import tarfile
# from pathlib import Path

# import numpy as np
# import nibabel as nib


# # -----------------------------
# # Tumor-centered crop
# # -----------------------------
# def tumor_center(mask: np.ndarray) -> np.ndarray:
#     idx = np.argwhere(mask > 0)
#     if idx.size == 0:
#         return np.array(mask.shape) // 2
#     return np.round(idx.mean(axis=0)).astype(int)


# def make_crop_slices(center: np.ndarray, in_shape: tuple[int, int, int],
#                      out_shape=(128, 128, 128)):
#     center = np.asarray(center, dtype=int)
#     in_shape = np.asarray(in_shape, dtype=int)
#     out_shape = np.asarray(out_shape, dtype=int)

#     start = center - out_shape // 2
#     end = start + out_shape

#     shift_low = np.minimum(0, start)
#     start -= shift_low
#     end -= shift_low

#     shift_high = np.maximum(0, end - in_shape)
#     start -= shift_high
#     end -= shift_high

#     start = np.maximum(0, start)
#     end = np.minimum(in_shape, end)

#     return tuple(slice(int(s), int(e)) for s, e in zip(start, end))


# def crop_or_pad(arr: np.ndarray, slices,
#                 out_shape=(128, 128, 128), pad_value=0):
#     cropped = arr[slices]

#     pad_width = []
#     for dim, target in zip(cropped.shape, out_shape):
#         missing = target - dim
#         pad_width.append((0, max(0, int(missing))))

#     if any(p != (0, 0) for p in pad_width):
#         cropped = np.pad(
#             cropped,
#             pad_width,
#             mode="constant",
#             constant_values=pad_value
#         )

#     return cropped[:out_shape[0], :out_shape[1], :out_shape[2]]


# def tumor_centered_crop(img: np.ndarray, mask: np.ndarray,
#                         out_shape=(128, 128, 128)):
#     c = tumor_center(mask)
#     sl = make_crop_slices(c, img.shape, out_shape=out_shape)
#     img_c = crop_or_pad(img, sl, out_shape=out_shape)
#     mask_c = crop_or_pad(mask, sl, out_shape=out_shape)
#     return img_c, mask_c


# # -----------------------------
# # BraTS normalization
# # -----------------------------
# def normalize_brats_volume(x: np.ndarray, p_low=1, p_high=99) -> np.ndarray:
#     x = x.astype(np.float32, copy=False)

#     mask_nonzero = x > 0
#     vals = x[mask_nonzero]

#     if vals.size < 10:
#         return x

#     lo, hi = np.percentile(vals, [p_low, p_high])
#     if hi <= lo:
#         return x

#     x = np.clip(x, lo, hi)

#     vals = x[mask_nonzero]
#     mean = vals.mean()
#     std = vals.std() + 1e-6

#     x = (x - mean) / std
#     x[~mask_nonzero] = 0.0
#     return x


# # -----------------------------
# # Case processing
# # -----------------------------
# def process_case(t1: np.ndarray, seg: np.ndarray, out_dir: Path, case_id: str,
#                  out_shape=(128, 128, 128)):
#     img_c, mask_c = tumor_centered_crop(t1, seg, out_shape=out_shape)
#     img_c = normalize_brats_volume(img_c)

#     out_patient_dir = out_dir / f"patient_{case_id}"
#     out_patient_dir.mkdir(parents=True, exist_ok=True)

#     np.save(out_patient_dir / f"{case_id}_T1_crop.npy", img_c.astype(np.float32))
#     np.save(out_patient_dir / f"{case_id}_seg_crop.npy", mask_c.astype(np.uint8))


# # -----------------------------
# # Download + extract helpers
# # -----------------------------
# def extract_all_tars(root: Path):
#     tars = sorted(root.rglob("*.tar"))
#     if not tars:
#         print(f"[extract] No .tar files found under: {root}")
#         return

#     print(f"[extract] Found {len(tars)} tar files. Extracting...")
#     for i, tar_path in enumerate(tars, 1):
#         out_dir = tar_path.parent
#         print(f"  [{i}/{len(tars)}] extracting: {tar_path.name} -> {out_dir}")
#         try:
#             with tarfile.open(tar_path, "r:*") as tf:
#                 tf.extractall(path=out_dir)
#         except Exception as e:
#             print(f"    !! Failed extracting {tar_path}: {repr(e)}")


# def download_to_raw_folder(raw_dir: Path) -> Path:
#     """
#     Downloads KaggleHub dataset into raw_dir, keeping raw_dir as the single
#     top-level location for the raw dataset.
#     """
#     import kagglehub

#     raw_dir.mkdir(parents=True, exist_ok=True)

#     print("[download] kagglehub.dataset_download(...)")
#     src = Path(kagglehub.dataset_download("dschettler8845/brats-2021-task1"))
#     print(f"[download] kagglehub cache path: {src}")

#     # Copy (or rsync-like) into raw_dir
#     # If raw_dir already has content, we keep it (so reruns don't explode).
#     # We'll copy only missing top-level items.
#     for item in src.iterdir():
#         dest = raw_dir / item.name
#         if dest.exists():
#             continue
#         if item.is_dir():
#             shutil.copytree(item, dest)
#         else:
#             shutil.copy2(item, dest)

#     return raw_dir


# def remove_ds_store(root: Path):
#     for p in root.rglob(".DS_Store"):
#         try:
#             p.unlink()
#         except Exception:
#             pass


# # -----------------------------
# # Main
# # -----------------------------
# def main():
#     """
#     Enforced layout:
#       base_dir/
#         1/        <-- raw dataset (downloaded + extracted)
#         t1_out/   <-- cropped outputs
#     """
#     base_dir = Path(".").resolve()
#     raw_dir = base_dir / "1"
#     out_dir = base_dir / "t1_out"

#     out_shape = (128, 128, 128)

#     print(f"[paths] base_dir = {base_dir}")
#     print(f"[paths] raw_dir  = {raw_dir}   (RAW dataset here)")
#     print(f"[paths] out_dir  = {out_dir}   (CROPPED outputs here)")
#     base_dir.mkdir(parents=True, exist_ok=True)
#     raw_dir.mkdir(parents=True, exist_ok=True)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # 1) Download to /cod_licenta/1
#     download_to_raw_folder(raw_dir)

#     # 2) Extract tar(s) into /cod_licenta/1
#     extract_all_tars(raw_dir)
#     remove_ds_store(raw_dir)

#     # 3) Find the actual case folder root inside /cod_licenta/1
#     print("[scan] Locating case directories...")
#     nii_segs = list(raw_dir.rglob("*seg*.nii*"))
#     if not nii_segs:
#         print("[error] Could not find any '*seg*.nii*' under raw_dir. Check extraction.")
#         sys.exit(2)

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
from __future__ import annotations

import sys
import shutil
import tarfile
from pathlib import Path

import numpy as np
import nibabel as nib

from cropping import process_case


# -----------------------------
# Download + extract helpers
# -----------------------------
def extract_all_tars(root: Path):
    tars = sorted(root.rglob("*.tar"))
    if not tars:
        print(f"[extract] No .tar files found under: {root}")
        return

    print(f"[extract] Found {len(tars)} tar files. Extracting...")
    for i, tar_path in enumerate(tars, 1):
        out_dir = tar_path.parent
        print(f"  [{i}/{len(tars)}] extracting: {tar_path.name} -> {out_dir}")
        try:
            with tarfile.open(tar_path, "r:*") as tf:
                tf.extractall(path=out_dir)
        except Exception as e:
            print(f"    !! Failed extracting {tar_path}: {repr(e)}")


def download_to_raw_folder(raw_dir: Path) -> Path:
    """
    Downloads KaggleHub dataset into raw_dir, keeping raw_dir as the single
    top-level location for the raw dataset.
    """
    import kagglehub

    raw_dir.mkdir(parents=True, exist_ok=True)

    print("[download] kagglehub.dataset_download(...)")
    src = Path(kagglehub.dataset_download("dschettler8845/brats-2021-task1"))
    print(f"[download] kagglehub cache path: {src}")

    # Copy into raw_dir, skip existing items
    for item in src.iterdir():
        dest = raw_dir / item.name
        if dest.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    return raw_dir


def remove_ds_store(root: Path):
    for p in root.rglob(".DS_Store"):
        try:
            p.unlink()
        except Exception:
            pass


# -----------------------------
# Main
# -----------------------------
def main():
    """
    Enforced layout:
      base_dir/
        1/        <-- raw dataset (downloaded + extracted)
        t1_out/   <-- cropped outputs
    """
    base_dir = Path(".").resolve()
    raw_dir = base_dir / "1"
    out_dir = base_dir / "t1_out"

    out_shape = (128, 128, 128)

    print(f"[paths] base_dir = {base_dir}")
    print(f"[paths] raw_dir  = {raw_dir}   (RAW dataset here)")
    print(f"[paths] out_dir  = {out_dir}   (CROPPED outputs here)")
    base_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Download to /cod_licenta/1
    download_to_raw_folder(raw_dir)

    # 2) Extract tar(s) into /cod_licenta/1
    extract_all_tars(raw_dir)
    remove_ds_store(raw_dir)

    # 3) Find the actual case folder root inside /cod_licenta/1
    print("[scan] Locating case directories...")
    nii_segs = list(raw_dir.rglob("*seg*.nii*"))
    if not nii_segs:
        print("[error] Could not find any '*seg*.nii*' under raw_dir. Check extraction.")
        sys.exit(2)

if __name__ == "__main__":
    main()
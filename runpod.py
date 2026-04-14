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
# from __future__ import annotations

# import sys
# import shutil
# import tarfile
# from pathlib import Path

# import numpy as np
# import nibabel as nib

# from cropping import process_case


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

#     # Copy into raw_dir, skip existing items
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



# nou multiprocess
# from __future__ import annotations

# import os
# import sys
# import shutil
# import tarfile
# from pathlib import Path

# from cropping import build_crops_parallel


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
#     import kagglehub

#     raw_dir.mkdir(parents=True, exist_ok=True)

#     print("[download] kagglehub.dataset_download(...)")
#     src = Path(kagglehub.dataset_download("dschettler8845/brats-2021-task1"))
#     print(f"[download] kagglehub cache path: {src}")

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
# # Dataset scan helper
# # -----------------------------
# def is_case_dir(case_dir: Path) -> bool:
#     if not case_dir.is_dir():
#         return False

#     has_t1 = any(case_dir.glob("*t1*.nii*"))
#     has_seg = any(case_dir.glob("*seg*.nii*"))
#     return has_t1 and has_seg


# def locate_case_root(raw_dir: Path) -> Path:
#     """
#     Return the directory whose immediate children are BraTS patient folders.
#     """
#     print("[scan] Locating case directories...")

#     candidates = []

#     for folder in raw_dir.rglob("*"):
#         if not folder.is_dir():
#             continue

#         subdirs = [p for p in folder.iterdir() if p.is_dir()]
#         if not subdirs:
#             continue

#         case_like = sum(1 for s in subdirs if is_case_dir(s))
#         if case_like > 0:
#             candidates.append((case_like, folder))

#     if not candidates:
#         print("[error] Could not find a folder that contains BraTS case directories.")
#         sys.exit(2)

#     candidates.sort(key=lambda x: x[0], reverse=True)
#     best_count, best_folder = candidates[0]

#     print(f"[scan] case root = {best_folder}")
#     print(f"[scan] detected {best_count} case folders")
#     return best_folder


# # -----------------------------
# # Main
# # -----------------------------
# def main():
#     base_dir = Path(".").resolve()
#     raw_dir = base_dir / "1"
#     out_dir = base_dir / "t1_out"
#     out_shape = (128, 128, 128)

#     num_workers = int(os.environ.get("RUNPOD_NUM_WORKERS", min(8, os.cpu_count() or 4)))

#     print(f"[paths] base_dir    = {base_dir}")
#     print(f"[paths] raw_dir     = {raw_dir}")
#     print(f"[paths] out_dir     = {out_dir}")
#     print(f"[paths] out_shape   = {out_shape}")
#     print(f"[paths] num_workers = {num_workers}")

#     base_dir.mkdir(parents=True, exist_ok=True)
#     raw_dir.mkdir(parents=True, exist_ok=True)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # 1) Download raw dataset
#     download_to_raw_folder(raw_dir)

#     # 2) Extract tar archives
#     extract_all_tars(raw_dir)
#     remove_ds_store(raw_dir)

# # 3) Skip crop if dataset already exists
#     existing = [p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith("patient_")]
#     if existing:
#         print(f"[crop] Dataset already exists ({len(existing)} patients). Skipping crop step.")
#     else:
#         build_crops_parallel(
#             dataset_root=raw_dir,
#             out_dir=out_dir,
#             out_shape=out_shape,
#             num_workers=num_workers,
#             debug_first_n=1,
#             base_seed=12345,
#         )

#     print("[done] t1_out dataset ready.")


# if __name__ == "__main__":
#     main()


# jos cand foloseam cropuri
 
# from __future__ import annotations

# import os
# import sys
# import shutil
# import tarfile
# from pathlib import Path

# from cropping import run as run_cropping


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
#     import kagglehub

#     raw_dir.mkdir(parents=True, exist_ok=True)

#     print("[download] kagglehub.dataset_download(...)")
#     src = Path(kagglehub.dataset_download("dschettler8845/brats-2021-task1"))
#     print(f"[download] kagglehub cache path: {src}")

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
# # Dataset scan helper
# # -----------------------------
# def is_case_dir(case_dir: Path) -> bool:
#     if not case_dir.is_dir():
#         return False

#     nii_files = list(case_dir.glob("*.nii*"))
#     if not nii_files:
#         return False

#     names = [p.name.lower() for p in nii_files]

#     has_t1 = any("t1." in n or "_t1" in n for n in names)
#     has_t1ce = any("t1ce" in n for n in names)
#     has_t2 = any("t2." in n or "_t2" in n for n in names)
#     has_flair = any("flair" in n for n in names)
#     has_seg = any("seg" in n for n in names)

#     return has_t1 and has_t1ce and has_t2 and has_flair and has_seg


# def locate_case_root(raw_dir: Path) -> Path:
#     print("[scan] Locating case directories...")

#     # First try: directories that are themselves case dirs
#     direct_case_dirs = [p for p in raw_dir.rglob("*") if is_case_dir(p)]
#     if direct_case_dirs:
#         parent_counts = {}
#         for case_dir in direct_case_dirs:
#             parent_counts[case_dir.parent] = parent_counts.get(case_dir.parent, 0) + 1

#         best_folder = max(parent_counts, key=parent_counts.get)
#         best_count = parent_counts[best_folder]

#         print(f"[scan] case root = {best_folder}")
#         print(f"[scan] detected {best_count} case folders")
#         return best_folder

#     print("[error] Could not find a folder that contains BraTS case directories.")
#     sys.exit(2)


# def cropped_dataset_exists(out_dir: Path) -> bool:
#     patient_dirs = [p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith("patient_")]
#     if not patient_dirs:
#         return False

#     sample = patient_dirs[0]
#     needed_patterns = [
#         "*_T1_crop.npy",
#         "*_T1ce_crop.npy",
#         "*_T2_crop.npy",
#         "*_FLAIR_crop.npy",
#         "*_seg_crop.npy",
#     ]
#     return all(any(sample.glob(pattern)) for pattern in needed_patterns)


# # -----------------------------
# # Main
# # -----------------------------
# def main():
#     base_dir = Path(".").resolve()
#     raw_dir = base_dir / "1"
#     out_dir = base_dir / "brats_crops"

#     num_workers = int(os.environ.get("RUNPOD_NUM_WORKERS", min(8, os.cpu_count() or 4)))

#     print(f"[paths] base_dir    = {base_dir}")
#     print(f"[paths] raw_dir     = {raw_dir}")
#     print(f"[paths] out_dir     = {out_dir}")
#     print(f"[paths] num_workers = {num_workers}")

#     base_dir.mkdir(parents=True, exist_ok=True)
#     raw_dir.mkdir(parents=True, exist_ok=True)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # 1) Download raw dataset
#     download_to_raw_folder(raw_dir)

#     # 2) Extract tar archives
#     extract_all_tars(raw_dir)
#     remove_ds_store(raw_dir)

#     # 3) Find case root
#     case_root = locate_case_root(raw_dir)

#     # 4) Skip crop if multimodal cropped dataset already exists
#     if cropped_dataset_exists(out_dir):
#         existing = [p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith("patient_")]
#         print(f"[crop] Multimodal dataset already exists ({len(existing)} patients). Skipping crop step.")
#     else:
#         print("[crop] Building multimodal cropped dataset...")
#         run_cropping(case_root, out_dir, num_workers)

#     print("[done] brats_crops dataset ready.")


# if __name__ == "__main__":
#     main()


# fara cropuri
#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import shutil
import tarfile
from pathlib import Path

from full_volume import run as run_full_volume


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
    import kagglehub

    raw_dir.mkdir(parents=True, exist_ok=True)

    print("[download] kagglehub.dataset_download(...)")
    src = Path(kagglehub.dataset_download("dschettler8845/brats-2021-task1"))
    print(f"[download] kagglehub cache path: {src}")

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
# Dataset scan helper
# -----------------------------
def is_case_dir(case_dir: Path) -> bool:
    if not case_dir.is_dir():
        return False

    nii_files = list(case_dir.glob("*.nii*"))
    if not nii_files:
        return False

    names = [p.name.lower() for p in nii_files]

    has_t1 = any(("t1ce" not in n) and ("t1" in n) for n in names)
    has_t1ce = any("t1ce" in n for n in names)
    has_t2 = any("t2" in n for n in names)
    has_flair = any("flair" in n for n in names)
    has_seg = any("seg" in n for n in names)

    return has_t1 and has_t1ce and has_t2 and has_flair and has_seg


def locate_case_root(raw_dir: Path) -> Path:
    print("[scan] Locating case directories...")

    direct_case_dirs = [p for p in raw_dir.rglob("*") if is_case_dir(p)]
    if direct_case_dirs:
        parent_counts = {}
        for case_dir in direct_case_dirs:
            parent_counts[case_dir.parent] = parent_counts.get(case_dir.parent, 0) + 1

        best_folder = max(parent_counts, key=parent_counts.get)
        best_count = parent_counts[best_folder]

        print(f"[scan] case root = {best_folder}")
        print(f"[scan] detected {best_count} case folders")
        return best_folder

    print("[error] Could not find a folder that contains BraTS case directories.")
    sys.exit(2)


def full_dataset_exists(out_dir: Path) -> bool:
    patient_dirs = [p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith("patient_")]
    if not patient_dirs:
        return False

    sample = patient_dirs[0]
    needed_patterns = [
        "*_T1_full.npy",
        "*_T1ce_full.npy",
        "*_T2_full.npy",
        "*_FLAIR_full.npy",
        "*_seg_full.npy",
    ]
    return all(any(sample.glob(pattern)) for pattern in needed_patterns)


# -----------------------------
# Main
# -----------------------------
def main():
    base_dir = Path(".").resolve()
    raw_dir = base_dir / "1"
    out_dir = base_dir / "brats_full_fixed"

    num_workers = int(os.environ.get("RUNPOD_NUM_WORKERS", min(8, os.cpu_count() or 4)))

    print(f"[paths] base_dir    = {base_dir}")
    print(f"[paths] raw_dir     = {raw_dir}")
    print(f"[paths] out_dir     = {out_dir}")
    print(f"[paths] num_workers = {num_workers}")

    base_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    download_to_raw_folder(raw_dir)
    extract_all_tars(raw_dir)
    remove_ds_store(raw_dir)

    case_root = locate_case_root(raw_dir)

    if full_dataset_exists(out_dir):
        existing = [p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith("patient_")]
        print(f"[full] Multimodal full-volume dataset already exists ({len(existing)} patients). Skipping build step.")
    else:
        print("[full] Building multimodal full-volume dataset...")
        run_full_volume(case_root, out_dir, num_workers)

    print("[done] brats_full_fixed dataset ready.")
    
if __name__ == "__main__":
    main()
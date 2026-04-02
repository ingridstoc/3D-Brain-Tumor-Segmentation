from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os

import nibabel as nib
import numpy as np


# -----------------------------
# Normalization
# -----------------------------
def normalize_brats_volume(
    x: np.ndarray,
    p_low: float = 1,
    p_high: float = 99,
) -> np.ndarray:
    x = x.astype(np.float32, copy=False)

    mask_nonzero = x > 0
    vals = x[mask_nonzero]

    if vals.size < 10:
        return x

    lo, hi = np.percentile(vals, [p_low, p_high])
    if hi <= lo:
        return x

    x = np.clip(x, lo, hi)

    vals = x[mask_nonzero]
    mean = vals.mean()
    std = vals.std() + 1e-6

    x = (x - mean) / std
    x[~mask_nonzero] = 0.0

    return x


# -----------------------------
# File helpers
# -----------------------------
def _find_modality_file(case_dir: Path, modality: str) -> Path:
    files = sorted(case_dir.glob("*.nii*"))

    if modality == "t1":
        matches = [p for p in files if ("t1ce" not in p.name.lower()) and ("t1" in p.name.lower())]
    elif modality == "t1ce":
        matches = [p for p in files if "t1ce" in p.name.lower()]
    elif modality == "t2":
        matches = [p for p in files if "t2" in p.name.lower()]
    elif modality == "flair":
        matches = [p for p in files if "flair" in p.name.lower()]
    elif modality == "seg":
        matches = [p for p in files if "seg" in p.name.lower()]
    else:
        raise ValueError(f"Unknown modality: {modality}")

    if not matches:
        raise FileNotFoundError(f"{modality} not found in {case_dir}")

    return sorted(matches)[0]


# -----------------------------
# Per-case processing
# -----------------------------
def process_case_full_from_disk(
    case_dir: Path,
    out_dir: Path,
):
    case_id = case_dir.name

    t1_path = _find_modality_file(case_dir, "t1")
    t1ce_path = _find_modality_file(case_dir, "t1ce")
    t2_path = _find_modality_file(case_dir, "t2")
    flair_path = _find_modality_file(case_dir, "flair")
    seg_path = _find_modality_file(case_dir, "seg")

    t1 = np.asanyarray(nib.load(str(t1_path)).dataobj).astype(np.float32)
    t1ce = np.asanyarray(nib.load(str(t1ce_path)).dataobj).astype(np.float32)
    t2 = np.asanyarray(nib.load(str(t2_path)).dataobj).astype(np.float32)
    flair = np.asanyarray(nib.load(str(flair_path)).dataobj).astype(np.float32)
    seg = np.asanyarray(nib.load(str(seg_path)).dataobj).astype(np.uint8)

    seg[seg == 4] = 3

    t1 = normalize_brats_volume(t1)
    t1ce = normalize_brats_volume(t1ce)
    t2 = normalize_brats_volume(t2)
    flair = normalize_brats_volume(flair)

    out_patient_dir = out_dir / f"patient_{case_id}"
    out_patient_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_patient_dir / f"{case_id}_T1_full.npy", t1.astype(np.float32))
    np.save(out_patient_dir / f"{case_id}_T1ce_full.npy", t1ce.astype(np.float32))
    np.save(out_patient_dir / f"{case_id}_T2_full.npy", t2.astype(np.float32))
    np.save(out_patient_dir / f"{case_id}_FLAIR_full.npy", flair.astype(np.float32))
    np.save(out_patient_dir / f"{case_id}_seg_full.npy", seg.astype(np.uint8))

    return case_id


def process_case_worker(args):
    case_dir, out_dir = args
    case_id = case_dir.name
    try:
        process_case_full_from_disk(case_dir=case_dir, out_dir=out_dir)
        return True, case_id, None
    except Exception as e:
        return False, case_id, repr(e)


# -----------------------------
# Parallel runner
# -----------------------------
def build_full_volumes_parallel(
    dataset_root: Path,
    out_dir: Path,
    num_workers: int = 4,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])

    print(f"[full] Found {len(case_dirs)} case folders")
    print(f"[full] Using {num_workers} worker processes")

    if not case_dirs:
        print("[full] No case folders found.")
        return

    tasks = [(case_dir, out_dir) for case_dir in case_dirs]

    ok_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(process_case_worker, task) for task in tasks]

        for i, fut in enumerate(as_completed(futures), 1):
            ok, case_id, err = fut.result()
            if ok:
                ok_count += 1
                if i % 25 == 0 or i == len(tasks):
                    print(f"[full] [{i}/{len(tasks)}] done... latest={case_id}")
            else:
                fail_count += 1
                print(f"[full] Failed {case_id}: {err}")

    print(f"[full] Done. success={ok_count}, failed={fail_count}")


def run(dataset_root: Path, out_dir: Path, workers: int = 8):
    build_full_volumes_parallel(
        dataset_root=dataset_root,
        out_dir=out_dir,
        num_workers=workers,
    )


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python full_volume.py <dataset_root> <output_folder> [num_workers]")
        sys.exit(1)

    dataset_root = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])

    if len(sys.argv) >= 4:
        num_workers = int(sys.argv[3])
    else:
        num_workers = min(8, os.cpu_count() or 4)

    run(dataset_root, out_dir, workers=num_workers)
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os

import nibabel as nib
import numpy as np


# -----------------------------
# Crop helpers
# -----------------------------
def tumor_center(mask: np.ndarray) -> np.ndarray:
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        return np.array(mask.shape) // 2
    return np.round(idx.mean(axis=0)).astype(int)


def tumor_bbox(mask: np.ndarray):
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        return None
    mins = idx.min(axis=0)
    maxs = idx.max(axis=0)
    return mins, maxs


def make_crop_slices(
    center: np.ndarray,
    in_shape: tuple[int, int, int],
    out_shape=(128, 128, 128),
):
    center = np.asarray(center, dtype=int)
    in_shape = np.asarray(in_shape, dtype=int)
    out_shape = np.asarray(out_shape, dtype=int)

    start = center - out_shape // 2
    end = start + out_shape

    shift_low = np.minimum(0, start)
    start -= shift_low
    end -= shift_low

    shift_high = np.maximum(0, end - in_shape)
    start -= shift_high
    end -= shift_high

    start = np.maximum(0, start)
    end = np.minimum(in_shape, end)

    return tuple(slice(int(s), int(e)) for s, e in zip(start, end))


def crop_or_pad(
    arr: np.ndarray,
    slices,
    out_shape=(128, 128, 128),
    pad_value=0,
):
    cropped = arr[slices]

    pad_width = []
    for dim, target in zip(cropped.shape, out_shape):
        missing = target - dim
        pad_width.append((0, max(0, int(missing))))

    if any(p != (0, 0) for p in pad_width):
        cropped = np.pad(
            cropped,
            pad_width,
            mode="constant",
            constant_values=pad_value,
        )

    return cropped[:out_shape[0], :out_shape[1], :out_shape[2]]


# -----------------------------
# Slice strategies
# -----------------------------
def tumor_centered_slice(
    mask: np.ndarray,
    out_shape=(128, 128, 128),
):
    c = tumor_center(mask)
    return make_crop_slices(c, mask.shape, out_shape=out_shape)


def random_slice(
    rng: np.random.Generator,
    in_shape: tuple[int, int, int],
    out_shape=(128, 128, 128),
):
    in_shape = np.asarray(in_shape, dtype=int)
    out_shape = np.asarray(out_shape, dtype=int)

    starts = []
    for dim_in, dim_out in zip(in_shape, out_shape):
        max_start = int(max(0, dim_in - dim_out))
        if max_start == 0:
            starts.append(0)
        else:
            starts.append(int(rng.integers(0, max_start + 1)))

    starts = np.asarray(starts, dtype=int)
    ends = np.minimum(starts + out_shape, in_shape)

    return tuple(slice(int(s), int(e)) for s, e in zip(starts, ends))


def shifted_tumor_slice(
    rng: np.random.Generator,
    mask: np.ndarray,
    out_shape=(128, 128, 128),
    max_shift=(24, 24, 24),
):
    c = tumor_center(mask)
    max_shift = np.asarray(max_shift, dtype=int)

    shift = np.array([
        int(rng.integers(-max_shift[0], max_shift[0] + 1)),
        int(rng.integers(-max_shift[1], max_shift[1] + 1)),
        int(rng.integers(-max_shift[2], max_shift[2] + 1)),
    ], dtype=int)

    return make_crop_slices(c + shift, mask.shape, out_shape=out_shape)


def boundary_slice(
    rng: np.random.Generator,
    mask: np.ndarray,
    out_shape=(128, 128, 128),
):
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        return random_slice(rng, mask.shape, out_shape=out_shape)

    half = np.asarray(out_shape, dtype=int) // 2
    voxel = idx[int(rng.integers(0, len(idx)))]
    axis = int(rng.integers(0, 3))
    sign = int(rng.choice([-1, 1]))

    center = voxel.copy()
    offset = half[axis] - int(rng.integers(8, 24))
    center[axis] = voxel[axis] + sign * offset

    return make_crop_slices(center, mask.shape, out_shape=out_shape)


# -----------------------------
# Normalization
# -----------------------------
def normalize_brats_volume(
    x: np.ndarray,
    p_low=1,
    p_high=99,
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
# Per-case processing
# -----------------------------
def process_case(
    t1: np.ndarray,
    t1ce: np.ndarray,
    t2: np.ndarray,
    flair: np.ndarray,
    seg: np.ndarray,
    out_dir: Path,
    case_id: str,
    rng: np.random.Generator,
    out_shape=(128, 128, 128),
):
    out_patient_dir = out_dir / f"patient_{case_id}"
    out_patient_dir.mkdir(parents=True, exist_ok=True)

    crop_specs = [
        ("crop",  lambda: tumor_centered_slice(seg, out_shape=out_shape)),
        ("rand1", lambda: shifted_tumor_slice(rng, seg, out_shape=out_shape)),
        ("rand2", lambda: shifted_tumor_slice(rng, seg, out_shape=out_shape)),
        ("rand3", lambda: boundary_slice(rng, seg, out_shape=out_shape)),
        ("rand4", lambda: random_slice(rng, seg.shape, out_shape=out_shape)),
    ]

    for crop_name, crop_fn in crop_specs:
        sl = crop_fn()

        t1_c = crop_or_pad(t1, sl, out_shape=out_shape)
        t1ce_c = crop_or_pad(t1ce, sl, out_shape=out_shape)
        t2_c = crop_or_pad(t2, sl, out_shape=out_shape)
        flair_c = crop_or_pad(flair, sl, out_shape=out_shape)
        seg_c = crop_or_pad(seg, sl, out_shape=out_shape)

        t1_c = normalize_brats_volume(t1_c)
        t1ce_c = normalize_brats_volume(t1ce_c)
        t2_c = normalize_brats_volume(t2_c)
        flair_c = normalize_brats_volume(flair_c)

        np.save(out_patient_dir / f"{case_id}_T1_{crop_name}.npy", t1_c.astype(np.float32))
        np.save(out_patient_dir / f"{case_id}_T1ce_{crop_name}.npy", t1ce_c.astype(np.float32))
        np.save(out_patient_dir / f"{case_id}_T2_{crop_name}.npy", t2_c.astype(np.float32))
        np.save(out_patient_dir / f"{case_id}_FLAIR_{crop_name}.npy", flair_c.astype(np.float32))
        np.save(out_patient_dir / f"{case_id}_seg_{crop_name}.npy", seg_c.astype(np.uint8))


# -----------------------------
# Parallel worker
# -----------------------------
def _stable_seed_from_case_id(case_id: str, base_seed: int = 12345) -> int:
    return (sum(ord(c) for c in case_id) + base_seed) % (2**32 - 1)


def _find_modality_file(case_dir: Path, modality: str) -> Path:
    files = sorted(case_dir.glob("*.nii*"))
    names = [p.name.lower() for p in files]

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


def process_case_from_disk(
    case_dir: Path,
    out_dir: Path,
    out_shape=(128, 128, 128),
    base_seed: int = 12345,
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

    rng = np.random.default_rng(_stable_seed_from_case_id(case_id, base_seed))

    process_case(
        t1=t1,
        t1ce=t1ce,
        t2=t2,
        flair=flair,
        seg=seg,
        out_dir=out_dir,
        case_id=case_id,
        rng=rng,
        out_shape=out_shape,
    )

    return case_id


def process_case_worker(args):
    case_dir, out_dir, out_shape, base_seed = args
    case_id = case_dir.name
    try:
        process_case_from_disk(
            case_dir=case_dir,
            out_dir=out_dir,
            out_shape=out_shape,
            base_seed=base_seed,
        )
        return True, case_id, None
    except Exception as e:
        return False, case_id, repr(e)


# -----------------------------
# Parallel runner
# -----------------------------
def build_crops_parallel(
    dataset_root: Path,
    out_dir: Path,
    out_shape=(128, 128, 128),
    num_workers: int = 4,
    base_seed: int = 12345,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([
        p for p in dataset_root.iterdir()
        if p.is_dir()
    ])

    print(f"[crop] Found {len(case_dirs)} case folders")
    print(f"[crop] Using {num_workers} worker processes")

    if not case_dirs:
        print("[crop] No case folders found.")
        return

    tasks = [
        (case_dir, out_dir, out_shape, base_seed)
        for case_dir in case_dirs
    ]

    ok_count = 0
    fail_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(process_case_worker, task) for task in tasks]

        for i, fut in enumerate(as_completed(futures), 1):
            ok, case_id, err = fut.result()
            if ok:
                ok_count += 1
                if i % 25 == 0 or i == len(tasks):
                    print(f"[crop] [{i}/{len(tasks)}] done... latest={case_id}")
            else:
                fail_count += 1
                print(f"[crop] Failed {case_id}: {err}")

    print(f"[crop] Done. success={ok_count}, failed={fail_count}")


def run(dataset_root: Path, out_dir: Path, workers: int = 8):
    build_crops_parallel(
        dataset_root=dataset_root,
        out_dir=out_dir,
        out_shape=(128, 128, 128),
        num_workers=workers,
        base_seed=12345,
    )


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python cropping.py <dataset_root> <output_folder> [num_workers]")
        sys.exit(1)

    dataset_root = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])

    if len(sys.argv) >= 4:
        num_workers = int(sys.argv[3])
    else:
        num_workers = min(8, os.cpu_count() or 4)

    run(dataset_root, out_dir, workers=num_workers)
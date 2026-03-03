from __future__ import annotations
from pathlib import Path
import numpy as np
import nibabel as nib


# -----------------------------
# 1) Tumor-centered fixed crop
# -----------------------------
def tumor_center(mask: np.ndarray) -> np.ndarray:
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        return np.array(mask.shape) // 2
    return np.round(idx.mean(axis=0)).astype(int)


def make_crop_slices(center: np.ndarray, in_shape: tuple[int, int, int],
                     out_shape=(128, 128, 128)):
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


def crop_or_pad(arr: np.ndarray, slices,
                out_shape=(128, 128, 128), pad_value=0):
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
            constant_values=pad_value
        )

    return cropped[:out_shape[0], :out_shape[1], :out_shape[2]]


def tumor_centered_crop(img: np.ndarray, mask: np.ndarray,
                        out_shape=(128, 128, 128)):
    c = tumor_center(mask)
    sl = make_crop_slices(c, img.shape, out_shape=out_shape)
    img_c = crop_or_pad(img, sl, out_shape=out_shape)
    mask_c = crop_or_pad(mask, sl, out_shape=out_shape)
    return img_c, mask_c


# -----------------------------
# 2) BraTS normalization
# -----------------------------
def normalize_brats_volume(x: np.ndarray,
                           p_low=1,
                           p_high=99) -> np.ndarray:
    """
    Robust per-volume normalization:
    - Uses nonzero voxels (background assumed 0)
    - Percentile clipping
    - Z-score
    """
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
# 3) Per-case processing
# -----------------------------
def process_case(
    t1: np.ndarray,
    seg: np.ndarray,
    out_dir: Path,
    case_id: str,
    out_shape=(128, 128, 128),
):
    img_c, mask_c = tumor_centered_crop(t1, seg, out_shape=out_shape)

    # Normalize AFTER cropping
    img_c = normalize_brats_volume(img_c)

    out_patient_dir = out_dir / f"patient_{case_id}"
    out_patient_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_patient_dir / f"{case_id}_T1_crop.npy",
            img_c.astype(np.float32))

    np.save(out_patient_dir / f"{case_id}_seg_crop.npy",
            mask_c.astype(np.uint8))


# -----------------------------
# 4) Batch runner
# -----------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python crop_t1.py <dataset_root> <output_folder>")
        sys.exit(1)

    dataset_root = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    print(f"Found {len(case_dirs)} case folders")

    for i, case_dir in enumerate(case_dirs, 1):
        case_id = case_dir.name

        try:
            t1_path = next(case_dir.glob("*t1*.nii*"))
            seg_path = next(case_dir.glob("*seg*.nii*"))

            t1 = np.asanyarray(nib.load(str(t1_path)).dataobj).astype(np.float32)
            seg = np.asanyarray(nib.load(str(seg_path)).dataobj).astype(np.uint8)

            process_case(
                t1,
                seg,
                out_dir=out_dir,
                case_id=case_id,
                out_shape=(128, 128, 128),
            )

            if i % 25 == 0:
                print(f"[{i}/{len(case_dirs)}] processed... latest={case_id}")

        except Exception as e:
            print(f"Failed {case_id}: {repr(e)}")

    print("Done.")
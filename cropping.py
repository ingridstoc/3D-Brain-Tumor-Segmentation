from __future__ import annotations
from pathlib import Path
import numpy as np
import nibabel as nib

def save_png(path: Path, arr: np.ndarray):
    """
    Save uint8 image using imageio if available, else matplotlib.
    arr can be (H,W) or (H,W,3).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        imageio.imwrite(str(path), arr)
    except Exception:
        import matplotlib.pyplot as plt
        plt.imsave(str(path), arr)

# -----------------------------
# 1) Tumor-centered fixed crop
# -----------------------------
def tumor_center(mask: np.ndarray) -> np.ndarray:
    """Center of mass of tumor voxels; fallback to center if empty."""
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        return np.array(mask.shape) // 2
    return np.round(idx.mean(axis=0)).astype(int)

def make_crop_slices(center: np.ndarray, in_shape: tuple[int, int, int], out_shape=(128, 128, 128)):
    center = np.asarray(center, dtype=int)
    in_shape = np.asarray(in_shape, dtype=int)
    out_shape = np.asarray(out_shape, dtype=int)

    start = center - out_shape // 2
    end = start + out_shape

    # shift to fit low bound
    shift_low = np.minimum(0, start)
    start = start - shift_low
    end = end - shift_low

    # shift to fit high bound
    shift_high = np.maximum(0, end - in_shape)
    start = start - shift_high
    end = end - shift_high

    start = np.maximum(0, start)
    end = np.minimum(in_shape, end)

    return tuple(slice(int(s), int(e)) for s, e in zip(start, end))

def crop_or_pad(arr: np.ndarray, slices, out_shape=(128, 128, 128), pad_value=0):
    cropped = arr[slices]

    pad_width = []
    for dim, target in zip(cropped.shape, out_shape):
        missing = target - dim
        pad_width.append((0, max(0, int(missing))))

    if any(p != (0, 0) for p in pad_width):
        cropped = np.pad(cropped, pad_width, mode="constant", constant_values=pad_value)

    # safety: enforce exact shape
    cropped = cropped[:out_shape[0], :out_shape[1], :out_shape[2]]
    return cropped

def tumor_centered_crop(img: np.ndarray, mask: np.ndarray, out_shape=(128, 128, 128)):
    c = tumor_center(mask)
    sl = make_crop_slices(c, img.shape, out_shape=out_shape)
    img_c = crop_or_pad(img, sl, out_shape=out_shape, pad_value=0)
    mask_c = crop_or_pad(mask, sl, out_shape=out_shape, pad_value=0)
    return img_c, mask_c, sl, c

# -----------------------------
# A) Single-slice debug helpers
# -----------------------------
def pick_center_slice_index(mask_c: np.ndarray, axis: int = 2) -> int:
    """
    Pick one slice index representative for the tumor: mean index of tumor voxels.
    axis=2 => axial slices for (H, W, D) volumes.
    """
    idx = np.argwhere(mask_c > 0)
    if idx.size == 0:
        return mask_c.shape[axis] // 2
    return int(np.round(idx[:, axis].mean()))

def to_uint8_robust(x: np.ndarray) -> np.ndarray:
    """Robust intensity scaling to uint8 for visualization."""
    x = x.astype(np.float32)
    nz = x[np.abs(x) > 0]
    ref = nz if nz.size > 0 else x.reshape(-1)
    lo, hi = np.percentile(ref, [1, 99])
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max() + 1e-6)
    x = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    return (x * 255).astype(np.uint8)

def overlay_mask_rgb(gray_u8: np.ndarray, mask_slice: np.ndarray) -> np.ndarray:
    """
    Multi-class overlay for BraTS-style labels:
      1 = NCR/NET
      2 = ED
      4 = ET
    Produces RGB image with different colors per class.
    """
    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1).astype(np.uint8)

    # Pick distinct colors (R,G,B). Change if you want.
    # Using strong, easily distinguishable colors:
    # label 4 (ET): red
    # label 1 (NCR/NET): blue
    # label 2 (ED): green
    colors = {
        4: (255, 0, 0),    # ET
        1: (0, 80, 255),   # NCR/NET
        2: (0, 255, 0),    # ED
    }

    # Slight dim factor for background where overlay applies (keeps anatomy visible)
    dim = 0.25

    for lab, (r, g, b) in colors.items():
        m = (mask_slice == lab)
        if not np.any(m):
            continue
        # dim underlying grayscale in those pixels
        rgb[m] = (rgb[m] * dim).astype(np.uint8)
        # add the class color
        rgb[m, 0] = r
        rgb[m, 1] = g
        rgb[m, 2] = b

    return rgb

def save_single_debug_slice(img_c: np.ndarray, mask_c: np.ndarray, out_png: Path, axis: int = 2):
    """
    Save ONE slice overlay (axial through tumor center) from the cropped 128^3 volume.
    """
    z = pick_center_slice_index(mask_c, axis=axis)

    if axis == 0:
        img_slice = img_c[z, :, :]
        mask_slice = mask_c[z, :, :]
    elif axis == 1:
        img_slice = img_c[:, z, :]
        mask_slice = mask_c[:, z, :]
    else:
        img_slice = img_c[:, :, z]
        mask_slice = mask_c[:, :, z]

    img_u8 = to_uint8_robust(img_slice)
    overlay = overlay_mask_rgb(img_u8, mask_slice)
    save_png(out_png, overlay)

# -----------------------------
# B) Per-case processing (T1 only)
# -----------------------------
def process_case(
    t1: np.ndarray,
    seg: np.ndarray,
    out_dir: Path,
    case_id: str,
    out_shape=(128, 128, 128),
):
    img_c, mask_c, sl, center = tumor_centered_crop(t1, seg, out_shape=out_shape)
    img_c = normalize_brats_volume(img_c)
    
    out_patient_dir = out_dir / f"patient_{case_id}"
    out_patient_dir.mkdir(parents=True, exist_ok=True)

    # Save 3D crops as .npy
    
    np.save(out_patient_dir / f"{case_id}_T1_crop.npy", img_c.astype(np.float32))
    np.save(out_patient_dir / f"{case_id}_seg_crop.npy", mask_c.astype(np.uint8))

    # Save ONE debug PNG: axial slice through tumor center
    debug_png = out_dir / "debug" / f"{case_id}_T1_centerSlice_overlay.png"
    save_single_debug_slice(img_c, mask_c, debug_png, axis=2)
    return {
        "case_id": case_id,
        "center": center.tolist(),
        "slices": [(s.start, s.stop) for s in sl],
        "crop_shape": list(img_c.shape),
        "tumor_voxels_in_crop": int((mask_c > 0).sum()),
        "debug_png": str(debug_png),
    }


def normalize_brats_volume(x: np.ndarray, mask_nonzero: np.ndarray | None = None,
                           p_low=1, p_high=99) -> np.ndarray:
    """
    Robust per-volume normalization for MRI.
    - Uses nonzero voxels by default (BraTS background is typically 0)
    - Clips to percentiles, then z-score
    - Keeps background at 0
    """
    x = x.astype(np.float32, copy=False)

    if mask_nonzero is None:
        mask_nonzero = x > 0

    vals = x[mask_nonzero]
    if vals.size < 10:
        return x  # nothing to normalize

    lo, hi = np.percentile(vals, [p_low, p_high])
    if hi <= lo:
        return x

    x_clip = np.clip(x, lo, hi)

    vals_clip = x_clip[mask_nonzero]
    mean = vals_clip.mean()
    std = vals_clip.std() + 1e-6

    x_norm = (x_clip - mean) / std

    # keep background exactly 0
    x_norm[~mask_nonzero] = 0.0
    return x_norm


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) != 3:
        print("Usage: python crop_t1.py <dataset_root> <output_folder>")
        sys.exit(1)

    dataset_root = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    print(f"Found {len(case_dirs)} case folders")

    logs = []
    failed = []

    for i, case_dir in enumerate(case_dirs, 1):
        print(f"a scris fisierul {i}...")
        case_id = case_dir.name
        try:
            t1_path = str(next(case_dir.glob("*t1*.nii*")))
            seg_path = str(next(case_dir.glob("*seg*.nii*")))

            
            t1 = np.asanyarray(nib.load(t1_path).dataobj).astype(np.float32)
            seg = np.asanyarray(nib.load(seg_path).dataobj).astype(np.uint8)

            info = process_case(t1, seg, out_dir=out_dir, 
                case_id=case_id, out_shape=(128, 128, 128))
            logs.append(info)

            if i % 25 == 0:
                print(f"[{i}/{len(case_dirs)}] processed... latest={case_id}")

        except StopIteration:
            failed.append((case_id, "Missing t1 or seg file (glob didn't match)"))
        except Exception as e:
            failed.append((case_id, repr(e)))

    with open(out_dir / "crop_log.json", "w") as f:
        json.dump({"processed": logs, "failed": failed}, f, indent=2)

    print(f"Done. OK={len(logs)}  Failed={len(failed)}")
    if failed:
        print("First few failures:", failed[:5])
# from __future__ import annotations
# from pathlib import Path
# import numpy as np
# import nibabel as nib


# # -----------------------------
# # 1) Tumor-centered fixed crop
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
# # 2) BraTS normalization
# # -----------------------------
# def normalize_brats_volume(x: np.ndarray,
#                            p_low=1,
#                            p_high=99) -> np.ndarray:
#     """
#     Robust per-volume normalization:
#     - Uses nonzero voxels (background assumed 0)
#     - Percentile clipping
#     - Z-score
#     """
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
# # 3) Per-case processing
# # -----------------------------
# def process_case(
#     t1: np.ndarray,
#     seg: np.ndarray,
#     out_dir: Path,
#     case_id: str,
#     out_shape=(128, 128, 128),
# ):
#     img_c, mask_c = tumor_centered_crop(t1, seg, out_shape=out_shape)

#     # Normalize AFTER cropping
#     img_c = normalize_brats_volume(img_c)

#     out_patient_dir = out_dir / f"patient_{case_id}"
#     out_patient_dir.mkdir(parents=True, exist_ok=True)

#     np.save(out_patient_dir / f"{case_id}_T1_crop.npy",
#             img_c.astype(np.float32))

#     np.save(out_patient_dir / f"{case_id}_seg_crop.npy",
#             mask_c.astype(np.uint8))


# # -----------------------------
# # 4) Batch runner
# # -----------------------------
# if __name__ == "__main__":
#     import sys

#     if len(sys.argv) != 3:
#         print("Usage: python crop_t1.py <dataset_root> <output_folder>")
#         sys.exit(1)

#     dataset_root = Path(sys.argv[1])
#     out_dir = Path(sys.argv[2])
#     out_dir.mkdir(parents=True, exist_ok=True)

#     case_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
#     print(f"Found {len(case_dirs)} case folders")

#     for i, case_dir in enumerate(case_dirs, 1):
#         case_id = case_dir.name

#         try:
#             t1_path = next(case_dir.glob("*t1*.nii*"))
#             seg_path = next(case_dir.glob("*seg*.nii*"))

#             t1 = np.asanyarray(nib.load(str(t1_path)).dataobj).astype(np.float32)
#             seg = np.asanyarray(nib.load(str(seg_path)).dataobj).astype(np.uint8)

#             process_case(
#                 t1,
#                 seg,
#                 out_dir=out_dir,
#                 case_id=case_id,
#                 out_shape=(128, 128, 128),
#             )

#             if i % 25 == 0:
#                 print(f"[{i}/{len(case_dirs)}] processed... latest={case_id}")

#         except Exception as e:
#             print(f"Failed {case_id}: {repr(e)}")

#     print("Done.")

# from __future__ import annotations
# from pathlib import Path
# import numpy as np
# import nibabel as nib


# # -----------------------------
# # 0) Debug PNG helpers
# # -----------------------------
# def _to_uint8_img(x: np.ndarray) -> np.ndarray:
#     """Robust convert float image to uint8 for PNG viewing."""
#     x = x.astype(np.float32, copy=False)
#     finite = np.isfinite(x)
#     if not np.any(finite):
#         return np.zeros_like(x, dtype=np.uint8)
#     x = x.copy()
#     x[~finite] = 0

#     # Use percentiles for contrast (works for z-scored images too)
#     lo, hi = np.percentile(x, [1, 99])
#     if hi <= lo:
#         return np.zeros_like(x, dtype=np.uint8)
#     x = np.clip(x, lo, hi)
#     x = (x - lo) / (hi - lo)
#     return (x * 255.0).round().astype(np.uint8)


# def _to_uint8_mask(m: np.ndarray) -> np.ndarray:
#     """Map labels to 0..255 for visibility."""
#     m = m.astype(np.uint8, copy=False)
#     # scale distinct labels a bit (0,1,2,3,4 etc)
#     return (m * 60).clip(0, 255).astype(np.uint8)


# def save_png(path: Path, arr2d_uint8: np.ndarray):
#     path.parent.mkdir(parents=True, exist_ok=True)
#     try:
#         import imageio.v2 as imageio
#         imageio.imwrite(str(path), arr2d_uint8)
#     except Exception:
#         import matplotlib.pyplot as plt
#         plt.imsave(str(path), arr2d_uint8, cmap="gray", vmin=0, vmax=255)


# def save_debug_slices(
#     debug_dir: Path,
#     case_id: str,
#     crop_name: str,
#     img_crop: np.ndarray,
#     seg_crop: np.ndarray,
# ):
#     """
#     Save ONLY the axial mid-slice (x,y) at z_mid for img + seg as PNG.
#     """
#     debug_case = debug_dir / f"patient_{case_id}"
#     debug_case.mkdir(parents=True, exist_ok=True)

#     img_u8 = _to_uint8_img(img_crop)
#     seg_u8 = _to_uint8_mask(seg_crop)

#     z_mid = img_u8.shape[2] // 2

#     save_png(
#         debug_case / f"{case_id}_{crop_name}_img_axial_z{z_mid}.png",
#         img_u8[:, :, z_mid]
#     )
#     save_png(
#         debug_case / f"{case_id}_{crop_name}_seg_axial_z{z_mid}.png",
#         seg_u8[:, :, z_mid]
#     )
# # -----------------------------
# # 1) Tumor-centered fixed crop
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
# # 1b) Random crops (not tumor-centered)
# # -----------------------------
# def make_random_slices(
#     rng: np.random.Generator,
#     in_shape: tuple[int, int, int],
#     out_shape=(128, 128, 128),
# ):
#     """
#     Uniform random crop location. If volume dim < crop dim, start=0 and crop_or_pad will pad.
#     """
#     in_shape = np.asarray(in_shape, dtype=int)
#     out_shape = np.asarray(out_shape, dtype=int)

#     starts = []
#     for dim_in, dim_out in zip(in_shape, out_shape):
#         max_start = int(max(0, dim_in - dim_out))
#         if max_start == 0:
#             starts.append(0)
#         else:
#             starts.append(int(rng.integers(0, max_start + 1)))
#     starts = np.asarray(starts, dtype=int)
#     ends = starts + out_shape

#     # Clamp to in_shape for slicing
#     ends = np.minimum(ends, in_shape)

#     return tuple(slice(int(s), int(e)) for s, e in zip(starts, ends))


# def random_crop_pair(
#     rng: np.random.Generator,
#     img: np.ndarray,
#     mask: np.ndarray,
#     out_shape=(128, 128, 128),
# ):
#     sl = make_random_slices(rng, img.shape, out_shape=out_shape)
#     img_c = crop_or_pad(img, sl, out_shape=out_shape)
#     mask_c = crop_or_pad(mask, sl, out_shape=out_shape)
#     return img_c, mask_c


# # -----------------------------
# # 2) BraTS normalization
# # -----------------------------
# def normalize_brats_volume(x: np.ndarray,
#                            p_low=1,
#                            p_high=99) -> np.ndarray:
#     """
#     Robust per-volume normalization:
#     - Uses nonzero voxels (background assumed 0)
#     - Percentile clipping
#     - Z-score
#     """
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
# # 3) Per-case processing
# # -----------------------------
# def process_case(
#     t1: np.ndarray,
#     seg: np.ndarray,
#     out_dir: Path,
#     case_id: str,
#     rng: np.random.Generator,
#     out_shape=(128, 128, 128),
#     num_random_crops: int = 4,
#     debug_dir: Path | None = None,
#     do_debug: bool = False,
# ):
#     out_patient_dir = out_dir / f"patient_{case_id}"
#     out_patient_dir.mkdir(parents=True, exist_ok=True)

#     # --- tumor-centered ---
#     img_c, mask_c = tumor_centered_crop(t1, seg, out_shape=out_shape)
#     img_c = normalize_brats_volume(img_c)

#     np.save(out_patient_dir / f"{case_id}_T1_crop.npy", img_c.astype(np.float32))
#     np.save(out_patient_dir / f"{case_id}_seg_crop.npy", mask_c.astype(np.uint8))

#     if do_debug and debug_dir is not None:
#         save_debug_slices(debug_dir, case_id, "tumor", img_c, mask_c)

#     # --- random crops ---
#     for k in range(1, num_random_crops + 1):
#         img_r, mask_r = random_crop_pair(rng, t1, seg, out_shape=out_shape)
#         img_r = normalize_brats_volume(img_r)

#         np.save(out_patient_dir / f"{case_id}_T1_rand{k}.npy", img_r.astype(np.float32))
#         np.save(out_patient_dir / f"{case_id}_seg_rand{k}.npy", mask_r.astype(np.uint8))

#         if do_debug and debug_dir is not None:
#             save_debug_slices(debug_dir, case_id, f"rand{k}", img_r, mask_r)


# # -----------------------------
# # 4) Batch runner
# # -----------------------------
# if __name__ == "__main__":
#     import sys

#     if len(sys.argv) != 3:
#         print("Usage: python crop_t1.py <dataset_root> <output_folder>")
#         sys.exit(1)

#     dataset_root = Path(sys.argv[1])
#     out_dir = Path(sys.argv[2])
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # Debug: saves PNGs only for first 2 patients
#     debug_dir = out_dir / "_debug_png"
#     debug_dir.mkdir(parents=True, exist_ok=True)

#     # Reproducible randomness (change seed if you want)
#     rng = np.random.default_rng(12345)

#     case_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
#     print(f"Found {len(case_dirs)} case folders")

#     for i, case_dir in enumerate(case_dirs, 1):
#         case_id = case_dir.name

#         try:
#             t1_path = next(case_dir.glob("*t1*.nii*"))
#             seg_path = next(case_dir.glob("*seg*.nii*"))

#             t1 = np.asanyarray(nib.load(str(t1_path)).dataobj).astype(np.float32)
#             seg = np.asanyarray(nib.load(str(seg_path)).dataobj).astype(np.uint8)

#             process_case(
#                 t1,
#                 seg,
#                 out_dir=out_dir,
#                 case_id=case_id,
#                 rng=rng,
#                 out_shape=(128, 128, 128),
#                 num_random_crops=4,
#                 debug_dir=debug_dir,
#                 do_debug=(i <= 2),
#             )

#             if i % 25 == 0:
#                 print(f"[{i}/{len(case_dirs)}] processed... latest={case_id}")

#         except Exception as e:
#             print(f"Failed {case_id}: {repr(e)}")

#     print("Done.")


from __future__ import annotations
from pathlib import Path
import numpy as np
import nibabel as nib


# -----------------------------
# 0) Debug PNG helpers
# -----------------------------
def _to_uint8_img(x: np.ndarray) -> np.ndarray:
    """Robust convert float image to uint8 for PNG viewing."""
    x = x.astype(np.float32, copy=False)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.uint8)
    x = x.copy()
    x[~finite] = 0

    # Use percentiles for contrast (works for z-scored images too)
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)
    return (x * 255.0).round().astype(np.uint8)


def _to_uint8_mask(m: np.ndarray) -> np.ndarray:
    """Map labels to 0..255 for visibility."""
    m = m.astype(np.uint8, copy=False)
    return (m * 60).clip(0, 255).astype(np.uint8)


def save_png(path: Path, arr2d_uint8: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        imageio.imwrite(str(path), arr2d_uint8)
    except Exception:
        import matplotlib.pyplot as plt
        plt.imsave(str(path), arr2d_uint8, cmap="gray", vmin=0, vmax=255)


def save_debug_slices(
    debug_dir: Path,
    case_id: str,
    crop_name: str,
    img_crop: np.ndarray,
    seg_crop: np.ndarray,
):
    """
    Save ONLY the axial mid-slice (x,y) at z_mid for img + seg as PNG.
    """
    debug_case = debug_dir / f"patient_{case_id}"
    debug_case.mkdir(parents=True, exist_ok=True)

    img_u8 = _to_uint8_img(img_crop)
    seg_u8 = _to_uint8_mask(seg_crop)

    z_mid = img_u8.shape[2] // 2

    save_png(
        debug_case / f"{case_id}_{crop_name}_img_axial_z{z_mid}.png",
        img_u8[:, :, z_mid]
    )
    save_png(
        debug_case / f"{case_id}_{crop_name}_seg_axial_z{z_mid}.png",
        seg_u8[:, :, z_mid]
    )


# -----------------------------
# 1) Crop helpers
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
    out_shape=(128, 128, 128)
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


def crop_or_pad(arr: np.ndarray, slices, out_shape=(128, 128, 128), pad_value=0):
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


def crop_from_center(
    img: np.ndarray,
    mask: np.ndarray,
    center: np.ndarray,
    out_shape=(128, 128, 128),
):
    sl = make_crop_slices(center, img.shape, out_shape=out_shape)
    img_c = crop_or_pad(img, sl, out_shape=out_shape)
    mask_c = crop_or_pad(mask, sl, out_shape=out_shape)
    return img_c, mask_c


def tumor_centered_crop(img: np.ndarray, mask: np.ndarray, out_shape=(128, 128, 128)):
    c = tumor_center(mask)
    return crop_from_center(img, mask, c, out_shape=out_shape)


# -----------------------------
# 1b) Crop strategies
# -----------------------------
def make_random_slices(
    rng: np.random.Generator,
    in_shape: tuple[int, int, int],
    out_shape=(128, 128, 128),
):
    """
    Uniform random crop location. If volume dim < crop dim, start=0 and crop_or_pad will pad.
    """
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
    ends = starts + out_shape
    ends = np.minimum(ends, in_shape)

    return tuple(slice(int(s), int(e)) for s, e in zip(starts, ends))


def random_crop_pair(
    rng: np.random.Generator,
    img: np.ndarray,
    mask: np.ndarray,
    out_shape=(128, 128, 128),
):
    sl = make_random_slices(rng, img.shape, out_shape=out_shape)
    img_c = crop_or_pad(img, sl, out_shape=out_shape)
    mask_c = crop_or_pad(mask, sl, out_shape=out_shape)
    return img_c, mask_c


def shifted_tumor_crop_pair(
    rng: np.random.Generator,
    img: np.ndarray,
    mask: np.ndarray,
    out_shape=(128, 128, 128),
    max_shift=(24, 24, 24),
    min_tumor_voxels: int = 50,
    max_tries: int = 50,
):
    """
    Crop near tumor but shifted, still keeping tumor overlap.
    """
    c = tumor_center(mask)
    bbox = tumor_bbox(mask)
    if bbox is None:
        return random_crop_pair(rng, img, mask, out_shape=out_shape)

    max_shift = np.asarray(max_shift, dtype=int)

    for _ in range(max_tries):
        shift = np.array([
            int(rng.integers(-max_shift[0], max_shift[0] + 1)),
            int(rng.integers(-max_shift[1], max_shift[1] + 1)),
            int(rng.integers(-max_shift[2], max_shift[2] + 1)),
        ], dtype=int)
        c_shift = c + shift
        img_c, mask_c = crop_from_center(img, mask, c_shift, out_shape=out_shape)
        if int((mask_c > 0).sum()) >= min_tumor_voxels:
            return img_c, mask_c

    return tumor_centered_crop(img, mask, out_shape=out_shape)


def boundary_crop_pair(
    rng: np.random.Generator,
    img: np.ndarray,
    mask: np.ndarray,
    out_shape=(128, 128, 128),
    min_tumor_voxels: int = 20,
    max_fraction: float = 0.25,
    max_tries: int = 80,
):
    """
    Crop that contains only part of tumor / tumor near edge of crop.
    """
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        return random_crop_pair(rng, img, mask, out_shape=out_shape)

    total_tumor = idx.shape[0]
    half = np.asarray(out_shape, dtype=int) // 2

    for _ in range(max_tries):
        voxel = idx[int(rng.integers(0, len(idx)))]
        axis = int(rng.integers(0, 3))
        sign = int(rng.choice([-1, 1]))

        center = voxel.copy()
        offset = half[axis] - int(rng.integers(8, 24))
        center[axis] = voxel[axis] + sign * offset

        img_c, mask_c = crop_from_center(img, mask, center, out_shape=out_shape)
        tumor_voxels = int((mask_c > 0).sum())

        if min_tumor_voxels <= tumor_voxels <= int(max_fraction * total_tumor):
            return img_c, mask_c

    return shifted_tumor_crop_pair(rng, img, mask, out_shape=out_shape)


# -----------------------------
# 2) BraTS normalization
# -----------------------------
def normalize_brats_volume(x: np.ndarray, p_low=1, p_high=99) -> np.ndarray:
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
    rng: np.random.Generator,
    out_shape=(128, 128, 128),
    debug_dir: Path | None = None,
    do_debug: bool = False,
):
    out_patient_dir = out_dir / f"patient_{case_id}"
    out_patient_dir.mkdir(parents=True, exist_ok=True)

    crop_specs = [
        ("center", lambda: tumor_centered_crop(t1, seg, out_shape=out_shape)),
        ("shift1", lambda: shifted_tumor_crop_pair(rng, t1, seg, out_shape=out_shape)),
        ("shift2", lambda: shifted_tumor_crop_pair(rng, t1, seg, out_shape=out_shape)),
        ("boundary", lambda: boundary_crop_pair(rng, t1, seg, out_shape=out_shape)),
        ("random", lambda: random_crop_pair(rng, t1, seg, out_shape=out_shape)),
    ]

    for crop_name, crop_fn in crop_specs:
        img_c, mask_c = crop_fn()
        img_c = normalize_brats_volume(img_c)

        np.save(out_patient_dir / f"{case_id}_T1_{crop_name}.npy", img_c.astype(np.float32))
        np.save(out_patient_dir / f"{case_id}_seg_{crop_name}.npy", mask_c.astype(np.uint8))

        if do_debug and debug_dir is not None:
            save_debug_slices(debug_dir, case_id, crop_name, img_c, mask_c)


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

    # Debug: saves PNGs only for first 1 patient
    debug_dir = out_dir / "_debug_png"
    debug_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(12345)

    case_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
    print(f"Found {len(case_dirs)} case folders")

    for i, case_dir in enumerate(case_dirs, 1):
        case_id = case_dir.name

        try:
            t1_path = next(case_dir.glob("*t1*.nii*"))
            seg_path = next(case_dir.glob("*seg*.nii*"))

            t1 = np.asanyarray(nib.load(str(t1_path)).dataobj).astype(np.float32)
            seg = np.asanyarray(nib.load(str(seg_path)).dataobj).astype(np.uint8)
            seg[seg == 4] = 3

            process_case(
                t1,
                seg,
                out_dir=out_dir,
                case_id=case_id,
                rng=rng,
                out_shape=(128, 128, 128),
                debug_dir=debug_dir,
                do_debug=(i == 1),
            )

            if i % 25 == 0:
                print(f"[{i}/{len(case_dirs)}] processed... latest={case_id}")

        except Exception as e:
            print(f"Failed {case_id}: {repr(e)}")

    print("Done.")
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation


def robust_normalize(x, p_low=1, p_high=99):
    x = x.astype(np.float32)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=np.float32)

    lo, hi = np.percentile(x[finite], [p_low, p_high])
    x = np.clip(x, lo, hi)
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def intensity_projection(vol3d, axis=2, mode="max"):
    if mode == "max":
        return np.max(vol3d, axis=axis)
    if mode == "mean":
        return np.mean(vol3d, axis=axis)
    raise ValueError("mode must be 'max' or 'mean'")


def label_depth_maps(seg3d, label, axis=2):
    """
    present2d: where label exists anywhere along projection axis
    depth2d: normalized depth of the closest occurrence along axis (0 closest, 1 farthest)
    """
    mask = (seg3d == label)
    present2d = np.any(mask, axis=axis)

    m = np.moveaxis(mask, axis, -1)  # (H,W,D)
    _, _, D = m.shape

    first_idx = np.argmax(m, axis=-1).astype(np.float32)  # 0 even if absent
    first_idx[~present2d] = np.nan

    if D > 1:
        depth2d = first_idx / (D - 1)
    else:
        depth2d = np.zeros_like(first_idx, dtype=np.float32)

    return present2d, depth2d


def mask_contour(mask2d, thickness=2):
    mask2d = mask2d.astype(bool)
    er = binary_erosion(mask2d)
    edge = mask2d & (~er)
    if thickness > 1:
        edge = binary_dilation(edge, iterations=thickness - 1)
    return edge


def alpha_over(bg_rgb, fg_rgb, fg_alpha):
    fg_alpha = fg_alpha[..., None]
    return fg_rgb * fg_alpha + bg_rgb * (1 - fg_alpha)


def auto_label_order_by_projected_area(seg3d, labels, axis=2):
    """
    Returns labels sorted by projected area (largest first).
    """
    areas = []
    for lab in labels:
        present2d = np.any(seg3d == lab, axis=axis)
        areas.append((lab, int(np.sum(present2d))))
    areas.sort(key=lambda x: x[1], reverse=True)
    return tuple(lab for lab, area in areas if area > 0)


def flatten_case_area_order(
    mri_nii_path,
    seg_nii_path,
    out_png_path="flattened.png",
    axis=2,
    mri_proj_mode="max",
    labels=(1, 2, 4),
    colors=None,
    base_alpha=0.45,
    contour_thickness=2,
    depth_shading=True,
    depth_gamma=1.5,
    tiny_area_boost=True,
    tiny_area_threshold=50,
    tiny_dilate_iters=2,
):
    """
    - Computes auto label order by projected area (largest -> smallest)
    - Overlays all labels so small/deep ones still show
    """
    if colors is None:
        colors = {
            2: (0.20, 0.80, 0.25),  # ED
            1: (1.00, 0.20, 0.20),  # NCR/NET
            4: (0.20, 0.35, 1.00),  # ET
        }

    mri = nib.load(str(mri_nii_path)).get_fdata()
    seg = nib.load(str(seg_nii_path)).get_fdata().astype(np.int16)

    # background
    bg2d = intensity_projection(mri, axis=axis, mode=mri_proj_mode)
    bg2d = robust_normalize(bg2d)
    comp = np.stack([bg2d, bg2d, bg2d], axis=-1)

    # automatic order: large -> small
    label_order = auto_label_order_by_projected_area(seg, labels, axis=axis)

    for lab in label_order:
        present2d, depth2d = label_depth_maps(seg, lab, axis=axis)
        if not np.any(present2d):
            continue

        vis_mask = present2d
        if tiny_area_boost and np.sum(vis_mask) < tiny_area_threshold:
            vis_mask = binary_dilation(vis_mask, iterations=tiny_dilate_iters)

        alpha_map = vis_mask.astype(np.float32) * base_alpha

        if depth_shading:
            strength = (1.0 - np.nan_to_num(depth2d, nan=1.0)) ** depth_gamma
            alpha_map *= (0.4 + 0.6 * strength)

        col = np.array(colors.get(lab, (1, 1, 0)), dtype=np.float32)

        fg_rgb = np.zeros_like(comp)
        fg_rgb[vis_mask] = col
        comp = alpha_over(comp, fg_rgb, alpha_map)

        edge = mask_contour(present2d, thickness=contour_thickness)
        if np.any(edge):
            edge_rgb = np.zeros_like(comp)
            edge_rgb[edge] = col
            comp = alpha_over(comp, edge_rgb, edge.astype(np.float32) * 0.95)

    plt.figure(figsize=(7, 7), dpi=180)
    plt.imshow(comp, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(str(out_png_path), bbox_inches="tight", pad_inches=0)
    plt.close()

    return out_png_path, label_order


def run_flatten_on_dataset(
    root_dir,
    subset="1",
    modality="flair",
    out_dir_name="flattened_2d",
    axis=2,
    labels=(1, 2, 4),
):
    root_dir = Path(root_dir)
    data_dir = root_dir / subset
    out_dir = root_dir / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("BraTS2021_")])
    if not case_dirs:
        raise FileNotFoundError(f"No case folders found in: {data_dir}")

    print(f"Found {len(case_dirs)} cases in {data_dir}")
    print(f"Saving PNGs to {out_dir}")

    ok, fail = 0, 0
    for case_dir in case_dirs:
        case_id = case_dir.name

        mri_path = case_dir / f"{case_id}_{modality}.nii.gz"
        seg_path = case_dir / f"{case_id}_seg.nii.gz"

        if not mri_path.exists():
            print(f"[SKIP] Missing MRI: {mri_path}")
            fail += 1
            continue
        if not seg_path.exists():
            print(f"[SKIP] Missing SEG: {seg_path}")
            fail += 1
            continue

        out_png = out_dir / f"{case_id}_{modality}_axis{axis}_flatten.png"

        try:
            _, order = flatten_case_area_order(
                mri_nii_path=mri_path,
                seg_nii_path=seg_path,
                out_png_path=out_png,
                axis=axis,
                mri_proj_mode="max",
                labels=labels,
            )
            print(f"[OK] {case_id} -> {out_png.name} | order={order}")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {case_id}: {e}")
            fail += 1

    print(f"Done. OK={ok}, FAIL={fail}")
    return str(out_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Flatten BraTS 3D volumes to 2D projections with tumor overlays.")
    p.add_argument("--root", type=str, default=".", help="Root directory (where subset folder like '1/' exists).")
    p.add_argument("--subset", type=str, default="1", help="Subset folder name (e.g., '1').")
    p.add_argument("--modality", type=str, default="flair", help="MRI modality: flair, t1ce, t1, t2.")
    p.add_argument("--out", type=str, default="flattened_2d", help="Output folder name under root.")
    p.add_argument("--axis", type=int, default=2, choices=[0, 1, 2], help="Projection axis: 0=sag,1=cor,2=axial.")
    p.add_argument("--labels", type=int, nargs="+", default=[1, 2, 4], help="Segmentation labels to overlay.")
    return p.parse_args()


def main():
    args = parse_args()
    run_flatten_on_dataset(
        root_dir=args.root,
        subset=args.subset,
        modality=args.modality,
        out_dir_name=args.out,
        axis=args.axis,
        labels=tuple(args.labels),
    )


if __name__ == "__main__":
    main()
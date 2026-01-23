#!/usr/bin/env python3
"""
Convert 128-channel LiDAR frames to 64-channel using a range-image (pseudo-ring) approach.

This script does NOT require a per-point ring column. Instead, it:
  - Projects points to a range image with V_src (default 128) vertical bins via elevation angle
    and H horizontal bins via azimuth angle.
  - Reduces the vertical resolution to V_tgt (default 64) by pairwise row reduction
    (default: choose the nearer point per (2 rows × each azimuth column)).
  - Reconstructs a downsampled point cloud from the selected cells.

Frames with point count <= threshold are treated as native 64-channel and are copied/symlinked.

Notes
-----
This is an approximation without ring. It preserves a 64-like vertical stratification but
cannot replicate the exact vendor-specific beam angles or scan ordering.

Usage example:
    python OpenPCDet/tools/convert_lidar_range_image.py \
        --dataset_root OpenPCDet/data/custom_av \
        --output_dir   OpenPCDet/data/custom_av_range64/points \
        --train_split  OpenPCDet/data/custom_av/ImageSets/train.txt \
        --val_split    OpenPCDet/data/custom_av/ImageSets/val.txt \
        --threshold 200000 --h_bins 2048 --v_src 128 --v_tgt 64
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


DEFAULT_POINT_THRESHOLD = 200_000


def load_split_ids(file_path: Path) -> list[str]:
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_points(point_dir: Path, sample_id: str) -> np.ndarray:
    pts = np.load(point_dir / f"{sample_id}.npy")
    if pts.ndim != 2 or pts.shape[1] < 4:
        raise ValueError(f"Expected (N,4+) array, got {pts.shape} for {sample_id}")
    return pts.astype(np.float32, copy=False)


def copy_or_symlink(src: Path, dst: Path, *, symlink: bool, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if overwrite:
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
        else:
            return
    if symlink:
        link_target = os.path.relpath(src.resolve(strict=True), start=dst.parent.resolve())
        os.symlink(link_target, dst)
    else:
        np.save(dst, np.load(src, mmap_mode="r"))


def spherical_angles(points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    xy = np.sqrt(x * x + y * y) + 1e-8
    # azimuth in [0, 2pi)
    az = np.arctan2(y, x)
    az = np.where(az < 0.0, az + 2.0 * math.pi, az)
    # elevation angle
    el = np.arctan2(z, xy)
    return az.astype(np.float32, copy=False), el.astype(np.float32, copy=False), r.astype(np.float32, copy=False)


def build_range_image(points: np.ndarray,
                      h_bins: int,
                      v_src: int,
                      el_range: Tuple[float, float] | None = None,
                      reduce: str = "nearest") -> tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Build a V_src × H range image and return (best_idx, best_r, el_edges, indices).

    - best_idx: (V_src, H) int64 indices into the original point array (or -1 if empty)
    - best_r:   (V_src, H) float32 nearest range for the selected point (inf if empty)
    - el_edges: vertical bin edges used to quantize elevation
    - indices:  (vi, hi) arrays of per-point indices for debugging/inspection
    """
    N = points.shape[0]
    az, el, r = spherical_angles(points[:, :3])

    # Horizontal bins (uniform in azimuth)
    hi = np.floor(az / (2.0 * math.pi) * h_bins).astype(np.int32)
    hi = np.clip(hi, 0, h_bins - 1)

    # Vertical range determination (percentiles to avoid extreme outliers)
    if el_range is None:
        el_min = float(np.percentile(el, 0.5))
        el_max = float(np.percentile(el, 99.5))
        if el_max <= el_min:  # degenerate, fall back to min/max
            el_min = float(el.min())
            el_max = float(el.max())
    else:
        el_min, el_max = float(el_range[0]), float(el_range[1])
    # Guard: ensure non-zero span
    if not math.isfinite(el_min) or not math.isfinite(el_max) or el_max - el_min < 1e-6:
        el_min = float(el.min())
        el_max = float(el.max())

    # Vertical bins (uniform in elevation angle, pseudo-ring)
    el_edges = np.linspace(el_min, el_max, v_src + 1, dtype=np.float32)
    vi = np.clip(np.digitize(el, el_edges) - 1, 0, v_src - 1).astype(np.int32)

    # Allocate buffers
    best_idx = np.full((v_src, h_bins), -1, dtype=np.int64)
    best_r = np.full((v_src, h_bins), np.inf, dtype=np.float32)

    # Populate nearest point per cell
    # Vectorized-by-chunk to limit memory pressure
    order = np.argsort(r)
    for idx in order:
        v = vi[idx]
        h = hi[idx]
        if r[idx] < best_r[v, h]:
            best_r[v, h] = r[idx]
            best_idx[v, h] = idx

    return best_idx, best_r, el_edges, (vi, hi)


def reduce_vertical(best_idx: np.ndarray, best_r: np.ndarray, v_tgt: int, method: str = "pair_nearest") -> np.ndarray:
    """Reduce V_src × H to V_tgt × H and return the chosen original indices (unique, 1D)."""
    v_src, h_bins = best_idx.shape
    if v_src % v_tgt != 0:
        # Default to pairwise downsample if non-multiple; take floor pairing.
        factor = v_src // v_tgt
        if factor < 2:
            factor = 2
    else:
        factor = v_src // v_tgt

    chosen = []
    for t in range(v_tgt):
        v0 = t * factor
        v1 = min((t + 1) * factor, v_src)
        # Slice source rows [v0:v1)
        idx_block = best_idx[v0:v1, :]
        r_block = best_r[v0:v1, :]
        if method == "pair_nearest" or method == "nearest":
            # Choose nearest among the group for each column
            argmin = np.argmin(r_block, axis=0)
            ri = r_block[argmin, np.arange(h_bins)]
            ii = idx_block[argmin, np.arange(h_bins)]
            mask = np.isfinite(ri) & (ii >= 0)
            if mask.any():
                chosen.append(ii[mask])
        elif method == "first_nonempty":
            # Pick first non-empty cell by row order
            valid = (idx_block >= 0)
            first_row = np.argmax(valid, axis=0)  # 0 where all False
            # However, where all False, keep -1
            cols = np.where(valid.any(axis=0))[0]
            if cols.size > 0:
                ii = idx_block[first_row[cols], cols]
                chosen.append(ii)
        else:
            raise ValueError(f"Unknown vertical reduction method: {method}")

    if not chosen:
        return np.empty((0,), dtype=np.int64)
    merged = np.unique(np.concatenate(chosen).astype(np.int64, copy=False))
    merged = merged[merged >= 0]
    return merged


def process_frame(points: np.ndarray,
                  h_bins: int,
                  v_src: int,
                  v_tgt: int,
                  el_range: Tuple[float, float] | None,
                  reduce_method: str) -> np.ndarray:
    best_idx, best_r, el_edges, (vi, hi) = build_range_image(points, h_bins, v_src, el_range=el_range)
    if reduce_method in ("row_even", "row_odd"):
        parity = 0 if reduce_method == "row_even" else 1
        mask = (vi % (v_src // v_tgt if v_src >= v_tgt and v_src % v_tgt == 0 else 2)) == parity
        # If factor is not clean, default to parity over 2
        chosen_idx = np.nonzero(mask)[0]
        if chosen_idx.size == 0:
            return points[:0]
        return points[chosen_idx]
    else:
        chosen_idx = reduce_vertical(best_idx, best_r, v_tgt, method=reduce_method)
        if chosen_idx.size == 0:
            return points[:0]
        return points[chosen_idx]


def convert_dataset(dataset_root: Path,
                    output_dir: Path,
                    train_split: Path,
                    val_split: Path,
                    threshold: int,
                    h_bins: int,
                    v_src: int,
                    v_tgt: int,
                    reduce_method: str,
                    overwrite: bool,
                    skip_existing: bool,
                    symlink_64: bool,
                    el_range: Tuple[float, float] | None,
                    max_frames: int | None) -> None:
    point_dir = dataset_root / "points"
    label_dir = dataset_root / "labels"  # unused; placeholder for future ROI-aware variant
    if not point_dir.exists():
        raise FileNotFoundError(f"Missing points dir: {point_dir}")

    train_ids = load_split_ids(train_split)
    val_ids = load_split_ids(val_split)
    all_ids = train_ids + val_ids

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    converted = 0
    copied = 0
    for sid in all_ids:
        if max_frames is not None and processed >= max_frames:
            break

        src = point_dir / f"{sid}.npy"
        dst = output_dir / f"{sid}.npy"
        if skip_existing and dst.exists():
            processed += 1
            continue

        pts = load_points(point_dir, sid)
        if pts.shape[0] <= threshold:
            copy_or_symlink(src, dst, symlink=symlink_64, overwrite=overwrite)
            copied += 1
            processed += 1
            continue

        # 128-like frame → range-image reduction
        down = process_frame(pts, h_bins=h_bins, v_src=v_src, v_tgt=v_tgt,
                             el_range=el_range, reduce_method=reduce_method)
        np.save(dst, down.astype(np.float32, copy=False))
        converted += 1
        processed += 1

    print(f"Done. Converted={converted}, Copied={copied}, Total processed={processed}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert 128→64 via range image (pseudo-ring)")
    ap.add_argument("--dataset_root", type=Path, required=True,
                    help="Dataset root containing points/ and labels/")
    ap.add_argument("--output_dir", type=Path, required=True,
                    help="Output directory for converted .npy points")
    ap.add_argument("--train_split", type=Path, required=True,
                    help="Path to train split txt")
    ap.add_argument("--val_split", type=Path, required=True,
                    help="Path to val split txt")
    ap.add_argument("--threshold", type=int, default=DEFAULT_POINT_THRESHOLD,
                    help="Point-count threshold: <= threshold treated as 64, > as 128")
    ap.add_argument("--h_bins", type=int, default=4096, help="Azimuth bins (columns)")
    ap.add_argument("--v_src", type=int, default=128, help="Source vertical bins (rows)")
    ap.add_argument("--v_tgt", type=int, default=64, help="Target vertical bins (rows)")
    ap.add_argument("--reduce", type=str, default="pair_nearest", choices=["pair_nearest", "nearest", "first_nonempty", "row_even", "row_odd"],
                    help="Row reduction method")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--skip_existing", action="store_true", help="Skip frames with existing outputs")
    ap.add_argument("--symlink_64", action="store_true", help="Symlink native-64 frames instead of copying them")
    ap.add_argument("--el_min_deg", type=float, default=None, help="Optional fixed elevation min (deg)")
    ap.add_argument("--el_max_deg", type=float, default=None, help="Optional fixed elevation max (deg)")
    ap.add_argument("--max_frames", type=int, default=None, help="Limit frames for this run")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    el_range = None
    if args.el_min_deg is not None and args.el_max_deg is not None:
        el_range = (math.radians(args.el_min_deg), math.radians(args.el_max_deg))

    convert_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        threshold=args.threshold,
        h_bins=args.h_bins,
        v_src=args.v_src,
        v_tgt=args.v_tgt,
        reduce_method=args.reduce,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        symlink_64=args.symlink_64,
        el_range=el_range,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()

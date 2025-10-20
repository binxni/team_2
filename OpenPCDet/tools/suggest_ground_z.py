#!/usr/bin/env python3
"""
Suggest a ground z-range from .npy point clouds by analyzing z distribution
near the sensor (within an XY radius). The script computes a z histogram to
find the dominant ground mode and estimates a tight band around it.

Example:
  python OpenPCDet/tools/suggest_ground_z.py \
      --root OpenPCDet/data/custom_av/points_test \
      --r-max 20 --sample 2000 --bins 400 --z-hist-range -5 5

Then use the suggested range with:
  --ground-z ZMIN ZMAX
in compare_intensity_groups.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def _iter_point_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.glob("*.npy")):
        if p.is_file():
            yield p


def suggest_ground_range(
    root: Path,
    r_max: float = 20.0,
    sample: int = 1000,
    bins: int = 400,
    z_hist_range: Tuple[float, float] = (-5.0, 5.0),
    window: float = 0.5,
) -> Tuple[float, float, dict]:
    lo, hi = z_hist_range
    edges = np.linspace(lo, hi, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = np.zeros(bins, dtype=np.int64)

    files = list(_iter_point_files(root))
    if len(files) == 0:
        raise FileNotFoundError(f"No .npy files found in {root}")
    if sample is not None and sample > 0:
        files = files[: min(sample, len(files))]

    # Pass 1: build z histogram within r <= r_max
    total_pts = 0
    for p in files:
        pts = np.load(p)
        if pts.ndim != 2 or pts.shape[1] < 4:
            continue
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        r = np.sqrt(x * x + y * y)
        m = r <= float(r_max)
        if not np.any(m):
            continue
        total_pts += int(np.count_nonzero(m))
        h, _ = np.histogram(z[m], bins=edges)
        counts += h.astype(np.int64, copy=False)

    if counts.sum() == 0:
        raise RuntimeError("No points within r_max; try increasing --r-max or adjusting z range.")

    # Mode (bin with max count)
    i_mode = int(np.argmax(counts))
    z_mode = float(centers[i_mode])

    # Pass 2 (on histogram): estimate dispersion around the mode within +/- window
    mask_window = (centers >= z_mode - window) & (centers <= z_mode + window)
    w = counts[mask_window].astype(float)
    c = centers[mask_window]
    if w.sum() == 0:
        # fallback to a fixed band around mode
        zmin, zmax = z_mode - 0.3, z_mode + 0.3
        stats = {"mode": z_mode, "mean": z_mode, "std": float('nan'), "points": total_pts}
        return zmin, zmax, stats

    mu = float(np.sum(w * c) / np.sum(w))
    var = float(np.sum(w * (c - mu) ** 2) / np.sum(w))
    sigma = float(np.sqrt(max(var, 1e-12)))

    # Recommend band: mean +/- max(2*sigma, 0.15), limited to +/- 0.6 m
    half = max(2.0 * sigma, 0.15)
    half = float(min(half, 0.6))
    zmin = mu - half
    zmax = mu + half

    stats = {
        "mode": z_mode,
        "mean": mu,
        "std": sigma,
        "points": total_pts,
        "half_width": half,
        "bins": bins,
        "r_max": r_max,
        "files_used": len(files),
    }
    return zmin, zmax, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Suggest ground z-range from nearby points")
    ap.add_argument("--root", type=Path, required=True, help="Directory with .npy point clouds")
    ap.add_argument("--r-max", type=float, default=20.0, help="Max XY radius (m) to consider")
    ap.add_argument("--sample", type=int, default=1000, help="Max number of files to scan (0=all)")
    ap.add_argument("--bins", type=int, default=400, help="Histogram bins for z")
    ap.add_argument("--z-hist-range", type=float, nargs=2, default=(-5.0, 5.0), metavar=("ZMIN", "ZMAX"))
    ap.add_argument("--window", type=float, default=0.5, help="Half window (m) around z-mode to estimate spread")
    args = ap.parse_args()

    zmin, zmax, stats = suggest_ground_range(
        root=args.root,
        r_max=args.r_max,
        sample=args.sample,
        bins=args.bins,
        z_hist_range=(args.z_hist_range[0], args.z_hist_range[1]),
        window=args.window,
    )

    print("Suggested ground z-range:")
    print(f"  --ground-z {zmin:.3f} {zmax:.3f}")
    print("Details:")
    print(f"  files_used: {stats['files_used']}")
    print(f"  points_in_radius: {stats['points']:,}")
    print(f"  r_max: {stats['r_max']} m, bins: {stats['bins']}")
    print(f"  mode_z: {stats['mode']:.3f}, mean_z: {stats['mean']:.3f}, std_z: {stats['std']:.3f}, half_width: {stats['half_width']:.3f}")


if __name__ == "__main__":
    main()


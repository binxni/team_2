#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np


def _iter_point_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.glob("*.npy")):
        if path.is_file():
            yield path


def _extract_numeric_id_from_stem(stem: str) -> Optional[int]:
    if stem.isdigit():
        return int(stem)
    m = re.findall(r"(\d+)", stem)
    if not m:
        return None
    try:
        return int(m[-1])
    except Exception:
        return None


def _id_in_ranges(nid: int, ranges: List[Tuple[int, int]]) -> bool:
    for lo, hi in ranges:
        if lo <= nid <= hi:
            return True
    return False


@dataclass
class HistResult:
    counts: np.ndarray
    edges: np.ndarray
    frames: int
    points: int
    per_distance_counts: Optional[List[np.ndarray]] = None
    per_distance_edges: Optional[np.ndarray] = None


def _compute_hist_for_files(
    files: Iterable[Path],
    bins: int,
    value_range: Tuple[float, float],
    per_distance_edges: Optional[List[float]] = None,
    z_range: Optional[Tuple[float, float]] = None,
    invert_z: bool = False,
    mask_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> HistResult:
    total_hist = np.zeros(bins, dtype=np.int64)
    frames = 0
    pts_total = 0
    lo, hi = value_range
    edges = np.linspace(lo, hi, bins + 1)

    per_dist_hists: Optional[List[np.ndarray]] = None
    if per_distance_edges is not None and len(per_distance_edges) >= 2:
        per_dist_hists = [np.zeros(bins, dtype=np.int64) for _ in range(len(per_distance_edges) - 1)]

    for p in files:
        pts = np.load(p)
        if pts.ndim != 2 or pts.shape[1] < 4:
            continue
        intens = pts[:, 3].astype(np.float32, copy=False)
        base_mask = None
        if z_range is not None:
            z = pts[:, 2]
            zmask = (z >= z_range[0]) & (z <= z_range[1])
            if invert_z:
                zmask = ~zmask
            base_mask = zmask if base_mask is None else (base_mask & zmask)
        if mask_func is not None:
            try:
                m = mask_func(pts)
            except Exception:
                m = None
            if m is not None:
                base_mask = m if base_mask is None else (base_mask & m)
        intens_used = intens if base_mask is None else intens[base_mask]
        h, _ = np.histogram(intens_used, bins=bins, range=value_range)
        total_hist += h.astype(np.int64, copy=False)
        frames += 1
        pts_total += int(intens_used.shape[0])

        if per_dist_hists is not None:
            r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
            for i, (r0, r1) in enumerate(zip(per_distance_edges[:-1], per_distance_edges[1:])):
                mask = (r >= r0) & (r < r1)
                if z_range is not None:
                    z = pts[:, 2]
                    zmask = (z >= z_range[0]) & (z <= z_range[1])
                    if invert_z:
                        zmask = ~zmask
                    mask = mask & zmask
                if base_mask is not None:
                    mask = mask & base_mask
                if not np.any(mask):
                    continue
                hh, _ = np.histogram(intens[mask], bins=bins, range=value_range)
                per_dist_hists[i] += hh.astype(np.int64, copy=False)

    return HistResult(
        counts=total_hist,
        edges=edges,
        frames=frames,
        points=pts_total,
        per_distance_counts=per_dist_hists,
        per_distance_edges=np.array(per_distance_edges, dtype=float) if per_dist_hists is not None else None,
    )


def _filter_files_by_groups(
    all_files: Iterable[Path],
    red_ranges: List[Tuple[int, int]],
    blue_ranges: List[Tuple[int, int]],
) -> Tuple[List[Path], List[Path]]:
    red_files: List[Path] = []
    blue_files: List[Path] = []
    for p in all_files:
        nid = _extract_numeric_id_from_stem(p.stem)
        if nid is None:
            continue
        if red_ranges and _id_in_ranges(nid, red_ranges):
            red_files.append(p)
        elif blue_ranges and _id_in_ranges(nid, blue_ranges):
            blue_files.append(p)
    return red_files, blue_files


def _compute_per_meter_hist_and_zeros(
    files: Iterable[Path],
    meter_min: int,
    meter_max: int,
    bins: int,
    value_range: Tuple[float, float],
    z_range: Optional[Tuple[float, float]] = None,
    invert_z: bool = False,
    mask_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Aggregate per-1m histograms over intensity and zero counts.

    Returns (meters, hist_per_meter, total_counts_per_meter, zero_counts_per_meter, frames)
    - meters: array of meter centers [meter_min, ..., meter_max]
    - hist_per_meter: shape (M, bins)
    - total_counts_per_meter: shape (M,)
    - zero_counts_per_meter: shape (M,)
    - frames: number of frames processed
    """
    if meter_max < meter_min:
        meter_min, meter_max = meter_max, meter_min
    # Edges for 1m bins: [meter_min, meter_min+1, ..., meter_max+1]
    r_edges = np.arange(meter_min, meter_max + 2, dtype=float)
    M = meter_max - meter_min + 1
    hist_pm = np.zeros((M, bins), dtype=np.int64)
    total_pm = np.zeros(M, dtype=np.int64)
    zero_pm = np.zeros(M, dtype=np.int64)
    frames = 0

    lo, hi = value_range
    i_edges = np.linspace(lo, hi, bins + 1)

    for p in files:
        pts = np.load(p)
        if pts.ndim != 2 or pts.shape[1] < 4:
            continue
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        intens = pts[:, 3].astype(np.float32, copy=False)
        r = np.sqrt(x * x + y * y)
        mask = (r >= meter_min) & (r < meter_max + 1)
        if z_range is not None:
            zmask = (z >= z_range[0]) & (z <= z_range[1])
            if invert_z:
                zmask = ~zmask
            mask = mask & zmask
        if mask_func is not None:
            try:
                m0 = mask_func(pts)
                if m0 is not None:
                    mask = mask & m0
            except Exception:
                pass
        if not np.any(mask):
            frames += 1
            continue
        r_sel = r[mask]
        i_sel = intens[mask]
        # 2D histogram over (r, intensity)
        h2d, _, _ = np.histogram2d(r_sel, i_sel, bins=(r_edges, i_edges))
        hist_pm += h2d.astype(np.int64, copy=False)
        # totals per meter bin (sum over intensity axis)
        t_local = np.histogram(r_sel, bins=r_edges)[0]
        total_pm += t_local.astype(np.int64, copy=False)
        # zeros per meter
        zmask0 = (i_sel == 0.0)
        if np.any(zmask0):
            z_local = np.histogram(r_sel[zmask0], bins=r_edges)[0]
            zero_pm += z_local.astype(np.int64, copy=False)
        frames += 1

    meters = (r_edges[:-1] + r_edges[1:]) / 2.0
    return meters, hist_pm, total_pm, zero_pm, frames


def _compute_per_meter_frame_mean(
    files: Iterable[Path],
    meter_min: int,
    meter_max: int,
    z_range: Optional[Tuple[float, float]] = None,
    invert_z: bool = False,
    mask_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Compute per-frame means per meter, then average across frames (frame-mean reduce)."""
    if meter_max < meter_min:
        meter_min, meter_max = meter_max, meter_min
    size = meter_max + 1
    sum_means = np.zeros(size, dtype=np.float64)
    frame_counts = np.zeros(size, dtype=np.int64)
    frames = 0
    pts_used = 0

    for p in files:
        pts = np.load(p)
        if pts.ndim != 2 or pts.shape[1] < 4:
            continue
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        intens = pts[:, 3].astype(np.float32, copy=False)
        r = np.sqrt(x * x + y * y)
        bins_r = np.floor(r).astype(np.int32)
        mask = (bins_r >= meter_min) & (bins_r <= meter_max)
        if z_range is not None:
            zmask = (z >= z_range[0]) & (z <= z_range[1])
            if invert_z:
                zmask = ~zmask
            mask = mask & zmask
        if mask_func is not None:
            try:
                m0 = mask_func(pts)
                if m0 is not None:
                    mask = mask & m0
            except Exception:
                pass
        if not np.any(mask):
            frames += 1
            continue
        bb = bins_r[mask]
        ii = intens[mask]
        sums_local = np.bincount(bb, weights=ii, minlength=size)
        counts_local = np.bincount(bb, minlength=size)
        with np.errstate(invalid='ignore', divide='ignore'):
            means_local = np.divide(sums_local, counts_local, out=np.zeros_like(sums_local, dtype=float), where=counts_local > 0)
        sel = counts_local > 0
        sum_means[sel] += means_local[sel]
        frame_counts[sel] += 1
        frames += 1
        pts_used += int(mask.sum())

    with np.errstate(invalid='ignore', divide='ignore'):
        means = np.divide(sum_means, frame_counts, out=np.full_like(sum_means, np.nan, dtype=float), where=frame_counts > 0)
    meters = np.arange(size)
    meters = meters[meter_min:meter_max + 1]
    means = means[meter_min:meter_max + 1]
    return meters, means, frames, pts_used


def _smooth_series_ignore_nan(y: np.ndarray, window: int) -> np.ndarray:
    if y is None:
        return y
    L = int(y.size)
    if L <= 1 or window is None or window <= 1:
        return y
    w = int(window)
    if w > L:
        w = L
    k = np.ones(w, dtype=float)
    mask = np.isfinite(y)
    y0 = np.where(mask, y, 0.0)
    num = np.convolve(y0, k, mode='same')
    den = np.convolve(mask.astype(float), k, mode='same')
    # Ensure output length matches input length (numpy 'same' returns max(L, w))
    if num.size != L:
        start = (num.size - L) // 2
        end = start + L
        num = num[start:end]
        den = den[start:end]
    out = np.divide(num, den, out=np.full(L, np.nan, dtype=float), where=den > 0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay intensity histograms for two ID-range groups (red vs blue)")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("OpenPCDet/data/custom_av/points_test"),
        help="Directory containing .npy point cloud files.",
    )
    parser.add_argument(
        "--root-red",
        type=Path,
        default=None,
        help="Optional alternate root directory for RED(Noise) group.",
    )
    parser.add_argument(
        "--root-blue",
        type=Path,
        default=None,
        help="Optional alternate root directory for BLUE(Clean) group.",
    )
    parser.add_argument("--bins", type=int, default=256, help="Number of histogram bins.")
    parser.add_argument(
        "--range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.0, 1.0),
        help="Intensity range for histogram (e.g., 0 1 or 0 255).",
    )
    parser.add_argument(
        "--red-range",
        type=int,
        nargs=2,
        metavar=("START_ID", "END_ID"),
        action="append",
        default=None,
        help="Inclusive ID ranges for the RED group; can repeat.",
    )
    parser.add_argument(
        "--blue-range",
        type=int,
        nargs=2,
        metavar=("START_ID", "END_ID"),
        action="append",
        default=None,
        help="Inclusive ID ranges for the BLUE group; can repeat.",
    )
    parser.add_argument("--output", type=Path, default=Path("OpenPCDet/tools/intensity_compare_hist.png"), help="Output PNG path.")
    parser.add_argument("--normalize", action="store_true", help="Plot normalized counts (sum=1) for shape comparison.")
    parser.add_argument(
        "--no-hist",
        action="store_true",
        help="Disable all histogram-style plots (overall and per-distance, including ground/RANSAC variants).",
    )
    parser.add_argument(
        "--per-distance",
        type=float,
        nargs='+',
        default=None,
        help="Optional distance bucket edges in meters, e.g., 0 30 50 70 100.",
    )
    parser.add_argument("--per-meter", action="store_true", help="Compute 1m-bin intensity stats vs distance and save plot(s).")
    parser.add_argument("--meter-min", type=int, default=0, help="Minimum distance (inclusive) for per-meter plots.")
    parser.add_argument("--meter-max", type=int, default=120, help="Maximum distance (inclusive) for per-meter plots.")
    parser.add_argument(
        "--per-meter-stat",
        type=str,
        nargs='+',
        default=["mean"],
        choices=["mean", "median", "p10", "p90"],
        help="Per-meter statistic(s) to plot. Multiple allowed.",
    )
    parser.add_argument(
        "--per-meter-reduce",
        type=str,
        choices=["point", "frame-mean"],
        default="point",
        help="Reduction for 'mean': point-weighted over all points, or frame-wise mean averaged across frames.",
    )
    parser.add_argument(
        "--per-meter-zero-fraction",
        action="store_true",
        help="Also plot zero-intensity fraction vs distance (per 1m bin).",
    )
    parser.add_argument(
        "--only-ransac-nonground-1m-mean",
        action="store_true",
        help="Output only a single plot: per-1m mean intensity vs distance for non-ground (via RANSAC), comparing RED vs BLUE. Skips all other outputs.",
    )
    parser.add_argument("--smooth-window", type=int, default=5, help="Moving average window for mean-vs-distance plots (<=1 disables).")
    # Ground splitting options
    parser.add_argument(
        "--ground-z",
        type=float,
        nargs=2,
        metavar=("ZMIN", "ZMAX"),
        default=None,
        help="If set, also generate plots for ground-only (ZMIN<=z<=ZMAX) and non-ground (outside).",
    )
    parser.add_argument("--ransac-ground", action="store_true", help="Also split using RANSAC plane (ground vs non-ground) and plot.")
    parser.add_argument("--rg-seed-radius", type=float, default=20.0, help="RANSAC: XY radius (m) for seed points used to fit plane.")
    parser.add_argument("--rg-max-iters", type=int, default=200, help="RANSAC: max iterations.")
    parser.add_argument("--rg-dist-thresh", type=float, default=0.2, help="RANSAC: inlier distance threshold (m).")
    parser.add_argument("--rg-slope-max-deg", type=float, default=20.0, help="RANSAC: max allowed slope angle from +Z (deg).")
    parser.add_argument("--rg-sample-limit", type=int, default=50000, help="RANSAC: max seed points sampled per frame.")

    args = parser.parse_args()

    # Resolve roots per group
    root_red = args.root_red if args.root_red is not None else args.root
    root_blue = args.root_blue if args.root_blue is not None else args.root
    if not root_red.exists():
        raise FileNotFoundError(f"RED group directory {root_red} does not exist.")
    if not root_blue.exists():
        raise FileNotFoundError(f"BLUE group directory {root_blue} does not exist.")

    # Normalize ranges
    red_ranges: List[Tuple[int, int]] = []
    blue_ranges: List[Tuple[int, int]] = []
    if args.red_range:
        for a, b in args.red_range:
            lo, hi = (a, b) if a <= b else (b, a)
            red_ranges.append((lo, hi))
    if args.blue_range:
        for a, b in args.blue_range:
            lo, hi = (a, b) if a <= b else (b, a)
            blue_ranges.append((lo, hi))

    # Collect files per group. If no ranges are given for a group, include all files from its root.
    red_files_all = list(_iter_point_files(root_red))
    blue_files_all = list(_iter_point_files(root_blue))
    if red_ranges:
        red_files = [p for p in red_files_all if (nid:=_extract_numeric_id_from_stem(p.stem)) is not None and _id_in_ranges(nid, red_ranges)]
    else:
        red_files = red_files_all
    if blue_ranges:
        blue_files = [p for p in blue_files_all if (nid:=_extract_numeric_id_from_stem(p.stem)) is not None and _id_in_ranges(nid, blue_ranges)]
    else:
        blue_files = blue_files_all

    per_distance_edges = None
    if args.per_distance is not None:
        if len(args.per_distance) < 2:
            raise ValueError("--per-distance requires at least two edge values (e.g., 0 30 60 90)")
        per_distance_edges = sorted(args.per_distance)

    # Fast-path: only RANSAC non-ground per-1m mean plot
    if args.only_ransac_nonground_1m_mean:
        if not args.ransac_ground:
            raise ValueError("--only-ransac-nonground-1m-mean requires --ransac-ground")

        # RANSAC mask definition (same as used later)
        import math
        def make_ransac_mask(select_ground: bool):
            def mask_func(pts: np.ndarray) -> np.ndarray:
                x = pts[:, 0]; y = pts[:, 1]; z = pts[:, 2]
                r = np.sqrt(x * x + y * y)
                seed = r <= float(args.rg_seed_radius)
                idx = np.nonzero(seed)[0]
                if idx.size == 0:
                    return np.zeros(pts.shape[0], dtype=bool) if select_ground else np.ones(pts.shape[0], dtype=bool)
                if idx.size > args.rg_sample_limit > 0:
                    idx = np.random.choice(idx, size=args.rg_sample_limit, replace=False)
                Xs = x[idx]; Ys = y[idx]; Zs = z[idx]
                best_inliers = None; best_count = -1
                for _ in range(int(args.rg_max_iters)):
                    if idx.size < 3: break
                    j = np.random.choice(idx, size=3, replace=False)
                    p1 = np.array([x[j[0]], y[j[0]], z[j[0]]], dtype=float)
                    p2 = np.array([x[j[1]], y[j[1]], z[j[1]]], dtype=float)
                    p3 = np.array([x[j[2]], y[j[2]], z[j[2]]], dtype=float)
                    v1 = p2 - p1; v2 = p3 - p1
                    n = np.cross(v1, v2); norm = np.linalg.norm(n)
                    if not np.isfinite(norm) or norm < 1e-6: continue
                    n = n / norm
                    if n[2] < 0: n = -n
                    angle_deg = math.degrees(math.acos(max(min(n[2], 1.0), -1.0)))
                    if angle_deg > float(args.rg_slope_max_deg): continue
                    d = -np.dot(n, p1)
                    dist = np.abs(n[0]*Xs + n[1]*Ys + n[2]*Zs + d)
                    inl = dist <= float(args.rg_dist_thresh)
                    cnt = int(np.count_nonzero(inl))
                    if cnt > best_count:
                        best_count = cnt; best_inliers = inl
                if best_inliers is None or best_count < 10:
                    zmed = float(np.median(Zs)) if Zs.size > 0 else 0.0
                    n = np.array([0.0, 0.0, 1.0], dtype=float); d = -zmed
                else:
                    sel = np.nonzero(best_inliers)[0]
                    Xi = Xs[sel]; Yi = Ys[sel]; Zi = Zs[sel]
                    A = np.stack([Xi, Yi, np.ones_like(Xi)], axis=1)
                    try:
                        a, b, c = np.linalg.lstsq(A, Zi, rcond=None)[0]
                        n = np.array([-a, -b, 1.0], dtype=float); n = n / np.linalg.norm(n)
                        if n[2] < 0: n = -n
                        d = -c
                    except Exception:
                        zmed = float(np.median(Zs)) if Zs.size > 0 else 0.0
                        n = np.array([0.0, 0.0, 1.0], dtype=float); d = -zmed
                dist_all = np.abs(n[0]*x + n[1]*y + n[2]*z + d)
                ground = dist_all <= float(args.rg_dist_thresh)
                return ground if select_ground else ~ground
            return mask_func

        # Compute non-ground per-1m mean for RED/BLUE
        if args.per_meter_reduce == "frame-mean":
            mx_r, my_r, _, _ = _compute_per_meter_frame_mean(red_files, args.meter_min, args.meter_max, mask_func=make_ransac_mask(False))
            mx_b, my_b, _, _ = _compute_per_meter_frame_mean(blue_files, args.meter_min, args.meter_max, mask_func=make_ransac_mask(False))
            if not np.array_equal(mx_r, mx_b):
                m_x = np.intersect1d(mx_r, mx_b)
                my_r = my_r[np.isin(mx_r, m_x)]; my_b = my_b[np.isin(mx_b, m_x)]
            else:
                m_x = mx_r
            y_r = my_r; y_b = my_b
            suffix = "_1m_mean_framemean_nonground_ransac"
            title = "Per-1m mean intensity (non-ground, RANSAC, frame-mean)"
        else:
            lohi = (args.range[0], args.range[1])
            mx_r, h_r, _, _, _ = _compute_per_meter_hist_and_zeros(red_files, args.meter_min, args.meter_max, args.bins, lohi, mask_func=make_ransac_mask(False))
            mx_b, h_b, _, _, _ = _compute_per_meter_hist_and_zeros(blue_files, args.meter_min, args.meter_max, args.bins, lohi, mask_func=make_ransac_mask(False))
            if not np.array_equal(mx_r, mx_b):
                m_x = np.intersect1d(mx_r, mx_b)
                h_r = h_r[np.isin(mx_r, m_x), :]
                h_b = h_b[np.isin(mx_b, m_x), :]
            else:
                m_x = mx_r
            edges_i = np.linspace(lohi[0], lohi[1], args.bins + 1)
            centers_i = (edges_i[:-1] + edges_i[1:]) / 2.0
            den_r = np.maximum(h_r.sum(axis=1), 1)
            den_b = np.maximum(h_b.sum(axis=1), 1)
            y_r = (h_r @ centers_i) / den_r
            y_b = (h_b @ centers_i) / den_b
            suffix = "_1m_mean_nonground_ransac"
            title = "Per-1m mean intensity (non-ground, RANSAC)"

        # Plot single figure
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise RuntimeError("matplotlib is required to plot per-meter mean graph.") from e
        yr_s = _smooth_series_ignore_nan(y_r, args.smooth_window)
        yb_s = _smooth_series_ignore_nan(y_b, args.smooth_window)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(m_x, yr_s, '-', color='red', linewidth=1.6, label='RED (non-ground)')
        ax.plot(m_x, yb_s, '-', color='blue', linewidth=1.6, label='BLUE (non-ground)')
        ax.set_xlabel('distance (m)'); ax.set_ylabel('mean intensity'); ax.set_title(title)
        ax.grid(True, alpha=0.3, linestyle='--'); ax.legend(); fig.tight_layout()
        out = args.output.with_name(args.output.stem + suffix + args.output.suffix)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"Saved ONLY non-ground RANSAC per-1m mean plot to {out}")
        return

    red_res = _compute_hist_for_files(
        red_files, args.bins, (args.range[0], args.range[1]), per_distance_edges=per_distance_edges
    )
    blue_res = _compute_hist_for_files(
        blue_files, args.bins, (args.range[0], args.range[1]), per_distance_edges=per_distance_edges
    )

    if red_res.frames == 0 and blue_res.frames == 0:
        raise RuntimeError("No files matched the given ranges.")

    # Prepare plot data
    edges = red_res.edges  # same binning for both
    red_y = red_res.counts.astype(float)
    blue_y = blue_res.counts.astype(float)
    if args.normalize:
        red_sum = red_y.sum()
        blue_sum = blue_y.sum()
        if red_sum > 0:
            red_y /= red_sum
        if blue_sum > 0:
            blue_y /= blue_sum

    # Plot overall histogram unless disabled
    if not args.no_hist:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise RuntimeError("matplotlib is required to plot the histogram.") from e
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.step(edges[:-1], red_y, where='post', color='red', label="RED", alpha=0.9)
        ax.step(edges[:-1], blue_y, where='post', color='blue', label="BLUE", alpha=0.9)
        ax.set_xlabel("intensity")
        ax.set_ylabel("normalized count" if args.normalize else "count")
        ax.set_title("Intensity histogram: RED vs BLUE groups")
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend()
        fig.tight_layout()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        plt.close(fig)

        print(f"Saved plot to {args.output}")
    print(f"RED  : files={len(red_files)}, frames={red_res.frames}, points={red_res.points:,}")
    print(f"BLUE : files={len(blue_files)}, frames={blue_res.frames}, points={blue_res.points:,}")

    # Per-distance plotting, if requested
    if per_distance_edges is not None and red_res.per_distance_counts is not None and blue_res.per_distance_counts is not None:
        # Use the minimum available bucket count among edges/red/blue to avoid mismatch
        nb_edges = len(per_distance_edges) - 1
        nb_red = len(red_res.per_distance_counts)
        nb_blue = len(blue_res.per_distance_counts)
        nb = min(nb_edges, nb_red, nb_blue)
        # Determine grid for subplots
        import math
        cols = 3 if nb >= 3 else nb
        rows = math.ceil(nb / cols) if cols > 0 else 1

        # Ensure plotting available for any per-distance figures
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise RuntimeError("matplotlib is required to plot per-distance graphs.") from e

        # Prepare figure for per-bucket hist overlays (unless disabled)
        if not args.no_hist:
            fig2, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
            for i in range(nb):
                r0, r1 = per_distance_edges[i], per_distance_edges[i + 1]
                ax2 = axes[i // cols][i % cols]
                ry = red_res.per_distance_counts[i].astype(float)
                by = blue_res.per_distance_counts[i].astype(float)
                if args.normalize:
                    rsum = ry.sum()
                    bsum = by.sum()
                    if rsum > 0:
                        ry /= rsum
                    if bsum > 0:
                        by /= bsum
                ax2.step(edges[:-1], ry, where='post', color='red', alpha=0.9)
                ax2.step(edges[:-1], by, where='post', color='blue', alpha=0.9)
                ax2.set_title(f"{int(r0)}–{int(r1)} m")
                ax2.grid(True, alpha=0.3, linestyle='--')
            for j in range(nb, rows * cols):
                fig2.delaxes(axes[j // cols][j % cols])
            fig2.suptitle("Per-distance intensity histograms (RED vs BLUE)")
            fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
            out2 = args.output.with_name(args.output.stem + "_per_distance" + args.output.suffix)
            fig2.savefig(out2, dpi=150)
            plt.close(fig2)
            print(f"Saved per-distance plot to {out2}")

        # Mean intensity vs distance summary
        # Compute mean from histograms using bin centers
        centers = (edges[:-1] + edges[1:]) / 2.0
        def mean_from_hist(h: np.ndarray) -> float:
            s = h.sum()
            return float(np.average(centers, weights=h)) if s > 0 else float('nan')

        red_means = [mean_from_hist(red_res.per_distance_counts[i]) for i in range(nb)]
        blue_means = [mean_from_hist(blue_res.per_distance_counts[i]) for i in range(nb)]
        x = [0.5 * (per_distance_edges[i] + per_distance_edges[i + 1]) for i in range(nb)]

        fig3, ax3 = plt.subplots(figsize=(7, 4))
        # Optional smoothing and no markers
        red_means_s = _smooth_series_ignore_nan(np.asarray(red_means, dtype=float), args.smooth_window)
        blue_means_s = _smooth_series_ignore_nan(np.asarray(blue_means, dtype=float), args.smooth_window)
        ax3.plot(x, red_means_s, '-', color='red', label='RED mean intensity', linewidth=1.6)
        ax3.plot(x, blue_means_s, '-', color='blue', label='BLUE mean intensity', linewidth=1.6)
        ax3.set_xlabel("distance (m)")
        ax3.set_ylabel("mean intensity")
        ax3.set_title("Mean intensity vs distance")
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.legend()
        fig3.tight_layout()
        out3 = args.output.with_name(args.output.stem + "_mean_vs_distance" + args.output.suffix)
        fig3.savefig(out3, dpi=150)
        plt.close(fig3)
        print(f"Saved mean-vs-distance plot to {out3}")

    # Per-1m intensity stats vs distance plot(s)
    if args.per_meter:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise RuntimeError("matplotlib is required to plot per-meter graphs.") from e

        # Frame-mean reduce (only for mean)
        if "mean" in args.per_meter_stat and args.per_meter_reduce == "frame-mean":
            mrx, mry, fr_r, _ = _compute_per_meter_frame_mean(red_files, args.meter_min, args.meter_max)
            mbx, mby, fr_b, _ = _compute_per_meter_frame_mean(blue_files, args.meter_min, args.meter_max)
            # Align x
            if not np.array_equal(mrx, mbx):
                m_x = np.intersect1d(mrx, mbx)
                mry = mry[np.isin(mrx, m_x)]
                mby = mby[np.isin(mbx, m_x)]
            else:
                m_x = mrx
            # Smooth
            ry_s = _smooth_series_ignore_nan(mry, args.smooth_window)
            by_s = _smooth_series_ignore_nan(mby, args.smooth_window)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(m_x, ry_s, '-', color='red', linewidth=1.6, label='RED (frame-mean)')
            ax.plot(m_x, by_s, '-', color='blue', linewidth=1.6, label='BLUE (frame-mean)')
            ax.set_xlabel('distance (m)')
            ax.set_ylabel('mean intensity')
            ax.set_title('Per-1m mean intensity (frame-mean reduce)')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend()
            fig.tight_layout()
            out = args.output.with_name(args.output.stem + "_1m_mean_framemean" + args.output.suffix)
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"Saved per-meter frame-mean plot to {out}")

        # Histogram-based stats (mean-point, median, p10, p90) and zero fraction
        need_hist = (args.per_meter_reduce == "point" and "mean" in args.per_meter_stat) or any(
            s in args.per_meter_stat for s in ("median", "p10", "p90")
        ) or args.per_meter_zero_fraction

        if need_hist:
            lohi = (args.range[0], args.range[1])
            mrx, h_r, tot_r, zer_r, _ = _compute_per_meter_hist_and_zeros(
                red_files, args.meter_min, args.meter_max, args.bins, lohi
            )
            mbx, h_b, tot_b, zer_b, _ = _compute_per_meter_hist_and_zeros(
                blue_files, args.meter_min, args.meter_max, args.bins, lohi
            )
            # Align x
            if not np.array_equal(mrx, mbx):
                m_x = np.intersect1d(mrx, mbx)
                sel_r = np.isin(mrx, m_x)
                sel_b = np.isin(mbx, m_x)
                h_r = h_r[sel_r, :]
                h_b = h_b[sel_b, :]
                tot_r = tot_r[sel_r]
                tot_b = tot_b[sel_b]
                zer_r = zer_r[sel_r]
                zer_b = zer_b[sel_b]
            else:
                m_x = mrx

            # Prepare intensity bin centers
            edges_i = np.linspace(lohi[0], lohi[1], args.bins + 1)
            centers_i = (edges_i[:-1] + edges_i[1:]) / 2.0

            def stat_from_hist(H: np.ndarray, stat: str) -> np.ndarray:
                # H: (M, B)
                if stat == "mean":
                    num = H @ centers_i
                    den = H.sum(axis=1)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        return np.divide(num, den, out=np.full(H.shape[0], np.nan), where=den > 0)
                # quantiles
                qmap = {"median": 0.5, "p10": 0.10, "p90": 0.90}
                q = qmap[stat]
                M = H.shape[0]
                out = np.full(M, np.nan)
                for i in range(M):
                    h = H[i]
                    c = h.cumsum()
                    tot = int(c[-1]) if h.size > 0 else 0
                    if tot <= 0:
                        continue
                    idx = int(np.searchsorted(c, q * tot, side='left'))
                    if idx >= h.size:
                        idx = h.size - 1
                    out[i] = centers_i[idx]
                return out

            # For each requested stat (hist-based)
            for stat_name in args.per_meter_stat:
                if stat_name == "mean" and args.per_meter_reduce != "point":
                    continue  # handled above
                y_r = stat_from_hist(h_r, stat_name)
                y_b = stat_from_hist(h_b, stat_name)
                y_r_s = _smooth_series_ignore_nan(y_r, args.smooth_window)
                y_b_s = _smooth_series_ignore_nan(y_b, args.smooth_window)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(m_x, y_r_s, '-', color='red', linewidth=1.6, label='RED')
                ax.plot(m_x, y_b_s, '-', color='blue', linewidth=1.6, label='BLUE')
                ax.set_xlabel('distance (m)')
                ax.set_ylabel(f'{stat_name} intensity')
                ax.set_title(f'Per-1m {stat_name} intensity (point-reduce)')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend()
                fig.tight_layout()
                out = args.output.with_name(args.output.stem + f"_1m_{stat_name}" + args.output.suffix)
                fig.savefig(out, dpi=150)
                plt.close(fig)
                print(f"Saved per-meter {stat_name} plot to {out}")

            # Zero-fraction plot
            if args.per_meter_zero_fraction:
                with np.errstate(invalid='ignore', divide='ignore'):
                    zr = np.divide(zer_r, tot_r, out=np.full_like(zer_r, np.nan, dtype=float), where=tot_r > 0)
                    zb = np.divide(zer_b, tot_b, out=np.full_like(zer_b, np.nan, dtype=float), where=tot_b > 0)
                zr_s = _smooth_series_ignore_nan(zr, args.smooth_window)
                zb_s = _smooth_series_ignore_nan(zb, args.smooth_window)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(m_x, zr_s, '-', color='red', linewidth=1.6, label='RED zero-fraction')
                ax.plot(m_x, zb_s, '-', color='blue', linewidth=1.6, label='BLUE zero-fraction')
                ax.set_xlabel('distance (m)')
                ax.set_ylabel('zero fraction')
                ax.set_title('Per-1m zero-intensity fraction')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend()
                fig.tight_layout()
                outz = args.output.with_name(args.output.stem + "_1m_zero_fraction" + args.output.suffix)
                fig.savefig(outz, dpi=150)
                plt.close(fig)
                print(f"Saved per-meter zero-fraction plot to {outz}")

    # Ground vs Non-ground plots if z-range provided
    if args.ground_z is not None:
        gz0, gz1 = args.ground_z
        zmin, zmax = (gz0, gz1) if gz0 <= gz1 else (gz1, gz0)
        # Overall histogram for ground only
        red_g = _compute_hist_for_files(
            red_files, args.bins, (args.range[0], args.range[1]), per_distance_edges=per_distance_edges, z_range=(zmin, zmax), invert_z=False
        )
        blue_g = _compute_hist_for_files(
            blue_files, args.bins, (args.range[0], args.range[1]), per_distance_edges=per_distance_edges, z_range=(zmin, zmax), invert_z=False
        )
        red_ng = _compute_hist_for_files(
            red_files, args.bins, (args.range[0], args.range[1]), per_distance_edges=per_distance_edges, z_range=(zmin, zmax), invert_z=True
        )
        blue_ng = _compute_hist_for_files(
            blue_files, args.bins, (args.range[0], args.range[1]), per_distance_edges=per_distance_edges, z_range=(zmin, zmax), invert_z=True
        )

        try:
            import matplotlib.pyplot as plt
        except Exception:
            plt = None

        if plt is not None and not args.no_hist:
            # Ground only - overall histogram
            edges = red_g.edges
            gy_r = red_g.counts.astype(float)
            gy_b = blue_g.counts.astype(float)
            if args.normalize:
                rs, bs = gy_r.sum(), gy_b.sum()
                if rs > 0:
                    gy_r /= rs
                if bs > 0:
                    gy_b /= bs
            figg, axg = plt.subplots(figsize=(9, 4.5))
            axg.step(edges[:-1], gy_r, where='post', color='red', label='RED (ground)', alpha=0.9)
            axg.step(edges[:-1], gy_b, where='post', color='blue', label='BLUE (ground)', alpha=0.9)
            axg.set_xlabel('intensity')
            axg.set_ylabel('normalized count' if args.normalize else 'count')
            axg.set_title(f'Intensity histogram (ground z∈[{zmin},{zmax}])')
            axg.grid(True, alpha=0.3, linestyle='--')
            axg.legend()
            figg.tight_layout()
            outg = args.output.with_name(args.output.stem + "_ground" + args.output.suffix)
            figg.savefig(outg, dpi=150)
            plt.close(figg)
            print(f"Saved ground-only histogram to {outg}")

            # Non-ground only - overall histogram
            edges = red_ng.edges
            ny_r = red_ng.counts.astype(float)
            ny_b = blue_ng.counts.astype(float)
            if args.normalize:
                rs, bs = ny_r.sum(), ny_b.sum()
                if rs > 0:
                    ny_r /= rs
                if bs > 0:
                    ny_b /= bs
            fign, axn = plt.subplots(figsize=(9, 4.5))
            axn.step(edges[:-1], ny_r, where='post', color='red', label='RED (non-ground)', alpha=0.9)
            axn.step(edges[:-1], ny_b, where='post', color='blue', label='BLUE (non-ground)', alpha=0.9)
            axn.set_xlabel('intensity')
            axn.set_ylabel('normalized count' if args.normalize else 'count')
            axn.set_title(f'Intensity histogram (non-ground z∉[{zmin},{zmax}])')
            axn.grid(True, alpha=0.3, linestyle='--')
            axn.legend()
            fign.tight_layout()
            outn = args.output.with_name(args.output.stem + "_nonground" + args.output.suffix)
            fign.savefig(outn, dpi=150)
            plt.close(fign)
            print(f"Saved non-ground histogram to {outn}")

            # Per-distance overlays for ground / non-ground if requested
            if not args.no_hist and per_distance_edges is not None and red_g.per_distance_counts is not None and blue_g.per_distance_counts is not None:
                nb = min(len(per_distance_edges) - 1, len(red_g.per_distance_counts), len(blue_g.per_distance_counts))
                import math
                cols = 3 if nb >= 3 else nb
                rows = math.ceil(nb / cols) if cols > 0 else 1
                # Ground
                fig2g, axesg = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
                for i in range(nb):
                    r0, r1 = per_distance_edges[i], per_distance_edges[i + 1]
                    ax2 = axesg[i // cols][i % cols]
                    ry = red_g.per_distance_counts[i].astype(float)
                    by = blue_g.per_distance_counts[i].astype(float)
                    if args.normalize:
                        rsum = ry.sum(); bsum = by.sum()
                        if rsum > 0: ry /= rsum
                        if bsum > 0: by /= bsum
                    ax2.step(edges[:-1], ry, where='post', color='red', alpha=0.9)
                    ax2.step(edges[:-1], by, where='post', color='blue', alpha=0.9)
                    ax2.set_title(f"{int(r0)}–{int(r1)} m (ground)")
                    ax2.grid(True, alpha=0.3, linestyle='--')
                for j in range(nb, rows * cols):
                    fig2g.delaxes(axesg[j // cols][j % cols])
                fig2g.suptitle("Per-distance histograms (ground)")
                fig2g.tight_layout(rect=[0, 0.03, 1, 0.95])
                out2g = args.output.with_name(args.output.stem + "_per_distance_ground" + args.output.suffix)
                fig2g.savefig(out2g, dpi=150)
                plt.close(fig2g)
                print(f"Saved ground per-distance plot to {out2g}")

                # Non-ground
                nb2 = min(len(per_distance_edges) - 1, len(red_ng.per_distance_counts), len(blue_ng.per_distance_counts))
                cols2 = 3 if nb2 >= 3 else nb2
                rows2 = math.ceil(nb2 / cols2) if cols2 > 0 else 1
                fig2n, axesn = plt.subplots(rows2, cols2, figsize=(5 * cols2, 3.5 * rows2), squeeze=False)
                for i in range(nb2):
                    r0, r1 = per_distance_edges[i], per_distance_edges[i + 1]
                    ax2 = axesn[i // cols2][i % cols2]
                    ry = red_ng.per_distance_counts[i].astype(float)
                    by = blue_ng.per_distance_counts[i].astype(float)
                    if args.normalize:
                        rsum = ry.sum(); bsum = by.sum()
                        if rsum > 0: ry /= rsum
                        if bsum > 0: by /= bsum
                    ax2.step(edges[:-1], ry, where='post', color='red', alpha=0.9)
                    ax2.step(edges[:-1], by, where='post', color='blue', alpha=0.9)
                    ax2.set_title(f"{int(r0)}–{int(r1)} m (non-ground)")
                    ax2.grid(True, alpha=0.3, linestyle='--')
                for j in range(nb2, rows2 * cols2):
                    fig2n.delaxes(axesn[j // cols2][j % cols2])
                fig2n.suptitle("Per-distance histograms (non-ground)")
                fig2n.tight_layout(rect=[0, 0.03, 1, 0.95])
                out2n = args.output.with_name(args.output.stem + "_per_distance_nonground" + args.output.suffix)
                fig2n.savefig(out2n, dpi=150)
                plt.close(fig2n)
                print(f"Saved non-ground per-distance plot to {out2n}")

            # Mean intensity vs distance (bucketed) for ground/non-ground
            if per_distance_edges is not None and red_g.per_distance_counts is not None and blue_g.per_distance_counts is not None:
                # Ensure plotting is available (may be needed even if hist overlays were disabled)
                try:
                    import matplotlib.pyplot as plt  # type: ignore
                except Exception as e:
                    raise RuntimeError("matplotlib is required to plot mean-vs-distance graphs.") from e
                centers = (edges[:-1] + edges[1:]) / 2.0
                def mean_from_hist(h: np.ndarray) -> float:
                    s = h.sum()
                    return float(np.average(centers, weights=h)) if s > 0 else float('nan')
                nb = min(len(per_distance_edges) - 1, len(red_g.per_distance_counts), len(blue_g.per_distance_counts))
                x = [0.5 * (per_distance_edges[i] + per_distance_edges[i + 1]) for i in range(nb)]
                # Ground
                red_means = [mean_from_hist(red_g.per_distance_counts[i]) for i in range(nb)]
                blue_means = [mean_from_hist(blue_g.per_distance_counts[i]) for i in range(nb)]
                red_means_s = _smooth_series_ignore_nan(np.asarray(red_means, dtype=float), args.smooth_window)
                blue_means_s = _smooth_series_ignore_nan(np.asarray(blue_means, dtype=float), args.smooth_window)
                fig3g, ax3g = plt.subplots(figsize=(7, 4))
                ax3g.plot(x, red_means_s, '-', color='red', label='RED mean (ground)', linewidth=1.6)
                ax3g.plot(x, blue_means_s, '-', color='blue', label='BLUE mean (ground)', linewidth=1.6)
                ax3g.set_xlabel('distance (m)')
                ax3g.set_ylabel('mean intensity')
                ax3g.set_title('Mean intensity vs distance (ground)')
                ax3g.grid(True, alpha=0.3, linestyle='--')
                ax3g.legend()
                fig3g.tight_layout()
                out3g = args.output.with_name(args.output.stem + "_mean_vs_distance_ground" + args.output.suffix)
                fig3g.savefig(out3g, dpi=150)
                plt.close(fig3g)
                print(f"Saved ground mean-vs-distance plot to {out3g}")

                # Non-ground
                nb2 = min(len(per_distance_edges) - 1, len(red_ng.per_distance_counts), len(blue_ng.per_distance_counts))
                x2 = [0.5 * (per_distance_edges[i] + per_distance_edges[i + 1]) for i in range(nb2)]
                red_means2 = [mean_from_hist(red_ng.per_distance_counts[i]) for i in range(nb2)]
                blue_means2 = [mean_from_hist(blue_ng.per_distance_counts[i]) for i in range(nb2)]
                red_means2_s = _smooth_series_ignore_nan(np.asarray(red_means2, dtype=float), args.smooth_window)
                blue_means2_s = _smooth_series_ignore_nan(np.asarray(blue_means2, dtype=float), args.smooth_window)
                fig3n, ax3n = plt.subplots(figsize=(7, 4))
                ax3n.plot(x2, red_means2_s, '-', color='red', label='RED mean (non-ground)', linewidth=1.6)
                ax3n.plot(x2, blue_means2_s, '-', color='blue', label='BLUE mean (non-ground)', linewidth=1.6)
                ax3n.set_xlabel('distance (m)')
                ax3n.set_ylabel('mean intensity')
                ax3n.set_title('Mean intensity vs distance (non-ground)')
                ax3n.grid(True, alpha=0.3, linestyle='--')
                ax3n.legend()
                fig3n.tight_layout()
                out3n = args.output.with_name(args.output.stem + "_mean_vs_distance_nonground" + args.output.suffix)
                fig3n.savefig(out3n, dpi=150)
                plt.close(fig3n)
                print(f"Saved non-ground mean-vs-distance plot to {out3n}")

            # Per-meter mean intensity for ground/non-ground
            if args.per_meter:
                # Ground
                if args.per_meter_reduce == "frame-mean":
                    gx, gy, _, _ = _compute_per_meter_frame_mean(red_files, args.meter_min, args.meter_max, z_range=(zmin, zmax), invert_z=False)
                    bx, by, _, _ = _compute_per_meter_frame_mean(blue_files, args.meter_min, args.meter_max, z_range=(zmin, zmax), invert_z=False)
                    if not np.array_equal(gx, bx):
                        mx = np.intersect1d(gx, bx)
                        gy = gy[np.isin(gx, mx)]
                        by = by[np.isin(bx, mx)]
                    else:
                        mx = gx
                else:
                    # point-reduce via histogram mean
                    lohi = (args.range[0], args.range[1])
                    gx, gh, _, _, _ = _compute_per_meter_hist_and_zeros(red_files, args.meter_min, args.meter_max, args.bins, lohi, z_range=(zmin, zmax), invert_z=False)
                    bx, bh, _, _, _ = _compute_per_meter_hist_and_zeros(blue_files, args.meter_min, args.meter_max, args.bins, lohi, z_range=(zmin, zmax), invert_z=False)
                    if not np.array_equal(gx, bx):
                        mx = np.intersect1d(gx, bx)
                        gh = gh[np.isin(gx, mx), :]
                        bh = bh[np.isin(bx, mx), :]
                    else:
                        mx = gx
                    edges_i = np.linspace(lohi[0], lohi[1], args.bins + 1)
                    centers_i = (edges_i[:-1] + edges_i[1:]) / 2.0
                    gy = (gh @ centers_i) / np.maximum(gh.sum(axis=1), 1)
                    by = (bh @ centers_i) / np.maximum(bh.sum(axis=1), 1)
                # Plot ground
                gy_s = _smooth_series_ignore_nan(gy, args.smooth_window)
                by_s = _smooth_series_ignore_nan(by, args.smooth_window)
                fig4g, ax4g = plt.subplots(figsize=(8, 4))
                ax4g.plot(mx, gy_s, '-', color='red', linewidth=1.6, label='RED (ground)')
                ax4g.plot(mx, by_s, '-', color='blue', linewidth=1.6, label='BLUE (ground)')
                ax4g.set_xlabel('distance (m)')
                ax4g.set_ylabel('mean intensity')
                ax4g.set_title(f'Per-1m mean intensity (ground z∈[{zmin},{zmax}])')
                ax4g.grid(True, alpha=0.3, linestyle='--')
                ax4g.legend()
                fig4g.tight_layout()
                out4g = args.output.with_name(args.output.stem + "_mean_vs_distance_1m_ground" + args.output.suffix)
                fig4g.savefig(out4g, dpi=150)
                plt.close(fig4g)
                print(f"Saved per-meter ground mean plot to {out4g}")

                # Non-ground
                if args.per_meter_reduce == "frame-mean":
                    gx2, gy2, _, _ = _compute_per_meter_frame_mean(red_files, args.meter_min, args.meter_max, z_range=(zmin, zmax), invert_z=True)
                    bx2, by2, _, _ = _compute_per_meter_frame_mean(blue_files, args.meter_min, args.meter_max, z_range=(zmin, zmax), invert_z=True)
                    if not np.array_equal(gx2, bx2):
                        mx2 = np.intersect1d(gx2, bx2)
                        gy2 = gy2[np.isin(gx2, mx2)]
                        by2 = by2[np.isin(bx2, mx2)]
                    else:
                        mx2 = gx2
                else:
                    lohi = (args.range[0], args.range[1])
                    gx2, gh2, _, _, _ = _compute_per_meter_hist_and_zeros(red_files, args.meter_min, args.meter_max, args.bins, lohi, z_range=(zmin, zmax), invert_z=True)
                    bx2, bh2, _, _, _ = _compute_per_meter_hist_and_zeros(blue_files, args.meter_min, args.meter_max, args.bins, lohi, z_range=(zmin, zmax), invert_z=True)
                    if not np.array_equal(gx2, bx2):
                        mx2 = np.intersect1d(gx2, bx2)
                        gh2 = gh2[np.isin(gx2, mx2), :]
                        bh2 = bh2[np.isin(bx2, mx2), :]
                    else:
                        mx2 = gx2
                    edges_i = np.linspace(lohi[0], lohi[1], args.bins + 1)
                    centers_i = (edges_i[:-1] + edges_i[1:]) / 2.0
                    gy2 = (gh2 @ centers_i) / np.maximum(gh2.sum(axis=1), 1)
                    by2 = (bh2 @ centers_i) / np.maximum(bh2.sum(axis=1), 1)
                gy2_s = _smooth_series_ignore_nan(gy2, args.smooth_window)
                by2_s = _smooth_series_ignore_nan(by2, args.smooth_window)
                fig4n, ax4n = plt.subplots(figsize=(8, 4))
                ax4n.plot(mx2, gy2_s, '-', color='red', linewidth=1.6, label='RED (non-ground)')
                ax4n.plot(mx2, by2_s, '-', color='blue', linewidth=1.6, label='BLUE (non-ground)')
                ax4n.set_xlabel('distance (m)')
                ax4n.set_ylabel('mean intensity')
                ax4n.set_title('Per-1m mean intensity (non-ground)')
                ax4n.grid(True, alpha=0.3, linestyle='--')
                ax4n.legend()
                fig4n.tight_layout()
                out4n = args.output.with_name(args.output.stem + "_mean_vs_distance_1m_nonground" + args.output.suffix)
                fig4n.savefig(out4n, dpi=150)
                plt.close(fig4n)
                print(f"Saved per-meter non-ground mean plot to {out4n}")

    # RANSAC-based ground vs non-ground splitting
    if args.ransac_ground:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            plt = None

        # Define RANSAC ground mask function factory
        import math
        def make_ransac_mask(select_ground: bool):
            def mask_func(pts: np.ndarray) -> np.ndarray:
                # Seed selection within radius
                x = pts[:, 0]; y = pts[:, 1]; z = pts[:, 2]
                r = np.sqrt(x * x + y * y)
                seed = r <= float(args.rg_seed_radius)
                idx = np.nonzero(seed)[0]
                if idx.size == 0:
                    return np.zeros(pts.shape[0], dtype=bool) if select_ground else np.ones(pts.shape[0], dtype=bool)
                # Subsample seed
                if idx.size > args.rg_sample_limit > 0:
                    idx = np.random.choice(idx, size=args.rg_sample_limit, replace=False)
                Xs = x[idx]; Ys = y[idx]; Zs = z[idx]

                best_inliers = None
                best_count = -1
                # RANSAC iterations
                for _ in range(int(args.rg_max_iters)):
                    if idx.size < 3:
                        break
                    j = np.random.choice(idx, size=3, replace=False)
                    p1 = np.array([x[j[0]], y[j[0]], z[j[0]]], dtype=float)
                    p2 = np.array([x[j[1]], y[j[1]], z[j[1]]], dtype=float)
                    p3 = np.array([x[j[2]], y[j[2]], z[j[2]]], dtype=float)
                    v1 = p2 - p1
                    v2 = p3 - p1
                    n = np.cross(v1, v2)
                    norm = np.linalg.norm(n)
                    if not np.isfinite(norm) or norm < 1e-6:
                        continue
                    n = n / norm
                    # Ensure upward normal (nz >= 0)
                    if n[2] < 0:
                        n = -n
                    # Slope constraint
                    angle_deg = math.degrees(math.acos(max(min(n[2], 1.0), -1.0)))
                    if angle_deg > float(args.rg_slope_max_deg):
                        continue
                    d = -np.dot(n, p1)
                    # Distances for seed
                    dist = np.abs(n[0] * Xs + n[1] * Ys + n[2] * Zs + d)
                    inl = dist <= float(args.rg_dist_thresh)
                    cnt = int(np.count_nonzero(inl))
                    if cnt > best_count:
                        best_count = cnt
                        best_inliers = inl
                # Fallback: constant-z plane at median z
                if best_inliers is None or best_count < 10:
                    zmed = float(np.median(Zs)) if Zs.size > 0 else 0.0
                    n = np.array([0.0, 0.0, 1.0], dtype=float)
                    d = -zmed
                else:
                    # Refit plane z = ax + by + c using inliers (least squares)
                    sel = np.nonzero(best_inliers)[0]
                    Xi = Xs[sel]; Yi = Ys[sel]; Zi = Zs[sel]
                    A = np.stack([Xi, Yi, np.ones_like(Xi)], axis=1)
                    try:
                        a, b, c = np.linalg.lstsq(A, Zi, rcond=None)[0]
                        # Convert to n·p + d = 0 form
                        n = np.array([-a, -b, 1.0], dtype=float)
                        n = n / np.linalg.norm(n)
                        if n[2] < 0:
                            n = -n
                        d = -c
                    except Exception:
                        zmed = float(np.median(Zs)) if Zs.size > 0 else 0.0
                        n = np.array([0.0, 0.0, 1.0], dtype=float)
                        d = -zmed

                # Final mask over all points
                dist_all = np.abs(n[0] * x + n[1] * y + n[2] * z + d)
                ground = dist_all <= float(args.rg_dist_thresh)
                return ground if select_ground else ~ground

            return mask_func

        # Compute overall histograms for ground and non-ground
        red_g = _compute_hist_for_files(
            red_files, args.bins, (args.range[0], args.range[1]), per_distance_edges=per_distance_edges, mask_func=make_ransac_mask(True)
        )
        blue_g = _compute_hist_for_files(
            blue_files, args.bins, (args.range[0], args.range[1]), per_distance_edges=per_distance_edges, mask_func=make_ransac_mask(True)
        )
        red_ng = _compute_hist_for_files(
            red_files, args.bins, (args.range[0], args.range[1]), per_distance_edges=per_distance_edges, mask_func=make_ransac_mask(False)
        )
        blue_ng = _compute_hist_for_files(
            blue_files, args.bins, (args.range[0], args.range[1]), per_distance_edges=per_distance_edges, mask_func=make_ransac_mask(False)
        )

        if plt is not None and not args.no_hist:
            # Ground overall hist
            edges = red_g.edges
            gy_r = red_g.counts.astype(float)
            gy_b = blue_g.counts.astype(float)
            if args.normalize:
                rs, bs = gy_r.sum(), gy_b.sum()
                if rs > 0: gy_r /= rs
                if bs > 0: gy_b /= bs
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.step(edges[:-1], gy_r, where='post', color='red', label='RED (ground-RANSAC)', alpha=0.9)
            ax.step(edges[:-1], gy_b, where='post', color='blue', label='BLUE (ground-RANSAC)', alpha=0.9)
            ax.set_xlabel('intensity'); ax.set_ylabel('normalized count' if args.normalize else 'count')
            ax.set_title('Intensity histogram (ground via RANSAC)')
            ax.grid(True, alpha=0.3, linestyle='--'); ax.legend(); fig.tight_layout()
            outg = args.output.with_name(args.output.stem + "_ground_ransac" + args.output.suffix)
            fig.savefig(outg, dpi=150); plt.close(fig)
            print(f"Saved RANSAC ground histogram to {outg}")

            # Non-ground overall hist
            edges = red_ng.edges
            ny_r = red_ng.counts.astype(float)
            ny_b = blue_ng.counts.astype(float)
            if args.normalize:
                rs, bs = ny_r.sum(), ny_b.sum()
                if rs > 0: ny_r /= rs
                if bs > 0: ny_b /= bs
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.step(edges[:-1], ny_r, where='post', color='red', label='RED (non-ground RANSAC)', alpha=0.9)
            ax.step(edges[:-1], ny_b, where='post', color='blue', label='BLUE (non-ground RANSAC)', alpha=0.9)
            ax.set_xlabel('intensity'); ax.set_ylabel('normalized count' if args.normalize else 'count')
            ax.set_title('Intensity histogram (non-ground via RANSAC)')
            ax.grid(True, alpha=0.3, linestyle='--'); ax.legend(); fig.tight_layout()
            outn = args.output.with_name(args.output.stem + "_nonground_ransac" + args.output.suffix)
            fig.savefig(outn, dpi=150); plt.close(fig)
            print(f"Saved RANSAC non-ground histogram to {outn}")

            # Per-distance overlays if requested
            if per_distance_edges is not None and red_g.per_distance_counts is not None and blue_g.per_distance_counts is not None:
                nb = min(len(per_distance_edges) - 1, len(red_g.per_distance_counts), len(blue_g.per_distance_counts))
                import math
                cols = 3 if nb >= 3 else nb
                rows = math.ceil(nb / cols) if cols > 0 else 1
                if not args.no_hist:
                    # Ground per-distance
                    figpdg, axg = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
                    for i in range(nb):
                        r0, r1 = per_distance_edges[i], per_distance_edges[i + 1]
                        a = axg[i // cols][i % cols]
                        ry = red_g.per_distance_counts[i].astype(float)
                        by = blue_g.per_distance_counts[i].astype(float)
                        if args.normalize:
                            rs, bs = ry.sum(), by.sum()
                            if rs > 0: ry /= rs
                            if bs > 0: by /= bs
                        a.step(edges[:-1], ry, where='post', color='red', alpha=0.9)
                        a.step(edges[:-1], by, where='post', color='blue', alpha=0.9)
                        a.set_title(f"{int(r0)}–{int(r1)} m (ground)")
                        a.grid(True, alpha=0.3, linestyle='--')
                    for j in range(nb, rows * cols): figpdg.delaxes(axg[j // cols][j % cols])
                    figpdg.suptitle('Per-distance histograms (ground, RANSAC)')
                    figpdg.tight_layout(rect=[0,0.03,1,0.95])
                    outpdg = args.output.with_name(args.output.stem + "_per_distance_ground_ransac" + args.output.suffix)
                    figpdg.savefig(outpdg, dpi=150); plt.close(figpdg)
                    print(f"Saved RANSAC ground per-distance plot to {outpdg}")

                    # Non-ground per-distance
                    figpdn, axn = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
                    for i in range(nb):
                        r0, r1 = per_distance_edges[i], per_distance_edges[i + 1]
                        a = axn[i // cols][i % cols]
                        ry = red_ng.per_distance_counts[i].astype(float)
                        by = blue_ng.per_distance_counts[i].astype(float)
                        if args.normalize:
                            rs, bs = ry.sum(), by.sum()
                            if rs > 0: ry /= rs
                            if bs > 0: by /= bs
                        a.step(edges[:-1], ry, where='post', color='red', alpha=0.9)
                        a.step(edges[:-1], by, where='post', color='blue', alpha=0.9)
                        a.set_title(f"{int(r0)}–{int(r1)} m (non-ground)")
                        a.grid(True, alpha=0.3, linestyle='--')
                    for j in range(nb, rows * cols): figpdn.delaxes(axn[j // cols][j % cols])
                    figpdn.suptitle('Per-distance histograms (non-ground, RANSAC)')
                    figpdn.tight_layout(rect=[0,0.03,1,0.95])
                    outpdn = args.output.with_name(args.output.stem + "_per_distance_nonground_ransac" + args.output.suffix)
                    figpdn.savefig(outpdn, dpi=150); plt.close(figpdn)
                    print(f"Saved RANSAC non-ground per-distance plot to {outpdn}")

                # Mean vs distance (bucketed) for ground and non-ground
                # Ensure plotting is available (could be needed even if hist overlays were disabled)
                try:
                    import matplotlib.pyplot as plt  # type: ignore
                except Exception as e:
                    raise RuntimeError("matplotlib is required to plot mean-vs-distance graphs.") from e
                centers = (edges[:-1] + edges[1:]) / 2.0
                def mean_from_hist(h: np.ndarray) -> float:
                    s = h.sum();
                    return float(np.average(centers, weights=h)) if s > 0 else float('nan')
                x = [0.5 * (per_distance_edges[i] + per_distance_edges[i + 1]) for i in range(nb)]
                # Ground
                rg_means = [mean_from_hist(red_g.per_distance_counts[i]) for i in range(nb)]
                bg_means = [mean_from_hist(blue_g.per_distance_counts[i]) for i in range(nb)]
                fg, axmg = plt.subplots(figsize=(7,4))
                axmg.plot(x, _smooth_series_ignore_nan(np.asarray(rg_means,float), args.smooth_window), '-', color='red', linewidth=1.6, label='RED mean (ground)')
                axmg.plot(x, _smooth_series_ignore_nan(np.asarray(bg_means,float), args.smooth_window), '-', color='blue', linewidth=1.6, label='BLUE mean (ground)')
                axmg.set_xlabel('distance (m)'); axmg.set_ylabel('mean intensity'); axmg.set_title('Mean intensity vs distance (ground, RANSAC)'); axmg.grid(True, alpha=0.3, linestyle='--'); axmg.legend(); fg.tight_layout()
                outmg = args.output.with_name(args.output.stem + "_mean_vs_distance_ground_ransac" + args.output.suffix)
                fg.savefig(outmg, dpi=150); plt.close(fg)
                print(f"Saved RANSAC ground mean-vs-distance plot to {outmg}")
                # Non-ground
                rng_means = [mean_from_hist(red_ng.per_distance_counts[i]) for i in range(nb)]
                bng_means = [mean_from_hist(blue_ng.per_distance_counts[i]) for i in range(nb)]
                fn, axmn = plt.subplots(figsize=(7,4))
                axmn.plot(x, _smooth_series_ignore_nan(np.asarray(rng_means,float), args.smooth_window), '-', color='red', linewidth=1.6, label='RED mean (non-ground)')
                axmn.plot(x, _smooth_series_ignore_nan(np.asarray(bng_means,float), args.smooth_window), '-', color='blue', linewidth=1.6, label='BLUE mean (non-ground)')
                axmn.set_xlabel('distance (m)'); axmn.set_ylabel('mean intensity'); axmn.set_title('Mean intensity vs distance (non-ground, RANSAC)'); axmn.grid(True, alpha=0.3, linestyle='--'); axmn.legend(); fn.tight_layout()
                outmn = args.output.with_name(args.output.stem + "_mean_vs_distance_nonground_ransac" + args.output.suffix)
                fn.savefig(outmn, dpi=150); plt.close(fn)
                print(f"Saved RANSAC non-ground mean-vs-distance plot to {outmn}")

            # Per-meter stats (mean and others) via the new engines
            if args.per_meter:
                # Mean (frame-mean reduce)
                if "mean" in args.per_meter_stat and args.per_meter_reduce == "frame-mean":
                    gx_r, gy_r, _, _ = _compute_per_meter_frame_mean(red_files, args.meter_min, args.meter_max, mask_func=make_ransac_mask(True))
                    gx_b, gy_b, _, _ = _compute_per_meter_frame_mean(blue_files, args.meter_min, args.meter_max, mask_func=make_ransac_mask(True))
                    mx = gx_r if np.array_equal(gx_r, gx_b) else np.intersect1d(gx_r, gx_b)
                    gy_r = gy_r[np.isin(gx_r, mx)]; gy_b = gy_b[np.isin(gx_b, mx)]
                    fig, ax = plt.subplots(figsize=(8,4))
                    ax.plot(mx, _smooth_series_ignore_nan(gy_r, args.smooth_window), '-', color='red', linewidth=1.6, label='RED (ground, frame-mean)')
                    ax.plot(mx, _smooth_series_ignore_nan(gy_b, args.smooth_window), '-', color='blue', linewidth=1.6, label='BLUE (ground, frame-mean)')
                    ax.set_xlabel('distance (m)'); ax.set_ylabel('mean intensity'); ax.set_title('Per-1m mean (ground, RANSAC)'); ax.grid(True, alpha=0.3, linestyle='--'); ax.legend(); fig.tight_layout()
                    out = args.output.with_name(args.output.stem + "_1m_mean_framemean_ground_ransac" + args.output.suffix)
                    fig.savefig(out, dpi=150); plt.close(fig); print(f"Saved RANSAC ground 1m frame-mean to {out}")

                    gx_r, gy_r, _, _ = _compute_per_meter_frame_mean(red_files, args.meter_min, args.meter_max, mask_func=make_ransac_mask(False))
                    gx_b, gy_b, _, _ = _compute_per_meter_frame_mean(blue_files, args.meter_min, args.meter_max, mask_func=make_ransac_mask(False))
                    mx = gx_r if np.array_equal(gx_r, gx_b) else np.intersect1d(gx_r, gx_b)
                    gy_r = gy_r[np.isin(gx_r, mx)]; gy_b = gy_b[np.isin(gx_b, mx)]
                    fig, ax = plt.subplots(figsize=(8,4))
                    ax.plot(mx, _smooth_series_ignore_nan(gy_r, args.smooth_window), '-', color='red', linewidth=1.6, label='RED (non-ground, frame-mean)')
                    ax.plot(mx, _smooth_series_ignore_nan(gy_b, args.smooth_window), '-', color='blue', linewidth=1.6, label='BLUE (non-ground, frame-mean)')
                    ax.set_xlabel('distance (m)'); ax.set_ylabel('mean intensity'); ax.set_title('Per-1m mean (non-ground, RANSAC)'); ax.grid(True, alpha=0.3, linestyle='--'); ax.legend(); fig.tight_layout()
                    out = args.output.with_name(args.output.stem + "_1m_mean_framemean_nonground_ransac" + args.output.suffix)
                    fig.savefig(out, dpi=150); plt.close(fig); print(f"Saved RANSAC non-ground 1m frame-mean to {out}")

                # Histogram-based stats (mean-point, median, p10, p90) and zero fraction for ground/non-ground
                need_hist = (args.per_meter_reduce == "point" and "mean" in args.per_meter_stat) or any(s in args.per_meter_stat for s in ("median","p10","p90")) or args.per_meter_zero_fraction
                if need_hist:
                    lohi = (args.range[0], args.range[1])
                    mx_r, hr_g, tr_g, zr_g, _ = _compute_per_meter_hist_and_zeros(red_files, args.meter_min, args.meter_max, args.bins, lohi, mask_func=make_ransac_mask(True))
                    mx_b, hb_g, tb_g, zb_g, _ = _compute_per_meter_hist_and_zeros(blue_files, args.meter_min, args.meter_max, args.bins, lohi, mask_func=make_ransac_mask(True))
                    if not np.array_equal(mx_r, mx_b):
                        m = np.intersect1d(mx_r, mx_b)
                        selr = np.isin(mx_r, m); selb = np.isin(mx_b, m)
                        mx = m; hr_g = hr_g[selr]; hb_g = hb_g[selb]; tr_g = tr_g[selr]; tb_g = tb_g[selb]; zr_g = zr_g[selr]; zb_g = zb_g[selb]
                    else:
                        mx = mx_r
                    edges_i = np.linspace(lohi[0], lohi[1], args.bins + 1)
                    centers_i = (edges_i[:-1] + edges_i[1:]) / 2.0
                    def stat_from_hist(H, stat):
                        if stat == 'mean':
                            num = H @ centers_i; den = H.sum(axis=1)
                            with np.errstate(invalid='ignore', divide='ignore'):
                                return np.divide(num, den, out=np.full(H.shape[0], np.nan), where=den>0)
                        qmap = {'median':0.5,'p10':0.10,'p90':0.90}
                        q = qmap[stat]; M = H.shape[0]; out = np.full(M, np.nan)
                        for i in range(M):
                            h = H[i]; c = h.cumsum(); tot = int(c[-1]) if h.size>0 else 0
                            if tot<=0: continue
                            idx = int(np.searchsorted(c, q*tot, side='left')); idx = min(max(idx,0), h.size-1)
                            out[i] = centers_i[idx]
                        return out
                    for stat in args.per_meter_stat:
                        if stat=='mean' and args.per_meter_reduce!='point':
                            continue
                        yr = stat_from_hist(hr_g, stat); yb = stat_from_hist(hb_g, stat)
                        fig, ax = plt.subplots(figsize=(8,4))
                        ax.plot(mx, _smooth_series_ignore_nan(yr, args.smooth_window), '-', color='red', linewidth=1.6, label=f'RED ({stat}, ground)')
                        ax.plot(mx, _smooth_series_ignore_nan(yb, args.smooth_window), '-', color='blue', linewidth=1.6, label=f'BLUE ({stat}, ground)')
                        ax.set_xlabel('distance (m)'); ax.set_ylabel(f'{stat} intensity'); ax.set_title(f'Per-1m {stat} (ground, RANSAC)'); ax.grid(True, alpha=0.3, linestyle='--'); ax.legend(); fig.tight_layout()
                        out = args.output.with_name(args.output.stem + f"_1m_{stat}_ground_ransac" + args.output.suffix)
                        fig.savefig(out, dpi=150); plt.close(fig); print(f"Saved RANSAC ground 1m {stat} to {out}")

                    if args.per_meter_zero_fraction:
                        with np.errstate(invalid='ignore', divide='ignore'):
                            zfr = np.divide(zr_g, tr_g, out=np.full_like(zr_g, np.nan, dtype=float), where=tr_g>0)
                            zfb = np.divide(zb_g, tb_g, out=np.full_like(zb_g, np.nan, dtype=float), where=tb_g>0)
                        fig, ax = plt.subplots(figsize=(8,4))
                        ax.plot(mx, _smooth_series_ignore_nan(zfr, args.smooth_window), '-', color='red', linewidth=1.6, label='RED zero-fraction (ground)')
                        ax.plot(mx, _smooth_series_ignore_nan(zfb, args.smooth_window), '-', color='blue', linewidth=1.6, label='BLUE zero-fraction (ground)')
                        ax.set_xlabel('distance (m)'); ax.set_ylabel('zero fraction'); ax.set_title('Per-1m zero-fraction (ground, RANSAC)'); ax.grid(True, alpha=0.3, linestyle='--'); ax.legend(); fig.tight_layout()
                        out = args.output.with_name(args.output.stem + "_1m_zero_fraction_ground_ransac" + args.output.suffix)
                        fig.savefig(out, dpi=150); plt.close(fig); print(f"Saved RANSAC ground 1m zero-fraction to {out}")

                    # Non-ground stats
                    mx_r, hr_ng, tr_ng, zr_ng, _ = _compute_per_meter_hist_and_zeros(red_files, args.meter_min, args.meter_max, args.bins, lohi, mask_func=make_ransac_mask(False))
                    mx_b, hb_ng, tb_ng, zb_ng, _ = _compute_per_meter_hist_and_zeros(blue_files, args.meter_min, args.meter_max, args.bins, lohi, mask_func=make_ransac_mask(False))
                    if not np.array_equal(mx_r, mx_b):
                        m = np.intersect1d(mx_r, mx_b)
                        selr = np.isin(mx_r, m); selb = np.isin(mx_b, m)
                        mx = m; hr_ng = hr_ng[selr]; hb_ng = hb_ng[selb]; tr_ng = tr_ng[selr]; tb_ng = tb_ng[selb]; zr_ng = zr_ng[selr]; zb_ng = zb_ng[selb]
                    else:
                        mx = mx_r
                    for stat in args.per_meter_stat:
                        if stat=='mean' and args.per_meter_reduce!='point':
                            continue
                        yr = stat_from_hist(hr_ng, stat); yb = stat_from_hist(hb_ng, stat)
                        fig, ax = plt.subplots(figsize=(8,4))
                        ax.plot(mx, _smooth_series_ignore_nan(yr, args.smooth_window), '-', color='red', linewidth=1.6, label=f'RED ({stat}, non-ground)')
                        ax.plot(mx, _smooth_series_ignore_nan(yb, args.smooth_window), '-', color='blue', linewidth=1.6, label=f'BLUE ({stat}, non-ground)')
                        ax.set_xlabel('distance (m)'); ax.set_ylabel(f'{stat} intensity'); ax.set_title(f'Per-1m {stat} (non-ground, RANSAC)'); ax.grid(True, alpha=0.3, linestyle='--'); ax.legend(); fig.tight_layout()
                        out = args.output.with_name(args.output.stem + f"_1m_{stat}_nonground_ransac" + args.output.suffix)
                        fig.savefig(out, dpi=150); plt.close(fig); print(f"Saved RANSAC non-ground 1m {stat} to {out}")

                    if args.per_meter_zero_fraction:
                        with np.errstate(invalid='ignore', divide='ignore'):
                            zfr = np.divide(zr_ng, tr_ng, out=np.full_like(zr_ng, np.nan, dtype=float), where=tr_ng>0)
                            zfb = np.divide(zb_ng, tb_ng, out=np.full_like(zb_ng, np.nan, dtype=float), where=tb_ng>0)
                        fig, ax = plt.subplots(figsize=(8,4))
                        ax.plot(mx, _smooth_series_ignore_nan(zfr, args.smooth_window), '-', color='red', linewidth=1.6, label='RED zero-fraction (non-ground)')
                        ax.plot(mx, _smooth_series_ignore_nan(zfb, args.smooth_window), '-', color='blue', linewidth=1.6, label='BLUE zero-fraction (non-ground)')
                        ax.set_xlabel('distance (m)'); ax.set_ylabel('zero fraction'); ax.set_title('Per-1m zero-fraction (non-ground, RANSAC)'); ax.grid(True, alpha=0.3, linestyle='--'); ax.legend(); fig.tight_layout()
                        out = args.output.with_name(args.output.stem + "_1m_zero_fraction_nonground_ransac" + args.output.suffix)
                        fig.savefig(out, dpi=150); plt.close(fig); print(f"Saved RANSAC non-ground 1m zero-fraction to {out}")


if __name__ == "__main__":
    main()

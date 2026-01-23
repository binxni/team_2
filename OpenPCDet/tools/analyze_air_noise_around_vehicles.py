#!/usr/bin/env python3
"""
Analyze the distribution of airborne noise points around vehicles in points_test.

This script:
  - Loads point clouds from `OpenPCDet/data/custom_av/points_test`
  - Loads detection results (result.pkl) to get vehicle boxes per frame
  - For selected frames (e.g., noisy indices), extracts points within the XY
    footprint (with margin) of each vehicle but above the top of the box
  - Aggregates heights above roof (dz), planar distances (rho), and angles
  - Fits simple distributions (Normal, Laplace, Exponential) to dz and
    reports K-S statistics to judge goodness-of-fit
  - Optionally saves histograms with fitted PDFs overlaid

Examples:
  python OpenPCDet/tools/analyze_air_noise_around_vehicles.py \
      --points-dir OpenPCDet/data/custom_av/points_test \
      --detections result.pkl \
      --frames 10000010-10000030,10000045,10000060-10000065 \
      --save-plots OpenPCDet/output/noise_plots \
      --z-offset 0.3 --z-max 3.0 --margin-xy 0.5

If you don’t specify --frames, the script uses ImageSets/test.txt in the same
dataset root as --points-dir.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ------------------------------
# Utilities
# ------------------------------

def parse_ranges(ranges_str: str) -> List[Tuple[int, int]]:
    """Parse "a-b,c-d,e" into [(a,b), (c,d), (e,e)]."""
    if not ranges_str:
        return []
    parts = [p.strip() for p in ranges_str.split(',') if p.strip()]
    out: List[Tuple[int, int]] = []
    for p in parts:
        if '-' in p:
            a, b = p.split('-', 1)
            ia, ib = int(a), int(b)
            if ib < ia:
                ia, ib = ib, ia
            out.append((ia, ib))
        else:
            i = int(p)
            out.append((i, i))
    return out


def in_any_range(x: int, ranges: Sequence[Tuple[int, int]]) -> bool:
    for a, b in ranges:
        if a <= x <= b:
            return True
    return False


def load_frame_ids_from_imageset(points_dir: Path) -> List[int]:
    # Expect .../data/custom_av/{points,points_test} and ImageSets/test.txt next to it
    root = points_dir.parent
    test_txt = root / 'ImageSets' / 'test.txt'
    if not test_txt.exists():
        raise FileNotFoundError(f"Cannot find ImageSets/test.txt next to {points_dir}")
    ids: List[int] = []
    with test_txt.open('r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                ids.append(int(s))
            except ValueError:
                continue
    return ids


def resolve_frame_ids(
    points_dir: Path,
    frames: Optional[str],
    frames_file: Optional[Path],
) -> List[int]:
    if frames_file is not None:
        if not frames_file.exists():
            raise FileNotFoundError(str(frames_file))
        ids: List[int] = []
        with frames_file.open('r') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                ids.append(int(s))
        return sorted(set(ids))

    if frames:
        ranges = parse_ranges(frames)
        # Enumerate frames that actually exist in points_dir
        existing: List[int] = []
        for name in os.listdir(points_dir):
            if not name.endswith('.npy'):
                continue
            try:
                idx = int(name[:-4])
            except ValueError:
                continue
            if in_any_range(idx, ranges):
                existing.append(idx)
        return sorted(set(existing))

    # Default to test.txt
    return load_frame_ids_from_imageset(points_dir)


@dataclass
class Detections:
    boxes: np.ndarray  # (N, 7) each [x,y,z,dx,dy,dz,yaw]
    names: List[str]   # len N


def load_detections(detections_pkl: Path) -> Dict[str, Detections]:
    """Return mapping frame_id(str) -> Detections."""
    with open(detections_pkl, 'rb') as f:
        data = pickle.load(f)
    out: Dict[str, Detections] = {}
    for it in data:
        frame_id = str(it.get('frame_id'))
        boxes = np.asarray(it.get('boxes_lidar'), dtype=np.float32)
        names = [str(x) for x in it.get('name', [])]
        # Some result dumps might not include 'name' but have 'pred_labels'. Map using known CLASS_NAMES ordering if present.
        if (not names) and ('pred_labels' in it):
            labels = np.asarray(it['pred_labels']).astype(int)
            # Default class mapping used in configs: ['Vehicle', 'Pedestrian', 'Cyclist']
            default_names = ['Vehicle', 'Pedestrian', 'Cyclist']
            names = [default_names[l - 1] if 1 <= l <= len(default_names) else f'cls{l}' for l in labels]
        out[frame_id] = Detections(boxes=boxes, names=names)
    return out


def rotate_points_xy(x: np.ndarray, y: np.ndarray, yaw: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate points by -yaw to align with the box axes.
    Returns (x_local, y_local).
    """
    c = math.cos(yaw)
    s = math.sin(yaw)
    # [x'; y'] = R(-yaw) * [x; y] with R(-yaw) = [[c, s], [-s, c]]
    x_local = c * x + s * y
    y_local = -s * x + c * y
    return x_local, y_local


def extract_airborne_points_around_box(
    pts: np.ndarray,
    box: np.ndarray,
    *,
    margin_xy: float = 0.5,
    z_offset: float = 0.3,
    z_max: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select airborne points above the top of the box, within XY footprint + margin.

    Args:
      pts: (N, >=3), columns [x,y,z,(intensity,...)]
      box: (7,) [cx,cy,cz,dx,dy,dz,yaw], cz is box center z.

    Returns:
      dz:  (M,) heights above the roof (z - (cz + dz/2))
      rho: (M,) planar radius from box center in box-aligned XY
      ang: (M,) angle atan2(y_local, x_local) in radians
    """
    cx, cy, cz, dx, dy, dz, yaw = [float(x) for x in box.tolist()]
    # Shift
    px = pts[:, 0] - cx
    py = pts[:, 1] - cy
    pz = pts[:, 2]
    # Rotate to local frame
    xloc, yloc = rotate_points_xy(px, py, yaw)
    # XY in footprint + margin
    half_x = 0.5 * float(dx) + float(margin_xy)
    half_y = 0.5 * float(dy) + float(margin_xy)
    m_xy = (np.abs(xloc) <= half_x) & (np.abs(yloc) <= half_y)
    if not np.any(m_xy):
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    # Above roof within [z_offset, z_max]
    z_roof = float(cz + 0.5 * dz)
    dz_vals = pz - z_roof
    m_z = (dz_vals >= float(z_offset)) & (dz_vals <= float(z_max))
    m = m_xy & m_z
    if not np.any(m):
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    sel_dz = dz_vals[m].astype(np.float32, copy=False)
    sel_xl = xloc[m].astype(np.float32, copy=False)
    sel_yl = yloc[m].astype(np.float32, copy=False)
    rho = np.sqrt(sel_xl * sel_xl + sel_yl * sel_yl)
    ang = np.arctan2(sel_yl, sel_xl)
    return sel_dz, rho.astype(np.float32, copy=False), ang.astype(np.float32, copy=False)


# ------------------------------
# Simple distribution fits and K-S
# ------------------------------

def fit_normal(x: np.ndarray) -> Tuple[float, float]:
    mu = float(np.mean(x)) if x.size else float('nan')
    sigma = float(np.std(x, ddof=1)) if x.size > 1 else float('nan')
    return mu, max(sigma, 1e-9) if np.isfinite(sigma) else sigma


def fit_laplace(x: np.ndarray) -> Tuple[float, float]:
    if x.size == 0:
        return float('nan'), float('nan')
    mu = float(np.median(x))
    b = float(np.mean(np.abs(x - mu)))
    return mu, max(b, 1e-9)


def fit_exponential_pos(x: np.ndarray) -> float:
    # Fit to Exponential on x>=0 (lambda = 1/mean(x))
    if x.size == 0:
        return float('nan')
    x_pos = x[x >= 0]
    if x_pos.size == 0:
        return float('nan')
    lam = 1.0 / float(np.mean(x_pos))
    return max(lam, 1e-9)


def ks_statistic(x: np.ndarray, cdf_fn) -> float:
    if x.size == 0:
        return float('nan')
    xs = np.sort(x)
    n = xs.size
    # Empirical CDF at each sample point (right-continuous)
    ecdf = (np.arange(1, n + 1)) / float(n)
    # Model CDF
    F = cdf_fn(xs)
    return float(np.max(np.abs(ecdf - F)))


def make_cdf_normal(mu: float, sigma: float):
    inv = math.sqrt(2.0)
    def _cdf(x: np.ndarray) -> np.ndarray:
        z = (x - mu) / sigma
        return 0.5 * (1.0 + erf_vec(z / inv))
    return _cdf


def make_cdf_laplace(mu: float, b: float):
    def _cdf(x: np.ndarray) -> np.ndarray:
        z = (x - mu) / b
        out = np.empty_like(z, dtype=np.float64)
        m = (x < mu)
        out[m] = 0.5 * np.exp(z[m])
        out[~m] = 1.0 - 0.5 * np.exp(-z[~m])
        return out
    return _cdf


def make_cdf_exponential(lam: float):
    def _cdf(x: np.ndarray) -> np.ndarray:
        # For x<0 we clamp to 0
        out = np.zeros_like(x, dtype=np.float64)
        xp = np.maximum(x, 0.0)
        out = 1.0 - np.exp(-lam * xp)
        return out
    return _cdf


def erf_vec(z: np.ndarray) -> np.ndarray:
    # Vectorized math.erf
    import math as _m
    vfunc = np.vectorize(_m.erf)
    return vfunc(z)


# ------------------------------
# Main
# ------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze airborne noise distribution around vehicles in points_test")
    ap.add_argument('--points-dir', type=Path, default=Path('OpenPCDet/data/custom_av/points_test'), help='Directory with .npy point clouds')
    ap.add_argument('--detections', type=Path, default=Path('result.pkl'), help='Path to detection results (pkl)')
    ap.add_argument('--frames', type=str, default='', help='Comma-separated frame ranges like 10000010-10000030,10000045')
    ap.add_argument('--frames-file', type=Path, default=None, help='Optional file with one frame id per line')
    ap.add_argument('--class', dest='cls_name', type=str, default='Vehicle', help='Class to consider')
    ap.add_argument('--margin-xy', type=float, default=0.5, help='XY margin (m) around box footprint')
    ap.add_argument('--z-offset', type=float, default=0.3, help='Min height above roof (m) to count as air point')
    ap.add_argument('--z-max', type=float, default=3.0, help='Max height above roof (m) to consider')
    ap.add_argument('--max-frames', type=int, default=None, help='Optional cap on number of frames to process')
    ap.add_argument('--save-plots', type=Path, default=None, help='Directory to save hist plots (optional)')
    ap.add_argument('--bins', type=int, default=80, help='Histogram bins for plots')
    args = ap.parse_args()

    if not args.points_dir.exists():
        raise FileNotFoundError(f"Points dir not found: {args.points_dir}")
    if not args.detections.exists():
        raise FileNotFoundError(f"Detections not found: {args.detections}")

    frame_ids = resolve_frame_ids(args.points_dir, args.frames, args.frames_file)
    if args.max_frames is not None and args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
    print(f"Frames to analyze: {len(frame_ids)} (first: {frame_ids[0] if frame_ids else 'N/A'})")

    det_map = load_detections(args.detections)
    missing = [fid for fid in frame_ids if str(fid) not in det_map]
    if missing:
        print(f"Warning: {len(missing)} frames have no detections in {args.detections}. Example: {missing[:5]}")

    all_dz: List[float] = []
    all_rho: List[float] = []
    all_ang: List[float] = []
    n_frames_used = 0
    n_veh = 0
    n_pts = 0

    for fid in frame_ids:
        key = str(fid)
        if key not in det_map:
            continue
        det = det_map[key]
        # Filter vehicle boxes
        if det.boxes.size == 0:
            continue
        names = det.names
        if len(names) != det.boxes.shape[0]:
            # Fallback: assume all vehicles if names missing (unlikely)
            sel = np.arange(det.boxes.shape[0])
        else:
            sel = np.array([i for i, nm in enumerate(names) if nm == args.cls_name], dtype=np.int64)
        if sel.size == 0:
            continue
        # Load points
        pc_path = args.points_dir / f"{fid}.npy"
        if not pc_path.exists():
            continue
        pts = np.load(pc_path)
        if pts.ndim != 2 or pts.shape[1] < 3:
            continue
        n_frames_used += 1

        for i in sel:
            box = det.boxes[i]
            dz, rho, ang = extract_airborne_points_around_box(
                pts, box, margin_xy=args.margin_xy, z_offset=args.z_offset, z_max=args.z_max
            )
            if dz.size == 0:
                continue
            all_dz.append(dz)
            all_rho.append(rho)
            all_ang.append(ang)
            n_pts += int(dz.size)
            n_veh += 1

    if n_pts == 0:
        print("No airborne points selected. Try reducing --z-offset or increasing --margin-xy/--z-max.")
        return

    dz = np.concatenate(all_dz)
    rho = np.concatenate(all_rho)
    ang = np.concatenate(all_ang)

    # Fits on dz (>=0 by construction)
    mu_n, sig_n = fit_normal(dz)
    mu_l, b_l = fit_laplace(dz)
    lam_e = fit_exponential_pos(dz)

    cdf_n = make_cdf_normal(mu_n, sig_n)
    cdf_l = make_cdf_laplace(mu_l, b_l)
    cdf_e = make_cdf_exponential(lam_e)
    ks_n = ks_statistic(dz, cdf_n)
    ks_l = ks_statistic(dz, cdf_l)
    ks_e = ks_statistic(dz, cdf_e)

    print("=== Selection Summary ===")
    print(f"frames_used        : {n_frames_used}")
    print(f"vehicles_considered: {n_veh}")
    print(f"airborne_points    : {n_pts}")
    print(f"dz stats (m)       : mean={float(np.mean(dz)):.3f} std={float(np.std(dz)):.3f} p50={float(np.percentile(dz,50)):.3f} p90={float(np.percentile(dz,90)):.3f} max={float(np.max(dz)):.3f}")
    print(f"rho stats (m)      : mean={float(np.mean(rho)):.3f} std={float(np.std(rho)):.3f} p90={float(np.percentile(rho,90)):.3f}")

    print("\n=== Distribution Fits for dz (height above roof) ===")
    print(f"Normal     : mu={mu_n:.4f}, sigma={sig_n:.4f},   KS={ks_n:.4f}")
    print(f"Laplace    : mu={mu_l:.4f}, b={b_l:.4f},        KS={ks_l:.4f}")
    print(f"Exponential: lambda={lam_e:.4f} (mean={1.0/lam_e:.4f}), KS={ks_e:.4f}")

    if args.save_plots is not None:
        out_dir = Path(args.save_plots)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib
            matplotlib.use('Agg')  # non-interactive backend
            import matplotlib.pyplot as plt

            # dz histogram with PDFs
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            # dz
            ax0 = ax[0]
            ax0.hist(dz, bins=args.bins, density=True, alpha=0.6, color='C0')
            xs = np.linspace(0, max(float(np.max(dz)), 1e-3), 400)
            # PDFs
            def pdf_normal(x):
                return (1.0 / (sig_n * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x - mu_n) / sig_n) ** 2)
            def pdf_laplace(x):
                return (1.0 / (2.0 * b_l)) * np.exp(-np.abs(x - mu_l) / b_l)
            def pdf_expo(x):
                return lam_e * np.exp(-lam_e * np.maximum(x, 0.0))
            ax0.plot(xs, pdf_normal(xs), label=f'Normal(mu={mu_n:.2f},sig={sig_n:.2f})')
            ax0.plot(xs, pdf_laplace(xs), label=f'Laplace(mu={mu_l:.2f},b={b_l:.2f})')
            ax0.plot(xs, pdf_expo(xs), label=f'Exp(lam={lam_e:.2f})')
            ax0.set_title('dz above roof (density)')
            ax0.set_xlabel('dz (m)')
            ax0.set_ylabel('density')
            ax0.legend()

            # rho histogram
            ax1 = ax[1]
            ax1.hist(rho, bins=args.bins, density=True, alpha=0.6, color='C1')
            ax1.set_title('rho around vehicle (density)')
            ax1.set_xlabel('rho (m)')
            ax1.set_ylabel('density')

            fig.tight_layout()
            fig.savefig(out_dir / 'air_noise_distribution.png', dpi=150)
            plt.close(fig)
            print(f"Saved plots to {out_dir / 'air_noise_distribution.png'}")
        except Exception as e:
            print(f"Plotting failed: {e}")


if __name__ == '__main__':
    main()


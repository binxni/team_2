import argparse
import os
import sys
import math
import random
from typing import Dict, List, Tuple

import numpy as np


def safe_import_helpers():
    here = os.path.dirname(__file__)
    if here not in sys.path:
        sys.path.append(here)
    try:
        from analyze_pointcloud_noise import (
            compute_frame_metrics,
            parse_ranges,
            in_any_range,
            list_frame_indices,
        )
        return compute_frame_metrics, parse_ranges, in_any_range, list_frame_indices
    except Exception as e:
        raise RuntimeError(
            'Failed to import helpers from analyze_pointcloud_noise.py. '
            'Make sure it is present in the same directory.'
        ) from e


def js_divergence(x: np.ndarray, y: np.ndarray, bins: int = 60) -> float:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float('nan')
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return float('nan')
    P, _ = np.histogram(x, bins=bins, range=(lo, hi), density=True)
    Q, _ = np.histogram(y, bins=bins, range=(lo, hi), density=True)
    P = P + 1e-12
    Q = Q + 1e-12
    P = P / np.sum(P)
    Q = Q / np.sum(Q)
    M = 0.5 * (P + Q)
    def KL(A, B):
        return float(np.sum(A * (np.log(A) - np.log(B))))
    return 0.5 * (KL(P, M) + KL(Q, M))


def ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(x[np.isfinite(x)])
    y = np.sort(y[np.isfinite(y)])
    if x.size == 0 or y.size == 0:
        return float('nan')
    # Merge-walk ECDF
    i = j = 0
    nx, ny = x.size, y.size
    d = 0.0
    while i < nx and j < ny:
        if x[i] <= y[j]:
            i += 1
        else:
            j += 1
        Fx = i / nx
        Fy = j / ny
        d = max(d, abs(Fx - Fy))
    # Tail remainder
    d = max(d, abs(1.0 - j / ny))
    d = max(d, abs(i / nx - 1.0))
    return float(d)


def quantile_wasserstein(x: np.ndarray, y: np.ndarray, qn: int = 101) -> float:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float('nan')
    qs = np.linspace(0.0, 1.0, qn)
    qx = np.quantile(x, qs)
    qy = np.quantile(y, qs)
    return float(np.mean(np.abs(qx - qy)))


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float('nan')
    ma, mb = float(np.mean(a)), float(np.mean(b))
    sa = float(np.std(a, ddof=1) if a.size > 1 else 0.0)
    sb = float(np.std(b, ddof=1) if b.size > 1 else 0.0)
    n1, n2 = a.size, b.size
    sp = math.sqrt(((n1 - 1) * sa * sa + (n2 - 1) * sb * sb) / max(n1 + n2 - 2, 1))
    if sp == 0:
        return float('nan')
    return (ma - mb) / sp


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, scores))
    except Exception:
        # Simple fallback: rank-based AUC
        ranks = np.argsort(np.argsort(scores)) + 1
        r_pos = float(np.sum(ranks[y_true == 1]))
        n1 = int(np.sum(y_true == 1))
        n0 = int(np.sum(y_true == 0))
        if n1 == 0 or n0 == 0:
            return float('nan')
        U = r_pos - n1 * (n1 + 1) / 2.0
        return float(U / (n1 * n0))


def list_npy_indices(points_dir: str) -> List[str]:
    names = []
    for n in os.listdir(points_dir):
        if n.endswith('.npy'):
            names.append(n[:-4])
    names.sort()
    return names


def fmt(x: float) -> str:
    try:
        return f"{x:.6f}" if np.isfinite(x) else 'nan'
    except Exception:
        return 'nan'


def main():
    compute_frame_metrics, parse_ranges, in_any_range, list_frame_indices = safe_import_helpers()

    ap = argparse.ArgumentParser(description='Compare points_test noisy frames to LISA-generated frames quantitatively')
    ap.add_argument('--a-points-dir', type=str, default='OpenPCDet/data/custom_av/points_test')
    ap.add_argument('--a-noise-ranges', type=str, default='10006432-10009318,10100400-10104398')
    ap.add_argument('--b-points-dir', type=str, default='OpenPCDet/data/points_lisa')
    ap.add_argument('--num-a', type=int, default=150, help='Max frames from A(noisy points_test)')
    ap.add_argument('--num-b', type=int, default=150, help='Max frames from B(LISA)')
    ap.add_argument('--subsample', type=int, default=12000)
    ap.add_argument('--knn-k', type=int, default=16)
    ap.add_argument('--curvature-sample', type=int, default=2500)
    ap.add_argument('--ror-radius', type=float, default=0.5)
    ap.add_argument('--ror-min-pts', type=int, default=3)
    # Weather params (pass-through)
    ap.add_argument('--near-r', type=float, default=3.0)
    ap.add_argument('--far-r', type=float, default=50.0)
    ap.add_argument('--az-bins', type=int, default=90)
    ap.add_argument('--linearity-thresh', type=float, default=0.8)
    ap.add_argument('--radial-align-thresh', type=float, default=0.95)
    ap.add_argument('--near-air-z', type=float, default=1.5)
    ap.add_argument('--near-air-r', type=float, default=10.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-csv', type=str, default='OpenPCDet/output/points_test_vs_lisa.csv')
    ap.add_argument('--summary', type=str, default='OpenPCDet/output/points_test_vs_lisa_summary.csv')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Collect A noisy indices
    if not os.path.isdir(args.a_points_dir):
        raise SystemExit(f"A points dir not found: {args.a_points_dir}")
    if not os.path.isdir(args.b_points_dir):
        raise SystemExit(f"B points dir not found: {args.b_points_dir}")

    all_a = list_frame_indices(args.a_points_dir)
    noise_ranges = parse_ranges(args.a_noise_ranges) if args.a_noise_ranges else []
    a_idxs = [i for i in all_a if in_any_range(i, noise_ranges)]
    if len(a_idxs) == 0:
        raise SystemExit('No A noisy frames matched the provided --a-noise-ranges')
    if len(a_idxs) > args.num_a:
        a_idxs = random.sample(a_idxs, args.num_a)

    # Collect B indices (use all .npy stems; may be zero-padded strings)
    b_stems = list_npy_indices(args.b_points_dir)
    if len(b_stems) == 0:
        raise SystemExit('No B frames found')
    if len(b_stems) > args.num_b:
        b_stems = random.sample(b_stems, args.num_b)

    print(f"A noisy frames: {len(a_idxs)} from {args.a_points_dir}")
    print(f"B LISA frames:  {len(b_stems)} from {args.b_points_dir}")

    rows: List[Dict[str, float]] = []
    # A group label 0, B group label 1
    for idx in a_idxs:
        path = os.path.join(args.a_points_dir, f"{idx}.npy")
        try:
            pts = np.load(path)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue
        try:
            m = compute_frame_metrics(
                pts,
                subsample_points=args.subsample,
                knn_k=args.knn_k,
                curvature_sample=args.curvature_sample,
                ror_radius=args.ror_radius,
                ror_min_pts=args.ror_min_pts,
                near_r=args.near_r,
                far_r=args.far_r,
                az_bins=args.az_bins,
                linearity_thresh=args.linearity_thresh,
                radial_align_thresh=args.radial_align_thresh,
                near_air_z=args.near_air_z,
                near_air_r=args.near_air_r,
            )
        except Exception as e:
            print(f"Failed metrics for {path}: {e}")
            continue
        m['index'] = idx
        m['label'] = 0
        m['source'] = 'points_test'
        rows.append(m)

    for stem in b_stems:
        path = os.path.join(args.b_points_dir, f"{stem}.npy")
        try:
            pts = np.load(path)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            continue
        try:
            m = compute_frame_metrics(
                pts,
                subsample_points=args.subsample,
                knn_k=args.knn_k,
                curvature_sample=args.curvature_sample,
                ror_radius=args.ror_radius,
                ror_min_pts=args.ror_min_pts,
                near_r=args.near_r,
                far_r=args.far_r,
                az_bins=args.az_bins,
                linearity_thresh=args.linearity_thresh,
                radial_align_thresh=args.radial_align_thresh,
                near_air_z=args.near_air_z,
                near_air_r=args.near_air_r,
            )
        except Exception as e:
            print(f"Failed metrics for {path}: {e}")
            continue
        m['index'] = stem
        m['label'] = 1
        m['source'] = 'lisa'
        rows.append(m)

    if not rows:
        raise SystemExit('No metrics computed for either group')

    # Keys and arrays
    keys = [k for k in rows[0].keys() if k not in ('index', 'label', 'source')]
    y = np.array([r['label'] for r in rows], dtype=int)
    print("\n=== A(noisy points_test) vs B(LISA) similarity ===")
    print("metric,a_mean,b_mean,cohen_d,auc,auc_dist,jsd,ks,wqdiff")
    # Save per-frame CSV compatible with plot_noise_metrics
    out_csv = args.out_csv
    # Create parent dir only if provided (handle plain filenames)
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    import csv
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['index', 'label', 'source'] + keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in ['index', 'label', 'source'] + keys})
    # Compute and save summary CSV
    summary_path = args.summary
    summary_dir = os.path.dirname(summary_path)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    with open(summary_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'a_mean', 'b_mean', 'cohen_d', 'auc', 'auc_dist', 'jsd', 'ks', 'wqdiff'])
        for k in keys:
            a_vals = np.array([r[k] for r in rows if r['label'] == 0 and np.isfinite(r[k])], dtype=float)
            b_vals = np.array([r[k] for r in rows if r['label'] == 1 and np.isfinite(r[k])], dtype=float)
            if a_vals.size == 0 or b_vals.size == 0:
                continue
            d = cohen_d(b_vals, a_vals)  # positive means LISA > points_test_noisy
            scores = np.array([r[k] if np.isfinite(r[k]) else np.nan for r in rows])
            mask = np.isfinite(scores)
            auc = roc_auc(y[mask], scores[mask]) if np.any(mask) else float('nan')
            auc_dist = abs(auc - 0.5) if np.isfinite(auc) else float('nan')
            jsd = js_divergence(a_vals, b_vals)
            ks = ks_statistic(a_vals, b_vals)
            wq = quantile_wasserstein(a_vals, b_vals)
            a_mean = float(np.mean(a_vals))
            b_mean = float(np.mean(b_vals))
            print(f"{k},{fmt(a_mean)},{fmt(b_mean)},{fmt(d)},{fmt(auc)},{fmt(auc_dist)},{fmt(jsd)},{fmt(ks)},{fmt(wq)}")
            w.writerow([k, fmt(a_mean), fmt(b_mean), fmt(d), fmt(auc), fmt(auc_dist), fmt(jsd), fmt(ks), fmt(wq)])

    print(f"\nSaved per-frame metrics to: {out_csv}")
    print(f"Saved summary to: {summary_path}")
    print("Use plot_noise_metrics.py to visualize distributions (label 0=A, 1=B).")


if __name__ == '__main__':
    main()

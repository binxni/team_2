import argparse
import os
import sys
import math
import random
from typing import Dict, List, Tuple

import numpy as np


def parse_ranges(ranges_str: str) -> List[Tuple[int, int]]:
    """Parse a string like "10006432-10009318,10100400-10104398" into list of (start,end)."""
    if not ranges_str:
        return []
    parts = [p.strip() for p in ranges_str.split(',') if p.strip()]
    ranges: List[Tuple[int, int]] = []
    for p in parts:
        if '-' in p:
            a, b = p.split('-', 1)
            try:
                start, end = int(a), int(b)
            except ValueError:
                raise ValueError(f"Invalid range component: {p}")
            if end < start:
                start, end = end, start
            ranges.append((start, end))
        else:
            # single index
            try:
                idx = int(p)
            except ValueError:
                raise ValueError(f"Invalid index component: {p}")
            ranges.append((idx, idx))
    return ranges


def in_any_range(x: int, ranges: List[Tuple[int, int]]) -> bool:
    for a, b in ranges:
        if a <= x <= b:
            return True
    return False


def robust_zscore(values: np.ndarray) -> np.ndarray:
    # Median and MAD-based z-score (scaled to match std for normal dist)
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad == 0:
        return np.zeros_like(values)
    return 0.67448975 * (values - med) / mad


def compute_frame_metrics(
    pts: np.ndarray,
    *,
    subsample_points: int = 12000,
    knn_k: int = 16,
    curvature_sample: int = 2500,
    ror_radius: float = 0.5,
    ror_min_pts: int = 3,
    # Weather-specific params
    near_r: float = 3.0,
    far_r: float = 50.0,
    az_bins: int = 90,
    linearity_thresh: float = 0.8,
    radial_align_thresh: float = 0.95,
    near_air_z: float = 1.5,
    near_air_r: float = 10.0,
) -> Dict[str, float]:
    """Compute per-frame summary metrics indicative of noise.

    pts: (N,4) array with x,y,z,intensity
    Returns: dict of summary statistics.
    """
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("Expected points of shape (N, >=3)")

    N = pts.shape[0]
    # Basic sampling to bound compute
    if N > subsample_points:
        sel = np.random.choice(N, subsample_points, replace=False)
        P = pts[sel, :3]
        I = pts[sel, 3] if pts.shape[1] > 3 else None
    else:
        P = pts[:, :3]
        I = pts[:, 3] if pts.shape[1] > 3 else None

    # Range stats
    r = np.linalg.norm(P, axis=1)
    r_mean = float(np.mean(r))
    r_std = float(np.std(r))
    r_p95 = float(np.percentile(r, 95))
    r_p99 = float(np.percentile(r, 99))
    r_max = float(np.max(r))

    # Intensity stats (if available)
    intensity_mean = intensity_std = intensity_p1 = intensity_p99 = float('nan')
    if I is not None:
        intensity_mean = float(np.mean(I))
        intensity_std = float(np.std(I))
        intensity_p1 = float(np.percentile(I, 1))
        intensity_p99 = float(np.percentile(I, 99))

    # Nearest neighbors distances and outlier fraction
    try:
        from sklearn.neighbors import NearestNeighbors, BallTree
    except Exception as e:
        raise RuntimeError("scikit-learn is required for this analysis") from e

    # Build NN structure once
    knn = NearestNeighbors(n_neighbors=min(knn_k + 1, len(P)), algorithm='kd_tree')
    knn.fit(P)
    dists, _ = knn.kneighbors(P, n_neighbors=min(knn_k + 1, len(P)), return_distance=True)
    if dists.shape[1] > 1:
        local_mean_dist = np.mean(dists[:, 1:], axis=1)  # exclude self-distance
    else:
        local_mean_dist = dists[:, 0]

    # Robust outlier score based on local mean distance
    z = robust_zscore(local_mean_dist)
    outlier_fraction = float(np.mean(np.abs(z) > 3.5))
    lmd_mean = float(np.mean(local_mean_dist))
    lmd_p95 = float(np.percentile(local_mean_dist, 95))
    lmd_p99 = float(np.percentile(local_mean_dist, 99))

    # Radius outlier removal proxy: fraction of points with < ror_min_pts neighbors within radius
    try:
        bt = BallTree(P)
        # Sample to reduce cost if needed
        if len(P) > 8000:
            ridx = np.random.choice(len(P), 8000, replace=False)
            Q = P[ridx]
        else:
            Q = P
        ind = bt.query_radius(Q, r=ror_radius, count_only=False)
        # ind is list of arrays of indices; convert to counts
        counts = np.array([len(ix) for ix in ind])
        ror_fraction = float(np.mean(counts < ror_min_pts))
    except Exception:
        ror_fraction = float('nan')

    # Curvature via neighborhood covariance eigenvalues
    # Sample a subset of points for curvature to keep runtime reasonable
    curv_p95 = float('nan')
    curv_mean = float('nan')
    linearity_mean = float('nan')
    radial_linear_aligned_ratio = float('nan')
    try:
        if len(P) >= 16 and curvature_sample > 0:
            m = min(curvature_sample, len(P))
            cidx = np.random.choice(len(P), m, replace=False)
            # Use existing neighbor graph to fetch neighbors up to knn_k
            # Recompute with sufficient neighbors for curvature
            k_curv = min(max(knn_k, 16), len(P))
            d2, n2 = knn.kneighbors(P[cidx], n_neighbors=k_curv, return_distance=True)
            curvs = []
            lin_vals = []
            radial_align_flags = []
            for i in range(m):
                nbrs = P[n2[i]]
                # Center and compute covariance
                C = np.cov(nbrs.T)
                # Numerical guard
                if not np.isfinite(C).all():
                    continue
                w, V = np.linalg.eigh(C)
                # sort descending
                order = np.argsort(w)[::-1]
                w = np.maximum(w[order], 1e-12)
                V = V[:, order]
                # Use standard curvature definition: lambda_min / (lambda_sum)
                lam_sum = float(np.sum(w))
                if lam_sum <= 0:
                    continue
                # after descending sort, lambda_min is w[-1]
                curv = float(w[-1] / lam_sum)
                curvs.append(curv)
                # linearity = (lambda1 - lambda2) / lambda1
                lin = float((w[0] - w[1]) / w[0]) if w[0] > 0 else 0.0
                lin_vals.append(lin)
                # radial alignment of principal direction
                v1 = V[:, 0]
                p0 = P[cidx[i]]
                nr = np.linalg.norm(p0)
                if nr > 0:
                    rhat = p0 / nr
                    align = float(abs(np.dot(v1, rhat)))
                    radial_align_flags.append(1.0 if (lin >= linearity_thresh and align >= radial_align_thresh) else 0.0)
            if curvs:
                curvs = np.array(curvs)
                curv_mean = float(np.mean(curvs))
                curv_p95 = float(np.percentile(curvs, 95))
            if lin_vals:
                linearity_mean = float(np.mean(lin_vals))
            if radial_align_flags:
                radial_linear_aligned_ratio = float(np.mean(radial_align_flags))
    except Exception:
        pass

    # Weather-oriented additional metrics
    # Range-intensity slope and R^2
    intensity_range_slope = float('nan')
    intensity_range_r2 = float('nan')
    if I is not None and len(P) >= 10:
        try:
            # Fit I = a * r + b
            coef = np.polyfit(r, I, 1)
            pred = np.polyval(coef, r)
            ss_res = float(np.sum((I - pred) ** 2))
            ss_tot = float(np.sum((I - np.mean(I)) ** 2))
            intensity_range_slope = float(coef[0])
            intensity_range_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        except Exception:
            pass

    # Near backscatter low-intensity ratio
    near_backscatter_lowI_ratio = float('nan')
    if I is not None:
        try:
            lowI_th = float(np.percentile(I, 10.0))
            mask_near = r <= near_r
            if np.any(mask_near):
                near_backscatter_lowI_ratio = float(np.mean(I[mask_near] <= lowI_th))
        except Exception:
            pass

    # Far range proportion (dropout indicator)
    far_range_ratio = float(np.mean(r > far_r)) if len(r) > 0 else float('nan')

    # Air ratio near sensor (points above certain z within near radius)
    near_air_ratio = float('nan')
    try:
        mask_near = r <= near_air_r
        if np.any(mask_near):
            near_air_ratio = float(np.mean(P[mask_near, 2] > near_air_z))
    except Exception:
        pass

    # Azimuthal uniformity of outliers (lower CV => more uniform, typical for weather)
    az_outlier_cv = float('nan')
    az_outlier_gini = float('nan')
    try:
        az = np.arctan2(P[:, 1], P[:, 0])  # [-pi, pi]
        # outlier mask from z above
        out_mask = np.abs(z) > 3.5
        if az_bins < 4:
            az_bins = 4
        edges = np.linspace(-np.pi, np.pi, az_bins + 1)
        counts = np.zeros(az_bins, dtype=np.float64)
        totals = np.zeros(az_bins, dtype=np.float64)
        idx_bin = np.digitize(az, edges) - 1
        idx_bin = np.clip(idx_bin, 0, az_bins - 1)
        for b in range(az_bins):
            m = idx_bin == b
            totals[b] = float(np.sum(m))
            if totals[b] > 0:
                counts[b] = float(np.sum(out_mask[m])) / totals[b]
        valid = totals > 0
        vals = counts[valid]
        if vals.size > 1 and np.mean(vals) > 0:
            az_outlier_cv = float(np.std(vals) / np.mean(vals))
            # Gini coefficient
            x = np.sort(vals)
            n = x.size
            cumx = np.cumsum(x)
            g = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n if cumx[-1] > 0 else 0.0
            az_outlier_gini = float(g)
    except Exception:
        pass

    return {
        'n_points': float(N),
        'range_mean': r_mean,
        'range_std': r_std,
        'range_p95': r_p95,
        'range_p99': r_p99,
        'range_max': r_max,
        'intensity_mean': intensity_mean,
        'intensity_std': intensity_std,
        'intensity_p1': intensity_p1,
        'intensity_p99': intensity_p99,
        'knn_mean_dist_mean': lmd_mean,
        'knn_mean_dist_p95': lmd_p95,
        'knn_mean_dist_p99': lmd_p99,
        'outlier_fraction': outlier_fraction,
        'ror_fraction': ror_fraction,
        'curv_mean': curv_mean,
        'curv_p95': curv_p95,
        # Added
        'linearity_mean': linearity_mean,
        'radial_linear_aligned_ratio': radial_linear_aligned_ratio,
        'intensity_range_slope': intensity_range_slope,
        'intensity_range_r2': intensity_range_r2,
        'near_backscatter_lowI_ratio': near_backscatter_lowI_ratio,
        'far_range_ratio': far_range_ratio,
        'near_air_ratio': near_air_ratio,
        'az_outlier_cv': az_outlier_cv,
        'az_outlier_gini': az_outlier_gini,
    }


def list_frame_indices(points_dir: str) -> List[int]:
    idxs = []
    for name in os.listdir(points_dir):
        if not name.endswith('.npy'):
            continue
        stem = name[:-4]
        try:
            idxs.append(int(stem))
        except ValueError:
            continue
    idxs.sort()
    return idxs


def cohen_d(a: List[float], b: List[float]) -> float:
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    ma, mb = float(np.mean(a)), float(np.mean(b))
    sa, sb = float(np.std(a, ddof=1) if len(a) > 1 else 0.0), float(np.std(b, ddof=1) if len(b) > 1 else 0.0)
    n1, n2 = len(a), len(b)
    # Pooled std
    s_p = math.sqrt(((n1 - 1) * sa * sa + (n2 - 1) * sb * sb) / max(n1 + n2 - 2, 1))
    if s_p == 0:
        return float('nan')
    return (ma - mb) / s_p


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, scores))
    except Exception:
        # Fallback simple AUC via Mann-Whitney U statistic
        pos = scores[y_true == 1]
        neg = scores[y_true == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float('nan')
        ranks = np.argsort(np.argsort(scores)) + 1
        r_pos = np.sum(ranks[y_true == 1])
        n1, n0 = len(pos), len(neg)
        U = r_pos - n1 * (n1 + 1) / 2
        return float(U / (n1 * n0))


def main():
    ap = argparse.ArgumentParser(description='Quantitative noise-vs-clean analysis for point clouds')
    ap.add_argument('--points-dir', type=str, default='OpenPCDet/data/custom_av/points_test', help='Directory with .npy point clouds')
    ap.add_argument('--noise-ranges', type=str, default='10006432-10009318,10100400-10104398', help='Comma-separated index ranges for noisy frames')
    ap.add_argument('--num-noise', type=int, default=120, help='Max noisy frames to sample')
    ap.add_argument('--num-clean', type=int, default=120, help='Max clean frames to sample')
    ap.add_argument('--subsample', type=int, default=12000, help='Points per frame to sample')
    ap.add_argument('--knn-k', type=int, default=16)
    ap.add_argument('--curvature-sample', type=int, default=2500)
    ap.add_argument('--ror-radius', type=float, default=0.5)
    ap.add_argument('--ror-min-pts', type=int, default=3)
    # Weather-specific params
    ap.add_argument('--near-r', type=float, default=3.0)
    ap.add_argument('--far-r', type=float, default=50.0)
    ap.add_argument('--az-bins', type=int, default=90)
    ap.add_argument('--linearity-thresh', type=float, default=0.8)
    ap.add_argument('--radial-align-thresh', type=float, default=0.95)
    ap.add_argument('--near-air-z', type=float, default=1.5)
    ap.add_argument('--near-air-r', type=float, default=10.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--csv', type=str, default='', help='Optional path to save per-frame metrics CSV')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.isdir(args.points_dir):
        print(f"Points dir not found: {args.points_dir}", file=sys.stderr)
        sys.exit(1)

    all_idxs = list_frame_indices(args.points_dir)
    if not all_idxs:
        print("No frames found.", file=sys.stderr)
        sys.exit(1)

    noise_ranges = parse_ranges(args.noise_ranges) if args.noise_ranges is not None else []

    if not noise_ranges:
        # Unsupervised mode: sample frames and rank by noise-like metrics
        total = min(len(all_idxs), args.num_noise + args.num_clean)
        sample = random.sample(all_idxs, total)
        print(f"Unsupervised mode: sampling {len(sample)} frames (no noise ranges provided)")
        rows = []
        for idx in sample:
            path = os.path.join(args.points_dir, f"{idx}.npy")
            try:
                pts = np.load(path)
            except Exception as e:
                print(f"Failed to load {path}: {e}", file=sys.stderr)
                continue
            try:
                metrics = compute_frame_metrics(
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
                print(f"Failed metrics for {path}: {e}", file=sys.stderr)
                continue
            metrics['index'] = idx
            rows.append(metrics)

        if not rows:
            print("No metrics computed.", file=sys.stderr)
            sys.exit(1)

        # Rank frames by several noise-sensitive metrics
        def topk(key: str, k: int = 10):
            srtd = sorted(rows, key=lambda r: (float('-inf') if not np.isfinite(r.get(key, float('nan'))) else r[key]), reverse=True)
            return srtd[:k]

        print("\nTop frames by outlier_fraction:")
        for r in topk('outlier_fraction'):
            print(f"index={r['index']} outlier_fraction={r['outlier_fraction']:.4f} knn_mean_dist_p99={r['knn_mean_dist_p99']:.4f} ror_fraction={r['ror_fraction']:.4f}")

        print("\nTop frames by knn_mean_dist_p99:")
        for r in topk('knn_mean_dist_p99'):
            print(f"index={r['index']} knn_mean_dist_p99={r['knn_mean_dist_p99']:.4f} outlier_fraction={r['outlier_fraction']:.4f}")

        print("\nTop frames by ror_fraction:")
        for r in topk('ror_fraction'):
            print(f"index={r['index']} ror_fraction={r['ror_fraction']:.4f} outlier_fraction={r['outlier_fraction']:.4f}")

        if args.csv:
            import csv
            keys = [k for k in rows[0].keys() if k != 'index']
            with open(args.csv, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['index'] + keys)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, '') for k in ['index'] + keys})
            print(f"\nSaved per-frame metrics to: {args.csv}")
        return

    # Supervised (weakly) mode using provided noise ranges
    noise_idxs = [i for i in all_idxs if in_any_range(i, noise_ranges)]
    clean_idxs = [i for i in all_idxs if not in_any_range(i, noise_ranges)]

    # Sample subsets for speed
    noise_sample = noise_idxs if len(noise_idxs) <= args.num_noise else random.sample(noise_idxs, args.num_noise)
    clean_sample = clean_idxs if len(clean_idxs) <= args.num_clean else random.sample(clean_idxs, args.num_clean)

    print(f"Frames: total={len(all_idxs)} noisy={len(noise_idxs)} clean={len(clean_idxs)}")
    print(f"Sampling: noisy={len(noise_sample)} clean={len(clean_sample)}")

    rows = []  # per-frame metrics
    for label, idxs in [(1, noise_sample), (0, clean_sample)]:
        for idx in idxs:
            path = os.path.join(args.points_dir, f"{idx}.npy")
            try:
                pts = np.load(path)
            except Exception as e:
                print(f"Failed to load {path}: {e}", file=sys.stderr)
                continue
            try:
                metrics = compute_frame_metrics(
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
                print(f"Failed metrics for {path}: {e}", file=sys.stderr)
                continue
            metrics['index'] = idx
            metrics['label'] = label
            rows.append(metrics)

    if not rows:
        print("No metrics computed.", file=sys.stderr)
        sys.exit(1)

    # Aggregate and compare
    keys = [k for k in rows[0].keys() if k not in ('index', 'label')]
    y = np.array([r['label'] for r in rows], dtype=np.int64)
    print("\n=== Metric Separation (noisy vs clean) ===")
    print("metric, noisy_mean, clean_mean, cohen_d, auc")
    for k in keys:
        nv = [r[k] for r in rows if r['label'] == 1 and np.isfinite(r[k])]
        cv = [r[k] for r in rows if r['label'] == 0 and np.isfinite(r[k])]
        if len(nv) == 0 or len(cv) == 0:
            continue
        d = cohen_d(nv, cv)
        # Use metric as score (higher presumed more noisy) for AUC
        scores = np.array([r[k] if np.isfinite(r[k]) else np.nan for r in rows])
        mask = np.isfinite(scores)
        auc = roc_auc(y[mask], scores[mask]) if mask.any() else float('nan')
        print(f"{k}, {np.mean(nv):.5f}, {np.mean(cv):.5f}, {d:.3f}, {auc:.3f}")

    # Optionally save per-frame CSV
    if args.csv:
        import csv
        fieldnames = ['index', 'label'] + keys
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, '') for k in fieldnames})
        print(f"\nSaved per-frame metrics to: {args.csv}")


if __name__ == '__main__':
    main()

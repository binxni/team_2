#!/usr/bin/env python3
"""
Check Gaussianity of point clouds across specified frame ID ranges.

- Loads .npy point clouds from a directory (e.g., OpenPCDet/data/custom_av/points_test)
- Selects frames whose numeric stem lies within provided ranges
- Reservoir-samples up to --max-points points over all selected frames
- Tests:
    * Per-axis normality (D’Agostino K^2 if SciPy is available, otherwise Jarque–Bera)
    * Multivariate (3D) normality via Mahalanobis distances D^2;
      reports mean/variance vs theoretical (k, 2k), KS vs Chi-square if SciPy available,
      and QQ correlation (sorted D^2 vs Chi-square quantiles)

Usage example:
  python OpenPCDet/tools/check_gaussianity_point_ranges.py \
      --points-dir OpenPCDet/data/custom_av/points_test \
      --ranges 10006432-10009318,10100400-10104398 \
      --cols 0,1,2 --max-points 500000
"""

from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def parse_ranges(ranges_str: str) -> List[Tuple[int, int]]:
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


def list_frames(points_dir: Path, ranges: Sequence[Tuple[int, int]]) -> List[int]:
    frames: List[int] = []
    for name in os.listdir(points_dir):
        if not name.endswith('.npy'):
            continue
        stem = name[:-4]
        try:
            fid = int(stem)
        except ValueError:
            continue
        if in_any_range(fid, ranges):
            frames.append(fid)
    frames.sort()
    return frames


def reservoir_sample_points(paths: List[Path], cols: Sequence[int], max_points: int, seed: int = 12345) -> np.ndarray:
    rng = random.Random(seed)
    reservoir: Optional[np.ndarray] = None
    total_seen = 0
    for p in paths:
        try:
            arr = np.load(p)
        except Exception:
            continue
        if arr.ndim != 2 or arr.shape[1] <= max(cols):
            continue
        pts = arr[:, cols]
        n = pts.shape[0]
        if n == 0:
            continue
        if reservoir is None:
            take = min(n, max_points)
            idx = np.arange(n)
            rng.shuffle(idx)
            idx = idx[:take]
            reservoir = pts[idx].copy()
            total_seen = take
            continue
        # Fill until reservoir is full, then do standard reservoir sampling
        if reservoir.shape[0] < max_points:
            rem = max_points - reservoir.shape[0]
            take = min(n, rem)
            idx = np.arange(n)
            rng.shuffle(idx)
            reservoir = np.vstack([reservoir, pts[idx[:take]]])
            total_seen += take
            n_remaining = n - take
            if n_remaining <= 0:
                continue
            # now perform reservoir on the rest
            start = take
        else:
            start = 0

        # Reservoir sampling for the remaining points
        # For j-th item seen (1-based), replace a random reservoir item with probability max_points / j
        if start < n:
            for j in range(start, n):
                total_seen += 1
                if rng.random() < (max_points / float(total_seen)):
                    k = rng.randrange(max_points)
                    reservoir[k] = pts[j]
    return reservoir if reservoir is not None else np.empty((0, len(cols)), dtype=np.float32)


def normality_test_1d(x: np.ndarray) -> Tuple[str, float, Optional[float]]:
    """Return (test_name, stat, pvalue_or_None). Uses SciPy if available, else JB."""
    x = np.asarray(x)
    n = x.size
    if n < 8:
        return ('insufficient_n', float('nan'), None)
    try:
        from scipy.stats import normaltest  # type: ignore
        stat, p = normaltest(x)
        return ('dagostino_k2', float(stat), float(p))
    except Exception:
        # Jarque–Bera
        m = float(np.mean(x))
        s = float(np.std(x, ddof=1)) if n > 1 else float('nan')
        if (not np.isfinite(s)) or s <= 0:
            return ('jarque_bera', float('nan'), None)
        z = (x - m) / s
        skew = float(np.mean(z ** 3))
        kurt = float(np.mean(z ** 4))
        jb = n / 6.0 * (skew ** 2 + (kurt - 3.0) ** 2 / 4.0)
        p: Optional[float] = None
        try:
            from scipy.stats import chi2  # type: ignore
            p = float(chi2.sf(jb, 2))
        except Exception:
            p = None
        return ('jarque_bera', float(jb), p)


def multivariate_gaussianity(X: np.ndarray, diagonal_only: bool = True) -> dict:
    X = np.asarray(X)
    n, k = X.shape
    mu = X.mean(axis=0)
    Xc = X - mu
    if diagonal_only:
        var = Xc.var(axis=0, ddof=1)
        var[var <= 1e-12] = 1e-12
        z = Xc / np.sqrt(var)
        d2 = np.sum(z * z, axis=1)
    else:
        cov = np.cov(Xc, rowvar=False)
        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(cov)
        d2 = np.einsum('ni,ij,nj->n', Xc, inv, Xc)
    out = {
        'n': int(n), 'k': int(k),
        'mean_d2': float(np.mean(d2)) if n > 0 else float('nan'),
        'var_d2': float(np.var(d2, ddof=1)) if n > 1 else float('nan'),
        'ks_D': None, 'ks_p': None, 'qq_corr': None,
    }
    # Try SciPy K-S vs Chi-square
    try:
        from scipy.stats import chi2, kstest  # type: ignore
        ks_stat, ks_p = kstest(d2, 'chi2', args=(k,))
        out['ks_D'] = float(ks_stat)
        out['ks_p'] = float(ks_p)
        xs = np.sort(d2)
        q = chi2.ppf((np.arange(1, n + 1) - 0.5) / n, df=k)
        out['qq_corr'] = float(np.corrcoef(xs, q)[0, 1])
    except Exception:
        # Compute QQ corr against approximate chi-square quantiles using normal approximation if SciPy missing
        try:
            xs = np.sort(d2)
            # Wilson-Hilferty transform approx: if Z~N(0,1), then X≈k*(1 - 2/(9k) + Z*sqrt(2/(9k)))^3
            m = np.arange(1, n + 1)
            p = (m - 0.5) / n
            # Approximate normal quantiles via inverse error function
            z = np.sqrt(2.0) * erfinv_clamped(2 * p - 1)
            q = k * (1 - 2 / (9 * k) + z * math.sqrt(2 / (9 * k))) ** 3
            out['qq_corr'] = float(np.corrcoef(xs, q)[0, 1])
        except Exception:
            pass
    return out


def erfinv_clamped(y: np.ndarray) -> np.ndarray:
    # Approximate inverse error function via scipy fallback or numpy polynomial approx
    try:
        from scipy.special import erfinv  # type: ignore
        return erfinv(y)
    except Exception:
        # Winitzki approximation for erfinv
        # erfinv(y) ≈ sign(y) * sqrt( sqrt( (2/(pi*a) + ln(1-y^2)/2)^2 - ln(1-y^2)/a ) - (2/(pi*a) + ln(1-y^2)/2) )
        a = 0.147
        y = np.clip(y, -0.999999, 0.999999)
        s = np.sign(y)
        ln = np.log(1 - y * y)
        first = 2 / (math.pi * a) + ln / 2
        inside = first * first - ln / a
        return s * np.sqrt(np.sqrt(inside) - first)


def main() -> None:
    ap = argparse.ArgumentParser(description='Check Gaussianity over frame ranges')
    ap.add_argument('--points-dir', type=Path, required=True, help='Directory with .npy point clouds')
    ap.add_argument('--ranges', type=str, required=True, help='Comma-separated frame ranges like 10006432-10009318,10100400-10104398')
    ap.add_argument('--cols', type=str, default='0,1,2', help='Comma-separated column indices for XYZ (default: 0,1,2)')
    ap.add_argument('--max-points', type=int, default=500000, help='Max total sampled points')
    ap.add_argument('--max-frames', type=int, default=400, help='Optional cap on number of frames to scan (sampled)')
    ap.add_argument('--frame-seed', type=int, default=2025, help='Random seed for frame sampling')
    ap.add_argument('--seed', type=int, default=12345, help='Random seed')
    ap.add_argument('--diagonal-only', action='store_true', help='Use diagonal covariance only (avoid heavy linalg)')
    args = ap.parse_args()

    cols = tuple(int(x.strip()) for x in args.cols.split(',') if x.strip())
    ranges = parse_ranges(args.ranges)
    frames = list_frames(args.points_dir, ranges)
    print(f"Found {len(frames)} frames in ranges {ranges} under {args.points_dir}")
    if args.max_frames is not None and args.max_frames > 0 and len(frames) > args.max_frames:
        rng = np.random.default_rng(args.frame_seed)
        sel_idx = rng.choice(len(frames), size=args.max_frames, replace=False)
        sel_idx.sort()
        frames = [frames[i] for i in sel_idx.tolist()]
        print(f"Sampling {len(frames)} frames for analysis (max_frames={args.max_frames})")
    if not frames:
        return
    paths = [args.points_dir / f"{fid}.npy" for fid in frames]
    sample = reservoir_sample_points(paths, cols, max_points=args.max_points, seed=args.seed)
    print(f"Sampled points: {sample.shape}")
    if sample.size == 0:
        print('No points sampled. Check directory and ranges.')
        return

    # Univariate tests
    for i, c in enumerate(cols):
        x = sample[:, i]
        tname, stat, p = normality_test_1d(x)
        mean, std = float(np.mean(x)), float(np.std(x, ddof=1))
        p_str = f"{p:.6f}" if (p is not None and np.isfinite(p)) else 'NA'
        print(f"Axis col={c}: mean={mean:.5f} std={std:.5f} | {tname} stat={stat:.5f} p={p_str}")

    # Multivariate
    mv = multivariate_gaussianity(sample, diagonal_only=args.diagonal_only or True)
    k = mv['k']
    print("--- Multivariate (Mahalanobis D^2) ---")
    print(f"n={mv['n']} k={k} | mean_d2={mv['mean_d2']:.4f} (target {k}) var_d2={mv['var_d2']:.4f} (target {2*k})")
    if mv['ks_p'] is not None:
        print(f"KS vs Chi2(df={k}): D={mv['ks_D']:.5f} p={mv['ks_p']:.6f}")
    if mv['qq_corr'] is not None:
        print(f"QQ corr(D^2, Chi2): {mv['qq_corr']:.6f}")


if __name__ == '__main__':
    main()

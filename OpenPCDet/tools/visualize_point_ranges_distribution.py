#!/usr/bin/env python3
"""
Visualize distributions for points in given frame ID ranges.

- Loads .npy point clouds under --points-dir
- Selects frames within --ranges (e.g., 10006432-10009318,10100400-10104398)
- Uniformly samples up to --max-frames frames and --max-points total points
- Saves plots to --out-dir:
    * axis_hist.png      : histograms for X,Y,Z with Normal(mu,sigma) PDF overlay
    * axis_qq.png        : QQ plots X,Y,Z vs Normal(mu,sigma)
    * pairs_hist2d.png   : 2D hist2d for (X,Y), (X,Z), (Y,Z)
    * d2_hist.png        : Histogram of diagonal Mahalanobis D^2 with Chi-square(df=3) PDF overlay
    * summary.txt        : Basic stats and JB-like metrics (no SciPy needed)

This script avoids heavy linear algebra and GUI backends.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import List, Sequence, Tuple

# Minimize BLAS threads to avoid sandbox SHM issues
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

# Matplotlib is optional; default to a lightweight PGM backend to avoid SHM issues.
try:
    import matplotlib  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    _MATPLOTLIB_OK = True
except Exception:
    _MATPLOTLIB_OK = False


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
        try:
            fid = int(name[:-4])
        except ValueError:
            continue
        if in_any_range(fid, ranges):
            frames.append(fid)
    frames.sort()
    return frames


def reservoir_sample(paths: List[Path], cols=(0, 1, 2), max_points: int = 120_000, seed: int = 12345) -> np.ndarray:
    rng = np.random.default_rng(seed)
    reservoir: np.ndarray | None = None
    filled = 0
    seen = 0
    for p in paths:
        try:
            arr = np.load(p)
        except Exception:
            continue
        if arr.ndim != 2 or arr.shape[1] <= max(cols):
            continue
        pts = arr[:, cols]
        n = pts.shape[0]
        if reservoir is None:
            take = min(n, max_points)
            idx = rng.permutation(n)[:take]
            reservoir = pts[idx].copy()
            filled = take
            seen = take
            continue
        # Fill first
        if filled < max_points:
            rem = max_points - filled
            take = min(n, rem)
            idx = rng.permutation(n)[:take]
            reservoir = np.vstack([reservoir, pts[idx]])
            filled += take
            seen += take
            if take == n:
                continue
            start = take
        else:
            start = 0
        # Reservoir sampling for remainder
        for j in range(start, n):
            seen += 1
            if rng.random() < (max_points / float(seen)):
                k = rng.integers(0, max_points)
                reservoir[k] = pts[j]
    if reservoir is None:
        return np.empty((0, len(cols)), dtype=np.float32)
    return reservoir


def jb_statistics(x: np.ndarray) -> Tuple[float, float, float, float]:
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    z = (x - m) / (s if s > 0 else 1.0)
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))
    n = x.size
    jb = n / 6.0 * (skew ** 2 + (kurt - 3.0) ** 2 / 4.0)
    return m, s, skew, kurt, jb


def erfinv_clamped(y: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import erfinv  # type: ignore
        return erfinv(y)
    except Exception:
        a = 0.147
        y = np.clip(y, -0.999999, 0.999999)
        s = np.sign(y)
        ln = np.log(1 - y * y)
        first = 2 / (math.pi * a) + ln / 2
        inside = first * first - ln / a
        return s * np.sqrt(np.sqrt(inside) - first)


def plot_axis_hist_matplotlib(sample: np.ndarray, out_path: Path, bins: int = 120) -> None:
    mu = sample.mean(axis=0)
    sd = sample.std(axis=0, ddof=1)
    xs = [sample[:, i] for i in range(sample.shape[1])]
    matplotlib.use('Agg')
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for i, a in enumerate(ax):
        a.hist(xs[i], bins=bins, density=True, alpha=0.6, color=f'C{i}')
        # Overlay Normal(mu, sd)
        xg = np.linspace(mu[i] - 4 * sd[i], mu[i] + 4 * sd[i], 400)
        pdf = (1.0 / (sd[i] * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((xg - mu[i]) / sd[i]) ** 2)
        a.plot(xg, pdf, 'k-', lw=1.5, label=f'N({mu[i]:.2f},{sd[i]:.2f})')
        a.set_title(f'Axis {i} hist')
        a.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_axis_qq_matplotlib(sample: np.ndarray, out_path: Path) -> None:
    matplotlib.use('Agg')
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    n = sample.shape[0]
    p = (np.arange(1, n + 1) - 0.5) / n
    zq = np.sqrt(2.0) * erfinv_clamped(2 * p - 1)
    for i, a in enumerate(ax):
        x = np.sort(sample[:, i])
        mu = float(np.mean(sample[:, i]))
        sd = float(np.std(sample[:, i], ddof=1))
        theo = mu + sd * zq
        a.plot(theo, x, '.', ms=2)
        # reference line
        lo, hi = np.percentile(theo, [25, 75])
        lo2, hi2 = np.percentile(x, [25, 75])
        slope = (hi2 - lo2) / (hi - lo + 1e-12)
        intercept = lo2 - slope * lo
        xx = np.linspace(theo.min(), theo.max(), 50)
        a.plot(xx, slope * xx + intercept, 'r-', lw=1)
        a.set_title(f'Axis {i} QQ')
        a.set_xlabel('Theoretical quantiles')
        a.set_ylabel('Sample quantiles')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pairs_hist2d_matplotlib(sample: np.ndarray, out_path: Path, bins: int = 200) -> None:
    matplotlib.use('Agg')
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    pairs = [(0, 1), (0, 2), (1, 2)]
    titles = ['X-Y', 'X-Z', 'Y-Z']
    for i, (a_idx, b_idx) in enumerate(pairs):
        h = ax[i].hist2d(sample[:, a_idx], sample[:, b_idx], bins=bins, cmap='viridis')
        fig.colorbar(h[3], ax=ax[i])
        ax[i].set_title(titles[i])
        ax[i].set_xlabel(f'Axis {a_idx}')
        ax[i].set_ylabel(f'Axis {b_idx}')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_d2_hist_matplotlib(sample: np.ndarray, out_path: Path, bins: int = 160) -> None:
    mu = sample.mean(axis=0)
    var = sample.var(axis=0, ddof=1)
    var[var <= 1e-12] = 1e-12
    z = (sample - mu) / np.sqrt(var)
    d2 = np.sum(z * z, axis=1)
    matplotlib.use('Agg')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(d2, bins=bins, density=True, alpha=0.6, color='C3')
    # Overlay Chi-square(df=3) pdf
    k = 3.0
    xg = np.linspace(0, np.percentile(d2, 99.9), 500)
    # pdf = 1/(2^{k/2} Gamma(k/2)) x^{k/2-1} e^{-x/2}
    coef = 1.0 / (2 ** (k / 2.0) * math.gamma(k / 2.0))
    pdf = coef * (xg ** (k / 2.0 - 1.0)) * np.exp(-xg / 2.0)
    ax.plot(xg, pdf, 'k-', lw=1.5, label='Chi-square(df=3)')
    ax.set_title('Diagonal Mahalanobis D^2')
    ax.set_xlabel('D^2')
    ax.set_ylabel('density')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_pgm(path: Path, img: np.ndarray) -> None:
    """Write a grayscale PGM image (P5). img must be uint8 HxW."""
    assert img.dtype == np.uint8 and img.ndim == 2
    h, w = img.shape
    with open(path, 'wb') as f:
        header = f'P5\n{w} {h}\n255\n'.encode('ascii')
        f.write(header)
        f.write(img.tobytes())


def plot_axis_hist_simple(sample: np.ndarray, out_path: Path, bins: int = 200, size: Tuple[int, int] = (220, 720)) -> None:
    H, W = size
    img = np.full((H, W), 255, dtype=np.uint8)
    cols = sample.shape[1]
    panel_w = W // cols
    for i in range(cols):
        x = sample[:, i]
        hist, edges = np.histogram(x, bins=bins)
        hist = hist.astype(np.float64)
        hist /= hist.max() + 1e-12
        bw = max(1, (panel_w - 20) // bins)
        for b, h in enumerate(hist):
            x0 = i * panel_w + 10 + b * bw
            x1 = min(i * panel_w + panel_w - 10, x0 + bw)
            hpx = int(h * (H - 30))
            img[H - 10 - hpx:H - 10, x0:x1] = 30
        # axis label bar
        img[H - 9:H - 7, i * panel_w + 10:i * panel_w + panel_w - 10] = 0
    write_pgm(out_path, img)


def plot_axis_qq_simple(sample: np.ndarray, out_path: Path, size: Tuple[int, int] = (640, 1920)) -> None:
    H, W = size
    img = np.full((H, W), 255, dtype=np.uint8)
    cols = sample.shape[1]
    panel_w = W // cols
    n = sample.shape[0]
    p = (np.arange(1, n + 1) - 0.5) / n
    zq = np.sqrt(2.0) * erfinv_clamped(2 * p - 1)
    for i in range(cols):
        x = np.sort(sample[:, i])
        mu = float(np.mean(sample[:, i]))
        sd = float(np.std(sample[:, i], ddof=1))
        theo = mu + sd * zq
        # Normalize to panel
        tmin, tmax = float(theo.min()), float(theo.max())
        xmin, xmax = float(x.min()), float(x.max())
        def norm(v, lo, hi, L):
            return np.clip((v - lo) / (hi - lo + 1e-12) * (L - 20) + 10, 10, L - 10)
        tx = norm(theo, tmin, tmax, panel_w)
        ty = norm(x, xmin, xmax, H)
        # Draw points
        for j in range(min(n, 6000)):
            cx = int(i * panel_w + tx[j])
            cy = int(H - ty[j])
            img[max(0, cy - 0):min(H, cy + 1), max(0, cx - 0):min(i * panel_w + panel_w, cx + 1)] = 0
        # Diagonal reference
        for u in range(panel_w - 20):
            cx = i * panel_w + 10 + u
            cy = int(H - (10 + u * (H - 20) / (panel_w - 20)))
            if 0 <= cy < H:
                img[cy, cx] = 100
    write_pgm(out_path, img)


def plot_pairs_hist2d_simple(sample: np.ndarray, out_path: Path, bins: int = 256, size: Tuple[int, int] = (256, 768)) -> None:
    H, W = size
    img = np.full((H, W), 255, dtype=np.uint8)
    pairs = [(0, 1), (0, 2), (1, 2)]
    panel_w = W // 3
    for pi, (a, b) in enumerate(pairs):
        x = sample[:, a]
        y = sample[:, b]
        # clamp range via percentiles to avoid outliers dominating
        xlo, xhi = np.percentile(x, [1, 99])
        ylo, yhi = np.percentile(y, [1, 99])
        H2, xe, ye = np.histogram2d(x, y, bins=bins, range=[[xlo, xhi], [ylo, yhi]])
        H2 = H2.T
        if H2.max() > 0:
            V = np.log1p(H2) / np.log1p(H2.max())
        else:
            V = H2
        # Resize to panel via simple bin replication
        hh, ww = V.shape
        sx = max(1, (panel_w) // ww)
        sy = max(1, H // hh)
        tile = np.kron(V, np.ones((sy, sx)))
        tile = tile[:H, :panel_w]
        tile_img = (255 - (tile * 255)).astype(np.uint8)
        img[:, pi * panel_w:(pi + 1) * panel_w] = tile_img
    write_pgm(out_path, img)


def plot_d2_hist_simple(sample: np.ndarray, out_path: Path, bins: int = 256, size: Tuple[int, int] = (220, 720)) -> None:
    mu = sample.mean(axis=0)
    var = sample.var(axis=0, ddof=1)
    var[var <= 1e-12] = 1e-12
    z = (sample - mu) / np.sqrt(var)
    d2 = np.sum(z * z, axis=1)
    H, W = size
    img = np.full((H, W), 255, dtype=np.uint8)
    # histogram on [0, q99.9]
    xmax = float(np.percentile(d2, 99.9))
    hist, edges = np.histogram(d2, bins=bins, range=(0, xmax), density=True)
    hist = hist.astype(np.float64)
    hist /= hist.max() + 1e-12
    bw = max(1, (W - 20) // bins)
    for b, h in enumerate(hist):
        x0 = 10 + b * bw
        x1 = min(W - 10, x0 + bw)
        hpx = int(h * (H - 30))
        img[H - 10 - hpx:H - 10, x0:x1] = 30
    write_pgm(out_path, img)


def main() -> None:
    ap = argparse.ArgumentParser(description='Visualize distributions in frame ranges')
    ap.add_argument('--points-dir', type=Path, required=True)
    ap.add_argument('--ranges', type=str, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--max-frames', type=int, default=80)
    ap.add_argument('--max-points', type=int, default=120_000)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--backend', type=str, choices=['simple', 'matplotlib'], default='simple', help='simple (PGM) avoids SHM issues')
    args = ap.parse_args()

    ranges = parse_ranges(args.ranges)
    frames = list_frames(args.points_dir, ranges)
    if not frames:
        raise SystemExit('No frames matched the given ranges.')
    rng = np.random.default_rng(args.seed)
    if args.max_frames > 0 and len(frames) > args.max_frames:
        sel = rng.choice(frames, size=args.max_frames, replace=False)
        frames = sorted(sel.tolist())

    paths = [args.points_dir / f'{fid}.npy' for fid in frames]
    sample = reservoir_sample(paths, cols=(0, 1, 2), max_points=args.max_points, seed=args.seed)
    if sample.size == 0:
        raise SystemExit('No points sampled. Check inputs.')

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    with (out_dir / 'summary.txt').open('w') as f:
        for i in range(3):
            m, s, sk, ku, jb = jb_statistics(sample[:, i])
            f.write(f'Axis {i}: mean={m:.6f} std={s:.6f} skew={sk:.6f} kurt={ku:.6f} JB={jb:.3f}\n')
        # D2 stats
        mu = sample.mean(axis=0)
        var = sample.var(axis=0, ddof=1)
        var[var <= 1e-12] = 1e-12
        z = (sample - mu) / np.sqrt(var)
        d2 = np.sum(z * z, axis=1)
        f.write('D2 stats: mean=%.6f var=%.6f p50=%.6f p90=%.6f max=%.6f\n' % (
            float(np.mean(d2)), float(np.var(d2, ddof=1)), float(np.percentile(d2, 50)), float(np.percentile(d2, 90)), float(np.max(d2))
        ))

    # Plots
    if args.backend == 'matplotlib' and _MATPLOTLIB_OK:
        plot_axis_hist_matplotlib(sample, out_dir / 'axis_hist.png')
        plot_axis_qq_matplotlib(sample, out_dir / 'axis_qq.png')
        plot_pairs_hist2d_matplotlib(sample, out_dir / 'pairs_hist2d.png')
        plot_d2_hist_matplotlib(sample, out_dir / 'd2_hist.png')
    else:
        plot_axis_hist_simple(sample, out_dir / 'axis_hist.pgm')
        plot_axis_qq_simple(sample, out_dir / 'axis_qq.pgm')
        plot_pairs_hist2d_simple(sample, out_dir / 'pairs_hist2d.pgm')
        plot_d2_hist_simple(sample, out_dir / 'd2_hist.pgm')

    print('Saved:')
    names = ['summary.txt']
    if args.backend == 'matplotlib' and _MATPLOTLIB_OK:
        names += ['axis_hist.png', 'axis_qq.png', 'pairs_hist2d.png', 'd2_hist.png']
    else:
        names += ['axis_hist.pgm', 'axis_qq.pgm', 'pairs_hist2d.pgm', 'd2_hist.pgm']
    for name in names:
        print(out_dir / name)


if __name__ == '__main__':
    main()

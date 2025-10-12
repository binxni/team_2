#!/usr/bin/env python3
"""
3D plotting utility for intensity over XY.

Modes:
  - scatter: plot (x, y, intensity) as a 3D scatter for a single .npy file
  - surface: aggregate over frames into XY bins and plot mean/max intensity surface

Examples:
  Single frame scatter (save PNG):
    python Subin/OpenPCDet/tools/plot_intensity_3d.py \
        --file Subin/OpenPCDet/data/custom_av/points_test/10000000.npy \
        --mode scatter --limit-points 50000 --save-png Subin/OpenPCDet/tools/intensity_3d_scatter.png

  Aggregated surface over XY (mean intensity):
    python Subin/OpenPCDet/tools/plot_intensity_3d.py \
        --root Subin/OpenPCDet/data/custom_av/points_test \
        --mode surface --bin-size 1.0 --stat mean --max-files 10 \
        --smooth-sigma 1.2 --upsample 2 \
        --save-png Subin/OpenPCDet/tools/intensity_3d_surface.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np


def _iter_point_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.glob('*.npy')):
        if p.is_file():
            yield p


def _load_points(npy_path: Path) -> np.ndarray:
    pts = np.load(npy_path)
    if pts.ndim != 2 or pts.shape[1] < 4:
        raise ValueError(f"Unexpected shape {pts.shape} in {npy_path}")
    return pts.astype(np.float32, copy=False)


def _maybe_clip(v: np.ndarray, clip: Optional[Tuple[float, float]]) -> np.ndarray:
    if clip is None:
        return v
    lo, hi = clip
    return np.clip(v, lo, hi)


def _gaussian_kernel1d(sigma: float, radius: Optional[int] = None) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float64)
    if radius is None:
        radius = max(1, int(np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k


def _fftconvolve2d_same(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """FFT-based 2D convolution with 'same' output and zero padding."""
    H, W = a.shape
    Kh, Kw = k.shape
    out_shape = (H + Kh - 1, W + Kw - 1)
    fa = np.fft.rfft2(a, out_shape)
    fk = np.fft.rfft2(k, out_shape)
    conv = np.fft.irfft2(fa * fk, out_shape)
    # center crop to original size
    i0 = Kh - 1
    j0 = Kw - 1
    return conv[i0:i0 + H, j0:j0 + W]


def smooth_gaussian_nanaware(Z: np.ndarray, sigma: float, upsample: int = 1) -> np.ndarray:
    """
    Apply NaN-aware Gaussian smoothing to Z.
    - sigma is in grid cell units
    - upsample>1 performs simple nearest upsampling before smoothing for a softer look
    """
    if sigma <= 0 and upsample <= 1:
        return Z

    Z_work = Z
    if upsample and upsample > 1:
        Z_work = np.kron(Z_work, np.ones((upsample, upsample), dtype=Z_work.dtype))

    valid = np.isfinite(Z_work).astype(np.float64)
    values = np.nan_to_num(Z_work, nan=0.0).astype(np.float64)
    k1 = _gaussian_kernel1d(max(1e-6, float(sigma)))
    K = np.outer(k1, k1)
    num = _fftconvolve2d_same(values, K)
    den = _fftconvolve2d_same(valid, K)
    out = num / np.maximum(den, 1e-12)
    out[den < 1e-8] = np.nan
    return out.astype(np.float32)


def plot_scatter(
    file: Path,
    save_png: Optional[Path],
    limit_points: int = 100000,
    vertical_axis: str = 'z',  # 'z' or 'y'
    clip: Optional[Tuple[float, float]] = None,
    elev: float = 30.0,
    azim: float = -60.0,
) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')  # headless
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D)
    except Exception as e:
        raise RuntimeError("matplotlib is required for plotting") from e

    pts = _load_points(file)
    x, y, z, i = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    i = _maybe_clip(i, clip)

    n = len(i)
    if limit_points is not None and n > limit_points:
        idx = np.random.choice(n, size=limit_points, replace=False)
        x, y, z, i = x[idx], y[idx], z[idx], i[idx]

    # Prepare axes values: vertical axis shows intensity
    if vertical_axis.lower() == 'z':
        X, Y, Z = x, y, i
        zlabel = 'intensity'
    elif vertical_axis.lower() == 'y':
        X, Y, Z = x, i, y
        zlabel = 'y (lateral)'
    else:
        raise ValueError("vertical_axis must be 'z' or 'y'")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    s = ax.scatter(X, Y, Z, c=i, s=1, cmap='viridis', alpha=0.8)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)' if vertical_axis.lower() == 'z' else 'intensity')
    ax.set_zlabel(zlabel)
    fig.colorbar(s, ax=ax, label='intensity')
    fig.tight_layout()

    if save_png is not None:
        fig.savefig(save_png, dpi=150)
    plt.close(fig)


def plot_surface(
    root: Path,
    save_png: Optional[Path],
    bin_size: float = 1.0,
    stat: str = 'mean',  # 'mean' or 'max'
    max_files: Optional[int] = None,
    clip: Optional[Tuple[float, float]] = None,
    smooth_sigma: float = 0.0,
    upsample: int = 1,
    elev: float = 40.0,
    azim: float = -50.0,
) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as e:
        raise RuntimeError("matplotlib is required for plotting") from e

    if bin_size <= 0:
        raise ValueError("bin_size must be > 0")

    # Accumulate per-cell stats
    sum_i = {}
    cnt = {}
    max_i = {}

    processed = 0
    for path in _iter_point_files(root):
        if max_files is not None and processed >= max_files:
            break
        pts = _load_points(path)
        i = _maybe_clip(pts[:, 3], clip)
        xy = np.floor(pts[:, :2] / bin_size).astype(np.int64)
        cells, inv, counts = np.unique(xy, axis=0, return_inverse=True, return_counts=True)

        # aggregate
        if stat == 'mean':
            sums = np.zeros(len(cells), dtype=np.float64)
            np.add.at(sums, inv, i)
            for (cx, cy), s, c in zip(cells, sums, counts):
                key = (int(cx), int(cy))
                sum_i[key] = sum_i.get(key, 0.0) + float(s)
                cnt[key] = cnt.get(key, 0) + int(c)
        elif stat == 'max':
            mx = np.full(len(cells), -np.inf, dtype=np.float64)
            np.maximum.at(mx, inv, i)
            for (cx, cy), m in zip(cells, mx):
                key = (int(cx), int(cy))
                max_i[key] = max(max_i.get(key, -np.inf), float(m))
        else:
            raise ValueError("Unsupported stat: choose 'mean' or 'max'")

        processed += 1

    if processed == 0:
        raise RuntimeError("No frames processed. Check --root or --max-files")

    # Build grid
    keys = list(sum_i.keys() if stat == 'mean' else max_i.keys())
    xs = [k[0] for k in keys]
    ys = [k[1] for k in keys]
    ix_min, ix_max = min(xs), max(xs)
    iy_min, iy_max = min(ys), max(ys)
    nx = ix_max - ix_min + 1
    ny = iy_max - iy_min + 1

    Z = np.full((nx, ny), np.nan, dtype=np.float32)
    for (cx, cy) in keys:
        if stat == 'mean':
            s = sum_i[(cx, cy)]
            c = cnt[(cx, cy)]
            val = s / max(c, 1)
        else:
            val = max_i[(cx, cy)]
        Z[cx - ix_min, cy - iy_min] = float(val)

    # Optional smoothing and upsampling
    Z_plot = Z
    if smooth_sigma > 0 or (upsample and upsample > 1):
        Z_plot = smooth_gaussian_nanaware(Z_plot, sigma=smooth_sigma, upsample=upsample)

    scale = 1.0 / float(upsample if upsample and upsample > 1 else 1)
    eff_bin = bin_size * scale
    nx_eff, ny_eff = Z_plot.shape
    Xc = (np.arange(nx_eff) + 0.5) * eff_bin + (ix_min * bin_size)
    Yc = (np.arange(ny_eff) + 0.5) * eff_bin + (iy_min * bin_size)
    X, Y = np.meshgrid(Xc, Yc, indexing='ij')

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    # To avoid rendering NaNs as zero, mask them
    Z_plot = np.ma.masked_invalid(Z_plot)
    surf = ax.plot_surface(X, Y, Z_plot, cmap='viridis', linewidth=0, antialiased=False)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel(f'{stat} intensity')
    fig.colorbar(surf, ax=ax, shrink=0.6, label='intensity')
    fig.tight_layout()
    if save_png is not None:
        fig.savefig(save_png, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='3D plot of intensity over XY')
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--file', type=Path, help='Single .npy point cloud (x,y,z,intensity)')
    src.add_argument('--root', type=Path, help='Directory with .npy files for aggregation')

    parser.add_argument('--mode', choices=['scatter', 'surface'], default='scatter')
    parser.add_argument('--save-png', type=Path, default=None, help='Output PNG path')
    parser.add_argument('--clip', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'), help='Clip intensity range')

    # scatter options
    parser.add_argument('--limit-points', type=int, default=100000, help='Max points to plot in scatter mode')
    parser.add_argument('--vertical-axis', choices=['z', 'y'], default='z', help='Which axis is vertical for scatter: z (default) or y')
    parser.add_argument('--elev', type=float, default=None, help='Elevation angle for view')
    parser.add_argument('--azim', type=float, default=None, help='Azimuth angle for view')

    # surface options
    parser.add_argument('--bin-size', type=float, default=1.0, help='XY bin size for surface mode (meters)')
    parser.add_argument('--stat', choices=['mean', 'max'], default='mean', help='Aggregation statistic for surface mode')
    parser.add_argument('--max-files', type=int, default=None, help='Max number of frames to aggregate')
    parser.add_argument('--smooth-sigma', type=float, default=0.0, help='Gaussian smoothing sigma in bin units (surface mode)')
    parser.add_argument('--upsample', type=int, default=1, help='Upsample factor applied before smoothing (surface mode)')

    args = parser.parse_args()

    clip = tuple(args.clip) if args.clip is not None else None

    if args.mode == 'scatter':
        if not args.file:
            raise ValueError('--file is required for scatter mode')
        plot_scatter(
            file=args.file,
            save_png=args.save_png,
            limit_points=args.limit_points,
            vertical_axis=args.vertical_axis,
            clip=clip,
            elev=30.0 if args.elev is None else args.elev,
            azim=-60.0 if args.azim is None else args.azim,
        )
    else:
        if not args.root:
            raise ValueError('--root is required for surface mode')
        plot_surface(
            root=args.root,
            save_png=args.save_png,
            bin_size=args.bin_size,
            stat=args.stat,
            max_files=args.max_files,
            clip=clip,
            smooth_sigma=float(args.smooth_sigma),
            upsample=int(args.upsample),
            elev=40.0 if args.elev is None else args.elev,
            azim=-50.0 if args.azim is None else args.azim,
        )


if __name__ == '__main__':
    main()

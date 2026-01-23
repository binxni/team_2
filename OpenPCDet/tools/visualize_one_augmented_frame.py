#!/usr/bin/env python3
"""
Visualize a single frame before/after augmentation (BEV), highlighting air-noise.

Usage example:
  python OpenPCDet/tools/visualize_one_augmented_frame.py \
      --dataset-root OpenPCDet/data/custom_av \
      --points-dir points \
      --aug-dir points_lisa \
      --frame-id 10006432 \
      --air-noise-frac 0.005 \
      --out OpenPCDet/output/preview_bev_10006432.ppm

Notes:
  - If augmented file is missing and --simulate-if-missing is set (default),
    the script appends synthetic air-noise to the original for preview only.
  - Output is a color PPM (P6) so it opens in most viewers; use .ppm extension.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Visualize one augmented frame (BEV)')
    ap.add_argument('--dataset-root', type=Path, required=True)
    ap.add_argument('--points-dir', type=Path, default=Path('points'))
    ap.add_argument('--aug-dir', type=Path, default=Path('points_lisa'))
    ap.add_argument('--frame-id', type=str, required=True, help='e.g., 10006432')
    ap.add_argument('--out', type=Path, required=True, help='Output image path (.ppm recommended)')
    ap.add_argument('--air-noise-count', type=int, default=None, help='Expected injected count; else derived from frac or diff')
    ap.add_argument('--air-noise-frac', type=float, default=0.0, help='Expected injected frac of original count')
    ap.add_argument('--x-range', type=float, nargs=2, default=(-50.0, 50.0))
    ap.add_argument('--y-range', type=float, nargs=2, default=(-50.0, 50.0))
    ap.add_argument('--res', type=float, default=0.1, help='Meters per pixel')
    ap.add_argument('--simulate-if-missing', action='store_true', help='Simulate air-noise if augmented file absent')
    # noise sim params (only used when --simulate-if-missing)
    ap.add_argument('--sim-rmin', type=float, default=1.5)
    ap.add_argument('--sim-rmax', type=float, default=80.0)
    ap.add_argument('--sim-min-z', type=float, default=0.3)
    ap.add_argument('--sim-max-z', type=float, default=3.0)
    ap.add_argument('--sim-r-scale', type=float, default=30.0)
    ap.add_argument('--sim-model', choices=('mixture', 'laplace', 'exponential'), default='mixture')
    ap.add_argument('--sim-cluster', action='store_true', help='Use clustered simulation (default iid)')
    ap.add_argument('--sim-cluster-mean', type=float, default=3.0)
    ap.add_argument('--sim-cluster-sigma', type=float, default=0.1)
    return ap.parse_args()


def write_ppm(path: Path, img: np.ndarray) -> None:
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8
    h, w, _ = img.shape
    with open(path, 'wb') as f:
        header = f'P6\n{w} {h}\n255\n'.encode('ascii')
        f.write(header)
        f.write(img.tobytes())


def _sample_radius(model: str, scale: float, size: int, rmin: float, rmax: float) -> np.ndarray:
    if model == "exponential":
        r = np.random.exponential(scale=scale, size=size)
    elif model == "laplace":
        b = max(1e-3, scale / 2.0)
        r = np.abs(np.random.laplace(loc=0.0, scale=b, size=size))
    else:  # mixture
        w = np.random.rand(size) < 0.3
        r = np.where(w, np.random.exponential(scale=max(1e-3, scale / 3.0), size=size), np.random.exponential(scale=scale, size=size))
    # clamp to [rmin,rmax] roughly via rejection
    out = np.empty(size, dtype=np.float32)
    filled = 0
    while filled < size:
        remain = size - filled
        rr = _sample_radius(model, scale, remain, rmin, rmax) if filled > 0 else r
        m = (rr >= rmin) & (rr <= rmax)
        take = min(remain, int(np.sum(m)))
        if take > 0:
            out[filled:filled + take] = rr[m][:take]
            filled += take
        else:
            break
    if filled < size:
        out[filled:] = (rmin + rmax) / 2.0
    return out


def _sample_height(model: str, min_z: float, max_z: float, size: int) -> np.ndarray:
    if model == "exponential":
        z = min_z + np.random.exponential(scale=max(1e-3, (max_z - min_z) / 3.0), size=size)
    elif model == "laplace":
        b = max(1e-3, (max_z - min_z) / 6.0)
        z = (min_z + max_z) / 2.0 + np.random.laplace(loc=0.0, scale=b, size=size)
    else:  # mixture
        w = np.random.rand(size) < 0.4
        z = np.empty(size, dtype=np.float32)
        z[w] = min_z + np.random.exponential(scale=max(1e-3, (max_z - min_z) / 2.5), size=int(np.sum(w)))
        z[~w] = (min_z + max_z) / 2.0 + np.random.laplace(loc=0.0, scale=max(1e-3, (max_z - min_z) / 6.0), size=int(np.sum(~w)))
    return np.clip(z, min_z, max_z).astype(np.float32)


def _iid_noise(n: int, rmin: float, rmax: float, min_z: float, max_z: float, r_scale: float, model: str) -> np.ndarray:
    r = _sample_radius(model, r_scale, n, rmin, rmax)
    th = np.random.uniform(-np.pi, np.pi, size=n).astype(np.float32)
    z = _sample_height(model, min_z, max_z, n)
    r2 = r.astype(np.float32) ** 2
    rz2 = np.clip(r2 - (z.astype(np.float32) ** 2), 0.0, None)
    rxy = np.sqrt(rz2)
    x = rxy * np.cos(th)
    y = rxy * np.sin(th)
    return np.stack([x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)], axis=1)


def simulate_noise(n: int, *, rmin: float, rmax: float, min_z: float, max_z: float, r_scale: float, model: str, cluster: bool, cluster_mean: float, cluster_sigma: float) -> np.ndarray:
    if not cluster:
        return _iid_noise(n, rmin, rmax, min_z, max_z, r_scale, model)
    mean_k = max(1.0, float(cluster_mean))
    p = min(1.0, max(1e-3, 1.0 / mean_k))
    k = max(1, int(np.ceil(n / mean_k)))
    sizes = np.random.geometric(p, size=k)
    total = int(np.sum(sizes))
    if total < n:
        sizes[0] += (n - total)
    elif total > n:
        over = total - n
        sizes[-1] = max(1, sizes[-1] - over)
    centers = _iid_noise(int(sizes.size), rmin, rmax, min_z, max_z, r_scale, model)
    xyz_list = []
    for i, sz in enumerate(sizes):
        center = centers[i]
        local = np.random.normal(loc=0.0, scale=cluster_sigma, size=(int(sz), 3)).astype(np.float32)
        pts = center.reshape(1, 3).astype(np.float32) + local
        pts[:, 2] = np.clip(pts[:, 2], min_z, max_z)
        r = np.sqrt(np.sum(pts * pts, axis=1))
        keep = (r >= rmin) & (r <= rmax)
        if not np.all(keep):
            pts = pts[keep]
            need = int(sz) - pts.shape[0]
            if need > 0:
                extra = _iid_noise(need, rmin, rmax, min_z, max_z, r_scale, model)
                pts = np.vstack([pts, extra])
        xyz_list.append(pts.astype(np.float32))
    xyz = np.vstack(xyz_list)
    if xyz.shape[0] > n:
        xyz = xyz[:n]
    elif xyz.shape[0] < n:
        extra = _iid_noise(n - xyz.shape[0], rmin, rmax, min_z, max_z, r_scale, model)
        xyz = np.vstack([xyz, extra])
    return xyz


def bev_raster(points: np.ndarray, x_range: Tuple[float, float], y_range: Tuple[float, float], res: float, color: Tuple[int, int, int], img: np.ndarray) -> None:
    xmin, xmax = x_range
    ymin, ymax = y_range
    W = int(round((xmax - xmin) / res))
    H = int(round((ymax - ymin) / res))
    # map to pixels
    xs = (points[:, 0] - xmin) / (xmax - xmin + 1e-12) * (W - 1)
    ys = (points[:, 1] - ymin) / (ymax - ymin + 1e-12) * (H - 1)
    xi = xs.astype(np.int32)
    yi = (H - 1 - ys).astype(np.int32)  # flip y for top-down
    m = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    xi = xi[m]
    yi = yi[m]
    img[yi, xi, 0] = np.maximum(img[yi, xi, 0], color[0])
    img[yi, xi, 1] = np.maximum(img[yi, xi, 1], color[1])
    img[yi, xi, 2] = np.maximum(img[yi, xi, 2], color[2])


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    orig_path = (root / args.points_dir / f"{args.frame_id}.npy").resolve()
    aug_path = (root / args.aug_dir / f"{args.frame_id}.npy").resolve()
    if not orig_path.exists():
        raise FileNotFoundError(f'Original file not found: {orig_path}')
    orig = np.load(orig_path)
    if orig.ndim != 2 or orig.shape[1] < 3:
        raise ValueError(f'Invalid original shape: {orig.shape}')
    orig_xyz = orig[:, :3].astype(np.float32)

    if aug_path.exists():
        augmented = np.load(aug_path)
        aug_xyz = augmented[:, :3].astype(np.float32)
    elif args.simulate_if_missing:
        # simulate augmented by appending synthetic noise
        n_noise = args.air_noise_count
        if (n_noise is None) and (args.air_noise_frac and args.air_noise_frac > 0):
            n_noise = max(0, int(round(args.air_noise_frac * orig_xyz.shape[0])))
        if n_noise is None or n_noise <= 0:
            n_noise = max(1, int(0.005 * orig_xyz.shape[0]))  # default 0.5%
        noise = simulate_noise(
            n_noise,
            rmin=float(args.sim_rmin), rmax=float(args.sim_rmax),
            min_z=float(args.sim_min_z), max_z=float(args.sim_max_z),
            r_scale=float(args.sim_r_scale), model=str(args.sim_model),
            cluster=bool(args.sim_cluster), cluster_mean=float(args.sim_cluster_mean), cluster_sigma=float(args.sim_cluster_sigma)
        )
        aug_xyz = np.vstack([orig_xyz, noise])
    else:
        raise FileNotFoundError(f'Augmented file not found: {aug_path}')

    # determine n_noise to highlight (assume noise appended at end)
    n_noise = args.air_noise_count
    if (n_noise is None) and (args.air_noise_frac and args.air_noise_frac > 0):
        n_noise = max(0, int(round(args.air_noise_frac * orig_xyz.shape[0])))
    if n_noise is None or n_noise < 0:
        diff = aug_xyz.shape[0] - orig_xyz.shape[0]
        n_noise = diff if diff > 0 else 0

    # prepare canvas
    xmin, xmax = args.x_range
    ymin, ymax = args.y_range
    W = int(round((xmax - xmin) / args.res))
    H = int(round((ymax - ymin) / args.res))
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # draw original (gray)
    bev_raster(orig_xyz, (xmin, xmax), (ymin, ymax), args.res, (120, 120, 120), img)

    # draw noise (red) from tail of augmented
    if n_noise > 0 and n_noise <= aug_xyz.shape[0]:
        noise_xyz = aug_xyz[-n_noise:, :]
        bev_raster(noise_xyz, (xmin, xmax), (ymin, ymax), args.res, (255, 60, 60), img)

    # also draw non-noise augmented additions (if any ambiguity) in blue (optional)
    # skipped for clarity

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_ppm(args.out, img)
    print(f'Saved BEV to {args.out}  (gray=original, red=air-noise)')


if __name__ == '__main__':
    main()

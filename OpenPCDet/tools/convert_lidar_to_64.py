"""Convert mixed 64/128-channel LiDAR frames to a unified 64-channel-like distribution.

This utility performs three main tasks:
  1. Analyse the dataset and split frame IDs into 64-channel (<= threshold points)
     and 128-channel (> threshold points).
  2. Compute distribution statistics from native 64-channel frames (point counts,
     intensity histogram, range histogram) and from 128-channel frames for
     intensity normalisation.
  3. Downsample each 128-channel frame so that its point count and per-point
     attributes resemble the statistics of 64-channel frames. The converted
     frames are saved to a separate output directory, while original 64-channel
     frames are symlinked (or copied) to keep the dataset complete.

Usage example:

    python tools/convert_lidar_to_64.py \
        --dataset_root data/custom_av \
        --output_dir data/custom_av_aligned64/points \
        --train_split ImageSets/train.txt \
        --val_split ImageSets/val.txt

The script expects `.npy` point clouds and KITTI-style label files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
# Torch is not required; using NumPy-based geometric checks keeps the script
# lightweight and avoids depending on compiled CUDA extensions.


DEFAULT_POINT_THRESHOLD = 200_000
INTENSITY_BINS = np.linspace(0.0, 1.0, 129)  # 128 bins
RANGE_BINS = np.linspace(0.0, 120.0, 241)    # 0.5 m resolution


@dataclass
class RunningStats:
    frame_count: int = 0
    point_count: int = 0
    sum_xyz: np.ndarray = None
    sum_sq_xyz: np.ndarray = None
    min_xyz: np.ndarray = None
    max_xyz: np.ndarray = None
    sum_i: float = 0.0
    sum_sq_i: float = 0.0
    min_i: float = math.inf
    max_i: float = -math.inf
    sum_r: float = 0.0
    sum_sq_r: float = 0.0
    min_r: float = math.inf
    max_r: float = -math.inf
    hist_i: np.ndarray = None
    hist_r: np.ndarray = None

    def __post_init__(self) -> None:
        if self.sum_xyz is None:
            self.sum_xyz = np.zeros(3, dtype=np.float64)
        if self.sum_sq_xyz is None:
            self.sum_sq_xyz = np.zeros(3, dtype=np.float64)
        if self.min_xyz is None:
            self.min_xyz = np.full(3, np.inf)
        if self.max_xyz is None:
            self.max_xyz = np.full(3, -np.inf)
        if self.hist_i is None:
            self.hist_i = np.zeros(len(INTENSITY_BINS) - 1, dtype=np.float64)
        if self.hist_r is None:
            self.hist_r = np.zeros(len(RANGE_BINS) - 1, dtype=np.float64)

    def update(self, points: np.ndarray) -> None:
        xyz = points[:, :3]
        intensity = points[:, 3]
        r = np.linalg.norm(xyz, axis=1)

        n = points.shape[0]
        self.frame_count += 1
        self.point_count += n

        self.sum_xyz += xyz.sum(axis=0)
        self.sum_sq_xyz += np.square(xyz).sum(axis=0)
        self.min_xyz = np.minimum(self.min_xyz, xyz.min(axis=0))
        self.max_xyz = np.maximum(self.max_xyz, xyz.max(axis=0))

        self.sum_i += float(intensity.sum())
        self.sum_sq_i += float(np.square(intensity).sum())
        self.min_i = min(self.min_i, float(intensity.min()))
        self.max_i = max(self.max_i, float(intensity.max()))

        self.sum_r += float(r.sum())
        self.sum_sq_r += float(np.square(r).sum())
        self.min_r = min(self.min_r, float(r.min()))
        self.max_r = max(self.max_r, float(r.max()))

        self.hist_i += np.histogram(intensity, bins=INTENSITY_BINS)[0]
        self.hist_r += np.histogram(r, bins=RANGE_BINS)[0]

    def finalize(self) -> dict:
        if self.point_count == 0:
            return {}
        mean_xyz = self.sum_xyz / self.point_count
        var_xyz = self.sum_sq_xyz / self.point_count - mean_xyz ** 2
        std_xyz = np.sqrt(np.clip(var_xyz, 0, None))

        mean_i = self.sum_i / self.point_count
        var_i = self.sum_sq_i / self.point_count - mean_i ** 2
        std_i = math.sqrt(max(var_i, 0.0))

        mean_r = self.sum_r / self.point_count
        var_r = self.sum_sq_r / self.point_count - mean_r ** 2
        std_r = math.sqrt(max(var_r, 0.0))

        return {
            "frame_count": self.frame_count,
            "point_count": self.point_count,
            "mean_xyz": mean_xyz.tolist(),
            "std_xyz": std_xyz.tolist(),
            "min_xyz": self.min_xyz.tolist(),
            "max_xyz": self.max_xyz.tolist(),
            "mean_intensity": mean_i,
            "std_intensity": std_i,
            "min_intensity": self.min_i,
            "max_intensity": self.max_i,
            "mean_range": mean_r,
            "std_range": std_r,
            "min_range": self.min_r,
            "max_range": self.max_r,
            "hist_intensity": self.hist_i.tolist(),
            "hist_range": self.hist_r.tolist(),
        }


@dataclass
class DatasetStats:
    counts64: List[int]
    counts128: List[int]
    stats64: dict
    stats128: dict


def load_split_ids(file_path: Path) -> List[str]:
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_points(point_dir: Path, sample_id: str) -> np.ndarray:
    return np.load(point_dir / f"{sample_id}.npy")


def load_boxes(label_dir: Path, sample_id: str) -> np.ndarray:
    label_path = label_dir / f"{sample_id}.txt"
    if not label_path.exists():
        return np.zeros((0, 7), dtype=np.float32)
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            x, y, z, dx, dy, dz, heading = map(float, parts[:7])
            boxes.append([x, y, z, dx, dy, dz, heading])
    if not boxes:
        return np.zeros((0, 7), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def compute_dataset_stats(point_dir: Path,
                          split_ids: Iterable[str],
                          threshold: int) -> DatasetStats:
    counts64: List[int] = []
    counts128: List[int] = []
    stats64 = RunningStats()
    stats128 = RunningStats()

    for idx in split_ids:
        points = load_points(point_dir, idx)
        n = points.shape[0]
        if n <= threshold:
            counts64.append(n)
            stats64.update(points)
        else:
            counts128.append(n)
            stats128.update(points)

    return DatasetStats(
        counts64=counts64,
        counts128=counts128,
        stats64=stats64.finalize(),
        stats128=stats128.finalize(),
    )


def percentile_from_hist(hist: np.ndarray, bins: np.ndarray, percent: float) -> float:
    total = hist.sum()
    if total <= 0:
        return float("nan")
    target = percent / 100.0 * total
    cdf = np.cumsum(hist)
    idx = np.searchsorted(cdf, target)
    idx = np.clip(idx, 1, len(bins) - 1)
    prev = cdf[idx - 1]
    bin_count = cdf[idx] - prev
    if bin_count == 0:
        return float(bins[idx])
    ratio = (target - prev) / bin_count
    return float(bins[idx - 1] + ratio * (bins[idx] - bins[idx - 1]))


def build_roi_mask(points_xyz: np.ndarray,
                   boxes: np.ndarray,
                   expand_lengths: Tuple[float, float, float]) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.zeros(points_xyz.shape[0], dtype=bool)

    mask = np.zeros(points_xyz.shape[0], dtype=bool)
    expand_dx, expand_dy, expand_dz = expand_lengths

    for box in boxes:
        cx, cy, cz, dx, dy, dz, heading = box
        dx += 2 * expand_dx
        dy += 2 * expand_dy
        dz += 2 * expand_dz

        cos_yaw = math.cos(-heading)
        sin_yaw = math.sin(-heading)

        rel = points_xyz - np.array([cx, cy, cz], dtype=np.float32)
        rot_x = rel[:, 0] * cos_yaw - rel[:, 1] * sin_yaw
        rot_y = rel[:, 0] * sin_yaw + rel[:, 1] * cos_yaw
        rot_z = rel[:, 2]

        mask_box = (
            np.abs(rot_x) <= dx * 0.5
            ) & (
            np.abs(rot_y) <= dy * 0.5
            ) & (
            np.abs(rot_z) <= dz * 0.5
        )
        mask |= mask_box

    return mask


def normalise_intensity(intensity: np.ndarray,
                        mean_src: float,
                        std_src: float,
                        mean_tgt: float,
                        std_tgt: float) -> np.ndarray:
    if std_src <= 0:
        return np.clip(np.full_like(intensity, mean_tgt, dtype=np.float32), 0.0, 1.0)
    normalised = (intensity - mean_src) / std_src
    remapped = normalised * std_tgt + mean_tgt
    return np.clip(remapped, 0.0, 1.0)


def downsample_points(points: np.ndarray,
                      roi_mask: np.ndarray,
                      target_count: int,
                      range_pdf: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
    total_points = points.shape[0]
    if target_count >= total_points:
        # Nothing to downsample; just shuffle to avoid ordering bias.
        shuffle_idx = rng.permutation(total_points)
        return points[shuffle_idx]

    roi_indices = np.nonzero(roi_mask)[0]
    non_roi_indices = np.nonzero(~roi_mask)[0]

    selected: List[int] = []

    if len(roi_indices) >= target_count:
        chosen = rng.choice(roi_indices, size=target_count, replace=False)
        return points[np.sort(chosen)]

    if len(roi_indices) > 0:
        selected.extend(roi_indices.tolist())

    remaining = target_count - len(selected)
    if remaining <= 0 or len(non_roi_indices) == 0:
        return points[np.sort(np.array(selected, dtype=np.int64))]

    non_roi_points = points[non_roi_indices]
    distances = np.linalg.norm(non_roi_points[:, :3], axis=1)
    bin_indices = np.digitize(distances, RANGE_BINS) - 1
    bin_indices = np.clip(bin_indices, 0, len(range_pdf) - 1)

    weights = range_pdf[bin_indices]
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones_like(weights)

    weights = weights / weights.sum()
    sampled = rng.choice(non_roi_indices, size=min(remaining, len(non_roi_indices)), replace=False, p=weights)
    selected.extend(sampled.tolist())

    selected_array = np.array(selected, dtype=np.int64)
    if selected_array.shape[0] > target_count:
        selected_array = selected_array[rng.choice(selected_array.shape[0], size=target_count, replace=False)]

    return points[np.sort(selected_array)]


def copy_points(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if overwrite:
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
        else:
            return
    np.save(dst, np.load(src, mmap_mode="r"))


def symlink_points(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if overwrite:
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
        else:
            return
    link_target = os.path.relpath(src.resolve(strict=True), start=dst.parent.resolve())
    os.symlink(link_target, dst)


def convert_dataset(dataset_root: Path,
                    output_dir: Path,
                    train_split: Path,
                    val_split: Path,
                    threshold: int,
                    expand_lengths: Tuple[float, float, float],
                    seed: int,
                    overwrite: bool,
                    skip_existing: bool,
                    stats_json: Path | None,
                    max_frames: int | None,
                    use_symlink_for_64: bool) -> None:
    rng = np.random.default_rng(seed)

    point_dir = dataset_root / "points"
    label_dir = dataset_root / "labels"

    train_ids = load_split_ids(train_split)
    val_ids = load_split_ids(val_split)
    all_ids = train_ids + val_ids

    output_dir.mkdir(parents=True, exist_ok=True)

    if not stats_json:
        default_stats = output_dir.parent / "aligned64_stats.json"
        if default_stats.exists():
            stats_json = default_stats

    if stats_json and stats_json.exists():
        with open(stats_json, "r") as f:
            stats_dict = json.load(f)
        stats = DatasetStats(
            counts64=stats_dict["counts64"],
            counts128=stats_dict.get("counts128", []),
            stats64=stats_dict["stats64"],
            stats128=stats_dict.get("stats128", {}),
        )
    else:
        stats = compute_dataset_stats(point_dir, all_ids, threshold)
        stats_json = output_dir.parent / "aligned64_stats.json"
        with open(stats_json, "w") as f:
            json.dump(asdict(stats), f, indent=2)

    hist64 = np.array(stats.stats64["hist_range"], dtype=np.float64)
    pdf64 = hist64 / hist64.sum() if hist64.sum() > 0 else np.ones_like(hist64) / len(hist64)

    mean_i64 = stats.stats64["mean_intensity"]
    std_i64 = stats.stats64["std_intensity"]
    mean_i128 = stats.stats128.get("mean_intensity", mean_i64)
    std_i128 = stats.stats128.get("std_intensity", std_i64)

    counts64_array = np.array(stats.counts64, dtype=np.int64)
    if counts64_array.size == 0:
        raise RuntimeError("No 64-channel frames found – cannot derive target distribution.")

    # Convert 128-channel frames
    converted_ids = set()
    processed = 0
    for idx in all_ids:
        if max_frames is not None and processed >= max_frames:
            break

        src_path = point_dir / f"{idx}.npy"
        dst_path = output_dir / f"{idx}.npy"

        points = np.load(src_path)
        if points.shape[0] <= threshold:
            if skip_existing and dst_path.exists():
                continue
            if use_symlink_for_64:
                symlink_points(src_path, dst_path, overwrite=overwrite)
            else:
                copy_points(src_path, dst_path, overwrite=overwrite)
            continue

        if skip_existing and dst_path.exists():
            continue

        boxes = load_boxes(label_dir, idx)
        roi_mask = build_roi_mask(points[:, :3], boxes, expand_lengths)

        target_count = int(rng.choice(counts64_array))
        sampled_points = downsample_points(points, roi_mask, target_count, pdf64, rng)
        sampled_points = sampled_points.astype(np.float32, copy=False)
        sampled_points[:, 3] = normalise_intensity(sampled_points[:, 3], mean_i128, std_i128, mean_i64, std_i64)

        np.save(dst_path, sampled_points)
        converted_ids.add(idx)
        processed += 1

    print(f"Converted {len(converted_ids)} frames to 64-channel distribution. Stats source: {stats_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert 128-channel LiDAR frames to 64-channel-like distribution.")
    parser.add_argument("--dataset_root", type=Path, required=True,
                        help="Path to dataset root containing points/ and labels/ directories.")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory where converted point clouds will be written.")
    parser.add_argument("--train_split", type=Path, required=True,
                        help="Path to train split text file.")
    parser.add_argument("--val_split", type=Path, required=True,
                        help="Path to val split text file.")
    parser.add_argument("--threshold", type=int, default=DEFAULT_POINT_THRESHOLD,
                        help="Point-count threshold separating 64- and 128-channel frames.")
    parser.add_argument("--expand_dx", type=float, default=0.5, help="ROI expansion along x axis (metres).")
    parser.add_argument("--expand_dy", type=float, default=0.5, help="ROI expansion along y axis (metres).")
    parser.add_argument("--expand_dz", type=float, default=0.2, help="ROI expansion along z axis (metres).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip conversion if output file already exists.")
    parser.add_argument("--stats_json", type=Path, default=None, help="Optional path to precomputed stats JSON.")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Limit the number of frames processed in this run (helpful for chunked execution).")
    parser.add_argument("--symlink_64", action="store_true",
                        help="Keep native 64-channel frames as symlinks instead of copying them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    convert_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        threshold=args.threshold,
        expand_lengths=(args.expand_dx, args.expand_dy, args.expand_dz),
        seed=args.seed,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        stats_json=args.stats_json,
        max_frames=args.max_frames,
        use_symlink_for_64=args.symlink_64,
    )


if __name__ == "__main__":
    main()

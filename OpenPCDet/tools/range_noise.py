#!/usr/bin/env python3
"""
Add range-dependent noise to LiDAR point clouds stored as numpy arrays.

Example
-------
    python tools/range_noise.py \
        --dataset-root data/custom_av_64 \
        --points-dir points \
        --output-dir points_noisy \
        --base-sigma 0.02 \
        --sigma-gain 0.15 \
        --dropout-max 0.3
        --start-file 000001.npy \
        --end-file 000500.npy \
        --shard-count 3 \
        --shard-index 0 \
        --end-index 500
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np

LOGGER = logging.getLogger("range_noise")

_WORKER_ARGS: SimpleNamespace | None = None
_WORKER_BASE_SEED: int | None = None


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject range-dependent radial noise into LiDAR point clouds."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/custom_av_64"),
        help="Root directory that contains the points folder (default: data/custom_av_64).",
    )
    parser.add_argument(
        "--points-dir",
        type=Path,
        default=Path("points"),
        help="Relative path (from dataset root) with input .npy point clouds (default: points).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("points_noisy"),
        help="Relative path (from dataset root) where noisy clouds will be written.",
    )
    parser.add_argument(
        "--base-sigma",
        type=float,
        default=0.02,
        help="Baseline radial noise standard deviation in meters (default: 0.02).",
    )
    parser.add_argument(
        "--sigma-gain",
        type=float,
        default=0.15,
        help="How much the radial noise grows with distance (default: 0.15).",
    )
    parser.add_argument(
        "--reference-distance",
        type=float,
        default=120.0,
        help="Distance in meters used to normalize noise growth (default: 120).",
    )
    parser.add_argument(
        "--min-range",
        type=float,
        default=1.0,
        help="Clamp noisy ranges to be at least this value in meters (default: 1.0).",
    )
    parser.add_argument(
        "--max-range",
        type=float,
        default=None,
        help="Clamp noisy ranges to be at most this value in meters (default: no clamp).",
    )
    parser.add_argument(
        "--dropout-start",
        type=float,
        default=30.0,
        help="Start distance in meters where dropout begins to increase (default: 30).",
    )
    parser.add_argument(
        "--dropout-max",
        type=float,
        default=0.3,
        help="Maximum probability to drop a point due to weather noise (default: 0.3).",
    )
    parser.add_argument(
        "--dropout-gamma",
        type=float,
        default=1.0,
        help="Exponent applied to dropout ramp for shaping the curve (default: 1.0).",
    )
    parser.add_argument(
        "--intensity-decay",
        type=float,
        default=0.0,
        help="Apply exponential intensity decay coeff (per reference distance). 0 disables.",
    )
    parser.add_argument(
        "--normalize-intensity",
        action="store_true",
        help="Normalize intensity column to [0, 1] if current max exceeds 1.0.",
    )
    parser.add_argument(
        "--intensity-max",
        type=float,
        default=255.0,
        help="Denominator used when normalizing intensity (default: 255).",
    )
    parser.add_argument(
        "--keep-extra-columns",
        action="store_true",
        help="Keep additional point attributes beyond XYZ(I). Dropped rows remove extras too.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible noise generation (default: 42).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of processes for parallel augmentation (default: 1). "
        "Set to 0 to use all available CPU cores.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Split the file list into this many shards and process only one shard (default: 1).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Which shard to process when --shard-count > 1 (0-indexed).",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Stop after processing files with enumeration index <= this value (inclusive).",
    )
    parser.add_argument(
        "--start-file",
        type=str,
        default=None,
        help="Start processing from this filename (inclusive). Earlier files are skipped.",
    )
    parser.add_argument(
        "--end-file",
        type=str,
        default=None,
        help="Stop processing after this filename (inclusive).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing noisy files instead of skipping them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse files and preview operations but skip writing outputs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args(list(argv))


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def normalize_intensity_if_needed(points: np.ndarray, args: argparse.Namespace, file_name: str) -> None:
    intensity = points[:, 3]
    max_intensity = float(np.max(intensity))
    if args.normalize_intensity and max_intensity > 1.0:
        denom = args.intensity_max if args.intensity_max > 0 else max_intensity
        LOGGER.debug("Normalizing intensity for %s (max %.3f, denom %.3f)", file_name, max_intensity, denom)
        points[:, 3] = np.clip(intensity / denom, 0.0, 1.0)
    elif not args.normalize_intensity and max_intensity > 1.0:
        LOGGER.warning(
            "Intensity max %.3f in %s exceeds 1.0 and --normalize-intensity was not set.",
            max_intensity,
            file_name,
        )


def apply_range_noise(points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected (N,>=3) array but got shape {points.shape}")

    extras = None
    if points.shape[1] == 3:
        points = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    elif points.shape[1] > 4 and not args.keep_extra_columns:
        extras = points[:, 4:]
        points = points[:, :4]
    elif points.shape[1] > 4:
        extras = points[:, 4:]

    normalize_intensity_if_needed(points, args, "current file")

    xyz = points[:, :3]
    ranges = np.linalg.norm(xyz, axis=1)

    ref_distance = args.reference_distance if args.reference_distance > 0 else max(float(np.max(ranges)), 1.0)
    sigma = args.base_sigma + args.sigma_gain * (ranges / ref_distance)
    radial_noise = np.random.normal(0.0, sigma)

    base_ranges = np.maximum(ranges, 1e-6)
    directions = xyz / base_ranges[:, None]
    noisy_ranges = ranges + radial_noise

    if args.min_range is not None:
        noisy_ranges = np.maximum(noisy_ranges, args.min_range)
    if args.max_range not in (None, float("inf")):
        noisy_ranges = np.minimum(noisy_ranges, args.max_range)

    noisy_xyz = directions * noisy_ranges[:, None]

    drop_prob = compute_dropout_probability(ranges, args, ref_distance)
    keep_mask = np.random.rand(points.shape[0]) > drop_prob

    out = np.concatenate([noisy_xyz, points[:, 3:4]], axis=1)
    if extras is not None:
        out = np.concatenate([out, extras], axis=1)

    out = out[keep_mask]

    if args.intensity_decay > 0.0 and out.size > 0:
        new_ranges = np.linalg.norm(out[:, :3], axis=1)
        decay = np.exp(-args.intensity_decay * (new_ranges / ref_distance))
        out[:, 3] *= decay

    return out.astype(np.float32, copy=False)


def compute_dropout_probability(ranges: np.ndarray, args: argparse.Namespace, ref_distance: float) -> np.ndarray:
    start = max(args.dropout_start, 0.0)
    slope_den = max(ref_distance - start, 1e-6)
    ramp = np.clip((np.maximum(ranges - start, 0.0) / slope_den), 0.0, 1.0)
    if args.dropout_gamma != 1.0:
        ramp = np.power(ramp, args.dropout_gamma)
    prob = ramp * max(args.dropout_max, 0.0)
    return np.clip(prob, 0.0, 1.0)


def _init_worker(worker_args: dict[str, object], base_seed: int) -> None:
    global _WORKER_ARGS, _WORKER_BASE_SEED
    _WORKER_ARGS = SimpleNamespace(**worker_args)
    _WORKER_BASE_SEED = int(base_seed)
    logging.basicConfig(
        level=logging.DEBUG if getattr(_WORKER_ARGS, "verbose", False) else logging.INFO,
        format="[%(levelname)s][PID %(process)d] %(message)s",
    )


def _process_file_task(task: tuple[int, str, str]) -> None:
    if _WORKER_ARGS is None:
        raise RuntimeError("Worker not initialized. _init_worker must run before processing tasks.")

    idx, input_path_str, output_path_str = task
    input_path = Path(input_path_str)
    output_path = Path(output_path_str)

    if _WORKER_BASE_SEED is not None:
        np.random.seed(_WORKER_BASE_SEED + idx)

    process_single_file(input_path, output_path, _WORKER_ARGS)


def process_single_file(points_path: Path, output_path: Path, args: argparse.Namespace | SimpleNamespace) -> None:
    LOGGER.debug("Loading %s", points_path.name)
    points = np.load(points_path)

    noisy_points = apply_range_noise(points, args)

    if args.dry_run:
        LOGGER.info("[dry-run] Would write %s (%d -> %d points)", output_path.name, points.shape[0], noisy_points.shape[0])
        return

    if output_path.exists() and not args.overwrite:
        LOGGER.info("Skipping %s (already exists). Use --overwrite to regenerate.", output_path.name)
        return

    np.save(output_path, noisy_points)
    LOGGER.info("Saved noisy cloud to %s (%d -> %d points)", output_path, points.shape[0], noisy_points.shape[0])


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    dataset_root = args.dataset_root.resolve()
    input_dir = (dataset_root / args.points_dir).resolve()
    output_dir = (dataset_root / args.output_dir).resolve()

    if not input_dir.exists():
        LOGGER.error("Input directory %s does not exist.", input_dir)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(input_dir.glob("*.npy"))
    if not npy_files:
        LOGGER.warning("No .npy files found under %s", input_dir)
        return 0

    total_files = len(npy_files)

    start_idx = 0
    if args.start_file:
        try:
            start_idx = next(i for i, path in enumerate(npy_files) if path.name == args.start_file)
            if start_idx > 0:
                LOGGER.info(
                    "Skipping %d files. Starting from %s.",
                    start_idx,
                    npy_files[start_idx].name,
                )
        except StopIteration:
            LOGGER.error("Start file %s not found under %s.", args.start_file, input_dir)
            return 1

    end_idx_file = None
    if args.end_file:
        try:
            end_idx_file = next(i for i, path in enumerate(npy_files) if path.name == args.end_file)
        except StopIteration:
            LOGGER.error("End file %s not found under %s.", args.end_file, input_dir)
            return 1
        if end_idx_file < start_idx:
            LOGGER.error("End file %s is before start file %s.", args.end_file, args.start_file)
            return 1

    indexed_files = [
        (idx, path)
        for idx, path in enumerate(npy_files)
        if idx >= start_idx and (end_idx_file is None or idx <= end_idx_file)
    ]

    if args.end_index is not None:
        if args.end_index < 0:
            LOGGER.error("--end-index must be non-negative if provided.")
            return 1
        indexed_files = [(idx, path) for idx, path in indexed_files if idx <= args.end_index]

    if args.shard_count <= 0:
        LOGGER.error("--shard-count must be positive.")
        return 1
    if not (0 <= args.shard_index < args.shard_count):
        LOGGER.error("--shard-index must satisfy 0 <= index < shard-count.")
        return 1
    if args.shard_count > 1:
        indexed_files = [
            (idx, path)
            for idx, path in indexed_files
            if idx % args.shard_count == args.shard_index
        ]

    if not indexed_files:
        LOGGER.warning("No files to process after applying shard/end-index filters.")
        return 0

    LOGGER.info(
        "Applying range noise to %d/%d files from %s -> %s (start-file=%s, end-file=%s, shard %d of %d, end-index=%s)",
        len(indexed_files),
        total_files,
        input_dir,
        output_dir,
        args.start_file or "auto",
        args.end_file or "auto",
        args.shard_index,
        args.shard_count,
        "none" if args.end_index is None else args.end_index,
    )

    use_workers = args.num_workers if args.num_workers != 0 else (os.cpu_count() or 1)
    if use_workers <= 1:
        for idx, npy_path in indexed_files:
            np.random.seed(args.seed + idx)
            output_path = output_dir / npy_path.name
            process_single_file(npy_path, output_path, args)
    else:
        LOGGER.info("Using %d worker processes.", use_workers)
        worker_args = {
            "base_sigma": args.base_sigma,
            "sigma_gain": args.sigma_gain,
            "reference_distance": args.reference_distance,
            "min_range": args.min_range,
            "max_range": args.max_range,
            "dropout_start": args.dropout_start,
            "dropout_max": args.dropout_max,
            "dropout_gamma": args.dropout_gamma,
            "intensity_decay": args.intensity_decay,
            "normalize_intensity": args.normalize_intensity,
            "intensity_max": args.intensity_max,
            "keep_extra_columns": args.keep_extra_columns,
            "dry_run": args.dry_run,
            "overwrite": args.overwrite,
            "verbose": args.verbose,
        }
        tasks = [
            (idx, str(path), str(output_dir / path.name))
            for idx, path in indexed_files
        ]

        with ProcessPoolExecutor(
            max_workers=use_workers,
            initializer=_init_worker,
            initargs=(worker_args, args.seed),
        ) as executor:
            for _ in executor.map(_process_file_task, tasks):
                pass

    LOGGER.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

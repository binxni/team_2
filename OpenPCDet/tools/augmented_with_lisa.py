#!/usr/bin/env python3
"""
Apply LISA augmentation to a directory of numpy point clouds.

Example
-------
    python tools/augmented_with_lisa.py \
        --dataset-root data/custom_av_64 \
        --points-dir points \
        --output-dir points_lisa \
        --atm-model rain \
        --rain-rate 10.0 \
        --normalize-intensity
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

try:
    from pylisa.lisa import Lisa
except ImportError as exc:  # pragma: no cover - imported at runtime
    raise ImportError(
        "Failed to import pylisa. Install it with `pip install -e LISA/` from the repo root."
    ) from exc


LOGGER = logging.getLogger("augment_with_lisa")


ATM_MODELS_MC: tuple[str, ...] = ("rain", "snow")
ATM_MODELS_AVG: tuple[str, ...] = ("chu_hogg_fog", "strong_advection_fog", "moderate_advection_fog")
ATM_MODELS_ALL: tuple[str, ...] = ATM_MODELS_MC + ATM_MODELS_AVG

_WORKER_LISA: Lisa | None = None
_WORKER_ARGS: SimpleNamespace | None = None
_WORKER_BASE_SEED: int | None = None


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment numpy point clouds with the LISA weather simulation."
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
        default=Path("points_lisa"),
        help="Relative path (from dataset root) where augmented clouds will be written.",
    )
    parser.add_argument(
        "--atm-model",
        choices=ATM_MODELS_ALL,
        default="rain",
        help="Atmospheric model to simulate (default: rain).",
    )
    parser.add_argument(
        "--mode",
        choices=("strongest", "last"),
        default="strongest",
        help="Return mode for LISA Monte-Carlo augmentation (default: strongest).",
    )
    parser.add_argument(
        "--rain-rate",
        type=float,
        default=None,
        help="Rain/snow rate in mm/hr. Required for rain/snow models.",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=905.0,
        help="LiDAR wavelength in nm (default: 905).",
    )
    parser.add_argument(
        "--rmax",
        type=float,
        default=200.0,
        help="Maximum LiDAR range in meters (default: 200).",
    )
    parser.add_argument(
        "--rmin",
        type=float,
        default=1.5,
        help="Minimum LiDAR range in meters (default: 1.5).",
    )
    parser.add_argument(
        "--bdiv",
        type=float,
        default=3e-3,
        help="Beam divergence in radians (default: 3e-3).",
    )
    parser.add_argument(
        "--dst",
        type=float,
        default=0.05,
        help="Droplet diameter starting point in mm (default: 0.05).",
    )
    parser.add_argument(
        "--dR",
        type=float,
        default=0.09,
        help="Range accuracy in meters (default: 0.09).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible Monte-Carlo sampling (default: 42).",
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
        "--keep-label-column",
        action="store_true",
        help="Retain the 5th column (LISA labels) in the saved output. "
        "By default only the first 4 columns are kept for OpenPCDet training.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing augmented files instead of skipping them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse files and initialize LISA but skip writing outputs.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of processes for parallel augmentation (default: 1). "
        "Set to 0 to use all available CPU cores.",
    )
    parser.add_argument(
        "--start-file",
        type=str,
        default=None,
        help="Start processing from this filename (inclusive). Earlier files are skipped.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit the number of files processed in this run (e.g., 1000 for chunked runs).",
    )
    return parser.parse_args(list(argv))


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


def build_lisa(args: argparse.Namespace) -> Lisa:
    lisa = Lisa(
        lam=args.lam,
        rmax=args.rmax,
        rmin=args.rmin,
        bdiv=args.bdiv,
        dst=args.dst,
        dR=args.dR,
        atm_model=args.atm_model,
        mode=args.mode,
    )
    LOGGER.debug("Initialized LISA with params: %s", lisa.__dict__)
    return lisa


def lisa_kwargs_from_args(args: argparse.Namespace) -> dict[str, float | str]:
    return {
        "lam": args.lam,
        "rmax": args.rmax,
        "rmin": args.rmin,
        "bdiv": args.bdiv,
        "dst": args.dst,
        "dR": args.dR,
        "atm_model": args.atm_model,
        "mode": args.mode,
    }


def _init_worker(
    lisa_kwargs: dict[str, float | str],
    worker_args: dict[str, object],
    base_seed: int,
) -> None:
    global _WORKER_LISA, _WORKER_ARGS, _WORKER_BASE_SEED
    _WORKER_LISA = Lisa(**lisa_kwargs)
    _WORKER_ARGS = SimpleNamespace(**worker_args)
    _WORKER_BASE_SEED = int(base_seed)
    logging.basicConfig(
        level=logging.DEBUG if getattr(_WORKER_ARGS, "verbose", False) else logging.INFO,
        format="[%(levelname)s][PID %(process)d] %(message)s",
    )


def _process_file_task(task: tuple[int, str, str]) -> None:
    if _WORKER_LISA is None or _WORKER_ARGS is None:
        raise RuntimeError("Worker not initialized. _init_worker must run before processing tasks.")

    idx, input_path_str, output_path_str = task
    input_path = Path(input_path_str)
    output_path = Path(output_path_str)

    if _WORKER_BASE_SEED is not None:
        np.random.seed(_WORKER_BASE_SEED + idx)

    augment_single_file(_WORKER_LISA, input_path, output_path, _WORKER_ARGS)


def normalize_intensity_if_needed(points: np.ndarray, args: argparse.Namespace, file_name: str) -> None:
    intensity = points[:, 3]
    max_intensity = float(np.max(intensity))
    if args.normalize_intensity and max_intensity > 1.0:
        denom = args.intensity_max if args.intensity_max > 0 else max_intensity
        LOGGER.debug("Normalizing intensity for %s (max %.3f, denom %.3f)", file_name, max_intensity, denom)
        points[:, 3] = np.clip(intensity / denom, 0.0, 1.0)
    elif not args.normalize_intensity and max_intensity > 1.0:
        LOGGER.warning(
            "Intensity max %.3f in %s exceeds 1.0 and --normalize-intensity was not set. "
            "LISA expects reflectivity in [0, 1].",
            max_intensity,
            file_name,
        )


def augment_single_file(
    lisa: Lisa,
    points_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    LOGGER.debug("Loading %s", points_path.name)
    points = np.load(points_path)
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(f"{points_path} does not contain an (N,4) array; got shape {points.shape}")

    points = points[:, :4].astype(np.float32, copy=False)
    normalize_intensity_if_needed(points, args, points_path.name)

    if lisa.atm_model in ATM_MODELS_MC:
        if args.rain_rate is None:
            raise ValueError("--rain-rate must be provided for rain/snow models.")
        augmented = lisa.augment(points, args.rain_rate)
    else:
        augmented = lisa.augment(points)

    if not args.keep_label_column and augmented.shape[1] > 4:
        augmented = augmented[:, :4]

    augmented = augmented.astype(np.float32, copy=False)

    if args.dry_run:
        LOGGER.info("[dry-run] Would write %s", output_path.name)
        return

    if output_path.exists() and not args.overwrite:
        LOGGER.info("Skipping %s (already exists). Use --overwrite to regenerate.", output_path.name)
        return

    np.save(output_path, augmented)
    LOGGER.info("Saved augmented cloud to %s", output_path)


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

    start_idx = 0
    if args.start_file:
        try:
            start_idx = next(i for i, path in enumerate(npy_files) if path.name == args.start_file)
            if start_idx > 0:
                LOGGER.info("Skipping %d files. Starting from %s.", start_idx, npy_files[start_idx].name)
        except StopIteration:
            LOGGER.error("Start file %s not found under %s.", args.start_file, input_dir)
            return 1

    npy_files = npy_files[start_idx:]

    if not npy_files:
        LOGGER.warning("No files to process after applying --start-file filter.")
        return 0

    if args.max_files is not None:
        if args.max_files <= 0:
            LOGGER.error("--max-files must be positive if provided.")
            return 1
        npy_files = npy_files[:args.max_files]

    if not npy_files:
        LOGGER.warning("No files to process after applying --max-files limit.")
        return 0

    indexed_files = [
        (idx, path) for idx, path in enumerate(npy_files, start=start_idx)
    ]

    LOGGER.info(
        "Augmenting %d files from %s -> %s with model=%s",
        len(npy_files),
        input_dir,
        output_dir,
        args.atm_model,
    )

    use_workers = args.num_workers if args.num_workers != 0 else (os.cpu_count() or 1)
    if use_workers <= 1:
        lisa = build_lisa(args)
        for file_idx, npy_path in indexed_files:
            np.random.seed(args.seed + file_idx)
            output_path = output_dir / npy_path.name
            augment_single_file(lisa, npy_path, output_path, args)
    else:
        LOGGER.info("Using %d worker processes.", use_workers)
        lisa_kwargs = lisa_kwargs_from_args(args)
        worker_args = {
            "normalize_intensity": args.normalize_intensity,
            "intensity_max": args.intensity_max,
            "rain_rate": args.rain_rate,
            "keep_label_column": args.keep_label_column,
            "dry_run": args.dry_run,
            "overwrite": args.overwrite,
            "verbose": args.verbose,
        }
        tasks = [
            (file_idx, str(path), str(output_dir / path.name))
            for file_idx, path in indexed_files
        ]

        with ProcessPoolExecutor(
            max_workers=use_workers,
            initializer=_init_worker,
            initargs=(lisa_kwargs, worker_args, args.seed),
        ) as executor:
            for _ in executor.map(_process_file_task, tasks):
                pass

    LOGGER.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

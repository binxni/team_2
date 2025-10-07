#!/usr/bin/env python3
"""Batch-apply the LIDSOR filter to point clouds and save the results."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from easydict import EasyDict

try:
    __import__("open3d")  # Ensure the dependency is available before work starts
except ImportError as exc:  # pragma: no cover - early feedback when Open3D is missing
    raise ImportError("Open3D is required to run the LIDSOR filter offline.") from exc

from pcdet.config import cfg_from_yaml_file
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply the LIDSOR filter to an entire point dataset.")
    parser.add_argument(
        "--cfg",
        type=Path,
        default=Path("tools/cfgs/dataset_configs/custom_av_dataset_aligned64.yaml"),
        help="Dataset config that contains the lidsor_filter step.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory of the dataset (fallbacks to DATA_PATH in the config).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory with raw point clouds (defaults to <data-root>/points).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store filtered point clouds (defaults to <data-root>/points_filter).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".npy",
        help="File extension to pick up and emit (default: .npy).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing outputs instead of skipping them.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N files (useful for smoke tests).",
    )
    return parser.parse_args()


def resolve_data_root(cfg_path: Path, override: Optional[Path], dataset_cfg: EasyDict) -> Path:
    if override is not None:
        return override.resolve()

    data_path = Path(dataset_cfg.DATA_PATH)
    if data_path.is_absolute():
        return data_path.resolve()

    cfg_dir = cfg_path.resolve().parent
    for candidate in [cfg_dir] + list(cfg_dir.parents):
        resolved = (candidate / data_path).resolve()
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        f"Unable to resolve DATA_PATH='{dataset_cfg.DATA_PATH}' relative to {cfg_path}. Provide --data-root explicitly."
    )


def enumerate_point_files(points_dir: Path, suffix: str, limit: Optional[int]) -> List[Path]:
    if not points_dir.exists():
        raise FileNotFoundError(f"Points directory not found: {points_dir}")

    files = sorted(path for path in points_dir.iterdir() if path.suffix == suffix)
    if limit is not None:
        files = files[:limit]
    if not files:
        raise RuntimeError(f"No files with suffix '{suffix}' found in {points_dir}")
    return files


def get_lidsor_config(processor_cfgs: Iterable) -> dict:
    for cfg in processor_cfgs:
        if cfg.NAME == "lidsor_filter":
            return cfg
    raise RuntimeError("The provided dataset config does not define a lidsor_filter step.")


def main() -> None:
    args = parse_args()

    dataset_cfg = EasyDict()
    cfg_from_yaml_file(args.cfg, dataset_cfg)

    data_root = resolve_data_root(args.cfg, args.data_root, dataset_cfg)
    points_dir = (args.input_dir or (data_root / "points")).resolve()
    output_dir = (args.output_dir or (data_root / "points_filter")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = enumerate_point_files(points_dir, args.suffix, args.limit)

    point_cloud_range = np.asarray(dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
    feature_encoder = PointFeatureEncoder(dataset_cfg.POINT_FEATURE_ENCODING, point_cloud_range)
    processor = DataProcessor(
        dataset_cfg.DATA_PROCESSOR,
        point_cloud_range=point_cloud_range,
        training=False,
        num_point_features=feature_encoder.num_point_features,
    )
    lidsor_cfg = get_lidsor_config(dataset_cfg.DATA_PROCESSOR)

    total_removed = 0
    print(f"Filtering {len(files)} files from {points_dir} -> {output_dir}")

    for src_path in files:
        dst_path = output_dir / src_path.name
        if dst_path.exists() and not args.overwrite:
            continue

        points = np.load(src_path)
        filtered = processor.lidsor_filter(data_dict={"points": points.copy()}, config=lidsor_cfg)
        filtered_points = filtered["points"]

        np.save(dst_path, filtered_points)
        total_removed += points.shape[0] - filtered_points.shape[0]

    print(f"Done. Removed a total of {total_removed} points.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # pragma: no cover - surfacing context on failure
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

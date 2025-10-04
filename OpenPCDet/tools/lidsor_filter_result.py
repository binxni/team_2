"""Apply the LIDSOR denoising filter to a single point cloud and save the result."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from easydict import EasyDict

from pcdet.config import cfg_from_yaml_file
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LIDSOR filter on one frame and dump the points")
    parser.add_argument(
        "--cfg",
        type=Path,
        default=Path("tools/cfgs/dataset_configs/custom_av_dataset.yaml"),
        help="Dataset config that defines DATA_PROCESSOR with lidsor_filter",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override dataset root (points/, ImageSets/). Defaults to path derived from config",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="ImageSets split file name (without .txt). Used when --frame-id is not provided",
    )
    parser.add_argument(
        "--frame-id",
        type=str,
        default=None,
        help="Frame identifier (e.g., 000001). If omitted, --frame-idx is used",
    )
    parser.add_argument(
        "--frame-idx",
        type=int,
        default=0,
        help="Index into the split file when frame-id is not supplied",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to store the filtered point cloud",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["npy", "txt"],
        default="npy",
        help="Output format",
    )
    parser.add_argument(
        "--save-original",
        type=Path,
        default=None,
        help="Optional path to save the raw (unfiltered) points",
    )
    return parser.parse_args()


def resolve_data_root(cfg_path: Path, override: Path | None, dataset_cfg: EasyDict) -> Path:
    if override is not None:
        return override.resolve()
    cfg_dir = cfg_path.parent
    return (cfg_dir / dataset_cfg.DATA_PATH).resolve()


def load_split_ids(data_root: Path, split: str) -> List[str]:
    split_file = data_root / "ImageSets" / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Cannot locate split file: {split_file}")
    return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]


def save_points(points: np.ndarray, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npy":
        np.save(path, points)
    else:
        np.savetxt(path, points, fmt="%.6f")


def main() -> None:
    args = parse_args()

    dataset_cfg = EasyDict()
    cfg_from_yaml_file(args.cfg, dataset_cfg)

    data_root = resolve_data_root(args.cfg, args.data_root, dataset_cfg)

    frame_id = args.frame_id
    if frame_id is None:
        split_ids = load_split_ids(data_root, args.split)
        if not split_ids:
            raise RuntimeError(f"Split file for '{args.split}' is empty")
        frame_id = split_ids[args.frame_idx % len(split_ids)]

    points_path = data_root / "points" / f"{frame_id}.npy"
    if not points_path.exists():
        raise FileNotFoundError(f"Point file not found: {points_path}")

    points = np.load(points_path)

    if args.save_original is not None:
        save_points(points, args.save_original, args.format)

    point_cloud_range = np.array(dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
    point_encoder = PointFeatureEncoder(dataset_cfg.POINT_FEATURE_ENCODING, point_cloud_range)
    processor = DataProcessor(
        dataset_cfg.DATA_PROCESSOR,
        point_cloud_range=point_cloud_range,
        training=False,
        num_point_features=point_encoder.num_point_features,
    )

    lidsor_cfg = None
    for proc_cfg in dataset_cfg.DATA_PROCESSOR:
        if proc_cfg.NAME == "lidsor_filter":
            lidsor_cfg = proc_cfg
            break
    if lidsor_cfg is None:
        raise RuntimeError("The provided config does not define a lidsor_filter step")

    filtered = processor.lidsor_filter(
        data_dict={"points": points.copy()},
        config=lidsor_cfg,
    )

    filtered_points = filtered["points"]
    save_points(filtered_points, args.output, args.format)

    reduction = points.shape[0] - filtered_points.shape[0]
    print(f"Frame {frame_id}: {points.shape[0]} -> {filtered_points.shape[0]} points (removed {reduction})")


if __name__ == "__main__":
    main()


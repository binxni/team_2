#!/usr/bin/env python3
"""Utility script to compare point cloud downsampling outputs.

This script loads a single point cloud, applies the standard data processor
pipeline (as defined in the dataset config) to obtain the voxel-based
representation, and then re-runs the pipeline with the newly added
`voxel_mean_downsample` processor to generate one representative point per
voxel. Both results are written to .npy files for offline inspection.
"""

import argparse
import copy
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from easydict import EasyDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump baseline and voxel-mean downsampled point clouds to npy files"
    )
    parser.add_argument("--cfg_file", required=True, help="Dataset config YAML path")
    parser.add_argument("--point_cloud", required=True, help="Point cloud file (.bin or .npy)")
    parser.add_argument("--baseline_output", required=True, help="Output path for baseline result (.npy)")
    parser.add_argument(
        "--voxel_mean_output",
        required=True,
        help="Output path for voxel-mean result (.npy)",
    )
    parser.add_argument(
        "--save_counts",
        action="store_true",
        help="If set, store per-voxel point counts alongside voxel-mean points",
    )
    return parser.parse_args()


def load_points(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        points = np.load(path)
    elif suffix == ".bin":
        raw = np.fromfile(str(path), dtype=np.float32)
        if raw.size % 4 != 0:
            raise ValueError(f"Unexpected bin size {raw.size}, cannot reshape to N x 4")
        points = raw.reshape(-1, 4)
    else:
        raise ValueError(f"Unsupported point cloud format: {suffix}")

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Point cloud must be (N, >=3), got {points.shape}")

    return points.astype(np.float32, copy=False)


def build_processors(dataset_cfg: EasyDict, save_counts: bool) -> Tuple[PointFeatureEncoder, DataProcessor, DataProcessor]:
    point_cloud_range = np.array(dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
    point_feature_encoder = PointFeatureEncoder(
        dataset_cfg.POINT_FEATURE_ENCODING,
        point_cloud_range=point_cloud_range,
    )

    baseline_processor = DataProcessor(
        dataset_cfg.DATA_PROCESSOR,
        point_cloud_range=point_cloud_range,
        training=False,
        num_point_features=point_feature_encoder.num_point_features,
    )

    voxel_cfg_list = copy.deepcopy(dataset_cfg.DATA_PROCESSOR)
    replaced = False
    for idx, proc_cfg in enumerate(voxel_cfg_list):
        if proc_cfg.NAME == "transform_points_to_voxels":
            voxel_mean_cfg = EasyDict({
                "NAME": "voxel_mean_downsample",
                "VOXEL_SIZE": copy.deepcopy(proc_cfg.VOXEL_SIZE),
                "MAX_POINTS_PER_VOXEL": proc_cfg.get("MAX_POINTS_PER_VOXEL", 5),
                "MAX_NUMBER_OF_VOXELS": copy.deepcopy(proc_cfg.MAX_NUMBER_OF_VOXELS),
            })
            if save_counts:
                voxel_mean_cfg["SAVE_NUM_POINTS"] = True
            voxel_cfg_list[idx] = voxel_mean_cfg
            replaced = True
            break

    if not replaced:
        raise ValueError("Config does not contain 'transform_points_to_voxels'; cannot derive voxel parameters")

    voxel_mean_processor = DataProcessor(
        voxel_cfg_list,
        point_cloud_range=point_cloud_range,
        training=False,
        num_point_features=point_feature_encoder.num_point_features,
    )

    return point_feature_encoder, baseline_processor, voxel_mean_processor


def main():
    args = parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    dataset_cfg = cfg.get('DATA_CONFIG', cfg)

    feature_encoder, baseline_processor, voxel_mean_processor = build_processors(dataset_cfg, args.save_counts)

    point_path = Path(args.point_cloud)
    points = load_points(point_path)

    data_dict = {"points": points}
    data_dict = feature_encoder.forward(data_dict)

    baseline_input = copy.deepcopy(data_dict)
    voxel_mean_input = copy.deepcopy(data_dict)

    baseline_processed = baseline_processor.forward(baseline_input)
    if "voxels" not in baseline_processed:
        raise RuntimeError("Baseline processor did not produce voxels; check DATA_PROCESSOR config")

    voxel_mean_processed = voxel_mean_processor.forward(voxel_mean_input)
    if "points" not in voxel_mean_processed:
        raise RuntimeError("Voxel-mean processor did not produce points; ensure voxel_mean_downsample is configured")

    voxels = baseline_processed["voxels"]
    voxel_num_points = baseline_processed["voxel_num_points"]
    max_points_per_voxel = voxels.shape[1]
    valid_mask = np.arange(max_points_per_voxel)[None, :] < voxel_num_points[:, None]
    baseline_points = voxels[valid_mask]

    baseline_payload = {
        "voxels": voxels,
        "voxel_coords": baseline_processed["voxel_coords"],
        "voxel_num_points": voxel_num_points,
        "points": baseline_points,
    }

    voxel_mean_payload = {"points": voxel_mean_processed["points"]}
    if args.save_counts and "mean_points_per_voxel" in voxel_mean_processed:
        voxel_mean_payload["mean_points_per_voxel"] = voxel_mean_processed["mean_points_per_voxel"]

    baseline_out = Path(args.baseline_output)
    voxel_mean_out = Path(args.voxel_mean_output)
    baseline_out.parent.mkdir(parents=True, exist_ok=True)
    voxel_mean_out.parent.mkdir(parents=True, exist_ok=True)

    np.save(baseline_out, baseline_payload)
    np.save(voxel_mean_out, voxel_mean_payload)

    print(f"Baseline voxels saved to {baseline_out}")
    print(f"Voxel-mean points saved to {voxel_mean_out}")


if __name__ == "__main__":
    main()

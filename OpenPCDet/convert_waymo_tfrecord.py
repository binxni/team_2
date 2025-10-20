#!/usr/bin/env python3
"""
Usage:
  python convert_waymo_tfrecord.py \
      --tfrecord path/to/segment-xxxx.tfrecord \
      --out-dir /path/to/output \
      --lasers TOP \
      --returns 1,2 \
      --sample 200000
"""

import argparse
import glob
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Dict, Any

import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils

LASER_NAME_MAP = {
    "TOP": dataset_pb2.LaserName.TOP,
    "FRONT": dataset_pb2.LaserName.FRONT,
    "SIDE_LEFT": dataset_pb2.LaserName.SIDE_LEFT,
    "SIDE_RIGHT": dataset_pb2.LaserName.SIDE_RIGHT,
    "REAR": dataset_pb2.LaserName.REAR,
}

# Human-friendly class names for Waymo label types (robust numeric map)
CLASS_NAME_MAP = {
    0: 'Unknown',
    1: 'Vehicle',
    2: 'Pedestrian',
    3: 'Sign',
    4: 'Cyclist',
}

# Keep only these classes when saving labels (numeric ids)
ALLOWED_LABEL_TYPES = {1, 2, 4}

def parse_returns(arg: str) -> Sequence[int]:
    tokens = [token.strip() for token in arg.split(",") if token.strip()]
    if not tokens:
        return [0]
    indices = []
    for token in tokens:
        if token not in {"1", "2"}:
            raise ValueError(f"Unsupported return index: {token}")
        indices.append(int(token) - 1)
    return indices

def parse_lasers(arg: str) -> Optional[Sequence[int]]:
    arg = arg.strip().upper()
    if arg == "ALL":
        return None
    indices = []
    for token in arg.split(","):
        token = token.strip().upper()
        if not token:
            continue
        if token not in LASER_NAME_MAP:
            valid = ", ".join(LASER_NAME_MAP.keys())
            raise ValueError(f"Unknown laser '{token}'. Valid: {valid}, ALL")
        indices.append(LASER_NAME_MAP[token])
    return indices or None

def maybe_subsample(points: np.ndarray, target: int) -> np.ndarray:
    if target <= 0 or target >= len(points):
        return points
    idx = np.random.choice(len(points), target, replace=False)
    return points[idx]

def frame_points(
    frame: dataset_pb2.Frame,
    return_indices: Sequence[int],
    allowed_lasers: Optional[Sequence[int]],
) -> List[np.ndarray]:
    # Local parser to avoid bytearray/bytes incompat in some Waymo versions
    def _parse_range_image_and_camera_projection_compat(frame: dataset_pb2.Frame):
        range_images = {}
        camera_projections = {}
        seg_labels = {}
        range_image_top_pose: dataset_pb2.MatrixFloat = dataset_pb2.MatrixFloat()
        for laser in frame.lasers:
            # First return
            if len(laser.ri_return1.range_image_compressed) > 0:
                range_image_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.range_image_compressed, 'ZLIB')
                ri = dataset_pb2.MatrixFloat()
                ri.ParseFromString(bytes(range_image_str_tensor.numpy()))
                range_images[laser.name] = [ri]

                if laser.name == dataset_pb2.LaserName.TOP:
                    range_image_top_pose_str_tensor = tf.io.decode_compressed(
                        laser.ri_return1.range_image_pose_compressed, 'ZLIB')
                    range_image_top_pose = dataset_pb2.MatrixFloat()
                    range_image_top_pose.ParseFromString(
                        bytes(range_image_top_pose_str_tensor.numpy()))

                camera_projection_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.camera_projection_compressed, 'ZLIB')
                cp = dataset_pb2.MatrixInt32()
                cp.ParseFromString(bytes(camera_projection_str_tensor.numpy()))
                camera_projections[laser.name] = [cp]

                if len(laser.ri_return1.segmentation_label_compressed) > 0:
                    seg_label_str_tensor = tf.io.decode_compressed(
                        laser.ri_return1.segmentation_label_compressed, 'ZLIB')
                    seg_label = dataset_pb2.MatrixInt32()
                    seg_label.ParseFromString(bytes(seg_label_str_tensor.numpy()))
                    seg_labels[laser.name] = [seg_label]

            # Second return
            if len(laser.ri_return2.range_image_compressed) > 0:
                range_image_str_tensor = tf.io.decode_compressed(
                    laser.ri_return2.range_image_compressed, 'ZLIB')
                ri = dataset_pb2.MatrixFloat()
                ri.ParseFromString(bytes(range_image_str_tensor.numpy()))
                # Ensure list exists from first return; otherwise start it
                range_images.setdefault(laser.name, []).append(ri)

                camera_projection_str_tensor = tf.io.decode_compressed(
                    laser.ri_return2.camera_projection_compressed, 'ZLIB')
                cp = dataset_pb2.MatrixInt32()
                cp.ParseFromString(bytes(camera_projection_str_tensor.numpy()))
                camera_projections.setdefault(laser.name, []).append(cp)

                if len(laser.ri_return2.segmentation_label_compressed) > 0:
                    seg_label_str_tensor = tf.io.decode_compressed(
                        laser.ri_return2.segmentation_label_compressed, 'ZLIB')
                    seg_label = dataset_pb2.MatrixInt32()
                    seg_label.ParseFromString(bytes(seg_label_str_tensor.numpy()))
                    seg_labels.setdefault(laser.name, []).append(seg_label)
        return range_images, camera_projections, seg_labels, range_image_top_pose

    # Use local compat parser instead of the package one to avoid bytearray issue
    range_images, camera_projections, _seg_labels, range_image_top_pose = _parse_range_image_and_camera_projection_compat(frame)

    # Local conversion adapted from OpenPCDet to avoid torch dependency
    def _convert_range_image_to_point_cloud_local(frame, range_images, camera_projections, range_image_top_pose, ri_index=(0,)):
        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        points = []
        cp_points = []
        points_NLZ = []
        points_intensity = []
        points_elongation = []

        frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
        )
        range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
            range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
            range_image_top_pose_tensor[..., 2])
        range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation,
            range_image_top_pose_tensor_translation)

        for c in calibrations:
            points_single, cp_points_single, points_NLZ_single, points_intensity_single, points_elongation_single = [], [], [], [], []
            for cur_ri_index in ri_index:
                try:
                    ri_list = range_images[c.name]
                    if ri_list is None:
                        continue
                    if isinstance(ri_list, (list, tuple)):
                        if cur_ri_index >= len(ri_list) or ri_list[cur_ri_index] is None:
                            continue
                        range_image = ri_list[cur_ri_index]
                    else:
                        # Unexpected structure, skip
                        continue
                except Exception:
                    continue

                if len(c.beam_inclinations) == 0:
                    beam_inclinations = range_image_utils.compute_inclination(
                        tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                        height=range_image.shape.dims[0])
                else:
                    beam_inclinations = tf.constant(c.beam_inclinations)

                beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
                extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

                range_image_tensor = tf.reshape(
                    tf.convert_to_tensor(range_image.data), range_image.shape.dims)
                pixel_pose_local = None
                frame_pose_local = None
                if c.name == dataset_pb2.LaserName.TOP:
                    pixel_pose_local = range_image_top_pose_tensor
                    pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                    frame_pose_local = tf.expand_dims(frame_pose, axis=0)
                range_image_mask = range_image_tensor[..., 0] > 0
                range_image_NLZ = range_image_tensor[..., 3]
                range_image_intensity = range_image_tensor[..., 1]
                range_image_elongation = range_image_tensor[..., 2]
                range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                    tf.expand_dims(range_image_tensor[..., 0], axis=0),
                    tf.expand_dims(extrinsic, axis=0),
                    tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
                    pixel_pose=pixel_pose_local,
                    frame_pose=frame_pose_local)

                range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
                points_tensor = tf.gather_nd(range_image_cartesian, tf.where(range_image_mask))
                points_NLZ_tensor = tf.gather_nd(range_image_NLZ, tf.compat.v1.where(range_image_mask))
                points_intensity_tensor = tf.gather_nd(range_image_intensity, tf.compat.v1.where(range_image_mask))
                points_elongation_tensor = tf.gather_nd(range_image_elongation, tf.compat.v1.where(range_image_mask))
                # Choose matching camera projection return if available
                cp_list = camera_projections.get(c.name, [])
                cp_idx = 0
                if isinstance(cp_list, (list, tuple)) and len(cp_list) > 1 and cur_ri_index < len(cp_list):
                    cp_idx = cur_ri_index
                cp = cp_list[cp_idx]
                cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
                cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))

                points_single.append(points_tensor.numpy())
                cp_points_single.append(cp_points_tensor.numpy())
                points_NLZ_single.append(points_NLZ_tensor.numpy())
                points_intensity_single.append(points_intensity_tensor.numpy())
                points_elongation_single.append(points_elongation_tensor.numpy())

            if points_single:
                points.append(np.concatenate(points_single, axis=0))
                cp_points.append(np.concatenate(cp_points_single, axis=0))
                points_NLZ.append(np.concatenate(points_NLZ_single, axis=0))
                points_intensity.append(np.concatenate(points_intensity_single, axis=0))
                points_elongation.append(np.concatenate(points_elongation_single, axis=0))
            else:
                points.append(np.zeros((0, 3), dtype=np.float32))
                cp_points.append(np.zeros((0, 6), dtype=np.float32))
                points_NLZ.append(np.zeros((0,), dtype=np.float32))
                points_intensity.append(np.zeros((0,), dtype=np.float32))
                points_elongation.append(np.zeros((0,), dtype=np.float32))

        return points, cp_points, points_NLZ, points_intensity, points_elongation

    clouds: List[np.ndarray] = []
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    for ri in return_indices:
        ri_tuple = (int(ri),)
        points_list, _cp_points, _nlz, intensity_list, _elongation = _convert_range_image_to_point_cloud_local(
            frame, range_images, camera_projections, range_image_top_pose, ri_index=ri_tuple
        )
        for calib, point_array, intensity_array in zip(calibrations, points_list, intensity_list):
            if allowed_lasers is not None and calib.name not in allowed_lasers:
                continue
            # x,y,z,intensity를 포함한 포인트 클라우드 생성
            if len(point_array) > 0 and len(intensity_array) > 0:
                # intensity를 4번째 컬럼으로 추가
                points_with_intensity = np.column_stack((
                    point_array.astype(np.float32, copy=False),
                    intensity_array.astype(np.float32, copy=False)
                ))
                clouds.append(points_with_intensity)
            else:
                # 빈 배열인 경우 4개 컬럼(x,y,z,intensity)으로 생성
                empty_cloud = np.zeros((0, 4), dtype=np.float32)
                clouds.append(empty_cloud)
    return clouds

def _expand_tfrecords(path_like: Path) -> List[str]:
    pattern = str(path_like)
    paths: List[str] = []
    if any(ch in pattern for ch in ['*', '?', '[']):
        paths = sorted(glob.glob(pattern))
    else:
        p = Path(pattern)
        if p.is_dir():
            paths = sorted(str(f) for f in p.glob('*.tfrecord'))
        elif p.is_file():
            paths = [str(p)]
    return [p for p in paths if Path(p).is_file()]


def convert(args: argparse.Namespace) -> None:
    record_paths = _expand_tfrecords(args.tfrecord)
    if not record_paths:
        raise FileNotFoundError(f"TFRecord not found: {args.tfrecord}. "
                                f"Provide an existing .tfrecord file, directory, or glob pattern.")
    dataset = tf.data.TFRecordDataset(record_paths, compression_type="")
    return_indices = parse_returns(args.returns)
    allowed_lasers = parse_lasers(args.lasers)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_dir: Optional[Path] = None
    if getattr(args, 'save_labels', False):
        # If a separate labels output dir is provided, use it; otherwise, default to <out-dir>/labels
        if getattr(args, 'labels_out_dir', None):
            labels_dir = Path(args.labels_out_dir)
        else:
            labels_dir = out_dir / 'labels'
        labels_dir.mkdir(parents=True, exist_ok=True)

    def save_labels_json(frame: dataset_pb2.Frame, path: Path) -> None:
        labels_out: List[Dict[str, Any]] = []
        for lab in frame.laser_labels:
            if int(lab.type) not in ALLOWED_LABEL_TYPES:
                continue
            box = lab.box
            cls_name = CLASS_NAME_MAP.get(int(lab.type), str(int(lab.type)))
            labels_out.append({
                "name": cls_name,
                "id": lab.id,
                "difficulty_det": int(getattr(lab, 'detection_difficulty_level', 0)),
                "difficulty_trk": int(getattr(lab, 'tracking_difficulty_level', 0)),
                "num_points": int(getattr(lab, 'num_lidar_points_in_box', 0)),
                "box": [
                    float(box.center_x), float(box.center_y), float(box.center_z),
                    float(box.length), float(box.width), float(box.height),
                    float(box.heading),
                ],
            })
        import json
        with path.open('w') as f:
            json.dump({
                "context_name": frame.context.name,
                "timestamp_micros": int(frame.timestamp_micros),
                "labels": labels_out,
            }, f)

    def save_labels_txt(frame: dataset_pb2.Frame, path: Path) -> None:
        """Save labels as plain lines: x y z dx dy dz yaw class"""
        lines: List[str] = []
        for lab in frame.laser_labels:
            if int(lab.type) not in ALLOWED_LABEL_TYPES:
                continue
            box = lab.box
            cls_name = CLASS_NAME_MAP.get(int(lab.type), 'Unknown')
            # x y z dx dy dz yaw class
            line = (
                f"{box.center_x:.6f} {box.center_y:.6f} {box.center_z:.6f} "
                f"{box.length:.6f} {box.width:.6f} {box.height:.6f} {box.heading:.6f} {cls_name}"
            )
            lines.append(line)
        with path.open('w', encoding='utf-8') as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    # Determine starting index by scanning existing files unless overridden
    def _find_next_index(data_dir: Path, lbl_dir: Optional[Path], labels_fmt: str) -> int:
        max_idx = 5000000 - 1  # 005로 시작하도록 기본값을 5000000-1로 설정
        def scan_dir(d: Optional[Path], exts):
            nonlocal max_idx
            if d is None or not d.exists():
                return
            for p in d.iterdir():
                if not p.is_file():
                    continue
                name = p.name
                for ext in exts:
                    if name.endswith(ext):
                        stem = name[:-len(ext)]
                        if len(stem) == 8 and stem.isdigit():
                            idx = int(stem)
                            if idx > max_idx:
                                max_idx = idx
        scan_dir(data_dir, exts=['.npy'])
        scan_dir(lbl_dir, exts=['.txt', '.json'] if labels_fmt not in ('txt', 'json') else [f'.{labels_fmt}'])
        return (max_idx + 1) if max_idx >= 5000000 - 1 else 5000000  # 최소 5000000부터 시작

    start_index = args.start_index if getattr(args, 'start_index', None) is not None and args.start_index >= 0 else _find_next_index(out_dir, labels_dir, getattr(args, 'labels_format', 'json'))
    # start_index가 5000000보다 작으면 5000000으로 설정 (005로 시작하도록)
    if start_index < 5000000:
        start_index = 5000000
    file_idx = start_index
    for frame_idx, frame_bytes in enumerate(dataset):
        frame = dataset_pb2.Frame()
        # Ensure `bytes` type for ParseFromString (avoid bytearray)
        frame.ParseFromString(bytes(frame_bytes.numpy()))

        segment = frame.context.name
        timestamp = frame.timestamp_micros

        clouds = frame_points(frame, return_indices, allowed_lasers)
        if not clouds:
            print(f"[Frame {frame_idx}] empty (check laser/return filters)")
            continue

        if getattr(args, 'save_all_clouds', False):
            for _idx, cloud in enumerate(clouds):
                cloud = maybe_subsample(cloud, args.sample)
                stem = f"{file_idx:08d}"
                save_path = out_dir / f"{stem}.npy"
                np.save(save_path, cloud)
                # Optional label sidecar with matching stem
                if getattr(args, 'save_labels', False) and labels_dir is not None:
                    if getattr(args, 'labels_format', 'json') == 'txt':
                        save_labels_txt(frame, labels_dir / f"{stem}.txt")
                    else:
                        save_labels_json(frame, labels_dir / f"{stem}.json")
                print(f"[Frame {frame_idx}] saved {cloud.shape[0]} points -> {save_path}")
                file_idx += 1
        else:
            cloud = maybe_subsample(clouds[0], args.sample)
            stem = f"{file_idx:08d}"
            save_path = out_dir / f"{stem}.npy"
            np.save(save_path, cloud)
            if getattr(args, 'save_labels', False) and labels_dir is not None:
                if getattr(args, 'labels_format', 'json') == 'txt':
                    save_labels_txt(frame, labels_dir / f"{stem}.txt")
                else:
                    save_labels_json(frame, labels_dir / f"{stem}.json")
            print(f"[Frame {frame_idx}] saved {cloud.shape[0]} points -> {save_path}")
            file_idx += 1

def main() -> None:
    parser = argparse.ArgumentParser(description="Waymo TFRecord → NPY point clouds")
    parser.add_argument("--tfrecord", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--save-labels",
        action="store_true",
        help="Save labels alongside point clouds (use --labels-format)"
    )
    parser.add_argument(
        "--labels-format",
        choices=["json", "txt"],
        default="json",
        help="Labels file format when using --save-labels"
    )
    parser.add_argument(
        "--labels-out-dir",
        type=Path,
        default=None,
        help="Directory to store labels (default: <out-dir>/labels)"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Starting index for output filenames (default: auto-continue from existing files)"
    )
    parser.add_argument(
        "--lasers",
        type=str,
        default="TOP",
        help="Comma-separated laser names (TOP,FRONT,SIDE_LEFT,SIDE_RIGHT,REAR or ALL)"
    )
    parser.add_argument(
        "--returns",
        type=str,
        default="1",
        help="Comma-separated returns to use (1, 2, or '1,2'). Default: 1"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Random subsample size per cloud (0 keeps all points)"
    )
    parser.add_argument(
        "--save-all-clouds",
        action="store_true",
        help="Save all available clouds per frame (default: only cloud0)"
    )
    args = parser.parse_args()
    convert(args)

if __name__ == "__main__":
    main()

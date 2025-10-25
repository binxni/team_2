#!/usr/bin/env python3
"""
Convert 128-channel LiDAR frames to 64-channel using a range-image (pseudo-ring) approach.

This script does NOT require a per-point ring column. Instead, it:
  - Projects points to a range image with V_src (default 128) vertical bins via elevation angle
    and H horizontal bins via azimuth angle.
  - Reduces the vertical resolution to V_tgt (default 64) by pairwise row reduction
    (default: choose the nearer point per (2 rows × each azimuth column)).
  - Reconstructs a downsampled point cloud from the selected cells.

Frames with point count <= threshold are treated as native 64-channel and are copied/symlinked.

Notes
-----
This is an approximation without ring. It preserves a 64-like vertical stratification but
cannot replicate the exact vendor-specific beam angles or scan ordering.

Usage example:
    # Process entire dataset
    python OpenPCDet/tools/convert_lidar_range_image.py \
    --dataset_root OpenPCDet/data/custom_av \
    --output_dir OpenPCDet/data/custom_av/points_1014 \
    --train_split OpenPCDet/data/custom_av/ImageSets/train.txt \
    --val_split OpenPCDet/data/custom_av/ImageSets/val.txt \
    --threshold 200000 --h_bins 2048 --v_src 128 --v_tgt 64 \
    --use_64ch_layout
    
    # Process single frame
    python convert_lidar_range_image.py \
    --dataset_root /home/ailab/git/Team_4/Ai_challenge/OpenPCDet/data/custom_av \
    --output_dir /home/ailab/git/Team_2/Seokjae/OpenPCDet/dataset_analysis/range_image \
    --frame_id 00202455 \
    --v_src 128 --v_tgt 64 --use_64ch_layout
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Iterable, Tuple
import numpy as np


DEFAULT_POINT_THRESHOLD = 200_000


def load_split_ids(file_path: Path) -> list[str]:
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_points(point_dir: Path, sample_id: str) -> np.ndarray:
    pts = np.load(point_dir / f"{sample_id}.npy")
    if pts.ndim != 2 or pts.shape[1] < 4:
        raise ValueError(f"Expected (N,4+) array, got {pts.shape} for {sample_id}")
    return pts.astype(np.float32, copy=False)


def copy_or_symlink(src: Path, dst: Path, *, symlink: bool, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if overwrite:
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
        else:
            return
    if symlink:
        link_target = os.path.relpath(src.resolve(strict=True), start=dst.parent.resolve())
        os.symlink(link_target, dst)
    else:
        np.save(dst, np.load(src, mmap_mode="r"))


def spherical_angles(points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    xy = np.sqrt(x * x + y * y) + 1e-8
    # azimuth in [0, 2pi)
    az = np.arctan2(y, x)
    az = np.where(az < 0.0, az + 2.0 * math.pi, az)
    # elevation angle
    el = np.arctan2(z, xy)
    return az.astype(np.float32, copy=False), el.astype(np.float32, copy=False), r.astype(np.float32, copy=False)


def create_128channel_elevation_angles() -> np.ndarray:
    """
    Create 128-channel LiDAR elevation angles based on real specifications.
    
    Returns elevation angles in radians for 128 channels with non-uniform spacing:
    - 0.125° resolution: -6° to +2° (64 channels)
    - 0.5° resolution: +2° to +14°, -6° to -24° (56 channels total: 24 + 36)
    - 1° resolution: +14° to +15°, -24° to -25° (2 channels total)
    
    Total: 122 channels covering -25° to +15° (40° FOV) 
    """
    angles = []
    
    # High-density zone: -6° to +2° with 0.125° resolution (64 channels)
    start_deg = -6.0
    end_deg = 2.0
    resolution_deg = 0.125
    n_channels = int((end_deg - start_deg) / resolution_deg)  # 64 channels
    for i in range(n_channels):
        angle_deg = start_deg + i * resolution_deg
        angles.append(angle_deg)
    
    # Medium-density zone: +2° to +14° with 0.5° resolution (24 channels)
    for i in range(24):
        angle_deg = 2.0 + i * 0.5  # 2.0, 2.5, 3.0, ..., 13.5
        angles.append(angle_deg)
    
    # Medium-density zone: -6° to -24° with 0.5° resolution (36 channels)
    # Note: -6° overlaps with high-density zone, so start from -6.5°
    for i in range(36):
        angle_deg = -6.5 - i * 0.5  # -6.5, -7.0, -7.5, ..., -24.0
        angles.append(angle_deg)
    
    # Low-density zone: +14° to +15° with 1° resolution (1 channel)
    angles.append(15.0)
    
    # Low-density zone: -24° to -25° with 1° resolution (1 channel)
    angles.append(-25.0)
    
    # Total should be 64 + 24 + 36 + 1 + 1 = 126 channels
    # Fill remaining 2 channels with intermediate angles
    angles.append(14.5)  # between 14° and 15°
    angles.append(-24.5)  # between -24° and -25°
    
    # Sort angles and convert to radians
    angles.sort()
    angles = angles[:128]  # Ensure exactly 128 channels
    
    return np.array([math.radians(angle) for angle in angles], dtype=np.float32)


def create_64channel_elevation_angles() -> np.ndarray:
    """
    Create 64-channel LiDAR elevation angles based on real specifications.
    
    Returns elevation angles in radians for 64 channels with non-uniform spacing:
    - 0.167° resolution: -6° to +2° (48 channels)
    - 1° resolution: +2° to +3°, -14° to -6° (9 channels)
    - 2° resolution: +3° to +5° (1 channel)
    - 3° resolution: +5° to +11° (2 channels)
    - 4° resolution: +11° to +15° (1 channel)
    - 5° resolution: +19° to +14° (1 channels) 
    - 6° resolution: -25° to -18° (1 channels)
    
    Total: 63 channels covering -25° to +15° (40° FOV)
    """
    angles = []
    
    # High-density zone: -6° to +2° with 0.167° resolution (48 channels)
    start_deg = -6.0
    end_deg = 2.0
    resolution_deg = 0.167
    n_channels = int((end_deg - start_deg) / resolution_deg)  # 48 channels
    for i in range(n_channels):
        angle_deg = start_deg + i * resolution_deg
        angles.append(angle_deg)
    
    # 1° resolution zones (9 channels total)
    # +2° to +3° (1 channel)
    angles.append(3.0)
    
    # -14° to -6° with 1° resolution (8 channels: -14, -13, -12, -11, -10, -9, -8, -7)
    for angle_deg in range(-14, -6):  # -14 to -7 (8 channels)
        angles.append(float(angle_deg))
    
    # 2° resolution: +3° to +5° (1 channel)
    angles.append(5.0)
    
    # 3° resolution: +5° to +11° (2 channels)
    angles.append(8.0)
    angles.append(11.0)
    
    # 4° resolution: +11° to +15° (1 channel)
    angles.append(15.0)
    
    # 5° resolution: +19° to +14° (1 channel) - Note: this seems backwards in comment
    # Interpreting as one channel in this range
    angles.append(14.0)
    
    # 6° resolution: -25° to -18° (1 channel)
    angles.append(-18.0)
    
    # Add final boundary channel
    angles.append(-25.0)
    
    # Total should be: 48 + 1 + 8 + 1 + 2 + 1 + 1 + 1 + 1 = 64 channels
    # Fill if needed to reach exactly 64
    while len(angles) < 64:
        angles.append(-22.0)  # intermediate angle
        break
    
    # Sort angles and convert to radians
    angles.sort()
    angles = angles[:64]  # Ensure exactly 64 channels
    
    return np.array([math.radians(angle) for angle in angles], dtype=np.float32)


def channel_mapping_128_to_64(points: np.ndarray) -> np.ndarray:
    """
    Convert 128-channel LiDAR points to 64-channel by selecting channels that best match
    the 64-channel elevation angle distribution.
    
    This method:
    1. Converts points to spherical coordinates
    2. Finds the closest 128-channel elevation angle for each point
    3. Maps selected 128-channel angles to corresponding 64-channel angles
    4. Returns points that correspond to the 64-channel distribution
    """
    N = points.shape[0]
    if N == 0:
        return points[:0]
    
    # Get spherical coordinates
    az, el, r = spherical_angles(points[:, :3])
    
    # Get the predefined elevation angles for both sensors
    angles_128ch = create_128channel_elevation_angles()
    angles_64ch = create_64channel_elevation_angles()
    
    # Find closest 128-channel for each point
    el_expanded = el[:, np.newaxis]  # Shape: (N, 1)
    angles_128ch_expanded = angles_128ch[np.newaxis, :]  # Shape: (1, 128)
    
    # Calculate distances and find closest channel
    distances = np.abs(el_expanded - angles_128ch_expanded)
    closest_128ch_idx = np.argmin(distances, axis=1)  # Shape: (N,)
    
    # Create mapping from 128-channel indices to 64-channel indices
    # For each 64-channel angle, find the closest 128-channel angle
    channel_mapping = {}
    used_128ch_indices = set()
    
    for i, angle_64ch in enumerate(angles_64ch):
        # Find closest unused 128-channel angle
        distances_to_64ch = np.abs(angles_128ch - angle_64ch)
        sorted_indices = np.argsort(distances_to_64ch)
        
        for idx_128ch in sorted_indices:
            if idx_128ch not in used_128ch_indices:
                channel_mapping[idx_128ch] = i
                used_128ch_indices.add(idx_128ch)
                break
    
    # Select points that belong to mapped 128-channel indices
    valid_mask = np.isin(closest_128ch_idx, list(channel_mapping.keys()))
    
    if not valid_mask.any():
        return points[:0]
    
    return points[valid_mask]


def build_range_image(points: np.ndarray,
                      h_bins: int,
                      v_src: int,
                      v_tgt: int = 64,
                      el_range: Tuple[float, float] | None = None,
                      reduce: str = "nearest",
                      use_64ch_layout: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Build a V_src × H range image and return (best_idx, best_r, el_edges, indices).

    - best_idx: (V_src, H) int64 indices into the original point array (or -1 if empty)
    - best_r:   (V_src, H) float32 nearest range for the selected point (inf if empty)
    - el_edges: vertical bin edges used to quantize elevation
    - indices:  (vi, hi) arrays of per-point indices for debugging/inspection
    - use_64ch_layout: if True and v_src==64, use real 64-channel non-uniform layout
    """
    N = points.shape[0]
    az, el, r = spherical_angles(points[:, :3])

    # Horizontal bins (uniform in azimuth)
    hi = np.floor(az / (2.0 * math.pi) * h_bins).astype(np.int32)
    hi = np.clip(hi, 0, h_bins - 1)

    # Vertical bins - choose between uniform and 64-channel layout
    if use_64ch_layout and v_tgt == 64:
        # Use real 64-channel non-uniform elevation layout for target
        angles_64ch = create_64channel_elevation_angles()
        # Create bin edges by adding boundaries
        el_edges = np.zeros(65, dtype=np.float32)
        el_edges[0] = angles_64ch[0] - 0.001  # Lower boundary
        el_edges[-1] = angles_64ch[-1] + 0.001  # Upper boundary
        
        # Create intermediate boundaries as midpoints
        for i in range(1, 64):
            el_edges[i] = (angles_64ch[i-1] + angles_64ch[i]) / 2.0
        
        # Use 64 bins for range image construction
        v_bins_for_image = 64
        print(f"Using 64-channel non-uniform elevation layout for downsampling")
        print(f"Elevation range: {math.degrees(el_edges[0]):.1f}° to {math.degrees(el_edges[-1]):.1f}°")
    else:
        # Vertical range determination (percentiles to avoid extreme outliers)
        if el_range is None:
            el_min = float(np.percentile(el, 0.5))
            el_max = float(np.percentile(el, 99.5))
            if el_max <= el_min:  # degenerate, fall back to min/max
                el_min = float(el.min())
                el_max = float(el.max())
        else:
            el_min, el_max = float(el_range[0]), float(el_range[1])
        # Guard: ensure non-zero span
        if not math.isfinite(el_min) or not math.isfinite(el_max) or el_max - el_min < 1e-6:
            el_min = float(el.min())
            el_max = float(el.max())

        # Vertical bins (uniform in elevation angle, pseudo-ring)
        el_edges = np.linspace(el_min, el_max, v_src + 1, dtype=np.float32)
        v_bins_for_image = v_src
    
    vi = np.clip(np.digitize(el, el_edges) - 1, 0, v_bins_for_image - 1).astype(np.int32)

    # Allocate buffers
    best_idx = np.full((v_bins_for_image, h_bins), -1, dtype=np.int64)
    best_r = np.full((v_bins_for_image, h_bins), np.inf, dtype=np.float32)

    # Populate nearest point per cell - vectorized approach for better performance
    # Create flat indices for (v, h) pairs
    flat_indices = vi * h_bins + hi
    unique_flat_indices, inverse_indices = np.unique(flat_indices, return_inverse=True)
    
    # For each unique (v, h) cell, find the nearest point
    for i, flat_idx in enumerate(unique_flat_indices):
        v = flat_idx // h_bins
        h = flat_idx % h_bins
        
        # Find all points in this cell
        cell_mask = (inverse_indices == i)
        cell_points = np.where(cell_mask)[0]
        
        if len(cell_points) > 0:
            # Find nearest point in this cell
            cell_ranges = r[cell_points]
            nearest_idx = cell_points[np.argmin(cell_ranges)]
            
            best_idx[v, h] = nearest_idx
            best_r[v, h] = r[nearest_idx]

    return best_idx, best_r, el_edges, (vi, hi)


def reduce_vertical(best_idx: np.ndarray, best_r: np.ndarray, v_tgt: int, method: str = "pair_nearest") -> np.ndarray:
    """Reduce V_src × H to V_tgt × H and return the chosen original indices (unique, 1D)."""
    v_src, h_bins = best_idx.shape
    if v_src % v_tgt != 0:
        # Default to pairwise downsample if non-multiple; take floor pairing.
        factor = v_src // v_tgt
        if factor < 2:
            factor = 2
    else:
        factor = v_src // v_tgt

    chosen = []
    for t in range(v_tgt):
        v0 = t * factor
        v1 = min((t + 1) * factor, v_src)
        # Slice source rows [v0:v1)
        idx_block = best_idx[v0:v1, :]
        r_block = best_r[v0:v1, :]
        if method == "pair_nearest" or method == "nearest":
            # Choose nearest among the group for each column
            argmin = np.argmin(r_block, axis=0)
            ri = r_block[argmin, np.arange(h_bins)]
            ii = idx_block[argmin, np.arange(h_bins)]
            mask = np.isfinite(ri) & (ii >= 0)
            if mask.any():
                chosen.append(ii[mask])
        elif method == "first_nonempty":
            # Pick first non-empty cell by row order
            valid = (idx_block >= 0)
            first_row = np.argmax(valid, axis=0)  # 0 where all False
            # However, where all False, keep -1
            cols = np.where(valid.any(axis=0))[0]
            if cols.size > 0:
                ii = idx_block[first_row[cols], cols]
                chosen.append(ii)
        else:
            raise ValueError(f"Unknown vertical reduction method: {method}")

    if not chosen:
        return np.empty((0,), dtype=np.int64)
    merged = np.unique(np.concatenate(chosen).astype(np.int64, copy=False))
    merged = merged[merged >= 0]
    return merged


def process_frame(points: np.ndarray,
                  h_bins: int,
                  v_src: int,
                  v_tgt: int,
                  el_range: Tuple[float, float] | None,
                  reduce_method: str,
                  use_64ch_layout: bool = False) -> np.ndarray:
    # Use direct channel mapping method for 128->64 conversion
    if v_src == 128 and v_tgt == 64 and reduce_method == "channel_mapping":
        return channel_mapping_128_to_64(points)
    
    # Fall back to range image method for other cases
    best_idx, best_r, el_edges, (vi, hi) = build_range_image(
        points, h_bins, v_src, v_tgt, el_range=el_range, use_64ch_layout=use_64ch_layout
    )
    if reduce_method in ("row_even", "row_odd"):
        parity = 0 if reduce_method == "row_even" else 1
        mask = (vi % (v_src // v_tgt if v_src >= v_tgt and v_src % v_tgt == 0 else 2)) == parity
        # If factor is not clean, default to parity over 2
        chosen_idx = np.nonzero(mask)[0]
        if chosen_idx.size == 0:
            return points[:0]
        return points[chosen_idx]
    else:
        chosen_idx = reduce_vertical(best_idx, best_r, v_tgt, method=reduce_method)
        if chosen_idx.size == 0:
            return points[:0]
        return points[chosen_idx]


def convert_dataset(dataset_root: Path,
                    output_dir: Path,
                    train_split: Path | None,
                    val_split: Path | None,
                    frame_id: str | None,
                    threshold: int,
                    h_bins: int,
                    v_src: int,
                    v_tgt: int,
                    reduce_method: str,
                    overwrite: bool,
                    skip_existing: bool,
                    symlink_64: bool,
                    el_range: Tuple[float, float] | None,
                    max_frames: int | None,
                    use_64ch_layout: bool = False) -> None:
    point_dir = dataset_root / "points"
    label_dir = dataset_root / "labels"  # unused; placeholder for future ROI-aware variant
    if not point_dir.exists():
        raise FileNotFoundError(f"Missing points dir: {point_dir}")

    # Determine which frames to process
    if frame_id is not None:
        # Process only the specified frame
        all_ids = [frame_id]
        print(f"Processing single frame: {frame_id}")
    else:
        # Process frames from split files
        if train_split is None or val_split is None:
            raise ValueError("train_split and val_split are required when frame_id is not specified")
        train_ids = load_split_ids(train_split)
        val_ids = load_split_ids(val_split)
        all_ids = train_ids + val_ids
    
    total_frames = len(all_ids)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 처리 시작 시간과 초기 정보 출력
    start_time = time.time()
    print(f"=" * 60)
    print(f"LiDAR 데이터 변환 시작")
    print(f"=" * 60)
    print(f"데이터셋 경로: {dataset_root}")
    print(f"출력 경로: {output_dir}")
    print(f"전체 프레임 수: {len(all_ids)} (최대 처리: {total_frames})")
    if frame_id is not None:
        print(f"단일 프레임 처리: {frame_id}")
    else:
        print(f"Train 프레임: {len(train_ids)}, Val 프레임: {len(val_ids)}")
    print(f"포인트 수 임계값: {threshold:,}")
    print(f"변환 방법: {v_src} → {v_tgt} 채널 (reduce: {reduce_method})")
    if reduce_method == "channel_mapping" and v_src == 128 and v_tgt == 64:
        print(f"128→64 채널 직접 매핑 적용:")
        print(f"  - 128ch: 0.125°~1° 해상도, -25°~+15° FOV")
        print(f"  - 64ch: 0.167°~5° 해상도, -25°~+15° FOV")
        print(f"  - 각 포인트를 가장 유사한 64ch 각도로 매핑")
    elif use_64ch_layout and v_tgt == 64:
        print(f"64채널 비균일 각도 분포 적용:")
        print(f"  - 0.167° 해상도: -6° ~ +2° (고밀도 구간)")
        print(f"  - 0.5° 해상도: +2° ~ +14°, -8° ~ -24° (중밀도 구간)")  
        print(f"  - 1° 해상도: +14° ~ +15°, -24° ~ -25° (저밀도 구간)")
    print(f"Azimuth bins: {h_bins}")
    if el_range:
        print(f"Elevation 범위: {math.degrees(el_range[0]):.1f}° ~ {math.degrees(el_range[1]):.1f}°")
    print(f"-" * 60)

    processed = 0
    converted = 0
    copied = 0
    skipped = 0
    total_points_before = 0
    total_points_after = 0
    
    for i, sid in enumerate(all_ids):
        if max_frames is not None and processed >= max_frames:
            break

        src = point_dir / f"{sid}.npy"
        dst = output_dir / f"{sid}.npy"
        
        # 진행률 출력 (매 10%마다)
        progress = (i + 1) / total_frames * 100
        if (i + 1) % max(1, total_frames // 10) == 0 or i == 0:
            print(f"진행률: {progress:.1f}% ({i + 1}/{total_frames}) - 현재 처리: {sid}")
        
        if skip_existing and dst.exists():
            processed += 1
            skipped += 1
            continue

        pts = load_points(point_dir, sid)
        original_points = pts.shape[0]
        total_points_before += original_points
        
        if pts.shape[0] <= threshold:
            copy_or_symlink(src, dst, symlink=symlink_64, overwrite=overwrite)
            copied += 1
            total_points_after += original_points
        else:
            # 128-like frame → range-image reduction
            down = process_frame(pts, h_bins=h_bins, v_src=v_src, v_tgt=v_tgt,
                                 el_range=el_range, reduce_method=reduce_method,
                                 use_64ch_layout=use_64ch_layout)
            np.save(dst, down.astype(np.float32, copy=False))
            converted += 1
            total_points_after += down.shape[0]
            
        processed += 1

    # 처리 완료 및 결과 출력
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n" + "=" * 60)
    print(f"LiDAR 데이터 변환 완료!")
    print(f"=" * 60)
    print(f"처리 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.1f}분)")
    print(f"처리 속도: {processed/elapsed_time:.1f} 프레임/초")
    print(f"")
    print(f"처리 결과:")
    print(f"  - 총 처리된 프레임: {processed:,}")
    print(f"  - 변환된 프레임 (128→64): {converted:,}")
    print(f"  - 복사된 프레임 (≤{threshold:,} 포인트): {copied:,}")
    if skipped > 0:
        print(f"  - 건너뛴 프레임 (기존재): {skipped:,}")
    print(f"")
    print(f"포인트 수 통계:")
    print(f"  - 처리 전 총 포인트: {total_points_before:,}")
    print(f"  - 처리 후 총 포인트: {total_points_after:,}")
    if total_points_before > 0:
        reduction_ratio = (total_points_before - total_points_after) / total_points_before * 100
        print(f"  - 포인트 감소율: {reduction_ratio:.1f}%")
        print(f"  - 평균 압축비: {total_points_before/total_points_after:.2f}:1")
    print(f"")
    if converted > 0:
        avg_before = total_points_before / processed if processed > 0 else 0
        avg_after = total_points_after / processed if processed > 0 else 0
        print(f"프레임당 평균 포인트:")
        print(f"  - 변환 전: {avg_before:,.0f}")
        print(f"  - 변환 후: {avg_after:,.0f}")
    print(f"=" * 60)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert 128→64 via range image (pseudo-ring)")
    ap.add_argument("--dataset_root", type=Path, required=True,
                    help="Dataset root containing points/ and labels/")
    ap.add_argument("--output_dir", type=Path, required=True,
                    help="Output directory for converted .npy points")
    ap.add_argument("--train_split", type=Path, required=False,
                    help="Path to train split txt")
    ap.add_argument("--val_split", type=Path, required=False,
                    help="Path to val split txt")
    ap.add_argument("--frame_id", type=str, default=None,
                    help="Specific frame ID to convert (if specified, splits are ignored)")
    ap.add_argument("--threshold", type=int, default=DEFAULT_POINT_THRESHOLD,
                    help="Point-count threshold: <= threshold treated as 64, > as 128")
    ap.add_argument("--h_bins", type=int, default=1800, help="Azimuth bins (columns)")
    ap.add_argument("--v_src", type=int, default=128, help="Source vertical bins (rows)")
    ap.add_argument("--v_tgt", type=int, default=64, help="Target vertical bins (rows)")
    ap.add_argument("--reduce", type=str, default="channel_mapping", choices=["channel_mapping", "pair_nearest", "nearest", "first_nonempty", "row_even", "row_odd"],
                    help="Row reduction method")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--skip_existing", action="store_true", help="Skip frames with existing outputs")
    ap.add_argument("--symlink_64", action="store_true", help="Symlink native-64 frames instead of copying them")
    ap.add_argument("--el_min_deg", type=float, default=-25, help="Optional fixed elevation min (deg)")
    ap.add_argument("--el_max_deg", type=float, default=15, help="Optional fixed elevation max (deg)")
    ap.add_argument("--max_frames", type=int, default=None, help="Limit frames for this run")
    ap.add_argument("--use_64ch_layout", action="store_true", 
                    help="Always use real 64-channel non-uniform elevation layout for downsampling")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    el_range = None
    if args.el_min_deg is not None and args.el_max_deg is not None:
        el_range = (math.radians(args.el_min_deg), math.radians(args.el_max_deg))

    convert_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        frame_id=args.frame_id,
        threshold=args.threshold,
        h_bins=args.h_bins,
        v_src=args.v_src,
        v_tgt=args.v_tgt,
        reduce_method=args.reduce,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        symlink_64=args.symlink_64,
        el_range=el_range,
        max_frames=args.max_frames,
        use_64ch_layout=args.use_64ch_layout,
    )


if __name__ == "__main__":
    main()
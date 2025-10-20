#!/usr/bin/env python3
"""
Prepare custom_av_noise2 dataset:
1. Copy scene 1,2,3 (non-003) data as-is to train set
2. Copy scene 4 (003) data to val set
3. Generate noised versions for augmentation
4. Copy test data (10) as-is
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


def get_scene_info(file_id: str) -> Tuple[int, str]:
    """Extract scene number from file ID (e.g., '00000001' -> scene 1)."""
    prefix = file_id[:3]
    scene_map = {
        "000": 1,
        "001": 2,
        "002": 3,
        "003": 4,
        "010": 0,  # Test set
    }
    scene = scene_map.get(prefix, -1)
    return scene, prefix


def get_files_by_scene(points_dir: Path) -> dict:
    """Organize point files by scene."""
    scenes = {1: [], 2: [], 3: [], 4: [], 0: []}
    
    for npy_file in sorted(points_dir.glob("*.npy")):
        file_id = npy_file.stem
        scene, _ = get_scene_info(file_id)
        if scene in scenes:
            scenes[scene].append(file_id)
    
    return scenes


def copy_original_data(
    src_points_dir: Path,
    dst_points_dir: Path,
    dst_labels_dir: Path,
    src_labels_dir: Path,
    file_ids: List[str],
    output_file: Path,
) -> int:
    """
    Copy original data files.
    Returns the next available output index.
    """
    with open(output_file, 'w') as f:
        for file_id in file_ids:
            src_file = src_points_dir / f"{file_id}.npy"
            dst_file = dst_points_dir / f"{file_id}.npy"
            
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
                
                # Also copy label if exists
                src_label = src_labels_dir / f"{file_id}.txt"
                if src_label.exists():
                    dst_label = dst_labels_dir / f"{file_id}.txt"
                    shutil.copy2(src_label, dst_label)
                
                f.write(f"{file_id}\n")
                print(f"Copied: {file_id}")
    
    return len(file_ids)


def generate_noised_data(
    src_points_dir: Path,
    dst_points_dir: Path,
    dst_labels_dir: Path,
    src_labels_dir: Path,
    src_scenes: List[int],
    sample_count: int,
    start_index: int,
    output_file: Path,
    base_sigma: float = 0.02,
    sigma_gain: float = 0.15,
    dropout_max: float = 0.3,
) -> int:
    """Generate noised versions of data using range_noise.py."""
    
    scene_prefixes = {1: "000", 2: "001", 3: "002", 4: "003"}
    
    noised_indices = []
    current_index = start_index
    
    for scene in src_scenes:
        prefix = scene_prefixes[scene]
        scene_files = sorted([f for f in src_points_dir.glob("*.npy") 
                             if f.stem.startswith(prefix)])
        
        if not scene_files:
            print(f"No files found for scene {scene} (prefix {prefix})")
            continue
        
        # Sample one file per scene for noise augmentation
        import random
        random.seed(42)
        sampled_file = random.choice(scene_files)
        file_id = sampled_file.stem
        
        print(f"\nGenerating noise for scene {scene}: {file_id}")
        
        # Read original point cloud
        points = np.load(sampled_file)
        
        # Apply range noise
        noised_points = apply_range_noise(
            points,
            base_sigma=base_sigma,
            sigma_gain=sigma_gain,
            dropout_max=dropout_max,
        )
        
        # Save noised version with new index
        noised_id = f"{current_index:08d}"
        noised_file = dst_points_dir / f"{noised_id}.npy"
        np.save(noised_file, noised_points)
        
        # Copy and save label with new index
        src_label = src_labels_dir / f"{file_id}.txt"
        if src_label.exists():
            dst_label = dst_labels_dir / f"{noised_id}.txt"
            shutil.copy2(src_label, dst_label)
        
        noised_indices.append(noised_id)
        print(f"Saved noised data: {noised_id}")
        
        current_index += 1
    
    # Append to output file
    with open(output_file, 'a') as f:
        for noised_id in noised_indices:
            f.write(f"{noised_id}\n")
    
    return current_index


def apply_range_noise(
    points: np.ndarray,
    base_sigma: float = 0.02,
    sigma_gain: float = 0.15,
    reference_distance: float = 120.0,
    min_range: float = 1.0,
    dropout_max: float = 0.3,
    dropout_start: float = 30.0,
    dropout_end: float = 100.0,
) -> np.ndarray:
    """Apply range-dependent noise to point cloud."""
    points = points.copy()
    
    # Extract x, y, z coordinates
    xyz = points[:, :3]
    
    # Calculate ranges
    ranges = np.linalg.norm(xyz, axis=1)
    
    # Calculate sigma based on range
    sigma = base_sigma + sigma_gain * (ranges / reference_distance)
    
    # Add Gaussian noise to ranges
    noise = np.random.normal(0, sigma)
    noised_ranges = np.maximum(ranges + noise, min_range)
    
    # Apply dropout based on range
    dropout_prob = np.zeros_like(ranges)
    mask = (ranges >= dropout_start) & (ranges <= dropout_end)
    dropout_prob[mask] = dropout_max * (ranges[mask] - dropout_start) / (dropout_end - dropout_start)
    
    dropout_mask = np.random.random(len(ranges)) < dropout_prob
    
    # Update point coordinates
    scale = noised_ranges / (ranges + 1e-8)
    points[:, :3] = xyz * scale[:, np.newaxis]
    
    # Remove dropped points
    points = points[~dropout_mask]
    
    return points


def main():
    parser = argparse.ArgumentParser(description="Prepare custom_av_noise2 dataset")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/ailab/git/Team_2/Subin/OpenPCDet/data"),
        help="Root data directory",
    )
    parser.add_argument(
        "--src-dataset",
        type=str,
        default="custom_av",
        help="Source dataset name",
    )
    parser.add_argument(
        "--dst-dataset",
        type=str,
        default="custom_av_noise2",
        help="Destination dataset name",
    )
    
    args = parser.parse_args()
    
    src_root = args.data_root / args.src_dataset
    dst_root = args.data_root / args.dst_dataset
    
    src_points_dir = src_root / "points"
    src_labels_dir = src_root / "labels"
    
    dst_points_dir = dst_root / "points"
    dst_labels_dir = dst_root / "labels"
    dst_imagesets = dst_root / "ImageSets"
    
    # Ensure directories exist
    dst_points_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)
    dst_imagesets.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Step 1: Organize source data by scene")
    print("=" * 80)
    scenes = get_files_by_scene(src_points_dir)
    
    for scene_num, files in scenes.items():
        print(f"Scene {scene_num}: {len(files)} files")
    
    print("\n" + "=" * 80)
    print("Step 2: Copy original scene 1,2,3 (train set)")
    print("=" * 80)
    
    train_file = dst_imagesets / "train.txt"
    train_files = scenes[1] + scenes[2] + scenes[3]
    train_count = copy_original_data(
        src_points_dir,
        dst_points_dir,
        dst_labels_dir,
        src_labels_dir,
        train_files,
        train_file,
    )
    print(f"Total train files: {train_count}")
    
    print("\n" + "=" * 80)
    print("Step 3: Copy original scene 4 (val set)")
    print("=" * 80)
    
    val_file = dst_imagesets / "val.txt"
    val_files = scenes[4]
    copy_original_data(
        src_points_dir,
        dst_points_dir,
        dst_labels_dir,
        src_labels_dir,
        val_files,
        val_file,
    )
    print(f"Total val files: {len(val_files)}")
    
    print("\n" + "=" * 80)
    print("Step 4: Generate noised data for train (1 per scene)")
    print("=" * 80)
    
    next_train_index = train_count
    train_count = generate_noised_data(
        src_points_dir,
        dst_points_dir,
        dst_labels_dir,
        src_labels_dir,
        src_scenes=[1, 2, 3],
        sample_count=1,
        start_index=next_train_index,
        output_file=train_file,
    )
    print(f"Total after noise augmentation: {train_count}")
    
    print("\n" + "=" * 80)
    print("Step 5: Generate noised data for val (all scene 4)")
    print("=" * 80)
    
    val_count = len(val_files)
    next_val_index = val_count
    generate_noised_data(
        src_points_dir,
        dst_points_dir,
        dst_labels_dir,
        src_labels_dir,
        src_scenes=[4],
        sample_count=len(val_files),
        start_index=next_val_index,
        output_file=val_file,
    )
    
    print("\n" + "=" * 80)
    print("Step 6: Copy test data (scene 10)")
    print("=" * 80)
    
    test_file = dst_imagesets / "test.txt"
    test_files = scenes[0]  # Scene 0 is test (010xxxxx)
    copy_original_data(
        src_points_dir,
        dst_points_dir,
        dst_labels_dir,
        src_labels_dir,
        test_files,
        test_file,
    )
    print(f"Total test files: {len(test_files)}")
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Train samples: {train_count}")
    print(f"Val samples: {val_count + len(val_files)}")  # Original + noised
    print(f"Test samples: {len(test_files)}")
    
    print("\n✓ Dataset preparation complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate noised samples to achieve 30% noise ratio in dataset.

Train: 7,360 noised samples (30% of 24,534 total)
Val:   685 noised samples (30% of 2,285 total)
Test:  0 noised samples (100% original)
"""

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np


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
    points = points.copy().astype(np.float32)
    
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


def get_scene_files(points_dir: Path, prefix: str) -> List[str]:
    """Get all files from a specific scene (by prefix)."""
    files = [f.stem for f in points_dir.glob(f"{prefix}*.npy")]
    return sorted(files)


def generate_noised_samples(
    src_points_dir: Path,
    src_labels_dir: Path,
    dst_points_dir: Path,
    dst_labels_dir: Path,
    original_file_ids: List[str],
    num_target: int,
    start_index: int,
    output_file: Path,
    append: bool = False,
) -> Tuple[int, List[str]]:
    """
    Generate noised samples by randomly sampling from original files.
    
    Returns:
        (next_available_index, list_of_generated_ids)
    """
    
    mode = 'a' if append else 'w'
    generated_ids = []
    current_index = start_index
    
    # Randomly sample files with replacement for noise generation
    random.seed(42)
    sampled_files = random.choices(original_file_ids, k=num_target)
    
    print(f"\nGenerating {num_target} noised samples...")
    print(f"Starting from index: {start_index}")
    
    for i, original_file_id in enumerate(sampled_files):
        # Load original point cloud
        src_file = src_points_dir / f"{original_file_id}.npy"
        if not src_file.exists():
            print(f"Warning: {src_file} not found, skipping...")
            continue
        
        points = np.load(src_file)
        
        # Apply noise
        noised_points = apply_range_noise(points)
        
        # Save noised version with new index
        noised_id = f"{current_index:08d}"
        noised_file = dst_points_dir / f"{noised_id}.npy"
        np.save(noised_file, noised_points)
        
        # Copy label if exists
        src_label = src_labels_dir / f"{original_file_id}.txt"
        if src_label.exists():
            dst_label = dst_labels_dir / f"{noised_id}.txt"
            dst_label.write_text(src_label.read_text())
        
        generated_ids.append(noised_id)
        current_index += 1
        
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_target} samples... (latest: {noised_id})")
    
    # Append to output file
    with open(output_file, mode) as f:
        for noised_id in generated_ids:
            f.write(f"{noised_id}\n")
    
    print(f"✓ Generated {len(generated_ids)} noised samples")
    return current_index, generated_ids


def main():
    parser = argparse.ArgumentParser(
        description="Generate noised samples for 30% noise augmentation"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/ailab/git/Team_2/Subin/OpenPCDet/data"),
    )
    parser.add_argument(
        "--src-dataset",
        type=str,
        default="custom_av",
    )
    parser.add_argument(
        "--dst-dataset",
        type=str,
        default="custom_av_noise2",
    )
    
    args = parser.parse_args()
    
    src_root = args.data_root / args.src_dataset
    dst_root = args.data_root / args.dst_dataset
    
    src_points_dir = src_root / "points"
    src_labels_dir = src_root / "labels"
    dst_points_dir = dst_root / "points"
    dst_labels_dir = dst_root / "labels"
    dst_imagesets = dst_root / "ImageSets"
    
    print("=" * 90)
    print("GENERATING NOISE AUGMENTATION FOR 30% NOISE RATIO")
    print("=" * 90)
    
    # ========================================================================
    # STEP 1: Prepare train data
    # ========================================================================
    print("\n" + "=" * 90)
    print("STEP 1: GENERATE TRAIN NOISED SAMPLES (7,357 more needed)")
    print("=" * 90)
    
    train_file = dst_imagesets / "train.txt"
    
    # Get all original train files (Scenes 1, 2, 3)
    scene1_files = get_scene_files(src_points_dir, "000")
    scene2_files = get_scene_files(src_points_dir, "001")
    scene3_files = get_scene_files(src_points_dir, "002")
    all_train_files = scene1_files + scene2_files + scene3_files
    
    print(f"Total original train files available: {len(all_train_files)}")
    print(f"  Scene 1: {len(scene1_files)}")
    print(f"  Scene 2: {len(scene2_files)}")
    print(f"  Scene 3: {len(scene3_files)}")
    
    # We already have 3 noised samples, need 7,357 more
    train_noised_target = 7360
    train_already_noised = 3
    train_still_needed = train_noised_target - train_already_noised
    
    # Start from index 00017177 (after current data)
    train_start_index = 17177
    
    current_index, train_generated = generate_noised_samples(
        src_points_dir,
        src_labels_dir,
        dst_points_dir,
        dst_labels_dir,
        all_train_files,
        train_still_needed,
        train_start_index,
        train_file,
        append=True,  # Append to existing file
    )
    
    # ========================================================================
    # STEP 2: Prepare validation data
    # ========================================================================
    print("\n" + "=" * 90)
    print("STEP 2: GENERATE VALIDATION NOISED SAMPLES (684 more needed)")
    print("=" * 90)
    
    val_file = dst_imagesets / "val.txt"
    
    # Get all original val files (Scene 4)
    scene4_files = get_scene_files(src_points_dir, "003")
    
    print(f"Total original val files available: {len(scene4_files)}")
    print(f"  Scene 4: {len(scene4_files)}")
    
    # We already have 1 noised sample, need 684 more
    val_noised_target = 685
    val_already_noised = 1
    val_still_needed = val_noised_target - val_already_noised
    
    current_index, val_generated = generate_noised_samples(
        src_points_dir,
        src_labels_dir,
        dst_points_dir,
        dst_labels_dir,
        scene4_files,
        val_still_needed,
        current_index,
        val_file,
        append=True,  # Append to existing file
    )
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    
    # Verify results
    train_ids = train_file.read_text().strip().split('\n')
    val_ids = val_file.read_text().strip().split('\n')
    
    train_noised_count = sum(1 for id in train_ids if int(id) >= 17177)
    val_noised_count = sum(1 for id in val_ids if int(id) >= 24534)  # After train noised
    
    train_total = len(train_ids)
    val_total = len(val_ids)
    
    print(f"\nTRAIN SET:")
    print(f"  Total samples: {train_total}")
    print(f"  Original: {train_total - train_noised_count}")
    print(f"  Noised: {train_noised_count}")
    print(f"  Noise ratio: {train_noised_count / train_total * 100:.2f}%")
    
    print(f"\nVALIDATION SET:")
    print(f"  Total samples: {val_total}")
    print(f"  Original: {val_total - val_noised_count}")
    print(f"  Noised: {val_noised_count}")
    print(f"  Noise ratio: {val_noised_count / val_total * 100:.2f}%")
    
    print("\n✓ Noise augmentation complete!")
    print("=" * 90)


if __name__ == "__main__":
    main()

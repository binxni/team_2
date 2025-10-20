#!/usr/bin/env python3
"""
Generate noised point clouds to achieve 30% noise ratio - Parallel version
병렬 처리로 빠르게 생성
"""

import os
import sys
import shutil
import random
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 노이즈 적용 함수
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
    
    xyz = points[:, :3]
    ranges = np.linalg.norm(xyz, axis=1)
    sigma = base_sigma + sigma_gain * (ranges / reference_distance)
    
    noise = np.random.normal(0, sigma)
    noised_ranges = np.maximum(ranges + noise, min_range)
    
    dropout_prob = np.zeros_like(ranges)
    mask = (ranges >= dropout_start) & (ranges <= dropout_end)
    dropout_prob[mask] = dropout_max * (ranges[mask] - dropout_start) / (dropout_end - dropout_start)
    
    dropout_mask = np.random.random(len(ranges)) < dropout_prob
    
    scale = noised_ranges / (ranges + 1e-8)
    points[:, :3] = xyz * scale[:, np.newaxis]
    points = points[~dropout_mask]
    
    return points


def process_single_file(args):
    """Process a single file - for parallel execution"""
    src_points_dir, src_labels_dir, dst_points_dir, dst_labels_dir, original_id, noised_id = args
    
    try:
        src_file = src_points_dir / f"{original_id}.npy"
        if not src_file.exists():
            return None
        
        points = np.load(src_file)
        noised_points = apply_range_noise(points)
        
        dst_file = dst_points_dir / f"{noised_id}.npy"
        np.save(dst_file, noised_points)
        
        src_label = src_labels_dir / f"{original_id}.txt"
        if src_label.exists():
            dst_label = dst_labels_dir / f"{noised_id}.txt"
            shutil.copy2(src_label, dst_label)
        
        return noised_id
    except Exception as e:
        print(f"Error processing {original_id}: {e}")
        return None


def generate_noised_dataset(
    src_root: Path,
    dst_root: Path,
    noise_ratio: float = 0.30,
    num_workers: int = 8,
):
    """Generate noised dataset with specified noise ratio using parallel processing"""
    
    src_points = src_root / "points"
    src_labels = src_root / "labels"
    dst_points = dst_root / "points"
    dst_labels = dst_root / "labels"
    dst_imagesets = dst_root / "ImageSets"
    
    print("=" * 90)
    print("GENERATING NOISED DATASET WITH 30% NOISE RATIO (PARALLEL)")
    print("=" * 90)
    
    # ========================================================================
    # TRAIN SET
    # ========================================================================
    print("\n[STEP 1] Processing TRAIN SET")
    print("-" * 90)
    
    train_file = dst_imagesets / "train.txt"
    
    original_train_ids = []
    with open(train_file, 'r') as f:
        for line in f:
            file_id = line.strip()
            if file_id.startswith(('000', '001', '002')):
                original_train_ids.append(file_id)
    
    print(f"Original train samples: {len(original_train_ids)}")
    
    total_train_needed = len(original_train_ids)
    noised_train_needed = int(total_train_needed * noise_ratio / (1 - noise_ratio))
    
    print(f"Target noise samples: {noised_train_needed}")
    print(f"Final total train samples: {total_train_needed + noised_train_needed}")
    print(f"Noise ratio: {noised_train_needed / (total_train_needed + noised_train_needed) * 100:.2f}%")
    
    # 기존 노이즈 샘플 삭제
    existing_noised = [f for f in dst_points.glob("000*.npy") 
                       if int(f.stem) >= 20000]
    for f in existing_noised:
        f.unlink()
    print(f"Cleaned up {len(existing_noised)} old noised files")
    
    # 병렬 처리를 위한 작업 리스트 준비
    print(f"\nGenerating {noised_train_needed} noised train samples...")
    
    random.seed(42)
    np.random.seed(42)
    
    tasks = []
    noised_train_ids = []
    next_index = 20000
    
    for i in range(noised_train_needed):
        original_id = random.choice(original_train_ids)
        noised_id = f"{next_index:08d}"
        tasks.append((src_points, src_labels, dst_points, dst_labels, original_id, noised_id))
        noised_train_ids.append(noised_id)
        next_index += 1
    
    # 병렬 처리 실행
    completed = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_file, task): task for task in tasks}
        
        with tqdm(total=len(tasks), desc="Processing train samples") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        completed += 1
                except Exception as e:
                    print(f"Task failed: {e}")
                pbar.update(1)
    
    print(f"Successfully generated {completed}/{noised_train_needed} noised train samples")
    
    # train.txt 재작성
    print(f"Updating train.txt...")
    with open(train_file, 'w') as f:
        for file_id in original_train_ids:
            f.write(f"{file_id}\n")
        for file_id in noised_train_ids:
            f.write(f"{file_id}\n")
    
    print(f"✓ Train set completed: {len(original_train_ids)} original + {len(noised_train_ids)} noised")
    
    # ========================================================================
    # VALIDATION SET
    # ========================================================================
    print("\n[STEP 2] Processing VALIDATION SET")
    print("-" * 90)
    
    val_file = dst_imagesets / "val.txt"
    
    original_val_ids = []
    with open(val_file, 'r') as f:
        for line in f:
            file_id = line.strip()
            if file_id.startswith('003'):
                original_val_ids.append(file_id)
    
    print(f"Original val samples: {len(original_val_ids)}")
    
    noised_val_needed = int(len(original_val_ids) * noise_ratio / (1 - noise_ratio))
    
    print(f"Target noise samples: {noised_val_needed}")
    print(f"Final total val samples: {len(original_val_ids) + noised_val_needed}")
    print(f"Noise ratio: {noised_val_needed / (len(original_val_ids) + noised_val_needed) * 100:.2f}%")
    
    # 새 노이즈 샘플 생성
    print(f"\nGenerating {noised_val_needed} noised val samples...")
    
    tasks_val = []
    noised_val_ids = []
    
    for i in range(noised_val_needed):
        original_id = random.choice(original_val_ids)
        noised_id = f"{next_index:08d}"
        tasks_val.append((src_points, src_labels, dst_points, dst_labels, original_id, noised_id))
        noised_val_ids.append(noised_id)
        next_index += 1
    
    # 병렬 처리 실행
    completed_val = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_file, task): task for task in tasks_val}
        
        with tqdm(total=len(tasks_val), desc="Processing val samples") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        completed_val += 1
                except Exception as e:
                    print(f"Task failed: {e}")
                pbar.update(1)
    
    print(f"Successfully generated {completed_val}/{noised_val_needed} noised val samples")
    
    # val.txt 재작성
    print(f"Updating val.txt...")
    with open(val_file, 'w') as f:
        for file_id in original_val_ids:
            f.write(f"{file_id}\n")
        for file_id in noised_val_ids:
            f.write(f"{file_id}\n")
    
    print(f"✓ Val set completed: {len(original_val_ids)} original + {len(noised_val_ids)} noised")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 90)
    print("DATASET GENERATION COMPLETE")
    print("=" * 90)
    
    total_train = len(original_train_ids) + len(noised_train_ids)
    total_val = len(original_val_ids) + len(noised_val_ids)
    test_count = len(list((dst_imagesets / "test.txt").read_text().strip().split('\n')))
    
    print(f"\n[TRAIN SET]")
    print(f"  Original: {len(original_train_ids)}")
    print(f"  Noised:   {len(noised_train_ids)}")
    print(f"  Total:    {total_train}")
    print(f"  Noise ratio: {len(noised_train_ids) / total_train * 100:.2f}%")
    
    print(f"\n[VALIDATION SET]")
    print(f"  Original: {len(original_val_ids)}")
    print(f"  Noised:   {len(noised_val_ids)}")
    print(f"  Total:    {total_val}")
    print(f"  Noise ratio: {len(noised_val_ids) / total_val * 100:.2f}%")
    
    print(f"\n[TEST SET]")
    print(f"  Total:    {test_count} (all original)")
    print(f"  Noise ratio: 0.00%")
    
    print(f"\n[OVERALL]")
    print(f"  Total samples: {total_train + total_val + test_count}")
    print(f"  Total noised: {len(noised_train_ids) + len(noised_val_ids)}")
    print(f"  Overall noise ratio (train+val): {(len(noised_train_ids) + len(noised_val_ids)) / (total_train + total_val) * 100:.2f}%")


if __name__ == "__main__":
    src_root = Path("/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av")
    dst_root = Path("/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av_noise2")
    
    # 사용 가능한 CPU 코어 수에 따라 워커 수 결정
    num_workers = min(16, os.cpu_count() or 8)
    print(f"Using {num_workers} workers for parallel processing")
    
    generate_noised_dataset(src_root, dst_root, noise_ratio=0.30, num_workers=num_workers)
    print("\n✓ All done!")

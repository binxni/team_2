#!/usr/bin/env python3
"""
Generate noised point clouds to achieve 30% noise ratio in the dataset.
30% 노이즈 비율을 달성하기 위해 필요한 데이터 생성
"""

import os
import sys
import shutil
import random
import numpy as np
from pathlib import Path
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


def generate_noised_dataset(
    src_root: Path,
    dst_root: Path,
    noise_ratio: float = 0.30,
):
    """Generate noised dataset with specified noise ratio"""
    
    src_points = src_root / "points"
    src_labels = src_root / "labels"
    dst_points = dst_root / "points"
    dst_labels = dst_root / "labels"
    dst_imagesets = dst_root / "ImageSets"
    
    print("=" * 90)
    print("GENERATING NOISED DATASET WITH 30% NOISE RATIO")
    print("=" * 90)
    
    # ========================================================================
    # TRAIN SET
    # ========================================================================
    print("\n[STEP 1] Processing TRAIN SET")
    print("-" * 90)
    
    train_file = dst_imagesets / "train.txt"
    
    # 기존 train.txt 읽기 (Scene 1,2,3만)
    original_train_ids = []
    with open(train_file, 'r') as f:
        for line in f:
            file_id = line.strip()
            if file_id.startswith(('000', '001', '002')):
                original_train_ids.append(file_id)
    
    print(f"Original train samples: {len(original_train_ids)}")
    
    # 30% 노이즈 비율 계산
    total_train_needed = len(original_train_ids)
    noised_train_needed = int(total_train_needed * noise_ratio / (1 - noise_ratio))
    
    print(f"Target noise samples: {noised_train_needed}")
    print(f"Final total train samples: {total_train_needed + noised_train_needed}")
    print(f"Noise ratio: {noised_train_needed / (total_train_needed + noised_train_needed) * 100:.2f}%")
    
    # 기존 노이즈 샘플 삭제
    existing_noised = [f for f in (dst_points).glob("*.npy") 
                       if f.stem in ['00017174', '00017175', '00017176']]
    for f in existing_noised:
        f.unlink()
        print(f"Removed existing noised: {f.stem}")
    
    # 새 노이즈 샘플 생성
    noised_train_ids = []
    next_index = 20000  # 새 노이즈 샘플의 시작 인덱스
    
    print(f"\nGenerating {noised_train_needed} noised train samples...")
    
    random.seed(42)
    np.random.seed(42)
    
    for i in tqdm(range(noised_train_needed)):
        # 원본 파일 중 랜덤 선택
        original_id = random.choice(original_train_ids)
        src_file = src_points / f"{original_id}.npy"
        
        if not src_file.exists():
            print(f"Warning: {src_file} not found")
            continue
        
        # 노이즈 적용
        points = np.load(src_file)
        noised_points = apply_range_noise(points)
        
        # 새 인덱스로 저장
        noised_id = f"{next_index:08d}"
        dst_file = dst_points / f"{noised_id}.npy"
        np.save(dst_file, noised_points)
        
        # 라벨도 복사
        src_label = src_labels / f"{original_id}.txt"
        if src_label.exists():
            dst_label = dst_labels / f"{noised_id}.txt"
            shutil.copy2(src_label, dst_label)
        
        noised_train_ids.append(noised_id)
        next_index += 1
    
    # train.txt 재작성
    print(f"\nUpdating train.txt...")
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
    
    # 기존 val.txt 읽기 (Scene 4만)
    original_val_ids = []
    with open(val_file, 'r') as f:
        for line in f:
            file_id = line.strip()
            if file_id.startswith('003'):
                original_val_ids.append(file_id)
    
    print(f"Original val samples: {len(original_val_ids)}")
    
    # 30% 노이즈 비율 계산
    noised_val_needed = int(len(original_val_ids) * noise_ratio / (1 - noise_ratio))
    
    print(f"Target noise samples: {noised_val_needed}")
    print(f"Final total val samples: {len(original_val_ids) + noised_val_needed}")
    print(f"Noise ratio: {noised_val_needed / (len(original_val_ids) + noised_val_needed) * 100:.2f}%")
    
    # 기존 노이즈 샘플 삭제
    existing_noised_val = [f for f in (dst_points).glob("*.npy") 
                           if f.stem == '00001600']
    for f in existing_noised_val:
        f.unlink()
        print(f"Removed existing noised: {f.stem}")
    
    # 새 노이즈 샘플 생성
    noised_val_ids = []
    
    print(f"\nGenerating {noised_val_needed} noised val samples...")
    
    for i in tqdm(range(noised_val_needed)):
        # 원본 파일 중 랜덤 선택
        original_id = random.choice(original_val_ids)
        src_file = src_points / f"{original_id}.npy"
        
        if not src_file.exists():
            print(f"Warning: {src_file} not found")
            continue
        
        # 노이즈 적용
        points = np.load(src_file)
        noised_points = apply_range_noise(points)
        
        # 새 인덱스로 저장
        noised_id = f"{next_index:08d}"
        dst_file = dst_points / f"{noised_id}.npy"
        np.save(dst_file, noised_points)
        
        # 라벨도 복사
        src_label = src_labels / f"{original_id}.txt"
        if src_label.exists():
            dst_label = dst_labels / f"{noised_id}.txt"
            shutil.copy2(src_label, dst_label)
        
        noised_val_ids.append(noised_id)
        next_index += 1
    
    # val.txt 재작성
    print(f"\nUpdating val.txt...")
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
    print(f"  Overall noise ratio: {(len(noised_train_ids) + len(noised_val_ids)) / (total_train + total_val) * 100:.2f}%")


if __name__ == "__main__":
    src_root = Path("/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av")
    dst_root = Path("/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av_noise2")
    
    generate_noised_dataset(src_root, dst_root, noise_ratio=0.30)
    print("\n✓ All done!")

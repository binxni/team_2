#!/usr/bin/env python3
import os
import shutil
import numpy as np
from pathlib import Path

def merge_datasets():
    # 경로 설정
    base_path = Path("/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av_64")
    points_path = base_path / "points"
    points_lisa_path = base_path / "points_lisa"
    labels_path = base_path / "labels"
    
    # 출력 경로
    output_base = Path("/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av_hybrid")
    output_points = output_base / "points"
    output_labels = output_base / "labels"
    output_imagesets = output_base / "ImageSets"
    
    # 출력 디렉토리 생성
    output_points.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    output_imagesets.mkdir(parents=True, exist_ok=True)
    
    # 새로운 frame_id들을 저장할 리스트
    new_frame_ids = []
    
    # 현재 새로운 frame_id
    new_frame_id = 0
    
    print("Processing /points directory...")
    # /points 디렉토리 처리
    if points_path.exists():
        npy_files = sorted(list(points_path.glob("*.npy")))
        print(f"Found {len(npy_files)} files in /points")
        
        for npy_file in npy_files:
            original_name = npy_file.stem  # 확장자 제외한 파일명
            new_name = f"{new_frame_id:08d}"  # 8자리 0 패딩
            
            # .npy 파일 복사
            src_npy = npy_file
            dst_npy = output_points / f"{new_name}.npy"
            shutil.copy2(src_npy, dst_npy)
            print(f"Copied: {src_npy.name} -> {dst_npy.name}")
            
            # 해당하는 라벨 파일 복사
            src_label = labels_path / f"{original_name}.txt"
            dst_label = output_labels / f"{new_name}.txt"
            if src_label.exists():
                shutil.copy2(src_label, dst_label)
                print(f"Copied label: {src_label.name} -> {dst_label.name}")
            else:
                print(f"Warning: Label file not found: {src_label}")
            
            # 새로운 frame_id 저장
            new_frame_ids.append(new_name)
            new_frame_id += 1
    
    print("\nProcessing /points_lisa directory...")
    # /points_lisa 디렉토리 처리
    if points_lisa_path.exists():
        npy_files = sorted(list(points_lisa_path.glob("*.npy")))
        print(f"Found {len(npy_files)} files in /points_lisa")
        
        for npy_file in npy_files:
            original_name = npy_file.stem  # 확장자 제외한 파일명
            new_name = f"{new_frame_id:08d}"  # 8자리 0 패딩
            
            # .npy 파일 복사
            src_npy = npy_file
            dst_npy = output_points / f"{new_name}.npy"
            shutil.copy2(src_npy, dst_npy)
            print(f"Copied: {src_npy.name} -> {dst_npy.name}")
            
            # 해당하는 라벨 파일 복사 (원본 labels 디렉토리에서)
            src_label = labels_path / f"{original_name}.txt"
            dst_label = output_labels / f"{new_name}.txt"
            if src_label.exists():
                shutil.copy2(src_label, dst_label)
                print(f"Copied label: {src_label.name} -> {dst_label.name}")
            else:
                print(f"Warning: Label file not found: {src_label}")
            
            # 새로운 frame_id 저장
            new_frame_ids.append(new_name)
            new_frame_id += 1
    
    print("\nSaving train file...")
    # train.txt 파일에 새로운 frame_id들만 저장
    train_file = output_imagesets / "train.txt"
    with open(train_file, 'w') as f:
        for frame_id in new_frame_ids:
            f.write(frame_id + '\n')
    
    print(f"Train file saved to: {train_file}")
    print(f"Total files processed: {len(new_frame_ids)}")
    
    # 통계 출력
    points_count = len(list(output_points.glob("*.npy")))
    labels_count = len(list(output_labels.glob("*.txt")))
    
    print(f"\nSummary:")
    print(f"- Output points: {points_count} files")
    print(f"- Output labels: {labels_count} files")
    print(f"- Train entries: {len(new_frame_ids)} files")
    print(f"- Output directory: {output_base}")

if __name__ == "__main__":
    merge_datasets()
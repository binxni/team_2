"""
points_noise 폴더에서 3개 묶음마다 첫 번째 파일만 선택하여 
points 및 labels 폴더로 복사하는 스크립트
"""

import os
import shutil
from pathlib import Path
import glob

def main():
    # 경로 설정
    base_dir = Path("/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av")
    points_noise_dir = base_dir / "points_noise"
    points_dir = base_dir / "points"
    labels_dir = base_dir / "labels"
    
    # points_noise 폴더의 모든 .npy 파일 가져오기 (정렬)
    noise_files = sorted(glob.glob(str(points_noise_dir / "*.npy")))
    print(f"Total files in points_noise: {len(noise_files)}")
    
    # 3개 묶음마다 첫 번째 파일만 선택 (인덱스 0, 3, 6, 9, ...)
    selected_files = [noise_files[i] for i in range(0, len(noise_files), 3)]
    print(f"Selected files (1/3): {len(selected_files)}")
    
    # points 폴더의 기존 마지막 파일 번호 찾기
    existing_points = sorted(glob.glob(str(points_dir / "*.npy")))
    if existing_points:
        last_file = Path(existing_points[-1]).stem
        last_number = int(last_file)
        print(f"Last existing file in points: {last_file}")
    else:
        last_number = -1
        print("No existing files in points folder")
    
    # 다음 시작 번호
    next_number = last_number + 1
    print(f"Starting from: {next_number:08d}")
    
    # 복사 작업
    copied_count = 0
    for idx, noise_file in enumerate(selected_files):
        # 새 파일명 생성
        new_number = next_number + idx
        new_filename = f"{new_number:08d}"
        
        # points_noise의 원본 파일명 (확장자 제외)
        original_name = Path(noise_file).stem
        
        # 포인트 클라우드 복사
        src_point = noise_file
        dst_point = points_dir / f"{new_filename}.npy"
        shutil.copy2(src_point, dst_point)
        
        # 대응하는 라벨 복사 (points_noise의 원본 파일명에 해당하는 라벨)
        # points_noise/00000000.npy -> labels/00000000.txt
        src_label = labels_dir / f"{original_name}.txt"
        dst_label = labels_dir / f"{new_filename}.txt"
        
        if src_label.exists():
            shutil.copy2(src_label, dst_label)
            copied_count += 1
            if (idx + 1) % 100 == 0:
                print(f"Copied {idx + 1}/{len(selected_files)} files...")
        else:
            print(f"Warning: Label not found for {original_name}")
    
    print(f"\nCopy completed!")
    print(f"Total files copied: {copied_count}")
    print(f"Points range: {next_number:08d} to {next_number + len(selected_files) - 1:08d}")

if __name__ == "__main__":
    main()

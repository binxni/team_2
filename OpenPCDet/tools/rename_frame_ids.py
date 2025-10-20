#!/usr/bin/env python3
"""
Rename frame IDs from 000xxxxx.npy/txt format to 005xxxxx.npy/txt format
and generate val.txt with the new frame IDs.
"""

import os
import shutil
from pathlib import Path
from typing import List


def rename_files_in_directory(directory: Path, old_prefix: str, new_prefix: str, extension: str) -> List[str]:
    """
    Rename files with old_prefix to new_prefix and return list of new frame IDs.
    
    Args:
        directory: Directory containing the files
        old_prefix: Old prefix (e.g., "000")
        new_prefix: New prefix (e.g., "005") 
        extension: File extension (e.g., ".npy", ".txt")
    
    Returns:
        List of new frame IDs (without extension)
    """
    if not directory.exists():
        print(f"Directory {directory} does not exist, skipping...")
        return []
    
    files = list(directory.glob(f"{old_prefix}*{extension}"))
    new_frame_ids = []
    
    print(f"Renaming {len(files)} {extension} files in {directory}...")
    
    for file_path in sorted(files):
        old_filename = file_path.name
        
        # Extract the numeric part after the prefix
        numeric_part = old_filename[len(old_prefix):].split('.')[0]
        
        # Create new filename with new prefix
        new_filename = f"{new_prefix}{numeric_part}{extension}"
        new_file_path = directory / new_filename
        
        # Rename the file
        file_path.rename(new_file_path)
        
        # Store frame ID (without extension) for val.txt
        frame_id = f"{new_prefix}{numeric_part}"
        new_frame_ids.append(frame_id)
        
        if len(new_frame_ids) % 100 == 0:
            print(f"  Processed {len(new_frame_ids)} files...")
    
    print(f"  Completed renaming {len(files)} {extension} files")
    return new_frame_ids


def create_val_txt(frame_ids: List[str], output_path: Path) -> None:
    """
    Create val.txt file with frame IDs, one per line.
    
    Args:
        frame_ids: List of frame IDs
        output_path: Path to output val.txt file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for frame_id in sorted(frame_ids):
            f.write(f"{frame_id}\n")
    
    print(f"Created {output_path} with {len(frame_ids)} frame IDs")


def main():
    # Paths
    base_dir = Path("/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av_range64/waymo_npy")
    npy_dir = base_dir
    labels_dir = base_dir / "labels"
    val_txt_path = base_dir / "val.txt"
    
    # Parameters
    old_prefix = "000"
    new_prefix = "005"
    
    print("=" * 60)
    print("Frame ID 변경 작업 시작")
    print("=" * 60)
    print(f"Base directory: {base_dir}")
    print(f"변경: {old_prefix}xxxxx → {new_prefix}xxxxx")
    print("-" * 60)
    
    # Rename .npy files
    print("\n1. .npy 파일 이름 변경 중...")
    npy_frame_ids = rename_files_in_directory(npy_dir, old_prefix, new_prefix, ".npy")
    
    # Rename .txt files in labels directory
    print("\n2. labels 디렉토리의 .txt 파일 이름 변경 중...")
    txt_frame_ids = rename_files_in_directory(labels_dir, old_prefix, new_prefix, ".txt")
    
    # Use npy_frame_ids for val.txt (should be the same as txt_frame_ids)
    frame_ids = npy_frame_ids if npy_frame_ids else txt_frame_ids
    
    # Create val.txt
    print("\n3. val.txt 파일 생성 중...")
    create_val_txt(frame_ids, val_txt_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("Frame ID 변경 작업 완료!")
    print("=" * 60)
    print(f"변경된 .npy 파일: {len(npy_frame_ids):,}개")
    print(f"변경된 .txt 파일: {len(txt_frame_ids):,}개")
    print(f"생성된 val.txt: {val_txt_path}")
    print(f"총 frame ID: {len(frame_ids):,}개")
    
    if frame_ids:
        print(f"\n예시:")
        print(f"  첫 번째 frame ID: {frame_ids[0]}")
        print(f"  마지막 frame ID: {frame_ids[-1]}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
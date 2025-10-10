#!/usr/bin/env python3
"""
Script to create custom hybrid dataset combining Scene 1 and Scene 3 data.
- Scene 1: files starting with 000xxxxx (even frame_id only)
- Scene 3: files starting with 002xxxxx (even frame_id only)
- Even frame_id: select frames ending with even numbers (0, 2, 4, 6, 8)
- points files: save with original frame_id
- points_lisa files: save with frame_id+1 to avoid naming conflicts
"""

import os
import shutil
import glob
from pathlib import Path

def create_directory_structure(base_path):
    """Create the hybrid dataset directory structure."""
    hybrid_path = Path(base_path) / "custom_av_hybrid"
    
    # Create main directories
    dirs_to_create = [
        hybrid_path / "points",
        hybrid_path / "labels",
        hybrid_path / "ImageSets"
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    return hybrid_path

def filter_even_frames(file_list):
    """Filter files to get even frame_id (0, 2, 4, 6, 8)."""
    filtered_files = []
    
    for file_path in file_list:
        filename = os.path.basename(file_path)
        # Extract the frame_id
        frame_id = filename.replace('.npy', '')
        if frame_id.isdigit() and len(frame_id) == 8:
            last_digit = int(frame_id[-1])
            # Select frames ending with even numbers
            if last_digit % 2 == 0:
                filtered_files.append(file_path)
    
    return sorted(filtered_files)

def get_all_scene_files(source_folder):
    """Get all scene files: Scene 1 (000xxxxx), Scene 2 (001xxxxx), Scene 3 (002xxxxx), Scene 4 (003xxxxx)."""
    all_files = glob.glob(os.path.join(source_folder, "*.npy"))
    
    scene_1_files = []
    scene_2_files = []
    scene_3_files = []
    scene_4_files = []
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        frame_id = filename.replace('.npy', '')
        
        if frame_id.startswith('000'):  # Scene 1
            scene_1_files.append(file_path)
        elif frame_id.startswith('001'):  # Scene 2
            scene_2_files.append(file_path)
        elif frame_id.startswith('002'):  # Scene 3
            scene_3_files.append(file_path)
        elif frame_id.startswith('003'):  # Scene 4
            scene_4_files.append(file_path)
    
    return scene_1_files, scene_2_files, scene_3_files, scene_4_files

def copy_files_with_rename(source_files, dest_folder, add_one=False, description=""):
    """Copy files with optional frame_id+1 renaming."""
    dest_folder.mkdir(parents=True, exist_ok=True)
    copied_count = 0
    copied_files_info = []
    
    for source_file in source_files:
        original_filename = os.path.basename(source_file)
        frame_id = original_filename.replace('.npy', '')
        
        if add_one:
            # Add 1 to frame_id for points_lisa files
            new_frame_id = str(int(frame_id) + 1).zfill(8)
            new_filename = f"{new_frame_id}.npy"
        else:
            new_filename = original_filename
            new_frame_id = frame_id
        
        dest_file = dest_folder / new_filename
        shutil.copy2(source_file, dest_file)
        copied_count += 1
        copied_files_info.append((original_filename, new_filename, new_frame_id))
    
    print(f"{description}: Copied {copied_count} files")
    return copied_files_info

def copy_labels_with_rename(label_source_folder, dest_folder, points_info, points_lisa_info):
    """Copy label files for both points and points_lisa with appropriate renaming."""
    dest_folder.mkdir(parents=True, exist_ok=True)
    copied_count = 0
    all_frame_ids = []
    missing_labels = []
    
    # Copy labels for points files (original frame_id)
    for original_name, new_name, new_frame_id in points_info:
        original_frame_id = original_name.replace('.npy', '')
        label_source = label_source_folder / f"{original_frame_id}.txt"
        label_dest = dest_folder / f"{new_frame_id}.txt"
        
        if label_source.exists():
            shutil.copy2(label_source, label_dest)
            copied_count += 1
            all_frame_ids.append(new_frame_id)
        else:
            missing_labels.append(f"{original_frame_id}.txt (for points)")
    
    # Copy labels for points_lisa files (frame_id+1)
    for original_name, new_name, new_frame_id in points_lisa_info:
        original_frame_id = original_name.replace('.npy', '')
        label_source = label_source_folder / f"{original_frame_id}.txt"
        label_dest = dest_folder / f"{new_frame_id}.txt"
        
        if label_source.exists():
            shutil.copy2(label_source, label_dest)
            copied_count += 1
            all_frame_ids.append(new_frame_id)
        else:
            missing_labels.append(f"{original_frame_id}.txt (for points_lisa)")
    
    print(f"Label files: Copied {copied_count} files")
    if missing_labels:
        print(f"Missing labels: {len(missing_labels)} files")
        print("First 10 missing labels:", missing_labels[:10])
    
    return all_frame_ids

def generate_imagesets(selected_frame_ids, imagesets_folder):
    """Generate train_origin.txt file with all frame IDs."""
    imagesets_folder.mkdir(parents=True, exist_ok=True)
    
    # Sort frame IDs for consistent ordering
    sorted_ids = sorted(selected_frame_ids)
    
    # Write all frame IDs to train_origin.txt
    file_path = imagesets_folder / "train_origin.txt"
    with open(file_path, 'w') as f:
        for frame_id in sorted_ids:
            f.write(f"{frame_id}\n")
    
    print(f"Generated train_origin.txt: {len(sorted_ids)} frames")

def main():
    # Base paths
    base_path = "/home/ailab/git/Team_2/Subin/OpenPCDet/data"
    source_dataset = Path(base_path) / "custom_av_64"
    
    print("Creating custom_av_hybrid dataset...")
    print("=" * 50)
    
    # Step 1: Create directory structure
    print("Step 1: Creating directory structure")
    hybrid_path = create_directory_structure(base_path)
    
    # Step 2: Process points folder (original point cloud data)
    print("\nStep 2: Processing points folder")
    points_source = source_dataset / "points"
    scene_1_points, scene_2_points, scene_3_points, scene_4_points = get_all_scene_files(points_source)
    
    # Filter frames: Scene 1,3 (even only), Scene 2,4 (all frames)
    scene_1_filtered = filter_even_frames(scene_1_points)
    scene_2_filtered = sorted(scene_2_points)  # All frames for Scene 2
    scene_3_filtered = filter_even_frames(scene_3_points)
    scene_4_filtered = sorted(scene_4_points)  # All frames for Scene 4
    
    print(f"Scene 1 points: {len(scene_1_points)} total -> {len(scene_1_filtered)} filtered (even only)")
    print(f"Scene 2 points: {len(scene_2_points)} total -> {len(scene_2_filtered)} filtered (all frames)")
    print(f"Scene 3 points: {len(scene_3_points)} total -> {len(scene_3_filtered)} filtered (even only)")
    print(f"Scene 4 points: {len(scene_4_points)} total -> {len(scene_4_filtered)} filtered (all frames)")
    
    # Copy filtered points files (original frame_id)
    all_points_files = scene_1_filtered + scene_2_filtered + scene_3_filtered + scene_4_filtered
    points_info = copy_files_with_rename(all_points_files, hybrid_path / "points", 
                                       add_one=False, description="Points files")
    
    # Step 3: Process points_lisa folder (noisy point cloud data)
    print("\nStep 3: Processing points_lisa folder")
    points_lisa_source = source_dataset / "points_lisa"
    scene_1_lisa, scene_2_lisa, scene_3_lisa, scene_4_lisa = get_all_scene_files(points_lisa_source)
    
    # Filter frames: Scene 1,3 (even only), Scene 2,4 (all frames)
    scene_1_lisa_filtered = filter_even_frames(scene_1_lisa)
    scene_2_lisa_filtered = sorted(scene_2_lisa)  # All frames for Scene 2
    scene_3_lisa_filtered = filter_even_frames(scene_3_lisa)
    scene_4_lisa_filtered = sorted(scene_4_lisa)  # All frames for Scene 4
    
    print(f"Scene 1 points_lisa: {len(scene_1_lisa)} total -> {len(scene_1_lisa_filtered)} filtered (even only)")
    print(f"Scene 2 points_lisa: {len(scene_2_lisa)} total -> {len(scene_2_lisa_filtered)} filtered (all frames)")
    print(f"Scene 3 points_lisa: {len(scene_3_lisa)} total -> {len(scene_3_lisa_filtered)} filtered (even only)")
    print(f"Scene 4 points_lisa: {len(scene_4_lisa)} total -> {len(scene_4_lisa_filtered)} filtered (all frames)")
    
    # Copy filtered points_lisa files (frame_id+1) to the same points folder
    all_lisa_files = scene_1_lisa_filtered + scene_2_lisa_filtered + scene_3_lisa_filtered + scene_4_lisa_filtered
    points_lisa_info = copy_files_with_rename(all_lisa_files, hybrid_path / "points", 
                                            add_one=True, description="Points_lisa files")
    
    # Step 4: Process labels folder
    print("\nStep 4: Processing labels folder")
    labels_source = source_dataset / "labels"
    
    # Copy labels for both points and points_lisa files
    all_frame_ids = copy_labels_with_rename(labels_source, hybrid_path / "labels", 
                                          points_info, points_lisa_info)
    
    # Step 5: Generate ImageSets
    print("\nStep 5: Generating ImageSets")
    generate_imagesets(all_frame_ids, hybrid_path / "ImageSets")
    
    # Step 6: Summary
    print("\n" + "=" * 50)
    print("HYBRID DATASET CREATION SUMMARY")
    print("=" * 50)
    print(f"Dataset location: {hybrid_path}")
    print(f"Total frames: {len(all_frame_ids)}")
    print(f"Scene 1 frames (even only): {len(scene_1_filtered)}")
    print(f"Scene 2 frames (all): {len(scene_2_filtered)}")
    print(f"Scene 3 frames (even only): {len(scene_3_filtered)}")
    print(f"Scene 4 frames (all): {len(scene_4_filtered)}")
    print(f"Original points files: {len(points_info)}")
    print(f"Points_lisa files (with frame_id+1): {len(points_lisa_info)}")
    print(f"Total points files in /points: {len(points_info) + len(points_lisa_info)}")
    print(f"Label files: {len(all_frame_ids)}")
    print("\nDataset structure created successfully!")

if __name__ == "__main__":
    main()
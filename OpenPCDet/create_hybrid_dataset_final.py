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

def generate_imagesets(id_mapping, imagesets_folder, source_imagesets_folder):
    """Generate train.txt and val.txt based on source dataset splits and ID mapping."""
    imagesets_folder.mkdir(parents=True, exist_ok=True)
    source_imagesets_folder = Path(source_imagesets_folder)
    
    # Read original train.txt and val.txt from custom_av_64
    train_file = source_imagesets_folder / "train.txt"
    val_file = source_imagesets_folder / "val.txt"
    
    original_train_ids = set()
    original_val_ids = set()
    
    if train_file.exists():
        with open(train_file, 'r') as f:
            original_train_ids = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(original_train_ids)} IDs from original train.txt")
    else:
        print("Warning: train.txt not found in source ImageSets")
    
    if val_file.exists():
        with open(val_file, 'r') as f:
            original_val_ids = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(original_val_ids)} IDs from original val.txt")
    else:
        print("Warning: val.txt not found in source ImageSets")
    
    # Map original IDs to new IDs based on train/val split
    new_train_ids = []
    new_val_ids = []
    unmapped_count = 0
    
    for original_id, new_ids_list in sorted(id_mapping.items()):
        if original_id in original_train_ids:
            # Add all new IDs (both points and points_lisa) to train
            new_train_ids.extend(new_ids_list)
        elif original_id in original_val_ids:
            # Add all new IDs (both points and points_lisa) to val
            new_val_ids.extend(new_ids_list)
        else:
            # If original ID not found in train/val, add to train by default
            new_train_ids.extend(new_ids_list)
            unmapped_count += 1
    
    if unmapped_count > 0:
        print(f"Warning: {unmapped_count} original IDs not found in train/val, added to train")
    
    # Sort for consistent ordering
    new_train_ids.sort()
    new_val_ids.sort()
    
    # Write new train.txt
    train_output = imagesets_folder / "train.txt"
    with open(train_output, 'w') as f:
        for frame_id in new_train_ids:
            f.write(f"{frame_id}\n")
    
    # Write new val.txt
    val_output = imagesets_folder / "val.txt"
    with open(val_output, 'w') as f:
        for frame_id in new_val_ids:
            f.write(f"{frame_id}\n")
    
    print(f"Generated train.txt: {len(new_train_ids)} frames")
    print(f"Generated val.txt: {len(new_val_ids)} frames")
    print(f"Total: {len(new_train_ids) + len(new_val_ids)} frames")

def merge_points(points_files, points_lisa_files, dest_dir, labels_source, label_dest):
    """
    Merge points and points_lisa files with proper renaming.
    
    Args:
        points_files: List of point cloud file paths
        points_lisa_files: List of noisy point cloud file paths
        dest_dir: Destination directory for merged points
        labels_source: Source directory for labels
        label_dest: Destination directory for labels
    
    Returns:
        all_frame_ids: List of all new frame IDs in the merged dataset
        id_mapping: Dictionary mapping original_id -> list of new_ids
    """
    dest_dir = Path(dest_dir)
    labels_source = Path(labels_source)
    label_dest = Path(label_dest)
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    label_dest.mkdir(parents=True, exist_ok=True)
    
    all_frame_ids = []
    id_mapping = {}  # original_id -> [new_id_for_points, new_id_for_lisa]
    index = 0
    
    # Create a mapping of original frame_id to files
    points_dict = {}
    for file_path in points_files:
        filename = os.path.basename(file_path)
        frame_id = filename.replace('.npy', '')
        points_dict[frame_id] = file_path
    
    points_lisa_dict = {}
    for file_path in points_lisa_files:
        filename = os.path.basename(file_path)
        frame_id = filename.replace('.npy', '')
        points_lisa_dict[frame_id] = file_path
    
    # Get all unique frame IDs
    all_original_frame_ids = sorted(set(points_dict.keys()) | set(points_lisa_dict.keys()))
    
    print(f"Processing {len(all_original_frame_ids)} unique frame IDs...")
    
    for original_frame_id in all_original_frame_ids:
        id_mapping[original_frame_id] = []
        
        # Process points file if exists
        if original_frame_id in points_dict:
            new_frame_id = str(index).zfill(8)
            
            # Copy point cloud file
            source_file = points_dict[original_frame_id]
            dest_file = dest_dir / f"{new_frame_id}.npy"
            shutil.copy2(source_file, dest_file)
            
            # Copy corresponding label file
            label_source_file = labels_source / f"{original_frame_id}.txt"
            if label_source_file.exists():
                label_dest_file = label_dest / f"{new_frame_id}.txt"
                shutil.copy2(label_source_file, label_dest_file)
                all_frame_ids.append(new_frame_id)
                id_mapping[original_frame_id].append(new_frame_id)
            else:
                print(f"Warning: Label not found for {original_frame_id}")
            
            index += 1
        
        # Process points_lisa file if exists
        if original_frame_id in points_lisa_dict:
            new_frame_id = str(index).zfill(8)
            
            # Copy point cloud file
            source_file = points_lisa_dict[original_frame_id]
            dest_file = dest_dir / f"{new_frame_id}.npy"
            shutil.copy2(source_file, dest_file)
            
            # Copy corresponding label file
            label_source_file = labels_source / f"{original_frame_id}.txt"
            if label_source_file.exists():
                label_dest_file = label_dest / f"{new_frame_id}.txt"
                shutil.copy2(label_source_file, label_dest_file)
                all_frame_ids.append(new_frame_id)
                id_mapping[original_frame_id].append(new_frame_id)
            else:
                print(f"Warning: Label not found for {original_frame_id} (lisa)")
            
            index += 1
    
    print(f"Merged {index} files successfully")
    print(f"Created mapping for {len(id_mapping)} original IDs")
    return all_frame_ids, id_mapping

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
    
    all_points_files = scene_1_filtered + scene_2_filtered + scene_3_filtered + scene_4_filtered
    
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
    
    all_lisa_files = scene_1_lisa_filtered + scene_2_lisa_filtered + scene_3_lisa_filtered + scene_4_lisa_filtered
    
    # Step 4: Merge points and points_lisa files with labels
    print("\nStep 4: Merging points, points_lisa, and labels")
    labels_source = source_dataset / "labels"
    
    all_frame_ids, id_mapping = merge_points(
        all_points_files, 
        all_lisa_files, 
        hybrid_path / "points",
        labels_source, 
        hybrid_path / "labels"
    )
    
    # Step 5: Generate ImageSets
    print("\nStep 5: Generating ImageSets")
    source_imagesets = source_dataset / "ImageSets"
    generate_imagesets(id_mapping, hybrid_path / "ImageSets", source_imagesets)
    
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
    print(f"Total original points files: {len(all_points_files)}")
    print(f"Total points_lisa files: {len(all_lisa_files)}")
    print(f"Total merged files: {len(all_frame_ids)}")
    print(f"\nImageSets created: train.txt and val.txt")
    print("\nDataset structure created successfully!")

if __name__ == "__main__":
    main()
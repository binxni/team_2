import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from pathlib import Path
import argparse

def load_sample_ids(txt_path):
    """ImageSets txt 파일에서 샘플 ID 목록 로드"""
    with open(txt_path, 'r') as f:
        sample_ids = [line.strip() for line in f.readlines() if line.strip()]
    return sample_ids

def load_pointcloud(data_path, sample_id):
    """포인트 클라우드 데이터 로드"""
    pc_file = Path(data_path) / 'points' / f'{sample_id}.npy'
    if pc_file.exists():
        return np.load(pc_file)
    else:
        print(f"Warning: {pc_file} not found")
        return None

def load_labels(data_path, sample_id):
    """라벨 데이터 로드"""
    label_file = Path(data_path) / 'labels' / f'{sample_id}.txt'
    if label_file.exists():
        gt_boxes = []
        gt_names = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:
                    gt_boxes.append([float(x) for x in parts[:7]])  # x,y,z,l,w,h,angle
                    gt_names.append(parts[7])
        return np.array(gt_boxes), gt_names
    return None, None

def visualize_pointcloud_2d(points, gt_boxes=None, title="Point Cloud Top View"):
    """2D 탑뷰로 포인트 클라우드 시각화"""
    plt.figure(figsize=(12, 10))
    
    # 포인트 클라우드 플롯
    plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=0.1, cmap='viridis', alpha=0.6)
    plt.colorbar(label='Height (Z)')
    
    # GT 박스 그리기
    if gt_boxes is not None and len(gt_boxes) > 0:
        for box in gt_boxes:
            x, y, z, l, w, h, angle = box
            # 박스 모서리 계산
            corners = np.array([
                [-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2], [-l/2, -w/2]
            ])
            # 회전 적용
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_corners = corners @ rot_matrix.T
            rotated_corners[:, 0] += x
            rotated_corners[:, 1] += y
            
            plt.plot(rotated_corners[:, 0], rotated_corners[:, 1], 'r-', linewidth=2)
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(title)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)

def analyze_scene_continuity(data_path, sample_ids, max_samples=20):
    """연속성 분석을 위해 여러 프레임의 통계 정보 확인"""
    print(f"Analyzing {min(len(sample_ids), max_samples)} samples from train_origin.txt")
    print("=" * 60)
    
    stats = []
    for i, sample_id in enumerate(sample_ids[:max_samples]):
        points = load_pointcloud(data_path, sample_id)
        gt_boxes, gt_names = load_labels(data_path, sample_id)
        
        if points is not None:
            # 통계 계산
            center_x, center_y = np.mean(points[:, 0]), np.mean(points[:, 1])
            std_x, std_y = np.std(points[:, 0]), np.std(points[:, 1])
            num_objects = len(gt_boxes) if gt_boxes is not None else 0
            
            stats.append({
                'sample_id': sample_id,
                'num_points': len(points),
                'center_x': center_x,
                'center_y': center_y,
                'std_x': std_x,
                'std_y': std_y,
                'num_objects': num_objects
            })
            
            print(f"Sample {i+1:2d}: {sample_id} | Points: {len(points):6d} | "
                  f"Center: ({center_x:6.1f}, {center_y:6.1f}) | Objects: {num_objects}")
    
    return stats

def visualize_multiple_frames(data_path, sample_ids, num_frames=6):
    """여러 프레임을 동시에 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i in range(min(num_frames, len(sample_ids))):
        sample_id = sample_ids[i]
        points = load_pointcloud(data_path, sample_id)
        gt_boxes, gt_names = load_labels(data_path, sample_id)
        
        if points is not None:
            ax = axes[i]
            
            # 포인트 클라우드 플롯
            scatter = ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], 
                               s=0.5, cmap='viridis', alpha=0.6)
            
            # GT 박스 그리기
            if gt_boxes is not None and len(gt_boxes) > 0:
                for box in gt_boxes:
                    x, y, z, l, w, h, angle = box
                    corners = np.array([
                        [-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2], [-l/2, -w/2]
                    ])
                    cos_a, sin_a = np.cos(angle), np.sin(angle)
                    rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                    rotated_corners = corners @ rot_matrix.T
                    rotated_corners[:, 0] += x
                    rotated_corners[:, 1] += y
                    
                    ax.plot(rotated_corners[:, 0], rotated_corners[:, 1], 'r-', linewidth=1.5)
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'Frame {i+1}: {sample_id}')
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize Custom AV Point Cloud Data')
    parser.add_argument('--data_path', type=str, default='/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av',
                       help='Path to custom_av dataset')
    parser.add_argument('--txt_file', type=str, default='train_origin.txt',
                       help='ImageSets txt file name')
    parser.add_argument('--max_samples', type=int, default=20,
                       help='Maximum number of samples to analyze')
    parser.add_argument('--visualize_frames', type=int, default=6,
                       help='Number of frames to visualize')
    
    args = parser.parse_args()
    
    # 데이터 경로 설정
    data_path = Path(args.data_path)
    txt_path = data_path / 'ImageSets' / args.txt_file
    
    if not txt_path.exists():
        print(f"Error: {txt_path} not found!")
        return
    
    # 샘플 ID 로드
    sample_ids = load_sample_ids(txt_path)
    print(f"Total samples in {args.txt_file}: {len(sample_ids)}")
    
    # 1. 연속성 분석
    print("\n1. Scene Continuity Analysis:")
    stats = analyze_scene_continuity(data_path, sample_ids, args.max_samples)
    
    # 2. 중심점 변화 분석
    if len(stats) > 1:
        print("\n2. Center Point Movement Analysis:")
        print("=" * 60)
        for i in range(1, len(stats)):
            prev_center = (stats[i-1]['center_x'], stats[i-1]['center_y'])
            curr_center = (stats[i]['center_x'], stats[i]['center_y'])
            movement = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                             (curr_center[1] - prev_center[1])**2)
            print(f"Frame {i} -> {i+1}: Movement = {movement:.2f}m")
    
    # 3. 여러 프레임 시각화
    print(f"\n3. Visualizing first {args.visualize_frames} frames:")
    visualize_multiple_frames(data_path, sample_ids, args.visualize_frames)
    
    # 4. 개별 프레임 상세 시각화 (첫 번째와 마지막 프레임)
    print("\n4. Detailed visualization of first and last frames:")
    
    # 첫 번째 프레임
    first_points = load_pointcloud(data_path, sample_ids[0])
    first_boxes, first_names = load_labels(data_path, sample_ids[0])
    if first_points is not None:
        visualize_pointcloud_2d(first_points, first_boxes, 
                               f"First Frame: {sample_ids[0]}")
        plt.show()
    
    # 마지막 프레임 (충분한 데이터가 있는 경우)
    if len(sample_ids) > 1:
        last_idx = min(len(sample_ids)-1, args.max_samples-1)
        last_points = load_pointcloud(data_path, sample_ids[last_idx])
        last_boxes, last_names = load_labels(data_path, sample_ids[last_idx])
        if last_points is not None:
            visualize_pointcloud_2d(last_points, last_boxes, 
                                   f"Frame {last_idx+1}: {sample_ids[last_idx]}")
            plt.show()
    
    # 5. 결론 출력
    print("\n5. Analysis Conclusion:")
    print("=" * 60)
    if len(stats) > 1:
        movements = []
        for i in range(1, len(stats)):
            prev_center = (stats[i-1]['center_x'], stats[i-1]['center_y'])
            curr_center = (stats[i]['center_x'], stats[i]['center_y'])
            movement = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                             (curr_center[1] - prev_center[1])**2)
            movements.append(movement)
        
        avg_movement = np.mean(movements)
        max_movement = np.max(movements)
        
        print(f"Average movement between consecutive frames: {avg_movement:.2f}m")
        print(f"Maximum movement between consecutive frames: {max_movement:.2f}m")
        
        if avg_movement < 5.0:  # 5m 이하면 연속된 scene으로 판단
            print("🎯 RESULT: This appears to be a CONTINUOUS DRIVING SCENE")
            print("   - Small movements between frames suggest temporal continuity")
            print("   - Recommended split strategy: Temporal splitting (avoid consecutive frames in different splits)")
        else:
            print("🎯 RESULT: This appears to be MULTIPLE DIFFERENT SCENES")
            print("   - Large movements between frames suggest different locations/scenes")
            print("   - Recommended split strategy: Random splitting is safe")

if __name__ == "__main__":
    main()
import numpy as np
import os
import argparse
from datetime import datetime

# Configuration
DATA_PATH = "/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av"
FRAME_LIST_FILE = os.path.join(DATA_PATH, "ImageSets", "train_origin.txt")
POINTS_FOLDER = os.path.join(DATA_PATH, "points")
LABELS_FOLDER = os.path.join(DATA_PATH, "labels")
OUTPUT_DIR = "/home/ailab/git/Team_2/Seokjae/OpenPCDet/dataset_analysis"  

# Target classes for analysis
TARGET_CLASSES = ["Vehicle", "Pedestrian", "Cyclist"]

def load_frame_ids():
    """프레임 ID 목록을 로드합니다."""
    with open(FRAME_LIST_FILE, 'r') as f:
        frame_ids = [line.strip() for line in f.readlines()]
    return frame_ids

def load_npy_pointcloud(file_path):
    """포인트 클라우드를 로드합니다."""
    if not os.path.exists(file_path):
        return None
    points = np.load(file_path)
    return points

def load_gt_labels(label_file_path):
    """GT 라벨 파일을 로드합니다."""
    if not os.path.exists(label_file_path):
        return []
    
    labels = []
    with open(label_file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 8:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    dx, dy, dz = float(parts[3]), float(parts[4]), float(parts[5])
                    rotation = float(parts[6])
                    class_name = parts[7]
                    
                    labels.append({
                        'center': [x, y, z],
                        'dimensions': [dx, dy, dz],
                        'rotation': rotation,
                        'class': class_name
                    })
    return labels

def points_in_box_3d_vectorized(points, box_center, box_dimensions, box_rotation):
    """벡터화된 3D 바운딩 박스 내부 포인트 검사 (NumPy 최적화)"""
    if len(points) == 0:
        return np.array([], dtype=bool)
    
    # 포인트를 numpy 배열로 변환
    points = np.array(points)
    if len(points.shape) == 1:
        points = points.reshape(1, -1)
    
    # 박스 중심으로 이동
    centered_points = points[:, :3] - np.array(box_center)
    
    # 회전 적용 (z축 회전만 고려)
    cos_r = np.cos(-box_rotation)
    sin_r = np.sin(-box_rotation)
    
    # 회전 행렬 적용
    rotated_x = centered_points[:, 0] * cos_r - centered_points[:, 1] * sin_r
    rotated_y = centered_points[:, 0] * sin_r + centered_points[:, 1] * cos_r
    rotated_z = centered_points[:, 2]
    
    # 박스 크기의 절반
    half_dims = np.array(box_dimensions) / 2.0
    
    # 박스 내부 확인 (벡터화)
    inside_x = np.abs(rotated_x) <= half_dims[0]
    inside_y = np.abs(rotated_y) <= half_dims[1]
    inside_z = np.abs(rotated_z) <= half_dims[2]
    
    return inside_x & inside_y & inside_z

def count_points_in_boxes_optimized(points, boxes):
    """최적화된 박스 내 포인트 개수 계산 (벡터화)"""
    if len(points) == 0 or len(boxes) == 0:
        return []
    
    box_point_counts = []
    points_array = np.array(points)
    
    for box in boxes:
        inside_mask = points_in_box_3d_vectorized(
            points_array, 
            box['center'], 
            box['dimensions'], 
            box['rotation']
        )
        box_point_counts.append(np.sum(inside_mask))
    
    return box_point_counts

def analyze_gt_box_points(points, labels):
    """GT 바운딩 박스 내 포인트 개수 분석"""
    if points is None or len(points) == 0:
        return None
    
    if len(labels) == 0:
        return {
            'total_points': len(points),
            'total_boxes': 0,
            'boxes_by_class': {cls: [] for cls in TARGET_CLASSES},
            'avg_points_per_box_by_class': {cls: 0.0 for cls in TARGET_CLASSES},
            'total_boxes_by_class': {cls: 0 for cls in TARGET_CLASSES},
            'status': 'No labels'
        }
    
    # 대상 클래스만 필터링
    target_boxes = [box for box in labels if box['class'] in TARGET_CLASSES]
    
    if len(target_boxes) == 0:
        return {
            'total_points': len(points),
            'total_boxes': 0,
            'boxes_by_class': {cls: [] for cls in TARGET_CLASSES},
            'avg_points_per_box_by_class': {cls: 0.0 for cls in TARGET_CLASSES},
            'total_boxes_by_class': {cls: 0 for cls in TARGET_CLASSES},
            'status': 'No target class boxes'
        }
    
    # 각 박스 내 포인트 개수 계산 (최적화된 버전)
    box_point_counts = count_points_in_boxes_optimized(points, target_boxes)
    
    # 클래스별 통계 계산
    boxes_by_class = {cls: [] for cls in TARGET_CLASSES}
    total_boxes_by_class = {cls: 0 for cls in TARGET_CLASSES}
    
    for i, box in enumerate(target_boxes):
        class_name = box['class']
        if class_name in TARGET_CLASSES:
            boxes_by_class[class_name].append(box_point_counts[i])
            total_boxes_by_class[class_name] += 1
    
    # 클래스별 평균 계산
    avg_points_per_box_by_class = {}
    for cls in TARGET_CLASSES:
        if total_boxes_by_class[cls] > 0:
            avg_points_per_box_by_class[cls] = np.mean(boxes_by_class[cls])
        else:
            avg_points_per_box_by_class[cls] = 0.0
    
    return {
        'total_points': len(points),
        'total_boxes': len(target_boxes),
        'boxes_by_class': boxes_by_class,
        'avg_points_per_box_by_class': avg_points_per_box_by_class,
        'total_boxes_by_class': total_boxes_by_class,
        'status': 'OK'
    }

def analyze_gt_box_statistics(start_idx=None, end_idx=None):
    """GT 바운딩 박스 포인트 통계 분석 메인 함수"""
    import time
    
    frame_ids = load_frame_ids()
    total_frames = len(frame_ids)
    
    # 범위 설정
    start_idx = start_idx if start_idx is not None else 0
    end_idx = end_idx if end_idx is not None else total_frames
    
    # 범위 검증
    start_idx = max(0, start_idx)
    end_idx = min(total_frames, end_idx)
    
    if start_idx >= end_idx:
        print("Invalid frame range!")
        return
    
    target_frame_count = end_idx - start_idx
    print(f"\n🚀 Starting GT box point analysis...")
    print(f"📊 Processing frames {start_idx} to {end_idx-1} ({target_frame_count:,} total frames)")
    print(f"📁 Data path: {DATA_PATH}")
    print(f"🎯 Target classes: {TARGET_CLASSES}")
    print("-" * 80)
    
    # 분석 결과 저장
    analysis_results = []
    start_time = time.time()
    
    for idx, i in enumerate(range(start_idx, end_idx)):
        frame_id = frame_ids[i]
        pc_path = os.path.join(POINTS_FOLDER, f"{frame_id}.npy")
        label_path = os.path.join(LABELS_FOLDER, f"{frame_id}.txt")
        
        # 진행률 계산 및 출력
        progress = (idx + 1) / target_frame_count
        elapsed_time = time.time() - start_time
        
        if idx % max(1, target_frame_count // 50) == 0 or idx == target_frame_count - 1:  # 2% 간격으로 출력
            eta = elapsed_time / (idx + 1) * (target_frame_count - idx - 1) if idx > 0 else 0
            print(f"\r⏳ Progress: {progress*100:5.1f}% [{idx+1:,}/{target_frame_count:,}] "
                  f"| Frame: {frame_id} | Elapsed: {elapsed_time:.1f}s | ETA: {eta:.1f}s", end="", flush=True)
        
        # 포인트 클라우드 로드
        points = load_npy_pointcloud(pc_path)
        
        # GT 라벨 로드
        labels = load_gt_labels(label_path)
        
        # GT 박스 포인트 분석
        box_stats = analyze_gt_box_points(points, labels)
        
        if box_stats is not None:
            box_stats['frame_idx'] = i
            box_stats['frame_id'] = frame_id
        else:
            box_stats = {
                'frame_idx': i,
                'frame_id': frame_id,
                'total_points': 0,
                'total_boxes': 0,
                'boxes_by_class': {cls: [] for cls in TARGET_CLASSES},
                'avg_points_per_box_by_class': {cls: 0.0 for cls in TARGET_CLASSES},
                'total_boxes_by_class': {cls: 0 for cls in TARGET_CLASSES},
                'status': 'File not found'
            }
        
        analysis_results.append(box_stats)
    
    print()  # 새 줄
    total_time = time.time() - start_time
    avg_time_per_frame = total_time / target_frame_count
    
    print(f"\n✅ Analysis completed successfully!")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📈 Average time per frame: {avg_time_per_frame:.3f} seconds")
    print(f"🚀 Processing speed: {target_frame_count/total_time:.1f} frames/second")
    
    # 간단한 통계 출력
    valid_results = [r for r in analysis_results if r['status'] == 'OK']
    if valid_results:
        total_points = sum(r['total_points'] for r in valid_results)
        total_boxes = sum(r['total_boxes'] for r in valid_results)
        print(f"\n📊 Quick Stats:")
        print(f"   Valid frames: {len(valid_results):,}")
        print(f"   Total points: {total_points:,}")
        print(f"   Total GT boxes: {total_boxes:,}")
        print(f"   Avg points per frame: {total_points/len(valid_results):,.0f}")
    
    # 결과 저장
    print(f"\n💾 Saving results to file...")
    output_file = save_results_to_txt(analysis_results, start_idx, end_idx)
    print(f"📄 Results saved to: {output_file}")
    
    return analysis_results

def save_results_to_txt(analysis_results, start_idx, end_idx):
    """분석 결과를 TXT 파일로 저장"""
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # TXT 파일 저장
    txt_filename = os.path.join(OUTPUT_DIR, f"gt_box_point_analysis_128.txt")
    
    with open(txt_filename, 'w', encoding='utf-8') as txtfile:
        txtfile.write(f"GT Bounding Box Point Analysis Report\n")
        txtfile.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        txtfile.write(f"Frames: {start_idx} to {end_idx-1} (total: {end_idx - start_idx})\n")
        txtfile.write("=" * 80 + "\n\n")
        
        txtfile.write("Configuration:\n")
        txtfile.write(f"  Target Classes: {TARGET_CLASSES}\n")
        txtfile.write("-" * 80 + "\n\n")
        
        # 개별 프레임 결과
        valid_results = [r for r in analysis_results if r['status'] == 'OK']
        
        for result in valid_results:
            txtfile.write(f"Frame {result['frame_idx']:4d}: {result['frame_id']}\n")
            txtfile.write(f"  Total points: {result['total_points']:,}\n")
            txtfile.write(f"  Total GT boxes: {result['total_boxes']:,}\n")
            
            # 클래스별 박스 개수 및 평균 포인트 수
            for cls in TARGET_CLASSES:
                box_count = result['total_boxes_by_class'][cls]
                avg_points = result['avg_points_per_box_by_class'][cls]
                txtfile.write(f"  {cls}: {box_count} boxes, avg {avg_points:.1f} points per box\n")
            
            txtfile.write("-" * 60 + "\n")
        
        # 전체 통계
        if valid_results:
            txtfile.write(f"\nOverall Statistics ({len(valid_results)} valid frames):\n")
            txtfile.write("=" * 80 + "\n")
            
            # 클래스별 전체 통계 계산
            for cls in TARGET_CLASSES:
                all_points_for_class = []
                total_boxes_for_class = 0
                
                for result in valid_results:
                    if cls in result['boxes_by_class']:
                        all_points_for_class.extend(result['boxes_by_class'][cls])
                        total_boxes_for_class += result['total_boxes_by_class'][cls]
                
                if total_boxes_for_class > 0:
                    avg_points = np.mean(all_points_for_class)
                    std_points = np.std(all_points_for_class)
                    min_points = np.min(all_points_for_class)
                    max_points = np.max(all_points_for_class)
                    
                    txtfile.write(f"\n{cls} (Total: {total_boxes_for_class} boxes):\n")
                    txtfile.write(f"  Average points per box: {avg_points:.2f}\n")
                    txtfile.write(f"  Standard deviation: {std_points:.2f}\n")
                    txtfile.write(f"  Min points per box: {min_points}\n")
                    txtfile.write(f"  Max points per box: {max_points}\n")
                else:
                    txtfile.write(f"\n{cls}: No boxes found\n")
            
            # 전체 평균
            all_boxes_all_classes = []
            for result in valid_results:
                for cls in TARGET_CLASSES:
                    if cls in result['boxes_by_class']:
                        all_boxes_all_classes.extend(result['boxes_by_class'][cls])
            
            if all_boxes_all_classes:
                overall_avg = np.mean(all_boxes_all_classes)
                txtfile.write(f"\nOverall average points per GT box: {overall_avg:.2f}\n")
    
    print(f"📄 Analysis report saved: {txt_filename}")
    return txt_filename

def parse_arguments():
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(description='Analyze GT bounding box point statistics for point cloud frames')
    parser.add_argument('--start', type=int, default=None, 
                       help='Start frame index (default: 0)')
    parser.add_argument('--end', type=int, default=None, 
                       help='End frame index (default: total frames)')
    parser.add_argument('--data_path', type=str, default=DATA_PATH,
                       help='Path to custom_av dataset')
    parser.add_argument('--split', type=str, default='train_128', 
                       choices=['train', 'train_128', 'train_origin', 'train_origin_old', 'val', 'test'],
                       help='Dataset split to analyze')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # 전역 변수 업데이트
    global DATA_PATH, FRAME_LIST_FILE, POINTS_FOLDER, LABELS_FOLDER
    DATA_PATH = args.data_path
    FRAME_LIST_FILE = os.path.join(DATA_PATH, "ImageSets", f"{args.split}.txt")
    POINTS_FOLDER = os.path.join(DATA_PATH, "points")
    LABELS_FOLDER = os.path.join(DATA_PATH, "labels")
    
    if not os.path.exists(FRAME_LIST_FILE):
        print(f"Frame list file not found: {FRAME_LIST_FILE}")
        return
    
    # GT 박스 통계 분석
    analyze_gt_box_statistics(args.start, args.end)

if __name__ == "__main__":
    main()
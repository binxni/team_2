import numpy as np
import os
import argparse
import csv
from datetime import datetime

# Configuration
DATA_PATH = "/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av"
FRAME_LIST_FILE = os.path.join(DATA_PATH, "ImageSets", "train_origin_old.txt")
POINTS_FOLDER = os.path.join(DATA_PATH, "points")
LABELS_FOLDER = os.path.join(DATA_PATH, "labels")
OUTPUT_DIR = "/home/ailab/git/Team_2/Seokjae/OpenPCDet/dataset_analysis"

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

def load_gt_boxes(label_path):
    """GT 바운딩 박스를 로드합니다."""
    if not os.path.exists(label_path):
        return []
    
    gt_boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                # x, y, z, l, w, h, angle, class_name
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                l, w, h = float(parts[3]), float(parts[4]), float(parts[5])
                angle = float(parts[6])
                class_name = parts[7] if len(parts) > 7 else "Unknown"
                
                gt_boxes.append({
                    'center': [x, y, z],
                    'size': [l, w, h],
                    'angle': angle,
                    'class': class_name,
                    'x_min': x - l/2, 'x_max': x + l/2,
                    'y_min': y - w/2, 'y_max': y + w/2,
                    'z_min': z - h/2, 'z_max': z + h/2
                })
    
    return gt_boxes

def find_extreme_boxes(gt_boxes):
    """각 축에서 가장 멀리 떨어진 바운딩 박스를 찾습니다."""
    if not gt_boxes:
        return None
    
    # X축에서 가장 멀리 떨어진 박스들 (절댓값 기준)
    x_distances = [max(abs(box['x_min']), abs(box['x_max'])) for box in gt_boxes]
    max_x_idx = np.argmax(x_distances)
    
    # Y축에서 가장 멀리 떨어진 박스들 (절댓값 기준)
    y_distances = [max(abs(box['y_min']), abs(box['y_max'])) for box in gt_boxes]
    max_y_idx = np.argmax(y_distances)
    
    # Z축에서 가장 멀리 떨어진 박스들 (절댓값 기준)
    z_distances = [max(abs(box['z_min']), abs(box['z_max'])) for box in gt_boxes]
    max_z_idx = np.argmax(z_distances)
    
    return {
        'extreme_x': {
            'box': gt_boxes[max_x_idx],
            'distance': x_distances[max_x_idx]
        },
        'extreme_y': {
            'box': gt_boxes[max_y_idx],
            'distance': y_distances[max_y_idx]
        },
        'extreme_z': {
            'box': gt_boxes[max_z_idx],
            'distance': z_distances[max_z_idx]
        }
    }

def calculate_effective_range(gt_boxes):
    """GT 바운딩 박스를 기준으로 실제 사용 범위를 계산합니다."""
    if not gt_boxes:
        return None
    
    # 모든 박스의 최소/최대 좌표 계산
    all_x_min = [box['x_min'] for box in gt_boxes]
    all_x_max = [box['x_max'] for box in gt_boxes]
    all_y_min = [box['y_min'] for box in gt_boxes]
    all_y_max = [box['y_max'] for box in gt_boxes]
    all_z_min = [box['z_min'] for box in gt_boxes]
    all_z_max = [box['z_max'] for box in gt_boxes]
    
    effective_range = {
        'x_min': min(all_x_min), 'x_max': max(all_x_max),
        'y_min': min(all_y_min), 'y_max': max(all_y_max),
        'z_min': min(all_z_min), 'z_max': max(all_z_max)
    }
    
    effective_range['x_range'] = effective_range['x_max'] - effective_range['x_min']
    effective_range['y_range'] = effective_range['y_max'] - effective_range['y_min']
    effective_range['z_range'] = effective_range['z_max'] - effective_range['z_min']
    
    return effective_range

def analyze_gt_based_ranges(start_idx=None, end_idx=None):
    """GT 바운딩 박스를 기준으로 실제 사용 범위를 분석합니다."""
    frame_ids = load_frame_ids()
    total_frames = len(frame_ids)
    
    # 범위 설정
    start_idx = start_idx if start_idx is not None else 0
    end_idx = end_idx if end_idx is not None else total_frames
    
    # 범위 검증
    start_idx = max(0, start_idx)
    end_idx = min(total_frames, end_idx)
    
    if start_idx >= end_idx:
        print(f"Error: Invalid range. start_idx ({start_idx}) should be less than end_idx ({end_idx})")
        return
    
    # 저장할 데이터 수집
    analysis_data = []
    
    for i in range(start_idx, end_idx):
        frame_id = frame_ids[i]
        label_path = os.path.join(LABELS_FOLDER, f"{frame_id}.txt")
        
        # GT 바운딩 박스 로드
        gt_boxes = load_gt_boxes(label_path)
        if not gt_boxes:
            analysis_data.append({
                'frame_idx': i,
                'frame_id': frame_id,
                'num_objects': 0,
                'status': 'No GT boxes'
            })
            continue
        
        # 극단적인 박스들 찾기
        extreme_boxes = find_extreme_boxes(gt_boxes)
        effective_range = calculate_effective_range(gt_boxes)
        
        analysis_data.append({
            'frame_idx': i,
            'frame_id': frame_id,
            'num_objects': len(gt_boxes),
            'extreme_boxes': extreme_boxes,
            'effective_range': effective_range,
            'status': 'OK'
        })
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # TXT 파일로 저장
    txt_filename = os.path.join(OUTPUT_DIR, f"gt_based_point_cloud_ranges.txt")
    
    with open(txt_filename, 'w', encoding='utf-8') as txtfile:
        txtfile.write(f"GT-based Point Cloud Range Analysis Report\n")
        txtfile.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        txtfile.write(f"Frames: {start_idx} to {end_idx-1} (total: {end_idx - start_idx})\n")
        txtfile.write("=" * 90 + "\n\n")
        
        # 개별 프레임 정보
        valid_data = [data for data in analysis_data if data['status'] == 'OK']
        
        for data in valid_data:
            txtfile.write(f"Frame {data['frame_idx']:4d}: {data['frame_id']}\n")
            txtfile.write(f"  Total GT Objects: {data['num_objects']}\n")
            txtfile.write(f"\n")
            
            # 실제 사용 범위
            eff_range = data['effective_range']
            txtfile.write(f"  Effective Point Cloud Range (based on all GT boxes):\n")
            txtfile.write(f"    X: {eff_range['x_min']:7.2f}m ~ {eff_range['x_max']:7.2f}m (Range: {eff_range['x_range']:6.2f}m)\n")
            txtfile.write(f"    Y: {eff_range['y_min']:7.2f}m ~ {eff_range['y_max']:7.2f}m (Range: {eff_range['y_range']:6.2f}m)\n")
            txtfile.write(f"    Z: {eff_range['z_min']:7.2f}m ~ {eff_range['z_max']:7.2f}m (Range: {eff_range['z_range']:6.2f}m)\n")
            txtfile.write(f"\n")
            
            # 각 축에서 가장 극단적인 박스들
            extreme = data['extreme_boxes']
            
            # X축 극단 박스
            x_box = extreme['extreme_x']['box']
            txtfile.write(f"  Most Extreme Box in X-axis:\n")
            txtfile.write(f"    Class: {x_box['class']}\n")
            txtfile.write(f"    Center: ({x_box['center'][0]:6.2f}, {x_box['center'][1]:6.2f}, {x_box['center'][2]:6.2f})\n")
            txtfile.write(f"    Size: L={x_box['size'][0]:5.2f}m, W={x_box['size'][1]:5.2f}m, H={x_box['size'][2]:5.2f}m\n")
            txtfile.write(f"    X-range: {x_box['x_min']:6.2f}m ~ {x_box['x_max']:6.2f}m\n")
            txtfile.write(f"    Max distance from X-origin: {extreme['extreme_x']['distance']:.2f}m\n")
            txtfile.write(f"\n")
            
            # Y축 극단 박스
            y_box = extreme['extreme_y']['box']
            txtfile.write(f"  Most Extreme Box in Y-axis:\n")
            txtfile.write(f"    Class: {y_box['class']}\n")
            txtfile.write(f"    Center: ({y_box['center'][0]:6.2f}, {y_box['center'][1]:6.2f}, {y_box['center'][2]:6.2f})\n")
            txtfile.write(f"    Size: L={y_box['size'][0]:5.2f}m, W={y_box['size'][1]:5.2f}m, H={y_box['size'][2]:5.2f}m\n")
            txtfile.write(f"    Y-range: {y_box['y_min']:6.2f}m ~ {y_box['y_max']:6.2f}m\n")
            txtfile.write(f"    Max distance from Y-origin: {extreme['extreme_y']['distance']:.2f}m\n")
            txtfile.write(f"\n")
            
            # Z축 극단 박스
            z_box = extreme['extreme_z']['box']
            txtfile.write(f"  Most Extreme Box in Z-axis:\n")
            txtfile.write(f"    Class: {z_box['class']}\n")
            txtfile.write(f"    Center: ({z_box['center'][0]:6.2f}, {z_box['center'][1]:6.2f}, {z_box['center'][2]:6.2f})\n")
            txtfile.write(f"    Size: L={z_box['size'][0]:5.2f}m, W={z_box['size'][1]:5.2f}m, H={z_box['size'][2]:5.2f}m\n")
            txtfile.write(f"    Z-range: {z_box['z_min']:6.2f}m ~ {z_box['z_max']:6.2f}m\n")
            txtfile.write(f"    Max distance from Z-origin: {extreme['extreme_z']['distance']:.2f}m\n")
            
            txtfile.write("=" * 90 + "\n\n")
    
    return txt_filename

def parse_arguments():
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(description='Analyze GT-based Point Cloud effective ranges')
    parser.add_argument('--start', type=int, default=None, 
                       help='Start frame index (default: 0)')
    parser.add_argument('--end', type=int, default=None, 
                       help='End frame index (default: total frames)')
    parser.add_argument('--data_path', type=str, default=DATA_PATH,
                       help='Path to custom_av dataset')
    parser.add_argument('--split', type=str, default='train_origin_old', 
                       choices=['train', 'train_origin', 'train_origin_old', 'val', 'test'],
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
        return
    
    if not os.path.exists(LABELS_FOLDER):
        return
    
    # GT 기반 범위 분석
    analyze_gt_based_ranges(args.start, args.end)

if __name__ == "__main__":
    main()
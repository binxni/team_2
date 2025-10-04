import numpy as np
import os
import argparse
import csv
from datetime import datetime

# Configuration
DATA_PATH = "/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av"
FRAME_LIST_FILE = os.path.join(DATA_PATH, "ImageSets", "test.txt")
POINTS_FOLDER = os.path.join(DATA_PATH, "points")
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

def calculate_point_cloud_range(points):
    """포인트 클라우드의 XYZ 범위를 계산합니다."""
    if points is None or len(points) == 0:
        return None
    
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    z_coords = points[:, 2]
    
    # 각 축의 min, max 값 계산
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    
    # 각 축의 범위(미터) 계산
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    return {
        'x_min': x_min, 'x_max': x_max, 'x_range': x_range,
        'y_min': y_min, 'y_max': y_max, 'y_range': y_range,
        'z_min': z_min, 'z_max': z_max, 'z_range': z_range,
        'total_points': len(points)
    }

def analyze_point_cloud_ranges(start_idx=None, end_idx=None):
    """지정된 범위의 프레임들에 대해 포인트 클라우드 범위를 분석하고 저장합니다."""
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
    
    print(f"Analyzing Point Cloud Ranges - Frames {start_idx} to {end_idx-1} (total: {end_idx - start_idx} frames)")
    print(f"Total available frames: {total_frames}")
    print("Calculating X/Y/Z ranges for each frame...")
    
    # 저장할 데이터 수집
    csv_data = []
    
    for i in range(start_idx, end_idx):
        frame_id = frame_ids[i]
        pc_path = os.path.join(POINTS_FOLDER, f"{frame_id}.npy")
        
        # 포인트 클라우드 로드
        points = load_npy_pointcloud(pc_path)
        if points is None:
            csv_data.append([i, frame_id, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "File not found"])
            continue
        
        # 범위 계산
        range_info = calculate_point_cloud_range(points)
        if range_info is None:
            csv_data.append([i, frame_id, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "Empty point cloud"])
            continue
        
        # CSV 데이터에 추가
        csv_data.append([
            i,                          # Frame_Index
            frame_id,                   # Frame_ID
            range_info['total_points'], # Total_Points
            range_info['x_min'],        # X_Min
            range_info['x_max'],        # X_Max
            range_info['x_range'],      # X_Range_m
            range_info['y_min'],        # Y_Min
            range_info['y_max'],        # Y_Max
            range_info['y_range'],      # Y_Range_m
            range_info['z_min'],        # Z_Min
            range_info['z_max'],        # Z_Max
            range_info['z_range'],      # Z_Range_m
            "OK"                        # Status
        ])
        
        # 진행상황 출력
        if (i - start_idx + 1) % 100 == 0 or i == end_idx - 1:
            print(f"  Processed {i - start_idx + 1}/{end_idx - start_idx} frames...")
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # TXT 파일로 저장 (사람이 읽기 쉬운 형태)
    txt_filename = os.path.join(OUTPUT_DIR, f"train_point_cloud_ranges.txt")
    
    with open(txt_filename, 'w', encoding='utf-8') as txtfile:
        txtfile.write(f"Point Cloud Range Analysis Report\n")
        txtfile.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        txtfile.write(f"Frames: {start_idx} to {end_idx-1} (total: {end_idx - start_idx})\n")
        txtfile.write("=" * 80 + "\n\n")
        
        # 개별 프레임 정보 (모든 유효한 프레임 출력)
        valid_data = [row for row in csv_data if row[12] == "OK"]
        for row in valid_data:
            txtfile.write(f"Frame {row[0]:4d}: {row[1]}\n")
            txtfile.write(f"  Total Points: {row[2]:,}\n")
            txtfile.write(f"  X-axis: {row[3]:7.2f}m ~ {row[4]:7.2f}m (Range: {row[5]:6.2f}m)\n")
            txtfile.write(f"  Y-axis: {row[6]:7.2f}m ~ {row[7]:7.2f}m (Range: {row[8]:6.2f}m)\n")
            txtfile.write(f"  Z-axis: {row[9]:7.2f}m ~ {row[10]:7.2f}m (Range: {row[11]:6.2f}m)\n")
            txtfile.write("-" * 60 + "\n")
    
    print(f"Results saved to TXT file: {txt_filename}")
    
    return txt_filename

def parse_arguments():
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(description='Analyze Point Cloud XYZ ranges for each frame')
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
    global DATA_PATH, FRAME_LIST_FILE, POINTS_FOLDER
    DATA_PATH = args.data_path
    FRAME_LIST_FILE = os.path.join(DATA_PATH, "ImageSets", f"{args.split}.txt")
    POINTS_FOLDER = os.path.join(DATA_PATH, "points")
    
    if not os.path.exists(FRAME_LIST_FILE):
        print(f"Error: Frame list file not found: {FRAME_LIST_FILE}")
        return
    
    # 포인트 클라우드 범위 분석
    analyze_point_cloud_ranges(args.start, args.end)

if __name__ == "__main__":
    main()
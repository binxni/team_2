import numpy as np
import os
import argparse
import csv
from datetime import datetime

# Configuration
DATA_PATH = "/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av"
FRAME_LIST_FILE = os.path.join(DATA_PATH, "ImageSets", "train_origin.txt")
POINTS_FOLDER = os.path.join(DATA_PATH, "points")
OUTPUT_DIR = "/home/ailab/git/Team_2/Seokjae/OpenPCDet/dataset_analysis"

def load_frame_ids():
    """프레임 ID 목록을 로드합니다."""
    with open(FRAME_LIST_FILE, 'r') as f:
        frame_ids = [line.strip() for line in f.readlines()]
    return frame_ids

def load_npy_pointcloud(file_path):
    """포인트 클라우드를 로드하고 포인트 개수를 반환합니다."""
    if not os.path.exists(file_path):
        return None
    points = np.load(file_path)
    return points

def check_points_height_near_origin(start_idx=None, end_idx=None):
    """지정된 범위의 프레임들에 대해 라이다 원점 반경 1m 이내 포인트들의 평균 높이를 확인하고 CSV로 저장합니다."""
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
    
    print(f"Analyzing train_origin.txt - Frames {start_idx} to {end_idx-1} (total: {end_idx - start_idx} frames)")
    print(f"Total available frames: {total_frames}")
    print("Calculating average height of points within 1m radius from LiDAR origin...")
    
    # CSV 저장을 위한 데이터 수집
    csv_data = []
    
    for i in range(start_idx, end_idx):
        frame_id = frame_ids[i]
        pc_path = os.path.join(POINTS_FOLDER, f"{frame_id}.npy")
        
        # 포인트 클라우드 로드
        points = load_npy_pointcloud(pc_path)
        if points is None:
            csv_data.append([i, frame_id, 0, 0, "File not found"])
            continue
        
        # 라이다 원점(0,0)으로부터 반경 1m 이내의 포인트 찾기
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        z_coords = points[:, 2]
        
        # 원점으로부터의 거리 계산 (XY 평면에서)
        distances = np.sqrt(x_coords**2 + y_coords**2)
        
        # 반경 1m 이내의 포인트 필터링
        within_1m_mask = distances <= 1.5
        points_within_1m = points[within_1m_mask]
        
        if len(points_within_1m) > 0:
            # 반경 1m 이내 포인트들의 평균 높이(Z값) 계산
            avg_height = np.mean(points_within_1m[:, 2])
            count_within_1m = len(points_within_1m)
            csv_data.append([i, frame_id, count_within_1m, avg_height, "OK"])
        else:
            # 반경 1m 이내에 포인트가 없는 경우
            csv_data.append([i, frame_id, 0, 0.0, "No points within 1m"])
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # CSV 파일 저장
    csv_filename = os.path.join(OUTPUT_DIR, f"train_point_height_1m_radius.txt")
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 헤더 작성
        writer.writerow(['Frame_Index', 'Frame_ID', 'Points_Within_1m', 'Avg_Height_m', 'Status'])
        
        # 데이터 작성
        for row in csv_data:
            writer.writerow(row)
    
    print(f"Results saved to CSV file: {csv_filename}")
    
    # 간단한 통계 출력
    valid_data = [row for row in csv_data if row[4] == "OK"]
    if valid_data:
        heights = [row[3] for row in valid_data if row[2] > 0]  # 포인트가 있는 경우만
        point_counts = [row[2] for row in valid_data]
        
        if heights:
            avg_height = np.mean(heights)
            min_height = min(heights)
            max_height = max(heights)
            std_height = np.std(heights)
            
            print(f"Summary:")
            print(f"  Valid frames: {len(valid_data)} / {len(csv_data)}")
            print(f"  Frames with points within 1m: {len(heights)} / {len(valid_data)}")
            print(f"  Average height within 1m radius: {avg_height:.3f}m")
            print(f"  Height range: {min_height:.3f}m ~ {max_height:.3f}m")
            print(f"  Height std deviation: {std_height:.3f}m")
            print(f"  Average points within 1m: {np.mean(point_counts):.1f}")
        else:
            print(f"No points found within 1m radius in any frame!")
    
    return csv_filename

def parse_arguments():
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(description='Analyze average height of points within 1m radius from LiDAR origin')
    parser.add_argument('--start', type=int, default=None, 
                       help='Start frame index (default: 0)')
    parser.add_argument('--end', type=int, default=None, 
                       help='End frame index (default: total frames)')
    parser.add_argument('--data_path', type=str, default=DATA_PATH,
                       help='Path to custom_av dataset')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # 전역 변수 업데이트
    global DATA_PATH, FRAME_LIST_FILE, POINTS_FOLDER
    DATA_PATH = args.data_path
    FRAME_LIST_FILE = os.path.join(DATA_PATH, "ImageSets", "train_origin.txt")
    POINTS_FOLDER = os.path.join(DATA_PATH, "points")
    
    if not os.path.exists(FRAME_LIST_FILE):
        print(f"Error: Frame list file not found: {FRAME_LIST_FILE}")
        return
    
    # 반경 1m 이내 포인트들의 평균 높이 확인 및 CSV 저장
    check_points_height_near_origin(args.start, args.end)

if __name__ == "__main__":
    main()
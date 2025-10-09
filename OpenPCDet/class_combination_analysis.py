import numpy as np
import os
import argparse
from datetime import datetime
from collections import defaultdict, Counter

# Configuration
DATA_PATH = "/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av_64"
FRAME_LIST_FILE = os.path.join(DATA_PATH, "ImageSets", "train_128.txt")
LABELS_FOLDER = os.path.join(DATA_PATH, "labels")
OUTPUT_DIR = "/home/ailab/git/Team_2/Seokjae/OpenPCDet/dataset_analysis"  

# Target classes for analysis
TARGET_CLASSES = ["Vehicle", "Pedestrian", "Cyclist"]

def load_frame_ids():
    """프레임 ID 목록을 로드합니다."""
    with open(FRAME_LIST_FILE, 'r') as f:
        frame_ids = [line.strip() for line in f.readlines()]
    return frame_ids

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
                    class_name = parts[7]
                    if class_name in TARGET_CLASSES:
                        labels.append(class_name)
    return labels

def get_class_combination(class_list):
    """클래스 리스트에서 조합을 생성합니다."""
    if not class_list:
        return "None"
    
    # 클래스별 개수 계산
    class_counts = Counter(class_list)
    
    # 존재하는 클래스만 추출 (개수 상관없이)
    present_classes = sorted(list(class_counts.keys()))
    
    # 조합 문자열 생성
    combination = "+".join(present_classes)
    
    return combination, class_counts

def analyze_class_combinations(start_idx=None, end_idx=None):
    """클래스 조합 분석 메인 함수"""
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
    print(f"\n🚀 Starting class combination analysis...")
    print(f"📊 Processing frames {start_idx} to {end_idx-1} ({target_frame_count:,} total frames)")
    print(f"📁 Data path: {DATA_PATH}")
    print(f"🎯 Target classes: {TARGET_CLASSES}")
    print(f"⚠️  Note: Frames with only 'Vehicle' will be excluded")
    print("-" * 80)
    
    # 결과 저장용 딕셔너리
    combination_data = defaultdict(lambda: {
        'frame_indices': [],
        'frame_ids': [],
        'class_totals': {cls: 0 for cls in TARGET_CLASSES}
    })
    
    excluded_frames = []  # Vehicle만 있는 프레임들
    start_time = time.time()
    
    for idx, i in enumerate(range(start_idx, end_idx)):
        frame_id = frame_ids[i]
        label_path = os.path.join(LABELS_FOLDER, f"{frame_id}.txt")
        
        # 진행률 출력
        progress = (idx + 1) / target_frame_count
        elapsed_time = time.time() - start_time
        
        if idx % max(1, target_frame_count // 50) == 0 or idx == target_frame_count - 1:  # 2% 간격으로 출력
            eta = elapsed_time / (idx + 1) * (target_frame_count - idx - 1) if idx > 0 else 0
            print(f"\r⏳ Progress: {progress*100:5.1f}% [{idx+1:,}/{target_frame_count:,}] "
                  f"| Frame: {frame_id} | Elapsed: {elapsed_time:.1f}s | ETA: {eta:.1f}s", end="", flush=True)
        
        # GT 라벨 로드
        class_list = load_gt_labels(label_path)
        
        if not class_list:
            continue
        
        # 클래스 조합 및 개수 계산
        combination, class_counts = get_class_combination(class_list)
        
        # Vehicle만 있는 경우 제외
        if combination == "Vehicle":
            excluded_frames.append({
                'frame_idx': i,
                'frame_id': frame_id,
                'vehicle_count': class_counts['Vehicle']
            })
            continue
        
        # 조합 데이터에 추가
        combination_data[combination]['frame_indices'].append(i)
        combination_data[combination]['frame_ids'].append(frame_id)
        
        # 클래스별 총 개수 누적
        for cls in TARGET_CLASSES:
            if cls in class_counts:
                combination_data[combination]['class_totals'][cls] += class_counts[cls]
    
    print()  # 새 줄
    total_time = time.time() - start_time
    
    print(f"\n✅ Analysis completed successfully!")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"🚫 Excluded frames (Vehicle only): {len(excluded_frames)}")
    print(f"📊 Valid combinations found: {len(combination_data)}")
    
    # 결과 저장
    print(f"\n💾 Saving results to file...")
    output_file = save_combination_results(combination_data, excluded_frames, start_idx, end_idx)
    print(f"📄 Results saved to: {output_file}")
    
    # 간단한 통계 출력
    print(f"\n📊 Quick Summary:")
    for combination in sorted(combination_data.keys()):
        data = combination_data[combination]
        frame_count = len(data['frame_indices'])
        print(f"   {combination}: {frame_count} frames")
    
    return combination_data, excluded_frames

def save_combination_results(combination_data, excluded_frames, start_idx, end_idx):
    """조합 분석 결과를 파일로 저장"""
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 파일명 생성
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(OUTPUT_DIR, f"class_combination_analysis_128.txt")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Class Combination Analysis Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Frames: {start_idx} to {end_idx-1} (total: {end_idx - start_idx})\n")
        f.write(f"Target Classes: {TARGET_CLASSES}\n")
        f.write(f"Note: Frames with only 'Vehicle' are excluded\n")
        f.write("=" * 100 + "\n\n")
        
        # 1. 조합별 상세 정보
        f.write("1. COMBINATION DETAILS\n")
        f.write("=" * 100 + "\n")
        
        for combination in sorted(combination_data.keys()):
            data = combination_data[combination]
            frame_count = len(data['frame_indices'])
            
            f.write(f"\nCombination: {combination}\n")
            f.write(f"Frame Count: {frame_count}\n")
            
            # 클래스별 총 개수
            f.write(f"Class Totals:\n")
            for cls in TARGET_CLASSES:
                total = data['class_totals'][cls]
                if total > 0:
                    avg = total / frame_count
                    f.write(f"  {cls}: {total} total ({avg:.1f} avg per frame)\n")
            
            # 프레임 인덱스 (처음 20개만 표시)
            f.write(f"Frame Indices: ")
            if frame_count <= 20:
                f.write(f"{data['frame_indices']}\n")
            else:
                f.write(f"{data['frame_indices'][:20]} ... (and {frame_count-20} more)\n")
            
            f.write("-" * 80 + "\n")
        
        # 2. 조합별 요약 테이블
        f.write(f"\n2. COMBINATION SUMMARY TABLE\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'Combination':<25} {'Frames':<8} {'Vehicle':<8} {'Pedestrian':<12} {'Cyclist':<8}\n")
        f.write("-" * 80 + "\n")
        
        for combination in sorted(combination_data.keys()):
            data = combination_data[combination]
            frame_count = len(data['frame_indices'])
            vehicle_total = data['class_totals']['Vehicle']
            pedestrian_total = data['class_totals']['Pedestrian']
            cyclist_total = data['class_totals']['Cyclist']
            
            f.write(f"{combination:<25} {frame_count:<8} {vehicle_total:<8} {pedestrian_total:<12} {cyclist_total:<8}\n")
        
        # 3. 제외된 프레임 정보
        f.write(f"\n3. EXCLUDED FRAMES (Vehicle Only)\n")
        f.write("=" * 100 + "\n")
        f.write(f"Total excluded frames: {len(excluded_frames)}\n")
        
        if excluded_frames:
            f.write(f"Vehicle count distribution in excluded frames:\n")
            vehicle_counts = [frame['vehicle_count'] for frame in excluded_frames]
            unique_counts = sorted(set(vehicle_counts))
            for count in unique_counts:
                count_freq = vehicle_counts.count(count)
                f.write(f"  {count} vehicles: {count_freq} frames\n")
            
            # 처음 50개 제외된 프레임 인덱스
            f.write(f"\nExcluded frame indices (first 50): ")
            excluded_indices = [frame['frame_idx'] for frame in excluded_frames[:50]]
            f.write(f"{excluded_indices}")
            if len(excluded_frames) > 50:
                f.write(f" ... (and {len(excluded_frames)-50} more)")
            f.write(f"\n")
        
        # 4. 전체 통계
        f.write(f"\n4. OVERALL STATISTICS\n")
        f.write("=" * 100 + "\n")
        
        total_valid_frames = sum(len(data['frame_indices']) for data in combination_data.values())
        total_analyzed_frames = total_valid_frames + len(excluded_frames)
        
        f.write(f"Total analyzed frames: {total_analyzed_frames}\n")
        f.write(f"Valid combination frames: {total_valid_frames}\n")
        f.write(f"Excluded frames (Vehicle only): {len(excluded_frames)}\n")
        f.write(f"Valid frame percentage: {total_valid_frames/total_analyzed_frames*100:.1f}%\n")
        
        # 각 클래스별 전체 통계
        f.write(f"\nClass totals across all valid combinations:\n")
        overall_totals = {cls: 0 for cls in TARGET_CLASSES}
        for data in combination_data.values():
            for cls in TARGET_CLASSES:
                overall_totals[cls] += data['class_totals'][cls]
        
        for cls in TARGET_CLASSES:
            total = overall_totals[cls]
            if total_valid_frames > 0:
                avg = total / total_valid_frames
                f.write(f"  {cls}: {total} total ({avg:.1f} avg per valid frame)\n")
        
        # 5. 차량만 있는 프레임을 제외한 모든 프레임의 전체 frame_id 목록
        f.write(f"\n5. NON-VEHICLE-ONLY FRAMES COMPLETE LIST\n")
        f.write("=" * 100 + "\n")
        
        # 모든 유효한 조합의 frame_id들을 수집
        all_valid_frame_ids = []
        for data in combination_data.values():
            all_valid_frame_ids.extend(data['frame_ids'])
        
        if all_valid_frame_ids:
            f.write(f"Total frames (excluding Vehicle-only): {len(all_valid_frame_ids)}\n\n")
            
            # 모든 차량만 있는 프레임을 제외한 프레임 ID를 한 줄에 1개씩 출력
            f.write(f"Complete frame ID list (one per line):\n")
            
            for frame_id in sorted(all_valid_frame_ids):
                f.write(f"{frame_id}\n")
        else:
            f.write(f"No valid frames found (all frames are Vehicle-only).\n")
    
    return filename

def parse_arguments():
    """명령줄 인수를 파싱합니다."""
    parser = argparse.ArgumentParser(description='Analyze class combinations in frames (excluding Vehicle-only frames)')
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
    global DATA_PATH, FRAME_LIST_FILE, LABELS_FOLDER
    DATA_PATH = args.data_path
    FRAME_LIST_FILE = os.path.join(DATA_PATH, "ImageSets", f"{args.split}.txt")
    LABELS_FOLDER = os.path.join(DATA_PATH, "labels")
    
    if not os.path.exists(FRAME_LIST_FILE):
        print(f"Frame list file not found: {FRAME_LIST_FILE}")
        return
    
    # 클래스 조합 분석 실행
    analyze_class_combinations(args.start, args.end)

if __name__ == "__main__":
    main()
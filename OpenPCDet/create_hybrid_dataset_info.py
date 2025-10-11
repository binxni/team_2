import os
import shutil
import pickle
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def detect_weather_from_filename(filename):
    """파일명에서 날씨 조건 추정"""
    filename_lower = filename.lower()
    
    if 'rain' in filename_lower:
        return 'rain'
    elif 'fog' in filename_lower:
        return 'fog'
    elif 'snow' in filename_lower:
        return 'snow'
    elif 'storm' in filename_lower:
        return 'storm'
    elif 'wet' in filename_lower:
        return 'rain'
    elif 'mist' in filename_lower:
        return 'fog'
    else:
        # 기본값: rain (points_lias가 주로 adverse weather)
        return 'rain'

def parse_label_file(label_path):
    """라벨 파일에서 객체 정보 추출 (KITTI 포맷)"""
    if not label_path.exists():
        return {
            'num_objects': 0,
            'objects': [],
            'gt_boxes': np.zeros((0, 7), dtype=np.float32),
            'gt_names': np.array([])
        }
    
    objects = []
    gt_boxes = []
    gt_names = []
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 15:  # KITTI 포맷 확인
                class_name = parts[0]
                
                # 3D bounding box 정보 (KITTI 포맷)
                h, w, l = float(parts[8]), float(parts[9]), float(parts[10])  # height, width, length
                x, y, z = float(parts[11]), float(parts[12]), float(parts[13])  # location
                rotation_y = float(parts[14])
                
                # OpenPCDet 포맷으로 변환: [x, y, z, dx, dy, dz, heading]
                gt_box = [x, y, z, l, w, h, rotation_y]
                
                obj_info = {
                    'class': class_name,
                    'truncated': float(parts[1]),
                    'occluded': int(parts[2]),
                    'alpha': float(parts[3]),
                    'bbox_2d': [float(x) for x in parts[4:8]],  # 2D bbox
                    'dimensions': [h, w, l],  # height, width, length
                    'location': [x, y, z],
                    'rotation_y': rotation_y,
                    'gt_box_3d': gt_box
                }
                
                objects.append(obj_info)
                gt_boxes.append(gt_box)
                gt_names.append(class_name)
    
    return {
        'num_objects': len(objects),
        'objects': objects,
        'gt_boxes': np.array(gt_boxes, dtype=np.float32),
        'gt_names': np.array(gt_names)
    }

def get_point_cloud_info(points_file):
    """포인트 클라우드 파일 정보 추출"""
    try:
        points = np.fromfile(points_file, dtype=np.float32).reshape(-1, 4)
        return {
            'num_points': len(points),
            'point_cloud_range': {
                'x_min': float(np.min(points[:, 0])),
                'x_max': float(np.max(points[:, 0])),
                'y_min': float(np.min(points[:, 1])),
                'y_max': float(np.max(points[:, 1])),
                'z_min': float(np.min(points[:, 2])),
                'z_max': float(np.max(points[:, 2]))
            }
        }
    except:
        return {'num_points': 0, 'point_cloud_range': {}}

def create_hybrid_dataset_with_weather(source_dir, target_dir, copy_files=True):
    """Weather 정보를 포함한 hybrid 데이터셋 생성
    
    Args:
        source_dir: custom_av_64 경로
        target_dir: 새로 생성할 custom_av_hybrid_v2 경로  
        copy_files: 파일을 실제로 복사할지 여부 (False면 info만 생성)
    """
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    print(f"Source directory: {source_path}")
    print(f"Target directory: {target_path}")
    
    # 소스 디렉토리 확인
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_path}")
    
    # 타겟 디렉토리 생성
    if copy_files:
        (target_path / 'points').mkdir(parents=True, exist_ok=True)
        (target_path / 'labels').mkdir(parents=True, exist_ok=True)
    else:
        target_path.mkdir(parents=True, exist_ok=True)
    
    infos = []
    frame_id = 0
    
    print("=" * 50)
    print("Processing Clean Data (points/)")
    print("=" * 50)
    
    # Clean data 처리
    clean_points_dir = source_path / 'points'
    clean_labels_dir = source_path / 'labels'
    
    if clean_points_dir.exists():
        clean_files = sorted(clean_points_dir.glob('*.bin'))
        print(f"Found {len(clean_files)} clean point cloud files")
        
        for point_file in tqdm(clean_files, desc="Processing clean data"):
            # 파일 복사 (옵션)
            if copy_files:
                target_point_file = target_path / 'points' / f'{frame_id:06d}.bin'
                shutil.copy2(point_file, target_point_file)
            else:
                target_point_file = target_path / 'points' / f'{frame_id:06d}.bin'
            
            # 라벨 파일 처리
            label_file = clean_labels_dir / point_file.name.replace('.bin', '.txt')
            target_label_file = target_path / 'labels' / f'{frame_id:06d}.txt'
            
            if copy_files and label_file.exists():
                shutil.copy2(label_file, target_label_file)
            
            # 포인트 클라우드 정보 추출
            pc_info = get_point_cloud_info(point_file)
            
            # Info 생성 (Weather 정보 포함)
            info = {
                'frame_id': frame_id,
                'point_cloud_idx': frame_id,
                'lidar_path': str(target_point_file.relative_to(target_path)),
                
                # Weather 정보
                'weather': 'clear',
                'weather_type': 'clean',
                'weather_condition': 'sunny',
                
                # 원본 파일 정보
                'original_file': str(point_file),
                'original_source': 'points',
                
                # 포인트 클라우드 정보
                'num_points_of_each_lidar': [pc_info['num_points']],
                **pc_info
            }
            
            # 라벨 정보 추가
            if label_file.exists():
                info['label_path'] = str(target_label_file.relative_to(target_path))
                label_info = parse_label_file(label_file)
                info.update(label_info)
            else:
                # 라벨이 없는 경우 기본값
                info.update({
                    'num_objects': 0,
                    'objects': [],
                    'gt_boxes': np.zeros((0, 7), dtype=np.float32),
                    'gt_names': np.array([])
                })
            
            infos.append(info)
            frame_id += 1
    else:
        print(f"Warning: Clean points directory not found: {clean_points_dir}")
    
    print("\n" + "=" * 50)
    print("Processing Adverse Data (points_lias/)")
    print("=" * 50)
    
    # Adverse data 처리
    adverse_points_dir = source_path / 'points_lias'
    adverse_labels_dir = source_path / 'labels_lias'
    
    if adverse_points_dir.exists():
        adverse_files = sorted(adverse_points_dir.glob('*.bin'))
        print(f"Found {len(adverse_files)} adverse point cloud files")
        
        for point_file in tqdm(adverse_files, desc="Processing adverse data"):
            # 파일 복사 (옵션)
            if copy_files:
                target_point_file = target_path / 'points' / f'{frame_id:06d}.bin'
                shutil.copy2(point_file, target_point_file)
            else:
                target_point_file = target_path / 'points' / f'{frame_id:06d}.bin'
            
            # Weather 타입 추정 (파일명 기반)
            weather_type = detect_weather_from_filename(point_file.name)
            
            # 라벨 파일 처리 (adverse용이 있으면 사용, 없으면 clean용 사용)
            label_file = None
            target_label_file = target_path / 'labels' / f'{frame_id:06d}.txt'
            
            # 1. adverse 라벨 디렉토리 확인
            if adverse_labels_dir.exists():
                adverse_label_file = adverse_labels_dir / point_file.name.replace('.bin', '.txt')
                if adverse_label_file.exists():
                    label_file = adverse_label_file
            
            # 2. adverse 라벨이 없으면 clean 라벨 사용
            if label_file is None:
                clean_label_file = clean_labels_dir / point_file.name.replace('.bin', '.txt')
                if clean_label_file.exists():
                    label_file = clean_label_file
            
            # 라벨 파일 복사
            if copy_files and label_file is not None:
                shutil.copy2(label_file, target_label_file)
            
            # 포인트 클라우드 정보 추출
            pc_info = get_point_cloud_info(point_file)
            
            # Info 생성 (Weather 정보 포함)
            info = {
                'frame_id': frame_id,
                'point_cloud_idx': frame_id,
                'lidar_path': str(target_point_file.relative_to(target_path)),
                
                # Weather 정보
                'weather': weather_type,
                'weather_type': 'adverse',
                'weather_condition': weather_type,
                
                # 원본 파일 정보
                'original_file': str(point_file),
                'original_source': 'points_lias',
                
                # 포인트 클라우드 정보
                'num_points_of_each_lidar': [pc_info['num_points']],
                **pc_info
            }
            
            # 라벨 정보 추가
            if label_file is not None:
                info['label_path'] = str(target_label_file.relative_to(target_path))
                label_info = parse_label_file(label_file)
                info.update(label_info)
            else:
                # 라벨이 없는 경우 기본값
                info.update({
                    'num_objects': 0,
                    'objects': [],
                    'gt_boxes': np.zeros((0, 7), dtype=np.float32),
                    'gt_names': np.array([])
                })
            
            infos.append(info)
            frame_id += 1
    else:
        print(f"Warning: Adverse points directory not found: {adverse_points_dir}")
    
    print("\n" + "=" * 50)
    print("Saving dataset info...")
    print("=" * 50)
    
    # Info 파일들 저장
    info_files = ['infos_train.pkl', 'infos_val.pkl', 'infos_test.pkl']
    
    for info_file in info_files:
        info_path = target_path / info_file
        with open(info_path, 'wb') as f:
            pickle.dump(infos, f)
        print(f"Saved: {info_path}")
    
    # 통계 정보 출력
    clean_count = sum(1 for info in infos if info['weather_type'] == 'clean')
    adverse_count = sum(1 for info in infos if info['weather_type'] == 'adverse')
    
    weather_stats = {}
    for info in infos:
        weather = info['weather']
        weather_stats[weather] = weather_stats.get(weather, 0) + 1
    
    print("\n" + "=" * 50)
    print("Dataset Creation Complete!")
    print("=" * 50)
    print(f"Total samples: {len(infos)}")
    print(f"Clean samples: {clean_count}")
    print(f"Adverse samples: {adverse_count}")
    print(f"Clean ratio: {clean_count/len(infos)*100:.1f}%")
    print(f"Adverse ratio: {adverse_count/len(infos)*100:.1f}%")
    print("\nWeather breakdown:")
    for weather, count in weather_stats.items():
        print(f"  {weather}: {count} ({count/len(infos)*100:.1f}%)")
    
    print(f"\nDataset saved to: {target_path}")
    if copy_files:
        print("Files were copied to target directory")
    else:
        print("Only info files were created (no file copying)")
    
    return infos

def main():
    parser = argparse.ArgumentParser(description='Create hybrid dataset with weather information')
    parser.add_argument('--source_dir', type=str, required=True,
                       help='Path to custom_av_64 directory')
    parser.add_argument('--target_dir', type=str, required=True,
                       help='Path to output hybrid dataset directory')
    parser.add_argument('--copy_files', action='store_true',
                       help='Copy point cloud and label files (default: False)')
    parser.add_argument('--info_only', action='store_true',
                       help='Create info files only without copying data files')
    
    args = parser.parse_args()
    
    # copy_files 옵션 설정
    copy_files = args.copy_files and not args.info_only
    
    try:
        infos = create_hybrid_dataset_with_weather(
            source_dir=args.source_dir,
            target_dir=args.target_dir,
            copy_files=copy_files
        )
        print("\n✅ Dataset creation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during dataset creation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # 기본 실행 (스크립트에서 직접 실행하는 경우)
    if len(os.sys.argv) == 1:
        # 기본 경로 설정 (실제 경로로 수정 필요)
        source_dir = '/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av_64'
        target_dir = '/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av_hybrid_v2'
        
        print("Running with default paths:")
        print(f"Source: {source_dir}")
        print(f"Target: {target_dir}")
        print("Copy files: True")
        
        create_hybrid_dataset_with_weather(
            source_dir=source_dir,
            target_dir=target_dir,
            copy_files=True
        )
    else:
        main()

import numpy as np
import os
import glob
from tqdm import tqdm
import math

def cartesian_to_spherical(points):
    """
    Cartesian coordinates (x, y, z)를 Spherical coordinates (r, theta, phi)로 변환
    - r: 거리
    - theta: 방위각 (azimuth, -pi ~ pi)
    - phi: 고도각 (elevation, -pi/2 ~ pi/2)
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)  # 방위각
    phi = np.arcsin(z / (r + 1e-8))  # 고도각 (elevation)
    
    return r, theta, phi

def get_128ch_channel_mapping():
    """
    128ch 라이다의 수직 각도 매핑 생성
    """
    channels = []
    
    # -25° ~ -24° : 1° 간격 (2개 채널)
    channels.extend(np.linspace(-25, -24, 2))
    
    # -24° ~ -8° : 0.5° 간격 (32개 채널)
    channels.extend(np.linspace(-24, -8, 32))
    
    # -8° ~ -6° : 0.125° 간격 (16개 채널)
    channels.extend(np.linspace(-8, -6, 16))
    
    # -6° ~ +2° : 0.125° 간격 (64개 채널)
    channels.extend(np.linspace(-6, 2, 64))
    
    # +2° ~ +14° : 0.5° 간격 (24개 채널)
    channels.extend(np.linspace(2, 14, 24))
    
    # +14° ~ +15° : 1° 간격 (2개 채널)
    channels.extend(np.linspace(14, 15, 2))
    
    return np.array(sorted(channels))

def get_64ch_channel_mapping():
    """
    64ch 라이다의 수직 각도 매핑 생성
    """
    channels = []
    
    # -25° ~ -18° : 5° 간격 (2개 채널)
    channels.extend([-25, -20])
    
    # -14° ~ -8° : 1° 간격 (7개 채널)
    channels.extend(np.linspace(-14, -8, 7))
    
    # -6° ~ +2° : 0.167° 간격 (48개 채널)
    channels.extend(np.linspace(-6, 2, 48))
    
    # +2° ~ +3° : 1° 간격 (2개 채널)
    channels.extend([2, 3])
    
    # +3° ~ +5° : 2° 간격 (2개 채널)
    channels.extend([3, 5])
    
    # +5° ~ +11° : 3° 간격 (3개 채널)
    channels.extend([5, 8, 11])
    
    # +11° ~ +15° : 4° 간격 (2개 채널)
    channels.extend([11, 15])
    
    return np.array(sorted(channels))

def find_closest_channels(channel_128, channel_64):
    """
    128ch 채널에서 64ch와 가장 유사한 채널들을 찾아 매핑
    """
    mapping = []
    for ch_64 in channel_64:
        closest_idx = np.argmin(np.abs(channel_128 - ch_64))
        mapping.append(closest_idx)
    return mapping

def downsample_128ch_to_64ch(points):
    """
    128ch 라이다 포인트 클라우드를 64ch로 다운샘플링
    """
    if len(points) == 0:
        return points
    
    # Cartesian to Spherical conversion
    r, theta, phi = cartesian_to_spherical(points)
    phi_degrees = np.degrees(phi)  # 고도각을 degree로 변환
    
    # 각 채널 매핑 생성
    channels_128 = get_128ch_channel_mapping()
    channels_64 = get_64ch_channel_mapping()
    
    # 각 포인트를 가장 가까운 128ch 채널에 할당
    channel_assignments = []
    for elevation in phi_degrees:
        closest_channel_idx = np.argmin(np.abs(channels_128 - elevation))
        channel_assignments.append(closest_channel_idx)
    
    channel_assignments = np.array(channel_assignments)
    
    # 64ch에 해당하는 채널 매핑 찾기
    selected_channels = find_closest_channels(channels_128, channels_64)
    
    # 선택된 채널의 포인트들만 유지
    mask = np.isin(channel_assignments, selected_channels)
    downsampled_points = points[mask]
    
    return downsampled_points

def process_lidar_files(input_dir, output_dir):
    """
    입력 디렉터리의 001, 002로 시작하는 .npy 파일들을 처리하여 64ch로 변환
    """
    # 출력 디렉터리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 001, 002로 시작하는 .npy 파일들 찾기
    pattern1 = os.path.join(input_dir, "001*.npy")
    pattern2 = os.path.join(input_dir, "002*.npy")
    
    files = glob.glob(pattern1) + glob.glob(pattern2)
    
    if len(files) == 0:
        print("No files starting with 001 or 002 found in the input directory.")
        print("Available files sample:")
        all_files = glob.glob(os.path.join(input_dir, "*.npy"))
        for i, f in enumerate(all_files[:10]):
            print(f"  {os.path.basename(f)}")
        return
    
    print(f"Found {len(files)} files to process")
    
    # 각 파일 처리
    for file_path in tqdm(files, desc="Processing files"):
        try:
            # 128ch 포인트 클라우드 로드
            points_128ch = np.load(file_path)
            
            # 64ch로 다운샘플링
            points_64ch = downsample_128ch_to_64ch(points_128ch)
            
            # 출력 파일 경로 생성
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, filename)
            
            # 64ch 포인트 클라우드 저장
            np.save(output_path, points_64ch)
            
            print(f"Processed {filename}: {len(points_128ch)} -> {len(points_64ch)} points")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

def test_single_file(file_path):
    """
    단일 파일에 대해 테스트
    """
    try:
        print(f"Testing file: {file_path}")
        points = np.load(file_path)
        print(f"Original shape: {points.shape}")
        
        if len(points) > 0:
            print(f"Point cloud bounds:")
            print(f"  X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
            print(f"  Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
            print(f"  Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
            
            # 다운샘플링 테스트
            downsampled = downsample_128ch_to_64ch(points)
            print(f"Downsampled shape: {downsampled.shape}")
            print(f"Reduction ratio: {len(downsampled)/len(points)*100:.1f}%")
            
        return points
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    # 경로 설정
    input_dir = "/home/ailab/git/Team_4/Ai_challenge/OpenPCDet/data/custom_av/points"
    output_dir = "/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av_range64/new_points"
    
    # 먼저 디렉터리에 어떤 파일들이 있는지 확인
    print("Checking input directory...")
    all_files = glob.glob(os.path.join(input_dir, "*.npy"))
    print(f"Total .npy files found: {len(all_files)}")
    
    if len(all_files) > 0:
        print("Sample files:")
        for i, f in enumerate(all_files[:5]):
            print(f"  {os.path.basename(f)}")
        
        # 첫 번째 파일로 테스트
        test_file = all_files[0]
        test_single_file(test_file)
    
    # 001, 002로 시작하는 파일들 처리
    print("\nProcessing 001, 002 files...")
    process_lidar_files(input_dir, output_dir)

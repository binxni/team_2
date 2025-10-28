import numpy as np
import os
import glob

def downsample_pointcloud_by_z(points, downsample_ratio=0.5):
    """
    포인트 클라우드를 z축 기준으로 다운샘플링
    z값을 기준으로 정렬한 후 2행 중 1행씩 삭제
    
    Args:
        points: numpy array (N, 3 or 4) - 포인트 클라우드 [x, y, z, intensity(optional)]
        downsample_ratio: float - 다운샘플링 비율 (0.5 = 50% 유지)
    
    Returns:
        downsampled_points: numpy array - 다운샘플링된 포인트 클라우드
    """
    if len(points) == 0:
        return points
    
    # z값을 기준으로 정렬
    z_sorted_indices = np.argsort(points[:, 2])
    sorted_points = points[z_sorted_indices]
    
    # 2행 중 1행씩 선택 (0, 2, 4, 6, ... 인덱스)
    # 이는 z축 기준으로 정렬된 상태에서 균등하게 샘플링
    step = int(1 / downsample_ratio)  # downsample_ratio=0.5이면 step=2
    selected_indices = np.arange(0, len(sorted_points), step)
    
    downsampled_points = sorted_points[selected_indices]
    
    return downsampled_points

def process_specific_pointcloud(input_dir, output_dir, frame_id, downsample_ratio=0.5):
    """
    특정 frame_id의 포인트 클라우드를 다운샘플링
    
    Args:
        input_dir: 입력 디렉터리 경로
        output_dir: 출력 디렉터리 경로
        frame_id: 처리할 frame_id (예: "001234", "002567")
        downsample_ratio: 다운샘플링 비율 (0.5 = 50% 유지)
    
    Returns:
        bool: 처리 성공 여부
    """
    # 해당 frame_id로 시작하는 .npy 파일 찾기
    file_pattern = os.path.join(input_dir, f"{frame_id}.npy")
    matching_files = glob.glob(file_pattern)
    
    if len(matching_files) == 0:
        print(f"No file found with frame_id: {frame_id}")
        # 비슷한 파일들 찾기
        pattern_prefix = frame_id[:3]  # 처음 3자리만 사용
        similar_files = glob.glob(os.path.join(input_dir, f"{pattern_prefix}*.npy"))
        if len(similar_files) > 0:
            print(f"Similar files found with prefix '{pattern_prefix}':")
            for f in similar_files[:10]:
                print(f"  {os.path.basename(f)}")
        return False
    
    file_path = matching_files[0]
    print(f"Found file: {os.path.basename(file_path)}")
    
    try:
        # 포인트 클라우드 로드
        points = np.load(file_path)
        print(f"Loaded point cloud with shape: {points.shape}")
        
        if len(points) == 0:
            print("Empty point cloud, skipping...")
            return False
        
        # 원본 포인트 클라우드 정보 출력
        print(f"Original point cloud bounds:")
        print(f"  X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"  Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"  Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
        print(f"  Total points: {len(points)}")
        
        # z축 기준 다운샘플링 수행
        print(f"\nPerforming downsampling with ratio: {downsample_ratio}")
        downsampled_points = downsample_pointcloud_by_z(points, downsample_ratio)
        
        # 다운샘플링된 포인트 클라우드 정보 출력
        print(f"Downsampled point cloud bounds:")
        print(f"  X: [{downsampled_points[:, 0].min():.2f}, {downsampled_points[:, 0].max():.2f}]")
        print(f"  Y: [{downsampled_points[:, 1].min():.2f}, {downsampled_points[:, 1].max():.2f}]")
        print(f"  Z: [{downsampled_points[:, 2].min():.2f}, {downsampled_points[:, 2].max():.2f}]")
        print(f"  Total points: {len(downsampled_points)}")
        
        # 출력 디렉터리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 다운샘플링된 파일 저장
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}_downsampled_{int(downsample_ratio*100)}percent.npy"
        output_path = os.path.join(output_dir, output_filename)
        
        np.save(output_path, downsampled_points)
        print(f"\nSaved downsampled point cloud to: {output_path}")
        
        # 통계 정보
        reduction_ratio = len(downsampled_points) / len(points)
        print(f"\n=== Downsampling Summary ===")
        print(f"Original points: {len(points)}")
        print(f"Downsampled points: {len(downsampled_points)}")
        print(f"Actual reduction ratio: {reduction_ratio:.3f}")
        print(f"Points removed: {len(points) - len(downsampled_points)}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def list_available_files(input_dir, prefix=""):
    """
    사용 가능한 파일들을 나열
    
    Args:
        input_dir: 입력 디렉터리 경로
        prefix: 파일명 접두사 (예: "001", "002")
    """
    if prefix:
        pattern = os.path.join(input_dir, f"{prefix}*.npy")
    else:
        pattern = os.path.join(input_dir, "*.npy")
    
    files = glob.glob(pattern)
    
    if len(files) == 0:
        print(f"No .npy files found in {input_dir}")
        if prefix:
            print(f"with prefix '{prefix}'")
        return []
    
    print(f"Found {len(files)} .npy files:")
    for i, f in enumerate(files[:20]):  # 최대 20개만 표시
        print(f"  {os.path.basename(f)}")
    
    if len(files) > 20:
        print(f"  ... and {len(files) - 20} more files")
    
    return files

if __name__ == "__main__":
    # 경로 설정
    input_dir = "/home/ailab/git/Team_4/Ai_challenge/OpenPCDet/data/custom_av/points"
    output_dir = "/home/ailab/git/Team_2/Seokjae/OpenPCDet/dataset_analysis/range_image"
    
    print("=== Point Cloud Z-axis Downsampling ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # 사용 가능한 파일들 확인
    print("\nChecking available files...")
    available_files = list_available_files(input_dir)
    
    if len(available_files) == 0:
        print("No files found. Exiting...")
        exit()
    
    # === 처리할 frame_id 설정 ===
    # 여기서 원하는 frame_id를 지정하세요
    target_frame_id = "00202455"  # 원하는 frame_id로 변경하세요
    downsample_ratio = 0.3  # 50% 유지 (2행 중 1행 삭제)
    
    print(f"\nProcessing frame_id: {target_frame_id}")
    print(f"Downsampling ratio: {downsample_ratio} (keeping {int(downsample_ratio*100)}% of points)")
    
    # 다운샘플링 수행
    success = process_specific_pointcloud(
        input_dir=input_dir,
        output_dir=output_dir,
        frame_id=target_frame_id,
        downsample_ratio=downsample_ratio
    )
    
    if success:
        print(f"\n✅ Successfully downsampled frame {target_frame_id}!")
    else:
        print(f"\n❌ Failed to process frame {target_frame_id}")
        print("\nTip: Check if the frame_id exists in the input directory")
        print("Available files with similar prefixes:")
        prefix = target_frame_id[:3]
        list_available_files(input_dir, prefix)
    
    print("\nPoint cloud downsampling completed!")
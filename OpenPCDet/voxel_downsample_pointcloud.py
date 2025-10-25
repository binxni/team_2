import numpy as np
import open3d as o3d
import os
import glob

def downsample_pointcloud_voxel(points, voxel_size=0.15):
    """
    복셀 기반 포인트 클라우드 다운샘플링
    
    Args:
        points: numpy array (N, 3 or 4) - 포인트 클라우드 [x, y, z, intensity(optional)]
        voxel_size: float - 복셀 크기 (m 단위)
    
    Returns:
        downsampled_points: numpy array - 다운샘플링된 포인트 클라우드
    """
    if len(points) == 0:
        return points
    
    # Open3D PointCloud 객체 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # x, y, z만 사용
    
    # 복셀 기반 다운샘플링 수행
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # numpy 배열로 변환
    downsampled_xyz = np.asarray(down_pcd.points)
    
    # intensity 정보가 있는 경우 처리
    if points.shape[1] > 3:
        # 원본 포인트에서 가장 가까운 점의 intensity 사용
        from scipy.spatial import cKDTree
        
        # KDTree로 최근접 이웃 찾기
        tree = cKDTree(points[:, :3])
        distances, indices = tree.query(downsampled_xyz)
        
        # 대응되는 intensity 값 추출
        intensities = points[indices, 3:]
        
        # xyz + intensity 결합
        downsampled_points = np.column_stack([downsampled_xyz, intensities])
    else:
        downsampled_points = downsampled_xyz
    
    return downsampled_points

def process_voxel_downsampling(input_dir, output_dir, frame_id, voxel_size=0.1):
    """
    특정 frame_id의 포인트 클라우드를 복셀 기반으로 다운샘플링
    
    Args:
        input_dir: 입력 디렉터리 경로
        output_dir: 출력 디렉터리 경로
        frame_id: 처리할 frame_id (예: "001234", "002567")
        voxel_size: 복셀 크기 (m 단위)
    
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
        print(f"\nOriginal point cloud bounds:")
        print(f"  X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}] (range: {points[:, 0].max() - points[:, 0].min():.2f}m)")
        print(f"  Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}] (range: {points[:, 1].max() - points[:, 1].min():.2f}m)")
        print(f"  Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}] (range: {points[:, 2].max() - points[:, 2].min():.2f}m)")
        print(f"  Total points: {len(points):,}")
        
        # 복셀 크기 대비 공간 범위 정보
        x_range = points[:, 0].max() - points[:, 0].min()
        y_range = points[:, 1].max() - points[:, 1].min()
        z_range = points[:, 2].max() - points[:, 2].min()
        
        estimated_voxels = (x_range / voxel_size) * (y_range / voxel_size) * (z_range / voxel_size)
        print(f"  Estimated max voxels with {voxel_size}m voxel size: {estimated_voxels:,.0f}")
        
        # 복셀 기반 다운샘플링 수행
        print(f"\nPerforming voxel downsampling with voxel_size: {voxel_size}m")
        downsampled_points = downsample_pointcloud_voxel(points, voxel_size)
        
        # 다운샘플링된 포인트 클라우드 정보 출력
        print(f"Downsampled point cloud bounds:")
        print(f"  X: [{downsampled_points[:, 0].min():.2f}, {downsampled_points[:, 0].max():.2f}]")
        print(f"  Y: [{downsampled_points[:, 1].min():.2f}, {downsampled_points[:, 1].max():.2f}]")
        print(f"  Z: [{downsampled_points[:, 2].min():.2f}, {downsampled_points[:, 2].max():.2f}]")
        print(f"  Total points: {len(downsampled_points):,}")
        
        # 출력 디렉터리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 다운샘플링된 파일 저장
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        voxel_size_str = f"{voxel_size:.3f}".replace(".", "p")  # 0.1 -> 0p1
        output_filename = f"{base_name}_voxel_{voxel_size_str}m.npy"
        output_path = os.path.join(output_dir, output_filename)
        
        np.save(output_path, downsampled_points)
        print(f"\nSaved voxel downsampled point cloud to: {output_path}")
        
        # 통계 정보
        reduction_ratio = len(downsampled_points) / len(points)
        compression_ratio = len(points) / len(downsampled_points)
        
        print(f"\n=== Voxel Downsampling Summary ===")
        print(f"Voxel size: {voxel_size}m")
        print(f"Original points: {len(points):,}")
        print(f"Downsampled points: {len(downsampled_points):,}")
        print(f"Reduction ratio: {reduction_ratio:.3f} (kept {reduction_ratio*100:.1f}%)")
        print(f"Compression ratio: {compression_ratio:.1f}:1")
        print(f"Points removed: {len(points) - len(downsampled_points):,}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def visualize_comparison(original_points, downsampled_points, title="Point Cloud Comparison"):
    """
    원본과 다운샘플링된 포인트 클라우드 비교 시각화
    
    Args:
        original_points: numpy array - 원본 포인트 클라우드
        downsampled_points: numpy array - 다운샘플링된 포인트 클라우드
        title: str - 시각화 창 제목
    """
    # 원본 포인트 클라우드 (빨간색)
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(original_points[:, :3])
    pcd_original.paint_uniform_color([1, 0, 0])  # 빨간색
    
    # 다운샘플링된 포인트 클라우드 (파란색)
    pcd_downsampled = o3d.geometry.PointCloud()
    pcd_downsampled.points = o3d.utility.Vector3dVector(downsampled_points[:, :3])
    pcd_downsampled.paint_uniform_color([0, 0, 1])  # 파란색
    
    # 시각화
    print(f"\nVisualizing: Red = Original ({len(original_points):,} points), Blue = Downsampled ({len(downsampled_points):,} points)")
    o3d.visualization.draw_geometries([pcd_original, pcd_downsampled], window_name=title)

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
    
    print("=== Voxel-based Point Cloud Downsampling ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # 사용 가능한 파일들 확인
    print("\nChecking available files...")
    available_files = list_available_files(input_dir)
    
    if len(available_files) == 0:
        print("No files found. Exiting...")
        exit()
    
    # === 처리할 frame_id 및 복셀 크기 설정 ===
    target_frame_id = "00202455"  # 원하는 frame_id로 변경하세요
    voxel_size = 0.2  # 복셀 크기 (m 단위) - 작을수록 더 세밀함
    enable_visualization = False  # True로 설정하면 3D 시각화 활성화
    
    print(f"\nProcessing frame_id: {target_frame_id}")
    print(f"Voxel size: {voxel_size}m")
    print(f"Visualization: {'Enabled' if enable_visualization else 'Disabled'}")
    
    # 복셀 다운샘플링 수행
    success = process_voxel_downsampling(
        input_dir=input_dir,
        output_dir=output_dir,
        frame_id=target_frame_id,
        voxel_size=voxel_size
    )
    
    # 시각화 (선택사항)
    if success and enable_visualization:
        try:
            # 원본과 다운샘플링된 포인트 로드
            file_path = os.path.join(input_dir, f"{target_frame_id}.npy")
            original_points = np.load(file_path)
            
            voxel_size_str = f"{voxel_size:.3f}".replace(".", "p")
            base_name = os.path.splitext(target_frame_id)[0]
            downsampled_path = os.path.join(output_dir, f"{base_name}_voxel_{voxel_size_str}m.npy")
            downsampled_points = np.load(downsampled_path)
            
            visualize_comparison(original_points, downsampled_points, 
                               f"Voxel Downsampling: {voxel_size}m")
        except Exception as e:
            print(f"Visualization error: {e}")
    
    if success:
        print(f"\n✅ Successfully processed frame {target_frame_id} with voxel downsampling!")
        print(f"💡 Tip: Adjust voxel_size for different compression levels:")
        print(f"   - 0.05m: Fine detail (less compression)")
        print(f"   - 0.1m:  Balanced (moderate compression)")
        print(f"   - 0.2m:  Coarse (high compression)")
    else:
        print(f"\n❌ Failed to process frame {target_frame_id}")
        print("\nTip: Check if the frame_id exists in the input directory")
        print("Available files with similar prefixes:")
        prefix = target_frame_id[:3]
        list_available_files(input_dir, prefix)
    
    print("\nVoxel-based point cloud downsampling completed!")
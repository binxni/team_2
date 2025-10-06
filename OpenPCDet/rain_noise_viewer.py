import open3d as o3d
import numpy as np
import os
import time
from rain_noise_utils import RainNoiseSimulator

# Configuration
DATA_PATH = "/home/ailab/git/Team_2/Subin/OpenPCDet/data/custom_av"
FRAME_LIST_FILE = os.path.join(DATA_PATH, "ImageSets", "train.txt")
POINTS_FOLDER = os.path.join(DATA_PATH, "points")
LABELS_FOLDER = os.path.join(DATA_PATH, "labels")
VIEW_FILE = os.path.join(DATA_PATH, "view.json")

# Viewer modes
VIEW_MODES = {
    'ORIGINAL': 0,
    'RAIN_NOISE': 1,
    'COMPARISON': 2
}

class RainNoiseViewer:
    def __init__(self):
        # Load frame IDs
        with open(FRAME_LIST_FILE, 'r') as f:
            self.frame_ids = [line.strip() for line in f.readlines()]
        
        # Initialize rain noise simulator
        self.rain_simulator = RainNoiseSimulator()
        
        # Viewer state
        self.frame_idx = 0
        self.view_mode = VIEW_MODES['ORIGINAL']
        self.rain_intensity = 0.8  # Default rain intensity
        self.noise_radius = 2.0    # Default noise radius (meters)
        self.last_key_time = 0
        self.key_delay = 0.1
        
        # Cached data
        self.current_original_points = None
        self.current_rain_points = None
        self.current_rain_intensity = 0.0
        
        # Colors for different point clouds
        self.original_color = [0.7, 0.7, 0.7]  # Gray
        self.rain_noise_color = [0.3, 0.8, 1.0]  # Light blue
        self.rain_droplet_color = [1.0, 0.3, 0.3]  # Red
    
    def load_npy_pointcloud(self, file_path):
        """포인트 클라우드를 로드합니다."""
        points = np.load(file_path)
        xyz = points[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        return pcd, points
    
    def create_bbox(self, center, size, yaw, color):
        """바운딩 박스를 생성합니다."""
        R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])
        box = o3d.geometry.OrientedBoundingBox(center, R, size)
        box.color = color
        return box
    
    def create_heading_arrow(self, center, yaw, length=2.0):
        """방향 화살표를 생성합니다."""
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.05,
            cone_radius=0.1,
            cylinder_height=length * 0.8,
            cone_height=length * 0.2
        )
        arrow.paint_uniform_color([1, 0, 0])  # Red

        # Step 1: 기본 방향 +Z → +X 로 눕히기
        R_to_x = o3d.geometry.get_rotation_matrix_from_xyz([-np.pi / 2, 0, 0])
        arrow.rotate(R_to_x, center=(0, 0, 0))

        # Step 2: yaw 회전 적용 (Z축 기준)
        R_yaw = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw - np.pi/2])
        arrow.rotate(R_yaw, center=(0, 0, 0))

        # Step 3: 중심 위치로 이동
        arrow.translate(center)
        return arrow
    
    def load_labels(self, label_path):
        """라벨을 로드합니다."""
        labels = []
        if not os.path.exists(label_path):
            return labels
            
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                x, y, z = map(float, parts[0:3])
                dx, dy, dz = map(float, parts[3:6])
                yaw = float(parts[6])
                color = [1, 0, 0]  # Red
                labels.append((x, y, z, dx, dy, dz, yaw, color))
        return labels
    
    def generate_rain_noise_data(self):
        """현재 프레임에 대한 비 노이즈 데이터를 생성합니다."""
        if self.current_original_points is None:
            return
        
        # 노이즈 반경 설정 적용
        self.rain_simulator.set_noise_radius(self.noise_radius)
        
        # 비 노이즈 시뮬레이션 적용
        noisy_points, actual_intensity = self.rain_simulator.simulate_rain_noise(
            self.current_original_points, 
            self.rain_intensity
        )
        
        self.current_rain_points = noisy_points
        self.current_rain_intensity = actual_intensity
    
    def create_pointcloud_with_color(self, points, color):
        """색상이 지정된 포인트 클라우드를 생성합니다."""
        if len(points) == 0:
            return o3d.geometry.PointCloud()
            
        xyz = points[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        
        # 모든 포인트에 같은 색상 적용
        colors = np.tile(color, (len(points), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    
    def create_comparison_pointclouds(self):
        """비교 모드용 포인트 클라우드들을 생성합니다."""
        pcds = []
        
        if self.current_original_points is not None:
            # 원본 포인트 클라우드
            original_pcd = self.create_pointcloud_with_color(
                self.current_original_points, 
                self.original_color
            )
            pcds.append(original_pcd)
        
        if self.current_rain_points is not None:
            # 비 노이즈가 추가된 포인트에서 원본 포인트와 노이즈 포인트 구분
            num_original = len(self.current_original_points) if self.current_original_points is not None else 0
            
            if len(self.current_rain_points) > num_original:
                # 원본 포인트 (감쇠된 버전)
                attenuated_points = self.current_rain_points[:num_original]
                attenuated_pcd = self.create_pointcloud_with_color(
                    attenuated_points, 
                    self.rain_noise_color
                )
                pcds.append(attenuated_pcd)
                
                # 비 노이즈 포인트
                noise_points = self.current_rain_points[num_original:]
                noise_pcd = self.create_pointcloud_with_color(
                    noise_points, 
                    self.rain_droplet_color
                )
                pcds.append(noise_pcd)
            else:
                # 모든 포인트가 감쇠된 원본 포인트
                rain_pcd = self.create_pointcloud_with_color(
                    self.current_rain_points, 
                    self.rain_noise_color
                )
                pcds.append(rain_pcd)
        
        return pcds
    
    def create_noise_radius_guide(self):
        """노이즈 반경을 시각화하는 원형 가이드를 생성합니다."""
        try:
            # 원점 중심의 원형 라인셋 생성
            num_points = 64
            angles = np.linspace(0, 2 * np.pi, num_points)
            
            # XY 평면에서 원 생성 (Z=0)
            circle_points = []
            for angle in angles:
                x = self.noise_radius * np.cos(angle)
                y = self.noise_radius * np.sin(angle)
                z = 0.0
                circle_points.append([x, y, z])
            
            # 라인셋 생성
            lines = []
            for i in range(num_points):
                lines.append([i, (i + 1) % num_points])
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(circle_points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            
            # 노란색으로 설정
            colors = [[1, 1, 0] for _ in range(len(lines))]  # Yellow
            line_set.colors = o3d.utility.Vector3dVector(colors)
            
            return line_set
        except Exception as e:
            print(f"Warning: Could not create noise radius guide: {e}")
            return None
    
    def get_view_mode_name(self):
        """현재 뷰 모드 이름을 반환합니다."""
        for name, value in VIEW_MODES.items():
            if value == self.view_mode:
                return name
        return "UNKNOWN"
    
    def update_scene(self, vis):
        """화면을 업데이트합니다."""
        vis.clear_geometries()
        
        frame_id = self.frame_ids[self.frame_idx]
        pc_path = os.path.join(POINTS_FOLDER, f"{frame_id}.npy")
        label_path = os.path.join(LABELS_FOLDER, f"{frame_id}.txt")
        
        # 포인트 클라우드 로드
        if os.path.exists(pc_path):
            _, self.current_original_points = self.load_npy_pointcloud(pc_path)
            
            # 비 노이즈 데이터 생성
            self.generate_rain_noise_data()
            
            # 뷰 모드에 따라 다른 포인트 클라우드 표시
            if self.view_mode == VIEW_MODES['ORIGINAL']:
                # 원본 포인트 클라우드만 표시
                pcd = self.create_pointcloud_with_color(
                    self.current_original_points, 
                    self.original_color
                )
                vis.add_geometry(pcd)
                
            elif self.view_mode == VIEW_MODES['RAIN_NOISE']:
                # 비 노이즈 포인트 클라우드만 표시
                if self.current_rain_points is not None:
                    pcd = self.create_pointcloud_with_color(
                        self.current_rain_points, 
                        self.rain_noise_color
                    )
                    vis.add_geometry(pcd)
                    
            elif self.view_mode == VIEW_MODES['COMPARISON']:
                # 비교 모드: 원본과 노이즈를 다른 색상으로 표시
                comparison_pcds = self.create_comparison_pointclouds()
                for pcd in comparison_pcds:
                    vis.add_geometry(pcd)
        
        # 라벨 로드 및 표시
        labels = self.load_labels(label_path)
        for x, y, z, dx, dy, dz, yaw, color in labels:
            center = [x, y, z]
            size = [dx, dy, dz]
            box = self.create_bbox(center, size, yaw, color)
            vis.add_geometry(box)

            arrow = self.create_heading_arrow(center, yaw)
            vis.add_geometry(arrow)
        
        # 좌표계 추가
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)
        
        # 노이즈 반경 시각화 (원점 중심의 원형 가이드)
        if self.view_mode != VIEW_MODES['ORIGINAL']:
            noise_circle = self.create_noise_radius_guide()
            if noise_circle is not None:
                vis.add_geometry(noise_circle)
        
        vis.poll_events()
        vis.update_renderer()
        
        # 뷰포인트 복원
        if os.path.exists(VIEW_FILE):
            params = o3d.io.read_pinhole_camera_parameters(VIEW_FILE)
            vis.get_view_control().convert_from_pinhole_camera_parameters(params)
        
        # 상태 정보 출력
        mode_name = self.get_view_mode_name()
        if self.current_rain_points is not None:
            rain_info = f" (Rain: {self.current_rain_intensity:.2f}, Radius: {self.noise_radius:.1f}m)"
        else:
            rain_info = f" (Radius: {self.noise_radius:.1f}m)"
        
        print(f"\rFrame {self.frame_idx}: {frame_id} | Mode: {mode_name}{rain_info} | "
              f"Original: {len(self.current_original_points) if self.current_original_points is not None else 0} pts | "
              f"Rain: {len(self.current_rain_points) if self.current_rain_points is not None else 0} pts", end="", flush=True)
    
    def save_viewpoint(self, vis):
        """뷰포인트를 저장합니다."""
        params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(VIEW_FILE, params)
        print(f"\nViewpoint saved to {VIEW_FILE}")
    
    def debounce(self):
        """키 입력 디바운싱"""
        now = time.time()
        if now - self.last_key_time >= self.key_delay:
            self.last_key_time = now
            return True
        return False
    
    def next_frame(self, vis):
        """다음 프레임으로 이동"""
        if self.debounce():
            self.frame_idx = (self.frame_idx + 1) % len(self.frame_ids)
            self.update_scene(vis)
        return False
    
    def prev_frame(self, vis):
        """이전 프레임으로 이동"""
        if self.debounce():
            self.frame_idx = (self.frame_idx - 1 + len(self.frame_ids)) % len(self.frame_ids)
            self.update_scene(vis)
        return False
    
    def toggle_view_mode(self, vis):
        """뷰 모드 전환"""
        if self.debounce():
            self.view_mode = (self.view_mode + 1) % len(VIEW_MODES)
            self.update_scene(vis)
        return False
    
    def increase_rain_intensity(self, vis):
        """비 강도 증가"""
        if self.debounce():
            self.rain_intensity = min(2.0, self.rain_intensity + 0.1)
            self.update_scene(vis)
        return False
    
    def decrease_rain_intensity(self, vis):
        """비 강도 감소"""
        if self.debounce():
            self.rain_intensity = max(0.1, self.rain_intensity - 0.1)
            self.update_scene(vis)
        return False
    
    def increase_noise_radius(self, vis):
        """노이즈 반경 증가"""
        if self.debounce():
            self.noise_radius = min(20.0, self.noise_radius + 0.5)
            self.update_scene(vis)
        return False
    
    def decrease_noise_radius(self, vis):
        """노이즈 반경 감소"""
        if self.debounce():
            self.noise_radius = max(0.5, self.noise_radius - 0.5)
            self.update_scene(vis)
        return False
    
    def save_view(self, vis):
        """뷰 저장"""
        self.save_viewpoint(vis)
        return False
    
    def print_help(self, vis):
        """도움말 출력"""
        print("\n" + "="*80)
        print("🌧️ Rain Noise Point Cloud Viewer - Help")
        print("="*80)
        print("Navigation:")
        print("  D/→     : Next frame")
        print("  A/←     : Previous frame")
        print("")
        print("View Modes:")
        print("  T       : Toggle view mode (Original → Rain Noise → Comparison)")
        print("            - Original: Show original point cloud only")
        print("            - Rain Noise: Show rain-affected point cloud only") 
        print("            - Comparison: Show both with different colors")
        print("")
        print("Rain Control:")
        print("  ↑       : Increase rain intensity (+0.1)")
        print("  ↓       : Decrease rain intensity (-0.1)")
        print("")
        print("Noise Range Control:")
        print("  +       : Increase noise radius (+0.5m)")
        print("  -       : Decrease noise radius (-0.5m)")
        print("")
        print("Colors in Comparison mode:")
        print("  Gray    : Original points")
        print("  Blue    : Rain-affected original points")
        print("  Red     : Rain droplet noise points")
        print("")
        print("Other:")
        print("  F       : Save current viewpoint")
        print("  H       : Show this help")
        print("  Q       : Quit viewer")
        print("="*80)
        return False
    
    def quit_viewer(self, vis):
        """뷰어 종료"""
        print("\nQuitting Rain Noise Viewer.")
        vis.close()
        return False
    
    def run(self):
        """뷰어 실행"""
        vis = o3d.visualization.VisualizerWithKeyCallback()
        if not vis.create_window(window_name='Rain Noise Point Cloud Viewer'):
            print("[ERROR] Failed to create Open3D window.")
            return
        
        # 배경색을 검정색으로 설정
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])  # 검정색 배경
        
        # 키 콜백 등록
        vis.register_key_callback(ord("D"), self.next_frame)
        vis.register_key_callback(ord("A"), self.prev_frame)
        vis.register_key_callback(ord("T"), self.toggle_view_mode)
        vis.register_key_callback(ord("F"), self.save_view)
        vis.register_key_callback(ord("H"), self.print_help)
        vis.register_key_callback(ord("Q"), self.quit_viewer)
        vis.register_key_callback(ord("+"), self.increase_noise_radius)  # + key
        vis.register_key_callback(ord("-"), self.decrease_noise_radius)  # - key
        vis.register_key_callback(ord("="), self.increase_noise_radius)  # = key (same as +)
        
        # 화살표 키 지원 (ASCII 코드)
        vis.register_key_callback(262, self.next_frame)     # Right arrow
        vis.register_key_callback(263, self.prev_frame)    # Left arrow
        vis.register_key_callback(264, self.decrease_rain_intensity)  # Down arrow
        vis.register_key_callback(265, self.increase_rain_intensity)  # Up arrow
        
        # 초기 화면 설정
        self.update_scene(vis)
        
        # 도움말 출력
        self.print_help(vis)
        
        # 뷰어 실행
        vis.run()

def main():
    """메인 함수"""
    print("🌧️ Starting Rain Noise Point Cloud Viewer...")
    
    # 데이터 경로 확인
    if not os.path.exists(FRAME_LIST_FILE):
        print(f"[ERROR] Frame list file not found: {FRAME_LIST_FILE}")
        return
    
    if not os.path.exists(POINTS_FOLDER):
        print(f"[ERROR] Points folder not found: {POINTS_FOLDER}")
        return
    
    # 뷰어 실행
    viewer = RainNoiseViewer()
    viewer.run()

if __name__ == "__main__":
    main()
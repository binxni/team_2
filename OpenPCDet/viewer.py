# Subin
import open3d as o3d
import numpy as np
import os
import time

# Configuration
DATA_PATH = "./data/custom_av"
FRAME_LIST_FILE = os.path.join(DATA_PATH, "ImageSets", "train.txt")
POINTS_FOLDER = os.path.join(DATA_PATH, "points")
LABELS_FOLDER = os.path.join(DATA_PATH, "labels")
VIEW_FILE = os.path.join(DATA_PATH, "view.json")

# Load frame IDs
with open(FRAME_LIST_FILE, 'r') as f:
    frame_ids = [line.strip() for line in f.readlines()]

def load_npy_pointcloud(file_path):
    points = np.load(file_path)
    xyz = points[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def create_bbox(center, size, yaw, color):
    R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])
    box = o3d.geometry.OrientedBoundingBox(center, R, size)
    box.color = color
    return box

def create_heading_arrow(center, yaw, length=2.0):
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




def load_labels(label_path):
    labels = []
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

def save_viewpoint(vis):
    params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(VIEW_FILE, params)
    print(f"Viewpoint saved to {VIEW_FILE}")

def deferred_load_viewpoint(vis):
    if os.path.exists(VIEW_FILE):
        params = o3d.io.read_pinhole_camera_parameters(VIEW_FILE)
        vis.get_view_control().convert_from_pinhole_camera_parameters(params)
        print(f"[Deferred] Viewpoint loaded from {VIEW_FILE}")
    else:
        print("[Warning] No saved viewpoint found.")

def visualize_frames():
    vis = o3d.visualization.VisualizerWithKeyCallback()
    if not vis.create_window(window_name='LiDAR Viewer'):
        print("[ERROR] Failed to create Open3D window.")
        return

    frame_idx = 0
    deferred_view_loaded = False
    last_key_time = 0
    key_delay = 0.1  # seconds

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    def update_scene():
        nonlocal deferred_view_loaded
        vis.clear_geometries()

        frame_id = frame_ids[frame_idx]
        pc_path = os.path.join(POINTS_FOLDER, f"{frame_id}.npy")
        label_path = os.path.join(LABELS_FOLDER, f"{frame_id}.txt")

        pcd = load_npy_pointcloud(pc_path)
        vis.add_geometry(pcd)

        labels = load_labels(label_path)
        for x, y, z, dx, dy, dz, yaw, color in labels:
            center = [x, y, z]
            size = [dx, dy, dz]
            box = create_bbox(center, size, yaw, color)
            vis.add_geometry(box)

            arrow = create_heading_arrow(center, yaw)
            vis.add_geometry(arrow)

        vis.add_geometry(coordinate_frame)

        vis.poll_events()
        vis.update_renderer()

        if os.path.exists(VIEW_FILE):
            params = o3d.io.read_pinhole_camera_parameters(VIEW_FILE)
            vis.get_view_control().convert_from_pinhole_camera_parameters(params)

        if not deferred_view_loaded:
            deferred_view_loaded = True

    def debounce():
        nonlocal last_key_time
        now = time.time()
        if now - last_key_time >= key_delay:
            last_key_time = now
            return True
        return False

    def next_frame(vis):
        nonlocal frame_idx
        if debounce():
            frame_idx = (frame_idx + 1) % len(frame_ids)
            update_scene()
        return False

    def prev_frame(vis):
        nonlocal frame_idx
        if debounce():
            frame_idx = (frame_idx - 1 + len(frame_ids)) % len(frame_ids)
            update_scene()
        return False

    def save_view(vis):
        save_viewpoint(vis)
        return False

    def quit_viewer(vis):
        print("Quitting viewer.")
        vis.close()
        return False

    vis.register_key_callback(ord("D"), next_frame)
    vis.register_key_callback(ord("A"), prev_frame)
    vis.register_key_callback(ord("F"), save_view)
    vis.register_key_callback(ord("Q"), quit_viewer)

    update_scene()
    vis.run()

if __name__ == "__main__":
    visualize_frames()
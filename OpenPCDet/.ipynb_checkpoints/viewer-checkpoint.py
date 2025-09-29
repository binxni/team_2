"""Notebook-friendly visualization utilities for the Custom AV dataset.

Adds headless rendering support so frames can be exported to images when a GUI
is unavailable (e.g. inside a container without an X display).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Sequence

import numpy as np
import open3d as o3d

try:
    from open3d.visualization import rendering
except ImportError:  # pragma: no cover
    rendering = None

DATA_PATH = Path("./data/custom_av")
FRAME_LIST_FILE = DATA_PATH / "ImageSets" / "train.txt"
POINTS_FOLDER = DATA_PATH / "points"
LABELS_FOLDER = DATA_PATH / "labels"


def load_frame_ids(list_file: Path = FRAME_LIST_FILE) -> List[str]:
    if not list_file.exists():
        raise FileNotFoundError(f"Frame list not found: {list_file}")
    ids = [line.strip() for line in list_file.read_text().splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"No frame ids found in {list_file}")
    return ids


FRAME_IDS = load_frame_ids()


def load_point_cloud(frame_id: str) -> o3d.geometry.PointCloud:
    npy_path = POINTS_FOLDER / f"{frame_id}.npy"
    if not npy_path.exists():
        raise FileNotFoundError(f"Point file missing: {npy_path}")
    points = np.load(npy_path)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    return cloud


def load_labels(frame_id: str) -> Sequence[Sequence[float]]:
    label_path = LABELS_FOLDER / f"{frame_id}.txt"
    if not label_path.exists():
        return []
    labels = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 8:
            continue
        x, y, z = map(float, parts[0:3])
        dx, dy, dz = map(float, parts[3:6])
        yaw = float(parts[6])
        labels.append((x, y, z, dx, dy, dz, yaw))
    return labels


def create_bbox(center, size, yaw, color=(1.0, 0.0, 0.0)):
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle([0.0, 0.0, yaw])
    obb = o3d.geometry.OrientedBoundingBox(center, rot, size)
    lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    colors = np.tile(np.asarray(color, dtype=np.float64), (len(lineset.lines), 1))
    lineset.colors = o3d.utility.Vector3dVector(colors)
    return lineset


def create_heading_arrow(center, yaw, length=2.0):
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.05,
        cone_radius=0.1,
        cylinder_height=length * 0.8,
        cone_height=length * 0.2,
    )
    arrow.paint_uniform_color([1.0, 0.0, 0.0])
    rot_to_x = o3d.geometry.get_rotation_matrix_from_xyz([-np.pi / 2.0, 0.0, 0.0])
    arrow.rotate(rot_to_x, center=(0.0, 0.0, 0.0))
    rot_yaw = o3d.geometry.get_rotation_matrix_from_axis_angle([0.0, 0.0, yaw - np.pi / 2.0])
    arrow.rotate(rot_yaw, center=(0.0, 0.0, 0.0))
    arrow.translate(center)
    return arrow


def build_geometries(frame_id: str, *, include_arrows: bool = True, include_axis: bool = True):
    geometries: List[o3d.geometry.Geometry] = []
    geometries.append(load_point_cloud(frame_id))
    for x, y, z, dx, dy, dz, yaw in load_labels(frame_id):
        center = [x, y, z]
        size = [dx, dy, dz]
        geometries.append(create_bbox(center, size, yaw))
        if include_arrows:
            geometries.append(create_heading_arrow(center, yaw))
    if include_axis:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0.0, 0.0, 0.0]))
    return geometries


def render_offscreen(
    geometries: Sequence[o3d.geometry.Geometry],
    output_path: Path,
    width: int,
    height: int,
    point_size: float,
    bg_color=(0.0, 0.0, 0.0, 1.0),
):
    if rendering is None:
        raise RuntimeError("Offscreen rendering requires 'open3d.visualization.rendering' support")

    renderer = rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background(bg_color)

    def add(name: str, geometry: o3d.geometry.Geometry):
        material = rendering.MaterialRecord()
        if isinstance(geometry, o3d.geometry.PointCloud):
            material.shader = "defaultUnlit"
            material.point_size = point_size
        elif isinstance(geometry, o3d.geometry.LineSet):
            material.shader = "defaultUnlitLine"
            material.line_width = 1.0
        elif isinstance(geometry, o3d.geometry.TriangleMesh):
            material.shader = "defaultUnlit"
        else:
            return
        renderer.scene.add_geometry(name, geometry, material)

    for idx, geom in enumerate(geometries):
        add(f"geom_{idx}", geom)

    aabb = renderer.scene.bounding_box
    if aabb.extent.max() == 0:
        center = np.asarray([0.0, 0.0, 0.0])
        radius = 10.0
    else:
        center = np.asarray(aabb.get_center())
        radius = np.linalg.norm(aabb.get_extent()) * 0.6 + 1e-3

    eye = center + np.array([0.0, -radius, radius])
    up = np.array([0.0, 0.0, 1.0])
    renderer.scene.camera.look_at(center, eye, up)

    image = renderer.render_to_image()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_image(str(output_path), image)


def draw_frame(
    frame_index: int = 0,
    *,
    notebook: bool = True,
    show: bool = True,
    width: int = 960,
    height: int = 720,
    point_size: float = 1.0,
    include_arrows: bool = True,
    include_axis: bool = True,
    save_path: Path | None = None,
) -> None:
    if not FRAME_IDS:
        raise RuntimeError("Frame id list is empty")
    frame_id = FRAME_IDS[frame_index % len(FRAME_IDS)]
    geometries = build_geometries(frame_id, include_arrows=include_arrows, include_axis=include_axis)

    if save_path is not None:
        render_offscreen(geometries, Path(save_path), width, height, point_size)
        print(f"Saved frame '{frame_id}' to {save_path}")

    if show:
        o3d.visualization.draw(
            geometries,
            title=f"Frame {frame_id}",
            width=width,
            height=height,
            point_size=point_size,
            show_ui=not notebook,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Custom AV frames with Open3D draw API")
    parser.add_argument("--frame", type=int, default=0, help="Index of the frame to visualize")
    parser.add_argument("--no-arrows", action="store_true", help="Hide heading arrows")
    parser.add_argument("--no-axis", action="store_true", help="Hide coordinate frame")
    parser.add_argument("--point-size", type=float, default=1.0, help="Point size for rendering")
    parser.add_argument("--desktop", action="store_true", help="Force desktop window")
    parser.add_argument("--save", type=Path, default=None, help="Path to save an off-screen rendered image")
    parser.add_argument("--no-view", action="store_true", help="Skip interactive viewer")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.desktop and os.environ.get("DISPLAY") is None:
        print("[WARN] DISPLAY is not set; desktop viewer may fail. Use --save or run inside a notebook.")

    show_viewer = not args.no_view
    notebook_mode = not args.desktop

    draw_frame(
        frame_index=args.frame,
        notebook=notebook_mode,
        show=show_viewer,
        point_size=args.point_size,
        include_arrows=not args.no_arrows,
        include_axis=not args.no_axis,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()

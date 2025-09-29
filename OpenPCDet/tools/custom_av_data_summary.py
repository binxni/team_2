#!/usr/bin/env python3
"""Utility to summarize the Custom AV dataset statistics and optional plots."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Custom AV dataset statistics")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "custom_av",
        help="Root directory that contains info files, points/, labels/",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="List of dataset splits to summarize (expects custom_av_infos_{split}.pkl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of frames (per split) to process",
    )
    parser.add_argument(
        "--point-stats",
        action="store_true",
        help="If set, load point clouds (*.npy) to report per-frame point statistics",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory to save visualization outputs (bar charts, histograms)",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=150,
        help="DPI setting for saved plots",
    )
    return parser.parse_args()


def load_infos(info_path: Path) -> List[dict]:
    if not info_path.exists():
        raise FileNotFoundError(f"Info file not found: {info_path}")
    with info_path.open("rb") as f:
        return pickle.load(f)


def compute_numeric_summary(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {}
    percentiles = np.percentile(arr, [25, 50, 75])
    summary = {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "25%": float(percentiles[0]),
        "50%": float(percentiles[1]),
        "75%": float(percentiles[2]),
        "max": float(arr.max()),
    }
    return summary


def summarize_split(
    infos: List[dict],
    data_root: Path,
    limit: int | None = None,
    with_point_stats: bool = False,
    collect_raw: bool = False,
) -> Tuple[Dict, Dict | None]:
    if limit is not None:
        infos = infos[:limit]

    num_frames = len(infos)
    class_to_boxes: Dict[str, List[np.ndarray]] = defaultdict(list)
    class_to_points: Dict[str, List[int]] = defaultdict(list)
    frames_with_annos = 0

    lidar_type_counter: Counter[str] = Counter()
    lidar_channel_counter: Counter[int] = Counter()

    points_per_frame: List[int] = []
    intensity_values: List[float] = []

    for info in infos:
        pc_meta = info.get("point_cloud", {})
        lidar_type = pc_meta.get("lidar_type")
        if lidar_type:
            lidar_type_counter[lidar_type] += 1
        num_channels = pc_meta.get("num_channels")
        if num_channels is not None:
            lidar_channel_counter[num_channels] += 1

        if "annos" in info:
            frames_with_annos += 1
            annos = info["annos"]
            names = annos.get("name", [])
            boxes = annos.get("gt_boxes_lidar", np.empty((0, 7), dtype=np.float32))
            num_points = annos.get("num_points_in_gt", np.empty((0,), dtype=np.int64))
            for name, box, pts in zip(names, boxes, num_points):
                class_to_boxes[name].append(box)
                class_to_points[name].append(int(pts))

        if with_point_stats:
            sample_idx = pc_meta.get("lidar_idx")
            if sample_idx is None:
                continue
            lidar_path = data_root / "points" / f"{sample_idx}.npy"
            if not lidar_path.exists():
                continue
            points = np.load(lidar_path, mmap_mode="r")
            points_per_frame.append(points.shape[0])
            if points.shape[1] > 3:
                intensity_values.extend(points[:, 3].tolist())

    total_boxes = sum(len(v) for v in class_to_boxes.values())
    class_distribution = {name: len(samples) for name, samples in class_to_boxes.items()}

    bbox_stats = {}
    for name, boxes in class_to_boxes.items():
        if not boxes:
            bbox_stats[name] = {}
            continue
        box_array = np.vstack(boxes)
        bbox_stats[name] = {
            "dimensions": {
                "length": compute_numeric_summary(box_array[:, 3]),
                "width": compute_numeric_summary(box_array[:, 4]),
                "height": compute_numeric_summary(box_array[:, 5]),
            },
            "center": {
                "x": compute_numeric_summary(box_array[:, 0]),
                "y": compute_numeric_summary(box_array[:, 1]),
                "z": compute_numeric_summary(box_array[:, 2]),
            },
            "yaw": compute_numeric_summary(box_array[:, 6]),
            "points_in_box": compute_numeric_summary(class_to_points[name]),
        }

    split_summary = {
        "num_frames": num_frames,
        "frames_with_annos": frames_with_annos,
        "boxes_per_frame_mean": float(total_boxes / num_frames) if num_frames else 0.0,
        "class_distribution": class_distribution,
        "bbox_stats": bbox_stats,
        "lidar_types": dict(lidar_type_counter),
        "num_channels": dict(lidar_channel_counter),
    }

    if with_point_stats and points_per_frame:
        split_summary["points_per_frame"] = compute_numeric_summary(points_per_frame)
        if intensity_values:
            split_summary["intensity"] = compute_numeric_summary(intensity_values)

    raw_data = None
    if collect_raw:
        raw_data = {
            "class_boxes": {
                name: (np.vstack(boxes) if boxes else np.empty((0, 7), dtype=np.float32))
                for name, boxes in class_to_boxes.items()
            },
            "class_points": {
                name: np.asarray(points, dtype=np.int32)
                for name, points in class_to_points.items()
            },
            "points_per_frame": np.asarray(points_per_frame, dtype=np.int64)
            if points_per_frame
            else np.empty((0,), dtype=np.int64),
            "intensity": np.asarray(intensity_values, dtype=np.float32)
            if intensity_values
            else np.empty((0,), dtype=np.float32),
        }

    return split_summary, raw_data


def format_summary(split: str, summary: Dict) -> str:
    lines: List[str] = []
    lines.append(f"Split: {split}")
    lines.append(f"  Frames: {summary['num_frames']} (with labels: {summary['frames_with_annos']})")
    lines.append(f"  Boxes / frame (mean): {summary['boxes_per_frame_mean']:.2f}")

    class_distribution = summary.get("class_distribution", {})
    if class_distribution:
        total = sum(class_distribution.values())
        lines.append("  Class distribution:")
        for cls_name, count in sorted(class_distribution.items(), key=lambda x: x[0]):
            ratio = (count / total * 100.0) if total else 0.0
            lines.append(f"    - {cls_name}: {count} ({ratio:.2f}%)")

    lidar_types = summary.get("lidar_types", {})
    if lidar_types:
        lines.append("  LiDAR types:")
        for lidar, count in sorted(lidar_types.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"    - {lidar}: {count}")

    num_channels = summary.get("num_channels", {})
    if num_channels:
        lines.append("  Channel counts:")
        for ch, count in sorted(num_channels.items()):
            lines.append(f"    - {ch}: {count}")

    if "points_per_frame" in summary:
        stats = summary["points_per_frame"]
        lines.append(
            "  Points per frame: mean={mean:.0f}, 50%={median:.0f}, max={max:.0f}".format(
                mean=stats.get("mean", 0.0),
                median=stats.get("50%", 0.0),
                max=stats.get("max", 0.0),
            )
        )

    lines.append("  Bounding box stats per class:")
    bbox_stats = summary.get("bbox_stats", {})
    for cls_name, stats in sorted(bbox_stats.items(), key=lambda x: x[0]):
        dims = stats.get("dimensions", {})
        pts = stats.get("points_in_box", {})
        lines.append(f"    * {cls_name}")
        if dims:
            length_stats = dims.get("length", {})
            width_stats = dims.get("width", {})
            height_stats = dims.get("height", {})
            if length_stats:
                lines.append(
                    "      dims (mean L/W/H): {l:.2f} / {w:.2f} / {h:.2f}".format(
                        l=length_stats.get("mean", 0.0),
                        w=width_stats.get("mean", 0.0),
                        h=height_stats.get("mean", 0.0),
                    )
                )
        if pts:
            lines.append(
                "      points per box: mean={mean:.0f}, 25%={p25:.0f}, 75%={p75:.0f}".format(
                    mean=pts.get("mean", 0.0),
                    p25=pts.get("25%", 0.0),
                    p75=pts.get("75%", 0.0),
                )
            )
    return "\n".join(lines)


def sanitize_name(name: str) -> str:
    return name.lower().replace(" ", "_")


def render_plots(split: str, summary: Dict, raw: Dict | None, plot_dir: Path, dpi: int) -> None:
    if raw is None:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"[WARN] matplotlib is required for plotting ({exc}). Skipping plots for {split}.")
        return
    except Exception as exc:
        print(f"[WARN] matplotlib failed to initialize ({exc}). Skipping plots for {split}.")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    def save_hist(data: np.ndarray, title: str, xlabel: str, filename: Path, bins: int = 30) -> None:
        if data.size == 0:
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(data, bins=bins, color="#2E86AB", edgecolor="#1B4F72")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(filename, dpi=dpi)
        plt.close(fig)

    class_distribution = summary.get("class_distribution", {})
    if class_distribution:
        classes = sorted(class_distribution)
        counts = [class_distribution[c] for c in classes]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(classes, counts, color="#27AE60", edgecolor="#1E8449")
        ax.set_title(f"{split} class distribution")
        ax.set_ylabel("Box count")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{split}_class_distribution.png", dpi=dpi)
        plt.close(fig)

    for cls_name, boxes in raw.get("class_boxes", {}).items():
        if boxes.size == 0:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        dims = ["length", "width", "height"]
        idxs = [3, 4, 5]
        for ax, dim, idx in zip(axes, dims, idxs):
            ax.hist(boxes[:, idx], bins=40, color="#F39C12", edgecolor="#B9770E")
            ax.set_title(f"{dim}")
            ax.set_xlabel("meters")
            ax.set_ylabel("Count")
        fig.suptitle(f"{split} {cls_name} box dimensions")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{split}_{sanitize_name(cls_name)}_box_dims.png", dpi=dpi)
        plt.close(fig)

        points = raw.get("class_points", {}).get(cls_name, np.empty((0,), dtype=np.int32))
        save_hist(
            points,
            title=f"{split} {cls_name} points per box",
            xlabel="Point count",
            filename=plot_dir / f"{split}_{sanitize_name(cls_name)}_points_per_box.png",
            bins=40,
        )

    save_hist(
        raw.get("points_per_frame", np.empty((0,), dtype=np.int64)),
        title=f"{split} points per frame",
        xlabel="Point count",
        filename=plot_dir / f"{split}_points_per_frame.png",
        bins=40,
    )

    intensity = raw.get("intensity", np.empty((0,), dtype=np.float32))
    save_hist(
        intensity,
        title=f"{split} intensity distribution",
        xlabel="Intensity",
        filename=plot_dir / f"{split}_intensity.png",
        bins=60,
    )


def main() -> None:
    args = parse_args()
    collect_raw = args.plot_dir is not None

    all_summaries: List[Tuple[str, Dict, Dict | None]] = []

    for split in args.splits:
        info_path = args.data_root / f"custom_av_infos_{split}.pkl"
        infos = load_infos(info_path)
        summary, raw = summarize_split(
            infos=infos,
            data_root=args.data_root,
            limit=args.limit,
            with_point_stats=args.point_stats,
            collect_raw=collect_raw,
        )
        all_summaries.append((split, summary, raw))

    for split, summary, raw in all_summaries:
        print(format_summary(split, summary))
        print()
        if args.plot_dir is not None:
            render_plots(split, summary, raw, args.plot_dir, args.plot_dpi)


if __name__ == "__main__":
    main()

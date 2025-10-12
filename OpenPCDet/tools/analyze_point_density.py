#!/usr/bin/env python3
"""
Utility to inspect point density statistics for custom_av point clouds.

Example:
    python tools/analyze_point_density.py \
        --root data/custom_av/points_test --bin-size 1.0 --top-k 10 --save-npz density_map.npz
"""
from __future__ import annotations

import argparse
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def _iter_point_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.glob("*.npy")):
        if path.is_file():
            yield path


def analyze_density(
    point_files: Iterable[Path],
    bin_size: float,
    max_files: int | None = None,
) -> dict:
    grid_counter: Counter[Tuple[int, int]] = Counter()
    per_frame_counts: list[int] = []
    per_frame_bounds: list[Tuple[float, float, float, float]] = []

    x_min = y_min = z_min = float("inf")
    x_max = y_max = z_max = float("-inf")

    processed = 0
    for path in point_files:
        if max_files is not None and processed >= max_files:
            break
        points = np.load(path)  # (N, 4) -> x, y, z, intensity
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Unexpected shape {points.shape} in {path}")

        processed += 1
        num_points = int(points.shape[0])
        per_frame_counts.append(num_points)

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        x_min = min(x_min, float(x.min()))
        y_min = min(y_min, float(y.min()))
        z_min = min(z_min, float(z.min()))

        x_max = max(x_max, float(x.max()))
        y_max = max(y_max, float(y.max()))
        z_max = max(z_max, float(z.max()))

        per_frame_bounds.append((float(x.min()), float(x.max()), float(y.min()), float(y.max())))

        if bin_size > 0.0:
            xy_bins = np.floor(points[:, :2] / bin_size).astype(np.int64)
            unique_bins, counts = np.unique(xy_bins, axis=0, return_counts=True)
            for key, count in zip(unique_bins, counts):
                grid_counter[(int(key[0]), int(key[1]))] += int(count)

    if processed == 0:
        raise RuntimeError("No point cloud files were processed. Check the dataset path.")

    total_points = sum(per_frame_counts)
    occupied_cells = len(grid_counter)
    bin_area = bin_size * bin_size if bin_size > 0 else None

    occupied_area = occupied_cells * bin_area if bin_area is not None else None
    total_area_bbox = (x_max - x_min) * (y_max - y_min)

    density_per_cell = None
    if occupied_cells > 0:
        density_per_cell = total_points / occupied_cells

    density_per_square_meter = None
    per_frame_density_per_square_meter = None
    if occupied_area and occupied_area > 0:
        density_per_square_meter = total_points / occupied_area
        per_frame_density_per_square_meter = density_per_square_meter / processed

    result = {
        "frames": processed,
        "total_points": total_points,
        "per_frame_counts": per_frame_counts,
        "per_frame_bounds": per_frame_bounds,
        "bbox": {
            "x": [x_min, x_max],
            "y": [y_min, y_max],
            "z": [z_min, z_max],
        },
        "grid_counter": grid_counter,
        "bin_size": bin_size,
        "occupied_cells": occupied_cells,
        "occupied_area": occupied_area,
        "bounding_box_area": total_area_bbox,
        "density_per_cell": density_per_cell,
        "density_per_square_meter": density_per_square_meter,
        "per_frame_density_per_square_meter": per_frame_density_per_square_meter,
    }
    return result


def summarize(result: dict, top_k: int) -> str:
    per_frame_counts = result["per_frame_counts"]
    frames = result["frames"]
    counter: Counter = result["grid_counter"]

    stats_lines = [
        f"Frames processed              : {frames}",
        f"Total points                  : {result['total_points']:,}",
        f"Points per frame (mean ± std) : {statistics.mean(per_frame_counts):.1f} ± {statistics.pstdev(per_frame_counts):.1f}",
        f"Points per frame (min / max)  : {min(per_frame_counts)} / {max(per_frame_counts)}",
    ]

    bbox = result["bbox"]
    stats_lines.append(
        "XYZ extent (min → max)        : "
        f"x [{bbox['x'][0]:.2f}, {bbox['x'][1]:.2f}], "
        f"y [{bbox['y'][0]:.2f}, {bbox['y'][1]:.2f}], "
        f"z [{bbox['z'][0]:.2f}, {bbox['z'][1]:.2f}]"
    )

    if result["occupied_cells"] > 0:
        stats_lines.append(
            f"Occupied XY cells             : {result['occupied_cells']} "
            f"(bin size {result['bin_size']} m)"
        )
    if result["occupied_area"]:
        stats_lines.append(
            f"Occupied XY area              : {result['occupied_area']:.2f} m² "
            f"(density {result['density_per_square_meter']:.1f} pts/m² total, "
            f"{result['per_frame_density_per_square_meter']:.1f} pts/m²/frame)"
        )
    stats_lines.append(
        f"XY bounding box area          : {result['bounding_box_area']:.2f} m²"
    )

    if counter and top_k > 0:
        stats_lines.append(f"Top {top_k} densest XY cells (bin indices, counts):")
        for (ix, iy), count in counter.most_common(top_k):
            stats_lines.append(
                f"  bin ({ix}, {iy}) -> {count} pts (center ~ [{(ix + 0.5) * result['bin_size']:.2f}, {(iy + 0.5) * result['bin_size']:.2f}])"
            )

    return "\n".join(stats_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze XY density for .npy point cloud files.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/custom_av/points_test"),
        help="Directory that contains .npy point cloud files.",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=1.0,
        help="XY grid resolution in meters used to estimate density.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on the number of files to process.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many of the densest cells to list in the summary.",
    )
    parser.add_argument(
        "--save-npz",
        type=Path,
        default=None,
        help="Optional path to save the accumulated XY density map as an .npz archive.",
    )

    args = parser.parse_args()
    root = args.root
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory {root} does not exist.")

    result = analyze_density(
        point_files=_iter_point_files(root),
        bin_size=args.bin_size,
        max_files=args.max_files,
    )
    print(summarize(result, args.top_k))

    if args.save_npz:
        counter: Counter = result["grid_counter"]
        if counter:
            indices = np.array(list(counter.keys()), dtype=np.int64)
            counts = np.array(list(counter.values()), dtype=np.int64)
        else:
            indices = np.zeros((0, 2), dtype=np.int64)
            counts = np.zeros(0, dtype=np.int64)
        np.savez(
            args.save_npz,
            indices=indices,
            counts=counts,
            bin_size=result["bin_size"],
            bbox=np.array(
                [
                    result["bbox"]["x"],
                    result["bbox"]["y"],
                    result["bbox"]["z"],
                ],
                dtype=np.float32,
            ),
            frames=result["frames"],
        )
        print(f"Saved density map to {args.save_npz}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze number of points per frame and per object.

Per-frame: counts points in each .npy point cloud file under a directory.
Per-object: aggregates `num_points_in_gt` from the ground-truth database info
            (created by OpenPCDet's create_groundtruth_database) for training.

Examples:
  - Per-frame on test split:
      python Subin/OpenPCDet/tools/analyze_points_per_frame_object.py \
          --frame-root Subin/OpenPCDet/data/custom_av/points_test \
          --save-csv-frame Subin/OpenPCDet/tools/points_per_frame.csv

  - Per-object (training db infos):
      python Subin/OpenPCDet/tools/analyze_points_per_frame_object.py \
          --dbinfo Subin/OpenPCDet/data/custom_av/custom_av_dbinfos_train.pkl \
          --save-csv-object Subin/OpenPCDet/tools/points_per_object.csv

  - Do both in one run (defaults already match these paths):
      python Subin/OpenPCDet/tools/analyze_points_per_frame_object.py
"""
from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _iter_point_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.glob("*.npy")):
        if path.is_file():
            yield path


@dataclass
class FrameStats:
    frames: int
    total_points: int
    per_frame_counts: List[int]

    @property
    def mean(self) -> float:
        return statistics.mean(self.per_frame_counts) if self.per_frame_counts else float("nan")

    @property
    def std(self) -> float:
        return statistics.pstdev(self.per_frame_counts) if len(self.per_frame_counts) > 1 else 0.0

    @property
    def min(self) -> int:
        return min(self.per_frame_counts) if self.per_frame_counts else 0

    @property
    def max(self) -> int:
        return max(self.per_frame_counts) if self.per_frame_counts else 0


def analyze_per_frame(root: Path, max_files: Optional[int] = None) -> FrameStats:
    counts: List[int] = []
    processed = 0
    for p in _iter_point_files(root):
        if max_files is not None and processed >= max_files:
            break
        pts = np.load(p)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"Unexpected point shape {pts.shape} in {p}")
        counts.append(int(pts.shape[0]))
        processed += 1
    if processed == 0:
        raise RuntimeError(f"No .npy point clouds found in {root}")
    return FrameStats(frames=processed, total_points=sum(counts), per_frame_counts=counts)


@dataclass
class ObjectStats:
    # Mapping class name -> list of per-object point counts
    per_class_counts: Dict[str, List[int]]

    @property
    def total_objects(self) -> int:
        return sum(len(v) for v in self.per_class_counts.values())

    def summary_rows(self) -> List[Tuple[str, int, float, float, int, int]]:
        rows = []
        for name, counts in sorted(self.per_class_counts.items()):
            if not counts:
                rows.append((name, 0, float("nan"), float("nan"), 0, 0))
                continue
            mean = float(statistics.mean(counts))
            std = float(statistics.pstdev(counts)) if len(counts) > 1 else 0.0
            rows.append((name, len(counts), mean, std, min(counts), max(counts)))
        return rows


def analyze_per_object_dbinfo(dbinfo_path: Path, max_objects: Optional[int] = None) -> ObjectStats:
    import pickle

    if not dbinfo_path.exists():
        raise FileNotFoundError(f"DB infos not found: {dbinfo_path}")

    with dbinfo_path.open("rb") as f:
        db = pickle.load(f)

    per_class: Dict[str, List[int]] = {}
    for cls_name, entries in db.items():
        counts: List[int] = []
        limit = max_objects if max_objects is not None else len(entries)
        for i, it in enumerate(entries):
            if i >= limit:
                break
            n = int(it.get("num_points_in_gt", 0))
            counts.append(n)
        per_class[str(cls_name)] = counts

    return ObjectStats(per_class_counts=per_class)


def save_frame_csv(stats: FrameStats, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "num_points"])
        for i, n in enumerate(stats.per_frame_counts):
            w.writerow([i, n])


def save_object_csv(stats: ObjectStats, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "num_points_in_gt"])
        for cls, counts in stats.per_class_counts.items():
            for n in counts:
                w.writerow([cls, n])


def describe_frame_stats(stats: FrameStats) -> str:
    return (
        "\n".join(
            [
                f"Frames processed       : {stats.frames}",
                f"Total points           : {stats.total_points:,}",
                f"Points/frame mean±std  : {stats.mean:.1f} ± {stats.std:.1f}",
                f"Points/frame min / max : {stats.min} / {stats.max}",
            ]
        )
    )


def describe_object_stats(stats: ObjectStats) -> str:
    lines = [f"Total objects: {stats.total_objects:,}", "Per-class summary:"]
    lines.append("  class         count    mean    std    min    max")
    for name, cnt, mean, std, mn, mx in stats.summary_rows():
        lines.append(f"  {name:<12s} {cnt:6d}  {mean:7.1f} {std:7.1f}  {mn:5d} {mx:6d}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze number of points per frame and per object.")
    parser.add_argument(
        "--frame-root",
        type=Path,
        default=Path("../data/custom_av/points_test"),
        help="Directory of .npy point clouds for per-frame stats.",
    )
    parser.add_argument(
        "--dbinfo",
        type=Path,
        default=Path("../data/custom_av/custom_av_dbinfos_train.pkl"),
        help="Path to *_dbinfos_train.pkl for per-object stats.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames for per-frame stats.")
    parser.add_argument("--max-objects", type=int, default=None, help="Limit number of objects per class for per-object stats.")
    parser.add_argument("--save-csv-frame", type=Path, default=None, help="Optional CSV path to save per-frame counts.")
    parser.add_argument("--save-csv-object", type=Path, default=None, help="Optional CSV path to save per-object counts.")

    args = parser.parse_args()

    # Per-frame
    if args.frame_root.exists():
        frame_stats = analyze_per_frame(args.frame_root, max_files=args.max_frames)
        print("[Per-frame]")
        print(describe_frame_stats(frame_stats))
        if args.save_csv_frame is not None:
            save_frame_csv(frame_stats, args.save_csv_frame)
            print(f"Saved per-frame CSV to {args.save_csv_frame}")
    else:
        print(f"[Per-frame] Skip: {args.frame_root} does not exist")

    # Per-object (from db infos)
    if args.dbinfo.exists():
        obj_stats = analyze_per_object_dbinfo(args.dbinfo, max_objects=args.max_objects)
        print("\n[Per-object]")
        print(describe_object_stats(obj_stats))
        if args.save_csv_object is not None:
            save_object_csv(obj_stats, args.save_csv_object)
            print(f"Saved per-object CSV to {args.save_csv_object}")
    else:
        print(f"[Per-object] Skip: {args.dbinfo} does not exist")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Generate missing label files for custom_av_hybrid using existing info PKLs.

Use-case: points_lisa frames (e.g., IDs like 00018810) have no label files,
but custom_av_infos_val.pkl contains their annotations. This script writes
labels/<id>.txt lines in the format expected by the dataset:

  x y z l w h angle class_name

It only creates labels for IDs that are missing on disk and present in the PKL.

Usage:
  python OpenPCDet/tools/generate_missing_labels_from_pkl.py \
    --data-root OpenPCDet/data --pkl custom_av_infos_val.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate missing labels from PKL infos")
    p.add_argument("--data-root", type=Path, default=Path("OpenPCDet/data"))
    p.add_argument("--pkl", type=str, default="custom_av_infos_val.pkl",
                   help="Info PKL filename under custom_av_hybrid (train/val)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def write_label_file(path: Path, boxes, names, dry: bool) -> None:
    # boxes: (N, 7) [x,y,z,l,w,h,angle]
    # names: (N,)
    if dry:
        print(f"[dry-run] write {path} with {len(names)} objects")
        return
    with open(path, "w") as f:
        for i in range(len(names)):
            x, y, z, l, w, h, a = boxes[i]
            name = names[i]
            f.write(f"{x} {y} {z} {l} {w} {h} {a} {name}\n")


def main() -> int:
    args = parse_args()
    root = args.data_root.resolve()
    hyb = (root / "custom_av_hybrid").resolve()
    labels_dir = hyb / "labels"
    imagesets = hyb / "ImageSets"
    pkl_path = hyb / args.pkl

    if not pkl_path.exists():
        print(f"ERROR: PKL not found: {pkl_path}")
        return 1
    if not labels_dir.exists():
        print(f"ERROR: labels dir not found: {labels_dir}")
        return 1

    # Build target ID list from val.txt (typical missing labels live here)
    val_list_path = imagesets / "val.txt"
    ids_target: list[str] = []
    if val_list_path.exists():
        with open(val_list_path, "r") as f:
            ids_target = [ln.strip() for ln in f if ln.strip()]
    else:
        print("WARNING: ImageSets/val.txt not found; will scan labels dir for missing ones")

    # Load infos
    with open(pkl_path, "rb") as f:
        infos = pickle.load(f)

    # Build index: id -> (boxes, names)
    index: dict[str, tuple] = {}
    for info in infos:
        if 'point_cloud' not in info:
            continue
        fid = str(info['point_cloud']['lidar_idx'])
        ann = info.get('annos')
        if ann is None:
            continue
        boxes = ann.get('gt_boxes_lidar')
        names = ann.get('name')
        if boxes is None or names is None:
            continue
        index[fid] = (boxes, names)

    created = 0
    missing_in_pkl = 0
    already = 0
    for fid in ids_target:
        out = labels_dir / f"{fid}.txt"
        if out.exists():
            already += 1
            continue
        item = index.get(fid)
        if item is None:
            missing_in_pkl += 1
            continue
        boxes, names = item
        write_label_file(out, boxes, names, args.dry_run)
        created += 1

    print(
        f"Created {created} label files; "
        f"already existed {already}; missing-in-pkl {missing_in_pkl}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


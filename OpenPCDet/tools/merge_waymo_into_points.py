#!/usr/bin/env python3
"""
Merge waymo_npy into custom_av_hybrid/points by renaming to unique 8-digit IDs,
and update ImageSets/val.txt to include both points_lisa and the renamed waymo IDs.

Strategy
- Keep existing train as all points_new (already moved to points)
- Keep existing val lisa IDs
- For every waymo_npy/<id>.npy, copy to points/<offset+id>.npy where offset is 80,000,000
  Also copy labels from custom_av_64/labels/<id>.txt to labels/<offset+id>.txt if available.

Usage: python OpenPCDet/tools/merge_waymo_into_points.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

OFFSET = 80_000_000  # produces 8-digit ids like 80000000+


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge waymo_npy into points with renamed IDs")
    p.add_argument("--data-root", type=Path, default=Path("OpenPCDet/data"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def write_list(path: Path, items: list[str], dry: bool) -> None:
    if dry:
        print(f"[dry-run] write {path} ({len(items)} lines)")
        return
    with open(path, "w") as f:
        for x in items:
            f.write(f"{x}\n")


def main() -> int:
    args = parse_args()
    root = args.data_root.resolve()
    hyb = (root / "custom_av_hybrid").resolve()
    points = hyb / "points"
    labels = hyb / "labels"
    imagesets = hyb / "ImageSets"
    waymo = hyb / "waymo_npy"
    lisa = hyb / "points_lisa"
    src64_labels = (root / "custom_av_64" / "labels").resolve()

    for p in [points, labels, imagesets, waymo]:
        if not p.exists():
            print(f"ERROR: missing path: {p}")
            return 1

    # Build current val list (prefer existing file)
    val_ids = []
    val_path = imagesets / "val.txt"
    if val_path.exists():
        with open(val_path, "r") as f:
            val_ids = [ln.strip() for ln in f if ln.strip()]
    else:
        # derive lisa ids directly if file missing
        if lisa.exists():
            val_ids = sorted(p.stem for p in lisa.glob("*.npy"))

    # Add waymo with offset renaming
    added_points = 0
    added_labels = 0
    renamed_ids: list[str] = []
    for p in sorted(waymo.glob("*.npy")):
        orig = p.stem
        try:
            new_id = f"{int(orig) + OFFSET:08d}"
        except ValueError:
            # Non-numeric id, prefix with '8' and pad/truncate to keep unique
            new_id = f"8{orig}"[-8:]
        dst_point = points / f"{new_id}.npy"
        if dst_point.exists():
            # extremely unlikely; skip to avoid overwrite
            continue
        if args.dry_run:
            print(f"[dry-run] copy {p.name} -> {dst_point.name}")
        else:
            shutil.copy2(p, dst_point)
        added_points += 1
        renamed_ids.append(new_id)

        # labels
        src_lab = src64_labels / f"{orig}.txt"
        if src_lab.exists():
            dst_lab = labels / f"{new_id}.txt"
            if args.dry_run:
                print(f"[dry-run] copy {src_lab.name} -> {dst_lab.name}")
            else:
                shutil.copy2(src_lab, dst_lab)
            added_labels += 1

    # Update val list: keep existing lisa ids, append renamed waymo
    new_val_ids = sorted(set(val_ids) | set(renamed_ids))
    write_list(val_path, new_val_ids, args.dry_run)

    print(
        f"Waymo merged: points+{added_points}, labels+{added_labels}. "
        f"New val size: {len(new_val_ids)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


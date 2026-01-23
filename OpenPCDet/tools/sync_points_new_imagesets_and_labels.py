#!/usr/bin/env python3
"""
Sync ImageSets and labels for points_new (1/3 LISA mix).

This script:
  1) Creates domain-aware ImageSets that match OpenPCDet/data/custom_av_hybrid/points_new
     using the original split from custom_av_64/ImageSets/{train,val}.txt
  2) Copies labels to OpenPCDet/data/custom_av_hybrid/labels_new with the same IDs
     present in points_new.

Domain assignment follows the same deterministic rule used when creating points_new:
  - Enumerate sorted custom_av_64/points files; for index i, pick LISA if
    (i % 3 == 0) and matching custom_av_64/points_lisa file exists; otherwise CLEAN.

Outputs under OpenPCDet/data/custom_av_hybrid/ImageSets:
  - train_points_new.txt, val_points_new.txt
  - train_clean_points_new.txt, train_lisa_points_new.txt
  - val_clean_points_new.txt, val_lisa_points_new.txt
  - domain_map_points_new.txt

Usage (from repo root):
  python OpenPCDet/tools/sync_points_new_imagesets_and_labels.py

Optional args:
  --data-root   Root folder containing custom_av_64 and custom_av_hybrid (default: OpenPCDet/data)
  --dry-run     Print planned actions without writing files
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sync ImageSets/labels for points_new")
    p.add_argument(
        "--data-root", type=Path, default=Path("OpenPCDet/data"),
        help="Root containing custom_av_64 and custom_av_hybrid (default: OpenPCDet/data)",
    )
    p.add_argument("--dry-run", action="store_true", help="Do not write files")
    return p.parse_args()


def load_id_list(txt_path: Path) -> list[str]:
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main() -> int:
    args = parse_args()
    root = args.data_root.resolve()

    src_points = (root / "custom_av_64" / "points").resolve()
    src_points_lisa = (root / "custom_av_64" / "points_lisa").resolve()
    src_imagesets = (root / "custom_av_64" / "ImageSets").resolve()

    hyb = (root / "custom_av_hybrid").resolve()
    points_new = (hyb / "points_new").resolve()
    # Use original labels from custom_av_64 to align with original IDs
    labels_src = (root / "custom_av_64" / "labels").resolve()
    labels_new = (hyb / "labels_new").resolve()
    imagesets_out = (hyb / "ImageSets").resolve()

    # Basic checks
    for p in [src_points, src_points_lisa, src_imagesets, points_new, labels_src, imagesets_out]:
        if not p.exists():
            print(f"ERROR: required path not found: {p}")
            return 1

    # Map id -> domain (clean|lisa) by reproducing the selection rule
    sorted_clean_files = sorted(src_points.glob("*.npy"))
    id_to_domain: dict[str, str] = {}
    total = 0
    lisa_count = 0
    for i, clean_path in enumerate(sorted_clean_files):
        fid = clean_path.stem  # '00000000'
        lisa_path = src_points_lisa / f"{fid}.npy"
        is_lisa = (i % 3 == 0) and lisa_path.exists()
        id_to_domain[fid] = "lisa" if is_lisa else "clean"
        total += 1
        if is_lisa:
            lisa_count += 1

    print(f"Domain map prepared for {total} ids (lisa={lisa_count}, clean={total - lisa_count})")

    # IDs actually present in points_new
    ids_points_new = sorted(p.stem for p in points_new.glob("*.npy"))
    if not ids_points_new:
        print(f"WARNING: no .npy files in {points_new}")
        return 0

    # Load original split from custom_av_64
    train_ids_src = set(load_id_list(src_imagesets / "train.txt"))
    val_ids_src = set(load_id_list(src_imagesets / "val.txt"))

    # Sanity: split coverage
    missing_in_split = [fid for fid in ids_points_new if fid not in train_ids_src and fid not in val_ids_src]
    if missing_in_split:
        print(f"Note: {len(missing_in_split)} ids in points_new are not in train/val split; assigning to train.")

    # Build new lists limited to points_new ids
    train_ids = []
    val_ids = []
    train_clean = []
    train_lisa = []
    val_clean = []
    val_lisa = []

    for fid in ids_points_new:
        assign_to = "train" if fid in train_ids_src or fid in missing_in_split else "val"
        domain = id_to_domain.get(fid, "clean")
        if assign_to == "train":
            train_ids.append(fid)
            (train_lisa if domain == "lisa" else train_clean).append(fid)
        else:
            val_ids.append(fid)
            (val_lisa if domain == "lisa" else val_clean).append(fid)

    # Write ImageSets files
    def write_list(path: Path, items: list[str]) -> None:
        if args.dry_run:
            print(f"[dry-run] write {path} ({len(items)} lines)")
            return
        with open(path, "w") as f:
            for x in items:
                f.write(f"{x}\n")

    write_list(imagesets_out / "train_points_new.txt", train_ids)
    write_list(imagesets_out / "val_points_new.txt", val_ids)
    write_list(imagesets_out / "train_clean_points_new.txt", train_clean)
    write_list(imagesets_out / "train_lisa_points_new.txt", train_lisa)
    write_list(imagesets_out / "val_clean_points_new.txt", val_clean)
    write_list(imagesets_out / "val_lisa_points_new.txt", val_lisa)

    # Domain map file
    if args.dry_run:
        print(f"[dry-run] write {imagesets_out / 'domain_map_points_new.txt'}")
    else:
        with open(imagesets_out / "domain_map_points_new.txt", "w") as f:
            for fid in ids_points_new:
                f.write(f"{fid} {id_to_domain.get(fid, 'clean')}\n")

    # Copy labels to labels_new
    labels_new.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = 0
    for fid in ids_points_new:
        src = labels_src / f"{fid}.txt"
        dst = labels_new / f"{fid}.txt"
        if not src.exists():
            missing += 1
            continue
        if args.dry_run:
            print(f"[dry-run] copy {src.name} -> labels_new/")
        else:
            shutil.copy2(src, dst)
        copied += 1

    print(
        f"ImageSets (points_new) written. Labels copied: {copied} files"
        + (f", missing labels: {missing}" if missing else "")
    )

    # Summary
    print(
        "SUMMARY: "
        f"train={len(train_ids)}, val={len(val_ids)}, "
        f"train_clean={len(train_clean)}, train_lisa={len(train_lisa)}, "
        f"val_clean={len(val_clean)}, val_lisa={len(val_lisa)}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

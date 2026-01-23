#!/usr/bin/env python3
"""
Rebase custom_av_hybrid to use points_new as the canonical points folder and
create new train/val splits:

  - Train: all frames from points_new
  - Val:   all frames from points_lisa and waymo_npy (minus any overlap with train)

Operations:
  1) Remove existing points and labels under custom_av_hybrid
  2) Move points_new -> points, labels_new -> labels
  3) Copy val point clouds into points and corresponding labels into labels
  4) Rewrite ImageSets/train.txt and val.txt

Assumptions:
  - Labels for points_new and val frames are available in custom_av_64/labels

Usage (from repo root):
  python OpenPCDet/tools/rebase_to_points_new_and_resplit.py

Optional args:
  --data-root  Base data dir (default: OpenPCDet/data)
  --dry-run    Print planned actions only
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebase to points_new and resplit train/val")
    p.add_argument("--data-root", type=Path, default=Path("OpenPCDet/data"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_ids_from_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]


def write_id_list(path: Path, ids: list[str], dry: bool) -> None:
    if dry:
        print(f"[dry-run] write {path} ({len(ids)} lines)")
        return
    with open(path, "w") as f:
        for fid in ids:
            f.write(f"{fid}\n")


def main() -> int:
    args = parse_args()
    root = args.data_root.resolve()

    src64_labels = (root / "custom_av_64" / "labels").resolve()

    hyb = (root / "custom_av_hybrid").resolve()
    hyb_points = hyb / "points"
    hyb_labels = hyb / "labels"
    hyb_imagesets = hyb / "ImageSets"
    hyb_points_new = hyb / "points_new"
    hyb_labels_new = hyb / "labels_new"
    hyb_points_lisa = hyb / "points_lisa"
    hyb_waymo = hyb / "waymo_npy"

    # Checks
    for p in [hyb_imagesets, hyb_points_new, hyb_labels_new, src64_labels]:
        if not p.exists():
            print(f"ERROR: missing required path: {p}")
            return 1
    if not hyb_points_lisa.exists():
        print(f"WARNING: points_lisa not found: {hyb_points_lisa}")
    if not hyb_waymo.exists():
        print(f"WARNING: waymo_npy not found: {hyb_waymo}")

    # Gather train ids (from points_new): use union of train/val helper lists to include ALL points_new
    train_ids = []
    tpn = load_ids_from_file(hyb_imagesets / "train_points_new.txt")
    vpn = load_ids_from_file(hyb_imagesets / "val_points_new.txt")
    if tpn or vpn:
        train_ids = sorted(set(tpn) | set(vpn))
    else:
        train_ids = sorted(p.stem for p in hyb_points_new.glob("*.npy"))

    # Gather val candidate ids from points_lisa and waymo_npy
    val_ids_set = set()
    if hyb_points_lisa.exists():
        val_ids_set.update(p.stem for p in hyb_points_lisa.glob("*.npy"))
    if hyb_waymo.exists():
        val_ids_set.update(p.stem for p in hyb_waymo.glob("*.npy"))

    # Exclude overlaps with train to avoid collisions
    train_set = set(train_ids)
    val_ids = sorted(fid for fid in val_ids_set if fid not in train_set)

    print(f"Train count (points_new, all): {len(train_ids)}")
    print(f"Val candidates (points_lisa+waymo): {len(val_ids_set)} -> after excluding overlaps: {len(val_ids)}")

    # 1) Remove old points/labels
    if hyb_points.exists():
        if args.dry_run:
            print(f"[dry-run] remove {hyb_points}")
        else:
            shutil.rmtree(hyb_points)
    if hyb_labels.exists():
        if args.dry_run:
            print(f"[dry-run] remove {hyb_labels}")
        else:
            shutil.rmtree(hyb_labels)

    # 2) Move points_new -> points, labels_new -> labels
    if args.dry_run:
        print(f"[dry-run] move {hyb_points_new} -> {hyb_points}")
        print(f"[dry-run] move {hyb_labels_new} -> {hyb_labels}")
    else:
        hyb_points_new.rename(hyb_points)
        hyb_labels_new.rename(hyb_labels)

    # 3) Copy val frames into points and labels
    copied_points = 0
    copied_labels = 0
    for fid in val_ids:
        dst_p = hyb_points / f"{fid}.npy"
        if dst_p.exists():
            # Should not happen due to overlap exclusion
            continue
        src_p = None
        cand1 = hyb_points_lisa / f"{fid}.npy"
        cand2 = hyb_waymo / f"{fid}.npy"
        if cand1.exists():
            src_p = cand1
        elif cand2.exists():
            src_p = cand2
        else:
            print(f"WARNING: no source point found for {fid}")
            continue
        if args.dry_run:
            print(f"[dry-run] copy {src_p} -> {dst_p}")
        else:
            shutil.copy2(src_p, dst_p)
        copied_points += 1

        # labels
        src_l = src64_labels / f"{fid}.txt"
        dst_l = hyb_labels / f"{fid}.txt"
        if src_l.exists():
            if args.dry_run:
                print(f"[dry-run] copy {src_l} -> {dst_l}")
            else:
                shutil.copy2(src_l, dst_l)
            copied_labels += 1
        else:
            print(f"WARNING: missing label for {fid}")

    print(f"Copied val points: {copied_points}, val labels: {copied_labels}")

    # 4) Rewrite ImageSets/train.txt and val.txt
    hyb_imagesets.mkdir(parents=True, exist_ok=True)
    write_id_list(hyb_imagesets / "train.txt", train_ids, args.dry_run)
    write_id_list(hyb_imagesets / "val.txt", val_ids, args.dry_run)
    print("ImageSets/train.txt and val.txt rewritten.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

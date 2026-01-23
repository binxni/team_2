#!/usr/bin/env python3
"""
Create a mixed point cloud folder where ~1/3 of frames use LISA-augmented
points and the rest use clean points, based on custom_av_64.

Outputs to: OpenPCDet/data/custom_av_hybrid/points_new

Selection rule: sort by filename and pick LISA for indices i where i % 3 == 0,
falling back to clean if the corresponding LISA file is missing.

This script assumes the following directories exist:
  - OpenPCDet/data/custom_av_64/points
  - OpenPCDet/data/custom_av_64/points_lisa

Usage (from repo root):
  python OpenPCDet/tools/create_points_new_one_third.py

Optional arguments:
  --src-root   Path to data root containing custom_av_64 (default: OpenPCDet/data)
  --dst-root   Path to data root containing custom_av_hybrid (default: OpenPCDet/data)
  --seed       Seed for deterministic shuffling (not used by default; selection is modular)
  --dry-run    Do not copy files, only print planned actions
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ~1/3 LISA mixed points_new")
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("OpenPCDet/data"),
        help="Path containing custom_av_64 (default: OpenPCDet/data)",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=Path("OpenPCDet/data"),
        help="Path containing custom_av_hybrid (default: OpenPCDet/data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be copied",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    src_points = (args.src_root / "custom_av_64" / "points").resolve()
    src_points_lisa = (args.src_root / "custom_av_64" / "points_lisa").resolve()
    dst_points_new = (args.dst_root / "custom_av_hybrid" / "points_new").resolve()

    if not src_points.exists():
        print(f"ERROR: {src_points} does not exist")
        return 1
    if not src_points_lisa.exists():
        print(f"ERROR: {src_points_lisa} does not exist")
        return 1

    dst_points_new.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in src_points.glob("*.npy")])
    if not files:
        print(f"WARNING: no .npy files found in {src_points}")
        return 0

    total = len(files)
    use_lisa = 0
    use_clean = 0

    print(f"Creating mixed folder: {dst_points_new}")
    print(f"Source clean: {src_points}")
    print(f"Source lisa : {src_points_lisa}")
    print(f"Total frames: {total}")

    for i, src_clean in enumerate(files):
        fname = src_clean.name
        src_lisa = src_points_lisa / fname
        dst = dst_points_new / fname

        choose_lisa = (i % 3 == 0) and src_lisa.exists()
        src = src_lisa if choose_lisa else src_clean

        if args.dry_run:
            print(f"[dry-run] {'LISA ' if choose_lisa else 'CLEAN'} -> {dst.name}")
        else:
            shutil.copy2(src, dst)
        if choose_lisa:
            use_lisa += 1
        else:
            use_clean += 1

    ratio = use_lisa / total if total > 0 else 0.0
    print(f"Done. Copied {total} files: LISA={use_lisa}, CLEAN={use_clean}, LISA ratio={ratio:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


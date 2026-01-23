#!/usr/bin/env python3
"""
Create a single train split that includes all point files found in a dataset's points/ directory.

This is useful when you want no train/val split and prefer to train on all frames.

Usage:
  python OpenPCDet/tools/make_all_train_split.py \
      --dataset_root OpenPCDet/data/custom_av_range64 \
      --ext npy

It writes ImageSets/train.txt under the dataset root.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate train.txt with all frames from points/")
    ap.add_argument("--dataset_root", type=Path, required=True,
                    help="Dataset root containing points/ (and optional labels/) directory")
    ap.add_argument("--ext", type=str, default="npy", choices=["npy", "bin"],
                    help="Point file extension to search for")
    ap.add_argument("--imagesets_dir", type=Path, default=None,
                    help="Optional ImageSets dir (default: <dataset_root>/ImageSets)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing train.txt")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ds_root: Path = args.dataset_root
    points_dir = ds_root / "points"
    if not points_dir.is_dir():
        raise FileNotFoundError(f"points directory not found: {points_dir}")

    ids = sorted(p.stem for p in points_dir.glob(f"*.{args.ext}") if p.is_file())
    if not ids:
        raise RuntimeError(f"No '*.{args.ext}' files found under {points_dir}")

    imagesets_dir = args.imagesets_dir or (ds_root / "ImageSets")
    imagesets_dir.mkdir(parents=True, exist_ok=True)
    train_txt = imagesets_dir / "train.txt"
    if train_txt.exists() and not args.overwrite:
        print(f"train.txt exists and --overwrite not set: {train_txt}")
        return

    with open(train_txt, "w") as f:
        for sid in ids:
            f.write(f"{sid}\n")

    print(f"Wrote {len(ids)} IDs to {train_txt}")


if __name__ == "__main__":
    main()


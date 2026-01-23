#!/usr/bin/env python3
"""
Regenerate ImageSets/{train,val}_{clean,lisa}.txt according to current
ImageSets/train.txt and ImageSets/val.txt.

Rules:
- If an ID exists in domain_map.txt, use that mapping ("clean" or "lisa").
- Otherwise, infer from point shape:
  * points/<id>.npy has shape (N, 3) -> "clean"
  * shape with >=4 columns -> "lisa"

This accommodates newly imported Waymo (3ch) and LISA-augmented (4ch).

Usage (from repo root or OpenPCDet dir):
  python OpenPCDet/tools/regenerate_imagesets_clean_lisa.py \
      --dataset-root OpenPCDet/data/custom_av_hybrid
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import numpy as np


def read_list(path: Path) -> List[str]:
    if not path.exists():
        return []
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def write_list(path: Path, items: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(f"{it}\n")


def load_domain_map(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not path.exists():
        return mapping
    with open(path, "r") as f:
        for line in f:
            s = line.strip().split()
            if len(s) >= 2 and s[0] and s[1]:
                mapping[s[0]] = s[1]
    return mapping


def infer_domain_from_points(points_dir: Path, sample_id: str) -> str:
    npy = points_dir / f"{sample_id}.npy"
    if not npy.exists():
        return "clean"  # default fall-back
    try:
        arr = np.load(npy, mmap_mode='r', allow_pickle=False)
        cols = arr.shape[1] if arr.ndim == 2 else 0
    except Exception:
        cols = 0
    return "lisa" if cols >= 4 else "clean"


def regenerate(dataset_root: Path) -> Tuple[int, int, int, int]:
    imagesets = dataset_root / "ImageSets"
    points_dir = dataset_root / "points"

    train_ids = read_list(imagesets / "train.txt")
    val_ids = read_list(imagesets / "val.txt")
    domain_map = load_domain_map(imagesets / "domain_map.txt")

    train_clean: List[str] = []
    train_lisa: List[str] = []
    for sid in train_ids:
        domain = domain_map.get(sid)
        if domain is None:
            domain = infer_domain_from_points(points_dir, sid)
        (train_clean if domain == "clean" else train_lisa).append(sid)

    val_clean: List[str] = []
    val_lisa: List[str] = []
    for sid in val_ids:
        domain = domain_map.get(sid)
        if domain is None:
            domain = infer_domain_from_points(points_dir, sid)
        (val_clean if domain == "clean" else val_lisa).append(sid)

    # Write outputs
    write_list(imagesets / "train_clean.txt", train_clean)
    write_list(imagesets / "train_lisa.txt", train_lisa)
    write_list(imagesets / "val_clean.txt", val_clean)
    write_list(imagesets / "val_lisa.txt", val_lisa)

    print(
        f"train_clean={len(train_clean)} train_lisa={len(train_lisa)} "
        f"val_clean={len(val_clean)} val_lisa={len(val_lisa)}"
    )
    return len(train_clean), len(train_lisa), len(val_clean), len(val_lisa)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Path to dataset root (contains points, labels, ImageSets)",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    # Resolve default relative to this file if not provided
    root_dir = Path(__file__).resolve().parents[1]
    dataset_root = (args.dataset_root if args.dataset_root is not None else (root_dir / "data/custom_av_hybrid")).resolve()

    if not (dataset_root / "ImageSets").exists():
        print(f"Invalid dataset root: {dataset_root}", file=sys.stderr)
        return 1

    regenerate(dataset_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


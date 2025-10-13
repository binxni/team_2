#!/usr/bin/env python3
"""
Merge custom_av_hybrid train/val into a single train split, and set validation
to Waymo npy samples (with 1/3 LISA-augmented duplicates). This script:

- Reads ImageSets/train.txt and val.txt under custom_av_hybrid and merges them
  into a new ImageSets/train.txt (backups created).
- Imports Waymo validation samples from `waymo_npy` directory by copying .npy
  and matching labels/*.txt into the dataset's points/ and labels/ with new IDs.
- Randomly selects ~1/3 of the imported Waymo samples and applies LISA-based
  adverse weather augmentation to create augmented duplicates (without touching
  originals). Augmented .npy are saved with new IDs and the corresponding
  labels are duplicated as-is.
- Writes a fresh ImageSets/val.txt listing the Waymo originals plus augmented copies.
- Optionally regenerates infos/gt_database through the dataset's info builder.

Usage (from repo root):
    python OpenPCDet/tools/update_splits_with_waymo_val.py \
        --dataset-root OpenPCDet/data/custom_av_hybrid \
        --waymo-root OpenPCDet/waymo_npy \
        --augment-ratio 0.3333 \
        --seed 42 \
        --run-lisa  # actually perform augmentation (default on)

Notes:
- LISA augmentation uses OpenPCDet/augmented_with_lisa.py. No network needed.
- To import `pylisa`, the script runs that tool with PYTHONPATH set to
  OpenPCDet/LISA.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple


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


def find_max_numeric_id(points_dir: Path) -> int:
    max_id = -1
    for p in points_dir.glob("*.npy"):
        stem = p.stem
        if stem.isdigit():
            try:
                num = int(stem)
            except ValueError:
                continue
            if num > max_id:
                max_id = num
    return max_id


def collect_waymo_pairs(waymo_root: Path) -> List[Tuple[Path, Path]]:
    label_dir = waymo_root / "labels"
    pairs: List[Tuple[Path, Path]] = []
    for npy in sorted(waymo_root.glob("*.npy")):
        lbl = label_dir / f"{npy.stem}.txt"
        if lbl.exists():
            pairs.append((npy, lbl))
    return pairs


def copy_pairs_to_dataset(
    pairs: List[Tuple[Path, Path]], points_dst: Path, labels_dst: Path, start_id: int
) -> List[str]:
    ids: List[str] = []
    cur = start_id
    for npy_src, lbl_src in pairs:
        cur += 1
        new_id = f"{cur:08d}"
        npy_dst = points_dst / f"{new_id}.npy"
        lbl_dst = labels_dst / f"{new_id}.txt"
        shutil.copy2(npy_src, npy_dst)
        shutil.copy2(lbl_src, lbl_dst)
        ids.append(new_id)
    return ids


def run_lisa_on_subset(
    dataset_points_dir: Path,
    subset_ids: List[str],
    tmp_root: Path,
    root_dir: Path,
    lisa_atm_model: str = "rain",
    lisa_rain_rate: float = 10.0,
    lisa_workers: int = 1,
    normalize_intensity: bool = True,
) -> Path:
    """
    Prepares a temporary dataset layout with only the selected IDs under tmp_root/points,
    runs augmented_with_lisa.py on it, and returns the path to tmp_root/points_lisa.
    """
    tmp_points = tmp_root / "points"
    tmp_points.mkdir(parents=True, exist_ok=True)

    # Symlink or copy the selected files into tmp_points
    for sid in subset_ids:
        src = dataset_points_dir / f"{sid}.npy"
        dst = tmp_points / f"{sid}.npy"
        try:
            if dst.exists():
                dst.unlink()
            os.symlink(src.resolve(), dst)
        except Exception:
            shutil.copy2(src, dst)

    # Build command for augmentation tool
    tool = (root_dir / "augmented_with_lisa.py").resolve()
    env = os.environ.copy()
    # Ensure pylisa is importable
    env["PYTHONPATH"] = str((root_dir / "LISA").resolve()) + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        "python", str(tool),
        "--dataset-root", str(tmp_root.resolve()),
        "--points-dir", "points",
        "--output-dir", "points_lisa",
        "--atm-model", lisa_atm_model,
        "--rain-rate", str(lisa_rain_rate),
        "--num-workers", str(lisa_workers),
    ]
    if normalize_intensity:
        cmd.append("--normalize-intensity")

    print("[LISA] Running:", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)
    return tmp_root / "points_lisa"


def duplicate_augmented_into_dataset(
    aug_dir: Path, labels_src_dir: Path, points_dst: Path, labels_dst: Path, start_id: int
) -> List[str]:
    ids: List[str] = []
    cur = start_id
    for npy in sorted(aug_dir.glob("*.npy")):
        cur += 1
        new_id = f"{cur:08d}"
        npy_dst = points_dst / f"{new_id}.npy"
        lbl_src = labels_src_dir / f"{npy.stem}.txt"
        lbl_dst = labels_dst / f"{new_id}.txt"
        shutil.copy2(npy, npy_dst)
        shutil.copy2(lbl_src, lbl_dst)
        ids.append(new_id)
    return ids


def regenerate_infos(dataset_cfg_yaml: Path, root_dir: Path) -> None:
    """Call the dataset info builder to regenerate train/val pkl and gt_database."""
    builder = (root_dir / "pcdet/datasets/custom_av_hybrid/custom_av_dataset_hybrid.py").resolve()
    cmd = [
        "python", str(builder), "create_custom_av_infos", str(dataset_cfg_yaml.resolve())
    ]
    print("[INFO] Regenerating infos and gt_database:\n ", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--waymo-root", type=Path, default=None)
    ap.add_argument("--augment-ratio", type=float, default=1.0 / 3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--atm-model", type=str, default="rain")
    ap.add_argument("--rain-rate", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-run-lisa", action="store_true", help="Skip LISA run (debug)")
    ap.add_argument("--regenerate-infos", action="store_true", help="Rebuild info pkl and gt_database")
    args = ap.parse_args()

    # Resolve project root (one level above tools/)
    root_dir = Path(__file__).resolve().parents[1]

    # Defaults if not provided
    dataset_root = (args.dataset_root if args.dataset_root is not None else (root_dir / "data/custom_av_hybrid")).resolve()
    waymo_root = (args.waymo_root if args.waymo_root is not None else (root_dir / "waymo_npy")).resolve()

    imagesets = dataset_root / "ImageSets"
    points_dir = dataset_root / "points"
    labels_dir = dataset_root / "labels"
    train_list = imagesets / "train.txt"
    val_list = imagesets / "val.txt"

    assert points_dir.exists() and labels_dir.exists(), f"Invalid dataset root: {dataset_root}"
    assert waymo_root.exists(), f"Waymo root does not exist: {waymo_root}"

    # 1) Merge existing train and val into train
    train_ids = read_list(train_list)
    val_ids_old = read_list(val_list)
    merged_train = train_ids + [v for v in val_ids_old if v not in train_ids]
    # Backup old lists
    if train_list.exists():
        shutil.copy2(train_list, imagesets / "train.txt.bak")
    if val_list.exists():
        shutil.copy2(val_list, imagesets / "val.txt.bak")
    write_list(train_list, merged_train)
    print(f"[SPLIT] Merged train({len(train_ids)}) + val({len(val_ids_old)}) -> train({len(merged_train)})")

    # 2) Import Waymo npy+labels into dataset with new IDs
    pairs = collect_waymo_pairs(waymo_root)
    if not pairs:
        raise RuntimeError(f"No Waymo npy+label pairs found under {waymo_root}")
    max_id = find_max_numeric_id(points_dir)
    start_id_waymo = max_id
    waymo_ids = copy_pairs_to_dataset(pairs, points_dir, labels_dir, start_id_waymo)
    print(f"[IMPORT] Copied {len(waymo_ids)} Waymo samples into dataset starting from {start_id_waymo+1:08d}")

    # 3) Randomly select ~1/3 for augmentation (duplicate, not replace)
    random.seed(args.seed)
    k = max(1, int(math.floor(len(waymo_ids) * args.augment_ratio)))
    subset = sorted(random.sample(waymo_ids, k))
    print(f"[AUG] Selecting {k}/{len(waymo_ids)} samples for LISA augmentation")

    aug_ids: List[str] = []
    if not args.no_run_lisa and k > 0:
        tmp_root = (root_dir / "tmp/aug_waymo_subset").resolve()
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True, exist_ok=True)

        aug_out = run_lisa_on_subset(
            dataset_points_dir=points_dir,
            subset_ids=subset,
            tmp_root=tmp_root,
            root_dir=root_dir,
            lisa_atm_model=args.atm_model,
            lisa_rain_rate=args.rain_rate,
            lisa_workers=args.workers,
            normalize_intensity=True,
        )
        # 4) Copy augmented files back into dataset with new IDs and duplicate labels
        start_id_aug = start_id_waymo + len(waymo_ids)
        aug_ids = duplicate_augmented_into_dataset(
            aug_dir=aug_out,
            labels_src_dir=labels_dir,
            points_dst=points_dir,
            labels_dst=labels_dir,
            start_id=start_id_aug,
        )
        print(f"[AUG] Duplicated {len(aug_ids)} augmented samples into dataset starting from {start_id_aug+1:08d}")
    else:
        print("[AUG] Skipping LISA run (--no-run-lisa set or no samples)")

    # 5) Write new val.txt: waymo originals + augmented duplicates
    final_val_ids = waymo_ids + aug_ids
    write_list(val_list, final_val_ids)
    print(f"[SPLIT] Wrote new val.txt with {len(final_val_ids)} entries")

    # 6) Regenerate infos and db if requested
    if args.regenerate_infos:
        regenerate_infos(root_dir / "tools/cfgs/dataset_configs/custom_av_dataset_hybrid.yaml", root_dir)
    else:
        print("[INFO] Skipped info regeneration. Run with --regenerate-infos to update PKLs and gt_database.")

    print("[DONE] Completed split update and Waymo val import/augmentation.")


if __name__ == "__main__":
    main()

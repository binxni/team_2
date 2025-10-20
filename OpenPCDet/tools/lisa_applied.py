#!/usr/bin/env python3
"""
Apply LISA-based adverse-weather noise to every k-th frame while copying others.

Behavior
- Reads all .npy point clouds from --src-points
- Writes outputs under --dst-root/points with identical filenames
- For every k-th file in sorted order (1-based), applies LISA via in-process call
  to augmented_with_lisa.augment_single_file; otherwise copies the file as-is.
- Optionally cycles models per augmented frame in the order: rain -> snow -> fog.

Example
  python OpenPCDet/tools/lisa_applied.py \
      --src-points OpenPCDet/data/custom_av/points \
      --dst-root OpenPCDet/data/custom_av_hybrid2 \
      --every-k 3 --cycle-models --fog-model chu_hogg_fog \
      --normalize-intensity --output-rmax 64 --post-thin-a 0.15 --post-thin-r0 45
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np


LOGGER = logging.getLogger("lisa_applied")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply LISA to every k-th frame, copy others")
    # IO
    p.add_argument("--src-points", type=Path, required=True, help="Source directory with .npy files")
    p.add_argument("--dst-root", type=Path, required=True, help="Destination dataset root; outputs under <dst-root>/points")
    p.add_argument("--every-k", type=int, default=3, help="Apply LISA to every k-th file (1-based order). Default: 3")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--dry-run", action="store_true", help="Parse and plan only; do not write outputs")
    p.add_argument("--verbose", action="store_true")

    # LISA core params (match augmented_with_lisa)
    p.add_argument("--atm-model", choices=("rain","snow","chu_hogg_fog","strong_advection_fog","moderate_advection_fog"), default="rain")
    p.add_argument("--cycle-models", action="store_true", help="If set, cycle models per augmented frame: rain -> snow -> fog-model")
    p.add_argument("--fog-model", choices=("chu_hogg_fog","strong_advection_fog","moderate_advection_fog"), default="chu_hogg_fog", help="Fog model to use when cycling models")
    p.add_argument("--mode", choices=("strongest","last"), default="strongest")
    p.add_argument("--rain-rate", type=float, default=None, help="mm/hr, required for rain/snow")
    p.add_argument("--enable-falling-particles", action="store_true", help="Enable enhanced falling rain/snow particles in air")
    p.add_argument("--lam", type=float, default=905.0)
    p.add_argument("--rmax", type=float, default=200.0)
    p.add_argument("--rmin", type=float, default=1.5)
    p.add_argument("--bdiv", type=float, default=3e-3)
    p.add_argument("--dst", type=float, default=0.05)
    p.add_argument("--dR", type=float, default=0.09)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--normalize-intensity", action="store_true")
    p.add_argument("--intensity-max", type=float, default=255.0)
    p.add_argument("--keep-label-column", action="store_true")

    # Post-processing (match augmented_with_lisa new options)
    p.add_argument("--output-rmax", type=float, default=None)
    p.add_argument("--post-thin-a", type=float, default=None)
    p.add_argument("--post-thin-r0", type=float, default=None)
    p.add_argument("--attenuate-intensity-beta", type=float, default=None)
    p.add_argument("--attenuate-intensity-noise-sigma", type=float, default=0.0)
    p.add_argument("--veil-near-r", type=float, default=None)
    p.add_argument("--veil-near-z", type=float, default=None)
    p.add_argument("--veil-density", type=float, default=0.0)
    p.add_argument("--veil-intensity-max", type=float, default=0.2)

    return p.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="[%(levelname)s] %(message)s")


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    src = args.src_points.resolve()
    if not src.exists() or not src.is_dir():
        LOGGER.error("Source directory not found: %s", src)
        return 1
    dst_points = (args.dst_root / "points").resolve()
    dst_points.mkdir(parents=True, exist_ok=True)
    dst_labels = (args.dst_root / "labels").resolve()
    dst_labels.mkdir(parents=True, exist_ok=True)

    # 원본 라벨/이미지셋 경로 추정
    src_labels = src.parent / "labels"
    src_imagesets = src.parent / "ImageSets"
    dst_imagesets = args.dst_root / "ImageSets"

    # train/val 인덱스 로딩
    train_indices, val_indices = set(), set()
    if src_imagesets.exists():
        dst_imagesets.mkdir(parents=True, exist_ok=True)
        train_path = src_imagesets / "train.txt"
        val_path = src_imagesets / "val.txt"
        if train_path.exists():
            with open(train_path) as f:
                train_indices = set(line.strip() for line in f if line.strip())
        if val_path.exists():
            with open(val_path) as f:
                val_indices = set(line.strip() for line in f if line.strip())

    hybrid_train, hybrid_val, hybrid_lisa = [], [], []

    npy_files = sorted(src.glob("*.npy"))
    if not npy_files:
        LOGGER.warning("No .npy files under %s", src)
        return 0

    from augmented_with_lisa import Lisa, augment_single_file

    def build(model: str):
        lisa_instance = Lisa(
            lam=args.lam,
            rmax=args.rmax,
            rmin=args.rmin,
            bdiv=args.bdiv,
            dst=args.dst,
            dR=args.dR,
            atm_model=model,
            mode=args.mode,
        )
        lisa_instance.atm_model = model  # Store model name for enhanced effects
        return lisa_instance

    if args.cycle_models:
        model_cycle = ("rain", "snow", args.fog_model)
        lisa_map = {m: build(m) for m in model_cycle}
    else:
        lisa_map = {args.atm_model: build(args.atm_model)}

    aug_ns = SimpleNamespace(
        normalize_intensity=args.normalize_intensity,
        intensity_max=args.intensity_max,
        rain_rate=args.rain_rate,
        keep_label_column=args.keep_label_column,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        verbose=args.verbose,
        output_rmax=args.output_rmax,
        post_thin_a=args.post_thin_a,
        post_thin_r0=args.post_thin_r0,
        attenuate_intensity_beta=args.attenuate_intensity_beta,
        attenuate_intensity_noise_sigma=args.attenuate_intensity_noise_sigma,
        veil_near_r=args.veil_near_r,
        veil_near_z=args.veil_near_z,
        veil_density=args.veil_density,
        veil_intensity_max=args.veil_intensity_max,
    )

    total = len(npy_files)
    LOGGER.info("Planning %d files; every %d-th will be LISA-augmented.", total, args.every_k)
    aug_count = 0
    mc_aug_count = 0
    RAIN_RATE_SET = (5.0, 10.0, 15.0, 20.0)
    lisa_idx = len(npy_files)  # LISA 증강본의 시작 인덱스
    for ordinal, src_path in enumerate(npy_files, start=1):
        # 1. 원본 포인트 복사
        dst_path = dst_points / f"{ordinal-1:08d}.npy"
        if not args.dry_run:
            shutil.copy2(src_path, dst_path)
        # 1-1. 원본 라벨 복사
        label_name = f"{ordinal-1:08d}.txt"
        src_label_path = src_labels / src_path.with_suffix('.txt').name
        dst_label_path = dst_labels / label_name
        if src_label_path.exists() and not args.dry_run:
            shutil.copy2(src_label_path, dst_label_path)

        # train/val 인덱스에 추가
        idx_str = f"{ordinal-1:08d}"
        if idx_str in train_indices:
            hybrid_train.append(idx_str)
        elif idx_str in val_indices:
            hybrid_val.append(idx_str)

        # 2. LISA 증강본 생성 (every-k 마다)
    if ordinal % args.every_k == 0:
            if args.cycle_models:
                models = list(lisa_map.keys())
                m = models[aug_count % len(models)]
            else:
                m = next(iter(lisa_map.keys()))
            lisa = lisa_map[m]
            aug_count += 1
            this_rain_rate = args.rain_rate
            if m in ("rain", "snow"):
                if this_rain_rate is None:
                    this_rain_rate = RAIN_RATE_SET[mc_aug_count % len(RAIN_RATE_SET)]
                    mc_aug_count += 1
            aug_ns.rain_rate = this_rain_rate
            if m in ("rain", "snow") and aug_ns.rain_rate is None:
                LOGGER.error("Rain/snow model selected but no rain-rate resolved.")
                return 2
            # 연속적인 파일명으로 저장
            lisa_name = f"{lisa_idx:08d}.npy"
            dst_lisa_path = dst_points / lisa_name
            LOGGER.info("[%d/%d] LISA augment (%s) -> %s", ordinal, total, m, dst_lisa_path.name)
            if not args.dry_run:
                np.random.seed(args.seed + ordinal)
                try:
                    if args.enable_falling_particles and m in ("rain", "snow"):
                        # Use enhanced augmentation with falling particles
                        points = np.load(src_path)
                        enhanced_points = lisa.augment_with_falling_effects(points, aug_ns.rain_rate)
                        # Save enhanced points (keep only first 4 columns for compatibility)
                        np.save(dst_lisa_path, enhanced_points[:, :4])
                        LOGGER.info("Applied falling particles enhancement")
                    else:
                        # Use standard LISA augmentation
                        augment_single_file(lisa, src_path, dst_lisa_path, aug_ns)
                except Exception as e:
                    LOGGER.error("Augment failed for %s: %s", src_path.name, e)
                    return 2
            # 2-1. LISA 라벨 복사 (동일한 번호)
            dst_lisa_label = dst_labels / f"{lisa_idx:08d}.txt"
            if src_label_path.exists() and not args.dry_run:
                shutil.copy2(src_label_path, dst_lisa_label)
            # LISA 증강본 인덱스도 train/val에 추가
            lisa_idx_str = f"{lisa_idx:08d}"
            if idx_str in train_indices:
                hybrid_train.append(lisa_idx_str)
            elif idx_str in val_indices:
                hybrid_val.append(lisa_idx_str)
            # lisa.txt에 기록
            hybrid_lisa.append(lisa_idx_str)
            lisa_idx += 1

    # hybrid2 ImageSets 저장
    if dst_imagesets.exists():
        for f in dst_imagesets.glob("*.txt"):
            f.unlink()
    dst_imagesets.mkdir(parents=True, exist_ok=True)
    with open(dst_imagesets / "train.txt", "w") as f:
        for idx in hybrid_train:
            f.write(f"{idx}\n")
    with open(dst_imagesets / "val.txt", "w") as f:
        for idx in hybrid_val:
            f.write(f"{idx}\n")
    with open(dst_imagesets / "lisa.txt", "w") as f:
        for idx in hybrid_lisa:
            f.write(f"{idx}\n")
    LOGGER.info("Done. Outputs under %s", dst_points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

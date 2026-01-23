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

    npy_files = sorted(src.glob("*.npy"))
    if not npy_files:
        LOGGER.warning("No .npy files under %s", src)
        return 0

    # Lazy import to avoid pylisa dependency unless actually applying LISA
    from OpenPCDet.augmented_with_lisa import Lisa, augment_single_file

    # Build Lisa instance(s)
    def build(model: str) -> Lisa:
        return Lisa(
            lam=args.lam,
            rmax=args.rmax,
            rmin=args.rmin,
            bdiv=args.bdiv,
            dst=args.dst,
            dR=args.dR,
            atm_model=model,
            mode=args.mode,
        )

    if args.cycle-models:
        model_cycle = ("rain", "snow", args.fog_model)
        lisa_map = {m: build(m) for m in model_cycle}
    else:
        lisa_map = {args.atm_model: build(args.atm_model)}

    # Prepare a Namespace compatible with augment_single_file
    aug_ns = SimpleNamespace(
        normalize_intensity=args.normalize_intensity,
        intensity_max=args.intensity_max,
        rain_rate=args.rain_rate,
        keep_label_column=args.keep_label_column,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        verbose=args.verbose,
        # Post options
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
    mc_aug_count = 0  # count of rain/snow augmented frames for cycling rain-rate
    RAIN_RATE_SET = (5.0, 10.0, 15.0, 20.0)
    for ordinal, src_path in enumerate(npy_files, start=1):
        dst_path = dst_points / src_path.name
        do_aug = (ordinal % args.every_k == 0)

        if dst_path.exists() and not args.overwrite:
            LOGGER.info("Skip existing %s", dst_path.name)
            continue

        if do_aug:
            # Choose model
            if args.cycle-models:
                models = list(lisa_map.keys())
                m = models[aug_count % len(models)]
            else:
                m = next(iter(lisa_map.keys()))
            lisa = lisa_map[m]
            aug_count += 1
            # Choose rain rate for rain/snow models if not provided
            this_rain_rate = args.rain_rate
            if m in ("rain", "snow"):
                if this_rain_rate is None:
                    this_rain_rate = RAIN_RATE_SET[mc_aug_count % len(RAIN_RATE_SET)]
                    mc_aug_count += 1
            # Set per-call rain rate
            aug_ns.rain_rate = this_rain_rate

            if m in ("rain", "snow") and aug_ns.rain_rate is None:
                LOGGER.error("Rain/snow model selected but no rain-rate resolved.")
                return 2

            rate_str = f", rr={aug_ns.rain_rate}mm/hr" if m in ("rain","snow") else ""
            LOGGER.info("[%d/%d] LISA augment (%s%s) -> %s", ordinal, total, m, rate_str, dst_path.name)
            if not args.dry_run:
                # Fix seed per-file for reproducibility across runs
                np.random.seed(args.seed + ordinal)
            try:
                augment_single_file(lisa, src_path, dst_path, aug_ns)
            except Exception as e:
                LOGGER.error("Augment failed for %s: %s", src_path.name, e)
                return 2
        else:
            LOGGER.info("[%d/%d] Copy -> %s", ordinal, total, dst_path.name)
            if not args.dry_run:
                shutil.copy2(src_path, dst_path)

    LOGGER.info("Done. Outputs under %s", dst_points)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

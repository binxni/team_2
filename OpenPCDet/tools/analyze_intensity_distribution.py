#!/usr/bin/env python3
"""
Analyze intensity distribution for .npy point clouds (x, y, z, intensity).

Examples:
  - Basic summary over a folder:
      python Subin/OpenPCDet/tools/analyze_intensity_distribution.py \
          --root Subin/OpenPCDet/data/custom_av/points_test

  - With distance-wise histograms and outputs:
      python Subin/OpenPCDet/tools/analyze_intensity_distribution.py \
          --root Subin/OpenPCDet/data/custom_av/points_test \
          --bins 256 --range 0 1 \
          --per-distance 0 30 50 70 100 \
          --save-csv intensity_hist.csv \
          --plot-png intensity_hist.png
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np


def _iter_point_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.glob("*.npy")):
        if path.is_file():
            yield path


def _compute_hist(
    values: np.ndarray,
    bins: int,
    value_range: Tuple[float, float],
) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    return hist.astype(np.int64, copy=False)


def analyze_intensity(
    point_files: Iterable[Path],
    bins: int = 256,
    value_range: Tuple[float, float] = (0.0, 1.0),
    per_distance_edges: Optional[List[float]] = None,
    max_files: Optional[int] = None,
) -> dict:
    total_hist = np.zeros(bins, dtype=np.int64)
    per_distance_hists: Optional[List[np.ndarray]] = None
    if per_distance_edges is not None and len(per_distance_edges) >= 2:
        per_distance_hists = [np.zeros(bins, dtype=np.int64) for _ in range(len(per_distance_edges) - 1)]

    per_frame_means: list[float] = []
    per_frame_stds: list[float] = []
    per_frame_counts: list[int] = []

    global_min = float("inf")
    global_max = float("-inf")

    processed = 0
    for path in point_files:
        if max_files is not None and processed >= max_files:
            break

        pts = np.load(path)  # (N, 4): x, y, z, intensity
        if pts.ndim != 2 or pts.shape[1] < 4:
            raise ValueError(f"Unexpected shape {pts.shape} in {path}")

        intens = pts[:, 3].astype(np.float32, copy=False)
        total_hist += _compute_hist(intens, bins=bins, value_range=value_range)

        per_frame_means.append(float(np.mean(intens)))
        per_frame_stds.append(float(np.std(intens)))
        per_frame_counts.append(int(intens.shape[0]))
        global_min = min(global_min, float(np.min(intens)))
        global_max = max(global_max, float(np.max(intens)))

        if per_distance_hists is not None:
            # radial distance in XY-plane
            r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
            for i, (r_min, r_max) in enumerate(zip(per_distance_edges[:-1], per_distance_edges[1:])):
                mask = (r >= r_min) & (r < r_max)
                if not np.any(mask):
                    continue
                per_distance_hists[i] += _compute_hist(intens[mask], bins=bins, value_range=value_range)

        processed += 1

    if processed == 0:
        raise RuntimeError("No .npy point clouds found or processed.")

    # Build bin centers for summary printing
    lo, hi = value_range
    bin_edges = np.linspace(lo, hi, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # quantiles from sampled frames (using per-frame intensities mean/std)
    # To avoid storing all intensities, we estimate quantiles from total_hist
    cumsum = np.cumsum(total_hist)
    total_count = int(cumsum[-1])
    def q_from_hist(q: float) -> float:
        if total_count == 0:
            return float('nan')
        target = q * total_count
        idx = int(np.searchsorted(cumsum, target, side='left'))
        idx = min(max(idx, 0), bins - 1)
        return float(bin_centers[idx])

    summary = {
        "frames": processed,
        "total_points": int(sum(per_frame_counts)),
        "intensity": {
            "range_observed": [global_min, global_max],
            "range_config": [lo, hi],
            "mean_per_frame": statistics.mean(per_frame_means),
            "std_per_frame": statistics.mean(per_frame_stds),
            "mean_overall": float(np.average(bin_centers, weights=total_hist)) if total_count > 0 else float('nan'),
            "p10": q_from_hist(0.10),
            "p50": q_from_hist(0.50),
            "p90": q_from_hist(0.90),
            "zero_fraction": float(total_hist[0] / total_count) if total_count > 0 else float('nan'),
        },
        "hist": {
            "bins": bins,
            "edges": bin_edges.tolist(),
            "counts": total_hist.tolist(),
        },
    }

    if per_distance_hists is not None:
        dist_bucket_summaries = []
        for i, h in enumerate(per_distance_hists):
            cnt = int(np.sum(h))
            mean = float(np.average(bin_centers, weights=h)) if cnt > 0 else float('nan')
            dist_bucket_summaries.append({
                "range": [float(per_distance_edges[i]), float(per_distance_edges[i+1])],
                "count": cnt,
                "mean": mean,
                "counts": h.tolist(),
            })
        summary["hist_per_distance"] = dist_bucket_summaries

    summary["per_frame_counts"] = per_frame_counts
    summary["per_frame_mean"] = per_frame_means
    summary["per_frame_std"] = per_frame_stds
    return summary


def save_csv(summary: dict, out_csv: Path) -> None:
    import csv
    edges = summary["hist"]["edges"]
    counts = summary["hist"]["counts"]
    headers = ["bin_left", "bin_right", "count_total"]
    per_dist = summary.get("hist_per_distance")
    if per_dist is not None:
        headers += [f"count_{int(b['range'][0])}-{int(b['range'][1])}m" for b in per_dist]

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(len(edges) - 1):
            row = [edges[i], edges[i + 1], counts[i]]
            if per_dist is not None:
                row += [b["counts"][i] for b in per_dist]
            writer.writerow(row)


def plot_png(summary: dict, out_png: Path, title: str = "Intensity Histogram") -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required for --plot-png") from e

    edges = np.array(summary["hist"]["edges"], dtype=float)
    counts = np.array(summary["hist"]["counts"], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(edges[:-1], counts, where='post', label='total')
    ax.set_title(title)
    ax.set_xlabel("intensity")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3, linestyle='--')

    per_dist = summary.get("hist_per_distance")
    if per_dist is not None:
        for b in per_dist:
            ax.step(edges[:-1], np.array(b["counts"], dtype=float), where='post', label=f"{int(b['range'][0])}-{int(b['range'][1])}m", alpha=0.7)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def print_summary(summary: dict) -> str:
    s = []
    s.append(f"Frames processed             : {summary['frames']}")
    s.append(f"Total points                 : {summary['total_points']:,}")
    ints = summary["intensity"]
    s.append(
        "Intensity range (observed)     : "
        f"[{ints['range_observed'][0]:.4f}, {ints['range_observed'][1]:.4f}] "
        f"(configured [{ints['range_config'][0]:.2f}, {ints['range_config'][1]:.2f}])"
    )
    s.append(
        f"Intensity mean (overall)      : {ints['mean_overall']:.5f}"
    )
    s.append(
        f"Intensity per-frame mean±std  : {ints['mean_per_frame']:.5f} ± {ints['std_per_frame']:.5f}"
    )
    s.append(
        f"Intensity p10 / p50 / p90     : {ints['p10']:.5f} / {ints['p50']:.5f} / {ints['p90']:.5f}"
    )
    if np.isfinite(ints.get("zero_fraction", float('nan'))):
        s.append(f"Zero-intensity fraction       : {ints['zero_fraction']:.5f}")

    if "hist_per_distance" in summary:
        s.append("Distance buckets (mean intensity, counts):")
        for b in summary["hist_per_distance"]:
            r0, r1 = b["range"]
            s.append(f"  {int(r0):2d}-{int(r1):3d} m -> mean {b['mean']:.5f}, count {b['count']:,}")

    return "\n".join(s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze intensity distribution for .npy point clouds.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("Subin/OpenPCDet/data/custom_av/points_test"),
        help="Directory containing .npy point cloud files.",
    )
    parser.add_argument("--bins", type=int, default=256, help="Number of histogram bins.")
    parser.add_argument(
        "--range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.0, 1.0),
        help="Intensity range for histogram (e.g., 0 1 or 0 255).",
    )
    parser.add_argument(
        "--per-distance",
        type=float,
        nargs='+',
        default=None,
        help="Optional distance bucket edges in meters, e.g., 0 30 50 70 100.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of frames to process.")
    parser.add_argument("--save-csv", type=Path, default=None, help="Optional path to save histogram(s) as CSV.")
    parser.add_argument("--plot-png", type=Path, default=None, help="Optional path to save histogram plot as PNG.")

    args = parser.parse_args()
    root = args.root
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory {root} does not exist.")

    per_distance = args.per_distance
    if per_distance is not None:
        if len(per_distance) < 2:
            raise ValueError("--per-distance requires at least two edge values (e.g., 0 30 60 90)")
        per_distance = sorted(per_distance)

    result = analyze_intensity(
        point_files=_iter_point_files(root),
        bins=args.bins,
        value_range=(args.range[0], args.range[1]),
        per_distance_edges=per_distance,
        max_files=args.max_files,
    )

    print(print_summary(result))

    if args.save_csv is not None:
        save_csv(result, args.save_csv)
        print(f"Saved CSV to {args.save_csv}")

    if args.plot_png is not None:
        plot_png(result, args.plot_png)
        print(f"Saved plot to {args.plot_png}")


if __name__ == "__main__":
    main()


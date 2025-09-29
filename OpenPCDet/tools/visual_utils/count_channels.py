import os
import sys
import pickle
import argparse
import numpy as np
from collections import Counter, defaultdict

def load_points(path, ext):
    if ext == "npy":
        return np.load(path)  # (N, C). 희망: [x,y,z,intensity,ring,...]
    elif ext == "bin":
        # 일반적으로 KITTI 스타일 float32 x,y,z,intensity (ring 없음) → 채널 판별 불가
        pts = np.fromfile(path, dtype=np.float32)
        if pts.size % 4 != 0:
            # 다른 포맷일 수 있음: 시도해보되, 컬럼 수 모르면 reshape 불가
            return pts.reshape(-1, pts.size // (pts.size // 4))
        return pts.reshape(-1, 4)
    else:
        raise ValueError(f"Unsupported ext: {ext}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True, help="path to *_infos_*.pkl")
    ap.add_argument("--points_dir", required=True, help="directory containing point files")
    ap.add_argument("--ext", default="npy", choices=["npy","bin"], help="point file extension")
    ap.add_argument("--limit", type=int, default=None, help="debug: only first N samples")
    args = ap.parse_args()

    with open(args.infos, "rb") as f:
        infos = pickle.load(f)

    counts = Counter()
    unknown_samples = []
    per_channels_samples = defaultdict(int)

    total = len(infos) if args.limit is None else min(args.limit, len(infos))
    for i, info in enumerate(infos[:total]):
        pc = info.get("point_cloud", {})
        lidar_idx = pc.get("lidar_idx")
        if lidar_idx is None:
            counts["no_lidar_idx"] += 1
            continue

        path = os.path.join(args.points_dir, f"{lidar_idx}.{args.ext}")
        if not os.path.isfile(path):
            counts["missing_file"] += 1
            unknown_samples.append(("missing", path))
            continue

        try:
            pts = load_points(path, args.ext)
        except Exception as e:
            counts["load_error"] += 1
            unknown_samples.append(("load_error", path, str(e)))
            continue

        # ring 컬럼 탐색: 보통 5번째 컬럼(index 4)
        n_channels = None
        if pts.ndim == 2 and pts.shape[1] >= 5:
            ring = pts[:, 4].astype(np.int32, copy=False)
            # ring 값이 합리적인지 sanity check
            rmax = int(ring.max())
            rmin = int(ring.min())
            # 채널 수 추정
            if 0 <= rmin <= rmax <= 1024:
                n_channels = rmax + 1

        if n_channels in (64, 128):
            counts[n_channels] += 1
            per_channels_samples[n_channels] += 1
        else:
            counts["unknown"] += 1
            if len(unknown_samples) < 20:
                unknown_samples.append(("unknown", path, pts.shape))

    print("=== Summary ===")
    print(f"Total scanned: {total}")
    for k in sorted([k for k in counts if isinstance(k, int)]) + [x for x in counts if not isinstance(x, int)]:
        print(f"{k}: {counts[k]}")

    if unknown_samples:
        print("\nExamples of unknown/missing:")
        for item in unknown_samples[:10]:
            print(" -", item)

    # 센서별 개수만 깔끔하게
    print("\nPandar64 frames:", counts.get(64, 0))
    print("Pandar128 frames:", counts.get(128, 0))
    print("Unknown frames:", counts.get("unknown", 0))

if __name__ == "__main__":
    main()

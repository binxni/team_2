import pickle
from collections import Counter
import numpy as np

points = np.load("/workspace/OpenPCDet/OpenPCDet/data/custom_av/points/00000000.npy")  # (N,4) [x,y,z,intensity]


def estimate_channels_from_xyz(points_xyzi, 
                               deg_min=-40.0, deg_max=40.0, 
                               bin_deg=0.05, smooth_win=5, 
                               peak_thresh_ratio=0.2, min_separation_bins=5, 
                               max_points=200000):
    """
    points_xyzi: (N,4) [x,y,z,intensity]
    Returns: 64, 128, or None
    """
    pts = points_xyzi
    if pts.ndim != 2 or pts.shape[1] < 3 or pts.shape[0] < 500:
        return None

    # 다운샘플로 속도/안정성 확보
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]

    x, y, z = pts[:,0], pts[:,1], pts[:,2]
    r_xy = np.sqrt(x*x + y*y)
    elev = np.degrees(np.arctan2(z, r_xy))  # [-90, 90]

    # 고도각 히스토그램
    bins = int((deg_max - deg_min) / bin_deg)
    hist, edges = np.histogram(np.clip(elev, deg_min, deg_max), bins=bins, range=(deg_min, deg_max))

    # 간단 스무딩(이동평균)
    if smooth_win > 1:
        k = np.ones(smooth_win, dtype=np.float32) / smooth_win
        hist = np.convolve(hist, k, mode='same')

    # 피크 탐지(간단판)
    thr = hist.max() * peak_thresh_ratio
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] >= thr and hist[i] >= hist[i-1] and hist[i] >= hist[i+1]:
            if not peaks or (i - peaks[-1]) >= min_separation_bins:
                peaks.append(i)

    n = len(peaks)
    # 스냅 규칙(근처값 보정)
    if 56 <= n <= 72:
        return 64
    if 112 <= n <= 136:
        return 128
    return None

if __name__ == '__main__' :
    channels = estimate_channels_from_xyz(points)
    print("추정 채널 수:", channels)
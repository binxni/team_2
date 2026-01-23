import argparse
import os
import math
import re
from typing import List, Dict, Tuple

import numpy as np


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def load_csv(path: str) -> Dict[str, np.ndarray]:
    """Load CSV into column arrays. Prefers pandas if available, falls back to csv module.
    Returns dict: {column_name: np.ndarray}
    """
    try:
        import pandas as pd  # type: ignore
        df = pd.read_csv(path)
        data = {c: df[c].to_numpy() for c in df.columns}
        return data
    except Exception:
        import csv
        data_list: List[Dict[str, str]] = []
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data_list.append(row)
        if not data_list:
            raise RuntimeError("CSV appears empty: " + path)
        cols = list(data_list[0].keys())
        out: Dict[str, List[float]] = {c: [] for c in cols}
        for r in data_list:
            for c in cols:
                v = r[c]
                if c in ('index', 'label'):
                    try:
                        out[c].append(int(v))
                    except Exception:
                        out[c].append(np.nan)
                else:
                    try:
                        out[c].append(float(v) if v != '' else np.nan)
                    except Exception:
                        out[c].append(np.nan)
        return {c: np.asarray(v) for c, v in out.items()}


def cohen_d(nv: np.ndarray, cv: np.ndarray) -> float:
    nv = nv[np.isfinite(nv)]
    cv = cv[np.isfinite(cv)]
    if nv.size == 0 or cv.size == 0:
        return float('nan')
    m1, m0 = float(np.mean(nv)), float(np.mean(cv))
    s1 = float(np.std(nv, ddof=1) if nv.size > 1 else 0.0)
    s0 = float(np.std(cv, ddof=1) if cv.size > 1 else 0.0)
    n1, n0 = nv.size, cv.size
    denom = math.sqrt(((n1 - 1) * s1 * s1 + (n0 - 1) * s0 * s0) / max(n1 + n0 - 2, 1))
    if denom == 0:
        return float('nan')
    return (m1 - m0) / denom


def compute_auc(y: np.ndarray, scores: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.metrics import roc_auc_score, roc_curve
    mask = np.isfinite(scores)
    if not np.any(mask) or len(np.unique(y[mask])) < 2:
        return float('nan'), np.array([]), np.array([]), np.array([])
    auc = float(roc_auc_score(y[mask], scores[mask]))
    fpr, tpr, thr = roc_curve(y[mask], scores[mask])
    return auc, fpr, tpr, thr


def sanitize(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\-]+', '_', name)


def can_cast_to_float(arr: np.ndarray) -> bool:
    try:
        a = np.asarray(arr)
        if a.dtype.kind in 'iuf':
            return True
        # Try casting
        _ = a.astype(float)
        return True
    except Exception:
        return False


def plot_hist(ax, vals_clean: np.ndarray, vals_noise: np.ndarray, title: str):
    bins = 50
    c = vals_clean[np.isfinite(vals_clean)]
    n = vals_noise[np.isfinite(vals_noise)]
    if c.size == 0 or n.size == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        ax.set_title(title)
        return
    lo = float(min(np.min(c), np.min(n)))
    hi = float(max(np.max(c), np.max(n)))
    if lo == hi:
        hi = lo + 1e-6
    ax.hist(c, bins=bins, range=(lo, hi), alpha=0.5, label='clean', color='#1f77b4')
    ax.hist(n, bins=bins, range=(lo, hi), alpha=0.5, label='noise', color='#d62728')
    ax.axvline(np.median(c), color='#1f77b4', linestyle='--', linewidth=1)
    ax.axvline(np.median(n), color='#d62728', linestyle='--', linewidth=1)
    ax.set_title(title)
    ax.legend()


def plot_box(ax, vals_clean: np.ndarray, vals_noise: np.ndarray, title: str):
    c = vals_clean[np.isfinite(vals_clean)]
    n = vals_noise[np.isfinite(vals_noise)]
    data = [c, n]
    ax.boxplot(data, labels=['clean', 'noise'], showfliers=False)
    ax.set_title(title)


def plot_roc(ax, fpr: np.ndarray, tpr: np.ndarray, auc: float, thr: np.ndarray):
    if fpr.size == 0:
        ax.text(0.5, 0.5, 'No ROC', ha='center', va='center')
        ax.set_title('ROC (n/a)')
        return
    ax.plot(fpr, tpr, label=f'AUC={auc:.3f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
    # Best Youden J threshold marker
    j = tpr - fpr
    k = int(np.argmax(j))
    ax.scatter([fpr[k]], [tpr[k]], color='red', s=20)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC')
    ax.legend()


def main():
    ap = argparse.ArgumentParser(description='Plot noise vs clean metric visualizations from CSV')
    ap.add_argument('--csv', type=str, required=True, help='Path to noise_analysis.csv')
    ap.add_argument('--outdir', type=str, default='OpenPCDet/output/noise_viz')
    ap.add_argument('--order', type=str, default='auc', choices=['auc', 'd'], help='Rank metrics by AUC distance or |Cohen d|')
    ap.add_argument('--top-k', type=int, default=8, help='Number of top metrics to plot')
    ap.add_argument('--scatter-x', type=str, default='')
    ap.add_argument('--scatter-y', type=str, default='')
    ap.add_argument('--max-scatter', type=int, default=4000, help='Subsample points for scatter for readability')
    args = ap.parse_args()

    data = load_csv(args.csv)
    if 'label' not in data:
        raise RuntimeError('CSV must contain a label column (0=clean,1=noise)')
    y = data['label'].astype(int)

    # Identify metric columns
    metric_cols = [
        c for c in data.keys()
        if c not in ('index', 'label', 'source') and can_cast_to_float(data[c])
    ]
    if not metric_cols:
        raise RuntimeError('No metric columns found')

    # Compute separations
    rankings: List[Tuple[str, float, float, float, float]] = []  # (metric, auc, auc_dist, d, noise_minus_clean)
    for m in metric_cols:
        x = data[m].astype(float)
        auc, fpr, tpr, thr = compute_auc(y, x)
        auc_dist = abs(auc - 0.5) if np.isfinite(auc) else float('nan')
        d = cohen_d(x[y == 1], x[y == 0])
        nm = float(np.nanmean(x[y == 1]))
        cm = float(np.nanmean(x[y == 0]))
        rankings.append((m, auc, auc_dist, d, nm - cm))

    if args.order == 'auc':
        rankings.sort(key=lambda t: (-(t[2] if np.isfinite(t[2]) else -1)))
    else:
        rankings.sort(key=lambda t: (-(abs(t[3]) if np.isfinite(t[3]) else -1)))

    safe_mkdir(args.outdir)
    # Save ranking CSV
    def fmt(x: float) -> str:
        try:
            return f"{x:.6f}" if np.isfinite(x) else 'nan'
        except Exception:
            return 'nan'

    rank_path = os.path.join(args.outdir, 'ranking.csv')
    with open(rank_path, 'w') as f:
        f.write('metric,auc,auc_dist,cohen_d,noise_minus_clean\n')
        for m, auc, ad, d, diff in rankings:
            f.write(f"{m},{fmt(auc)},{fmt(ad)},{fmt(d)},{fmt(diff)}\n")

    # Plot top-k metrics
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    top = rankings[: max(1, args.top_k)]
    for m, auc, auc_dist, d, diff in top:
        x = data[m].astype(float)
        c = x[y == 0]
        n = x[y == 1]

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        plot_hist(axes[0], c, n, f"{m} (hist)")

        plot_box(axes[1], c, n, f"{m} (box)")

        auc_val, fpr, tpr, thr = compute_auc(y, x)
        plot_roc(axes[2], fpr, tpr, auc_val, thr)
        axes[2].set_title(f"ROC (AUC={auc_val:.3f})")

        fig.suptitle(f"{m} | d={d:.2f} | Δ={diff:.2f}")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = os.path.join(args.outdir, f"metric_{sanitize(m)}.png")
        fig.savefig(out, dpi=160)
        plt.close(fig)

    # Scatter of best two metrics by AUC distance
    if args.scatter_x and args.scatter_y:
        mx, my = args.scatter_x, args.scatter_y
        if mx not in metric_cols or my not in metric_cols:
            print(f"Scatter metrics not found in CSV: {mx}, {my}")
        else:
            X = data[mx].astype(float)
            Yv = data[my].astype(float)
            mask = np.isfinite(X) & np.isfinite(Yv)
            idx = np.where(mask)[0]
            if idx.size > args.max_scatter:
                idx = np.random.choice(idx, args.max_scatter, replace=False)
            fig, ax = plt.subplots(figsize=(5.2, 4.5))
            sc0 = ax.scatter(X[idx][y[idx] == 0], Yv[idx][y[idx] == 0], s=8, alpha=0.5, label='clean', c='#1f77b4')
            sc1 = ax.scatter(X[idx][y[idx] == 1], Yv[idx][y[idx] == 1], s=8, alpha=0.5, label='noise', c='#d62728')
            ax.set_xlabel(mx)
            ax.set_ylabel(my)
            ax.set_title('Scatter')
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(args.outdir, f"scatter_{sanitize(mx)}_vs_{sanitize(my)}.png"), dpi=160)
            plt.close(fig)
    else:
        # Auto-pick top-2 by AUC distance
        if len(rankings) >= 2:
            mx, my = rankings[0][0], rankings[1][0]
            X = data[mx].astype(float)
            Yv = data[my].astype(float)
            mask = np.isfinite(X) & np.isfinite(Yv)
            idx = np.where(mask)[0]
            if idx.size > args.max_scatter:
                idx = np.random.choice(idx, args.max_scatter, replace=False)
            fig, ax = plt.subplots(figsize=(5.2, 4.5))
            ax.scatter(X[idx][y[idx] == 0], Yv[idx][y[idx] == 0], s=8, alpha=0.5, label='clean', c='#1f77b4')
            ax.scatter(X[idx][y[idx] == 1], Yv[idx][y[idx] == 1], s=8, alpha=0.5, label='noise', c='#d62728')
            ax.set_xlabel(mx)
            ax.set_ylabel(my)
            ax.set_title('Scatter (top-2 by AUC)')
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(args.outdir, f"scatter_{sanitize(mx)}_vs_{sanitize(my)}.png"), dpi=160)
            plt.close(fig)

    print(f"Saved plots and ranking to: {args.outdir}")


if __name__ == '__main__':
    main()

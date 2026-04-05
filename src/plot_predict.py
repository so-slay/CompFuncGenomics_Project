"""
plots_predictions.py

Visualises the output of predict.py.
Reads predictions/chr{c}_predictions.tsv.gz for chr3, chr10, chr17.

Figures generated:
  1. Score distribution per TF per chromosome  (KDE + histogram grid)
  2. Cumulative distribution of scores per TF  (CDF grid)
  3. ATAC vs score scatter / box plots         (are ATAC-open bins scoring higher?)
  4. Per-TF top-bin count bar chart            (bins with score ≥ threshold)

All figures saved to plots/ directory.

Usage: python src/plots_predictions.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive; safe for headless runs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
PRED_DIR   = "predictions"
PLOT_DIR   = "plots"
PRED_CHRS  = [3, 10, 17]
TF_LIST    = ["CTCF", "REST", "EP300"]
THRESHOLD  = 0.5          # for counting "predicted bound" bins

# Publication-style colour palette
CHR_COLORS = {3: "#4C72B0", 10: "#55A868", 17: "#C44E52"}
TF_COLORS  = {"CTCF": "#4C72B0", "REST": "#DD8452", "EP300": "#55A868"}


# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
def load_predictions() -> dict[int, pd.DataFrame]:
    dfs = {}
    for c in PRED_CHRS:
        path = os.path.join(PRED_DIR, f"chr{c}_predictions.tsv.gz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run predict.py first."
            )
        df = pd.read_csv(path, sep="\t", compression="gzip")
        # normalise ATAC column to 0/1 int if still string
        if df["ATAC"].dtype == object:
            df["ATAC"] = df["ATAC"].map({"B": 1, "U": 0}).fillna(0).astype(int)
        dfs[c] = df
        print(f"  Loaded chr{c}: {len(df):,} bins")
    return dfs


# ─────────────────────────────────────────────
#  FIGURE 1 — Score distribution (KDE)
# ─────────────────────────────────────────────
def plot_score_distributions(dfs: dict[int, pd.DataFrame], out_dir: str):
    """
    3-row × 3-col grid: rows = chromosomes, cols = TFs.
    Each panel: histogram (bins=60) + KDE overlay.
    """
    fig, axes = plt.subplots(
        len(PRED_CHRS), len(TF_LIST),
        figsize=(14, 10), sharey=False
    )
    fig.suptitle("Prediction Score Distributions", fontsize=14, fontweight="bold")

    for row, c in enumerate(PRED_CHRS):
        df = dfs[c]
        for col, tf in enumerate(TF_LIST):
            ax = axes[row][col]
            scores = df[tf].values.astype(float)
            color  = TF_COLORS[tf]

            # histogram
            ax.hist(scores, bins=60, color=color, alpha=0.35,
                    density=True, edgecolor="none")

            # KDE (skip if too few unique values)
            if len(np.unique(scores)) > 10:
                try:
                    kde = gaussian_kde(scores, bw_method="scott")
                    xs  = np.linspace(0, 1, 400)
                    ax.plot(xs, kde(xs), color=color, lw=2)
                except Exception:
                    pass

            # threshold line
            ax.axvline(THRESHOLD, color="black", lw=0.8, ls="--", alpha=0.6)

            pct_bound = 100 * (scores >= THRESHOLD).mean()
            ax.set_title(
                f"chr{c} — {tf}\n≥0.5: {pct_bound:.1f}%",
                fontsize=9
            )
            ax.set_xlim(0, 1)
            ax.set_xlabel("Score" if row == len(PRED_CHRS) - 1 else "")
            ax.set_ylabel("Density" if col == 0 else "")
            ax.tick_params(labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "pred_score_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
#  FIGURE 2 — CDF per TF across chromosomes
# ─────────────────────────────────────────────
def plot_cdfs(dfs: dict[int, pd.DataFrame], out_dir: str):
    """
    1 row × 3 cols: each col = one TF, lines = chromosomes.
    """
    fig, axes = plt.subplots(1, len(TF_LIST), figsize=(14, 4))
    fig.suptitle("Cumulative Distribution of Prediction Scores",
                 fontsize=13, fontweight="bold")

    for col, tf in enumerate(TF_LIST):
        ax = axes[col]
        for c in PRED_CHRS:
            scores = np.sort(dfs[c][tf].values.astype(float))
            cdf    = np.arange(1, len(scores) + 1) / len(scores)
            ax.plot(scores, cdf, color=CHR_COLORS[c],
                    lw=1.8, label=f"chr{c}")

        ax.axvline(THRESHOLD, color="black", lw=0.8, ls="--", alpha=0.6,
                   label=f"thr={THRESHOLD}")
        ax.set_title(tf, fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Score")
        ax.set_ylabel("CDF" if col == 0 else "")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, "pred_score_cdfs.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
#  FIGURE 3 — Score by ATAC status (box plots)
# ─────────────────────────────────────────────
def plot_atac_stratified(dfs: dict[int, pd.DataFrame], out_dir: str):
    """
    3-row × 3-col: rows=TFs, cols=chromosomes.
    Each panel: box plot for ATAC=0 vs ATAC=1.
    Checks whether open-chromatin bins get higher binding scores.
    """
    fig, axes = plt.subplots(
        len(TF_LIST), len(PRED_CHRS),
        figsize=(12, 9), sharey="row"
    )
    fig.suptitle("Binding Score by ATAC Status",
                 fontsize=13, fontweight="bold")

    for row, tf in enumerate(TF_LIST):
        for col, c in enumerate(PRED_CHRS):
            ax   = axes[row][col]
            df   = dfs[c]
            open_  = df.loc[df["ATAC"] == 1, tf].values.astype(float)
            closed = df.loc[df["ATAC"] == 0, tf].values.astype(float)

            bp = ax.boxplot(
                [closed, open_],
                labels=["Closed", "Open"],
                patch_artist=True,
                medianprops=dict(color="black", lw=2),
                flierprops=dict(marker=".", markersize=1,
                                alpha=0.3, linestyle="none"),
                widths=0.5,
            )
            bp["boxes"][0].set_facecolor("#AEC6CF")
            bp["boxes"][1].set_facecolor(TF_COLORS[tf])

            ax.set_title(f"chr{c}", fontsize=9)
            if col == 0:
                ax.set_ylabel(tf, fontsize=10, fontweight="bold")
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # annotate n
            ax.text(0.98, 0.97,
                    f"open n={len(open_):,}\nclosed n={len(closed):,}",
                    transform=ax.transAxes, fontsize=6,
                    va="top", ha="right", color="grey")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "pred_atac_stratified.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
#  FIGURE 4 — Predicted-bound bin counts
# ─────────────────────────────────────────────
def plot_bound_counts(dfs: dict[int, pd.DataFrame], out_dir: str):
    """
    Grouped bar chart: for each TF, bars = chromosomes.
    Shows how many bins pass the threshold.
    """
    x       = np.arange(len(TF_LIST))
    width   = 0.22
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(f"Predicted-Bound Bins (score ≥ {THRESHOLD})",
                 fontsize=13, fontweight="bold")

    for i, (c, offset) in enumerate(zip(PRED_CHRS, offsets)):
        df     = dfs[c]
        counts = [(df[tf] >= THRESHOLD).sum() for tf in TF_LIST]
        bars   = ax.bar(x + offset, counts, width,
                        color=CHR_COLORS[c], label=f"chr{c}", alpha=0.85)
        # value labels on bars
        for bar, val in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 30,
                    f"{val:,}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(TF_LIST, fontsize=11)
    ax.set_ylabel("Number of bins")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(out_dir, "pred_bound_counts.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
#  FIGURE 5 — Score correlation between TFs
# ─────────────────────────────────────────────
def plot_tf_correlation(dfs: dict[int, pd.DataFrame], out_dir: str):
    """
    3 chromosomes × 3 TF-pair scatter panels.
    TF pairs: (CTCF, REST), (CTCF, EP300), (REST, EP300).
    """
    pairs = [("CTCF", "REST"), ("CTCF", "EP300"), ("REST", "EP300")]
    fig, axes = plt.subplots(
        len(PRED_CHRS), len(pairs),
        figsize=(12, 10)
    )
    fig.suptitle("Score Correlation Between TFs",
                 fontsize=13, fontweight="bold")

    rng = np.random.default_rng(42)

    for row, c in enumerate(PRED_CHRS):
        df = dfs[c]
        for col, (tf_x, tf_y) in enumerate(pairs):
            ax = axes[row][col]

            x = df[tf_x].values.astype(float)
            y = df[tf_y].values.astype(float)

            # subsample for plotting speed (max 5000 points)
            n = len(x)
            if n > 5000:
                idx = rng.choice(n, 5000, replace=False)
                x, y = x[idx], y[idx]

            ax.scatter(x, y, s=2, alpha=0.3,
                       color=CHR_COLORS[c], edgecolors="none")

            # Pearson r on full data
            full_x = df[tf_x].values.astype(float)
            full_y = df[tf_y].values.astype(float)
            r = np.corrcoef(full_x, full_y)[0, 1]
            ax.text(0.05, 0.92, f"r = {r:.3f}",
                    transform=ax.transAxes, fontsize=8,
                    color="black", va="top")

            if row == 0:
                ax.set_title(f"{tf_x} vs {tf_y}", fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"chr{c}", fontsize=9, fontweight="bold")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel(tf_x if row == len(PRED_CHRS) - 1 else "", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "pred_tf_correlation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    print(f"\n{'='*50}")
    print("  plots_predictions.py")
    print(f"{'='*50}")

    print("\nLoading prediction files...")
    dfs = load_predictions()

    print("\nGenerating figures...")
    plot_score_distributions(dfs, PLOT_DIR)
    plot_cdfs(dfs, PLOT_DIR)
    plot_atac_stratified(dfs, PLOT_DIR)
    plot_bound_counts(dfs, PLOT_DIR)
    plot_tf_correlation(dfs, PLOT_DIR)

    print(f"\nAll figures saved to: {os.path.abspath(PLOT_DIR)}/")

    # ── Print a quick text summary table ─────
    print(f"\n{'─'*65}")
    print(f"  {'Chr':>4}  {'TF':>6}  {'min':>6} {'max':>6} "
          f"{'mean':>6} {'≥0.5 %':>7} {'≥0.5 n':>8}")
    print(f"{'─'*65}")
    for c in PRED_CHRS:
        df = dfs[c]
        for tf in TF_LIST:
            p   = df[tf].values.astype(float)
            pct = 100 * (p >= THRESHOLD).mean()
            n   = (p >= THRESHOLD).sum()
            print(f"  {c:>4}  {tf:>6}  {p.min():>6.3f} {p.max():>6.3f} "
                  f"{p.mean():>6.3f} {pct:>7.2f} {n:>8,}")
    print(f"{'─'*65}\n")


if __name__ == "__main__":
    main()
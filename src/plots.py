"""
plots.py

Generates publication-style plots from metrics saved by train.py.
Produces three figures:
  1. Validation ROC-AUC per epoch per config (C-Origami style)
  2. Validation PR-AUC per epoch per config
  3. ROC curves on test set per config
  4. Precision-Recall curves on test set per config

Input:  models/checkpoints/metrics.json
Output: plots/val_roc_curves.png
        plots/val_pr_curves.png
        plots/test_roc_curves.png
        plots/test_prc_curves.png

Usage: python src/plots.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import roc_curve, auc, precision_recall_curve

METRICS_PATH = "models/checkpoints/metrics.json"
PLOTS_DIR    = "plots"

# consistent color per config across all figures
CONFIG_COLORS = {
    "DNA":               "#4e79a7",
    "DNA_ATAC":          "#f28e2b",
    "DNA_ATAC_METH":     "#59a14f",
    "DNA_ATAC_PWM":      "#e15759",
    "DNA_ATAC_METH_PWM": "#b07aa1",
}

TF_NAMES = ["CTCF", "REST", "EP300"]

def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)

# ---------------- FIG 1 & 2: val metrics per epoch ----------------
def plot_val_curves(history, metric_key, ylabel, title, outpath):
    """
    Plots val metric over epochs for each config.
    Mirrors C-Origami style: one line per config, x=epoch, y=metric.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, h in history.items():
        color = CONFIG_COLORS.get(name, "grey")
        epochs = h["epoch"]
        values = h[metric_key]
        ax.plot(epochs, values, label=name, color=color,
                linewidth=2, marker="o", markersize=4)

        # mark best epoch
        best_ep  = epochs[int(np.argmax(values))]
        best_val = max(values)
        ax.scatter(best_ep, best_val, color=color, s=80, zorder=5)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Saved {outpath}")

# ---------------- FIG 3: test ROC curves ----------------
def plot_test_roc(results, outpath):
    """
    One subplot per TF, one ROC curve per config.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ti, tf in enumerate(TF_NAMES):
        ax = axes[ti]
        for name, res in results.items():
            y_true = np.array(res["test_y"])[:, ti]
            y_pred = np.array(res["test_preds"])[:, ti]

            if len(np.unique(y_true)) < 2:
                continue

            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc     = auc(fpr, tpr)
            color       = CONFIG_COLORS.get(name, "grey")

            ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})",
                    color=color, linewidth=2)

        ax.plot([0,1],[0,1], "k--", linewidth=1, alpha=0.5)
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(f"ROC — {tf}", fontsize=12)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Test Set ROC Curves by Config", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {outpath}")

# ---------------- FIG 4: test PRC curves ----------------
def plot_test_prc(results, outpath):
    """
    One subplot per TF, one PR curve per config.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ti, tf in enumerate(TF_NAMES):
        ax = axes[ti]
        for name, res in results.items():
            y_true = np.array(res["test_y"])[:, ti]
            y_pred = np.array(res["test_preds"])[:, ti]

            if len(np.unique(y_true)) < 2:
                continue

            prec, rec, _ = precision_recall_curve(y_true, y_pred)
            pr_auc       = auc(rec, prec)
            color        = CONFIG_COLORS.get(name, "grey")

            ax.plot(rec, prec, label=f"{name} (AUC={pr_auc:.3f})",
                    color=color, linewidth=2)

        # baseline: random classifier = fraction of positives
        y_true_all = np.array(list(results.values())[0]["test_y"])[:, ti]
        baseline   = y_true_all.mean()
        ax.axhline(baseline, color="k", linestyle="--",
                   linewidth=1, alpha=0.5, label=f"Baseline ({baseline:.3f})")

        ax.set_xlabel("Recall", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_title(f"PRC — {tf}", fontsize=12)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Test Set Precision-Recall Curves by Config", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {outpath}")

# ---------------- MAIN ----------------
def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading metrics...")
    data    = load_metrics()
    history = data["history"]
    results = data["results"]

    print("Generating plots...")
    plot_val_curves(
        history, "val_roc",
        ylabel  = "Validation ROC-AUC",
        title   = "Validation ROC-AUC per Epoch by Feature Config",
        outpath = os.path.join(PLOTS_DIR, "val_roc_curves.png")
    )
    plot_val_curves(
        history, "val_pr",
        ylabel  = "Validation PR-AUC",
        title   = "Validation PR-AUC per Epoch by Feature Config",
        outpath = os.path.join(PLOTS_DIR, "val_pr_curves.png")
    )
    plot_test_roc(results,  os.path.join(PLOTS_DIR, "test_roc_curves.png"))
    plot_test_prc(results,  os.path.join(PLOTS_DIR, "test_prc_curves.png"))

    # print final summary table
    print(f"\n{'Config':<25} {'val_ROC':>8} {'test_ROC':>9} {'test_PR':>8}")
    print("-" * 55)
    for name, res in results.items():
        print(f"  {name:<23} {res['val_roc']:>8.4f} "
              f"{res['test_roc']:>9.4f} {res['test_pr']:>8.4f}")

if __name__ == "__main__":
    main()
"""
predict.py

Final prediction pipeline for TF binding on chr3, chr10, chr17.
Trains on ALL 19 labeled chromosomes, then predicts on unknown bins.

Key differences from train.py:
  - Hardcoded +ALL config (DNA + ATAC + METH + PWM)
  - Negative undersampling per chromosome (NEG_RATIO:1 neg:pos)
    to drastically cut epoch time with minimal AUC impact
  - Unknown chromosomes loaded from chr{c}_200bp_bins_unknown.tsv
    (no label columns expected)
  - TTA (fwd + RC) at inference
  - Full logging to predictions/predict.log
  - Training loss curve + per-TF prediction distribution plots

Input:  data/processed/chr*_methylation.npy
        data/processed/chr*_pwm.npy
        data/raw/tsv/chr*_200bp_bins.tsv          (labeled)
        data/raw/tsv/chr*_200bp_bins_unknown.tsv  (prediction targets)
        data/raw/FASTAs/chr*_200bp_bins.fa

Output: predictions/chr3_predictions.tsv.gz
        predictions/chr10_predictions.tsv.gz
        predictions/chr17_predictions.tsv.gz
        predictions/predict.log
        predictions/plots/loss_curve.png
        predictions/plots/pred_distributions.png
        predictions/plots/binding_rates.png

Usage:  python src/predict.py
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from noGarbageIn import (
    TRAIN_CHRS, VAL_CHRS, TEST_CHRS, PRED_CHRS, TF_LIST,
    load_chromosome, load_split_by_chr,
    encode_batch, reverse_complement,
)
from model import CNN, FocalLoss


# ─────────────────────────── CONFIG ───────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
EPOCHS     = 20
PATIENCE   = 3
LR         = 3e-4
NEG_RATIO  = 5          # negatives per positive; None = no undersampling

CKPT_DIR   = "models/checkpoints"
PRED_DIR   = "predictions"
PLOT_DIR   = os.path.join(PRED_DIR, "plots")
LOG_PATH   = os.path.join(PRED_DIR, "predict.log")

# +ALL config — hardcoded, no selection needed
CFG   = {"name": "ALL", "use_atac": True, "use_meth": True, "use_pwm": True}
IN_CH = 4 + 1 + 1 + 3   # DNA=4, ATAC=1, METH=1, PWM=3 → 9

ALL_LABELED_CHRS = TRAIN_CHRS + VAL_CHRS + TEST_CHRS   # 19 chromosomes


# ─────────────────────────── LOGGING ──────────────────────────
def setup_logging():
    os.makedirs(PRED_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_PATH, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

log = logging.getLogger(__name__)


# ──────────────────────── UNDERSAMPLING ───────────────────────
def undersample(X, y, neg_ratio):
    """
    Keep all positive bins; sample neg_ratio × n_pos negatives.
    A bin is positive if ANY TF is bound.
    Returns contiguous arrays — safe for torch.from_numpy.
    stats_dict always has 'ratio' key (None if no-op).
    """
    if neg_ratio is None:
        return X, y, {"n_orig": len(X), "n_sub": len(X), "ratio": None}

    pos_mask = y.any(axis=1)
    pos_idx  = np.where(pos_mask)[0]
    neg_idx  = np.where(~pos_mask)[0]

    n_pos  = len(pos_idx)
    n_keep = min(neg_ratio * n_pos, len(neg_idx))

    if n_keep == 0 or n_pos == 0:
        return X, y, {"n_orig": len(X), "n_sub": len(X), "ratio": None}

    chosen_neg = np.random.choice(neg_idx, size=n_keep, replace=False)
    keep_idx   = np.sort(np.concatenate([pos_idx, chosen_neg]))

    # fancy indexing → non-contiguous; force contiguous for from_numpy
    X_sub = np.ascontiguousarray(X[keep_idx])
    y_sub = np.ascontiguousarray(y[keep_idx])

    return (
        X_sub, y_sub,
        {"n_orig": len(X), "n_sub": len(keep_idx),
         "n_pos": n_pos, "n_neg_kept": n_keep, "ratio": neg_ratio},
    )


# ──────────────────────── TRAIN ONE EPOCH ─────────────────────
def train_epoch(model, optimizer, criterion,
                seqs_by_chr, atac_by_chr, meth_by_chr,
                pwm_by_chr, y_by_chr, scaler):
    """
    One training epoch over all labeled chromosomes.
    Encodes one chromosome at a time — keeps RAM bounded.
    encode_batch returns (N, 200, C) — matches how train.py feeds the CNN.
    Undersamples negatives, then batches with AMP.
    """
    model.train()
    total_loss, n_batches = 0.0, 0
    chr_order = np.random.permutation(len(seqs_by_chr))

    for ci in tqdm(chr_order, desc="  Chromosomes", leave=False):
        # (N, 200, C) — channels-last, consistent with train.py
        X_c = encode_batch(
            seqs_by_chr[ci], atac_by_chr[ci],
            meth_by_chr[ci], pwm_by_chr[ci],
            CFG["use_atac"], CFG["use_meth"], CFG["use_pwm"],
        )
        y_c = y_by_chr[ci]

        # undersample — returns contiguous arrays
        X_c, y_c, stats = undersample(X_c, y_c, NEG_RATIO)
        if stats["ratio"] is not None:
            tqdm.write(
                f"    chr undersample: {stats['n_orig']} → {stats['n_sub']} "
                f"({stats['n_pos']} pos, {stats['n_neg_kept']} neg)"
            )

        idx = np.random.permutation(len(X_c))
        for i in range(0, len(X_c), BATCH_SIZE):
            batch_idx = idx[i : i + BATCH_SIZE]

            # permuted fancy index → non-contiguous; force contiguous
            xb = torch.from_numpy(
                np.ascontiguousarray(X_c[batch_idx])
            ).float().to(DEVICE, non_blocking=True)
            
            assert xb.shape[1] == IN_CH, f"Expected {IN_CH} channels, got {xb.shape}"

            yb = torch.from_numpy(
                np.ascontiguousarray(y_c[batch_idx])
            ).float().to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches  += 1

        del X_c

    torch.cuda.empty_cache()
    return total_loss / max(n_batches, 1)


# ──────────────────────── RETRAIN ON ALL LABELED ──────────────
def retrain_full(seqs_all, atac_all, meth_all, pwm_all, y_all):
    """
    Trains +ALL on all 19 labeled chromosomes.
    Early stopping on train loss plateau (no val set).
    Returns trained model and per-epoch losses.
    """
    model     = CNN(in_ch=IN_CH, use_attn=True).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = FocalLoss(gamma=2)
    scaler    = torch.amp.GradScaler("cuda")

    log.info(
        f"Retraining +ALL on {len(seqs_all)} chromosomes | "
        f"in_ch={IN_CH} | EPOCHS={EPOCHS} | "
        f"PATIENCE={PATIENCE} | NEG_RATIO={NEG_RATIO} | device={DEVICE}"
    )

    epoch_losses = []
    best_loss    = float("inf")
    patience_ctr = 0
    os.makedirs(CKPT_DIR, exist_ok=True)

    for ep in range(1, EPOCHS + 1):
        loss = train_epoch(
            model, optimizer, criterion,
            seqs_all, atac_all, meth_all, pwm_all, y_all,
            scaler=scaler,
        )
        scheduler.step()
        epoch_losses.append(loss)

        lr_now = scheduler.get_last_lr()[0]
        log.info(f"Epoch {ep:02d}/{EPOCHS} | loss={loss:.4f} | lr={lr_now:.6f}")

        if loss < best_loss - 1e-5:
            best_loss    = loss
            patience_ctr = 0
            torch.save(
                model.state_dict(),
                os.path.join(CKPT_DIR, "retrained_model.pt"),
            )
        else:
            patience_ctr += 1
            log.info(f"  No improvement ({patience_ctr}/{PATIENCE})")
            if patience_ctr >= PATIENCE:
                log.info(f"Early stopping at epoch {ep}")
                break

    # reload best weights
    model.load_state_dict(
        torch.load(
            os.path.join(CKPT_DIR, "retrained_model.pt"),
            map_location=DEVICE,
            weights_only=True,
        )
    )
    log.info("Best retrained weights reloaded.")
    return model, epoch_losses


# ──────────────────────── PREDICT ONE CHR ─────────────────────
def predict_chromosome(model, c):
    """
    Predicts TF binding probabilities for one unknown chromosome.
    TTA: averages forward + reverse-complement predictions.
    encode_batch returns (N, 200, C) — consistent with train.py.
    Returns preds (N, 3) in [0, 1] and bin dataframe.
    """
    seqs, atac, meth, pwm, _, df = load_chromosome(
        c, test=True, augment=False
    )

    model.eval()

    def run_model(X):
        """Run model on (N, 200, C) array, return (N, 3) sigmoid probs."""
        out = []
        with torch.no_grad():
            for i in range(0, len(X), BATCH_SIZE):
                xb = torch.from_numpy(
                    np.ascontiguousarray(X[i : i + BATCH_SIZE])
                ).float().to(DEVICE, non_blocking=True)
                
                assert xb.shape[1] == IN_CH, f"Expected {IN_CH} channels, got {xb.shape}"

                with torch.amp.autocast("cuda"):
                    out.append(torch.sigmoid(model(xb)).cpu().numpy())
        return np.vstack(out)

    # forward pass — (N, 200, C)
    X_fwd     = encode_batch(seqs, atac, meth, pwm,
                             CFG["use_atac"], CFG["use_meth"], CFG["use_pwm"])
    preds_fwd = run_model(X_fwd)
    del X_fwd

    # reverse complement pass (TTA) — (N, 200, C)
    rc_seqs   = [reverse_complement(s) for s in seqs]
    X_rc      = encode_batch(rc_seqs, atac, meth, pwm,
                             CFG["use_atac"], CFG["use_meth"], CFG["use_pwm"])
    preds_rc  = run_model(X_rc)
    del X_rc

    torch.cuda.empty_cache()

    preds = (preds_fwd + preds_rc) / 2.0

    assert preds.min() >= 0.0 and preds.max() <= 1.0, (
        f"chr{c}: predictions out of [0,1]: "
        f"[{preds.min():.4f}, {preds.max():.4f}]"
    )
    return preds, df


# ──────────────────────── WRITE PREDICTIONS ───────────────────
def write_predictions(df, preds, c):
    """
    Writes chr{c}_predictions.tsv.gz:
        chr | start | end | ATAC | CTCF | REST | EP300
    TF columns are probabilities in [0, 1], 6 decimal places.
    """
    os.makedirs(PRED_DIR, exist_ok=True)

    out_df = df[["chr", "start", "end", "ATAC"]].copy()
    for i, tf in enumerate(TF_LIST):
        out_df[tf] = preds[:, i].round(6)

    out_path = os.path.join(PRED_DIR, f"chr{c}_predictions.tsv.gz")
    out_df.to_csv(out_path, sep="\t", index=False, compression="gzip")

    for tf in TF_LIST:
        col = out_df[tf]
        log.info(
            f"  chr{c} | {tf}: "
            f"min={col.min():.4f}  max={col.max():.4f}  "
            f"mean={col.mean():.4f}  "
            f">0.5: {(col > 0.5).sum()} / {len(col)}"
        )
    log.info(f"  → {out_path}  ({len(out_df)} bins)")
    return out_df


# ─────────────────────────── PLOTS ────────────────────────────
def plot_loss_curve(epoch_losses):
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses,
            marker="o", linewidth=2, color="#2196F3")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FocalLoss (train)")
    ax.set_title(f"Training Loss — +ALL config | NEG_RATIO={NEG_RATIO}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "loss_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Loss curve → {path}")


def plot_pred_distributions(all_preds):
    """Histogram of predicted probabilities per TF per chromosome."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    n_tfs  = len(TF_LIST)
    n_chrs = len(all_preds)
    colors = ["#E91E63", "#4CAF50", "#FF9800"]

    fig, axes = plt.subplots(
        n_tfs, n_chrs,
        figsize=(4 * n_chrs, 3 * n_tfs),
        sharey=False,
    )
    # always 2-D — guard against squeeze when n_tfs or n_chrs == 1
    axes = np.array(axes).reshape(n_tfs, n_chrs)

    for col_i, (c, (preds, _)) in enumerate(all_preds.items()):
        for row_i, (tf, color) in enumerate(zip(TF_LIST, colors)):
            ax   = axes[row_i, col_i]
            vals = preds[:, row_i]
            ax.hist(vals, bins=60, color=color, alpha=0.8, edgecolor="none")
            ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
            n_pos = (vals > 0.5).sum()
            ax.set_title(
                f"chr{c} | {tf}\n>{0.5}: {n_pos} ({100*n_pos/len(vals):.1f}%)",
                fontsize=9,
            )
            ax.set_xlabel("P(bound)", fontsize=8)
            ax.set_ylabel("Bins", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle("+ALL Model — Prediction Distributions",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "pred_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Prediction distributions → {path}")


def plot_summary_bar(all_preds):
    """Bar chart: % bins predicted bound per TF per chromosome."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    chrs   = list(all_preds.keys())
    x      = np.arange(len(chrs))
    width  = 0.25
    colors = ["#E91E63", "#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (tf, color) in enumerate(zip(TF_LIST, colors)):
        fracs = [(all_preds[c][0][:, i] > 0.5).mean() * 100 for c in chrs]
        ax.bar(x + i * width, fracs, width, label=tf, color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"chr{c}" for c in chrs])
    ax.set_ylabel("% bins predicted bound (p > 0.5)")
    ax.set_title("+ALL Model — Predicted Binding Rates")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "binding_rates.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Binding rates → {path}")


# ─────────────────────────── MAIN ─────────────────────────────
def main():
    setup_logging()

    log.info("=" * 60)
    log.info("predict.py — +ALL config | TTA | undersampling")
    log.info(f"  TFs       : {TF_LIST}")
    log.info(f"  Predict   : chr{PRED_CHRS}")
    log.info(f"  Train chrs: {len(ALL_LABELED_CHRS)} (all labeled)")
    log.info(f"  NEG_RATIO : {NEG_RATIO}")
    log.info(f"  EPOCHS    : {EPOCHS}  PATIENCE={PATIENCE}")
    log.info(f"  DEVICE    : {DEVICE}")
    log.info("=" * 60)

    # 1. load all 19 labeled chromosomes (no augment — halves RAM)
    log.info(f"\nLoading {len(ALL_LABELED_CHRS)} labeled chromosomes...")
    seqs_all, atac_all, meth_all, pwm_all, y_all = load_split_by_chr(
        ALL_LABELED_CHRS, augment=False
    )

    # 2. retrain +ALL
    model, epoch_losses = retrain_full(
        seqs_all, atac_all, meth_all, pwm_all, y_all
    )

    # save run metadata
    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(os.path.join(CKPT_DIR, "retrained_config.json"), "w") as f:
        json.dump(
            {**CFG, "in_ch": IN_CH,
             "epochs_run": len(epoch_losses),
             "neg_ratio": NEG_RATIO,
             "final_loss": round(epoch_losses[-1], 6)},
            f, indent=2,
        )

    # 3. loss curve
    plot_loss_curve(epoch_losses)

    # 4. predict on unknown chromosomes
    log.info(f"\nPredicting on chromosomes {PRED_CHRS}...")
    all_preds = {}

    for c in PRED_CHRS:
        log.info(f"\n  chr{c}:")
        preds, df    = predict_chromosome(model, c)
        out_df       = write_predictions(df, preds, c)
        all_preds[c] = (preds, out_df)

    # 5. summary plots
    plot_pred_distributions(all_preds)
    plot_summary_bar(all_preds)

    log.info("\n" + "=" * 60)
    log.info("Done.")
    log.info(f"  Predictions → {PRED_DIR}/")
    log.info(f"  Plots       → {PLOT_DIR}/")
    log.info(f"  Log         → {LOG_PATH}")
    log.info(f"  Checkpoint  → {CKPT_DIR}/retrained_model.pt")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
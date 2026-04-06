"""
predict.py

Self-contained train + predict pipeline.
No model selection — hardcoded +ALL config (DNA + ATAC + METH + PWM).

Training chromosomes: all labeled chrs except 3, 10, 17
  [1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22]

Prediction chromosomes: 3, 10, 17

Output:
  predictions/chr{3,10,17}_predictions.tsv.gz
  predictions/loss_curve.png
  predictions/pred_distributions.png
  predictions/predict.log
  models/checkpoints/retrained_model.pt
  models/checkpoints/retrained_config.json

Usage: python src/predict.py
"""

import os
import sys
import copy
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from noGarbageIn import (
    PRED_CHRS,
    load_chromosome,
    encode_batch,
    reverse_complement,
)
from model import CNN, FocalLoss


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
EPOCHS     = 20
PATIENCE   = 5          # early stopping on train loss
LR         = 3e-4
NEG_RATIO  = 5          # negatives per positive per chromosome; None = use all

# All chromosomes except the three prediction targets
TRAIN_CHRS = [1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22]

# +ALL feature config — hardcoded, no selection
USE_ATAC = True
USE_METH = True
USE_PWM  = True
IN_CH    = 4 + 1 + 1 + 3   # DNA=4, ATAC=1, METH=1, PWM=3 → 9

TF_LIST  = ["CTCF", "REST", "EP300"]

CKPT_DIR = "models/checkpoints"
PRED_DIR = "predictions"
LOG_PATH = os.path.join(PRED_DIR, "predict.log")


# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
def setup_logging():
    os.makedirs(PRED_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_PATH, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  UNDERSAMPLING
# ─────────────────────────────────────────────
def undersample(X, y, neg_ratio):
    """
    Keep all positive bins; sample neg_ratio x n_pos negatives.
    A bin is positive if ANY TF is bound (any column == 1).
    Returns contiguous arrays safe for torch.from_numpy.
    """
    if neg_ratio is None:
        return X, y

    pos_mask = y.any(axis=1)
    pos_idx  = np.where(pos_mask)[0]
    neg_idx  = np.where(~pos_mask)[0]

    n_pos  = len(pos_idx)
    n_keep = min(neg_ratio * n_pos, len(neg_idx))

    if n_pos == 0 or n_keep == 0:
        return X, y

    chosen_neg = np.random.choice(neg_idx, size=n_keep, replace=False)
    keep_idx   = np.sort(np.concatenate([pos_idx, chosen_neg]))

    return (
        np.ascontiguousarray(X[keep_idx]),
        np.ascontiguousarray(y[keep_idx]),
    )


# ─────────────────────────────────────────────
#  TRAIN ONE EPOCH
# ─────────────────────────────────────────────
def train_epoch(model, optimizer, criterion,
                seqs_by_chr, atac_by_chr, meth_by_chr, pwm_by_chr, y_by_chr):
    """
    Encodes one chromosome at a time — keeps RAM bounded.
    encode_batch returns (N, 200, C).
    CNN.forward() does permute(0,2,1) internally — do NOT permute here.
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0
    chr_order  = np.random.permutation(len(seqs_by_chr))

    for ci in tqdm(chr_order, desc="  chrs", leave=False):
        # (N, 200, C) — CNN.forward permutes to (N, C, 200) internally
        X_c = encode_batch(
            seqs_by_chr[ci], atac_by_chr[ci],
            meth_by_chr[ci], pwm_by_chr[ci],
            USE_ATAC, USE_METH, USE_PWM,
        )
        y_c = y_by_chr[ci]

        X_c, y_c = undersample(X_c, y_c, NEG_RATIO)

        idx = np.random.permutation(len(X_c))

        for i in range(0, len(X_c), BATCH_SIZE):
            batch_idx = idx[i : i + BATCH_SIZE]

            # fancy index → non-contiguous; force contiguous for from_numpy
            xb = torch.from_numpy(
                np.ascontiguousarray(X_c[batch_idx])
            ).float().to(DEVICE)

            yb = torch.from_numpy(
                np.ascontiguousarray(y_c[batch_idx])
            ).float().to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        del X_c
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return total_loss / max(n_batches, 1)


# ─────────────────────────────────────────────
#  LOAD ALL TRAINING CHROMOSOMES
# ─────────────────────────────────────────────
def load_train_data():
    """
    Loads raw sequences + scalars for all training chromosomes.
    Does NOT encode — encoding happens per-chromosome in train_epoch.
    augment=True doubles data with reverse complements for free.
    """
    seqs_list, atac_list, meth_list, pwm_list, y_list = [], [], [], [], []

    for c in TRAIN_CHRS:
        seqs, atac, meth, pwm, y, _ = load_chromosome(c, augment=True)
        seqs_list.append(seqs)
        atac_list.append(atac)
        meth_list.append(meth)
        pwm_list.append(pwm)
        y_list.append(y)
        log.info(f"  Loaded chr{c}: {len(seqs):,} sequences")

    return seqs_list, atac_list, meth_list, pwm_list, y_list


# ─────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────
def train(seqs_all, atac_all, meth_all, pwm_all, y_all):
    """
    Trains +ALL config on all TRAIN_CHRS.
    Early stopping on train loss — no val set, using all data.
    Saves best weights to CKPT_DIR/retrained_model.pt.
    Returns (trained model, list of per-epoch losses).
    """
    os.makedirs(CKPT_DIR, exist_ok=True)

    model     = CNN(in_ch=IN_CH, use_attn=True).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = FocalLoss(gamma=2)

    log.info(f"\nTraining +ALL | in_ch={IN_CH} | "
             f"{len(TRAIN_CHRS)} chromosomes | DEVICE={DEVICE}")
    log.info(f"EPOCHS={EPOCHS} | PATIENCE={PATIENCE} | NEG_RATIO={NEG_RATIO}")

    best_loss    = float("inf")
    best_state   = None
    patience_ctr = 0
    epoch_losses = []

    for ep in range(1, EPOCHS + 1):
        loss = train_epoch(
            model, optimizer, criterion,
            seqs_all, atac_all, meth_all, pwm_all, y_all,
        )
        scheduler.step()
        epoch_losses.append(loss)

        lr_now = scheduler.get_last_lr()[0]
        log.info(f"Epoch {ep:02d}/{EPOCHS} | loss={loss:.4f} | lr={lr_now:.6f}")

        if loss < best_loss - 1e-5:
            best_loss    = loss
            best_state   = copy.deepcopy(model.state_dict())
            patience_ctr = 0
            torch.save(best_state,
                       os.path.join(CKPT_DIR, "retrained_model.pt"))
        else:
            patience_ctr += 1
            log.info(f"  No improvement ({patience_ctr}/{PATIENCE})")
            if patience_ctr >= PATIENCE:
                log.info(f"  Early stopping at epoch {ep}")
                break

    # reload best weights before returning
    model.load_state_dict(best_state)
    model.eval()
    log.info(f"Best train loss: {best_loss:.4f}")
    return model, epoch_losses


# ─────────────────────────────────────────────
#  INFERENCE HELPERS
# ─────────────────────────────────────────────
def _run_model(model, X):
    """
    Inference pass on pre-encoded X of shape (N, 200, C).
    CNN.forward() does the permute — do NOT permute here.
    Returns sigmoid probabilities (N, 3).
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = torch.from_numpy(
                np.ascontiguousarray(X[i : i + BATCH_SIZE])
            ).float().to(DEVICE)
            preds.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.vstack(preds)   # (N, 3)


def predict_chromosome(model, c):
    """
    Loads unknown chromosome, runs fwd + RC TTA, returns (preds, df).
    atac/meth/pwm scalars are strand-invariant — reused for RC pass.
    """
    # augment=False — we handle RC manually for TTA
    seqs, atac, meth, pwm, _, df = load_chromosome(c, augment=False)
    log.info(f"  chr{c}: {len(seqs):,} bins")

    # forward pass
    X_fwd = encode_batch(seqs, atac, meth, pwm, USE_ATAC, USE_METH, USE_PWM)
    p_fwd = _run_model(model, X_fwd)
    del X_fwd

    # reverse complement pass (TTA)
    rc_seqs = [reverse_complement(s) for s in seqs]
    X_rc    = encode_batch(rc_seqs, atac, meth, pwm, USE_ATAC, USE_METH, USE_PWM)
    p_rc    = _run_model(model, X_rc)
    del X_rc

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    preds = (p_fwd + p_rc) / 2.0    # (N, 3)

    for i, tf in enumerate(TF_LIST):
        col = preds[:, i]
        log.info(f"    {tf}: min={col.min():.4f}  max={col.max():.4f}  "
                 f"mean={col.mean():.4f}  "
                 f">0.5: {(col > 0.5).sum():,} / {len(col):,}")

    return preds, df


# ─────────────────────────────────────────────
#  WRITE OUTPUT TSV
# ─────────────────────────────────────────────
def write_predictions(df, preds, c):
    out_df = df[["chr", "start", "end", "ATAC"]].copy()
    for i, tf in enumerate(TF_LIST):
        out_df[tf] = preds[:, i].round(6)

    out_path = os.path.join(PRED_DIR, f"chr{c}_predictions.tsv.gz")
    out_df.to_csv(out_path, sep="\t", index=False, compression="gzip")
    log.info(f"  Saved → {out_path}")
    return out_df


# ─────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────
def plot_loss_curve(epoch_losses):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses,
            marker="o", lw=2, color="#2196F3")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FocalLoss (train)")
    ax.set_title(f"+ALL | NEG_RATIO={NEG_RATIO} | "
                 f"{len(TRAIN_CHRS)} train chromosomes")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path = os.path.join(PRED_DIR, "loss_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Loss curve → {path}")


def plot_pred_distributions(all_preds):
    colors = {"CTCF": "#4C72B0", "REST": "#DD8452", "EP300": "#55A868"}
    n_chrs = len(all_preds)

    fig, axes = plt.subplots(
        len(TF_LIST), n_chrs,
        figsize=(4 * n_chrs, 3 * len(TF_LIST)),
    )
    axes = np.array(axes).reshape(len(TF_LIST), n_chrs)

    for col_i, (c, (preds, _)) in enumerate(all_preds.items()):
        for row_i, tf in enumerate(TF_LIST):
            ax   = axes[row_i, col_i]
            vals = preds[:, row_i]
            ax.hist(vals, bins=60, color=colors[tf], alpha=0.8, edgecolor="none")
            ax.axvline(0.5, color="black", ls="--", lw=1)
            n_pos = (vals > 0.5).sum()
            ax.set_title(
                f"chr{c} | {tf}  >0.5: {n_pos:,} ({100*n_pos/len(vals):.1f}%)",
                fontsize=9,
            )
            ax.set_xlabel("P(bound)", fontsize=8)
            ax.set_ylabel("Bins", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.suptitle("+ALL — Prediction Score Distributions",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PRED_DIR, "pred_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Prediction distributions → {path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    setup_logging()

    log.info("=" * 60)
    log.info("predict.py — +ALL config | self-contained train+predict")
    log.info(f"  Train chrs : {TRAIN_CHRS}")
    log.info(f"  Pred chrs  : {list(PRED_CHRS)}")
    log.info(f"  IN_CH      : {IN_CH}  (DNA + ATAC + METH + PWM)")
    log.info(f"  DEVICE     : {DEVICE}")
    log.info("=" * 60)

    # 1. load training data (raw sequences, no encoding yet)
    log.info(f"\nLoading {len(TRAIN_CHRS)} training chromosomes...")
    seqs_all, atac_all, meth_all, pwm_all, y_all = load_train_data()

    # 2. train
    model, epoch_losses = train(seqs_all, atac_all, meth_all, pwm_all, y_all)

    # free training RAM before prediction
    del seqs_all, atac_all, meth_all, pwm_all, y_all
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # save run metadata
    with open(os.path.join(CKPT_DIR, "retrained_config.json"), "w") as f:
        json.dump({
            "config":     "+ALL",
            "use_atac":   USE_ATAC,
            "use_meth":   USE_METH,
            "use_pwm":    USE_PWM,
            "in_ch":      IN_CH,
            "train_chrs": TRAIN_CHRS,
            "pred_chrs":  list(PRED_CHRS),
            "epochs_run": len(epoch_losses),
            "neg_ratio":  NEG_RATIO,
            "final_loss": round(epoch_losses[-1], 6),
        }, f, indent=2)

    plot_loss_curve(epoch_losses)

    # 3. predict on unknown chromosomes
    log.info(f"\nPredicting on chromosomes {list(PRED_CHRS)}...")
    all_preds = {}

    for c in PRED_CHRS:
        log.info(f"\nchr{c}:")
        preds, df    = predict_chromosome(model, c)
        out_df       = write_predictions(df, preds, c)
        all_preds[c] = (preds, out_df)

    plot_pred_distributions(all_preds)

    log.info("\n" + "=" * 60)
    log.info("Done.")
    log.info(f"  Predictions → {PRED_DIR}/")
    log.info(f"  Checkpoint  → {CKPT_DIR}/retrained_model.pt")
    log.info(f"  Log         → {LOG_PATH}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
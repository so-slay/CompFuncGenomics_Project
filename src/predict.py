"""
predict.py

Self-contained train + predict pipeline.
No model selection — hardcoded +ALL config (DNA + ATAC + METH + PWM).

Memory strategy: chromosomes are loaded from disk one at a time inside
each epoch and immediately deleted after. Nothing is held in RAM between
chromosomes. Peak RAM = one encoded chromosome at a time (~200-400 MB).

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
import gc
import copy
import json
import logging
import numpy as np
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
PATIENCE   = 5
LR         = 3e-4
NEG_RATIO  = 5      # negatives per positive; None = use all

# All chromosomes except prediction targets
TRAIN_CHRS = [1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22]

# +ALL feature config — hardcoded
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
    A bin is positive if ANY TF is bound.
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
#  TRAIN ONE EPOCH  — streams from disk
# ─────────────────────────────────────────────
def train_epoch(model, optimizer, criterion):
    """
    Loads each chromosome from disk, encodes it, trains on it, deletes it.
    Peak RAM = one encoded chromosome (~200-400 MB).
    augment=True inside load_chromosome doubles sequences with RC.
    encode_batch returns (N, 200, C) — CNN.forward() permutes internally.
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0
    chr_order  = np.random.permutation(len(TRAIN_CHRS))

    for ci in tqdm(chr_order, desc="  chrs", leave=False):
        c = TRAIN_CHRS[ci]

        # ── load from disk ──────────────────────────────────
        seqs, atac, meth, pwm, y, _ = load_chromosome(c, augment=True)

        # ── encode (N, 200, 9) ──────────────────────────────
        X = encode_batch(seqs, atac, meth, pwm, USE_ATAC, USE_METH, USE_PWM)

        # sequences no longer needed after encoding
        del seqs, atac, meth, pwm
        gc.collect()

        # ── undersample ─────────────────────────────────────
        X, y = undersample(X, y, NEG_RATIO)

        # ── mini-batch SGD ──────────────────────────────────
        idx = np.random.permutation(len(X))

        for i in range(0, len(X), BATCH_SIZE):
            batch_idx = idx[i : i + BATCH_SIZE]

            xb = torch.from_numpy(
                np.ascontiguousarray(X[batch_idx])
            ).float().to(DEVICE)

            yb = torch.from_numpy(
                np.ascontiguousarray(y[batch_idx])
            ).float().to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        # ── free chromosome memory before next chr ──────────
        del X, y
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return total_loss / max(n_batches, 1)


# ─────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────
def train():
    """
    Trains +ALL config on all TRAIN_CHRS.
    Early stopping on train loss — using all data, no val set.
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
    log.info("Memory mode: streaming from disk — one chromosome at a time")

    best_loss    = float("inf")
    best_state   = None
    patience_ctr = 0
    epoch_losses = []

    for ep in range(1, EPOCHS + 1):
        loss = train_epoch(model, optimizer, criterion)
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
            log.info(f"  New best — checkpoint saved")
        else:
            patience_ctr += 1
            log.info(f"  No improvement ({patience_ctr}/{PATIENCE})")
            if patience_ctr >= PATIENCE:
                log.info(f"  Early stopping at epoch {ep}")
                break

    model.load_state_dict(best_state)
    model.eval()
    log.info(f"Best train loss: {best_loss:.4f}")
    return model, epoch_losses


# ─────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────
def _run_model(model, X):
    """
    Inference on pre-encoded X of shape (N, 200, C).
    CNN.forward() does permute internally — do NOT permute here.
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
    return np.vstack(preds)


def predict_chromosome(model, c):
    """
    Loads unknown chromosome, runs fwd + RC TTA, returns (preds, df).
    Encodes one strand at a time — frees before encoding the other.
    """
    # augment=False — RC handled manually for TTA
    seqs, atac, meth, pwm, _, df = load_chromosome(c, test=True, augment=False)
    log.info(f"  chr{c}: {len(seqs):,} bins")

    # forward pass
    X_fwd = encode_batch(seqs, atac, meth, pwm, USE_ATAC, USE_METH, USE_PWM)
    p_fwd = _run_model(model, X_fwd)
    del X_fwd
    gc.collect()

    # RC pass
    rc_seqs = [reverse_complement(s) for s in seqs]
    del seqs   # free original sequences before allocating RC array
    gc.collect()

    X_rc  = encode_batch(rc_seqs, atac, meth, pwm, USE_ATAC, USE_METH, USE_PWM)
    p_rc  = _run_model(model, X_rc)
    del X_rc, rc_seqs, atac, meth, pwm
    gc.collect()

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    preds = (p_fwd + p_rc) / 2.0

    for i, tf in enumerate(TF_LIST):
        col = preds[:, i]
        log.info(f"    {tf}: min={col.min():.4f}  max={col.max():.4f}  "
                 f"mean={col.mean():.4f}  "
                 f">0.5: {(col > 0.5).sum():,} / {len(col):,}")

    return preds, df


# ─────────────────────────────────────────────
#  WRITE OUTPUT
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
    log.info("predict.py — +ALL | streaming from disk | train+predict")
    log.info(f"  Train chrs : {TRAIN_CHRS}")
    log.info(f"  Pred chrs  : {list(PRED_CHRS)}")
    log.info(f"  IN_CH      : {IN_CH}  (DNA + ATAC + METH + PWM)")
    log.info(f"  DEVICE     : {DEVICE}")
    log.info("=" * 60)

    ckpt_path = os.path.join(CKPT_DIR, "retrained_model.pt")

    if os.path.exists(ckpt_path):
        log.info(f"\nCheckpoint found — skipping training.")
        model = CNN(in_ch=IN_CH, use_attn=True).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval()
        epoch_losses = []
    else:
        model, epoch_losses = train()

    # save run metadata
    os.makedirs(CKPT_DIR, exist_ok=True)
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
            "final_loss": round(epoch_losses[-1], 6) if epoch_losses else None,
        }, f, indent=2)

    if epoch_losses:
        plot_loss_curve(epoch_losses)

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
    log.info(f"  Checkpoint  → {ckpt_path}")
    log.info(f"  Log         → {LOG_PATH}")
    log.info("=" * 60)

    # Safety check:
    assert not any(c in TRAIN_CHRS for c in PRED_CHRS), \
        "Prediction chromosomes leaked into training!"


if __name__ == "__main__":
    main()

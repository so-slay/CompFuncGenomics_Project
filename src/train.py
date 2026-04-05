"""
train.py

Global training pipeline for TF binding prediction.
Pools all training chromosomes, validates on VAL_CHRS, tests on TEST_CHRS.
Trains five input configs independently, selects best by val ROC-AUC.
Saves per-epoch metrics (loss, val_roc, val_pr) to JSON for plotting.
Uses CosineAnnealingLR, early stopping (patience=5), FocalLoss.
TTA (fwd + RC average) applied at evaluation.
Fully CUDA-aware. Encodes one chromosome at a time during training.

Input:  data processed by noGarbageIn.py, model from model.py
Output: models/checkpoints/best_model.pt
        models/checkpoints/best_config.json
        models/checkpoints/metrics.json      ← for plots.py

Usage: python src/train.py
"""

import os
import json
import copy
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from noGarbageIn import (
    TRAIN_CHRS, VAL_CHRS, TEST_CHRS,
    load_chromosome, load_split,
    encode_batch, reverse_complement
)
from model import CNN, FocalLoss

# ---------------- CONFIG ----------------
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
EPOCHS     = 20
LR         = 1e-3
PATIENCE   = 5        # was 3 — gives model more room before early stopping
CKPT_DIR   = "models/checkpoints"

CONFIGS = [
    {"name": "DNA",               "use_atac": False, "use_meth": False, "use_pwm": False},
    {"name": "DNA_ATAC",          "use_atac": True,  "use_meth": False, "use_pwm": False},
    {"name": "DNA_ATAC_METH",     "use_atac": True,  "use_meth": True,  "use_pwm": False},
    {"name": "DNA_ATAC_PWM",      "use_atac": True,  "use_meth": False, "use_pwm": True},
    {"name": "DNA_ATAC_METH_PWM", "use_atac": True,  "use_meth": True,  "use_pwm": True},
]

# ---------------- HELPERS ----------------
def n_channels(use_atac, use_meth, use_pwm):
    return 4 + int(use_atac) + int(use_meth) + (3 if use_pwm else 0)


def load_split_by_chr(chrs, augment=False):
    """
    Returns per-chromosome lists (not concatenated) to keep
    memory bounded during training. Each list entry = one chromosome.
    """
    seqs_list, atac_list, meth_list, pwm_list, y_list = [], [], [], [], []
    for c in chrs:
        seqs, atac, meth, pwm, y, _ = load_chromosome(c, augment=augment)
        seqs_list.append(seqs)
        atac_list.append(atac)
        meth_list.append(meth)
        pwm_list.append(pwm)
        y_list.append(y)
        print(f"  Loaded chr{c}: {len(seqs)} sequences")
    return seqs_list, atac_list, meth_list, pwm_list, y_list


# ---------------- TRAIN ONE EPOCH ----------------
def train_epoch(model, optimizer, criterion,
                seqs_by_chr, atac_by_chr, meth_by_chr, pwm_by_chr, y_by_chr,
                use_atac, use_meth, use_pwm):
    """
    Encodes and trains one chromosome at a time to keep RAM bounded.
    Frees encoded array immediately after each chromosome.
    Chromosomes processed in random order each epoch.
    """
    model.train()
    total_loss = 0
    n_batches  = 0
    chr_order  = np.random.permutation(len(seqs_by_chr))

    for ci in tqdm(chr_order, desc="  Chromosomes"):
        X_c = encode_batch(
            seqs_by_chr[ci],
            atac_by_chr[ci],
            meth_by_chr[ci],
            pwm_by_chr[ci],
            use_atac, use_meth, use_pwm
        )
        y_c = y_by_chr[ci]
        idx = np.random.permutation(len(X_c))

        for i in range(0, len(X_c), BATCH_SIZE):
            batch_idx = idx[i:i + BATCH_SIZE]
            xb = torch.tensor(X_c[batch_idx]).to(DEVICE)
            yb = torch.tensor(y_c[batch_idx]).to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        del X_c
        torch.cuda.empty_cache()

    return total_loss / n_batches


# ---------------- EVALUATE ----------------
def evaluate(model, X, y,
             atac=None, meth=None, pwm=None,
             use_atac=False, use_meth=False, use_pwm=False,
             seqs=None):
    """
    Evaluates model on pre-encoded X, y.
    If seqs provided, applies TTA: averages predictions on
    forward + reverse complement sequences.
    Returns mean ROC-AUC, mean avg-precision, and raw predictions (N, 3).
    """
    model.eval()
    preds_fwd = []

    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = torch.tensor(X[i:i + BATCH_SIZE]).to(DEVICE)
            preds_fwd.append(torch.sigmoid(model(xb)).cpu().numpy())

    preds_fwd = np.vstack(preds_fwd)

    # --- TTA ---
    if seqs is not None:
        rc_seqs  = [reverse_complement(s) for s in seqs]
        X_rc     = encode_batch(rc_seqs, atac, meth, pwm,
                                use_atac, use_meth, use_pwm)
        preds_rc = []
        with torch.no_grad():
            for i in range(0, len(X_rc), BATCH_SIZE):
                xb = torch.tensor(X_rc[i:i + BATCH_SIZE]).to(DEVICE)
                preds_rc.append(torch.sigmoid(model(xb)).cpu().numpy())
        preds_rc  = np.vstack(preds_rc)
        preds_fwd = (preds_fwd + preds_rc) / 2
        del X_rc

    # --- metrics per TF ---
    roc_scores, pr_scores = [], []
    for i in range(3):
        if len(np.unique(y[:, i])) < 2:
            continue
        roc_scores.append(roc_auc_score(y[:, i], preds_fwd[:, i]))
        pr_scores.append(average_precision_score(y[:, i], preds_fwd[:, i]))

    return np.mean(roc_scores), np.mean(pr_scores), preds_fwd


# ---------------- TRAIN ONE CONFIG ----------------
def train_config(cfg, in_ch,
                 seqs_tr, atac_tr, meth_tr, pwm_tr, y_tr,
                 X_val, y_val, seqs_val, atac_val, meth_val, pwm_val):
    """
    Trains a single feature config with early stopping and LR scheduling.
    Records per-epoch metrics for plotting.
    Returns best val ROC, best state dict, in_ch, and epoch metrics dict.
    """
    use_atac = cfg["use_atac"]
    use_meth = cfg["use_meth"]
    use_pwm  = cfg["use_pwm"]
    name     = cfg["name"]

    model     = CNN(in_ch=in_ch, use_attn=True).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = FocalLoss(gamma=2)

    best_val_roc   = 0
    best_state     = None
    patience_count = 0

    # per-epoch record for plots.py
    history = {
        "epoch": [], "loss": [], "val_roc": [], "val_pr": [], "lr": []
    }

    print(f"\n{'='*50}")
    print(f"Config: {name} | in_ch={in_ch} | device={DEVICE}")
    print(f"{'='*50}")

    for ep in range(1, EPOCHS + 1):
        loss = train_epoch(
            model, optimizer, criterion,
            seqs_tr, atac_tr, meth_tr, pwm_tr, y_tr,
            use_atac, use_meth, use_pwm
        )
        val_roc, val_pr, _ = evaluate(
            model, X_val, y_val,
            atac=atac_val, meth=meth_val, pwm=pwm_val,
            use_atac=use_atac, use_meth=use_meth, use_pwm=use_pwm,
            seqs=seqs_val
        )
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        print(f"  Epoch {ep:02d} | loss={loss:.4f} | "
              f"val_ROC={val_roc:.4f} | val_PR={val_pr:.4f} | "
              f"lr={current_lr:.6f}")

        history["epoch"].append(ep)
        history["loss"].append(float(loss))
        history["val_roc"].append(float(val_roc))
        history["val_pr"].append(float(val_pr))
        history["lr"].append(float(current_lr))

        if val_roc > best_val_roc:
            best_val_roc   = val_roc
            best_state     = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stopping at epoch {ep} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    return best_val_roc, best_state, in_ch, history


# ---------------- MAIN ----------------
def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"\nDevice: {DEVICE}")

    # --- Load training data per chromosome ---
    print("\nLoading training chromosomes (raw, no encoding)...")
    seqs_tr, atac_tr, meth_tr, pwm_tr, y_tr = load_split_by_chr(
        TRAIN_CHRS, augment=True
    )

    # --- Val and test: small enough to load fully ---
    print("\nLoading validation split...")
    seqs_val, atac_val, meth_val, pwm_val, y_val, _ = load_split(
        VAL_CHRS, augment=False
    )

    print("\nLoading test split...")
    seqs_te, atac_te, meth_te, pwm_te, y_te, _ = load_split(
        TEST_CHRS, augment=False
    )

    results    = {}
    all_history = {}
    best_roc   = 0
    best_cfg   = None
    best_state = None
    best_in_ch = None

    for cfg in CONFIGS:
        torch.cuda.empty_cache()

        use_atac = cfg["use_atac"]
        use_meth = cfg["use_meth"]
        use_pwm  = cfg["use_pwm"]
        name     = cfg["name"]
        in_ch    = n_channels(use_atac, use_meth, use_pwm)

        # encode val and test once per config
        print(f"\nEncoding val/test for {name}...")
        X_val = encode_batch(seqs_val, atac_val, meth_val, pwm_val,
                             use_atac, use_meth, use_pwm)
        X_te  = encode_batch(seqs_te,  atac_te,  meth_te,  pwm_te,
                             use_atac, use_meth, use_pwm)

        val_roc, state, in_ch, history = train_config(
            cfg, in_ch,
            seqs_tr, atac_tr, meth_tr, pwm_tr, y_tr,
            X_val, y_val, seqs_val, atac_val, meth_val, pwm_val
        )

        # evaluate on test with best epoch weights + TTA
        test_model = CNN(in_ch=in_ch, use_attn=True).to(DEVICE)
        test_model.load_state_dict(state)
        test_roc, test_pr, test_preds = evaluate(
            test_model, X_te, y_te,
            atac=atac_te, meth=meth_te, pwm=pwm_te,
            use_atac=use_atac, use_meth=use_meth, use_pwm=use_pwm,
            seqs=seqs_te
        )

        print(f"\n  {name} | val_ROC={val_roc:.4f} | "
              f"test_ROC={test_roc:.4f} | test_PR={test_pr:.4f}")

        results[name] = {
            "val_roc":    float(val_roc),
            "test_roc":   float(test_roc),
            "test_pr":    float(test_pr),
            "test_preds": test_preds.tolist(),   # saved for ROC/PRC plots
            "test_y":     y_te.tolist(),
        }
        all_history[name] = history

        if val_roc > best_roc:
            best_roc   = val_roc
            best_cfg   = cfg
            best_state = state
            best_in_ch = in_ch

        del X_val, X_te, test_model
        torch.cuda.empty_cache()

    # --- Save best model ---
    torch.save(best_state, os.path.join(CKPT_DIR, "best_model.pt"))

    config_out = {
        "name":     best_cfg["name"],
        "use_atac": best_cfg["use_atac"],
        "use_meth": best_cfg["use_meth"],
        "use_pwm":  best_cfg["use_pwm"],
        "in_ch":    best_in_ch,
        "val_roc":  float(best_roc),
    }
    with open(os.path.join(CKPT_DIR, "best_config.json"), "w") as f:
        json.dump(config_out, f, indent=2)

    # --- Save all metrics for plots.py ---
    metrics_out = {
        "history": all_history,   # per-epoch loss/roc/pr per config
        "results": results,       # final test metrics + predictions
    }
    with open(os.path.join(CKPT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    # --- Summary ---
    print(f"\n{'='*50}")
    print("RESULTS SUMMARY")
    print(f"{'='*50}")
    for name, scores in results.items():
        marker = " ← BEST" if name == best_cfg["name"] else ""
        print(f"  {name:25s} val={scores['val_roc']:.4f}  "
              f"test_ROC={scores['test_roc']:.4f}  "
              f"test_PR={scores['test_pr']:.4f}{marker}")

    print(f"\nBest config : {best_cfg['name']}")
    print(f"Saved to    : {CKPT_DIR}/")


if __name__ == "__main__":
    main()
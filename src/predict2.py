"""
predict.py

Loads best_model.pt + best_config.json and predicts TF binding
probabilities on PRED_CHRS (chr3, chr10, chr17).

Applies TTA (forward + reverse complement average).
Encodes each chromosome independently — RAM safe.

Output: predictions/chr{c}_predictions.tsv.gz
        predictions/chr{c}_predictions_summary.txt  (score stats)

Usage: python src/predict.py
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch

# ── make sure src/ is on the path so imports work whether you
#    run from project root OR from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from noGarbageIn import (
    PRED_CHRS,
    load_chromosome,
    encode_batch,
    reverse_complement,
)
from model import CNN

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512          # safe for GTX 1650 4 GB; lower to 256 if OOM
CKPT_DIR   = "models/checkpoints"
PRED_DIR   = "predictions"
TF_LIST    = ["CTCF", "REST", "EP300"]


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def _forward_pass(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    """
    Run inference on pre-encoded array X of shape (N, 200, C).
    CNN.forward() does the (N,200,C) → permute → (N,C,200) internally.
    Returns sigmoid probabilities, shape (N, 3), float32 numpy.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            # X is already (N, 200, C) — CNN.forward permutes it
            xb = torch.tensor(X[start : start + BATCH_SIZE],
                               dtype=torch.float32).to(DEVICE)
            out = torch.sigmoid(model(xb))   # (B, 3)
            preds.append(out.cpu().numpy())
    return np.vstack(preds)                  # (N, 3)


def predict_with_tta(
    model: torch.nn.Module,
    seqs: list,
    atac: np.ndarray,
    meth: np.ndarray,
    pwm: np.ndarray,
    use_atac: bool,
    use_meth: bool,
    use_pwm: bool,
) -> np.ndarray:
    """
    Encode forward strand, run inference.
    Encode reverse complement, run inference.
    Return average — TTA (test-time augmentation).

    atac/meth/pwm scalars are strand-invariant so we reuse them for RC.
    """
    # ── Forward ──────────────────────────────
    X_fwd  = encode_batch(seqs, atac, meth, pwm, use_atac, use_meth, use_pwm)
    p_fwd  = _forward_pass(model, X_fwd)
    del X_fwd
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ── Reverse complement ────────────────────
    rc_seqs = [reverse_complement(s) for s in seqs]
    X_rc    = encode_batch(rc_seqs, atac, meth, pwm, use_atac, use_meth, use_pwm)
    p_rc    = _forward_pass(model, X_rc)
    del X_rc
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return (p_fwd + p_rc) / 2.0             # (N, 3)


def _score_summary(preds: np.ndarray, c: int) -> str:
    lines = [f"chr{c} prediction summary ({len(preds):,} bins)\n" + "─" * 50]
    for i, tf in enumerate(TF_LIST):
        p = preds[:, i]
        n_bound = (p >= 0.5).sum()
        lines.append(
            f"  {tf:6s}  min={p.min():.4f}  max={p.max():.4f}  "
            f"mean={p.mean():.4f}  median={np.median(p):.4f}  "
            f"≥0.5: {n_bound:,} ({100*n_bound/len(p):.2f}%)"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    os.makedirs(PRED_DIR, exist_ok=True)

    # ── Load config ───────────────────────────
    cfg_path = os.path.join(CKPT_DIR, "best_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"Config not found at {cfg_path}. Run train.py first."
        )
    with open(cfg_path) as f:
        cfg = json.load(f)

    use_atac = cfg["use_atac"]
    use_meth = cfg["use_meth"]
    use_pwm  = cfg["use_pwm"]
    in_ch    = cfg["in_ch"]
    name     = cfg["name"]

    print(f"\n{'='*55}")
    print(f"  TF Binding Prediction — predict.py")
    print(f"{'='*55}")
    print(f"  Device      : {DEVICE}")
    print(f"  Best config : {name}  (in_ch={in_ch})")
    print(f"  use_atac={use_atac}  use_meth={use_meth}  use_pwm={use_pwm}")
    print(f"  PRED_CHRS   : {PRED_CHRS}")
    print(f"  BATCH_SIZE  : {BATCH_SIZE}")

    # ── Sanity-check in_ch matches what encode_batch will produce ─
    expected_ch = 4 + int(use_atac) + int(use_meth) + (3 if use_pwm else 0)
    if in_ch != expected_ch:
        raise ValueError(
            f"Config in_ch={in_ch} but feature flags imply {expected_ch} channels. "
            f"Check best_config.json."
        )

    # ── Load model ────────────────────────────
    ckpt_path = os.path.join(CKPT_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. Run train.py first."
        )

    # use_attn=True matches train.py — must match what was saved
    model = CNN(in_ch=in_ch, use_attn=True).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"\n  Loaded checkpoint: {ckpt_path}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    # ── Predict each chromosome ───────────────
    t_total = time.time()

    for c in PRED_CHRS:
        print(f"\n{'─'*55}")
        print(f"  chr{c} — loading...")
        t0 = time.time()

        # load_chromosome returns (seqs, atac, meth, pwm, y, df)
        # For PRED_CHRS, y is None (unknown TSV has no labels)
        seqs, atac, meth, pwm, y, df = load_chromosome(c, augment=False)

        print(f"  chr{c} — {len(seqs):,} bins  "
              f"(loaded in {time.time()-t0:.1f}s)")

        # Guard: check that the number of rows in df matches seqs
        if len(seqs) != len(df):
            raise ValueError(
                f"chr{c}: {len(seqs)} FASTA sequences but {len(df)} TSV rows. "
                "Data may be misaligned."
            )

        print(f"  chr{c} — predicting (TTA fwd+RC)...")
        t1 = time.time()

        preds = predict_with_tta(
            model, seqs, atac, meth, pwm,
            use_atac, use_meth, use_pwm,
        )
        # preds: (N, 3) float32, values in [0, 1]

        print(f"  chr{c} — done in {time.time()-t1:.1f}s")

        # ── Write scores into df ──────────────
        for i, tf in enumerate(TF_LIST):
            df[tf] = preds[:, i].astype(np.float32)

        # ── Save TSV.GZ ───────────────────────
        out_path = os.path.join(PRED_DIR, f"chr{c}_predictions.tsv.gz")
        df.to_csv(out_path, sep="\t", index=False, compression="gzip")
        print(f"  Saved → {out_path}")

        # ── Print score summary ───────────────
        summary = _score_summary(preds, c)
        print(summary)

        # ── Save summary txt ──────────────────
        txt_path = os.path.join(PRED_DIR, f"chr{c}_summary.txt")
        with open(txt_path, "w") as f:
            f.write(summary + "\n")

    print(f"\n{'='*55}")
    print(f"  All chromosomes done in {time.time()-t_total:.1f}s")
    print(f"  Output directory: {os.path.abspath(PRED_DIR)}/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
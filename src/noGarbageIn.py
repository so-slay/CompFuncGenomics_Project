"""
noGarbageIn.py

Handles all data loading and encoding for the TF binding prediction pipeline.
Reads per-chromosome FASTA, TSV, methylation and PWM precomputed arrays and
encodes them into (N, 200, channels) float32 tensors ready for the CNN.
Supports four input configs (DNA / +ATAC / +METH / +PWM) for model selection.
Reverse complement augmentation is applied at load time for training data only.

Input:  data/raw/FASTAs/chr{c}_200bp_bins.fa
        data/raw/tsv/chr{c}_200bp_bins[_unknown].tsv
        data/processed/chr{c}_methylation.npy
        data/processed/chr{c}_pwm.npy
Output: X array (N, 200, channels), y array (N, 3), df with bin coords

Usage: imported by train.py and predict.py
"""

import os
import numpy as np
import pandas as pd

# ---------------- SPLIT CONSTANTS ----------------
TRAIN_CHRS = [1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]
VAL_CHRS   = [18, 19, 20]
TEST_CHRS  = [21, 22]
PRED_CHRS  = [3, 10, 17]

TF_LIST    = ["CTCF", "REST", "EP300"]

# ---------------- PATHS ----------------
FASTA_DIR = "data/raw/FASTAs"
TSV_DIR   = "data/raw/tsv"
PROC_DIR  = "data/processed"

# ---------------- FASTA ----------------
def read_fasta(path):
    seqs, seq = [], ""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq:
                    seqs.append(seq)
                    seq = ""
            else:
                seq += line
        if seq:
            seqs.append(seq)
    return seqs

# ---------------- REVERSE COMPLEMENT ----------------
rc_map = str.maketrans("ACGT", "TGCA")

def reverse_complement(seq):
    return seq.translate(rc_map)[::-1]

# ---------------- ENCODING ----------------
BASE_MAP = {
    "A": [1,0,0,0], "C": [0,1,0,0],
    "G": [0,0,1,0], "T": [0,0,0,1]
}

def encode_batch(seqs, atac=None, meth=None, pwm=None,
                 use_atac=False, use_meth=False, use_pwm=False):
    """
    Encodes sequences and optional features into (N, 200, channels) float32.

    Channel layout:
        0-3 : one-hot DNA                        always
        4   : ATAC scalar broadcast over 200bp   if use_atac
        5   : methylation scalar                 if use_meth
        6-8 : PWM scores (CTCF, REST, EP300)     if use_pwm
    """
    N        = len(seqs)
    n_ch     = 4
    if use_atac: n_ch += 1
    if use_meth: n_ch += 1
    if use_pwm:  n_ch += 3

    X = np.zeros((N, 200, n_ch), dtype=np.float32)

    for i, seq in enumerate(seqs):
        for j, base in enumerate(seq[:200]):
            X[i, j, :4] = BASE_MAP.get(base.upper(), [0,0,0,0])

        idx = 4
        if use_atac:
            X[i, :, idx] = atac[i]
            idx += 1
        if use_meth:
            X[i, :, idx] = meth[i]
            idx += 1
        if use_pwm:
            X[i, :, idx]     = pwm[i, 0]   # CTCF
            X[i, :, idx + 1] = pwm[i, 1]   # REST
            X[i, :, idx + 2] = pwm[i, 2]   # EP300 (zeros)

    return X

# ---------------- LOAD ONE CHROMOSOME ----------------
def load_chromosome(c, test=False, augment=False):
    """
    Loads all data for chromosome c.

    Args:
        c       : chromosome number
        test    : if True, loads unknown TSV (no TF labels)
        augment : if True, appends reverse complement sequences
                  (use for training only, not val/test/predict)

    Returns:
        seqs  : list of str
        atac  : (N,) float32
        meth  : (N,) float32 or None
        pwm   : (N, 3) float32 or None
        y     : (N, 3) float32 or None (None if test=True)
        df    : DataFrame with chr, start, end columns
    """
    # --- TSV ---
    if test:
        tsv_path = os.path.join(TSV_DIR, f"chr{c}_200bp_bins_unknown.tsv")
        if not os.path.exists(tsv_path):
            tsv_path = os.path.join(TSV_DIR, f"chr{c}_200bp_bins.tsv")
    else:
        tsv_path = os.path.join(TSV_DIR, f"chr{c}_200bp_bins.tsv")

    df   = pd.read_csv(tsv_path, sep="\t")
    atac = df["ATAC"].map({"B": 1.0, "U": 0.0}).values.astype(np.float32)

    y = None
    if not test:
        y = df[TF_LIST].apply(lambda col: col.map({"B": 1.0, "U": 0.0})).astype(np.float32).values

    # --- FASTA ---
    fasta_path = os.path.join(FASTA_DIR, f"chr{c}_200bp_bins.fa")
    seqs       = read_fasta(fasta_path)

    # --- METHYLATION ---
    meth_path = os.path.join(PROC_DIR, f"chr{c}_methylation.npy")
    meth      = np.load(meth_path).astype(np.float32) if os.path.exists(meth_path) else None

    # --- PWM ---
    pwm_path = os.path.join(PROC_DIR, f"chr{c}_pwm.npy")
    pwm      = np.load(pwm_path).astype(np.float32) if os.path.exists(pwm_path) else None

    # --- AUGMENTATION (training only) ---
    if augment:
        rc_seqs = [reverse_complement(s) for s in seqs]
        seqs    = seqs + rc_seqs
        atac    = np.concatenate([atac, atac])
        if y    is not None: y    = np.concatenate([y,    y])
        if meth is not None: meth = np.concatenate([meth, meth])
        if pwm  is not None: pwm  = np.concatenate([pwm,  pwm])
        df      = pd.concat([df, df], ignore_index=True)

    return seqs, atac, meth, pwm, y, df

# ---------------- LOAD MULTIPLE CHROMOSOMES ----------------
def load_split(chrs, test=False, augment=False):
    """
    Loads and concatenates data for a list of chromosomes.
    Used to build train/val/test splits.

    Returns same structure as load_chromosome but pooled.
    """
    all_seqs, all_atac, all_meth, all_pwm, all_y, all_df = [], [], [], [], [], []

    for c in chrs:
        seqs, atac, meth, pwm, y, df = load_chromosome(c, test=test, augment=augment)

        all_seqs  += seqs
        all_atac.append(atac)
        if meth is not None: all_meth.append(meth)
        if pwm  is not None: all_pwm.append(pwm)
        if y    is not None: all_y.append(y)
        all_df.append(df)

        print(f"  Loaded chr{c}: {len(seqs)} bins")

    all_atac = np.concatenate(all_atac)
    all_meth = np.concatenate(all_meth) if all_meth else None
    all_pwm  = np.concatenate(all_pwm)  if all_pwm  else None
    all_y    = np.concatenate(all_y)    if all_y    else None
    all_df   = pd.concat(all_df, ignore_index=True)

    return all_seqs, all_atac, all_meth, all_pwm, all_y, all_df

def load_split_by_chr(chrs, augment=False):
    """
    Loads chromosomes separately — returns lists, one entry per chromosome.
    Prevents memory explosion from concatenating all chromosomes at once.
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


# ---------------- SANITY CHECK ----------------
if __name__ == "__main__":
    print("Loading chr1 as sanity check...")
    seqs, atac, meth, pwm, y, df = load_chromosome(1, augment=True)

    print(f"  Sequences : {len(seqs)}")
    print(f"  ATAC      : {atac.shape}, mean={atac.mean():.3f}")
    print(f"  Meth      : {meth.shape if meth is not None else 'None'}")
    print(f"  PWM       : {pwm.shape  if pwm  is not None else 'None'}")
    print(f"  Labels    : {y.shape    if y    is not None else 'None'}")

    X = encode_batch(seqs, atac, meth, pwm,
                     use_atac=True, use_meth=True, use_pwm=True)
    print(f"  Encoded X : {X.shape}")   # expect (2*num_bins, 200, 9)
    print("Sanity check passed.")
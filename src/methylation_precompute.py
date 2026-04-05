"""
methylation_precompute.py

Precomputes per-bin CpG methylation features from ENCODE WGBS data (ENCFF660IHA, gemBS bed9+ format).
Filters CpG sites by minimum coverage (default=5), then for each 200bp genomic bin computes
the mean methylation fraction across all CpG positions within that bin using vectorized searchsorted.
Outputs one float32 numpy array of shape (num_bins,) per chromosome to data/processed/,
named chr{c}_methylation.npy. Already-processed chromosomes are skipped automatically.

Input:  data/raw/methylation/ENCFF660IHA.bed.gz
        data/raw/tsv/chr{c}_200bp_bins[_unknown].tsv
Output: data/processed/chr{c}_methylation.npy

Usage: python src/methylation_precompute.py
"""


import os
import numpy as np
import pandas as pd

RAW_METH = "data/raw/methylation/ENCFF660IHA.bed.gz"
TSV_DIR  = "data/raw/tsv"
OUT_DIR  = "data/processed"
MIN_COVERAGE = 5

TRAIN_CHRS = [c for c in range(1, 23) if c not in [3, 10, 17]]
TEST_CHRS  = [3, 10, 17]
ALL_CHRS   = TRAIN_CHRS + TEST_CHRS

def load_methylation_file(path, min_cov=5):
    """
    Load entire CpG bed file once into memory.
    Returns a dict: { "chr1": pd.Series(meth_frac, index=position), ... }
    """
    print("Loading methylation file (this takes a minute)...")

    meth = pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[0, 1, 9, 10],
        names=["chrom", "pos", "coverage", "meth_pct"],
        dtype={"chrom": str, "pos": int, "coverage": int, "meth_pct": float}
    )

    meth = meth[meth["coverage"] >= min_cov].copy()
    meth["meth_frac"] = meth["meth_pct"] / 100.0

    print(f"  Retained {len(meth):,} CpG sites (coverage >= {min_cov})")

    # Split by chromosome into dict of Series for fast lookup
    meth_by_chr = {}
    for chrom, group in meth.groupby("chrom"):
        meth_by_chr[chrom] = group.set_index("pos")["meth_frac"]

    return meth_by_chr


def get_tsv_path(c):
    for name in [f"chr{c}_200bp_bins.tsv", f"chr{c}_200bp_bins_unknown.tsv"]:
        p = os.path.join(TSV_DIR, name)
        if os.path.exists(p):
            return p
    return None


def precompute_methylation(meth_by_chr):
    """
    For each chromosome, compute mean CpG methylation per 200bp bin.
    Output: (num_bins,) float32 array, saved as chr{c}_methylation.npy
    """
    for c in ALL_CHRS:
        out_path = os.path.join(OUT_DIR, f"chr{c}_methylation.npy")

        if os.path.exists(out_path):
            print(f"chr{c} methylation already exists, skipping")
            continue

        tsv = get_tsv_path(c)
        if tsv is None:
            print(f"No TSV for chr{c}, skipping")
            continue

        bins = pd.read_csv(tsv, sep="\t", usecols=["chr", "start", "end"])
        chrom_key = f"chr{c}"
        chr_meth  = meth_by_chr.get(chrom_key, pd.Series(dtype=float))

        arr = np.zeros(len(bins), dtype=np.float32)

        if len(chr_meth) > 0:
            # vectorized: for each bin find all CpG positions within [start, end)
            cpg_pos    = chr_meth.index.values          # sorted positions
            cpg_values = chr_meth.values

            starts = bins["start"].values
            ends   = bins["end"].values

            # searchsorted gives us the slice of CpGs inside each bin instantly
            left  = np.searchsorted(cpg_pos, starts, side="left")
            right = np.searchsorted(cpg_pos, ends,   side="left")

            for i in range(len(bins)):
                if right[i] > left[i]:
                    arr[i] = cpg_values[left[i]:right[i]].mean()
                # else stays 0.0 — no CpG in bin

        np.save(out_path, arr)
        nonzero = (arr > 0).sum()
        print(f"  chr{c}: {nonzero}/{len(arr)} bins have CpG data — saved to {out_path}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    meth_by_chr = load_methylation_file(RAW_METH, MIN_COVERAGE)
    precompute_methylation(meth_by_chr)

    print("Done.")
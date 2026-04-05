"""
pwm_precompute.py

Computes per-bin PWM match scores for CTCF and REST using JASPAR motifs
(MA0139.1 and MA0138.2 respectively). For each 200bp bin, scans both strands
of the DNA sequence and records the maximum log-odds score across all positions.
EP300 has no DNA-binding domain so its channel is set to zeros throughout.
Scores are computed against a uniform background (0.25 per base).
Outputs one float32 array of shape (num_bins, 3) per chromosome where
columns are [CTCF_score, REST_score, EP300_zeros].

Input:  data/raw/FASTAs/chr{c}_200bp_bins.fa
Output: data/processed/chr{c}_pwm.npy

Usage: python src/pwm_precompute.py
"""



import os
import numpy as np
import urllib.request
from Bio import motifs

# ---------------- CONFIG ----------------
FASTA_DIR = "data/raw/FASTAs"
OUT_DIR   = "data/processed"

TRAIN_CHRS = [c for c in range(1, 23) if c not in [3, 10, 17]]
TEST_CHRS  = [3, 10, 17]
ALL_CHRS   = TRAIN_CHRS + TEST_CHRS

JASPAR_IDS = {
    "CTCF": "MA0139.1",
    "REST": "MA0138.2",
}
BACKGROUND = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
PSEUDOCOUNT = 0.1

# ---------------- MOTIF LOADING ----------------
def fetch_pwm(tf_name, jaspar_id):
    """
    Downloads motif from JASPAR API and returns a log-odds PWM as a
    dict of numpy arrays keyed by nucleotide {A, C, G, T},
    each of shape (motif_length,).
    """
    url  = f"https://jaspar.elixir.no/api/v1/matrix/{jaspar_id}/?format=jaspar"
    path = f"/tmp/{jaspar_id}.jaspar"

    if not os.path.exists(path):
        print(f"  Downloading {tf_name} ({jaspar_id})...")
        urllib.request.urlretrieve(url, path)

    with open(path) as f:
        motif = motifs.read(f, "jaspar")

    pwm = motif.counts.normalize(pseudocounts=PSEUDOCOUNT)
    pssm = pwm.log_odds(background=BACKGROUND)

    # Convert to plain numpy dict for fast scanning
    return {
        base: np.array(pssm[base], dtype=np.float32)
        for base in "ACGT"
    }, len(motif)


# ---------------- SCANNING (VECTORIZED) ----------------
BASE_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}

rc_map = str.maketrans("ACGT", "TGCA")

def build_pssm_matrix(pssm):
    """Convert pssm dict to (4, motif_len) numpy matrix for vectorized scoring."""
    return np.array([pssm["A"], pssm["C"], pssm["G"], pssm["T"]], dtype=np.float32)

def reverse_complement(seq):
    return seq.translate(rc_map)[::-1] 
  

def encode_sequence_fast(seq):
    """Faster encoding using lookup table."""
    lookup = np.full(128, -1, dtype=np.int8)
    for base, idx in BASE_IDX.items():
        lookup[ord(base)] = idx
    arr = np.frombuffer(seq.upper().encode(), dtype=np.uint8)
    return lookup[arr]


def score_sequence_vectorized(seq, pssm_matrix, motif_len):
    """
    Scans both strands with numpy stride tricks.
    All windows scored simultaneously — no Python loop over positions.
    Returns max log-odds score across all positions and both strands.
    """
    best = -np.inf

    seq      = seq.upper()
    strands  = [seq, reverse_complement(seq)]

    for s in strands:
        enc = encode_sequence_fast(s)
        n   = len(enc) - motif_len + 1

        if n <= 0:
            continue

        # Build (n_windows, motif_len) view with stride tricks — no copy
        shape   = (n, motif_len)
        strides = (enc.strides[0], enc.strides[0])
        windows = np.lib.stride_tricks.as_strided(enc, shape=shape, strides=strides)

        # Mask windows containing N (-1)
        valid_mask = (windows >= 0).all(axis=1)   # (n_windows,)

        if not valid_mask.any():
            continue

        valid_windows = windows[valid_mask]        # (n_valid, motif_len)

        # Score: for each position j, look up pssm_matrix[base, j]
        # positions index: (n_valid, motif_len)
        pos_idx      = np.arange(motif_len)
        scores       = pssm_matrix[valid_windows, pos_idx].sum(axis=1)  # (n_valid,)

        best = max(best, scores.max())

    return float(best) if best != -np.inf else 0.0


# ---------------- FASTA READING ----------------
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


# ---------------- MAIN ----------------
def precompute_pwm(pssms, motif_lens):

    pssm_matrices = {
        tf: build_pssm_matrix(pssm)
        for tf, pssm in pssms.items()
    }

    for c in ALL_CHRS:
        out_path = os.path.join(OUT_DIR, f"chr{c}_pwm.npy")

        if os.path.exists(out_path):
            print(f"chr{c} PWM already exists, skipping")
            continue

        fasta_path = os.path.join(FASTA_DIR, f"chr{c}_200bp_bins.fa")
        if not os.path.exists(fasta_path):
            for suffix in ["_unkown.fa", "_unknown.fa"]:
                alt = os.path.join(FASTA_DIR, f"chr{c}_200bp_bins{suffix}")
                if os.path.exists(alt):
                    fasta_path = alt
                    print(f"  chr{c}: using alternate FASTA name: {os.path.basename(fasta_path)}")
                    break

        if not os.path.exists(fasta_path):
            print(f"No FASTA for chr{c}, skipping")
            continue

        seqs     = read_fasta(fasta_path)
        num_bins = len(seqs)

        # shape: (num_bins, 3) — CTCF, REST, EP300
        arr = np.zeros((num_bins, 3), dtype=np.float32)

        for i, seq in enumerate(seqs):
            arr[i, 0] = score_sequence_vectorized(seq, pssm_matrices["CTCF"], motif_lens["CTCF"])
            arr[i, 1] = score_sequence_vectorized(seq, pssm_matrices["REST"],  motif_lens["REST"])
            # arr[i, 2] stays 0.0 — EP300

            if i % 10000 == 0:
                print(f"  chr{c}: {i}/{num_bins} bins scored")

        np.save(out_path, arr)
        print(f"  chr{c}: saved {out_path}, shape {arr.shape}")
        print(f"    CTCF score range: {arr[:,0].min():.2f} to {arr[:,0].max():.2f}")
        print(f"    REST score range: {arr[:,1].min():.2f} to {arr[:,1].max():.2f}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Fetching motifs from JASPAR...")
    pssms, motif_lens = {}, {}
    for tf, jid in JASPAR_IDS.items():
        pssms[tf], motif_lens[tf] = fetch_pwm(tf, jid)
        print(f"  {tf}: motif length {motif_lens[tf]}")

    print("\nScanning sequences...")
    precompute_pwm(pssms, motif_lens)

    print("\nDone.")
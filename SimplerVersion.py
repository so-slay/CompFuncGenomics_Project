import sys
import math
from pathlib import Path
import pandas as pd
import numpy as np

"""
Usage:
    python simplerVersion.py <FASTA file> <Markov order>

Output:
    Prints one log-likelihood score per sequence to stdout.
"""

# Note importing the markov module causes issues with sys
# Copied the same function here...
def markov_k(seq_list, k):
    nucleotides = ["A", "T", "G", "C"]
    counts = {}

    # Consider a sequence of Length = L
    # for a sliding window, we can go from i = 1 to i = L-k (1 based)
    # in 0-based indexing this is i = 0  to L-1-k
    for seq in seq_list:
        seq = seq.upper()  # Ensure uppercase letters
        for i in range(len(seq) - k):
            k_prefix = seq[i:i+k]
            k_letter = seq[i+k]

            # Build nested dictionary of outer 'k_prefix'es
            if k_prefix not in counts:
                counts[k_prefix] = {}

            # Then count each occurrence of the 'next'==kth letter
            if k_letter not in counts[k_prefix]:
                counts[k_prefix][k_letter] = 0

            counts[k_prefix][k_letter] += 1


    markov_model = {}
    # Convert raw counts into probabilities
    # pseudocounts (For postmidsem- Bayesian stuff)
    pseudoc = 1
    for k_prefix, vals in counts.items():
        total = sum(vals.get(nt, 0) + pseudoc for nt in nucleotides)  # Ensure all nucleotides included

        markov_model[k_prefix] = {}
        for nt in nucleotides:  # Refer line 70 for nucleotides
            count = vals.get(nt, 0) + pseudoc
            prob = count / total
            markov_model[k_prefix][nt] = math.log(prob)

    return markov_model

def loglikelihod_all_fast(seq, markov_model, order_k):
    """
    Faster log-likelihood calculation using numpy indexing
    """
    seq = seq.upper()
    k = order_k
    default_log_p = math.log(1/4)

    # Build arrays of k-prefixes and next letters
    prefixes = [seq[i:i+k] for i in range(len(seq) - k)]
    next_letters = [seq[i+k] for i in range(len(seq) - k)]

    # Vectorized scoring
    scores = np.array([
        markov_model.get(pref, {}).get(nt, default_log_p)
        for pref, nt in zip(prefixes, next_letters)
    ], dtype=float)

    return np.sum(scores)

def main(fasta, order):
    order = int(order)
    # Read FASTA
    sequences = []
    with open(fasta, "r") as f:
        seq = ""
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq:
                    sequences.append(seq)
                    seq = ""
                continue
            seq += line
        if seq:
            sequences.append(seq)

    df = pd.DataFrame(sequences, columns=["seq"])

    # Learn markov model
    model = markov_k(df["seq"].tolist(), order)

    # Score each sequence using the fast function
    df["llr"] = df["seq"].apply(lambda seq: loglikelihod_all_fast(seq, model, order))

    # Print scores
    for _, row in df.iterrows():
        print(f"Log-likelihood score for sequence {row['seq']}: {row['llr']}")

if __name__=="__main__":
    if len(sys.argv) != 3:
        print("Usage: python simplerVersion.py <FASTA file> <Markov order>; no quotes needed")
        sys.exit(1)
    fasta_file = Path(sys.argv[1])  # Convert to Path object
    markov_order = int(sys.argv[2])
    main(fasta_file, markov_order)

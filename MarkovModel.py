import pandas as pd
import numpy as np
import math
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def read_fasta(fasta_file):
    """
    Returns:
        dict : bin_id -> sequence
    """
    sequences = {}
    current_id = None

    with open(fasta_file) as f:
        for line in f:
            line = line.strip()

            if line.startswith(">"):
                current_id = line[1:]
                sequences[current_id] = ""
            else:
                sequences[current_id] += line.upper()

    return sequences


def load_data(fasta_file, tsv_file, tf_name):

    
    fasta_sequences = read_fasta(fasta_file)

    
    sequences = list(fasta_sequences.values())
    
    
    df = pd.read_csv(tsv_file, sep="\t")

    labels = df[tf_name].map({"B": 1, "U": 0}).tolist()


    
    if len(sequences) != len(labels):
        raise ValueError(
            f"Mismatch: {len(sequences)} sequences but {len(labels)} labels"
        )

    return sequences, labels




def k_fold_split(seqs, labels, k):

    idx = np.arange(len(seqs))
    np.random.shuffle(idx)

    folds = np.array_split(idx, k)
    splits = []

    for i in range(k):

        test_idx = folds[i]
        train_idx = np.hstack(folds[:i] + folds[i+1:])

        splits.append((
            [seqs[j] for j in train_idx],
            [labels[j] for j in train_idx],
            [seqs[j] for j in test_idx],
            [labels[j] for j in test_idx]
        ))

    return splits


def train_markov(seqs, m):

    bases = ["A", "C", "G", "T"]

    # counts[context][base]
    counts = defaultdict(lambda: defaultdict(int))

    for seq in seqs:
        seq = seq.upper()

        # context length = m (CORRECT)
        for i in range(m, len(seq)):
            context = seq[i-m:i]
            base = seq[i]

            if base in bases:
                counts[context][base] += 1

    # convert to log probabilities
    model = {}

    for context in counts:

        total = sum(counts[context][b] + 1 for b in bases)

        model[context] = {
            b: math.log((counts[context][b] + 1) / total)
            for b in bases
        }

    return model



def score_sequence(seq, model_pos, model_neg, m):

    score = 0.0
    seq = seq.upper()

    for i in range(m, len(seq)):

        context = seq[i-m:i]
        base = seq[i]

        lp = model_pos.get(context, {}).get(base, math.log(1e-6))
        ln = model_neg.get(context, {}).get(base, math.log(1e-6))

        score += (lp - ln)

    return score


def run_one_m(seqs, labels, m, k=5):

    start_time = time.time()

    folds = k_fold_split(seqs, labels, k)

    roc_aucs = []
    pr_aucs = []

    plt.figure(figsize=(12, 5))

    
    plt.subplot(1, 2, 1)

    for i, (tr_s, tr_l, te_s, te_l) in enumerate(folds):

        pos = [s for s, y in zip(tr_s, tr_l) if y == 1]
        neg = [s for s, y in zip(tr_s, tr_l) if y == 0]

        model_pos = train_markov(pos, m)
        model_neg = train_markov(neg, m)

        scores = [score_sequence(s, model_pos, model_neg, m)
                  for s in te_s]

        fpr, tpr, _ = roc_curve(te_l, scores)
        roc_auc = auc(fpr, tpr)

        roc_aucs.append(roc_auc)

        plt.plot(fpr, tpr, label=f"Fold {i+1}")

    plt.plot([0, 1], [0, 1], "k--")
    plt.title(f"ROC (m={m})")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()

   
    plt.subplot(1, 2, 2)

    for i, (tr_s, tr_l, te_s, te_l) in enumerate(folds):

        pos = [s for s, y in zip(tr_s, tr_l) if y == 1]
        neg = [s for s, y in zip(tr_s, tr_l) if y == 0]

        model_pos = train_markov(pos, m)
        model_neg = train_markov(neg, m)

        scores = [score_sequence(s, model_pos, model_neg, m)
                  for s in te_s]

        precision, recall, _ = precision_recall_curve(te_l, scores)
        pr_auc = auc(recall, precision)

        pr_aucs.append(pr_auc)

        plt.plot(recall, precision, label=f"Fold {i+1}")

    plt.title(f"Precision-Recall (m={m})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"ROC_PR_m{m}.png")
    plt.close()

    elapsed = time.time() - start_time

    return np.mean(roc_aucs), np.mean(pr_aucs), elapsed



def main(fasta_file, tsv_file, tf_name):

    seqs, labels = load_data(fasta_file, tsv_file, tf_name)

    print("\nRunning m = 0 → 10\n")

    for m in range(11):

        roc_auc, pr_auc, runtime = run_one_m(
            seqs, labels, m, k=5
        )

        print(
            f"m={m} | ROC AUC={roc_auc:.3f} "
            f"| PR AUC={pr_auc:.3f} "
            f"| time={runtime:.2f}s"
        )



if __name__ == "__main__":
    import sys

    
    main(sys.argv[1], sys.argv[2], sys.argv[3])

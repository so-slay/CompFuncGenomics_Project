import os
import numpy as np
import pandas as pd

TRAIN_CHRS = [1,2,4,5,6,7,8,9,11,12,13,14,15,16]
VAL_CHRS   = [18,19,20]
TEST_CHRS  = [21,22]
PRED_CHRS  = [3,10,17]

TF_LIST = ["CTCF","REST","EP300"]

FASTA_DIR = "data/raw/FASTAs"
TSV_DIR   = "data/raw/tsv"
PROC_DIR  = "data/processed"
CACHE_DIR = "cache"

os.makedirs(CACHE_DIR, exist_ok=True)

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

def encode_batch_fast(seqs, atac, meth, pwm):

    N = len(seqs)

    seq_array = np.frombuffer("".join(seqs).encode("ascii"), dtype="S1")
    seq_array = seq_array.reshape(N, 200)

    X = np.zeros((N,200,4), dtype=np.float32)
    X[:,:,0] = (seq_array == b"A")
    X[:,:,1] = (seq_array == b"C")
    X[:,:,2] = (seq_array == b"G")
    X[:,:,3] = (seq_array == b"T")

    if atac is not None:
        X = np.concatenate([X, np.repeat(atac[:,None,None],200,axis=1)], axis=2)

    if meth is not None:
        X = np.concatenate([X, np.repeat(meth[:,None,None],200,axis=1)], axis=2)

    if pwm is not None:
        X = np.concatenate([X, np.repeat(pwm[:,None,:],200,axis=1)], axis=2)

    return X.astype(np.float32)

def load_chromosome(c, test=False, augment=False):

    cache_path = f"{CACHE_DIR}/chr{c}_{test}_{augment}.npz"

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"], data["df"]

    # TSV
    if test:
        tsv_path = os.path.join(TSV_DIR, f"chr{c}_200bp_bins_unknown.tsv")
        if not os.path.exists(tsv_path):
            tsv_path = os.path.join(TSV_DIR, f"chr{c}_200bp_bins.tsv")
    else:
        tsv_path = os.path.join(TSV_DIR, f"chr{c}_200bp_bins.tsv")

    df = pd.read_csv(tsv_path, sep="\t")

    atac = df["ATAC"].map({"B":1.0,"U":0.0}).values.astype(np.float32)

    y = None
    if not test:
        y = df[TF_LIST].apply(lambda col: col.map({"B":1.0,"U":0.0})).values.astype(np.float32)

    seqs = read_fasta(os.path.join(FASTA_DIR, f"chr{c}_200bp_bins.fa"))

    meth = None
    mp = os.path.join(PROC_DIR, f"chr{c}_methylation.npy")
    if os.path.exists(mp):
        meth = np.load(mp)

    pwm = None
    pp = os.path.join(PROC_DIR, f"chr{c}_pwm.npy")
    if os.path.exists(pp):
        pwm = np.load(pp)

    X = encode_batch_fast(seqs, atac, meth, pwm)

    np.savez(cache_path, X=X, y=y, df=df)

    return X, y, df

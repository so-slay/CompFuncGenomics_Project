# cnn_version2.py 
import os
import time
import numpy as np
import pandas as pd

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Scikit-klearn


from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score, average_precision_score

# Visualization
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------

DATA_DIR = os.getcwd()
FASTA_DIR = os.path.join(DATA_DIR, "FASTAs")
TSV_DIR = os.path.join(DATA_DIR, "projectData")

# Toggle this while developing
DEBUG_CHR = 1      # set to e.g. 1 for single chromosome
USE_ALL_CHR = False


# -------------------------
# CHROMOSOME HANDLER
# -------------------------

def get_chromosomes():
    """
    Returns list of chromosomes to use.
    """
    if USE_ALL_CHR:
        return [i for i in range(1, 23) if i not in [3, 10, 17]]
    else:
        return [DEBUG_CHR]


# -------------------------
# FILE PATHS
# -------------------------

def get_files(chr_num):
    """
    Returns file paths for a chromosome.
    """
    fasta_path = os.path.join(FASTA_DIR, f"chr{chr_num}_200bp_bins.fa")
    tsv_path = os.path.join(TSV_DIR, f"chr{chr_num}_200bp_bins.tsv")

    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"Missing FASTA: {fasta_path}")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"Missing TSV: {tsv_path}")

    return fasta_path, tsv_path


# -------------------------
# FASTA READER
# -------------------------

def read_fasta(fasta_file):
    sequences = []
    seq = ""

    with open(fasta_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq:
                    sequences.append(seq)
                    seq = ""
            else:
                seq += line.upper()

        if seq:
            sequences.append(seq)

    return sequences


# -------------------------
# LOAD DATA
# -------------------------

def load_data(chr_num):
    """
    Loads data for a chromosome.
    """
    fasta_file, tsv_file = get_files(chr_num)

    sequences = read_fasta(fasta_file)
    df = pd.read_csv(tsv_file, sep="\t")

    if len(sequences) != len(df):
        raise ValueError(
            f"chr{chr_num}: {len(sequences)} seqs vs {len(df)} rows mismatch"
        )

    return sequences, df

# -------------------------
# SETTINGS 
# -------------------------

TF_LIST = ["CTCF", "REST", "EP300"]

BATCH_SIZE = 512
EPOCHS = 2
LR = 1e-3

MAX_CV_SAMPLES = 30000
MAX_TRAIN_SAMPLES = 80000

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")

# Simple timer
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# -------------------------
# ENCODING (DNA + flexible features, including global)
# -------------------------
def encode(seq, feature_data, feature_list):
    """
    Encode a DNA sequence with optional per-base or global features.
    
    Args:
        seq (str): DNA sequence (length N)
        feature_data (dict): Dict of features keyed by name, e.g.
            {"ATAC": [1,0,1,...] or 1, "METH": [0,1,...]}
        feature_list (list of str): Which features to include, e.g.
            ["DNA"], ["DNA","ATAC"], ["DNA","ATAC","METH"]
    
    Returns:
        np.ndarray: shape (N, channels)
    
    Notes:
        - DNA: 4 one-hot channels (A,C,G,T)
        - Per-base features: array of length N
        - Global features: single value, broadcast to length N
    """
    seq = seq.upper()
    N = len(seq)
    
    # DNA one-hot encoding
    mapping = {"A":[1,0,0,0],"C":[0,1,0,0],"G":[0,0,1,0],"T":[0,0,0,1]}
    seq_onehot = np.array([mapping.get(b, [0,0,0,0]) for b in seq], dtype=np.float32)
    
    channels = []
    if "DNA" in feature_list:
        channels.append(seq_onehot)
    
    # Add other features
    for feat in feature_list:
        if feat == "DNA":
            continue
        val = feature_data[feat]
        # Broadcast single scalar to length N
        if np.isscalar(val):
            arr = np.full(N, val, dtype=np.float32)
        else:
            arr = np.array(val, dtype=np.float32)
            if len(arr) != N:
                raise ValueError(f"Feature '{feat}' length mismatch: {len(arr)} vs {N}")
        channels.append(arr.reshape(-1,1))
    
    encoded = np.concatenate(channels, axis=1)
    return encoded



# -------------------------
# CNN MODEL
# -------------------------
class CNN(nn.Module):
    def __init__(self, input_channels=5, num_outputs=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 128, kernel_size=15)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=7)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=5)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_outputs)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)  # logits

# -------------------------
# TRAIN FOR ONE EPOCH
# -------------------------
def train_epoch(model, X, y, optimizer, criterion, batch_size=512, device="cpu"):
    model.train()
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    running_loss = 0.0

    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[idx[i:i+batch_size]], dtype=torch.float32).to(device)
        yb = torch.tensor(y[idx[i:i+batch_size]], dtype=torch.float32).to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(xb)

    return running_loss / len(X)

# -------------------------
# PREDICTION
# -------------------------
def predict_model(model, X, batch_size=512, device="cpu"):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(device)
            out = torch.sigmoid(model(xb)).cpu().numpy()
            preds.append(out)
    return np.vstack(preds)

# -------------------------
# CROSS-VALIDATION
# -------------------------
def cross_validate(X, y, model_class=CNN, n_splits=5, batch_size=512, lr=1e-3, pos_weight=None, device="cpu"):
    """
    Perform k-fold cross-validation on a dataset.

    Returns:
        dict: {tf_name: {"roc": [per_fold], "pr": [per_fold]}}
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    num_outputs = y.shape[1]
    TF_LIST = [f"TF{i}" for i in range(num_outputs)]  # placeholder names
    roc_scores = {tf: [] for tf in TF_LIST}
    pr_scores = {tf: [] for tf in TF_LIST}

    for fold, (tr, te) in enumerate(kf.split(X), 1):
        print(f"[CV] Fold {fold}/{n_splits}")
        model = model_class(input_channels=X.shape[2], num_outputs=num_outputs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)
        # Train one epoch at a time (for speed; can loop multiple)
        train_epoch(model, X[tr], y[tr], optimizer, criterion, batch_size, device)
        # Predict on test fold
        preds = predict_model(model, X[te], batch_size, device)
        for i, tf in enumerate(TF_LIST):
            fpr, tpr, _ = roc_curve(y[te][:,i], preds[:,i])
            p, r, _ = precision_recall_curve(y[te][:,i], preds[:,i])
            roc_scores[tf].append(auc(fpr, tpr))
            pr_scores[tf].append(auc(r, p))
        # Plot first fold ROC
        if fold == 1:
            plt.figure(figsize=(8,4))
            for i, tf in enumerate(TF_LIST):
                fpr, tpr, _ = roc_curve(y[te][:,i], preds[:,i])
                plt.plot(fpr, tpr, label=tf)
            plt.title("CV Fold 1 ROC")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.legend()
            plt.savefig("cv_fold1_ROC.png")
            plt.close()
    return {"roc": roc_scores, "pr": pr_scores}

# -------------------------
# SMALL IN-CHROMOSOME CROSS-VALIDATION
# -------------------------
def cross_validate_chr(X, y, folds=5):
    """
    Perform K-fold cross-validation on a single chromosome dataset.
    
    Args:
        X (np.ndarray): input features, shape (N, L, C)
        y (np.ndarray): labels, shape (N, n_tf)
        folds (int): number of CV folds
    
    Returns:
        dict: average ROC and PR per TF
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import roc_curve, auc, precision_recall_curve

    roc_scores = {tf: [] for tf in TF_LIST}
    pr_scores = {tf: [] for tf in TF_LIST}

    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X), 1):
        log(f"CV Fold {fold}/{folds}")
        model = CNN().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        pos_weight = torch.tensor([5.0]*len(TF_LIST)).to(DEVICE)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_epoch(model, X[tr_idx], y[tr_idx], opt, crit)
        preds = predict(model, X[te_idx])

        for i, tf in enumerate(TF_LIST):
            fpr, tpr, _ = roc_curve(y[te_idx][:,i], preds[:,i])
            p, r, _ = precision_recall_curve(y[te_idx][:,i], preds[:,i])
            roc_scores[tf].append(auc(fpr, tpr))
            pr_scores[tf].append(auc(r, p))

    # Average results
    avg_results = {tf: {"ROC": np.mean(roc_scores[tf]), "PR": np.mean(pr_scores[tf])} for tf in TF_LIST}
    return avg_results
# -------------------------
# CV ACROSS CHROMOSOMES
# -------------------------

def cross_validate_all_chromosomes(model_class=CNN, n_splits=5, batch_size=512, lr=1e-3, pos_weight=None, device="cpu"):
    all_X, all_y = [], []
    for c in get_chromosomes():
        seqs, df = load_data(c)
        X_chr = np.array([encode(seqs[i], {"ATAC": df["ATAC"].map({'B':1,'U':0}).values[i]}, ["DNA","ATAC"]) 
                          for i in range(len(seqs))])
        y_chr = df[TF_LIST].replace({'B':1,'U':0}).values.astype(np.float32)
        all_X.append(X_chr)
        all_y.append(y_chr)
    X = np.vstack(all_X)
    y = np.vstack(all_y)
    return cross_validate(X, y, model_class, n_splits, batch_size, lr, pos_weight, device)

# -------------------------
# TRAIN ACROSS CHROMOSOMES
# -------------------------
def train_chromosomes(chr_list, get_data_fn, model_class=CNN, epochs=2, batch_size=512, lr=1e-3, pos_weight=None, device="cpu"):
    """
    Train a model sequentially across multiple chromosomes.
    """
    device = torch.device(device)
    input_channels = None
    num_outputs = None
    model = None
    optimizer = None
    criterion = None

    for c in chr_list:
        print(f"[TRAIN] Chromosome {c}")
        X, y = get_data_fn(c)
        if X is None or y is None or len(X) == 0:
            continue

        if input_channels is None:
            input_channels = X.shape[2]
            num_outputs = y.shape[1]
            model = model_class(input_channels=input_channels, num_outputs=num_outputs).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)

        for epoch in range(1, epochs+1):
            t0 = time.time()
            loss = train_epoch(model, X, y, optimizer, criterion, batch_size, device)
            t1 = time.time()
            print(f"  Epoch {epoch}/{epochs} | Loss: {loss:.4f} | Time: {t1-t0:.1f}s")

    return model

# -------------------------
# SAVE PREDICTIONS
# -------------------------
def save_predictions(df, preds, TF_LIST, out_file):
    """
    Save predictions to a TSV file, matching columns to TF_LIST.
    """
    for i, tf in enumerate(TF_LIST):
        df[tf] = preds[:,i]
    df.to_csv(out_file, sep="\t", index=False)
    print(f"[SAVE] Predictions saved to {out_file}")




# -------------------------
# MAIN
# -------------------------
# -------------------------
# MAIN: full workflow
# -------------------------
def main():
    import time
    start_time = time.time()

    # 1. Cross-validation on all training chromosomes or a subset
    log("Starting cross-validation...")
    cv_start = time.time()
    cross_validate_all_chromosomes()  # uses your updated CV function that handles all chromosomes
    log(f"Cross-validation done in {time.time() - cv_start:.2f} sec\n")

    # 2. Train final model on selected chromosomes (training set)
    log("Training final model on selected chromosomes...")
    train_start = time.time()
    model = train_final()
    log(f"Final training done in {time.time() - train_start:.2f} sec\n")

    # 3. Predict on unseen chromosomes (always 3,10,17)
    log("Predicting on unseen chromosomes (3,10,17)...")
    predict_start = time.time()
    for chr_num in [3,10,17]:
        predict_chr(model, chr_num)
    log(f"Prediction completed in {time.time() - predict_start:.2f} sec\n")

    log(f"Total runtime: {time.time() - start_time:.2f} sec")


# -------------------------
# DEBUG MAIN: single chromosome, quick CV
# -------------------------
def debug_main():
    """
    Quick debug workflow:
    - Loads DEBUG_CHR
    - Performs in-chromosome CV on small subset
    - Trains final model on all other chromosomes
    - Predicts on unseen (3,10,17)
    """
    import time
    start_time = time.time()

    debug_chr = DEBUG_CHR
    log(f"Loading chromosome {debug_chr} for in-chromosome CV")
    seqs, df = load_data(debug_chr)

    # Example: use DNA + ATAC
    X = np.array([encode(seqs[i], {"ATAC": df["ATAC"].map({'B':1,'U':0}).values[i]}, ["DNA","ATAC"]) for i in range(len(seqs))])
    y = df[TF_LIST].replace({'B':1,'U':0}).values.astype(np.float32)

    # Optionally subsample for faster CV
    if len(X) > MAX_CV_SAMPLES:
        idx = np.random.choice(len(X), MAX_CV_SAMPLES, replace=False)
        X, y = X[idx], y[idx]

    # 1. In-chromosome CV
    log("Running in-chromosome CV...")
    cv_start = time.time()
    cv_results = cross_validate_chr(X, y)
    for tf, metrics in cv_results.items():
        log(f"{tf}: ROC={metrics['ROC']:.3f}, PR={metrics['PR']:.3f}")
    log(f"In-chromosome CV done in {time.time() - cv_start:.2f} sec\n")

    # 2. Train final model on all other chromosomes
    log("Training final model on all chromosomes except 3,10,17...")
    train_start = time.time()
    model = train_final()
    log(f"Final training done in {time.time() - train_start:.2f} sec\n")

    # 3. Predict on unseen chromosomes
    log("Predicting on unseen chromosomes (3,10,17)...")
    predict_start = time.time()
    for c in [3,10,17]:
        predict_chr(model, c)
    log(f"Prediction completed in {time.time() - predict_start:.2f} sec\n")

    log(f"Total runtime (debug): {time.time() - start_time:.2f} sec")


# -------------------------
# ENTRY POINT
# -------------------------
if __name__ == "__main__":
    if USE_ALL_CHR:
        main()
    else:
        debug_main()
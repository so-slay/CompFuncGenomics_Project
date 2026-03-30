import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
import time

# -------------------------
# SETTINGS
# -------------------------
TF_LIST = ["CTCF", "REST", "EP300"]
BATCH_SIZE = 512
EPOCHS = 2
MAX_CV_SAMPLES = 30000
MAX_TRAIN_SAMPLES = 80000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# FASTA READER
# -------------------------
def read_fasta(file):
    seqs, seq = [], ""
    with open(file) as f:
        for line in f:
            if line.startswith(">"):
                if seq:
                    seqs.append(seq)
                    seq = ""
            else:
                seq += line.strip().upper()
        if seq:
            seqs.append(seq)
    return seqs

# -------------------------
# ENCODING (SEQ + ATAC)
# -------------------------
def encode(seq, atac):
    mapping = {'A':[1,0,0,0], 'C':[0,1,0,0], 'G':[0,0,1,0], 'T':[0,0,0,1]}
    seq_enc = [mapping.get(b,[0,0,0,0]) for b in seq]
    return np.array([s + [atac] for s in seq_enc], dtype=np.float32)

# -------------------------
# MODEL
# -------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(5, 128, kernel_size=15)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(128, 256, kernel_size=7)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(256, 256, kernel_size=5)

        self.global_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = x.permute(0,2,1)

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))

        x = self.global_pool(x).squeeze(-1)

        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)   # logits

# -------------------------
# TRAIN
# -------------------------
def train_epoch(model, X, y, opt, crit):

    model.train()
    idx = np.arange(len(X))
    np.random.shuffle(idx)

    for i in range(0, len(X), BATCH_SIZE):
        xb = torch.tensor(X[idx[i:i+BATCH_SIZE]], dtype=torch.float32).to(DEVICE)
        yb = torch.tensor(y[idx[i:i+BATCH_SIZE]], dtype=torch.float32).to(DEVICE)

        preds = model(xb)
        loss = crit(preds, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

# -------------------------
# PREDICT
# -------------------------
def predict(model, X):

    model.eval()
    preds = []

    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = torch.tensor(X[i:i+BATCH_SIZE], dtype=torch.float32).to(DEVICE)
            out = torch.sigmoid(model(xb)).cpu().numpy()
            preds.append(out)

    return np.vstack(preds)

# -------------------------
# LOAD FILE HANDLER
# -------------------------

def get_files(chr_num):
    # Define directories
    fasta_dir = "FASTAs"
    tsv_dir = "projectData"

    # Build file paths
    fasta = os.path.join(fasta_dir, f"chr{chr_num}_200bp_bins.fa")
    tsv = os.path.join(tsv_dir, f"chr{chr_num}_200bp_bins.tsv")

    # Fallback if files don't exist
    if not os.path.exists(fasta):
        fasta = os.path.join(fasta_dir, f"chr{chr_num}_200bp_bins_unknown.fa")
    if not os.path.exists(tsv):
        tsv = os.path.join(tsv_dir, f"chr{chr_num}_200bp_bins_unknown.tsv")

    return fasta, tsv


# -------------------------
# PREP DATA
# -------------------------
def load_data(fasta, tsv):

    seqs = read_fasta(fasta)
    df = pd.read_csv(tsv, sep="\t")

    atac = df["ATAC"].map({'B':1,'U':0}).values
    X = np.array([encode(seqs[i], atac[i]) for i in range(len(seqs))])

    if all(tf in df.columns for tf in TF_LIST):
        y = df[TF_LIST].replace({'B':1,'U':0}).astype(np.float32).values
    else:
        y = None

    return X, y, df

# -------------------------
# FAST CV
# -------------------------
def cross_validate():

    print("\nRunning CV on chr1...")

    fasta, tsv = get_files(1)
    X, y, _ = load_data(fasta, tsv)

    if len(X) > MAX_CV_SAMPLES:
        idx = np.random.choice(len(X), MAX_CV_SAMPLES, replace=False)
        X, y = X[idx], y[idx]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    roc_scores = {tf: [] for tf in TF_LIST}
    pr_scores = {tf: [] for tf in TF_LIST}

    for fold,(tr,te) in enumerate(kf.split(X),1):
        print(f"Fold {fold}")

        model = CNN().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        pos_weight = torch.tensor([5.0,5.0,5.0]).to(DEVICE)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_epoch(model, X[tr], y[tr], opt, crit)

        preds = predict(model, X[te])

        for i,tf in enumerate(TF_LIST):
            fpr,tpr,_ = roc_curve(y[te][:,i], preds[:,i])
            p,r,_ = precision_recall_curve(y[te][:,i], preds[:,i])

            roc_scores[tf].append(auc(fpr,tpr))
            pr_scores[tf].append(auc(r,p))

        if fold==1:
            plt.figure(figsize=(10,4))
            for i,tf in enumerate(TF_LIST):
                fpr,tpr,_ = roc_curve(y[te][:,i], preds[:,i])
                plt.plot(fpr,tpr,label=tf)
            plt.legend(); plt.title("ROC")
            plt.savefig("cnn_ROC_fold1.png")
            plt.close()

    with open("auc_results.txt","w") as f:
        for tf in TF_LIST:
            f.write(f"{tf} ROC={np.mean(roc_scores[tf]):.3f}, PR={np.mean(pr_scores[tf]):.3f}\n")

# -------------------------
# TRAIN FINAL
# -------------------------
def train_final():

    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pos_weight = torch.tensor([5.0,5.0,5.0]).to(DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for c in range(1,23):
        if c in [3,10,17]:
            continue

        fasta, tsv = get_files(c)
        if not os.path.exists(fasta):
            continue

        print(f"Training chr{c}")

        X,y,_ = load_data(fasta,tsv)
        if y is None:
            continue

        if len(X) > MAX_TRAIN_SAMPLES:
            idx = np.random.choice(len(X), MAX_TRAIN_SAMPLES, replace=False)
            X,y = X[idx], y[idx]

        train_epoch(model, X, y, opt, crit)

    return model

# -------------------------
# PREDICT FINAL
# -------------------------
def predict_chr(model, c):

    fasta, tsv = get_files(c)
    X,_,df = load_data(fasta,tsv)

    preds = predict(model, X)

    for i,tf in enumerate(TF_LIST):
        df[tf] = preds[:,i]

    out = f"chr{c}_predictions.tsv"
    df.to_csv(out, sep="\t", index=False)

    print(f"Saved {out}")

# -------------------------
# MAIN
# -------------------------
def main():

    cross_validate()

    print("\nTraining final model...")
    model = train_final()

    print("\nGenerating predictions...")
    for c in [3,10,17]:
        predict_chr(model, c)

if __name__ == "__main__":
    main()

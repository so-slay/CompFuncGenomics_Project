import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt


TF_LIST = ["CTCF","REST","EP300"]
BATCH_SIZE = 256
EPOCHS = 2
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = os.getcwd()
FASTA_DIR = os.path.join(DATA_DIR, "FASTAs")
TSV_DIR = os.path.join(DATA_DIR, "projectData")


def read_fasta(file):
    seqs, seq = [], ""
    with open(file) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq: seqs.append(seq); seq=""
            else:
                seq += line
        if seq: seqs.append(seq)
    return seqs


mapping = {
    "A":[1,0,0,0],
    "C":[0,1,0,0],
    "G":[0,0,1,0],
    "T":[0,0,0,1]
}

def encode_batch(seqs, atac):
    N = len(seqs)
    X = np.zeros((N,200,5),dtype=np.float32)

    for i,seq in enumerate(seqs):
        for j,b in enumerate(seq):
            X[i,j,:4] = mapping.get(b,[0,0,0,0])
        X[i,:,4] = atac[i]
    return X


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(5,64,15)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64,128,7)
        self.conv3 = nn.Conv1d(128,128,5)
        self.gap = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64,3)

    def forward(self,x):
        x = x.permute(0,2,1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train(model,X,y):
    opt = torch.optim.Adam(model.parameters(),lr=LR)
    crit = nn.BCEWithLogitsLoss()

    model.train()
    idx = np.arange(len(X))
    np.random.shuffle(idx)

    for i in range(0,len(X),BATCH_SIZE):
        xb = torch.tensor(X[idx[i:i+BATCH_SIZE]]).to(DEVICE)
        yb = torch.tensor(y[idx[i:i+BATCH_SIZE]]).to(DEVICE)

        opt.zero_grad()
        loss = crit(model(xb),yb)
        loss.backward()
        opt.step()


def predict(model,X):
    model.eval()
    preds=[]
    with torch.no_grad():
        for i in range(0,len(X),BATCH_SIZE):
            xb = torch.tensor(X[i:i+BATCH_SIZE]).to(DEVICE)
            preds.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.vstack(preds)


def load_chr(c):
    fasta = os.path.join(FASTA_DIR,f"chr{c}_200bp_bins.fa")
    tsv = os.path.join(TSV_DIR,f"chr{c}_200bp_bins.tsv")

    seqs = read_fasta(fasta)
    df = pd.read_csv(tsv,sep="\t")

    atac = df["ATAC"].map({'B':1,'U':0}).values
    y = df[TF_LIST].replace({'B':1,'U':0}).values.astype(np.float32)

    return seqs, atac, y, df


def train_final():
    model = CNN().to(DEVICE)

    for c in range(1,23):
        if c in [3,10,17]: continue
        print(f"Training chr{c}")

        seqs, atac, y, _ = load_chr(c)

        # subsample to avoid memory blow
        idx = np.random.choice(len(seqs),min(40000,len(seqs)),replace=False)
        seqs = [seqs[i] for i in idx]
        atac = atac[idx]
        y = y[idx]

        X = encode_batch(seqs,atac)
        train(model,X,y)

        del X  # free memory

    return model


def predict_chr(model,c):
    print(f"Predicting chr{c}")
    fasta = os.path.join(FASTA_DIR,f"chr{c}_200bp_bins.fa")
    tsv = os.path.join(TSV_DIR,f"chr{c}_200bp_bins_unknown.tsv")

    seqs = read_fasta(fasta)
    df = pd.read_csv(tsv,sep="\t")

    atac = df["ATAC"].map({'B':1,'U':0}).values

    preds_all = []

    for i in range(0,len(seqs),10000):
        batch_seqs = seqs[i:i+10000]
        batch_atac = atac[i:i+10000]

        X = encode_batch(batch_seqs,batch_atac)
        preds = predict(model,X)
        preds_all.append(preds)

        del X

    preds = np.vstack(preds_all)

    for i,tf in enumerate(TF_LIST):
        df[tf] = preds[:,i]

    out = f"chr{c}_predictions.tsv"
    df.to_csv(out,sep="\t",index=False)
    print("Saved",out)


def main():
    model = train_final()

    for c in [3,10,17]:
        predict_chr(model,c)

if __name__=="__main__":
    main()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import os
import glob
from sklearn.model_selection import train_test_split

# -------------------------
# SETTINGS
# -------------------------
TF_LIST = ["CTCF", "REST", "EP300"]
BATCH_SIZE = 512
EPOCHS = 2
MAX_SAMPLES = 80000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FASTA_DIR = "FASTAs"
DATA_DIR = "projectData"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

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
# ENCODING
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
        self.conv1 = nn.Conv1d(5,128,15)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(128,256,7)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(256,256,5)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(256,128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128,3)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

# -------------------------
# TRAIN/VALIDATION
# -------------------------
def train_epoch(model, X, y, opt, crit):
    model.train()
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    for i in range(0,len(X),BATCH_SIZE):
        xb = torch.tensor(X[idx[i:i+BATCH_SIZE]],dtype=torch.float32).to(DEVICE)
        yb = torch.tensor(y[idx[i:i+BATCH_SIZE]],dtype=torch.float32).to(DEVICE)
        preds = model(xb)
        loss = crit(preds,yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

def evaluate(model,X,y):
    model.eval()
    preds=[]
    with torch.no_grad():
        for i in range(0,len(X),BATCH_SIZE):
            xb=torch.tensor(X[i:i+BATCH_SIZE],dtype=torch.float32).to(DEVICE)
            out=torch.sigmoid(model(xb)).cpu().numpy()
            preds.append(out)
    preds=np.vstack(preds)
    roc_scores, pr_scores={},{}
    for i,tf in enumerate(TF_LIST):
        fpr,tpr,_=roc_curve(y[:,i],preds[:,i])
        p,r,_=precision_recall_curve(y[:,i],preds[:,i])
        roc_scores[tf]=auc(fpr,tpr)
        pr_scores[tf]=auc(r,p)
    return roc_scores, pr_scores, preds

# -------------------------
# FILE HANDLER
# -------------------------
def get_files(chr_num):
    fasta=os.path.join(FASTA_DIR,f"chr{chr_num}_200bp_bins.fa")
    if not os.path.exists(fasta):
        fasta=os.path.join(FASTA_DIR,f"chr{chr_num}_200bp_bins_unknown.fa")
    tsv_candidates=glob.glob(os.path.join(DATA_DIR,f"chr{chr_num}*.tsv"))
    bed_candidates=glob.glob(os.path.join(DATA_DIR,f"chr{chr_num}*.bed"))
    if tsv_candidates and bed_candidates:
        print(f"Warning: Both TSV and BED files found for chr{chr_num}. Using TSV.")
        tsv=tsv_candidates[0]
    elif tsv_candidates:
        tsv=tsv_candidates[0]
    elif bed_candidates:
        tsv=bed_candidates[0]
    else:
        tsv=None
    return fasta, tsv

# -------------------------
# LOAD DATA
# -------------------------
def load_data(fasta, tsv):
    if fasta is None or tsv is None:
        return None,None,None
    seqs=read_fasta(fasta)
    df=pd.read_csv(tsv,sep="\t")
    atac=df["ATAC"].map({'B':1,'U':0}).values
    X=np.array([encode(seqs[i],atac[i]) for i in range(len(seqs))])
    if all(tf in df.columns for tf in TF_LIST):
        y=df[TF_LIST].replace({'B':1,'U':0}).astype(np.float32).values
    else:
        y=None
    return X,y,df

# -------------------------
# TRAIN FINAL WITH SPLITS
# -------------------------
def train_final(chrom_list):
    model=CNN().to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    pos_weight=torch.tensor([5.0,5.0,5.0]).to(DEVICE)
    crit=nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for c in chrom_list:
        fasta, tsv = get_files(c)
        if not fasta or not tsv:
            print(f"Skipping chr{c}, data missing")
            continue
        print(f"\nTraining on chr{c}")
        X,y,_ = load_data(fasta, tsv)
        if y is None:
            print(f"Skipping chr{c}, labels missing")
            continue
        if len(X) > MAX_SAMPLES:
            idx=np.random.choice(len(X),MAX_SAMPLES,replace=False)
            X,y=X[idx],y[idx]

        # Split train/val/test
        X_train,X_temp,y_train,y_temp=train_test_split(X,y,test_size=VAL_RATIO+TEST_RATIO,random_state=42)
        val_size=VAL_RATIO/(VAL_RATIO+TEST_RATIO)
        X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=1-val_size,random_state=42)

        # Training epochs
        for e in range(EPOCHS):
            train_epoch(model,X_train,y_train,opt,crit)

        # Evaluate
        val_roc,val_pr,_ = evaluate(model,X_val,y_val)
        test_roc,test_pr,_ = evaluate(model,X_test,y_test)
        print(f"Validation ROC: {val_roc}, PR: {val_pr}")
        print(f"Test ROC: {test_roc}, PR: {test_pr}")

    return model

# -------------------------
# PREDICT
# -------------------------
def predict_chr(model, c):
    fasta, tsv = get_files(c)
    X,_,df = load_data(fasta,tsv)
    if X is None:
        print(f"No data for chr{c}")
        return
    preds=predict(model,X)
    for i,tf in enumerate(TF_LIST):
        df[tf]=preds[:,i]
    out=f"chr{c}_predictions.tsv"
    df.to_csv(out,sep="\t",index=False)
    print(f"Saved {out}")

# -------------------------
# CLI
# -------------------------
def select_chromosomes():
    chroms=input("Enter chromosome(s) to train on (e.g., 1,4,7) or 'all': ").strip()
    if chroms.lower()=="all":
        return list(range(1,23))
    else:
        return [int(c) for c in chroms.split(",")]

# -------------------------
# MAIN
# -------------------------
def main():
    chrom_list=select_chromosomes()
    model=train_final(chrom_list)
    print("\nGenerating predictions for selected chromosomes...")
    for c in chrom_list:
        predict_chr(model,c)

if __name__=="__main__":
    main()
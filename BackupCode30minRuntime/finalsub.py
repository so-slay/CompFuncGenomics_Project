

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
TF_LIST = ["CTCF","REST","EP300"]
BATCH_SIZE = 256
EPOCHS = 3
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = os.getcwd()
FASTA_DIR = os.path.join(DATA_DIR, "FASTAs")
TSV_DIR = os.path.join(DATA_DIR, "projectData")

# ---------------- FASTA ----------------
def read_fasta(file):
    seqs, seq = [], ""
    with open(file) as f:
        for line in f:
            line=line.strip()
            if line.startswith(">"):
                if seq:
                    seqs.append(seq)
                    seq=""
            else:
                seq+=line
        if seq:
            seqs.append(seq)
    return seqs

# ---------------- REVERSE COMPLEMENT ----------------
rc_map = str.maketrans("ACGT","TGCA")
def reverse_complement(seq):
    return seq.translate(rc_map)[::-1]

# ---------------- ENCODING ----------------
mapping = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1]}

def encode_batch(seqs, atac=None, meth=None, use_atac=False, use_meth=False):
    N=len(seqs)
    channels = 4 + (1 if use_atac else 0) + (1 if use_meth else 0)

    X = np.zeros((N,200,channels),dtype=np.float32)

    for i,seq in enumerate(seqs):
        for j,b in enumerate(seq):
            X[i,j,:4] = mapping.get(b,[0,0,0,0])

        idx=4
        if use_atac:
            X[i,:,idx]=atac[i]; idx+=1
        if use_meth:
            X[i,:,idx]=meth[i]

    return X

# ---------------- FOCAL LOSS ----------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma=gamma

    def forward(self, logits, targets):
        BCE = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-BCE)
        return ((1-pt)**self.gamma * BCE).mean()

# ---------------- RESIDUAL BLOCK ----------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch,out_ch,7,padding=3)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch,out_ch,5,padding=2)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.skip = nn.Conv1d(in_ch,out_ch,1) if in_ch!=out_ch else None

    def forward(self,x):
        identity = x if self.skip is None else self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

# ---------------- MODEL ----------------
class CNN(nn.Module):
    def __init__(self,in_ch):
        super().__init__()
        self.block1 = ResBlock(in_ch,64)
        self.block2 = ResBlock(64,128)
        self.block3 = ResBlock(128,128)
        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(128,64)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64,3)

    def forward(self,x):
        x=x.permute(0,2,1)
        x=self.pool(self.block1(x))
        x=self.pool(self.block2(x))
        x=self.block3(x)
        x=self.gap(x).squeeze(-1)
        x=self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

# ---------------- TRAIN / EVAL ----------------
def train_epoch(model,X,y,opt,crit):
    model.train()
    idx=np.random.permutation(len(X))
    total=0

    for i in tqdm(range(0,len(X),BATCH_SIZE),desc="Train"):
        xb=torch.tensor(X[idx[i:i+BATCH_SIZE]]).to(DEVICE)
        yb=torch.tensor(y[idx[i:i+BATCH_SIZE]]).to(DEVICE)

        opt.zero_grad()
        loss=crit(model(xb),yb)
        loss.backward()
        opt.step()
        total+=loss.item()

    return total

def evaluate(model,X,y):
    model.eval()
    preds=[]

    with torch.no_grad():
        for i in range(0,len(X),BATCH_SIZE):
            xb=torch.tensor(X[i:i+BATCH_SIZE]).to(DEVICE)
            preds.append(torch.sigmoid(model(xb)).cpu().numpy())

    preds=np.vstack(preds)

    roc_scores=[]
    pr_scores=[]

    for i in range(3):
        fpr,tpr,_=roc_curve(y[:,i],preds[:,i])
        p,r,_=precision_recall_curve(y[:,i],preds[:,i])
        roc_scores.append(auc(fpr,tpr))
        pr_scores.append(auc(r,p))

    return np.mean(roc_scores), np.mean(pr_scores), preds

# ---------------- LOAD ----------------
def load_chr(c, test=False):
    fasta=os.path.join(FASTA_DIR,f"chr{c}_200bp_bins.fa")

    if test:
        tsv=os.path.join(TSV_DIR,f"chr{c}_200bp_bins_unknown.tsv")
        if not os.path.exists(tsv):
            tsv=os.path.join(TSV_DIR,f"chr{c}_200bp_bins.tsv")
    else:
        tsv=os.path.join(TSV_DIR,f"chr{c}_200bp_bins.tsv")

    seqs=read_fasta(fasta)
    df=pd.read_csv(tsv,sep="\t")

    atac=df["ATAC"].map({'B':1,'U':0}).values

    y=None
    if not test:
        y=df[TF_LIST].replace({'B':1,'U':0}).astype(np.float32).values

    meth_file=f"chr{c}_methylation.npy"
    meth=np.load(meth_file) if os.path.exists(meth_file) else None

    return seqs, atac, meth, y, df

# ---------------- MAIN PIPELINE ----------------
def main():

    auc_file=open("auc_results.txt","w")

    best_model=None
    best_cfg=None
    best_score=0

    configs=[("DNA",False,False),("DNA_ATAC",True,False),("DNA_ATAC_METH",True,True)]

    for name,use_atac,use_meth in configs:

        losses=[]
        val_scores=[]

        for c in range(1,23):
            if c in [3,10,17]: continue

            seqs,atac,meth,y,_=load_chr(c)

            idx=np.random.choice(len(seqs),min(30000,len(seqs)),replace=False)
            seqs=[seqs[i] for i in idx]
            atac=atac[idx]
            y=y[idx]

            if use_meth and meth is not None:
                meth=meth[idx]

            seqs+= [reverse_complement(s) for s in seqs]
            atac=np.concatenate([atac,atac])
            y=np.concatenate([y,y])
            if use_meth and meth is not None:
                meth=np.concatenate([meth,meth])

            X=encode_batch(seqs,atac,meth,use_atac,use_meth)

            Xtr,Xval,ytr,yval=train_test_split(X,y,test_size=0.2)

            model=CNN(X.shape[2]).to(DEVICE)
            opt=torch.optim.Adam(model.parameters(),lr=LR)
            crit=FocalLoss()

            for ep in range(EPOCHS):
                loss=train_epoch(model,Xtr,ytr,opt,crit)
                roc,pr,_=evaluate(model,Xval,yval)

                losses.append(loss)
                val_scores.append(roc)

                print(f"{name} chr{c} Epoch{ep+1} ROC={roc:.3f}")

                auc_file.write(f"{name} chr{c} epoch{ep+1} ROC={roc:.3f} PR={pr:.3f}\n")

            # save ROC curve
            fpr,tpr,_=roc_curve(yval[:,0],evaluate(model,Xval,yval)[2][:,0])
            plt.plot(fpr,tpr)
            plt.savefig(f"roc_chr{c}.png")
            plt.close()

            p,r,_=precision_recall_curve(yval[:,0],evaluate(model,Xval,yval)[2][:,0])
            plt.plot(r,p)
            plt.savefig(f"pr_chr{c}.png")
            plt.close()

            if roc>best_score:
                best_score=roc
                best_model=model
                best_cfg=(use_atac,use_meth)

        plt.plot(losses,label="Loss")
        plt.plot(val_scores,label="Val ROC")
        plt.legend()
        plt.savefig(f"training_curve_{name}.png")
        plt.close()

    auc_file.close()

    print("BEST CONFIG:",best_cfg)

    for c in [3,10,17]:
        seqs,atac,meth,_,df=load_chr(c,test=True)

        preds_all=[]
        for i in tqdm(range(0,len(seqs),10000)):
            X=encode_batch(seqs[i:i+10000],
                           atac[i:i+10000],
                           None if meth is None else meth[i:i+10000],
                           *best_cfg)

            xb=torch.tensor(X).to(DEVICE)
            with torch.no_grad():
                preds=torch.sigmoid(best_model(xb)).cpu().numpy()

            preds_all.append(preds)

        preds=np.vstack(preds_all)

        for i,tf in enumerate(TF_LIST):
            df[tf]=preds[:,i]

        df.to_csv(f"chr{c}_predictions.tsv",sep="\t",index=False)

if __name__=="__main__":
    main()

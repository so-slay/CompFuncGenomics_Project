# cnn_version3.py

import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = os.getcwd()
FASTA_DIR = os.path.join(DATA_DIR, "FASTAs")
TSV_DIR = os.path.join(DATA_DIR, "projectData")

TF_LIST = ["CTCF", "REST", "EP300"]
BATCH_SIZE = 512
EPOCHS = 5  # Increased slightly to show scheduler impact
LR = 1e-3
USE_RC_AUGMENTATION = True # Toggle for RC encoding

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")

# -------------------------
# VECTORIZED ENCODING
# -------------------------
def encode_vectorized(seq, feature_dict, feature_list):
    """
    Fast vectorized encoding of DNA and additional features.
    """
    # 1. DNA One-Hot
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_indices = np.array([mapping.get(b, 4) for b in seq.upper()])
    
    # Create identity matrix + 1 row for N/Unknown (all zeros)
    eye = np.eye(4, dtype=np.float32)
    dna_encoded = np.vstack([eye, np.zeros(4, dtype=np.float32)])[seq_indices]
    
    channels = []
    if "DNA" in feature_list:
        channels.append(dna_encoded)
    
    # 2. Add extra features
    for feat in feature_list:
        if feat == "DNA": continue
        val = feature_dict[feat]
        if np.isscalar(val):
            arr = np.full((len(seq), 1), val, dtype=np.float32)
        else:
            arr = np.array(val, dtype=np.float32).reshape(-1, 1)
        channels.append(arr)
        
    return np.concatenate(channels, axis=1)

def get_reverse_complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return "".join(complement.get(base, 'N') for base in reversed(seq.upper()))

# -------------------------
# CUSTOM PYTORCH DATASET
# -------------------------
class GenomicDataset(Dataset):
    def __init__(self, sequences, df, feature_list=["DNA", "ATAC"], augment_rc=False):
        self.sequences = sequences
        self.df = df
        self.feature_list = feature_list
        self.augment_rc = augment_rc
        # Map labels B/U to 1/0
        self.labels = self.df[TF_LIST].replace({'B':1, 'U':0}).values.astype(np.float32)
        # Pre-calculate ATAC or other features if they exist
        self.atac_data = self.df["ATAC"].map({'B':1, 'U':0}).values

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        features = {"ATAC": self.atac_data[idx]}
        
        # Apply RC augmentation if toggled
        if self.augment_rc and np.random.random() > 0.5:
            seq = get_reverse_complement(seq)
            
        x = encode_vectorized(seq, features, self.feature_list)
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# -------------------------
# CNN MODEL
# -------------------------
class CNN(nn.Module):
    def __init__(self, input_channels=5, num_outputs=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 128, kernel_size=15)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=7)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=5)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, num_outputs)

    def forward(self, x):
        x = x.permute(0, 2, 1) # (B, L, C) -> (B, C, L)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv3(x))
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

# -------------------------
# UPDATED TRAINING LOGIC
# -------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    return running_loss / len(loader.dataset)

def train_model(train_loader, input_channels, num_outputs):
    model = CNN(input_channels=input_channels, num_outputs=num_outputs).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Scheduler: Reduces LR if loss doesn't improve for 2 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    
    pos_weight = torch.tensor([5.0] * num_outputs).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        scheduler.step(loss) # Update scheduler based on loss
        print(f"Epoch {epoch} Loss: {loss:.4f} | LR: {optimizer.param_groups[0]['lr']}")
        
    return model

# -------------------------
# HELPERS (RETAINED)
# -------------------------
def read_fasta(fasta_file):
    sequences = []
    seq = ""
    with open(fasta_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq: sequences.append(seq)
                seq = ""
            else: seq += line.upper()
        if seq: sequences.append(seq)
    return sequences

def load_data(chr_num):
    fasta_path = os.path.join(FASTA_DIR, f"chr{chr_num}_200bp_bins.fa")
    tsv_path = os.path.join(TSV_DIR, f"chr{chr_num}_200bp_bins.tsv")
    sequences = read_fasta(fasta_path)
    df = pd.read_csv(tsv_path, sep="\t")
    return sequences, df

def main():
    # Example: Training on Chromosome 1
    seqs, df = load_data(1)
    
    dataset = GenomicDataset(seqs, df, augment_rc=USE_RC_AUGMENTATION)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # input_channels = 4 (DNA) + 1 (ATAC) = 5
    model = train_model(loader, input_channels=5, num_outputs=len(TF_LIST))
    
    torch.save(model.state_dict(), "cnn_v2_final.pth")
    print("Training complete and model saved.")

if __name__ == "__main__":
    main()
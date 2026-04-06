import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from NoGarbageIn_op import load_chromosome, TRAIN_CHRS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1024
EPOCHS = 6
LR = 1e-3

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(9,64,15)
        self.conv2 = nn.Conv1d(64,128,7)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128,3)

    def forward(self,x):
        x = x.permute(0,2,1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

def train():
    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss()

    X_all, y_all = [], []

    print("Preloading...")
    for c in TRAIN_CHRS:
        X, y, _ = load_chromosome(c)
        X_all.append(X)
        y_all.append(y)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}")

        for ci in range(len(X_all)):
            X = X_all[ci]
            y = y_all[ci]

            idx = np.random.permutation(len(X))

            for i in tqdm(range(0,len(X),BATCH_SIZE)):
                xb = torch.from_numpy(X[idx[i:i+BATCH_SIZE]]).to(DEVICE)
                yb = torch.from_numpy(y[idx[i:i+BATCH_SIZE]]).to(DEVICE)

                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()

if __name__ == "__main__":
    train()

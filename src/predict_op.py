import numpy as np
import torch
from tqdm import tqdm
from noGarbageIn_fast import load_chromosome, PRED_CHRS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(model):

    model.eval()

    for c in PRED_CHRS:
        print(f"\nPredicting chr{c}")

        X, _, df = load_chromosome(c, test=True)

        preds_all = []

        for i in tqdm(range(0,len(X),10000)):
            xb = torch.from_numpy(X[i:i+10000]).to(DEVICE)
            with torch.no_grad():
                preds = torch.sigmoid(model(xb)).cpu().numpy()
            preds_all.append(preds)

        preds = np.vstack(preds_all)

        for i,tf in enumerate(["CTCF","REST","EP300"]):
            df[tf] = preds[:,i]

        df.to_csv(f"chr{c}_predictions.tsv",sep="\t",index=False)

        print(f"Saved chr{c}_predictions.tsv")

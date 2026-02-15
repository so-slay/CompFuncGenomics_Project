import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import GetFASTAfromTSV as gf 

file = pd.read_csv("outputs/0-Order-5-foldCV_loglikelihoods-chr4_200bp_bins.tsv", sep="\t")

def plotter(file):
    # Load your data
    df = pd.read_csv(file, sep="\t" )

    # Choose target column (change as needed)
    target = gf.config.get("which_factor")

    # True labels
    y_true = df[target].values

    # Prediction scores (use llr; if lower = stronger signal, multiply by -1)
    y_score = -df["llr"].values  # flip sign if needed

    # -------------------
    # ROC Curve
    # -------------------
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # -------------------
    # Precision-Recall Curve
    # -------------------
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    # -------------------
    # Plot
    # -------------------
    plt.figure(figsize=(12,5))

    # ROC
    plt.subplot(1,2,1)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.5f}")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    # PR
    plt.subplot(1,2,2)
    plt.plot(recall, precision, label=f"AP = {pr_auc:.5f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()


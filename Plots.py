import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import CrossValidationScores as cv
import markovNull as mm
import GetFASTAfromTSV as gf
from pathlib import Path


def plotter(file):
    # Load your data
    
    df = pd.read_csv(file, sep="\t" )

    # Choose target column (change as needed)
    target = gf.config.get("which_factor")

    # True labels
    y_true = df[target].values

    # Prediction scores (use llr; if lower = stronger signal, multiply by -1)
    y_score = df["llr"].values  # Sometimes -df["llr"] shows positive AUC

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

    return fpr, tpr, roc_auc, precision, recall, pr_auc

def main(files: list):
    # List to store ROC curve data for plotting all on the same graph
    all_fpr = []
    all_tpr = []
    all_roc_auc = []

    # List to store metrics for printing
    metrics = []
    
    output_dir = Path('outputs/')

    # Loop through all TSV files in the 'outputs/' directory
    for file_path in files:
        file_path = Path(file_path)
        # Get the ROC and Precision-Recall data
        fpr, tpr, roc_auc, precision, recall, pr_auc = plotter(file_path)

        # Store ROC data for later plotting
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_roc_auc.append(roc_auc)

        # Collect metrics to print later, use the filename here
        metrics.append(f"File {file_path.name}: ROC AUC = {roc_auc:.5f}, PR AUC = {pr_auc:.5f}")

        # -------------------
        # Plot PR curve (individual)
        # -------------------
        plt.figure(figsize=(12, 5))

        # ROC curve
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.5f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve: {file_path.name}")
        plt.legend()

        # Precision-Recall curve
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, label=f"AP = {pr_auc:.5f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve: {file_path.name}")
        plt.legend()

        plt.tight_layout()
        plt.show()

    # Optionally print out metrics for each file
    print("\nMetrics for each file:")
    for metric in metrics:
        print(metric)

    # -------------------
    # Plot ROC curves for all files on the same graph
    # -------------------
    plt.figure(figsize=(10, 8))

slice_at = []
if __name__ == "__main__":
    tsv_files = cv.pipeline_cv(mm.dict_of_dfs)
    main(tsv_files)
    




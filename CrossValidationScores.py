from pathlib import Path
import time

# Helper libs I wrote
import markovNull as mm 
import GetFASTAfromTSV as gf

# standard ds libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

from sklearn.model_selection import StratifiedKFold

start = time.time()

# GOALS: k-fold cross validation

which_factor = gf.config.get("which_factor")
markov_order = int(gf.config.get("markov_order"))
kfolds  = int(gf.config.get("k_fold"))

"""
Logic: 

randomly shuflle data (rows of each df)

For a specified order
Split in to kfolds

In each such partitioning:
    Train by bound vs. unbound label for specified TF

    Score the left-out set
Average all scores 
Plot stats
"""

def loglikely_cv(train_df, val_df, which_factor, markov_order):
        
    # Define bound vs unbound sequences in training and validation sets:

    bound_seqs = train_df.loc[train_df[which_factor] == 1, "sequence"].tolist()
    unbound_seqs = train_df.loc[train_df[which_factor] == 0, "sequence"].tolist()

    # Learn markov models with training set
    bdd_model = mm.markov_k(bound_seqs, markov_order)
    unb_model = mm.markov_k(unbound_seqs, markov_order)

    # Score bound and unbound regions in validation set
    llr = val_df["sequence"].apply(
        lambda seq: mm.sequence_score(seq, bdd_model, unb_model, markov_order)
    )

    return llr

def kfold_cv(df, kfolds=kfolds, markov_order=markov_order, which_factor=which_factor):
    # Placeholder for loglikelihood scores
    log_li_s = np.zeros(len(df), dtype=float) 

    X = df["sequence"]  # just a placeholder, we only need indices
    y = df[which_factor].to_numpy()

    skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=42)

    for train_idx, cv_idx in skf.split(X, y):
        train_df = df.iloc[train_idx]
        cv_df = df.iloc[cv_idx]

        log_li_s[cv_idx] = loglikely_cv(train_df, cv_df, which_factor, markov_order)

    return log_li_s

# Test function
# for k, df in mm.dict_of_dfs.items():
#     # learn Markov model of order k FOR EACH CLASS
#     # Write log-likelihood sequence scores to sequences in these classes
#     # Calculate scores in bound vs. unbound sets

#     df["llr"] = kfold_cv(df)
#     print(df.head())


# Write outputs 
base_dir = gf.input_dir
output_dir = base_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

runtimes = []

def main():
    for iterable_markov_order in range(0, 11):
        subTime = time.time()
        for k, df in mm.dict_of_dfs.items():
            df["llr"] = kfold_cv(df, kfolds=5, markov_order=iterable_markov_order)
            # Write output
            outfile = output_dir / f"{iterable_markov_order}-Order-{kfolds}-foldCV_loglikelihoods-{k}.tsv"
            df.to_csv(outfile, sep="\t", index=False)
        
        endsubTime = time.time()
        runTime = endsubTime- subTime
        runtimes.append((iterable_markov_order, runTime))
        print(f"Runtime at order {iterable_markov_order}: {runTime:.4f} seconds")
        
    end = time.time()

    print(f"Runtime for 11 orders: {end - start} seconds")

    # I used ChatGPT for coding the dataframe below and plotting the runtimes
    # Write each df out into cache
    runtime_df = pd.DataFrame(runtimes, columns=["order", "helper_runtime"])
    runtime_df["helper_constant"] = mm.helperRunTime

    runtime_df.to_csv(output_dir / "runtimes.out", sep="\t", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(runtime_df["order"], runtime_df["helper_runtime"], marker='o', label="Helper Runtime")
    plt.plot(runtime_df["order"], runtime_df["helper_constant"], marker='x', label="Constant Helper Runtime")

    plt.xlabel("Markov Order")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtimes per Markov Order")
    plt.legend()
    plt.grid(True)
    plt.show()

    tsv_files = list(output_dir.glob("*.tsv"))
    if tsv_files: 
        import Plots 
        Plots.main(tsv_files)
    else:
        print("Error plotting ROCs: no files found")



def pipeline_cv(dict):
    for k, df in mm.dict_of_dfs.items():
        output_files = []
        df["llr"] = kfold_cv(df)
        # Write output
        outfile = output_dir / f"{markov_order}-Order-{kfolds}-foldCV_loglikelihoods-{k}.tsv"
        print(f"{markov_order}-Order-{kfolds}-foldCV_loglikelihoods-{k}.tsv written to folder {output_dir}")
        df.to_csv(outfile, sep="\t", index=False)
        output_files.append(outfile)
    return output_files


if __name__ == "__main__":
    main()

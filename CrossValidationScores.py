from pathlib import Path
import time

# Helper libs I wrote
import markovNull as mm 
import GetFASTAfromTSV as gf

# standard ds libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


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
    ll_bdd = val_df["sequence"].apply(lambda seq: mm.sequence_score(seq, bdd_model))
    ll_unb = val_df["sequence"].apply(lambda seq: mm.sequence_score(seq, unb_model))

    llr = ll_bdd - ll_unb
    return llr

def kfold_cv(df, kfolds=kfolds, markov_order=markov_order, which_factor=which_factor):
    #Shuffle df
    df = df.sample(frac=1).reset_index(drop=1)

    # Folds:
    fold_sizes = np.full(kfolds, len(df)//kfolds)
    fold_sizes[:len(df)%kfolds] += 1

    tally = 0
    folds = []
    for i in fold_sizes:
        folds.append(df.iloc[tally:tally+i].index.tolist())
        tally += i
    
    # Placeholder for loglikelihood scores
    log_li_s = pd.Series(index=df.index, dtype=float)

    
    for i in range(kfolds):
        cv_ids = folds[i]
        # Define which indices are trainig vs. validation
        train_lists = folds[:i] + folds[i+1:]
        train_ids = [i for fold in train_lists for i in fold]
        # Map these indices to df    
        train_df = df.loc[train_ids]
        cv_df = df.loc[cv_ids]

        log_li_s.loc[cv_ids] = loglikely_cv(train_df, cv_df, which_factor, markov_order)

    return log_li_s

# Tes function
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

for markov_order in range(0, 11):
    subTime = time.time()

    for k, df in mm.dict_of_dfs.items():
        df["llr"] = kfold_cv(df)
        # Write output
        outfile = output_dir / f"{markov_order}-Order-{kfolds}-foldCV_loglikelihoods-{k}.tsv"
        df.to_csv(outfile, sep="\t", index=False)
    
    endsubTime = time.time()
    runTime = endsubTime- subTime
    runtimes.append((markov_order, runTime))
    print(f"Runtime at order {markov_order}: {runTime:.4f} seconds")

end = time.time()

print(f"Runtime for 11 orders: {end - start} seconds")

# I used ChatGPT for building the dataframe below and plotting the runtimes
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

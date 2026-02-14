import pandas as pd
import numpy as np
import subprocess
import GetFASTAfromTSV as gt
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

s = pd.Series(["B", "U", "B", "U"])
r = "ADHFHGJKKKKKKGKBMGKMGKHKHKKKKKHKHNBJGKBMKKKHKBMGKKKHBMKG"
print(len(r))
k = 2
counts = defaultdict(Counter)
for seq in [r]:
    for i in range(len(r) - k):
        
        context = seq[i:i+k]
        next_nt = seq[i+k]
        print(defaultdict[next_nt])
        counts[context][context] += 1

    print(counts)


# Work for tomorrow:
"""
1. evolution assignment
TRAIN SEPARATE MODELS
2. Cross validation over k 0 to 10
3. AUC, ROC
4. RUN whole on all.

"""

"""
Archive:

1. Clean-up/Preprocess from markov null

for k, df in dict_of_dfs.items():
    # Set mask to ignore NaNs
    df['unmapped'] = df.iloc[:, 0].isna() # Bool 1 for NaN
    mask = ~df['unmapped'] # Bool 0 for NaN, 1 for mapped regions

    # Convert U B to Boolean with bind_Boolmap
    df[cols_to_Bool] = df[cols_to_Bool].apply(lambda col: col.map(bind_Boolmap))

    # Count using Counter for 0-order markov model:
    df.loc[mask, 'counts'] = df.loc[mask, 'sequence'].apply(Counter)

    # Convert counts to probabilities:
    df.loc[mask, "probs"] = df.loc[mask, "counts"].apply(lambda c: {k: v/sum(c.values()) for k,v in c.items()})


2. old order 0 markovs
for k, df in dict_of_dfs.items():    
    # Make ATCG counts a df
    df = df[df["probs"].apply(lambda x: isinstance(x, dict))] # drop None instances
    prob_df = df["probs"].apply(lambda d: {nt: d.get(nt, 0) for nt in nucleotides}).apply(pd.Series)
    
    # Create a mask from parent df:
    bound_mask  = df[which_factor] == 1    

    # Calculate averages for bound, unbound
    bound_avg = prob_df[bound_mask].mean(axis=0) 
    unbound_avg = prob_df[~bound_mask].mean(axis=0) 
    bg_stats[k] = {"bound":bound_avg, "unbound":unbound_avg}
    print(f"{df} stats: Bound Probs: {bound_avg} , Unbound Probs:{unbound_avg}")



"""
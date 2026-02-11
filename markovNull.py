from pathlib import Path
import math
import numpy as np 
import pandas as pd 
import GetFASTAfromTSV as gf 
from collections import Counter, defaultdict

header = ['chr', 'start', 'stop', 'ATAC', 'CTCF', 'REST', 'EP300', 'sequence' ]

# Script Objectives: Read Fasta files, and count ATGC nucleotides

# Inputs: .bed & .fa files with sequences 
# Basics: Read the bed files, and make a data frame
# Append the sequence info from corresponding FASTA file
# Handle "NaNs"

# Choose one TF
# "DO you want to use ATAC info?" > Select only B in ATAC and then train
# Separate U and B
config_path = gf.config_path
print(config_path)
input_dir = Path(gf.config["input_dir"] or ".").resolve()
bed_files = list(input_dir.glob("*.bed"))
fasta_files= list(Path("FASTAs").glob("*.fa")) # strip extenstions

# Compose ditionary of {bed_file:Fasta File}
bed_fasta_dict = {}
fasta_dict = {f.stem: f for f in fasta_files}
for bed in bed_files: 
    bstem = bed.stem
    bed_fasta_dict[bed] = fasta_dict[bstem]
    # This should link same-named bed and fasta files to each other

# Data structuring: list of dfs
# Each sub df is a combination of bed and fasta info.
dict_of_dfs = {}
for bed, fasta in bed_fasta_dict.items():
    df_bed = pd.read_csv(bed, sep="\t", header=0)
    df_fasta = pd.read_csv(fasta, header =None)
    df_fastest = df_fasta.iloc[1::2].reset_index(drop=1) # Keep the sequences (only present in lines 1,3..)
    df = pd.concat([df_bed, df_fastest], axis =1)
    df.columns = header
    print(f'bed: {bed}; fasta: {fasta} concatenated to give: {df}')
    # Create a dictionary so dfs can be named
    dict_of_dfs[Path(bed).stem] = df

# Note: for a very robust pipeline, the lines 0,2,4... of the fasta file can be used to map to the right bin in bed. (perhaps later)

bind_Boolmap = {"B":1, "U":0}
cols_to_Bool = ["ATAC", "CTCF", "REST", "EP300"]

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


## Markov Model of Order 0: Calculate nucleotide frequencies (background) for bound, unbound
which_factor = "CTCF"
# which_factors = gf.config["which_factor"]
# atac = gf.config["USE ATAC info?"]

nucleotides = ["A", "C", "G", "T"]

bg_stats = {k:{"bound":pd.Series(dtype=float) , "unbound":pd.Series(dtype=float)}for k in dict_of_dfs.keys()}

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


# Define a k-th order markov model fn, where k is an integer from 1 to 10
# k = gf.config["MarkovOrder"]
order_k = int(3)

# Build a function to work on any sequence string
# Iterate over list of sequences
# do this in bound/unbound sets for specified factor

def markov_k(seq_list, k=order_k):
    pseudoc = int(1)
    counts = defaultdict(Counter)

    # Consider a sequence of Length = L
    # for a sliding window, we can go from i = 1 to i = L-k (1 based)
    # in 0-based indexing this is i = 0  to L-1-k
    for seq in seq_list:
        for i in range(len(seq) - k):
            k_prefix = seq[i:i+k] # These functions not inclusive of upper arg == until i+k-1 etc
            k_letter = seq[i+k]
            # Build nested dictionary of outer 'k_prefix'es
            # then count each occurence of the 'next'==kth letter 
            counts[k_prefix][k_letter] +=1
    markov_model = {}
    
    for k_prefix,vals in counts.items():
        total = sum(v + pseudoc for v in vals.values())
        markov_model[k_prefix] = {}

        for nt in nucleotides:
            count = vals.get(nt, 0) + pseudoc
            prob = count/total
            markov_model[k_prefix][nt] = math.log(prob)
    return markov_model

def log_likelihood_scorer(seq, markov_model, k=order_k):
    default_log_p = -100
    log_lhood = 0.0
    for i in range(len(seq) -k):
        k_prefix = seq[i:i+k] 
        k_letter = seq[i+k]
        log_lhood += markov_model.get(k_prefix, {}).get(k_letter, default_log_p)
    return log_lhood

for k, df in dict_of_dfs.items():
    # learn Markov model of order k 
    # Write log_likelihood scores
    # Calculate scores in bound vs. unbound sets
    sequences = df["sequence"].tolist()
    row_markov = markov_k(sequences, order_k)
    # ll is short for Log-Likelihood :)
    df["ll"] = df["sequence"].apply(lambda c: log_likelihood_scorer(c, row_markov, order_k))

    # define bound set:
    bound_mask  = df[which_factor] == 1 
    bound_avg = df.loc[bound_mask, "ll"].mean()
    unbound_avg = df.loc[~bound_mask, "ll"].mean()

    bg_stats[k] = {"bound":bound_avg, "unbound":unbound_avg}
    print(f"{df} stats: Bound Probs: {bound_avg} , Unbound Probs:{unbound_avg}")

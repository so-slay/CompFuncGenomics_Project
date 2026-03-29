from pathlib import Path
import math
import numpy as np 
import pandas as pd 
import GetFASTAfromTSV as gf 
import time 
start = time.time()

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
    df_fasta = pd.read_csv(fasta, header =None, comment='>')
    df = pd.concat([df_bed, df_fasta], axis =1)
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

    # Yeet NaNs in col1
    df2 = df.copy() 
    df2 = df2.dropna(subset=[df2.columns[0]])
    dict_of_dfs[k] = df2

## Markov Model of Order k:
# Define a k-th order markov model fn, where k is an integer from 1 to 10

order_k = int(gf.config.get("markov_order"))
which_factor = gf.config.get("which_factor")

# which_factors = gf.config["which_factor"]
# atac = gf.config["USE ATAC info?"]

nucleotides = ["A", "C", "G", "T"]

bg_stats = {k:{"bound":pd.Series(dtype=float) , "unbound":pd.Series(dtype=float)}for k in dict_of_dfs.keys()}


# Build a function to work on any sequence string
# Iterate over list of sequences
# do this in bound/unbound sets for specified factor in all dfs

def markov_k(seq_list, k=order_k):

    counts = {}

    # Consider a sequence of Length = L
    # for a sliding window, we can go from i = 1 to i = L-k (1 based)
    # in 0-based indexing this is i = 0  to L-1-k
    for seq in seq_list:
        for i in range(len(seq) - k):
            k_prefix = seq[i:i+k] 
            k_letter = seq[i+k]

            # Build nested dictionary of outer 'k_prefix'es 
            if k_prefix not in counts:
                counts[k_prefix] = {}

            # Then count each occurence of the 'next'==kth letter
            if k_letter not in counts[k_prefix]:
                counts[k_prefix][k_letter] = 0

            counts[k_prefix][k_letter] +=1

    
    markov_model = {}
    # Convert raw counts into probabilities
    # pseudocounts (For postmidsem- Bayesian stuff) 
    pseudoc = int(1)
    for k_prefix,vals in counts.items():
        total = sum(vals.get(nt, 0) + pseudoc for nt in nucleotides)

        markov_model[k_prefix] = {}

        for nt in nucleotides: #Refer line 70 for nucleotides
            count = vals.get(nt, 0) + pseudoc
            prob = count/total
            markov_model[k_prefix][nt] = math.log(prob)

    return markov_model

def sequence_score(seq, markov_pos_model, markov_neg_model, k=order_k):

    # Score each sequence in the list using the trained models 
    default_log_p = math.log(1/4) # Assumes any nucleotide is equally likely
    log_lhood = 0.0

    for i in range(len(seq) - k):
        k_prefix = seq[i:i+k] 
        k_letter = seq[i+k]

        bdd_score = markov_pos_model.get(k_prefix, {}).get(k_letter, default_log_p)
        unb_score = markov_neg_model.get(k_prefix, {}).get(k_letter, default_log_p)
        log_lhood += (bdd_score - unb_score)
    return log_lhood


# Pls Ignore stuff below, older function not deleting because useful syntactically

def loglikely(df, bdd_mask):

    # Input: df, bound and unbound markov models
    # Output: log_likelihood ratio for each sequence in 

    # Extract sequence col as list
    bound_seqs = df.loc[bdd_mask, "sequence"].tolist()
    unbound_seqs = df.loc[~bdd_mask, "sequence"].tolist()

    # learn Markov model of order k on it
    bdd_model = markov_k(bound_seqs, order_k)
    unb_model = markov_k(unbound_seqs, order_k)

    llr = df["sequence"].apply(
        lambda seq: sequence_score(seq, bdd_model, unb_model)
    )

    return llr


# for k, df in dict_of_dfs.items():
#     # learn Markov model of order k FOR EACH CLASS
#     # Write log-likelihood sequence scores to sequences in these classes
#     # Calculate scores in bound vs. unbound sets
#     bound_mask  = df[which_factor] == 1 

#     df["llr"] = loglikely(df, bound_mask)
    
#     print(df.head())

end = time.time()
# Write each df out into cache
helperRunTime = end - start
print(f"Runtime (loading markov helpers): {helperRunTime} seconds")

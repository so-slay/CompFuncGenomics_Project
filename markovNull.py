from pathlib import Path
import numpy as np 
import pandas as pd 
import GetFASTAfromTSV as gf 
from collections import Counter

header = ['chr', 'start', 'stop', 'ATAC', 'CTCF', 'REST', 'EP300', 'sequence' ]

# Script Objectives: Read Fasta files, and count ATGC nucleotides

# Inputs: .bed & .fa files with sequences 
# Basics: Read the bed files, and make a data frame
# Append the sequence info from corresponding FASTA file
# Handle "Ns"

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
chrom_dfs = []
for bed, fasta in bed_fasta_dict.items():
    df_bed = pd.read_csv(bed, sep="\t", header=0)
    df_fasta = pd.read_csv(fasta, header =None)
    df_fastest = df_fasta.iloc[1::2].reset_index(drop=1) # Keep the sequences (only present in lines 1,3..)
    df = pd.concat([df_bed, df_fastest], axis =1)
    df.columns = header
    print(f'bed: {bed}; fasta: {fasta} concatenated to give: {df}')
    chrom_dfs.append(df)

# Note: for a very robust pipeline, the lines 0,2,4... of the fasta file can be used to map to the right bin in bed. (perhaps later)


# Count using Counter for 0 order markov model:
for df in chrom_dfs:
    df['counts'] = df['sequence'].apply(Counter)
# Comprehension version of the same as above:
# chrom_dfs = [df.assign(counts=df['sequence'].apply(Counter)) for df in chrom_dfs]
print(chrom_dfs)


#markov_null 
# Figure out how to do this for A | G etc... 


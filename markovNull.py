from pathlib import Path
import numpy as np 
import pandas as pd 

header = ['chr', 'start', 'stop', 'ATAC', 'CTCF', 'REST', 'EP300' ]

# Script Objectives: Read Fasta files, and count ATGC nucleotides

# Inputs: .bed & .fa files with sequences 
# Basics: Read the bed files, and make a data frame
# Append the sequence info from corresponding FASTA file
# Handle "Ns"

# Choose one TF
# "DO you want to use ATAC info?" > Select only B in ATAC and then train
# Separate U and B

bed_files = Path("data").glob("*.bed")
fasta_files = Path("FASTAs").glob("*.fa") # strip extenstions


# Compose ditionary of {bed_file:Fasta File}
# for i in Bed:fasta
for bed,fasta in 
with open(bed_files) as f:
    df = pd.read_csv(f, sep='\t')
with open#BEDFILE names) as g: 
    df2 = pd.read_csv(fasta)
    full_df = pd.concat([df, df2], axis=1)


# use collections.Counter to count A T C G 
# Figure out how to do this for A | G etc..

# for row in dfdf[counts] = Counter(df.[sequence]) 


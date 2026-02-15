# CompFuncGenomics_Project
CFG Project Repo: Spring 2026
1. TSV converted to BED for using Bedtools Getfasta tool
2. run GetFASTAfromBED.sh on these files 
    This used bedtools to fetch the fasta sequences
3. Read the FASTA and bed files, concatenate them into a pandas dataframe
3.1: Preprocess Data to remove NaNs or Unmapped regions
4. Markov times:
read whether the TF is Bound or unbound and calculate probabilities 
4.1: Null Markov model: Raw counts of each nucleotide
4.2: Order-1 MM: Look at all 2-mers and count how many times, say AA out of AA+AT+AC+AG etc. and calculate log likelihoods
4.3: Order-k MM: Look at all (k+1)-mers and look at how many times the last letter is say, A, given the prefix is the same. Calculate log likelihoods.


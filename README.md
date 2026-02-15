# CompFuncGenomics_Project (CFG)

**CFG Project Repo — Spring 2026**  

#### Python-based pipeline to process genomic sequences and calculate Markov model probabilities for transcription factor (TF) binding analysis.
---
## Configuration

### Quick Start
In a hurry?  
The simplest way to run this code:

1. Place the `.tsv` file(s) of interest in the same folder as `00_Main.py`.  
2. Run `00_Main.py` with no arguments 
- Automatically process **all `.tsv` files** in this directory.


Default parameters are set in `01Default_config.py`:

| Parameter       | Default Value               | Description |
|-----------------|----------------------------|-------------|
| `input_dir`     | (user-specified folder)     | Base folder for input FASTA/TSV files and outputs |
| `files`         | chr4_200bp_bins.tsv         | TSV files to process (comma-separated if multiple) |
| `markov_order`  | 10                          | Order for Markov chain |
| `k_fold`        | 5                           | Number of folds for cross-validation |
| `which_factor`  | CTCF                        | Choose TF: CTCF, REST, or EP300 |

**Notes:**  
- All outputs (TSV, plots, logs) are written to subfolders of `input_dir`.  
- Filenames should **not** include quotes.  

## Detailed Run Instructions:

### Quick Tweaks
For small changes, you can directly edit the arguments in `01Default_config.txt`.

### Custom/Reproducible Runs

For record-keeping/choosing specific files from a list of files and other complications:

1. **Create a new config file**  
   - Make a copy of `default_config.txt` and modify the arguments as needed.

2. **Run with custom config**  
   - Pass your config file as an argument to `00_Main.py` like so:

   ```bash
   python3 00_Main.py your_config.txt
---

## Project Workflow

1. **TSV → BED conversion**  
   Convert input TSV files into BED format to be used with Bedtools `getfasta`.  

2. **Fetch sequences**  
   Run `GetFASTAfromBED.sh` to fetch FASTA sequences using Bedtools.  

3. **Read and preprocess data**  
   - Load FASTA and BED files into a pandas DataFrame.  
   - Clean the data: remove `NaN`s and unmapped regions.  

4. **Markov model analysis**  
   - Determine whether each TF is **Bound** or **Unbound**.  
   - Calculate nucleotide probabilities with different order Markov models:  

   **4.1 Null Markov model**  
   - Raw counts of each nucleotide converted to **Probabilities** (simply normalize by totat count)

   **4.2 Order-1 Markov model**  
   - Count all 2-mers (e.g., AA, AT, AC, AG) and compute log-likelihoods.  

   **4.3 Order-k Markov model**  
   - Count all (k+1)-mers.  
   - For a given prefix, calculate the probability of the last nucleotide and compute log-likelihoods.  

5. **Markov model training**  
   - Train order 0–10 Markov chains on bound vs unbound sequences.  
   - Implement k-fold cross-validation  

6. **Performance evaluation**  
   - Plot ROC, AUC, and PRC using `scikit-learn`.  

---

## Dependencies

### System-level tools
- **bedtools**: v2.31.1  

### Python
- **Python**: 3.12.7  

### Python Libraries

#### Standard / built-in libraries
- `os`, `sys`, `time`, `math`, `pathlib`, `subprocess`  

#### Installed libraries (via pip / conda)
- `numpy`  
- `pandas`  
- `matplotlib`  
- `scikit-learn`  
- `tqdm`  (soon)

#### Helper libraries
- `GetFASTAfromTSV` (`gf`)  
- `markovNull` (`mm`)  
- `CrossValidationScores`(`cv`)
- `Plots`

---
## Blame:

#### **Worklfow implementation, Markov Recursion, Cross-Validation** - S. V. Ananthakrishna
#### **ROCs, Statististics, Plots** -Saahil Dholakia
---


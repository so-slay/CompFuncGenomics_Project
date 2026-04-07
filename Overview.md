# CFG Project Jan 2026
## VINCA: A CNN-based model to predict TF-binding probabilities
Task at hand: 
Predicting binding probability of CTCF, REST, EP300 on chr3, chr10, chr17 (K562, hg38)
Final output: chr3/10/17_predictions.tsv.gz with probability scores per TF per 200bp bin

---

## Project Structure
```project/
├── data/
│   ├── raw/
│   │   ├── FASTAs/         
│   │   ├── tsv/             
│   │   └── methylation/     
│   └── processed/
│       ├── chr{c}_methylation.npy   
│       ├── chr{c}_pwm.npy           
│       └── ...
├── models/
│   └── checkpoints/
│       ├── best_model.pt            
│       └── ...         
├── predictions/
│   ├── chr3_predictions.tsv.gz
│   ├── chr10_predictions.tsv.gz
│   └── chr17_predictions.tsv.gz
├── src/
│   ├── methylation_precompute.py    
│   ├── pwm_precompute.py            
│   ├── dataset.py                   
│   ├── model.py                     
│   ├── train.py                     
│   └── predict.py                   
└── main.py                          
```
---

## Code Overview 

### methylation_precompute.py
- Source: ENCFF660IHA.bed.gz (ENCODE WGBS, K562, gemBS bed9+ CpG format)
- Columns used: col0=chrom, col1=start, col9=coverage, col10=meth_pct
- Filters CpG sites with coverage < 5 (keep if more)
- For each 200bp bin: mean methylation fraction across all CpG sites in bin
- Vectorized with searchsorted — fast
- Output: data/processed/chr{c}_methylation.npy, shape (num_bins,), float32
- Skips already-processed chromosomes
- Run: python src/methylation_precompute.py

### CNN code - model.py, train.py and predict.py
1. evaluate() called 3x per chromosome — fixed by saving preds from final epoch
2. Model reinit bug — fixed using state_dict() + copy.deepcopy()
   - Tracks best epoch within each chromosome
   - Tracks best model globally across all chromosomes and configs
   - Saves best_in_ch for model reconstruction at predict time
3. Model reconstruction before prediction loop:
   best_model = CNN(best_in_ch).to(DEVICE)
   best_model.load_state_dict(best_state_dict)


### JASPAR PWMs: pwm_precompute.py
- CTCF: MA0139.1 
- REST: MA0138.2 
- EP300: zeros (no direct DNA-binding domain) 
- Cell line context: K562
- For each bin: max PWM score across all positions in the 200bp window
- Output: data/processed/chr{c}_pwm.npy, shape (num_bins, 3), float32
  where col0=CTCF, col1=REST, col2=EP300(zeros)
- Requires: pip install biopython

---

## Detailed workflow:

### STEP 1 — Precompute all features 
python src/methylation_precompute.py  
python src/pwm_precompute.py        
  

### STEP 2 — noGarbageIn.py
Responsibilities:
- read_fasta(path) → list of sequences
- reverse_complement(seq)
- encode_batch(seqs, atac, meth, pwm, use_atac, use_meth, use_pwm)
  Input channels:
    0-3:  one-hot DNA          always on
    4:    ATAC scalar          optional
    5:    methylation scalar   optional
    6-8:  PWM scores (3 TFs)  optional (EP300 all zero currently)
- load_chromosome(c, test=False) → seqs, atac, meth, pwm, y, df

### STEP 3 — model.py
- Dilated convolutions in ResBlock for longer-range context
- Configurable input channels (based on active features)
- FocalLoss stays (handles class imbalance well)
- DEVICE-aware throughout

### STEP 4 — train.py
- GLOBAL training: pools all training chromosomes
- Hold out one chromosome for validation (e.g. chr1)
- Choose batch size(512) and number of epochs(20) with early stopping (with patience=3 on val ROC)
- CosineAnnealingLR scheduler
- Config-based model selection stays: DNA / DNA+ATAC / DNA+ATAC+METH / DNA+ATAC+METH+PWM
  → best config selected by val ROC, worst configs pruned
- Test-time augmentation: average predictions on fwd + reverse complement
- DEVICE-aware throughout
- Warning: Streaming not yet implemented, the script uses tons of RAM currently

### STEP 5 — predict.py
- Load best_model.pt + best_config.json
- Predict on chr3, chr10, chr17
- TTA (fwd + RC average)
- Output: predictions/chr{c}_predictions.tsv.gz

#### Behaviour without best_model.pt
- A key feature of this script is the fast forward feature: 
You might already know what choice of inputs you want OR frequently one does not have the time and memory to train and choose the best model
- training data is streamed to disk and read, to be able to run on even basic computers.
- running predict.py without running train.py first generates a model in the ALL configuration: DNA + ATAC + JASPAR, trains on this for 20 Epochs, and predicts with this model.


### STEP 6 — main.py
- Calls precompute → train → predict in sequence
- Checks if processed files already exist before recomputing

---

## Key Constraints (Guidelines)
- No public ENCODE-DREAM challenge code
- No other ChIP-seq/ChIP-exo/CUT&RUN data
- JASPAR PWMs allowed 
- K562 cell line, hg38 genome
- Test chromosomes: 3, 10, 17
- Train chromosomes: all others (1-22 except 3,10,17)
- Output must be probability scores (higher = more likely bound)

---

## Dependencies
torch, numpy, pandas, scikit-learn, matplotlib, biopython (for PWM)

## Compute Requirements:
DO
Python 3.10 Nvidia GTX 1650 (Mobile) with Cuda took around 8 hours for 20 - Epochs of training (config: all inputs)

---

## Input Feature Channels
| Channel | Source                | Shape per bin  | Status     |
|---------|-----------------------|----------------|------------|
| 0-3     | One-hot DNA           | (200, 4)       | done    |
| 4       | ATAC scalar           | (1,)           | done    |
| 5       | CpG methylation       | (1,)           | done    |
| 6       | CTCF PWM score        | (1,)           | done    |
| 7       | REST PWM score        | (1,)           | done    |
| 8       | EP300 PWM (zeros)     | (1,)           | all 0    |
| 9       | RNA-seq log TPM       | (1,)           | toDo(Future work) |

---

## Strategy: 

- Based primarily on discussions in class: Computational Functional Genomics taught by Prof, Leelavati Narlikar at IISER Pune. Credit also to Claude AI for countless debugs and optimizations that allowed this code to be written; and the book 'An introduction to statistical learning' (Hastie et al) for insights and motivation with handling data.


**Split strategy:** Not doing: Chromosome-level 70/15/15 split: not random (nearby bins are correlated)
Train: chr1,2,4,5,6,7,8,9,11,12,13,14,15,16. Val: chr18,19,20. - or randomly hold out 3
Test: chr21,22. Predict: chr3,10,17 (no labels). Val selects best config, test gives honest AUC.

**Ablation Study / Config-based model selection:** Four input configs trained independently: DNA only,
DNA+ATAC, DNA+ATAC+METH, DNA+ATAC+METH+PWM. Best config selected by val ROC-AUC.
This answers: which combination of features actually helps? If a feature doesn't contribute,
it gets pruned by the selection process.
- inspired by the paper C_Origami that implements this beautifully (\cite[])

**FIMO/ZOOPS decision:** Decided against. FIMO gives discrete hit/no-hit with p-value vs
our continuous max log-odds score — marginal gain for well-characterised motifs (CTCF, REST).
ZOOPS is a discovery model, not useful when motifs are already known. Revisit only if PWM
features don't improve ROC over DNA-only baseline.

**RNA-seq expression (Future work):** K562 RNA-seq TPM from ENCODE mapped to nearest
gene TSS per bin → log TPM scalar input channel. Particularly useful for EP300 (no motif signal)
since EP300 binds active enhancers which correlate with nearby gene expression. Requires hg38
GTF + K562 RNA-seq TPM file (both ENCODE, both allowed by project rules).
Implemented in rna_precompute.py after core pipeline is working.

**Hi-C:** Decided against. 200bp bins require ultra-deep Hi-C for reliable
contacts — resolution mismatch too severe. ATAC is a better proxy.

**Model:** 1D ResNet CNN with dilated convolutions for longer-range context, configurable
input channels, FocalLoss for class imbalance, CosineAnnealingLR, early stopping (patience=3).
Test-time augmentation: average predictions on forward + reverse complement strands.
CUDA-aware throughout. Precompute scripts are CPU-only (I/O bound).

---

### What's vinca?
Variable Input Network of Convolutions Affinity-mapper
Inspired by the latin: vincire "to bind"



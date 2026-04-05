# TF Binding Prediction — CFG Course Project Jan 2026
Predicting binding probability of CTCF, REST, EP300 on chr3, chr10, chr17 (K562, hg38)
Final output: chr3/10/17_predictions.tsv.gz with probability scores per TF per 200bp bin

---

## Project Structure
project/
├── data/
│   ├── raw/
│   │   ├── FASTAs/          # chr{c}_200bp_bins.fa
│   │   ├── tsv/             # chr{c}_200bp_bins[_unknown].tsv
│   │   └── methylation/     # ENCFF660IHA.bed.gz (gemBS bed9+, CpG only)
│   └── processed/           # all precomputed .npy files go here
│       ├── chr{c}_methylation.npy   # shape: (num_bins,) float32
│       ├── chr{c}_pwm.npy           # shape: (num_bins, 3) float32 — COMING SOON
│       └── ...
├── models/
│   └── checkpoints/
│       ├── best_model.pt            # state dict of best model
│       └── best_config.json         # which feature config won
├── predictions/
│   ├── chr3_predictions.tsv.gz
│   ├── chr10_predictions.tsv.gz
│   └── chr17_predictions.tsv.gz
├── src/
│   ├── methylation_precompute.py    # ✅ DONE
│   ├── pwm_precompute.py            # 🔲 NEXT (after CNN)
│   ├── dataset.py                   # 🔲 TODO
│   ├── model.py                     # 🔲 TODO
│   ├── train.py                     # 🔲 TODO
│   └── predict.py                   # 🔲 TODO
└── main.py                          # 🔲 TODO

---

## Status

### ✅ DONE: methylation_precompute.py
- Source: ENCFF660IHA.bed.gz (ENCODE WGBS, K562, gemBS bed9+ CpG format)
- Columns used: col0=chrom, col1=start, col9=coverage, col10=meth_pct
- Filters CpG sites with coverage < 5
- For each 200bp bin: mean methylation fraction across all CpG sites in bin
- Vectorized with searchsorted — fast
- Output: data/processed/chr{c}_methylation.npy, shape (num_bins,), float32
- Skips already-processed chromosomes
- Run: python src/methylation_precompute.py

### ✅ DONE: Bug fixes to original CNN code
1. evaluate() called 3x per chromosome — fixed by saving preds from final epoch
2. Model reinit bug — fixed using state_dict() + copy.deepcopy()
   - Tracks best epoch within each chromosome
   - Tracks best model globally across all chromosomes and configs
   - Saves best_in_ch for model reconstruction at predict time
3. Model reconstruction before prediction loop:
   best_model = CNN(best_in_ch).to(DEVICE)
   best_model.load_state_dict(best_state_dict)

### 🔲 NEXT SESSION: CNN Overhaul (do this before PWM)
See detailed plan below.

### 🔲 AFTER CNN: pwm_precompute.py
- CTCF: MA0139.1 (confirmed working via JASPAR API)
- REST: MA0138.2 (to be confirmed)
- EP300: zeros (no DNA-binding domain — coactivator)
- Cell line context: K562
- For each bin: max PWM score across all positions in the 200bp window
- Output: data/processed/chr{c}_pwm.npy, shape (num_bins, 3), float32
  where col0=CTCF, col1=REST, col2=EP300(zeros)
- Requires: pip install biopython

---

## Full Planned Workflow

### STEP 1 — Precompute all features (run once)
python src/methylation_precompute.py   # ✅ done
python src/pwm_precompute.py           # 🔲 after CNN

### STEP 2 — dataset.py
Responsibilities:
- read_fasta(path) → list of sequences
- reverse_complement(seq)
- encode_batch(seqs, atac, meth, pwm, use_atac, use_meth, use_pwm)
  Input channels:
    0-3:  one-hot DNA          always on
    4:    ATAC scalar          optional
    5:    methylation scalar   optional
    6-8:  PWM scores (3 TFs)  optional (EP300 always zero)
- load_chromosome(c, test=False) → seqs, atac, meth, pwm, y, df

### STEP 3 — model.py
Planned architecture improvements:
- Dilated convolutions in ResBlock for longer-range context
- Configurable input channels (based on active features)
- FocalLoss stays (handles class imbalance well)
- DEVICE-aware throughout

### STEP 4 — train.py
Planned training improvements:
- GLOBAL training: pool all training chromosomes, don't train per-chromosome
- Hold out one chromosome for validation (e.g. chr1)
- More epochs (15-20) with early stopping (patience=3 on val ROC)
- CosineAnnealingLR scheduler
- Config-based model selection stays: DNA / DNA+ATAC / DNA+ATAC+METH / DNA+ATAC+METH+PWM
  → best config selected by val ROC, worst configs pruned
- Test-time augmentation: average predictions on fwd + reverse complement
- DEVICE-aware throughout

### STEP 5 — predict.py
- Load best_model.pt + best_config.json
- Predict on chr3, chr10, chr17
- TTA (fwd + RC average)
- Output: predictions/chr{c}_predictions.tsv.gz

### STEP 6 — main.py
- Calls precompute → train → predict in sequence
- Checks if processed files already exist before recomputing

---

## Key Constraints (from project brief)
- No public ENCODE-DREAM challenge code
- No other ChIP-seq/ChIP-exo/CUT&RUN data
- JASPAR PWMs allowed ✅
- K562 cell line, hg38 genome
- Test chromosomes: 3, 10, 17
- Train chromosomes: all others (1-22 except 3,10,17)
- Output must be probability scores (higher = more likely bound)

---

## Dependencies
torch, numpy, pandas, scikit-learn, matplotlib, biopython (for PWM)

## Device
CUDA-aware throughout model/train/predict. Precompute scripts are CPU-only (I/O bound).

---

## To Resume
Start a new Claude session and paste this block:

"Predicting TF binding (CTCF, REST, EP300) on 200bp bins, K562, hg38.
Methylation precompute done (ENCFF660IHA, per-bin mean CpG fraction, searchsorted).
Three CNN bugs fixed (evaluate caching, state_dict saving, model reconstruction).
Now building CNN overhaul: global training, dilated convolutions, early stopping,
LR scheduling, TTA, config-based feature selection.
File structure is clean (see README). Starting with model.py."


# TF Binding Prediction — CFG Course Project Jan 2026
Predicting binding probability of CTCF, REST, EP300 on chr3, chr10, chr17 (K562, hg38)
Final output: chr3/10/17_predictions.tsv.gz with probability scores per TF per 200bp bin

---

## Project Structure
project/
├── data/
│   ├── raw/
│   │   ├── FASTAs/          # chr{c}_200bp_bins.fa
│   │   ├── tsv/             # chr{c}_200bp_bins[_unknown].tsv
│   │   └── methylation/     # ENCFF660IHA.bed.gz (gemBS bed9+, CpG only)
│   └── processed/           # all precomputed .npy files go here
│       ├── chr{c}_methylation.npy   # shape: (num_bins,) float32
│       ├── chr{c}_pwm.npy           # shape: (num_bins, 3) float32
│       └── ...
├── models/
│   └── checkpoints/
│       ├── best_model.pt            # state dict of best model
│       └── best_config.json         # which feature config won
├── predictions/
│   ├── chr3_predictions.tsv.gz
│   ├── chr10_predictions.tsv.gz
│   └── chr17_predictions.tsv.gz
├── src/
│   ├── methylation_precompute.py    # ✅ DONE
│   ├── pwm_precompute.py            # ✅ DONE
│   ├── noGarbageIn.py               # 🔲 NEXT — dataset loading
│   ├── model.py                     # 🔲 TODO
│   ├── train.py                     # 🔲 TODO
│   ├── predict.py                   # 🔲 TODO
│   └── rna_precompute.py            # 🔲 TODO (post-core)
└── main.py                          # 🔲 TODO

---

## Status

### ✅ DONE: methylation_precompute.py
- Source: ENCFF660IHA.bed.gz (ENCODE WGBS, K562, gemBS bed9+ CpG format)
- Columns used: col0=chrom, col1=start, col9=coverage, col10=meth_pct
- Filters CpG sites with coverage < 5
- For each 200bp bin: mean methylation fraction across all CpG positions in bin
- Vectorized with searchsorted — fast
- Output: data/processed/chr{c}_methylation.npy, shape (num_bins,), float32
- Skips already-processed chromosomes
- Run: python src/methylation_precompute.py

### ✅ DONE: pwm_precompute.py
- CTCF: MA0139.1 (19bp motif, confirmed from JASPAR API)
- REST: MA0138.2 (21bp motif, confirmed from JASPAR API)
- EP300: zeros — coactivator, no DNA-binding domain, no motif assigned
- Scans both strands of each 200bp bin, records max log-odds score per bin
- Vectorized using numpy stride tricks (as_strided) — fast
- Background: uniform (0.25 per base), pseudocount=0.1
- Output: data/processed/chr{c}_pwm.npy, shape (num_bins, 3), float32
  col0=CTCF, col1=REST, col2=EP300(zeros)
- Skips already-processed chromosomes
- Run: python src/pwm_precompute.py

### ✅ DONE: Bug fixes to original CNN code
1. evaluate() called 3x per chromosome — fixed by caching preds from final epoch
2. Model reinit bug — fixed using state_dict() + copy.deepcopy()
   - Tracks best epoch within each chromosome
   - Tracks global best across all chromosomes and configs
3. Model reconstruction before prediction:
   best_model = CNN(best_in_ch).to(DEVICE)
   best_model.load_state_dict(best_state_dict)

---

## Architecture Decisions (key ideas, dense)

**Split strategy:** Chromosome-level 70/15/15 split (NOT random — nearby bins are correlated,
random splitting leaks signal). Train: chr1,2,4,5,6,7,8,9,11,12,13,14,15,16. Val: chr18,19,20.
Test: chr21,22. Predict: chr3,10,17 (no labels). Val selects best config, test gives honest AUC.

**Config-based model selection:** Four input configs trained independently — DNA only,
DNA+ATAC, DNA+ATAC+METH, DNA+ATAC+METH+PWM. Best config selected by val ROC-AUC.
This answers: which combination of features actually helps? If a feature doesn't contribute,
it gets pruned by the selection process.

**FIMO/ZOOPS decision:** Decided against. FIMO gives discrete hit/no-hit with p-value vs
our continuous max log-odds score — marginal gain for well-characterised motifs (CTCF, REST).
ZOOPS is a discovery model, not useful when motifs are already known. Revisit only if PWM
features don't improve ROC over DNA-only baseline.

**RNA-seq expression (planned, post-core):** K562 RNA-seq TPM from ENCODE mapped to nearest
gene TSS per bin → log TPM scalar input channel. Particularly useful for EP300 (no motif signal)
since EP300 binds active enhancers which correlate with nearby gene expression. Requires hg38
GTF + K562 RNA-seq TPM file (both ENCODE, both allowed by project rules).
Implemented in rna_precompute.py after core pipeline is working.

**Model:** 1D ResNet CNN with dilated convolutions for longer-range context, configurable
input channels, FocalLoss for class imbalance, CosineAnnealingLR, early stopping (patience=3).
Test-time augmentation: average predictions on forward + reverse complement strands.
CUDA-aware throughout model/train/predict. Precompute scripts are CPU-only (I/O bound).

---

## Input Feature Channels
| Channel | Source                | Shape per bin  | Status     |
|---------|-----------------------|----------------|------------|
| 0-3     | One-hot DNA           | (200, 4)       | ✅ done    |
| 4       | ATAC scalar           | (1,)           | ✅ done    |
| 5       | CpG methylation       | (1,)           | ✅ done    |
| 6       | CTCF PWM score        | (1,)           | ✅ done    |
| 7       | REST PWM score        | (1,)           | ✅ done    |
| 8       | EP300 PWM (zeros)     | (1,)           | ✅ done    |
| 9       | RNA-seq log TPM       | (1,)           | 🔲 planned |

---

## Full Planned Workflow

### STEP 1 — Precompute all features (run once)
python src/methylation_precompute.py   # ✅ done
python src/pwm_precompute.py           # ✅ done
python src/rna_precompute.py           # 🔲 post-core

### STEP 2 — noGarbageIn.py (dataset)
- read_fasta(path) → list of sequences
- reverse_complement(seq)
- encode_batch(seqs, atac, meth, pwm, use_atac, use_meth, use_pwm)
- load_chromosome(c, test=False) → seqs, atac, meth, pwm, y, df
- Chromosome-level split constants defined here:
    TRAIN_CHRS = [1,2,4,5,6,7,8,9,11,12,13,14,15,16]
    VAL_CHRS   = [18,19,20]
    TEST_CHRS  = [21,22]
    PRED_CHRS  = [3,10,17]

### STEP 3 — model.py
- ResBlock with dilated convolutions
- Configurable input channels
- FocalLoss

### STEP 4 — train.py
- Pool all train chromosomes, validate on VAL_CHRS, test on TEST_CHRS
- Config-based model selection
- Early stopping (patience=3), CosineAnnealingLR
- Test-time augmentation (fwd + RC average)
- Save best_model.pt + best_config.json

### STEP 5 — predict.py
- Load best_model.pt + best_config.json
- Predict on PRED_CHRS (chr3, chr10, chr17)
- Output: predictions/chr{c}_predictions.tsv.gz

### STEP 6 — main.py
- Orchestrates precompute → train → predict

---

## Key Constraints (from project brief)
- No public ENCODE-DREAM challenge code
- No other ChIP-seq/ChIP-exo/CUT&RUN data
- JASPAR PWMs ✅, RNA-seq ✅, Hi-C ✅, PWM databases ✅
- K562 cell line, hg38 genome
- Output must be probability scores (higher = more likely bound)

---

## Dependencies
torch, numpy, pandas, scikit-learn, matplotlib, biopython

## Device
CUDA-aware throughout model/train/predict. Precompute scripts are CPU-only (I/O bound).

---

## To Resume
Paste this into a new Claude session:

"Predicting TF binding (CTCF, REST, EP300) on 200bp bins, K562, hg38.
Done: methylation_precompute.py (CpG mean per bin, searchsorted vectorized),
pwm_precompute.py (CTCF MA0139.1, REST MA0138.2, EP300 zeros, stride-trick vectorized).
CNN bugs fixed (evaluate caching, state_dict, model reconstruction).
Split: chr-level 70/15/15 — train chr1,2,4,5,6,7,8,9,11,12,13,14,15,16 / val chr18,19,20 /
test chr21,22 / predict chr3,10,17.
Config selection: DNA / +ATAC / +METH / +PWM — best by val ROC.
Now writing noGarbageIn.py (dataset loading + encoding)."


# TF Binding Prediction — CFG Course Project Jan 2026
Predicting binding probability of CTCF, REST, EP300 on chr3, chr10, chr17 (K562, hg38)
Final output: chr3/10/17_predictions.tsv.gz with probability scores per TF per 200bp bin

---

## Project Structure
project/
├── data/
│   ├── raw/
│   │   ├── FASTAs/          # chr{c}_200bp_bins.fa
│   │   ├── tsv/             # chr{c}_200bp_bins[_unknown].tsv
│   │   └── methylation/     # ENCFF660IHA.bed.gz (gemBS bed9+, CpG only)
│   └── processed/           # all precomputed .npy files go here
│       ├── chr{c}_methylation.npy   # shape: (num_bins,) float32
│       └── chr{c}_pwm.npy           # shape: (num_bins, 3) float32
├── models/
│   └── checkpoints/
│       ├── best_model.pt            # state dict of best model
│       ├── best_config.json         # winning feature config
│       └── metrics.json             # per-epoch history + test preds for plots
├── plots/                           # output of plots.py
│   ├── val_roc_curves.png
│   ├── val_pr_curves.png
│   ├── test_roc_curves.png
│   └── test_prc_curves.png
├── predictions/
│   ├── chr3_predictions.tsv.gz
│   ├── chr10_predictions.tsv.gz
│   └── chr17_predictions.tsv.gz
├── src/
│   ├── methylation_precompute.py    # ✅ DONE
│   ├── pwm_precompute.py            # ✅ DONE
│   ├── noGarbageIn.py               # ✅ DONE
│   ├── model.py                     # ✅ DONE
│   ├── train.py                     # ✅ DONE (running)
│   ├── plots.py                     # ✅ DONE (run after train)
│   ├── predict.py                   # 🔲 TODO
│   └── rna_precompute.py            # 🔲 TODO (post-core)
└── main.py                          # 🔲 TODO

---

## Status

### ✅ DONE: methylation_precompute.py
- Source: ENCFF660IHA.bed.gz (ENCODE WGBS, K562, gemBS bed9+ CpG format)
- Columns: col0=chrom, col1=start, col9=coverage, col10=meth_pct
- Filters CpG sites coverage < 5, mean methylation per 200bp bin
- Vectorized with searchsorted
- Output: data/processed/chr{c}_methylation.npy, shape (num_bins,), float32

### ✅ DONE: pwm_precompute.py
- CTCF: MA0139.1 (19bp), REST: MA0138.2 (21bp), EP300: zeros
- Both strands scanned, max log-odds score per bin
- Vectorized via numpy stride tricks
- Output: data/processed/chr{c}_pwm.npy, shape (num_bins, 3), float32

### ✅ DONE: noGarbageIn.py
- Chromosome-level 70/15/15 split:
    TRAIN_CHRS = [1,2,4,5,6,7,8,9,11,12,13,14,15,16]
    VAL_CHRS   = [18,19,20]
    TEST_CHRS  = [21,22]
    PRED_CHRS  = [3,10,17]
- Vectorized ASCII lookup encoding
- load_chromosome(), load_split(), load_split_by_chr(), encode_batch()
- RC augmentation at load time (training only)
- Pandas fix: .apply(lambda col: col.map()) for compatibility

### ✅ DONE: model.py
- 1D ResNet CNN, 445k parameters, CUDA-aware
- Dilated ResBlocks: dilation 1→2→4
- ChannelAttention: 4-head MultiheadAttention after block3
- FocalLoss gamma=2 for class imbalance (~6% positive rate)
- Confirmed: device=cuda, (32,200,9)→(32,3)

### ✅ DONE: train.py
- Five configs trained independently:
    DNA / DNA+ATAC / DNA+ATAC+METH / DNA+ATAC+PWM / DNA+ATAC+METH+PWM
- Per-chromosome encoding during training (RAM safe, peak ~1-2GB)
- EPOCHS=20, PATIENCE=5 (not 3 epochs total — early stopping after
  5 epochs of no val ROC improvement, up to 20 max)
- CosineAnnealingLR, FocalLoss, TTA at eval (fwd + RC average)
- Saves: best_model.pt, best_config.json, metrics.json
- metrics.json contains per-epoch history and test predictions for plotting

### ✅ DONE: plots.py
- Reads metrics.json, generates four publication-style figures
- Fig 1: Val ROC-AUC per epoch per config (C-Origami style)
- Fig 2: Val PR-AUC per epoch per config
- Fig 3: Test ROC curves, one subplot per TF
- Fig 4: Test PRC curves, one subplot per TF, baseline shown
- Run after train.py: python src/plots.py

### 🔲 TODO: predict.py
### 🔲 TODO: rna_precompute.py (post-core)
### 🔲 TODO: main.py

---

## Architecture Decisions (dense)

**Split:** Chromosome-level 70/15/15 — random splits leak signal between
correlated nearby bins. Val selects best config, test gives honest AUC estimate,
chr3/10/17 are predict-only.

**Config selection:** Five configs trained independently, best by val ROC-AUC.
Answers: does ATAC help? Does methylation add over ATAC? Does PWM add?
Does PWM work without methylation? All answered by the plots.

**Memory:** Training encodes one chromosome at a time (~1-2GB peak),
frees immediately after. Val/test encoded fully per config (safe, ~300k seqs).
Prevented 19GB OOM from pre-encoding all training chromosomes.

**Early stopping:** PATIENCE=5 means model can plateau for up to 5 epochs
before stopping, giving time to escape local minima. EPOCHS=20 is the ceiling.

**FIMO/ZOOPS:** Not implemented — continuous max log-odds captures same signal
as discrete hit counts for well-characterised motifs. Revisit if PWM doesn't help.

**RNA-seq:** Planned post-core. K562 TPM → nearest TSS → log TPM scalar.
Particularly useful for EP300 (no motif signal).

**Hi-C:** Decided against. 200bp bins require ultra-deep Hi-C for reliable
contacts — resolution mismatch too severe. ATAC is a better proxy.

---

## Input Feature Channels
| Ch  | Source            | Shape     | Status     |
|-----|-------------------|-----------|------------|
| 0-3 | One-hot DNA       | (200, 4)  | ✅ done    |
| 4   | ATAC scalar       | (1,)      | ✅ done    |
| 5   | CpG methylation   | (1,)      | ✅ done    |
| 6   | CTCF PWM score    | (1,)      | ✅ done    |
| 7   | REST PWM score    | (1,)      | ✅ done    |
| 8   | EP300 PWM (zeros) | (1,)      | ✅ done    |
| 9   | RNA-seq log TPM   | (1,)      | 🔲 planned |

---

## Key Constraints
- No ENCODE-DREAM public code
- No ChIP-seq/ChIP-exo/CUT&RUN data
- JASPAR ✅, RNA-seq ✅, Hi-C ❌ (resolution mismatch)
- K562, hg38, output = probability scores

---

## Dependencies
torch, numpy, pandas, scikit-learn, matplotlib, biopython

## Device
CUDA (GTX 1650 4GB VRAM) for model/train/predict.
CPU only for precompute scripts.

---

## Resume Prompt
"Predicting TF binding (CTCF, REST, EP300), 200bp bins, K562, hg38.
Done: methylation_precompute, pwm_precompute, noGarbageIn, model, train, plots.
train.py running overnight — 5 configs, EPOCHS=20, PATIENCE=5.
Next: predict.py, then rna_precompute, then main.py.
GTX 1650 4GB, 15GB RAM, BATCH_SIZE=128."
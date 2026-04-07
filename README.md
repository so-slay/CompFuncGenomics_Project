## Computational Functional Genomics - Project

# VINCA: A CNN-based predictive model for TF-binding

Predicts transcription factor (TF) binding (`CTCF`, `REST`, `EP300`) using DNA sequence, ATAC-seq, CpG methylation, and PWM features.

---

## Features

- Multi-TF CNN model with sequence + ATAC + methylation + PWM input**
- Input encoding: (N × 200 × 9) tensor**
- Residual CNN + dilated convolutions + attention
- Reverse complement augmentation (training + inference)
- Chromosome-wise training (no leakage)
- Handles class imbalance using:
  - Negative sampling (`NEG_RATIO`)
  - Focal Loss
- Trains on chromosomes **1–22 (excluding test chromosomes)**
- Generates predictions for:
  - chr3
  - chr10
  - chr17
- Saves:
  - Prediction `.tsv.gz` files
  - Loss curve
  - Prediction distribution plots

---

## Dependencies

- Python 3.10+
- numpy
- pandas
- torch
- scikit-learn
- matplotlib
- tqdm


## Features
- Multi-TF CNN model with sequence + ATAC input  
- Cross-validation on chromosome 1  
- Final model training on chromosomes [1–22] 
- Generates predictions for specific chromosomes (3, 10, 17)  
- Saves ROC curves and AUC results  

---
## Running the Model

### 1. Full Run (Train → Select Best Model → Predict)
Execute the main pipeline:

```python
python3 main.py
```
### 2. Fast Run (Prediction Only)

Use this mode when confident about required inputs and/or if you want to skip model selection.

**Steps:**

1. Clear the checkpoints directory:

```bash
rm -r models/checkpoints/*
```

```python
python3 predict.py
```

## Detailed info on VINCA: 
See the model [Overview](Overview.md)


---
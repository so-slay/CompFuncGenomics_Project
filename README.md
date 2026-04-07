# CNN-based TF Binding Prediction with ATAC-seq

# CNN-based TF Binding Prediction with ATAC + Epigenomics

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

Install:

```bash
pip install numpy pandas torch scikit-learn matplotlib tqdm
---

## Features
- Multi-TF CNN model with sequence + ATAC input  
- Cross-validation on chromosome 1  
- Final model training on chromosomes 1–22  
- Generates predictions for specific chromosomes (3, 10, 17)  
- Saves ROC curves and AUC results  

---

## Dependencies
- Python 3.10
- numpy  
- pandas  
- torch  
- scikit-learn  
- matplotlib  

---

## Project Structure
.
├── model.py           # CNN + residual + dilated conv + attention
├── noGarbageIn.py     # Data loading + encoding pipeline
├── predict.py         # Training + inference script
├── data/
│   ├── chr*_200bp_bins.fa
│   ├── chr*_200bp_bins.tsv
│   ├── *.npy (methylation + PWM)
├── outputs/
│   ├── chr3_predictions.tsv.gz
│   ├── chr10_predictions.tsv.gz
│   ├── chr17_predictions.tsv.gz
│   ├── loss_curve.png
│   ├── prediction_histograms.png



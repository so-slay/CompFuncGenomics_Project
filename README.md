# CNN-based TF Binding Prediction with ATAC-seq

Predicts transcription factor (TF) binding (`CTCF`, `REST`, `EP300`) using DNA sequence and ATAC-seq data.

---

## Features
- Multi-TF CNN model with sequence + ATAC input  
- Cross-validation on chromosome 1  
- Final model training on chromosomes 1–22  
- Generates predictions for specific chromosomes (3, 10, 17)  
- Saves ROC curves and AUC results  

---

## Dependencies
- Python ≥ 3.8  
- numpy  
- pandas  
- torch  
- scikit-learn  
- matplotlib  

---

## Project Structure

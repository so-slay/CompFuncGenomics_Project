# DNA + Genomic Feature Encoder

## Overview

The `encode` function transforms DNA sequences and associated genomic features into a numerical format suitable for convolutional neural networks (CNNs). It is designed to be flexible and biologically interpretable, supporting both **per-base** and **global** features.  

The main purpose is to prepare input for models predicting transcription factor (TF) binding, epigenetic states, or other genomic properties.

---

## Features

- **DNA One-Hot Encoding**:  
  Each nucleotide is encoded as a 4-dimensional one-hot vector (A, C, G, T). This allows convolutional kernels to learn motif patterns directly from sequence.

- **Optional Features**:  
  Additional genomic features can be included, such as:
  - `ATAC`: Chromatin accessibility (per-base or global)
  - `METH`: DNA methylation (per-base or global)
  - Future features: ChIP-seq signals, conservation scores, etc.

- **Flexible Feature Selection**:  
  Users can specify which features to include via a `feature_list`. The order of features is preserved, which is critical for reproducibility and consistency when initializing motif-based convolutional filters.

- **Global Feature Broadcasting**:  
  Scalar features (e.g., average accessibility over the region) are broadcast across the sequence to be compatible with the per-base CNN input.

---

## Usage

```python
# DNA only
enc = encode("ACGTACGT", {}, ["DNA"])

# DNA + global ATAC
enc = encode("ACGTACGT", {"ATAC":1}, ["DNA","ATAC"])

# DNA + per-base ATAC + per-base methylation
atac = [1,0,1,0,1,0,1,0]
meth = [0,1,0,1,0,1,0,1]
enc = encode("ACGTACGT", {"ATAC":atac,"METH":meth}, ["DNA","ATAC","METH"])
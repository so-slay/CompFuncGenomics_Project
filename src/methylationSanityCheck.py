import numpy as np

m = np.load("data/processed/chr1_methylation.npy")
print(m.shape)           # (num_bins,)
print(m.max())           # should be <= 1.0
print(m.min())           # should be >= 0.0
print((m > 0).mean())    # fraction of bins with CpG — expect 0.3 to 0.6

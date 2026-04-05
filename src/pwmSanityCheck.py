import numpy as np
p = np.load("data/processed/chr1_pwm.npy")
print(p.shape)        # (num_bins, 3)
print(p[:,0].max())   # CTCF — expect positive values ~10-20 for strong hits
print(p[:,1].max())   # REST — similar
print(p[:,2].max())   # EP300 — must be 0.0
print((p[:,0] > 5).mean())  # fraction of bins with strong CTCF motif
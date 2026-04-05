import torch
import numpy as np
import matplotlib.pyplot as plt
from finalsub import CNN   # <-- your model file
from finalsub import encode_batch, load_chr  # reuse your functions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- HOOK CLASS --------
class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = {}

        # Register hooks
        self.model.block1.conv1.register_forward_hook(self.save_output("block1_conv1"))
        self.model.block2.conv1.register_forward_hook(self.save_output("block2_conv1"))
        self.model.block3.conv1.register_forward_hook(self.save_output("block3_conv1"))

    def save_output(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach().cpu()
        return hook


# -------- LOAD MODEL --------
def load_model(path, in_channels):
    model = CNN(in_channels)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# -------- VISUALIZE --------
def plot_feature_maps(feature_tensor, title, max_channels=6):
    """
    feature_tensor shape: (1, C, L)
    """
    feature_tensor = feature_tensor[0]  # remove batch dim
    C, L = feature_tensor.shape

    plt.figure(figsize=(12, 8))

    for i in range(min(C, max_channels)):
        plt.subplot(max_channels, 1, i + 1)
        plt.plot(feature_tensor[i])
        plt.title(f"{title} - Channel {i}")

    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.close()


# -------- MAIN --------
def main():

    # ---- CONFIG ----
    MODEL_PATH = "best_model.pth"   # <-- change this
    CHR = 1
    SAMPLE_IDX = 0

    use_atac = True
    use_meth = False

    # ---- LOAD DATA ----
    seqs, atac, meth, y, _ = load_chr(CHR)

    seq = [seqs[SAMPLE_IDX]]
    atac_sample = atac[SAMPLE_IDX:SAMPLE_IDX+1]

    if use_meth and meth is not None:
        meth_sample = meth[SAMPLE_IDX:SAMPLE_IDX+1]
    else:
        meth_sample = None

    X = encode_batch(seq, atac_sample, meth_sample, use_atac, use_meth)

    X = torch.tensor(X).to(DEVICE)

    # ---- LOAD MODEL ----
    in_channels = X.shape[2]
    model = load_model(MODEL_PATH, in_channels)

    # ---- EXTRACT FEATURES ----
    extractor = FeatureExtractor(model)

    with torch.no_grad():
        _ = model(X)

    # ---- PLOT ----
    for name, feat in extractor.features.items():
        plot_feature_maps(feat, name)


if __name__ == "__main__":
    main()
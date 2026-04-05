"""
model.py

1D ResNet CNN with dilated convolutions for TF binding prediction.
Takes (N, 200, channels) input, outputs (N, 3) logits for CTCF, REST, EP300.
Dilated convolutions expand receptive field without increasing parameters.
Optional self-attention layer after CNN blocks captures long-range dependencies.
FocalLoss handles severe class imbalance (bound sites are rare).
Fully CUDA-aware — call model.to(DEVICE) after instantiation.

Input:  (N, 200, channels) float32 tensor
Output: (N, 3) float32 logits — pass through sigmoid for probabilities

Usage: imported by train.py and predict.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- FOCAL LOSS ----------------
class FocalLoss(nn.Module):
    """
    Focal loss downweights easy negatives, focuses training on hard examples.
    Critical here because bound bins are ~5-10% of data for most TFs.
    gamma=2 is standard; higher gamma = more focus on hard examples.
    """
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt  = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


# ---------------- RESIDUAL BLOCK ----------------
class ResBlock(nn.Module):
    """
    1D residual block with two convolutions and a skip connection.
    First conv uses standard kernel, second uses dilated kernel to expand
    receptive field. Dilation=1 means no dilation (standard convolution).
    BatchNorm + ReLU after each conv, skip connection handles channel mismatch.
    """
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel_size=7,
                               padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=5,
                               padding=2 * dilation, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.skip  = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


# ---------------- ATTENTION ----------------
class ChannelAttention(nn.Module):
    """
    Single-head self-attention over sequence positions.
    Applied after CNN blocks to capture long-range dependencies within the bin.
    At L=200 the O(L^2) cost is trivial (~40k operations).
    Particularly helps REST whose motif has complex multi-part structure.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,      # expects (N, L, channels)
            dropout=0.1
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: (N, channels, L) from CNN — transpose for attention
        x_t = x.permute(0, 2, 1)            # (N, L, channels)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        out = self.norm(x_t + attn_out)      # residual + norm
        return out.permute(0, 2, 1)          # back to (N, channels, L)


# ---------------- MODEL ----------------
class CNN(nn.Module):
    """
    Full model: three ResBlocks with increasing dilation + optional attention
    + global max pooling + classifier head.

    Architecture:
        Input (N, 200, in_ch)
        → permute to (N, in_ch, 200)          [CNN expects channels first]
        → ResBlock(in_ch → 64,  dilation=1)   [local motif detection]
        → MaxPool(2) → (N, 64, 100)
        → ResBlock(64 → 128, dilation=2)      [medium-range context]
        → MaxPool(2) → (N, 128, 50)
        → ResBlock(128 → 128, dilation=4)     [longer-range context]
        → ChannelAttention(128)               [global dependencies]
        → AdaptiveMaxPool → (N, 128)
        → FC(128 → 64) + Dropout(0.5)
        → FC(64 → 3)                          [one logit per TF]

    Args:
        in_ch      : number of input channels (9 for full feature set)
        use_attn   : whether to include attention layer (default True)
    """
    def __init__(self, in_ch, use_attn=True):
        super().__init__()
        self.block1  = ResBlock(in_ch, 64,  dilation=1)
        self.block2  = ResBlock(64,    128, dilation=2)
        self.block3  = ResBlock(128,   128, dilation=4)
        self.pool    = nn.MaxPool1d(2)
        self.gap     = nn.AdaptiveMaxPool1d(1)
        self.use_attn = use_attn
        if use_attn:
            self.attn = ChannelAttention(128, num_heads=4)
        self.fc1     = nn.Linear(128, 64)
        self.drop    = nn.Dropout(0.5)
        self.fc2     = nn.Linear(64, 3)

    def forward(self, x):
        x = x.permute(0, 2, 1)              # (N, 200, ch) → (N, ch, 200)
        x = self.pool(self.block1(x))        # (N, 64,  100)
        x = self.pool(self.block2(x))        # (N, 128,  50)
        x = self.block3(x)                   # (N, 128,  50)
        if self.use_attn:
            x = self.attn(x)                 # (N, 128,  50)
        x = self.gap(x).squeeze(-1)          # (N, 128)
        x = self.drop(F.relu(self.fc1(x)))   # (N, 64)
        return self.fc2(x)                   # (N, 3)


# ---------------- SANITY CHECK ----------------
if __name__ == "__main__":
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    model = CNN(in_ch=9, use_attn=True).to(DEVICE)

    # count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # forward pass
    x = torch.randn(32, 200, 9).to(DEVICE)
    out = model(x)
    print(f"Input  shape: {x.shape}")     # (32, 200, 9)
    print(f"Output shape: {out.shape}")   # (32, 3)
    print("model.py sanity check passed.")
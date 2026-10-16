"""
Step 6: Scaled Dot-Product Self-Attention (single head).

Input:  (batch, seq_len, embed_dim)  = (32, 256, 384)
Output: (batch, seq_len, embed_dim)  = (32, 256, 384)

The output has the same shape as the input, so attention can be stacked.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Single-head causal self-attention.

    Every token produces a Query, Key, and Value vector via three learned
    linear projections. Attention weights are computed from Q @ K^T and
    applied to V.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        # Three projections of the SAME input x.
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

        # Note: no softmax stored here — we call F.softmax directly.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, C)  where B=batch, T=seq_len, C=embed_dim
        B, T, C = x.shape

        # ---- 1. Project to Q, K, V --------------------------------------
        q = self.W_q(x)   # (B, T, C)
        k = self.W_k(x)   # (B, T, C)
        v = self.W_v(x)   # (B, T, C)

        # ---- 2. Attention scores: Q @ K^T -------------------------------
        # k.transpose(-2, -1) swaps last two dims -> (B, C, T)
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        scores = q @ k.transpose(-2, -1)   # (B, T, T)

        # ---- 3. Scale by 1/sqrt(d_k) -----------------------------------
        # d_k == C for single-head attention
        scores = scores / (C ** 0.5)       # (B, T, T)

        # ---- 4. Causal mask --------------------------------------------
        # Lower-triangular matrix of ones: mask[i, j] = 1 if j <= i else 0
        # We set masked positions to -inf so softmax gives them 0 weight.
        mask = torch.tril(torch.ones(T, T, device=x.device))   # (T, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # ---- 5. Softmax over the last dim ------------------------------
        attn = F.softmax(scores, dim=-1)   # (B, T, T)

        # ---- 6. Weighted sum with V ------------------------------------
        # (B, T, T) @ (B, T, C) -> (B, T, C)
        out = attn @ v                     # (B, T, C)

        return out


# ---------------------------------------------------------------------
# Demo: run on a small tensor first, then the real (32, 256, 384) batch
# ---------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    B, T, C = 2, 4, 8   # small for readability

    x = torch.randn(B, T, C)
    print(f"Input shape: {tuple(x.shape)}")

    attn_layer = SelfAttention(embed_dim=C)
    out = attn_layer(x)
    print(f"Output shape: {tuple(out.shape)}")

    # ---- Inspect the attention matrix for batch 0 ------------------
    # Recompute manually to inspect intermediate values
    q = attn_layer.W_q(x)
    k = attn_layer.W_k(x)
    scores = q @ k.transpose(-2, -1) / (C ** 0.5)
    mask = torch.tril(torch.ones(T, T))
    scores_masked = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores_masked, dim=-1)

    print(f"\nAttention matrix, batch 0, shape {tuple(attn[0].shape)}:")
    print(attn[0])

    print(f"\nRow sums (should all be 1.0):")
    print(attn[0].sum(dim=-1))

    # ---- Realistic shapes ------------------------------------------
    print("\n" + "=" * 60)
    print("Realistic shapes: (32, 256, 384)")
    print("=" * 60)

    B, T, C = 32, 256, 384
    x_big = torch.randn(B, T, C)
    attn_big = SelfAttention(embed_dim=C)
    out_big = attn_big(x_big)

    print(f"Input:  {tuple(x_big.shape)}")
    print(f"Output: {tuple(out_big.shape)}")

    n_params = sum(p.numel() for p in attn_big.parameters())
    print(f"Params in this attention layer: {n_params:,}")

    # ---- Leakage test ----------------------------------------------
    print("\n" + "=" * 60)
    print("Causal leakage test")
    print("=" * 60)

    B, T, C = 1, 6, 8
    x_a = torch.randn(B, T, C)
    x_b = x_a.clone()

    # Modify ONLY the last token (position 5). Earlier positions must
    # produce identical outputs, because they can't attend to the future.
    x_b[0, 5] = torch.randn(C)

    layer = SelfAttention(embed_dim=C)
    out_a = layer(x_a)
    out_b = layer(x_b)

    # Positions 0..4 should be identical
    same_early = torch.allclose(out_a[0, :5], out_b[0, :5], atol=1e-6)
    # Position 5 should differ (it changed and it can attend to itself)
    different_last = not torch.allclose(out_a[0, 5], out_b[0, 5], atol=1e-6)

    print(f"Early positions unchanged after modifying future: {same_early}")
    print(f"Last position changed:                            {different_last}")
    assert same_early, "LEAKAGE: future token affected earlier outputs!"
    assert different_last, "Something is wrong: last token should have changed."
    print("Causality verified. No leakage.")
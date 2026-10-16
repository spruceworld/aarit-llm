"""
Step 7: Multi-Head Causal Self-Attention.

Splits embed_dim into h heads of size head_dim = embed_dim // h.
Each head does Step 6 attention independently, then we concatenate
and apply a final output projection.

Input:  (B, T, C)  e.g. (32, 256, 384)
Output: (B, T, C)  same shape — attention preserves shape
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.embed_dim = embed_dim
        self.n_heads   = n_heads
        self.head_dim  = embed_dim // n_heads

        # One combined projection for Q, K, V (efficient form).
        # Output shape: (B, T, 3 * embed_dim)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)

        # Output projection: mixes information across heads.
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        h, d    = self.n_heads, self.head_dim

        # ---- 1. Project to Q, K, V for all heads at once --------------
        qkv = self.qkv_proj(x)              # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)      # each: (B, T, C)

        # ---- 2. Split into heads and reorder --------------------------
        # (B, T, C) -> (B, T, h, d) -> (B, h, T, d)
        q = q.view(B, T, h, d).transpose(1, 2)   # (B, h, T, d)
        k = k.view(B, T, h, d).transpose(1, 2)   # (B, h, T, d)
        v = v.view(B, T, h, d).transpose(1, 2)   # (B, h, T, d)

        # ---- 3. Scaled dot-product attention per head -----------------
        # scores: (B, h, T, T)
        scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)

        # ---- 4. Causal mask -------------------------------------------
        mask = torch.tril(torch.ones(T, T, device=x.device))   # (T, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # ---- 5. Softmax + dropout -------------------------------------
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # ---- 6. Weighted sum with V -----------------------------------
        # (B, h, T, T) @ (B, h, T, d) -> (B, h, T, d)
        out = attn @ v

        # ---- 7. Merge heads -------------------------------------------
        # (B, h, T, d) -> (B, T, h, d) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # ---- 8. Output projection -------------------------------------
        out = self.out_proj(out)
        return out


# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    # ---- Small example --------------------------------------------
    B, T, C = 2, 4, 8
    n_heads = 2

    x = torch.randn(B, T, C)
    print(f"Input: {tuple(x.shape)}")

    mha = MultiHeadAttention(embed_dim=C, n_heads=n_heads)
    out = mha(x)
    print(f"Output: {tuple(out.shape)}  (same shape)")

    n_params = sum(p.numel() for p in mha.parameters())
    print(f"Params: {n_params:,}")

    # Expected: qkv_proj = 3 * C * C, out_proj = C * C, total = 4 * C * C
    expected = 4 * C * C
    print(f"Expected: {expected:,}  Match: {n_params == expected}")

    # ---- Leakage test -------------------------------------------
    print("\n" + "=" * 60)
    print("Causal leakage test")
    print("=" * 60)

    B, T, C = 1, 8, 16
    x_a = torch.randn(B, T, C)
    x_b = x_a.clone()
    x_b[0, -1] = torch.randn(C)   # change ONLY the last token

    layer = MultiHeadAttention(embed_dim=C, n_heads=4)
    out_a = layer(x_a)
    out_b = layer(x_b)

    same_early   = torch.allclose(out_a[0, :-1], out_b[0, :-1], atol=1e-6)
    diff_last    = not torch.allclose(out_a[0, -1], out_b[0, -1], atol=1e-6)
    print(f"Early positions unchanged: {same_early}")
    print(f"Last position changed:     {diff_last}")
    assert same_early, "LEAKAGE DETECTED"
    assert diff_last, "Last token should have changed"
    print("Causality verified.")

    # ---- Realistic shapes ---------------------------------------
    print("\n" + "=" * 60)
    print("Realistic shapes: (32, 256, 384), 6 heads")
    print("=" * 60)

    B, T, C = 32, 256, 384
    n_heads = 6
    head_dim = C // n_heads
    print(f"head_dim = {head_dim}")

    x_big = torch.randn(B, T, C)
    mha_big = MultiHeadAttention(embed_dim=C, n_heads=n_heads)
    out_big = mha_big(x_big)

    print(f"Input:  {tuple(x_big.shape)}")
    print(f"Output: {tuple(out_big.shape)}")

    n_params = sum(p.numel() for p in mha_big.parameters())
    print(f"Params: {n_params:,}")
    print(f"Breakdown: qkv_proj={3*C*C:,}  out_proj={C*C:,}")
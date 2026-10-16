"""
Step 8: Feed-Forward MLP, LayerNorm, and residual connections.

These three pieces, combined with attention from Step 7, form a
complete Transformer block (Step 9 will assemble them).

Shapes throughout: (B, T, C) = (batch, seq_len, embed_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. Feed-Forward MLP
# =====================================================================
class FeedForward(nn.Module):
    """
    Standard 2-layer MLP: Linear(C -> 4C) -> GELU -> Linear(4C -> C).

    This is where the model's 'knowledge' is mostly stored.
    Operates on each token independently — no mixing across positions.
    """

    def __init__(self, embed_dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = embed_dim * expansion

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = self.fc1(x)          # (B, T, 4C)
        x = self.act(x)          # (B, T, 4C)
        x = self.fc2(x)          # (B, T, C)
        x = self.dropout(x)
        return x


# =====================================================================
# 2. Complete Transformer Block (Pre-LN style)
# =====================================================================
class TransformerBlock(nn.Module):
    """
    One full Transformer block, Pre-LN layout:

        x = x + Attn(LN(x))
        x = x + FFN(LN(x))

    (Step 9 will use this. We define it here so you can test the
    full block before moving on.)
    """

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        # Import the MHA from step 7
        from step7_multihead import MultiHeadAttention

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout=dropout)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, expansion=4, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sublayer with residual
        x = x + self.attn(self.ln1(x))

        # FFN sublayer with residual
        x = x + self.ffn(self.ln2(x))
        return x


# =====================================================================
# 3. Demo + verification
# =====================================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    # ---- Small example --------------------------------------------
    B, T, C = 2, 8, 16
    print("=" * 60)
    print("FFN shape check")
    print("=" * 60)
    ffn = FeedForward(embed_dim=C, expansion=4)
    x = torch.randn(B, T, C)
    out = ffn(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print(f"FFN params: {sum(p.numel() for p in ffn.parameters()):,}")
    print(f"Expected:   {2 * C * 4 * C:,}")

    # ---- Residual identity check ----------------------------------
    print("\n" + "=" * 60)
    print("Residual identity check")
    print("=" * 60)
    # With a residual, x is preserved in the output. If we zero out the
    # sublayer's output, output should equal input.
    x = torch.randn(B, T, C)
    y = x + torch.zeros_like(x)
    print(f"x + 0 == x: {torch.allclose(y, x)}")

    # ---- Full Transformer block -----------------------------------
    print("\n" + "=" * 60)
    print("Full Transformer block")
    print("=" * 60)
    C = 32
    block = TransformerBlock(embed_dim=C, n_heads=4)
    x = torch.randn(B, T, C)
    out = block(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")
    n_params = sum(p.numel() for p in block.parameters())
    print(f"Block params: {n_params:,}")
    # Expected: attn (4*C^2) + ffn (2 * 4 * C^2) + 4*C (two LayerNorms)
    expected = 4 * C * C + 8 * C * C + 4 * C
    print(f"Expected:     {expected:,}  Match: {n_params == expected}")

    # ---- Leakage test at the block level --------------------------
    print("\n" + "=" * 60)
    print("Causal leakage test (through the whole block)")
    print("=" * 60)
    B, T, C = 1, 8, 16
    block = TransformerBlock(embed_dim=C, n_heads=4)
    x_a = torch.randn(B, T, C)
    x_b = x_a.clone()
    x_b[0, -1] = torch.randn(C)

    out_a = block(x_a)
    out_b = block(x_b)

    same_early = torch.allclose(out_a[0, :-1], out_b[0, :-1], atol=1e-6)
    diff_last  = not torch.allclose(out_a[0, -1], out_b[0, -1], atol=1e-6)
    print(f"Early positions unchanged: {same_early}")
    print(f"Last position changed:     {diff_last}")
    assert same_early, "LEAKAGE — check masking in attention"
    assert diff_last, "Last token should have changed"
    print("Causality verified.")

    # ---- Realistic shapes --------------------------------------
    print("\n" + "=" * 60)
    print("Realistic: (32, 256, 384), 6 heads, full block")
    print("=" * 60)
    B, T, C = 32, 256, 384
    block = TransformerBlock(embed_dim=C, n_heads=6)
    x = torch.randn(B, T, C)
    out = block(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print(f"Block params: {sum(p.numel() for p in block.parameters()):,}")
    print(f"  attention:  {sum(p.numel() for p in block.attn.parameters()):,}")
    print(f"  FFN:        {sum(p.numel() for p in block.ffn.parameters()):,}")
    print(f"  2x LN:      {sum(p.numel() for p in block.ln1.parameters()) + sum(p.numel() for p in block.ln2.parameters()):,}")
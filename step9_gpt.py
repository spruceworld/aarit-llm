"""
Step 9: The Complete GPT Model.

Combines:
  - Token & Position Embeddings (Step 5)
  - N x Transformer Blocks (Step 8, which uses Step 7 attention)
  - Final LayerNorm
  - Output Head (LM Head) to produce logits
"""

import torch
import torch.nn as nn
from step8_ffn_layernorm import TransformerBlock


class GPTConfig:
    """Holds all hyperparameters in one place."""

    def __init__(self, vocab_size, context_window, embed_dim, n_heads, n_layers, dropout=0.0):
        self.vocab_size = vocab_size
        self.context_window = context_window
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # 1. Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.context_window, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

        # 2. Transformer Blocks (N layers)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.n_heads, dropout=config.dropout)
            for _ in range(config.n_layers)
        ])

        # 3. Final LayerNorm
        self.ln_f = nn.LayerNorm(config.embed_dim)

        # 4. Output Head (LM Head)
        # Maps embed_dim -> vocab_size to predict next token logits.
        # bias=False because we won't use it.
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        idx: (B, T) tensor of token IDs
        targets: (B, T) tensor of target token IDs (optional, for loss)
        """
        B, T = idx.shape
        assert T <= self.config.context_window, f"Sequence length {T} exceeds context window {self.config.context_window}"

        # 1. Get embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, C)
        pos_ids = torch.arange(T, device=idx.device)  # (T,)
        pos_emb = self.position_embedding(pos_ids)  # (T, C)

        x = self.dropout(tok_emb + pos_emb)  # (B, T, C)

        # 2. Pass through all Transformer blocks
        for block in self.blocks:
            x = block(x)  # (B, T, C)

        # 3. Final LayerNorm
        x = self.ln_f(x)  # (B, T, C)

        # 4. Project to vocabulary size
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # 5. Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten the batch and sequence dimensions for CrossEntropy
            # logits: (B*T, vocab_size), targets: (B*T)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss


# =====================================================================
# Demo
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Building the GPT Model")
    print("=" * 60)

    # Define our configuration
    config = GPTConfig(
        vocab_size=50257,
        context_window=256,
        embed_dim=384,
        n_heads=6,
        n_layers=6,
        dropout=0.1
    )

    model = GPT(config)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
    print(f"  Token Emb:      {config.vocab_size * config.embed_dim:,}")
    print(f"  Position Emb:   {config.context_window * config.embed_dim:,}")
    print(f"  Transformer (x{config.n_layers}): {sum(p.numel() for p in model.blocks.parameters()):,}")
    print(
        f"  Final LN + Head:{sum(p.numel() for p in model.ln_f.parameters()) + sum(p.numel() for p in model.lm_head.parameters()):,}")

    # Create dummy batch
    B, T = 2, 64  # Small batch and sequence for quick test
    dummy_x = torch.randint(0, config.vocab_size, (B, T))
    dummy_y = torch.randint(0, config.vocab_size, (B, T))

    print(f"\nRunning forward pass with dummy data...")
    print(f"Input shape:  {tuple(dummy_x.shape)}")

    logits, loss = model(dummy_x, dummy_y)

    print(f"Logits shape: {tuple(logits.shape)}  (B, T, vocab_size)")
    print(f"Loss:         {loss.item():.4f}")
    print("Model built successfully!")
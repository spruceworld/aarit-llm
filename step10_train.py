"""
Step 10: The Training Loop + Generation.

Loads data from Step 4, builds the model from Step 9, trains it,
saves checkpoints, and generates text.
"""

import torch
import numpy as np
import time
import os
import tiktoken
from step9_gpt import GPT, GPTConfig

# =====================================================================
# 1. Configuration
# =====================================================================
torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device.upper()}")

# Model config (matches Step 9)
config = GPTConfig(
    vocab_size=50257,
    context_window=256,
    embed_dim=384,
    n_heads=6,
    n_layers=6,
    dropout=0.1
)

# Training config
BATCH_SIZE = 16  # Drop from 32 to 16 to save RAM on CPU
LEARNING_RATE = 3e-4  # Standard for small GPTs
MAX_STEPS = 5  # Stop after 500 batches (upgrade to 5000+ later)
EVAL_INTERVAL = 50  # Evaluate on val set every 50 steps
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =====================================================================
# 2. Load Data (from Step 4)
# =====================================================================
train_ids = np.load("train_ids.npy")
val_ids = np.load("val_ids.npy")


def get_batch(data: np.ndarray, split: str):
    """Grab a random batch of (x, y) pairs."""
    max_start = len(data) - config.context_window - 1
    starts = np.random.randint(0, max_start, size=BATCH_SIZE)
    x = np.stack([data[s: s + config.context_window] for s in starts])
    y = np.stack([data[s + 1: s + config.context_window + 1] for s in starts])

    x_t = torch.from_numpy(x.astype(np.int64)).to(device)
    y_t = torch.from_numpy(y.astype(np.int64)).to(device)
    return x_t, y_t


# =====================================================================
# 3. Initialize Model & Optimizer
# =====================================================================
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# =====================================================================
# 4. Text Generation Function
# =====================================================================
enc = tiktoken.get_encoding("gpt2")


@torch.no_grad()
def generate(model, prompt: str, max_new_tokens: int = 100, temperature: float = 1.0):
    """Generate text autoregressively from a prompt."""
    model.eval()

    # Encode prompt -> tensor
    idx = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        # Crop context to context_window
        idx_cond = idx[:, -config.context_window:]

        # Forward pass
        logits, _ = model(idx_cond)

        # Focus only on the last position (we want the next token)
        logits = logits[:, -1, :] / temperature

        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1)

        # Append to sequence
        idx = torch.cat([idx, next_token], dim=1)

    model.train()
    return enc.decode(idx[0].tolist())


# =====================================================================
# 5. Training Loop
# =====================================================================
print("\nStarting training...")
model.train()
best_val_loss = float('inf')

for step in range(MAX_STEPS):
    t0 = time.time()

    # --- 1. Forward pass ---
    xb, yb = get_batch(train_ids, "train")
    logits, loss = model(xb, yb)

    # --- 2. Backward pass ---
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # --- 3. Gradient clipping ---
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # --- 4. Update weights ---
    optimizer.step()

    t1 = time.time()

    # --- Print progress ---
    if step % 10 == 0:
        dt = (t1 - t0) * 1000
        print(f"Step {step:4d} | Loss: {loss.item():.4f} | Time: {dt:.0f}ms")

    # --- Evaluate ---
    if step > 0 and step % EVAL_INTERVAL == 0:
        model.eval()
        with torch.no_grad():
            val_losses = []
            for _ in range(10):  # Average over 10 batches
                xv, yv = get_batch(val_ids, "val")
                _, v_loss = model(xv, yv)
                val_losses.append(v_loss.item())
            avg_val_loss = sum(val_losses) / len(val_losses)

        print(f"\n--- EVAL | Step {step} | Val Loss: {avg_val_loss:.4f} ---")

        # Generate sample text
        sample = generate(model, "ROMEO: ", max_new_tokens=50, temperature=0.8)
        print(f"Sample:\n{sample}\n" + "-" * 40)

        # Save checkpoint if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            print(f"Saved best checkpoint to {CHECKPOINT_DIR}/best_model.pt\n")

        model.train()

print("\nTraining finished!")
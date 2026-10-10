"""
Step 4: Sliding-window dataset + DataLoader.

Turns the flat token stream into (input, target) training pairs,
then batches them for the training loop.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------
# 1. The Dataset class
# ---------------------------------------------------------------
class ShakespeareDataset(Dataset):
    """
    A PyTorch Dataset that yields fixed-length (input, target) pairs
    from a flat array of token IDs using a sliding window.
    """

    def __init__(self, token_ids: np.ndarray, context_window: int, stride: int | None = None):
        self.data = torch.from_numpy(token_ids.astype(np.int64))
        self.context_window = context_window
        self.stride = stride if stride is not None else context_window

        # How many windows fit in the data?
        self.n_windows = (len(self.data) - context_window - 1) // self.stride + 1

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int):
        start = idx * self.stride
        end = start + self.context_window

        x = self.data[start:end]          # input:  [t0 ... t_{n-1}]
        y = self.data[start + 1:end + 1]  # target: [t1 ... t_n]

        return x, y


# ---------------------------------------------------------------
# 2. Load the tokenized corpus we saved in Step 3
# ---------------------------------------------------------------
train_ids = np.load("train_ids.npy")
val_ids   = np.load("val_ids.npy")

print(f"Train tokens: {len(train_ids):,}")
print(f"Val tokens:   {len(val_ids):,}")


# ---------------------------------------------------------------
# 3. Create Dataset objects
# ---------------------------------------------------------------
CONTEXT_WINDOW = 256
BATCH_SIZE     = 32

train_ds = ShakespeareDataset(train_ids, CONTEXT_WINDOW)
val_ds   = ShakespeareDataset(val_ids,   CONTEXT_WINDOW)

print(f"\nTrain windows: {len(train_ds):,}")
print(f"Val windows:   {len(val_ds):,}")


# ---------------------------------------------------------------
# 4. Sanity check: inspect one example
# ---------------------------------------------------------------
x_sample, y_sample = train_ds[0]
print(f"\nSingle example shapes:")
print(f"  input  x: {x_sample.shape}")
print(f"  target y: {y_sample.shape}")
print(f"  x[0:10]:  {x_sample[:10].tolist()}")
print(f"  y[0:10]:  {y_sample[:10].tolist()}")
print(f"  (y is x shifted by 1: {torch.equal(x_sample[1:], y_sample[:-1])})")


# ---------------------------------------------------------------
# 5. DataLoaders
# ---------------------------------------------------------------
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=0,      # Windows-safe; 0 = load in main process
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True,
    num_workers=0,
)

print(f"\nBatches per epoch (train): {len(train_loader)}")
print(f"Batches per epoch (val):   {len(val_loader)}")


# ---------------------------------------------------------------
# 6. Inspect one batch
# ---------------------------------------------------------------
xb, yb = next(iter(train_loader))
print(f"\nOne batch from DataLoader:")
print(f"  xb shape: {tuple(xb.shape)}  (batch_size, context_window)")
print(f"  yb shape: {tuple(yb.shape)}")
print(f"  xb dtype: {xb.dtype}")
print(f"  value range in batch: [{xb.min().item()}, {xb.max().item()}]")


# ---------------------------------------------------------------
# 7. Decode a batch back to text to prove it works end to end
# ---------------------------------------------------------------
import tiktoken
enc = tiktoken.get_encoding("gpt2")

print(f"\nFirst sequence in batch (decoded):")
print("-" * 60)
print(enc.decode(xb[0].tolist()))
print("-" * 60)

print(f"\nIts target (next-token shifted, decoded):")
print("-" * 60)
print(enc.decode(yb[0].tolist()))
print("-" * 60)
"""
Step 3: Tokenize the corpus with BPE and save train/val splits.
"""

import os
import urllib.request
import numpy as np
import tiktoken


# ---------------------------------------------------------------
# 1. Download the corpus (skip if already downloaded)
# ---------------------------------------------------------------
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
PATH = "tinyshakespeare.txt"

if not os.path.exists(PATH):
    print("Downloading Tiny Shakespeare...")
    urllib.request.urlretrieve(URL, PATH)
    print("Done.")

with open(PATH, "r", encoding="utf-8") as f:
    text = f.read()

print(f"Corpus size:  {len(text):,} characters")


# ---------------------------------------------------------------
# 2. Encode the entire corpus with GPT-2 BPE
# ---------------------------------------------------------------
enc = tiktoken.get_encoding("gpt2")
ids = enc.encode(text)
arr = np.array(ids, dtype=np.uint16)

print(f"Tokens (BPE): {len(arr):,}")
print(f"Compression:  {len(text) / len(arr):.2f} chars/token")
print(f"Vocab size:   {enc.n_vocab}")


# ---------------------------------------------------------------
# 3. Save the full corpus
# ---------------------------------------------------------------
np.save("tinyshakespeare_ids.npy", arr)
print(f"\nSaved tinyshakespeare_ids.npy ({len(arr):,} tokens)")


# ---------------------------------------------------------------
# 4. Train / Val split (90% / 10%)
# ---------------------------------------------------------------
n = len(arr)
split = int(0.9 * n)

train_ids = arr[:split]
val_ids   = arr[split:]

np.save("train_ids.npy", train_ids)
np.save("val_ids.npy",   val_ids)

print(f"Saved train_ids.npy ({len(train_ids):,} tokens)")
print(f"Saved val_ids.npy   ({len(val_ids):,} tokens)")


# ---------------------------------------------------------------
# 5. Verify everything round-trips
# ---------------------------------------------------------------
train_check = np.load("train_ids.npy")
val_check   = np.load("val_ids.npy")

print(f"\nVerification:")
print(f"  train_ids.npy dtype: {train_check.dtype}, shape: {train_check.shape}")
print(f"  val_ids.npy   dtype: {val_check.dtype}, shape: {val_check.shape}")
print(f"  Combined length matches full corpus: {len(train_check) + len(val_check) == len(arr)}")

print("\nDone. Now run step4_dataset.py.")
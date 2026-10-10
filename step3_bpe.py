import tiktoken

with open("tinyshakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

enc = tiktoken.get_encoding("gpt2")

# Encode the entire corpus into token IDs
ids = enc.encode(text)

print(f"Characters:      {len(text):,}")
print(f"Tokens (BPE):    {len(ids):,}")
print(f"Compression:     {len(text) / len(ids):.2f} chars/token")
print(f"Vocab size:      {enc.n_vocab}")
print(f"\nFirst 20 tokens: {ids[:20]}")
print(f"Decoded back:    {enc.decode(ids[:20])!r}")

import numpy as np

arr = np.array(ids, dtype=np.uint16)   # uint16 = 2 bytes, enough for 50257
np.save("tinyshakespeare_ids.npy", arr)
print(f"Saved {arr.shape[0]:,} tokens to tinyshakespeare_ids.npy")

# Verify
loaded = np.load("tinyshakespeare_ids.npy")
print(f"Reloaded: {loaded.shape}, dtype={loaded.dtype}")
assert (loaded == arr).all()
print("Round-trip verified.")

"""
Step 2: Build a character-level tokenizer from scratch.

A tokenizer is just a two-way map:
    text  ->  integer IDs
    integer IDs  ->  text
"""

# 1. A small training corpus. In real projects this would be gigabytes.
text = "hello world. this is my first llm. i am building it from scratch!"

print("=" * 60)
print("RAW TEXT")
print("=" * 60)
print(repr(text))
print()

# 2. Find all unique characters in the corpus.
#    sorted() makes the order deterministic (same every run).
chars = sorted(set(text))
vocab_size = len(chars)

print("=" * 60)
print("VOCABULARY")
print("=" * 60)
print(f"Unique characters: {chars}")
print(f"Vocab size: {vocab_size}")
print()

# 3. Build the two mappings (the "dictionary" of the tokenizer).
#    stoi = "string to integer"  (encode)
#    itos = "integer to string"  (decode)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

print("=" * 60)
print("MAPPINGS")
print("=" * 60)
print("stoi (char -> id):", stoi)
print("itos (id -> char):", itos)
print()

# 4. The two core functions every tokenizer must have.
def encode(s: str) -> list[int]:
    """Convert a string into a list of integer token IDs."""
    return [stoi[c] for c in s]

def decode(ids: list[int]) -> str:
    """Convert a list of integer token IDs back into a string."""
    return "".join(itos[i] for i in ids)

# 5. Test the round trip.
sample = "hello llm"
encoded = encode(sample)
decoded = decode(encoded)

print("=" * 60)
print("ROUND TRIP TEST")
print("=" * 60)
print(f"Original: {sample!r}")
print(f"Encoded:  {encoded}")
print(f"Decoded:  {decoded!r}")
print(f"Match?    {sample == decoded}")
print()

# 6. Encode the entire training text — this is what the model will train on.
full_encoded = encode(text)
print("=" * 60)
print("FULL CORPUS ENCODED")
print("=" * 60)
print(f"Total characters: {len(text)}")
print(f"Total tokens:     {len(full_encoded)}")
print(f"First 30 token IDs: {full_encoded[:30]}")
print()

# 7. Show how each token maps back to a character.
print("=" * 60)
print("TOKEN-BY-TOKEN DECODE (first 20)")
print("=" * 60)
for i, tid in enumerate(full_encoded[:20]):
    ch = itos[tid]
    print(f"  token #{i:2}  id={tid:2}  char={ch!r}")
# Challenge 1: What happens if you encode a character NOT in the vocab?
try:
    print(encode("z"))
except KeyError as e:
    print(f"Error! Character {e} is not in the vocabulary.")

# Challenge 2: Which character appears most often in the text?
from collections import Counter
counts = Counter(text)
print(f"Most common character: {counts.most_common(3)}")
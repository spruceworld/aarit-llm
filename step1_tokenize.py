import tiktoken

# Load GPT-2's tokenizer (a standard BPE tokenizer)
enc = tiktoken.get_encoding("gpt2")

text = "Hello, I am building an LLM from scratch!"

# Encode: text -> list of integer token IDs
token_ids = enc.encode(text)
print("Token IDs:", token_ids)

# Decode: integer token IDs -> text
decoded = enc.decode(token_ids)
print("Decoded:", decoded)

# See the actual tokens (subwords)
tokens = [enc.decode([tid]) for tid in token_ids]
print("Individual tokens:", tokens)

print()
print(f"Vocab size: {enc.n_vocab}")
print(f"Text length (chars): {len(text)}")
print(f"Token count: {len(token_ids)}")
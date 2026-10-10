import os
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
PATH = "tinyshakespeare.txt"

if not os.path.exists(PATH):
    print("Downloading Tiny Shakespeare...")
    urllib.request.urlretrieve(URL, PATH)
    print("Done.")
else:
    print("Already downloaded.")

with open(PATH, "r", encoding="utf-8") as f:
    text = f.read()

print(f"File size:   {os.path.getsize(PATH) / 1e6:.2f} MB")
print(f"Characters:  {len(text):,}")
print(f"First 300 chars:\n{text[:300]}")
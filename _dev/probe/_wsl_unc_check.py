"""Can Windows reach the WSL ext4 copy right now?

Needed to repair the damaged header and to verify unbuffered reads against an
independent copy.
"""

import os

BS = chr(92)
ROOT = BS * 2 + "wsl.localhost" + BS + "Ubuntu" + BS + "home" + BS + "hong" \
    + BS + "models"
SUB = "Qwen3.8-Flash-Next-AutoRound-3bpw-MTP"
NAME = "model-00001-of-00011.safetensors"

path = os.path.join(ROOT, SUB, NAME)
print("path:", path)
print("exists:", os.path.exists(path))
try:
    print("size:", os.path.getsize(path))
except Exception as e:
    print("getsize fail:", type(e).__name__, e)
try:
    with open(path, "rb") as f:
        head = f.read(8)
    print("open+read 8 B ok:", head)
except Exception as e:
    print("open fail:", type(e).__name__, e)
for base in (BS * 2 + "wsl.localhost", ROOT, os.path.dirname(path)):
    try:
        entries = sorted(os.listdir(base))
        print(f"listdir {base}: {len(entries)} entries -> "
              f"{[e for e in entries if 'Qwen3.8' in e][:5]}")
    except Exception as e:
        print(f"listdir {base} fail: {type(e).__name__}: {e}")

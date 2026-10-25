#!/usr/bin/env bash
# Stage the W4A16 checkpoint on D: reusing the existing 95.37 GiB PLE shard via
# a hardlink (same volume), so only the ~73 GiB of weight shards need copying.
set -uo pipefail
SRC=/home/hong/pp-recipe/models/Qwen3.8-Flash-Next-W4A16-AutoRound
DST=/mnt/d/models/Qwen3.8-Flash-Next-W4A16-AutoRound
mkdir -p "$DST"
echo "=== small files (configs, tokenizer, index) ==="
for f in "$SRC"/*; do
  b=$(basename "$f")
  case "$b" in
    model-*-of-00017.safetensors) continue ;;
  esac
  cp -f "$f" "$DST/$b" && echo "  copied $b"
done
echo "=== hardlink the PLE shard ==="
TARGET=/mnt/d/models/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP/model-00001-of-00011.safetensors
LN="$DST/model-00016-of-00017.safetensors"
if [ -e "$LN" ] || [ -L "$LN" ]; then rm -f "$LN"; fi
ln "$TARGET" "$LN" && echo "  linked $(basename "$TARGET") -> $(basename "$LN")"
ls -l "$LN"
echo "=== read check through the new path ==="
python3 - "$LN" <<'PY'
import hashlib, json, struct, sys
p = sys.argv[1]
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n).decode())
hdr.pop("__metadata__", None)
key = [k for k in hdr if k.endswith("shard_64.weight")][0]
off = hdr[key]["data_offsets"]
h = hashlib.sha256()
with open(p, "rb") as f:
    f.seek(8 + n + off[0])
    rem = off[1] - off[0]
    while rem:
        c = f.read(min(64 << 20, rem))
        if not c:
            break
        h.update(c)
        rem -= len(c)
print(f"  {key.split('.')[-2]}: {h.hexdigest()[:16]} (expect 21b372c328c8e326)")
PY
echo "=== dir state ==="
ls "$DST" | wc -l
du -sh --apparent-size "$DST"
df -h /mnt/d | tail -1

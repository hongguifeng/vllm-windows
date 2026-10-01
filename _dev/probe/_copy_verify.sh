#!/usr/bin/env bash
# Spot-check the WSL -> D: copy of the Flash-Next checkpoint.
# Full sha256 over 95 GiB costs ~20 min of disk reads on both ends; head+tail
# plus the metadata files catch truncation, ordering errors and offset drift.
set -u
SRC=/home/hong/models/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP
DST=/mnt/d/models/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP

echo "== metadata (full sha256) =="
for f in config.json model.safetensors.index.json tokenizer.json tokenizer_config.json generation_config.json; do
  a=$(sha256sum "$SRC/$f" | awk '{print $1}')
  b=$(sha256sum "$DST/$f" | awk '{print $1}')
  if [ "$a" = "$b" ]; then echo "  MATCH    $f"; else echo "  MISMATCH $f  src=$a win=$b"; fi
done

echo "== all shards (size check) =="
n_bad=0
for f in "$SRC"/model-*.safetensors "$SRC"/mtp-model-*.safetensors; do
  base=$(basename "$f")
  s1=$(stat -c %s "$f")
  s2=$(stat -c %s "$DST/$base" 2>/dev/null || echo missing)
  if [ "$s1" != "$s2" ]; then echo "  SIZE MISMATCH $base src=$s1 win=$s2"; n_bad=$((n_bad+1)); fi
done
echo "  size mismatches: $n_bad"

echo "== PLE shard (95.37 GiB), head and tail 1 GiB =="
S="$SRC/model-00001-of-00011.safetensors"
D="$DST/model-00001-of-00011.safetensors"
for region in head tail; do
  if [ "$region" = head ]; then
    a=$(dd if="$S" bs=1M count=1024 2>/dev/null | sha256sum | awk '{print $1}')
    b=$(dd if="$D" bs=1M count=1024 2>/dev/null | sha256sum | awk '{print $1}')
  else
    a=$(tail -c 1073741824 "$S" | sha256sum | awk '{print $1}')
    b=$(tail -c 1073741824 "$D" | sha256sum | awk '{print $1}')
  fi
  if [ "$a" = "$b" ]; then echo "  MATCH    $region 1 GiB"; else echo "  MISMATCH $region src=$a win=$b"; fi
done

echo "== safetensors header sanity (windows side, via python) =="
/d/code/vllm-windows/.venv/Scripts/python.exe - <<'PY'
import json, struct, os
p = r"D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP\model-00001-of-00011.safetensors"
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n).decode())
keys = [k for k in hdr if k != "__metadata__"]
ple = [k for k in keys if "ngram" in k or "ple" in k]
print(f"  tensors={len(keys)}  ple-related={len(ple)}  file={os.path.getsize(p)/2**30:.2f} GiB")
mx = max(hdr[k]["data_offsets"][1] for k in keys)
print(f"  header max data offset = {mx} (file size - header = {os.path.getsize(p)-8-n})  -> {'OK' if mx == os.path.getsize(p)-8-n else 'TRUNCATED'}")
PY

#!/usr/bin/env bash
# Copy the 15 weight shards (73.4 GiB) of the W4A16 checkpoint to D:.
set -uo pipefail
SRC=/home/hong/pp-recipe/models/Qwen3.8-Flash-Next-W4A16-AutoRound
DST=/mnt/d/models/Qwen3.8-Flash-Next-W4A16-AutoRound
LOG=/home/hong/pp-recipe/copy_to_d.log
exec >>"$LOG" 2>&1
echo "=== $(date -Is) copy start"
total=0
for f in "$SRC"/model-*-of-00017.safetensors; do
  b=$(basename "$f")
  [ "$b" = "model-00016-of-00017.safetensors" ] && continue   # hardlinked already
  if [ -f "$DST/$b" ] && [ "$(stat -c %s "$DST/$b")" = "$(stat -c %s "$f")" ]; then
    echo "  skip $b (present)"; continue
  fi
  t0=$(date +%s)
  cp -f "$f" "$DST/$b" || { echo "  FAILED $b"; continue; }
  dt=$(( $(date +%s) - t0 ))
  sz=$(stat -c %s "$DST/$b")
  total=$((total + sz))
  echo "  $b  $((sz/1000000)) MB in ${dt}s ($((sz/1000000/(dt>0?dt:1))) MB/s)  $(date +%H:%M:%S)"
done
echo "=== $(date -Is) copy done, $((total/1000000000)) GB written"
ls -l "$DST" | wc -l
du -sh --apparent-size "$DST"
df -h /mnt/d | tail -1

#!/usr/bin/env bash
# Download the W4A16 checkpoint file by file (sequential, resumable), then the
# engine image and the MTP draft head. Gentle on the SSD: one write stream only.
set -uo pipefail
DEST=/home/hong/pp-recipe/models/Qwen3.8-Flash-Next-W4A16-AutoRound
REV=4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897
REPO=Intel/Qwen3.8-Flash-Next-W4A16-AutoRound
LOG=/home/hong/pp-recipe/curl_dl.log
HF=${HF_ENDPOINT:-https://huggingface.co}
mkdir -p "$DEST"
exec >>"$LOG" 2>&1
echo "=== $(date -Is) start, endpoint $HF"
df -h / | tail -1
if ! curl -sf --max-time 180 "$HF/api/models/$REPO/tree/$REV?recursive=true" -o /tmp/pp_tree.json; then
  echo "FATAL cannot list files"
  exit 1
fi
python3 - <<'PY' > /tmp/pp_files.txt
import json
t = json.load(open('/tmp/pp_tree.json'))
for x in t:
    if x.get('type') == 'file' and x.get('size'):
        print(x['size'], x['path'])
PY
echo "files: $(wc -l < /tmp/pp_files.txt), bytes: $(awk '{s+=$1} END{print s}' /tmp/pp_files.txt)"
n=0; bad=0
while read -r size path; do
  n=$((n + 1))
  out="$DEST/$path"
  mkdir -p "$(dirname "$out")"
  want=$size
  for attempt in $(seq 1 60); do
    have=0
    [ -f "$out" ] && have=$(stat -c %s "$out")
    if [ "$have" -gt "$want" ]; then rm -f "$out"; have=0; fi
    [ "$have" -eq "$want" ] && break
    curl -L -C - --retry 4 --retry-delay 5 --connect-timeout 30 \
         --speed-time 90 --speed-limit 2048 -o "$out" \
         "$HF/$REPO/resolve/$REV/$path" || true
    have=$(stat -c %s "$out" 2>/dev/null || echo 0)
    [ "$have" -eq "$want" ] && break
    echo "retry $attempt $path: $have/$want"
    sleep 5
  done
  have=$(stat -c %s "$out" 2>/dev/null || echo 0)
  if [ "$have" -eq "$want" ]; then
    echo "OK   [$n] $path $want  ($(date +%H:%M:%S))"
  else
    echo "BAD  [$n] $path $have/$want"
    bad=$((bad + 1))
  fi
done < /tmp/pp_files.txt
echo "=== $(date -Is) download stage done, bad=$bad"
du -sh "$DEST"
find "$DEST" -type f -name '*.safetensors' -printf '%s\n' | awk '{s+=$1} END{printf "safetensors: %.1f GiB\n", s/1073741824}'
if [ "$bad" -ne 0 ]; then echo "=== STOP: $bad file(s) incomplete"; exit 1; fi

echo "=== $(date -Is) pull image stage"
cd /home/hong/pp-recipe
. ./versions.env
for attempt in $(seq 1 15); do
  echo "=== $(date -Is) pull attempt $attempt"
  if docker pull "$ENGINE_IMAGE"; then echo "=== $(date -Is) PULL OK"; break; fi
  sleep 15
done
docker images --format '{{.Repository}} {{.ID}} {{.Size}}' | grep -i alexbrandes || true

echo "=== $(date -Is) draft head stage"
DRAFT_HEAD=/home/hong/pp-recipe/models/draft-head-w4a16.safetensors
if [ ! -f "$DRAFT_HEAD" ]; then
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v /home/hong/pp-recipe/models:/models -v "$DEST:/model" \
    -v /home/hong/pp-recipe/tools:/tools:ro --entrypoint python3 "$ENGINE_IMAGE" \
    /tools/quantize_lm_head.py --model-dir /model \
    --out /models/draft-head-w4a16.safetensors --mse-scale-search
fi
if [ -f "$DRAFT_HEAD" ]; then
  got=$(sha256sum "$DRAFT_HEAD" | cut -d' ' -f1)
  echo "=== draft head $got want $DRAFT_HEAD_SHA256"
  [ "$got" = "$DRAFT_HEAD_SHA256" ] && echo "=== ALL DONE (draft head OK)" || echo "=== draft head MISMATCH"
fi
df -h / | tail -1

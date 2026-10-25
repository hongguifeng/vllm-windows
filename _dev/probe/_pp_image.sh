#!/usr/bin/env bash
# Pull the engine image (gentle concurrency), build the int4 MTP draft head,
# verify its pinned sha256.
set -uo pipefail
cd /home/hong/pp-recipe
. ./versions.env
DEST=$PWD/models/Qwen3.8-Flash-Next-W4A16-AutoRound
LOG=$PWD/image.log
exec >>"$LOG" 2>&1
echo "=== $(date -Is) pull start (max 2 concurrent layers)"
for attempt in $(seq 1 20); do
  echo "--- attempt $attempt $(date +%H:%M:%S)"
  if docker pull "$ENGINE_IMAGE"; then echo "PULL OK"; break; fi
  echo "--- attempt $attempt failed, sleep 20"
  sleep 20
done
if ! docker images --format '{{.Repository}} {{.ID}} {{.Size}}' | grep -qi alexbrandes; then
  echo "PULL FAILED after retries"; exit 1
fi
docker images --format '{{.Repository}} {{.ID}} {{.Size}}' | grep -i alexbrandes
echo "=== $(date -Is) draft head"
H=$PWD/models/draft-head-w4a16.safetensors
if [ ! -f "$H" ]; then
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$PWD/models:/models" -v "$DEST:/model" -v "$PWD/tools:/tools:ro" \
    --entrypoint python3 "$ENGINE_IMAGE" /tools/quantize_lm_head.py \
    --model-dir /model --out /models/draft-head-w4a16.safetensors --mse-scale-search
fi
if [ -f "$H" ]; then
  got=$(sha256sum "$H" | cut -d' ' -f1)
  echo "sha256 $got want $DRAFT_HEAD_SHA256"
  if [ "$got" = "$DRAFT_HEAD_SHA256" ]; then echo "DRAFT HEAD OK"; else echo "DRAFT HEAD MISMATCH"; fi
  ls -l "$H"
else
  echo "DRAFT HEAD MISSING"
fi
echo "=== $(date -Is) done"
df -h / | tail -1

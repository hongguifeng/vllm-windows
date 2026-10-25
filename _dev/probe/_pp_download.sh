#!/usr/bin/env bash
# Pull the engine image, download the W4A16 checkpoint, build the MTP draft head.
set -uo pipefail
cd /home/hong/pp-recipe
. ./versions.env
[ -f .env ] && . ./.env
MODELS_DIR=${MODELS_DIR:-$PWD/models}
MODEL_DIR=$MODELS_DIR/Qwen3.8-Flash-Next-W4A16-AutoRound
DRAFT_HEAD=$MODELS_DIR/draft-head-w4a16.safetensors
LOG=$PWD/download.log
mkdir -p "$MODEL_DIR"
exec >>"$LOG" 2>&1
echo "=== $(date -Is) start (pid $$)"
echo "=== $(date -Is) pull $ENGINE_IMAGE"
docker pull "$ENGINE_IMAGE" || { echo "FATAL pull failed"; exit 1; }
echo "=== $(date -Is) pull ok"
for attempt in 1 2 3 4 5 6; do
  echo "=== $(date -Is) snapshot attempt $attempt"
  if docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
      -v "$MODELS_DIR:/models" -v "$MODEL_DIR:/model" \
      --entrypoint python3 "$ENGINE_IMAGE" -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL_REPO', revision='$MODEL_REVISION', local_dir='/model', max_workers=4)
print('model ready')"; then
    echo "=== $(date -Is) snapshot ok"
    break
  fi
  echo "=== $(date -Is) attempt $attempt failed, sleep 120"
  sleep 120
done
du -sh "$MODEL_DIR"
if [ ! -f "$DRAFT_HEAD" ]; then
  echo "=== $(date -Is) build draft head"
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$MODELS_DIR:/models" -v "$MODEL_DIR:/model" -v "$PWD/tools:/tools:ro" \
    --entrypoint python3 "$ENGINE_IMAGE" /tools/quantize_lm_head.py \
    --model-dir /model --out /models/draft-head-w4a16.safetensors --mse-scale-search
fi
if [ -f "$DRAFT_HEAD" ]; then
  got=$(sha256sum "$DRAFT_HEAD" | cut -d' ' -f1)
  echo "=== $(date -Is) draft head sha256 $got (want $DRAFT_HEAD_SHA256)"
  [ "$got" = "$DRAFT_HEAD_SHA256" ] && echo "=== DRAFT HEAD OK" || echo "=== DRAFT HEAD MISMATCH"
fi
echo "=== $(date -Is) DONE"
df -h /home | tail -1

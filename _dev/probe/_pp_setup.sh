#!/usr/bin/env bash
# Prepare ~/pp-recipe for the AlexBrandes PP recipe inside WSL (ext4).
set -euo pipefail
DEST=/home/hong/pp-recipe
if [ ! -d "$DEST/.git" ]; then
  rm -rf "$DEST"
  git clone --depth 1 https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next "$DEST"
fi
mkdir -p "$DEST/models"
cat > "$DEST/.env" <<'EOF'
GPUS=1,2
HOST_TUNE=0
MODELS_DIR=/home/hong/pp-recipe/models
PORT=8000
EOF
cd "$DEST"
echo "--- repo ---"
ls
echo "--- hf api ---"
curl -s -o /dev/null -w "api model: %{http_code}\n" --max-time 30 \
  "https://huggingface.co/api/models/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/revision/4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897"
curl -s -o /dev/null -w "api tree: %{http_code}\n" --max-time 30 \
  "https://huggingface.co/api/models/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/tree/4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897?recursive=true"
echo "--- disk ---"
df -h /home | tail -1

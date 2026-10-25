#!/usr/bin/env bash
# Route comparison with redirects followed (-L), which the first pass missed.
REPO=Intel/Qwen3.8-Flash-Next-W4A16-AutoRound
REV=4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897
F=model-00003-of-00017.safetensors
R="0-31457280"
probe() { # label, curl args...
  local label=$1; shift
  echo "--- $label"
  curl -L -s -r "$R" -o /dev/null --max-time 30 \
    -w "  speed=%{speed_download} B/s http=%{http_code} host=%{url_effective}\n" "$@" \
    "https://huggingface.co/$REPO/resolve/$REV/$F"
}
probe "proxied HF" 
probe "direct HF" --noproxy '*'
echo "--- direct hf-mirror (no proxy, -L)"
curl --noproxy '*' -L -s -r "$R" -o /dev/null --max-time 30 \
  -w "  speed=%{speed_download} B/s http=%{http_code} host=%{url_effective}\n" \
  "https://hf-mirror.com/$REPO/resolve/$REV/$F"
echo "--- proxied hf-mirror (-L)"
curl -L -s -r "$R" -o /dev/null --max-time 30 \
  -w "  speed=%{speed_download} B/s http=%{http_code} host=%{url_effective}\n" \
  "https://hf-mirror.com/$REPO/resolve/$REV/$F"
echo "--- redirect chain (proxied HF)"
curl -s -I --max-time 20 "https://huggingface.co/$REPO/resolve/$REV/$F" | grep -i -e '^HTTP' -e '^location' | head -6

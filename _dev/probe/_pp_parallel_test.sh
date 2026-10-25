#!/usr/bin/env bash
# 1) kill the sequential downloader, 2) measure 16-way parallel aggregate,
# 3) check whether ModelScope mirrors the checkpoint.
pkill -f _pp_curl_dl.sh 2>/dev/null
sleep 2
REPO=Intel/Qwen3.8-Flash-Next-W4A16-AutoRound
REV=4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897
F=model-00003-of-00017.safetensors
URL="https://huggingface.co/$REPO/resolve/$REV/$F"
echo "--- 16 parallel 8 MiB ranges (proxied) ---"
start=$(date +%s.%N)
for i in $(seq 0 15); do
  s=$((i * 8388608)); e=$((s + 8388607))
  curl -L -s -r "$s-$e" -o /dev/null --max-time 40 "$URL" &
done
wait
end=$(date +%s.%N)
awk -v s="$start" -v e="$end" 'BEGIN{printf "  aggregate: %.2f MB/s over %.1f s (16 x 8 MiB)\n", 16*8/(e-s), (e-s)}'
echo "--- ModelScope repo ---"
for m in "Intel/Qwen3.8-Flash-Next-W4A16-AutoRound" "Qwen/Qwen3.8-Flash-Next-W4A16-AutoRound"; do
  code=$(curl -s -o /tmp/ms.json -w '%{http_code}' --max-time 25 --noproxy '*' "https://modelscope.cn/api/v1/models/$m")
  echo "  $m -> http=$code"
done
echo "--- ModelScope file list (if any) ---"
curl -s --max-time 25 --noproxy '*' "https://modelscope.cn/api/v1/models/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/repo/files?Revision=master&Root=" | head -c 400
echo

#!/usr/bin/env bash
# ModelScope throughput with redirects followed, plus the redirect target.
set -uo pipefail
MS=https://modelscope.cn
M=Intel/Qwen3.8-Flash-Next-W4A16-AutoRound
BIG=model-00016-of-00017.safetensors
U="$MS/models/$M/resolve/master/$BIG"
echo "--- redirect target ---"
curl --noproxy '*' -s -I --max-time 20 "$U" | grep -i -e '^HTTP' -e '^location' | head -4
echo "--- 1 connection, 60 MiB (direct, -L) ---"
curl --noproxy '*' -L -s -r 0-62914560 -o /dev/null --max-time 40 \
  -w "  speed=%{speed_download} B/s http=%{http_code}\n" "$U"
echo "--- 4 connections, 32 MiB each (direct, -L) ---"
start=$(date +%s.%N)
for i in 0 1 2 3; do
  s=$((i * 33554432)); e=$((s + 33554431))
  curl --noproxy '*' -L -s -r "$s-$e" -o /dev/null --max-time 40 "$U" &
done
wait
end=$(date +%s.%N)
awk -v s="$start" -v e="$end" 'BEGIN{printf "  aggregate(4): %.2f MB/s over %.1f s\n", 4*32/(e-s), (e-s)}'
echo "--- 8 connections, 16 MiB each (direct, -L) ---"
start=$(date +%s.%N)
for i in $(seq 0 7); do
  s=$((i * 16777216)); e=$((s + 16777215))
  curl --noproxy '*' -L -s -r "$s-$e" -o /dev/null --max-time 40 "$U" &
done
wait
end=$(date +%s.%N)
awk -v s="$start" -v e="$end" 'BEGIN{printf "  aggregate(8): %.2f MB/s over %.1f s\n", 8*16/(e-s), (e-s)}'

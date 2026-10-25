#!/usr/bin/env bash
# Compare download routes for one HF file: proxied, direct, hf-mirror, and a
# multi-connection sample of the mirror.
REPO=Intel/Qwen3.8-Flash-Next-W4A16-AutoRound
REV=4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897
F=model-00003-of-00017.safetensors
range="0-31457280"
echo "--- proxied huggingface.co ---"
curl -s -r "$range" -o /dev/null -w "  %{speed_download} B/s http=%{http_code}\n" --max-time 25 "https://huggingface.co/$REPO/resolve/$REV/$F"
echo "--- direct huggingface.co ---"
curl --noproxy '*' -s -r "$range" -o /dev/null -w "  %{speed_download} B/s http=%{http_code}\n" --max-time 25 "https://huggingface.co/$REPO/resolve/$REV/$F"
echo "--- proxied hf-mirror.com ---"
curl -s -r "$range" -o /dev/null -w "  %{speed_download} B/s http=%{http_code}\n" --max-time 25 "https://hf-mirror.com/$REPO/resolve/$REV/$F"
echo "--- direct hf-mirror.com ---"
curl --noproxy '*' -s -r "$range" -o /dev/null -w "  %{speed_download} B/s http=%{http_code}\n" --max-time 25 "https://hf-mirror.com/$REPO/resolve/$REV/$F"
echo "--- 4 parallel ranges, proxied mirror ---"
for i in 0 1 2 3; do
  s=$((i * 8388608)); e=$((s + 8388607))
  curl -s -r "$s-$e" -o /dev/null -w "  conn$i %{speed_download}\n" --max-time 25 "https://hf-mirror.com/$REPO/resolve/$REV/$F" &
done
wait
echo "--- current live file growth ---"
D=/home/hong/pp-recipe/models/Qwen3.8-Flash-Next-W4A16-AutoRound/model-00001-of-00017.safetensors
a=$(stat -c %s "$D" 2>/dev/null || echo 0); sleep 20; b=$(stat -c %s "$D" 2>/dev/null || echo 0)
awk -v a="$a" -v b="$b" 'BEGIN{printf "  live: %.2f MB/s\n",(b-a)/20/1e6}'

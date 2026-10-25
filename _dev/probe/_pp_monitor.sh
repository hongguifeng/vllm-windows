#!/usr/bin/env bash
# One-shot progress report for the PP recipe download.
LOG=/home/hong/pp-recipe/download.log
MOD=/home/hong/pp-recipe/models
if grep -q "snapshot attempt" "$LOG" 2>/dev/null; then
  echo "snapshot: STARTED"
else
  echo "snapshot: not yet (image pull in progress)"
fi
echo "pull complete layers: $(grep -c 'Pull complete' "$LOG" 2>/dev/null)"
echo "--- tail ---"
tail -3 "$LOG" 2>/dev/null
echo "--- docker disk ---"
docker system df --format '{{.Type}} {{.Size}} reclaimable {{.Reclaimable}}' 2>/dev/null | head -3
a=$(du -sb "$MOD" | cut -f1)
sleep 60
b=$(du -sb "$MOD" | cut -f1)
awk -v a="$a" -v b="$b" 'BEGIN{printf "model dir: %.2f -> %.2f GiB in 60s, rate %.1f MB/s\n", a/1073741824, b/1073741824, (b-a)/60/1e6}'
du -sh "$MOD"
df -h /home | tail -1

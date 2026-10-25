#!/usr/bin/env bash
# Compact status for the ModelScope checkpoint download.
set -uo pipefail
DEST=/home/hong/pp-recipe/models/Qwen3.8-Flash-Next-W4A16-AutoRound
LOG=/home/hong/pp-recipe/ms_dl.log
TOTAL=$((1688 * 1073741824 / 10))   # 168.8 GiB
cur=$(du -sb "$DEST" 2>/dev/null | cut -f1 || echo 0)
sleep 30
cur2=$(du -sb "$DEST" 2>/dev/null | cut -f1 || echo 0)
rate=$(awk -v a="$cur2" -v b="$cur" 'BEGIN{printf "%.2f", (a-b)/30/1e6}')
pct=$(awk -v c="$cur2" -v t="$TOTAL" 'BEGIN{printf "%.1f", 100*c/t}')
rem=$(awk -v c="$cur2" -v t="$TOTAL" 'BEGIN{printf "%.1f", (t-c)/1073741824}')
eta=$(awk -v c="$cur2" -v t="$TOTAL" -v r="$rate" 'BEGIN{ if (r+0 > 0.1) printf "%.1f h", (t-c)/1e6/r/3600; else print "n/a" }')
if pgrep -f _pp_ms_dl.sh >/dev/null; then proc="running (pid $(pgrep -f _pp_ms_dl.sh | head -1))"; else proc="NOT RUNNING"; fi
stall="no"
[ "$(awk -v r="$rate" 'BEGIN{print (r<0.3)?1:0}')" = "1" ] && [ "$proc" = "running" ] && stall="POSSIBLE (rate < 0.3 MB/s)"
echo "bytes: $cur2 ($pct% of 168.8 GiB), remaining $rem GiB"
echo "rate: $rate MB/s, eta $eta"
echo "downloader: $proc"
echo "stall: $stall"
echo "files done: $(grep -c '^OK ' "$LOG" 2>/dev/null || echo 0), skipped: $(grep -c '^SKIP ' "$LOG" 2>/dev/null || echo 0), bad: $(grep -cE '^(BADHASH|INCOMPLETE)' "$LOG" 2>/dev/null || echo 0)"
echo "last: $(grep -E '^(GET|OK|SKIP|BAD)' "$LOG" 2>/dev/null | tail -2 | tr '\n' ' | ')"
df -h / | tail -1

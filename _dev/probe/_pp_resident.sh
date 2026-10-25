#!/usr/bin/env bash
# How much of the 95.37 GiB PLE shard is actually resident right now?
echo "=== page cache / mem ==="
free -g | head -2
echo "=== worker processes ==="
docker top hong-pc -o pid,cmd 2>/dev/null | grep -E "VllmWorker|EngineCore" | head -5
echo "=== resident bytes of the PLE shard mapping ==="
total=0
for pid in $(pgrep -f "VllmWorker|EngineCore" 2>/dev/null); do
  rss=$(awk '
    /model-00016-of-00017/ {inblk=1; next}
    inblk && /^Rss:/ {print $2; inblk=0}
  ' /proc/$pid/smaps 2>/dev/null | awk '{s+=$1} END {print s+0}')
  if [ "${rss:-0}" -gt 0 ]; then
    echo "  pid $pid: $((rss/1024)) MiB resident of the PLE shard"
    total=$((total + rss))
  fi
done
[ "$total" -eq 0 ] && echo "  (mapping not found in smaps; workers may be in another pid namespace)"
echo "  total resident: $((total/1024/1024)) GiB of 95.37 GiB"
echo "=== read IOPS while idle (5s) ==="
a=$(awk '{print $1}' /sys/block/sdd/stat); sleep 5; b=$(awk '{print $1}' /sys/block/sdd/stat)
echo "  idle reads/s: $(( (b-a)/5 ))"

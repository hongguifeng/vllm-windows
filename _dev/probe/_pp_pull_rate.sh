#!/usr/bin/env bash
# Measure the live throughput of the docker pull.
pid=$(pgrep -f "docker pull ghcr.io/alexbrandes" | head -1)
echo "pull pid: ${pid:-none}"
if [ -n "${pid:-}" ]; then
  a=$(awk '/^rchar/{print $2}' /proc/$pid/io)
  sleep 30
  b=$(awk '/^rchar/{print $2}' /proc/$pid/io)
  awk -v a="$a" -v b="$b" 'BEGIN{printf "pull throughput: %.2f MB/s (30s window)\n",(b-a)/30/1e6}'
else
  echo "no docker pull process"
fi
echo "--- recent log lines ---"
tail -6 /home/hong/pp-recipe/download.log 2>/dev/null
echo "--- snapshot started? ---"
grep -c "snapshot attempt" /home/hong/pp-recipe/download.log 2>/dev/null || true

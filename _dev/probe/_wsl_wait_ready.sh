#!/bin/bash
# Wait for the WSL-side hyperqwen container to become healthy, then dump state.
# Run from Windows:  wsl.exe -d Ubuntu -- bash -c "tr -d '\r' < /mnt/d/.../_wsl_wait_ready.sh > /tmp/w.sh && bash /tmp/w.sh"
CNAME=qwen38-27b-rtx3090-single-1
for i in $(seq 1 160); do
  s=$(docker inspect --format '{{.State.Health.Status}}' "$CNAME" 2>/dev/null)
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  echo "$(date +%T) health=${s:-?} vram=${mem}MiB"
  if [ "$s" = "healthy" ]; then echo "HEALTHY"; break; fi
  sleep 15
done
echo "=== compose ps ==="
cd /home/hong/testcode/qwen38-27b-rtx3090 && docker compose --profile single ps 2>&1 | tail -3
echo "=== log tail ==="
docker compose --profile single logs --tail 45 single 2>&1 | tail -50
echo "=== df shm ==="
df -h /dev/shm | tail -1

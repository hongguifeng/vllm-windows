#!/usr/bin/env bash
# Watch the PP engine boot: health, container state, last log line each cycle.
for i in $(seq 1 12); do
  if curl -fs --max-time 3 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "HEALTH OK at $(date +%H:%M:%S)"
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -q '^hong-pc$'; then
    echo "CONTAINER EXITED at $(date +%H:%M:%S)"
    break
  fi
  echo "[$i $(date +%H:%M:%S)] $(docker logs --tail 1 hong-pc 2>&1 | tail -1 | cut -c1-150)"
  sleep 45
done
echo "--- last 15 ---"
docker logs --tail 15 hong-pc 2>&1 | cut -c1-190

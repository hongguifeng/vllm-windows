#!/bin/bash
# Stop pier only. Never touch the model-server container.
for p in $(pgrep -f "pier run"); do
  kill -9 "$p" 2>/dev/null || true
done
sleep 2
docker ps -a --format '{{.Names}}' | grep -E -- '-(main-1|pier-egress-proxy-1)$' | xargs -r docker rm -f >/dev/null 2>&1
echo "pier procs: $(pgrep -fc 'pier run' || true)"
echo "containers: $(docker ps -aq | wc -l)"

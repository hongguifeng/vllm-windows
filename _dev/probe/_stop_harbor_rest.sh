#!/bin/bash
# Stop the interrupted Harbor rest-job only; leave Pier trials and other containers alone.
pkill -f "harbor run -p datasets/pro20" || true
sleep 2
pgrep -af "harbor run -p datasets/pro20" || true
# Remove only containers whose name starts with a pro20 trial prefix (instance_...), never the model server.
docker ps -aq --filter "name=instance_" | xargs -r docker rm -f >/dev/null 2>&1
docker ps -a --format "{{.Names}}" | grep -c "^instance_" || true

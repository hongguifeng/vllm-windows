#!/usr/bin/env bash
# Restart the image pull job with the unsupported flag removed.
S=/mnt/d/code/vllm-windows/_dev/probe/_pp_image.sh
pkill -f "$S" 2>/dev/null
sleep 1
sed -i 's|docker pull --max-concurrent-downloads 2 |docker pull |' "$S"
echo "--- pull line now ---"
grep -n 'docker pull' "$S"
echo "--- leftovers ---"
pgrep -af "_pp_image" | grep -v "pp_restart" || echo none
df -h / | tail -1

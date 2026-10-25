#!/bin/bash
pkill -f "pier run" || true
sleep 2
pgrep -af "pier run" || true
docker ps -aq | xargs -r docker rm -f >/dev/null 2>&1
echo "remaining pier procs: $(pgrep -fc 'pier run' || true)"

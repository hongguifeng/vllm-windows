#!/usr/bin/env bash
# Vision + 71680 + no-async ITL probe (discriminator: is the 4x step penalty
# tied to the tower's presence, or was yesterday's bench run an anomaly?)
set -u
REPO=/d/code/vllm-windows
REPO_WIN=D:/code/vllm-windows
PY="$REPO/.venv/Scripts/python.exe"
PORT=8017
LOG="$REPO/_vis_itl.log"
: > "$LOG"

AB_PORT=$PORT AB_VISION_OFFLOAD=1 AB_MAX_LEN=71680 AB_NO_ASYNC=1 \
  bash "$REPO/_ab_serve_win.sh" >>"$LOG" 2>&1 &
SRV=$!
ready=0
for _ in $(seq 1 120); do
  sleep 5
  if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
  kill -0 "$SRV" 2>/dev/null || break
done
if [ "$ready" != 1 ]; then
  echo "SERVER NOT READY"; tail -15 "$LOG"
else
  grep -o "Graph capturing finished in [0-9]* secs, took [0-9.]* GiB" "$LOG"
  "$PY" "$REPO_WIN/_dev/bench/_itl_client.py" --port $PORT --tokens 512 --label vis_noasync
  grep -o "Avg generation throughput: [0-9.]*" "$LOG" | tail -1
fi
kill -9 "$SRV" 2>/dev/null
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
sleep 4
echo "=== done ==="

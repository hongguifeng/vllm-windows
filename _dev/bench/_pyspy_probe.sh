#!/usr/bin/env bash
# Sample EngineCore + API server for 25s each during a decode request, then
# aggregate where wall-time goes (queue.get vs GPU sync vs zmq).
set -u
REPO=/d/code/vllm-windows
REPO_WIN=D:/code/vllm-windows
PY="$REPO/.venv/Scripts/python.exe"
SPY="$REPO/.venv/Scripts/py-spy.exe"
PORT=8017
LOG="$REPO/_pyspy_probe.log"
: > "$LOG"

AB_PORT=$PORT AB_TEXT_ONLY=1 AB_MAX_LEN=71680 AB_NO_ASYNC=1 \
  bash "$REPO/_ab_serve_win.sh" >>"$LOG" 2>&1 &
SRV=$!
ready=0
for _ in $(seq 1 120); do
  sleep 5
  if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
  kill -0 "$SRV" 2>/dev/null || break
done
[ "$ready" = 1 ] || { echo "SERVER NOT READY"; exit 1; }

API_PID=$(MSYS_NO_PATHCONV=1 netstat -ano | grep ":$PORT" | grep LISTENING | head -1 | awk '{print $NF}')
ENG_PID=$(grep -o "EngineCore pid=[0-9]*" "$LOG" | head -1 | grep -o "[0-9]*")
echo "api=$API_PID eng=$ENG_PID"

( "$PY" "$REPO_WIN/_dev/bench/_itl_client.py" --port $PORT --tokens 1024 --label rec > "$REPO/_dev/out/_pyspy_itl.txt" 2>&1 ) &
sleep 8
"$SPY" record --pid "$ENG_PID" --duration 25 --rate 100 --format speedscope \
  -o "$REPO/_spy_engine.json" && echo "engine recorded"
"$SPY" record --pid "$API_PID" --duration 25 --rate 100 --format speedscope \
  -o "$REPO/_spy_api.json" && echo "api recorded"
wait
cat "$REPO/_dev/out/_pyspy_itl.txt"
kill -9 "$SRV" 2>/dev/null
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
sleep 3
echo "=== done ==="

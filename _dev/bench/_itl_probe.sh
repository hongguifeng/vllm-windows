#!/usr/bin/env bash
# Bisect the no-async-scheduling slowdown: boot the A/B server once per config,
# run the tiny ITL client, shut down. All configs chained in one process so a
# long-running background task cannot be killed mid-warmup.
#
# Usage: bash _itl_probe.sh          # runs the 2-config bisect
#        CFGS="a b" bash _itl_probe.sh
set -u
REPO=/d/code/vllm-windows
REPO_WIN=D:/code/vllm-windows
PY="$REPO/.venv/Scripts/python.exe"
PORT=8017
LOG="$REPO/_itl_probe.log"

run_cfg() {  # run_cfg <name> <env...>
  local name=$1; shift
  echo "########## $name ##########"
  : > "$LOG"
  env "$@" AB_PORT=$PORT bash "$REPO/_ab_serve_win.sh" >>"$LOG" 2>&1 &
  local srv=$!
  local ready=0
  for _ in $(seq 1 120); do
    sleep 5
    if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
    kill -0 "$srv" 2>/dev/null || break
  done
  if [ "$ready" != 1 ]; then
    echo "[$name] SERVER NOT READY"; tail -15 "$LOG"
  else
    "$PY" "$REPO_WIN/_dev/bench/_itl_client.py" --port $PORT --label "$name"
    grep -o "async_scheduling': [A-Za-z]*" "$LOG" | head -1
    grep -c "speculative_config" "$LOG" | xargs echo "spec-config-lines:"
  fi
  kill -9 "$srv" 2>/dev/null
  pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
  sleep 4
}

CFGS=${CFGS:-"nospec nospec_dflash"}
for cfg in $CFGS; do
  case $cfg in
    nospec)        run_cfg text_noasync_nospec  AB_TEXT_ONLY=1 AB_MAX_LEN=71680 AB_NO_ASYNC=1 AB_NOSPEC=1 ;;
    nospec_dflash) run_cfg text_noasync_dflash  AB_TEXT_ONLY=1 AB_MAX_LEN=71680 AB_NO_ASYNC=1 ;;
    async_dflash)  run_cfg text_async_dflash    AB_TEXT_ONLY=1 AB_MAX_LEN=71680 ;;
  esac
done
echo "=== all done ==="

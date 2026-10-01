#!/usr/bin/env bash
# Start the A/B server at a chosen context/pin, print the KV *ledger*, benchmark,
# then shut it down -- all inside one process so a long-running background task
# cannot be killed from the outside mid-warmup (see AB_WSL_VS_WINDOWS.md 4.3).
#
# Usage:
#   AB_MAX_LEN=71680 AB_KV_BYTES=6000000000 bash _ab_kv_probe.sh <label>
#   AB_FAIL_ON_PADDING=1  exit non-zero if the full/GDN buckets still get padded
#
# Env: AB_MAX_LEN, AB_KV_BYTES, AB_PORT, AB_SKIP_BENCH=1, AB_ONLY, AB_LENS,
#      plus everything _ab_serve_win.sh reads (AB_TEXT_ONLY, AB_VISION_OFFLOAD..).
set -u

REPO=/d/code/vllm-windows
# Native python needs a Windows path: /d/... would be read as C:\d\... (see the
# project's tool gotchas), so hand it the D:/ form.
REPO_WIN=D:/code/vllm-windows
PY="$REPO/.venv/Scripts/python.exe"
LABEL=${1:-probe}
PORT=${AB_PORT:-8000}
LOG="$REPO/_ab_kv_${LABEL}.log"

cd "$REPO"
: > "$LOG"
AB_PORT="$PORT" bash "$REPO/_ab_serve_win.sh" >>"$LOG" 2>&1 &
SRV=$!

ready=0
for _ in $(seq 1 120); do          # up to 10 min: a cold compile can take 4
  sleep 5
  if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
  kill -0 "$SRV" 2>/dev/null || break
done

echo "=== KV ledger ($LABEL: max-len ${AB_MAX_LEN:-60928} / pin ${AB_KV_BYTES:-6000000000}) ==="
grep -n "Initial free memory\|padding layers\|Sliding-window bucket\|GPU KV cache size" "$LOG" || true

rc=0
if [ "$ready" != 1 ]; then
  echo "=== SERVER NEVER BECAME READY ==="
  tail -30 "$LOG"
  rc=1
else
  if [ "${AB_SKIP_BENCH:-0}" != 1 ]; then
    "$PY" "$REPO_WIN/_dev/bench/_ab_bench.py" --port "$PORT" --label "$LABEL" \
      --outdir "$REPO_WIN/_dev/out/_ab_results" --only "${AB_ONLY:-warmup,prefill,cohort}" \
      --lens "${AB_LENS:-1024,16384}"
  fi
  if [ "${AB_FAIL_ON_PADDING:-0}" = 1 ] && \
     grep -q "Add 4 padding layers\|Add 2 padding layers" "$LOG"; then
    echo "=== STILL PADDING THE FULL/ATTENTION + GDN BUCKETS ==="
    rc=1
  fi
fi

kill -9 "$SRV" 2>/dev/null
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
sleep 2
echo "=== done (rc=$rc), log: $LOG ==="
exit "$rc"

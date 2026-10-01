#!/usr/bin/env bash
# Boot-lottery A/B: does zeroing scratch memory before DFlash2 graph capture
# eliminate the boot-to-boot device-side assert / partial-capture lottery?
# Arm Z = ZEROFILL_EXP=1 (patched zeroing active), arm C = control (patch inert).
set -u
REPO=/d/code/vllm-windows
OUT="$REPO/_dev/out/_boot_lottery_results.txt"
: >> "$OUT"   # resume: keep results from boots 01/02

cleanup() {
  # kill ONLY project-venv pythons + spawned EngineCore (never VS Code LSPs)
  local round i used
  for round in 1 2 3; do
    MSYS_NO_PATHCONV=1 wmic process where "name='python.exe' and (ExecutablePath like '%.venv%' or commandline like '%spawn_main%')" get processid 2>/dev/null \
      | grep -oE '[0-9]+' | while read p; do
        MSYS_NO_PATHCONV=1 taskkill -F -PID "$p" >/dev/null 2>&1
      done
    for i in $(seq 1 45); do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
      [ "${used:-99999}" -lt 1500 ] && break
      sleep 2
    done
    if [ "${used:-99999}" -lt 1500 ]; then break; fi
    echo "[cleanup] round $round stuck at ${used:-?} MiB; holders:" >> "$OUT"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> "$OUT" 2>/dev/null
  done
  echo "[cleanup] final vram=${used:-?} MiB" >> "$OUT"
}

one_boot() {
  local arm="$1" tag="$2"
  cleanup
  local log="$REPO/_dev/out/_boot_lottery_${tag}.log"
  if [ "$arm" = "Z" ]; then
    ZEROFILL_EXP=1 AB_TEXT_ONLY=1 AB_MAX_LEN=71680 AB_KV_BYTES=6000000000 \
      bash "$REPO/_dev/bench/_ab_serve_win.sh" > "$log" 2>&1 &
  else
    AB_TEXT_ONLY=1 AB_MAX_LEN=71680 AB_KV_BYTES=6000000000 \
      bash "$REPO/_dev/bench/_ab_serve_win.sh" > "$log" 2>&1 &
  fi
  local ok=0 t0
  t0=$(date +%s)
  for i in $(seq 1 60); do
    sleep 5
    if curl -sf -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1; then ok=1; break; fi
  done
  local t1 cap zf verdict
  t1=$(date +%s)
  cap=$(grep -o "Graph capturing finished in [0-9]* secs, took [0-9.]* GiB" "$log" | head -1)
  zf=$(grep -c "zerofill-exp. zeroed" "$log")
  if [ "$ok" = 1 ]; then
    verdict=HEALTHY
  elif grep -q "device-side assert" "$log"; then
    verdict=CRASH_ASSERT
  else
    verdict=DEAD_OTHER
  fi
  echo "$tag arm=$arm result=$verdict boot_s=$((t1-t0)) zerofill_lines=$zf | $cap" >> "$OUT"
  tail -2 "$OUT"
  cleanup
}

# deconfounded arm order for the remaining 8 boots (5Z/5C overall):
# Z at ordinals 1,5,6,8,9 -- C at 2,3,4,7,10
for spec in C:03 C:04 Z:05 Z:06 C:07 Z:08 Z:09 C:10; do
  one_boot "${spec%%:*}" "${spec##*:}"
done
echo "ALL_DONE" >> "$OUT"
cat "$OUT"

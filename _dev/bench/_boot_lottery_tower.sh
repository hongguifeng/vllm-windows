#!/usr/bin/env bash
# Discriminator: historical boot-time device-side asserts -- do they need the
# vision tower in the capture path? 6 tower-on boots, alternating C/Z arms.
set -u
REPO=/d/code/vllm-windows
OUT="$REPO/_dev/out/_boot_lottery_tower.txt"
: > "$OUT"

cleanup() {
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
    echo "[cleanup] round $round stuck at ${used:-?} MiB" >> "$OUT"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> "$OUT" 2>/dev/null
  done
  echo "[cleanup] final vram=${used:-?} MiB" >> "$OUT"
}

one_boot() {
  local arm="$1" tag="$2" zf
  if [ "$arm" = "Z" ]; then zf=1; else zf=0; fi
  cleanup
  local log="$REPO/_dev/out/_boot_lottery_${tag}.log"
  env ZEROFILL_EXP=$zf AB_VISION_OFFLOAD=1 AB_TEXT_ONLY=0 \
      AB_MAX_LEN=71680 AB_KV_BYTES=6000000000 \
      bash "$REPO/_dev/bench/_ab_serve_win.sh" > "$log" 2>&1 &
  local ok=0 t0 t1 cap zf_lines verdict
  t0=$(date +%s)
  for i in $(seq 1 72); do
    sleep 5
    if curl -sf -m 3 http://127.0.0.1:8000/health >/dev/null 2>&1; then ok=1; break; fi
  done
  t1=$(date +%s)
  cap=$(grep -o "Graph capturing finished in [0-9]* secs, took [0-9.]* GiB" "$log" | head -1)
  zf_lines=$(grep -c "zerofill-exp. zeroed" "$log")
  if [ "$ok" = 1 ]; then
    verdict=HEALTHY
  elif grep -q "device-side assert" "$log"; then
    verdict=CRASH_ASSERT
  else
    verdict=DEAD_OTHER
  fi
  echo "$tag arm=$arm result=$verdict boot_s=$((t1-t0)) zerofill_lines=$zf_lines | $cap" >> "$OUT"
  tail -2 "$OUT"
  cleanup
}

for spec in C:11 Z:12 C:13 Z:14 C:15 Z:16; do
  one_boot "${spec%%:*}" "${spec##*:}"
done
echo "ALL_DONE" >> "$OUT"

#!/usr/bin/env bash
# Wait up to ~8 minutes for the image pull to finish, printing progress.
LOG=/home/hong/pp-recipe/image.log
for i in $(seq 1 16); do
  if grep -q "PULL OK" "$LOG" 2>/dev/null; then echo "state: PULL OK"; break; fi
  if grep -q "PULL FAILED" "$LOG" 2>/dev/null; then echo "state: PULL FAILED"; break; fi
  echo "[$i] layers=$(grep -c 'Pull complete' "$LOG" 2>/dev/null) $(date +%H:%M:%S)"
  sleep 30
done
echo "--- tail ---"
tail -6 "$LOG"

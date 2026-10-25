#!/usr/bin/env bash
# Download the W4A16 checkpoint from ModelScope (domestic, fast), one file at a
# time, resume + sha256 verification against the ModelScope manifest.
set -uo pipefail
MS=https://modelscope.cn
M=Intel/Qwen3.8-Flash-Next-W4A16-AutoRound
DEST=/home/hong/pp-recipe/models/Qwen3.8-Flash-Next-W4A16-AutoRound
LOG=/home/hong/pp-recipe/ms_dl.log
mkdir -p "$DEST"
exec >>"$LOG" 2>&1
echo "=== $(date -Is) ModelScope download start"
curl -s --max-time 60 --noproxy '*' "$MS/api/v1/models/$M/repo/files?Revision=master&Root=" -o /tmp/ms_files.json || { echo FATAL list; exit 1; }
python3 - <<'PY' > /tmp/ms_list.txt
import json
d = json.load(open('/tmp/ms_files.json'))['Data']['Files']
for f in sorted(d, key=lambda x: x['Size']):
    print(f['Size'], f['Path'], f.get('Sha256') or '-')
PY
total=$(awk '{s+=$1} END{print s}' /tmp/ms_list.txt)
python3 -c "print('files %d, total %.1f GiB' % (int(open('/tmp/ms_list.txt').read().count(chr(10))), $total/2**30))"
done_bytes=0
while read -r size path sha; do
  out="$DEST/$path"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$size" ]; then
    if [ "$sha" != "-" ]; then
      if [ "$(sha256sum "$out" | cut -d' ' -f1)" = "$sha" ]; then
        echo "SKIP $path (verified)"; done_bytes=$((done_bytes + size)); continue
      else
        echo "HASH-BAD existing $path, restarting"; rm -f "$out"
      fi
    else
      echo "SKIP $path (size ok)"; done_bytes=$((done_bytes + size)); continue
    fi
  fi
  echo "GET  $path $size ($(date +%H:%M:%S))"
  for attempt in $(seq 1 20); do
    curl --noproxy '*' -L -C - -s -S --retry 3 --retry-delay 5 --connect-timeout 30 \
         --speed-time 120 --speed-limit 1024 -o "$out" \
         "$MS/models/$M/resolve/master/$path"
    have=$(stat -c %s "$out" 2>/dev/null || echo 0)
    [ "$have" -eq "$size" ] && break
    echo "  retry $attempt $path $have/$size"
    sleep 5
  done
  if [ "$(stat -c %s "$out" 2>/dev/null || echo 0)" = "$size" ]; then
    if [ "$sha" != "-" ]; then
      if [ "$(sha256sum "$out" | cut -d' ' -f1)" = "$sha" ]; then
        echo "OK   $path $(date +%H:%M:%S)"
      else
        echo "BADHASH $path"
      fi
    else
      echo "OK   $path (no sha) $(date +%H:%M:%S)"
    fi
    done_bytes=$((done_bytes + size))
    python3 -c "print('  progress %.1f / %.1f GiB' % ($done_bytes/2**30, $total/2**30))"
  else
    echo "INCOMPLETE $path $(stat -c %s "$out" 2>/dev/null || echo 0)/$size"
  fi
done < /tmp/ms_list.txt
echo "=== $(date -Is) download done"
du -sh "$DEST"
df -h / | tail -1

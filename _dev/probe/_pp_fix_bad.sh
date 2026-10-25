#!/usr/bin/env bash
# Repair pass: re-fetch any file whose size or sha256 does not match the
# ModelScope manifest. No resume (a resumed file can end up with mixed bytes).
set -uo pipefail
MS=https://modelscope.cn
M=Intel/Qwen3.8-Flash-Next-W4A16-AutoRound
DEST=/home/hong/pp-recipe/models/Qwen3.8-Flash-Next-W4A16-AutoRound
LOG=/home/hong/pp-recipe/ms_fix.log
[ -f /tmp/ms_files.json ] || curl -s --max-time 60 --noproxy '*' "$MS/api/v1/models/$M/repo/files?Revision=master&Root=" -o /tmp/ms_files.json
exec >>"$LOG" 2>&1
echo "=== $(date -Is) verify pass"
list=/tmp/ms_fix_list.txt
python3 - <<'PY' > "$list"
import json
d = json.load(open('/tmp/ms_files.json'))['Data']['Files']
for f in sorted(d, key=lambda x: x['Size']):
    print(f['Size'], f['Path'], f.get('Sha256') or '-')
PY
ok=0; fixed=0; fail=0
while read -r size path sha; do
  out="$DEST/$path"
  have=0
  [ -f "$out" ] && have=$(stat -c %s "$out")
  need=0
  if [ "$have" != "$size" ]; then
    need=1
  elif [ "$sha" != "-" ] && [ "$(sha256sum "$out" | cut -d' ' -f1)" != "$sha" ]; then
    need=1
  fi
  if [ "$need" = 0 ]; then ok=$((ok + 1)); continue; fi
  echo "REFETCH $path have=$have want=$size"
  rm -f "$out"
  mkdir -p "$(dirname "$out")"
  curl --noproxy '*' -L -s -S --retry 5 --retry-delay 5 --connect-timeout 30 \
       --speed-time 120 --speed-limit 1024 -o "$out" "$MS/models/$M/resolve/master/$path"
  have=$(stat -c %s "$out" 2>/dev/null || echo 0)
  if [ "$have" = "$size" ] && { [ "$sha" = "-" ] || [ "$(sha256sum "$out" | cut -d' ' -f1)" = "$sha" ]; }; then
    echo "FIXED $path"; fixed=$((fixed + 1))
  else
    echo "STILL-BAD $path $have/$size"; fail=$((fail + 1))
  fi
done < "$list"
echo "=== $(date -Is) verify done: ok=$ok fixed=$fixed fail=$fail"
du -sh "$DEST"

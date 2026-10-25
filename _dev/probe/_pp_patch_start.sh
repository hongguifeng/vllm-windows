#!/usr/bin/env bash
# WSL fix: docker's --gpus device= selector is ignored here (the container sees
# every GPU), so pin the cards with CUDA_VISIBLE_DEVICES instead.
set -euo pipefail
S=/home/hong/pp-recipe/start.sh
cp -n "$S" "$S.orig"
python3 - "$S" <<'PY'
import sys
p = sys.argv[1]
src = open(p).read()
old = 'docker run -d --name "$NAME" --gpus "\\"device=$GPUS\\"" --shm-size 1g'
new = 'docker run -d --name "$NAME" --gpus all -e CUDA_VISIBLE_DEVICES="$GPUS" --shm-size 1g'
if old in src:
    src = src.replace(old, new)
    open(p, 'w').write(src)
    print("patched")
elif 'CUDA_VISIBLE_DEVICES' in src:
    print("already patched")
else:
    print("PATTERN NOT FOUND - inspect manually")
PY
echo "--- result ---"
grep -n "docker run -d" "$S"
bash -n "$S" && echo "syntax ok"

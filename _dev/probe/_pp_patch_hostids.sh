#!/usr/bin/env bash
# WSL lacks pinned host memory, so the fork's PLE host-ids path cannot bind;
# fall back to the stock sync path by turning the flag off.
set -euo pipefail
S=/home/hong/pp-recipe/start.sh
python3 - "$S" <<'PY'
import sys
p = sys.argv[1]
src = open(p).read()
if 'VLLM_PLE_HOST_IDS=0' in src:
    print('already off')
elif 'VLLM_PLE_HOST_IDS=1' in src:
    open(p, 'w').write(src.replace('VLLM_PLE_HOST_IDS=1', 'VLLM_PLE_HOST_IDS=0'))
    print('patched to 0')
else:
    print('marker not found')
PY
grep -n "VLLM_PLE_HOST_IDS\|CUDA_VISIBLE_DEVICES" "$S"
bash -n "$S" && echo "syntax ok"

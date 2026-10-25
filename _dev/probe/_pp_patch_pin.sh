#!/usr/bin/env bash
# This WSL2 kernel supports pinned memory but vLLM gates it behind
# VLLM_WSL2_ENABLE_PIN_MEMORY (default 0). Enable it, and restore the fork's
# PLE host-ids path now that UVA-backed buffers exist.
set -euo pipefail
S=/home/hong/pp-recipe/start.sh
python3 - "$S" <<'PY'
import sys
p = sys.argv[1]
src = open(p).read()
old = '  -e VLLM_PLE_HOST_IDS=0 \\\n'
new = '  -e VLLM_PLE_HOST_IDS=1 \\\n  -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 \\\n'
if 'VLLM_WSL2_ENABLE_PIN_MEMORY' in src:
    print('already set')
elif old in src:
    open(p, 'w').write(src.replace(old, new))
    print('patched: pinned memory on, host-ids on')
else:
    print('marker not found - dumping env block')
PY
grep -n "VLLM_PLE_HOST_IDS\|VLLM_WSL2_ENABLE_PIN_MEMORY\|VLLM_PLE_MMAP" "$S"
bash -n "$S" && echo "syntax ok"

#!/usr/bin/env bash
# Run the 27B benchmark twice and keep the second (warm) run.
#   usage: _bench.sh <tag> [extra env assignments...]
set -u
TAG="${1:-run}"
shift || true

# MSYS /d/... paths are meaningless to the Windows interpreter -- hand it C:\...
REPO_WIN="$(cygpath -w /d/code/vllm-windows)"
PY="/d/code/vllm-windows/.venv/Scripts/python.exe"
BENCH_WIN="$REPO_WIN\\_dev\\bench\\_bench_27b.py"
LOG_DIR="/d/code/vllm-windows/_dev/out"

export PATH="/d/code/vllm-windows/.venv/Scripts:$PATH"
export PYTHONUNBUFFERED=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

export VBENCH_MODEL="D:\\models\\Qwen3.8-27B-W4A16-AutoRound-fast"
export VMAXLEN=16384
export VMEM=0.90
export VMAXSEQS=8
export VBATCHED=4096

cd /c/Users/hong || exit 1

for i in 1 2; do
  export VBENCH_TAG="${TAG}_r${i}"
  echo "########## ${TAG} run ${i} ##########"
  "$PY" -u "$BENCH_WIN" > "$LOG_DIR/_bench_${TAG}_r${i}.log" 2>&1
  echo "rc=$? log=_bench_${TAG}_r${i}.log"
done

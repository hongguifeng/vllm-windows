#!/usr/bin/env bash
# Native-Windows launcher that mirrors the WSL project's `.env` single-user
# profile as closely as 0.29 allows. This exists for the WSL<->Windows A/B;
# `_run.ps1` is still the everyday launcher.
#
# Every flag here has a source in `~/testcode/qwen38-27b-rtx3090/single-user/
# start_qwen.sh` + `.env`; the mapping and the things that do NOT port are
# written up in AB_WSL_VS_WINDOWS.md.
#
# Vision: the .env's VISION=1 needs the 0.85 GiB tower out of GPU memory, which
# WSL does with patches/vision-tower-cpu-offload.patch. That patch is not
# portable *as a patch* (it edits qwen3_vl.py/__init__ internals), but it is not
# needed either: 0.29 upstreamed the same thing as OffloadConfig.
#
#   --cpu-offload-gb / --cpu-offload-params  -> UVAOffloader
#   interfaces.py::_mark_tower_model         -> auto-routes the tower through it
#   qwen3_5.py:515 vs :523                   -> tower is built before the LM, so
#                                               the tower eats the budget first
#                                               (the patch's tower-only budget)
#
# AB_VISION_OFFLOAD=1 turns that on (requires dropping AB_TEXT_ONLY).
# VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1 selects the bulk-copy path; the patch
# sets the same flag in code ("UVA zero-copy makes every GEMM reread operand
# tiles over PCIe and is much slower here").
#
# The one remaining deviation is --max-model-len: DFLASH_MAX_LEN=71680 costs
# 6.37 GiB of KV where the .env's 6000000000-byte pin leaves 5.58 GiB (0.28 fit
# 75,181 tokens in that same 5.58 GiB; 0.29 charges ~20% more per token).
# 60928 is the ceiling 0.29 reports for that pin. AB_VISION_OFFLOAD=1 frees the
# tower's budget again, which is what puts 71680 back in reach.
#
#   AB_DROP_MAMBA=1   leave out --mamba-ssm-cache-dtype float16
#                     (project note: it turns the fused GDN kernel into Triton)
#   AB_TEXT_ONLY=1    --language-model-only instead of the vision tower
#   AB_VISION_OFFLOAD=1
#                     --cpu-offload-gb (AB_VISION_OFFLOAD_GB, default 1) at
#                     AB_VISION_OFFLOAD_PARAMS (default "visual")
#   AB_PORT=...       default 8000
set -u

REPO=/d/code/vllm-windows
PY="$REPO/.venv/Scripts/python.exe"
PORT=${AB_PORT:-8000}

export PATH="$REPO/.venv/Scripts:$PATH"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONUTF8=1
export TORCH_CUDA_ARCH_LIST=8.0
export VLLM_TARGET_DEVICE=cuda

# .env: INT8_ACT=int8 INT8_LAYERS=mlp
export VLLM_MARLIN_INPUT_DTYPE=int8
export VLLM_MARLIN_INT8_INCLUDE_RE=mlp
# .env: CTX=fast -> the launcher exports VLLM_SPEC_DECODE_ATTN=1
export VLLM_SPEC_DECODE_ATTN=1
# .env: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
# .env: VLLM_WSL2_ENABLE_PIN_MEMORY=1 -- a no-op on native Windows, where
# CudaPlatform.is_pin_memory_available() returns True before reading it.

MAMBA=()
[ "${AB_DROP_MAMBA:-0}" = 1 ] || MAMBA=(--mamba-ssm-cache-dtype float16)

if [ "${AB_TEXT_ONLY:-0}" = 1 ]; then
  VISION=(--language-model-only)
else
  # .env: VISION=1 with the shipped pixel cap.
  VISION=(--limit-mm-per-prompt '{"image":{"count":32}}'
          --mm-processor-kwargs '{"size":{"shortest_edge":65536,"longest_edge":1048576}}')
fi

# 0.29's upstream equivalent of vision-tower-cpu-offload.patch; see the header.
# --cpu-offload-params takes nargs='+', so this group must stay last in argv.
VISION_OFFLOAD=()
if [ "${AB_VISION_OFFLOAD:-0}" = 1 ]; then
  if [ "${AB_TEXT_ONLY:-0}" = 1 ]; then
    echo "[ab] AB_VISION_OFFLOAD=1 is meaningless with AB_TEXT_ONLY=1" >&2
  fi
  export VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1   # bulk-copy path, not zero-copy
  VISION_OFFLOAD=(--cpu-offload-gb "${AB_VISION_OFFLOAD_GB:-1}"
                  --cpu-offload-params ${AB_VISION_OFFLOAD_PARAMS:-visual})
  echo "[ab] vision tower offload ON: ${VISION_OFFLOAD[*]} (UVA disabled)" >&2
fi

cd /c/Users/hong   # convention 5: keep the repo's vllm/ source tree off sys.path

# .env: EXTRA_ARGS carries --kv-offloading-size 16 --kv-offloading-backend native.
# That backend mmaps /dev/shm/vllm_offload_<engine_id>.mmap with no platform
# check (vllm/v1/kv_offload/cpu/shared_offload_region.py:98), so on native
# Windows it dies with FileNotFoundError before serving. WSL2 reported
# kv_offload_cpu_cache_usage_perc=0.0 for every call in this workload, so the
# region was never exercised -- leaving it out does not move the numbers.
OFFLOAD=()
if [ -d /dev/shm ]; then
  OFFLOAD=(--kv-offloading-size 16 --kv-offloading-backend native)
else
  echo "[ab] /dev/shm absent: dropping --kv-offloading-* (Linux-only backend)" >&2
fi

# WSL 0.28 default-runs --no-async-scheduling for DFlash2 (ASYNC_SCHED=0 in
# start_qwen.sh). Set AB_NO_ASYNC=1 to mirror that and A/B the async path.
if [ "${AB_NO_ASYNC:-0}" = 1 ]; then
  ASYNC=(--no-async-scheduling)
else
  ASYNC=(--async-scheduling)
fi

# AB_NOSPEC=1 drops DFlash2 entirely (bisect: spec-decode x no-async penalty).
SPEC=()
if [ "${AB_NOSPEC:-0}" != 1 ]; then
  SPEC=(--speculative-config '{"method":"dflash","model":"D:/models/Qwen3.8-27B-DFlash2-W4A16","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}')
fi

if [ "${AB_DRYRUN:-0}" = 1 ]; then
  set -- "$PY" -u -m vllm.entrypoints.openai.api_server \
    --model "D:/models/Qwen3.8-27B-W4A16-AutoRound-fast" \
    --served-model-name qwen3.8-27b \
    --host 127.0.0.1 --port "$PORT" \
    --gpu-memory-utilization 0.93 \
    --max-model-len "${AB_MAX_LEN:-60928}" \
    --max-num-seqs 8 --api-server-count 1 \
    "${VISION[@]}" \
    --kv-cache-dtype bfloat16 "${MAMBA[@]}" \
    --max-num-batched-tokens 2048 \
    "${OFFLOAD[@]}" --kv-cache-memory-bytes "${AB_KV_BYTES:-6000000000}" \
    "${VISION_OFFLOAD[@]}"
  printf '%s\n' "$@"
  exit 0
fi

exec "$PY" -u -m vllm.entrypoints.openai.api_server \
  --model "D:/models/Qwen3.8-27B-W4A16-AutoRound-fast" \
  --served-model-name qwen3.8-27b \
  --host 127.0.0.1 --port "$PORT" \
  --gpu-memory-utilization 0.93 \
  --max-model-len "${AB_MAX_LEN:-60928}" \
  --max-num-seqs 8 \
  --api-server-count 1 \
  "${VISION[@]}" \
  --attention-backend FLASH_ATTN --kv-cache-dtype bfloat16 \
  "${MAMBA[@]}" \
  "${ASYNC[@]}" \
  --max-num-batched-tokens 2048 \
  "${SPEC[@]}" \
  --compilation-config '{"max_cudagraph_capture_size":64,"custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  "${OFFLOAD[@]}" \
  --kv-cache-memory-bytes "${AB_KV_BYTES:-6000000000}" \
  --enable-prefix-caching --mamba-cache-mode align \
  --sse-keep-alive-interval 30 \
  "${VISION_OFFLOAD[@]}"

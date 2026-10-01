#!/usr/bin/env bash
# One-off check that start_server.ps1's argv actually comes up. Mirrors the runtime
# half of _env.ps1 (the build half -- INCLUDE/LIB/MAX_JOBS -- is not needed to
# serve, but cl.exe must stay reachable for the flashinfer/torch.compile JIT).
set -u

REPO=/d/code/vllm-windows
PY="$REPO/.venv/Scripts/python.exe"
LOG="$REPO/_dev/out/_serve_live.log"
MSVC='C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207'
SDK='C:\Program Files (x86)\Windows Kits\10'
SDKVER=10.0.26100.0
CUDA='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'

export PATH="$MSVC/bin/Hostx64/x64:$SDK/bin/$SDKVER/x64:$REPO/.venv/Scripts:$CUDA/bin:$PATH"
export VCToolsInstallDir="$MSVC/"
export VCToolsVersion=14.44.35207
export WindowsSdkDir="$SDK/"
export WindowsSDKVersion=$SDKVER
export UCRTVersion=$SDKVER
export WindowsSdkBinPath="$SDK/bin/$SDKVER/"
export DISTUTILS_USE_SDK=1
export TORCH_CUDA_ARCH_LIST=8.0
export VLLM_TARGET_DEVICE=cuda
export CUDA_HOME="$CUDA" CUDA_PATH="$CUDA" CUDA_ROOT="$CUDA"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONUTF8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export VLLM_SPEC_DECODE_ATTN=1
export VLLM_MARLIN_INPUT_DTYPE=int8
export VLLM_MARLIN_INT8_INCLUDE_RE=mlp
export VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1

cd /c/Users/hong   # convention 5

"$PY" -u -m vllm.entrypoints.openai.api_server \
  --model 'D:\models\Qwen3.8-27B-W4A16-AutoRound-fast' \
  --served-model-name qwen3.8-27b \
  --host 127.0.0.1 --port 8000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.93 \
  --max-model-len 71680 \
  --max-num-seqs 8 --max-num-batched-tokens 2048 \
  --attention-backend FLASH_ATTN --kv-cache-dtype bfloat16 \
  --async-scheduling --enable-prefix-caching --mamba-cache-mode align \
  --enable-prompt-tokens-details \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --compilation-config '{"max_cudagraph_capture_size":64,"custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --kv-cache-memory-bytes 6000000000 --sse-keep-alive-interval 30 \
  --limit-mm-per-prompt '{"image":{"count":32}}' \
  --mm-processor-kwargs '{"size":{"shortest_edge":65536,"longest_edge":1048576}}' \
  --mamba-ssm-cache-dtype float16 \
  --speculative-config '{"method": "dflash", "model": "D:/models/Qwen3.8-27B-DFlash2-W4A16", "num_speculative_tokens": 7, "draft_sample_method": "probabilistic"}' \
  --cpu-offload-gb 1 --cpu-offload-params visual \
  > "$LOG" 2>&1 &
SRV=$!

for i in $(seq 1 45); do
  if curl -sf -m 3 http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "HEALTHY after $((i * 10))s"
    break
  fi
  sleep 10
done

curl -sf -m 5 http://127.0.0.1:8000/v1/models | head -c 400
echo
kill $SRV 2>/dev/null
sleep 5
pkill -9 -f "vllm.entrypoints.openai.api_server" 2>/dev/null
echo "--- key log lines ---"
grep -n "Sliding-window bucket\|padding layers\|GPU KV cache size\|Model loading took\|GDN decode kernel\|Application startup complete\|Traceback" "$LOG" | head -20

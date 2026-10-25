#!/usr/bin/env bash
# Inspect the pinned-memory fallback and the CUDA-graph copy guard.
IMG=ghcr.io/alexbrandes/cmp170hx-qwen38-flash-next@sha256:d6fc872d943b3e9e5cc9eb583b51d72af200e039fbbb7401700bb38d665f0957
docker run --rm --entrypoint bash "$IMG" -c '
set -e
echo "=== guard string ==="
grep -rn "CPU tensor is pinned" /usr/local/lib/python3.12/dist-packages/vllm | head -3 || true
echo "=== buffer_utils 25-60 ==="
sed -n "25,60p" /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/buffer_utils.py
echo "=== is_pin_memory_available ==="
grep -rn "def is_pin_memory_available" -A 14 /usr/local/lib/python3.12/dist-packages/vllm/utils/ | head -40
'

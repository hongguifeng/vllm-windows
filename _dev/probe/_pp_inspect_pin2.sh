#!/usr/bin/env bash
# Why does the CUDA platform report pinned memory as unavailable here?
IMG=ghcr.io/alexbrandes/cmp170hx-qwen38-flash-next@sha256:d6fc872d943b3e9e5cc9eb583b51d72af200e039fbbb7401700bb38d665f0957
docker run --rm --entrypoint bash "$IMG" -c '
set -e
echo "=== cuda platform impl ==="
grep -rn "is_pin_memory_available" -A 8 /usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py
echo "=== interface default ==="
grep -rn "def is_pin_memory_available" -A 8 /usr/local/lib/python3.12/dist-packages/vllm/platforms/interface.py
echo "=== wsl detection ==="
grep -rn "def is_wsl" -A 12 /usr/local/lib/python3.12/dist-packages/vllm/platforms/*.py | head -30
'

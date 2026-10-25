#!/usr/bin/env bash
# Full body of the WSL pinned-memory gate plus this kernel's version string.
IMG=ghcr.io/alexbrandes/cmp170hx-qwen38-flash-next@sha256:d6fc872d943b3e9e5cc9eb583b51d72af200e039fbbb7401700bb38d665f0957
echo "kernel: $(uname -r)"
docker run --rm --entrypoint bash "$IMG" -c '
set -e
sed -n "300,345p" /usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py
echo "=== _get_wsl_kernel_version ==="
grep -rn "def _get_wsl_kernel_version" -A 20 /usr/local/lib/python3.12/dist-packages/vllm/ | head -30
'

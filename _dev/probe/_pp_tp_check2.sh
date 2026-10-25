#!/usr/bin/env bash
# What happens when tp_size != 1 in the PLE mmap backend?
IMG=ghcr.io/alexbrandes/cmp170hx-qwen38-flash-next@sha256:d6fc872d943b3e9e5cc9eb583b51d72af200e039fbbb7401700bb38d665f0957
docker run --rm --entrypoint bash "$IMG" -c '
set -e
F=/usr/local/lib/python3.12/dist-packages/vllm_ple_mmap_qwen4.py
echo "=== around 925-950 ==="
sed -n "925,950p" "$F"
echo "=== around 1000-1025 ==="
sed -n "1000,1025p" "$F"
'

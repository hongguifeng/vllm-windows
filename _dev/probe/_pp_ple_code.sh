#!/usr/bin/env bash
# Does the PLE mmap backend drop pages after reading, or bypass the page cache?
IMG=ghcr.io/alexbrandes/cmp170hx-qwen38-flash-next@sha256:d6fc872d943b3e9e5cc9eb583b51d72af200e039fbbb7401700bb38d665f0957
docker run --rm --entrypoint bash "$IMG" -c '
F=/usr/local/lib/python3.12/dist-packages/vllm_ple_mmap_qwen4.py
echo "=== cache-bypass / drop hints ==="
grep -nE "fadvise|DONTNEED|O_DIRECT|madvise|MADV_|drop_cache|posix_fadvise" "$F" | head -20
echo "=== mmap setup ==="
grep -nE "mmap\(|ACCESS_|PROT_|MAP_" "$F" | head -10
echo "=== how many n-gram orders per lookup ==="
grep -nE "ngram|orders|for k in|range\(.*gram" "$F" | head -20
'

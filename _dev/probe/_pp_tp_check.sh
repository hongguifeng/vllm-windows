#!/usr/bin/env bash
# Does the fork support TP for this model, or only PP?
IMG=ghcr.io/alexbrandes/cmp170hx-qwen38-flash-next@sha256:d6fc872d943b3e9e5cc9eb583b51d72af200e039fbbb7401700bb38d665f0957
docker run --rm --entrypoint bash "$IMG" -c '
set -e
PKG=/usr/local/lib/python3.12/dist-packages
echo "=== PLE backend: parallel-config assumptions ==="
grep -nE "tensor_parallel|tp_size|pipeline_parallel|pp_size|pp_rank|stage" $PKG/vllm_ple_mmap_qwen4.py | head -20
echo
echo "=== model_state overlay: gates ==="
grep -nE "tensor_parallel|tp_size|pipeline_parallel|pp_size|pp_rank|raise " $PKG/vllm/models/qwen4_exp/nvidia/model_state.py | head -20
echo
echo "=== qwen4_exp model: parallelism assumptions ==="
grep -rnE "tensor_parallel_size|pipeline_parallel_size" $PKG/vllm/models/qwen4_exp/nvidia/*.py | head -15
'

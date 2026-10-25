#!/usr/bin/env bash
# Determine how to restrict the container to the two 170HX cards under WSL.
IMG=ghcr.io/alexbrandes/cmp170hx-qwen38-flash-next@sha256:d6fc872d943b3e9e5cc9eb583b51d72af200e039fbbb7401700bb38d665f0957
CODE='import torch;print(torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'
echo "--- baseline: --gpus all ---"
docker run --rm --gpus all --entrypoint python3 "$IMG" -c "$CODE"
echo "--- -e CUDA_VISIBLE_DEVICES=1,2 ---"
docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES=1,2 --entrypoint python3 "$IMG" -c "$CODE"
echo "--- -e NVIDIA_VISIBLE_DEVICES=1,2 ---"
docker run --rm -e NVIDIA_VISIBLE_DEVICES=1,2 --entrypoint python3 "$IMG" -c "$CODE"
echo "--- --gpus device=1,2 (as start.sh does) ---"
docker run --rm --gpus '"device=1,2"' --entrypoint python3 "$IMG" -c "$CODE"

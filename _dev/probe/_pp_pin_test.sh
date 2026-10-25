#!/usr/bin/env bash
# Does pinned host memory actually work in this WSL container?
IMG=ghcr.io/alexbrandes/cmp170hx-qwen38-flash-next@sha256:d6fc872d943b3e9e5cc9eb583b51d72af200e039fbbb7401700bb38d665f0957
cat > /tmp/pintest.py <<'PY'
import torch, traceback
print('torch', torch.__version__)
print('pin supported flag:', getattr(torch.cuda, 'is_pinned_supported', lambda: 'n/a')())
try:
    t = torch.empty(8, dtype=torch.long).pin_memory()
    print('pin_memory ok, is_pinned =', t.is_pinned())
except Exception as e:
    print('pin_memory raised:', type(e).__name__, e)
    traceback.print_exc()
try:
    import torch.cuda
    print('cudaHostAlloc probe:', end=' ')
    t2 = torch.empty(8, dtype=torch.long, pin_memory=True)
    print('ok', t2.is_pinned())
except Exception as e:
    print('raised', type(e).__name__, e)
PY
docker run --rm --gpus all -e CUDA_VISIBLE_DEVICES=1,2 -v /tmp/pintest.py:/tmp/pintest.py:ro \
  --entrypoint python3 "$IMG" /tmp/pintest.py 2>&1 | tail -15
echo "--- where the graph-capture guard lives ---"
docker run --rm --entrypoint bash "$IMG" -c "grep -rn 'during CUDA graph capture unless the CPU tensor is pinned' /usr/local/lib/python3.12/dist-packages/vllm | head -3"

#!/usr/bin/env bash
# Post-boot sanity: model list, one short completion, KV pool, GPU usage.
echo "=== /v1/models ==="
curl -s --max-time 10 http://127.0.0.1:8000/v1/models | head -c 400; echo
echo "=== KV / memory lines ==="
docker logs hong-pc 2>&1 | grep -E "KV cache size|GPU KV cache|Available KV|free memory|graph capture|CUDA graph|max_num|Starting vLLM" | tail -8
echo "=== one completion ==="
curl -s --max-time 60 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-w4a16","messages":[{"role":"user","content":"What is 3+5? Answer with the number only."}],"max_tokens":24,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  | head -c 500; echo
echo "=== gpu ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

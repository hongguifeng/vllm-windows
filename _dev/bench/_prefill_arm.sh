#!/usr/bin/env bash
# Prefill TTFT arm probe: one engine, three lengths, warmup excluded, counters verified.
#   bash _dev/bench/_prefill_arm.sh <model-dir> <served-name> <port> <tag>
# Results: _dev/out/prefill_<tag>_<len>.json plus _dev/out/prefill_<tag>_summary.txt
set -uo pipefail
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL="*"
REPO="D:/code/vllm-windows"
VLLM="$REPO/.flashnext-root/.venv/Scripts/vllm.exe"
MODEL="$1"; NAME="$2"; PORT="$3"; TAG="$4"
NP="${5:-3}"; SEED_BASE="${6:-40200}"
OUT="$REPO/_dev/out"
LENS=(2048 8192 32768)

snapshot() { curl -s --max-time 20 "http://127.0.0.1:$PORT/metrics" > "$OUT/prefill_${TAG}_$1.metrics"; }

for L in "${LENS[@]}"; do
  # Warm the shape with a distinct prompt that is never measured (separate seed).
  "$VLLM" bench serve --backend openai --model "$MODEL" --served-model-name "$NAME" \
    --host 127.0.0.1 --port "$PORT" --endpoint /v1/completions \
    --dataset-name random --random-input-len "$L" --random-output-len 1 \
    --num-prompts 1 --max-concurrency 1 --temperature 0.0 --seed $((39000 + L)) \
    --disable-tqdm --num-warmups 0 --ready-check-timeout-sec 0 \
    > "$OUT/prefill_${TAG}_${L}_warm.log" 2>&1 || echo "warmup FAILED len=$L"
done

snapshot before
for L in "${LENS[@]}"; do
  "$VLLM" bench serve --backend openai --model "$MODEL" --served-model-name "$NAME" \
    --host 127.0.0.1 --port "$PORT" --endpoint /v1/completions \
    --dataset-name random --random-input-len "$L" --random-output-len 1 \
    --num-prompts "$NP" --max-concurrency 1 --temperature 0.0 --seed $((SEED_BASE + L)) \
    --disable-tqdm --num-warmups 0 --ready-check-timeout-sec 0 \
    --save-result --save-detailed --result-dir "$OUT" --result-filename "prefill_${TAG}_${L}.json" \
    > "$OUT/prefill_${TAG}_${L}_bench.log" 2>&1 || echo "bench FAILED len=$L"
done
snapshot after

python - "$OUT" "$TAG" <<'PY' | tee "$OUT/prefill_${TAG}_summary.txt"
import json, re, sys
out, tag = sys.argv[1], sys.argv[2]

def val(path, name):
    want = "vllm:" + name + "{"
    for line in open(path, "rb").read().decode("utf-8", "ignore").splitlines():
        if line.startswith(want):
            return float(line.rsplit(" ", 1)[1])
    return 0.0

print(f"==== arm {tag} ====")
print(f"{'len':<7}{'median_s':<11}{'rounds_s':<26}{'tok/s':<10}{'prefill_s':<10}")
for L in (2048, 8192, 32768):
    d = json.load(open(f"{out}/prefill_{tag}_{L}.json"))
    rounds = [round(t, 3) for t in d["ttfts"]]
    print(f"{L:<7}{d['median_ttft_ms']/1000:<11.3f}{str(rounds):<26}"
          f"{d['total_token_throughput']:<10.1f}{d['mean_ttft_ms']/1000:<10.3f}")

b, a = f"{out}/prefill_{tag}_before.metrics", f"{out}/prefill_{tag}_after.metrics"
for m in ("request_prompt_tokens_sum", "request_prefill_kv_computed_tokens_sum",
          "prefix_cache_hits_total", "request_num_preemptions_sum",
          "request_prefill_time_seconds_sum", "time_to_first_token_seconds_sum"):
    print(f"delta {m} = {val(a, m) - val(b, m):.2f}")
PY

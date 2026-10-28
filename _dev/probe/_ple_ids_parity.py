"""Are the CPU and CUDA `compute_ngram_ids` bit-identical?

Moving the PLE id computation to the host (so the SSD row read no longer waits
for an ids D2H) only works if the host math reproduces the Triton kernel
exactly: the SSD path *consumes* the prefetched rows, so a wrong id is a silently
wrong embedding, not a crash. This probe runs both paths over the shapes the
engine actually schedules -- single-token decode, DFlash-width decode batches,
chunked prefill, mixed requests, EOS inside the context and inside the chunk --
using the served checkpoint's real hash buffers, and reports the first mismatch.

    python _dev/probe/_ple_ids_parity.py --fuzz 200 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from safetensors import safe_open

MODEL = Path(r"D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP")


def load_ngram(cls, device: torch.device):
    cfg = json.loads((MODEL / "config.json").read_text(encoding="utf-8"))
    text = cfg.get("text_config", cfg)
    eos = int(text["eos_token_id"]) if "eos_token_id" in text else int(cfg["eos_token_id"])
    ngram_size = int(text["ngram_size"])
    heads_per_ngram = int(text["heads_per_ngram"])
    index = json.loads((MODEL / "model.safetensors.index.json").read_text(encoding="utf-8"))
    wanted = {
        "layer_multipliers": "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
        "ngram_heads_offsets": "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
        "ngram_heads_vocab_sizes": "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
    }
    module = cls.__new__(cls)
    torch.nn.Module.__init__(module)
    for attr, key in wanted.items():
        shard = index["weight_map"][key]
        with safe_open(str(MODEL / shard), framework="pt", device="cpu") as f:
            module.register_buffer(attr, f.get_tensor(key))
    module.ngram_size = ngram_size
    module.heads_per_ngram = heads_per_ngram
    module.ngram_heads = (ngram_size - 1) * heads_per_ngram  # type: ignore[assignment]
    module.eos_token_id = eos
    module.split_ngram_parts = int(text.get("split_ngram_parts", 1))
    # Move the hash buffers only; ngram ids need nothing else.
    for attr in wanted:
        setattr(module, attr, getattr(module, attr).to(device))
    return module, eos, ngram_size


def case_tensors(query_lens, contexts, base, eos_positions, eos, dtype, device):
    qsl = torch.tensor([0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=dtype, device=device)
    n = int(qsl[-1])
    ids = torch.arange(base, base + n, dtype=dtype, device=device)
    for pos in eos_positions:
        ids[pos] = eos
    ctx = torch.tensor(contexts, dtype=dtype, device=device)
    return ids, qsl, ctx


def compare(module_cpu, module_gpu, ids, qsl, ctx, device):
    cpu = module_cpu.compute_ngram_ids(ids.cpu(), qsl.cpu(), ctx.cpu())
    gpu = module_gpu.compute_ngram_ids(ids.to(device), qsl.to(device), ctx.to(device))
    return cpu, gpu.cpu()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fuzz", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261028)
    args = ap.parse_args()
    device = torch.device(args.device)

    from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpNGramEmbedding

    module_cpu, eos, ngram_size = load_ngram(Qwen4ExpNGramEmbedding, torch.device("cpu"))
    module_gpu, _, _ = load_ngram(Qwen4ExpNGramEmbedding, device)
    ctx_len = ngram_size - 1
    print(
        f"buffers: {module_cpu.ngram_heads} heads, ctx_len={ctx_len}, eos={eos}, "
        f"vocab sizes {module_cpu.ngram_heads_vocab_sizes[:4].tolist()}..."
    )

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    cases: list[tuple[str, list[int], list[list[int]], int, list[int]]] = [
        ("decode/1req/1tok", [1], [[1000, 1001]], 5000, []),
        ("decode/4req/8tok (dflash width)", [8] * 4, [[11, 12], [13, 14], [15, 16], [17, 18]], 7000, []),
        ("decode/eos-in-context", [6, 6], [[eos, 42], [43, eos]], 8000, []),
        ("decode/eos-in-chunk", [8, 8], [[1, 2], [3, 4]], 9000, [3, 11]),
        ("prefill/1req/2048tok", [2048], [[101, 102]], 10_000, []),
        ("prefill/chunk2-of-4", [2048], [[20_001, 20_002]], 20_000, []),
        ("mixed/1+7+2048", [1, 7, 2048], [[31, 32], [33, 34], [35, 36]], 40_000, [40]),
        ("int32-ceiling", [3, 5], [[2_000_000_000, 2_000_000_001], [2_000_000_002, 2_000_000_003]], 2_000_000_004, []),
    ]

    failures = 0
    for name, lens, contexts, base, eos_pos in cases:
        ids, qsl, ctx = case_tensors(lens, contexts, base, eos_pos, eos, torch.int32, device)
        cpu, gpu = compare(module_cpu, module_gpu, ids, qsl, ctx, device)
        ok = torch.equal(cpu, gpu)
        failures += 0 if ok else 1
        print(f"  {'OK  ' if ok else 'FAIL'} {name:28} tokens={int(qsl[-1]):5d} shape={tuple(cpu.shape)}")
        if not ok:
            bad = (cpu != gpu).any(dim=1).nonzero().flatten()[:5].tolist()
            print(f"       first bad token rows: {bad}")
            for r in bad[:2]:
                print(f"       row {r}: cpu={cpu[r].tolist()} gpu={gpu[r].tolist()}")

    print(f"fuzzing {args.fuzz} random batches ...")
    for i in range(args.fuzz):
        nreq = rng.randint(1, 8)
        lens = [rng.choice([0, 1, 1, 2, 3, 5, 7, 8, 8, 16, 31, 64, 257]) for _ in range(nreq)]
        if not any(lens):
            lens[0] = 1
        ctxs = []
        for _ in range(nreq):
            row = [rng.randrange(0, 250_000) for _ in range(ctx_len)]
            if rng.random() < 0.15:
                row[rng.randrange(ctx_len)] = eos
            ctxs.append(row)
        base = rng.choice([0, 1, 1000, 200_000, 248_000])
        total = sum(lens)
        eos_pos = [p for p in range(total) if rng.random() < 0.02]
        ids, qsl, ctx = case_tensors(lens, ctxs, base, eos_pos, eos, torch.int32, device)
        cpu, gpu = compare(module_cpu, module_gpu, ids, qsl, ctx, device)
        if not torch.equal(cpu, gpu):
            failures += 1
            bad = (cpu != gpu).any(dim=1).nonzero().flatten()[:3].tolist()
            print(f"  FAIL fuzz #{i}: lens={lens} base={base} eos_pos={eos_pos[:5]} rows={bad}")
            break

    print(f"\nRESULT: {'PARITY OK' if failures == 0 else f'{failures} FAILURES'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())


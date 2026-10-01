"""Does flashinfer.top_k ever return out-of-range indices?

vLLM's DFlash2 candidate path calls exactly:

    flashinfer.top_k(logits, k=16, sorted=True, deterministic=True)

with logits of shape [num_reqs*7, 248320] (bf16), and feeds the returned
indices straight into a [248320, 256] codebook gather.  Anything outside
[0, 248320) is a device-side assert.

This script hammers that exact call with a handful of adversarial inputs.
"""

import torch
import flashinfer

torch.manual_seed(0)
V = 248320
K = 16
dev = "cuda"


def check(tag, x, k=K):
    v, idx = flashinfer.top_k(x, k, sorted=True, deterministic=True)
    lo, hi = int(idx.min()), int(idx.max())
    bad = ((idx < 0) | (idx >= V)).sum().item()
    tv, ti = torch.topk(x, k, dim=-1, sorted=True)
    # index sets must match; compare the *values* they select instead of ids
    sel_flash = torch.gather(x, -1, idx.clamp(0, V - 1).clamp(min=0))
    # (clamped gather is only a sanity check, ids themselves are what matter)
    mismatch = (sel_flash != tv).sum().item() if bad == 0 else -1
    print(
        f"{tag:34s} rows={x.shape[0]:3d} idx_range=[{lo},{hi}] out_of_range={bad:4d}"
        f" value_mismatch_vs_torch={mismatch}"
    )
    return bad


print("flashinfer:", flashinfer.__version__, "| cuda:", torch.cuda.get_device_name(0))

for rows in (7, 14, 28, 56):
    x = torch.randn(rows, V, dtype=torch.bfloat16, device=dev)
    check(f"randn bf16 rows={rows}", x)

# tail padding to -inf (get_top_k_tokens does this when num_org_vocab_padding > 0)
x = torch.randn(14, V, dtype=torch.bfloat16, device=dev)
x[:, -275:] = float("-inf")
check("tail 275 x -inf", x)

# an all -inf row
x = torch.randn(14, V, dtype=torch.bfloat16, device=dev)
x[3, :] = float("-inf")
check("row3 all -inf", x)

# NaN contamination
x = torch.randn(14, V, dtype=torch.bfloat16, device=dev)
x[5, 1000:1010] = float("nan")
check("row5 has NaN", x)

# heavy ties across the whole row
x = torch.zeros(14, V, dtype=torch.bfloat16, device=dev)
check("all-zero row (full tie)", x)

# huge magnitudes (overflow-ish bf16)
x = torch.randn(14, V, dtype=torch.bfloat16, device=dev) * 3e38
check("randn * 3e38 (bf16 inf)", x)

# repeat determinism
x = torch.randn(56, V, dtype=torch.bfloat16, device=dev)
first = flashinfer.top_k(x, K, sorted=True, deterministic=True)[1]
worst = 0
for _ in range(20):
    cur = flashinfer.top_k(x, K, sorted=True, deterministic=True)[1]
    worst = max(worst, (cur != first).sum().item())
    worst = max(worst, int(cur.max()))
print(f"{'20x repeat':34s} max_id_seen={worst} (deterministic=True)")

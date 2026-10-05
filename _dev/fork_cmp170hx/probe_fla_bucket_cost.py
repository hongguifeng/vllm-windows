# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""How much the T=64 warmup costs a real GDN prefill on this card.

The GDN layer warms every autotuner with a single T=FLA_CHUNK_SIZE (64) pass, and
the autotune key does not contain the token count, so the config picked for a
64-token launch is what every later prefill length uses. This measures the
difference: time a 2048-token prefill with the cache warmed at T=64, then clear
the cache, warm at T=2048, and time the same launch again.

Usage:
    CUDA_VISIBLE_DEVICES=1 python _dev/fork_cmp170hx/probe_fla_bucket_cost.py
"""

import torch

from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.pinned_autotune import REGISTRY
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
from vllm.triton_utils import triton

HEAD_K_DIM = 128
HEAD_V_DIM = 128
NUM_K_HEADS = 16
NUM_V_HEADS = 48
T_REAL = 2048


def _tensors(t: int, device: str = "cuda"):
    dtype = torch.bfloat16
    q = torch.randn(1, t, NUM_K_HEADS, HEAD_K_DIM, device=device, dtype=dtype)
    k = torch.randn(1, t, NUM_K_HEADS, HEAD_K_DIM, device=device, dtype=dtype)
    v = torch.randn(1, t, NUM_V_HEADS, HEAD_V_DIM, device=device, dtype=dtype)
    g = -torch.rand(1, t, NUM_V_HEADS, device=device, dtype=dtype).abs()
    beta = torch.rand(1, t, NUM_V_HEADS, device=device, dtype=dtype)
    state = torch.zeros(
        1, NUM_V_HEADS, HEAD_V_DIM, HEAD_K_DIM, device=device, dtype=torch.float32
    )
    cu_seqlens = torch.tensor([0, t], device=device, dtype=torch.int32)
    return dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )


def _clear_caches() -> None:
    for tuner in REGISTRY.values():
        if hasattr(tuner, "cache"):
            tuner.cache.clear()


def _winners() -> dict[str, str]:
    out = {}
    for name, tuner in REGISTRY.items():
        for key, config in getattr(tuner, "cache", {}).items():
            kwargs = dict(config.all_kwargs())
            out[name] = (
                f"BV={kwargs.get('BV')} BK={kwargs.get('BK')} "
                f"num_warps={config.num_warps} num_stages={config.num_stages}"
            )
    return out


def main() -> None:
    warm64 = _tensors(FLA_CHUNK_SIZE)
    real = _tensors(T_REAL)

    _clear_caches()
    chunk_gated_delta_rule(**warm64)
    torch.accelerator.synchronize()
    t_warmed_64 = triton.testing.do_bench(
        lambda: chunk_gated_delta_rule(**real), warmup=5, rep=20
    )
    w64 = _winners()

    _clear_caches()
    chunk_gated_delta_rule(**real)
    torch.accelerator.synchronize()
    t_warmed_real = triton.testing.do_bench(
        lambda: chunk_gated_delta_rule(**real), warmup=5, rep=20
    )
    wreal = _winners()

    print(f"T={T_REAL} prefill, cache warmed at T=64 : {t_warmed_64:.3f} ms")
    print(f"T={T_REAL} prefill, cache warmed at T={T_REAL}: {t_warmed_real:.3f} ms")
    print(
        f"cost of the T-bucket mismatch: "
        f"{100.0 * (t_warmed_64 / t_warmed_real - 1.0):+.1f}%"
    )
    print("\nwinners when warmed at T=64:")
    for name in sorted(w64):
        print(f"  {name}: {w64[name]}")
    print(f"\nwinners when warmed at T={T_REAL}:")
    for name in sorted(wreal):
        print(f"  {name}: {wreal[name]}")


if __name__ == "__main__":
    main()

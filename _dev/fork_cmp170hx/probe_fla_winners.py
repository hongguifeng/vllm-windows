# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dump the fla autotune winners for the GDN prefill shapes this rig runs.

Runs ``chunk_gated_delta_rule`` the way the Qwen3.8 GDN layer does (T=64 warmup,
then a real prefill length) and prints, for every autotuned kernel in the path,
the config the Autotuner settled on for each cache key. Run it in several fresh
processes: if the winners differ between processes, the config -- and with it the
reduction order -- is not a function of the launch shape, which is what
VLLM_FLA_PIN_AUTOTUNE is meant to fix.

Usage:
    CUDA_VISIBLE_DEVICES=1 python _dev/fork_cmp170hx/probe_fla_winners.py
"""

import os

import torch

from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.pinned_autotune import REGISTRY
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

HEAD_K_DIM = 128
HEAD_V_DIM = 128
NUM_K_HEADS = 16
NUM_V_HEADS = 48


def _launch(t: int, device: str = "cuda") -> None:
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
    chunk_gated_delta_rule(
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
    torch.accelerator.synchronize()


def main() -> None:
    print(f"pid={os.getpid()} FLA_CHUNK_SIZE={FLA_CHUNK_SIZE}")
    _launch(FLA_CHUNK_SIZE)
    for t in (1152, 2048):
        _launch(t)

    for name in sorted(REGISTRY):
        tuner = REGISTRY[name]
        cache = getattr(tuner, "cache", {})
        for key, config in sorted(cache.items(), key=repr):
            kwargs = dict(config.all_kwargs())
            print(
                f"{name} key={key} -> BV={kwargs.get('BV')} "
                f"BK={kwargs.get('BK')} BT={kwargs.get('BT')} "
                f"num_warps={config.num_warps} num_stages={config.num_stages}"
            )


if __name__ == "__main__":
    main()

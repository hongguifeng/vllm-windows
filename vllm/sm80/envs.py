# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment variables of the Windows sm_80 fork.

``vllm/envs.py`` calls :func:`register_env_vars` at the end of its import, so
every name below is an ordinary vLLM environment variable in every process.
Only the EXL3 variables are carried over from the Linux fork; the GLM-5 /
sparse-MLA / all-reduce switches stay out until those stacks are ported.
"""

import os
from collections.abc import Callable
from typing import Any


def _env_with_choices(
    name: str, default: str, choices: list[str]
) -> Callable[[], str]:
    def read() -> str:
        value = os.getenv(name, default)
        if value not in choices:
            raise ValueError(f"{name}={value!r} must be one of {choices}")
        return value

    return read


FORK_ENV_VARS: dict[str, Callable[[], Any]] = {
    # EXL3 (vllm/sm80/kernels/exl3/quant.py, csrc/exl3).
    #
    # Dense EXL3 launches of 1..32 rows of one Qwen3.8-Flash-Next rank at TP2
    # take the split-K operator's kernel (csrc/exl3/dense_splitk.cu).
    "VLLM_EXL3_DENSE_SPLITK": lambda: bool(
        int(os.getenv("VLLM_EXL3_DENSE_SPLITK", "0"))
    ),
    # Tensor-core blocks for the split-K kernel's 1..32 row launches
    # (csrc/exl3/dense_mma.cu); requires VLLM_EXL3_DENSE_SPLITK.
    "VLLM_EXL3_DENSE_MMA": lambda: bool(int(os.getenv("VLLM_EXL3_DENSE_MMA", "0"))),
    # Diagnostic for VLLM_EXL3_DENSE_MMA: "MT,KW,SPLITS" plan override.
    "VLLM_EXL3_DENSE_MMA_SHAPE_OVERRIDE": lambda: os.getenv(
        "VLLM_EXL3_DENSE_MMA_SHAPE_OVERRIDE", ""
    ),
    # Dense EXL3 prefill launches over 64 rows (csrc/exl3/dense_prefill_tc.cu).
    "VLLM_EXL3_DENSE_PREFILL_TC": lambda: bool(
        int(os.getenv("VLLM_EXL3_DENSE_PREFILL_TC", "0"))
    ),
    # Diagnostic for VLLM_EXL3_DENSE_PREFILL_TC: "planned" or "whole".
    "VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE": lambda: _env_with_choices(
        "VLLM_EXL3_DENSE_PREFILL_TC_SCHEDULE", "planned", ["planned", "whole"]
    )(),
    # EXL3 routed experts on the tensor cores for calls of at most 64 rows
    # (csrc/exl3/moe_v2.cu).
    "VLLM_EXL3_MOE_V2": lambda: bool(int(os.getenv("VLLM_EXL3_MOE_V2", "0"))),
    # EXL3 routed experts on the tensor cores for 65..3460 row calls
    # (csrc/exl3/moe_prefill.cu).
    "VLLM_EXL3_MOE_PREFILL_TC": lambda: bool(
        int(os.getenv("VLLM_EXL3_MOE_PREFILL_TC", "0"))
    ),
    # Where the compressed PLE table lives: SSD row reads, host RAM, or VRAM.
    "VLLM_EXL3_PLE_PLACEMENT": lambda: _env_with_choices(
        "VLLM_EXL3_PLE_PLACEMENT", "ssd", ["ssd", "ram", "device"]
    )(),
    # Start an SSD run with the PLE table dropped from the page cache.
    "VLLM_EXL3_PLE_COLD": lambda: bool(int(os.getenv("VLLM_EXL3_PLE_COLD", "0"))),
    # Retries (1 ms backoff, doubling) for a failed PLE row read.
    "VLLM_EXL3_PLE_READ_RETRIES": lambda: int(
        os.getenv("VLLM_EXL3_PLE_READ_RETRIES", "3")
    ),
    # Diagnostic: time the host side of every SSD PLE prefetch.
    "VLLM_EXL3_PLE_TIMING": lambda: bool(
        int(os.getenv("VLLM_EXL3_PLE_TIMING", "0"))
    ),
    # Read the PLE rows at the PLE layer, overlapping the preceding layer.
    "VLLM_EXL3_PLE_OVERLAP": lambda: bool(
        int(os.getenv("VLLM_EXL3_PLE_OVERLAP", "0"))
    ),
    # One native call for a step's SSD rows instead of a Python thread pool.
    "VLLM_EXL3_PLE_NATIVE_READ": lambda: bool(
        int(os.getenv("VLLM_EXL3_PLE_NATIVE_READ", "0"))
    ),
    # Queue the SSD row reads as a host function on the PLE stream.
    "VLLM_EXL3_PLE_HOSTFN_READ": lambda: bool(
        int(os.getenv("VLLM_EXL3_PLE_HOSTFN_READ", "0"))
    ),
    # sm_80 thin-M BF16 GEMM (vllm/sm80/kernels/thin_gemm/) for the layers an
    # AutoRound 2-bit block leaves in BF16: hyper-connection mix/inject, router,
    # shared expert, GDN in_proj_a/b, PLE and lm_head.
    "VLLM_THIN_GEMM": lambda: bool(int(os.getenv("VLLM_THIN_GEMM", "0"))),
    # Largest row count dispatched to thin_gemm; above it cuBLAS wins on this part.
    "VLLM_THIN_GEMM_MAX_TOKENS": lambda: int(
        os.getenv("VLLM_THIN_GEMM_MAX_TOKENS", "32")
    ),
}


def register_env_vars(environment_variables: dict[str, Callable[[], Any]]) -> None:
    """Add the fork's variables to vLLM's table.

    Raises if upstream vLLM declares one of the fork's names itself, so a
    rebase that brings such a name cannot silently swap the reader.
    """
    for name, getter in FORK_ENV_VARS.items():
        existing = environment_variables.get(name)
        if existing is not None and existing is not getter:
            raise ValueError(
                f"{name} is declared by vLLM and by vllm/sm80/envs.py; "
                "keep one declaration"
            )
    environment_variables.update(FORK_ENV_VARS)

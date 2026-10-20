# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-tree registrations of the Windows sm_80 fork.

vLLM calls :func:`register` from :func:`vllm.plugins.load_general_plugins` in
every process (API server, engine core and each worker), so the EXL3
quantization method and load format exist before any engine config resolves
them. The function is idempotent.
"""

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_registered = False


def register() -> None:
    """Register the fork's quantization method and load format."""
    global _registered
    if _registered:
        return
    _register_exl3()
    _registered = True
    logger.info("sm80 plugin registered (EXL3 quantization and load format)")


def _register_exl3() -> None:
    """EXL3 checkpoints: quantization method "exl3" and load format "exl3"."""
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.model_loader import register_model_loader
    from vllm.sm80.kernels.exl3.loader import Exl3ModelLoader
    from vllm.sm80.kernels.exl3.quant import Exl3Config

    register_quantization_config("exl3")(Exl3Config)
    register_model_loader("exl3")(Exl3ModelLoader)


def is_registered() -> bool:
    """Whether :func:`register` ran in this process."""
    return _registered

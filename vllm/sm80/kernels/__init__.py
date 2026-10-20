# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernels tuned for NVIDIA sm_80 parts (GA100 class, e.g. the CMP 170HX).

``exl3`` is the quantization method and load format "exl3": its ``quant``,
``loader`` and ``utils`` modules are imported when the plugin registers it in
``vllm/sm80/plugin.py``, and ``ops`` (which loads the compiled
``vllm._exl3_C``) only when an EXL3 model is built or an EXL3 layer processes
its weights.
"""

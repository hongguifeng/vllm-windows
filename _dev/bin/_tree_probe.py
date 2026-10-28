# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Report which ``vllm`` the server would import, and whether it carries the two
Marlin patches that int8 activations depend on. Prints one marked, machine-
readable line (vLLM's import-time chatter can land on stdout, so the marker is
what lets the caller pick this line out of the noise):

    TREE|version|package_file|layer_select|negative_scales

``VLLM_MARLIN_INPUT_DTYPE=int8`` ("int8 act") needs both of them:

  layer_select  ``VLLM_MARLIN_INT8_INCLUDE_RE`` / ``_EXCLUDE_RE`` narrow the W4A8
                path to the MLP projections. Without them the regex is an unknown
                env var, every Linear runs W4A8 (decode -5.7%, init +143 s) and
                nothing warns.
  negative_scales  AutoRound symmetric exports ~50% negative group scales and the
                int8 Marlin kernel reads them as unsigned int16 -> garbage output.

Both are syv patches carried in the *installed* package. The checked-out ``vllm/``
tree in this repo has neither, and that tree wins ``sys.path[0]`` whenever python
is started inside the repo, so scrub both this script's directory and the caller's
cwd before importing -- same trick as ``_envcheck.py``.
"""

import os
import pathlib
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_CWD = os.path.abspath(os.getcwd())
sys.path = [p for p in sys.path if os.path.abspath(p or ".") not in (_HERE, _CWD)]

import vllm  # noqa: E402
import vllm.envs as envs  # noqa: E402

_SELECT = ("VLLM_MARLIN_INT8_INCLUDE_RE", "VLLM_MARLIN_INT8_EXCLUDE_RE")
layer_select = all(hasattr(envs, n) for n in _SELECT)

_marlin = (
    pathlib.Path(vllm.__file__).resolve().parent
    / "model_executor"
    / "kernels"
    / "linear"
    / "mixed_precision"
    / "marlin.py"
)
negative_scales = False
if _marlin.exists():
    _src = _marlin.read_text(encoding="utf-8", errors="ignore")
    negative_scales = "_ws.data < 0" in _src or "Fold the sign" in _src

print(
    "TREE|"
    + "|".join(
        [
            vllm.__version__,
            str(vllm.__file__),
            str(int(layer_select)),
            str(int(negative_scales)),
        ]
    )
)

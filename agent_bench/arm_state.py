# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

r"""Print "<preds> <empty>" for an arm's predictions file.

Used by supervise_arm.ps1 to decide when inference is complete. The two numbers
must be space separated; the supervisor parses `^(\d+)\s+(\d+)$`.
"""

import argparse
import json
import os

p = argparse.ArgumentParser()
p.add_argument("--out", required=True, help="arm output directory holding preds.json")
a = p.parse_args()

path = os.path.join(a.out, "preds.json")
if not os.path.exists(path):
    print("0 0")
else:
    with open(path) as f:
        data = json.load(f)
    empty = [k for k, v in data.items() if not (v.get("model_patch") or "").strip()]
    print(len(data), len(empty))

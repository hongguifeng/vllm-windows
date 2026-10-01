"""Execute the injected [idcheck-exp] block on CPU with fake ids.

Catches typos / bad tensor ops in the probe before spending a 4-minute boot.
"""

import os
import sys
import textwrap
import types

sys.path.insert(0, r"D:\code\vllm-windows")
import _idcheck_exp as M  # noqa: E402

BLOCK = textwrap.dedent(M.BLOCK)

os.environ["DFLASH2_IDCHECK"] = "1"

import torch  # noqa: E402


class FakeLogger:
    def warning(self, fmt, *a):
        print("  LOG:", (fmt % a) if a else fmt)


class FakeInner:
    def __init__(self):
        self.candidate_selector = types.SimpleNamespace(
            predecessor_codebook=torch.empty(248320, 1, dtype=torch.bool)
        )


class FakeModel:
    def __init__(self):
        self.model = FakeInner()


class FakeSelf:
    def __init__(self):
        self.model = FakeModel()
        self._anchor_indices = torch.arange(8) * 8


def run(cand, anchor, num_reqs, label):
    st = FakeSelf()
    ns = {
        "torch": torch,
        "self": st,
        "candidate_ids": cand,
        "anchor_token_ids": anchor,
        "num_reqs": num_reqs,
        "__import__": __import__,
    }
    print(label)
    for _ in range(25):
        exec(BLOCK, ns)
    return st


torch.manual_seed(0)
# 1) all in range -> only the every-25-steps range line
run(
    torch.randint(0, 248320, (2, 7, 16), dtype=torch.int64),
    torch.randint(0, 248320, (2,), dtype=torch.int32),
    2,
    "case 1: in-range",
)

# 2) garbage anchor (>= vocab) -> dump path must fire once
bad_anchor = torch.randint(0, 248320, (4,), dtype=torch.int32)
bad_anchor[2] = 2147483647
run(
    torch.randint(0, 248320, (4, 7, 16), dtype=torch.int64),
    bad_anchor,
    4,
    "case 2: anchor id = INT32_MAX",
)

# 3) garbage candidate (negative deep) -> dump path
bad_cand = torch.randint(0, 248320, (2, 7, 16), dtype=torch.int64)
bad_cand[0, 3, 5] = -2_147_483_648
run(bad_cand, torch.randint(0, 248320, (2,), dtype=torch.int32), 2, "case 3: cand id = INT32_MIN")

# 4) repro of the real kernel semantics: what does inductor's wrap do with those?
V = 248320
for v in (-1, -2, -V, -V - 1, 248320, 2147483647, -2_147_483_648):
    t = torch.tensor([v], dtype=torch.int64)
    wrapped = torch.where(t < 0, t + V, t)
    ok = bool(((wrapped >= 0) & (wrapped < V)).all())
    print(f"  wrap({v:>12}) -> {int(wrapped)}  in_range={ok}")

print("probe block OK")

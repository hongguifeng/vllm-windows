"""Offline self-test for the _clamp_exp.py blocks (no server, no model).

Checks the two things that decide whether the 26-minute repro run is worth
anything:

  1. BLOCK_A counts out-of-range ids in the right buckets and clamps them.
  2. BLOCK_A leaves in-range ids untouched.
  3. BLOCK_B reads the counter back from the "eager" side and logs it.

Run from OUTSIDE the repo tree (the source tree would shadow site-packages):

    cd C:/Users/hong
    D:/code/vllm-windows/.venv/Scripts/python.exe D:/code/vllm-windows/_selftest_clamp.py
"""

import importlib.util
import os
import sys
import textwrap
import types

os.environ["DFLASH2_CLAMP"] = "1"

import torch  # noqa: E402

# Load _clamp_exp.py by path: putting the repo root on sys.path would make the
# (uncompiled) source tree shadow site-packages and break `import vllm`.
_spec = importlib.util.spec_from_file_location(
    "_clamp_exp", r"D:\code\vllm-windows\_clamp_exp.py"
)
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

WARNINGS = []


class _FakeLog:
    def warning(self, fmt, *a):
        WARNINGS.append(fmt % a if a else fmt)


class _FakeCandidateSelector:
    def __init__(self, vocab):
        self.predecessor_codebook = torch.zeros(vocab, 256)


class _FakeModel:
    def __init__(self, vocab):
        self.candidate_selector = _FakeCandidateSelector(vocab)


class _FakeSpec:
    def __init__(self, vocab):
        # mirrors the real nesting used by the blocks:
        #   self.model.model.candidate_selector.predecessor_codebook
        self.model = types.SimpleNamespace(model=_FakeModel(vocab))
        self.input_buffers = types.SimpleNamespace()


def run_a(spec, cand, anchor):
    ns = {"self": spec, "torch": torch, "candidate_ids": cand,
          "anchor_token_ids": anchor, "num_reqs": int(anchor.shape[0])}
    exec(textwrap.dedent(M.BLOCK_A), ns)
    return ns["candidate_ids"], ns["anchor_token_ids"]


def run_b(spec, num_reqs=2, cg="FULL"):
    batch_desc = types.SimpleNamespace(cg_mode=cg)
    ns = {"self": spec, "torch": torch, "num_reqs": num_reqs,
          "batch_desc": batch_desc}
    exec(textwrap.dedent(M.BLOCK_B), ns)


def main() -> int:
    fails = 0

    # --- case 1: dirty ids must be counted per bucket and clamped -----------
    spec = _FakeSpec(vocab=8)
    cand = torch.tensor([[[0, 7, -1]], [[3, 8, 100]]], dtype=torch.int64)
    anchor = torch.tensor([[-5], [5]], dtype=torch.int64)
    c2, a2 = run_a(spec, cand, anchor)
    ctr = spec._clamp_ctr.tolist()
    print("case1 counter:", ctr)
    print("case1 clamped cand:", c2.tolist(), "anchor:", a2.tolist())
    for want, got, what in [(1, ctr[0], "cand<0"), (2, ctr[1], "cand>=vocab"),
                            (1, ctr[2], "anchor<0"), (0, ctr[3], "anchor>=vocab")]:
        if want != got:
            print(f"  FAIL {what}: want {want} got {got}")
            fails += 1
    if int(c2.min()) < 0 or int(c2.max()) > 7:
        print("  FAIL clamped candidate_ids still out of range")
        fails += 1
    if int(a2.min()) < 0 or int(a2.max()) > 7:
        print("  FAIL clamped anchor_token_ids still out of range")
        fails += 1

    # --- case 2: clean ids must not move the counter ------------------------
    spec2 = _FakeSpec(vocab=8)
    _c, _a = run_a(spec2, torch.tensor([[[0, 7, 3]]], dtype=torch.int64),
                   torch.tensor([[2]], dtype=torch.int64))
    ctr2 = spec2._clamp_ctr.tolist()
    print("case2 counter:", ctr2)
    if any(ctr2):
        print("  FAIL clean ids bumped the counter")
        fails += 1

    # --- case 3: the eager reader sees the counter --------------------------
    spec3 = _FakeSpec(vocab=8)
    run_a(spec3, torch.tensor([[[9, 0, 0]]], dtype=torch.int64),
          torch.tensor([[0]], dtype=torch.int64))
    real_logger = M.__dict__.get("__builtins__")
    # stub the logger import so the test does not need a vllm import
    import vllm.logger as _vl  # noqa: F401
    _vl.init_logger = lambda name: _FakeLog()  # type: ignore[assignment]
    for _ in range(50):
        run_b(spec3)
    print("case3 reads:", spec3._clamp_reads, "warnings:", len(WARNINGS))
    if spec3._clamp_reads != 50 or not WARNINGS or "ge=1" not in WARNINGS[-1]:
        print("  FAIL reader did not report the dirty bucket")
        fails += 1
    else:
        print("  reader line:", WARNINGS[-1])

    print("FAILS:", fails)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

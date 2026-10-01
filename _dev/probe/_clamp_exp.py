"""In-graph guard for the qwen3_dflash2._score_edges crash (2.2c).

Why this exists
---------------
The 20k-step eager probe (_idcheck_exp.py + _patch_draft_eager.py) came back
completely clean: cum min/max of both `candidate_ids` and `anchor_token_ids`
stayed inside [0, 248320) for 20,150 decode steps across 4 cohort passes.
Combined with "tower + 71680 + async never crashes while the drafter runs
eager" (8 passes), the bad value must be produced *inside the replayed
drafter CUDA graph*, not on the Python/eager path.

But a Python probe cannot see inside a replay: `_generate_draft` is traced
into the graph at capture time and its Python body is never executed again.
The only way to instrument a replay is to put the instrumentation **in the
graph** — i.e. ordinary torch ops that get recorded at capture, and let them
re-execute on every replay.

What it does
------------
BLOCK_A (dflash2/speculator.py, `_generate_draft`, right after
`anchor_token_ids` is read) is captured into the drafter graph:

    counter[0] += (candidate_ids < 0).sum()      counter[1] += (>= vocab)
    counter[2] += (anchor_ids    < 0).sum()      counter[3] += (>= vocab)
    candidate_ids   = candidate_ids.clamp(0, vocab - 1)
    anchor_ids      = anchor_ids.clamp(0, vocab - 1)

BLOCK_B (dflash/speculator.py, `propose()`, just before the final return)
reads that counter from the eager side — `propose` runs in Python every
step even when the draft itself is a FULL-graph replay — and logs it every
50 steps.

How to read the result
----------------------
* crash gone + `cand/anchor neg=0 ge=0` throughout  -> the bad id did NOT
  come through these two tensors; the assert is fed by something else.
* crash gone + counter > 0                          -> cause proven: the
  replayed graph really does see out-of-range ids, and the clamp is a
  (lossy) mitigation.
* still crashes                                     -> the guard never fired
  on the failing path, so the index is fed by a different expression than
  `predecessor_table[predecessor_ids]`.

Nothing changes unless DFLASH2_CLAMP=1.

Usage:
    python _clamp_exp.py --apply    # inject into both files (idempotent)
    python _clamp_exp.py --undo     # remove (idempotent)
    python _clamp_exp.py --status   # report current state

Edits BOTH copies per house rule:
  vllm/...  (repo tree)  and  .venv/Lib/site-packages/vllm/...  (runtime)
"""

import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # ...\_dev\probe
DEV  = HERE.parent                                     # ...\_dev
REPO = DEV.parent                                      # D:\code\vllm-windows
TARGET_A_REL = "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py"
TARGET_B_REL = "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"

MARK_A = "        # [clamp-exp]"
ANCHOR_A = (
    "        anchor_token_ids = "
    "self.input_buffers.input_ids[self._anchor_indices[:num_reqs]]\n"
)

MARK_B = "        # [clamp-exp-reader]"
ANCHOR_B = "        return self.draft_tokens[:num_reqs]\n\n\n@triton.jit\n"

BLOCK_A = '''        # [clamp-exp] In-graph guard for the _score_edges bounds assert.
        # Captured into the drafter CUDA graph, so it re-executes on every
        # replay -- which is exactly where the eager probe came back clean.
        # The counter must outlive the capture, hence the
        # is_current_stream_capturing() dance: creating a tensor *inside* a
        # capture would hand the counter a buffer that the graph does not own.
        if __import__("os").environ.get("DFLASH2_CLAMP") == "1":
            _clamp_vocab = (
                self.model.model.candidate_selector.predecessor_codebook.shape[0]
            )
            _clamp_ctr = getattr(self, "_clamp_ctr", None)
            if _clamp_ctr is None and not torch.cuda.is_current_stream_capturing():
                _clamp_ctr = torch.zeros(
                    4, dtype=torch.int64, device=candidate_ids.device
                )
                self._clamp_ctr = _clamp_ctr
            if _clamp_ctr is not None:
                _clamp_ctr[0] += (candidate_ids < 0).sum()
                _clamp_ctr[1] += (candidate_ids >= _clamp_vocab).sum()
                _clamp_ctr[2] += (anchor_token_ids < 0).sum()
                _clamp_ctr[3] += (anchor_token_ids >= _clamp_vocab).sum()
            candidate_ids = candidate_ids.clamp(0, _clamp_vocab - 1)
            anchor_token_ids = anchor_token_ids.clamp(0, _clamp_vocab - 1)
'''

BLOCK_B = '''        # [clamp-exp-reader] `propose()` runs in Python every step even when
        # the draft itself is a FULL-graph replay, so it is the only place
        # that can observe the counter the captured graph keeps bumping.
        if __import__("os").environ.get("DFLASH2_CLAMP") == "1":
            _clamp_ctr = getattr(self, "_clamp_ctr", None)
            if _clamp_ctr is not None and not torch.cuda.is_current_stream_capturing():
                _clamp_n = getattr(self, "_clamp_reads", 0) + 1
                self._clamp_reads = _clamp_n
                if _clamp_n % 50 == 0:
                    _clamp_v = _clamp_ctr.tolist()
                    __import__(
                        "vllm.logger", fromlist=["init_logger"]
                    ).init_logger("vllm.dflash2.clamp").warning(
                        "[clamp] reads=%d num_reqs=%d cg=%s | cand neg=%d ge=%d | "
                        "anchor neg=%d ge=%d",
                        _clamp_n, num_reqs, batch_desc.cg_mode,
                        _clamp_v[0], _clamp_v[1], _clamp_v[2], _clamp_v[3],
                    )
'''

# (relative path, anchor, block, mark, insert_after_anchor)
JOBS = [
    (TARGET_A_REL, ANCHOR_A, BLOCK_A, MARK_A, True),
    (TARGET_B_REL, ANCHOR_B, BLOCK_B, MARK_B, False),
]


def targets(rel: str) -> list[Path]:
    return [REPO / rel, REPO / ".venv/Lib/site-packages" / rel]


def main() -> None:
    undo = "--undo" in sys.argv
    if "--apply" not in sys.argv and not undo and "--status" not in sys.argv:
        print(__doc__)
        raise SystemExit(2)

    for rel, anchor, block, mark, after in JOBS:
        for path in targets(rel):
            if not path.exists():
                print(f"MISSING: {path}")
                continue
            src = io.open(path, encoding="utf-8").read()
            n = src.count(mark.strip())
            tag = path.relative_to(REPO).as_posix()
            if "--status" in sys.argv:
                print(f"{'applied' if n else 'absent '}: {tag}")
                continue
            if undo:
                if not n:
                    print(f"nothing to undo: {tag}")
                    continue
                if block not in src:
                    print(f"BLOCK NOT FOUND verbatim in {tag}", file=sys.stderr)
                    raise SystemExit(1)
                io.open(path, "w", encoding="utf-8", newline="").write(
                    src.replace(block, "", 1)
                )
                print(f"undone: {tag}")
            else:
                if n:
                    print(f"already applied: {tag}")
                    continue
                if anchor not in src:
                    print(f"ANCHOR NOT FOUND in {tag}", file=sys.stderr)
                    raise SystemExit(1)
                new = anchor + block if after else block + anchor
                io.open(path, "w", encoding="utf-8", newline="").write(
                    src.replace(anchor, new, 1)
                )
                print(f"applied: {tag}")

    if "--status" not in sys.argv:
        for rel, _a, _b, _m, _af in JOBS:
            a, b = targets(rel)
            if a.exists() and b.exists():
                same = io.open(a, encoding="utf-8").read() == io.open(
                    b, encoding="utf-8"
                ).read()
                print(f"sync {'OK' if same else 'MISMATCH'}: {Path(rel).name}")


if __name__ == "__main__":
    main()

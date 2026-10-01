"""Run the DFlash2 DRAFTER eagerly so the [idcheck] probe can actually see decode ids.

Why this exists (2026-09-21 22:10, after the 5th clean shot):

The probe injected by `_idcheck_exp.py` sits in
`dflash2/speculator.py::_generate_draft`, and on the 21:59 run it logged
NOTHING -- not even the "value unchanged" case.  Reading the call path explains
why the design could not have worked:

    v1/worker/gpu/spec_decode/dflash/speculator.py:~465
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            self.query_cudagraph_manager.run_fullgraph(batch_desc)   # REPLAY
        else:
            self._generate_draft(...)                                # python

The drafter captures FULL CUDA graphs for every decode bucket
(1,2,4,...,64) and replays them.  A replay executes only the recorded GPU
kernels -- `_generate_draft`'s Python body does not run, so neither the probe
nor any other Python instrumentation can observe the ids of a decode step.
And since `dispatch()` returns cg_mode=NONE only when no graph matches, every
real decode batch takes the replay branch.

This patch adds an env-gated override at the top of
`DFlashSpeculator.init_cudagraph_manager`: with DFLASH2_NO_DRAFT_CUDAGRAPH=1
the drafter is built with CUDAGraphMode.NONE, so `dispatch()` never returns
FULL, `_generate_draft` runs eagerly every step, and the probe is live.

It is a DISCRIMINATOR, not a fix:

  crash + [idcheck] shows an OOB tensor  -> the garbage id is real data;
                                            root cause named, one shot
  crash + [idcheck] all zeros            -> the OOB is created inside the
                                            captured graph only
  no crash                                -> the crash needs the drafter's
                                            FULL-graph replay path: the static
                                            input buffers / address reuse
                                            become the prime suspect

Cost: decode gets slower (draft forward no longer graph-replayed).  It changes
timing, so it may mask a race -- that is exactly the signal described above.

Usage:
    python _patch_draft_eager.py --apply
    python _patch_draft_eager.py --undo
    python _patch_draft_eager.py --status

Edits BOTH copies per house rule:
  vllm/...  (repo tree)  and  .venv/Lib/site-packages/vllm/...  (runtime)
"""

import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # ...\_dev\probe
DEV  = HERE.parent                                     # ...\_dev
REPO = DEV.parent                                      # D:\code\vllm-windows
REL = "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"
TARGETS = [REPO / REL, REPO / ".venv/Lib/site-packages" / REL]

ANCHOR = (
    "    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:\n"
    "        wants_full = cudagraph_mode.decode_mode() == CUDAGraphMode.FULL\n"
)

BLOCK = (
    "    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:\n"
    "        # [draft-eager-exp] Diagnostic override -- see _patch_draft_eager.py.\n"
    "        # Forcing NONE keeps dispatch() from ever returning FULL, so\n"
    "        # _generate_draft() runs in Python every step instead of being\n"
    "        # replayed from a captured graph, which is the only way the\n"
    "        # DFLASH2_IDCHECK probe can observe decode-time ids.\n"
    '        if __import__("os").environ.get("DFLASH2_NO_DRAFT_CUDAGRAPH") == "1":\n'
    "            cudagraph_mode = CUDAGraphMode.NONE\n"
    "        wants_full = cudagraph_mode.decode_mode() == CUDAGraphMode.FULL\n"
)


def main() -> None:
    undo = "--undo" in sys.argv
    if "--apply" not in sys.argv and not undo and "--status" not in sys.argv:
        print(__doc__)
        raise SystemExit(2)

    for path in TARGETS:
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        src = io.open(path, encoding="utf-8", newline="").read()
        n = src.count("[draft-eager-exp]")
        if "--status" in sys.argv:
            print(f"{'applied' if n else 'absent '}: {path}")
            continue
        if undo:
            if not n:
                print(f"nothing to undo: {path}")
                continue
            if BLOCK not in src:
                print(f"BLOCK NOT FOUND verbatim in {path}", file=sys.stderr)
                raise SystemExit(1)
            io.open(path, "w", encoding="utf-8", newline="").write(
                src.replace(BLOCK, ANCHOR, 1)
            )
            print(f"undone: {path}")
        else:
            if n:
                print(f"already applied: {path}")
                continue
            if ANCHOR not in src:
                print(f"ANCHOR NOT FOUND in {path}", file=sys.stderr)
                raise SystemExit(1)
            io.open(path, "w", encoding="utf-8", newline="").write(
                src.replace(ANCHOR, BLOCK, 1)
            )
            print(f"applied: {path}")

    if "--status" not in sys.argv:
        a = io.open(TARGETS[0], encoding="utf-8", newline="").read()
        b = io.open(TARGETS[1], encoding="utf-8", newline="").read()
        print(f"sync {'OK' if a == b else 'MISMATCH'}: {TARGETS[1].name}")


if __name__ == "__main__":
    main()

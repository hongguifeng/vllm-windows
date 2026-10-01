r"""Instrument the DFlash2 candidate path to prove (or kill) the garbage-id theory.

Root cause chain now localised (2026-09-21 21:43 crash):

    vllm/model_executor/models/qwen3_dflash2.py:181
        predecessors = predecessor_table[predecessor_ids]
    -> triton_poi_fused__unsafe_view_cat_expand_index_mul_slice_unsqueeze_0
       tl.device_assert((0 <= tmp17) & (tmp17 < 248320))   <- inductor bounds check
    -> AssertionError -> EngineDeadError -> every in-flight request 500s

`predecessor_ids` is `cat([anchor_token_ids.expand(-1,1,16), candidate_ids[:, :-1]])`,
i.e. exactly the two tensors built in

    v1/worker/gpu/spec_decode/dflash2/speculator.py::_generate_draft

    candidate_ids  from flashinfer.top_k(lm_head logits, k=16) -> int64
    anchor_token_ids = input_buffers.input_ids[anchor_indices] -> int32

Both must stay in [0, 248320).  A value outside that window is what trips the
assert, and the assert is *nondeterministic* (4 boots, 3 clean) -- so the value
looks like garbage rather than a shape bug.

This patch answers the only question that matters: WHICH tensor carries the bad
value, and what the value is.  It accumulates min/max/negative-count/too-big
count on the GPU (no per-step sync, so the async overlap window is untouched)
and flushes through one small sync every N steps.  With DFLASH2_IDCHECK_EVERY=1
the flush lands on every step, i.e. on the crashing step too -- the assert fires
right after this block (inside candidate_selector), so the offending values are
in the log even though the engine dies one line later.

It only sees anything if the drafter runs eagerly, i.e. together with
_patch_draft_eager.py (DFLASH2_NO_DRAFT_CUDAGRAPH=1).  Under the default
FULL cudagraph the whole _generate_draft body is *captured*, and replay
skips the Python entirely -- so a plain Python probe is blind there.

Nothing changes unless DFLASH2_IDCHECK=1.

Two knobs:
    DFLASH2_IDCHECK=1         arm the probe (required)
    DFLASH2_IDCHECK_EVERY=N   flush cadence, default 25 steps; 1 = every step

Armed runs always log one `[idcheck] ARMED ...` line at step 1, so
`grep -c '\[idcheck\]' <serve log>` == 0 unambiguously means "never armed"
(2026-09-21 21:59: the probe stayed silent and start_server.ps1's env whitelist
dropped DFLASH2_IDCHECK, so the log could not prove which of the two it was).

Usage:
    python _idcheck_exp.py --apply      # inject (idempotent)
    python _idcheck_exp.py --undo       # remove (idempotent)
    python _idcheck_exp.py --status     # report current state

Edits BOTH copies per house rule:
  vllm/...  (repo tree)  and  .venv/Lib/site-packages/vllm/...  (runtime)
"""

import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # ...\_dev\probe
DEV  = HERE.parent                                     # ...\_dev
REPO = DEV.parent                                      # D:\code\vllm-windows
REL = "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py"
TARGETS = [REPO / REL, REPO / ".venv/Lib/site-packages" / REL]

MARK = "        # [idcheck-exp]"
ANCHOR = (
    "        anchor_token_ids = "
    "self.input_buffers.input_ids[self._anchor_indices[:num_reqs]]\n"
)

BLOCK = '''        # [idcheck-exp] Diagnostics for the qwen3_dflash2._score_edges crash.
        # `candidate_ids` (flashinfer top-k) and `anchor_token_ids`
        # (input_buffers.input_ids) both index the [vocab, rank] codebooks, and
        # an out-of-range value makes inductor's generated bounds assert kill
        # the engine.  Everything below is accumulated on-device and flushed
        # every 25 steps, so no per-step sync disturbs the async overlap.
        if __import__("os").environ.get("DFLASH2_IDCHECK") == "1":
            try:
                _vocab = self.model.model.candidate_selector.predecessor_codebook.shape[0]
                _st = getattr(self, "_idcheck_state", None)
                if _st is None:
                    _st = {
                        "acc": torch.zeros(8, dtype=torch.int64, device=candidate_ids.device),
                        "step": 0,
                        "last": None,
                        "dumped": False,
                        # DFLASH2_IDCHECK_EVERY=1 -> flush every step (tightest
                        # coverage, one sync per step: perturbs the async
                        # overlap, so use it for a targeted pass only).
                        "every": max(1, int(__import__("os").environ.get(
                            "DFLASH2_IDCHECK_EVERY") or 25)),
                        "log": __import__(
                            "vllm.logger", fromlist=["init_logger"]
                        ).init_logger("vllm.dflash2.idcheck"),
                    }
                    self._idcheck_state = _st
                _acc = _st["acc"]
                _acc[0] = torch.minimum(_acc[0], torch.amin(candidate_ids))
                _acc[1] = torch.maximum(_acc[1], torch.amax(candidate_ids))
                _acc[2] += (candidate_ids < 0).sum()
                _acc[3] += (candidate_ids >= _vocab).sum()
                _acc[4] = torch.minimum(_acc[4], torch.amin(anchor_token_ids))
                _acc[5] = torch.maximum(_acc[5], torch.amax(anchor_token_ids))
                _acc[6] += (anchor_token_ids < 0).sum()
                _acc[7] += (anchor_token_ids >= _vocab).sum()
                _st["step"] += 1
                # 2026-09-21 22:0x: the probe produced ZERO lines on the 21:59
                # run, and there was no way to tell "clean run" from "never
                # armed" -- start_server.ps1's env dump whitelist drops
                # DFLASH2_IDCHECK, so the log could not prove either way.  Emit
                # one unconditional ARMED line on the first step: from now on
                # `grep -c idcheck` == 0 means "not armed", >=1 means "armed".
                if _st["step"] == 1:
                    _st["log"].warning(
                        "[idcheck] ARMED num_reqs=%d vocab=%d cand.shape=%s anchor.shape=%s",
                        num_reqs, _vocab, tuple(candidate_ids.shape),
                        tuple(anchor_token_ids.shape),
                    )
                if _st["step"] % _st["every"] == 0:
                    # 2026-09-21 22:19 run: one line at step 25, then silence for
                    # the remaining 22 min of decode.  That run gated this log on
                    # `_v != _st["last"]`, so silence was ambiguous -- "probe not
                    # called any more" vs "min/max simply never moved".  Always
                    # log now; the step counter itself then proves liveness.
                    _v = _acc.tolist()
                    _st["last"] = _v
                    _st["log"].warning(
                        "[idcheck] steps=%d every=%d num_reqs=%d vocab=%d | "
                        "cum cand min=%d max=%d neg=%d ge=%d | "
                        "cum anchor min=%d max=%d neg=%d ge=%d | "
                        "now cand[%d,%d] anchor[%d,%d]",
                        _st["step"], _st["every"], num_reqs, _vocab,
                        _v[0], _v[1], _v[2], _v[3], _v[4], _v[5], _v[6], _v[7],
                        int(torch.amin(candidate_ids)), int(torch.amax(candidate_ids)),
                        int(torch.amin(anchor_token_ids)),
                        int(torch.amax(anchor_token_ids)),
                    )
                    if (_v[2] or _v[3] or _v[6] or _v[7]) and not _st["dumped"]:
                        _st["dumped"] = True
                        _st["log"].warning("[idcheck] candidate_ids=%s", candidate_ids.tolist())
                        _st["log"].warning("[idcheck] anchor_ids=%s", anchor_token_ids.tolist())
                        _st["log"].warning(
                            "[idcheck] anchor_slots=%s",
                            self._anchor_indices[:num_reqs].tolist(),
                        )
            except Exception as _e:
                # one-shot: never spam the log from inside the decode loop
                if not getattr(self, "_idcheck_skip", False):
                    self._idcheck_skip = True
                    __import__("logging").getLogger("vllm.dflash2.idcheck").warning(
                        "[idcheck] DISABLED after error: %r", _e
                    )
'''


def main() -> None:
    undo = "--undo" in sys.argv
    if "--apply" not in sys.argv and not undo and "--status" not in sys.argv:
        print(__doc__)
        raise SystemExit(2)

    for path in TARGETS:
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        src = io.open(path, encoding="utf-8").read()
        n = src.count("[idcheck-exp]")
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
            io.open(path, "w", encoding="utf-8", newline="").write(src.replace(BLOCK, "", 1))
            print(f"undone: {path}")
        else:
            if n:
                print(f"already applied: {path}")
                continue
            if ANCHOR not in src:
                print(f"ANCHOR NOT FOUND in {path}", file=sys.stderr)
                raise SystemExit(1)
            io.open(path, "w", encoding="utf-8", newline="").write(
                src.replace(ANCHOR, ANCHOR + BLOCK, 1)
            )
            print(f"applied: {path}")

    if "--status" not in sys.argv:
        a = io.open(TARGETS[0], encoding="utf-8").read()
        b = io.open(TARGETS[1], encoding="utf-8").read()
        print(f"sync {'OK' if a == b else 'MISMATCH'}: {TARGETS[1].name}")


if __name__ == "__main__":
    main()

"""Boot-lottery experiment: zero all scratch memory before DFlash2 graph capture.

Hypothesis under test: the ~1/3-of-boots `device-side assert` during DFlash2 CUDA
graph capture comes from kernels reading UNINITIALIZED device memory whose garbage
content differs boot to boot. If we zero every scratch tensor in the drafter's
input_buffers plus all free VRAM before capture, the lottery should disappear.

Usage:
    python _zerofill_exp.py --apply   # inject the block (idempotent)
    python _zerofill_exp.py --undo    # remove it again (idempotent)

The injected code only runs when env ZEROFILL_EXP=1, so apply is safe to leave
in place; arm selection is done per-boot via the environment variable.

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

MARK = "        # [zerofill-exp]"
ANCHOR = ('    def capture(self) -> None:\n'
          '        logger.info("Capturing model for %s speculator...", '
          'self._speculator_name)\n')

BLOCK = '''        # [zerofill-exp] A/B experiment: make every scratch byte deterministic.
        # Zeroes (1) every CUDA tensor reachable from input_buffers and
        # (2) all free VRAM kept inside the torch allocator pool, so that any
        # uninitialized read during graph capture sees zeros on every boot.
        # Only active with ZEROFILL_EXP=1.
        if __import__("os").environ.get("ZEROFILL_EXP") == "1":
            try:
                def _zf_zero(obj, _seen):
                    if id(obj) in _seen:
                        return
                    _seen.add(id(obj))
                    if isinstance(obj, torch.Tensor):
                        if obj.is_cuda:
                            obj.zero_()
                    elif isinstance(obj, (list, tuple)):
                        for _x in obj:
                            _zf_zero(_x, _seen)
                    elif isinstance(obj, dict):
                        for _x in list(obj.values()):
                            _zf_zero(_x, _seen)
                    elif hasattr(obj, "__dict__"):
                        for _x in list(vars(obj).values()):
                            _zf_zero(_x, _seen)
                _zf_zero(self.input_buffers, set())
                _free_b, _ = torch.cuda.mem_get_info()
                _n = int(max(0, _free_b - int(1.5e9)) // 4)
                if _n > 0:
                    _buf = torch.empty(_n, dtype=torch.float32, device="cuda")
                    _buf.zero_()
                    torch.cuda.synchronize()
                    del _buf
                logger.info("[zerofill-exp] zeroed input_buffers + %.2f GiB free VRAM",
                            _n * 4 / 2 ** 30)
            except Exception as _e:
                logger.warning("[zerofill-exp] skipped: %s", _e)
'''


def main() -> None:
    undo = "--undo" in sys.argv
    for path in TARGETS:
        src = io.open(path, encoding="utf-8").read()
        if undo:
            if "[zerofill-exp]" not in src:
                print(f"nothing to undo: {path}")
                continue
            if BLOCK not in src:
                print(f"BLOCK NOT FOUND verbatim in {path}", file=sys.stderr)
                sys.exit(1)
            src = src.replace(BLOCK, "", 1)
            io.open(path, "w", encoding="utf-8", newline="").write(src)
            print(f"undone: {path}")
        else:
            if "[zerofill-exp]" in src:
                print(f"already applied: {path}")
                continue
            if ANCHOR not in src:
                print(f"ANCHOR NOT FOUND in {path}", file=sys.stderr)
                sys.exit(1)
            src = src.replace(ANCHOR, ANCHOR + BLOCK, 1)
            io.open(path, "w", encoding="utf-8", newline="").write(src)
            print(f"applied: {path}")

    for path in TARGETS:
        a = io.open(REPO / REL, encoding="utf-8").read()
        b = io.open(path, encoding="utf-8").read()
        print(f"sync {'OK' if a == b else 'MISMATCH'}: {path.name}")


if __name__ == "__main__":
    main()

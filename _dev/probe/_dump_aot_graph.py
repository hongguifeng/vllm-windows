"""Dump a torch.compile / inductor AOT artifact back to readable source.

When a device-side assert fires inside an inductor-generated kernel, the log only
shows something like

    ~/.cache/vllm/torch_compile_cache/torch_aot_compile/<h>/inductor_cache/7z/xxx.py:51:
      Assertion `index out of bounds: 0 <= tmp17 < 248320` failed

The compiled artifact next to it (under `fxgraph/**`) is a pickled
`torch._inductor.output_code.CompiledFxGraph`.  `inductor_post_grad_graph_str`
prints the graph with **`# File: <model>.py:LINE in func, code: ...` comments**,
which is how you map a fused kernel back to the model source line.

Usage:
    python _dump_aot_graph.py                  # newest artifact under the cache
    python _dump_aot_graph.py <file> [outdir]  # explicit artifact
"""

import io
import pickle
import sys
from pathlib import Path

DEFAULT_ROOT = Path.home() / ".cache" / "vllm" / "torch_compile_cache" / "torch_aot_compile"
ATTRS = ("runnable_graph_str", "inductor_post_grad_graph_str", "source_code")


def newest(root: Path) -> Path:
    # CompiledFxGraph pickles live under .../inductor_cache/fxgraph/<xx>/<hash>/<hash>
    # (they have no file extension).  inductor_cache/**/*.py are generated kernels.
    cands = [p for p in root.rglob("*") if p.is_file() and "fxgraph" in p.parts]
    if not cands:
        raise SystemExit(f"no CompiledFxGraph artifacts under {root}")
    return max(cands, key=lambda p: p.stat().st_mtime)


def main() -> None:
    # argv[1] = artifact (optional), argv[2] = output dir (optional).
    # A bare directory as the first arg means "auto-discover, write there".
    artifact = None
    outdir = Path.cwd()
    for a in sys.argv[1:]:
        p = Path(a)
        if p.is_dir():
            outdir = p
        elif artifact is None:
            artifact = p
        else:
            outdir = p
    target = artifact or newest(DEFAULT_ROOT)
    print("artifact:", target)

    with open(target, "rb") as f:
        obj = pickle.load(f)
    print("type:", type(obj).__name__)

    wrote = []
    for attr in ATTRS:
        v = getattr(obj, attr, None)
        if v is None:
            continue
        if not isinstance(v, str):
            v = repr(v)
        out = outdir / f"{target.name}.{attr}.txt"
        io.open(out, "w", encoding="utf-8", newline="\n").write(v)
        wrote.append(out)
        print(f"  {attr:32s} {len(v):>9,d} chars -> {out}")

    g = getattr(obj, "guards_expr", None)
    if g:
        print("guards_expr:", str(g)[:400])
    if not wrote:
        print("no dumpable attribute (is this really a CompiledFxGraph?)")


if __name__ == "__main__":
    main()

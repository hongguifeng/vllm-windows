"""Point every reference at the serve script's latest name.

2026-09-22, round 1: `_dev\\bin\\_serve.ps1` -> repo root `Serve.ps1`.
2026-09-22, round 2: `Serve.ps1` -> repo root `start_server.ps1`.

This rewrites the remaining references: absolute Windows paths in the
self-authored docs, and the prose mentions in comments.

    python _fix_serve_refs.py [--apply]

It deliberately does NOT list itself in FILES -- its own PAIRS constants contain
the old spellings, and rewriting them would erase the tool's record of what it
did.
"""
import os
import sys

FILES = [
    r"D:\code\vllm-windows\_dev\docs\AB_WSL_VS_WINDOWS.md",
    r"D:\code\vllm-windows\_dev\docs\BENCH_170HX_BASELINE.md",
    r"D:\code\vllm-windows\_dev\docs\RUN_QWEN38_27B_W4A16.md",
    r"D:\code\vllm-windows\_dev\docs\_skill_168c_new.md",
    r"D:\code\vllm-windows\_dev\README.md",
    r"D:\code\vllm-windows\_dev\bin\_serve_verify.sh",
    r"D:\code\vllm-windows\_dev\bin\_serve_cli_matrix.ps1",
    r"D:\code\vllm-windows\_dev\bench\_bench_suite.py",
    r"D:\code\vllm-windows\_dev\patches\fix_offload_release.py",
    r"D:\code\vllm-windows\_dev\probe\_check_serve_defaults.ps1",
    r"D:\code\vllm-windows\_dev\probe\_idcheck_exp.py",
    r"D:\code\vllm-windows\_dev\probe\_repro_async_tower.ps1",
    r"D:\code\vllm-windows\_dev\test\_vision_pressure.py",
    r"D:\code\vllm-windows\start_server.ps1",
    # the durable knowledge base lives outside the repo; a stale path there is
    # the worst kind, because that is what future sessions read first.
    r"C:\Users\hong\.workbuddy\skills\vllm-windows-source-build\SKILL.md",
]

# Order matters only for the absolute path, which is the one that has to stay
# runnable; the bare-name patterns below are disjoint from it and from each
# other (`_serve.ps1` is not a substring of `Serve.ps1` -- that is case
# sensitive -- and `_serve_cli_matrix.ps1` matches neither).
PAIRS = [
    (r"D:\code\vllm-windows\_dev\bin\_serve.ps1", r"D:\code\vllm-windows\start_server.ps1"),
    (r"D:\code\vllm-windows\Serve.ps1", r"D:\code\vllm-windows\start_server.ps1"),
    ("_serve.ps1", "start_server.ps1"),
    ("Serve.ps1", "start_server.ps1"),
]


def main():
    apply = "--apply" in sys.argv
    total = 0
    for p in FILES:
        if not os.path.exists(p):
            print(f"missing: {p}")
            continue
        s = open(p, encoding="utf-8", newline="").read()
        orig = s
        hits = []
        for a, b in PAIRS:
            if a in s:
                hits.append(f"{a!r} x{s.count(a)}")
                s = s.replace(a, b)
        if s == orig:
            print(f"unchanged: {p}")
            continue
        total += 1
        print(f"{p}\n    " + ", ".join(hits))
        if apply:
            open(p, "w", encoding="utf-8", newline="").write(s)
    print(f"{total} file(s) {'rewritten' if apply else 'would change'}")


if __name__ == "__main__":
    main()

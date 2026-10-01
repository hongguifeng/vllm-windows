"""Find stray control characters that broke Windows paths in the project docs.

An earlier session wrote some .md files through a string literal that ate the
backslash escapes, so `D:\\code\\vllm-windows\\_dev\\bin\\_serve.ps1` ended up on
disk as `D:\\code<VT>llm-windows<BS>in<BS>_serve.ps1` -- invisible in a terminal,
and the copied path does not exist.  This scans for the culprits (and can repair
them).

    python _scan_ctrlchars.py            # report only
    python _scan_ctrlchars.py --fix      # rewrite the files
"""
import os
import sys

BAD = {0x08: br"\b", 0x0B: br"\v", 0x0C: br"\f", 0x07: br"\a", 0x00: br"\0"}
EXT = (".md", ".ps1", ".py", ".sh", ".txt", ".json", ".yaml", ".yml", ".patch")
ROOTS = [
    r"D:\code\vllm-windows\_dev",
    r"D:\code\vllm-windows\.workbuddy",
    r"C:\Users\hong\.workbuddy\skills\vllm-windows-source-build",
]
SKIP = (".venv", ".git", "node_modules", "__pycache__", "\\out\\", "/out/")


def _skip(dp: str) -> bool:
    # `...\_dev\out` has no trailing separator, so a plain "\\out\\" substring
    # test misses every file that sits directly in it -- which is exactly how an
    # earlier run "repaired" a raw probe dump full of NULs.  Check the leaf too.
    leaf = os.path.basename(os.path.normpath(dp)).lower()
    return leaf in ("out", "logs") or any(s in dp for s in SKIP)


def walk():
    seen = set()
    for root in ROOTS:
        for dp, dn, fn in os.walk(root):
            if _skip(dp):
                continue
            for f in fn:
                if f.lower().endswith(EXT):
                    p = os.path.join(dp, f)
                    if p not in seen:
                        seen.add(p)
                        yield p


def repair(b: bytes) -> bytes:
    # `\v` -> VT, `\b` -> BS: the byte that landed on disk is the control code,
    # so the two-character escape is what has to go back.
    for c, esc in BAD.items():
        b = b.replace(bytes([c]), esc)
    return b


def main():
    fix = "--fix" in sys.argv
    n = 0
    for p in walk():
        try:
            b = open(p, "rb").read()
        except OSError:
            continue
        n += 1
        counts = {BAD[c]: b.count(bytes([c])) for c in BAD if b.count(bytes([c]))}
        if not counts:
            continue
        print(f"{p}\n    " + ", ".join(f"{k} x{v}" for k, v in counts.items()))
        if fix:
            open(p, "wb").write(repair(b))
            print("    -> repaired")
    print(f"scanned {n} files; fix={fix}")


if __name__ == "__main__":
    main()

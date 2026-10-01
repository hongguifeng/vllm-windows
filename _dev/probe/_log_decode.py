#!/usr/bin/env python3
"""Print a vLLM host log regardless of how PowerShell encoded it.

Redirecting a native command's stream through PowerShell can leave an ANSI
prefix followed by UTF-16LE text, which no single decoding handles. Split at the
first NUL byte, decode each half, and emit UTF-8.
"""
import sys

MAX_BYTES = 64 * 1024 * 1024


def decode(raw: bytes) -> str:
    if raw.startswith(b"\xff\xfe"):
        # A UTF-16LE byte-order mark means the whole file is wide-character
        # text; splitting on the first NUL would misalign every pair after it.
        tail = raw[2:] if len(raw) % 2 else raw
        return tail.decode("utf-16", "replace")
    cut = raw.find(b"\x00")
    if cut < 0:
        return raw.decode("utf-8", "replace")
    head = raw[:cut].decode("utf-8", "replace")
    tail = raw[cut:]
    if len(tail) % 2:
        tail += b"\x00"
    return head + "\n" + tail.decode("utf-16-le", "replace")


def main() -> None:
    path = sys.argv[1]
    # Console output on Windows defaults to a legacy code page and dies on the
    # block characters in progress bars.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover - very old Python
        pass
    with open(path, "rb") as fh:
        raw = fh.read(MAX_BYTES)
    text = decode(raw).replace("\r", "")
    pattern = sys.argv[2] if len(sys.argv) > 2 else None
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if pattern and pattern not in line:
            continue
        print(line)


if __name__ == "__main__":
    main()

"""Open every safetensors file of the EXL3 checkpoint, twice concurrently.

Reproduces the TP2 worker pattern that failed with Windows error 1455
(ERROR_COMMITMENT_LIMIT) at the loader's dict comprehension of safe_open().
"""

import subprocess
import sys
import time
from pathlib import Path

MODEL = Path(r"D:\models\Qwen3.8-Flash-Next-exl3")
CODE = (
    "import sys, time\n"
    "from pathlib import Path\n"
    "from safetensors import safe_open\n"
    "opened = []\n"
    "for p in sorted(Path(sys.argv[1]).glob('*.safetensors')):\n"
    "    opened.append(safe_open(str(p), framework='pt', device='cpu'))\n"
    "print('opened', len(opened), flush=True)\n"
    "time.sleep(20)\n"
)


def commit_mib() -> float:
    import ctypes
    from ctypes import wintypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.ullAvailPageFile / 2**20


def main() -> int:
    procs = [
        subprocess.Popen([sys.executable, "-c", CODE, str(MODEL)])
        for _ in range(2)
    ]
    peak = commit_mib()
    while any(p.poll() is None for p in procs):
        peak = min(peak, commit_mib())
        time.sleep(0.2)
    codes = [p.returncode for p in procs]
    print(f"commit free start/peak MiB: {commit_mib():.0f} / {peak:.0f}")
    print("exit codes", codes)
    return 0 if all(c == 0 for c in codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())

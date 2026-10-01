#!/usr/bin/env python3
"""Report Windows host memory through GlobalMemoryStatusEx.

Steady-state rates move about a quarter with host pressure alone, so every
measurement block needs to say what the host looked like when it ran. This reads
the same value the memory manager uses, so it does not depend on a Python
package being installed in the serving environment.
"""
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


def main() -> None:
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    gib = 1024 ** 3
    print(
        f"load={status.dwMemoryLoad}% "
        f"total={status.ullTotalPhys / gib:.1f} GiB "
        f"avail={status.ullAvailPhys / gib:.1f} GiB "
        f"avail_pagefile={status.ullAvailPageFile / gib:.1f} GiB"
    )


if __name__ == "__main__":
    main()

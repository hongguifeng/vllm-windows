"""Does NO_BUFFERING work on a file we created ourselves?

Distinguishes "AV/filter intercepts reads of model files" from "the whole
unbuffered path is rejected on this machine". Writes are tested too, because
NO_BUFFERING constrains them identically.

  python _dev/probe/_ple_nobuffer_freshfile.py
"""

import ctypes
import os
import tempfile

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.restype = ctypes.c_void_p
k32.CreateFileW.argtypes = [
    ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
]
k32.WriteFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                          ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p]
k32.WriteFile.restype = ctypes.c_int
k32.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                         ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p]
k32.ReadFile.restype = ctypes.c_int
k32.SetFilePointerEx.argtypes = [ctypes.c_void_p, ctypes.c_longlong,
                                 ctypes.POINTER(ctypes.c_longlong), ctypes.c_uint]
k32.SetFilePointerEx.restype = ctypes.c_int
k32.CloseHandle.argtypes = [ctypes.c_void_p]

GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
FILE_SHARE_READ = 0x00000001
CREATE_ALWAYS = 2
OPEN_EXISTING = 3
NO_BUFFERING = 0x40000000
RANDOM_ACCESS = 0x10000000
INVALID = ctypes.c_void_p(-1).value


def open_path(path, access, flags):
    h = k32.CreateFileW(path, access, FILE_SHARE_READ, None, OPEN_EXISTING, flags, None)
    if h in (None, INVALID):
        return None, ctypes.WinError(ctypes.get_last_error())
    return h, None


def main() -> int:
    path = os.path.join(tempfile.gettempdir(), "nbtest_fresh.bin")
    MB = 1024 * 1024
    h = k32.CreateFileW(path, GENERIC_WRITE | GENERIC_READ, FILE_SHARE_READ, None,
                        CREATE_ALWAYS, 0, None)
    if h in (None, INVALID):
        print("create failed:", ctypes.WinError(ctypes.get_last_error()))
        return 1
    got = ctypes.c_uint(0)
    ok = k32.WriteFile(h, bytes(MB), MB, ctypes.byref(got), None)
    print(f"buffered write {MB} B -> ok={ok} bytes={got.value}")
    k32.CloseHandle(h)

    print(f"file: {path} ({os.path.getsize(path)} B)")
    for name, flags in (
        ("plain", RANDOM_ACCESS),
        ("NO_BUFFERING", NO_BUFFERING | RANDOM_ACCESS),
    ):
        h, err = open_path(path, GENERIC_READ, flags)
        if h is None:
            print(f"  {name:14s} CreateFile -> {err}")
            continue
        moved = ctypes.c_longlong(0)
        k32.SetFilePointerEx(h, 4096, ctypes.byref(moved), 0)
        buf = ctypes.create_string_buffer(4096)
        got = ctypes.c_uint(0)
        ok = k32.ReadFile(h, buf, 4096, ctypes.byref(got), None)
        print(f"  {name:14s} read @4096 4096B -> "
              f"{'ok ' + str(got.value) if ok else str(ctypes.WinError(ctypes.get_last_error()))}")
        k32.CloseHandle(h)

    for name, flags in (("plain", 0), ("NO_BUFFERING", NO_BUFFERING)):
        h, err = open_path(path, GENERIC_WRITE, flags)
        if h is None:
            print(f"  {name:14s} CreateFile(w) -> {err}")
            continue
        moved = ctypes.c_longlong(0)
        k32.SetFilePointerEx(h, 4096, ctypes.byref(moved), 0)
        got = ctypes.c_uint(0)
        chunk = bytes(4096)
        ok = k32.WriteFile(h, chunk, 4096, ctypes.byref(got), None)
        print(f"  {name:14s} write @4096 4096B -> "
              f"{'ok ' + str(got.value) if ok else str(ctypes.WinError(ctypes.get_last_error()))}")
        k32.CloseHandle(h)
    os.remove(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

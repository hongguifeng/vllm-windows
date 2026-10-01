"""Which alignment does FILE_FLAG_NO_BUFFERING actually demand on this volume?

The earlier probe concluded that NO_BUFFERING was blocked by the environment.
That conclusion came from a path which never seeked, so nothing about the
alignment rules was ever tested. This walks the candidate combinations against
the real PLE shard and reports the Win32 error for each.

  python _dev/probe/_ple_nobuffer_align.py [--file <safetensors>]
"""

import argparse
import ctypes
import os
from pathlib import Path

GENERIC_READ = 0x80000000
FILE_SHARE_READ_WRITE = 0x00000007
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x40000000
FILE_FLAG_RANDOM_ACCESS = 0x10000000
MEM_COMMIT_RESERVE = 0x00003000
PAGE_READWRITE = 0x04
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

DEFAULT_FILE = Path(
    r"D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP\model-00001-of-00011.safetensors"
)

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.restype = ctypes.c_void_p
k32.CreateFileW.argtypes = [
    ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
]
k32.CloseHandle.argtypes = [ctypes.c_void_p]
k32.SetFilePointerEx.argtypes = [ctypes.c_void_p, ctypes.c_longlong,
                                 ctypes.POINTER(ctypes.c_longlong), ctypes.c_uint]
k32.SetFilePointerEx.restype = ctypes.c_int
k32.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
                         ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p]
k32.ReadFile.restype = ctypes.c_int
k32.VirtualAlloc.restype = ctypes.c_void_p
k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_uint]
k32.GetFileAttributesW.argtypes = [ctypes.c_wchar_p]
k32.GetFileAttributesW.restype = ctypes.c_uint
k32.DeviceIoControl.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p,
                                ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint,
                                ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p]
k32.DeviceIoControl.restype = ctypes.c_int

FILE_ATTRIBUTE_COMPRESSED = 0x00000800
FILE_ATTRIBUTE_SPARSE_FILE = 0x00000200
IOCTL_DISK_GET_DRIVE_GEOMETRY_EX = 0x000700A0
FILE_BEGIN = 0


def geometry():
    """bytesPerSector of the disk backing D:, if we are allowed to ask."""
    for letter in ("C", "D"):
        h = k32.CreateFileW(f"\\\\.\\{letter}:", GENERIC_READ, FILE_SHARE_READ_WRITE,
                            None, OPEN_EXISTING, 0, None)
        if h in (None, INVALID_HANDLE_VALUE):
            print(f"  \\\\.\\{letter}: open failed -> {ctypes.WinError(ctypes.get_last_error())}")
            continue
        buf = (ctypes.c_ubyte * 64)()
        returned = ctypes.c_uint(0)
        ok = k32.DeviceIoControl(h, IOCTL_DISK_GET_DRIVE_GEOMETRY_EX, None, 0,
                                 buf, ctypes.sizeof(buf), ctypes.byref(returned), None)
        if not ok:
            print(f"  \\\\.\\{letter}: IOCTL_DISK_GET_DRIVE_GEOMETRY_EX -> "
                  f"{ctypes.WinError(ctypes.get_last_error())}")
        else:
            disk_geom_size = 24
            bytes_per_sector = int.from_bytes(
                bytes(buf[4:8]), "little", signed=False)
            total = int.from_bytes(bytes(buf[24:32]), "little", signed=False)
            print(f"  \\\\.\\{letter}: bytesPerSector={bytes_per_sector} "
                  f"total={total/2**30:.1f} GiB")
        k32.CloseHandle(h)


def attrs(path: Path):
    a = k32.GetFileAttributesW(str(path))
    if a == 0xFFFFFFFF:
        print(f"  GetFileAttributesW failed -> {ctypes.WinError(ctypes.get_last_error())}")
        return
    flags = ", ".join(
        name for name, bit in (
            ("COMPRESSED", FILE_ATTRIBUTE_COMPRESSED),
            ("SPARSE", FILE_ATTRIBUTE_SPARSE_FILE),
        ) if a & bit) or "none"
    print(f"  attributes: {flags} (raw 0x{a:08x})")


def try_read(path: Path, offset: int, count: int, align: int) -> str:
    flags = FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS
    h = k32.CreateFileW(str(path), GENERIC_READ, FILE_SHARE_READ_WRITE, None,
                        OPEN_EXISTING, flags, None)
    if h in (None, INVALID_HANDLE_VALUE):
        return f"CreateFile: {ctypes.WinError(ctypes.get_last_error())}"
    try:
        buf = k32.VirtualAlloc(None, max(count, align), MEM_COMMIT_RESERVE, PAGE_READWRITE)
        if not buf:
            return f"VirtualAlloc: {ctypes.WinError(ctypes.get_last_error())}"
        # Push the buffer address up to the requested alignment if VirtualAlloc
        # gave us more than we need (it always gives 64 KiB granularity).
        addr = buf if isinstance(buf, int) else buf.value
        delta = (-addr) % align
        use = addr + delta
        moved = ctypes.c_longlong(0)
        if not k32.SetFilePointerEx(h, offset, ctypes.byref(moved), FILE_BEGIN):
            return f"Seek: {ctypes.WinError(ctypes.get_last_error())}"
        got = ctypes.c_uint(0)
        ok = k32.ReadFile(h, use, count, ctypes.byref(got), None)
        if not ok:
            return f"ReadFile: {ctypes.WinError(ctypes.get_last_error())}"
        return f"ok, {got.value} B"
    finally:
        k32.CloseHandle(h)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(DEFAULT_FILE))
    args = ap.parse_args()
    path = Path(args.file)

    print("==== NO_BUFFERING alignment probe ====")
    print(f"file : {path}  ({path.stat().st_size/2**30:.2f} GiB)")
    print()
    print("-- volume / file properties --")
    print(f"  volume of {path.drive}\\ : ", end="")
    print(os.path.realpath(str(path))[:3])
    geometry()
    attrs(path)
    try:
        free, total, freeu = ctypes.c_ulonglong(), ctypes.c_ulonglong(), ctypes.c_ulonglong()
        if k32.GetDiskFreeSpaceExW(str(path.drive) + "\\",
                                   ctypes.byref(free), ctypes.byref(total), ctypes.byref(freeu)):
            print(f"  GetDiskFreeSpaceExW {path.drive}\\ cluster bytes via fsutil is admin-only; "
                  f"free={free.value/2**30:.1f} GiB")
    except Exception as e:
        print(f"  GetDiskFreeSpaceExW: {e}")
    print()
    print("-- reads: (offset align, byte count, buffer align) --")
    cases = [
        (0, 4096, 4096),
        (4096, 4096, 4096),
        (8 * 4096, 4096, 4096),
        (512, 512, 512),
        (4096, 512, 512),
        (4096, 4096, 512),
        (1024, 4096, 4096),
        (4096, 4096, 8192),
        (4096, 8192, 4096),
        (4096, 1024, 4096),
    ]
    for offset, count, align in cases:
        print(f"  off {offset:6d}  count {count:5d}  buf align {align:4d} : "
              f"{try_read(path, offset, count, align)}")
    print()
    print("-- the shape PLE actually needs: 4096-aligned page covering a 320 B row --")
    for row in (0, 1, 7, 1000, 12345):
        off = 8 + row * 320
        page = off & ~4095
        nbytes = ((off - page) + 320 + 4095) & ~4095
        print(f"  row {row:6d}  page {page}  count {nbytes} : "
              f"{try_read(path, page, nbytes, 4096)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

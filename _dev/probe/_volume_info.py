"""Volume identity: filesystem name, label, reported sector size, flags."""

import ctypes

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.GetVolumeInformationW.argtypes = [
    ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint,
    ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint),
    ctypes.POINTER(ctypes.c_uint), ctypes.c_wchar_p,
]
k32.GetVolumeInformationW.restype = ctypes.c_int

FLAGS = {
    0x00000001: "FS_CASE_IS_PRESERVED",
    0x00000002: "FS_UNICODE_STORED_ON_DISK",
    0x00000004: "FS_PERSISTENT_ACLS",
    0x00000008: "FS_FILE_COMPRESSION",
    0x00000010: "FS_VOL_IS_COMPRESSED",
    0x00010000: " Supports object IDs",
    0x00020000: " Supports named streams",
    0x00040000: " Supports sparse files",
    0x00080000: " Supports hard links",
    0x00100000: " Supports compression",
}

for root in ("C:\\", "D:\\"):
    fs = ("c_wchar" * 0) if False else ctypes.create_unicode_buffer(256)
    label = ctypes.create_unicode_buffer(256)
    sfo = ctypes.c_uint(0)
    cs = ctypes.c_uint(0)
    flags = ctypes.c_uint(0)
    ok = k32.GetVolumeInformationW(root, label, 256, ctypes.byref(sfo),
                                   ctypes.byref(cs), ctypes.byref(flags), fs)
    if ok:
        on = ", ".join(n for v, n in FLAGS.items() if flags.value & v)
        print(f"{root}  FS={fs.value}  label={label.value}  "
              f"sector-per-cluster={sfo.value}  component-count={cs.value}  "
              f"flags=0x{flags.value:08x} [{on}]")
    else:
        print(f"{root}  -> {ctypes.WinError(ctypes.get_last_error())}")

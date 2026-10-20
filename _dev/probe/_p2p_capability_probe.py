"""Decide whether PCIe peer-to-peer between the CMP 170HX pair is available.

Three independent witnesses, because they fail for different reasons and the
distinction matters when deciding what (if anything) to try next:

  * CUDA runtime   -- cudaDeviceCanAccessPeer / cudaDeviceEnablePeerAccess
  * NVML           -- nvmlDeviceGetP2PStatus, which names *why* (GPU vs chipset
                      vs regkey vs platform)
  * measured cost  -- what the no-P2P fallback (host bounce through copy
                      engines) actually delivers, i.e. what P2P would have to
                      beat

Usage:
  python _p2p_capability_probe.py            # capability only
  python _p2p_capability_probe.py --bench    # + cross-device copy bench
"""

import argparse
import ctypes
import glob
import os
import sys
import time

P2P_STATUS = {
    0: "OK",
    1: "CHIPSET_NOT_SUPPORTED",
    2: "GPU_NOT_SUPPORTED",
    3: "IOH_TOPOLOGY_NOT_SUPPORTED",
    4: "DISABLED_BY_REGKEY",
    5: "NOT_SUPPORTED",
    6: "UNKNOWN",
}
P2P_CAPS = ((0, "READ"), (1, "WRITE"), (2, "ATOMICS"))


def cudart():
    import torch

    libdir = os.path.join(os.path.dirname(torch.__file__), "lib")
    cands = sorted(glob.glob(os.path.join(libdir, "cudart64*.dll")))
    if not cands:
        raise SystemExit(f"no cudart dll under {libdir}")
    return ctypes.WinDLL(cands[0]), cands[0]


def probe_cuda():
    import torch

    lib, path = cudart()
    lib.cudaGetErrorString.restype = ctypes.c_char_p
    print(f"cudart: {path}")
    n = torch.cuda.device_count()
    print(f"torch {torch.__version__} / cuda {torch.version.cuda} / devices {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name} sm_{p.major}{p.minor} {p.total_memory / 2**30:.1f}GiB uuid={p.uuid}")

    print("\n== cudaDeviceCanAccessPeer ==")
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            can = ctypes.c_int(0)
            rc = lib.cudaDeviceCanAccessPeer(ctypes.byref(can), i, j)
            print(f"  {i}->{j}: rc={rc} can={can.value}")

    print("\n== cudaDeviceEnablePeerAccess ==")
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lib.cudaSetDevice(i)
            rc = lib.cudaDeviceEnablePeerAccess(j, 0)
            msg = lib.cudaGetErrorString(rc).decode()
            # A failed enable leaves the sticky error set; without this the next
            # unrelated kernel launch surfaces "peer access is not supported".
            lib.cudaGetLastError()
            print(f"  {i}->{j}: rc={rc} ({msg})")
    lib.cudaSetDevice(0)


def probe_nvml():
    try:
        nvml = ctypes.WinDLL("nvml.dll")
    except OSError as exc:
        print(f"nvml unavailable: {exc}")
        return
    if nvml.nvmlInit_v2() != 0:
        print("nvmlInit_v2 failed")
        return
    try:
        buf = ctypes.create_string_buffer(80)
        nvml.nvmlSystemGetNVMLVersion(buf)
        ver = buf.value.decode()
        nvml.nvmlSystemGetDriverVersion(buf)
        print(f"NVML {ver} / driver {buf.value.decode()}")

        n = ctypes.c_uint(0)
        nvml.nvmlDeviceGetCount_v2(ctypes.byref(n))
        handles = []
        for i in range(n.value):
            h = ctypes.c_void_p()
            nvml.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h))
            handles.append(h)
            name = ctypes.create_string_buffer(96)
            nvml.nvmlDeviceGetName(h, name, 96)
            print(f"  [{i}] {name.value.decode()}")

        print("\n== nvmlDeviceGetP2PStatus ==")
        for i in range(len(handles)):
            for j in range(len(handles)):
                if i == j:
                    continue
                cells = []
                for cap, capname in P2P_CAPS:
                    st = ctypes.c_uint(99)
                    rc = nvml.nvmlDeviceGetP2PStatus(handles[i], handles[j], cap, ctypes.byref(st))
                    cells.append(f"{capname}={P2P_STATUS.get(st.value, st.value) if rc == 0 else f'rc{rc}'}")
                print(f"  {i}<->{j}: " + " ".join(cells))

        print("\n== nvmlDeviceGetTopologyCommonAncestor ==")
        for i in range(len(handles)):
            for j in range(i + 1, len(handles)):
                anc = ctypes.c_uint(99)
                rc = nvml.nvmlDeviceGetTopologyCommonAncestor(handles[i], handles[j], ctypes.byref(anc))
                # rc=2 (NVML_ERROR_NOT_SUPPORTED) is expected: topology is Linux-only.
                print(f"  {i},{j}: rc={rc} ancestor={anc.value if rc == 0 else 'n/a'}")
    finally:
        nvml.nvmlShutdown()


def bench(src_dev, dst_dev):
    import torch

    print(f"\n== cross-device copy, cuda:{src_dev} -> cuda:{dst_dev} ==")
    print(f"{'size':>8} {'direct ms':>10} {'staged ms':>10} {'staged GB/s':>12}")
    for nbytes, iters in ((4 << 10, 200), (64 << 10, 200), (1 << 20, 100), (16 << 20, 40), (256 << 20, 10)):
        src = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{src_dev}")
        dst = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{dst_dev}")
        host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        src.fill_(1)

        for _ in range(3):
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize(dst_dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize(dst_dev)
        direct = (time.perf_counter() - t0) / iters

        for _ in range(3):
            host.copy_(src, non_blocking=True)
            dst.copy_(host, non_blocking=True)
        torch.cuda.synchronize(dst_dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            host.copy_(src, non_blocking=True)
            dst.copy_(host, non_blocking=True)
        torch.cuda.synchronize(dst_dev)
        staged = (time.perf_counter() - t0) / iters

        print(f"{nbytes / 1024:>7.0f}K {direct * 1e3:>10.4f} {staged * 1e3:>10.4f} {nbytes / staged / 1e9:>12.3f}")
        del src, dst, host
        torch.cuda.empty_cache()

    a = torch.randn(512, 512, device=f"cuda:{src_dev}")
    print(f"sanity matmul on cuda:{src_dev}: {(a @ a).sum().item():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--src", type=int, default=1)
    ap.add_argument("--dst", type=int, default=2)
    args = ap.parse_args()

    probe_cuda()
    print()
    probe_nvml()
    if args.bench:
        bench(args.src, args.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())

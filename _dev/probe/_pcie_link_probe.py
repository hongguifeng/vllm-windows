"""Measure the PCIe path each visible GPU actually negotiates and delivers.

Two questions this answers before any engine starts:
  * which generation/width does the link train to under load (idle readings lie);
  * what do H2D / D2H / GPU->GPU copies of PP-boundary sizes actually cost.

Run it with CUDA_VISIBLE_DEVICES already narrowed to the cards under test:

  $env:CUDA_VISIBLE_DEVICES='1,2'
  python _dev/probe/_pcie_link_probe.py
"""

import statistics
import subprocess
import threading
import time

import torch

MIB = 2**20


def gbps(nbytes: int, seconds: float) -> float:
    return nbytes / seconds / 1e9


def timed(dev: int, fn, reps=10, warmup=3) -> list[float]:
    torch.cuda.set_device(dev)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(dev)
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(dev)
        out.append(time.perf_counter() - t0)
    return out


def sample_link(stop: threading.Event, seen: dict) -> None:
    while not stop.is_set():
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,pcie.link.gen.current,pcie.link.width.current",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        ).stdout
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            uuid, gen, width = parts
            key = uuid.removeprefix("GPU-")
            prev = seen.get(key)
            cur = (int(gen), int(width))
            if prev is None or cur > prev:
                seen[key] = cur
        time.sleep(0.2)


def main() -> None:
    n = torch.cuda.device_count()
    print(f"torch {torch.__version__} cuda {torch.version.cuda} visible={n}")
    props = [torch.cuda.get_device_properties(i) for i in range(n)]
    for i, p in enumerate(props):
        print(
            f"  logical {i}: {p.name} uuid={p.uuid} bus={p.pci_bus_id} "
            f"total={p.total_memory / 2**30:.2f} GiB sm={p.multi_processor_count}"
        )

    # Raise every link with a sustained H2D stream, sampling what it trains to.
    host = torch.empty(512 * MIB, dtype=torch.uint8, pin_memory=True)
    devs = [torch.empty(512 * MIB, dtype=torch.uint8, device=f"cuda:{i}") for i in range(n)]
    stop = threading.Event()
    seen: dict = {}
    sampler = threading.Thread(target=sample_link, args=(stop, seen), daemon=True)
    sampler.start()
    for _ in range(6):
        for i in range(n):
            torch.cuda.set_device(i)
            devs[i].copy_(host, non_blocking=True)
            torch.cuda.synchronize(i)
    stop.set()
    sampler.join(timeout=3)

    print("\nlink under load (max observed over the stream):")
    for i, p in enumerate(props):
        gen, width = seen.get(str(p.uuid), (0, 0))
        print(f"  logical {i} bus={p.pci_bus_id} gen={gen} width=x{width}")

    print("\nhost<->device bandwidth (512 MiB pinned, median):")
    for i in range(n):
        k = 512 * MIB
        h2d = timed(i, lambda i=i: devs[i].copy_(host, non_blocking=True))
        d2h = timed(i, lambda i=i: host.copy_(devs[i], non_blocking=True))
        print(
            f"  logical {i}: H2D {gbps(k, statistics.median(h2d)):.2f} GB/s  "
            f"D2H {gbps(k, statistics.median(d2h)):.2f} GB/s"
        )

    if n >= 2:
        print("\nGPU->GPU copies (no P2P on this box, so these stage through host):")
        for label, elems in (("20 MiB", 20 * MIB // 4), ("4 KiB", 1024)):
            src = torch.ones(elems, dtype=torch.float32, device="cuda:0")
            dst = torch.empty(elems, dtype=torch.float32, device="cuda:1")
            times = timed(0, lambda: dst.copy_(src), reps=20, warmup=3)
            med = statistics.median(times)
            print(
                f"  {label} cuda:0->cuda:1: med {med * 1e3:.2f} ms "
                f"({gbps(elems * 4, med):.2f} GB/s)"
            )

    print("\nVRAM still free:")
    for i in range(n):
        free, total = torch.cuda.mem_get_info(i)
        print(f"  logical {i}: {free / 2**30:.2f} / {total / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()

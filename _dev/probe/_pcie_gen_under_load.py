"""Does the 170HX PCIe link train higher when the GPU is actually busy?

Copy-engine-only traffic may leave the card in a low power state; the engine
benchmarks are compute-bound, so measure H2D with and without a matmul load and
watch pstate + link gen at the same time.
"""

import statistics
import subprocess
import threading
import time

import torch

MIB = 2**20


def smi(fields: str) -> list[str]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def bench_h2d(dev: int, nbytes: int = 512 * MIB, reps: int = 5) -> float:
    torch.cuda.set_device(dev)
    host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    gpu = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{dev}")
    for _ in range(2):
        gpu.copy_(host, non_blocking=True)
    torch.cuda.synchronize(dev)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        gpu.copy_(host, non_blocking=True)
        torch.cuda.synchronize(dev)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def burn(dev: int, stop: threading.Event) -> None:
    torch.cuda.set_device(dev)
    a = torch.randn(4096, 4096, device=f"cuda:{dev}", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device=f"cuda:{dev}", dtype=torch.bfloat16)
    while not stop.is_set():
        for _ in range(20):
            a @ b
        torch.cuda.synchronize(dev)


def main() -> None:
    n = torch.cuda.device_count()
    print("idle:")
    for line in smi("index,pstate,pcie.link.gen.current,pcie.link.width.current"):
        print("  " + line)
    for i in range(n):
        dt = bench_h2d(i)
        print(f"  logical {i} idle H2D {512 * MIB / dt / 1e9:.2f} GB/s")

    stop = threading.Event()
    threads = [threading.Thread(target=burn, args=(i, stop), daemon=True) for i in range(n)]
    for t in threads:
        t.start()
    time.sleep(5)
    print("busy:")
    for line in smi("index,pstate,pcie.link.gen.current,pcie.link.width.current"):
        print("  " + line)
    for i in range(n):
        dt = bench_h2d(i)
        print(f"  logical {i} busy H2D {512 * MIB / dt / 1e9:.2f} GB/s")
    print("busy (after copies):")
    for line in smi("index,pstate,pcie.link.gen.current,pcie.link.width.current"):
        print("  " + line)
    stop.set()
    for t in threads:
        t.join(timeout=10)


if __name__ == "__main__":
    main()

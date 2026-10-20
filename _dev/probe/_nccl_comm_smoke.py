"""Windows NCCL smoke test v2: cross-card comm, transports, and the price of
each primitive.  Each rank is given its own single-GPU CUDA_VISIBLE_DEVICES so
the physical card is unambiguous, and the device UUID is printed.

Timed separately:
  * all_reduce of 16 MiB  -> what a TP layer boundary would cost, per layer
  * send/recv of 10 MiB    -> one PP stage boundary for a 2048-token prefill
                             chunk at hidden_size=2560 (2048*2560*2 B)
  * send/recv of 4 KiB     -> one PP stage boundary for decode

Safety: refuses to run if the visible card has less than MIN_FREE_MIB free, so
it cannot crowd out the :9393 service.  No weights, no engine.
"""

import os
import statistics
import time

import torch
import torch.distributed as dist

MIN_FREE_MIB = 700
rank = int(os.environ["RANK"])

torch.cuda.set_device(0)
props = torch.cuda.get_device_properties(0)
free_b, total_b = torch.cuda.mem_get_info()
print(
    f"[r{rank}] pid={os.getpid()} visible={os.environ.get('CUDA_VISIBLE_DEVICES')} "
    f"name={props.name} uuid={getattr(props, 'uuid', 'n/a')} "
    f"free={free_b / 2**20:.0f} MiB total={total_b / 2**20:.0f} MiB",
    flush=True,
)
if free_b < MIN_FREE_MIB * 2**20:
    raise SystemExit(f"[r{rank}] REFUSING: only {free_b / 2**20:.0f} MiB free")

dist.init_process_group(backend="gloo")
device = torch.device("cuda:0")

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator  # noqa: E402

t0 = time.perf_counter()
comm = PyNcclCommunicator(group=dist.group.WORLD, device=device)
print(
    f"[r{rank}] pynccl available={comm.available} disabled={comm.disabled} "
    f"version={comm.nccl_version} commInitRank={time.perf_counter() - t0:.2f}s",
    flush=True,
)
if comm.disabled:
    raise SystemExit(f"[r{rank}] pynccl disabled, nothing to test")

free_i, _ = torch.cuda.mem_get_info()
print(f"[r{rank}] VRAM after comm init: consumed {(free_b - free_i) / 2**20:.0f} MiB", flush=True)


def bench(name, fn, warmup=2, reps=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    print(
        f"[r{rank}] {name}: min={min(times):.2f} med={statistics.median(times):.2f} "
        f"max={max(times):.2f} ms (n={reps})",
        flush=True,
    )


def guarded(name, fn):
    try:
        bench(name, fn)
    except Exception as exc:  # noqa: BLE001
        print(f"[r{rank}] {name}: FAIL {type(exc).__name__}: {exc}", flush=True)


tiny = torch.ones(1024, dtype=torch.float32, device=device)  # 4 KiB
ar = torch.ones(4 * 1024 * 1024, dtype=torch.float32, device=device)  # 16 MiB
p2p = torch.ones(5 * 1024 * 1024, dtype=torch.float32, device=device)  # 20 MiB
gather_out = torch.empty(2 * 1024, dtype=torch.float32, device=device)
scatter_out = torch.empty(1024, dtype=torch.float32, device=device)
scatter_in = torch.ones(2 * 1024, dtype=torch.float32, device=device)

guarded("all_reduce 16MiB (x48/step ~= TP layer boundary)", lambda: comm.all_reduce(ar))
guarded("all_gather 4KiB", lambda: comm.all_gather(gather_out, tiny))
guarded("reduce_scatter 8KiB", lambda: comm.reduce_scatter(scatter_out, scatter_in))
guarded("send/recv 4KiB (PP decode boundary)", lambda: (comm.send(tiny, 1) if rank == 0 else comm.recv(tiny, 0)))
guarded("send/recv 20MiB (PP prefill-chunk boundary)", lambda: (comm.send(p2p, 1) if rank == 0 else comm.recv(p2p, 0)))
guarded("broadcast 4KiB", lambda: comm.broadcast(tiny, 0))

free_e, _ = torch.cuda.mem_get_info()
print(f"[r{rank}] VRAM at end: consumed {(free_b - free_e) / 2**20:.0f} MiB", flush=True)
dist.barrier()
dist.destroy_process_group()
print(f"[r{rank}] done", flush=True)

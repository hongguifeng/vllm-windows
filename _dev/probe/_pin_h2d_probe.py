#!/usr/bin/env python3
"""Is pinned host memory actually faster here than pageable H2D?

Both platforms in this comparison claim a pinned-memory effect: on WSL2 vLLM
disables pinning by default (``VLLM_WSL2_ENABLE_PIN_MEMORY`` gates it), and the
native WSL2 notes say decode drops 30-35% without it. This measures the raw
H2D path on whatever platform runs the script, so the Windows and WSL numbers
can be put side by side.

Small buffers on purpose: nothing here competes with a running engine.
"""
import time

import torch


def timed(fn, iters=200):
    """CPU submission cost only: sync happens once, after the loop."""
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    dt = (time.perf_counter() - t0) / iters
    torch.cuda.synchronize()
    return dt * 1e3  # ms


def timed_sync(fn, iters=50):
    """Real transfer cost: every iteration is fenced by a synchronize."""
    fn()
    torch.cuda.synchronize()
    best = None
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        best = dt if best is None else min(best, dt)
    return best * 1e3  # ms


def main() -> None:
    print("platform check")
    print("  torch:", torch.__version__)
    print("  device:", torch.cuda.get_device_name(0))
    print("  pin_memory available:",
          torch.cuda.get_device_properties(0))

    mb = 8
    nbytes = mb * 1024 * 1024
    pageable = torch.randn(nbytes // 4, dtype=torch.float32)
    pinned = pageable.pin_memory()
    dst = torch.empty_like(pageable, device="cuda")
    print(f"  buffer: {mb} MiB fp32; pinned={pinned.is_pinned()}")

    t_page = timed(lambda: dst.copy_(pageable, non_blocking=True))
    t_pin = timed(lambda: dst.copy_(pinned, non_blocking=True))
    s_page = timed_sync(lambda: dst.copy_(pageable, non_blocking=True))
    s_pin = timed_sync(lambda: dst.copy_(pinned, non_blocking=True))
    print(f"H2D pageable   submit {t_page:7.3f} ms   fenced {s_page:7.3f} ms  "
          f"({nbytes / s_page / 1e6:6.1f} GB/s)")
    print(f"H2D pinned     submit {t_pin:7.3f} ms   fenced {s_pin:7.3f} ms  "
          f"({nbytes / s_pin / 1e6:6.1f} GB/s)")
    print(f"  fenced pinned/pageable speedup: {s_page / s_pin:5.2f}x")

    tiny = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    tiny_p = tiny.pin_memory()
    dtiny = torch.empty_like(tiny, device="cuda")
    t_tiny_page = timed(lambda: dtiny.copy_(tiny, non_blocking=True), 2000)
    t_tiny_pin = timed(lambda: dtiny.copy_(tiny_p, non_blocking=True), 2000)
    s_tiny_page = timed_sync(lambda: dtiny.copy_(tiny, non_blocking=True), 300)
    s_tiny_pin = timed_sync(lambda: dtiny.copy_(tiny_p, non_blocking=True), 300)
    print(f"4-elem H2D pageable submit {t_tiny_page:7.4f}  fenced {s_tiny_page:7.4f} ms")
    print(f"4-elem H2D pinned   submit {t_tiny_pin:7.4f}  fenced {s_tiny_pin:7.4f} ms")

    # what vLLM's async_tensor_h2d does per step
    from vllm.utils.torch_utils import PIN_MEMORY, async_tensor_h2d
    print("vllm PIN_MEMORY constant:", PIN_MEMORY)
    t_async = timed(lambda: async_tensor_h2d([[1, 2, 3, 4]], dtype=torch.long,
                                             device=torch.device("cuda")), 500)
    print(f"async_tensor_h2d (1x4) {t_async:7.3f} ms")


if __name__ == "__main__":
    main()

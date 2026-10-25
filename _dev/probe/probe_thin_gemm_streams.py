"""Repro probe: does thin_gemm corrupt results when two streams share a lane?

Our decode path runs the shared expert on an aux stream while the main stream
runs the rest of the layer (vllm/model_executor/layers/fused_moe/runner/
shared_experts.py). thin_gemm keeps its FP32 split-K partials and arrival
counters in a per-(split_k, M, N) workspace, one lane per stream *when the
caller selects one* (``workspace_lane``). If both streams use the default lane,
two concurrent calls with the same key share the partials and the counters.

This probe runs the same call on two streams, concurrently, and compares each
result against a single-stream reference. A mismatch is the race.

  $env:CUDA_VISIBLE_DEVICES=2; $env:VLLM_THIN_GEMM=1; `
    .venv\\Scripts\\python.exe _dev\\probe\\probe_thin_gemm_streams.py
"""

import sys

import torch


def main() -> int:
    dev = torch.device("cuda", 0)
    import vllm.sm80.kernels.thin_gemm as tg

    if not tg.use_ampere_thin_gemm():
        print("VLLM_THIN_GEMM must be set and the part must be sm_80")
        return 1

    shapes = [(320, 10240), (640, 2560), (512, 2560), (48, 2560), (4, 10240)]
    ms = (1, 2, 4, 8, 16)
    for n, k in shapes:
        w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) * 0.02
        for m in ms:
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            want = torch.nn.functional.linear(x, w)
            single = tg.ampere_thin_gemm(None, x, w, None)
            single_ok = torch.equal(single, want) or (
                (single.float() - want.float()).abs().max().item()
                <= 4 * want.float().abs().max().item() / 128
            )

            # Two streams, same (M, N, K), same lane, launched back to back.
            s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
            out = {}
            torch.cuda.synchronize()
            with torch.cuda.stream(s1):
                out["a"] = tg.ampere_thin_gemm(None, x, w, None)
            with torch.cuda.stream(s2), tg.use_lane(1):
                out["b"] = tg.ampere_thin_gemm(None, x, w, None)
            torch.cuda.synchronize()
            ok_ab = all(torch.equal(out[k], single) for k in out)
            err = max(
                (out[k].float() - single.float()).abs().max().item() for k in out
            )
            print(f"N={n:<6} K={k:<6} M={m:<3} single={'OK' if single_ok else 'BAD'} "
                  f"two_stream={'SAME' if ok_ab else 'DIFF'} max_dev={err:.4f}")
            if not single_ok or not ok_ab:
                return 1
    print("OK: single-stream and two-stream results are identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())

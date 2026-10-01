"""离线证明：UVAOffloader 的 non-UVA 分支跑过一次 forward 之后，
被卸载的权重副本会以 CUDA caching allocator 的形式常驻显存（不还给驱动）。

这解释了实测现象：--cpu-offload-gb 1 --cpu-offload-params visual 只买到
"启动时静态少占 879 MiB"；一旦真跑过一张图，这 879 MiB 会以 reserved
（nvidia-smi 看到的 used）的形式回到卡上，并且此后一直占着 —— 在 KV pin
已经贴顶的配置下，这一步就是把 decode 推过界、开始每步换页的触发点。

不依赖 vLLM，纯 torch 复刻 vllm/model_executor/offloader/uva.py:111-149 的逻辑。
用一个小模块（默认 512 MiB）即可，不需要真的加载模型。
"""

import argparse

import torch
import torch.nn as nn
from torch.func import functional_call


class Blk(nn.Module):
    """最小可复刻的模块：一个 bf16 参数 + 用它的 forward。"""

    def __init__(self, nbytes: int):
        super().__init__()
        n = nbytes // 2  # bf16 = 2 bytes
        self.w = nn.Parameter(torch.ones(n, dtype=torch.bfloat16, device="cuda"))

    def forward(self, x):
        return x + self.w[: x.numel()].reshape(x.shape).to(x.dtype)


def stat(tag: str):
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated() / 2**20
    r = torch.cuda.memory_reserved() / 2**20
    print(f"  {tag:28} allocated={a:8.1f} MiB   reserved={r:8.1f} MiB", flush=True)
    return a, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=512, help="卸载的权重体积 (MiB)")
    ap.add_argument("--iters", type=int, default=4)
    args = ap.parse_args()

    print(f"device = {torch.cuda.get_device_name(0)}")
    print(f"offloaded weight = {args.mb} MiB\n")

    mod = Blk(args.mb * 2**20).cuda()
    stat("module built on GPU")

    # --- 复刻 _maybe_offload_to_cpu（uva.py:111-121，uva_offloading=False 分支）---
    cpu_data = mod.w.data.to("cpu").pin_memory()
    mod.w.data = cpu_data
    torch.cuda.empty_cache()  # 相当于启动刚完成、还没跑过图的干净状态
    print("\nafter offload + empty_cache:")
    stat("  baseline")

    # --- 复刻 forward 包装（uva.py:124-149）---
    original_forward = mod.forward

    def forward(*a, **kw):
        mod.forward = original_forward
        device_state = {
            k: v.to("cuda", non_blocking=True) for k, v in mod.state_dict().items()
        }
        out = functional_call(mod, device_state, args=a, kwargs=kw, tie_weights=False)
        mod.forward = forward
        return out

    mod.forward = forward

    x = torch.ones(1024, dtype=torch.bfloat16, device="cuda")
    print("\n每次 forward（模拟每张图走一次视觉塔）:")
    prev_r = None
    for i in range(args.iters):
        y = mod(x)
        del y
        _, r = stat(f"  forward #{i + 1}")
        if prev_r is not None:
            print(f"       -> 相比上一次 reserved 变化 {r - prev_r:+.1f} MiB")
        prev_r = r

    print("\n结论判据：")
    print("  ・forward #1 的 reserved 增量 ≈ 权重体积  -> 权重被搬上卡并留成 allocator 缓存")
    print("  ・forward #2+ 增量 ≈ 0                    -> 此后一直复用，占用不释放")
    print("  ・weights 仍在 host（p.data 是 CPU）      -> 卸载没被撤销，但显存也没真的省下来")


if __name__ == "__main__":
    main()

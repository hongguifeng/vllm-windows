# Windows 上的 CPU KV 缓存卸载（PinnedCPUOffloadingSpec）

对象：`D:\code\vllm-windows`（原生 Windows，vLLM 0.29.1.dev0，CUDA 13.3 运行时）
目的：让多轮对话 / 多 agent 场景下被淘汰的旧 KV 落到主机内存，前缀命中时再拷回 GPU，
从而避免整段对话重新 prefill。

---

## 0. 结论摘要

**可以用了。** 上游的 KV offloading 在 Windows 上原本是**完全不可用**的，原因有两个，
而且两个都是硬性的：

| 组件 | 为什么在 Windows 上不行 |
|---|---|
| `SharedOffloadRegion` | 硬编码 `/dev/shm/...` 做 mmap + `MADV_POPULATE_WRITE`。Windows 没有 `/dev/shm`，也没有 `madvise`。 |
| `swap_blocks_batch` / `cuMemcpyBatchAsync` | CUDA 12.8+ 的批量拷贝 API。符号能经 `cuGetProcAddress` 解析到，但实际调用在 WDDM 下返回 `CUDA_ERROR_INVALID_VALUE`（err 1）；用 v13010 符号则是 `CUDA_ERROR_NOT_SUPPORTED`（err 400）。 |

所以本仓库新增了一个 spec：`vllm/v1/kv_offload/cpu/pinned_spec.py`
（`PinnedCPUOffloadingSpec`），它

- 用**普通 pinned host tensor**（`torch.pin_memory=True`）代替 `/dev/shm` mmap；
- 用 **`Tensor.copy_(..., non_blocking=True)`** 在专用 CUDA 流上逐块拷贝，
  代替 `cuMemcpyBatchAsync`。

调度器侧（`OffloadingConnectorScheduler`、`CPUOffloadingManager`、LRU/ARC 淘汰、
命中统计、job 生命周期）**一行没改**——那部分本来就是纯 Python，与平台无关。
只换了 worker。

### 实测结果

用 Qwen2.5-0.5B-Instruct（`gpu_memory_utilization=0.12`、`max_model_len=512`、
`kv_offloading_size=0.25`）跑两轮相同前缀：

```
R1: {'MISS': 1, 'prepare_store_keys': 25, 'store_jobs': 1, 'store_blocks': 25}
== reset_prefix_cache() ==
R2: {'HIT': 25, 'prepare_load_keys': 25, 'load_jobs': 1, 'load_blocks': 25}
```

第二轮 **25/25 命中，0 miss**——旧 KV 确实是从主机内存取回的，没有重新 prefill。

**数据正确性**：把 offload 路径产出的 token id 和"完全不启用 offload"的参考实现对比，
`TOKEN IDS MATCH: True`。即拷贝是逐字节正确的，不是"看起来能跑但 KV 是乱的"。

### 性能

- 大块连续传输（2 MiB）实测 **~1.5 GB/s**，与 WDDM 下 `cudaMemcpyAsync` 的上限一致。
- 小块（128 KiB）降到 ~0.87 GB/s，是每次拷贝固定开销导致的。
- **不打算靠它提速**：它的价值是少做 prefill。传输跑在低优先级旁路流上，
  且 store 被推迟到下一个 engine step（上游 `prepare_store_kv` 的设计），
  所以稳态 decode 不应该为它买单——但**上线前必须用 ITL/TTFT 对着
  `-KvOffloadGiB 0` 做一次 A/B**，别信这句话。

---

## 1. 怎么用

```powershell
# 开 16 GiB CPU 池
& D:\code\vllm-windows\start_server.ps1 -KvOffloadGiB 16

# 默认关闭 —— 不传这个参数，argv 里不会有任何 --kv-offloading-* 
& D:\code\vllm-windows\start_server.ps1
```

不带 `start_server.ps1` 时，直接给 vLLM 传参即可：

```powershell
python -m vllm.entrypoints.openai.api_server `
  --model <model> --kv-offloading-size 16 --kv-offloading-backend native
```

`--kv-offloading-backend native` 在 Windows 上会被 `vllm/config/vllm.py`
自动路由到 `PinnedCPUOffloadingSpec`，**无需手动指定 `spec_name`**。
若显式写了 `kv_connector_extra_config={"spec_name": ...}`，则尊重用户的选择。

### 观测指标

```
vllm:kv_offload_cpu_cache_usage_perc          # CPU 池占用比例，>0 说明真的在用
vllm:kv_offload_load_bytes / load_time        # CPU->GPU 读回
vllm:kv_offload_store_bytes / store_time      # GPU->CPU 存入
```

---

## 2. 实现要点

`vllm/v1/kv_offload/cpu/pinned_spec.py`

- **`_DirectionHandler`**：一个方向（GPU→CPU 或 CPU→GPU）。
  持有自己的 `torch.cuda.Stream`，用 `wait_stream(current_stream)` 保证
  读的是已经写好的 KV，再用 `wait_event(上一笔的 end_event)` 保证同一方向的
  传输严格串行（上游 invariant）。
- **`PinnedCPUOffloadingWorker`**：组合两个 handler，实现
  `submit_store` / `submit_load` / `get_finished` / `wait` / `shutdown`。
- **`PinnedCPUOffloadingSpec`**：`num_blocks` 由 `cpu_bytes_to_use` // 对齐后的
  `kv_bytes_per_chunk` 算出；scheduler 侧直接复用 `CPUOffloadingManager`。

CPU block 按 4 KiB 对齐分配（拷贝是按 `page_size_bytes` 切片做的，
对齐与否不影响正确性，但便于以后接 O_DIRECT 的 disk tier）。

### 已知限制

- `blocks_per_chunk > 1` 且两侧都 >1 时抛 `NotImplementedError`。
  默认 `blocks_per_chunk=1`，本仓库的用法也只用默认值。
- 只支持 CUDA-like / XPU；CPU-only 平台在 `get_worker` 就会报错。
- 与 WSL 无关：WSL 的 `/dev/shm` 是正常的，本来就能跑原版
  `CPUOffloadingSpec`（虽然本仓库的 `.env` 里它报 0% 占用，见
  `AB_WSL_VS_WINDOWS.md` §0）。

---

## 3. 改动清单

| 文件 | 改动 |
|---|---|
| `vllm/v1/kv_offload/cpu/pinned_spec.py` | 新增，全部实现 |
| `vllm/v1/kv_offload/factory.py` | 注册 `PinnedCPUOffloadingSpec` |
| `vllm/config/vllm.py` | Windows / `VLLM_KV_OFFLOAD_PINNED=1` 时默认用 pinned spec；用户显式 `spec_name` 时不覆盖 |
| `vllm/envs.py` | 新增 `VLLM_KV_OFFLOAD_PINNED`（默认 `"0"`） |
| `tests/v1/kv_offload/cpu/test_pinned_spec.py` | 新增 7 个测试 |
| `start_server.ps1` | 新增 `-KvOffloadGiB`，默认 0（关闭） |

`VLLM_KV_OFFLOAD_PINNED=1` 是为了在 Linux 上也能强制走 pinned 路径做对照实验。

---

## 4. 测试

```
tests/v1/kv_offload/cpu/test_pinned_spec.py ....... 7 passed
```

覆盖：spec 注册、`num_blocks` 推导（含向下取整）、manager 类型、
worker 分配 pinned 张量、**GPU→CPU→GPU 逐字节往返**、传输字节数上报。

跑法（Windows 上必须从仓库**外**运行，否则源码树会遮蔽已安装的包）：
见 `_dev` 里对应的临时脚本模式；GPU 需要 `CUDA_VISIBLE_DEVICES` 为空
（PowerShell 里能看到卡，bash 里看不到）。

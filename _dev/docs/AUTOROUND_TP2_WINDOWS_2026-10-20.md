# AutoRound 3bpw-MTP 跑 TP=2（Windows / 双 CMP 170HX）

日期 2026-10-20。目标：让 `Qwen3.8-Flash-Next-AutoRound-3bpw-MTP` 在两张 170HX 上做
TP=2。此前唯一闸门是 `vllm/models/qwen4_exp/nvidia/ple_ssd.py` 的 `TP=DP=1` 硬断言。

---

## 1. 改动

PLE 表有 95.37 GiB（320,001,536 行 × 320 B），既放不进单卡也不放不进主机内存
（TP=2 若用 pinned-host 每进程要 47.68 GiB，两进程 95.37 GiB），所以 SSD 流式必须保留，
只能把它改成 TP-aware。改动两处，共约 40 行：

| 文件 | 改动 |
|---|---|
| `vllm/models/qwen4_exp/nvidia/ple_ssd.py` | 断言从 `TP=DP=1` 放宽到 `DP=1`（另加 ETP 不得跨 DP 的检查）；`PLESSDTable` 接 `owner=(row_start, row_end)`，非本 rank 的行**不读盘、不进 cache**，直接零填；`_finalize_prefetch` 在拷进 `output` 前对本 rank 的行做 ETP all-reduce；启动日志打出本 rank 的行区间 |
| `vllm/models/qwen4_exp/nvidia/ngram_embedding.py` | `_reduce_etp_embeddings` 从 pinned-host 类上移到基类 `Qwen4ExpPLEEmbedding`，两个后端共用（逻辑未变） |

语义与 pinned-host 路径一致：每行只有一个 owner，owner 查自己的行、其余零填，
再在 ETP 组上求和 → 结果等于整表查询。TP=1 时 owner 是整表、`tp_size==1` 直接返回，
所以单卡路径等价于改动前（下面第 3 节验证）。

配套脚本：`_dev/probe/_autoround_tp2_smoke.ps1`（生产 argv + TP=2 + 无 P2P 开关）、
`_dev/probe/_autoround_tp2_stop.ps1`（从记录的 pid 按子进程后序杀）。
`_dev/out/logs/autoround_tp2_20261020_230821.stdout.log` 是这次 TP2 的原始日志。

---

## 2. 验证

### 2.1 行所有权（CPU，不占 GPU）

`_dev/probe/_ple_ssd_owner_probe.py`，真实 checkpoint、真实 128 个 PLE 分片：

- 表：`rows=320001536 dim=160 row_bytes=320` → 95.37 GiB；
- `owner=(0, half)` 与 `owner=(half, rows)` 各读一遍：本 rank 的行与整表读取**逐字节相同**，
  非本 rank 的行全零；
- 两个 owner 的输出相加**逐字节复现整表**（`owner sum reproduces the table`）。

### 2.2 TP=2 起服务

`health=200`，冷启动 5 分 24 秒（含 5 s 图捕获）。日志证据：

```
Worker_TP0: PLE SSD: 95.37 GiB on disk, 47.68 GiB in rows [0, 160000768) of 320001536
Worker_TP1: PLE SSD: 95.37 GiB on disk, 47.68 GiB in rows [160000768, 320001536) of 320001536
Worker_TP0/1: Actual usage is 29.06 GiB for consumed memory (weights)
GPU KV cache size: 2,172,916 tokens, Maximum concurrency for 262,144 tokens per request: 8.29x
Adaptive SD: decode CUDA graphs captured for draft counts [1, 2, 3].
```

行区间首尾相接、无重叠无缺口，每卡权重 29.06 GiB（单卡是 62.7 GiB），MTP 与
breakable CUDA graph 都正常。短请求里 spec decode 计数 29 drafts / 60 draft tokens /
22 accepted（接受率 36.7%，平均接受长度 2.76）。

### 2.3 单卡（TP=1）未被改坏的回归

同一棵树、`-Gpu 2 -Port 9402`：

- 贪心输出与生产（快照）**逐字节相同**（19 prompt / 51 completion）；
- `GPU KV cache size: 419,430`、`1.60x`，与生产一致；
- PLE 行区间 `[0, 320001536)`，即整表；
- bench 与生产在噪声内（见下表）。

---

## 3. 数字

口径：`_dev/probe/_bench_port.py`，prefill 4096 tok 单请求、4 并发各 128 tok、MTP k=3
+ adaptive_k、graphs 开、`max-model-len 262144`、`mem-util 0.96`。

| 臂 | 权重/卡 | KV tokens | 262k 并发 | prefill TTFT / 速度 | 4 路 decode | ITL 中位 |
|---|---|---|---|---|---|---|
| 单卡（生产，快照） | 62.7 GiB | 419,430 | 1.60× | 0.509 s / **8,041 tok/s** | **108.8 tok/s** | 27.7 ms |
| 单卡（本树，含本次改动） | — | 419,430 | 1.60× | 0.495 s / **8,273 tok/s** | **108.0 tok/s** | 28.3 ms |
| **TP=2（本树）** | 29.06 GiB | **2,172,916** | **8.29×** | 1.737 s / **2,358 tok/s** | **37.8 tok/s** | 40.9 ms |

结论：**TP=2 在这台机器上是拿速度换容量**——KV 池 **5.2×**（419k → 2.17M tokens，
262k 上下文并发 1.60× → 8.29×），代价是 decode **−65%**、prefill **−71%**。
原因与 `_dev/docs/PP_VS_TP_WINDOWS.md` 的推算一致：两卡之间没有 P2P，NCCL 走
`SHM/CE` 主机中转，48 层每层每步一次 all-reduce 把 GPU 等成瓶颈。

---

## 4. 边界与后续

- 前提：仓库树里的编译产物（`vllm\*.pyd`、`vllm\vllm_flash_attn\` 的 `layers/ops/cute`、PLE SSD 的
  native dll）都不在 git 里，换机器要先按
  [`EXL3_WINDOWS_PORT_STATUS.md`](EXL3_WINDOWS_PORT_STATUS.md) §6 补齐。
- TP=2 需要两张 170HX 全空 → 必须停掉单卡生产服务（本次停机约 12 分钟，已恢复 :9393）。
- SSD PLE 仍不支持 ETP 跨 DP（会显式报错）；要支持得再加 `_gather_dp_ids`/`_select_embeddings`。
- 未单独测量 PLE reduce 那一次集合通信的开销（每步每 PLE 层约 10 KiB，量级可忽略）。
- 想把 TP=2 的速度拉回来，只有三条路：真 P2P（硬件/AER 改不了）、
  换 PP（`config.py` 里 PLE 要求 `pp=1`，是更大的改动），或减少每层集合通信
  （`--enable-expert-parallel` 会改变 MoE 通信形态，值得单独 A/B）。

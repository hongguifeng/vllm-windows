# EXL3 recipe in WSL2 — 落地状态（2026-10-20）

目标：把 `Morrowmake/qwen38-flash-next-cmp170hx-recipe`（EXL3 5.05 bpw，
`turboderp/Qwen3.8-Flash-Next-exl3@7cef615f`）在这台机器上跑起来。
**它不能在当前 Windows 原生 fork 里跑**（无 EXL3 量化/内核、引擎是另一个 Linux 分支、
recipe 要求 TP2 而我们 PLE SSD 路径要求 TP=DP=1）；走 WSL2 + Docker 独立栈。

## 已完成（步骤 1）

- WSL 克隆：`~/qwen38-recipe`（LF 正常；Windows 侧 D:\code\_scratch 的克隆是 CRLF，不能直接用）。
- 引擎镜像已拉：`ghcr.io/morrowmake/vllm-cmp170hx@sha256:5b45c4d6...`（9.0 GiB 压缩）。
- 容器内验证：
  - `vllm 0.32.0rc1.dev249+g18305948a`（recipe 钉的 commit `18305948a`）。
  - `vllm/_exl3_C.abi3.so` 可 import；`vllm/sm80/kernels/exl3/{loader,quant,ops,dense_*}.py`、
    `vllm/models/qwen4_exp/nvidia/exl3_ple.py`、`csrc/exl3` 均在。
  - `VLLM_EXL3_*` 环境变量齐全。
  - `CUDA_VISIBLE_DEVICES=1,2` 时 torch 只看到两张 170HX（63.61 GiB each）。
- 镜像拉取第一次遇到 `short read`，重试后续传成功。

## 步骤 2：下载 checkpoint（进行中）

HF API 的 tree 列表被**共享 IP 限流**（429，0/500 remaining，hf-mirror 也把 API 302 回
huggingface.co），但 **resolve 文件 URL 不限流**（302→CDN，实测单流 ~9 MB/s）。
因此绕过 `hf download`：

1. `GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch 5.05bpw_h6_ng6` 到 `/tmp/hfrepo`
   （只拿 29 个文件的清单和 LFS 指针，不拉 115 GiB）。
2. `_dev` 外的脚本 `/mnt/d/code/_scratch/dl_model.sh`：按清单逐个
   `curl -L -C - https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/resolve/<rev>/<file>`
   下载到 `~/qwen38-recipe/models/Qwen3.8-Flash-Next-exl3/`，按 LFS 指针里的 size 校验。
   日志：`~/qwen38-recipe/logs/dl_model.log`。
3. 总量 **114.8 GiB**（11 个模型 shard ~80 GiB + ngram_embedding 39 GiB + 其余）。
   单流 ~9-11 MiB/s，预计 **3.5-4 小时**。
4. **坑**：`curl --retry ... -C -` 在内部重试时会从启动时快照的 offset 重来，
   36 GiB 的 `ngram_embedding.safetensors` 因此从 19.7 GiB 被截断重下。
   改为 `dl_ngram.sh`：**外层 shell 循环**，每次按 `stat` 当前文件长度显式
   `curl -C <have>`，不交给 curl 的 `--retry`；日志 `logs/dl_ngram.log`。
   范围请求的进度条 Total 是"剩余字节"，不是文件总大小，属正常。

当前进度见 `du -sb ~/qwen38-recipe/models/Qwen3.8-Flash-Next-exl3`；完成后
`bash docker/prepare.sh check` 应为 0（有 config、有 shard、index 里的 shard 齐全、
没有 `.cache/huggingface/download/*.incomplete`）。

## 步骤 3：启动（下载完成后）

```bash
cd ~/qwen38-recipe
# 冒烟：32K / eager / 无 MTP
# serve.sh 要求 FIRST_LIGHT=1 必须同时 SPEC_MODE=none、GRAPH_MODE=NONE、MAX_LEN=32768；
# README 明确 no-MTP 要 BOOT_CHECK=0（启动检查预期有 drafting）。
MIN_GPU_MIB=24000 CUDA_VISIBLE_DEVICES=1,2 P2P=off \
  FIRST_LIGHT=1 SPEC_MODE=none BOOT_CHECK=0 READY_TIMEOUT=3600 bash start.sh
# 正式：262144 ctx / MTP k=3 / PIECEWISE graph
MIN_GPU_MIB=24000 CUDA_VISIBLE_DEVICES=1,2 P2P=off BOOT_CHECK=1 READY_TIMEOUT=3600 bash start.sh
```

- `MIN_GPU_MIB=24000` 必须：否则 3090（24 GiB）会让 preflight 直接 die。
- `P2P=off`：WSL/Windows 驱动没有 peer access，recipe 的 auto 检查会失败并降级。
- 首次启动要冷编译 EXL3/FLA kernel，可能十几到几十分钟；`READY_TIMEOUT` 调大。
- 日志 `~/qwen38-recipe/logs/serve.log`；`bash start.sh status` 看 `/health` 和 KV 行。
- 冒烟 `bash start.sh smoke`；API 在 127.0.0.1:8000（Windows 侧 localhost 转发，
  或 `wsl -e curl`）。

## 预期与风险

- 卡是 **x8**（recipe 头条是 x16 + P2P）；`P2P=off` 下 TP2 的每层 all-reduce 走
  host-bounce。**152.8 tok/s(c1)/674.7(c8) 不可能复现**，先实测。
- WSL 内存 47 GiB；若加载/PLE 爆内存，再改 `.wslconfig` 调大（需 `wsl --shutdown`，
  会中断下载，放到下载完成后）。
- 跑起来会占住两张 170HX，与 Windows 侧 `:9393`/PP 测试不能同时用；端口 8000 不冲突。

## 步骤 3 实测（2026-10-20 07:00，服务已起来）

### 两个启动坑

1. **checkpoint 缺文件**：`dl_model.sh` 在 ngram（字母序最后一个大文件）处被我们替换成
   `dl_ngram.sh`，`preprocessor_config.json / quantization_config.json / tokenizer.json /
   tokenizer_config.json / vocab.json` 等 7 个小文件没下。`prepare.sh check` 只校验
   config+shards，所以 auto-start 提前放行 → 引擎报 `ReasoningConfig: failed to tokenize
   reasoning strings`。补下这 7 个文件后通过。
2. **`--prefix-match-unit 64` 与 KV block size 冲突**：本机 block size 被 mamba align
   推成 **1568**，不是 64 的倍数 → `ValueError: ... block sizes must be divisible by
   prefix_match_unit. Got [1568, 1568, 1568, 1568, 4, 1568]`。把 `serve.sh:283` 改成
   **`--prefix-match-unit 32`**（1568 = 32×49）后正常。这是本机环境差异，已记录。

### 正式模式（262144 ctx / MTP k=3 / PIECEWISE）

- 容器 `qwen38-flash-next`，`127.0.0.1:8000->8000`，Windows 侧
  `http://127.0.0.1:8000/v1/models` 返回 200，model=`qwen3.8-flash-next`，
  `max_model_len=262144`。
- **KV 池 1,051,409 tokens**（262144 ctx 下并发 4.01×），与 recipe 的 1,044,325 同量级。
- `init engine` 139.6 s（含 graph capture）；模型权重加载 39.69 GiB/卡、165 s。
- MTP 接受长度 2.3-4.0（boot check 的简单 prompt 到 4.0）。

### 实测速度（`_dev/probe/_exl3_bench.py`）

| 指标 | 本机（x8 + P2P off） | recipe 头条（x16 + P2P） |
|---|---|---|
| decode c1（计数 prompt，256 tok） | **74.1 tok/s**（72.1/76.1） | 152.8 |
| decode c4 聚合 | **247.1 tok/s**（211/283） | 357-459 |
| 冷预填 ~19.4k tok | **1,930 tok/s**（10.05 s） | 2,651 |
| TTFT（c1） | 0.077-0.183 s | — |

- 日志明确：`Custom allreduce is disabled because your platform lacks GPU P2P capability`
  → TP2 每层 all-reduce 走 NCCL SHM，是相对头条数字减半的主因；x8 是第二因素。
- 对比 Windows 侧 PP=2（AutoRound 3bpw）：c1 80 tok/s 略高，但 c4 只有 ~92-146 tok/s，
  EXL3 TP2 的并发吞吐明显更好；代价是 c1 和预填不如 PP。

### 运行/停止

```bash
# 现在正在跑（detached）
cd ~/qwen38-recipe && bash start.sh status
bash start.sh stop          # 只停这个 checkout 的容器
bash start.sh               # 再起（P2P=off 等环境变量要重带）
```

API 无 key、`127.0.0.1:8000`；`examples/` 里有 chat/stream/tool_call 示例。

## PP 原型（2026-10-20 10:04-10:46）：功能能跑，但更慢

目标：用 PP=2/TP=1 绕开 TP2 每层 all-reduce。做法是复制一套 `start_pp.sh` /
`serve_pp.sh`（LAYOUT=pp2 → TP=1, PP=2），并把 4 个补丁文件挂进容器覆盖原文件：

| 文件 | 补丁 |
|---|---|
| `sm80/kernels/exl3/loader.py` | 去掉两处 `pipeline_parallel_size != 1` 拒绝；按 `layers.N` 与本 rank 的 `start_layer/end_layer` 跳过其他 stage 的张量；其余非层张量先尝试 `model.load_weights`，失败再按 PP 跳过 |
| `models/qwen4_exp/nvidia/model_state.py` | 只有拥有 PLE 层的 rank 才建 n-gram context，其余返回 `uses_ngram_embedding=False` |
| `model_executor/models/config.py` | 删掉 PLE 的 PP 闸门（原假设"非首 rank 没有 input_ids"，在 V1 下不成立） |
| `v1/core/kv_cache_planning.py` | 构造 KV 张量时按 `group.layer_names` 过滤 `UniformTypeKVCacheSpecs` 里的 rank-agnostic 层；否则 PP 末 stage 为不属于它的 PLE 层建张量 → `allocate_kv_cache` 的 `StopIteration` |

结果：

- **PP=2 + 无 MTP：起得来并服务**，KV **1,404,025 tokens**（5.36×@262k），无报错。
- 速度反而更慢：c1 **29.9 tok/s**、c4 51.7、冷预填 1,270 tok/s；TP2 是 72.5 / 300.8 / 1,955。
- 原因在日志里：`VLLM_EXL3_DENSE_MMA is set but the gate is closed: (tensor, pipeline)
  parallel size (1, 2), not (2, 1)`，`DENSE_PREFILL_TC`、`MOE_PREFILL_TC` 同样被关；
  EXL3 的快 kernel **只支持 TP2/PP1**，PP 下退回慢路径。
- **PP=2 + MTP 仍然崩**：末 stage 在 memory profiling 阶段 native 崩溃（worker 退出码
  空），与 MTP drafter 在 PP 下的 dummy run 有关；要修得改 speculator/MTP（类似 Windows
  侧 `mtp.py` 的 PP 补丁）。
- 结论：**PP 功能上可支持（4 个小补丁），但速度不如 TP2**；要让它快，得把 EXL3 的
  dense/MoE kernel gate 从 (2,1) 扩到 (1,2)，并补 MTP×PP——是 kernel 工程，不是加载器补丁。
  因此最终仍用 TP2 + `VLLM_HOST_ALLREDUCE` + MTP k=4。

## 速度优化实验（2026-10-20 08:47-10:05）

当前最优配置（正在运行）：`GPU_UTIL=0.95、SPEC_N=4、VLLM_HOST_ALLREDUCE=1`，
`prefix-match-unit=16`；KV **1,020,688 tokens**（262k 下 3.89×）。

| GPU_UTIL | MTP k | HOSTSHM | c1 tok/s | c4 tok/s | KV tokens |
|---|---|---|---|---|---|
| 0.95 | 3 | off | 68.0-74.1 | 263.7 | 1,051,409 |
| 0.90 | 3 | off | 70.5 | 267.0 | 837,443 |
| 0.85 | 3 | off | 70.5 | 273.5 | 623,477 |
| 0.80 | 3 | off | 74.5 | 258.9 | 410,928 |
| 0.85 | 4 | off | 76.5 | 222.3* | 598,190 |
| 0.85 | 4 | on | 72.3 | 260.2 | 603,767 |
| 0.80 | 4 | on | 73.0 | 263.7 | 394,610 |
| **0.95** | **4** | **on** | **72.5** | **300.8** | **1,020,688** |

\* 该次 c4 有排队，per-request 与其它配置相同（~68-78 tok/s）。

结论：

1. **降显存利用率不是提速手段**：0.95→0.80 的 c1 变化在 ±10% 噪声内，c4 不升反降，
   而 KV 池从 1.05M 掉到 395k。卡在 62.4/65.5 GB 时没有 paging/降频：SM 1485 MHz、
   功耗 100-137 W（上限 250 W）、温度 40-42°C。用户记忆中的"显存满→变慢"在这个栈里
   没复现；已恢复 0.95 保容量。
2. **真正的瓶颈是 TP2 每层 all-reduce**：NCCL 40 KiB all-reduce 实测 **0.129 ms**，
   48 层 ≈ **6.2 ms/token**，占 ~13 ms/步的一半。`NCCL_ALGO=Ring`/`PROTO=LL`/单 channel
   都没有改善（LL 反而更差）。根源是 **没有 P2P**：`Custom allreduce is disabled ...
   lacks GPU P2P capability`，只能走 SHM。
3. **唯一可重复的提升**：`VLLM_HOST_ALLREDUCE=1`（fork 自带的 CMP 170HX 专用
   host-staged all-reduce，走 POSIX shm + 零拷贝 PCIe 读）+ **MTP k=4**：
   c4 263.7 → **300.8**（+14%），c1 与基线持平，KV 只少 3%。
4. **k=5 不可用**：`QSA ring capacity 12 must divide the attention block size 1616`；
   k=4 时 block=1616，要求 `prefix-match-unit=16`；k=3 时 block=1568，用 32。
5. **PP 不能用**：`sm80/kernels/exl3/loader.py` 明确要求 `pipeline_parallel_size == 1`
   （两处），加上 `qwen4_exp/nvidia/model_state.py` 的 PLE/PP 闸门；要 PP 必须改 EXL3
   loader（按 PP rank 切权重），不是一个参数。
6. 因此 recipe 头条 152.8(c1)/674.7(c8) 是 x16 + P2P 的；本机 x8 + 无 P2P 下
   c4 300.8 已到头条的 65-84%，但 c1/c8 只有 ~47%。

## 回滚

- 删容器：`cd ~/qwen38-recipe && bash start.sh stop`（只停它自己 checkout 的容器）。
- 删镜像/模型：`docker rmi <image>`；`rm -rf ~/qwen38-recipe/models/Qwen3.8-Flash-Next-exl3`。
- 不影响 `D:\code\vllm-windows` 与 `:9393`。

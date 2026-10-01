# WSL `.env` 单用户配置迁到原生 Windows：可行性与性能对照

对象：`~/testcode/qwen38-27b-rtx3090/.env` + `single-user/start_qwen.sh`（WSL，vLLM 0.28.0，Docker）
对照：本仓库 `D:\code\vllm-windows`（原生 Windows，vLLM 0.29.1.dev0）
结果文件：`_dev/out/_ab_results/wsl.json`、`_dev/out/_ab_results/win.json`、`_dev/out/_ab_results/win_fused.json`、
`_dev/out/_ab_results/padded71680.json`

---

## 0. 结论摘要

**能跑，而且 `.env` 三个"不可移植项"现在全部解决，`.env` 可完整还原。**

- `VISION=1` —— 那个 `vision-tower-cpu-offload.patch` **不必移植**，0.29 已把它上游化成
  `--cpu-offload-gb 1 --cpu-offload-params visual`（外加 `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1`）。
  实测视觉塔正常出图，且**对文本吞吐零成本**（ITL 28.72 vs 无塔基线 28.67 ms）。
- `--max-model-len 71680` —— **已完整恢复，且不花额外显存**。根因是 0.29 删掉了 0.28（实为 syv 补丁）
  的"滑动窗口桶优先"分桶特判，导致补齐层从 +3 膨胀到 full-attn 16→20 + mamba 48→50（计费层页数
  2992 → 3650，+22%）。把该特判用 `_dev/patches/port_wsl_patches.py --batch 3` 装回后，**同一个 6e9 pin 直接报出
  75,181 token / 1.05x，与 WSL 逐字一致**；吞吐相比 60928 无一处下降（C2/C4 反而 +8%）。
- `--kv-offloading-size 16 --kv-offloading-backend native` —— 后端硬编码 `/dev/shm`（Linux-only），
  而且 WSL 自己报 `kv_offload_cpu_cache_usage_perc=0.0`，本负载压根没用上，**删掉**。

在可用的配置下跑同一套负载，**原生 Windows 与 WSL2 基本打平**：

| 维度 | win/wsl | 说明 |
|---|---|---|
| prefill（1k / 4k / 16k，C=1） | 0.99x / 1.05x / 1.03x | 打平 |
| decode C=1 / C=2 / C=4 | 1.00x / 0.97x / 0.98x | 打平 |
| decode C=8 | **1.63x** | Windows 明显更快，因为 WSL 侧这一档在排队 |
| 接受长度 tok/step | 3.36–3.56 vs 3.40–3.46 | 完全一致 |
| TTFT（C=2/4/8） | 0.85x / 0.81x / 0.67x | Windows 更低 |

最值得注意的一点：**两边 tok/step 一模一样**，说明 DFlash2 投机解码路径在 Windows 上是**正确地**跑起来了，不是"跑得快但在胡说"。

> 注意：WSL 仓库 README 里 127 tok/s 的数字是 **250W 功耗墙**下测的（`nvidia-smi -pl 250`，容器改不了）。
> 本次 A/B 两侧都在本机当前 370W 下，所以**两侧互比有效**，但和 README 的历史数字不能直接比。

---

## 1. 实验装置

- 同一台机器、同一块 RTX 3090 24 GB，**分时独占**（一侧跑完停掉再跑另一侧）。
- WSL 侧：`docker compose --profile single up -d` 起的 `ghcr.io/syv-ai/qwen38-27b-rtx3090`，vLLM **0.28.0**，端口 18020。
- Windows 侧：本仓库 `_dev/bench/_ab_serve_win.sh` → 仓库 `.venv` 里的 vLLM **0.29.1.dev0**，端口 8000。
- **同一个客户端**打两边：本仓库 0.29 的 `vllm bench serve`（`_dev/bench/_ab_bench.py`）。
  这样不存在"谁的内置 bench 更偏心谁"的问题。
- 协议逐条对齐 WSL 的 `bench/run_benchmarks.sh single`：
  - warmup：`random` 256 in / 256 out，16 prompts，C=8
  - prefill 矩阵：`random` **1 个输出 token**，1024→16 prompts、4096→8、16384→4，全部 C=1
  - 真实 prompt 队列：`custom` + `prompts_real.jsonl`，`--custom-output-len 1024`，8 prompts，C=1/2/4/8，`--temperature 0`（贪心）
  - **每次 random 调用都用独立的 `--seed`**。WSL 的 runner 里明确写了原因：bench 默认 seed=0 会让每次调用拿到同样 prompt，
    叠加 `--enable-prefix-caching` 后后续调用变成"隐式部分前缀命中"，实测会把 prefill 读数带偏 15–20%。
  - `tok/step` 由 Prometheus 的 `spec_decode_num_drafts_total` / `spec_decode_num_accepted_tokens_total` 增量算出。

---

## 2. 参数映射：`.env` → Windows

传得过去、而且语义一致的：

| `.env` | WSL 生效形式 | Windows | 状态 |
|---|---|---|---|
| `SPEC=dflash2` | `--speculative-config {"method":"dflash","model":<DFlash2>,"num_speculative_tokens":7,"draft_sample_method":"probabilistic"}` | 完全相同 | ✅ |
| `PREFIX_CACHE=1` | `--enable-prefix-caching` | 相同 | ✅ |
| `CTX=fast` + `DFLASH_MAX_LEN=71680` | `--max-model-len 71680` | 60928 或 71680 | ⚠️ 见 2.2（能恢复，但有代价） |
| `KV_MEM=6000000000` | `--kv-cache-memory=6000000000` | `--kv-cache-memory-bytes 6000000000` | ✅（0.29 改了名，语义相同） |
| `VISION=1` / `VISION_OFFLOAD=1` / `VLLM_VISION_CPU_OFFLOAD_GB=1` | 靠 `vision-tower-cpu-offload.patch` | `--cpu-offload-gb 1 --cpu-offload-params visual` + `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1` | ✅ 见 2.1（**0.29 原生，不用移植补丁**） |
| `INT8_ACT=int8` / `INT8_LAYERS=mlp` | W4A8 Marlin（`VLLM_MARLIN_INPUT_DTYPE=int8` + `VLLM_MARLIN_INT8_INCLUDE_RE=mlp`） | 相同两个环境变量 | ✅ |
| `GPU_UTIL=0.93` | `--gpu-memory-utilization 0.93` | 相同 | ✅（但因 KV 手工 pin 而实际被跳过，见 4.2） |
| `ENABLE_THINKING=false` | `--default-chat-template-kwargs '{"enable_thinking": false}'` | 相同 | ✅ |
| `SSE_KEEP_ALIVE=30` | `--sse-keep-alive-interval 30` | 相同 | ✅（但有坑，见 4.1） |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` | 同 | 同 | ✅ |
| `CTX=fast` 附带的 `VLLM_SPEC_DECODE_ATTN=1` | start_qwen.sh 里 export | 相同 | ✅ |
| （start_qwen.sh） | `--mamba-ssm-cache-dtype float16` | 相同 | ✅ 传得过去，但**建议去掉**，见第 5 节 |
| `VLLM_WSL2_ENABLE_PIN_MEMORY=1` | WSL2 pinned-memory 变通 | 无需处理 | ➖ 原生 Windows 上 `is_pin_memory_available()` 直接返回 True，是 no-op |

**需要处理的只有两项**——真正删掉的只有一项（2.3），另一项是"能恢复但不建议"（2.2）：

### 2.1 `VISION=1` → **可以原样跑：0.29 已把那个补丁上游化了**

WSL 侧 `VISION=1` 能启动，靠的是 `patches/vision-tower-cpu-offload.patch` 把 0.85 GiB 的
视觉塔压在 pinned host RAM 里。这个补丁**本身是可移植的**——103 行、只动两个文件
（`envs.py` 加一个开关，`model_executor/models/qwen3_vl.py` 在 `Qwen3_VisionTransformer.__init__`
里包一层 offloader），而那两个改动区的锚点在 0.29 里**逐字符一致**。

**但不需要移植**，因为 0.29 已经把它做成了正式功能。三条事实拼起来就是等价物：

| 环节 | 0.29 的现成机制 | 位置 |
|---|---|---|
| 参数入口 | `--cpu-offload-gb` / `--cpu-offload-params` / `--offload-backend` | `vllm/config/offload.py` |
| 塔自动接入 | `_mark_tower_model` 在 context 退出时调 `get_offloader().wrap_modules(..., prefix=name)` | `models/interfaces.py:384` |
| 预算归属 | 塔在 `:515` 构造、语言模型在 `:523` → **塔先吃预算**；LM 层因参数名不含 `visual` 而完全不被卸载 | `models/qwen3_5.py` |

第 3 条正是原补丁"塔专属预算、与语言模型卸载预算互不干扰"的语义——不用自己写。
再补一个 `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1` 走批量拷贝路径，这是补丁在代码里做的同一件事
（`vision_offloader.uva_offloading = False`，原注释写着"UVA 零拷贝会让每个 GEMM 反复从 PCIe 读
操作数分块，这里慢得多"）。

实测（`--cpu-offload-gb 1 --cpu-offload-params visual`）：

```
INFO [base.py:138]  Offloader set to UVAOffloader
INFO [uva.py:65]    Total CPU offloaded parameters: 0.86
INFO [model_runner.py:404] Model loading took 15.35 GiB memory   (不卸载时 16.21 GiB)
```

一次真实图片推理（448×448 合成图，黄底 + 蓝圆）：

```
[image] 6.0s  content='Yellow, circle, 1111'
[image] usage={'prompt_tokens': 242, ..., 'multimodal_tokens': {'image': 196}}
```

塔确实在跑，而且输出由图像内容驱动（196 个 image token）。背景色与形状都答对；
"1111" 是我画了四根白色竖条代替字形，模型这么读是合理的。

> **注意**：第 3 节那批性能数字仍然是**纯文本**路径下取的（当时用 `--language-model-only`），
> 本节证明的是**配置层面 `VISION=1` 不再是"不可移植项"**，不是重新测过的多模态吞吐。

### 2.2 `--max-model-len 71680`：**装上卸载之后可以恢复**

同一个 `KV_MEM=6000000000`（5.59 GiB），0.28 装得下 71,680 上下文，0.29 装不下：

| | 0.28（WSL 容器） | 0.29（本仓库） |
|---|---|---|
| 71,680 上下文需要 | ≈5.2 GiB（估算）→ **过关** | **6.37 GiB** → 超出 5.59 GiB，拒绝启动 |
| 引擎实际报出 | 75,181 token / 1.05x | 降到 60,928 才有：61,303 token / 1.01x |

**原因不是"每个 token 的 KV 变大"，而是分组的补齐层被当成真实 KV 计入了每请求预算。**

第一，**页面本身的单价两边完全相同**——两版日志逐字一致：

```
INFO [interface.py:918] Setting attention block size to 448 tokens to ensure that attention page size is >= mamba page size.
INFO [interface.py:942] Padding mamba page size by 3.23% to ensure that mamba page size and attention page size are exactly equal.
```

（448 token/块 = 448 × 4 KV 头 × 256 head_dim × 2 (K,V) × 2 B = 每层每块 1,835,008 B；
mamba 页从 1,777,600 B 向上补齐到这同一个值。0.28 侧是 `interface.py:928/952`，内容相同。）
所以"贵"的不是 token，是**分桶补齐**。差别在层怎么分桶、补谁：

| | 0.28 | 0.29 |
|---|---|---|
| 日志 | `Sliding-window bucket(s) are the smallest; using group_size 8 (pads the sliding-window group **instead of** the full-attention/mamba layers)` + `Add 3 padding layers, 60.00%` | `Add 4 padding layers, 25.00%` + `Add 2 padding layers, 4.17%`（**0.29 删掉了那段滑动窗口特判**，`Sliding-window bucket` 全树 0 命中） |
| 补谁 | 只补 5 层的滑动窗口小桶（+3） | 补 full-attention 桶 16→20、mamba 桶 48→50 |
| 每请求计费层页数 | 160×16 + 9×48 = **2992** | 160×20 + 9×50 = **3650** → **+22%** |

推导依据：`max_memory_usage_bytes = cdiv(max_model_len, block_size) × page_size_bytes`，
而 `page_size_bytes` 是**该组所有层（含补齐层）页大小之和**。所以补齐层按真实 KV 一样向每个请求收费，
但永远不存 token。`group_size` 由启发式算出、**没有对外开关**，绕不过去。

关键点：`--kv-cache-memory-bytes` 是**硬上限**（它会跳过显存 profile，见 4.2），
所以日志里的"可用 KV 显存"就是 pin 值本身——**卸载省下的 0.86 GiB 不会自动进池子**。
把 pin 提到 `7000000000`（6.52 GiB）之后：

```
INFO [kv_cache_utils.py:2032] GPU KV cache size: 73,315 tokens,
                              Maximum concurrency for 71,680 tokens per request: 1.02x
```

即 **71,680 上下文与 `VISION=1` 可以同时恢复**，KV 池达到 WSL 的 97.5%（73,315 vs 75,181）。

**但"能跑"不等于"该这么跑"。** `pin=7000000000` 时 `nvidia-smi` 只剩 **196 MiB** 空闲显存，
实测立刻退化（同一客户端、同样 512 in / 128 out、C=1）：

| 配置 | Median ITL | Mean TTFT | 空闲显存 |
|---|---|---|---|
| 纯文本基线（无塔，60928/6e9） | 28.67 ms/step | — | — |
| **带塔 + 卸载（60928/6e9）** | **28.72 / 28.73 ms** | 382 / 414 ms | 820 MiB |
| 带塔 + 卸载（71680/**7e9**） | **128.8 ms/step** | **~10,000 ms** | **196 MiB** |

两件事一起看：

1. **视觉塔卸载对文本吞吐几乎零成本**（28.72 vs 28.67 ms/step，+0.2%）——
   因为 `--cpu-offload-params visual` 让语言模型层完全不进 offloader，
   塔的批量拷贝只在真正的图片请求里发生一次（`[image]` 4.6 s 含编码）。
2. 71680 那一档的 4.5 倍退化**不是卸载造成的，是 pin 顶到墙上**：
   `--kv-cache-memory-bytes` 跳过显存 profile（见 4.2），pin 多大就真占多大，
   剩 196 MiB 时 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` 的分配器只能反复
   cudaMalloc/cudaFree —— 表现为 TTFT 从 0.38 s 变成 10 s。

**所以：`VISION=1` 建议照搬（配 60928/6e9）；71680 要恢复的话得先给 pin 留余量，
不能把 `.env` 的 pin 直接抬到刚好够。**

### 2.2b 把那段特判改回去了：71680 现在在 **6e9** 就能跑，且吞吐不降

上面那段"滑动窗口桶优先"的特判**是 syv 自己的补丁**（`hybrid-sw-block-promote` /
`hybrid-kv-groups-v2-cudagraph`），不是 0.28 上游自带。所以可以整段搬回本树。
已通过 `_dev/patches/port_wsl_patches.py --batch 3` 落地到 `vllm/v1/core/kv_cache_utils.py`
（幂等，仓库树与 `.venv` 两份同步）。装上之后 **同一个 6e9 pin** 的日志：

```
INFO  [kv_cache_utils.py:1240] Sliding-window bucket(s) are the smallest;
      using group_size 8 (pads the sliding-window group instead of the full-attention/mamba layers)
WARNING [kv_cache_utils.py:1375] Add 3 padding layers, may waste at most 60.00% KV cache memory
INFO  [kv_cache_utils.py:2081] GPU KV cache size: 75,181 tokens,
      Maximum concurrency for 71,680 tokens per request: 1.05x
```

**75,181 token —— 与 WSL 0.28 报出的数字逐字一致**（1.05x 也一致）。
补齐从 0.29 的 `Add 4 (25.00%) + Add 2 (4.17%)` 变回只补 5 层滑动窗口桶 3 层，
计费层页数从 3650 回落到 2992（-18%）。

| 配置 | max_model_len | pin | KV 容量 | 并发 |
|---|---|---|---|---|
| 0.29 未打补丁 | 60,928 | 6e9 | 61,303 | 1.01x |
| 0.29 未打补丁，硬抬 pin | 71,680 | **7e9** | 73,315 | 1.02x |
| **0.29 + 补丁** | **71,680** | **6e9** | **75,181** | **1.05x** |
| 0.28 WSL 参照 | 71,680 | 6e9 | 75,181 | 1.05x |

也就是说**补丁把 71680 从"要多吃 0.93 GiB 显存"变回"零额外显存"**，
4.2 那节记载的 pin=7e9 显存墙（196 MiB 空闲 → ITL 128.8 ms）不再需要去碰。

而且**长上下文本身没有拖慢任何东西**——补丁后 71680 与补丁前 60928 同客户端对比：

| | 60928（未打补丁） | 71680（打补丁后） | 变化 |
|---|---|---|---|
| prefill 1k | 1838.8 t/s | 1889.7 t/s | +2.8% |
| prefill 16k | 1813.9 t/s | 1832.7 t/s | +1.0% |
| decode C1 | 121.33 t/s | 121.24 t/s | ±0 |
| decode C2 | 195.5 t/s | 212.9 t/s | +8.9% |
| decode C4 | 300.8 t/s | 324.2 t/s | +7.8% |
| decode C8 | 354.8 t/s | 364.3 t/s | +2.7% |
| Med ITL C1/C8 | 28.46 / 53.83 ms | 26.94 / 52.48 ms | 略优 |
| tok/step | 3.36–3.56 | 3.40–3.67 | 一致 |

C2/C4 的 +8% 在单次测量范围内，但**至少没有一处变差**，
`tok/step` 与 `accept_len` 两边一致说明 DFlash2 路径没有因为分组形态改变而退化。
（唯一的告警 "may waste at most 60.00%" 指的是那个 5 层小桶自己被补到 8，
绝对量远小于 0.29 补 16→20 + 48→50。）

⚠️ 两个已知边界：
1. 该补丁在 **MTP 配置下是 no-op**（函数提前返回），只对 DFlash2/无投机路径生效；
2. `group_size 8` 是启发式结果，换模型（层类型分布不同）后要重看日志确认。

### 2.2c 带视觉塔时 **71680 会崩**，60928 不会（实测 2/2 复现）—— **已修，见节末 2026-09-22 定案**

把 2.1 的塔卸载和 2.2b 的 71680 合在一起跑吞吐，得到的不是"慢一点"，是**引擎直接死**：

| 配置 | `device-side assert` 次数 | cohort 结果 |
|---|---|---|
| `--language-model-only` + 71680 | 0 | C1–C8 全通 |
| **塔在（卸载到 RAM）+ 71680** | **3 / 3（两次都是）** | **C2 或 C4 崩，之后全 500** |
| 塔在（卸载到 RAM）+ 60928 | 0 | C1–C8 全通 |

崩溃形态：`CUDA error: invalid argument` 出现在 `--async-scheduling` 的
`async_utils.py:168 copy_event.synchronize()`，但那是**异步上报点**，真正的错在前面——
日志里有 3 条 `device-side assert triggered`。表现为所有在途请求返回 500，
`EngineDeadError`，服务不可恢复。

**不是显存不够。** 四次运行的账本逐字相同：

```
Model loading took 15.35 GiB / Initial free memory 22.76 GiB
GPU KV cache size: 75,181（两个 71680 档完全一致）
Graph capturing finished in 6 secs, took 1.30 GiB
```

崩溃时 KV 只用到 26.5%，离满还差得远。所以是"塔在 + 71680"这个组合本身的问题，
具体哪个 kernel 越界没查（需要 `CUDA_LAUNCH_BLOCKING=1` 重跑，代价太大，没做）。

**实测吞吐（塔在 + 60928，与两个纯文本档对比）**：

| | TextOnly 60928 | TextOnly 71680 | **塔在 + 60928** |
|---|---|---|---|
| prefill 1k / 16k | 1839 / 1814 | 1890 / 1833 | 1879 / 1826 |
| C1 e2e / ITL | 121.3 / 28.46 ms | 121.2 / 26.94 ms | **121.6 / 26.93 ms** |
| C2 e2e | 195.5 | 212.9 | 203.6 |
| C4 e2e | 300.8 | 324.2 | 314.9 |
| C8 e2e / decode | 354.8 / 534.8 | 364.3 / 596.1 | **405.6 / 597.9** |
| tok/step | 3.36–3.56 | 3.40–3.67 | 3.27–3.72 |

**结论：塔卸载对文本吞吐零成本，再次确认**（C1 121.6 vs 纯文本 121.3，ITL 26.93 vs 28.46）。
塔在 + 60928 与纯文本 + 71680 基本等价，C2/C4 那 4% 落在波动内。
C8 的 405.6 是四次里最高，但单次测量，不单列结论。

根目录 `start_server.ps1` 因此把**带塔档的默认上下文定为 60928**，并拒绝 `vision + 71680`
（除非显式 `-AllowUnstable`）。
> ⚠️ **2026-09-22 起已翻回 `71680 + async`** —— 那条限制的根因既不是 async 也不是塔，
> 是 flashinfer top-k 在 CUDA 图重放下的缺陷。见本节末"2026-09-22 定案"。

**根因定位（判定实验，`vis71680noasync`）：是 `--async-scheduling`，不是塔也不是 71680 本身。**
WSL 那边的 `start_qwen.sh` 对 DFlash2 默认就是 `ASYNC_SCHED=0` → `--no-async-scheduling`
（注释：0.28 里 async 才是默认，是它主动关掉的）。本树所有基准一直开着 async。
把这一个开关翻过来重跑"塔在 + 71680"：

| | async ON（之前） | **async OFF（判定实验）** |
|---|---|---|
| 塔在 + 71680 + C1–C8 | **崩 2/2**（C2/C4 device-side assert） | **全通，0 次 assert** |

→ 触发条件 = **async-scheduling × 塔在 × 71680**，三个缺一不可（TextOnly+71680+async、
塔+60928+async、塔+71680+no-async 都干净）。为什么塔明明没被调用（cohort 是纯文本）
还会参与崩溃，没查到 kernel 级根因。

**但别急着改用 no-async——后续深挖（`_dev/bench/_itl_probe.sh` / `_dev/bench/_vis_itl_probe.sh` / py-spy）发现它是个"启动彩票"**：

| boot（全部 no-async + DFlash2 + 71680） | Graph capturing | 每步耗时 |
|---|---|---|
| 健康（py-spy 两次 + 带塔一次） | 5–6 s / **1.30 GiB** | **28.8–29.7 ms**（≈85 tok/s） |
| 半病（`vis71680noasync` 基准） | 10 s / 1.30 GiB | 131 ms（28.6 tok/s） |
| 重病（`_dev/bench/_itl_probe.sh` 首跑） | **23 s / 0.46 GiB** | **459 ms**（5.5 tok/s） |

- **重病 boot 有实锤：graph 只捕到 0.46 GiB（正常 1.30），且日志没有任何报错** ——
  静默的部分捕获让 decode 大面积落回 eager，慢 15 倍。与已知"1/3 启动在捕获期
  device-side assert"是同一个家族：有的 boot 崩，有的 boot 静默残废。
- **健康 boot 上 no-async 只比 async 慢 ~10%**（29 vs 27 ms/步），且与 WSL 的 no-async
  （27.9 ms/步）持平 —— **"Windows 特有 4.7 倍慢"是误判**，那是半病 boot 的数字。
  py-spy 也排除了 pin memory（原生 Windows 本来就 pin）。
- 机制（**2026-09-22 按代码订正**，原来写"重叠"不准确）：no-async 每步要把 draft token ids
  从 GPU 拷进 pinned CPU 缓冲（`gpu_model_runner.py:4966 _copy_draft_token_ids_to_cpu`：
  side stream + `wait_stream(default)` + `copy_(non_blocking=True)` + `event.record()`），
  再由 `post_step` 取走 —— `_get_draft_token_ids_cpu()` 里 `draft_token_ids_event.synchronize()`
  + `.tolist()`（`:5011`；`engine/core.py:696`）。`event.synchronize()` 等的是那条 copy stream，
  而它刚 `wait_stream(default)` 过 ⇒ **等于每步把整条流水线抽干一次**：CPU 等 GPU 收工 →
  建 Python list → 再重新提交下一步，GPU 在这段里空转。
  **async 不是把这段藏起来，是整段不执行**：`_copy_draft_token_ids_to_cpu` 在
  `use_async_scheduling and not (has_structured_output_requests or output_token_ids)` 时直接
  `return`（`:4974`），于是 `_draft_token_req_ids` 始终为 None → `take_draft_token_ids()` 早退
  （`:4961`），`post_step` 侧另有 `not self.async_scheduling` 守卫（`core.py:696`）。
  ⇒ 那 ~2 ms/步 = 每步一次 D2H + 一次 event sync + 一次 tolist 的**完全移除**。
  可证伪推论：**请求带结构化输出 / penalties / bad_words 时 async 照拷**，这 10% 应打折。
- ⚠️ **"Windows async ≈ WSL no-async"是个净额，别读成平台等价**（2026-09-22 澄清）：
  同开关口径下 Windows no-async（28.8–29.7）仍比 WSL no-async（27.9）慢 3~6%（≈1 ms/步），
  async 那 ~2 ms/步只是刚好把它抵掉，净 +3%（探针 27 vs 27.9；bench C1 26.94 vs 27.11）。
  **WSL 的 async 从未测过** —— 若这 10% 平台无关，WSL async 该落在 ~25 ms（即 Windows
  真落后 ~1 ms/步）；若 WSL async 也停在 27.9，则这 10% 是 Windows 特有的同步税。
  判别很便宜：WSL 侧把 `start_qwen.sh` 的 `ASYNC_SCHED` 翻成 1，用同一客户端跑一遍。

**修订后的实用结论**（⚠️ **2026-09-22 已过期**：崩溃根因已定位并修复，"async 会崩"与
"带塔必须 60928"两条都不再成立 —— 见本节末定案）：Windows 上这个模型，async 会崩（塔+71680）、no-async 会抽奖
（30→460 ms/步）。**带塔 60928 + async、TextOnly 71680 + async 仍是最稳组合**；
如果哪天必须 no-async，起服后核对 `Graph capturing finished ... took 1.30 GiB`，
数字明显偏小或捕获时间翻倍就重启服务。

**2026-09-21 凌晨补充——启动彩票实验（16 连测）：当夜的树没有复现任何一次。**
为验证"未初始化显存"假设，给 `speculator.py::capture` 注入了可开关的清零块
（`_dev/probe/_zerofill_exp.py --apply`，`ZEROFILL_EXP=1` 才激活）：清零 drafter 的全部
`input_buffers` + 空闲显存池。结果 **16 枪 0 崩溃、0 部分捕获**
（TextOnly 10 枪 + 带塔 6 枪，清零/对照各半，捕获全部 6–7 s / 1.30 GiB；
唯一异常是一次 13 s / 1.63 GiB，单次未解）。若历史"1/3"属实，
16 连健康的概率 ≈ 0.15% → **彩票不是当前树的固有属性**，它依赖某个今夜不存在的
条件：最可疑是当年的启动发生在代码/编译缓存频繁变动的时段（冷缓存），其次是
历史小样本把崩溃率估高了。用户长期日常使用 WSL 从未遇崩，与此相容。
清零补丁保留作诊断开关：**以后再遇启动 assert，设 `ZEROFILL_EXP=1` 重试——
如果它让原本必崩的 boot 通过，就是活体证据。** 实验脚本 `_boot_lottery*.sh`，
注意连环重启时 WDDM 显存释放可能滞后 >90 s，脚本必须轮询 nvidia-smi 等归零。

**2026-09-21 晚补充——2.2c 崩溃的上游对照（还没复现抓栈，先在案）**：

上游有三个高相关的 async × 多模态 × 投机崩溃族，崩溃矩阵与我们逐条吻合：

| 上游 issue | 组合 | 断言点 | 修复 | 本树状态 |
|---|---|---|---|---|
| #36906 | EAGLE3 × mm × async × 并发 | `embed_input_ids → F.embedding` 越界 token id（-1 占位符） | #37092 clamp | **已在**（`gpu_model_runner.py:3590`） |
| #38551 | MTP × mm × async | encoder cache 被投机过的 num_computed_tokens 提前驱逐 | #38907/#39543/#39544/#38622 | **部分在**（`scheduler.py:2327` 已按 `spec_lookahead - num_output_placeholders` 延迟释放） |
| #24799 | async × spec（无 mm） | `_prepare_input_ids` 的 `input_ids.gpu.scatter_` 索引越界 | 未见落地修复 | 未修（本树 `gpu_model_runner.py:1900`） |

⚠️ 但注意一个与上游都不完全吻合的点：**我们的 cohort 是纯文本**（2.2c 崩时没有任何
图片请求在飞），塔只是"加载着没调用"。所以 mm 竞态两族不直接适用；塔在场改变的是
文本请求的代码路径（走 mm 感知的 `_preprocess` 分支：`embed_input_ids(...,
multimodal_embeddings=[], is_multimodal=全 False)`）而非纯 token embedding 路径。
→ 当前第一嫌疑是 #24799 家族（async × spec 的 scatter/占位符），塔+71680 可能只是
把某个竞态窗口或缓冲区边界推过临界。
> ⚠️ **2026-09-22 结案：上表三族都不是真凶**，真凶是 flashinfer 的 radix top-k
> 在图重放下返回越界 candidate_id。见本节末"2026-09-22 定案"。

**下一步（复现包已备好）**：`.\_repro_async_tower.ps1` 用 `CUDA_LAUNCH_BLOCKING=1`
重跑原崩溃配置，拿到真实的出错 kernel + python 栈（现有日志的
`async_utils.py:168 copy_event.synchronize()` 只是异步上报点，不是病灶）。
`-Spec none` 变体做判别实验：塔+71680+async+无投机 如果全通，则投机解码是
必要成分，范围缩到 spec 路径。拿到栈后回查上游对应修复 PR 移植（走
`_dev/patches/port_wsl_patches.py` 的幂等补丁流程）。

**2026-09-21 20:40 第一枪结果：服务器侧健康，客户端打空（结论待重跑）**。
`serve_20260921_202101.log` 显示崩溃配置这次起得**完全正常**——补丁生效
（`Sliding-window bucket(s) are the smallest`）、`GPU KV cache size: 75,181 tokens ... 1.05x`、
`Graph capturing finished in 9 secs, took 1.30 GiB`（非半捕）。但 cohort 请求全被拒：
日志尾部是成片的 `POST /v1/completions → 404`，一个 CUDA 错误都没有。

原因不在服务端：`launcher.py:70` 打的路由表里 `/v1/completions POST` **存在**。
vLLM 在请求体 model 名与 `--served-model-name` 不匹配时回的是
**404 NotFoundError**，访问日志与"路由不存在"逐字相同。`_dev/bench/_bench_suite.py` 的 `SERVED`
常量此前漂成了完整模型路径，而根目录 `start_server.ps1` 传的是 `--served-model-name qwen3.8-27b`
→ 8/8 全废，四档 cohort 全是 0.0。（对照 `_dev/out/_ab_results/cohort_c1.log` 09-20 22:06 版：
那里 `served_model_name='qwen3.8-27b'`，是对的。）

已修：`_dev/bench/_bench_suite.py` 改为从 `/v1/models` 自动探测 model 名（`--served` 可覆盖）、
`Successful requests: 0` 时打印服务端错误正文并以 rc=1 退出；`_dev/probe/_repro_async_tower.ps1`
固定 `VBENCH_OUT=_dev/out/_bench_repro`，复现跑不再写进基线目录（本次曾覆盖
`_dev/out/_bench_base` 的 4 个 cohort 日志 + `SUMMARY.md`，已从 `_dev/out/_bench_base_170hx/` 原样还原，
数字另存于 `_dev/out/_ab_results/*.json`）。**结论：第一枪对崩溃问题没有结论，需重跑。**
教训：接口 404 先怀疑名字不匹配，再怀疑路由；跑测脚本必须限定输出目录。

**2026-09-21 20:47 第二枪：装置正确，但崩溃没复现——而 blocking 本身是嫌疑**。
`serve_20260921_204716.log`（422 行）：启动账本与崩溃时代逐字相同（`75,181 tokens ... 1.05x`、
`Graph capturing finished in 8 secs, took 1.30 GiB`），C1–C8 四档 **8/8 成功、0 失败**，
`CUDA error / device-side assert / Traceback` 计数 **0**。与当年崩溃配置逐项比对
（`_dev/bench/_ab_serve_win.sh` 的 `--max-num-seqs 8`、`--kv-cache-memory-bytes 6e9`、同一
compilation-config、同一 mm-processor-kwargs、`--cpu-offload-params visual`、async 开；
崩溃时代账本 `Initial free memory 22.76 GiB` 对应同是 3090）→ **唯一差别是
`CUDA_LAUNCH_BLOCKING=1`**。

推论：blocking 把每个 CUDA 调用串行化，**恰好抹掉了 async 调度出问题所需的重叠**——
"用 blocking 换精确栈"这个手段会吃掉它要抓的 bug。已于 `_dev/probe/_repro_async_tower.ps1`
加 `-Blocking 0`（真正的崩溃配置）；若在该模式下复现，下一步改用 compute-sanitizer
（`--tool memcheck --target-processes all`，覆盖独立 spawn 的 EngineCore），
而不是继续依赖 blocking。若 `-Blocking 0` 也不复现，则说明该崩溃在当前树上已消失，
需回查 09-20 之后的改动（首要嫌疑：`_dev/probe/_zerofill_exp.py` 注入 `speculator.py` 导致
torch.compile 缓存整体重编）。

**2026-09-21 21:02 第三枪：真崩溃配置（blocking OFF）仍然干净——但装置本身漏了一段状态**。
`serve_20260921_210224.log`：`-AllowUnstable True -Async True -MaxLen 71680`，且 env 段确认
**没有** `CUDA_LAUNCH_BLOCKING`（这是真配置，不是被 blocking 掩盖的那次）。C1–C8 四档
8/8 成功、0 失败，服务端 `CUDA error / device-side assert / Traceback` 计数为 0。
把能比的轴都比了——argv、启动账本（`Initial free memory 22.76 GiB`、`Model loading
15.35 GiB`、`75,181 tokens`、graph `1.30 GiB`，与崩溃时代逐字相同）、env（差异只有 VS Code
带来的 `PYTHON_BASIC_REPL`/`PYTHONSTARTUP`）、客户端（两边同一份 `vllm bench serve`、
同一 prompt 文件、同参数）、机器（同一张 3090）——**全部打平**。源码树唯一差异是
`speculator.py` 里 09-21 00:02 注入的清零块（`ZEROFILL_EXP` 未设时是空操作）。

**真正的差异在阶段序列，而且是我的复现装置漏的**：崩溃时代的 `_dev/bench/_ab_bench.py` 默认
`--only warmup,prefill,cohort`，**在 cohort 之前先跑 warmup 和长 prefill 阶梯**
（`--lens 1024,16384` → 1024×16 与 16384×4；即 cohort 开跑前 KV 里已经躺过 4 条
16,384 token 的请求），而这三枪全部用了 `--only cohort`。崩溃档的 JSON 里 prefill 行确实
排在失败的 cohort 行之前，所以那个长上下文状态是**每次崩溃都在场、而我的复现每次都缺**
的变量。已修：`_dev/bench/_bench_suite.py` 新增 `--lens`（默认矩阵不变，可覆盖成当年形状——
两者的 warmup/prefill 参数经比对逐字相同），`_dev/probe/_repro_async_tower.ps1` 默认改为
`-Stages warmup,prefill,cohort -Lens 1024,16384`。下一枪按时代序列跑。

**2026-09-22 定案：根因不在 vLLM 的缓冲里，在上游 flashinfer 的 top-k——已修。**
前面所有线索（阶段序列 / boot 状态 / 清零块 / 竞态）都是真的，但都不是原因。真正的因果链：

drafter 的候选选择走 `LogitsProcessor.get_top_k_tokens` → `_topk` → `flashinfer.top_k`。
本模型形状（vocab 248320、k=16）被 flashinfer 判给 `RadixTopKMultiCTA`：
`ctas_per_group = ceil(V / max_chunk_elements) = 11`，一行输出由一个 11-CTA 的 group
经**全局显存里的软件屏障**（`arrival_counter`）同步，工作区取自进程级
`_get_cache_buf("radix_topk_row_states_<dev>")`，输出写进 `torch.empty`。
**该算子在 CUDA 图重放时屏障阶段会错位，collect 少写/不写，没人写的槽位就是 `torch.empty`
的残留** → 越界 candidate_id → `qwen3_dflash2.py:181` 的 codebook gather 触发
device-side assert → 引擎死、在途请求 500。

三条互相独立的证据：
1. **离线最小复现**（`_dev/probe/_repro_topk_graph.py`：不加载 vLLM，一张只含 top-k 一个节点的图，
   2400 次重放）命中 1 行错索引——`vals` 与参考一致、`idx` 多出第 17 名少了第 16 名；
   同一循环 eager 干净。
2. **不可重入**（`_dev/probe/_repro_topk_streams.py`）：两条流各重放一张图，**第一轮 200 s 都跑不完**
   （挂住）；同一 harness 换 `torch.topk` 是 ~1 ms/轮、零错。两个并发 launch 共用同一个
   `arrival_counter` / histogram / det_scratch，屏障永远等不到目标值。这解释了为什么只有
   `--async-scheduling` 会踩、且引擎里的命中率远高于任何串行 harness。
3. **引擎内地面真值**：图内守卫（`_dev/probe/_clamp_exp.py`）在 flashinfer 趟数出 `cand neg=15 / ge=23`，
   同窗 eager 20,150 步零越界；换成 `torch.topk` 后同 argv、同 boot 指纹下 **0/0**。

修复：`_dev/patches/fix_flashinfer_topk_graph_replay.py` 让 `_flashinfer_topk()` 默认返回 `None`
（即走 `torch.topk`，本来就是该函数的 fallback），`DFLASH2_TOPK_IMPL=flashinfer` 可切回。
代价图内实测 2.25x(1 req) → 1.62x(8 req)，即 **+0.06~0.18 ms/草稿步，占一个 decode step <1%**。
flashinfer 的 filtered 路径**不能**替代：`CanImplementFilteredTopK()` 要 128 KB smem/SM，
3090 只有 100 KB，flashinfer 自认本卡不可用。

验证（`serve_20260922_011603.log`；`-Async -MaxLen 71680 -AllowUnstable` + 塔，`-Clamp -Repeat 4`）：
4 趟 C1–C8、**零 assert / 零 500 / 零 `Never received`**；403 行 `[clamp]` 读到 `reads=20150`、
`cg=FULL`（证明走的是图重放路径而非退化的 eager），值恒 `cand neg=0 ge=0 | anchor neg=0 ge=0`；
启动账本与 no-async 路径逐项相同（75,181 tokens / 1.05x、6 s / 1.30 GiB）。

**连带后果：根目录 `start_server.ps1` 的默认值已于 2026-09-22 翻回 `async + 71680`。**
守卫不再是"绑 async"，而是"绑 `DFLASH2_TOPK_IMPL=flashinfer`"——它守的是那个已知 bug，
而不是一个与 bug 无关的开关（`-AllowUnstable` 仍可绕过）。

**两条被证伪的推断，记下来免得重走**：
- ~~"计数增量集中在批次边界 ⇒ 某个静态输入缓冲有槽位没写"~~ —— 那只是重放时序变了的相关性。
- ~~"C8 慢 2× 是 flashinfer multi-CTA stall"~~ —— flashinfer 无守卫时它就是基线
  （20.8 s / 52.08 ms）；慢的是"图内守卫 × flashinfer"这个组合，换掉 top-k 后守卫是免费的。

### 2.3 `EXTRA_ARGS` 里的 `--kv-offloading-size 16 --kv-offloading-backend native` → 删掉

这个后端把 `/dev/shm/vllm_offload_<engine_id>.mmap` 直接 mmap，**没有任何平台判断**
（`vllm/v1/kv_offload/cpu/shared_offload_region.py:98`），原生 Windows 上会
`FileNotFoundError: '/dev/shm/vllm_offload_....mmap'`，起不来。

而且**它在本负载里根本没被用到**：WSL 侧每个请求的 `kv_offload_cpu_cache_usage_perc` 都是 `0.0`。
所以删掉它不改变任何数字。`_dev/bench/_ab_serve_win.sh` 里用 `[ -d /dev/shm ]` 做守卫，
将来在 WSL/Linux 上再跑这个脚本会自动把它加回来。

---

## 3. 性能对照（同一客户端、同一协议）

### 3.1 prefill（C=1，纯算力）

| 输入长度 | WSL tok/s | Windows tok/s | win/wsl | WSL TTFT | Win TTFT |
|---|---|---|---|---|---|
| 1024 | 1855 | 1839 | 0.99x | 551 ms | 557 ms |
| 4096 | 1866 | 1954 | **1.05x** | 2195 ms | 2096 ms |
| 16384 | 1761 | 1814 | **1.03x** | 9306 ms | 9032 ms |

**打平。** 而且注意 WSL 的 README 里写 "prefill 1k ≈1850–1940 tok/s（带 `INT8_ACT=int8`）"，
我这次 WSL 实测 1855 —— 说明这套 A/B 装置本身是准的。

### 3.2 decode（真实 prompt，贪心，1024 输出）

| 并发 | 指标 | WSL | Windows | win/wsl |
|---|---|---|---|---|
| C=1 | decode tok/s | 125.8 | 125.6 | 1.00x |
| C=1 | med ITL (ms) | 27.11 | 28.46 | 1.05x |
| C=1 | tok/step | 3.40 | 3.56 | 1.05x |
| C=2 | decode tok/s | 237.8 | 229.6 | 0.97x |
| C=2 | tok/step | 3.40 | 3.36 | 0.99x |
| C=4 | decode tok/s | 406.9 | 396.8 | 0.98x |
| C=4 | tok/step | 3.42 | 3.43 | 1.00x |
| C=8 | decode tok/s | 328.1 | **534.8** | **1.63x** |
| C=8 | med ITL (ms) | 89.15 | 53.83 | 0.60x |
| C=8 | tok/step | 3.46 | 3.49 | 1.01x |

三点观察：

1. **C1–C4 完全打平**（±3%）。C1 单流 125.6 vs 125.8 tok/s，差 0.2%。
2. **tok/step 全程一致**（3.36–3.56 vs 3.40–3.46）→ DFlash2 的草稿/验证路径两边一致，
   这排除了"Windows 侧投机解码静默降级成逐个 token"这类最坏情况。
3. **C=8 反而 Windows 快 63%**。WSL 侧这一档是**退化**的（med ITL 89 ms，mean TTFT 2668 ms），
   明显在排队/抢占；Windows 侧 53.8 ms、TTFT 1783 ms。这不是 Windows 优化了什么，
   而是 WSL 那套 0.28 的调度在 8 并发 + 投机解码时把序列挤住了。
   所以"WSL 单用户配置"在 8 并发下并不是它的强项。

### 3.3 启动开销

| | WSL | Windows |
|---|---|---|
| `Model loading took` | — | 15.33 GiB / 34.9 s |
| `init engine (profile, create kv cache, warmup model)` | 52.45 s | **62.19 s**（其中 compilation 46.12 s） |
| KV 池 | 75,181 token @ 71,680 | 61,303 token @ 60,928 |

Windows 侧 init 慢约 10 s，基本就是 torch.compile 的差额；两边都开了 CUDA Graph，
`Capturing dflash2 CUDA graphs (FULL)` 在 Windows 上正常出现（草稿模型也是图模式）。

---

## 4. 过程中踩到的三个坑（都会让"看起来能跑"变成"数据是假的"）

### 4.1 `--sse-keep-alive-interval 30` 会把长 prefill 的 bench 请求判死

这是 `.env` 的 `SSE_KEEP_ALIVE=30` 直接带过来的。0.29 的 bench 客户端
（`vllm/_benchmarks/endpoint_request_func.py`）收到 `: keep-alive` 注释帧时，
会把该请求标记为 `Never received a valid chunk to calculate TTFT` 而**判为失败**。

判别实验（WSL 侧）：32768 token（TTFT 17.7 s）**通过**；49152（TTFT 33 s）和
65536（TTFT 48 s）**失败**。阈值正是 30 s。

所以 prefill 矩阵只能测到 32768 为止，超出就得把 `--sse-keep-alive-interval` 调大或去掉。
本次对照两侧都止于 32768（WSL 那条 65536 记录为失败行，已在 `wsl.json` 里标成 0）。

### 4.2 `--kv-cache-memory-bytes` 会**跳过显存 profile**

日志原话：

```
Initial free memory 22.76 GiB, reserved 5.59 GiB memory for KV Cache as specified by
kv_cache_memory_bytes config and skipped memory profiling. This does not respect the
gpu_memory_utilization config. ...
```

意思是：手工 pin KV 之后，vLLM **不再验证**高并发下的激活峰值能不能放得下。
所以 `GPU_UTIL=0.93` 这个参数在此配置下是**名义上的**，真正决定可用显存的是 `KV_MEM`。
实测该配置稳态只剩 ~820–950 MiB 余量，高并发下属于"没有安全边际"的状态。

**这条在 2.2 节被验证成了一个可量化的悬崖**：把 pin 从 6e9 抬到 7e9（为了换回 71,680 上下文）后，
空闲显存只剩 196 MiB，同一份负载的 TTFT 从 382 ms 变成 ~10,000 ms、ITL 从 28.7 ms 变成 128.8 ms，
而日志里没有任何 OOM 或报错——**它不会告诉你它变慢了**。所以调 `KV_MEM` 时务必
`nvidia-smi` 确认余量，别只看"起没起来"。

### 4.3 首轮 warmup 会吃 Triton JIT 编译尖峰

日志里有：

```
WARNING Triton kernel JIT compilation during inference: _prepare_dflash_inputs_kernel
WARNING Triton kernel JIT compilation during inference: _topk_topp_kernel
```

这两个内核**不在** init 期的 warmup 覆盖范围内，第一次真实请求时才编译。
WSL 的 runner 也写了同一件事（"第一次跑完重启再跑，只取第二次的数字，首轮会低 30–50%"）。
本次对照两侧都先跑了一轮 warmup 才取数。

---

## 5. 额外发现：`--mamba-ssm-cache-dtype float16` 在 0.29 上关掉了融合 GDN 解码内核

`.env` 的 `start_qwen.sh` 固定带 `--mamba-ssm-cache-dtype float16`。0.29 日志里能看到后果：

```
WARNING Qwen3.5 model specifies mamba_ssm_dtype='float32' in its config,
        but --mamba-ssm-cache-dtype='float16' was passed. Using the user-specified value.

Falling back to the Triton GDN decode path: the fused CUDA kernel requires a BF16 GDN
model with K=V=128, SiLU or sigmoid gating, non-interleaved GQA layout,
BF16 convolution cache, BF16 or FP32 recurrent state, and a GPU with compute capability 8.0+

GDN decode kernel: triton
```

判据在 `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`：

```python
FUSED_GDN_STATE_DTYPES = (torch.float32, torch.bfloat16)
...
or recurrent_state_dtype not in FUSED_GDN_STATE_DTYPES   # float16 -> 回退 Triton
```

最后一道门是 `hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp")`。
本机 `_C_stable_libtorch.pyd` 里**确实编进了**这个算子，所以只要把 `float16` 去掉、
让循环状态回到模型自己声明的 `float32`，就能直接启用融合内核（实测日志变成
`GDN decode kernel: cuda`）。

**但要诚实说明一个不对称**：这套"融合 vs Triton"的选择机制是 **0.29 才有的**。
我在 WSL 镜像里对着 `qwen_gdn_linear_attn.py`（1748 行）grep，
`FUSED_GDN` 出现 **0 次**，也没有 `GDN decode kernel:` 这行日志 —— 0.28 里根本不存在这个选择器
（0.28 只有 `third_party/flash_linear_attention/ops/fused_gdn_prefill_post_conv`，那是 **prefill** 算子）。

所以：

- 第 3 节的对照**不能**说成"两边都走 Triton 的同类比较"。准确说法是：
  **0.28 用什么 GDN decode 路径不去干预，Windows 侧因为 `.env` 这个 flag 被限制在 Triton 上，
  两边仍然打平。**
- 反过来说，如果 Windows 侧去掉这个 flag 能拿到明显收益，那说明 **WSL 那套 `.env` 参数本身
  在 0.29 上就不是最优解**，而不是 Windows 移植有短板。

### 5.1 去掉 float16 的代价：上下文从 60,928 掉到 52,416

不能只讲收益。同一个 `KV_MEM=6000000000` 下换成 float32 循环状态，引擎直接拒绝启动：

```
ValueError: To serve at least one request with the model's max seq len (60928),
(6.27 GiB KV cache is needed, which is larger than the available KV cache memory (5.57 GiB).
Based on the available memory, the estimated maximum model length is 52416.
```

`--max-model-len` 降到 52416 才起得来（KV 池 52,565 token）。
**即：融合 GDN 内核要用约 8.5k token 的上下文来换。** 是否值得取决于用途。

### 5.2 换内核到底快多少：**没能测出来**

我试了一轮（`AB_DROP_MAMBA=1 AB_MAX_LEN=52416`，日志确认 `GDN decode kernel: cuda`，
`init engine 61.01 s`），但这一轮**数据不可信**，列出来只为标记"待重测"：

| 并发 | win（Triton）e2e | win_fused（cuda）e2e | 比值 |
|---|---|---|---|
| C=1 | 121.3 | （客户端中断，缺） | — |
| C=2 | 195.5 | 183.6 | 0.94x |
| C=4 | 300.8 | 264.1 | 0.88x |
| C=8 | 354.8 | 262.4 | 0.74x |

判断"不可信"的依据：

- **C=1 与全部 prefill 测点直接失败**（客户端进程中途消失、无报错、无结果；同一时刻服务端
  日志显示请求正常 200 返回、`generated_tokens=1024 elapsed_s≈10`，即服务端没问题）。
- **C=8 的 `mean TTFT` 是 8202 ms**（Triton 那轮是 1783 ms，4.6 倍），说明这一档在严重排队，
  它那个看起来更高的 `decode 655 tok/s` 是测量窗口的产物，不能当收益。
- 三个并发档的 **e2e 吞吐一致地更差**（0.94/0.88/0.74），而 TTFT 一致地更差，
  更像整轮测量被污染，而不是"换个内核变慢了"。

**结论：融合 GDN 的机制与代价是确凿的；收益未被证实。** 要下结论必须重跑一轮干净的对照
（同一台机器、服务端与客户端都稳定、先跑一轮 warmup 取第二轮数字）。本次不建议据此改配置。

---

### 5.3 环境层面的坑（影响复现，不影响上面的结论）

跑这批对照时，**Windows 侧的长驻服务端进程两次被外部终止**（一次在 warmup 中途，
一次在 fused 复测中途）。特征是：

- 服务端日志结尾是 vLLM 自己的 shutdown 路径，最后抛
  `PermissionError: [WinError 5] 拒绝访问`（`kill_process_tree` 阶段），
  说明它是**收到终止信号后进入关闭流程**，不是自己崩的；
- 客户端侧表现为 `ConnectionRefusedError: [WinError 1225]`，或客户端进程静默消失、日志无结果段。

**这不影响第 3 节的数字**（那一轮 `win.json` 是完整跑完的，服务端全程 `health=200`）。
但要复现/继续测的话，建议**把服务端和基准客户端放进同一个后台任务里串联执行**
（起服务 → 轮询 `/health` → 跑基准 → 关服务），不要拆成两个长驻后台任务 —— 拆开更容易被误伤。

---

## 6. 复现方式

### 6.0 日常启动：根目录 `start_server.ps1`

`_dev/bench/_ab_serve_win.sh` 是给 A/B 用的（参数都写死在脚本里）。日常起服务用根目录 `start_server.ps1`，
它把上面验证过的一套参数做成开关，并在启动前自动跑补丁、做前置体检：

```powershell
& D:\code\vllm-windows\start_server.ps1                 # 27B W4A16 + DFlash2 + 视觉塔，71,680 + async（默认）
& D:\code\vllm-windows\start_server.ps1 -TextOnly       # 去掉视觉塔
& D:\code\vllm-windows\start_server.ps1 -NoAsync        # 回到 --no-async-scheduling（decode 慢 ~10%）
& D:\code\vllm-windows\start_server.ps1 -Spec mtp       # MTP 代替 DFlash2（并发高时用，自动降到 60,928）
& D:\code\vllm-windows\start_server.ps1 -Spec none      # 不要投机解码
& D:\code\vllm-windows\start_server.ps1 -Force          # 强杀占住 29550 / 8000 的残留进程
& D:\code\vllm-windows\start_server.ps1 -DryRun         # 只打印 argv
```

启动前它会：跑 `_dev/patches/port_wsl_patches.py --batch 3` + `_dev/patches/fix_winloop_import.py`（幂等）、
检查 29550 与 8000 是否被占、按 free VRAM 判断 `权重 15.8 GiB + KV pin` 是否装得下。
启动后按提示去日志里核对两行：

```
Sliding-window bucket(s) are the smallest   <- 补丁生效
GPU KV cache size: 75,181 tokens ... 1.05x
```

实测 160 s 到 healthy。**启动偶发**：约 1/3 的启动曾在 DFlash2 的 CUDA graph 捕获阶段
抛 `CUDA error: device-side assert triggered`（重试即可，从未连续两次）。2026-09-22 定位的
根因（flashinfer top-k 在图重放中返坏 id）**很可能也解释了它** —— 捕获会实际执行一遍图，
走的就是同一条路径 —— 但那仍是推断，没有真机复现。见 §2.2c 节末定案。

> **2026-09-22 起这里的默认值已翻成 `--async-scheduling` + 71,680**（§2.2c 的崩溃已修）。
> 唯一残留的守卫：显式 `DFLASH2_TOPK_IMPL=flashinfer` 回退旧路径时，上下文自动降回 60,928。

### 6.1 A/B 两侧的原始命令

```bash
# WSL 侧：起容器
wsl -d Ubuntu -u hong bash -lc 'cd ~/testcode/qwen38-27b-rtx3090 && docker compose --profile single up -d'

# 两侧都用同一个客户端（cwd 必须在仓库之外，见项目约定 5）
cd /c/Users/hong
python D:/code/vllm-windows/_dev/bench/_ab_bench.py --port 18020 --label wsl --lens 1024,4096,16384 \
       --outdir D:/code/vllm-windows/_dev/out/_ab_results

# Windows 侧：起服务（纯文本、max-len 自适应）
AB_TEXT_ONLY=1 bash /d/code/vllm-windows/_ab_serve_win.sh > /d/code/vllm-windows/_ab_win_serve.log 2>&1 &
python D:/code/vllm-windows/_dev/bench/_ab_bench.py --port 8000 --label win --lens 1024,4096,16384 \
       --outdir D:/code/vllm-windows/_dev/out/_ab_results

# 出表
python /d/code/vllm-windows/_ab_compare.py wsl win

# 71680 全量复现（2.2b）：先装分桶补丁，再一次性跑服务+基准
python /d/code/vllm-windows/port_wsl_patches.py --batch 3
AB_TEXT_ONLY=1 AB_MAX_LEN=71680 AB_KV_BYTES=6000000000 AB_LENS=1024,16384 \
  bash /d/code/vllm-windows/_ab_kv_probe.sh padded71680
```

`_dev/bench/_ab_serve_win.sh` 的开关：

| 开关 | 作用 |
|---|---|
| `AB_TEXT_ONLY=1` | `--language-model-only`，去掉视觉塔（第 3 节的对照就是这组） |
| `AB_VISION_OFFLOAD=1` | 视觉塔压到 host RAM：加 `--cpu-offload-gb`（默认 1）+ `--cpu-offload-params visual`，并设 `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1`。**与 `AB_TEXT_ONLY` 互斥** |
| `AB_VISION_OFFLOAD_GB=<n>` | 卸载预算，默认 1。塔总共 0.86 GiB，给再多也只卸载这么多 |
| `AB_VISION_OFFLOAD_PARAMS=<段名>` | 默认 `visual`（整塔）。想复刻原补丁的"只卸 merger+blocks"可用 `blocks merger` |
| `AB_MAX_LEN=<n>` / `AB_KV_BYTES=<n>` | 覆盖 `--max-model-len`（默认 60928）/ `--kv-cache-memory-bytes`（默认 6000000000） |
| `AB_DROP_MAMBA=1` | 去掉 `--mamba-ssm-cache-dtype float16`（见第 5 节） |
| `AB_PORT=<n>` | 默认 8000 |
| `AB_DRYRUN=1` | 只打印 argv 不启动（调参时用） |

推荐（= `.env` 的完整等价配置）：装好 `--batch 3` 补丁后
`AB_VISION_OFFLOAD=1 AB_MAX_LEN=71680`，pin 保持默认 6e9 即可（2.2b）。
不装补丁的话退回 `AB_VISION_OFFLOAD=1`（60928/6e9）；**别用
`AB_KV_BYTES=7000000000` 硬顶 71680** —— 读 2.2 / 4.2，余量会被吃光。

视觉自检：`python _dev/test/_vision_smoke.py --port 8000`（生成一张合成图，查背景色/形状/内容，
并打印 `multimodal_tokens` 确认塔真的跑了）。

---

## 7. 没做完的部分

- 第 3 节的性能对照是**纯文本**路径（`--language-model-only`）。多模态已证明可用（2.1），
  但**多模态吞吐没测**——图片编码在带卸载时多一次 0.86 GiB 的 H2D 拷贝，代价多大还没量。
- prefill 只测到 16384（32768 那档受 4.1 的客户端 bug 限制，需要先改 keep-alive 才能测）。
- 并发只到 8，与 `.env` 的 `max_num_seqs=8` 一致；batch 档（64 并发）没测。
- **融合 GDN 内核的收益待重测**（见 5.2）：机制与上下文代价已确认，性能差值没有拿到可信数据。
- WSL 侧的 `Model loading took` 没记（容器日志当时没留存），所以 3.3 节那一格是空的。
- 2.2b 的 71680 数据是**单次测量**（`_dev/out/_ab_results/padded71680.json`），C2/C4 的 +8% 落在
  运行间波动范围内，没有重复验证；"不比 60928 差"这个结论是稳的，"更快"不是。
- 2.2b 的补丁只在 **DFlash2/无投机**路径生效，MTP 配置下是 no-op，未验证 MTP + 71680 的组合。
- 2.2b 的基准是 **纯文本**跑的（`--language-model-only`）。根目录 `start_server.ps1` 的默认档带视觉塔，
  只验证到"能起来、71680 生效"（160 s healthy，见 6.0），**没有跑吞吐**；
  带塔 + 71680 的实际性能仍然是第 7 节第 1 条那个未测项。
- 启动偶发 `device-side assert`（见 6.0）：约 1/3 概率，重试即过。**根因没查**，
  只确认了它与配置无关、不与"补丁是否正确"相关（成功的那次 KV 账本完全正确）。
  ⚠️ 注意这**与 2.2c 的崩溃不是一回事**：2.2c 是"塔在 + 71680 + 并发≥2"必现（2/2），
  启动偶发则与配置无关、重试即过。
- **2.2c 崩溃的具体越界点没定位**：需要 `CUDA_LAUNCH_BLOCKING=1` 重跑（慢很多，没做）。
  目前只知道"塔在 + 71680"是触发条件、不是 OOM。
- 2.2c 的 C8 405.6 tok/s 是单次测量，未复现。

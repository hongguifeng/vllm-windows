# PP 能不能替代 TP（Windows / 双 CMP 170HX / 本 fork）

日期 2026-10-15。代码证据全部来自本树 `vllm/distributed/` 与 `vllm/v1/worker/gpu/`；未跑任何 GPU 实验。
配套事实：`nccl.dll` 已构建并能 `ctypes.CDLL` 加载（NCCL 2.29.7），见 `CMP170HX_LANDING_PLAN_TOP2.md` Tier 0 N1。

---

## 1. 先纠正前提："TP 在 Windows 下不支持"在本树的代码里**不成立**

`vllm/distributed/parallel_state.py:1866-1874`：
```python
if not torch.distributed.is_backend_available(backend):
    logger.warning("Distributed backend %s is not available; falling back to gloo.", backend)
    assert torch.distributed.is_gloo_available(), "Fallback Gloo backend is not available."
    backend = "gloo"
```
即：**torch 没有 NCCL 时，vLLM 会把 device_group 建成 gloo**（CUDA 平台默认 `dist_backend="nccl"`，`vllm/platforms/cuda.py:230`），然后在需要搬 GPU 张量时改走 pynccl：

`parallel_state.py:1199-1206` 的注释与判据是逐字写着 Windows 的：
```python
# Check if we need to use pynccl for CUDA tensors because
# torch.distributed NCCL backend is not available (e.g. on Windows).
_use_pynccl = (device_communicator.pynccl_comm is not None
               and not pynccl_comm.disabled
               and torch.distributed.get_backend(group) != "nccl")
```
`:1218-1219` → `if _use_pynccl and tensor.is_cuda: self.device_communicator.pynccl_comm.send(tensor, dst)`（收端对称逻辑在 `:1355`）。

pynccl 的建连方式也天生跨平台：`pynccl.py:147-162` 用 `ncclGetUniqueId()` + **在 gloo cpu_group 上广播那 128 字节** → 再 `ncclCommInitRank`；库本身走 `ctypes.CDLL(VLLM_NCCL_SO_PATH)`（`pynccl_wrapper.py:381`）——也就是我们刚编出来的那个 dll。

TP 的集合操作同样落在 pynccl：`cuda_communicator.py:422-434`（`all_gather` → pynccl，否则才退回基类/torch）、`pynccl.py` 提供 `all_reduce / all_gather / all_gatherv / reduce_scatter / reduce_scatterv / send / recv / broadcast / group_start / group_end`（`:268-567`），而这七个符号在我们的 dll 里都导出了。

**所以真正的门槛不是"TP 不被支持"，而是两件还没证的事**：① `ncclCommInitRank` 能不能在这对卡上成功；② 这个移植里有哪些 transport（SHM？P2P？只剩 socket？）。TP 和 PP 卡在同一处。

---

## 2. 但 PP 明显比 TP 更适合这台机器 —— 理由是可算的通信量

| | TP=2 | PP=2 |
|---|---|---|
| 通信原语 | 每层每步 all-reduce（+ MoE 的 all-gather/reduce-scatter） | **只有 p2p send/recv**，每个 stage 边界一次 |
| 热路径在哪 | 48 层 × 每步 | 1 个边界 × 每步 |
| 没有 P2P 时（本机正是） | 只能 host-bounce，带宽 = PCIe 有效带宽 | 同样 host-bounce，但**字节数少一到两个数量级** |

按我们的模型算（`config.json`：`hidden_size=2560`、`full_attention_interval=4`；argv：`--max-num-batched-tokens 2048`）：

- **PP=2**：一个 prefill chunk 的 hidden state = 2048 × 2560 × 2 B ≈ **10.0 MiB**，只有 1 个边界 → 每 chunk 一次。decode 阶段每 token 只有 2560 × 2 B ≈ **5 KiB**，4 并发也就 20 KiB/步。按 PCIe Gen2 x4 的有效带宽 ~1.5-2 GB/s 估计（`[推算]`），prefill 多花 **~6 ms/chunk**，decode 基本看不见。
- **TP=2**：同样的 10 MiB 要在**每一层**都过一次线 → 48 层 ≈ **480 MiB / prefill chunk**。按同一带宽估计，一个 chunk 光通信就要 **~300 ms**。我们现在的 4k-token prefill 全程才 ~1.8 s（`[我们复测]`，2,238 tok/s）。**这就是为什么 TP 在这台机器上是错的方向** —— 不是"Windows 不支持"，而是**没有 P2P 的 PCIe 撑不住每层一次的 all-reduce**。

推论：**要跨卡，就选 PP；TP 只在我们搞到真 P2P 之后才值得一试。**

### 2.5 实测（2026-10-15 23:44-23:49，`[我们复测]`）——上面两个数字现在有了实测替身

工具：`_dev/probe/_nccl_comm_smoke.py`；原始日志 `_dev/out/smoke_c_r0.log` / `smoke_c_r1.log`。
拓扑：rank0 = GPU1（整卡空）、rank1 = GPU0（`:9393` 那张，只用呅 1.4 GiB 空闲），rendezvous 走 gloo，通信走 pynccl + 我们自建的 `nccl.dll`。

| 原语 | 尺寸 | min / med / max（n=8） | 对我们意味着什么 |
|---|---|---|---|
| `ncclCommInitRank` | — | **0.40 s 成功** | 本机的最大未证前提被消掉了 |
| `all_reduce` | 16 MiB | **8.53 / 8.69 / 10.31 ms** | TP 每层边界的价；按我们 10 MiB/chunk 插值 ≈ 6.6 ms → 48 层 ≈ **~317 ms / 2048-token prefill chunk**（我们 4k prefill 全程才 ~1.8 s） |
| `send/recv` | 4 KiB | **0.08 / 0.10 / 0.12 ms** | PP 的 decode 边界：**~0.1 ms/步**，可忽略 |
| `send/recv` | 20 MiB | **6.57 / 6.64 / 6.77 ms** | PP 的一个 2048-token prefill chunk 边界：**~6.6 ms**（≈ 3 GB/s 有效） |
| `all_gather` / `reduce_scatter` | 4-8 KiB | 0.09 / 0.10 / 0.14 ms | 小消息往返很便宜 |
| `broadcast` | 4 KiB | 0.05 / 0.07 / 0.10 ms | `PPHandler` 那个广播有可行路 |
| 显存代价 | — | GPU1 rank：188→240 MiB；GPU0 rank：374→478 MiB | 均在 1.4 GiB 空闲内；退出后全部释放 |

**transport 直接写进日志了**：`Channel 00/01 : 0[1] -> 1[0] via SHM/CE/CE`（宿主内存 + copy engine），以及 `P2P is disabled between connected GPUs 0 and 1`。NET/Socket 也在位，但它绑的是 TUN（`Mihomo:198.18.0.1`）—— 万一哪天 SHM 被禁，socket 路径会穿过 TUN，这是个隐患。

**两个修正性的发现**：
1. **A 档（两 rank 同卡）被 NCCL 自己否证**：`NCCL WARN Duplicate GPU detected : rank 0 and rank 1 both on CUDA device c000` → `nccl invalid usage`。同卡两 rank 这条路根本不存在，不是“没测到”。
2. **CUDA 报的空闲显存与 nvidia-smi 严重不一致**：GPU0 被 `:9393` 占了 63.3 GiB（nvidia-smi free 1,431 MiB），而 `torch.cuda.mem_get_info()` 报 **free = 63,701 MiB** → 我的安全 guard（700 MiB）在这张卡上完全失真。这正是 Maya 文档里“free-VRAM 读数偏高导致超填→显存分页到 RAM”的那个坑的放大版，任何靠 free-VRAM 做决策的代码在本机都不可信。

---

### 2.6 引擎级 A/B：PP=2 与 TP=2 在本 fork 上都能跑（2026-10-16 00:06-00:40，`[我们复测]`）

窗口：双卡全空（用户手动停掉 :9393/:9394，我没动过这两个服务）。
模型：`Qwen3.8-27B-W4A16-AutoRound-fast` 的 config + `--load-format dummy` —— 同形状随机权重，**FLOPs 与真权重一致，所以通信/并行开销可比；绝对质量无意义**。eager（无 CUDA graph），`--max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 2048`，文本模式，fork 树（每轮都用日志 banner 核对过 `v0.1.dev21672+g4faed433c.d20261001`）。
工具：`_dev/probe/_pp_tp_smoke.ps1`（启）、`_dev/probe/_bench_port.py`（流式口径：TTFT / 聚合 tok/s / 每请求 ITL）。

| 臂 | prefill 4096 tok | 4 并发 decode | KV 池（8192 ctx） |
|---|---|---|---|
| 单卡 | TTFT 1.040 s / **3,940 tok/s** | **47.9 tok/s** / ITL **73.8 ms** | — |
| **PP=2** | 1.154 s / **3,550 tok/s**（−10%） | 22.1 tok/s / 172.5 ms（**−54%**） | 961,355 tok |
| **TP=2** | 2.149 s / **1,906 tok/s**（**−52%**） | 36.1 tok/s / 93.1 ms（−25%） | 989,786 tok |

1. **两条路都真的能跑**：PP=2 与 TP=2 都 `health=200` 并连续生成，走的正是 §1 里那条 gloo 兜底 + pynccl + 自建 `nccl.dll` 的路。**PP 的 sampled-token 广播（`PPHandler` 的 sibling 组）没有挡住它** —— §4 里我把它标为“真缺口”，在 27B 这个模型上是过了的（它到底走了 pynccl 还是 cpu_group，还需在目标模型上再看）。
2. **代价分布正好相反**：**TP 罚 prefill**（每层一次 all-reduce，−52%），**PP 罚 decode**（阶段串行化，−54%）。这与 §2 的 `[推算]` 方向一致，量级也对得上（PP 的 prefill 只因为一次边界跳 −10%）。
   → 推论：**长上下文 prefill 为主的工作负载选 PP；decode 吞吐为主选 TP**。两者都比单卡慢 —— **跨卡买的是容量（每卡权重减半、KV 池近翻倍），不是速度**。
   → 未测但值一个实验：把并发提到 16/32 后 PP 的 decode 惩罚应随 microbatch 填充而收缩。

#### 目标模型的落地闸门（两条，都在我们自己的树里）

- **PP**：`vllm/model_executor/models/config.py:879-884` → `NotImplementedError: Qwen4Exp N-gram PLE embedding requires pipeline_parallel_size=1 because non-first pipeline ranks do not receive the raw input_ids` （注释还说 `Qwen4ExpModelState` 里会再查一次）。这就是 clee `q38fn/ple-pp3` 要拆的那道闸门。
- **TP**：`vllm/models/qwen4_exp/nvidia/ple_ssd.py:574` → `ValueError: PLE SSD offload currently requires TP=DP=1`。而 PLE 表 **95.37 GiB** 既放不进任何单卡（63.61 GiB）也放不进主机内存（87.92 GiB 总量，还要留给两个引擎），**所以 offload 关不掉** → TP 对目标模型要求把 offload 改成 TP-aware，这不是小改。

#### 这轮顺手踩到的三个 Windows 坑（记下来）

1. **`.venv/Lib/site-packages/vllm/` 里躺着一份旧上游拷贝**（`0.29.1.dev0+g13e844c86`，还有个被中断卸载的 `~llm-...dist-info`）。cwd 不是仓库根时 `import vllm` 会拿到它 —— 第一轮 PP=2 因此跑的是**旧引擎**，靠日志 `[core.py:185]` 对不上 fork 的 `:137` 才露馅。探针现在显式 pin `PYTHONPATH=<repo>`，而且**每轮都要核对日志 banner**。
2. **flashinfer 的 JIT 用 `text=True` 读编译器输出**：中文 locale 的 MSVC 吐 GBK → `UnicodeDecodeError`；`VSLANG=1033` 解决。另外 JIT 需要 `ninja.exe`（在 venv `Scripts` 下）与 `cl.exe`（`_env.ps1`）在 PATH 上，否则 `FileNotFoundError [WinError 2]`。
3. **失败或被杀的实例会留下子进程**，占住 `--data-parallel-rpc-port`（默认 **29550**）与显存 → 下一次启动报 `zmq.error.ZMQError: Address in use (tcp://127.0.0.1:29550)`。每轮前按 `CommandLine like '*vllm.entrypoints*'` 清理。

#### 这轮实验**没能**回答的

- 目标模型（Flash-Next）自己的 PP/TP 行为：被上面两道闸门挡住，**不改 `vllm/` 就没法测**（本轮约束禁止改）。dummy 臂只能验并行机制，不能验 INC/Humming/Marlin/PLE 那些 kernel 在多卡下的正确性。
- 生产口径（CUDA graph + MTP + 262144 ctx）下的数字：本轮全是 eager + 8192 ctx。

---

## 3. PP 在我们模型上会带来什么（`[推算]`，需要实测才能定）

- 每卡只装自己那 ~24 层的权重 → 我们 48.28 GiB 的权重变成每卡 ~24 GiB 级，**腾出的显存可以给 KV**；我们的 KV 现在是 12.13 GiB / 419,430 tokens / 1.60× 并发@262K，PP=2 理论上能把 KV 池显著抬高（每卡只为自己的 full-attention 层分配，`full_attention_interval=4` → 每卡 12 个注意力层）。
- **速度不会变快**：一个 token 仍要过完 48 层，只是分两段顺序做；step time 与单卡同量级。PP 买的是**容量**（能装下 > 63.61 GiB 的模型）和 **KV 空间**，不是吞吐。
- 需要把 batch 切成 `pp_size` 个 microbatch：我们 `max_num_seqs=4` → 2+2，流水利用率尚可。
- PLE（95.37 GiB SSD 流式）与 Humming MoE 会落到"拥有该层的 stage"上 → 我们现有的 SSD 流式路径要按 stage 重新核算。

---

## 4. 要把 PP 落到 Windows，真正要动的地方（按优先级）

| # | 位置 | 为什么 | 性质 |
|---|---|---|---|
| 1 | `vllm/v1/worker/gpu/pp_utils.py:54, 84` —— `PPHandler` 用 `make_sibling_device_group()` 建"专用 **NCCL** communicator"做 sampled-token 广播；而 `parallel_state.py:598-603` 那个方法是 `torch.distributed.new_group(backend=self.torch_distributed_backend)` | Windows 上 backend 已是 gloo → 用 gloo 广播 **CUDA** 张量会失败或退化成 CPU 往返。这条是**真正的 Windows 缺口**，不是 NCCL 本身 | 需改成走 `pynccl_comm.broadcast`（或先 `.cpu()` 走 cpu_group） |
| 2 | `--disable-custom-all-reduce` + pynccl 侧 `NCCL_P2P_DISABLE=1` | 我们没有 P2P；custom all-reduce 依赖 CUDA IPC/P2P。clee 在 PP=3 上也是这么配的（`NCCL_P2P_DISABLE=1` + `--disable-custom-all-reduce`，原文可查） | 启动参数，零代码 |
| 3 | clee 的 `q38fn/ple-pp3`：**10 文件 +215/−49** —— PP>1 下的 PLE/checkpoint 加载（gate removal、per-worker `set_current_vllm_config` at load、rank-unowned tensor skip） | 我们也有 PLE 与外部 SSD 库，PP 下加载路径同样会破 | 需照抄思路，不是照抄 patch（他们的模块布局与我们不同） |
| 4 | `VLLM_DISTRIBUTED_USE_SPLIT_GROUP` **保持不设** | `_env.ps1` 的注释说得对：那条路径硬编码 `cpu:gloo,cuda:nccl`，会**要 torch 自己的 NCCL** → 一开就把 Windows 这条路打死 | 配置纪律，零代码 |
| 5 | MTP ×3 + adaptive k、CUDA graph `FULL_AND_PIECEWISE`、async scheduling 在 PP 下的正确性 | 我们的基线三件套都假设单 stage；clee 在 PP=3 上**明确没启用 MTP** | 待判（可能要先关 MTP 跑通，再逐步开） |

---

## 5. 有一条**完全绕开 NCCL** 的变体，而且本树已经支持它

`parallel_state.py:1222`（收端对称在 `:1355`）：`comm_group = metadata_group if tensor.is_cpu else group` —— **CPU 张量走 cpu_group（gloo）**，元数据也走 gloo（`isend_object`，`:943-963`）。
所以只要在 stage 边界把 `IntermediateTensors` 先 `.cpu()`、收到后再 `.cuda()`，PP 就**一个 NCCL 字节都不需要**：gloo 在 Windows 上是有的，我们实测 `is_nccl_available()==False` 时 gloo 可用。

代价（`[推算]`）：每个边界一次 D2H + 一次 H2D，10 MiB/chunk → 仍是几 ms 级，跟走 pynccl-over-PCIe 差不多，但**没有 NCCL 建连的不确定性**，也没有 `ncclCommInitRank` 这个未证前提。改动量约十几行，且只在 PP 路径上。

**这条值得优先做**：它能让我们在被"NCCL transport 到底存不存在"卡住之前，先把 PP=2 的功能面（PLE/MoE/MTP/graph 在 PP 下对不对）全部验一遍。

---

## 6. 验证顺序（第 2 步已于 23:44 实测完成，不需要停机窗口）

1. **CPU-staging PP=2**（不需要 NCCL）→ 判功能：能不能起、KV 多大、MTP/graph 是否要关。
2. ~~`ncclCommInitRank` 冒烟~~ **已做，已过**（comm init 0.40 s；all_reduce/all_gather/reduce_scatter/send/recv/broadcast 全部可用；transport = `SHM/CE`，`P2P is disabled`）。
3. 若 2 成立：关掉 CPU-staging，让 PP 走 pynccl p2p → 量 prefill chunk 的边界开销。
4. 若 2 不成立：PP 只能用 CPU-staging 版长期跑，并接受每边界几 ms。
5. TP=2 只在拿到真 P2P 之后才重新考虑（本机的 P2P 需要 Linux 驱动级解锁，见评估报告 W8）。

## 7. 一句话结论

**能用 PP 替代 TP，而且比 TP 好得多 —— 现在有实测支撑**：跨卡 `ncclCommInitRank` 在本机 Windows 上 **0.40 s 就起来了**，transport 是宿主内存 + copy engine（`SHM/CE`），无 P2P。实测价：PP 的一个 prefill chunk 边界 **~6.6 ms**、decode 边界 **~0.1 ms**；而 TP 的同一尺寸每层都要过一遍，48 层≈ **~317 ms / chunk**。PP 只需要 p2p、不需要 all-reduce，还能把模型装进“两卡合计 127.22 GiB”；本树已经为 Windows 写好了 gloo 兜底 + pynccl-over-standalone-dll 这条路。剩下的真缺口是 `PPHandler` 那个 sibling 广播组，以及我们自己的 PLE/MTP/graph 在 PP 下的正确性。**CPU-staging PP 仍然是最稳的起手式** —— 它把 NCCL 完全绕开，而且我们刚测出的那个 free-VRAM 失真使“靠空闲显存做决策”的所有路径都变得可疑。

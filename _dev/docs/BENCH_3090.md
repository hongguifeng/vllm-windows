# RTX 3090 回归测试报告（2026-09-20）

显卡从 CMP 170HX（GA100 / sm_80 / 64 GB）换回 **RTX 3090（GA102 / sm_86 / 24 GB）** 后的
全量复测。所有对比对象都是**同一份代码树**（WSL 补丁已应用）在 170HX 上的历史日志，
不是原厂基线，故差异只反映硬件。

- 设备：`NVIDIA GeForce RTX 3090`，cc 8.6，82 SM，24576 MiB（测试时整卡独占，0 MiB 占用）
- 驱动 591.86 / torch 2.11.0+cu130 / vLLM `0.29.1.dev0+g13e844c86`
- 构建产物仍是 **sm_80 cubin**（`TORCH_CUDA_ARCH_LIST=8.0`）：sm_80 二进制在同一大版本内
  向下兼容 sm_86，实测全部扩展模块正常加载，**无需重编译**
- 数值正确性探针通过，Marlin int8 负 scale 补丁生效（见 §6）

---

## 1. 引擎级基准（`_dev/bench/_bench_27b.py`）

`max_model_len=16384`，`max_num_seqs=8`，`max_num_batched_tokens=4096`，
`language_model_only=True`，CUDA graph 开启。

| 指标 | 170HX (patched r2) | 3090 r1 | 3090 r2 | 3090/170HX |
|---|---|---|---|---|
| 权重加载 | 23.9 s | 18.3 s | 17.9 s | 0.75x（3090 更快） |
| torch.compile | 22.0 s | 35.6 s | 31.6 s | 1.5x |
| profiling/warmup | 6.08 s | **85.96 s** | **3.89 s** | 一次性，见 §4 |
| init engine | 37.5 s | 148.1 s | 49.1 s | |
| `engine_init_s`（脚本口径） | 69.9 s | 176.3 s | 77.1 s | 1.10x |
| **prefill**（1×8 881 tok） | 2 183.5 t/s | 1 253.2 t/s | 1 247.6 t/s | **0.57x** |
| **decode bs=1** | 66.3 t/s | 50.8 t/s | 50.9 t/s | **0.77x** |
| **decode bs=8** | 461.8 t/s | 346.3 t/s | 347.9 t/s | **0.75x** |
| batch prefill（8×4 465 tok） | 2 262.8 t/s | 1 292.8 t/s | 1 290.3 t/s | 0.57x |
| KV cache 容量 | 530 962 tok / 42.5 GiB | 84 347 tok / 6.69 GiB | 92 235 tok / 7.31 GiB | 0.17x |
| 16k 上下文并发路数 | 32.4x | 5.15x | 5.63x | 0.17x |
| peak reserved | 57.55 GiB | 21.77 GiB | 22.39 GiB | — |

**复现性**：3090 两轮全部指标相差 ≤0.5%，与 170HX 的 r1/r2 离散度（≤0.6%）同级。

两个口径上的说明：

- `batch prefill` 这一项在两张卡上**都是假的批处理**：8 条 4 465 token 的 prompt 大于
  `max_num_batched_tokens=4096`，调度器只能一条一条 prefill（日志里每条间隔 ~5.8 s），
  所以它约等于单流 prefill。要真正测批 prefill 需要 `VBATCHED=8192`，
  但那会连带改变 compile range 与 KV 划分，跨卡对比时要一起改。
- KV 容量差异不是配置造成的，是 24 GB 减去 14 GB 权重之后的必然结果。

## 1b. 为什么 prefill 也是 170HX 快 —— 先修正一个常见误判

直觉上"3090 是新一代消费卡，算力应该更强，只是显存带宽吃亏"，所以 prefill 该更快、
decode 该更慢。**这个前提不成立：170HX 在算力和带宽两个轴上都强于 3090。**

| | CMP 170HX | RTX 3090 |
|---|---|---|
| 芯片 | **GA100**（A100 同族） | GA102 |
| SM / CUDA core / Tensor core | 70 / 4 480 / 280 | 82 / 10 496 / 328 |
| Boost 频率 | 1 410 MHz | 1 695 MHz |
| **FP16 张量（dense）** | **~101 TFLOPS** | **~71 TFLOPS** |
| INT8 张量（dense） | ~202 TOPS | ~142 TOPS |
| 显存 | 64 GB HBM2e | 24 GB GDDR6X |
| **带宽** | **1.49–1.56 TB/s** | **936 GB/s** |
| L2 | 10 MB | 6 MB |
| TDP | 250 W | 350 W |

关键在**每 SM 每时钟的 FP16 张量吞吐：GA100 是 GA102 的 2 倍**（A100 级 SM 的
张量核心规格高于消费级 GA102）。所以 170HX 用更少的 SM（70 vs 82）和更低的频率
（1.41 vs 1.70 GHz），总张量算力仍然高出约 **42%**；带宽则高出约 **60%**。

把纸面比例和实测比例摆在一起，整体是自洽的：

| 轴 | 纸面比（3090/170HX） | 实测比 |
|---|---|---|
| 张量算力 | 0.70 | prefill **0.57**（比纸面还低 19%） |
| 显存带宽 | 0.63 | decode **0.77**（反而比纸面高 22%） |

- **prefill 比算力比例还差**：3090 在持续张量负载下受 350 W 功耗墙限制，实际能维持的
  频率低于 1 695 MHz 标称值；加上 L2 只有 6 MB（170HX 10 MB），权重复用更差。
- **decode 比带宽比例还好**：等效带宽 760 GB/s 已达 3090 峰值的 81%（接近实用上限），
  而 170HX 只跑出 990 GB/s、约为峰值的 66% —— HBM2e 的峰值在 250 W 功耗墙下吃不满。

**结论：换回 3090 之后，prefill 和 decode 双双变慢是预期内的，不是配置或移植出错。**

### W4A8 的直接证据：3090 的 prefill 确实被张量核卡住

| 项目 | W4A16（默认） | W4A8（int8 激活） | 变化 |
|---|---|---|---|
| prefill 8.88k | 1 247.6 t/s | **2 434.2 t/s** | **+95%** |
| batch prefill | 1 290.3 t/s | **2 633.5 t/s** | **+104%** |
| decode bs=1 | 50.9 t/s | 48.8 t/s | −4.1% |
| decode bs=8 | 347.9 t/s | 334.3 t/s | −3.9% |
| engine init | 77.1 s | 220.7 s | +143 s（一次性） |
| KV cache | 92 235 tok | 83 740 tok | −9% |

W4A16 走的是 FP16 张量核，**int8 张量核全程闲着**；打开 int8 激活后 prefill 直接翻倍，
说明 3090 的 prefill 是**张量核吞吐瓶颈**，不是带宽瓶颈（带宽已由 §5 排除）。

对照 170HX：它的 W4A8 只拿到 **+31%**（2 183.5 → 2 856.9）。差异的原因还是
GA100 的 FP16 路径本身够快，而 GA102 的 FP16 路径相对更吃亏。

> **所以 3090 上的配置建议和 170HX 不一样：`-W4A8` 从"取舍"变成了近乎必开**
> —— 用 4% 的 decode 换 95% 的 prefill。

## 2. 服务级基准（`vllm bench serve`，`_dev/bench/_bench_suite.py --only warmup,cohort`）

真实 prompt 数据集 `_dev/bench/_prompts_real.jsonl`，`out=1024`，`T=0`，`--max-num-seqs 4`，
**MTP 投机解码开启**（与 170HX 历史基线一致，两侧 `tok/step` 都是 ~1.80）。

| C | e2e tok/s 170HX | e2e tok/s 3090 | 比 | medTPOT ms 170HX | medTPOT ms 3090 | tok/step |
|---|---|---|---|---|---|---|
| 1 | 108.4 | 73.1 | 0.67 | 9.05 | 13.33 | 1.780 / 1.802 |
| 2 | 194.3 | 132.4 | 0.68 | 9.42 | 13.88 | 1.790 / 1.803 |
| 4 | 352.6 | 253.3 | 0.72 | 10.18 | 14.39 | 1.810 / 1.802 |
| 8 | 353.3 | 253.4 | 0.72 | 10.27 | 14.38 | 1.800 / 1.808 |

TTFT（ms）：170HX 126.5 / 170.1 / 219.6 vs 3090 301.4 / 270.3 / 454.9 ——
3090 差在 **prefill**，和 §1 的 0.57x 完全对得上。
C=8 两卡都是 4+4 两批（`--max-num-seqs 4`），总时长与 C=4 相同可佐证。

> 服务端比 `tok/step` 是识别"有没有开 MTP"的关键：关掉 MTP 时这个值会掉到 n/a。
> 第一次跑 3090 服务端时没开 MTP，e2e 只有 170HX 的 0.47x，完全是配置不对齐导致的误判。

原始日志：`_dev/out/_bench_base_3090_mtp/`（3090 主结果）、`_dev/out/_bench_base_3090/`（3090 无 MTP 对照）、
`_dev/out/_bench_base_170hx/`（170HX 基线备份）。

## 3. MTP 在 3090 上的收益（同配置，唯一变量）

| C | 无 MTP | 有 MTP | 提升 |
|---|---|---|---|
| 1 | 50.5 t/s | 73.1 t/s | **+44.8%** |
| 2 | 87.2 t/s | 132.4 t/s | **+51.8%** |
| 4 | 164.9 t/s | 253.3 t/s | **+53.6%** |

单步延迟从 19.56 ms 涨到 24.05 ms（+23%），但每步吐 1.80 个 token，净赚 ~47%。
**MTP 在这张卡上值得常开**，`mtp` 头随 checkpoint 自带，不需要额外 drafter。
KV 代价：`block_size` 784→800，`num_gpu_blocks` 140→120，32k 上下文下 KV 78 643 token。

## 4. 启动成本（换卡后最容易被误读的一项）

| 阶段 | 时延 | 说明 |
|---|---|---|
| 权重加载 | ~18 s | 比 170HX 的 24 s 快，纯磁盘 |
| torch.compile | 31–37 s | 170HX 是 20–22 s |
| profiling/warmup | **同配置连跑：3.89 s；换配置后首跑：73–86 s** | 一次性 |
| 服务起到 `/health` 返回 200 | 130 s（无 MTP）/ 145 s（含 MTP） | 含上面各项 |

**那个 86 秒不是 3090 的问题**，是同一配置在新设备上的首次 kernel 编译成本，证据：

| 配置 | profiling/warmup |
|---|---|
| 170HX stock | 6.01 s |
| 170HX + WSL 补丁 r1 | 91.06 s |
| 170HX + WSL 补丁 r2 | **6.08 s** |
| 3090 + WSL 补丁 r1 | 85.96 s |
| 3090 + WSL 补丁 r2 | **3.89 s** |

同一份补丁树在两张卡上都是"首跑 ~86–91 s、二跑 ~4–6 s"，曲线形状一致。
170HX 的服务日志里只有 8.88 s，是因为它当年反复用同一套 `200000/8192` 配置，
这笔成本早被摊掉了。

**实用结论：这台机器上别频繁改启动参数**，改一次要付一次 ~70–90 s 的编译成本；
参数固定后重启约 77 s（引擎）/ 130–145 s（服务）。

## 5. 带宽核算

| | 3090 | 170HX |
|---|---|---|
| decode bs=1 | 50.9 t/s | 66.3 t/s |
| 每 token 需读权重 | 13.91 GiB = 14.94 GB | 同 |
| 实测等效带宽 | **760 GB/s** | 990 GB/s |
| 显存峰值 | 936 GB/s（GDDR6X） | HBM2e，更高 |
| 占峰值比 | **81%** | — |

**3090 的 decode 已经贴着显存带宽天花板（81% 峰值）**，这是 27B W4A16 在这张卡上的
硬上限：任何不减少"每 token 权重字节"的改动都拿不到 decode 收益。
要更快的 decode 只能上更小的量化（W3/W2）或更激进的投机解码。

prefill 是唯一还有余量的一侧（0.57x，且 3090 的算力比不该这么低），
推测与 6 MB L2（vs 170HX 的 40 MB）导致的权重复用变差有关 —— 但 §1 显示
prefill 远未到带宽瓶颈（14.9 GB/次 ÷ 936 GB/s = 16 ms，而实测单次 7.1 s），
所以更像是算力/occupancy 层面的问题，值得后续单独查。

## 6. 移植成果在 sm_86 上的验证清单

| 项 | 状态 |
|---|---|
| sm_80 cubin 在 sm_86 加载 | ✅ 6 个扩展模块全部正常 |
| Marlin（`MarlinLinearKernel for CompressedTensorsWNA16`） | ✅ |
| Marlin int8 负 scale 补丁（本 checkpoint 50.5% group scale 为负） | ✅ 探针输出正常，非胡话 |
| GDN 解码内核 | ✅ `GDN decode kernel: cuda`（未退化到 triton） |
| FlashAttention 2 | ✅ `Using FlashAttention version 2` |
| CUDA graph（FULL + PIECEWISE） | ✅ 捕获成功，0.06–0.11 GiB |
| torch.compile | ✅ |
| MTP 投机解码 | ✅ `tok/step ≈ 1.80`，与 170HX 一致 |
| 数值正确性探针 | ✅ 巴黎/4/100°C 三项全对 |

## 7. 口径说明：别和 HyperQwen 的 3090 数字混比

本仓库早前记录过一组 3090 数字（batch prefill 1 810 / single 1 440 / int8 激活 1 850–1 940、
64 并发），那是 WSL 里 **HyperQwen（`qwen38-27b-rtx3090`）** 自己跑的，跑的是
**DFlash2 投机解码（7 drafts/pass，`tok/step` 4.51）+ int8 激活**的完整栈，
和本项目的 MTP（1 draft，`tok/step` 1.80）不是同一套软件。

本报告的 3090 数字全部是**本项目 Windows 原生栈**的结果，用来跟同一份代码树在 170HX 上的
日志对比，反映的是硬件差异。两者不可交叉引用。

顺带一个真实差距：那张表里 3090 的 `tok/step` 是 4.51，本项目是 1.80 —— DFlash2 的
多 draft 收益远大于 MTP，这是后续值得单独评估的方向（sm_80/86 有开放 fault 风险，
且需要额外 drafter）。

## 8. 推荐的 3090 启动配置

```powershell
# 长上下文 / 生成优先（对话、RAG 深度思考）
& D:\code\vllm-windows\_dev\bin\_run.ps1 -Serve `
    -Model D:\models\Qwen3.8-27B-W4A16-AutoRound-fast `
    -TextOnly -MaxLen 32768 -MaxSeqs 4 -BatchedTokens 4096 -MemUtil 0.92 -MTP

# prefill 优先（长文档喂入、批量摘要）—— 在上一行基础上加 -W4A8
& D:\code\vllm-windows\_dev\bin\_run.ps1 -Serve `
    -Model D:\models\Qwen3.8-27B-W4A16-AutoRound-fast `
    -TextOnly -MaxLen 32768 -MaxSeqs 4 -BatchedTokens 4096 -MemUtil 0.92 -MTP -W4A8
```

默认给：KV 78 643 token，单请求最长 32k，并发 2.4 路（降到 16k 可到 5.6 路）。

- **`-TextOnly` 必须开**：24 GB 下省掉 0.92 GB 视觉塔，直接变成 KV
- **`-MTP` 建议常开**：+45~54% 服务端吞吐，代价是每步 +23% 延迟
- **`-W4A8` 看负载**：prefill +95% / decode −4%。提示词长就开；纯长生成可以不开。
  代价是一次性 ~143 s 额外编译，所以**别来回切**
- **别加 `--mamba-ssm-cache-dtype float16`**：会关掉融合 GDN 内核，退回 Triton
- 还想再多要 KV：`-MemUtil 0.95`。vLLM 提示的 0.9253 只是补偿 CUDA graph 估算，
  不是上限
- 显存预算参考：`util 0.92` → 权重 13.91 + 激活峰值 0.74–1.36 + graph 0.06
  + KV 6.69–7.31 GiB
- **不要照抄 170HX 的 `-MaxLen 200000`**：24 GB 放不下，能开的上限约 84k


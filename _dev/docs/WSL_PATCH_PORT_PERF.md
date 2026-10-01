# 从 HyperQwen 移植优化到本项目 —— 实测结果

来源：WSL `~/testcode/qwen38-27b-rtx3090`（上游 `syv-ai/HyperQwen`，41 条补丁，基于 v0.28.0）
目标：本项目 `D:\code\vllm-windows`（vLLM 0.29.0 源码树 + Windows/sm_80 构建，CMP 170HX 64 GB）
测试模型：`D:\models\Qwen3.8-27B-W4A16-AutoRound-fast`

分诊依据见 `WSL_3090_PATCH_PORT_REPORT.md`（40 条补丁逐条评估）。本文只讲**实际移植并测出结果**的部分。

---

## 一、结论先行

| 项目 | 结果 |
|---|---|
| **唯一实际有性能收益的** | **W4A8（int8 激活）** —— prefill **+29%**，代价是 decode **−6%** |
| 只开 MLP 层的折中档 | prefill **+18%**，decode **−3%** |
| `marlin-repack-staged-sm80` | **零收益**（这条是为 CPU fallback 路径准备的，我们的 stock 路径本来就快） |
| 服务端三条补丁 | 无性能影响，但**已验证工作**（每请求收尾日志、长 prompt 不再被 400 拒） |
| 重新编译 | **不需要** —— 移植的全部是 `.py`，零 C++/CUDA 改动 |

---

## 二、移植的 6 条

用 `_dev/patches/port_wsl_patches.py` 幂等应用（改源码树 → 同步 site-packages 两份，重装重编不丢）：

| 补丁 | 作用 | 实测价值 |
|---|---|---|
| `marlin-int8-layer-select` | 按层名正则选哪些层走 W4A8 | 前置条件，本身无性能影响 |
| `marlin-int8-negative-scales` | AutoRound 导出的负 group scale 被 Marlin 当无符号读 | **必需**，见下 |
| `marlin-repack-staged-sm80` | sm80 repack 走单个可增长暂存缓冲 | **零收益**，+0.58 GiB 常驻 |
| `engine-completion-log` | 每请求一行无门控收尾日志 | 已验证生效 |
| `engine-stall-sentinel` | 引擎停止推进时告警一次 | 默认关（`VLLM_ENGINE_STALL_SENTINEL_S=0`） |
| `no-output-token-reservation` | 长 prompt + 大 `max_tokens` 不再被 400 拒 | 已验证生效 |

**`marlin-int8-negative-scales` 对本 checkpoint 是必需的，不是可选。** 我实际读了权重验证：

```
model.language_model.layers.0.mlp.gate_proj.weight_scale   neg=352953/696320
model.language_model.layers.0.mlp.down_proj.weight_scale   neg=351300/696320
negative group scales: 1511115/2990080 = 50.5%
```

50.5% 的 group scale 是负值（AutoRound 对称导出的典型特征）。没有这条补丁，int8 Marlin 内核会把一半的 scale 读成垃圾 —— 补丁头部原话是"模型输出胡话，但基准测试看起来一切正常"。

---

## 三、明确不移植的，以及判定依据

| 类别 | 补丁 | 为什么不 |
|---|---|---|
| 投机解码 | `dflash2-*`、`spec-decode-*`、`vllm-pr50021`、`dspark-*` | HyperQwen README 明确记载 **sm80（含 CMP 170HX）在任意 k 上都有未修复的投机解码 fault**（issue #98/#72），且 drafter 是按模型单独准备的 checkpoint |
| 省显存 | `int4-kv-per-token-head`、KVarN、`offload-*`、`vision-tower-cpu-offload` | 全是"24 GB 塞不下"逼出来的手段；我们 64 GB，收益接近零。vision 那条我们跑 `language_model_only` |
| FP8 | `triton-spec-attn-fp8-kv` | **sm89+ 专属，GA100 没有 FP8**，Triton 会在 SM89 以下直接拒绝 |
| WSL2 专属 | `offload-wsl2-devptr` | UVA/device pointer 行为，原生 Windows 不是同一回事 |
| **已不适用** | `hybrid-sw-block-promote` | 目标场景是滑动窗口层。**实测本模型没有**：`layer_types` = 48 × `linear_attention` + 16 × `full_attention`，`sliding_window=None` |
| **0.29 已有** | `mamba-align-checkpoint-order`、`mamba-align-retire-null-gaps`、`xgrammar-spec-terminated`、`vllm-pr54282-draft-gumbel-salt`、`sse-keep-alive`、`hybrid-kv-groups-v2-cudagraph` | 日志已证实 `Mamba cache mode is set to 'align'`（align 模式本身就是上游机制，那两条补丁早于它）；其余按 `PATCHES.md` 自己写的 retire 条件已满足 |
| 未测 | `sampler-small-topk-fast-softmax` | 头部说明收益来自**投机解码**单步（多 verify 行）；我们没有投机解码，且基准用贪心。真要评估得先有用 top_k/top_p 的采样负载 |
| 未测 | `marlin-tune-table` | 需要本地编译一个可调 Marlin 扩展，默认关 |

---

## 四、基准方法（`_dev/bench/_bench_27b.py` + `_dev/bench/_bench.sh`）

每次配置跑两遍，取第二遍（torch.compile 缓存已热）。分离三个量：

- **engine init** —— 包住 `LLM(...)`，Marlin repack 在这里
- **prefill** —— 8881 token 的单流 prompt + 1 个输出 token；另有一组 8 条**互不相同**的 ~4450 token prompt 测批量 prefill
- **decode** —— 256 token 贪心生成，bs=1 与 bs=8
- **正确性探针** —— 3 个短 prompt 的贪心输出，用于抓"能跑但输出乱码"这类问题

### 我踩并修掉的两个测量陷阱

1. **前缀缓存把批量 prefill 变成了查表。** 第一版用 8 条相同 prompt，聚合出 9680 tok/s（单流只有 2194），明显不合理 —— 7 条命中了缓存。改成互不相同的 prompt 后，批量 prefill 变成 2266 tok/s，与单流 2197 一致，合理了。
2. **预热自己把缓存填了。** 第二版加了"用同一批 prompt 预热再计时"，结果计时那次**整批**命中缓存，反而更假（17784 tok/s）。最终改为每次调用都带 `time.time_ns()` 盐值，保证计时那批必定冷启动。

顺带：**没有**关掉前缀缓存来回避这个坑 —— 因为本模型的 Mamba cache mode 是**由前缀缓存推导出来的**（`align` when on），关掉就在测另一个引擎了。

### 一个必要的对照

四组配置是按顺序跑的（baseline → patched → w4a8 → w4a8mlp），decode 的下降有可能只是机器随时间的漂移。所以在最后又重跑了一次"无 int8"作为对照：

| | prefill | decode bs=1 | decode bs=8 |
|---|---|---|---|
| baseline（最早） | 2197 | 66.9 | 462.3 |
| patched（第二） | 2184 | 66.3 | 461.8 |
| **对照（最后）** | **2187** | **66.5** | **462.6** |

对照回到原值 → **没有漂移，W4A8 的 decode 回归是真实的**。

---

## 五、结果

全部为第二遍（缓存已热）的数字。`max_model_len=16384`，`gpu_memory_utilization=0.90`，GPU 独占。

| 配置 | engine init | prefill 8.8k | 批量 prefill | decode bs=1 | decode bs=8 |
|---|---|---|---|---|---|
| W4A16（默认） | 69.3 s | 2 197 t/s | 2 266 t/s | 66.9 t/s | 462.3 t/s |
| + repack-staged 补丁 | 69.9 s | 2 184 t/s | 2 263 t/s | 66.3 t/s | 461.8 t/s |
| 对照（同上，最后跑） | 66.4 s | 2 187 t/s | 2 268 t/s | 66.5 t/s | 462.6 t/s |
| **W4A8 全部层** | 105.9 s | **2 833 t/s (+28.9%)** | **2 999 t/s (+32.4%)** | 63.1 t/s (−5.7%) | 431.6 t/s (−6.6%) |
| **W4A8 仅 MLP** | 85.2 s | **2 602 t/s (+18.4%)** | **2 735 t/s (+20.7%)** | 64.8 t/s (−3.1%) | 445.4 t/s (−3.7%) |

### 为什么是"prefill 涨、decode 跌"

不是猜的，是算出来的：

- **decode 是显存带宽瓶颈。** bs=1 时 66.9 tok/s × 13.91 GiB 权重 ≈ **930 GB/s** 的权重读取速率，已经贴近这块卡的带宽上限。int8 激活**不减少权重字节**，所以帮不上；反而每层多一次 per-token 激活量化，且 int8 GEMM 在 M=1 没有优势 → 净跌 6%。
- **prefill 是算力瓶颈。** 2197 tok/s × 2 × 27e9 ≈ **119 TFLOPS**；GA100 70 SM 的 fp16 峰值约 200 TFLOPS（A100 108 SM 是 312），即约 59% MFU。int8 tensor core 在这块卡上峰值翻倍 → 有空间，实测吃到 +29%。

### 为什么 repack-staged 完全没有收益

补丁头部给的 A/B 是 **44.9 s（staged）对 403.7 s（CPU fallback）**。我们的 stock 路径**不是**那个 CPU fallback，本来就只要 23.9 s：

| | load weights | model loading | 常驻 |
|---|---|---|---|
| baseline | 21.88 s | 23.95 s / 13.91 GiB | — |
| patched | 21.85 s | 23.92 s / **14.49 GiB** | **+0.58 GiB** |

加载耗时一模一样，只有那 0.58 GiB 的暂存缓冲被分配了 —— 这反而**证明补丁确实在生效**，只是没有可优化的空间。

### W4A8 额外多出的 36 s 初始化

| 阶段 | W4A16 | W4A8 | 差 |
|---|---|---|---|
| loading weights | 21.85 s | 22.39 s | +0.5 |
| model loading 合计 | 23.92 s | 27.04 s | +3.1 |
| torch.compile | 21.99 s | 39.08 s | **+17.1** |
| 初始 profiling/warmup | 6.08 s | 15.96 s | **+9.9** |
| graph capture | 2 s | 2 s | 0 |

都是一次性（每进程）成本，不是每请求。

---

## 六、正确性

### W4A8 的数值

**W4A8、W4A16 两个配置的贪心输出逐字符一致：**

```
'The capital of France is' -> 'Paris.\nThe capital of Germany is Berlin.\nThe capital of Italy is Rome...'
'2 + 2 ='                  -> '4, 4 + 4 = 8, 8 + 8 = 16, 16 + 16 = 32, 32 + 32 = 64, 6...'
'Water boils at'           -> '100°C at sea level, but at higher altitudes, the boiling point decreases...'
```

这正是 `marlin-int8-negative-scales` 的预期效果（头部声明"数值等价，除了极罕见的 code −8 变 −7"）。**如果这条补丁没打，这里就会是乱码。**

### `no-output-token-reservation` 实测（`_dev/test/_test_token_reserve.py`）

```
max_model_len    = 16384
max_tokens asked = 8000        ->  stock 只允许 16384 - 8000 = 8384 prompt tokens
actual prompt    = 14441 tokens   (超出 stock 限制 6057 tokens)
RESULT: ACCEPTED
  generated 1943 tokens, finish_reason='length'
```

14441 + 1943 = 16384，正好卡在上下文边界并返回 `length` —— 与补丁声明的行为完全一致：输入检查只看 prompt 自身长度，生成由下游 clamp 到边界。

### `engine-completion-log` 已生效

同一份日志里每请求一行（这正是它加的东西）：

```
INFO [scheduler.py:2067] Request finished: req 0-a2e3ee19 finish_reason=length
     prompt_tokens=14441 generated_tokens=1943 elapsed_s=162.15
```

---

## 七、怎么用

```powershell
# 默认：W4A16，decode 最快
& D:\code\vllm-windows\_dev\bin\_run.ps1 -Model D:\models\Qwen3.8-27B-W4A16-AutoRound-fast `
                               -TextOnly -MaxLen 16384 -MaxSeqs 8 -BatchedTokens 4096

# prompt 为主（RAG / 文档问答 / 长上下文）：开 W4A8
& D:\code\vllm-windows\_dev\bin\_run.ps1 ... -W4A8

# 折中：只对 MLP 开（保住 ~2/3 收益，decode 只跌 3%）
& D:\code\vllm-windows\_dev\bin\_run.ps1 ... -W4A8 -W4A8Layers mlp
```

选型建议：**prompt 占比高就开 W4A8，生成占比高就保持默认。** 服务端 `-Serve` 同样吃这两个开关（走的是环境变量，不是 CLI 参数）。

重装 / 重编后重新应用补丁：

```powershell
python D:\code\vllm-windows\_dev\patches\port_wsl_patches.py          # 幂等
python D:\code\vllm-windows\_dev\patches\port_wsl_patches.py --check  # 只看不动
```

回退：改动前的工作副本在 `_dev/patches/wsl/_backup_pre/`（9 个文件）。

---

## 八、尚未验证

- **视觉（多模态）路径** —— 全程 `language_model_only`
- **`sampler-small-topk-fast-softmax` 的收益** —— 已移植（第二批），但需要带 top_k/top_p
  的采样负载才测得出来；本次基准用贪心
- **`spec-decode-attn` 的收益** —— 已移植（第二批），但需要 MTP + 长上下文的服务端场景
- **长上下文（200k）下 W4A8 的表现** —— 本次基准固定在 16384

---

## 九、第二批：为 RTX 3090 重新分诊（2026-09-20）

### 为什么会有第二批

第一轮分诊是在 **CMP 170HX（64 GB）** 视角下做的，所以有一整类补丁被以"你有 64 GB，
这些省显存的手段收益接近零"为由拒掉，还有一类被归入"投机解码"整体否决。

换回 **RTX 3090（24 GB，sm_86）** 后，这两条理由**都失效了**：

- 24 GB 重新成为约束，省显存的手段重新有意义；
- 被否决的投机解码理由写的是"sm80 有未修复的 fault"，而 3090 是 **sm_86**，
  且本项目已在 3090 上实测跑通 MTP（见 `BENCH_3090.md`），前提不成立。

更关键的是：**这批补丁有几条明确就是为 3090 写的**，其头部直接写明 3090 的 SM 数、
功耗和几何参数。当时只是因为在 170HX 上评估而没被认真对待。

### 本批移植的 5 条

| 补丁 | 当时的否决理由 | 现在 |
|---|---|---|
| `prefill-attn-int8` | "几何门控写死 24/4/256，非该几何回退；依赖 Windows Triton JIT" → 搁置 | 几何**完全命中**（24 heads / 4 kv / head_dim 256），Triton JIT 在 3090 上可用 |
| `spec-decode-attn` | 归入"投机解码"整体否决 | 否决理由是 sm80 fault，**不适用 sm_86**；我们已在跑 MTP（k+1=5 queries） |
| `sampler-small-topk-fast-softmax` | "收益来自投机解码单步，我们没有投机解码" | **已经有 MTP**；词表 248 320 与其头部写的一致 |
| `mamba-chunked-prefill-align` | 未测 | 正确性修复（chunked prefill 跨 Mamba 块边界丢状态 / NaN），我们用 chunked prefill + GDN |
| `vllm-pr50021-gdn-spec-bounds` | 未测 | 安全修复（GDN/KDA 投机解码越界，上游 PR 未合并），我们用 MTP + GDN |

仍然**不移植**，且理由与显卡无关：`triton-spec-attn-fp8-kv`（sm89+ 专属，sm_86 同样没有 FP8）、
`offload-wsl2-devptr`（WSL2 专属）、`hybrid-sw-block-promote`（本模型无 sliding window 层）、
DFlash2 全家族（需要按模型单独准备的 drafter）、int4-KV 链（依赖 `hybrid-sw-block-promote`）。

### 移植过程中发现的两件事

**1. 补丁之间有真实依赖顺序。** `prefill-attn-int8` 的 hunk 上下文里带着
`spec-decode-attn` 插入的三行注释，`spec-decode-attn` 的又带着 `sampler` 的。
按 `sampler → spec-decode-attn → prefill-attn-int8` 顺序应用全干净，乱序则全部冲突。
这也解释了为什么单独 `git apply --check` 一条会失败。

**2. 0.29 在 `flash_attn.py` 里长出了 10 行。** 两条注意力补丁的 hunk 上下文都是 0.28 的代码
（`causal = not has_window` 直接跟 `flash_attn_varlen_func(`），而 0.29 在中间插入了
`num_splits = attn_metadata.max_num_splits` 和 FA4 hd256 的页对齐逻辑 —— 所以冲突不是行号漂移，
是上下文真的变了，`-C1` 也救不了。两条在该处都是**纯新增**，因此 `_dev/patches/port_wsl_patches.py`
用 `added_block()` 把新增行从补丁里原样取出、按锚点重新落位，其余 hunk 照常 `git apply`。

### 实测：`prefill-attn-int8`

同参数（`VMEM=0.92` / `VMAXSEQS=8` / `VBATCHED=4096`），单流 prompt，取第二次运行：

| prompt 长度 | FA2（`VLLM_PREFILL_ATTN=`） | int8（`=int8`） | 变化 |
|---|---|---|---|
| 8.8k | **1 275.8 t/s** | 1 229.8 t/s | **−3.6%** |
| 30.0k | **1 178.9 t/s** | 1 192.4 t/s | **+1.1%** |

内核确实生效 —— 日志里有 `Triton kernel JIT compilation during inference: _prefill_attn_kernel`
（连同 `_k_stats_kernel`、`_k_quant_kernel`，首次约 4 s，r1 因此被低估到 1 037 t/s）。
数值探针在两种配置下逐字符一致。

**结论：没有实质收益。** 补丁头声称 "~1.3-1.35x FA2 at 16-51k context"，但那是
**250 W 的 3090**；本机功耗上限 **370 W**。功耗墙越宽松，FA2 越不受限，
int8 tensor core 的理论优势就越难兑现 —— 与第一批里 W4A8 在 170HX 上只有 +29%、
在 3090 上却有 +95% 是同一个机制的反面。

**处理方式：留着，但默认关闭**（`VLLM_PREFILL_ATTN=""`），与 `marlin-repack-staged-sm80`
一致 —— 默认关，无副作用，需要时可用环境变量打开。

### 本批的安全性验证

对照组（5 条补丁在场、开关全关）与第一批基线一致：

| | 第一批基线 | 本批对照 |
|---|---|---|
| prefill 8.8k | 1 247.6 t/s | 1 275.8 t/s |
| decode bs=1 | 50.9 t/s | 50.7 t/s |
| batch prefill | 1 290.3 t/s | 1 294.0 t/s |

差异 ≤2%（噪声内），三个数值探针逐字符一致 → **5 条补丁不破坏 baseline**。

### 怎么用

```powershell
# 第二批随第一批一起应用（幂等）
python D:\code\vllm-windows\_dev\patches\port_wsl_patches.py              # 两批
python D:\code\vllm-windows\_dev\patches\port_wsl_patches.py --batch 2    # 只第二批
python D:\code\vllm-windows\_dev\patches\port_wsl_patches.py --check      # 只看不动
```

新增开关（都默认关，除 sampler 的 draft 截断默认开）：

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `VLLM_PREFILL_ATTN` | `""`（关） | `int8` / `fp16` 启用 hd256 Triton prefill 内核 |
| `VLLM_SPEC_DECODE_ATTN` | `0`（关） | 1 启用投机解码 split-KV verify 内核 |
| `VLLM_SPEC_DECODE_ATTN_QMAX` | `0` | 该内核的 query-token 上限（0 = 1 + num_speculative_tokens） |
| `VLLM_SPEC_ATTN_BLOCK_M` | `0` | 强制 query-row tile（0 = 按行数自选） |
| `VLLM_DRAFT_TOPK_TOPP` | `1`（开） | 草稿侧按 target 的 top-k/top-p 截断 |
| `VLLM_DRAFT_TEMP_SCALE` | `1.0` | 草稿分布温度缩放 |

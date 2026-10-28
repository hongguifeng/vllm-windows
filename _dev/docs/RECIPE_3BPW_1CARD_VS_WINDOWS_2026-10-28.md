# 3bpw 单卡配方（`ElectroHobo/qwen38-flash-next-3bpw-1x-cmp170hx-recipe`）对照与可移植性

日期 2026-10-28。对象：`https://github.com/ElectroHobo/qwen38-flash-next-3bpw-1x-cmp170hx-recipe`
（release 0.1.0，2026-10-23 发布），与本仓库 `flash-next-win` 分支的生产单卡服务（`:9393`）。

## 0. 结论（先看这五条）

1. **配方不是"另一种更快的内核"，它是同一个模型、同一份 3bpw AutoRound 权重，换掉了三样东西**：
   FP8 n-gram 表（48 GiB 而不是 95 GiB BF16）放进**主机 pinned 内存**、DFlash 2 草稿 7/5/3 按并发调度、
   GDN 层用 RecoverSSM 只留一份状态。引擎是 Morrowmake 的 `vllm-cmp170hx`（pin 在一个 commit 上）跑在容器里。
2. **它的头条数字被"口径"放大了**。用配方自己的 harness（`bench/cmp_recipe_bench.py`，同样的三条 prompt、
   400 token、temperature 0、thinking off、min_tokens+ignore_eos、5 次取中位）打我们的 `:9393`，实测：

   | 1 用户 | 本项目 `:9393`（DFlash-7） | 配方（DFlash 7/5/3） | 差 |
   |---|---:|---:|---:|
   | structured | 167.7 | 197.8 | −15 % |
   | code | 229.1 | 296.4 | −23 % |
   | prose | 113.6 | 135.5 | −16 % |

   | 4 用户（聚合） | 本项目 `:9393` | 配方 | 差 |
   |---|---:|---:|---:|
   | structured | 444.4 | 515.6 | −14 % |
   | code | **588.5** | **587.1** | **±0 %** |
   | prose | 286.0 | 370.9 | −23 % |

   即：**同协议下是 1.0–1.3×，不是 2–3×**。"我们 113 tok/s、它 296"那个印象来自另一套口径
   （MTP-2、200/128 token、中文 prompt、非流式中位数）；同协议下它的 296 对应我们 229。
3. **剩下的 15–25 % 缺口有一个具体的、可验证的机制**：我们的 PLE 表在 SSD 上，解码步被切成
   `BreakableCUDAGraphCapture` 的多段图（每段之间要跑主机侧取行），配方把表放主机内存后是
   **整步一张 `torch.cuda.CUDAGraph()`**（`vllm/v1/worker/gpu/cudagraph_utils.py:541-551`）。
   本仓库自己的账：PLE 净代价 ~2.5 ms/步、约 17 %（`FLASH_NEXT_ROUTE_A_BUILD.md` §14.4、§15.1）。
   这与实测的 15–23 % 缺口对得上。
4. **可移植性：核心那条几乎不用改引擎代码。** 本树（上游 0.29 + Windows + 已移植的 Morrow 特性）**已经带**
   pinned-host PLE（`Qwen4ExpPLEPinnedHostEmbedding`）、UVA 取行 kernel、FP8 表 + 单全局 scale
   （`ple_embedding_dtype: float8_e4m3fn`）、`--engram-config cpu_offload`（默认开）、
   `num_speculative_tokens_per_batch_size`、`--mamba-ssm-cache-dtype`。
   要做的只是：**造一份 FP8 表的 checkpoint + 把 `ple_ssd_offload` 关掉**。
   首次在 Windows 上验证了这条路径可用（§6.1 探针：取行结果逐字节一致）。
5. **真正的门槛是 RAM，不是代码。** 48 GiB pinned 表 + 一个引擎 ≈ 74 GiB 常驻（含 Windows）；本机 87.9 GiB，
   但当前**两个**引擎各 26 GiB 常驻、只剩 9.5 GiB 空闲。也就是说：**这份配方在本机是"单服务"配置**，
   想让 `:9393` 和 `:9394` 并存就拿不出 48 GiB。要么接受单服务，要么加内存/做 FP4 表，
   要么只拿"免费档"（§6.2：DFlash 草稿数按并发调度，零内存代价）。`n   **2026-10-28 实测更新：本机 pin 上限 41 GiB，48 GiB 表根本放不下（§10）——这条路在本机已否决。**

---

## 1. 配方是什么

| 项 | 值 |
|---|---|
| 引擎 | `Morrowmake/vllm-cmp170hx` @ `18305948ac`（容器镜像按 digest pin，只读挂 8 个 patch 文件） |
| 权重 | `klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`（与本项目**同一份**，专家 3bit g128 / 2bit g64，2.998 bpw，卡上 47 GiB） |
| n-gram 表 | `halt95/...-W4A16-Merlin` 的 **FP8** 表（一个全局 scale），**48 GiB 主机 pinned 内存**，替换掉 klee100 的 95.37 GiB BF16 表 |
| 草稿 | inco.ai DFlash 2（与本项目同一份，revision `a22a1744`），7/5/3 按并发 |
| 状态 | **RecoverSSM**：GDN 每请求一份状态（而不是每草稿位一份），状态 bf16 |
| 图 | `FULL_AND_PIECEWISE` + `VLLM_USE_BREAKABLE_CUDAGRAPH=1`，**整步一张图** |
| 调度 | `--max-num-seqs 48 --max-num-batched-tokens 3460 --long-prefill-token-threshold 0 --block-size 912 --prefix-match-unit 64` |
| 其它 | `--kv-cache-dtype bfloat16`、`--attention-config {"indexer_kv_dtype":"bf16"}`、`--mamba-ssm-cache-dtype bfloat16` |

它的 8 个 patch：DFlash 捕获（2 个文件）、RecoverSSM（1 个 CUDA kernel + 4 处挂钩）、
n-gram 表「INC 也能加载 + 多服务共享一份」（`PLE_SHARED_DIR`）、INC 逐层 bits/MTP 层解析、COW 分块拷贝。
**这些 patch 是相对 Morrow 那棵树写的**，不是相对上游，不能直接 `git apply` 到本树。

## 2. 本项目当前在跑什么

`:9393`（`:9394` 同构），`start_qwen38_flash_next.ps1` 默认档：

```text
--model D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP      # 同一份 3bpw 权重，143 GiB（含 95.37 GiB BF16 表）
--max-model-len 262144 --max-num-seqs 4 --max-num-batched-tokens 2048
--gpu-memory-utilization 0.96 --mamba-cache-mode align
--block-size 1680 --prefix-match-unit 16
--kv-transfer-config {... cpu_bytes_to_use 16 GiB ...}        # 主机/磁盘 KV tier
--additional-config {ple_ssd_offload: true, ple_ssd_cache_mb 8192, ple_ssd_io_depth 512,
                     ple_ssd_workers 16, ple_ssd_prefetch_tokens 16384}
--speculative-config {method dflash, num_speculative_tokens 7}  # 固定 7，无调度
+ vision、thin-GEMM、fla skip-spilling、alloc healer
```

与配方逐项对照：

| 维度 | 本项目 | 配方 | 差异性质 |
|---|---|---|---|
| PLE 表 | **SSD 流式** + 8 GiB 行缓存（`ple_ssd_io_win.dll`），BF16 95.37 GiB | **主机 pinned**，FP8 47.68 GiB | 结构性（本节的主角） |
| 解码图 | SSD 路径强制 `BreakableCUDAGraphCapture`（多段 + 主机取行） | 整步单图 | 结构性 |
| 草稿数 | 固定 7 | 7 / 5 / 3 按并发 | 配置，`num_speculative_tokens_per_batch_size` 本树已支持 |
| GDN 状态 | 每草稿位一份，`mamba_ssm_cache_dtype` 未设（auto） | RecoverSSM + bf16 | 引擎缺失（RecoverSSM） |
| 并发上限 | 4 | 48 | 配置 + RecoverSSM |
| block / prefix unit | 1680 / 16（本机实测 c4 +10.6 %） | 912 / 64 | 配置，方向相反，不能盲抄 |
| KV 池 | ~42 万 token（262k ctx） | 292,784 token | 配方把余量让给了 48 并发 |
| 主机 KV tier | 16 GiB（磁盘文件） | 默认关 | 配置 |
| 硬件 | 74 SM CMP ×2 + 3090，5700X，87.9 GiB RAM，x8 | 1 卡，2016 Broadwell，204 GiB，x16 | 它的 CPU 更差、内存更宽 |

## 3. 同协议实测（本节是校准，不是调优）

工具直接用配方的 `bench/cmp_recipe_bench.py`，只把输出目录改到本仓库
（`_dev/probe/_cmp_recipe_bench.py`，原始 JSON 在 `_dev/out/cmp_recipe_bench/`）：

```powershell
cd D:\code\vllm-windows
& .\.venv\Scripts\python.exe _dev\probe\_cmp_recipe_bench.py --port 9393 `
    --model Qwen3.8-Flash-Next --tag win9393-dec --decode-only --concurrency 1,4 --tokens 400
```

结果见 §0 两张表。**读这张表要知道的三件事**：

1. 我们的数字是在**非独占**机器上取的：两个引擎都活着（各 26 GiB 常驻、私有 75.8 GiB），
   物理内存只剩 9.5 GiB；配方的数字来自 fresh clone + 独占机器。这一项对我们不利，估计 5–10 %。
2. 两类数字的 **prompt 完全相同**（配方的三条：40 行 markdown 表 / LRU cache 模块 / 技术散文），
   所以差距里不含 prompt 成分。之前仓库里 DFlash-7 code=157.6 那一档是另一套 prompt + 200/128 token，
   **不能**与 296.4 直接比。
3. 我们的 `max-num-seqs` 是 4，所以 4 用户那两行正好压在并发上限上；配方是 48。
   4 用户 code 打平，说明"多并发"不是它这 20 % 的来源。

## 4. 差异归因（按证据强度排序）

### 4.1 PLE 表的位置 —— 唯一有机制级证据的一项

- 本树在 `ple_ssd_offload` 打开时，解码图走 `BreakableCUDAGraphCapture`（该架构默认开 breakable），每段之间要做主机侧取行
  （`vllm/v1/worker/gpu/cudagraph_utils.py:541-551`；`vllm/models/qwen4_exp/nvidia/ple_ssd.py`）；
  不开 SSD 时是普通 `torch.cuda.CUDAGraph()`。
- 本仓库实测的每步账（`FLASH_NEXT_ROUTE_A_BUILD.md` §14.3、§15.1）：`Future.result()` 等 **9.8–10.4 ms**，
  其中 `ngram ids` 的 D2H **8.9–9.5 ms**，真正的取行只有 0.38–0.95 ms（缓存命中 77–80 %）；
  与 PLE-free 结构视图对比，**净代价 ~2.5 ms/步 ≈ 17 %**。
- 配方把这一步变成图内的 UVA gather（几十万字节/步，PCIe 上可忽略），因此**没有任何图外等待**。
- 这正好解释 c1 三行的 15–23 % 缺口；也解释 4 用户 code 为何打平（4 用户时 step 被 GPU 主导，
  图外等待被摊薄）。

### 4.2 草稿数按并发调度 —— 可能是 4 用户 prose −23 % 的主因（待 A/B）

配方在 3–4 用户时只发 5 个草稿，我们从 1 到 48 都是 7。验证代价随宽度线性涨，prose 接受率低时
7 草稿更亏。本树**已支持**该字段（`num_speculative_tokens_per_batch_size`，P3 实测生效），
属于**零内存代价**的 A/B。

### 4.3 硬件/口径

- 配方是一张卡独占 + 204 GiB RAM + x16；我们是两卡两引擎 + 9.5 GiB 空闲 + x8。
- 配方 `MAX_SEQS=48`，我们是 4；262k ctx 下我们的池只够 ~1.6× 并发，这是两种取舍，不是提速。

### 4.4 不是差距来源（别被名字骗了）

| 项 | 为什么不是 |
|---|---|
| `--block-size 912` / `--prefix-match-unit 64` | 配方为 48 并发和 DFlash 环选的值；我们 1680/16 在本机 c1/c4 实测更优（+5.7 %/+10.6 %） |
| `--max-num-batched-tokens 3460` | 只影响 prefill 切片（配方自己说 8192 更快但池更小）；对 decode 步时无效（我们已 A/B 过 8192） |
| `VLLM_QWEN4_QSA_TOPK_CANONICAL` / `VLLM_CUSTOM_ALLREDUCE_*` / `VLLM_EXL3_*` | Morrow 引擎的开关，单卡 AutoRound 路径用不到 |
| thin-GEMM / FLA skip-spilling | 我们已移植且默认开 |
| `VLLM_USE_BREAKABLE_CUDAGRAPH=1` | 本树对 Qwen4Exp **默认就置 1**（`vllm/config/vllm.py:100,775-784`）；但 FULL 模式是否切成多段，由图外的 SSD 取行决定（`cudagraph_utils.py:541-551`），不是这个 env |
| INC 逐层 bits / MTP 层解析 | 我们这份 checkpoint 已经能加载（走 `VLLM_PROBE_INC_HUMBING_LOWBIT` + humming），只是实现路径不同 |
| COW 分块拷贝 | 只有 32 并发共享 prompt 时才会 OOM；我们 `max-num-seqs 4` 用不到 |

## 5. 本树已经有什么、缺什么

| 能力 | 本树状态 | 证据 |
|---|---|---|
| pinned-host PLE 表 | **有**（`Qwen4ExpPLEPinnedHostEmbedding`，`torch.empty(pin_memory=True)`） | `vllm/models/qwen4_exp/nvidia/ngram_embedding.py:405,448` |
| UVA 取行 kernel（图内） | **有**（Triton，读 pinned 表的 device view） | 同文件 `:375-402,463-500` |
| FP8 表 + 单全局 scale | **有**（`ple_embedding_dtype: float8_e4m3fn` → `Qwen4ExpPLEFp8EmbeddingMethod`） | 同文件 `:191-192,286-340,705` |
| `--engram-config cpu_offload` | **有，且默认开** | `vllm/config/engram.py:44`、`vllm/envs.py:165` |
| 草稿数按并发调度 | **有** | `num_speculative_tokens_per_batch_size`（P3 已实测） |
| `--mamba-ssm-cache-dtype bfloat16` | **有**（默认 auto，未设） | `vllm/config/cache.py:174` |
| RecoverSSM | **缺**（无 `vllm/sm80/kernels/gdn/`，`qwen_gdn_linear_attn.py` 无 recover 路径） | `git grep recover` 为空 |
| 多服务共享一张表（`PLE_SHARED_DIR`） | **缺**，且配方实现是 Linux `fcntl`+`mmap` | `patches/ple-table.diff` |
| COW 分块（256 MiB） | **缺**（`copy_kv_cache_blocks_inplace` 一次性 gather） | `vllm/v1/worker/utils.py:685` |

**关键判断：配方里和我们"速度"有关的那条主路径（pinned host + FP8 + UVA），本树是上游自带、我们从未启用的代码**；
配方之所以要打 patch，是因为 Morrow 那棵树的 `from_quant_config` 里 INC 分支挡住了它
（`patches/ple-table.diff` 的第一段）。我们这棵树先判 `ple_embedding_dtype`，不会被 INC 挡住，**不需要那个 patch**。

## 6. 可移植性

### 6.1 主线：FP8 表进主机内存（**§10 已实测否决：本机 pin 上限 41 GiB**）

**需要的动作**

1. 拿 FP8 表，二选一：
   - **下载** `halt95` 的 FP8 表分片（~48 GiB，配方验证过的那一份）；
   - **本地转换**：把 `model-00001-of-00011.safetensors`（95.37 GiB，128 个 `shard_i.weight`，每个 `[2500012, 160]` BF16，
     合计 320,001,536 行 × 160 维）
     按一个全局 absmax 量化成 `float8_e4m3fn`（47.68 GiB）并写 `weight_scale`（fp32 标量）。
     需要两遍读（求 absmax、再量化），NVMe 上约 30–60 min，D: 余量 291 GiB 够。
2. 造"服务用"目录：hard link 我们的全部分片**除** `model-00001`，加入新表分片、合并 `model.safetensors.index.json`、
   在 `config.json` 的 `text_config` 写 `ple_embedding_dtype: float8_e4m3fn`。
   做法与配方的 `docker/build_checkpoint.py`、仓库里 `_dev/probe/_flashnext_trim.py` 同构，不改权重。
3. 启动时**去掉** `--additional-config {ple_ssd_offload:...}`，显式 `--engram-config {"cpu_offload":true}`，
   其余保持；`ple_ssd_cache_mb` 那 8 GiB 和 16 GiB KV tier 可以同时收回。
4. 校验：`HumanEval`/`GSM8K` 各 200（配方 163/164、197/200 与 MTP 同级）、`agent_bench` 子集、
   以及 §3 的 harness 复测。

**RAM 账（本机实测）**

| 项 | 数值 |
|---|---|
| 物理内存 | 87.9 GiB（提交上限 315.9 GiB，pagefile C: + F: 共 226 GiB） |
| 当前空闲 | 9.5 GiB；两个引擎各 **26 GiB 常驻 / 75.8 GiB 私有** |
| FP8 表 | 47.68 GiB **pinned**（= 物理内存锁死，不可换出） |
| 单服务 + 表 | 47.68 + ~18（引擎非 PLE 部分）+ ~8（Windows）≈ **74 GiB**，可行，余量 ~14 GiB |
| 双服务 + 表 | 47.68 + 2×18 + 8 ≈ **92 GiB**，**不可行**（除非表只一份且两个引擎再瘦身） |

结论：**这是"单服务"形态的配方**。本机要跑，要么停掉 `:9394`，要么等内存扩容，
要么只把表压到 24 GiB 级（FP4，质量未验证，不建议上生产）。

**已做的可行性探针（`_dev/probe/_ple_host_uva_probe.py`，RTX 3090，x4 槽）**

```text
torch 2.11.0+cu130, device NVIDIA GeForce RTX 3090
is_uva_available: True
table 200000x2560 fp8 = 488.3 MiB pinned; device view ok
lookup of 4096 rows: MATCH
UVA gather 4096 rows x 2560 dim: 6.646 ms (10.5 MB ≈ 1.6 GB/s，x4 槽)
UVA gather 4096 rows x 160 dim（真实行宽）: 0.485 ms (8.4 M rows/s)
pinned 4.00 GiB: allocate 0.9 s, device view 0.0 s     # 48 GiB 外推 ~11 s
```

即：**Windows + WDDM 下，pinned 主机内存的 UVA 取行可用且逐字节正确**。真实行宽 160 B、
约 500 行/步时取行 ~0.06 ms/步（对比 SSD 路径的 ~2.5 ms/步净代价），CMP 的 x8 链只会更快。
这还是首次在这台机器上验证这条路径。

**风险与未知（按严重度）**

1. **47.68 GiB 能不能一次性 pin 住**：4 GiB 可以，48 GiB 必须在"只剩一个引擎"的窗口里实测；
   配方说 torch 的 pinned allocator 会把它凑到 64 GiB（未解释），若真如此，本机直接不可行 ——
   这是第一个要打的靶子（探针 `--gib 48`，停 `:9394` 后跑）。
2. **长上下文 prefill 的取行量**：表在主机内存后，prefill 每 chunk 要读 `tokens × row` 字节；
   按行 160 B、约 50 行/token 估，4k token ≈ 32 MB/chunk，x8 链上 ~10 ms 级，可被 prefill 计算重叠。
   配方报 cold prefill 4,243 tok/s，我们当前 ~3,250（`_dev/docs` 旧口径），预期同档或略好。
3. **质量**：FP8 单全局 scale 是配方的既有做法并通过了 HumanEval/GSM8K；本地自转的表要自己测。
4. **退化路径**：`engram cpu_offload=False` + `--additional-config ple_ssd_offload` 原样可用，
   启动脚本加一个 `-PleHostRam` / `-PleSsd` 互斥开关即可回滚。

### 6.2 免费档：先把不需要内存的那一半拿下（建议先做）

| 动作 | 依据 | 代价 | 预期 |
|---|---|---|---|
| DFlash 草稿按并发调度 `[[1,2,7],[3,4,5],[5,48,3]]` | 本树已支持；配方 4 用户用 5 个草稿 | 改一行 spec JSON + 重启 | 4 用户 prose 的 −23 % 里可能拿回一半 |
| `--mamba-ssm-cache-dtype bfloat16` | 配方默认；状态页减半、步时 −1~3 % | 一个参数 | KV/并发小幅改善 |
| `--max-num-seqs` 4 → 8（仅 DFlash-7 不行，7 草稿只够 ~6 槽；要 MTP-3 才到 12+） | 配方 48 靠 RecoverSSM | 配置 | 只在我们需要 >4 并发时才有意义 |
| block 1680 / prefix 16 的 A/B 复核 | 我们已测；配方 912/64 是为 48 并发 | 0 | 不要盲抄 |

这几项**不碰 RAM、不碰引擎代码**，可以先用 §3 的 harness 做 A/B，再把结论写回启动脚本。

### 6.3 RecoverSSM：只在要 >6 并发时才值得

- 它解决的是"每草稿位一份 GDN 状态"：7 草稿时每请求每层 8 页，单卡只装得下 ~6 请求；
  配方用 verify+commit 把状态压回每请求一份，才有 48 并发。
- 移植量：一个 CUDA kernel（`gdn_recover_verify.cu`，首次编译 ~95 s）+ 一个 Triton commit +
  GDN/PLE 层/model_state 三处挂钩，且要**逐位对齐**验证（配方列了随机输入、1–7 草稿、
  denormal 边界、prefix 边界四类用例）。
- 在 Windows 上还要过 `torch.utils.cpp_extension` + MSVC/nvcc 的构建（本仓库已经干过 `ple_ssd_io_win.dll`
  和 thin-GEMM，路是通的），估计 **3–5 天**含验证。
- 我们的用量是 4 并发：**不建议现在做**。若将来要 16–48 并发，它才是前置条件。`n- 注意：RecoverSSM 与 §10 的 pin 上限无关，它省的是**显存里的状态页**，不受 WDDM 限制。

### 6.4 多服务共享一张表（`PLE_SHARED_DIR`）：Windows 上不建议做

配方的实现是 tmpfs 文件 + `fcntl.flock` + `mmap` + `cudaHostRegister`，三个原语在 Windows 上都要换
（`CreateFileMapping`/命名互斥/`MapViewOfFile`），而且"把一个文件映射注册成 CUDA host 内存"
在 WDDM 下是否稳定**没有证据**。加上 §6.1 的账（48 + 2×18 + 8 > 88 GiB），
共享也只能省一份表、省不出第二份引擎。**结论：不做；宁可单服务。**

### 6.5 明确不要照搬

- 配方把 `MAX_LEN` 锁 262144、`GPU_UTIL` 锁 0.95、`EXTRA_ARGS` 白名单——那是它的发布契约，不是性能项。
- `--long-prefill-token-threshold 0` 与本树默认一致。
- `--load-format auto`、`--mm-encoder-tp-mode data`、`--kv-cache-dtype bfloat16`、`--attention-config indexer bf16`：
  我们已是等价配置（vision 默认开、bf16 KV）。
- `block 912`、`batched 3460`、`QSA_TOPK_CANONICAL`、`EXL3_*`、`CUSTOM_ALLREDUCE_*`：见 §4.4。

## 7. 建议顺序

**P0（半天，不停生产）**
1. 用 `:9394` 做 §6.2 的 A/B：DFlash 调度 7/5/3 + `--mamba-ssm-cache-dtype bfloat16`，用 §3 harness 复测 c1/c4。
   通过就写回 `start_qwen38_flash_next.ps1` 的默认档（新增 `-SpecSchedule` 传参，默认 `[[1,2,7],[3,4,5],[5,48,3]]`）。
2. 把 §3 的 harness 纳入 `_dev/bench/`（它已经是配方发布的同款口径，正好补上"可对外比较"的空缺）。

**P1（1–2 天，需要停机窗口，建议先只停 `:9394`）**
3. 探针 `_ple_host_uva_probe.py --gib 48`：量 47.68 GiB pinned 的真实可行性、耗时、`torch` 是否凑整到 64 GiB。
   **这是 go/no-go 开关**；失败则 P1 终止，只保留 P0。
4. 造 FP8 表（先试下载 halt95；不行再本地转），按 §6.1 造 checkpoint。
5. 单服务起 `:9394`（不动 `:9393`），`--engram-config cpu_offload` 且无 `ple_ssd_offload`，
   用 §3 harness 对 `:9393` A/B；同时看 prefill TTFT、KV 池、GPU util。
6. 质量：HumanEval/GSM8K 各 200 + `agent_bench` 子集。

**P2（仅当需要 >8 并发）**

7. 移植 RecoverSSM（§6.3），再谈 `MAX_SEQS=48`。

**验收口径**（避免重复本轮的口径错）：同一 harness、同一 prompt、同一 token 数、同一并发、
独占机器（至少停掉另一个引擎）、记 `completion_tokens` 而不是 chunk 数。

## 8. 复现与文件

| 文件 | 内容 |
|---|---|
| `_dev/probe/_cmp_recipe_bench.py` | 配方 harness 的本地副本（只改输出目录） |
| `_dev/out/cmp_recipe_bench/cmp-recipe-bench-win9393-dec.json` | 本次 c1/c4 原始请求记录 |
| `_dev/probe/_ple_host_uva_probe.py` | pinned/UVA 取行 + 大页 pin 探针（`--gib 48` 为 go/no-go） |
| `_dev/docs/FLASH_NEXT_ROUTE_A_BUILD.md` §14–15 | 每步账（PLE 等待 10 ms、净代价 2.5 ms、GPU 6 ms） |
| `_dev/docs/FLASHNEXT_SINGLE_CARD_ACCEL_2026-10-21.md` | 上一轮（W4A16 两卡配方）移植结论 |
| `vllm/v1/worker/gpu/cudagraph_utils.py:541` | SSD 路径 → breakable 多段图的分叉点 |
| `vllm/models/qwen4_exp/nvidia/ngram_embedding.py:191,375,405` | FP8 选择 / UVA kernel / pinned 分配 |
| 配方 `patches/ple-table.diff`、`docker/build_checkpoint.py`、`serve.sh` | 表更换、patch 清单、全部启动参数 |

## 9. 一句话

配方值得抄的是**"把 48 GiB FP8 表放进主机内存，让整步回到一张 CUDA graph"**，而且本树早就带这条路；
它的 296 tok/s 不是"新内核"，同协议下我们是它的 77–100 %。移植的收益是单用户 +15~25 %、prefill/TTFT 改善，
代价是**本机只能跑一个服务**；而 4 用户 code 我们已经打平。所以先做零代价的草稿调度 A/B，
再用 `--gib 48` 探针决定要不要为这 15–25 % 付掉第二个服务。







---

## 10. 追加实测（2026-10-28，停掉两个服务后）：主机内存**放不下**这张表

用户授权停掉 `:9393`/`:9394` 后，按 §7 P1 的 go/no-go 直接量了本机的 pin 上限。
结论：**配方的"48 GiB pinned 表"在本机不成立**，而且不是"再挤一挤"的问题。

### 10.1 三个上限（`_dev/probe/_pin_memory_cap_probe.py`、`_cudart_hostalloc_probe.py`）

| 测量 | 结果 |
|---|---|
| 分块 `VirtualAlloc` + `cudaHostRegister`（1 GiB/次） | 到 **41 GiB** 成功，第 42 GiB 返回 `cudaErrorMemoryAllocation`；当时空闲物理内存约 62 GiB |
| 单次 `cudaHostAlloc`（3090 与 CMP 都测） | **14 GiB 成功，16 GiB 失败** |
| `torch.empty(..., pin_memory=True)`（引擎实际走的路） | **8 GiB 成功，12 GiB 失败** |
| 配方需要 | **47.68 GiB，单块连续** |

- 41 GiB ≈ 87.9 GiB 的 47 %，与 WDDM「共享显存 = 一半物理内存」的口径吻合；它随内存总量走，
  所以加内存是唯一能抬高它的办法（128 GiB → 约 64 GiB）。
- 16 GiB 的单次分配上限**不随内存总量走**（3090 与 CMP、x4 与 x8 都一样），
  这是平台级限制；即使内存加到 128 GiB，引擎现在"一个 `[320M,160]` 张量"的分配方式仍然过不去。
- torch 的 pinned 分配器还比裸 `cudaHostAlloc` 低一截（8 vs 14 GiB），这一层要单独绕。

### 10.2 这意味着什么

1. **本机（87.9 GiB）无论怎么改代码都放不下**：总预算 41 GiB < 47.68 GiB。
2. 加内存到 128 GiB 也不够"直接跑"：还要把表的分配**切成 ≤14 GiB 的若干块**，
   并让 UVA 取行 kernel 知道每块的行偏移——这已经不是"换表 + 关开关"，而是
   `Qwen4ExpPLEPinnedHostEmbedding` 的定制改造（分块权重、分块 weight_loader、多基址 gather）。
3. 因此 §6.1 从"1–2 天、单服务可行"降级为：**不加内存不做；加了内存还要再加分块改造**。
   本机保持 SSD 路径是正确选择，之前的 15–25 % 缺口在当前硬件上无法用这条路补。
4. 剩下的可行动空间就是 §6.2 的免费档（草稿调度、SSM 精度、block/prefix），已在下面 10.3 实测。

### 10.3 免费档 A/B（独占机器，:9393，同 harness）

停掉两个服务后，空机上依次起两个臂，各用 §3 的 harness 打一遍（每臂 c1 五次取中位、c4 三次取中位）：

| 臂 | c1 structured / code / prose | c4 structured / code / prose |
|---|---|---|
| A：生产档（DFlash 固定 7） | 181.0 / 213.3 / 125.8 | 447.2 / 546.3 / 290.9 |
| B：+ 草稿调度 7/5/3 | 176.3 / 242.3 / 120.9 | 462.5 / 533.6 / 317.1 |
| B′：B 再跑一遍（同一实例） | 203.7 / 245.0 / 120.6 | 452.7 / 513.3 / 321.1 |

**噪声比效应大**：B 与 B′ 是同一份配置，c1 structured 差 **15.5 %**、c4 code 差 3.8 %；
而 c1 的草稿调度按定义就是 7 个（调度表 `[1,2,7]`），A 与 B 的 c1 本应逐位相同，实测差 13.6 %。
可稳定复现的只有 c1 code ≈ 242–245、c1 prose ≈ 120–126、c4 prose ≈ 317–321。

因此：

- **草稿调度没有可测出的收益**（c4 prose 一次 +9 %、一次 +1 %；c4 code 反而 −3.8 %）。
  在本机噪声下（±15 %），配方 4 用户 prose 的 370.9 不能归因于调度。
- 第一臂有系统性偏慢的嫌疑：A 是长时间空闲后的第一次起服，PLE 走的 SSD 文件缓存是冷的，
  B/B′ 是热的。以后同机 A/B 必须先跑一遍丢弃（或两臂都先热身）。
- 空机（B/B′）与配方相比：c1 code 242–245 vs 296.4（−18 %）、c4 code 513–533 vs 587.1（−9~13 %）、
  c4 prose 317–321 vs 370.9（−13~14 %）。比第一版报告（在双引擎负载下取的）缩小了 5–10 个百分点，
  说明 §3 那个"非独占"折价是真实的。

### 10.4 结论更新

- 配方与本项目在**引擎能力**上的差距，比第一版报告估计的还要小：pinned-host PLE 这条路的
  代码我们本来就有，卡住它的是 **Windows/WDDM 的 pin 上限**，而不是缺失的实现。
- 配方能跑，是因为 Linux 上 `cudaHostAlloc` 没有这层 WDDM 预算（它用的是 `mlock` 语义），
  且它要求 96 GB 以上内存；这正是"Linux 容器配方"与"Windows 原生"的**平台性差异**，
  不是配置差异。



### 10.5 如果内存加到 128 GiB：两个实例共享一份表（关键机制已实测）

128 GiB 解掉的是**总量**（共享预算 ≈64 GiB > 47.68），但还留着两个 Windows 特有的坑：
单次 pin 上限 ~16 GiB、以及"两个进程怎么共用一份物理页"。这两个都用探针验过了：

| 探针 | 结果 |
|---|---|
| `_ple_chunked_pin_probe.py --gib 2 --chunks 4` | 一次 `VirtualAlloc` 预留、按 4 个**页对齐**块分别 `cudaHostRegister`：**最后一块里的行**通过**同一个基址**读回，`MATCH`。说明分块注册后设备侧地址仍连续，**取行 kernel 不用改** |
| `_section_pin_probe.py --mib 512` | 页文件支持的命名 section 映射两个 view，两次 `cudaHostRegister` 都成功，两个 view 的取行都 `MATCH` |
| 同上 `--spawn-child` | **另一个进程**按名字打开同一 section、自己注册、自己取行 → `MATCH`。即"一份物理页、两个进程各注册一次"在 Windows 上成立 |
| `_cudart_hostalloc_probe.py` | 单次 14 GiB 成功 / 16 GiB 失败（与 §5 KV tier 当年 17.16 GB `cudaHostRegister` 失败是同一堵墙） |

于是 128 GiB 的账：

| 项 | GiB |
|---|---|
| FP8 表（一份，两个实例共享） | 47.68 |
| 两个引擎的非 PLE 主机内存 | ~2 × 18 |
| Windows + 桌面 | ~10 |
| 合计 | ~94（余 ~34 给页缓存/KV tier） |
| 每张卡的共享预算 | ~64（47.68 < 64，两张卡各映射一次） |

**要写的代码**（仍然不是"零改动"，但比 §6.1 原估的"多基址改造"小）：

1. `Qwen4ExpPLEPinnedHostEmbedding.allocate_embedding_weight`：`VirtualAlloc` 预留 → 按 ≤8 GiB **页对齐**块 `cudaHostRegister` → `torch.frombuffer` + `get_accelerator_view_from_cpu_tensor` 交给原 kernel。约 40–60 行，env 门控（如 `VLLM_PLE_PIN_CHUNK_MB`）。
2. Windows 版共享表（替换配方 patch 的 `fcntl`+`mmap`）：`CreateFileMapping(INVALID_HANDLE_VALUE, 名字)` + `MapViewOfFile` + 分块注册 + 命名 Mutex + 逐行填充/校验（复用树里已有的 `compute_ple_shard_overlap`）。约 150–250 行。
3. 启动顺序：第一个实例填表，第二个实例只校验；表不同就拒绝启动（配方同款）。

**剩下的风险**：两个 GPU 是否把同一物理页各算一次 aperture（按 `QueryVideoMemoryInfo` 是 per-adapter，且 47.68 < 64，大概率无碍，但未实测）；页文件支持的 section 在重启后丢失（每次起服重读 48 GiB，约 2–4 min，可接受）。

---

## 11. 主机侧 ngram ids：第一里程碑（2026-10-28 晚）

目标：把 PLE 的 ids 计算从图内 GPU 挪到主机，消掉 `_ids_ready.synchronize()` 等待的那 8.9–9.5 ms
（`FLASH_NEXT_ROUTE_A_BUILD.md` §14.3）。第一步先证"两条路算出来的 ids 逐位相同"。

### 11.1 一致性（`_dev/probe/_ple_ids_parity.py`）

用**实际 checkpoint**的三个 hash buffer（`layer_multipliers` / `ngram_heads_offsets` /
`ngram_heads_vocab_sizes`，16 头、ctx_len=2、eos=248044），把 `compute_ngram_ids` 的
CPU 分支与 Triton `ple_ngram_ids` 逐位对比：

```text
OK decode/1req/1tok            tokens=    1
OK decode/4req/8tok (dflash)   tokens=   32
OK decode/eos-in-context       tokens=   12
OK decode/eos-in-chunk         tokens=   16
OK prefill/1req/2048tok        tokens= 2048
OK prefill/chunk2-of-4         tokens= 2048
OK mixed/1+7+2048              tokens= 2056
OK int32-ceiling               tokens=    8
fuzzing 200 random batches ...
RESULT: PARITY OK
```

结论：**算法层没有风险**。参考实现就在树里——`compute_ngram_ids` 的非 CUDA 分支
（上游 `84030bbe3d`）、测试里的纯 Python `_reference_ngram_ids`（`test_ple.py:1015`）、
以及 `test_fused_ngram_ids_correctness`（:1168）对 Triton kernel 的逐位断言。

### 11.2 主机侧成本（单线程，5700X）

| 每步形状 | CPU 时间 |
|---|---|
| 4 请求 × 8 token（DFlash 宽度） | **0.34 ms** |
| 1 请求 × 8 token | 0.33 ms |
| 2048-token prefill chunk | 0.56 ms |

相对它要省掉的 9 ms 等待，成本可忽略。

### 11.3 数据通路（已确认的与待验的）

- `compute_ngram_ids` 只需要 `input_ids` / `query_start_loc` / `ngram_context` 三样；
- `RequestState.all_token_ids` 是 **UVA 缓冲**（`StagedWriteTensor(uva_instead_of_gpu=True)`，
  底层是 pinned CPU 张量 + GPU 别名，`buffer_utils.py:34-39,160-163`）——**主机可读**；
- 但**追加是 GPU 写的**：`add_request` 只在建请求时 `stage_write` 一次整段 prompt
  （`states.py:112`），之后每步把"上一步采样的 token + 本步 draft"写进
  `all_token_ids_ptr + total_len` 的是**输入准备 kernel**（`input_batch.py:440-452,586`）。
  因此 `prepare_inputs`（主机侧）直接读 UVA，最后 1 个 token 必然读到旧值；
  探针的结论是：**不能"直接读 UVA"，必须有一个主机镜像**，或在采样后把 token 抄回镜像。
- `InputBatch.input_ids` 只有 GPU 版（`input_batch.py:30`），没有现成的主机镜像。

### 11.4 落地接线（5 处，估 ~110 行）

| 文件 | 改动 |
|---|---|
| `model_state.py:prepare_inputs` | 从 UVA `all_token_ids` 读本批上下文+token，调 CPU `compute_ngram_ids`，产出 `ple_ngram_ids`（CPU int64） |
| `model.py`（Qwen4ExpModel.forward） | 接受可选 `ple_ngram_ids`，只传给 PLE 层 |
| `ngram_embedding.py:start_prefetch/forward` | 有预计算 ids 时不再调 `compute_ngram_ids` |
| `ple_ssd.py:start_prefetch` | 跳过 `_device_ids.copy_` / `_ids.copy_` / `_ids_ready`，直接用 host ids |
| 调试开关 `VLLM_PLE_IDS_VERIFY=1` | 同一批仍走 GPU 路径，比对一次/步并记 mismatch |

### 11.5 修正后的落地路径（2026-10-28 深夜）

UVA 直接读被否掉之后，主机 ids 需要一个**显式的 token 镜像**，接缝有两处可选：

| 方案 | 接缝 | 代价 |
|---|---|---|
| M1（推荐） | 采样返回的 host 张量 `sampler_output.sampled_token_ids`（`model_runner.py:2234`）——每步把 accepted token 追加进 `RequestState` 的 Python 列表；多请求/多 token 都在这条路上 | 改 `states.py` + runner 的 append 点；镜像与 GPU 写入必须同源同序 |
| M2 | 只镜像 decode：用 `ngram_context`（2 个 token，D2H 只要几 µs）推下一步，草稿位另算 | 只能覆盖 1 草稿或非投机；DFlash-7 下不完整 |

关键约束：投机解码的**草稿位**也要 ids，而 draft 在 GPU（`req_states.draft_tokens`），
host 侧要拿同样的一批 draft（调度侧 `SchedulerOutput.spec_decode_tokens` 本来就在主机上，
但 worker 不直接持有）。这是 M1 的主要工作量：**把"本步 accepted + 本步 draft"在主机侧复现成一份列表**，
与 GPU 的 `input_batch` 写入逐位对齐（用 verify 开关每步断言）。

所以工作量从"5 处 ~110 行"修正为 **~2–3 天**（含 verify 与边界用例），
比 §6.1 的表进 RAM（2–3 天 + 内存钱）差不多，但不花硬件钱。

### 11.6 下一步（按顺序）

1. 在 `RequestState` 加 host 镜像（M1），先把"镜像 vs `input_batch` 里实际 GPU token"逐位对齐跑通
   （verify 开关，不起生产）。
2. 接通 5 处：`model_state.prepare_inputs` → `model.forward` → `ple_layer.start_prefetch`
   → `ngram_embedding.start_prefetch` → `ple_ssd.start_prefetch`（跳过 D2H 与 `_ids_ready`）。
3. `VLLM_PLE_IDS_VERIFY=1` 跑一天真实流量，再关掉做独占 A/B。

注：`tests/` 是 package（有 `__init__.py`），pytest 会把仓库根加进 `sys.path`，
从而优先 import 仓库里的 `vllm/`（未编译扩展），所以本机只能用
`.flashnext-root/.venv` + 探针验证；正式 pytest 用例要等有编译环境的机器再加。


### 11.7 第一版实现与实测（2026-10-28 深夜）

实现（全部 `VLLM_PLE_HOST_IDS` 默认关，行为不变）：

| 文件 | 改动 |
|---|---|
| `model_state.py` | `prepare_inputs` 把 `input_ids`/`qsl`/`ngram_context` 异步拷进 pinned 暂存（`_ids_stream` + event），随 `ple_host_ids` 传下去 |
| `model.py` | 外层 `forward` 的 `**kwargs` 透传 `ple_host_ids`（第一次跑就是因为这里丢参数，服务起不来） |
| `ple_layer.py` / `ngram_embedding.py` | 透传；有 host ids 时 `event.synchronize()` + CPU 分支算 ids（并把三个 hash buffer 缓存一份 CPU 副本——第二次失败是 `layer_multipliers` 还在 cuda:0） |
| `ple_ssd.py` | 收到 CPU ids 就跳过 `_device_ids.copy_`、D2H 和 `_ids_ready` 等待 |

实测（`:9394`，独占 GPU2，同 harness）：

```text
INFO [ngram_embedding.py:934] PLE host ids match the GPU kernel (step 0)   ← 逐位一致
c1  结构化/代码/散文 = 189.4 / 238.0 / 125.8
c4  结构化/代码/散文 = 456.1 / 525.0 / 287.4
```

对照基线（§10.3 的 B/B′：c1 176–204 / 242–245 / 121–126，c4 452–462 / 513–533 / 317–321）：

- **没有可测出的收益**（差值都在噪声带内，c4 散文还低 ~9%）；
- 而且 verify 只在 step 0 打了一条，跑完整个 bench 也没出现 step 200 ⇒ `ple_host_ids` 很可能**不是每步都有**：
  `model_inputs` 在部分步上是复用的，那条路径拿不到 host ids，自动回落到 GPU 路径（正确，但也就没有收益）。
- 也就是说：第一版**证明了两条路数值一致、接线能跑通**，但没证明"消掉等待能换来吞吐"。
  下一版要把"每步都产出 host ids"钉住（在 runner 的 per-step 输入组装处，而不是 `prepare_inputs`），
  并同时量 `event.synchronize()` 的实际耗时才值得继续。

当前状态：snapshot 已回滚到原文件，`:9393`/`:9394` 都以生产默认档运行；
`VLLM_PLE_HOST_IDS` 的实现以默认关闭的形式留在仓库里，可随时续做或整体回退。

### 11.8 回滚踩的坑与最终状态（2026-10-28 23:40）

**坑（下次别再踩）**：`.flashnext-root/.venv/Lib/site-packages/vllm/**.orig` 是
`_patch_vllm_windows_runtime.py` 在 2026-10-01 留下的**上游原版**，不是"当前运行版"。
拿它回滚会把 DFlash/Eagle 补丁一起抹掉，服务起不来（`AssertionError: Model instance must
inherit from EagleModelMixin`）。正确回滚源是 `git show HEAD:vllm/...`。
另外 `Set-Content -NoNewline` 会把多行文本拼成一行（`git show | Set-Content -NoNewline`
直接产出一个 4.9 万字节的单行文件，import 时报 `cannot import name Qwen4ExpForCausalLM`），
要用 `[IO.File]::WriteAllText` 或普通 `Set-Content`。

最终状态：

- `:9393` / `:9394` 都以生产默认档运行（`:9394` 352 s 起服，DFlash 正常出草稿）；
- snapshot 的 5 个文件已回到 `git HEAD`（与改动前等价；`model_state.py` 多出的
  `VLLM_PLE_ALLOW_PP` 门在 PP=1 下是 no-op）；
- `VLLM_PLE_HOST_IDS` 的实现留在仓库里、默认关闭（不改行为），探针与文档都在。

### 11.9 为什么第一版没有收益：找到了（2026-10-29 00:20）

加了一组只读计数器（`VLLM_PLE_IDS_DEBUG=1`，外层 `ngram_embedding.start_prefetch` 每 100 次打一条，
内层 `ple_ssd.start_prefetch` 每 100 次打一条），跑 400 token × 单用户 × 3 类：

```text
PLE SSD prefetch hook: 1600 calls, last ids on cuda      ← 内层钩子每步都在跑，ids 是 CUDA 的
（没有任何 "PLE ids debug" 行）                          ← 外层包了一层 Python，稳态里一次都没跑
```

**结论：host 分支根本没有在稳态执行。** 原因在 CUDA graph 的结构：

- 解码步是 breakable CUDA graph，每步重放时只有被 `@eager_break_during_capture(always=True)`
  标记的 **`ple_ssd.start_prefetch`** 会被重新调用（它在图段之间执行主机侧的读行）；
- 而 `model.forward → _start_layer_ple_prefetch → ple_layer.start_prefetch →
  ngram_embedding.start_prefetch` 这层普通 Python 只在**捕获**时跑一次；
- 我把 host 分支挂在了最外层 `ngram_embedding.start_prefetch`，于是：
  - 稳态里永远走不到它 → 每步仍是 GPU 算 ids + D2H；
  - `VLLM_PLE_IDS_VERIFY` 那条 "step 0 match" 正是**捕获时**那一次（它证明了数值一致，但不代表生效）。

这也解释了实测：instrumented 版本的 c1 是 196.3 / 231.6 / 119.1，与基线（176–204 / 242–245 / 121–126）
没有区别——因为行为完全没变。

**正确的落点**（下一版）：

1. 每步都要产出的主机数据，必须由**每步都执行**的代码产出：`model_state.prepare_inputs`
   在 `model_runner.py:2005` 每步被调用（已核实），可以在那里暂存 + 后台算 ids；
2. 交付方式不能是 `model.forward` 的 kwargs（kwargs 只在捕获时被读一次）——要用一个**可变槽**：
   `model_state` 在 `__init__` 里扫出 PLE 模块（和 `ple_prompt_prefetchers` 的做法一样），
   每步把 `module.ngram_embedding.pending_host_ids = <ids|future>` 写进去；
   内层钩子 `ple_ssd.start_prefetch` 每步读这个槽，取不到就回落到 GPU ids；
3. 验收标准：debug 行里 `outer == host == SSD hook calls`，且 `sync=~0us/step`、`cpu≈300us/step`，
   再谈吞吐；上限仍然是那 2.5 ms/步（§14.4）。

教训：在这套 breakable CUDA graph 里，**"我在 Python 路径上加了代码"不等于"每步都会执行"**；
任何 per-step 的东西，先证明钩子调用次数与步数相等，再谈效果。

### 11.10 可变槽版本：终于生效了，收益是"分类型的小幅"（2026-10-29 00:20–00:40）

按 §11.9 改成"每步在 `model_state.prepare_inputs` 里把 provider 写进
`ple_ssd.pending_host_ids`，内层钩子消费"之后：

```text
PLE SSD prefetch hook: 2900 calls, host=2843, avg host cost 13885us, last ids on cuda
```

- **命中率 97–98%**（前 ~57 次是捕获/预热），说明槽确实每步被消费；
- 但 **avg host cost ≈ 11–14 ms**——这不是算 ids 的成本（CPU 只要 0.34 ms），
  而是 `event.synchronize()` 等"暂存拷贝被执行"的时间。暂存拷贝是在 `prepare_inputs` 里
  排在当前流上的，而 CPU 每步领先 GPU ~9 ms，所以**在钩子里等它就是等 CPU 的领先量**。
  换句话说：把 D2H 换成了 event 等待，**依赖本身没消失**。

实测（同 harness，`VLLM_PLE_HOST_IDS=1`，两次）：

| | 基线 B/B′ | 槽版第 1 次 | 槽版第 2 次 |
|---|---|---|---|
| c1 结构化/代码/散文 | 176–204 / 242–245 / 121–126 | 191.6 / 247.2 / **135.6** | 192.7 / 250.3 / **134.9** |
| c4 聚合 结构化/代码/散文 | 453–462 / 513–533 / 317–321 | 496.0 / 579.6 / 303.1 | 485.3 / 507.3 / 327.5 |

读法：

- **c1 散文稳定 +11%**（135 两次，基线三次都是 121–126），是这轮唯一可复现的收益；
- c1 代码/结构化在噪声内；c4 结构化/代码平均 +4~7% 但单次波动 ±14%，不能单独下结论；
- 每步仍有 11–14 ms 的宿主等待，只是被 GPU 掩盖——负载越高越难掩盖，这大概是 c4 散文没跟上的原因。

结论：**"把 ids 的 D2H 换掉"本身只值一点点**，因为真正的依赖是"主机要等 GPU 把 token/上下文算出来"。
要让这条路径真正快，必须做到**零 GPU 依赖**：主机从调度侧的 token 列表直接算 ids（就是把 §11.5 的
M1 host 镜像补上），这样钩子里不再有 event 等待。当前版本作为"opt-in、有微弱正收益"的实现保留在仓库里。

### 11.11 opt-in 落地（2026-10-29 00:50）

- 启动器新增 `-PleHostIds`（`start_qwen38_flash_next.ps1` 与内层脚本），设置
  `VLLM_PLE_HOST_IDS=1`；**默认关**，不传时该 env 被显式置 0，生产路径逐字节不变。
- 当前部署：`:9394` 带 `-PleHostIds`（试验臂），`:9393` 保持默认（对照臂）；两端口 health 200。
- 部署验证：c1 = 187.6 / 230.1 / **132.0**（散文对基线 121–126 仍是正的，幅度 5–11%）。

**为什么默认关**（而不是"有收益就默认开"）：

1. 收益只落在部分类别（散文 +5~11%、代码 0、4 用户不稳，c4 散文出现过 303 vs 327 的反向波动）；
2. 每步多一次 11–14 ms 的宿主等待，靠 CPU 领先量掩盖；并发或宿主负载升高时可能变成关键路径；
3. 只验了 ids 逐位一致，没有端到端质量（HumanEval/GSM8K）；这条路径的失败模式是**静默错行**；
4. 仓库惯例：新旋钮默认关，成为默认要过"干净 A/B + 质量 + 一周流量"。

### 11.12 质量抽查：opt-in 无退化（2026-10-29 01:10）

先量了引擎的确定性：同一 prompt、greedy、`:9393` 连发 3 次 → **2 次相同、1 次不同**
（与仓库此前"MoE padding/归约顺序导致 logprob 抖动"的结论一致），所以**不能**用逐字节相等做质量判据。

改用**差分可验证套件**（`_dev/probe/_quality_diff.py`）：同一套题同时打 `:9393`（默认）与
`:9394`（host ids），按通过数比较：

| 类别 | 内容 | 基线 :9393 | opt-in :9394 |
|---|---|---:|---:|
| math | 24 道程序化生成、整数答案的应用题 | 19/24（seed 7）、16/24（seed 11） | 19/24、16/24 |
| code | 6 道带单测的函数题（实际执行判分） | 6/6 | 6/6 |
| format | 4 道格式约束（JSON/项目符号/重复词） | 4/4 | 4/4 |
| **合计** | 68 题（两个 seed） | **55/68** | **55/68** |

两个 seed、每个 8 题一次的中间计数也**逐点相同**，说明差别远小于套件分辨率。
绝对分不高（math 67–79%）是套件本身多步算术偏难，不是模型退化；关键是**两臂完全一致**。

结论：host-ids 路径在可验证任务上**没有质量退化**，可以继续留在 `:9394` 观察。

### 11.13 转默认开（2026-10-29 01:20）

- 两个启动脚本都改为：**host ids 默认开**（`VLLM_PLE_HOST_IDS=1`），新增 `-NoPleHostIds` 显式回到 GPU ids；
  旧的 `-PleHostIds` 仍被接受（no-op）。`-DryRun` 会打印实际生效值：
  `env VLLM_PLE_HOST_IDS=1` / `-NoPleHostIds` 时 `0`（两条都验过）。
- `:9394` 本来就是带 host ids 起的，**不需要重启**；`:9393` 按用户要求**保持不动**（进程里仍是旧代码 + env=0）。
  注意：snapshot 已换成含该特性的构建，所以 `:9393` 将来一旦重启，**默认也会带上 host ids**；
  要让它继续做对照，重启时显式加 `-NoPleHostIds`。
- 质量侧：§11.12 的差分套件（68 题，两个 seed）两臂完全一致；性能侧：散文 +5~11%、代码中性、4 用户不稳。

### 11.14 当前状态下的即时对照（2026-10-29 01:35）

两个服务同时在跑、各自独占一张卡，同一 harness 顺序打（非并行，避免抢卡）：

| | `:9393` 默认（GPU ids） | `:9394` host ids | 差 |
|---|---:|---:|---:|
| c1 结构化 / 代码 / 散文 | 166.9 / 231.9 / 118.6 | 188.4 / 226.7 / **135.6** | +12.9% / −2.2% / **+14.3%** |
| c4 聚合 结构化 / 代码 / 散文 | 450.3 / 559.5 / 303.5 | 496.6 / **509.9** / 326.3 | +10.3% / **−8.9%** / +7.5% |

汇总今天所有测量（含之前的分时数据）：

- **散文**：host ids 稳定 +8~14%（`135.6` 出现了三次，基线 118.6–126）；
- **结构化**：当前对照 +10~13%（此前分时测试在噪声内，倾向于正）；
- **代码**：c1 中性（−2%），**c4 明显退化 −9%**；此前分时的 c4 代码也在 507–580 之间大幅摆动，均值与基线接近。

读法：host ids 每步多出的 11–14 ms 宿主等待，在 4 用户、GPU 更忙、草稿接受率高的（代码）场景下更容易顶进关键路径；
散文/结构化的步数更多、单步 GPU 占比更低，收益才显出来。

**结论**：默认开对"散文/长文为主"的使用是赚的；对"4 并发代码 agent"是略亏的。若要两者兼顾，
按服务分工：代码面服务用 `-NoPleHostIds`，长文面服务保持默认。

### 11.15 上下文口径与长上下文补测（2026-10-29 01:40）

之前的“散文 +14%、代码 −9%、c4 +10%”**全部是短上下文**：输入 53–80 token + 400 token 输出，
单请求实际上下文约 450–480 token（`max-model-len 262144` 只是配置上限，bench 从未接近）。

补测长上下文（`_dev/probe/_longctx_ab.py`，冷前缀缓存、每请求 fresh salt、各 2 次取中位）：

| 上下文 | prefill（两臂相同） | decode `:9393`（GPU ids） | decode `:9394`（host ids） |
|---|---:|---:|---:|
| 34k | 8.4 s（~4.1k tok/s） | 112.9 | 120.5 |
| 100k | 24 s（~4.2k tok/s） | 132.0 | 224.0 |
| 200k | 51.5 s（~3.7–4.1k tok/s） | 173.3 | 208.7 |

**但原始 decode 吞吐不可直接比**：同一 prompt、temp 0 下引擎本身非确定，生成的续写不同 →
DFlash 接受率大幅波动。用 `/metrics` 归一化后（100k，各 1 次冷跑）：

| | decode | steps | acc/step | tok/step | **step** |
|---|---:|---:|---:|---:|---:|
| `:9393` GPU ids | 118.1 | 48 | 1.65 | 2.67 | **22.4 ms** |
| `:9394` host ids | 217.8 | 30 | 3.33 | 4.27 | **19.4 ms** |

即：100k 下那个“+70%”主要是**接受率差异**（tok/step 2.67 vs 4.27），
host ids 真正买到的是**步时 22.4 → 19.4 ms（−13%）**——比短上下文（步时差异在噪声内）更明显，
因为被去掉的 D2H 等待随上下文变长而变大。

口径总结：**短上下文看 tok/s ±10% 就够；长上下文必须同时看 tok/step 与步时**，
否则会把接受率差异误读成路径差异。

### 11.16 日常场景（单/双并发 + 长上下文）实测（2026-10-29 02:00）

用户明确：日常是**单并发、最多双并发**的长上下文（KV 400k，多并发容易超），
所以短上下文 4 并发那组数据不适用于日常决策。补测（`_dev/probe/_longctx_conc.py`，
冷前缀缓存、每请求 fresh salt、用 `/metrics` 归一化）：

| 场景 | `:9393` GPU ids | `:9394` host ids | 步时 |
|---|---:|---:|---:|
| C=1, 32k | 22.7 ms（148.3 tok/s，3.37 tok/step） | 21.1 ms（160.1，3.37） | **−7%** |
| C=1, 64k | 24.1 ms（197.1，4.74） | 20.2 ms（244.1，4.92） | **−16%** |
| C=1, 100k | 22.4 ms（—，2.67） | 19.4 ms（—，4.27） | **−13%** |
| C=2, 100k | 10.9 ms（346.9，3.76） | 10.2 ms（386.9，3.94） | **−6%** |

（32k/64k 两臂的 tok/step 基本一致，差异来自步时；100k C=1 那次接受率不同，只看步时。）

结论：

1. **日常场景（单/双并发长上下文）当前版本就是赢的，6–16% 步时收益**；
2. **M1 对这种场景的边际收益≈0**：它要去掉的 11–14 ms 等待在长上下文步里已被 GPU 工作掩盖
   （否则步时不会变短），删掉一个不在关键路径上的等待不会改变端到端；
3. M1 唯一有价值的场景是"短上下文 + 高接受率 + 多并发"（本次实测未覆盖日常），可以先不做；
4. 该场景真正的吞吐杠杆是**草稿接受率**（tok/step 2.7–4.9 波动远大于步时差），
   下一步值得做的是长上下文下 DFlash 7 草稿 vs MTP 3 草稿的 A/B。

### 11.17 长上下文下的草稿选择：DFlash-7 vs MTP-3（2026-10-29 02:20）

之前那次 DFlash/MTP A/B（`FLASHNEXT_SINGLE_CARD_ACCEL_2026-10-21.md` §2）是 **200/128 token 短上下文**；
日常是长上下文，所以补测。两臂都在 `:9394`、都开 host ids，只改草稿方式（MTP 用 adaptive_k≤3）：

| 场景 | DFlash-7 步时 / tok/步 / tok/s | MTP-3 步时 / tok/步 / tok/s |
|---|---|---|
| C=1, 32k | 21.1 ms / 3.37 / 160.1 | **18.5 ms** / 2.85 / 153.7 |
| C=1, 100k | 19.4 ms / **4.27** / **224** | **18.1 ms** / 3.12 / 172.7 |

读法：MTP 每步便宜 1–3 ms（少验证 4 个 token），但 **DFlash 每步多产出 0.5–1.1 个 token**，
在长上下文里这个差距足以反超——**DFlash-7 仍是日常（长上下文单并发）的更优选择**，
当前默认不用改。（100k 只各 1 次，内容波动大；但 32k 两次重复也是同一方向。）

顺带确认：长上下文下 tok/步 的波动（2.8–4.3）比步时（18–21 ms）对吞吐的影响大得多，
所以这类对比必须看 tok/step，不能只看 tok/s。

### 11.18 长上下文下的 DFlash 草稿数：3 / 5 / 7（2026-10-29 02:40）

同一端口（`:9394`，host ids），只改 `-DflashDrafts`，C=1，冷缓存，每点 1–2 次：

| 草稿数 | 32k 步时 / tok/步 / tok/s | 100k 步时 / tok/步 / tok/s |
|---|---|---|
| 3 | **17.2 ms** / 2.84 / 165.2 | **16.7 ms** / 3.24 / 193.8 |
| 5 | 18.9 ms / 3.12 / 165.6 | 18.5 ms / 3.88 / 209.9 |
| 7（默认） | 21.1 ms / 3.37 / 160.1 | 19.4 ms / **4.27** / **224.0** |

读法：

- **32k**：三者 tok/s 基本持平（165/166/160），草稿多 → 步时变长、tok/步也变多，互相抵消；
- **100k**：草稿越多越赢（194 → 210 → 224），因为 tok/步 的增幅（3.24 → 3.88 → 4.27）
  大于步时的增幅（16.7 → 18.5 → 19.4 ms）。

结论：**长上下文单并发下 7 个草稿仍是最优，默认不改。** 想按上下文长度动态调草稿数，
现有 `num_speculative_tokens_per_batch_size` 是按并发数调的，不是按上下文长度，做不到这件事；
真要做得改引擎（按 `max_model_len`/实际上下文选宽度），收益在 100k 约 +7%（7 vs 5），优先级低。

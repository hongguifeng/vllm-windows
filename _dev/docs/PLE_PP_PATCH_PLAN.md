# PLE × PP 补丁计划（把目标模型跑到双卡）

状态：**侦察完成，尚未动代码**。用户已授权（"好"）修改 `vllm/`，但本轮会话上下文耗尽，实际改动 + 验证留给下一轮。
前提事实：双卡各 63.61 GiB、全空可用；目标模型 `D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`；PLE 表 95.37 GiB 走 SSD 流式（`ple_ssd_io_win.dll`）。

## 1. 两道闸门（就是它们挡住 PP）

| # | 位置 | 内容 |
|---|---|---|
| A | `vllm/model_executor/models/config.py:879-884` | `if text_config.ple_layer_ids and parallel_config.pipeline_parallel_size > 1: raise NotImplementedError("Qwen4Exp N-gram PLE embedding requires pipeline_parallel_size=1 because non-first pipeline ranks do not receive the raw input_ids it needs. Please run with PP=1.")`；上一行注释写着 "Checked again in Qwen4ExpModelState" |
| B | `vllm/models/qwen4_exp/nvidia/model_state.py:51-56` | `if vllm_config.parallel_config.pipeline_parallel_size > 1: raise RuntimeError("N-gram PLE embedding currently requires pipeline_parallel_size=1 because non-first pipeline ranks do not receive the raw input_ids required by PLE. Please run with PP=1.")`（触发条件：`self.uses_ngram_embedding = bool(config.ple_layer_ids)`） |

## 2. 闸门的前提**很可能是过期的** —— 这是本轮最有价值的发现

`vllm/v1/worker/gpu_model_runner.py`：
- `:3626` `input_ids = self.input_ids.gpu[:num_input_tokens]`（text-only 路径，**无 rank 条件**）
- `:3644` `if is_first_rank: intermediate_tensors = None else: sync_and_gather_intermediate_tensors(...)`（**只有中间张量按 rank 区分**）
- `:3539` `is_first_rank = get_pp_group().is_first_rank`，同段落里 `input_ids` 的构造与它无关

即：vLLM v1 里每个 PP rank 都会拿到同一份 `input_ids`（所有 PP rank 执行同一个 `SchedulerOutput`，PP 只切层）。所以"非首 rank 没有 raw input_ids"这个理由，对**本树**看起来不成立 —— 但那两个 raise 是有人踩过坑才加的，**必须用实验判定到底卡在哪**，不能只删 raise 就宣布成功。

## 3. PLE 的数据流（补丁要保护的东西）

- `Qwen4ExpModelState`（`vllm/models/qwen4_exp/nvidia/model_state.py:44-75`）建 `ngram_context`（int32，`[max_num_reqs, ngram_size-1]`，初值 `eos_token_id`）+ `ngram_context_offsets`；并给 PLE 挂 `PLEPromptPrefetcher`。注释提到这些 buffer 必须在 CUDA graph 捕获区里保持固定形状/地址。
- 模型侧：`Qwen4ExpModel.forward(..., ngram_context=...)`（`vllm/models/qwen4_exp/nvidia/model.py:577-586`），逐层 PLE gather 在 `model.py:290/307-315`；`ngram_context` 由 `kwargs.get("ngram_context")` 注入（`model.py:1145`）。
- 相关文件：`.../nvidia/ngram_embedding.py`、`ops/ple.py`、`mtp.py`、`ple_ssd.py`（SSD offload，含 `requires TP=DP=1` 闸门 `:574`）。

**未解的关键问题（动刀前必须先回答）**：`ngram_context` 每步是**谁**生产的？全仓只有 `vllm/models/qwen4_exp/**` 提及它，`vllm/v1/` 里没有；最可能的汇合点是 `gpu_model_runner.py:1090` 的 `_init_model_kwargs()`（把模型声明的额外输入合进 model_kwargs）。如果生产者只在首 rank 上跑，PP=2 会让 stage 1 拿到**陈旧/全零的上下文** → 输出会静默变坏（比崩溃更危险）。**先查这里，再改闸门。**

## 4. 建议的补丁形态（三选一，按风险升序）

1. **最小**：删/旁路两道 raise，`PP=2 + --additional-config` 原样跑，用**连贯性测试**判定。若 stage 1 的上下文本就来自本地 `input_ids`，这就能成。
2. **显式广播**：每步把首 rank 的 `input_ids`（2048 token × int32 = 8 KB）广播给其他 PP rank；代价可忽略（实测 4 KiB send/recv 只要 0.10 ms）。
3. **首 rank 预计算**：把 PLE gather 结果塞进 `IntermediateTensors`。**不推荐**：体积 = 层数 × tokens × hidden，比 hidden state 大一个量级。

## 5. 验证流程（每轮约 5-10 分钟；双卡必须全空）

1. `_dev/probe/_pp_tp_smoke.ps1 -Mode pp -Size 2 -NoGraphs -NoMtp -MaxLen 8192`（PLE **on**，即默认）
2. 起得来后：连贯性（短对话能否正常应答）+ `curl /v1/models`
3. 再开 MTP（`-Mtp`）、再开 graph（去掉 `-NoGraphs`）、再放大 ctx
4. 基准：`_dev/probe/_bench_port.py --port 9401`，与 §2.6 的单卡/PP/TP 三臂数字对齐比较

## 6. 纪律（本项目硬约束）

- 只动这两处 raise + 必要的最小配套；**先备份原文件**，把 `git diff` 原文记进 `_dev/out/`，验完要么提交到独立分支、要么还原成 0 改动。
- 不动 `:9393` / `:9394` 的生产启动脚本与配置；不下载任何权重。
- 每轮启动前：按 `CommandLine like '*vllm.entrypoints*'` 清理残留（否则 29550 被占 → `ZMQError: Address in use`），并核对日志 banner 是 `v0.1.dev21672+g4faed433c.d20261001`（`.venv` 的 site-packages 里有一份旧上游 vLLM 会顶掉 fork 树）。

## 7. 已知的 Windows 侧坑（本轮实测，详见 `PP_VS_TP_WINDOWS.md` §2.6）

`VSLANG=1033`（否则 flashinfer JIT 读 GBK 编译器输出炸 `UnicodeDecodeError`）；PATH 需带 venv 的 `ninja.exe` 与 `cl.exe`（否则 `FileNotFoundError [WinError 2]`）；`PYTHONPATH=<repo>` 必须显式 pin。

---

## 8. 第一轮实测（2026-10-16 00:46-00:51）：闸门放行了，但 worker 崩了

补丁：**+11 / −2**，两处 raise 各自加 `and not os.environ.get("VLLM_PLE_ALLOW_PP")`，未设该 env 时**生产行为完全不变**；备份在 `_dev/out/ple_pp_backup/{config,model_state}.py.orig`。

跑到哪一步：
- 两道 gate 都没拦（error 里已无 NotImplementedError / RuntimeError 的那句）。
- 两个 worker 各装载 **25.1 GiB** 权重 = 48.28/2 ✓（PP=2 的切分生效）。
- KV 分配完成：**32.67 GiB/卡、963,925 tokens**（8192 ctx）。
- 然后：`Worker proc VllmWorker-0 died unexpectedly (exit code: -15)` → `RuntimeError: Worker failed with error ''`（**错误串是空的** → 像原生层崩溃，不是 Python 异常）。

下一轮先看 `_dev/out/logs/pp2_real_20261016_004659.stdout.log` 里 worker 死前最后几行与两个 worker 自己的 stderr，怀疑点按序：
0. **已定位到精确失败点**：`StopIteration` 在 `vllm/v1/worker/utils.py:426` —— `kv_cache_tensors` 里某个张量的 `tensor.layers[0]` 在 `kv_cache_config.kv_cache_groups` 里**找不到任何包含它的 group**。也就是说 PP=2 下“张量集合”与“分组集合”不一致（不是 PLE 的问题）。再往下追到本树唯一的 PP 感知 KV 分组逻辑：`vllm/v1/core/kv_cache_utils.py:1309-1336`（注释："Return a Mamba group count whose PP projections fit the MLA slots"）—— 它用 `get_pp_indices` 逐 rank 算，**当某个 rank 有 mamba 层但没有 MLA 层时 `return None`**（`if not num_mla: return None`）。我们模型正是这种 hybrid（GDN 线性注意力 + 每 4 层一个全注意力），PP=2 的切分极易踩到那个分支。下一步：找它的调用方，看 `None` 怎么被消费（那里就是真正的修点），以及 `kv_cache_tensors` 是按什么范围生成的。
1. **PLE SSD offload 在 PP>1 下每 rank 都开一份表**（95.37 GiB 表 + 87.9 GiB 主机 RAM 总量）→ 先把 `--additional-config` 里的 offload 只留给一个 rank，或调小 `ple_ssd_cache_mb`；
2. 原生库（`ple_ssd_io_win.dll`）的句柄/线程在多进程下竞争；
3. 排除法最快：**`-NoPle` + PP=2 + 真模型** —— dummy 已证明 PP=2 机制本身 OK，若 NoPle 能起则问题锁定在 offload 路径。

---

## 9. 已解决（2026-10-16 01:20-01:55）：PLE × PP=2 在真模型上跑通

**结论：跑通了。** 真模型 `Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`（48 层，`ple_layer_ids=[2]`，仅 1 个 PLE 层位于 layer_idx=1），PP=2 + PLE SSD offload：**120-150 s 起服务**，KV **32.67 GiB/卡 = 963,925 tokens**，输出连贯（能正确回”3+5=8”并先做推理）。

### 两处修复（共 **+18 / −2**，三个文件，生产 PP=1 路径位级不变）

1. **两道 gate 加开关** `VLLM_PLE_ALLOW_PP`（不设=原行为、照旧拒绝；设了才放行）：
   `vllm/model_executor/models/config.py:879` + `vllm/models/qwen4_exp/nvidia/model_state.py:51`
2. **KV 分组/张量不对称**：`vllm/v1/core/kv_cache_utils.py:1927` 附近（`get_kv_cache_config_from_groups`）。

   根因链：`UniformTypeKVCacheSpecs` 分支原本**遍历 `group_spec.kv_cache_specs` 字典**、**完全不看 `group.layer_names`**。而组规格字典是 **rank-agnostic**（全局 PLE 层在里面），组的 `layer_names` 才是按 rank 切的 —— PP=2 时 rank 1 上那个 PLE 组是**空 `[]`**，却照样产出 `layers.1.ple` 这张张量 → 无组可归属 → `vllm/v1/worker/utils.py:426` 的 `next(...)` 抛 `StopIteration`。

   修复：`if skip_unowned and layer_name not in group.layer_names: continue`，其中 `skip_unowned = pipeline_parallel_size > 1`（PP=1 下连判断都不做）。

### 诊断手法（可复用，强烈建议以后先这么做）

1. 在 `worker/utils.py:426` 那个 `next(...)` **之前**打印孤儿层的名字 + 完整组列表 → 一眼看到“空组 + rank 不拥有的层名”。
2. 给 `KVCacheTensor` 加临时 `__post_init__`，对含 `.ple` 的层打印 `traceback.format_stack()` → **一次跑就直接链出创建者**：`engine/core.py:339 → kv_cache_utils.py:2893 get_kv_cache_configs → :1942 get_kv_cache_config_from_groups`。比读几百行代码快得多。

两个临时诊断**已撤除**，当前 `git diff -- vllm` = `+18 / −2`（三文件）。

### 性能（真模型，8192 ctx，eager，无 MTP，conc=1，4k 预填；两次一致）

| 指标 | PP=2 真模型（本次） | 备注 |
|---|---|---|
| 预填 TTFT | **0.691 / 0.730 s** | = 5,790 / 5,482 tok/s |
| 解码 | **8.6-8.7 tok/s，ITL ≈ 111 ms** | conc=1 |

⚠️ **不可与生产基线 medTPOT 9.05 ms 直接比**：基线是**单卡 + CUDA graph + MTP(k≤3)**，本次是**双卡 + eager + 无 MTP**。同设置的 dummy 三臂对照（§2.6）才是干净比较：PP=2 预填 −10%、解码 −54%。

### 尚未验证（下一轮）

- 开 MTP（`-Mtp`）、开 CUDA graph（去掉 `-NoGraphs`）；
- 262144 ctx 与长上下文（PLE SSD 压力真正上来时）；
- **同参数单卡对照**（当前 `_pp_tp_smoke.ps1` 只支持 `pp|tp`，要加 single 模式才能 A/B）；
- 第二处修复只对本模型生效路径（kv_cache_utils）；`vllm/v1/hisparse/layout.py:265` 有同构代码，若要支持 hisparse 模型需同样处理（本模型不走那条路）。

### 回滚

- 备份：`_dev/out/ple_pp_backup/{config,model_state}.py.orig`
- 一键复原：`git checkout -- vllm/`（三处改动全部还原）

---

## 9.1 追加实测（2026-10-16 08:45-09:00，`[我们复测]`）

### ✓ CUDA graph + PP=2：通过

150 s 起服务，`Breakable CUDA graph enabled`，KV **959,010 tokens**。

**开图后连贯性仍然正确**：同一道题先推理再答“北京” → PLE 的 n-gram 上下文在 graph 捕获＋PP=2 下是对的（这是关键，因为 PLE 状态用固定地址 buffer）。

### 解码性能（真模型，8192 ctx，4k 预填，**无 MTP**，conc=1/4）

| 配置 | conc | 聚合 tok/s | 每请求 ITL |
|---|---|---|---|
| PP=2 **eager** | 1 | 8.6 | 111.3 ms |
| PP=2 **graph** | 1 | **45.0** | **17.1 ms** |
| PP=2 **graph** | 4 | 60.2 | 36.3 ms |

两点结论：
1. **开图对解码约 5.2×**（ITL 111 → 17.1 ms）。
2. **并发 1→4 只把聚合从 45.0 提到 60.2（1.34×），每请求 ITL 反而 17 → 36 ms** —— PP 的串行化气泡**没有**被提高并发摊薄。

   ⚠️ **这推翻了本文件 §5 之前那个预测**（“`max-num-seqs` 提到 16/32，微批气泡会变小、−54% 会摊薄”）。至少 1→4 这个区间不成立；不建议再指望靠并发救 PP 的解码。

### ✗ MTP × PP=2：失败（**第三道独立缺口**，尚未解决）

profiling 阶段就炸（所以没到 KV 分配）：

```
model_runner.py:980 profile_run → :913 _dummy_run
  → spec_decode/autoregressive/speculator.py:332 propose → :490 _prefill → :462 _run_model
    → self.model(**{input_ids, positions, hidden_states, inputs_embeds})
      → AssertionError: assert intermediate_tensors is not None
```

根因：`speculator.py` **全文件没有任何 `intermediate_tensors`/PP 处理**，`_run_model` 只传四个输入 → 非首 rank 的模型实例必然 assert。

附带一条无关的 `logger.warning`：`kv_cache_utils.py:2366` “Speculative decoding (method=mtp) is enabled but no KV cache group could be identified as the draft model's.”（它只看 `group.is_eagle_group` 标记，**与本次两处修复无关**；mamba 混合模型的既有限制）。

参考：**生产 PP=1 + MTP 是正常的**，所以这是 PP 引入的缺口，不是 MTP 本身。

---

## 9.2 ✓ MTP × PP=2：**已修复**（1 行），`[我们复测]`

### 修法

`vllm/models/qwen4_exp/nvidia/mtp.py:294`，分支由看 rank 改为**看数据**：

```python
# 前
if get_pp_group().is_first_rank:
# 后
if hidden_states is not None or get_pp_group().is_first_rank:
```

### 为何这样是对的

1. `propose()` **只在末 rank 跑**：`pp_utils.py` 的 `broadcast_drafts` 带 `assert self.is_last_rank`（草稿算完再广播给非末 rank 去 embed）。
2. 末 rank 上 `get_pp_group().is_first_rank` **恒为 False** → 原分支总是去要 `intermediate_tensors`，而**在“propose 只在末 rank 跑”的设计下没有任何人会发给它**（speculator 没有任何 PP 处理）。
3. 真正可用的输入早就到了：speculator 的 `_run_model` 把目标模型的 hidden 以 `hidden_states=` 传进来（`speculator.py:450`）。数据驱动后，末 rank 走的是与 PP=1 **完全相同**的预处理（`pre_fc_norm_hidden` → `fc_hidden`）路径。
4. **PP=1 行为位级不变**（PP=1 时 `hidden_states` 本来就不为 None，条件真假不变）。

### 实测结果

- PP=2 + CUDA graph + MTP：**210 s 起服务**；输出连贯（“北京”）✓
- 服务端指标：**`Mean acceptance length: 2.58`**（生产单卡基线 1.78）
- 同 prompt 同 ctx（8192，conc 1，256 max tokens）A/B：

  | 臂 | TTFT | 总耗时 | 解码 |
  |---|---|---|---|
  | **+ MTP** | 0.212 s | **3.455 s** | **~327 chars/s** |
  | 无 MTP | 0.167 s | 4.785 s | ~195 chars/s |

  → MTP 给 PP=2 解码约 **1.68×**（代价：TTFT 0.167 → 0.212 s）。

### ⚠️ 方法论教训（重要）

用 `_bench_port.py` 那种 32-token 短样本测 MTP 会得到**相反的错误结论**（本次先得出“MTP 更慢 26 vs 45 tok/s”）：请求 14 token 就 EOS 撞断，且 MTP 下 token 成簇到达，中位 ITL 失去意义。
**测 MTP 必须用长生成** → 新工具 `_dev/probe/_longgen.py`（固定长 prompt + 流式计时 + 预热）。

### 当前累计改动（4 文件，`+19 / −2`，PP=1 路径位级不变）

| 文件 | 改什么 |
|---|---|
| `vllm/model_executor/models/config.py` | gate 加 `VLLM_PLE_ALLOW_PP` 开关 |
| `vllm/models/qwen4_exp/nvidia/model_state.py` | 同上 |
| `vllm/v1/core/kv_cache_utils.py` | KV 张量按组层名过滤（仅 PP>1）|
| `vllm/models/qwen4_exp/nvidia/mtp.py` | 草稿模型分支数据驱动 |

---

## 9.3 MTP 开启后的 prefill / decode 精确数字（同配置 A/B）

配置：真模型，PP=2 + CUDA graph，MaxLen 8192，conc=1，greedy，预填 4000 token，解码 256 token，长 prompt（数数）。

| 指标 | **+ MTP** | 无 MTP | MTP 效果 |
|---|---|---|---|
| 预填 TTFT @4k | **0.668 s**（5,987 tok/s）| 0.577 s（6,932 tok/s）| 无差异（噪声内；两臂历史区间 5,482-6,932）|
| 解码（3 次均值）| **79.3 tok/s**（78.7 / 79.3 / 79.8）| 57.5 tok/s（58.4 / 56.6）| **1.38×** |
| 端到端（含 TTFT）| 73.5-75.2 tok/s | 54.6-56.4 tok/s | 1.35× |
| 首 token | 0.192-0.230 s | 0.158-0.162 s | 贵约 40 ms |

有效性校验（服务端累计计数）：drafted 918 / accepted 534 = **58.2%** 草稿 token 被接受；平均接受长度 **≈2.6**（生产单卡基线 1.78）。

**因 MTP 而变慢/变快的分界：** 预填不加速也不明显受损；**收益全在解码**；首 token 多花 ~40 ms（要先跑草稿）。

注意：32k 预填在此配置下**测不了**（`MaxLen=8192` 直接拒绝超长 prompt）；要做长上下文预填需重启并调大 `-MaxLen`。

---

## 9.4 2026-10-19 换槽复测（一张 170HX 进芯片组 x4 槽）

完整数据：`_dev/docs/BENCH_PP2_SLOT_SWAP_2026-10-19.md`。

- 拓扑：170HX `06011618` 在芯片组 `04:00.0`（x4），`f897f9ac` 在 CPU `0C:00.0`（x8）；
  3090 占 `0B:00.0`（CPU x8）。CUDA 顺序是 0=3090、1=@0C、2=@04，PP 必须
  `CUDA_VISIBLE_DEVICES=1,2`（stage 0 放 CPU 卡，PLE 走宽链）。
- 跨卡 20 MiB PP 预填边界 **6.64 → 13.06 ms（1.97×）**，正好是 x8→x4；4 KiB decode 边界持平。
  主机拷文件时同一测量会假性变成 25.6 ms、MTP 解码腰斩 —— 必须在静默窗口测。
- 真模型 8k：预填 **0.640 s**、MTP 解码 **76.1 tok/s**（基线 0.668 s / 79.3）→ −4% 内；
  KV 池 712,157 / 959,010 与基线逐字节相同。
- 无 MTP：0.622 s / 51.7 tok/s / ITL 39.45 ms（基线 0.577 / 57.5 / 36.3）→ 慢 8-10%，
  **超出边界字节数能解释的量**（边界只值 +3.2 ms/chunk、decode 为 0），疑似 3090 回归与
  主机侧干扰，待 PP=1 同机对照。
- 启动前还要 `VLLM_PLE_ALLOW_PP=1`（§8 的闸门开关），否则直接 `NotImplementedError`。

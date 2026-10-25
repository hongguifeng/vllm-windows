# Fork 二次调研：`Morrowmake/vllm-cmp170hx` —— 能引入什么（2026-10-22）

对 [FORK_CMP170HX_IMPROVEMENTS.md](FORK_CMP170HX_IMPROVEMENTS.md)（2026-10-05 首次调研）的
第二次扫描。这 2.5 周里 fork 前进了很多：它现在跑的**就是我们这台机器的同一个模型**
（**Qwen3.8-Flash-Next**，EXL3 5.05 bpw，2x CMP 170HX，TP2，262K 上下文），代码整体迁进了
`vllm/sm80/`，而老调研点名的若干 opt-in 在上游又被删掉了。

依据：git 考古 + 端口差异（我们 HEAD `7b987a8b1e` vs `morrow/ampere` `f02c7a709a`）、fork
自己每次 rebase 重新生成的 `FORK.md`、以及
[qwen38 配方 README](https://github.com/Morrowmake/qwen38-flash-next-cmp170hx-recipe)。
机器可读底稿：`_scratch/morrow_commits.txt`、`_scratch/bucketB_new_bytheme.txt`、
`_scratch/perf_stats.txt`、`_scratch/_port_scan/PORT_DIFF_REPORT.md`，以及
`_scratch/recipe_src/sm80/` —— fork `vllm/sm80` 树的**字节级最新**副本（83/83 文件与 fork
HEAD 一致，移植时直接从那儿抄）。

注意事项：`morrow/ampere` 是**浅抓取**（边界 `84030bbe`，09-11）。Ampere 会不断 rebase 到
更新的上游（基点依次为 `81c993658e` → `c5422f4c46` → `970dc63478` → `4e0681ed56`），所以每个
fork hash 都比老调研看到的新；老调研引用的 hash 大多已悬空。分析窗口：2026-09-18
（merge-base `71fc70d3ae`）到 2026-10-19。

## 1. fork 现状

| | |
| --- | --- |
| 顶端 | `f02c7a709a`（2026-10-19） |
| 上游基点 | `4e0681ed56`（2026-10-13）—— 我们的同步停在 2026-09-18 |
| 新增支持的第二模型 | **Qwen3.8-Flash-Next，EXL3 5.05 bpw，TP2，262K ctx** —— 配方发布 1.0.0：decode **152.8 tok/s @C=1**、**674.7 @C=8**、冷 prefill **2,651 tok/s**、KV 池 **1,044,325 token** |
| 链路前提 | 配方全部测在 **PCIe x16** 上；我们是 Gen2 **x8**（实测中继 3.22 GB/s）。他们的 TP2 数字不可平移到这里 |
| 代码布局 | `vllm/sm80/` 83 文件 + `csrc/exl3`（独立构建 `_exl3_C`）+ 打补丁的 `vllm/models/qwen4_exp/**`；`plugins/sm80/` 只是打包/许可，不是代码 |
| 体量 | fork 自有产品文件 127 个 / 38,804 行；被改的上游文件 89 个（+4,021/−803）；**87 个环境变量开关**，默认全关 |

相对我们分支，ampere-only commit 共 **1,564** 个 = **1,401 个纯上游**（2026-09-18..10-14；其中
557 个晚于我们的老调研；只有 2 个与我们已单独 cherry-pick 的主题重合）+ **163 个 fork 原创**
（其中 **162 个在 2026-10-05 之后**：Morrowmake 162 + 社区 1）。这 162 个的主题分布：EXL3 61、
thin GEMM/Gluon 20、fla autotune 18、PLE 9、QSA/sparse 7、KV-memory 6、DFlash/MTP 3、MoE 3、
all-reduce 2、Marlin 2、plugin/platform 8、docs/tests 13。

## 2. 我们已经带的东西（先查重，再动手）

经 cherry-pick 进来的 fork commit：**0 个**。我们用的每一条 fork 思路都是手写移植，而且
**开关名和 fork 各不相同**。端口差异结论见 `_scratch/_port_scan/PORT_DIFF_REPORT.md`：

- **EXL3 整套：移植忠实。** `csrc/exl3` 25 个文件里 **23 个与 fork HEAD 字节级相同**（10 个
  kernel + `ops.h` + `ple_read.cpp` 全在其中）；op 面一致（15 个 `m.def` / 8 个 CUDA + 1 个 CPU
  `m.impl`）；Windows 侧多出的只有 `win_compat.h`、`ple_read_win.cpp`、一个 MSVC ABI shim 和
  编译参数。Python 胶水 `vllm/sm80/kernels/exl3/` 8/8 路径齐全（5 个字节相同；`loader.py` /
  `ops.py` / `quant.py` 的差异**只是** Windows 补丁：懒式一次只开一个 shard（safetensors 视图
  计入 Windows commit 上限，全开会吃掉 114 GiB/进程并以错误 1455 杀掉 TP worker）、用模块级
  字典代替我们树里没有的 persistent workspace manager、`quantize_lm_head` + `_map_name`）。
  `vllm/sm80/envs.py`（101 行）带 14 个 EXL3 开关，声明行与 fork 逐字节相同，经
  `vllm/envs.py:2503-2507` 注册。7 个开关判 PORTED：MOE_V2、DENSE_SPLITK、DENSE_MMA、
  DENSE_PREFILL_TC、MOE_PREFILL_TC、PLE_OVERLAP、PLE_NATIVE_READ（降级：串行）。
- 与老调研各 tier 对应的独立移植（全部已落地）：公平 chunked prefill（`a118c9c4e0`）、
  hybrid-KV scratch + KV in-flight 预留（`62438e437d`、`917a65418e`）、采样器词表钳位 +
  indexer top-k 平票修复（`91b8c7d55d`）、MoE 确定性对齐 + 计数排序 + padding mask
  （`c6972b3a7d`、`aae54d6945`）、主机中转 all-reduce 的 Windows 适配 + 实测
  （`0f64c3616f`、`b332b53b63`）、pinned FLA autotune（自写赢家表，`952a7d8b41`）、自适应
  draft 数（`9611389a89` + 默认开 `4ca47b14b6`）、DFlash2 drafter（`d4cfba2eaa`）、PLE SSD
  卸载 TP 感知（`1794ce0a4d`）+ 行缓存默认（`18a47333ec`）、并行 Triton warmup
  （`69e3ec7cb9`）、profiling 期间分配器 split 上限（`_scoped_allocator_max_split`，20 MiB）、
  Mamba/MTP prompt-tail 前缀缓存命中（`drop_eagle_checkpoint_block`）、DFlash 边界缓存
  （`0b0ad49693`）、KV-offload 的 Windows 层（`6423806d21`）、block 1680 / prefix-match 16
  两个旋钮（`21f1811d01`）。
- 按设计 ABSENT 的开关：THIN_GEMM(+_GLUON)、FLA_AUTOTUNE_SKIP_SPILLING、QSA_TOPK_CANONICAL、
  CUSTOM_ALLREDUCE_FLAGS/PCIE_P2P、SHARED_EXPERT_REORDER —— fork 83 个 `vllm/sm80` 文件里
  **71 个从未移植**（−22,309 行）。

## 3. 值得引入的缺口，按价值/成本排序

### T0 —— 把我们已经移植了一半的东西做完（价值/成本最高）

1. **把 PLE 行读搬到 PLE stream 上**（`b45889c243` + `604c0e42bd`，2026-10-14）。
   我们的 `ple_read.cpp` 与 fork 字节相同 —— 但 **Windows 编译的是 `ple_read_win.cpp`**，
   它既没有读线程池（`max_threads` 在 L299 校验了却从不使用），也没有 `cudaLaunchHostFunc`。
   所以 `VLLM_EXL3_PLE_HOSTFN_READ` 是个 PARTIAL 开关：门在、`PleHostReader` 类在、`run()`
   却是同步的。fork 那一整族细化（每步一次原生调用 `db8c707248`、先分配再起线程
   `df1d5829c0`、线程起不来时内联回退 `320cf16584`）同样到不了我们的构建。树内注释已经点名
   快路径：`ple_ssd_io_win.c` 里的 IOCP。收益：PLE 行读与下一层重叠；配方把
   `PLE_NATIVE_READ` 算进下面那个 decode 惩罚里。
2. **thin GEMM** —— 两个独立事实：
   - `vllm/ampere_thin_gemm/thin_gemm.py`（37 KB，自 `291e2d0b13` 起被 HEAD 跟踪）是**死代码**：
     全树没有任何地方 import 它。把 BF16 未量化投影（decode 行 ≤32 的 qkv/o 投影，加上 drafter
     的投影）接上，是小活儿。
   - fork 的 **Gluon** 版 thin GEMM（`521e41a634`）明确写着是为 **Qwen3.8 的 decode 形状**做的，
     而且我们 `.venv` 的 `triton 3.6.0` **确实有** `triton.experimental.gluon`，包括
     `language.nvidia.ampere` —— Windows 可行性看着很好。把那个包（`_scratch/recipe_src/sm80/
     kernels/thin_gemm/`，6 文件 / 2,930 行）连同他们 10-14 的新调度规则一起拿：capture 计数
     决定路由（`f7ee6ebee6`）、eager 走 `F.linear`（`a2c3310f7a`）、未编译布局回退 cuBLAS
     （`200a0abbc9`）、rows 1..32 全部 warm 且 warmup 失败即拒绝启动（`c6f905d6a1`、
     `8d42a1e21d`）。
   配方跑着 `VLLM_THIN_GEMM` + `VLLM_THIN_GEMM_GLUON`；这六个性能开关全关时实测 **每个 decode
   步 +18%（C=1）、+43/53/56%（C=2..8）** —— thin GEMM 是其中一部分。

### T1 —— 小、便宜、确定性/正确性

3. **FLA autotune 卫生三连**（`ca7da03c36`、`42ff7bba32`、`819146023f`；约 +610/−187）：把
   **spilling** 配置挡在 sm_80 autotune 之外、拒绝全 infinite 的候选、并复查缓存的赢家。
   可以干净地叠在我们现有的 `pinned_autotune.py` 上；配方默认开
   `VLLM_FLA_AUTOTUNE_SKIP_SPILLING`。保护首次启动、GDN 稳态速度、以及 autotune 偷走的 KV 空间。
4. **异步调度下失败输入的安全**（`e336673de0` 让 failed-token mask 活到 host copy 真正跑过 +
   `ccbe1c2d1d` 测试 + `8598b79184`）：只碰 `vllm/v1/outputs.py`（+6）和 `async_utils.py`
   （+21）。我们跑 `--async-scheduling` 且 graph buffer 会复用 —— 这正是会咬我们的那一类隐患。
5. **QSA 规范化 top-k**（`bdd090a46d` + `0c70d3521e` + `e89e160dbf`）：平票按规范顺序破 +
   排序选择，藏在 `VLLM_QWEN4_QSA_TOPK_CANONICAL` 后面；他们"同一请求每轮返回相同 token 与
   log-probability"的可重复性说法依赖它。我们的 top-k 走 `torch.ops._C.{cooperative,
   persistent}_topk`，所以他们的 Triton `topk_tiefix` **不能机械照搬**。**先探针**（用
   `_dev/fork_cmp170hx/probe_logprob_repeat.py` 重放同一 prompt N 次，比 token/logprob），
   只有平票确实不稳定才落地。
6. **DFlash drafter 每个 KV group 各取一份 spec**（`61495e7b81`）：DFlash2 跑在 hybrid KV-group
   模型上正是我们的生产配置；对照一下我们的路径是否缺这个修正。
7. **诊断件**：`state_index_check.py`（299 行，`VLLM_STATE_INDEX_CHECK`，GDN state 索引）和
   `kv_planning.py`（610 行）+ `VLLM_KV_PLAN_LOG`（auto-KV 预算）—— 两者都是对我们现有
   `VLLM_MEM_ATTRIBUTION` 的补充。

### T2 —— 中等成本，先测再信

8. **Qwen 精度与 draft 归一化开关**（`3d9c630883`、`c960bb4c92`、`71e65d2c74`、`710f4df691`）：
   `VLLM_QWEN4_ROUTER_FP32`（512 专家 / top-10 的路由 logits 走 fp32）、
   `VLLM_QWEN4_MTP_PER_STREAM_NORM`、MTP drafter 的 `get_top_tokens`。质量 + 接受率的杠杆；
   以评测为准（swe-bench / prose A/B）。
9. **shared-expert reorder**（`VLLM_SHARED_EXPERT_REORDER`，`layers/moe/` 603 行 + core 钩子）：
   decode 阶段让共享专家与路由专家重叠。上游的 stream 阈值只管 prefill；这个是 decode 向的。
   对我们的 AutoRound/Marlin 路径是否有效不明 —— 需要一次实测。
10. **PP hop 打包**（fork 的 `pp_hop.py` / `pp_utils.py` 补丁）：每 hop 一次、免元数据的交接。
    这台机器的正确布局就是 PP（见 `PP_VS_TP_WINDOWS.md`）；实测边界 6.6 ms / 0.1 ms 的 decode
    消息 —— 在 3.22 GB/s 下，压字节是唯一还剩的杠杆。
11. **fork 历史里已经内嵌、且碰到我们文件的上游漂移 PR**（ampere 对象已在本地，可单独
    cherry-pick）：`bc3ddb198a` **#60602 Ampere 上的 FP8 QSA 主 KV cache** —— 缩小 QSA KV 字节，
    直接的容量收益；`a686c8e1b7` **#60502** Triton kernel 迁离 `tl.make_block_ptr` —— 我们的 FLA
    副本比上游慢这一拍；`656d0d4506` **#61355** GDN decode tiles（SM100-family —— 跳过）；
    `a6131b0e65` **#57914** `/dev/shm` 缺失回退（只有当我们那样共享表时才对 Windows 有意义）。
12. **rebase 问题。** 我们同步点与他们基点之间隔着 1,401 个上游 commit（557 个晚于老调研）。
    逐条落 = 上面 1–11 在做的事；把 `flash-next-win` rebase 到 ~2026-10-14+ 的上游能一次全拿，
    但要重放我们 ~60 个 Windows/Flash-Next commit，得单独出计划（他们的基点是 ampere-only
    对象，我们本地已有；更早的历史需要 unshallow）。

### T3 —— EXL3 上不上生产

EXL3 整套已完整移植，但只在 probe 里跑过（`_dev/probe/_exl3_tp2_smoke.ps1` 开了 8 个开关里的
7 个；生产启动脚本一个都不开 —— 生产跑的是 AutoRound 3bpw checkpoint）。如果哪天单卡跑 EXL3
checkpoint，六个性能开关全开（DENSE_MMA、DENSE_PREFILL_TC、MOE_PREFILL_TC、PLE_NATIVE_READ、
THIN_GEMM、THIN_GEMM_GLUON）+ `FLA_AUTOTUNE_SKIP_SPILLING`，跟 AutoRound 做 A/B；配方的 TP2
布局（x16 链路、P2P 验证过）在这里复现不了，所以**单卡才是该做的那个实验**。

### T4 —— 不要引入

- Sparse-MLA / DSA / GLM-5 kernel（mHC、KDA、router v2、prologue fuse）、编译版 Ampere **Marlin**
  （`decode_gemm_v2` 形状锁死在 SM8.0 / bf16 activations / 288 experts / top-8 / hidden 4096 /
  uint4b8 g128；我们的 512-exp top-10、hidden 2560、AutoRound g64 不合格）。
- **P2P 与 custom all-reduce**（`VLLM_CUSTOM_ALLREDUCE_*`）：这台机器上三层独立证据都说
  不行（`CMP_P2P_FEASIBILITY_2026-10-20.md`：`cudaDeviceCanAccessPeer` 6/6=0、NVML
  `GPU_NOT_SUPPORTED`、注册表键不存在），而且即便解锁，TP 的算术也不改善（120 ms/chunk 不变）。
  主机中转路径已移植并实测。
- NVFP4（#56273 就是 fork 现在的基点 —— Blackwell 格式）、ROCm 路径、Engram hugepage、GLM-only
  开关（`VLLM_GLM5_*`、`VLLM_AMPERE_*`）、mimo 社区配方。
- fork 自己**已经删掉**的 opt-in：sm_80 sparse-MLA decode v2（`52ad0916b5` 加了、`97a7dbafba`
  又删了），以及 fork README「Switches」一节列为已删的那些路径 —— 别去追。`efd1f8dbfc` 甚至
  把我们带的 adaptive-k capture flag 也拿掉了 —— 落地时记得去重。

## 4. 落地顺序（每条配一个便宜探针）

| # | 条目 | 成本 | 前/后探针 |
| --- | --- | --- | --- |
| 1 | PLE 的 IOCP + PLE stream 上的 `cudaLaunchHostFunc`（`ple_read_win.cpp`） | M | EXL3 这条臂上跑 `_dev/probe/_itl_canary.py`；`VLLM_EXL3_PLE_TIMING=1` 看 host 侧时间 |
| 2 | 接线 thin GEMM；评估来自 `_scratch/recipe_src` 的 Gluon 包 | M | C=1/C=4 的 decode ITL50 A/B，`-NoAsync` 关，MTP 与 DFlash 两臂 |
| 3 | FLA skip-spilling + 缓存赢家复查 | S | 首次启动日志（`autotune winners`）、32K-prefill 的 TTFT |
| 4 | failed-token mask 生命周期 | S | 异步压测：N 个用户 + 被拒 draft + 畸形输入 |
| 5 | QSA canonical top-k | S/M | `probe_logprob_repeat.py` —— 只有平票不稳定才做 |
| 6 | DFlash 每 KV group 一份 spec | S | DFlash2 在 C=2..8 的接受率日志 |
| 7 | state-index check + KV plan log | S | 仅启动诊断 |
| 8 | router fp32 / MTP per-stream norm | M | swebench + prose A/B，看接受率差值 |
| 9 | shared-expert reorder | M | C=1 的 decode step time，走 Marlin 路径 |
| 10 | PP hop packing | M | PP2 slot-swap 基准（`BENCH_PP2_SLOT_SWAP_2026-10-19.md`） |
| 11 | #60602 FP8 QSA KV cache（+ #60502） | S/M | KV 池大小 + QSA 精度抽查 |
| 12 | rebase 到 ~2026-10-14 的上游 | L | 单独出计划，不属于这张清单 |

老调研那条规矩不变：每个功能都留在开关后面、默认关，进启动路径之前先在
`_dev/bin/_flashnext_struct_serve.ps1` 的 kill-switch 表里登记一行。

## 5. 只看 AutoRound 3bpw 生产路径时的适用范围（同日补充）

范围收窄：只优化当前 AutoRound 3bpw 生产模型，EXL3 与其它一律不要。于是 §3 要按
"这条改动依不依赖权重量化 / 依不依赖 EXL3" 再过一遍。

先摆 checkpoint 事实（`D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP\config.json`）：
`bits: 2`、`group_size: 64`、`sym: true`、`auto_round:auto_gptq`、
`block_name_to_quantize = model.language_model.layers`；`extra_config` 里 38,607 个 2-bit
条目（routed experts）+ **85 个 bits=16/fp 例外**，外加逐层 regex 例外。目录名里的
"3bpw" 是平均比特，routed experts 本身是 2-bit。

**（2026-10-22 晚复核，实测推翻初稿）** 初稿说 "`self_attn.*` 全部 fp16" 是**错的**。
直接用 safetensors header 读 dtype（`_dev/probe/probe_3bpw_bf16_shapes.py`）：凡是
AutoGPTQ 量化的层都同时有 `qweight`(I32)/`qzeros`(I32)/`scales`(F16)，**没有"原始 .weight
还是 bf16"这种中间态**。实测结果：

- **被量化（2-bit）**：`self_attn.{q,k,v,o}_proj`（12 个全注意力层 + MTP 层）、
  `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`、routed experts、`self_attn.indexer.index_qk_proj`。
- **留作 bf16 的**（每层 13 个 2-D GEMM；全注意力层 11 个，无 `in_proj_a/b`）：
  mHC 的 `input_mix_weight_down`(320,10240)、`input_mix_weight_up`(10240,320)、
  `block_inject_weight`(4,10240)，attn 与 mlp 各一套 = 6 个；`mlp.gate` router (512,2560)；
  `mlp.shared_expert.{gate,up}_proj`(640,2560)×2 + `down_proj`(2560,640)；
  `mlp.shared_expert_gate`(1,2560)；GDN 层的 `linear_attn.in_proj_a/in_proj_b`(48,2560)。
- **模型级 / 其它**：`ple.key_proj`(10240,2560)、`ple.value_proj`(2560,2560)、
  `hyper_connection_mixer`(320,10240 + 10240,320)、`lm_head`(248320,2560)、
  `embed_tokens`、`model.visual.*`、MTP 层同一套 11 个。
- 一个正向确认：这些 bf16 权重是真的 `BF16`，生产脚本也确实是 `--dtype bfloat16`
  （`_dev/bin/_flashnext_struct_serve.ps1` L300），所以 thin GEMM 的
  `weight.dtype is bfloat16` 前置条件成立。

每步 BF16 权重流量（48 层 × 上述 13 个）≈ **1.79 GiB/step**（mHC 6 个占约 1.2 GiB），
这就是 thin GEMM 要抢的那块时间。

这直接改写了两条结论：

1. **thin GEMM 就是为这种检查点准备的，升到 3bpw 范围的第一位。** fork 的挂钩点是
   `vllm/model_executor/layers/utils.py::dispatch_unquantized_gemm`，注释原话就是
   "the layers a W4A16 checkpoint leaves unquantized"；先读 `VLLM_THIN_GEMM` 这个 bool，
   未设置时连 triton 都不 import；Gluon 版（`521e41a634`）更是点名 **Qwen3.8 的 decode
   形状**。我们的 decode step 每层有 13 个 rows≤32 的 BF16 GEMM（全注意力层 11 个）≈ 600 次调用
   （mHC 6、router 1、shared_expert 3、shared_gate 1、GDN `in_proj_a/b` 2），再加 PLE、
   lm_head、MTP 层。**复核更正**：我们仓库里那份
   `vllm/ampere_thin_gemm/thin_gemm.py`（661 行，37 KB）**不是 fork 现在发的内核**——它是
   更早的候选版本（配置表是 GLM-5.3 的 6,416/6,144 那些形状，`dispatch_threshold()` 还从
   `os.environ["THIN_GEMM_MAX_M"]` 读），fork 现在发的是 `vllm/sm80/kernels/thin_gemm/`
   下的包：`__init__.py`(199 行，`ampere_thin_gemm`/`thin_linear`/`thin_gemm_supported`/
   `_CUBLAS_FROM_M`)、`thin_gemm_v74.py`(1,037 行，74-SM 重扫的调度表 + `_rotate_k`)、
   `warmup.py`(152 行，`_discover_nk` 从活模型上抓形状)。所以接线不是"给死代码加个门"，
   而是**替换 + 接门**：内核 3 文件 1,388 行（可选 `concat.py` 275 + `gluon.py` 299 +
   `thin_gemm_gluon.py` 968）。warmup 必须放在 `capture_model()` 之前（fork 在
   `kernel_warmup.py` 里调，理由是 split-K 的 FP32 partials 与 arrival 计数器落在
   allocate-once 的模块级 cache 里，capture 里首次分配会进 graph 私有池）。
2. **shared expert 是 fp16 GEMM**，所以 T2-9 的 reorder 对我们有意义，而且它**与量化无关**：
   `Sm80SharedExperts.adopt()` 只是把上游 `SharedExperts` 的类就地换掉，改 aux-stream 的
   **入队顺序**（上游 #52033 是先入队共享专家再跑路由；fork 改成路由排前面、共享专家在 aux
   stream 上排在后面，再 join）。我们 decode 的行数落在 `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`
   以内，上游的 overlap 本来就开着，这条改的只是"谁先谁后"。

| §3 条目 | 在 3bpw 范围适用？ | 依据 |
| --- | --- | --- |
| T0-1 PLE host-fn / IOCP | **不适用** | EXL3 的 PLE reader 专属；我们 3bpw 的 PLE SSD 走 `ple_ssd.py` + `ple_ssd_io_win.c`（IOCP，已在跑），不欠 fork 任何东西 |
| T0-2 thin GEMM | **适用，且成为第一位** | 上面第 1 条 |
| T1-3 FLA skip-spilling 三连 | 适用 | GDN 的 Triton kernel，与权重量化无关 |
| T1-4 failed-token mask 生命周期 | 适用 | async scheduling + spec decode 的核心路径 |
| T1-5 QSA canonical top-k | 适用（先探针） | QSA/indexer 是模型机制；`index_qk_proj` 本来就是 fp16 |
| T1-6 DFlash 每 KV group 一份 spec | 适用 | 我们的生产 spec 方法 |
| T1-7 两个诊断件 | 适用 | 与量化无关 |
| T2-8 router fp32 / MTP per-stream norm / `get_top_tokens` | 适用（评测门槛） | router gate 已是 fp16，升 fp32 是小事 |
| T2-9 shared-expert reorder | 适用（需 A/B） | 量化无关的入队顺序改动；见上面第 2 条 |
| T2-10 PP hop 打包 | 适用（若继续走 PP2） | 与量化无关 |
| T2-11 #60602 FP8 QSA KV cache / #60502 | 适用 | 改的是 KV cache dtype 与 Triton 代码生成，不是权重量化 |
| T2-12 rebase 到 ~10-14 上游 | 适用 | 与量化无关 |
| T3 EXL3 上生产 | 按你的范围：**不做** | — |
| EXL3 六个性能开关、loader/quant 细化 | **不适用** | EXL3-only |
| Marlin `decode_gemm_v2`、P2P/all-reduce、NVFP4、ROCm、GLM/sparse-MLA、fork 已删 opt-in | 不适用 | 同 §3 T4 |

一句话：**§3 的 T0 在 3bpw 范围只剩 thin GEMM 一条**；T1/T2 几乎全是与量化无关的核心/
模型改动，照单全收，落地顺序不变（thin GEMM 提到最前，PLE host-fn 那条划掉）。

## 6. 优先级总表（3bpw 生产范围，2026-10-22）

工作量刻度：**S** = 半天内可完成并验证；**M** = 1–2 天；**L** = 一周以上（含重放/回归）。
收益标注：**实测** = fork 或我们自己量过的数字；**推导** = 由机制/代码位置推断，方向可信、
幅度待测；**待 A/B** = 只有上机对比才知道值不值。

| 优先级 | 改进内容 | 工作量 | 预期收益 |
| --- | --- | --- | --- |
| 1 | **移植 thin GEMM 并接线**（**已落地 2026-10-22，生产默认开、bound=8**，见 §8）：fork 的 `vllm/sm80/kernels/thin_gemm/`（`__init__.py` + `thin_gemm_v74.py` + `warmup.py` = 1,388 行）+ 2 个开关 + `dispatch_unquantized_gemm` env-first 门 + `capture_model()` 前 warmup + aux-stream 独立 workspace lane + 启动脚本 `-NoThinGemm` kill switch | **已完成** | **实测(整步 A/B，同卡交替重启)**：C=1 ITL 20.29 → 19.47 ms（**-4.3%**）、47.7 → 49.1 tok/s；C=4 用 bound=8 打平、bound=32 则 -8%。eager 单算子曾给出 1.03–1.60×/7–9% 的乐观预测，进 graph 后未兑现（§8.4） |
| 2 | **Gluon thin GEMM**（叠加在 1 之上）：fork 的 `gluon.py`+`thin_gemm_gluon.py`，`SHAPES` 就是我们的 7 个未量化 (N,K)、`SM_COUNT=74` 也对得上 | **不做（已探明，收益不足）** | **实测**：Triton 3.7.1 在 Windows 上确实可用（`triton-windows==3.7.1.post27`，金丝雀 venv 启动成功、KV 415,306 与 3.6.0 一字不差、C=1 ITL 20.27 vs 19.99 ms、C=4 25.54 vs 26.12 ms → 升级本身零收益），而 Gluon 相对我们已上线 v74（1.03–1.60×，中位 1.25×）只多 ~11% GEMM 时间，折算上限 ~4.5% ITL、现实 1–2%，且 M=16（C=4）这一档救不了。为 1–2% 换整套 Triton 底座 + humming 补丁，不划算。详见 §10.2 |
| 3 | **FLA autotune 防 spilling**（**已落地 2026-10-23，生产默认开**，见 §9）：`pinned_autotune.py` 加 `SkipSpillingAutotuner` + `LOCAL_BYTES_LIMIT=1024`、`sm80/envs.py` 声明开关、启动脚本 `-NoFlaSkipSpilling` | **已完成** | **实测(本机)**：护栏生效（日志打出 gate），但**一个 config 都没被跳过**、KV 池 415,306 一字不差、吞吐不变——fork 那条 1.5 GiB 的账不适用我们的配置集。价值是 fail-closed 护栏，不是收益 |
| 4 | **失败 token 掩码生命周期**：fork 是 `vllm/sm80/failed_inputs.py`（`take_failed_tokens`/`handle_failed_inputs`，约 150 行）+ `scheduler.py` 一处钩子 + `ModelRunnerOutput.failed_req_ids` 字段 | S（但**前置**：我们的 PLE SSD reader 要先能报失败行） | **推导**：这条本质是"worker 造不出某 token 的输入时只失败那些请求，并把它们的 KV 从 prefix cache 摘掉"；fork 的触发源就是 EXL3 PLE 读盘失败。我们的 `ple_ssd.py`+IOCP 同样会读失败，但**目前不产生 mask**，所以单独移植它等于空转——要么先给 SSD reader 加失败上报，要么不做 |
| 5 | **QSA 规范化 top-k**：fork 两个开关——`VLLM_QWEN_QSA_TOPK_SORTED`（**默认 True**）与 `VLLM_QWEN4_QSA_TOPK_CANONICAL`（默认 False，规范破票序）；先跑重放探针 | S(探针)/M(移植) | **待 A/B**：fork 的"同一请求每轮返回相同 token 与 log-prob"可重复性依赖它；我们树里没有任何 `TOPK_SORTED/CANONICAL` 开关，top-k 走自己的 kernel，不能机械照搬 |
| 6 | **DFlash 每 KV group 各取一份 spec**：对照 fork `61495e7b81` 检查我们的 DFlash2-on-hybrid 路径 | S，但**前提未证实** | **暂缓**：fork 的 `vllm/sm80/spec_decode/` 里没有任何 kv-group 代码（只有 adaptive_k/aux_fc_fold/dflash2_model/lilicorr），GitHub API 被限流拿不到那个 commit；我们树里也没有 dflash × group/slot 的痕迹。要么拿到那个 commit 的 diff，要么从"hybrid KV 6 组 + DFlash2"现场反推 bug 形态，再决定要不要动 |
| 7 | **`VLLM_QWEN4_ROUTER_FP32`**：路由 logits 走 fp32（router gate 本就是 bf16）；fork 实现在 `vllm/models/qwen4_exp/nvidia/precision.py` + `vllm/sm80/envs.py`，我们树里**没有** `precision.py`（已确认）。注意我们已有的 `csrc/libtorch_stable/fp32_router_gemm.cu` 只支持 (3072,256)/(6144,128)，(2560,512) 不命中，不能直接复用 | S | **待 A/B**：512 专家 / top-10 的路由精度；影响输出质量与 draft 接受率，必须过评测 |
| 8 | **MTP per-stream norm + `get_top_tokens`**：draft 归一化变体 + drafter 取词接口 | M | **待 A/B**：draft 质量/接受率杠杆；评测门槛 |
| 9 | **shared-expert reorder**：`Sm80SharedExperts.adopt()` 就地换掉上游 `SharedExperts` 的类，改 aux-stream 入队顺序（路由先行、共享专家在其后跑再 join） | M | **待 A/B**：与量化无关；shared expert 是 fp16 GEMM，我们 decode 已落在上游 overlap 阈值内，这条只改 SM 争用时序 |
| 10 | **#60602 Ampere 上的 FP8 QSA 主 KV cache**（+ #60502 迁离 `tl.make_block_ptr`） | S–M | **推导**：KV cache dtype 与权重量化无关 → 直接扩池容量（bytes 减半量级）+ 跟随上游 Triton 代码生成；QSA 精度需抽查 |
| 11 | **PP hop 打包**：每 hop 一次免元数据交接 | M | **推导**：PP2 是我们的正确布局；实测边界 6.6 ms / 0.1 ms @3.22 GB/s，压字节是唯一剩的杠杆 |
| 12 | **诊断件**：`VLLM_KV_PLAN_LOG`（KV 预算逐组明细）已落地（§11）；`state_index_check.py`（GDN/Mamba state 索引范围检查）留待单独做 | **部分完成** | **实测(本机)**：启动日志一行给出每组 spec 类型/层数/block/page/charged/scratch 与最终 concurrency（本模型：4×MambaSpec + CircularBufferSpec(13 层, block=8) + MLAAttentionSpec(26 层, 157 block/请求) = 178 block/请求、其中 12 scratch，concurrency 1.5843，KV 415,306 与之前一致）。纯诊断、不改变行为 |
| 13 | **rebase 到 ~2026-10-14+ 的上游** | L | **实测(数量)**：一次拿 1,401 上游 commit（557 个晚于老调研）；但要重放我们 ~60 个 Windows/Flash-Next commit，单独出计划 |

已划掉、不进这张表：EXL3 六个性能开关与 loader/quant 细化、EXL3 上生产的 A/B、PLE
host-fn/`ple_read_win.cpp`（EXL3 专属，我们 3bpw 的 PLE SSD 已走 IOCP）、Marlin
`decode_gemm_v2`（形状锁死）、P2P/custom all-reduce（本机不可能）、NVFP4、ROCm、GLM/sparse-MLA、
fork 自己已删的 opt-in。

## 7. 复核记录（2026-10-22 晚，上机实测）

对 §5/§6 逐条复核，凡是"读代码能确认"的都重读了一遍，凡是"能不能跑/值多少"的都在本机
74-SM CMP 170HX 上量了。新增探针（都在 `_dev/probe/`）：

| 脚本 | 作用 |
| --- | --- |
| `probe_3bpw_bf16_shapes.py` | 只读 safetensors header，列出 3bpw 检查点里真正 bf16 的 2-D 权重、dtype、形状、层数，并算出每步 BF16 权重流量 |
| `probe_thin_gemm_triton360.py` | fork `thin_gemm_v74` 在 **Triton 3.6.0** 下的编译/正确性/warmup 冒烟 |
| `bench_thin_gemm_3bpw.py` | 用真实形状做 cuBLAS vs thin GEMM 对比（L2 规避、3 轮交替取中位） |

### 7.1 硬件与软件事实（推翻了初稿的隐含假设）

- 本机三张卡：0 号 RTX 3090（sm_86, 82 SM，**不是**目标），1/2 号 **CMP 170HX（sm_80,
  **74** SM, 63.6 GiB）**。内核 docstring 里写的 "70 SMs" 是别处的 70-SM 料号，**我们这张是 74**，
  正好等于 fork `gluon.py` 的 `SM_COUNT = 74`。
- Triton **3.6.0**（torch 2.11.0+cu130）：`thin_gemm_v74` **能编译、能跑、warmup 能过**
  （M=4/1 的 5 个形状 + 3 形状 × M 1..8 全部 OK，run-to-run 位重复），所以内核 docstring
  里 "Triton 3.7.1" 对这条路径不是硬约束。**但 Gluon 不行**：`gl.barrier` 在 3.6.0 的
  `triton.experimental.gluon.language` 里不存在，`thin_gemm_gluon` 每个形状都
  `CompilationError`。**这正是 §6 第 2 行从 "M、待 A/B" 改判为 "阻塞、等 Triton ≥3.7" 的原因。**
- 生产 dtype：`--dtype bfloat16`（脚本 L300），检查点里未量化权重确实是 `BF16`
  → `thin_gemm_supported()` 的 dtype 门通过。

### 7.2 单算子实测（`bench_thin_gemm_3bpw.py`，74 SM）

方法：真实形状 × M∈{1,2,4,8,16,32}，每个形状 4/2 份权重轮换 + 320 MB flush 破 L2，
cuBLAS 与 thin GEMM 交替跑 3 轮、各 30 次取中位；块间离散 ≤1 µs。

| 形状 (N,K) | 每步调用 | 最好/最差加速 | M=16 时省 | 备注 |
| --- | --- | --- | --- | --- |
| (320,10240) mHC down | 96 | 1.21–1.43× | 393 µs | M=1 最好 1.43× |
| (10240,320) mHC up | 96 | 1.11–1.25× | 197 µs | 稳定 |
| (4,10240) mHC inject | 96 | 1.14–1.54× | 492 µs | 单次最省 |
| (640,2560) shared gate+up | 96 | 1.17–1.50× | 590 µs | M=16 最好 |
| (2560,640) shared down | 48 | 1.00–1.15× | 49 µs | M=32 打平 |
| (512,2560) router | 48 | 1.15–1.50× | 295 µs | 稳定 |
| (48,2560) GDN in_proj_a/b | 72 | 1.22–1.60× | 295 µs | 最稳的形状 |
| (1,2560) shared_gate | 48 | 1.08–1.45× | 98 µs | GEMV |
| (10240,2560) PLE key | 1 | 1.04–1.17× | 7 µs | 一次/步 |
| (2560,2560) PLE value | 1 | **0.86–1.12×** | −1 µs | M≥16 变慢 |
| (248320,2560) lm_head | 1 | 1.03–1.10× | 46 µs | 带宽已饱和，几乎没肉 |

按真实调用次数折算到一步（48 层，单卡）：

| M（=请求数×(1+spec)） | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| 省下 | **1.89 ms** | 2.05 | 2.05 | 2.03 | **2.46** | 1.51 |

对生产 ITL 中位 **27.7 ms**（C=4、MTP k=3，见 `AUTOROUND_TP2_WINDOWS_2026-10-20.md`）：
**≈ 7–9%**，即 108.8 tok/s → 约 117–119 tok/s，KV 池不变。M=32 收益掉到 1.5 ms 也正好验证
fork 把 `VLLM_THIN_GEMM_MAX_TOKENS` 默认设 32 是有依据的（他们测到 40/48/64 是 0.80/0.76/0.72×）。

### 7.3 结论与更正清单

1. §6 第 1 行：内容从"接死代码"改为"**换内核 + 接门**"（1,388 行起），工作量 S–M → **M**，
   收益从"推导/目标 5–15%"改为"**实测(单算子) +1.9–2.5 ms/step ≈ 7–9% ITL**"。端到端 A/B
   仍未做（要上服务），但不再是无根据的估计。
2. §6 第 2 行：Gluon 从"待 A/B"改为"**阻塞：Triton 3.6.0 缺 `gl.barrier`**"，且顺手确认了
   它的 7 个 `SHAPES` 就是我们这个模型的 7 个未量化形状、`SM_COUNT=74` 与我们一致的卡相同 →
   将来升 Triton 后是直接可用，不是"还要重调"。
3. §5 的检查点事实：`self_attn.*`/`linear_attn.in_proj_qkv,z,out`/`index_qk_proj` **是 2-bit
   量化**，不是 fp16；真正留 bf16 的是 mHC 6 个 + router + shared_expert 3 + shared_gate +
   GDN `in_proj_a/b`，每层 13 个（全注意力层 11 个）。
4. §6 第 4 行：真实落点是 `vllm/sm80/failed_inputs.py`（不是 `outputs.py`+`async_utils.py`），
   而且**前置条件是 PLE 读盘要能报失败行**——我们现在的 IOCP reader 不报，所以单独移植是空转。
5. §6 第 5 行：补上 fork 的两个开关名（`TOPK_SORTED` 默认 True、`TOPK_CANONICAL` 默认 False）。
6. §6 第 7 行：确认我们树里没有 `precision.py`；也确认我们现成的 `fp32_router_gemm.cu`
   **不覆盖** (2560,512)，不能拿它冒充这条改进。
7. 尚未复核（留给落地前）：第 3 行 FLA skip-spilling 的具体收益数字（只有 fork 的"默认开"）、
   第 6 行 DFlash 每 KV group（fork 的 `61495e7b81` 在这份 shallow fetch 里查不到 commit
   对象，需要在完整克隆或 fork 网页上核对）、第 10 行 #60602/#60502 的 diff 内容、第 11 行
   PP hop 打包的字节账。

## 8. 落地记录：thin GEMM 已移植并接线（2026-10-22 晚）

### 8.1 落地了什么

| 位置 | 内容 |
| --- | --- |
| `vllm/sm80/kernels/thin_gemm/{__init__,thin_gemm_v74,warmup}.py` | 从 fork 抄三件（1,388 行），去掉 `concat.py`/`gluon.py` 两个入口（Gluon 需 `gl.barrier`，Windows 的 Triton 3.6.0 没有；见 §7.1） |
| `vllm/sm80/envs.py` | 新增 `VLLM_THIN_GEMM`（默认 0）、`VLLM_THIN_GEMM_MAX_TOKENS`（默认 32） |
| `vllm/model_executor/layers/utils.py` | `dispatch_unquantized_gemm` 里 env-first 门（未设开关时不 import triton，直接返回 upstream callable） |
| `vllm/model_executor/warmup/kernel_warmup.py` | `capture_model()` 之前跑 `warmup_ampere_thin_gemm`，并 warm lanes 0/1 |
| `vllm/model_executor/layers/fused_moe/runner/shared_experts.py` | aux stream 的 GEMM 显式走 lane 1（见 §8.2） |
| `_dev/bin/_flashnext_struct_serve.ps1` | 新开关 `-ThinGemm`（默认关，且强制 `VLLM_THIN_GEMM=0/1`，防止 shell 里继承的值泄漏）+ `-ThinGemmMaxTokens`，启动横幅打印 `thin-GEMM on/off` |
| `_dev/probe/_deploy_thin_gemm_overlay.py` | 服务实际 import 的是 `.flashnext-root\.venv`（junction → `.venv-flashnext`），不是本仓库；这个脚本按最小 diff 把包和 4 处补丁写进那个树，逐文件备份、幂等、`--restore` 可回滚 |
| `_dev/probe/probe_thin_gemm_dispatch.py` | 开关两臂 + 9 个形状的数值检查（M 超过 bound 必须回落 cuBLAS） |
| `_dev/probe/probe_thin_gemm_streams.py` | 双流并发正确性（§8.2 的复现/回归） |
| `_dev/probe/_ab_thin_gemm.py`、`_ab_pair.py`、`_dev/bin/_ab_thin_gemm_restart.ps1` | 端到端 A/B（交替重启，同一张卡） |

### 8.2 一个必须修的坑：split-K 工作区跨流竞争

`thin_gemm` 把 split-K 的 FP32 partials 和到达计数器放在"按 lane 分开"的模块级工作区里，
只在**同一条流**的发射串行时才安全。我们这条 3bpw 路径的 shared expert 在 aux stream 上
跑（`torch.cuda.stream(self._stream)`，decode 时行数落 `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`
以内就开），于是主流的 mHC/router 与 aux 流的 shared expert 会同时用同一个 lane：

- **复现**（`probe_thin_gemm_streams.py`）：同一 (N,K,M) 在两条流上背靠背发射，M=16 时
  结果与单流参考最大偏差 **6.59**（M≤8 因为跑完了没重叠，看不出来）。
- **线上症状**：开关打开、overlap 默认开时，spec decode 接受率从 ~1.5/草稿塌到 **0.011**
  （1,561 草稿只接受 18 个），aggregate 掉到 65 tok/s；把
  `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=0`（关 overlap）后恢复到 1.36——即"慢"是
  假象，真因是结果被写坏。
- **修法**：`__init__.py` 里加 `use_lane(lane)`（开关关时是 `nullcontext()`，不 import 内核），
  shared expert 的 aux-stream 块改成 `with torch.cuda.stream(self._stream), _thin_gemm_lane(1)`
  ——不认识的调用方仍是 lane 0。warmup 两个 lane 都跑（否则第一次分配会落在 capture 里）。
  修完 `probe_thin_gemm_streams.py` 全形状 `SAME`（偏差 0.0000），线上接受率回到 1.50。
- **成本**：两个 lane 的 split-K 工作区合计 **16.1 MiB**（日志实测），KV 池
  418,252 vs 开/关两臂 415,306–419,724，差异在 profiling 噪声内。

### 8.3 端到端 A/B

同一张卡（GPU2，CMP 170HX 74 SM）、交替重启、`_ab_thin_gemm.py` 同一个客户端，
prefill 4096 tok、4 并发各 128 tok、MTP k=3、CUDA graphs、PLE SSD 等参数与生产一致，
每臂 4 次取中位：

| 臂 | C=1 ITL | C=1 tok/s | C=4 ITL | C=4 tok/s |
| --- | --- | --- | --- | --- |
| off（第 1 轮） | 19.68 ms | 47.7 | 24.62 ms | 145.2 |
| on，bound=32（第 1 轮） | 19.41 | 49.5 | 28.04 | 124.7 |
| off（第 2 轮） | 20.89 | 46.0 | 26.28 | 136.8 |
| on，bound=32（第 2 轮） | 19.52 | 48.5 | 27.15 | 140.2 |
| **on，bound=8** | **19.42** | **49.1** | **25.70** | **143.0** |

读法（同一张卡的 off 两臂之间就有 6.7% 漂移，所以只看同向重复）：

- **C=1（M=4）：on 两轮都低于 off 两轮**（19.41 / 19.52 vs 19.68 / 20.89）→ ITL **-4.3%**
  （取中位 19.47 vs 20.29），throughput +3.5%。
- **C=4（M=16）：bound=32 的 on 两轮都高于 off 两轮**（28.04 / 27.15 vs 24.62 / 26.28）→
  约 **+8%**，是**真的退步**，不是噪声（两个 on 臂都慢于两个 off 臂）。
- **bound=8**（M≤8 才走 thin GEMM，M=16 回落 cuBLAS）：C=4 回到 25.70 ms（落在 off 的
  24.62–26.28 之间，等于打平），C=1 保住 19.42 ms。→ 生产该用的就是这个配置。
- 因此把 `-ThinGemm` 的默认 `-ThinGemmMaxTokens` 从 fork 的 **32 改成 8**。

### 8.4 §7 的预测错在哪

§7.2 用 eager 单算子微基准预测"整步省 1.9–2.5 ms、约 7–9%"。实测只兑现了 C=1 的 4%，
C=4 反而退步，原因很具体：

1. **eager 计时把 launch 开销算进了两边**。进 CUDA graph 后 launch 间隔被吃掉，cuBLAS 的
   小 M 惩罚跟着变小，thin GEMM 相对优势缩水；M 越大，缩得越多。
2. M=16 时 cuBLAS 已经换到还算像样的 tile，而 thin GEMM 的 split-K 是给 M≤8 的窄形状调的；
   §7.2 表里 (2560,2560)/(2560,640) 在 M≥16 已经反转，整步层面就是那条尾巴。
3. 整步里其他部分（MoE 2-bit、attention、PLE SSD、all-reduce 之外的 kernel）占比大，
   2 ms 的单算子账进不了 27 ms 的步长——除非把所有 BF16 GEMM 都换掉，而 M≥16 换不动。

**教训**：thin GEMM 这类"窄形状专用 kernel"的收益必须用**进 graph 后的整步**来定，
eager 微基准只能用来排候选、定 bound。


### 8.5 结论与建议配置


1. **生产默认开**（2026-10-22 晚定）：生产是 C=1 单流交互，正是唯一测到收益的档位。
   `-NoThinGemm` 是 kill switch；`VLLM_THIN_GEMM` 由启动脚本强制为 0/1，shell 里继承的
   值改不动它。`vllm/sm80/envs.py` 里代码默认仍是 0（不放开关就完全惰性），
   开着这条路只在启动脚本这一处。
2. **默认 bound 是 8，不是 fork 的 32**：C=1（M=4）ITL 中位 20.29 → 19.47 ms（**-4.3%**，
   两轮 on 臂都低于两轮 off 臂），throughput 47.7 → 49.1 tok/s；C=4（M=16）用 bound=8
   时与关闭打平（25.70 vs off 的 24.62–26.28），用 32 则 **-8%**。
3. **默认开的安全网**：启动脚本在 `VLLM_THIN_GEMM=1` 时先检查目标 venv 里有没有
   `vllm/sm80/kernels/thin_gemm`——本仓库的 `.venv` 是 2026-09-26 的旧安装、没有 sm80，
   默认 `-Venv` 指向它时会 **fail fast** 并提示用 `.flashnext-root` 或 `-NoThinGemm`。
4. **两个必须先修的坑已在代码里**：split-K 工作区跨流竞争（§8.2，不修会静默算错、spec
   接受率塌到 0.01）；以及 warmup 必须放在 `capture_model()` 之前并把 lane 0/1 都 warm。
5. **上线后实测（生产 :9393，默认配置）**：C=1 **20.24 ms / 47.4 tok/s**（4 次中位，
   KV 池 415,306 不变、workspace 2.9 MiB），C=4 **24.66 ms / 142.4 tok/s**。当晚机器整体
   漂了 5–8%（同卡 off 臂从 24.6 漂到 26.3），所以这两个数不能直接和 §8.3 的表逐格比；
   能说的是：默认开**没有带来 C=4 的退化**，C=1 落在 off 的 19.7–20.9 区间内。
6. **下一步**（若还想从这条线榨收益）：
   - 用 `VLLM_THIN_GEMM_GLUON` 的路子（fork 的 7 个 (N,K) 表就是我们的形状）——被 Triton
     3.6.0 的 `gl.barrier` 缺位挡住，升 Triton ≥3.7 后是现成的；
   - 把 M=16 也做对：要么给 v74 补 M=16 的 74-SM 调度行（fork 的表是按 GLM 形状扫的），
     要么在 M=16 只对胜出的几个形状（mHC inject、GDN in_proj、router）开；
   - 别再用 eager 微基准判 kernel 收益，直接上 `_dev/bin/_ab_thin_gemm_restart.ps1`。

## 9. 落地记录：FLA autotune 防 spilling（2026-10-23 凌晨）

§6 第 3 行。fork 的 `vllm/sm80/fla/pinned_autotune.py` 比我们的
`vllm/third_party/flash_linear_attention/ops/pinned_autotune.py` 多一个分支：
`VLLM_FLA_AUTOTUNE_SKIP_SPILLING=1` 时把每个 `Autotuner` 换成
`SkipSpillingAutotuner`——benchmark 前先编译每个 config、读它的 local memory
（`n_spills * 4` B/线程），超过 `LOCAL_BYTES_LIMIT=1024` 的直接跳过（记 inf，永不 launch），
全 inf 就抛错而不是随便挑一个；Triton autotune cache 命中时只编译赢家，若它也超限就重新
benchmark。理由是 CUDA 上下文会把 local 大小增长到"本进程 launch 过的最大值"且**永不回收**，
一次 benchmark 就能永久吃掉 KV 池（fork 举的例子：`chunk_fwd_kernel_o` 两 warp + 128×128
tile 11,664 B ≈ 1.5 GiB / 74 SM 卡）。

**移植**：`pinned_autotune.py` 加 `LOCAL_BYTES_LIMIT`/`_is_sm80`/`local_bytes`/`_skip_gate_open`/
`SkipSpillingAutotuner` 与包装器分支（我们本来就有 `PinnedAutotuner` + pin 表，缺的只是这一支）；
`vllm/sm80/envs.py` 声明开关；启动脚本加 `-NoFlaSkipSpilling`（默认开，强制 0/1）。

**实测（生产 :9393，默认开）**：启动日志出现
`fla autotune on sm_80 skips configs whose build needs more than 1024 B ...`，
但整轮启动**一个 config 都没被跳过**（`fla autotune skips` 计数 0），KV 池
**415,306 tokens 与之前完全一致**（415,306）——也就是说 fork 那条 1.5 GiB 的账**不适用于我们的
配置集**：我们的 `chunk_fwd_kernel_o` 用的是 BK=64/BV=64/4 warps，不是他们那个 128×128 两 warp
的 tile。吞吐 sanity：C=1 19.49 ms / 48.4 tok/s，C=4 26.97 ms / 133.1 tok/s，与开关前同带。

**结论**：这条在我们这儿是**护栏而不是收益**（0 skip、KV 不变、吞吐不变）。保留默认开——它
fail-closed（真的出现全 spilling 的 config 会拒绝启动而不是带着 1.5 GiB 的窟窿跑），成本只有
cache 命中时编译一次赢家。等以后新增 FLA config 时（比如 §6 第 10 行的 Triton 代码生成）再回看。

## 10. Triton 3.7.1 在 Windows 上可用（2026-10-23 验证）

§6 第 2 行的 Gluon 阻塞点是 Triton 版本，不是硬件。查证与验证结果：

- **官方 `triton` 包在 Windows 上没有发行版**：`pip index versions triton` → `No matching distribution found`。Windows 的 Triton 由 `triton-windows`（PyPI，维护者 woctordho）发布，装进去仍是 `triton` 这个导入名。
- **`triton-windows` 有 3.7.x**：`3.7.0.post26`、**`3.7.1.post27`**、以及 `3.8.0.post29`（当前 latest）。生产 venv 现在是 `3.6.0.post26`（`triton 3.6.0`）。
- **隔离 venv 实测**（`_scratch\_gluon371`，python 3.12.13 + `torch==2.11.0+cu130` 与生产同版本 + `triton-windows==3.7.1.post27`，2.8 GiB）：
  - `triton 3.7.1`，GPU 认成 CMP 170HX / 74 SM 正常；
  - `triton.experimental.gluon.language.barrier` = **True**；`NVMMADistributedLayout` / `DotOperandLayout` / `SwizzledSharedLayout` / `allocate_shared_memory` 全在；`ampere.async_copy`、`ampere.mma_v2` 可导入 → **fork 的 `gluon.py` 前置检查全部通过**；
  - 我们已上线的 `thin_gemm_v74` 在 3.7.1 下**编译、数值、warmup、位重复性全过**（而 v74 的 docstring 本来写的就是 Triton 3.7.1）。
- **结论**：Gluon 现在只差“要不要升 Triton”这一个决定。升级的代价是一次全量回归（FLA/GDN、QSA、PLE、采样器、thin GEMM 全在这套 Triton 上）+ `torch.compile` 路径未测（生产是 `CompilationMode.NONE` + 图捕获，影响面小）。升级本身是 venv 内一条命令，可回滚（先记 `triton-windows==3.6.0.post26`）。

**建议路径**：复制一份 venv（或 `.venv-flashnext` 的 clone）升到 3.7.1.post27，先在 GPU2 上跑“启动 + KV 池 + C=1/C=4 bench + 接受率”四项对照；干净再换生产，然后才接 Gluon 那 6 个文件。

### 10.1 升级实测：3.7.1 能装能导，但**生产在 humming 这一步起不来**（2026-10-23）

按上面的建议做了金丝雀：复制生产 venv 到 `_scratch\t371\.venv`（6.28 GiB，robocopy 72 s），
`uv pip install triton-windows==3.7.1.post27`，用同一套启动参数在 GPU2 上启：

- **Triton 层全过**：`triton 3.7.1` + `torch 2.11.0+cu130`，GPU 认成 74 SM CMP 170HX；
  `gl.barrier` True；Gluon ampere API 齐全；`thin_gemm_v74` 编译/数值/warmup/位重复性全过（§10）。
- **启动挂在量化 kernel 栈（humming）**：模型加载到 `process_weights_after_loading` 时
  `humming` 决定“重建原生产物”，走 `humming/build.py::build_native()`，用
  `compiler = os.environ["CXX"] or which("g++") or which("c++")` 取到 `g++`，再执行
  `g++ -O3 -std=c++17 -fPIC -shared …`（POSIX 分支），本机没有 g++ →
  `FileNotFoundError: [WinError 2]`，引擎在 KV profiling 之前就死了。同一份 venv 在
  **3.6.0** 下同样的启动参数是好的，所以触发点是 humming 里跟着 triton 版本走的“原生产物
  是否需要重建”判断，不是 wheel 本身。
- 金丝雀 venv 里 `humming/_native/x86_64/` 的四个产物（`libcubinpatch.dll`、
  `libhumming_launcher.so`、`nvrtc_compile.exe`、`_device_info.pyd`）与 `manifest.json`
  哈希与生产**完全一致**，所以不是复制漏文件。
- 诊断手段留在 `_scratch\t371\.venv\Lib\site-packages\sitecustomize.py`（包装
  `subprocess.run/Popen` 打印失败命令），只在金丝雀里，生产没碰。

**结论**：Gluon 的阻塞点从“Triton 有没有 3.7 轮子”变成了“**humming 在 3.7 上要走重建分支**”。
要继续只有两条路：①找到 humming 的“需要重建”判据并满足它（或把仓库里
`_dev/bin/_humming_launcher.ps1` / `_humming_cubinpatch.ps1` 用 MSVC 预编译成它要的签名）；
②先不上 Gluon，留在 3.6.0。生产目前留在 3.6.0，未做任何升级。

### 10.2 humming 的“需要重建”判据找到了，而且能绕过去（2026-10-23）

**判据**（`humming/utils/jit.py`）：

```python
def hash_path_content(path, releative=False, text_only=True):
    if os.path.isfile(path):
        filename = path.split("/")[-1] if releative else path   # ← 这里
```

`hash_path_content` 对**文件**类型的源用 `path.split("/")[-1]` 取“相对名”。Windows 路径用反斜杠，
`split("/")` 不切分，于是“文件名”变成了**整条绝对路径**——哈希带上 venv 路径。后果：

1. `get_precompiled_artifact_path()` 拿 `manifest.json[artifact]` 比这个带路径的哈希 →
   **换一个目录的 venv 必然 miss**（生产自身其实也 miss：`_humming_*.ps1` 打过补丁后
   source 变了、manifest 没重写，所以它一直走 cache）；
2. 于是落到 `~/.humming/cache/<hash>` —— 同样带路径 → 新 venv 又 miss；
3. 最后走 `build_device_info_extension()` / `build_cubin_patcher()`，它们的命令行是
   **POSIX** 的（`g++ -O3 -std=c++17 -fPIC -shared -lcuda`，`device.py:106` 的
   `CXX or "g++"` 没有 win32 分支），Windows 上没有 g++ → `FileNotFoundError`。
   目录型源（`libhumming_launcher.so` 的 `csrc/launcher`）用的是 `os.path.relpath`，
   所以只有它是对的（两个 venv 的哈希都是 `9f27a9d0850315d7`）——这也解释了为什么只有
   文件型的三个产物出事。

**绕过方式（不需要任何编译器，只改金丝雀 venv）**：

1. `humming/utils/jit.py` 第 146 行改成 `os.path.basename(path) if releative else path`
   → 文件哈希变成“内容 + 文件名”，与路径无关；
2. 用改过的函数重写 `humming/_native/x86_64/manifest.json`（三个文件型产物 + launcher 目录），
   目录里本来就有 `_device_info.pyd` / `libcubinpatch.dll` / `nvrtc_compile.exe`；
3. 验证：`get_precompiled_artifact_path()` 三个都 **HIT**，`_get_device_info_extension()`
   与 `may_build_cubin_patcher()` 都能加载，**一次编译都不触发**。

**金丝雀（triton 3.7.1）最终启动成功**：KV 池 **415,306**（与 3.6.0 一字不差）、
thin GEMM warmup 35 形状/M≤8/2.9 MiB、FLA 防 spilling gate 正常，日志里只剩一条无害的
`rocm-sdk path --root` 探测失败。

**同卡同场 A/B（GPU2，canary 3.7.1 vs 生产 3.6.0，各 4 次中位）**：

| 臂 | C=1 ITL | C=1 tok/s | C=4 ITL | C=4 tok/s |
| --- | --- | --- | --- | --- |
| triton **3.7.1**（金丝雀） | 20.27 ms | 46.4 | **25.54 ms** | **143.9** |
| triton **3.6.0**（生产同款） | 19.99 ms | 46.8 | 26.12 ms | 133.9 |

C=1 基本打平（+1.4%），C=4 **-2.2%**（更快），KV 池相同、接受率都在 1.2–1.5 的噪声带里。
结论：**3.7.1 在这套栈上无回归**，可以作为接 Gluon 的底座。

**顺带发现的一个隐患**：这个 `hash_path_content` 的 Windows bug 意味着本项目的 venv
**一换目录就起不来**（复制/迁移/换盘都算）。重启部署流程时应在装完 humming 后打这一行补丁
并重写 manifest，否则 `_humming_*` 预编译的产物形同虚设。

**下一步（若要接 Gluon）**：把这两个补丁（`jit.py` basename + manifest 重写）落到生产
`.venv-flashnext`，装 `triton-windows==3.7.1.post27`，再拷 fork 的
`concat.py`/`gluon.py`/`thin_gemm_gluon.py` 并把 `VLLM_THIN_GEMM_GLUON` 支路接进启动脚本。

## 11. 落地记录：`VLLM_KV_PLAN_LOG` KV 预算诊断（2026-10-23）

fork 的 `kv_planning.py` 大部分我们在上游已经有了（`get_max_concurrency_for_kv_cache_config`、
`speculative_scratch_bytes`、`_pool_concurrency_limit`，连 `VLLM_KV_MAMBA_INFLIGHT_STATES`
那两条 `_log_*` 都在），缺的只是**逐组明细的那一行日志**。补上：

- `vllm/sm80/envs.py` 声明 `VLLM_KV_PLAN_LOG`（默认 0）；
- `vllm/v1/core/kv_cache_utils.py::get_max_concurrency_for_kv_cache_config` 在算
  `charged/scratch` 的循环里顺手拼一行，`logger.info_once` 打出来；纯日志，默认关；
- 启动脚本加 `-KvPlanLog`（非诊断臂强制 `VLLM_KV_PLAN_LOG=0`）。

本模型实测输出（GPU2，与生产同参数）：

```
KV cache plan (VLLM_KV_PLAN_LOG): num_blocks=282, 178 blocks per request of which
12 scratch, max_num_seqs=4, concurrency 1.5843;
[0] MambaSpec          layers=12 block=1680 page=3440640 charged=5  scratch=3;
[1] MambaSpec          layers=12 block=1680 page=3440640 charged=5  scratch=3;
[2] MambaSpec          layers=12 block=1680 page=3440640 charged=5  scratch=3;
[3] MambaSpec          layers=1  block=1680 page=3440640 charged=5  scratch=3;
[4] CircularBufferSpec layers=13 block=8    page=2240    charged=1  scratch=0;
[5] MLAAttentionSpec   layers=26 block=1680 page=107520  charged=157 scratch=0
GPU KV cache size: 415,306 tokens, Maximum concurrency for 262,144 tokens per
request: 1.58x
```

读法（对以后调 KV 有用）：

- **块预算几乎全被 MLA 组吃掉**：26 层每请求 157 个 block（page 107,520 B），其余五组加起来 21；
- Mamba 组 page 3,440,640 B 是 giant page，但每请求只要 5 个 block（其中 3 个 scratch，只在
  running 时占）——`max_num_seqs=4` 时 scratch 总量被 4 倍封顶，这正是
  `_pool_concurrency_limit` 的账；
- `CircularBufferSpec`（13 层、block=8）每请求 1 个 block，几乎不花钱；
- 想扩并发就动 MLA 组（block size / 层数 / KV dtype），Mamba 组不是瓶颈。

部署与回滚同样走 `_dev/probe/_deploy_thin_gemm_overlay.py`（本轮新增两处
`kv_cache_utils.py` 补丁，幂等、带备份）。

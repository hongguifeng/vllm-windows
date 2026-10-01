# HyperQwen（原 qwen38-27b-rtx3090）补丁系列 → vllm-windows 移植评估

评估对象：WSL `~/testcode/qwen38-27b-rtx3090`（上游 `syv-ai/HyperQwen`，fork `hongguifeng/qwen38-27b-rtx3090`）
目标：本项目 `D:\code\vllm-windows`（vLLM 0.29.0 源码树 + Windows/sm_80 构建）
评估方式：把 40 个补丁**按 `patches/series` 顺序**实打到 0.29 树的干净副本上，逐条统计冲突。

---

## 一、两个项目的定位差异

| 维度 | HyperQwen (WSL 3090) | 本项目 (vllm-windows) |
|---|---|---|
| 本质 | **部署调优仓库**：pinned vLLM wheel + 41 个补丁 + 模型预处理流水线 + Docker/compose | **平台移植工程**：把 vLLM 源码编译到 Windows/sm_80 并跑通 |
| 基线 | fork 的 `qwen38/0.28` 线（v0.28.0 + 每主题一 commit） | vLLM **0.29.0** release（`13e844c86d`），`SystemPanic/vllm-for-windows` |
| GPU | RTX 3090，**sm_86**，24 GB，250 W | CMP 170HX = GA100，**sm_80**，64 GB |
| 平台 | WSL2 / Ubuntu / Docker | **原生 Windows**，MSVC 14.44 + CUDA 13.3 |
| 目标模型 | Qwen3.8-27B（W4A16 AutoRound + DFlash2 drafter） | 通用（目前烟测 Qwen2.5-0.5B） |
| 优化方向 | **省显存 + 投机加速**：int4/int8 KV、KVarN、vision offload、KV offload 层、DFlash2 | **让 vLLM 在 Windows/sm_80 上正确编译与运行** |
| 交付物 | `.env` + `start_qwen.sh` + `verify.sh` | `.whl`（187 MB）+ `_dev/bin/_build.ps1` / `_dev/bin/_run.ps1` |

一句话：**那套是"24 GB 卡上怎么把 27B 跑快"，本项目是"这块卡怎么把 vLLM 跑起来"。** 两者的痛点几乎不重叠，但补丁系列里有几条是纯工程修复，与你显卡无关地适用。

> 注意：HyperQwen 的 `docs/gotchas.md` 第 37 条与 `patches/marlin-repack-staged-sm80.patch` 头部**直接针对 CMP 170HX / GA100**，且引用了另一个 170HX 专用仓库 `ahnguyen17/cmp-170hx-vllm`。这是两个项目唯一的实质性交汇点。

---

## 二、移植可行性的决定性事实

1. **补丁全是纯 Python。** 92 个被触及的文件，扩展名统计：`py` × 92，**C++/CUDA/Cython 源文件 0 个**。
   → 移植**不需要重新编译任何扩展模块**，`_dev/bin/_build.ps1` 那 37 分钟的流程完全不涉及。
   （例外：Triton 内核靠运行期 JIT，本机已装 `triton_windows 3.6.0.post26`，需要 `cl.exe`/`ninja` 在 PATH 里——你已踩过同类坑并解决了。）

2. **0.29 树里补丁的目标路径全部存在。** 包括 `v1/worker/gpu/spec_decode/dflash2/`、`v1/attention/ops/`、
   `model_executor/kernels/linear/mixed_precision/marlin.py` 等。上游 0.29 已自带 DFlash2，缺的是 fork 的扩展。

3. **按系列顺序整体实打结果：23/39 完全干净，16 条有冲突，其中 4 条只卡在 `envs.py`。**
   单独打会大面积失败——因为每个补丁都往 `envs.py` 同一区域插 knob，上下文互相依赖。
   **必须按 `patches/series` 顺序、以 `-p1 --directory=vllm` 应用**（补丁是相对 `site-packages/vllm` 生成的）。

4. **Marlin 内核在你本机构建里是有的**：`_C_stable_libtorch.pyd` 中 grep 到 `gptq_marlin_repack`、
   `awq_marlin_repack`、`marlin_gemm`、`scaled_mm`。W4A8/W4A16 路径的前提成立。

---

## 三、逐条补丁分诊

### A 组 — 100% 干净应用（23 条）

| 补丁 | 对你的价值 | 说明 |
|---|---|---|
| **marlin-repack-staged-sm80** | ★★★ | **默认只为 compute capability == 8.0 打开**，170HX 上已验证：repack 窗口 44.9 s（CPU fallback 403.7 s），engine ready ~90 s（~540 s）。核心文件已干净落入 0.29 |
| **marlin-int8-layer-select** | ★★★ | 用正则挑哪些层跑 W4A8 Marlin；GA100 有 int8 tensor core，prefill 理论 +19~30% |
| **marlin-int8-negative-scales** | ★★★ | AutoRound 导出的负 group scale 被 Marlin 当无符号读，修数值错误 |
| **engine-completion-log** | ★★ | 每个请求一行无门控 INFO 收尾日志 |
| **engine-stall-sentinel** | ★★ | 引擎停止推进时告警一次，排查 hang |
| **no-output-token-reservation** | ★★ | 长 prompt + 大 `max_tokens` 不再被 400 拒；0.29 的 `renderers/params.py:210` 同构，已干净应用 |
| **vllm-pr50021-gdn-spec-bounds** | ★★ | GDN/KDA 投机解码越界检查（上游 PR，未合并） |
| qwen3_5-embed-quant | ★ | 跑 Qwen3.5/3.8 系才有用 |
| qwen3_5-mtp-draft-vocab | ★ | 同上，MTP 草稿头词表截断 |
| sampler-small-topk-fast-softmax | ★ | 小 k 的免排序 top-k/top-p + 多块行 softmax |
| hybrid-sw-block-promote | ★ | 滑窗层 block 对齐，省页 |
| offload-dflash-eagle-groups | ★ | 只在用 offloading connector + 投机时相关 |
| dflash2-ngram-chains / spec-decode-int4-kv-mq3d | — | DFlash2 / int4 KV 配套，见 D 组 |
| dflash2-z-adaptive-emitted / dspark-draft-quant-config | — | 投机解码配套 |
| marlin-tune-table | ★ | 挂本地编译的可调 Marlin 扩展，默认关 |
| spec-decode-scratch-token-units / within-budget | — | mq3d 配套 |
| mamba-chunked-prefill-align | ★ | Mamba/GDN chunked prefill 丢状态与 NaN |
| offload-wsl2-devptr | ✗ | WSL2 专属，原生 Windows 不适用 |
| offload-mtp-serve | ★ | offloading connector 在 MTP/EAGLE 下服务命中 |

### B 组 — 只卡 `envs.py`，手工挪 2 行（4 条）

`speed-knobs-envs`、`vision-tower-cpu-offload`、`int4-mq3d-envs`、`triton-spec-attn-fp8-kv`
→ 冲突都在类型注解块与 `os.environ.get` 登记表两处，把 knob 登记行插到 0.29 对应位置即可。

### C 组 — 调用点漂移，需看 1 个文件（8 条）

| 补丁 | 冲突文件 | 落点难度 |
|---|---|---|
| spec-decode-attn | `v1/attention/backends/flash_attn.py` | 中（FA2 split-KV verify，sm80 上理论可用） |
| prefill-attn-int8 | `v1/attention/backends/flash_attn.py` | 中（新文件 `prefill_attn_hd256.py` 242 行已干净落地） |
| int4-kv-per-token-head | `v1/worker/gpu/attn_utils.py` | 中（新文件 `int4_per_token_head.py` 1300 行已落地） |
| hybrid-kv-groups-v2-cudagraph | `v1/worker/gpu/model_runner.py` | 低——0.29 **已自己 profile V2 CUDA graph**，该 hunk 按 PATCHES.md 本就该退休 |
| spec-sampler-prewarm | `model_executor/warmup/kernel_warmup.py` | 低 |
| mamba-align-checkpoint-order | `v1/core/single_type_kv_cache_manager.py` | 低（上游 #45238 未合并） |
| mamba-align-retire-null-gaps | 同上 | 低——上游 #55450 已合并，**大概率该丢** |
| sse-keep-alive | 2 个 `api_router.py` | 建议**直接丢掉**——0.29 已有 `entrypoints/openai/sse_keep_alive.py` |

### D 组 — 需要重写或已上游（3 条）

- `xgrammar-spec-terminated`：0.29 的 `backend_xgrammar.py` 已有 `_is_terminated` 逻辑 → **丢**
- `vllm-pr54282-draft-gumbel-salt`：0.29 的 `gumbel.py` 已有 salt 偏移 → **丢**（9 文件冲突正是因为它已经在了）
- `dflash2-lookup-drafting`：冲突在 `cudagraph_utils.py`、`dflash/speculator.py`、`dflash2/speculator.py` → 需要真重写
- `dflash2-backport`：RETIRED，本来就跳过

---

## 四、对 CMP 170HX 的实际价值判断

**值得做的（省事、有直接收益）**

1. `marlin-repack-staged-sm80` —— 唯一一条为 sm80 精确默认开启、且已在 170HX 上回归过的补丁。
   收益是**加载时间**（44.9 s vs 403.7 s 的 repack 窗口），代价是常驻最多 ~1.2 GiB 暂存缓冲，对 64 GB 卡无所谓。
   （补丁头部明确声明：Xid-31 的真因是解锁驱动把 WPR2/GSP 保留区注册进了页分配器，**不是** vLLM 的 bug；
   驱动修好后 stock 路径也能干净启动。所以它的价值是分配卫生 + 加载提速，不是"修 bug"。）
2. `marlin-int8-*` 三条 + `VLLM_MARLIN_INPUT_DTYPE=int8` —— prefill 是 24 GB 卡上最缺的，你这边 64 GB 也不嫌快。
3. 三条服务端质量补丁（completion-log / stall-sentinel / no-output-token-reservation）——与硬件无关，纯粹是工程改进。

**需要先验证前提的**

- `prefill-attn-int8`：几何门控写死 `num_heads==24 && num_kv_heads==4 && head_size==256`，
  非该几何自动回退 FA2。要泛化才能用在你跑的模型上，且依赖 Windows 上的 Triton JIT。
- `spec-decode-attn` / `spec-decode-int8-kv`：投机解码相关。**风险点**：HyperQwen README 明确写着
  sm80（A100/A30/CMP 170HX）在**任意 k 上都有一个开放中的投机解码 fault**（issue #98、#72），
  两位 sm80 用户正在 bisect。在你的卡上开投机解码要先接受这个不确定性。

**不要碰的**

- `dflash2-*` 全家族：drafter 是**按模型单独准备的 checkpoint**，且同样落在上面那个 sm80 fault 里。
- KVarN、`int4-kv-per-token-head`、`vision-tower-cpu-offload`、`offload-*`：
  全部是"24 GB 塞不下"逼出来的省显存手段，你有 64 GB，收益接近零。
- `triton-spec-attn-fp8-kv`：**sm89+ 专属**，GA100 没有 FP8，Triton 会在 SM89 以下直接拒绝 fp8 KV。
- `offload-wsl2-devptr`：WSL2 的 UVA/device pointer 行为，原生 Windows 不是同一回事。

---

## 五、落地方式（重要）

补丁改的是 `vllm/*.py`，这些文件**会被打进 wheel**。所以：

- 不要只在 `site-packages` 里改——下次 `_dev/bin/_build.ps1` 重编会被覆盖。
- 应当照现有 `fix_cutlass_msvc.py` / `_dev/patches/fix_flash_attn_msvc.py` 的成例，
  写一个**幂等补丁脚本**，在构建流程里调用，这样 `.deps` 或 wheel 重建后修改不会丢。
- 应用命令固定为：`git apply -p1 --directory=vllm <patch>`，**必须按 series 顺序**。
- 不建议手工编辑补丁文件本身（上游 `scripts/export-patch.sh` 生成的，行号是原树行号）。

已把 40 个补丁原样复制到 `_dev/patches/wsl/` 作为移植工作区。

---

## 六、结论

**技术上完全可移植，且成本比预期低**：零 C++/CUDA 改动、零重新编译、23 条即插即用。

**但价值上只需要其中一小部分**：两个仓库解决的是相反的问题（省显存 vs 平台兼容），
真正对你这块 170HX 有意义的是 **Marlin/sm80 三条 + 服务端质量三条**，其余大部分是为 24 GB 设计的，
在你 64 GB 的卡上要么无用、要么前提不成立（sm80 无 FP8、sm80 投机解码有开放 fault）。

建议先做 A 组里那 6 条，验证方式是跑一个 W4A16 量化模型对比：加载耗时、Xid 计数、prefill 吞吐。

# TODO — 从 `Morrowmake/vllm-cmp170hx` 引入的改动

分析依据：`_dev/docs/FORK_CMP170HX_IMPROVEMENTS.md`。
上游 commit 已抽取为 `_dev/fork_cmp170hx/patches/*.patch`；`patches/` 与
`fork_commits.txt` 都是生成物、已 gitignore，可从 `morrow` remote 重新导出。

运行环境：生产运行时是 `.flashnext-root/.venv`（junction → `.venv-flashnext`），
**非 editable 安装**，所以 Python 改动要么走 `_dev/bin/_setup_flashnext.ps1 -Execute`
全量重建，要么用 `_dev/fork_cmp170hx/deploy_runtime.py` 把改动的 `.py` 直接部署进
site-packages（会留 `.orig` 回滚副本）。服务停止后才能部署。

测试机：GPU1 空闲。**GPU0 上跑着我们的生产服务**（`:9393`，2026-10-03 21:32 起的，
uncensored 权重，占 ~63.5 GiB / 47% util）——它就是基线，别停。
验证一律用第二个实例：`start_qwen38_flash_next.ps1 -Unc -Gpu 1 -Port 9394`。
**当前只有 `D:\models\Qwen3.8-Flash-Next-Uncensored-AutoRound-3bpw-MTP` 可用**
（默认那个 AutoRound-3bpw-MTP 已被删除）。
deploy 时用 `--port 9394` 绕开脚本对 `:9393` 的存活检查（那个进程已经加载过代码，
改 site-packages 不影响它）。

状态图例：`[ ]` 未开始 · `[~]` 进行中 · `[x]` 已落地并验证 · `[-]` 决定不做

---

## A. Tier 1：小而干净，`vllm/` 侧 `git apply -3` 无冲突

| # | 状态 | 上游 commit | 改动 | 我们的收益 | 验证方式 |
| --- | --- | --- | --- | --- | --- |
| A1 | `[x]` | `d5051abaf1` (#58368) | `MambaManager`：prompt 尾部 hash 边界改用 `drop_eagle_checkpoint_block` | MTP 下恢复 prompt 尾部前缀命中，最多省 1 个 hash 单元 = **1600 token ≈ 0.6 s TTFT** | `Prefix cache hit rate` 行 + 重复/续写请求的 TTFT |
| A2 | `[x]` | `48d8880d09` (#58489) | `Qwen4ExpNGramEmbedding`：常驻 `_prefetch_ids`，别把 PLE 预取 id 留在 CUDA graph pool | 消除侧流 PLE 查表读到被后续图段复用的 storage → 错误 PLE 行/垃圾 token | 已跟踪文件，`_patch_vllm_windows_runtime.py --check`；GPU1 boot 后跑重复请求看输出一致 |
| A3 | `[x]` | `62a08ef13b` | Gumbel / rejection-sampler local argmax / residual resample 的输出 id 钳到词表 | 稳健性：`vocab_size=248,320` 非 block 宽度整数倍 | `tests/v1/worker/test_gumbel_vocab_clamp.py`（CPU） |
| A4 | `[x]` | `e6c07ea576` (#51694) | 增量多模态 block 哈希修正 | vision 默认开 + prefix caching：图片 prompt 的命中正确性（既防漏命中也防错命中） | `tests/v1/core/test_kv_cache_utils.py`；GPU1 上跑带图请求两次比对 |
| A5 | `[x]` | `f6aa2919bc` (#58288) | partial-block KV event 带上所有多模态特征 | 同上（KV event / connector 路径） | `tests/v1/core/prefix_cache/test_partial_prefix_cache_primitives.py` |
| A6 | `[x]` | `70fc359d25` (#57453) | KV offload 事件元数据跨 batch 翻译保留 | 备用保险（64 GB 上 KV 全驻留，暂未开 Windows KV offload） | `tests/v1/kv_connector/unit/offloading_connector/test_events.py` |
| A7 | `[x]` | `6491f481a7` (#58430) | `Worker._scoped_allocator_max_split` 重写：只在 native allocator 生效、保留并原样恢复其余 `PYTORCH_CUDA_ALLOC_CONF` | KV 自动档（`KvGiB 0`）下 profile 峰值不再被钉住的 segment 抬高 → KV 池更大；同时我们显式设的 `expandable_segments:False` 之前被两次重置抹掉 | A/B `Available KV cache memory` / `GPU KV cache size` 行 |
| A8 | `[x]` | `71891bdb94` (#58582) | Triton JIT 启动 warmup 用 `ThreadPoolExecutor` 并行编译（`VLLM_TRITON_JIT_WARMUP_NUM_THREADS`，默认 4） | 冷 compile cache 的那次 boot（~90 s 编译的一段） | boot 时间对比（冷 cache） |

## B. Tier 2：功能类，需要解决冲突或自建数据

| # | 状态 | 上游 commit | 改动 | 我们的收益 | 验证方式 |
| --- | --- | --- | --- | --- | --- |
| B1 | `[x]` | `d26e7d2548` | **fair chunked prefill**：`--prefill-chunk-with-decodes N`、`--max-num-partial-prefills N`（默认 0 = 与上游逐字相同） | 实测见 Log：基线是 1600-token 对齐的 prefill chunk 独占整步，6.2 s 的 prefill 里有 ~16 次 300–550 ms 空档；cap 1024 摊平成 ~200 ms（>300 ms 空档 15.7 → 2.3，p99 367 → 296 ms），插队者 TTFT +35%；cap 384 太碎（TTFT ×2.8） | `tests/v1/core/test_fair_prefill.py` + GPU1 三 arm A/B（`probe_itl.py`） |
| B2 | `[x]` | `ac877d6ad7` | `VLLM_GLM5_MEM_ATTRIBUTION`：启动显存按模块/profile 阶段归因 | 排查 46.53 GiB 权重 + PLE + 图 + workspace 的分布，服务 0.96 上限与 KV 自动预算 | boot 日志里的归因表 |
| B3 | `[x]` | `63dac5ab8f` | hybrid KV：`speculative_scratch_bytes` 按 `max_num_seqs` 记一次而非按并发槽 | **262K 上下文下是 no-op**（引擎报 `Maximum concurrency ... 1.89x` < 4，新分支不触发）。只有把 `max_model_len` 调短（如 71,680 → ~7x）才有 ~10–15% 容量收益 | `tests/v1/core/test_kv_cache_utils.py`；若做，用短 MaxLen 的 arm 验证 |
| B4 | `[x]` | `d508456f85`, `abe9524dba` | `VLLM_KV_MAMBA_INFLIGHT_STATES`、`VLLM_KV_SWA_INFLIGHT_SCRATCH`（默认 off） | 我们开了 async scheduling，in-flight prefill chunk 的 state 预留更准；opt-in、一行日志 | 同上，flag 打开前后对比容量行 |
| B5 | `[~]` | `031d63da62`, `9210eca1e6` | `VLLM_GLM5_FLA_PIN_AUTOTUNE`：`PinnedAutotuner` + 按 key 的配置表 | **跨 boot 输出稳定性**：我们的 GDN 算子 `key=["H","K","V","BT"]` 不含 seqlen，首个形状赢者通吃，候选项改 BV/warps/stages = 改归约顺序；顺带省掉首次 prefill 的 autotune benchmark。表要按我们形状（H=16/48、K=V=128、BT=64）在本卡重扫 | 两次 boot 同一 prompt 的 logprob 逐字对比 + prefill tok/s |
| B6 | `[x]` | `46cc42f176` | `VLLM_GLM5_MOE_MASK_PADDING`：CUDA-graph padding 行路由到"无 expert" | 可重复性：真实 batch padding 到 6/8/12/16/24 时 stale padding 行占 expert slot、改 grouped-GEMM 分块与累加顺序 → logprob 抖动 | 同一请求 padding batch 下重复多次，比对 logprob |
| B7 | `[x]` | `3ddd38cf3e`, `638709c3d6` | `moe_align_block_size` 确定性 expert 内顺序 + counting sort 排名 | 同上（确定性），第二个 commit 把开销还回来 | MoE 单测 + decode tok/s 不回退 |
| B8 | `[x]` | `d945300e8d` | `adaptive_k`（`--speculative-config`）：按接受率逐步选验证宽度 | 收益不确定：我们 tok/step ~1.9（k=2–3），省的主要是 attention/sampling 开销；代价是每个 k 一张 uniform decode graph（capture sizes 要覆盖 k 区间） | 先测固定 k=3 与 adaptive 的 tok/s/TPOT 再决定 |
| B9 | `[~]` | `561bfdcccd`, `04730e8270` | DFlash：边界复用已定稿 context、context K/V 预计算进 draft graph、draft confidence 日志 | **不属于 flash-next**（我们用 MTP drafter）；收益在 `Qwen3.8-27B` + DFlash2 那个服务 | 归到 27B 服务那边验证 |

## C. 明确不做

* TP/PP 与 all-reduce 一整套（`0ec2a7af3d`、`61f9ac9328`、`f671f1932d`、`997b6ea14a`、PP 系列）——TP=1 时这台机器上根本没有 all-reduce，恒等于 0；且实现是 Linux `/dev/shm`。
* sparse-MLA / DSA indexer kernels、`TRITON_MLA_SPARSE`、GLM 的 mHC/KDA/router/prologue/thin-GEMM 融合 kernel——形状与模型族都不对。
* 编译版 Ampere Marlin（`160db6239f`）——门槛是 288 experts/top-8/hidden 4096/uint4b8 group-128，我们是 512 experts/top-10/hidden 2560/AutoRound group-64；还要 MSVC+nvcc 单独构建。
* Engram huge-page（`6936e77ba4`）——Linux `MADV_HUGEPAGE`，且目标是 `deepseek_v41`。
* `6a0dd75b52`（PP deferred block tables）——我们不用 PP。

---

## 执行与记录约定

1. 一次一个 commit：`git apply -3 <patch>` → 解决冲突 → 过 lint → 跑对应的 CPU 单测 →
   单独提交（commit message 带 `(cherry picked from commit <sha>)` 与来源说明）。
2. 每落地一项就更新本文件的 `状态` 列，并在下面的 **Log** 段写一行结论（命令 + 观察到的数字）。
3. GPU1 验证集中在有意义的节点：A1+A2+A7 落地后 boot 一次看 TTFT/hit rate/容量行；
   B1 落地后专门做一次 C=2 长 prompt 插队的 ITL A/B；B5 需要两次 boot 的逐字对比。
4. 部署 Python 改动：`python _dev/fork_cmp170hx/deploy_runtime.py <repo-relative paths>`，
   已跟踪的 13 个文件仍走 `_dev/bin/_patch_vllm_windows_runtime.py`。

## 开发环境注记（本次建立）

* 源码树原本是**不可导入**的（缺编译扩展），而生产运行时是 site-packages 里的非 editable
  安装，且 `tests/conftest.py` 会把仓库根塞进 `sys.path` → 树会遮蔽安装包。
  为了让 `pytest` 直接跑到我改的源码，把已编译产物从 site-packages 复制进了源码树：
  `vllm/{_C_stable_libtorch,_moe_C_stable_libtorch,cumem_allocator,fs_io_C,spinloop}.pyd`、
  `vllm/_version.py`、`vllm/vllm_flash_attn/_vllm_fa2_C.pyd`，
  并把 `vllm/*.pyd` 写进了 `.git/info/exclude`（不污染 status）。
* 给 `.venv-flashnext` 装了测试依赖：`pytest==9.1.1`、`tblib==3.2.2`（生产 venv 之前没有）。
* lint 走 `.venv-maintenance/Scripts/pre-commit.exe`（ruff / mypy / SPDX 等钩子齐全）。

## Log

（下面按时间追加，一行一条：做了什么 / 怎么验的 / 数字）

* **A1 落地** `f1ad606eb9`（cherry-pick `d5051abaf1`, 上游 #58368）：
  `MambaManager` 的 prompt 尾部 hash junction 判断从 `self.use_eagle` 改成
  `self.drop_eagle_checkpoint_block`。验证：
  `.venv-flashnext/Scripts/python.exe -m pytest tests/v1/core/prefix_cache/test_mamba_eagle_resume_checkpoint.py -q`
  → **13 passed**（含新加的 `test_mamba_prefix_cache_drops_hash_block` 两个参数化）。
  pre-commit 全套钩子通过。GPU 上的收益（TTFT / hit rate）留到后面一起 boot 验。
* **A2 落地** `f97183cc7e`（cherry-pick `48d8880d09`, 上游 #58489）：
  `Qwen4ExpNGramEmbedding` 给 PLE 预取 id 建了一个常驻 buffer。
  **偏差修正**：上游写法 `torch.empty(max_total_tokens, ngram_heads, dtype=long)`
  不带 device，而我们的模型是在 **meta device** 上构的（boot 日志
  `weight_device=meta`）、`_prefetch_ids` 又是普通属性（`module.to()` 不会搬它），
  实测会落在 CPU 上 → Triton 报 `Pointer argument (at 6) cannot be accessed from
  Triton (cpu tensor?)`。所以我们用 `get_current_vllm_config().device_config.device`
  显式放到运行设备上。
  顺手把 `additional_config.get("ple_ssd_offload")` 改成 config 自己用的
  `isinstance(..., dict)` 写法（mypy-3.10/3.12 在该文件上的错误是这行来的，
  已确认 HEAD 上也报错 = 预存在）。
  验证：`pytest tests/models/qwen4_exp/test_ple.py -q` → **56 passed / 4 skipped / 1 failed**，
  失败的是 `test_ssd_prompt_read_ahead_preserves_context_and_cancels`，
  已用 stash 回到 HEAD 复跑确认 **预存在**（与本次改动无关）。
  另外写了个直接探针：用新的 MemPool 在下一次同尺寸分配后比对预取 id，
  `survives pool reuse: True`、ids 落在 `cuda:0`。
  pre-commit（含 mypy）全绿。该文件是 13 个跟踪运行时文件之一，
  下次重建时由 `_patch_vllm_windows_runtime.py` 部署。
* **A3 落地** `31c4e591b4`（fork 自有 `62a08ef13b`，非上游 PR）：
  `gumbel.py` / `rejection_sampler_utils.py` 的块内 argmax 以及 residual resample
  的输出 id 一律钳到 `vocab_size-1`。我们的 `vocab_size=248,320` 不是 n-gram
  `BLOCK_SIZE=32` 的整数倍，最后一个 head 的块会越界到不存在的 token id。
  验证：`pytest tests/v1/worker/test_gumbel_vocab_clamp.py -q` → **15 passed**。
* **A4 落地** `a0a5faa6da`（cherry-pick `e6c07ea576`, 上游 #51694）：
  `kv_cache_utils.py` 增量多模态 block 哈希不再取"最后一个 mm feature"
  （`curr_mm_idx = -1`），改用 `get_mm_features_in_window` 取落在被哈希窗口里的
  特征。vision 默认开 + prefix caching 都开着，所以这条路径在跑。
  验证：`pytest tests/v1/core/test_kv_cache_utils.py -q` → **118 passed / 2 failed**；
  两个失败（`..._mamba_hybrid_sharing_pp_group_count_bump` /
  `..._pp_starved_stage`）用 stash 回到 HEAD 复跑确认 **预存在**。
* **A5 落地** `f9b51b318d`（cherry-pick `f6aa2919bc`, 上游 #58288）：
  `block_pool.py` 的 partial-block KV event 用同一套窗口选择逻辑。
  验证：`pytest tests/v1/core/prefix_cache/test_partial_prefix_cache_primitives.py -q`
  → **15 passed**。
* **A6 落地** `34b1229a32`（cherry-pick `70fc359d25`, 上游 #57453）：
  offload event tracker 在同一 batch 里"先删后存"时保留 payload，batch 翻译完了
  再把 `active_residencies` 为空的 key 丢掉。我们自己没开 Windows KV offload，
  这条属于备用保险。验证：
  `pytest tests/v1/kv_connector/unit/offloading_connector/test_events.py -q`
  → **25 passed**。
* **A7 落地** `347a092499`（cherry-pick `6491f481a7`, 上游 #58430）：
  `_scoped_allocator_max_split` 现在只在 native allocator 上动作，且读出当前
  allocator settings、把 `max_split_size_mb` 追加上去、退出时**原样恢复**。
  旧实现进去时只传 `max_split_size_mb:20`（setter 会把其他项重置掉），出来时只
  恢复 `max_split_size_mb`，于是我们在 `start_qwen38_flash_next.ps1` 里设的
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` 被静默抹掉、
  expandable segments 被打开且再也不会关回去。
  验证：`pytest tests/v1/worker/test_gpu_worker.py -q` → **18 passed**（含 3 个新测试，
  其中一个直接断言 settings 跨成功/失败的 profile 保住了）。
  冲突处理：`tests/v1/worker/test_gpu_worker.py` 用 `-3` 合时有 1 处冲突，
  只保留新增的 3 个测试，丢掉上游的
  `test_load_model_preserves_compiled_graphs_at_runtime`（我们的 `gpu_worker` 没有
  `set_torch_threads_for_runtime` 可 monkeypatch），并补上 `import torch`。
* **A8 落地** `69e3ec7cb9`（cherry-pick `71891bdb94`, 上游 #58582）：
  Triton launcher warmup 走线程池，`VLLM_TRITON_JIT_WARMUP_NUM_THREADS`（默认 4，
  设 1 = 串行），并从 `compile_factors()` 的 ignored 集里排除该变量以免失效编译缓存。
  验证：`pytest tests/model_executor/test_jit_warmup_triton_launcher.py -q` →
  **15 passed**。
  **附带的预存在修复**：`vllm/envs.py` 的 `DEFAULT_MP_METHOD` 之前是
  `"fork" if platform.system() != "Windows" else "spawn"`，mypy 推成 `str`，
  于是任何碰 `envs.py` 的提交都会被 `mypy-3.10` 钩子挡下（HEAD 上同样报错，
  已确认）。加了 `Literal["fork", "spawn"]` 注解后钩子全绿。

### Tier 1 GPU1 验证（一次性 boot，端口 9394）

部署：`_patch_vllm_windows_runtime.py`（13 个跟踪文件）+ `deploy_runtime.py --port 9394
<git diff --name-only 4faed433c HEAD -- vllm/>` → 12 个文件进 site-packages。

`start_qwen38_flash_next.ps1 -Unc -Gpu 1 -Port 9394`（日志
`_dev/out/logs/qwen38-flash-next-win_20261005_112104.stdout.log`）：

| 指标 | 基线（GPU0 `:9393`，10-03 21:32，改动前） | 本次（GPU1，Tier 1） | 说明 |
| --- | --- | --- | --- |
| Available KV cache memory | 12.26 GiB | 12.22 GiB | 自动档；A7 没把 KV 抬高（噪声级） |
| GPU KV cache size | 423,681 tok / 1.62x | 422,264 tok / 1.61x | −0.34%，来自 peak activation 0.53→0.57 GiB |
| consumed / peak / graph | 48.28 / 0.53 GiB | 48.28 / 0.57 / 0.15 GiB | 权重与非-torch 完全一致 |
| 权重加载 | 171.4 + 12.1 = 207 s | 189.6 + 16.1 = 233 s | 两个实例读同一 47 GiB checkpoint、同一块 PLE SSD；磁盘/RAM 争用，非代码回退 |
| init engine | 18.73 s | 24.44 s | 同上争用 |

行为探针（`_dev/fork_cmp170hx/probe_service.py`，GPU1 空载）：

* `Reply with exactly: PING-7341` → 精确 `PING-7341`，无乱码（A2 PLE buffer、A3 钳位 OK）。
* 长 prompt（~2700 tok）重复 4 次：`2.41 s` → `0.57 / 0.58 / 0.59 s`，四次输出逐字相同。
  同一 prompt 打到基线 `:9393`：`2.92 s` → `0.60 / 0.61 / 0.61 s`（GPU0 当时 47% util）。
  前缀命中路径两边一致；A1 的收益在 hash 单元的 checkpoint 边界上，这个 prompt 只有一个
  满 block，看不出来（单测已覆盖）。
* 图片 prompt 两次：`0.194 / 0.193 s`，输出逐字相同，`MM cache hit rate: 50.0%`（A4/A5 OK）。
* boot 期间 `Prefix cache hit rate: 28.6% → 32.5%`。
* 停止：`stop_vllm.ps1 -Port 9394` 只杀 9394 的 3 个进程，GPU0 服务未动，GPU1 回到 2 MiB。

**A7 的行为变化要记住**：旧实现进 profile 窗口时把 `PYTORCH_CUDA_ALLOC_CONF` 重置成只剩
`max_split_size_mb:20`，退出时又只恢复 `max_split_size_mb`，于是启动器设的
`expandable_segments:False` 被抹掉、expandable segments 被打开且再也不会关回去。GPU0 那个
从 10-03 跑到现在的服务实际是 expandable segments **开着** 的；新代码之后是 **关着**。
容量行没变，但长期碎片 / `alloc-heal` 的行为会不同，GPU0 服务下次重启会切到新行为。

### B1 落地 + GPU1 A/B（fair chunked prefill）

先落地两个前置（我们的树缺，patch 是写在它们之上的）：
`b892c01e64` = `c64b15cde5`（上游 #57951，threshold 变成 `schedule()` 里的局部变量，
并且 waiting 循环也开始用它）；`f4085c38c7` = `6697a7dd30`（上游 #58459，adaptive floor）。
然后 `abc1f10f8b` = `d26e7d2548`（fair prefill），`3d4cdf149d`（启动器 knob）。
patch 里作为 context 带进来的 `_select_adaptive_k` 已丢掉（那是 B8 的事）。

测试：`tests/v1/core/test_fair_prefill.py` **29 passed**；
`tests/v1/core/test_scheduler.py` **177 passed**（两个 GPU 都可见时全绿；只给一个 GPU 时
`test_async_scheduling_pp_*` 会因为 "World size (2) > available GPUs (1)" 失败 = 环境问题）。

A/B（GPU1 `:9394`，uncensored，`max_num_batched_tokens=2048`，MTP k=3，C=2：
victim 解码 1200 token，1 s 后插一条**冷**的 26,913-token prefill，3 次重复，
`_dev/fork_cmp170hx/probe_itl.py`）：

| arm | 配置 | victim 中位 | p90 | p99 | max | >100 ms 步 | >300 ms 步 | 插队者的 TTFT |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 默认（fair off） | 21.1 ms | 23.1 ms | **367 ms** | 444 ms | 17.7 | **15.7** | **6.20 s** (~4340 tok/s) |
| B | cap 384 + partial 2 | 22.0 ms | 210 ms | 239 ms | 329 ms | 84 | 1.0 | 17.3 s (~1560 tok/s) |
| C | cap 1024（不切分） | 21.7 ms | 192 ms | 296 ms | 364 ms | 34 | 2.3 | 8.4 s (~3200 tok/s) |

机制（和先前文档里写的"~0.8 s 步长 / ITL 800 ms"不一样，已按实测纠正）：
mamba 的 state/hash block 是 **1600 token**，所以 prefill chunk 被对齐切成 1600，
一个整步就被它独占（~380–550 ms），decode 只能在**下一个**步里走 —— 于是基线的形状是
"380 ms 空档 + 21 ms 空档"交替，6.2 s 的 prefill 里出现约 16 次 300 ms+ 空档。
fair prefill 把这些长空档摊平成 ~200 ms 的一致步长，代价是插队者的 TTFT。

结论/建议：**用 `-PrefillChunkWithDecodes 1024`、`-MaxNumPartialPrefills 0`**。
300 ms+ 空档 15.7 → 2.3（6.8x），p99 367 → 296 ms，插队者 TTFT 只 +35%。
cap 384 把 1600-token block 切成 384（还再被 partial-prefills 二分成 192/步），
prefill 吞吐从 ~4340 掉到 ~1560 tok/s，TTFT 翻 2.8 倍，除非"短 prompt 排在 30K prompt 后面"
这个场景真的出现，否则不划算。C=1（没有并发的 decode）时两个 knob 都不生效，
单条 prompt 的 prefill 行为逐字不变。

### B2 落地 `f06b8fe0e3`（cherry-pick `ac877d6ad7`）

`VLLM_MEM_ATTRIBUTION=1`（默认 0）在启动时按模块组打 resident bytes、按 profile
阶段打峰值、打 KV 预算项、打 KV 定容之后 attention metadata builder 的分配。
纯日志，KV 容量不变。**改名**：fork 用的是 `VLLM_GLM5_MEM_ATTRIBUTION`，我们的树没有
GLM5 env 家族，而且这功能是模型无关的 → `VLLM_MEM_ATTRIBUTION`。
`envs.py` 的两处冲突就是 fork 那一整块 GLM5 当 context 进来了，只留了 attribution 一条。
顺手给 `Worker._timing_cm` 加了 `Any` 注解（`model_runner.py` 里一个**预存在**的
mypy-3.10 错误，不然这次提交被钩子挡住）。
测试 `tests/v1/worker/test_mem_attribution.py` → **4 passed**。
启动时验证留到下一次 boot（`VLLM_MEM_ATTRIBUTION=1`）。

### B5 落地 `952a7d8b41` + `b122887f28`（cherry-pick `031d63da62`, `9210eca1e6`）

`VLLM_FLA_PIN_AUTOTUNE=1`（默认 0）把 vendored fla 算子里所有 `@triton.autotune`
的 kernel 换成 `PinnedAutotuner`：每个 autotune key 只认表里的一个 config，
于是 boot 不再 benchmark，赢的那个 config（也就是归约顺序）是形状纯函数。
第二个 commit 把 lookup 扩成 `(kernel, key, H, T bucket)` —— key 里没有 token 数和
（部分 kernel 没有）head 数，而最快的 config 取决于这两者，单 config 在短 launch 上
kernel time 差到 17%。
**为什么对我们重要**：我们的 GDN 算子 `key=["H","K","V","BT"]` 不含 seqlen，第一个
撞上某个 key 的形状就赢者通吃整个进程，候选项差在 `BV`/`num_warps`/`num_stages`
= 差在归约顺序 → 两次 boot 同一 prompt 的 logprob 可能不同。
测试 `tests/kernels/test_fla_autotune_pin.py` → **12 passed**。
两处我们自己造的坑：fork 的测试用默认 locale 读 kernel 源码（这里是 GBK，读到
0x92 就炸）→ 显式 `encoding="utf-8"`；直接 `import triton`/`import re`（本仓库钩子禁）
→ 走 `vllm.triton_utils` / `regex`。
**状态 `[~]`**：机制 + fork 的表落地了，但那是他们 `heads [16, 64]` 的测量（表里
`PROVISIONAL` 只对没测的 kernel 生效），我们的 GDN 跑 `H=16` 和 `H=48`，`H=48` 现在
落 `DEFAULTS`。要真开这个 flag，得先按我们形状在本卡重扫一遍表。

### B6 落地 `aae54d6945`（cherry-pick `46cc42f176`）

`VLLM_MOE_MASK_PADDING=1`（默认 0）：CUDA-graph padding 行的 `topk_ids` 在进 fused MoE
前置 `-1`，alignment 直接丢掉它们。那些行的 hidden state 来自 stale buffer，它们落在
哪个 expert block 会改变旁边真实行的 grouped-GEMM 分块与累加顺序 → 同一请求
last-bit 不同的 MoE 输出。我们的 decode batch 被 capture sizes（6/8/12/16/24）
频繁 padding，所以这是除 autotune 顺序之外 run-to-run logprob 抖动的主要来源。
测试 `tests/kernels/moe/test_moe_mask_padding.py` → **12 passed**。
**丢掉** fork 的 sm_80 fused-decode 豁免（`_MASK_MIN_ROWS` 来自
`VLLM_GLM5_DECODE_KERNELS`/`VLLM_GLM5_DECODE_MOE_MAX_TOKENS`，我们没这两个 env）；
原来那条"小 batch 跳过"的测试改成"任何 padding 尺寸都遮"。
`458678cbf8` 要动 `vllm/ampere_decode/`（我们没这个包）→ 不做。
`d40c8df8bf` 是同一特性的测试/实现跟进，等 B7 一起看。

### B7 落地 `c853729ee0` + `f0953dcf43`（cherry-pick `3ddd38cf3e`, `638709c3d6`）

`VLLM_DETERMINISTIC_MOE_ALIGN`（现在是 int：`0` = 上游 CUDA op 默认，`1` = 确定性
Triton counting-sort kernel，`2` = 确定性 torch 路径，留着做 A/B）。
CUDA 的 `moe_align_block_size` 只在 `numel < 1024` 且 `num_experts <= 64` 时走单 kernel；
我们 512 experts，所以每次都走两 kernel，第二个 kernel 用全局 `atomicAdd` 排名 →
expert 段内顺序 = 线程到达顺序，fused MoE 按那个顺序累加每个 expert → 两次相同调用
低位不同。这条路径改成按扁平 routed-row 下标升序排名，顺序有定义。
`638709c3d6` 用分块 counting sort 替掉 argsort：比较次数从 `O(numel^2)` 降到
`O(numel * BLOCK_C)`。
**本卡实测**（GPU1，`numel=92160` = 9216 token chunk × top-10，E=512，block=16，每次调用）：
上游 `0.087 ms`、counting sort `0.197 ms`、torch argsort `0.679 ms`；
48 层 MoE 下 prefill chunk 级代价 ~9.5 ms（kernel）vs ~28 ms（argsort）。
测试 `tests/kernels/test_deterministic_moe_align.py` → **77 passed**。
丢掉的：`@overload` 对 + `scatter_idx` 返回（我们的 `ops.moe_align_block_size` 只有 7
个参数、没有 scatter_idx），以及他们 `ampere_decode`/`ampere_thin_gemm` 的 warmup
调用（我们没这两个包）。钩子相关：`with torch.cuda.device` 被本仓库禁 → 去掉（worker
的 current device 已经等于 tensor 的 device）；测试里的 `torch.cuda.synchronize` →
`torch.accelerator.synchronize`；`kernel_warmup.py` 里两处**预存在**的 E501/E722
（133 字符 TODO、bare except）被顺手修掉，否则这次提交过不了钩子。

**修一个我自己造的回归**：`aae54d6945`（B6）解 `envs.py` 冲突时取了 fork 那一整块，
把 `VLLM_MEM_ATTRIBUTION` 和 `VLLM_FLA_PIN_AUTOTUNE` 一起冲掉了。`c853729ee0` 里三条
都补回来了，现在 `envs.py` 四条 env 齐全（已核对）。

### B3 落地 `359413db3f`（cherry-pick `63dac5ab8f`）

`KVCacheSpec.speculative_scratch_bytes()`（默认 0，MambaSpec 返回 speculative
blocks 那部分）+ `_pool_concurrency_limit()`：先按 `max_num_seqs` 一次性留出
scratch，剩下的再按"每个驻留请求的净成本"除。`all_running <= max_num_seqs`
时两条分支一致 = 老公式。
**本卡 no-op**：基线日志 `Maximum concurrency for 262,144 tokens per request:
1.62x`（Tier 1 后 1.89x 那次是另一份日志），`max_num_seqs=4` → 走老分支。
收益要短 `MaxLen` 才看得见。
**为什么还是做**：B4 那两个记账改动全挂在 `_pool_concurrency_limit` 上，
不做 B3 就没法把两处记账做成一致的。
测试 `tests/v1/core/test_kv_cache_utils.py` 里 5 条新用例（scratch/concurrency）
→ **5 passed**；整文件 **159 passed**，只剩 2 条预存在的 PP 失败。
丢掉的 fork 用例：`test_trailing_layer_fallback_requires_exact_partition`（依赖我们
没有的 `_qwen3_5_hybrid_specs`），以及他们加在
`test_deepseek_v4_annotation_requires_model_type` 里的 `caplog_vllm` 断言（我们的
annotator 那条路径不打这条 warning，断言在我们这必然失败）。

### B4 落地 `64da3f3aba` + `2923ee6d25`（cherry-pick `d508456f85`, `abe9524dba`）

两条都默认 off，都只动记账/容量报告，不动 block pool、不动 admission cap。

**B4-1 `VLLM_KV_MAMBA_INFLIGHT_STATES`（对我们是真问题）**：align 模式下 prefill
chunk 把 end state 写进"最后一个 token 所在 block"，而给最老那个 in-flight chunk
播种的 block 要等 processed-token 越过它才释放。同步调度下已预留的 2 个 block
（seed + new state）刚好够；我们的服务开 `--async-scheduling`
（`_flashnext_struct_serve.ps1:315`）→ `max_concurrent_batches=2`，一个 prefilling
请求可能在每个 in-flight batch 里各有一个 chunk，于是多持有 1 个 state block
（每组）。欠预留的后果是**长 prefill 中途 block pool 耗尽**，不是报告里一个错数字。
代价（估算，待 boot 实测）：GDN state = 48×128×128 fp32 ≈ 3.0 MiB/层 + 3×10240
bf16 conv ≈ 60 KiB/层，每组 12 层、4 个 align 组（日志 group sizes
`[1600, 1600, 1600, 1600, 8, 1600]`）→ 一页 ~37 MiB；多 1 页/组/请求 = 4×37 MiB，
`max_num_seqs=4` 时 ~590 MiB ≈ 12.26 GiB KV 预算的 ~4%。
测试 `tests/v1/core/test_mamba_inflight_states.py` → **27 passed**。

**B4-2 `VLLM_KV_SWA_INFLIGHT_SCRATCH`（flash-next 上 inert）**：我们模型没有
sliding-attention 层 → 建不出 `SlidingWindowSpec`，hook 恒返回 0。但对
`Qwen3.8-27B` + DFlash2 那个服务是活的（draft config `sliding_window=2048`、
`layer_types=['sliding_attention']`），所以值得落而不是跳过。带 KV connector 时
不启用（异步 KV load / 延迟 free 会把 block 挂在非 running 请求上）。
测试 `tests/v1/core/test_swa_inflight_scratch.py` → **14 passed**。
新文件用 `git apply --include=...` 单独打（`-3` 对新文件会报
`does not exist in index`）。他们的测试要 `list[...]` 注解才能过 mypy。

**envs.py 老坑（第三次）**：fork 把整块 `VLLM_GLM5_*` 当 context 拖进来，`-3`
冲突时取 theirs 会把我们的条目冲掉。这轮 B3/B4 两次都手动只留了新增条目，并顺手
补上我们缺的 `VLLM_MEM_ATTRIBUTION` TYPE_CHECKING 注解。现在 6 条 env 齐全
（`environment_variables` 316 个 key，已核对）。

### GPU1 boot 验证（`qwen38-flash-next-win_20261005_140519` / `_141411`）

把 `4faed433c..HEAD` 的 36 个 `.py` 部署进 site-packages（`--port 9394` 绕过
`:9393` 存活检查；`--check` 复跑零 STALE/MISSING），然后两次 boot：

**A. `VLLM_MEM_ATTRIBUTION=1 VLLM_KV_MAMBA_INFLIGHT_STATES=1`**（日志 `..._140519`）
B2 归因表真的打出来了（这是它第一次上真机）：

* load：`routed_experts` **38,700 MiB**（权重绝对大头）、`linear_attn.in_proj_qkvz`
  1,462.5、**drafter** `routed_experts` 1,246.9（MTP k=3 的 draft 权重也要 1.2 GiB）、
  `embed_tokens` 1,212.5、`lm_head` 1,212.5、`linear_attn.out_proj` 548.4 …
* profile：`mm_encoder_profile` peak +164.1 MiB；`lm_dummy_run(2048)` peak
  **+459.7 MiB**（workspace 200 MiB）= 峰值 owner；`sampler` +236.1 MiB；
  日志明说 margin 只有 223.5 MiB —— "砍 owner 阶段的开销只能省下这么多 KV"。
* kv 预算行：`requested=62529.7 weights=48458.2 torch_peak_increase=460.2
  non_torch_increase=88.0 total_consumed=49438.0 transient_headroom=242.9
  non_kv=49680.9 cudagraph_estimate=338.0 (applied) available_kv=12510.8
  workspace=200.0` MiB。
* builders：KV 定容之后 attention metadata builder 只分配 0.1 MiB（两次都是）。

B4-1 的日志行与**实测代价**：
`Mamba in-flight states: 4 extra state blocks per request (1 per group x 4
align-mode groups) ... (max_concurrent_batches=2, max_in_flight_tokens=4096),
charged once per running request`
→ `GPU KV cache size: 413,327 tokens`（Tier-1 后不开这个 flag 是 422,264）
= **−8,937 tokens，−2.1%**；并发 1.61x → 1.58x。
之前按 state 尺寸估的 ~4% 偏大，实测就是 2.1%（一页 ~16.5 MiB，不是 37 MiB）。

功能探针（同一次 boot）：`PING-7341` 逐字正确；长 prompt 三次重复输出完全一致
（TTFT 0.809 → 0.429/0.433 s，前缀命中）；图片两次输出完全一致。

**B. `VLLM_MOE_MASK_PADDING=1 VLLM_DETERMINISTIC_MOE_ALIGN=1`**（日志 `..._141411`）
`Warmed up the deterministic MoE alignment kernels for 512 experts.` → 路径真的
接上了（`BLOCK_E=512` 正好卡在我们的 expert 数上）。KV 容量不变（422,264）。

### 一个负面结果：B6/B7 并没有治好 run-to-run 的 logprob 抖动

同一个 greedy 请求（`temperature=0`，`logprobs=1`，16 token）打 8–10 次，
比较每次返回的 per-token logprob 向量：

| arm | 结果 |
| --- | --- |
| GPU1 默认（B6/B7 off） | 10 次 10 个不同签名；97/112 token 的 logprob 不同，worst \|Δ\| **0.101** |
| GPU1 `MASK_PADDING=1` + `DETERMINISTIC_MOE_ALIGN=1` | 同样 97/112 不同，worst \|Δ\| **0.245** |
| GPU0 生产服务 `:9393`（老代码） | 8 次 8 个不同签名 |

也就是说：这两个补丁是**对的**（对齐顺序有定义、padding 行不再占 expert slot），
但它们不是这次抖动的主因。而且 `cudagraph_capture_sizes=[1,2,4,8,16,24,32]`
里有 size 1 → 单请求 decode **根本没有 padding 行**，mask padding 在这条路径上
无事可做。剩下最像的嫌疑是 INC/humming 低位宽 MoE GEMM 的 split-K 原子累加
（顺序不定 → 累加结果低位不同）；fork 那 555 个 commit 里没有任何一条动
humming 的确定性（只有 `#57421` 共享 workspace、`#58054` 版本号、`#58427` W4A8
oracle）。**结论：两条都保持默认 off，值不改；抖动留给 humming 侧查。**
探针 `_dev/fork_cmp170hx/probe_logprob_repeat.py`（completions 端点回的是 legacy
`token_logprobs`/`tokens`，不是 chat 的 `content`）。

### B8 落地 `9611389a89` + launcher `b2ac182d11`（cherry-pick `d945300e8d`）

`--speculative-config '{"method":"mtp","num_speculative_tokens":3,"adaptive_k":
{"min":1,"max":3,"log_interval":200}}'`；launcher 侧 `-AdaptiveK N`（0 = 默认，
JSON 与上游逐字相同；N 是策略可选的最大宽度，必须 ≤ `-MtpTokens`）。
我们的 drafter 是 MTP = `autoregressive/speculator.py`，里面
`supports_variable_num_steps = True`，所以 `num_steps` 覆写这条路是活的。
"验证宽度"与"起草宽度"是**故意两个数**：验证宽度被手上已有的 draft 数卡住
（uniform decode batch 每个请求必须交同样多的 draft，取 batch 最小值，并且在
`num_tokens_with_spec` 被读之前就把长 draft 列表截掉，后面所有尺寸跟着走），
起草宽度是策略那个不加封顶的选择 —— 分开才能让 k 重新爬上去。

GPU1 A/B（`probe_decode_tps.py`，`min_tokens=512` + `ignore_eos` 的长输出，
3 个 trial；GPU0 生产服务同时在跑）：

| arm | batch 1 tok/s | batch 1 TPOT | batch 4 tok/s | batch 4 TPOT |
| --- | --- | --- | --- | --- |
| A 固定 k=3（`..._155426`） | 42.5 / 46.8 / 47.5（均值 **45.6**） | 22.5 / 21.1 / 20.8（**21.5 ms**） | 134.2 / 144.5 / 136.7（**138.5**） | 27.5 / 26.4 / 27.3（**27.1 ms**） |
| B `-AdaptiveK 3`（`..._160401`） | 46.9 / 50.3 / 50.8（均值 **49.3**） | 20.3 / 19.6 / 19.5（**19.8 ms**） | 150.7 / 152.1 / 149.0（**150.6**） | 25.7 / 25.3 / 25.8（**25.6 ms**） |

→ **+8.2% tok/s（batch 1）/ +8.7%（batch 4）**，TPOT **−7.9% / −5.5%**。
策略自己打的东西（`log_interval=200`）：mean k 在 **2.23–2.60** 之间晃，
`accepted-draft prior` 1.53–1.74，分布大约 k=2 占 40–77%、k=3 占 23–60%、
k=1 几乎不出现（0–4%）。也就是"3 个 draft 平均只中 1.7 个"这个实测接受率把
宽度拉到 ~2.4，而不是恒 3。
输出仍然正确：`PING-7341` 逐字对、长 prompt 三次一致、图片两次一致。

启动日志两条要点：
`Adaptive SD: decode CUDA graphs captured for draft counts [1, 2, 3]`（target 和
speculator 各打一次）；`Acceptance-adaptive speculative decoding enabled:
draft counts [1, 2, 3], ema=0.80, margin=1.00, quantile=0.50`。
"uniform decode batch … 没有捕获图、跑了 eager" 只在 capture 刚结束的那几步出现
（`num_reqs=4 tokens/req=4` 那几条），真正服务期间没有再出现。

**决定**：代码默认 off（fork 的约定），要开就用 `-AdaptiveK 3`。
`k=1` 那档基本用不上，所以捕获图集合 `[1,2,3]` 的额外显存/捕获时间代价要留意，
但这次 boot 没看到 KV 容量被吃掉（422,264 tokens 那一档不变）。

### B5 实测：这台卡上 autotune 本来就稳，pin 没有可测的收益（决定不开）

两个探针（GPU1，无服务抢占）：

1. `probe_fla_winners.py` —— **3 个全新进程**跑我们形状（H=48=
   `linear_num_value_heads`、K=V=128、BT=64=`FLA_CHUNK_SIZE`）的
   `chunk_gated_delta_rule`，打印每个 autotuned kernel 落定的 config：
   **三次结果逐字相同**（只有 pid 行不同）。也就是说跨 boot 的"赢家不同"这件事
   在我们形状上没发生。
   赢家：`chunk_delta_h` BV=32/warps=2/stages=2；`chunk_o` BK=64,BV=64/warps=4/
   stages=2；`chunk_scaled_dot_kkt` BK=64/warps=8/stages=3；
   `chunk_local_cumsum_scalar` warps=2/stages=3；
   `solve_tril.merge_16x16_to_64x64` warps=8/stages=2；`wy_fast` warps=4/stages=3。
2. `probe_fla_bucket_cost.py` —— 先按 **T=64** warmup（GDN 层就是这么干的：
   `T = FLA_CHUNK_SIZE` 一趟过，而 autotune key 里没有 token 数，所以 64-token
   上选出的 config 之后所有长度都用），再给 **T=2048** 计时；然后清 cache、按
   T=2048 warmup、再计时：
   **0.872 ms vs 0.872 ms，差异 −0.0%**，而且两种 warmup 的赢家完全一样。
   → "T 桶不匹配"在我们这儿一分钱不花。

结论：`VLLM_FLA_PIN_AUTOTUNE` 唯一剩下的好处是省掉 boot 时那几秒 autotune
benchmark，代价是 **H=48 落 `DEFAULTS`**，而 `DEFAULTS` 跟我们实测的赢家不一样
（`chunk_delta_h` DEFAULTS BV=64/warps 4/stages 3 vs 实测 BV=32/warps 2/stages 2；
`chunk_o` stages 3 vs 2；`kkt` warps 4 vs 8；`cumsum` warps 4 vs 2；
`solve_tril merge` warps 4/stages 3 vs 8/stages 2）—— 拿没验证过的 config 去跑
H=48 大概率是**变慢**而不是变快。要真开就得按我们形状重扫表（fork 那张表是
"Generated file，别手改"，heads 只有 [16, 64]）。
**所以：代码留着、flag 保持默认 off，H=48 的表不补**（本卡收益已被实测否定）。
状态仍是 `[~]`：机制落地，开关不值得开。

顺带一条相关事实（`qwen_gdn_linear_attn.py` boot 日志）：我们用
`Using Triton/FLA GDN prefill kernel (requested=auto, head_k_dim=128)` ——
不是 FlashInfer、不是 CuteDSL，所以 fla 那套 autotune kernel 确实在路上。

### B9 DFlash：两条已落地，端到端验证卡在 27B 环境本身

已落地（GPU1 单测全绿，`380 passed`，只剩 3 条 PP/环境失败）：

* `a54e5f35e4` ← `04730e8270`（**带上游 PR 号 #57632**）：把 DFlash 的 context
  K/V 预计算收进 draft CUDA graph。我们树里 `dflash/cudagraph.py` +
  `dflash/speculator.py` **零冲突**。
* `0b0ad49693` ← `561bfdcccd`（fork 原创）：`VLLM_DFLASH_BOUNDARY_CACHE`
  （改名，我们没 `VLLM_GLM5_*` env 家族）。三条我们自己得补的东西：
  1. 我们 `scheduler.py` 没有 module-level `envs` import → 每个调度步
     `NameError: name 'envs' is not defined`，**173 条测试**挂在这上面，补 import 后恢复；
  2. `KVCacheManager.__init__` 把我们的 `enable_mamba_fine_grained_prefix_cache`
     **留着**，跟 fork 新加的两个参数并列，不是替换；
  3. 新测试给 `Scheduler._mamba_block_aligned_split` 造的 stub 少了
     `mamba_fine_grained_prefix_cache`（我们树的 splitter 读它）→ 14 条用例
     `AttributeError`；现在按 `Scheduler.__init__` 的推导方式补上，跟
     `tests/v1/core/prefix_cache/test_mamba_eagle_resume_checkpoint.py` 里那个
     邻居 stub 一致。**20 passed**。
* `5a57fd033d`（frozen draft confidence + 向量化 draft masking）**没做**：它改的是
  fork 自己的 `dflash2/speculator.py`（我们树里这文件根本不存在，`dflash2/` 只有
  `__init__.py`），fork 那份用 `candidate_sampler.scores` 做 confidence，
  我们的 `DFlashSpeculator` 是从 `draft_logits` gumbel argmax，语义不一样，
  要移植就得重写 `predict()`。那条 `draft_skip_mask` 在 fork 里是跟 confidence
  绑在一起的（只有 `confidence.coefficients is not None` 时才设 mask）。

**为什么端到端验证做不了（不是 GPU1 的问题）**：27B 服务用的解释器是
`D:\codellm-windows\.venv\Scripts\python.exe`，它的 vLLM 是**另一份、更老的
快照**，跟我们仓库树不是一套代码 —— 实测 `vllm/**/*.py` 里
**1409 个文件不同、1021 个相同、263 个只在我们仓库里有**。
两次尝试都是把仓库文件硬塞进那份快照，结果：

1. 先塞全量 → `ImportError: cannot import name 'HiSparseConfig' from
   vllm.config.attention`（它那份 `attention.py` 没这东西）；
2. 回滚到只留两个 dflash 文件 → `ImportError: cannot import name
   'maybe_prepare_dcp_local_seq_lens' from vllm.v1.worker.gpu.cp_utils`
   （我们 `dflash/speculator.py` 依赖的 `cp_utils` 它那份里没有）。
也就是说要在 27B 上验这两条，得先把 `.venv` 那套环境按我们仓库重建；而 27B
启动器自己会在每次 boot 打一堆 Windows patch（这次还报
`MISSING hybrid-kv-groups-v2-cudagraph`），那些 patch 是按它那套老快照写的，
整体覆盖有把它整套 patch 流水线打乱的风险。
**已把 `.venv` 恢复到会话前的状态**（33 个文件回滚到 `.orig`、6 个新增文件删掉、
两个 dflash 文件也从 `.orig` 还原；剩下 2 个跟 `.orig` 不同的文件是启动器自己
每次 boot 打的 patch，跟本次会话无关）。GPU1 已空（2 MiB），`:9393` 没动过（200）。

### B8 开关默认值翻转（`4ca47b14b6`）

`-AdaptiveK` 现在默认 **-1 = 跟随 `-MtpTokens`**（即开机就开），传 `0` 才回到
固定 k、且此时 `--speculative-config` JSON 跟上游**字节级一致**。三种 dry-run
已验：默认 → `"adaptive_k": {"min": 1, "max": 3, "log_interval": 200}`；
`-AdaptiveK 0` → `{"method": "mtp", "num_speculative_tokens": 3}`；
`-AdaptiveK 2` → `"max": 2`。`limits :` 那行日志现在会打 `adaptive k<=N` 或
`fixed k`，开机一眼能看出跑的是哪种。

**翻默认之前要知道的两笔代价**（都是实测，不是推测）：

* **KV 容量 −0.67%**：`422,264` → `419,430` tokens（可用 KV `12.22` → `12.13` GiB）。
  来源是给 draft counts `[1,2,3]` 各捕一套 decode graph（target + speculator），
  跟 `cudagraph_capture_sizes` 那套是两套东西。
* **启动多约 1 s**：arm A 捕获 4 s + 4 s，arm B 3 s + 5 s。
* 那 3 条 `no captured CUDA graph and ran eager` 警告只在捕获完头几步出现
  （16:07:32–16:07:33），整个 log 里总共就 3 条，服务起来之后没有再出现。

没动上游默认（`adaptive_k: dict | None = None` 还是 None），只动我们 launcher，
所以 27B/DFlash 那条服务的行为不受影响。

### 启动失败排障：`ZMQError: Address in use (tcp://127.0.0.1:29550)`（`3c1bd7987e`）

18:42 那次重启死在 EngineCore 握手：`zmq.error.ZMQError: Address in use`。
**跟 adaptive_k 无关**，根因是我 16:59 那次 27B 尝试留下的残留进程（PID 26748，
API server 在 `ImportError` 后没退，1.64 GiB 挂着占住 29550）。

**为什么 29550 这么关键**：它是 `data_parallel_rpc_port` 的**类默认值**
（`vllm/config/parallel.py:148`，`default=29550`），而 `launch_core_engines` 里
`if parallel_config.enable_elastic_ep or platform.system() == "Windows":
handshake_local_only = False`（v0.26.0 就有，我们的 base 和 upstream main 都有）
让 Windows 走固定 TCP 地址 `get_engine_client_zmq_addr(False, host, 29550)`
而不是 per-process uuid。所以它是**机器级单例**。
补充实测：健康实例**启动完成后会释放** 29550（`netstat` 查不到），
所以两个实例先后启动不冲突；卡死/失败的实例会永久占住它。

**修法**（`3c1bd7987e`）：`--data-parallel-rpc-port` 按 `-Port` 推导
（`0` = 推导，`29550 + $Port % 1000`），9393→29943、9394→29944、9395→29945，
实例间彻底独立、且永不碰 29550。已实测：线上 9393 在跑时，GPU1 起 9394
绑上自己的 29944 并 healthy，全程 9393 不受影响。

**同时补了端口守卫**（从 `start_server.ps1` 移植，flash-next 这条链原本没有）：
不带 `-Force` 就快速失败并报占用者 PID（而不是 60 s 后才崩），
`-DryRun` 只报告绝不杀进程（线上 9393 实测存活），`-Force` 才杀。
`-Force` 从 `start_qwen38_flash_next.ps1` 透传。

**重启后的生产状态**：health 200，`limits : ... adaptive k<=3`，
`Adaptive SD: decode CUDA graphs captured for draft counts [1, 2, 3]`，
KV `419,430`（= 自适应臂的值，比固定 k 的 `422,264` 少 0.67%，符合预期）。
策略实测 `mean k=2.58/2.47, accepted-draft prior=2.01/1.82, k=2 42%/53%, k=3 58%/47%`。
探针：PING 精确、长 prompt 三次一致、图片两次一致。
`probe_decode_tps`：batch1 `44.7/49.4` tok/s、batch4 `138.3/138.0` tok/s。
**注意**：batch4 落在固定 k 臂水平（138）而不是我 GPU1 测的 150.6，原因是这轮
接受率更高（prior 1.82–2.01 vs 1.53–1.74）→ 策略更多选 k=3 → 可省的无效功更少。
自适应的收益随负载接受率变化，不是恒定 +8%。

（`stop_vllm.ps1 -Port 9394` 行为正确：把 9393 的进程列为 "still running" 而不杀。）

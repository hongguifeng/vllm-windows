# TOP2 落地计划 — `mw00/project-maya`（推荐） + `Flun/glm53-flash-cmp170hx-exl3`（次选）

依据：`_dev/docs/CMP170HX_MODEL_CANDIDATES_SURVEY_2026-10.md`；逐仓库证据在 `_dev/out/candidate_survey/2{a..f}_*.md`。
上游材料已浅克隆在**仓库外**：`D:\code\_scratch\mw00_project-maya`（HEAD `d187195`，VERSION 1.1.11）、`D:\code\_scratch\Flun_glm53-flash-cmp170hx-exl3`（HEAD 见 manifest）。
风格对齐 `_dev/fork_cmp170hx/TODO.md`。状态图例：`[ ]` 未开始 · `[~]` 进行中 · `[x]` 已落地并验证 · `[-]` 决定不做。

> **判定口径（用户 2026-10-15 更正）**：两张 170HX 的全部显存（127.22 GiB）都可以留给目标模型，即**接受停掉/替换 :9393**；RAM 仍按 87.92 GiB 算。本轮调研我仍然没碰 :9393；本计划里凡需要两卡全空的验证（N1 的 TP=2、Flun / Maya-M/L 的实跑）都归到**停机窗口**里执行。

> **与已落地项的关系**：本计划里没有任何一条与 `_dev/fork_cmp170hx/TODO.md` 的 `[x]` 项重复。
> 特别注意：`B9 DFlash`（`[~]`，归 27B 服务）与下面的 A 组**不冲突**，A 组针对的是 PLE 的 SSD 随机读，不是投机解码。

---

## A0. Tier 0 — 本轮自查推翻的一条：NCCL 不是墙（`nccl.dll` 已编出，跨卡 comm init 已实测可用：SHM/CE，无 P2P）

| # | 状态 | 上游/本地出处 | 改动 | 收益假设 | 验证方式 |
|---|---|---|---|---|---|
| N1 | `[x]` | `_dev/bin/_build_nccl.ps1:1-15`（注释原文：“Windows torch ships WITHOUT NCCL, but vLLM's pynccl does not go through torch: it dlopen()s the library itself via ctypes, using VLLM_NCCL_SO_PATH. So a local nccl.dll is all that is missing for --tensor-parallel-size to work.”；源 `github.com/SystemPanic/nccl-windows` 的 **`nccl-windows`** 分支；`$Dir = 'C:\nccl-windows'`；`-Arch '80;86'`）、`_dev/bin/_env.ps1:34-38`（DLL 存在即自动设 `VLLM_NCCL_SO_PATH`）、`vllm/utils/nccl.py:23` | 跑一次 `& _dev\bin\_build_nccl.ps1 -Arch 80`（只要本卡，更快），把 `C:\nccl-windows\install\bin\nccl.dll` 造出来 | 解锁 TP=2 / PP 一类候选；也让 TODO 里因"无 NCCL"被判不做的条目重新变成可选 | 先确认 `_env.ps1` 把 `VLLM_NCCL_SO_PATH` 设上；**TP=1/pynccl 冒烟可在 GPU1 第二实例上做**；TP=2 需两张卡都空 → 与 :9393 冲突，只能等停机窗口。**已完成部分（2026-10-15 23:17）**：configure + build 通过（用 `vcvars64.bat` 绕开 `_env.ps1` 的 `$MSVC` 空展开，见报告 §8.7）；产物 `C:\nccl-windows\install\bin\nccl.dll` 34,019,840 B；纯 CPU `ctypes.CDLL` 加载成功，`ncclGetVersion` → **22907**（NCCL 2.29.7），`GetUniqueId / CommInitRank / AllReduce / Broadcast / Send / Recv / CommDestroy` 全部导出；`_env.ps1` 现已自动把 `VLLM_NCCL_SO_PATH` 设上。**跨卡 `ncclCommInitRank` 已于 23:44 实测通过**（见 N4）。构建日志：`_dev/out/nccl_build2.log` |
| N2 | `[ ]` | 外部事实：NVIDIA 官方 NCCL **无 Windows 构建**；PR **NVIDIA/nccl#1922**（Windows platform layer）至 2026-05-15 仍 **open、未合并**；issue #1287 里有人放弃 MSVC 原生移植 | 把它当**第三方维护、会漂移**的风险记账：每次升 vLLM/NCCL 后重验 dlopen 与 transport | — | 升版后重跑 N1 的冒烟 + `tests/distributed/test_pynccl.py` |
| N3 | `[x]` | 同上；clee 的 PP 靠 **NCCL SHM transport**，nguyenthimy 的 TEP2 靠**真 P2P** | 造出 dll **不等于能用**：SHM / P2P transport 在那个移植里到底存不存在，本轮无法从仓库判定 | 决定 PP/TEP2 路线到底能不能开 | **已由 N4 回答**：SHM 存在（`via SHM/CE/CE`）、P2P 不存在（`P2P is disabled between connected GPUs 0 and 1`） |
| N4 | `[x]` | `_dev/probe/_nccl_comm_smoke.py` + `_dev/out/smoke_c_r{0,1}.log`（23:44，rank0=GPU1 / rank1=GPU0，rendezvous 走 gloo） | **顺手得到的定价**：跨卡 `ncclCommInitRank` **0.40 s 成功**；transport = `Channel 00/01 : 0[1] -> 1[0] via SHM/CE/CE`，`P2P is disabled between connected GPUs 0 and 1` | PP 可行、TP 不可行：all_reduce 16 MiB med **8.69 ms**（→ TP 48 层 ≈ ~317 ms/2048-token prefill chunk）；send/recv 20 MiB med **6.64 ms**（PP 一个 prefill chunk 边界）、4 KiB **0.10 ms**（PP decode 边界） | 已实测；详情见 `_dev/docs/PP_VS_TP_WINDOWS.md` §2.5 |
| N5 | `[x]` | `_dev/probe/_pp_tp_smoke.ps1`、`_dev/probe/_bench_port.py`、`_dev/out/logs/**`（2026-10-16 00:06-00:40，双卡全空窗口，用户手动停的服务） | 引擎级 A/B：27B config + `--load-format dummy`（同形状随机权重，FLOPs 同真权重），eager，8192 ctx，fork 树（banner 校验过） | **PP=2 与 TP=2 在本 fork 上都起来了**；但目标模型各有闸门：PP → `config.py:879-884` 的 PLE/PP gate（= clee `ple-pp3` 要拆的）；TP → `ple_ssd.py:574` 的 `requires TP=DP=1`（而 PLE 表 95.37 GiB 放不进单卡也放不进 RAM，offload 关不掉） | 已实测；数字与解读见 `_dev/docs/PP_VS_TP_WINDOWS.md` §2.6 |

---

## A. Tier 1 — 来自 project-maya / Strata：推翻我们已认定的另一条 Windows 硬墙

运行时：生产是 `.flashnext-root/.venv`（非 editable）。Python 侧改动走 `_dev/bin/_setup_flashnext.ps1 -Execute` 全量重建，或用 `_dev/fork_cmp170hx/deploy_runtime.py` 直部署 `.py`（留 `.orig` 回滚副本，服务停止后才能部署）。C 侧（我们的 `ple_ssd_io_win.dll`）改完需单独重编。
验证一律用第二个实例：`start_qwen38_flash_next.ps1 -Unc -Gpu 1 -Port 9394`。**GPU0 的 :9393 是基线，别停。**

| # | 状态 | 上游出处 | 改动 | 我们的收益假设 | 验证方式 |
|---|---|---|---|---|---|
| A1 | `[ ]` | 我们自己的 `_dev/out/ple_ssd_io/backup/ple_ssd_io_win.c.orig`：`:40-41`（`direct` / `buffered` 两个 handle）、`:158-159`（`CreateFileW(..., FILE_FLAG_NO_BUFFERING \| FILE_FLAG_RANDOM_ACCESS, ...)`）、`:120-122`（`VirtualAlloc` 的 64 KiB granule 保证页对齐，即 NO_BUFFERING 的要求）、`:221-250`（只有跨 EOF 的行才退回 buffered exact 读）、头部注释（**IO Completion Port 在本机被 `ERROR_INVALID_PARAMETER` 拒**）；对照 Maya `src/platform/direct_file.cpp:166`（再加 `FILE_FLAG_OVERLAPPED`）与 `include/strata/platform/direct_file.hpp:34`（`alignment()==4096`） | **先确认生产里到底走的是哪个 handle**：direct 还是静默退回 buffered。再把 Maya 的 `FILE_FLAG_OVERLAPPED` + 多 issue 线程（Windows 默认 4 / Linux 16）拿来对比我们的 16 worker / QD 512 | 若现在实际在用 buffered：切回 unbuffered 可让 95.37 GiB 表不进系统页缓存 → RAM 不被表污染、`prefetch 16384` 可开更大、长 prefill 的 TTFT 有望再降 | 在 DLL 里加 direct/buffered 读次数计数（改 C 源码 = 越界，需用户授权），GPU1 第二实例跑一次长 prefill 看比例；`_dev/docs/FLASH_NEXT_WINDOWS_FEASIBILITY.md` §2.3 的"全盘失败 87"已确认是 **Python 堆缓冲 + 同步 ReadFile** 的探测产物，不代表 native 路径 |
| A2 | `[ ]` | `src/kernels/ngram.cpp:15, 256, 329, 343`（注释明写「`PrefetchVirtualMemory` 是这次 gather 改动的全部意义」；把 16 个 range 一次性交给一次调用并行完成）；`src/core/expert_source.cpp:483,490`（`PrefetchVirtualMemory` 在 Windows 8 才进 SDK → 用 `GetProcAddress(kernel32, ...)` 运行时解析，不硬链） | 给我们的 PLE 行预取加 **`PrefetchVirtualMemory` 批量 hint**（一次 syscall 提交多个 range），而不是逐个 issue overlapped read | 长 prefill 的 PLE 行就绪更早、I/O 提交开销更低；与 A1 叠加收益最大 | A/B `probe_itl.py` + boot 后的 TTFT（8192 / 32768 token 两档），看是否复现 `_dev/docs/PREFILL_KNOB_CLAIM_AB.md` 那类量级变化 |
| A3 | `[ ]` | `src/core/pinned.cu:41-57`（`MEM_LARGE_PAGES` 需要 `SeLockMemoryPrivilege`，普通账号没有 → `VirtualAlloc` 直接失败；必须在**自己的 token** 里 `AdjustTokenPrivileges` 之后再 `VirtualAlloc`） | 审我们 Windows 侧所有 pinned/large-page 相关路径（`PinnedCPUOffloadingSpec`、PLE row cache 的 host 侧）是否**静默退化**成非 pinned | 若现在是静默退化，修好可省掉一次 memcpy 并让大池真的可用 | `probe_service.py` 看 KV offload / PLE row cache 的分配行；对照 `VLLM_GLM5_MEM_ATTRIBUTION`（B2，已 `[x]`）的归因表 |
| A4 | `[ ]` | `src/core/expert_source.cpp` 的 Windows 分支注释、`STRATA_IO_THREADS` 默认 **4（Windows overlapped submits）/ 16（Linux）**；`direct_file.cpp` 的 `Pending` 队列说明「单线程 issue 是 n-gram 表读的上限：~12 µs kernel time/读、~80K reads/s，而 NVMe 在 depth 下能服务好几倍」 | 复核我们 `ple_ssd_workers=16` / `ple_ssd_io_depth=512` 在 Windows 上**是否真的打满**，还是被单线程 issue 卡住 | 若被卡住：调 issue 线程数与 depth 是纯配置收益，直接体现在 prefill TTFT | `_dev/fork_cmp170hx/probe_fla_bucket_cost.py` / `probe_itl.py` 在 GPU1 上跑两档 worker 数对比 |

> **两条审计带来的补充事实（影响 A3/B 组的定价）**：
> ① `mw00_project-maya/docs/SETTINGS.md:38` —— `STRATA_GLM_CPU_PIN` 在 **Windows 上永远是 off**（“on Windows the engine pins nothing whatever this says”）。所以 A3 的焦点不是"打开锁页"，而是**确认我们的 Windows 锁页路径有没有在静默失败**。
> ② `mw00_project-maya/README.md:45-46` —— Maya **能直接读 EXL3 pack**（`--model EXL3-3.05bpw`，125.3 GB，NVIDIA only / experimental）→ 将来真要试 GLM-5.3-Flash，**不一定得换到 ExLlamaV3 引擎**，B 组的价值因此上调。

## B. Tier 2 — 来自 Flun / ExLlamaV3：设计思想，不是补丁

| # | 状态 | 上游出处 | 改动 | 收益假设 | 验证方式 |
|---|---|---|---|---|---|
| B1 | `[ ]` | `glm/glm_dflash.py`（drafter pinning + **8K-token GPU ring cache** + fused-export 保留）、`glm/glm_dflash_prefix.py`（10.9 KB，opt-in 的"上一请求 prefix 命中"，`serve.sh` 默认开） | 借 **draft-KV 环形复用** 的设计：长上下文请求不必配等额的 draft cache | 我们 262K 上下文 + MTP×3 下，draft scratch 是 KV 预算里的固定税；环形复用能把这税从"随上下文线性"变成"固定 8K" | 先读我们的 `SpeculativeConfig`/scratch 记账（与已 `[x]` 的 B3 hybrid KV scratch 记账对照），再在 GPU1 上用短 `max_model_len` arm 量 KV 容量差 |
| B2 | `[ ]` | `glm/glm_q8_layout.py` + `glm/glm_q8_staging.py`（lm_head 钉 GPU0、bounded staging、batch-1 decode workspace） | 只借"**把 lm_head 钉在某张卡 + 有界 staging**"这一条布局思路，评估我们是否需要 | 我们 TP=1 单卡，收益不确定；列为待判 | 用 `VLLM_GLM5_MEM_ATTRIBUTION`（已 `[x]`）看 lm_head 现在占多少 |
| B3 | `[-]` | `dflash2/convert_drafter.py`（BF16 → EXL3 6bpw，36 个 quantized linear，~0.96 GiB drafter） | EXL3 格式转换工具链 | 我们不用 EXL3 引擎；且需要下载权重 → 本轮约束下禁止（**若你放开下载 + 给停机窗口，可重开**：新口径下它的显存侧已成立） | — |
| B4 | `[-]` | `kernels/k_hcfuse.py`（mHC fused decode，README Requirements 里写 **Linux** + nvcc compute_80） | GLM 家族专用内核 | 模型族不对（我们是 Flash-Next），且 Windows 需重编 | — |

## C. 明确不做（判定时同样适用）

* **换到 GLM-5.3-Flash 本体**（无论 EXL3 还是 Maya）：见报告 §6 —— 要下载 100-165 GB、作废 13 项已验证成果、tool-call 质量档最差。（停 :9393 已不是否决理由，但上面三条仍是。）**不做。**
* **Maya-M / Maya-L**：新口径下显存侧可行（两卡全给模型），但 RAM 侧要吃掉几乎全部空闲、且要下载 **116 GB（Maya-M）/ 156.3 GB（Maya-L）**（上游 README 实测数字）→ 仍 **[-] 不做**。
* **clee 的 `ple-nvme-staged` / `ple-pp3`**：`preadv` + `mmap_swap` 在 Windows 要重写，而我们的 PLE SSD 路线已跑通；PP 需要 `ncclCommInitRank` 真能跑（`nccl.dll` 已编出，见 N1；torch 内置 NCCL 仍 `is_nccl_available()==False`）。**不做。**
* **nguyenthimy 的 GDS/TEP2/PCIe 通信模块**：GDS 官方不支持 Windows、TEP2 需要**真 P2P**（NCCL 已实测可用但只有 `SHM/CE`，且 NCCL 自己报 `P2P is disabled between connected GPUs 0 and 1`）、P2P 需 Linux 驱动级解锁。**不做。**
* **polyuij42 / gavinxym 的全部补丁**：前者目标模块 `qwen3_8_flash_next`/`ple_offload_layer` 在我们树里不存在（`ImportError`）+ 依赖 `MADV_RANDOM`；后者**仓库里根本没有补丁文件**。**不做。**
* **clee 的 `pr54327-fs-cap`（KV fs tier `max_bytes` + LRU）**：这条分支**未发布**，拿不到代码；若要做只能照 upstream PR #54327 自己打 —— 留作未来开 Windows KV 卸载时的备选项，**本轮不做**。

---

## D. 建议的执行顺序（先花小钱证伪）

1. **N1/N3/N4 已完成**（23:17 构建出 `nccl.dll`；23:44 跨卡 comm init 0.40 s 成功，transport = SHM/CE、无 P2P；PP 边界 6.6 ms / 0.1 ms，TP 每层 8.7 ms）。**N5 也已完成**（00:06-00:40 双卡全空窗口内实测：PP=2 / TP=2 在本 fork 都能服务；单卡 3,940 tok/s vs PP 3,550 vs TP 1,906 的 4096-token prefill，decode 则反转）。只剩 N2（升版后重验）与 A1（一串 I/O 收益，需用户授权 GPU1 探针）。**下一个真正的工作项是把目标模型的 PLE/PP gate 拆掉**（Tier 1 C1）。
2. A4（配置档，零代码）→ A2（一次 syscall 换 16 次）→ A3（权限审计）。
3. B1 读代码 + 短上下文 arm 验证。
4. 只有当 A 组做完仍判不出推荐项可行性时，才动用 GPU1 探针（约束：单实例、只用 `D:\models` 现有权重、≤1 小时、`:9393` boot 时间戳不得变）。

## E. 验收口径

* 每条落地项都要给出**可观察的容量/延迟行**（boot 日志的 `Available KV cache memory` / `GPU KV cache size` / `Maximum concurrency`，或 `probe_itl.py` 的 ITL/TTFT 分布），不接受"作者自报"作为收益证据。
* 任何 A/B 都要留两 arm 的原始日志到 `_dev/out/`，并标注 `[我们复测]`。

# 双 CMP 170HX 大模型候选仓库评估报告（2026-10-15）

调研对象：用户给出的 6 个候选仓库。
运行前提：Windows + 当前 `vllm-windows` 分叉 + 双 CMP 170HX 64G（2×63.61 GiB 显存、87.92 GiB 主机内存）。
本轮**未下载任何权重**、**未做任何 GPU benchmark**、**未触碰 GPU0 上的 :9393 生产服务**。
逐仓库的完整证据在 `_dev/out/candidate_survey/2{a..f}_*.md`，本报告是汇总结论。

---

## 0. 结论速览

### 0.1 对你三个问题的直接答复

| 你的要求 | 答复 |
|---|---|
| **① Windows 下能否用当前项目跑这些仓库的改动** | **六个里没有一个能直接拓进来。** NCCL 已经不是墙（见 W1：本机的 `nccl.dll` **已经编出来并能被 dlopen**，NCCL 2.29.7），真正的墙是另一些：**GDS 在 Windows 官方不支持**、**`/dev/shm` 不存在**、**`MADV_*` 旋钮不存在**、**Docker/Ubuntu 依赖**、**补丁只存在于文字**（gavinxym）、**补丁目标模块在我们树里不存在**（polyuij42）、**可工作的分支未发布**（clee）。（按你 2026-10-15 的更正：**两卡显存全留给模型是允许的**，所以显存/卡数不再是否决理由；本轮调研我仍然没碰 :9393。）真正能在 Windows 上装起来的只有 `Flun/…-exl3`（ExLlamaV3 有官方 win 轮子、不依赖 NCCL）与 `mw00/project-maya`（原生 MSVC+nvcc），但它们**不是当前这个项目**，是另起引擎。 |
| **② "不需要真的把模型下载下来"这个前提如何满足** | 评估全部靠**读代码/文档/commit** 完成，已做到零下载。但要注意：任何"换到 GLM-5.3-Flash"的路线（EXL3 需 ~125 GB、Maya-S 需 96.5 GB）**都必然要下载权重才能真的跑**。所以在"不下载"约束下，唯一可执行的路径是：**继续用 `D:\models` 现有权重 + 把候选的思路/补丁移植进来**。 |
| **③ 内存与显存够不够** | 我们**现有**部署够：单卡 48.28 GiB 权重 + 0.65 act + 0.24 graph + 12.13 GiB KV，主机侧 8 GiB PLE row cache + prefetch，都在 87.92 GiB 之内。按你 2026-10-15 的口径更正：**两张卡的显存全部留给目标模型是被允许的**（= 接受停掉/替换 :9393；本轮调研我仍然没碰它）。在此口径下：我们现有部署（48.28 GiB 权重 + 12.13 GiB KV + 主机 8 GiB）本来就够；EXL3 的 ~122 GiB 也在 **127.22 GiB** 预算内。真正的瓶颈不再是"显存不够"，而是剩下两条：**每卡只有 63.61 GiB（空卡时 free 只 62.21 GiB）**与 **RAM 只有 87.92 GiB**。前者让 Flun 那道 ~64 GB 空闲门槛变得可疑，后者把 96 GB pinned RAM 一票否决（RAM 没变）。Maya-S 单卡形态能塞，但要向它让出主机内存并显式封顶。 |

### 0.2 排名（综合"能落地到 Windows + 对我们现役部署的边际价值"）

| 名次 | 仓库 | 一句话判词 |
|---|---|---|
| **1（推荐）** | `mw00/project-maya` | 唯一**今天就能在这台 Windows 机器上装起来**的大模型路线（原生 MSVC+nvcc 构建、不需要 NCCL），也是唯一带来**新知识点**的来源：它的母体 Strata 原生实现的就是我们现役 Flash-Next 的形状，其 Windows+SSD 随机读经验可直接移植。 |
| **2（次选）** | `Flun/glm53-flash-cmp170hx-exl3` | 新口径下**显存侧成立**（~122 GiB < 127.22 GiB），但它自身还有三个没消的坑：README 自陈要求 **Linux**、那道「每卡 free ~64 GB」门槛与代码对不上（待验）、首跑要下 125 GB。对我们最值钱的仍是 **DFlash2 的 8K-token draft ring cache**。 |
| 3 | `clee/qwen3.8-flash-next-3x170hx` | 技术上最扎实（真补丁 + 真 bench 工件），但绑死 3 卡 + Linux + `/dev/shm`（NCCL 已实测可用，但只有 SHM、无 P2P），且**可工作的那条分支 `q38fn/stack` 根本没发布**。对我们只剩 FP8-KV 那 146 行。 |
| 4 | `polyuij42-del/qwen38-flash-next-w4a16-cmp170hx` | 头号卖点（NVMe PLE + 小内存）**我们已经实现并在跑**；其补丁目标模块在我们树里不存在 → `ImportError`。 |
| 5 | `nguyenthimy2022kg-alt/Qwen-Flash-SM80-170HX` | 最激进也最 Windows-hostile：GDS 官方明确"不支持 Windows"、P2P 要 Linux 驱动级解锁、TEP2 需要真 P2P（本机已确认 `P2P is disabled`）—— 三面硬墙。可借的只有 NVFP4→Marlin。 |
| 6 | `gavinxym/170hx-2-qwen3.8-flash-next` | **纯文字手册**，仓库里只有 README，补丁只有文字描述（作者自己把它改成纯文字版），无 LICENSE、无数据 → 无法落地。 |

### 0.3 关于"要不要换到 GLM-5.3-Flash"：**不换**

理由见 §6。摘要：只剩单卡可用（8 核、双通道 DDR4、PCIe 很可能 Gen2 x4），而 GLM-5.3-Flash 的所有漂亮数字都建立在多卡 + 高内存带宽上；**draft 块投机需要 ≥2 卡**（单卡只有 `STRATA_GLM_MTP_DECODE`，而作者自己在分层形态下说"usually slower"）；更关键的是 Maya 的量化**恰好在 tool calls 上最差**。真正的收获不是换模型，而是**把它/Strata 的 SSD 分层与 Windows 随机读经验移植进我们的 PLE 路径**。

---

## 1. 本机基线（所有数字都可回溯到工具输出）

| 项 | 值 | 来源 |
|---|---|---|
| GPU | 2× NVIDIA CMP 170HX 64G（SM80），驱动 610.88 / CUDA UMD 13.3 | `nvidia-smi` |
| 每卡显存 | 63.61 GiB；启动时 free 62.21 GiB | boot 日志 `gpu_worker.py:923` |
| 主机内存 | 94,399,016,960 B ≈ **87.92 GiB** | `Win32_ComputerSystem.TotalPhysicalMemory` |
| CPU | AMD Ryzen 7 5700X 8 核 | `Win32_Processor` |
| 生产服务 | GPU0 `:9393` 已用 **63,523 MiB / free 1,201 MiB**、util 85%；你起的 `:9394` 占了 GPU1（已用 63,307 MiB / free 1,417 MiB）→ 现状**两卡都不空**；本轮全程未干扰两者 | `_dev/out/hw_snapshot_20261015_2200.txt`（原始 nvidia-smi 输出，23:09 复测） |
| **真实可用显存（现状）** | 现状两卡均被占（见上一行）；**判定口径按用户 2026-10-15 的更正：两卡全部 65,536 MiB × 2 = 127.22 GiB 都留给目标模型**（接受停/替换两个服务） | 快照文件 + 用户口径 |
| 引擎 | `v0.1.dev21672+g4faed433c.d20261001`，TP=1/PP=1/DP=1，`quantization=inc`（AutoRound W4A16），bf16，CUDA graph `FULL_AND_PIECEWISE`，async scheduling，prefix caching | boot 日志 |
| 权重 | `D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`（176B-A6B MoE 家族） | argv |
| 显存分解 | **48.28 GiB** consumed（权重+non-torch）+ 0.65 GiB peak activation + 0.24 GiB CUDAGraph + **12.13 GiB KV** | boot 日志 |
| KV 容量 | **419,430 tokens**；262,144 上下文并发 **1.60×** | `kv_cache_utils.py:2544` |
| PLE 表 | 磁盘 **95.37 GiB**；16 worker、8192 MiB row cache、QD 512、prefetch 16384 tokens | `ple_ssd.py:641` |
| 我们的实测性能 | prefill **2,238 tok/s @4k**（1,515 @100k）、medTPOT **9.05 ms**（短）、4 并发 e2e **352.6 tok/s**、tok/step **1.78** | `[我们复测]` `_dev/docs/BENCH_170HX_BASELINE.md` |

### 1.1 我们已经有的东西（这决定了候选的边际价值）

- **PLE n-gram 表已在 Windows 上做 SSD 流式**：自研 `_dev/out/ple_ssd_io/ple_ssd_io_win.dll`，Windows overlapped read。**已在生产跑通。**
- W4A16 AutoRound（`quantization=inc`）+ **Marlin** linear kernel（boot 日志确认 `MarlinLinearKernel` / `AutoGPTQLinearMethod`）。
- MoE 后端 **Humming**（`HummingIndexedExperts`）；全注意力 **FlashAttention 2**；线性注意力 GDN **Triton/FLA**；视觉 tower FLASH_ATTN。
- MTP ×3 + **acceptance-adaptive k**；tools（`qwen3_xml`）+ reasoning（`qwen3`）；prefix caching；chunked prefill + **fair chunked prefill**；确定性 MoE 顺序；CUDA graph。
- **YaRN** 在本树可用（`vllm/config/model.py:2468`、`transformers_utils/config.py:628`）→ 1M 上下文是配置题不是移植题。
- Windows 上的 CPU KV 卸载：我们自己的 `PinnedCPUOffloadingSpec`（`_dev/docs/KV_OFFLOAD_WINDOWS.md`）。
- 已从 `Morrowmake/vllm-cmp170hx` 移植并验证 13 项（见 `_dev/fork_cmp170hx/TODO.md` 的 `[x]`）。

---

## 2. Windows 硬墙清单（本机实测，不是推测）

| # | 机制 | 实测状态 | 出处 | 后果 |
|---|---|---|---|---|
| W1 | **torch 内置 NCCL** | `is_nccl_available() == False`；`torch.cuda.nccl.version()` → `AttributeError`；无 MPI，只有 gloo（三个 venv 全中：`.venv`、`.venv-flashnext`、生产 `.flashnext-root/.venv`，torch 2.11.0+cu130） | 本机实测（2026-10-15） | **torch 的 NCCL 后端不可用，但 vLLM 的 `pynccl` 不走 torch** —— `vllm/utils/nccl.py:23` 用 ctypes 按 `VLLM_NCCL_SO_PATH` dlopen。本仓库已有构建脚本 `_dev/bin/_build_nccl.ps1`（源 `SystemPanic/nccl-windows` 的 **`nccl-windows`** 分支 → `C:\nccl-windows\install\bin\nccl.dll`），`_env.ps1:37-38` 会在 DLL 存在时自动设上。**2026-10-15 23:17 已构建**：`C:\nccl-windows\install\bin\nccl.dll` 34,019,840 B；纯 CPU `ctypes.CDLL` 加载成功，`ncclGetVersion → 22907`（NCCL 2.29.7），`GetUniqueId / CommInitRank / AllReduce / Broadcast / Send / Recv / CommDestroy` 全部导出；`_env.ps1` 现已自动把 `VLLM_NCCL_SO_PATH` 设上。**23:44 实测：`ncclCommInitRank` 0.40 s 成功；all_reduce / all_gather / reduce_scatter / send / recv / broadcast 全部可用；transport = `SHM/CE`（宿主内存 + copy engine），`P2P is disabled between connected GPUs 0 and 1`** |
| W2 | **GDS / cuFile / `nvidia-fs`** | Windows 上不存在 | NVIDIA 官方 GDS 页原文：「GDS is currently supported on Linux x86-64 distributions; **it is not supported on Windows**」；`NVIDIA/gds-nvidia-fs` issue #67 仍 open | 任何"SSD 直读显存"方案在 Windows 上只能退化成主机缓冲 |
| W3 | `os.pread` / `os.posix_fadvise` / `os.O_DIRECT` / `/proc/self/fd/3` | Windows Python 全部 **ABSENT** | `_dev/docs/FLASH_NEXT_WINDOWS_FEASIBILITY.md` §2.3 探测 | Linux 风格的 preadv/fadvise 读路径必须重写 |
| W4 | `FILE_FLAG_NO_BUFFERING`（O_DIRECT 等价物） | **分两层：Python 侧同步探测（堆 `bytes` 缓冲，只有 16 B 对齐）全盘 `ERROR_INVALID_PARAMETER (87)`；但原生路径可用 —— 我们的 DLL 用 `VirtualAlloc` 页对齐缓冲 + overlapped，已在生产使用 `direct` handle** | 探测表：`FLASH_NEXT_WINDOWS_FEASIBILITY.md` §2.3；反证：`_dev/out/ple_ssd_io/backup/ple_ssd_io_win.c.orig` `:40-41, 120-122, 158-159, 221-250` + boot 日志「native Windows overlapped reads, queue depth 512」 | **Windows 上绕页缓存直读是可行的，我们已经在用**；不可用的是 IO Completion Port（同文件头部注释：本机被 `ERROR_INVALID_PARAMETER` 拒），故改用 per-handle overlapped wait |
| W5 | `MADV_HUGEPAGE` / `MADV_RANDOM` / read-ahead 旋钮 | Linux 专有，Windows 无对应 | 内核接口 | 依赖 madvise 调 readahead 的发布配置在 Windows 上复现不了 |
| W6 | `/dev/shm` tmpfs | Windows 无 | POSIX 接口 | 任何用 `/dev/shm` 做中转/KV gateway 的方案不可移植 |
| W7 | Triton 后端硬件 FP8 KV cache | `triton_attn.py:487` 硬性要求 **SM89+**，SM80 直接 `ValueError` | 代码 | 本卡上 FP8 KV 只能走"存 uint8 + 软件 e4m3 解码" |
| W8 | GPU P2P / BAR1 解锁 / IOMMU 设置 | 需要 Linux 驱动级操作（cmpunlocker `--p2p --no-iommu`、静态 BAR1） | `docs/CMP_P2P.md`（nguyenthimy 仓库） | Windows 上没有对应的解锁/验证路径 |

**推论**：
1. 凡"必须跨卡"的候选 → **墙已经矮了一大截**：`nccl.dll` 于 23:17 构建完成且能被 `ctypes.CDLL` 加载（NCCL 2.29.7）。门槛已经实测扫清：跨卡 `ncclCommInitRank` 在本机 Windows 上 0.40 s 成功，transport = **`SHM/CE`（宿主内存 + copy engine），无 P2P**。剩下的门槛是：① clee 还需要**第三张卡**；② 我们自己的 PLE/MTP/CUDA graph 在 PP 下的正确性。**ExLlamaV3 那种逐层 device-map 分片、根本不需要集合通信的路线，连 NCCL 都不需要。**
2. 按用户 2026-10-15 的口径更正：**两张卡的显存全部留给目标模型是被允许的**（接受停掉/替换 :9393）。所以"需要 110-120 GiB 显存"不再是否决理由；本轮调研我依然没碰 :9393。剩下的硬约束只有两条：**每卡 63.61 GiB**（空卡 free 62.21 GiB）与 **RAM 87.92 GiB**。
3. 凡"要求 pinned 90+ GiB 主机内存"的候选 → 在 87.92 GiB 物理内存下**不可能**。
4. **更正（本轮自查发现）**：`_dev/docs/FLASH_NEXT_WINDOWS_FEASIBILITY.md` §2.3 把「NO_BUFFERING 全盘失败」当成了硬结论，但那个探测是 **Python 堆缓冲 + 同步 ReadFile**；我们现役的 `ple_ssd_io_win.c` 早就用 `VirtualAlloc`（页对齐）+ overlapped + `FILE_FLAG_NO_BUFFERING` 开了 `direct` handle，并且只有跨 EOF 的行才退回 buffered。所以**"Windows 绕不过页缓存"这句是错的**，本报告已按事实改写 W4；那份旧文档我没有改（超出本轮边界），只在此指出分歧点。真正在本机被拒的是 **IO Completion Port**。

---

## 3. 六仓库横向对比

> 完整版（逐条出处、证据等级、容量三态、最值得移植的前 3）在附录文件里；本节只给结论矩阵。
> 附录：`_dev/out/candidate_survey/2a_clee.md`、`2b_polyuij42.md`、`2d_nguyenthimy.md`、`2e_maya.md`、`2f_flun_exl3.md`、以及 gavinxym 的 `subagent-artifacts/outputs/377f7262-…/task-3.c4ffeec2.md`。

| 维度 | clee | polyuij42-del | gavinxym | nguyenthimy | **mw00/maya** | **Flun/exl3** |
|---|---|---|---|---|---|---|
| 仓库形态 | 文档+launcher+bench 工件；代码在 fork 的 4 个 `q38fn/*` 分支 | 复现清单/数据为主，7.5 MB | **只有 README.md（12,644 B）** | Python 源码 + 双语部署文档 | 独立 C/CUDA 引擎（48 MB） | 独立 Python 胶水层（278 KB），引擎是 ExLlamaV3 |
| 引擎 | vLLM（钉 nightly `7d0b4e57`） | vLLM fork | vLLM（只有文字描述） | vLLM（钉**镜像 digest**） | **自带引擎** | **ExLlamaV3 1.5.4** |
| 模型 | Qwen4Exp 180B-A6B Intel INT4 | Qwen3.8-Flash-Next W4A16 | Qwen3.8-Flash-Next W4A16 | Qwen3.8-Flash-Next **NVFP4** | **GLM-5.3-Flash 321B** | **GLM-5.3-Flash 3.05bpw** |
| 跨卡方式 | **PP=3** | PP=2 | PP=2 | **TEP2（TP+EP）** | 多卡可选，**单卡可跑** | device-map 逐层分片，**不需要 NCCL/P2P** |
| 真补丁？ | ✅ 我亲自 diff 出三档 | 部分（目标模块我们树里没有） | ❌ 只有文字 | ✅ 自研内核（不可核，钉 digest） | ✅ 完整引擎源码 | ✅ 全是 process-local monkeypatch |
| 可复现性能工件 | ✅ `tests/bench-results-*.md` + `bench.py`/`needle.py` | ✅ `data/environment.txt` 类清单 | ❌ 引用了不在仓库里的 log | ⚠️ 验证文档有，原始数据无 | ✅ `bench/results/MAYA-{S,M,L}.md` | ❌ README 自称 harness/数据不含 |
| 提到 Windows | 否（Linux/Docker） | 否 | 否 | **全仓 0 次** | ✅ `docs/WINDOWS.md` + `START-MAYA.bat` | ⚠️ README 写 Requirements 里有 **Linux** |

### 3.1 每仓小结

**clee/qwen3.8-flash-next-3x170hx** —— 我亲自跑了分支 diff：
`q38fn/fp8-kv-sm80` **+146/−12**（5 文件，FP8 KV 落在 QSA 稀疏注意力路径 + `platforms/interface.py`）、
`q38fn/ple-nvme-staged` **+1196/−64**（新增 `ple_preadv.py`、`vllm/v1/ple_offload/mmap_swap.py`）、
`q38fn/ple-pp3` **+215/−49**。它 README 钉住的可工作 tip **`q38fn/stack` 与 `pr54327-fs-cap` 在远端不存在**（`git ls-remote` 只有 4 个 `q38fn/*`）→ **配方不可复现**（另注：README `:63` 书签表里写的是 `q38fn/ple-pipeline-parallel`，远端实际发布的是 `q38fn/ple-pp3`，两者关系作者未说明）。性能：`[有可复现工件]` 只覆盖 74 tok/s short 与 73 @1M（`tests/bench-results-2026-10-10.md` 里 6 行与 README 表逐行一致，审计轮逐字核过）；**KV 池 7.36M tokens 只出现在 README `:13` 的能力表，bench 文件里没有 ⇒ 改判 `[作者自报·叙述性]`**；README `:145` 还明写 **MTP 未启用**（draft 层 KV 要吃 ~16% 池）⇒ 他们的 tok/s 与我们 MTP×3 的口径**根本不同**，但口径是 3 卡 PP=3 + 1M 上下文，与我们 4 并发/262K 不可比。

**polyuij42-del/…** —— 卖点「NVMe mmap 按需分页、稳态 RAM≈11 GiB、不需 pinned」我们**已经实现并在生产跑**（`ple_ssd_io_win.dll`）。其具体部署判跑不了：PP=2 需要 `ncclCommInitRank` 真跑得起来（`nccl.dll` 已于 23:17 编出、comm init 已实测可用（SHM/CE），见 Tier 0 N1；真跑还需两卡全空）、补丁目标模块 `qwen3_8_flash_next`/`ple_offload_layer` 在我们树里不存在（`ImportError`）、docker/`blockdev`/udev 缺失、Windows 上没有 `MADV_RANDOM` 与 read-ahead 旋钮。**但这道墙的分寸审计轮纠出来了**：`README:36` 说发布数字是用 `VLLM_PLE_MMAP_RANDOM=1` 测的（我们复现不了），可同一行又说**线上自 2026-09-16 起跑 `=0`**，因为盘上 16 KiB 预读已到位、两档实测差不多（`data/environment.txt:61`：systemd drop-in 把它覆盖成 0）⇒ 它**挡住的只是复现他们的发布数字，不是他们的线上形态**。性能 `[作者自报]`（104 tok/s、774 @23 并发）—— 与我们 `max_num_seqs=4`、4 并发 352.6 tok/s 不同口径。

**gavinxym/…** —— HEAD 提交信息就是决定性证据：「改为纯文字版：去掉对补丁源码/脚本/数据的依赖」。工作树只有 `README.md`，**没有 `.patch/.diff/.py/.sh`、没有 LICENSE**；引用的 bench log 不在仓库内 → 所有速度数字**无法第三方复核**。其路线 PP=2 + MTP k=4 + PLE 96 GB 常驻 pinned RAM，三条前提本机逐项不成立。

**nguyenthimy/…** —— `Dockerfile:2` 钉的是**镜像 digest** 而非可 diff 的 commit；`DEPLOYMENT.md:6` 原文要求「Run the commands on **the Linux server**」。TEP2 = 张量并行+专家并行（`README.md:84`），要 Marlin + **GPU P2P**；GDS/cuFile 在 Windows **官方不支持**；全仓**一次都没提到 Windows**。性能 `[作者自报]`，且 170.826 tok/s 那条被 README 自己标注为「不同阶段的历史对比，并非同条件原版 vLLM 对照」。唯一对我们有用的一条：**NVFP4 在 SM80 上可以走 Marlin**（vLLM PR **#45306**、**#19563**）—— 而 Marlin 我们已经装上了。

**mw00/project-maya** —— Windows 构建路径是**真的**：`START-MAYA.bat` → `maya.py` → venv 内装 CMake/Ninja → `vswhere` 找 VS2022 Build Tools 的 `vcvars64.bat` → nvcc + MSVC 本地编译；**没有预编译包、没有 CI**，靠 Windows 用户实测报告支撑。代价是一串 Windows 专属失效模式（commit 内存记账、free-VRAM 读数偏高 ~1 GB 导致超填→显存分页到 RAM→decode 掉几倍、page-locking 上限、EcoQoS）。**最高价值的发现**：`docs/GLM5-FLASH.md` 显示它的母体 Strata 原生实现的就是**我们现役 Flash-Next 的形状**（n_embd 2560 / 48 层 / 512 experts×top-10 / n_ff 640 / PLE 28.8 GB n-gram / MTP），并记录了 Windows+SSD 随机读的实测教训（random-access hint 慢 14×、MSVC `ifstream` 4095 B 拆分、mmap vs pinned 2.2×、`PrefetchVirtualMemory`）。

**Flun/glm53-flash-cmp170hx-exl3** —— 目标权重 **~116.6 GiB 双卡全驻留**、Q8 MLA KV @384K ≈ **4.51 GiB**、drafter ~0.96 GiB + 8K-token GPU ring cache。**本轮被审计推翻了两条我自己的断言**：
① 我写过「torch cu130 无现成轮子 ⇒ 必须自建」—— **错**。`build.yml:50-54, 72-73` 把 `windows-2022` 与 `cuda 13.2.0 + torch 2.11.0`（py3.12/3.13.3/3.14）列进了矩阵，即**引擎在 Windows 上有现成轮子可用**。
② 我写过「若允许停服务，两张卡各 ~64 GiB → 127.2 GiB 刚好塞得下」—— **错**。README `:67` 是**启动门槛**：「The server refuses to start if either GPU has less than ~64 GB free」，而本机每卡只有 **63.61 GiB**（boot 时 free 62.21 GiB）→ **即使停掉 :9393 也过不了它自己的门**。
再加上 README `:65` 自己就把 **Linux** 写进 Requirements（`setup.sh`/`serve.sh`、`k_hcfuse.py` 要 nvcc compute_80），所以它的**部署路线在本机不成立**。
**但它真正值钱的东西仍在**：`glm/glm_dflash.py` 的 8K-token GPU ring draft cache，以及一个审计带来的发现 —— **Maya 自己能直接读 EXL3 pack**（`mw00_project-maya/README.md:45-46`：`--model EXL3-3.05bpw`，125.3 GB，"NVIDIA only, experimental"），所以"走 EXL3 就必须换引擎"这个前提也不成立。**它不依赖 NCCL/P2P**（host-bounce D2D，README `:4,40`）这点依然是真的。

---

## 4. 显存 / 内存核算（判定口径已按用户 2026-10-15 的更正：2×63.61 GiB = 127.22 GiB 显存全部留给目标模型；RAM 仍 87.92 GiB）

| 候选 | 权重 | PLE/辅助表 | KV | 需要的显存 | 需要的 RAM | **三态结论** | 数字来源 |
|---|---|---|---|---|---|---|---|
| **我们现况** | 48.28 GiB | PLE 95.37 GiB 在盘，主机 8 GiB row cache | 12.13 GiB = 419,430 tok | 单卡 ~61 GiB | 8 GiB 级 | **能跑（已在跑）** | `[我们复测]` boot 日志 |
| clee (PP=3) | ~169 GiB 检查点（NVMe 上） | PLE 95.4 GiB 走 NVMe | 池 7.36M tok；NVMe tier 128 GiB | **3 卡各 ~56 GiB** | 128 GB 级（作者机器） | **跑不了**：无第三张卡；NCCL 已构建且 comm init 已实测可用（SHM/CE，无 P2P）；无 `/dev/shm`；要 Linux | `[作者自报]`+config |
| polyuij42 | 169 GB 模型（含 ~102 GB PLE） | NVMe mmap 按需 | 未给 | PP=2 → 两卡（新口径下显存侧可行） | 稳态 11 GiB / 峰值 20 GiB | **跑不了**：补丁目标模块在我们树里不存在、docker/`blockdev`/udev 缺失、madvise 复现不了发布口径（NCCL 已构建，见 Tier 0 N1）；**但它的抽象我们已经实现** | `[作者自报]` |
| gavinxym | 181.2 GB | **PLE 96 GB 必须常驻 CPU 内存** | 未给 | PP=2 → 两卡（新口径下显存侧可行） | **≥96 GB pinned** | **跑不了**：96 GB pinned > 87.92 GiB —— **这条在新口径下依然成立，因为放松的是显存不是 RAM**；补丁只有文字；NCCL 已构建、comm init 已实测可用 | `[作者自报]` |
| nguyenthimy | NVFP4 主模型（双卡 TEP2） | PLE 51.2 GB 留 SSD（省 47.68 GiB 整表显存） | 未给 | **两卡都要**（TEP2+P2P；新口径下显存侧可行） | 需 GDS 环境 | **跑不了**：GDS 官方不支持 Windows、P2P 需 Linux 驱动级解锁、要 Linux；NCCL 已构建且 comm init 已实测可用，但本机 **P2P 确认不可用**（NCCL 自己报 `P2P is disabled`）→ TEP2 仍死 | `[作者自报]` |
| **Flun EXL3** | **116.6 GiB 双卡全驻留** | 无（全显存） | Q8 4.51 GiB @384K | 116.6 + 4.51 + 0.96 ≈ **122 GiB < 127.22 GiB** | 低（全显存路线） | **显存侧能跑（新口径）**，但三件事没消：① README `:67` 那道「每卡 free < ~64 GB 就拒启」与**本仓代码对不上** —— 我只在 `glm/glm_q8_layout.py:71-72` 找到 staging 计算，没找到 64 GB 硬判据，所以它是"作者自陈"还是 ExLlamaV3 运行时行为**待验**（空卡 free 只有 62.21 GiB）；② README `:65` 要求 **Linux** + nvcc compute_80 + ~140 GB 盘；③ 首跑自动下 125 GB；④ **cu130 与我们现役 torch 能不能直接用 cu128.1 / cu132.0 的 Windows 轮子，未经证实** —— AUDIT 里唯一被评为"严重度高"的一条，矩阵里 `cudaarch` 也含 `8.0`。引擎本身有 Windows 轮子（§3） | `[有仓库原文]` + `[推算]` |
| **Maya-S** | 96.5 GB（分层：部分 VRAM / RAM / NVMe） | — | 1M 上下文按需 | **单卡 63.61 GiB 可容纳其 VRAM 侧** | **要吃掉几乎全部空闲 RAM** | **单卡有条件能跑**：`STRATA_GLM_RAM_GB` 显式封顶 + 大 pagefile + `--gpu 1` + `STRATA_GLM_SPLIT=0`；**Maya-M/L 在新口径下显存侧可行**（两卡全给模型），但 RAM 侧要吃掉几乎全部空闲且需下载权重 | `[有仓库原文]`+推算 |

**叠加检查（口径更正后）**：可用显存按 **127.22 GiB** 计（两卡全给模型）。但每卡只有 **63.61 GiB**，且**空卡时 free 只有 62.21 GiB**（boot 日志）→ 任何"每卡要 ~64 GB 空闲"的门槛在本机都过不去，Flun 那条正卡在这上面（见 §3/§4）；RAM 侧的 87.92 GiB 不因口径变更而放松。

---

## 5. Windows 移植性矩阵

图例：**✅ 可替代** / **❌ 不可替代** / **🔧 需重写**

| Linux-only 机制 | clee | polyuij42 | gavinxym | nguyenthimy | maya | Flun |
|---|---|---|---|---|---|---|
| **NCCL all-reduce / TP / PP / EP** | ✅ comm init 已实测可用（SHM/CE，无 P2P；PP=3 还需第三张卡） | ✅ 已实测 | ✅ 已实测 | ❌（TEP2 需真 P2P，本机已确认无 P2P） | ✅ 单卡不需要 | ✅ device-map 分片，不需要 |
| **GDS / cuFile / nvidia-fs** | — | — | — | **❌ 官方不支持 Windows** | — | — |
| **`/dev/shm` tmpfs** | ❌（8 GiB KV gateway 用它） | — | — | — | — | — |
| **`MADV_HUGEPAGE` / `MADV_RANDOM`** | — | 🔧（发布口径依赖它） | — | — | — | — |
| **pinned host memory 大块** | ✅（改走 SSD 流式） | ✅（已实现） | ❌（96 GB pinned > 87.92 GiB） | ✅（留 SSD） | 🔧（受 commit/页锁定上限限制） | ✅（不需要） |
| **`mmap` 按需分页语义** | 🔧（`mmap_swap.py`） | 🔧（核心卖点在 Windows 语义不同） | — | — | 🔧（它自己的 mmap vs pinned 数据） | — |
| **O_DIRECT / 绕页缓存** | 🔧（`preadv` 在 Windows 不存在） | 🔧 | — | ❌（靠 GDS） | ✅ 它自带 Windows 原生实现 | — |
| **GPU P2P / BAR1 解锁 / IOMMU** | ❌（作者实测跨 root bus DMA 挂死） | — | — | ❌（TEP2 依赖真 P2P） | ✅（不依赖） | ✅（README 明说 **no P2P required**，host-bounce D2D） |
| **NVMe KV tier** | 🔧（我们已有 CPU offload 路线） | 🔧 | — | — | ✅ 原生分层设计 | — |
| **Docker / bash 脚本** | 🔧 | 🔧 | 🔧 | 🔧 | ✅ 原生 `.bat` + Python 构建 | 🔧（`setup.sh`/`serve.sh`） |
| **硬件 FP8 KV（SM89+）** | 🔧（他们用软件 e4m3 绕开） | — | — | — | — | ❌（Q8 MLA latent 需另行判定） |

> **注 1（NCCL 行）**：`_dev/bin/_build_nccl.ps1` 的构建**已经跑通**（我用 `vcvars64.bat` 绕开 `_env.ps1` 的 `$MSVC` 空展开，见 §8.7；源是 SystemPanic 的 `nccl-windows` 分支，不是官方 NCCL —— 官方至今无 Windows 构建，PR #1922 仍未合并）。产物能被 `ctypes.CDLL` 加载、NCCL 2.29.7、七个主符号全部导出，但**能加载不等于能用**：PP 靠的 NCCL **SHM transport** 与 P2P transport 在那个移植里到底存不存在，transport 也已经从日志里读出来了：`Channel 00/01 : 0[1] -> 1[0] via SHM/CE/CE` → **PP 要的 SHM 存在**；`P2P is disabled between connected GPUs 0 and 1` → **TEP2 要的 P2P 不存在**。
> **注 2（O_DIRECT 行）**：我们已经有 Windows 版的"绕页缓存直读"（`ple_ssd_io_win.c` 的 `direct` handle，`FILE_FLAG_NO_BUFFERING \| FILE_FLAG_RANDOM_ACCESS`，`VirtualAlloc` 页对齐缓冲）。所以 clee/polyuij42 那些 `preadv`/`mmap` 读路径对我们不是"做不到"，而是"得改写成调用我们自己的 DLL" —— 这才是"需重写"的真实含义。

---

## 6. 要不要换到 GLM-5.3-Flash：**不换**

| 判据 | 结论 | 依据 |
|---|---|---|
| 哪个量化档塞得进新口径（127.22 GiB VRAM / 87.92 GiB RAM） | **Maya-S（96.5 GB，分层）** 单卡就行；**Maya-M/L 与 EXL3 3.05bpw（116.6 GiB）需要两卡全给模型**；RAM 侧无论哪一档都得让出来 | `[有仓库原文]` / `[作者自报]` |
| 精度损失 | Maya 官方称 97.7–99.2% zero-shot，**但 tool calls 那一档最差（KL 0.850、同 token 76.5%）** —— 而 tool calls 正是我们的主要用法 | `[有仓库原文]`（maya `bench/results/`） |
| 许可 | GLM-5.3-Flash 是 zai-org、MIT；Maya 的量化由其从 FP8 制作 | `[有仓库原文]`（`LICENSE`） |
| 与 :9393 共存 | 按新口径：**可以停掉/替换 :9393**，所以"共存"不再是硬约束。EXL3 的 ~64 GB 空闲门槛仍需单独判定；Maya-S 若要共存则必须给它封顶 RAM，而 :9393 自己也要 RAM（PLE row cache 8 GiB + prefetch） | 用户 2026-10-15 口径更正 + boot 日志 |
| 我们已移植的 Flash-Next 专有机能会不会失效 | **会大半失效**：PLE SSD 流式、PLE row cache/prefetch、fair prefill 的 PLE 旋钮、`VLLM_GLM5_*` 系列开关、Humming MoE 后端、GDN 线性注意力 —— GLM-5.3-Flash 是**另一个模型族**，这 13 项已验证成果直接作废 | 我们的 TODO/boot 日志 |
| 单卡能力 | **draft 块投机需要 ≥2 卡**（`docs/SETTINGS.md:66-67` 的 `STRATA_GLM_MTP_GGUF`/`MTP_KEEP` 都写着 "two GPUs or more"）；单卡只有 `STRATA_GLM_MTP_DECODE`（`:68`，一卡装下整模型时才快，默认仅 Ryzen AI Max APU 开，作者自说在专家从 RAM 流的形态下"usually slower"）。而本机只剩单卡可用 | `[有仓库原文]`（已按审计更正，初稿把这句话说重了） |
| 硬件天花板 | 8 核 Ryzen 7 5700X + 双通道 DDR4，PCIe 很可能 Gen2 x4；它的漂亮数字（118 tok/s @4×4090、81 @RTX PRO 6000、61.3 @2×CMP170HX）都建立在**多卡 + 高内存带宽**上 | `[作者自报]` + 本机配置 |

**结论：不换模型。** 换过去的代价（停服务、下载 ~100-125 GB、作废 13 项已验证成果、tool-call 质量变差）大于收益。
**真正该做的是反向移植**：Maya/Strata 在 Windows+SSD 随机读上的实测教训，以及 Flun 的 draft-KV 环形缓存设计 —— 它们服务于我们现在的模型。

---

## 7. 为什么是 project-maya 第一、Flun 第二

排名用两个轴：**能不能在 Windows 上落地**（硬墙数）× **对我们现役部署的边际价值**（是不是我们已经有的东西）。

- **project-maya（推荐）**：唯一不需要 NCCL、且有原生 Windows 构建路径（`START-MAYA.bat` → CMake/Ninja → vcvars → nvcc+MSVC）的大模型路线；而且它**提供了我们没有的知识**（Strata 的 Windows SSD 随机读实测：random-access hint 慢 14×、MSVC `ifstream` 4095 B 拆分、mmap vs pinned 2.2×、`PrefetchVirtualMemory`），并且 Strata 原生实现的就是**我们这个模型形状** → 这些经验能直接用到我们的 `ple_ssd_io_win.dll` 上。
- **Flun/exl3（次选）**：它自己的**部署路线在本机不成立**（README `:65` 要 Linux、`:67` 要每卡 ~64 GB 空闲而我们只有 63.61 GiB），但它贡献了六个里对我们最有用的一个设计：`glm/glm_dflash.py` 的 **8K-token GPU ring draft cache** —— 直击我们 262K 上下文 + MTP×3 下 draft scratch 的 KV 预算。另外它不依赖 NCCL/P2P（host-bounce D2D）。
- 后面四名被压制的共同原因：**它们要的东西不是“NCCL 能不能用”就能绕开的**（NCCL 已实测可用，但只有 SHM、无 P2P）；各自额外原因：clee 的可工作分支未发布、polyuij42 的卖点已被我们实现、nguyenthimy 的 GDS/P2P 双墙、gavinxym 没有可执行工件。

**诚实的补充判断**：如果你的真实目标是"把现在这个服务做得更好"，那么**最高性价比的动作不在这六个仓库里**，而是：
1. 把 Strata 的 Windows SSD 随机读经验用到我们 PLE I/O 上；
2. 试 **NVFP4 → Marlin** 量化档（PR #45306/#19563 已把它放到 SM80）；
3. 借 Flun 的 draft-KV 环形缓存思路改善长上下文 MTP。

---

## 8. 调研过程中的更正与事故（必须留档）

1. **`Flun/glm53-flash-cmp170hx-exl3` 其实存在。** 早期一次 API 拉取返回 404 是因为**我漏了 `-exl3` 后缀**。按用户给的精确 URL 单次拉取：`api.github.com/repos/Flun/glm53-flash-cmp170hx-exl3` → **200**，`raw.githubusercontent.com/.../main/README.md` → **200**，`github.com/...` HTML → 504。因此按你最初的意图（六个都深挖）把它**纳入评估与排名**，而不是按"仓库不存在"排除。这与目标里 Success criteria (2) 的原文相矛盾 —— 事实变了，我按事实处理，并在此明确披露。
2. **并行 subagent 全军覆没。** 首轮 5 个只读 researcher 在 30 分钟全部超时、产物为 35 字节占位符（它们被 TUN fake-IP 的 SSRF 守卫拦住了 GitHub 抓取）。我逐个 resume 并把仓库浅克隆到 `D:\code\_scratch\`（仓库外）喂给它们，其中 gavinxym 与 project-maya 两份真的交付了。其余按你的"最多 2 个并行"指令被中断，改由主会话自查。
3. **没有授权任何用户级配置改动。** 有子 agent 请求在 `C:\Users\hong\.pi\agent\web-search.json` 加 SSRF 白名单，我拒绝了（那是你的安全配置，需要你本人同意）。唯一改动是你让我接入 :9394 后，我在 `models.json` 里加了 provider `vllm-win-9394`（备份 `models.json.bak-9394`），并补了它缺的 `name` 字段。
4. **`clee/vllm-sm80` 的可工作分支未发布**（`git ls-remote` 只有 4 个 `q38fn/*`，没有 `q38fn/stack` / `pr54327-fs-cap`）→ 它的"working recipe"无法从仓库复现。
5. **后台 detached `pi` 进程在这个 shell 里不存活**（`&` 起的那次没活下来，日志 0 字节）→ nguyenthimy 那一份改由主会话完成，未使用 :9394。
6. **审计轮（2026-10-15）**：`_dev/out/candidate_survey/AUDIT.md`（21,386 B，128 行）回来了，列出 P1–P8 若干条。我**自己复核并承认了两条硬性事实错误**：(a) NCCL 被我当成了硬墙，实际上我们有现成的构建脚本，而且**已经真的把它编出来了**（W1 更正 + §8.7）；(b) Flun 的两条断言（轮子匹配、停服务就能塞）被 README/build.yml 证否（见 §3/§4 更正）。审计还纠正了我两处措辞：Maya 的单卡 MTP（见 §6）与 `STRATA_GLM_RAM_RESIDENT` 的写法。**审计里那些它自己无法复现的条目（`git ls-remote`、md5、上游 PR 状态、文件计数）我的标注已改为 `[主会话亲跑 git/curl，审计轮未复核]** —— 因为这些是我在主会话里用能工作的 `git`/`curl` 亲自跑的，而子 agent 的 fetch 被 SSRF 守卫拦住了。
7. **构建 NCCL 时发现了 `_dev/bin/_env.ps1` 的真实缺陷（用户授权后已修）**：直接用 `powershell` 跑 `_build_nccl.ps1` 会在 configure 就挂——`nvcc fatal : Cannot find compiler 'cl.exe' in PATH`。根因不是版本过期（toolset 14.44.35207、SDK 10.0.26100.0 **都真的装着**），而是文件是 **UTF-8 无 BOM**，Windows PowerShell 5.1 按 ANSI 解码中文注释后把 `$MSVC = ...` 那一行吃掉了（实测：`MSVC=[]`、`PATH0=\bin\Hostx64\x64`、`INCLUDE0=\include`；改成按 UTF-8 读后同一行就完整了）。修法：改成 **vswhere + 目录发现**自动选 toolset/SDK/CUDA（带硬编码回退）+ 补上 BOM；改前原件从 git 取到 `_dev/bin/_env.ps1.orig`。修后实测：`cl.exe` 能找到、`INCLUDE0`/`LIB0`/`PATH0` 都正确、`SDKVER=10.0.26100.0`、`CUDAROOT=...v13.3`（与旧硬编码一致）。`_dev/bin` 里其余 `.ps1` 扫了一遍，**没有同类风险**。

---

## 9. 未决问题 / 建议的下一步（不在本轮执行）

| # | 未决 | 怎么决 | 成本 |
|---|---|---|---|
| U0 | ~~`nccl.dll` 到底能不能造出来~~ **已造出来并且能加载** | 剩下的问题是：SHM/P2P transport 在不在、TP=2 实跑能不能过（现在两卡都被 :9393 / :9394 占满 → 需停机窗口） | 构建已完成；运行验证高 |
| U1 | FP8 KV（软件 e4m3）在我们哪条后端能吃 | 先确认我们全注意力是 FA2、线性注意力是 Triton/FLA；clee 的实现落在 QSA 路径 → 需要判定我们的 QSA 是否在现役路径上 | 读代码，无 GPU |
| U2 | NVFP4 → Marlin 档是否真有可获取的权重 | 需要**下载权重**才能验（本轮禁止）→ 等你决定是否开下载 | 高（磁盘+时间） |
| U3 | 真要试 GLM-5.3-Flash 的话走哪条引擎 | 现在有**两条**：EXL3 runtime（本机的 64 GB 门槛过不了），或 **Maya 直接读 EXL3 pack**（`maya README:45-46`，`--model EXL3-3.05bpw`，125.3 GB，NVIDIA only/experimental）→ 两者都要下载权重 | 高 |
| U4 | ~~Strata 的 SSD 随机读经验能否直接迁到我们的 DLL~~ **本轮已部分回答**：我们的 `ple_ssd_io_win.c` 早就在用 `NO_BUFFERING` + `VirtualAlloc` 页对齐缓冲；Strata 多一个 `FILE_FLAG_OVERLAPPED` 与 `PrefetchVirtualMemory` 批量 hint（`ngram.cpp:256/329/343`），且 Windows 下只开 4 个 issue 线程（`direct_file.cpp` 的 `Pending` 注释） | 剩下的问题：生产里到底走 direct 还是 buffered？在 DLL 里加计数才能回答 | 低（需授权改 DLL） |
| U5 | Flun 的 ring draft cache 对我们 MTP 的适用性 | 读 `glm/glm_dflash.py` + `glm_dflash_prefix.py`，对照我们 `speculative_config` | 低 |
| ~~U6~~ **已回答**：SHM transport **存在**（`via SHM/CE/CE`），P2P transport **不存在**（`P2P is disabled between connected GPUs`） | 23:44 的跨卡冒烟（`_dev/probe/_nccl_comm_smoke.py`） | 已完成 |

**证据等级图例**：`[作者自报]` = 只见于对方 README/文档；`[有仓库原文]` = 我在克隆里读到原文；`[有可复现工件]` = 对方仓库里有 bench 数据/脚本；`[推算]` = 由容量算术得出；`[我们复测]` = 我们自己跑出来的。本报告不以任何 `[作者自报]` 数字作为优劣判据。

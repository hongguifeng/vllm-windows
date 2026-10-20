# EXL3 + MTP 移植到 Windows fork：可行性报告

日期 2026-10-20。目标：把 recipe 镜像里的 EXL3 5.05 bpw + MTP 栈（vLLM 0.32.0rc1.dev249+g18305948a，
Morrowmake `vllm-cmp170hx`）移植到 `D:\code\vllm-windows`（SystemPanic/vllm-windows，
`flash-next-win`），在 Windows 上原生跑 2×CMP 170HX。

全部结论都来自本机实测或本仓库/镜像源码，未跑任何会中断现状服务的实验。

---

## 0. 结论

**可行，而且最关键的未知已经证伪为"能"：**

1. `csrc/exl3` 已在 Windows 上**编译并链接成功**（MSVC 14.44 + CUDA 13.3 + torch 2.11+cu130，sm_80），
   产物 `_exl3_C.pyd` 4,783,616 B，`build_info()` 与运行时完全匹配。
2. 19 个 op schema 全部注册；CPU 算子返回正确结果；`ple_decode_out` 在 **cuda:1（170HX，74 SM）**
   上真实执行，输出 512/512 finite。
3. 剩余工作是 Python 集成（约 5.3k 行）+ PLE 基类对接 + MTP 两个 hook + 权重搬迁，
   **估 8–12 人日**；如果采用 PLE=ram 捷径，可压到 **5–8 人日**（见 §5）。
4. **不要期待更快**：Windows 同样没有 P2P，TP2 每层 all-reduce 仍是瓶颈。
   预期 c1 50–90 / c4 150–250 tok/s，与 WSL 现状（72.5 / 300.8）同档或略低。
   移植买到的是"原生 Windows + 与 :9393 同栈 + PLE 变成 TP-aware"，不是吞吐。

---

## 1. 移植面（量化）

| 部分 | 文件 | 行数 | 性质 |
|---|---:|---:|---|
| `csrc/exl3`（CUDA 扩展） | 24 | 7,338 | 纯 CUDA/C++，无 cutlass 依赖，POSIX 只在 `ple_read.cpp` |
| `vllm/sm80/kernels/exl3/*`（quant/loader/ops/utils/dense_mma/prefill_tc/step_phase） | 7 | 2,859 | 自包含，只 import 上游 vLLM API |
| `vllm/sm80/{__init__,plugin,envs,guard}` | 4 | 1,445 | 只需 `_register_exl3()` 与 `VLLM_EXL3_*` 子集（14 个） |
| `vllm/models/qwen4_exp/nvidia/exl3_ple.py` | 1 | 948 | PLE 的 EXL3 实现，依赖 POSIX IO |
| 模型侧 hook（`ngram_embedding.from_quant_config` 等） | 3 | ~10 | 每处 2–4 行 |
| MTP hook（`set_mtp_start_layer_idx`、`mtp_shares_target_embeddings`） | 1 | ~15 | Windows `mtp.py` 缺这两个 |
| 注册 | — | 2 | `register_quantization_config("exl3")` + `register_model_loader("exl3")` |

不需要的：`sm80/{attention,distributed,fla,kv_planning,layers,platform,worker,spec_decode}`、
`kernels/{attention,indexer,kda,marlin_moe,mhc,moe,thin_gemm}` —— 它们服务 GLM-5/稀疏 MLA/P2P，
EXL3 dense/MoE 一个都不 import（已逐文件核对 import 闭包）。

---

## 2. 已完成的验证

### 2.1 构建 spike（决定性）

环境：MSVC 14.44.35207、CUDA 13.3.73、torch 2.11.0+cu130、Python 3.12.13、ninja 1.13.2、
`TORCH_CUDA_ARCH_LIST=8.0`、`MAX_JOBS=6`。源码在 `_scratch/exl3_win/src`，
构建脚本 `_scratch/exl3_win/build_win.py`。

结果：**12 个 TU 全部编译、链接出 `_exl3_C.pyd`（4.78 MB），耗时约 4 分钟**
（11:34:29 → 11:38:32）。`gemm.cu` 是最慢的一个（约 3.5 分钟），其余 10–40 s/TU。

`build_info()`：

```
{'abi_version': 1, 'torch_version': '2.11.0', 'cuda_version': 13030, 'cxx11_abi': 1}
```

`_runtime_build_info()` 期望的正是 torch 2.11.0 / CUDA 13.x（比较时按主版本归一）/
`torch._C._GLIBCXX_USE_CXX11_ABI`——Windows torch 报 `True`，所以扩展必须报 1（见 §2.2 #3）。
四项全部对齐后，`ops.ensure_extension()` 的 `build_info` 校验会通过；剩余唯一不通过的是
`PleHostReader` class（spike 未移植 POSIX 读取层）。

### 2.2 Windows 适配只有三处

| # | 问题 | 必须的改动 |
|---|---|---|
| 1 | CUDA 13.3 的 CCCL 拒绝 MSVC 传统预处理器：`fatal error C1189: ... pass /Zc:preprocessor` | nvcc 加 `-Xcompiler=/Zc:preprocessor`，cl 加 `/Zc:preprocessor` |
| 2 | `windows.h`/`rpcndr.h` 把 `small`、`large` 定义成宏 → torch 头 `c10/cuda/CUDACachingAllocator.h:105`（`bool small`）与 `dense_mma.cu:471`（`int large`）直接编译失败 | 每个 TU 强制先 include 一个 `win_compat.h`（先 include `windows.h` 占住 include guard，再 `#undef small/large/near/far/interface`）；nvcc 用 `--pre-include=`，cl 用 `/FI` |
| 3 | `module.cpp` 的 `build_info()` 引用 `_GLIBCXX_USE_CXX11_ABI`（MSVC 无此宏） | `#if defined(_MSC_VER) && !defined(...)` → **定义为 1**：Windows torch 的 `torch._C._GLIBCXX_USE_CXX11_ABI` 是 `True`，而 `ops._runtime_build_info()` 会逐项比较（定义 0 会被判 mismatch 并拒绝加载） |

另外把 Linux 构建参数换成 Windows 等价物：`-std=gnu++20` → `/std:c++20`（nvcc `-std=c++20`）、
去掉 `-ffile-prefix-map`、`.abi3.so` → `.pyd`、`is_python_module=True`（源码里有 `PyInit__exl3_C`）。
`Py_LIMITED_API` 可省。

### 2.3 运行时验证（Windows，真卡）

```
0 NVIDIA GeForce RTX 3090      sm_86  82 SM  free=23336MiB
1 NVIDIA CMP 170HX 64G         sm_80  74 SM  free=63701MiB   <- 74 = dense_mma 的 VALIDATED_SMS
2 NVIDIA CMP 170HX 64G         sm_80  74 SM  free=63701MiB
```

- 19 个 schema 全部可解析（含 `set_dense_mma`/`linear_mma_plan` 等快速路径）。
- CPU 算子：`linear_workspace_size(1,128,128)=256`、`moe_workspace_size(...)=229700`、
  `linear_splitk_splits(...)=2`。
- GPU 内核：`ple_decode_out`（dim=128, K=6, rows=4）在 cuda:1 执行成功，
  `out` 512/512 finite，`row0[:6] = [0.181, -0.551, 0.922, -0.734, 1.383, 0.867]`。
- 唯一缺的是 `torch.classes._exl3_C.PleHostReader`（POSIX 版 `ple_read.cpp` 未移植），
  spike 里用 stub 替掉；`ops._validate_extension()` 会因为这个 class 缺失而拒绝加载，必须补。

> 注意：Windows 上 `torch.cuda.mem_get_info()` 对 170HX 读数不可信（两张卡都报 63.7 GiB free，
> 实际 WSL 里已占 60 GiB），小分配能成功，但任何"按 free 显存做决策"的逻辑不要照搬。

---

## 3. 还没解决的三块

### 3.1 PLE 读取层（最大的一块）

`exl3_ple.py` 的 SSD 路径用了 `mmap` + `mincore`（page-cache residency）、`os.preadv`、
`os.posix_fadvise`、`os.O_CLOEXEC` —— Windows 全没有；`ops.ple_read_rows`/`PleHostReader`
来自 POSIX 的 `ple_read.cpp`（`pread`/`nanosleep`/`sys/uio.h`，549 行）。

**但本仓库已经有现成的 Windows 等价物**：`vllm/models/qwen4_exp/nvidia/ple_ssd_io_win.c`
用 IOCP + overlapped `ReadFile` 做定长行读取，行宽是参数（≤ 4096 B），
已用 `_dev/bin/_ple_ssd_io_win_build.ps1` 编译成 `ple_ssd_io_win.dll` 并在用。
EXL3 的 PLE 行是 **122 B**（见 §4），完全落在它的能力里。

两条路：
- **A（推荐，MVP）**：`VLLM_EXL3_PLE_PLACEMENT=ram`。把 36.4 GiB 压缩表放主机内存，
  用 UVA 视图直接喂 `ple_decode_out`，**不做 SSD 行读取**。Windows 原生
  `is_pin_memory_available()==True`、`is_uva_available()==True`（已实测）。
  仍需要一层很小的 shim（实测 Windows 没有 `os.pread/preadv`、`os.O_CLOEXEC`、
  `os.posix_fadvise`、`os.O_DIRECT`）：`setup_file()` 无条件调用
  `_log_residency()`→`mincore`，`Exl3PLERowReader.__init__` 用 `os.O_CLOEXEC` 打开文件，
  `read_tensor()` 读少量 auxiliary 用 `os.preadv`——这三处改成 Windows 版本/空实现即可
  （约 20–40 行）。前提是能 `cudaHostAlloc` 36 GiB（88 GiB 总内存，紧但可行；失败就退回 B）。
- **B（正式）**：把 `ple_read.cpp` 的接口用 IOCP 重写（约 250–350 行 C，
  复用 `ple_ssd_io_win.c` 的 slot/续读逻辑），并在 `module.cpp` 里注册
  `PleHostReader` 自定义 class；Python 侧的 `mincore` 诊断路径改成 Windows 空实现。

### 3.2 PLE 基类 API 已经分叉

recipe 把基类挪到 `common/ngram_embedding.py` 并改了协议：

| | Windows（`nvidia/ngram_embedding.py`） | recipe（`common/ngram_embedding.py`） |
|---|---|---|
| 预取 | `start_prefetch(hidden_states, ngram_ids)` | `start_prefetch(ngram_ids)` + `finalize_prefetch(total_tokens)` |
| 构造 | 无 `reduce_results` | `PLEVocabParallelEmbedding(..., reduce_results=)` |
| 选型 | `from_quant_config()` 只认 FP8/ModelOpt/INC | 先 `getattr(quant_config, "get_ple_embedding_method", None)` |

Windows 的 `PLEVocabParallelEmbedding` 也没有 `reduce_results` 参数。
所以不能只搬 `exl3_ple.py`，要么把 recipe 的 `common/ngram_embedding.py`（747 行）+
`nvidia/ngram_embedding.py`（474 行）+ `ple_layer.py`（417 行）一起搬过来并适配调用点，
要么给 `exl3_ple.py` 写一层适配。这是**工作量最不可控**的一块（2–4 人日）。

### 3.3 MTP × EXL3

Windows `mtp.py` 有通用的 draft quant 配置，但缺 recipe 的两个 hook：
`set_mtp_start_layer_idx`（draft 层号接在 target 之后）与 `mtp_shares_target_embeddings`
（决定 draft 复用 target 的 embedding/lm_head）。EXL3 loader 靠这两个值判定
`shared`/`is_draft` 张量归属（`loader.py:128-150`），缺了会报 missing/unexpected。

### 3.4 版本血统

Windows 是 0.29 代（没有 `vllm/v1/core/kv_cache_planning.py`，没有 `common/` 那次重构），
recipe 是 0.32 代。已逐项核对的部分：`QuantizationConfig` 钩子、`FusedMoEMethodBase`/
`modular_kernel`/`RoutedExperts`、`DefaultModelLoader`、`register_*`、`forward_context`、
`breakable_cudagraph` —— 签名都兼容（低风险）；`--load-format exl3` 也已看代码
（`load_format` 是普通 `str`，`register_model_loader("exl3")` 即可，低风险）。
未核对：KV planner 与 spec-decode 的细节。

---

## 4. 权重与运行前提

- `~/qwen38-recipe/models/Qwen3.8-Flash-Next-exl3` 共约 **115 GiB**，在 WSL ext4 里。
  Windows 侧要复制到 NTFS（D: 剩 489 GiB），`\\wsl$` 的 9p 读 115 GiB 不可接受。
- PLE 表（`ngram_embedding.safetensors`）：`rows=320001536`、`K=6`、`row_dim=160`、
  codec `mul1` → 每行 `1 + 160*6/16 = 61` 个 int16 = **122 B**，表 **36.4 GiB**。
- 2×170HX = 127.2 GiB；EXL3 快速 kernel 的验证条件是 **TP=2 / PP=1 / 74 SM**，本机正好满足。
- Windows TP2 走 pynccl + 自建 `nccl.dll`（NCCL 2.29.7，已实测可用，
  `ncclCommInitRank` 0.40 s，transport `SHM/CE`，`P2P is disabled`）。
- 设备顺序：0=3090，1/2=170HX；启动要 `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2`。

---

## 5. 方案对比

| 方案 | 工作量 | 结果 | 判断 |
|---|---|---|---|
| **A. 直接移植**（本报告） | 8–12 人日（MVP 5–8） | 原生 Windows，TP2，EXL3+MTP | **推荐** |
| B. 合并 `morrow/ampere` | — | — | 不可行：与 HEAD 分叉 20,425 vs 570 commits，merge 冲突不可控 |
| C. 权重转成 AutoRound/GPTQ | 数天 + 校准 | 质量下降，且 5.05→3 bpw 不是同一模型 | 不划算；bf16/fp8 放不下（364/182 GiB） |
| D. 维持 WSL + 代理 | 0 | 现状 72.5 / 300.8 tok/s | 速度已最好；只有"必须原生"时才需要 A |

一个附带收益：Windows 现有 PLE SSD 被 `ple_ssd.py` 硬限制在 `TP=DP=1`
（`raise ValueError("PLE SSD offload currently requires TP=DP=1")`），
而 EXL3 的 PLE 是按 rank 切行（`owner_lo/owner_hi`）的 TP-aware 实现 ——
移植 EXL3 顺带解决"2 卡跑 PLE 模型"这件事。

---

## 6. 分阶段计划（含验收）

| 阶段 | 内容 | 估时 | 验收 |
|---|---|---:|---|
| S0 | 把 Windows 构建脚本落进仓库（`csrc/exl3/build_standalone.py` 加 Windows 分支 + `_dev/bin/_build_exl3.ps1`） | 0.5 d | `python -c "import vllm._exl3_C"` 通过，`build_info()` 匹配 |
| S1 | 搬 `vllm/sm80/kernels/exl3/*`、env 子集、plugin 注册、`ngram_embedding.from_quant_config` 钩子 | 1–2 d | `Exl3Config()` 可构造；`--load-format exl3` 被接受；dummy 权重下模型能 build |
| S2 | loader 干跑：只读 manifest/header，校验 115 GiB 权重的张量名、分片、TP 切分（不驻留） | 0.5–1 d | 所有 target tensor 都有归属；missing/unexpected 为 0 |
| S3a | PLE=ram：搬 `exl3_ple.py` 的 ram 分支 + 基类适配（最小改法） | 1–2 d | 单层 PLE 前向与 WSL 同输入输出对齐（bf16，误差 0） |
| S3b | （可选）IOCP 版 `PleHostReader`，切回 `PLE_PLACEMENT=ssd` | 2–3 d | 320M 行随机抽样 1e5 行，与 ram 路径逐字节一致 |
| S4 | MTP 两个 hook + draft 加载 + TP2 端到端 | 2–3 d | `/health=200`，MTP 接受率与 WSL 同档，c1/c4/prefill 有数 |
| S5 | 正确性/性能对照（同 prompt、同采样，与 WSL 逐 token 比） | 1 d | 前 200 token 完全一致；性能表落文档 |

MVP（S0–S2 + S3a + S4）= **5–8 人日**。S3b 可以后补。

---

## 7. 风险表

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| PLE 基类合并把 Windows 现有 AutoRound PLE 改坏 | 中 | 高 | 单独分支；S3a 只加不改，保留 `ple_ssd.py` 路径；回归 :9393 |
| `cudaHostAlloc` 36 GiB 失败 / 内存不足 | 中 | 中 | 退回 S3b（SSD）；或先 `PLE_PLACEMENT=ssd` 用 Python 线程读（慢但能跑） |
| 0.29↔0.32 API 细节（KV planner/spec decode） | 中 | 中 | S1/S4 用最小配置（先关 MTP、单序列）逐步开 |
| 性能低于 WSL（pynccl 小消息延迟更差） | 中 | 低 | 先测 c1；不行就继续用 WSL 跑生产，Windows 只做备份 |
| MTP×EXL3 在 TP2 下数值不一致 | 低 | 中 | S5 逐 token 对照；先 `SPEC_N=0` 跑通再加 |
| 115 GiB 复制/加载时间 | 高 | 低 | 一次性；NVMe 复制约 10–20 min |

---

## 附录 A：证据位置

- 构建 spike：`_scratch/exl3_win/{build_win.py,src/,build/_exl3_C.pyd,build.log}`
  （`src/ple_win_stub.cpp` 是 POSIX PLE 读取层的占位；`src/win_compat.h` 是宏补丁）
- 运行 smoke：`_scratch/exl3_win/{smoke_ops.py,smoke_gpu.py}`
- recipe 源码副本：`_scratch/recipe_src/{exl3,sm80,qwen4_exp,setup.py,CMakeLists.txt}`
- Windows TP/PP 实测：`_dev/docs/PP_VS_TP_WINDOWS.md` §2.5–2.6
- WSL EXL3 现状：`_dev/docs/EXL3_RECIPE_WSL_STATUS.md`

## 附录 B：关键命令

```powershell
# 构建（约 4 分钟）
. .\_dev\bin\_env.ps1
$env:TORCH_DONT_CHECK_COMPILER_ABI='1'; $env:MAX_JOBS='6'
.\.venv\Scripts\python.exe _scratch\exl3_win\build_win.py

# 注册/GPU smoke
.\.venv\Scripts\python.exe _scratch\exl3_win\smoke_ops.py
.\.venv\Scripts\python.exe _scratch\exl3_win\smoke_gpu.py
```

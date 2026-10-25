# EXL3 + MTP 移植落地状态（Windows fork）

日期 2026-10-20。配套可行性报告：`_dev/docs/EXL3_WINDOWS_PORT_FEASIBILITY_2026-10-20.md`。
本轮目标：把 EXL3 的 CUDA 扩展、Python 量化/加载栈、环境变量与注册落进
`D:\code\vllm-windows`，并让 `vllm._exl3_C` 在 Windows 上构建、加载、通过 ops 校验。

---

## 1. 已落地

### 1.1 CUDA 扩展（S0）

| 路径 | 内容 |
|---|---|
| `csrc/exl3/` | recipe 的 24 个源文件（`*.cu/*.cuh/*.h/*.cpp`，7,338 行） |
| `csrc/exl3/win_compat.h` | 强制先 include `windows.h` 并 `#undef small/large/near/far/interface`（否则 torch 头与 `dense_mma.cu` 编译失败） |
| `csrc/exl3/module.cpp` | `_GLIBCXX_USE_CXX11_ABI` 在 MSVC 下定义为 **1**（匹配 Windows torch 的 `True`，否则 `ops._runtime_build_info()` 判 mismatch） |
| `csrc/exl3/ple_read_win.cpp` | POSIX `ple_read.cpp` 的 Windows 版：`ple_read_rows` + `PleHostReader`（`_lseeki64`/`_read`，同 owner/vocab 语义、同零填充、同重试退避） |
| `csrc/exl3/build_standalone.py` | 跨平台构建脚本：Windows 走 MSVC 分支（`/std:c++20 /bigobj /utf-8 /Zc:preprocessor`、`--pre-include=win_compat.h`、`is_python_module=True`、输出 `.pyd`），Linux 分支保持原样 |
| `_dev/bin/_build_exl3.ps1` | 构建入口：source `_env.ps1`、设 `TORCH_DONT_CHECK_COMPILER_ABI=1` 与 `MAX_JOBS`、产物默认落 `vllm/` |

构建：`powershell -File _dev\bin\_build_exl3.ps1`（首次约 5 分钟，增量 <1 分钟）。
产物：`vllm/_exl3_C.pyd`（4,837,376 B；MSVC 14.44 + CUDA 13.3 + torch 2.11，sm_80）。
`.gitignore` 的 `*.py[cod]` 已覆盖 `.pyd`，不会误跟踪。

### 1.2 Python 栈（S1）

| 路径 | 内容 |
|---|---|
| `vllm/sm80/__init__.py` | 包说明（无 import 副作用） |
| `vllm/sm80/envs.py` | 14 个 `VLLM_EXL3_*` 变量 + `register_env_vars()`（撞名即报错） |
| `vllm/sm80/plugin.py` | `register()`：`register_quantization_config("exl3")` + `register_model_loader("exl3")`，幂等 |
| `vllm/sm80/kernels/__init__.py` | 包说明 |
| `vllm/sm80/kernels/exl3/*.py` | recipe 的 7 个文件（quant 950 / loader 270 / ops 731 / utils 382 / dense_mma 151 / dense_prefill_tc 316 / step_phase 59 行） |
| `vllm/envs.py` | 文件末尾调用 `register_env_vars(environment_variables)`（+5 行） |
| `vllm/plugins/__init__.py` | `load_general_plugins()` 末尾调用 `vllm.sm80.plugin.register()`（+5 行），保证 API server/engine core/worker 都先注册再解析 quant |

---

## 2. 验证（本轮实测）

```
$ . _dev\bin\_env.ps1; $env:PYTHONPATH='D:\code\vllm-windows'
$ python -c "...load_general_plugins(); Exl3Config(); ensure_extension()..."
vllm D:\code\vllm-windows\vllm\__init__.py
Registered model loader <class 'vllm.sm80.kernels.exl3.loader.Exl3ModelLoader'> with load format `exl3`
sm80 plugin registered (EXL3 quantization and load format)
exl3 method True
loader Exl3ModelLoader
placement ssd retries 3
cfg Exl3Config exl3 ssd
extension 256 class True
```

- `build_info()` = `{abi_version: 1, torch_version: 2.11.0, cuda_version: 13030, cxx11_abi: 1}`，
  与 `ops._runtime_build_info()` 逐项一致；12 个 operator schema 与 `PleHostReader` class 全部存在。
- `ple_read_rows` / `PleHostReader` 功能测试（`_dev/probe/_exl3_ple_read_win_test.py`）：
  64 行 × 122 B、`base=8` 偏移的表，行字节逐字节一致；owner 外与 head 词表外零填充；
  `enqueue/take` 返回 `[sequence, count, status, failed]` 正确、可 drain。
- `ple_decode_out` 之前已在 cuda:1（170HX，74 SM）跑通（512/512 finite）。

日志：`_dev/out/exl3_port/S0_S1_verify.log`；spike 阶段日志：`_scratch/exl3_win/build.log`。

---

## 3. 与 recipe 的差异（Windows 侧）

| 点 | recipe | 本移植 |
|---|---|---|
| 扩展命名 | `_exl3_C.abi3.so`（`is_python_module=False`） | `_exl3_C.pyd`（`is_python_module=True`，`PyInit__exl3_C`） |
| 预处理 | GCC 传统预处理器 | `/Zc:preprocessor`（CUDA 13.3 CCCL 强制） |
| `small`/`large` | 无冲突 | `win_compat.h` 强制 include + undef |
| `cxx11_abi` | 0（有 `_GLIBCXX_USE_CXX11_ABI` 宏） | 1（MSVC 无此宏，取 torch 的值） |
| PLE host reader | `ple_read.cpp`：`pread`/`preadv` + 线程池 | `ple_read_win.cpp`：`_lseeki64`/`_read`，调用线程同步读；IOCP 优化留作后续 |
| 构建入口 | docker `native_build.py` | `_dev/bin/_build_exl3.ps1` + `build_standalone.py` |
| Python 注册 | `vllm.general_plugins` entry point | `load_general_plugins()` 内直接调用（免打包 entry point） |

---

## 4. 下一步

### S2（已完成）权重落地与 manifest 干跑
- checkpoint 已完整复制到 `D:\models\Qwen3.8-Flash-Next-exl3`（123,252,418,628 B，与 WSL 源一致，25 个文件）。
- NTFS 上跑 manifest 干跑（只读 header，不驻留权重），**18.81 s** 通过：
  `tensors=304113 matrices=75751 ple_tables=1 unused=54`，payload **114.61 GiB**；
  PLE 表 `rows=320001536 dim=160 words=61 row_bytes=122 heads=16`；
  TP2 的 rank0 切分（`aligned_shard`）在 `lm_head`/`in_proj_qkv`/`in_proj_z` 上都落在 H128 边界。
  脚本：`_dev/probe/_exl3_manifest_win_test.py`，日志 `_dev/out/exl3_port/S2_manifest.log`。
  （同一脚本走 `\\wsl.localhost` 的 UNC 路径要 202 s，NTFS 快 10 倍。）

### S3（已完成）PLE 接入
落地内容：
- `vllm/models/qwen4_exp/nvidia/exl3_ple.py`（recipe 的 948 行移植）：基类改 `nvidia.ngram_embedding`、
  `start_prefetch(hidden_states, ngram_ids)`、`forward()` 走 `_finalize_prefetch`、
  `reduce_results` 缺省兼容、`O_CLOEXEC`/`O_BINARY` 兼容、`posix_fadvise` 有则用、
  `mincore` 在 Windows 返回 `(0, 0)`、`preadv` → `lseek`+`read`。
- `ngram_embedding.py`：`from_quant_config()` 先看 `quant_config.get_ple_embedding_method()`；
  类选择点先看 `quant_config.ple_embedding_cls`（两处共 ~10 行）。
- `Exl3Config` 在 recipe 里本来就带 `get_ple_embedding_method()` / `ple_embedding_cls` / MTP 两个 hook，
  无需改动。

验证（`_dev/probe/_exl3_ple_module_win_test.py`）：
- `ModelConfig` 解析 checkpoint → `quantization=exl3`；`VllmConfig` → `Exl3Config`。
- 真实构造 `Qwen4ExpNGramEmbedding`（TP=1，cuda:1）→ 选中 `Exl3PLEEmbedding`，
  日志 `placement=ssd owner=[0,320001536) row_bytes=122 capacity=512`。
- `setup_file` 成功 join 真实 36 GiB 表（消耗 5 个 key）。
- `start_prefetch` + `forward` 输出 `(4, 2560) bf16` 且 finite；
  与对同一批行的直接 `ple_decode_out` 比对 **10240 个值 0 mismatch**。
- 另外 `_exl3_ple_decode_win_test.py` 用 numpy 复刻 `ple.cu` 的 mul1 码本 + ring 位序，
  对真实行做逐位对照：1120 值 0 mismatch（含 owner 外零填充）。

**过程中修掉一个真实的并发 bug**：Windows 没有 `pread/preadv`，`lseek`+`read` 会共享
文件指针；PLE 行读取跑在线程池上，最初出现随机错行（每次运行 mismatch 数不同）。
`pread_exact_into()` 的 Windows 分支加了一把全局锁后变为确定性且逐位正确。

### S4（已完成）MTP
`vllm/models/qwen4_exp/nvidia/mtp.py` 补了三处：
- `_make_draft_vllm_config()` 里 `draft_quant_config.set_mtp_start_layer_idx(mtp_start_layer_idx)`；
- `Qwen4ExpMultiTokenPredictor` 的 `embed_tokens` 与 `Qwen4ExpMTP` 的 `lm_head`：
  `quant_config.mtp_shares_target_embeddings(config)` 为真时用 `PPMissingLayer()`。
真实 draft 已验证：`loader.py:282` 打印
`EXL3 packed format ... draft model ... sources: 3131 consumed, 0 file-backed, 300982 unused`，
随后 `Capturing model for speculator...` 成功，服务 `health=200`。

### S5（已完成）端到端（Windows 原生，2×170HX，TP=2）
- 启动参数见 `_dev/probe/_exl3_tp2_smoke.ps1`（`--load-format exl3 --tensor-parallel-size 2
  --enable-expert-parallel --disable-custom-all-reduce --mm-encoder-tp-mode data`，
  `CUDA_VISIBLE_DEVICES=1,2`，`NCCL_P2P_DISABLE=1`）。
- 权重加载约 5 分钟（114.61 GiB，逐分片惰性打开）；`init engine` 28–49 s。
- KV：`gpu_memory_utilization=0.95`（eager）→ **1,028,915 tokens**（32k 上下文 31.4×）；
  `0.92`（PIECEWISE graphs）→ 908,765 tokens。
- 生成质量：中文回答连贯（首条 chat completion 正常）。
- 性能（128 输出 token，`_dev/probe/_exl3_ws_bench.py`）：

| 配置 | c1 | c4 聚合 | 备注 |
|---|---:|---:|---|
| eager + 库默认 kernel | 7.8 | 29.2 | 慢基线 |
| eager + recipe kernel 集 | 7.6 | 28.7 | kernel 不是瓶颈 |
| PIECEWISE graphs + recipe kernel 集 | 32.5 | 107.9 | 图捕获是关键 |
| + MTP k=4 | **47.8** | **136.6** | 当前最好 |
| WSL（对照，MTP k=4 + host all-reduce） | 72.5 | 300.8 | 无 P2P |

- 冷预填（2212 token）：1,513 tok/s（WSL 约 1,955）。
- MTP 生效证据（服务端 metrics）：`Mean acceptance length 2.47–2.77`，
  `Avg Draft acceptance rate 36.7–44.1%`，per-position `0.70/0.52/0.30/0.24`。
- 瓶颈：decode 时 GPU 利用率只有 ~24%（峰值 92–98%），是**每层 all-reduce 的等待**。
  Windows 只能用 `PYNCCL`（无 P2P，host-bounce）；WSL 用的是 fork 的
  `vllm/sm80/distributed/host_shm_all_reduce.py`。这是剩余 1.5–2.2× 差距的最可能来源，
  属于后续优化，不是功能缺口。

### 运行/停止
- 启动：`powershell -File _dev\probe\_exl3_tp2_smoke.ps1 -MemUtil 0.92 -Mtp -MtpTokens 4`
  （服务 `http://127.0.0.1:9400`，`Qwen3.8-Flash-Next-EXL3`，TP2 + MTP k=4 + PIECEWISE graphs）。
- 停止：`powershell -File _dev\probe\_exl3_tp2_stop.ps1`（按进程树递归杀，避免 EngineCore 孤儿占卡）。
- **2026-10-20 晚：能力对照测试已结束，所有服务已停、GPU 空闲**。重新启动任一服务前先确认端口/显卡。
- WSL 的 recipe 服务已停（TP2 需要独占两张 170HX）；要恢复它：
  `wsl -d Ubuntu -- bash -lc 'cd ~/qwen38-recipe && bash start.sh'`（先停 Windows 侧）。

### 能力对照（SWE-bench Verified）
EXL3 5.05bpw、AutoRound 3bpw、推理预算（cap × step limit × effort）与 DeepSeek 参照的
完整对照见 [`agent_bench/RESULTS_qwen38_flash_next_swebench.md`](../../agent_bench/RESULTS_qwen38_flash_next_swebench.md)：
同协议 1536/60 下 EXL3 31/66 vs AutoRound 38/66（p=0.14）；同样的 xhigh 30000/200 下
**EXL3 64/66 vs AutoRound 61/66**（discordant 4/1，p=0.38）。早期所有 47–61% 的低分
主要是预算假象，不是量化或模型能力。

### 端到端过程中修掉的 Windows/版本适配问题
1. `WeightsMapper.map_name` → Windows 只有 `_map_name`（`quant.py`）。
2. 共享专家 640 不能被 TP2 的 H128 块整除 → `qwen3_next.py` 补
   `getattr(quant_config, "shared_expert_shard_group_size", ...)`，配合 `--enable-expert-parallel`
   走复制路径。
3. `lm_head` 在 Windows 模型里没传 `quant_config` → checkpoint 的 `lm_head.*` 无主；
   按 `quant_config.quantize_lm_head` 条件传入（不影响 AutoRound）。
4. 一次映射全部分片会耗尽 Windows commit（`OSError 1455`）→ loader 改为**逐分片惰性打开**。
5. `WorkspaceManager.get_persistent_resource` 在 Windows 不存在 → EXL3 scratch 改模块级缓存。
6. repo 的 `vllm/vllm_flash_attn/` 缺构建产物 `layers/`、`cute/`、`ops/`
   （`.gitignore` 忽略，安装时才拷进 site-packages）→ 从 build 目录补进仓库树。
7. `--enable-expert-parallel` 是 EXL3 TP2 的必需项（recipe serve.sh 一直在传）。

---

## 5. 当前环境事实（可复现）

- Windows：MSVC 14.44.35207、CUDA 13.3.73、torch 2.11.0+cu130、Python 3.12.13、ninja 1.13.2。
- GPU：0=3090（sm_86）、1/2=CMP 170HX（sm_80，**74 SM**，正好等于 `dense_mma.VALIDATED_SMS`）。
- Windows torch 的 `is_pin_memory_available()` 与 `is_uva_available()` 均为 True（PLE=ram 可用）。
- 运行 Python 时必须 `PYTHONPATH=D:\code\vllm-windows`，否则会 import 到
  `.venv/Lib/site-packages/vllm` 的旧上游拷贝（0.29）。

---

## 6. 从 git 还原到一台干净机器

仓库里**只有源码**：`git ls-files '*.pyd' '*.dll'` 为空（`.gitignore` 的 `*.py[cod]`、`_dev/out/`
与 `_scratch/` 规则），模型权重也不在库里。clone 之后按顺序补齐，缺一项就起不来：

| # | 补什么 | 用什么 | 落到哪 |
|---|---|---|---|
| 1 | 工具链 + venv | `_dev\bin\_env.ps1`（自动发现 MSVC / Windows SDK / CUDA 13.3） | 环境变量；跑任何命令都带 `PYTHONPATH=D:\code\vllm-windows` |
| 2 | NCCL | `_dev\bin\_build_nccl.ps1` | `C:\nccl-windows\install\bin\nccl.dll`（只有 TP/PP 需要；`_env.ps1` 会把它塞进 `VLLM_NCCL_SO_PATH`） |
| 3 | vllm 自身的 C/CUDA 扩展 | `_dev\bin\_flashtest_build.ps1`（构建 `.flashnext-root` worktree + venv，生产用的就是它）或 `_dev\bin\_build.ps1`（构建本仓库树，装进 `.venv`）；都是 `pip install . --no-build-isolation`，首次约 37 分钟，之后 ninja 增量 | 装在各自 venv 的 `site-packages\vllm`；要让**仓库树**自己也 import 得动，把 `_C_stable_libtorch`、`_moe_C_stable_libtorch`、`cumem_allocator`、`fs_io_C`、`spinloop` 这 5 个 `.pyd` 从那里拷到 `vllm\` |
| 4 | flash-attn 的包体 | 把 `vllm\vllm_flash_attn\build\lib.win-amd64-cpython-312\vllm\vllm_flash_attn\` 下的 `layers\`、`ops\`、`cute\`（以及 `_vllm_fa2_C.pyd`，若在）拷回 | `vllm\vllm_flash_attn\`（缺了报 `No module named 'vllm.vllm_flash_attn.layers'`） |
| 5 | EXL3 扩展 | `_dev\bin\_build_exl3.ps1`（首次约 5 分钟，增量 <1 分钟） | `vllm\_exl3_C.pyd`（4.84 MB，需与本机 torch 2.11 / CUDA 13.3 的 ABI 一致） |
| 6 | PLE/SSD 原生读 | `_dev\bin\_ple_ssd_io_win_build.ps1`（`-OutDir` 可改落点） | `_dev\out\ple_ssd_io\ple_ssd_io_win.dll`；服务以**绝对路径**引用它，见 `start_qwen38_flash_next.ps1` 的 `--additional-config` |
| 7 | 权重 | 不在 git | `D:\models\Qwen3.8-Flash-Next-exl3`（123 GB，25 个文件）；`D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`（142 GB，11 个分片，其中 102 GB 那片的 PLE 表走 SSD 流式） |

顺序：1 → 3 → 4 → 5（EXL3 扩展依赖 `_env.ps1` 的编译器与 torch 头），2 与 6 互相独立。
本机现有的事实基准可以直接对照：`vllm\` 下 6 个 `.pyd`（含 `_exl3_C`）、
`vllm\vllm_flash_attn\` 下 `layers/ ops/ cute/ _vllm_fa2_C.pyd`、`_dev\out\ple_ssd_io\ple_ssd_io_win.dll`（106,496 B）。

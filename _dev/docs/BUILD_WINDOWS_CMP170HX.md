# 在 Windows + CMP 170HX 上从源码构建并运行 vLLM

**结论：构建成功，推理跑通。**

- 产出 wheel：`vllm-0.29.1.dev0+g13e844c86.d20260919.cu133-cp312-cp312-win_amd64.whl`（187 MB）
- 安装版本：`vllm 0.29.1.dev0+g13e844c86.d20260919`（Python 3.12，torch 2.11.0+cu130）
- 全量构建耗时：约 **37 分钟**（`MAX_JOBS=10`），0 错误
- 端到端烟测：`SMOKE TEST OK`，eager 与 CUDA Graph 两种模式均通过

## 硬件与工具链

| 项目 | 值 |
|---|---|
| GPU | CMP 170HX = NVIDIA GA100，**sm_80**，70 SM，64 GB（已解锁） |
| 驱动 / CUDA 运行时 | 596.36 / 13.2 |
| 编译用 CUDA Toolkit | 13.3（V13.3.73） |
| 主机编译器 | MSVC 14.44.35207（VS 2022 BuildTools） |
| Windows SDK | 10.0.26100.0 |
| 构建 arch | `TORCH_CUDA_ARCH_LIST=8.0`（只编 sm_80） |

## 一键使用

```powershell
# 端到端烟测（自动按空闲显存推导 gpu_memory_utilization）
& D:\code\vllm-windows\_dev\bin\_run.ps1

# 换模型
& D:\code\vllm-windows\_dev\bin\_run.ps1 -Model Qwen/Qwen2.5-1.5B-Instruct

# 起 OpenAI 兼容服务
& D:\code\vllm-windows\_dev\bin\_run.ps1 -Serve

# 全量/增量重编
& D:\code\vllm-windows\_dev\bin\_build.ps1
```

## 为了让 MSVC 编过，改了 4 个地方

### 1. `csrc/libtorch_stable/moe/grouped_topk_kernels.cu`（第 672、858 行）
MSVC 把 `INFINITY` 展开成 `(float)(1e+300*1e+300)`，触发
`floating-point value does not fit in required floating-point type`。

```cpp
// 改前
const float invalidScoreFloat = -INFINITY;
// 改后
const float invalidScoreFloat = -cuda::std::numeric_limits<float>::infinity();
```
（同文件第 390 行本来就是这种写法，保持一致。）

### 2. `fix_cutlass_msvc.py` — 新增 CuTe `stride.hpp` 补丁
MSVC 在 `/std:c++20` 下解析不了成员别名模板 `typename Lambda::template seq<Shape>`，
回退到外层 `cute::seq`（参数包），报 `error C3545: 'Ints': parameter pack requires
a non-type template argument`。

**不能靠降回 C++17 绕过**：csrc 用了 C++20 指定初始化器
（`cudaLaunchConfig_t config{.gridDim=...}`）。改为直接展开映射：

```cpp
using Seq = std::conditional_t<std::is_same<Major, LayoutLeft>::value,
                               tuple_seq<Shape>, tuple_rseq<Shape>>;
```

补丁会同时作用于两处 cutlass 副本：`.deps/cutlass-src` 与
`.deps/vllm-flash-attn-src/csrc/cutlass`。

### 3. `_dev/patches/fix_flash_attn_msvc.py`（**新增文件**）— `M_LOG2E` 未定义
flash-attn 的 `flash_api.cpp` 报 `error C2065: 'M_LOG2E': undeclared identifier`。
MSVC 只在 `<math.h>` **之前**定义 `_USE_MATH_DEFINES` 才暴露 POSIX 数学常量，
而 flash-attn 是在一堆 torch 头之后才 `#include <cmath>`，此时补宏已经太晚。

按 vLLM 自身 `csrc/mamba/selective_scan_fwd.cu` 的惯用写法插入常量守卫：
```cpp
#ifndef M_LOG2E
#define M_LOG2E 1.44269504088896340736
#endif
```
作用于 `csrc/flash_attn/flash_api.cpp`（宿主）和 `csrc/flash_attn/src/softmax.h`（设备）。

### 4. `cmake/external_projects/vllm_flash_attn.cmake` — 挂载上面的补丁
在 `FetchContent_MakeAvailable(vllm-flash-attn)` 之后调用补丁脚本。
这一步很关键：`.deps/` 被清空重拉时补丁会自动重新应用，**手工改 `.deps` 里的文件会丢**。

## 构建前的两个必做项

```bash
git config --global core.longpaths true
```
不做会在拉 flash-attn 子模块时报 `fatal: Filename too long`。
注意：注册表 `LongPathsEnabled=1` 不够，git 自己的 `core.longpaths` 必须单独开。

```powershell
$env:CODEBUDDY_SAFE_DELETE_ENABLED = '0'
```
WorkBuddy 注入的 `sitecustomize.py` 会接管 `os.unlink`，累计删除约 50 个文件后抛
`SystemExit`，表现为 pip 覆盖 wheel 时莫名的 `[Errno 13] Permission denied`。
所有构建脚本里都要设。

## 运行期的三个坑（与编译无关）

这三条都已写进 `_dev/bin/_run.ps1`，正常用不需要手动处理：

| 现象 | 原因 | 对策 |
|---|---|---|
| `ModuleNotFoundError: No module named 'vllm._C_stable_libtorch'` | 脚本放在仓库根目录，Python 把脚本目录放进 `sys.path[0]`，导入命中了**未编译的源码树** | cwd 切到仓库外，并从 `sys.path` 剔除仓库根 |
| `zmq.error.ZMQError: Address in use (tcp://127.0.0.1:29550)` | 上次被中断的运行留下 EngineCore 子进程占着端口 | `VLLM_ENABLE_V1_MULTIPROCESSING=0` 走单进程，彻底绕开 ZMQ |
| flashinfer `FileNotFoundError: [WinError 2]`（崩在 `jit/cpp_ext.py: run_ninja`） | flashinfer 首次用到 top-k/top-p 采样时 JIT 编译，找不到 `ninja` | `VLLM_USE_FLASHINFER_SAMPLER=0` 回退到原生 PyTorch 采样器；并把 `.venv\Scripts` 加进 PATH |

## 显存注意事项

构建/运行期间本机有一个 `gpu_burn.exe`
（`D:\Portable Program\cmp170hx\gpu-burn-win-v0.2\`）占着 **58 GB 显存 + 100% GPU**，
所以 vLLM 启动时只有 6.56 GiB 可用。

`gpu_memory_utilization` 要按**空闲**显存算，而不是总显存，否则报
`Free memory on device cuda:0 (6.56/63.63 GiB) on startup is less than desired
GPU memory utilization`。`_dev/bin/_run.ps1` 已改为启动时用 `nvidia-smi` 自动推导。

## 实测结果

```
INFO [cuda.py:492] Using FLASH_ATTN attention backend out of potential backends:
                   ['FLASH_ATTN', 'FLASHINFER', 'TRITON_ATTN', 'FLEX_ATTENTION'].
INFO [flash_attn.py:897] Using FlashAttention version 2
INFO [model_runner.py:404] Model loading took 0.93 GiB memory and 3.243 seconds
INFO [gpu_worker.py:626] Available KV cache memory: 4.22 GiB
INFO [kv_cache_utils.py:2032] GPU KV cache size: 368,624 tokens

PROMPT: The capital of France is
OUTPUT: :\n\nA) Paris\nB) Lyon\nC) London\nD) Paris\n...
SMOKE TEST OK
```

- attention 后端自动选中 **FLASH_ATTN（FA2，即本次编译出的 `_vllm_fa2_C.pyd`）**，说明自编译内核生效
- 6 个扩展全部构建成功：`_C_stable_libtorch`、`_moe_C_stable_libtorch`、
  `vllm_flash_attn/_vllm_fa2_C`、`cumem_allocator`、`fs_io_C`、`spinloop`
- `enforce_eager=0`（CUDA Graph）：`cudagraph_mode=FULL_AND_PIECEWISE`，
  `max_cudagraph_capture_size=512`，CUDAGraph 占用 0.18 GiB，同样 `SMOKE TEST OK`
- 解码头速度约 130–195 tok/s（0.5B 模型，且 GPU 被判占用满，**不是**真实性能上限）

## 尚未验证

- FA3 / FA4（Hopper/Blackwell 专用，sm_80 上无意义）
- 量化内核：marlin、scaled_mm 等（编译通过了，但未做数值验证）
- DeepGEMM、FlashMLA 等外部子项目
- 更大模型与并发压测

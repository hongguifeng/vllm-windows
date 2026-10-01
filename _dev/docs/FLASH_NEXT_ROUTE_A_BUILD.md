# 路线 A：在 worktree 上把 Flash-Next 树编成 Windows 版本

记录 `D:\code\vllm-flashtest` 这条分支怎么搭起来、编到哪儿、踩了哪些 MSVC 的坑。
可行性判断本身在 `FLASH_NEXT_WINDOWS_FEASIBILITY.md`，这份只记施工过程。

## 1. 分支构成

`flash-next-win` @ `D:\code\vllm-flashtest`（git worktree，共享主仓库的对象库）：

| commit | 内容 |
|---|---|
| `3ce95f3619` | `origin/main` 基线 |
| `254c2ff5db` | `qwen38-ple-ssd.patch`（13 文件，PLE/SSD + INC RoutedExperts hunk） |
| `e5be8868ec` | cherry-pick `3d3a44e4a8`（fork 的 v0.26.0 Windows 主体，66 文件） |
| `4cb989c1d7` | cherry-pick `b137929e44`（tilelang win32 pin） |
| `5a96be5446` | cherry-pick `160f6b2b76`（torchvision win32 pin + DeepGEMM _C wheel） |
| `2efee6a517` | 修正：非 win32 的 torchvision 别被 hunk 顺手降到 0.26.0 |
| `3b72ba792b` | cherry-pick `4eebec3745`（v0.27.1 + 若干 Windows 修复） |
| `e88f64dd52` | cherry-pick `13e844c86d`（v0.29.0） |
| `1cda82db11` | 修正：build/cuda.txt 非 win32 的 torch 别被降到 2.11 |
| `82d8f5334b` | deepgemm 只取 Windows fork 真有的 submodule |
| `2300d15f11` | deepseek_v4 融合 kernel 在 MSVC 下改成 stub |
| `4317591913` | `-INFINITY` 不进 constexpr；Windows 上 cute 固定到 4.4.2 |
| `1c0bc23087` | Windows 上 CUDA 对象用 C++17；flash-attn 补 `_USE_MATH_DEFINES` |

cherry-pick 的选择理由：`git merge-base 13e844c86d origin/main` = `f5c3cc240b`，fork 线上 22 个
commit 里只有 5 个是 Javier 的 Windows 工作，另外 17 个是上游自己的 commit（`origin/main` 已经有了）。
逐文件 `git apply --check` 量出来的口径：**65 个文件干净通过，16 个需要手工**，其中真正麻烦的
只有 `requirements/cuda.txt`、`CMakeLists.txt`、两个 `*.cmake`、`csrc/fs_io.cpp`、四个 `.cu`、
`shm_broadcast.py`、`flashinfer.py`。

cherry-pick 期间只有一处真冲突：`vllm/v1/attention/backends/flashinfer.py`（上游已经改成 kwargs
风格的调用，fork 那边是旧的整行复制）。解决方式是把 Windows 的降级逻辑写成 kwargs 差量：

- `flashinfer_xqa_batch_decode_with_kv_cache`：flashinfer-windows ≤0.6.11 不认 `q_cu_seq_lens`，
  所以 ragged varlen decode 在 Windows 上不进 XQA，落到 trtllm-gen，并 `warning_once` 说明。
- `trtllm_batch_decode_with_kv_cache`：同样版本不认 `lse`/`return_lse`，Windows 上不给这两个参数，
  并给 DCP 合并路径塞一个零 `lse`，让它继续跑。

## 2. 需要警惕的"静默降级"

fork 的 hunk 是整行替换，落到今天的 `origin/main` 上会顺手把**非 win32** 的 pin 也降掉。已经抓到两处并
修掉：`requirements/cuda.txt` 的 torchvision `0.28.0 → 0.26.0`、`requirements/build/cuda.txt` 的 torch
`2.13.0 → 2.11.0`。win32 这一侧仍然是 torch 2.11.0+cu130（这台机器验证过的版本；上游要 2.13，
`torch-2.13.0+cu130-cp312-win_amd64.whl` 确实存在，真需要时再升）。

另外 `vllm/utils/system_utils.py` 里 `find_loaded_library()` 的上游精确匹配逻辑被换回了旧的宽松版本。
对我们无害（Windows 上没有 `/proc/self/maps`），但它是 Linux 侧的一个回退，别把它带上 PR。

## 3. 环境与构建脚本

- `_dev/bin/_flashtest_env.ps1`：MSVC 14.44.35207 + SDK 10.0.26100 + CUDA 13.3 的等价 vcvars，
  PATH 指向 **worktree 自己的 venv**，主仓库 `.venv`（跑 27B 的那个）完全不参与。
- `_dev/bin/_flashtest_provision.ps1`：`uv venv --python 3.12` + 四步安装。
- `_dev/bin/_flashtest_build.ps1`：configure + 编译，日志 `_dev/out/_flashtest_build.log`。

两个坑值得记住：
1. **`uv venv` 建的 venv 里没有 pip**，所有安装都得走 `uv pip install --python <venv python>`。
2. cu130 索引上 `cmake` 只有 3.25.0、`packaging` 只有 24.1，而 uv 默认**只看第一个提到这个包的索引**，
   所以必须 `--index-strategy unsafe-best-match`，否则解析直接失败。

venv 里的关键版本（都 import 成功）：torch 2.11.0+cu130（看到 2 张卡）、triton-windows 3.6.0、
flashinfer-windows 0.6.11.post3（`flashinfer.norm.gemma_rmsnorm` 在，qwen4_exp 的 QSA indexer 只有
这一个 flashinfer 依赖）、humming-kernels 0.1.15（从 SystemPanic/humming-windows git 现编）、
xformers 0.0.35、winloop 0.6.3。

## 4. 编译期已经踩到的四个坑（都有实测证据）

| 症状 | 根因 | 处置 |
| --- | --- | --- |
| `error: pathspec 'third-party/deep_jit' did not match any file(s)` | FetchContent 按上游的 submodule 列表去初始化，而 SystemPanic/DeepGEMM-windows 只有 `third-party/cutlass` 和 `third-party/fmt`（查了该 commit 的 `.gitmodules`） | WIN32 分支单独给 `GIT_SUBMODULES` |
| `fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu(351): error: expected a declaration` ×5，后面 `absmax/laneId/kNumHeadsQPadded` 全部 undefined | nvcc 前端在 cl.exe 驱动下把这个 unroll 密集的 kernel 的作用域弄丢了（预处理后的 .cu 内容花括号净值为 -1，即提前闭合） | MSVC 下整个 kernel `#else` 掉，只留 ops.h 声明的四个入口做 stub 并 `STD_TORCH_CHECK(false, ...)`；DeepSeek V4 不在 Windows 上服务，`torch_bindings` 仍能链接 |
| `grouped_topk_kernels.cu(856): error: floating-point value does not fit in required floating-point type` | `static constexpr float InvalidScore = -INFINITY;` 在 EDG+MSVC 下不合法（Linux/gcc 不报） | 改成 `-cuda::std::numeric_limits<float>::infinity()`（文件里第 390 行已经这么用了） |
| `cute/stride.hpp(299): error C3545: "Ints": 参数包需要一个非类型模板参数` → `layout.hpp(98) C2062` → 一串 `C2976 cute::Layout 参数太少` | cl.exe 在 `/std:c++20` 下错误解析 `cute::Ints<...>`；同一份 `scaled_mm_c2x.cu`（blob 与 fork 完全相同）配 cutlass 4.7.1 和 4.4.2 都炸 | 单文件实测：换成 `-std=c++17` → `rc=0`，零 error。于是 Windows 上 `CMAKE_CUDA_STANDARD 17`；顺手把 `CUTLASS_REVISION` 在 WIN32 降到 4.4.2 |
| `flash_api.cpp(159): error C2065: "M_LOG2E": 未声明` | MSVC 不暴露 `M_LOG2E` 除非先定义 `_USE_MATH_DEFINES` | `vllm_flash_attn.cmake` 的 WIN32 分支和 target_compile_options 都加上 `/D_USE_MATH_DEFINES` |

`/Zc:preprocessor` 不能拿掉：CCCL（CUDA 13 自带）在传统预处理器下直接 `#error`
（`cccl/v2/cuda/std/__cccl/preprocessor.h`），而 fork 的 Windows 块本来就靠它。试过 `-O2`、`-Ob1`、
`-Xptxas=-O2`、`/permissive-`、`--diag_suppress=2482`，都救不了上面那些错误。

## 5. 还没证明的事

编译通不通是一回事，跑不跑得起来是另一回事。仍然未知：

- QSA（12 层）在 flashinfer-windows 0.6.11 上是否真能执行（只验证了 `gemma_rmsnorm` 这个符号在）。
- humming 的 2/3-bit MoE kernel 在 sm_80 + Windows 上的实际执行（结构烟测只证明了 method 建得起来）。
- GDN 36 层 + `hc_count=4` 的完整前向、CUDA graph 捕获、KV 预算。
- PLE/SSD 的 Windows 读路径 —— 参考实现是 `os.pread` + `posix_fadvise` + `/proc/self/fd` +
  `O_DIRECT` 的 Linux AIO，本机 `FILE_FLAG_NO_BUFFERING` 被环境级拦截（`ERROR_INVALID_PARAMETER`），
  所以只能换成 buffered seek+read，见 `container_ple_ssd.py` 的 `PLESSDNativeReader` 与 `_read_rows`。

## 6. 编译结果：通过

`uv pip install . --no-build-isolation` 到 flashtest venv，rc=0（23:01）。产出的扩展：

```
_C_stable_libtorch.pyd  _moe_C_stable_libtorch.pyd  cumem_allocator.pyd
fs_io_C.pyd  spinloop.pyd  vllm_flash_attn/_vllm_fa2_C.pyd
```

没有 `_vllm_fa3_C.pyd`，因为 FA3 只出 Hopper/Blackwell 的 kernel，sm_80 不需要它。
import 侧全部到位：`vllm.vllm_flash_attn.flash_attn_varlen_func` 可用，
`is_flash_attn_varlen_func_available()` 返回 True（QSA 靠的是 flash-attn，不是 flashinfer），
`INCConfig` + `CUDA_HUMMING_SUPPORTED_BITS = {2,3,5,6,7}` 在树内（PLE 补丁带的 hunk），
`vllm.models.qwen4_exp.nvidia.model` / `ple_ssd` 都能 import。

顺带两个必须记住的坑：

1. `vllm.entrypoints.launchers.api_server.entry.main()` 在平台分支之前无条件 `import uvloop`，
   Windows 上直接 ImportError（fork 那边用 `import winloop as uvloop_impl`）。已修。
2. 结构烟测要用 PLE-free 视图时，`AutoWeightsLoader` 会因为在 checkpoint 里看到
   `layers.1.ple.*` 而模型里没有这个模块而 `ValueError`。checkpoint 里的 PLE 权重是 hardlink
   视图删不掉的（会动到原件），所以走 `WeightsMapper` 的丢弃语义：
   `ple_layer_ids` 为空时把 `".ple."` 加进 `skip_substrs`。已修。

## 7. 结构烟测：走到哪一步

GPU1（空闲），PLE-free 视图，`--enforce-eager --language-model-only`，4096 ctx / 4 seq / util 0.9。

| 阶段 | 结果 |
| --- | --- |
| 48 层全部构造（36 `linear_attention` + 12 `qwen_sparse_attention`，`hc_count=4`） | 通过 |
| Marlin 用于 AutoGPTQ / mixed-precision linear | `Using MarlinLinearKernel` |
| GDN：prefill Triton/FLA（head_k_dim=128）、decode cuda kernel | 通过 |
| MoE method 选择 | `Using HummingIndexedExperts Humming MoE backend` |
| FlashAttention 版本 | `Using FlashAttention version 2` |
| 47.14 GiB 权重加载（12 shard，~19 s/shard） | 通过 |
| `process_weights_after_loading` → humming | **死在这里** |

两个连续的错误：

```
humming/utils/device.py -> build_device_info_extension -> subprocess
FileNotFoundError: [WinError 2] 系统找不到指定的文件      # g++ 不存在
```
修掉之后（见下）：

```
humming/jit/compiler.py:_compile -> may_build_nvrtc_compile_binary
RuntimeError: Could not locate libnvrtc.so in CUDA path
```

## 8. humming 在 Windows 上的真实缺口

humming 的 upstream 是 `inclusionAI/humming`；`SystemPanic/humming-windows` 的 `main` 就是
`v0.1.15`（`git ls-remote` 只有到 v0.1.15 的 tag，没有更新的分支），upstream 到 `v0.1.16`。
把 v0.1.16 的 `utils/nvrtc.py` 与 `utils/device.py` 拉下来 grep `win32|msvc|nvrtc64|dll|.pyd`：**零命中**。
也就是说 humming 的运行时原生件从来没被移植到 Windows：

| 原生件 | 用途 | Windows 下的障碍 |
| --- | --- | --- |
| `_device_info`（CPython 扩展） | `prepare_layer_config` 要的设备属性 | 运行时用 g++ 编 `.so`；Windows 的 importlib 只认 `.pyd` |
| `nvrtc_compile`（独立 exe） | JIT 出 cubin（humming 全靠 JIT） | `#include <dlfcn.h>` + `dlopen/dlsym`；找库只找 `libnvrtc.so`，Windows 是 `nvrtc64_*.dll` |
| `libcubinpatch.so` | 改 cubin 的 ELF 段 | ctypes 加载，名字随便，主要是换编译器 |
| `libhumming_launcher.so` | 载 cubin + 发射 kernel | `launcher/mapped_file.h` 用 `sys/mman.h`/`mmap`/`unistd.h`；构建走 `torch.utils.cpp_extension.load`（Windows 可用），但要 mmap 替身 |

好消息：`_device_info` 已经打通 —— `_dev/bin/_humming_device_info.ps1` 用 cl.exe 把同一份
`device_info.cpp` 编成 `_device_info.pyd`，放进 `humming/_native/x86_64/` 并写 `manifest.json`
（humming 只认 hash 匹配的预编工件），同时把 `device.py` 里的工件名在 win32 上改成 `.pyd`
（备份 `device.py.orig`）。验证：`_get_device_info_extension()` 返回
`loaded ...\humming\_native\x86_64\_device_info.pyd`。

剩下的三个都得自己移植，其中 `nvrtc_compile`（LoadLibrary + 找 dll）和 `libcubinpatch`（换
/cl.exe）是小事，`launcher` 的 mmap 替身是正事。另一个背景事实：`import humming` 会 fire-and-forget
地 spawn 两个子进程去预编 nvrtc_compile 和 launcher（上游 issue vllm#44904 在抱怨这个泄漏），
所以在 Windows 上这两件事现在都是静默失败。

## 9. 四个原生件移植完之后：烟测通过

把 `_device_info.pyd` / `nvrtc_compile.exe` / `libcubinpatch.dll` / `libhumming_launcher.so` 四个
件都放进 `humming/_native/x86_64/` 并登记 `manifest.json` 之后，服务器起来了，而且不是“勉强活着”：

| | eager | CUDA graphs |
| --- | --- | --- |
| `/health` | READY | READY |
| 短请求 | `finish=stop`、`Paris`、0.38 s | 0.19 s |
| 3542 token prefill（> 2048 → 分块 prefill 真的跑了） | finish=length、2.50 s | 0.94 s |
| 两个并发 | 0.56 s | 0.34 s |
| engine init | 50.10 s | 11.74 s（humming JIT 缓存已热） |
| KV | 11.53 GiB | 11.35 GiB（权重 45.82 GiB，graph 池 0.06 GiB） |

`CUDA graphs (FULL): 3/3` 全部捕获成功，共 0.06 GiB。48 层（36 GDN + 12 QSA）、`hc_count=4`、
Marlin、FA2、humming 3-bit MoE、分块 prefill、并发批处理、sampler —— 在 sm_80 + Windows 上都是
真的跑通了。**还缺的只有 PLE。**

复现这四个件：

```powershell
& D:\code\vllm-windows\_dev\bin\_humming_device_info.ps1
& D:\code\vllm-windows\_dev\bin\_humming_nvrtc.ps1
& D:\code\vllm-windows\_dev\bin\_humming_cubinpatch.ps1
& D:\code\vllm-windows\_dev\bin\_humming_launcher.ps1     # -Rebuild 可以丢掉整个 JIT 缓存
```

一个必须记住的环境变量：`CUDA_PATH`。humming 靠它找到 `nvrtc64_130_0.dll`；不设就会报
`Could not locate libnvrtc.so in CUDA path`（MoE 直接死）。`_flashnext_struct_serve.ps1` 现在已经
会设它，并且在启服前检查 `_native\x86_64\` 里的四个件齐不齐，缺就直接报错并打印上面四条命令。

### 9.1 site-packages 里的改动与回滚

humming 的 Python/C++ 源码改了 7 个文件，每个旁边都留了 `.orig`：

```
humming/utils/device.py        # win32 上把预编工件名从 .so 换成 .pyd
humming/utils/nvrtc.py         # 在 <CUDA>/bin/x64 里找 nvrtc*.dll；工件/exe 名字带 .exe；cl 构建分支
humming/utils/cubin.py         # libcubinpatch.dll；cl 构建分支；补 import sys
humming/ops/utils.py           # Windows 链 cuda.lib + torch_cuda.lib + c10_cuda.lib
humming/csrc/launcher/mapped_file.h  # CreateFileMapping/MapViewOfFile 替 mmap，并加 NOMINMAX
humming/csrc/nvrtc_compile.cpp # LoadLibrary/GetProcAddress 替 dlopen/dlsym
humming/csrc/patch_cubin.cpp   # 两个入口加 __declspec(dllexport)
```

回滚（PowerShell，在 flashtest venv 的 site-packages 里）：

```powershell
$sp = 'D:\code\vllm-flashtest\.venv\Lib\site-packages\humming'
foreach ($f in 'utils\device.py','utils\nvrtc.py','utils\cubin.py','ops\utils.py',
               'csrc\launcher\mapped_file.h','csrc\nvrtc_compile.cpp','csrc\patch_cubin.cpp') {
  Copy-Item "$sp\$f.orig" "$sp\$f" -Force
}
```

另外两处 site-packages 的补丁只是源码 commit 的影子，重装 wheel 就会刷新：
`vllm/entrypoints/launchers/api_server/entry.py`（去掉多余的 `import uvloop`）和
`vllm/models/qwen4_exp/nvidia/model.py`（PLE 关掉时丢弃 PLE 权重），两者都已经在 worktree 提交。

---

## 10. PLE/SSD 后端：Windows 原生读路径已经跑通

这是最后一个缺口。结论：**全量 143 GiB 权重 + PLE 搬到 SSD，在 Windows 上能起服、能出对的
输出、磁盘读确实在发生**。

### 10.1 为什么 `NO_BUFFERING` 在 Python 里报 87

| 观察 | 结论 |
|---|---|
| 纯 C 程序 `nbtest.exe` 同步 unbuffered 读写成功 | 卷本身支持 NO_BUFFERING（512 B 扇区，NTFS） |
| 同一个 C 代码编成 shim DLL，在 `python.exe` 里同步读仍 87 | 不是 ctypes 编组问题，是**进程相关** |
| 同一 shim 里 overlapped（异步）unbuffered 读成功返回 `ERROR_IO_PENDING` | 异步路径不被拦 |
| `CreateIoCompletionPort(file, 已有 port)` 恒返回 87 | IOCP 关联这条路在本机不通 |

所以 Windows helper 用 **每请求一个 auto-reset event + `WaitForMultipleObjects`**，不关联
任何完成端口，异步提交。同步 buffered 读正常，故 Python 侧只需把 `os.pread` 换成
`lseek`+`read`。

### 10.2 helper 设计（`vllm/models/qwen4_exp/nvidia/ple_ssd_io_win.c`）

- 与 Linux `ple_ssd_io.c` 同接口：`rows_open/rows_bind/rows_read/rows_close`，另加
  `rows_depth`（真实深度）和 `rows_fault`（哪个 slot/哪一行/哪条路径失败）。
- 每行仍是一次页对齐读：`round_up(row_in_page + row_bytes, 4096)`，行内偏移不改读边界。
- 缓冲来自 `VirtualAlloc`（64 KiB 粒度 ⇒ 每个 slot 天然页对齐）。
- 越过 EOF 的最后几行走 buffered 异步读，直写调用方输出（NO_BUFFERING 句柄不能读越 EOF）。
- **`WaitForMultipleObjects` 最多接受 64 个句柄（`MAXIMUM_WAIT_OBJECTS`）** ⇒ 深度被压到 64；
  Linux 的 `ple_ssd_io_depth=256` 在这个模式下复现不了，日志会写明 "queue depth 64 (asked 256)"。
- 收割是流水线：一个 slot 完成立刻接下一行，不再整批等待（早先批处理版 9.5k rows/s，
  瓶颈就在批栅栏）。
- `rows_bind` 由 Python 传入 CRT fd + 路径；helper 自己开两个句柄（NO_BUFFERING+RANDOM_ACCESS
  与 RANDOM_ACCESS）。

### 10.3 Python 侧改了三件事

1. `_open_read_fd()`：Windows 用 `CreateFileW`（`GENERIC_READ` + 全 share + RANDOM_ACCESS）再
   `msvcrt.open_osfhandle(handle, os.O_BINARY)`。CRT 默认 deny-all 共享，被杀软/索引器占住就
   `PermissionError`；且**必须 `O_BINARY`**，否则文本模式会把张量字节里的 CRLF 改掉。
2. `_pread()`：没有 `os.pread` ⇒ `lseek`+`read`，因此 Windows 下每个 fd 一把锁（POSIX 不需要）。
3. `posix_fadvise` 用 `hasattr` 兜住；`PLESSDNativeReader` 分平台：Windows 走 `rows_bind`，
   不再碰 `/proc/self/fd` 与 `O_DIRECT`。

### 10.4 实测证据

```
PLE SSD: 95.37 GiB on disk, 8 I/O workers, 128 MiB row cache
PLE SSD uses native Windows overlapped reads, queue depth 64 (asked 256)
Model loading took 45.01 GiB memory and 201.319288 seconds
Initial free memory 62.58 GiB, reserved 16.0 GiB memory for KV Cache
GPU KV cache size: 147,456 tokens, Maximum concurrency for 4,096 tokens per request: 36.00x
```

- `45.01 GiB` ≈ 非 PLE 权重全部（3-bit 打包后），PLE 的 ~12 GiB 不在里面 ⇒ GPU 上确实没有 PLE。
- KV 147,456 tokens / 36.00x，对比 PLE-free 图模式那次 100,912 tokens / 24.64x：多出约 46%。
- GPU1 收尾 63,917 MiB / 64 GiB 卡：45.01 + 16 + 激活 + 上下文 ≈ 62.4 GiB，几乎占满。
- 4 路并发各跑完 558/700/148/700 tokens，单路约 8.2 tok/s，聚合 ~34 tok/s；输出与结构视图
  一致（`Paris` / `alpha` / `beta`，2792-token prefill 走分块）。
- 生成窗口内磁盘读：均值 15.5 MB/s，峰值 38.6 MB/s，整窗 0.15 GiB；可用内存稳定在 ~54.4 GiB
  （没有泄漏）。

### 10.5 顺带挖出的一个真 bug：PowerShell 5.1 吃掉 JSON 里的引号

`--additional-config '{"a": 1}'` 到 python 变成 `{a: 1}` ⇒ argparse 直接报
`cannot be converted to <function loads>`。**`--speculative-config`（MTP）和
`--limit-mm-per-prompt`（vision）同样中招**，所以这两项之前没跑通不只是"没测"。

修法：传原生程序前把 `"` 写成 `\"`（Win32 命令行的转义约定），脚本里的 `EscJson` 做这件事。
已验证：

```
argv[2] = '{"ple_ssd_offload": true, "ple_ssd_native_library": "D:/a.dll"}'
```

### 10.6 已知限制

- 深度上限 64（`MAXIMUM_WAIT_OBJECTS`）。想更高得换完成端口或分组等待，本机 IOCP 关联被 87 拒。
- helper 出错即 `CancelIoEx` + 标记 dead（与 Linux `io_destroy` 语义一致），一次坏读会让后续读全失败。
- 现在 GPU1 只剩 ~1.5 GiB 余量；换更大 KV 或图模式之前要先确认不会挤爆。

---

## 11. 解码吞吐归因（实测，不是猜）

### 11.1 慢的主因是 `--enforce-eager`，不是 SSD

同一个配置（全量权重 + PLE SSD + KV 14 GiB，GPU1），只改图开关：

| 模式 | n=1 | n=2 聚合 | n=4 聚合 | GPU util 均值 | 每步耗时 |
|---|---|---|---|---|---|
| `--enforce-eager` | 9.2 tok/s | 8.5 | 8.4 | **8.9%**（峰 33%） | ~118 ms |
| CUDA graphs | 65.4 tok/s | 72.2 | 74.8 | 35.9→44.9%（峰 95%） | ~15 ms |

 eager 下 GPU 九成时间是空的，118 ms 里只有约 10 ms 真在 GPU 上跑 ⇒ **纯粹是 CPU/kernel launch
 开销**（48 层 hybrid + MoE，kernel 数量很多，Windows 的 launch 又更贵）。开图后立刻 **7.8x**。

### 11.2 PLE + SSD 的真实代价：约 16%

两者都开图、非 PLE 权重相同：

| 配置 | n=1 | n=4 聚合 | GPU util 均值 |
|---|---|---|---|
| PLE-free 视图（无 SSD 读） | 77.6 tok/s | 80.6 | 43.7→47.6%（峰 99%） |
| 全量 + PLE on SSD | 65.4 tok/s | 74.8 | 35.9→44.9%（峰 91-95%） |

所以 SSD 读路径吃掉约 16% 吞吐、约 8 个百分点的 GPU 利用率——重要但不是数量级问题。
先把基准归位：**c1 111 / c16 667 是参考机（另一台机器，SATA SSD + 15 GiB RAM）上 MTP=1** 的数字，
不是本机、也不是 MTP=2。本机 WSL2 原生（pass3）MTP=2 是 **c1 131.19 / c4 聚合 305.42**。
我们无 MTP 的 65-80 跟这两个都不是同一个配置，不能直接互比。

### 11.3 剩下那半时间在干什么（py-spy，520 样本，图模式，PLE-free）

叶子帧占比分组（很平，没有单一热点）：

| 组 | 占叶子样本 |
|---|---|
| Triton 派发（`jit.py` cache key + `driver.py:752` 的 launch + 生成的 `dynamic_func`） | **11.7%** |
| `mamba_get_block_table_tensor`（GDN/mamba 每步建 block table） | **9.4%** |
| ZMQ `send`（API server ↔ EngineCore IPC） | 3.5% |
| `torch.cuda` `synchronize` | 3.5% |
| `copy_to_uva`（PLE/UVA 暂存拷贝） | 3.5% |
| `gdn_attn.build`（每步重建 metadata） | 2.5% |
| `default_unquantized_gemm`（Python 层派发） | 2.9% |
| `tensor // x`（`torch\_tensor.py:1161` 的索引算术） | 2.1% |
| `async_tensor_h2d` / `async_copy_to_np` | 4.3% |

几乎全部样本都在 `run_busy_loop → _process_engine_step → execute_model` 里，即 CPU 一直在
模型执行路径上派发工作。GPU util ~45% 说明**一步里约一半时间是 CPU**。

### 11.4 顺手确认的一件事

`VLLM_ENABLE_V1_MULTIPROCESSING=0` 对 `api_server` **无效**：OpenAI server 路径总是把
EngineCore 另起进程（`llm_engine.py:162` 的那条只在离线 `LLM` 路径上）。所以 ZMQ 那 3.5%
想省就得改用离线 `LLM` 跑分，而不是 http server。

### 11.5 下一步候选（按性价比）

1. `--max-num-seqs` 从 4 提到 16，看聚合能不能像容器那样 6x 扩展（现在是 1→4 只涨 14%）。
2. 压 CPU：Triton 派发与 mamba block table 都在图外重做，看能否挪进图里或缓存。
3. 给 `ple_ssd.py` 加计数（rows/s、命中率、`rows_read` 墙钟），否则那 16% 只能反复 A/B 估。

---

## 12. 单路提速：MTP 是主杠杆（实测）

同一个全量 + PLE SSD 配置，只改开关，每行都是一次独立重启后的实跑（512 token/请求，n=1/2/4）：

| 配置 | c1 tok/s | n=4 聚合 | tok/step | 草稿接受率 | GPU util 均值 | 每步耗时 |
|---|---|---|---|---|---|---|
| graphs，无 MTP | 65.4 | 74.8 | 1.00 | — | 35.9→44.9% | ~15.3 ms |
| graphs + MTP×1 | **103.8** | 115.4 | 1.94 | 1741/1863 = **93.5%** | 37.9→42.7% | ~18.7 ms |
| graphs + MTP×1 + `--async-scheduling` | 102.6 | 117.4 | ~1.94 | ~93% | 35.5→43.0% | 无变化 |
| graphs + MTP×2（batched 4096） | 102.3 | **140.8** | **2.79** | 2298/2562 = 89.7%（pos0 92.0%，pos1 87.4%） | **72.3→83.0%** | ~27.2 ms |
| （参照）eager，无 MTP | 9.2 | 8.4 | 1.00 | — | 8.9% | ~118 ms |

读法：

- **MTP×1 把单路抬了 +59%**（65.4 → 103.8）。同机同深度的本机 WSL2 原生 MTP=1 是 **106.3–116.0**
  （OPS.md §9.25 的公平 A/B）⇒ 我们 **≈ 90–98%**，单路上基本追平；真正的坑在并发（见 §13.8）。
- **MTP×2 对单路无效**：tok/step 从 1.94 到 2.79（+44%），但每步多一遍 MTP 层 forward，开销 +46%
  （18.7 → 27.2 ms），两者相互抵消。不过它把 GPU util 从 ~40% 拉到 **83%**，聚合 +22% ⇒
  **并发场景该用 MTP×2**，而且此时瓶颈从 CPU 派发翻转为 GPU。
- `--async-scheduling` 在 MTP×1 上**量不出增益**（102.6 vs 103.8，噪声内）。
- MTP×1 下 GPU util 仍只有 ~40%，即单路每步仍有约六成时间在 CPU 上——§11.3 那张表（Triton 派发
  11.7% + mamba block table 9.4%）仍然是单路的下一个杠杆。

输出正确性：三种配置的烟测输出完全一致（`Paris` / `alpha` / `beta`），投机解码没有把输出说坏。

### 12.1 一个副作用：prefill 分块边界会改变贪心输出

`--max-num-batched-tokens` 从 2048 改成 4096 后，2792 token 的 prompt 不再分块，同一条 prompt 的
贪心输出从 "It appears you have provided a template…" 变成 "I cannot generate 200 unique
sentences…"。这不是 bug，是分块边界改变 bf16 累加顺序的结果，但它提醒我们：**换 batched-token
预算等于换数值路径**，A/B 时要把它算进变量里。烟测脚本里 "prompt>2048 就意味着分块 prefill 跑过"
那个断言在新预算下已经不再成立。

---

## 13. 两个 CPU 热点的优化研究（带微基准数字）

### 13.1 Triton 派发到底花在哪（triton-windows 3.6）

每次 `kernel[grid](...)` 都要跑 `JITFunction.run()` 的全套：
`get_current_device/stream` → `binder(*args)`（逐参数算 specialization：指针 16B 对齐、
int ==1 / >=2**31 等）→ `compute_cache_key` → `kernel_cache.get` → **遍历 `used_global_vals`
检查全局变量没变** → grid 归一化 → `launch_metadata` → C launcher。

微基准（8 个指针 + 2 个 constexpr，空 kernel，取 3 轮最小值）：

| 怎么发 | us/call |
|---|---|
| `kernel[grid](...)`（现状） | **11.76** |
| 缓存 `CompiledKernel` 直接 `ck.run(...)`（跳过 binder/cache-key/globals） | **4.10** |
| 同上 + 重新校验 16B 对齐（可用的安全版） | **5.29** |
| `do_not_specialize=["n"]` | 11.46（**只省 2.6%，不值得**） |

⇒ **prepared launch 能把派发砍掉 55-65%**，而 `do_not_specialize` 在这类形状上基本无效（负结果，
值得记下来）。
代价是必须自己保证 specialization 没变（对齐、int 特例、全局变量），否则编译好的 kernel 会拿着
失效的前提跑。

### 13.2 谁在触发它：全是步前的输入/元数据簿记，不是模型层

把 §11.3 里落在 triton 帧上的样本按"谁触发的"归组（520 样本，共 13.7%）：

| 触发者 | 占全部样本 |
|---|---|
| `qsa_cache.py:687 build` | 2.7% |
| `input_batch.py:386 prepare_pos_seq_lens` | 2.1% |
| `input_batch.py:636 post_update` | 1.2% |
| `mamba_hybrid.py:353 postprocess_state` | 1.0% |
| `block_table.py:204 compute_slot_mappings` | 1.0% |
| `mamba_utils.py:1212 run_fused_precopy` | 1.0% |
| `block_table.py:166 gather_block_tables` | 0.8% |
| `input_batch.py:482 combine_sampled_and_draft_tokens`（MTP 路径） | 0.8% |
| `gumbel_sample` / `get_num_sampled_and_rejected` / `preprocess_state` … | 2.4% |\n
这些 kernel 本身已经是融合的 Triton kernel，问题只是**每步在图外用 Python 重发约 30 次**。

### 13.3 `mamba_get_block_table_tensor`：91.8 µs 换 9.8 µs

`align` 模式每步做的 6 个 aten op（`utils.py:1155-1173`）实测分解：

| 操作 | us |
|---|---|
| `(seq_lens - 1)` | 7.8 |
| `// block_size` | 9.6 |
| `clamp_(min=0)` | 4.9 |
| `torch.arange(2)`（每次新建） | 6.9 |
| `unsqueeze(1) + offsets` | 9.4 |
| `.to(int64)` | 7.9 |
| `torch.gather` | 8.5 |
| **整段合起来（含 3 次分配）** | **91.8** |
| 把 `arange` 提到外面 | 74.7（−19%） |
| **换成一个融合 Triton kernel** | **9.8（9.4x，省 82 µs）** |

交叉校核：§11.3 给它 9.4% 的 CPU 样本；按每步 on-CPU ≈ 2.6 ms 算就是 ≈ 244 µs/步，
÷ 91.8 µs = **每步约 2.7 次调用**（gdn_attn / linear_attn / mamba_attn 三个入口）✓✓ 对得上。

### 13.4 其它同量级的簿记开销

| 项 | 实测 |
|---|---|
| `static[:4].copy_(src, non_blocking=True)` | 8.3 µs |
| `static[4:].fill_(1)` | 7.1 µs |
| 纯切片视图 | 1.8 µs |
| `torch.cuda.synchronize()` | 3.4 µs |

`gdn_attn.build` 的图分支里有 ~2 组 `copy_`+`fill_`+4 个切片 ≈ 35 µs/步；投机分支里是 ~6 组
≈ 90 µs/步。`synchronize` 占 3.5% 的 CPU 样本 ≈ 90 µs/步。

### 13.5 排序（收益/工作量/风险）

| 方案 | 每步净省 | 工作量 | 风险 |
|---|---|---|---|
| 把 align 分支融合成一个 Triton kernel | ~220 µs（2.7×82） | 一个文件 + 一个 kernel | 低：语义完全照搬（含 0 长请求的 clamp） |
| prepared-launch 助手（`vllm/triton_utils/` 里加） | ~195 µs（356×55%） | 中：逐 kernel 改造 | **中**：specialization 过期会拿错编译产物，必须重查对齐/int 特例/全局变更 |
| 合并 `gdn_attn.build` 里的 copy_/fill_ | 35-90 µs | 中 | 低 |
| `synchronize()` → event 查询 | ~90 µs | 低 | 低 |
| `do_not_specialize` | 0 | — | **实测无效，不要做** |
| 结构性：把整段步前簿记搬进图/搬到设备端 | 未知，可能最多 | 大 | 上游级改动 |

### 13.6 诚实的预期

这些 CPU 簿记 ≈ 600 µs/步，是 on-CPU 2.6 ms 的 ~23%，但整步是 15.3 ms，而 GPU 只忙 6 ms
——**还有约 6.7 ms 既不是 CPU 也不是 GPU 执行时间**。所以：

- 省下的 CPU 时间只在"它串行排在 GPU 工作前面"时才直接缩短整步，即 +4% 左右的吞吐；
- 真正的未知是那 6.7 ms 的空洞。一个强线索：单路速率 65.4，但 **n=2 时每条流反而更快
  （72.4）**、n=4 74.8——只有"每步有一笔固定、跨流分摊的开销"才会这样。下一步该用
  `py-spy record --idle` 把阻塞态也采进来，问"那 6.7 ms 在等谁"。



### 13.7 同一份微基准，Windows vs WSL2 原生（同卡、同盘、各自的 venv，都是空卡 GPU1）

`_dev/probe/_launch_overhead_probe.py` 和 `_pin_h2d_probe.py` 在两边各跑一遍。
Windows：torch 2.11.0+cu130 / triton-windows 3.6.0；WSL2 原生：torch 2.13.0+cu130 / triton 3.7.1。
**库版本不同是个混杂因素，读表时要记住。**

| 项 | Windows | WSL2 原生 | 读法 |
|---|---|---|---|
| triton JIT call | **11.76 µs** | **14.66 µs** | Linux 反而更慢 ⇒ 这**不是** Windows 的毛病 |
| prepared launch | 4.10 | 4.32 | 基本一样 |
| prepared + 对齐校验 | 5.29 | 6.69 | Windows 略快 |
| `do_not_specialize` | 11.46 | 17.54 | Linux 上这个开关是**负**收益 |
| block-table 那 6 个 aten op | **91.8 µs** | **75.3 µs** | **Windows 贵 22%** |
| 融合成一个 triton kernel | 9.81 | 9.65 | 一样（融合后两边都不敏感） |
| `slice.copy_` | 8.26 | 6.23 | **Windows 贵 32%** |
| `slice.fill_` | 7.06 | 5.86 | **Windows 贵 20%** |
| `cuda.synchronize()` | 3.44 | 3.05 | 贵 13% |
| 8 MiB H2D，pinned（每轮加同步围栏） | 2.509 ms / 3.3 GB/s | 2.330 ms / 3.6 GB/s | 贵 8%，两边都被 PCIe gen2×8 钉死 |
| 同上 pageable | 2.930 ms | 2.755 ms | pin 只值 **1.17x / 1.18x**，两边一样 |
| `async_tensor_h2d(1×4)` | **32 µs** | **16 µs** | **Windows 2x** |
| vllm 的 `PIN_MEMORY` 常量 | **True** | **False**（WSL 默认关，靠 `VLLM_WSL2_ENABLE_PIN_MEMORY=1` 打开） | 这项 Windows 白送 |

⇒ **Triton 派发不是 Windows 特有的问题**（Linux 上更慢，虽然版本也不同）。真正的 Windows 税是
"每个 aten op 贵 20–30%" 加 "小块 H2D helper 贵一倍"——按每步约 30 次 aten 派发、约 10 次小 H2D 算，
这才是我们跟 WSL2 差距里属于平台的那一部分，而不是那两个热点本身。
另外：他们那句"不开 pin 掉 30–35%"在我们这台机器上**不可能重现**——Windows 默认就开着 pin，而且
原始传输是 PCIe 带宽受限，pin 只值 1.17x。那条数字量的是他们配置里别的环节。

### 13.8 基准数字到底是谁测的（我之前说错了，这里归位）

| 来源 | 机器 | MTP | c1 解码 | c4 聚合 |
|---|---|---|---|---|
| `docs/RESULTS.md`（pass2，2026-09-19） | **参考机**，不是本机：64 GiB SM80 @180 W、15 GiB RAM + 31 GiB swap、企业级 SATA SSD | **1** | 110.99 | 277.22 |
| `docs/RESULTS-WSL2.md`（pass3，2026-09-29） | **本机** WSL2 原生（无 docker） | **2** | **131.19** | **305.42** |
| `ops/OPS.md` §9.25 公平 A/B | 本机 WSL2 原生 | 1 | 106.3–116.0 | — |

**同机同配置**（都 MTP=2、都图模式、都 `--max-num-seqs 4`）：

| 指标 | 本机 WSL2 原生 | 我们（Windows） | 我们/他们 |
|---|---|---|---|
| c1 解码 | 131.19 | 102.3 | **78%** |
| c4 聚合 | 305.42 | 140.8 | **46%** |
| 每步 p50 | 17.2–17.6 ms | 27.2 ms | **慢 1.55x** |
| 每步出 token | 2.20 | 2.79 | 我们更高（接受率 89.7% vs 51.6–64.9%） |

⇒ 差距在**并发扩展**和**每步耗时**上，不在 Triton 派发。一个便宜又可验的候选：他们的 PLE 是
**16 workers / 512 MiB row cache / AIO depth 256 / prefetch 16384**，那次测量**解码期磁盘读 0 MiB**
（PLE 完全不在关键路径）；我们是 8 workers / 128 MiB / 深度 64 / 无 prefetch，解码期实测
~15.5 MB/s 真实磁盘读。

他们记下的**测量纪律**（我们几次测量没做到）：① 预热 ≥2500 个解码 token；② `vmmemWSL < 32 GB`
（跑 `bin/drop_host_cache.sh` 后等 20–60 s）。缺这两条同一配置能测出 **±25% 的假差异**——他们亲眼看到
步时 22.2 ms → 17.87 ms、120.1 tok/s。另外我们有几项微基准是在一个没杀掉的服务器同卡时测的
（那个孤儿进程已清理，GPU1 现已归零）。

## 14. 单流的真瓶颈：PLE 的 ids 依赖，不是磁盘、不是派发

### 14.1 先把测量噪声压下去
256-token 短测的 run-to-run 是 **±4~6%**，任何小于这个幅度的改动都量不出来。换成
`_dev/probe/_c1_traffic.py`：一条接一条地发（始终只有一条在跑），丢掉前 2 条，用整个窗口的
稳态速率。**当前单流稳态 = 113.2 tok/s**（4200 token / 37.1 s；融合 align 开、PLE 512 MiB/16 workers）。
对照本机 WSL2 原生 MTP=1 的 106.3–116.0 ⇒ **我们已经在他们的区间里**，之前"78%"那个说法是短测噪声
+ MTP 深度不对齐造出来的假象。

### 14.2 `py-spy --idle`：步线程在等什么
`--idle` 把阻塞线程也采进来（30 s，23,984 样本；1 样本 ≈ 某线程 10 ms 墙钟）：

| 事实 | 数值 |
|---|---|
| 步线程整段都在 `execute_model` 栈里 | 2,998 样本 ≈ 30 s（≈ 全程） |
| 其中阻塞在 `threading.wait` | 1,596（**53%**） |
| 这些 wait 的直接调用者 | `Future.result (concurrent/futures/_base.py:451)` = **99.6%** |
| 再往上一层 | `_finalize_prefetch (ple_ssd.py)` |

⇒ 步线程不是在做 CPU 工作，而是在**等 PLE 的 prefetch Future**。

### 14.3 把这个等待拆开（`VLLM_PLE_SSD_STATS=1`，每 200 步一行）

| 每步 | 实测 |
|---|---|
| `Future.result()` 等待 | **9.8–10.4 ms** |
| 其中 `_ids_ready.synchronize()`（等 ngram ids 落到 host） | **8.9–9.5 ms** |
| 真正的取行（缓存 + 磁盘） | **0.38–0.95 ms** |
| 行数 | ~350–520 行/步；**77–80% 命中缓存** |

所以：**磁盘无罪**（命中率高、读量已降到 3.3 MB/s）、**Python 取行无罪**（<1 ms）、
**Triton/aten 派发无罪**（融合后吞吐动不了，因为它排在等待后面）。临界路径是
**ids 的 D2H 依赖顺序**：`start_prefetch` 里 `self._stream.wait_stream(torch.cuda.current_stream())`
让侧流排在主流已排队的一切之后，于是 PLE 读只能等前一步的队列排空，然后步线程再同步等这个 Future。
GPU 利用率只有 ~40%（每步约 7 ms 真在跑）——**17 ms 的一步里 ~7 ms 是 GPU，~10 ms 是在等**。
**§18.2 把这个机制确认到了源码层面**：`wait_stream` 只覆盖当时已排进主流的活，而 ids
生产者本身就排在一步 GPU 工作的后段（它来自 MTP draft），所以那 9 ms 很大程度
是上游计算依赖，不是可拆的等待。

### 14.4 先把上限量出来（免得对着一个错误的乐观数字做设计）
同一个低噪声口径，把 PLE 整个拿掉（`Qwen3.8-Flash-Next-struct` 视图，图模式 + MTP×1）：
**132.4 / 133.0 tok/s**（两次，复现到 ±0.5%），而全量 + PLE 是 **113.2**。

换算成步时：132.4 → ~14.6 ms/步，113.2 → ~17.1 ms/步 ⇒ **PLE 真正吃掉的是 ~2.5 ms/步（+17%）**，
不是那 10 ms 的等待本身——**那个等待大部分与引擎本来就要等的东西重叠，删掉它并不能让步时缩短 10 ms**。
（这跟 §11.2 早先 A/B 估的 16% 对得上 ✓✓）**但 §18.4 改了口径**：结构视图这一档
同时改了模型输出与内存/图状态，所以 133 vs 113 **不能叫天花板**，只能当筛选；要
给 ids 依赖真正定价需要记录/回放式的 oracle 实验。

⇒ 修正后的结论：
- "把 ids 依赖拆掉" 的最好结果是把 113 抬到 **~133**，不是 230；仍然值得做（133 会让他们 MTP=1 的
  106–116 区间相形见绌，而我们是 MTP×1），但要按 +17% 来评估投入。
- **仍有 ~8 ms/步既不是 GPU（util 40–47%，约 6 ms）也不是 PLE 等待**。这才是下一个要钉死的东西。
  工具：在 worker 里按每 200 步报一次 execute_model 的墙钟 p50/max，把"步内 CPU 胶水"与
  "步间开销（调度、ZMQ、输出回传、synchronize）"分开——现在这两个还没有被区分过。
- 已实测无效/无效的候选（别再试）：`OMP_NUM_THREADS=1`、PLE cache 512 MiB + 16 workers + prefetch
  （磁盘读降 4.7x 但吞吐不动）、`do_not_specialize`、`--async-scheduling`、融合 align（bit-identical
  且 112→32 µs/次，但吞吐动不了，因为它排在等待之后）。

## 15. 每步到底花在哪（`VLLM_STEP_STATS=1`，实测）

跑的是 `step_with_batch_queue`（批队列流水化），不是 `step()`——所以内置的
`--enable-logging-iteration-details` 只量到 `future.result()` 那一小段（p50 0.00 ms），
别拿它当步时。自己拆的四段，稳态 114.9 tok/s：

| 段 | p50 | p95 | max |
|---|---|---|---|
| **submit**（schedule + 提交 execute/sample） | **15.3 ms** | 17.7 | 217 ms（偶发长尾） |
| block（等最老的 future） | 0.10 ms | 0.13 | 11.6 |
| update（`update_from_output`） | 0.00 ms | 0.00 | 0.01 |
| gap（忙循环本身） | 0.05 ms | 0.06 | 0.31 |

⇒ **整步 = submit 那 15 ms 的 CPU 工作**；GPU 在后面异步追（util ~40%，每步约 6 ms 真在跑）。
`block`/`update`/`gap` 合计 0.15 ms ⇒ 调度和输出回传**都不是**瓶颈（内置计时也印证：调度 ~20 µs）。
PLE 的 10 ms 等待（§14.3）就发生在这 15 ms 的 submit 窗口内部（第 2 层的 `_finalize_prefetch`）。

### 15.1 现在自洽的模型
- CPU 每步提交花 15 ms，GPU 只需 ~6 ms ⇒ CPU 领先 GPU 约 9 ms。
- `start_prefetch` 让侧流 `wait_stream(current)`，于是 ids 的 D2H 要排在**上一步的 GPU 尾巴 +
  本步第 2 层之前的工作**后面 ⇒ 实测 `ids wait` ≈ 9 ms ✓✓ 与"领先量 ~9 ms"正好对上。
- PLE 走显存的那条路要花 ~7.5 ms 的 GPU 工作；SSD 路径把它换成 ~10 ms 的等待 ⇒ **净差 2.5 ms**，
  这就是 §14.4 里 PLE-free 只快 17% 的原因（而不是等待本身等于 10 ms 的收益）。

### 15.2 剩下的选项（按"能省多少 / 要付多少"）
| 选项 | 上限 | 代价 | 状态 |
|---|---|---|---|
| GPU 常驻热点行缓存（~512 MiB ≈ 1.66M 行，命中率参照 host 缓存 78%） | 消掉 ids 的 D2H 依赖 + 大部分等待，天花板 133 | 要写 kernel + 淘汰策略 + miss 回落路径 | 多天的活 |
| 在 CPU 上算 ngram ids（采样后 token 已知） | 直接消掉 9 ms ids 等待 | 要把 ngram_context 在 CPU 上镜像，并跟 MTP draft 保持一致 | 风险高，校验必须逐位对齐 |
| 削 submit 里的派发胶水（prepared launch、合并 copy_/fill_、用 event 代替 synchronize） | ~1 ms / 15.3 ms ≈ **+6%** | 中 | **做完即死心**：省了 ~1 ms submit 墙钟，吞吐不动，反向加贵也不变慢 ⇒ 派发不在关键路径（见 §16） |
| 合并 `gdn_attn.build` 的 aten 簿记（~9% 的 submit CPU ≈ 0.65 ms/步） | 未验算 | 中 | 尚未做；但按 §16 的教训，省 CPU 时间未必等于提速 |
| 已实测死心 | PLE 512 MiB/16 workers/prefetch、`OMP_NUM_THREADS=1`、`do_not_specialize`、`--async-scheduling`、融合 align（已并进代码，属"严格更好"但不是提速项） | — | — |

---

## 16. Prepared Triton launch：做了、验了、吞吐不动（负结果，2026-09-30）

§15.2 把它列为"最便宜的剩余项"（§13 的微基准：11.76 µs → 5.29 µs/次，每步约 30 次派发 ⇒
理论可省 0.5–1 ms）。做完以后：**省下的 CPU 时间确实省下了，吞吐一点没动**，而且方向相反的
实验（故意把派发加贵）也没让它变慢 ⇒ 派发不在关键路径上。

### 16.1 实现
`vllm/triton_utils/prepared.py`，在 `gpu_worker.init_device` 里安装，开关
`VLLM_TRITON_PREPARED_LAUNCH=1`（默认关）。

- 机制：在 `JITFunction.run` 前面加一条快速路径。命中缓存的 `CompiledKernel` 后直接
  `compiled.run(gx, gy, gz, stream, function, packed_metadata, None, None, None, *values)`，
  跳过 binder、cache key、globals 复查、`launch_metadata`。
- 廉价 key：张量取 `dtype` + 指针是否 16B 对齐；整数取 `==1` / `%16==0` / `>=2**31` 三个类别；
  constexpr 取**精确值**；compile options 原样进 key。它比 Triton 的 specialization 更细，
  所以命中意味着"这些轴全部没变"。
- 安全阀（一律退回 Triton 原路径）：specialization 里出现我们不认识的 token；
  `used_global_vals` 非空；hook chain 里有真实调用；异步编译还没落地；参数类型不认识。

踩过的坑：**Triton 3.6 默认就给 `launch_enter_hook` / `launch_exit_hook` 挂两个空的
`HookChain`**，所以"有没有 hook"必须看 `.calls`，不能看 `is not None`。按 `is not None` 判断时
所有 kernel 都被挡住，`fast` 计数为 0——这也顺手解释了 py-spy 里那个 `knobs.py:443` 帧。

### 16.2 三层验证
1. `_dev/probe/_prepared_launch_test.py`：**17/17**。对抗用例包括指针对齐翻转、`n` 不再整除 16、
   `stride` 从 1 变 2、constexpr 改变、options 改变、中途装 launch hook、引用模块全局的 kernel、
   换一批张量数据；每一项都必须和 Triton 原路径结果一致。
2. STRICT 模式（`VLLM_TRITON_PREPARED_LAUNCH_STRICT=1`）：每次命中都再跑一遍 Triton 自己的
   dispatch（`warmup=True`，不启动 kernel），比对它返回的 `CompiledKernel` 对象是不是同一个。
   **引擎级实测：68,000 次命中，0 mismatch**，覆盖稳态解码、1592 token 的 chunked prefill、
   并发烟测两路。
3. 结构烟测在这台 strict 服务器上通过。

假警报记录：strict 最初报 2 次 mismatch，根因是**校验器自己写错了**——`kernel_cache` 的 key
串带完整 options repr（`...{'debug': False, 'instrumentation_mode': ''}`），而 binder 在调用方
不传 options 时给的是 `{}`，于是反查永远落空。改成"直接问 Triton 的 dispatch 会选哪个 kernel"
之后归零。

### 16.3 吞吐 A/B（严格协议：`_c1_traffic.py --tokens 900 --skip 3`，≥2700 解码 token 预热；
两次都是全新重启，起服参数逐字节相同）

| 配置 | c1 稳态 | submit p50 | Triton 派发样本/30 s |
|---|---|---|---|
| 基线（无 prepared） | 116.8 / 115.7 | ~16.5 ms | 134（≈0.78 ms/步） |
| prepared | 114.9 | **15.5 ms** | 248（其中 89 落在 `prepared.py:255`，即不可约的 launcher 成本） |
| prepared + strict（每次命中多付一整趟 ~12 µs 的 JIT） | 116.3 | 15.6 ms | — |

两个方向夹击：
- 省掉 ~1 ms 的 submit 墙钟 ⇒ 吞吐不变；
- 反过来给每次命中**加**一整趟 JIT 派发 ⇒ 吞吐也不变。

（降级一句：两台基线的 submit p50 本身在 16.0–17.9 ms 之间摆，那 1 ms 落在摆动范围内，不该当成
实测增益。§17.4 还有一条更硬的修正。）

⇒ **CPU 派发时间不是单流的约束**；约束仍是 PLE 的 ids 依赖（§14）。py-spy 的旁证：submit 的
on-CPU 样本从 1206 升到 1343，而 submit 墙钟下降 ⇒ 阻塞变少、CPU 变多，正好是自洽的图像。

### 16.4 方法论事故（必须记住）
**temperature=0 的贪心输出在同一台服务器上重复同一请求都会逐字不同**（前缀缓存命中会改变
prefill 的切分，reduce 顺序随之变，贪心的近似平局就会翻）。所以"输出逐字一致"**不能**用作
A/B 的正确性判据；这次的正确性判据换成 STRICT 的 kernel 同一性 + 对抗单测。

### 16.5 处置
代码留在 `flash-next-win` 分支，env 默认关，和融合 align 同一待遇：**严格更便宜，但不是提速项**。
它对高并发可能仍有价值（那时 CPU 余量才是稀缺资源），但按当前约定不做并发实验。
单流剩下唯一的杠杆仍然是 §15.2 的前两条（GPU 常驻热点行缓存 / CPU 上算 ngram ids，天花板 133）。

---

## 17. MTP 深度复测：MTP×2 值 +24%；prepared launch 在两个深度都无效（2026-09-30）

### 17.1 测量条件
今天所有服务器都逐条查过日志里的 `num_speculative_tokens`。MTP×2 的两台与 MTP×1 基线台的
起服参数**除深度外逐字节相同**（把 `speculative_config` 摘掉后 diff 过）。
协议同 §16.3：`_c1_traffic.py --tokens 900 --skip 3 --window 70`，≥2700 解码 token 预热。
接受率用引擎自己每 10 s 打的 `SpecDecoding metrics`，按流量窗口取平均。

### 17.2 结果

| 配置 | c1 稳态 tok/s | 平均接受长度 / 接受率 | submit p50 | 推算步频 / 步时 |
|---|---|---|---|---|
| MTP×1 基线（两台） | 116.8 / 115.7 | 1.91–2.00 / 91–100% | ~16.5 ms | 59.6 步/s ⇒ 16.8 ms |
| MTP×1 prepared | 114.9 | — | 15.5 ms | — |
| MTP×1 prepared+strict | 116.3 | — | 15.6 ms | — |
| **MTP×2 基线** | **144.0** | 2.73 / 86.4% | 17.4 ms | 52.7 步/s ⇒ 19.0 ms |
| **MTP×2 prepared** | **143.5** | 2.73 / 86.7% | 17.45 ms | 52.6 步/s ⇒ 19.0 ms |

- **MTP 1→2 = +23.8%**（144.0 / 116.3）。
- prepared launch 在 MTP×2 同样无效（-0.35%，在 ±1% 噪声内）。两次 MTP×2 的接受长度和接受率
  **一模一样**（2.73 / 86.4% vs 2.73 / 86.7%），所以这次比较没有"接受率漂移"这个混淆项。
- submit p50 在 MTP×2 下纹丝不动（17.4 → 17.45）⇒ prepared 连墙钟都没省，跟 MTP×1 那次不同。

### 17.3 跟他们同机公平 A/B 对齐（OPS §9.25）
他们：MTP×1 106.3–116.0（tok/step 1.70，step p50 15.2–15.7 ms，接受率 63.5–77.1%）
→ MTP×2 118.4–130.9（tok/step 2.20，p50 17.2–17.6 ms，接受率 51.6–64.9%）= **+13.5%**。

- **每步成本我们跟他们齐平**：MTP×2 我们 p50 17.4 ms，他们 17.2–17.6 ms ✓✓ 引擎侧没有差距。
- 差距全在**接受率**：我们这套流量（短 prompt、重复句式）MTP×1 接受长度 1.95、MTP×2 2.73
  （第二个 draft 位置仍接受 0.80–0.88）；他们 1.70 / 2.20。tok/step 高 24% ⇒ token 速率高 24%。
- 拆开：MTP 1→2 让我们 tok/step +40%（1.95→2.73）、步时 +13%（16.8→19.0 ms）⇒ 净 +24%；
  他们 tok/step +29%、步时 +12.5% ⇒ 净 +13.5%。**机制相同，幅度由接受率决定。**
- 推论：他们 pass3 的 **131.19 不是我们的靶子**——那是他们负载下 MTP×2 的数。我们负载下
  MTP×2 已经 144。反过来，我们这 86% 的接受率部分是"prompt 高度重复"喂出来的，真实业务
  流量上增益会缩水，别拿 144 当业务承诺。

### 17.4 一条更硬的观察（修正 §16 的乐观版本）
把 py-spy 样本换算成 CPU 毫秒（100 Hz ⇒ 1 样本 = 10 ms CPU），再除以窗口内的步数：

| | 落在 Triton 派发路径上的 CPU |
|---|---|
| MTP×1 基线 | 134 样本 / 1572 步 = **0.85 ms/步** |
| MTP×1 prepared | 248 样本 / 1570 步 = **1.58 ms/步** |
| MTP×2 基线 | 180 样本 / 1581 步 = **1.14 ms/步** |
| MTP×2 prepared | 246 样本 / 1573 步 = **1.56 ms/步** |

⇒ **在真实引擎里 prepared launch 让 Triton 路径的 CPU 变多了**，微基准那次 11.76 → 5.29 µs 的
省法在引擎里没有兑现。原因很可能是微基准用的是 8 指针 + 2 constexpr 的小 kernel，而 vLLM 的真
kernel 参数多、带 global scratch 和 tensor descriptor：**一次派发的成本大头在 launcher 本身
（scratch 分配 + ctypes + `cuLaunchKernel`），prepared launch 只拆得掉 Python 前端，拆不动 launcher。**

于是 §16.3 的结论要重写成两条：
1. 吞吐在两个 MTP 深度上都不动 ⇒ 派发不是约束（这条不变，且有反向实验支撑）；
2. 我们以为省下的派发 CPU，按原位测量其实是**增加了** ⇒ 这条实现的净收益是负的，默认关是对的决定。

### 17.5 处置
- **生产口径该用 MTP×2**（+23.8%）。上一阶段"Windows 上 MTP×2 反而更慢"（c1 102.3、p50 27.2 ms）
  的结论作废：那是在 PLE 配置对齐之前、插桩与负载都不同的条件下测的。
- prepared launch：两个深度都无收益、原位 CPU 还变差 ⇒ 保留代码但 env 默认关，
  验证结论（68k 命中 0 mismatch）仍然有效，只是现在没有理由开它。
- 单流剩下的杠杆仍然只有 §15.2 前两条（消掉 ids 的 D2H 依赖）。注意 PLE-free 的上限
  （133 vs 113）只在 MTP×1 上量过；若要走这条路，得在 MTP×2 上重量一次天花板。

## 18. 两轮 gpt-6 会诊纪要

材料：`consultations/astra_brief1.md` ↔ `astra_reply1.md`（17.9 KB / 20.4 KB），
`astra_brief2.md` ↔ `astra_reply2.md`（11.5 KB / 27.6 KB）。两轮都按 AGENTS 的会诊
约定跑（只读评审、不碰 GPU0、不重启服务）。**下面每条都经过我们独立核实**，标 ✗ 的
是被测量或源码否证的。

### 18.1 它抓到、我们核实为真并已修的

1. **PLE reader 的 EOF 判定不精确**（`ple_ssd_io_win.c`）：分支用行的**逻辑末尾**
   `page + delta + row_bytes > size`，而真正发出的非缓冲请求是**页取整后**的长度。
   safetensors 张量末尾不页对齐 ⇒ 该形状在真实权重里可达。
   ✗ 但它的断言"非缓冲读越 EOF 必失败"被我们否证：探针里 offset 20480、请求 8192、
   文件 20481 ⇒ `lpBytesReturned = 8192`，这台机器上越过 EOF 的读**返回完整取整长度**。
   所以性质是"依赖了没有文档保证的行为"，不是"已经读错字节"。
   已改为用取整后的长度判定（把该情形送去精确缓冲读），重建 DLL
   sha256 `4E59A3BA2E87CA7EDFB9B70C53D453E3FBEA4F1397F13865CEB8A113105D1EC4`，
   探针仍 **0 failure**。
2. **两个句柄都没有 `FILE_FLAG_OVERLAPPED`**，却给 `ReadFile` 传 `OVERLAPPED`。实测：

   | 旗标 | ReadFile | GetLastError | 事件立刻 | 200 ms 内 | GetOverlappedResult | 字节 |
   |---|---|---|---|---|---|---|
   | shipped（NO_BUFFERING\|RANDOM_ACCESS） | TRUE | 0 | **已置位** | 已置位 | TRUE | 8192 |
   | +FILE_FLAG_OVERLAPPED | FALSE | 997 `ERROR_IO_PENDING` | 未置位 | **置位** | TRUE | 8192 |

   ⇒ 现路径是**同步完成但确实置事件**，所以 `WaitForMultipleObjects` 收割能工作。
   它第二轮找到 MSDN 依据（`ReadFile` 明说同步句柄+非空 `lpOverlapped` 时偏移取自该
   结构且调用直到完成才返回；`OVERLAPPED.hEvent` 明说完成时置位）——**这纠正了我们
   第一轮"无文档依据"的说法**。代价：**原生深度队列没有 I/O 并行**，真并行只来自
   Python 线程池（与"worker 8→16 把读盘速率降 4.7x、而 depth 64 早已封顶"自洽）。
3. **`WAIT_MS=60000` 约束不了同步 `ReadFile` 的耗时**：卡住的同步读根本到不了超时
   分支 ⇒ 我们现在**没有可靠的 I/O 截止**。这条我们原先没意识到。
4. **strict 校验不是 fail-closed**：①发现不一致只打日志、照样启动 prepared kernel
   （已修成"不一致就退回 Triton dispatch"，并加了强制不一致的单测：fallback 计数动、
   fast 计数不动、结果仍正确）；②校验器在异常或拿不到 reference 时**返回 True**（已
   改成 False）。单测 22/22。
5. ✗ 它提醒 py-spy 归因不可全信：快路径改变了可见栈帧，所以"prepared 之后 Triton 路径
   CPU 从 0.85→1.58 ms/步"**当作未证实**。

只读探针 `_dev/probe/_ple_win_sem.c` 直接驱动真 DLL：文件长度含页对齐与
**非页对齐**（4096+37、3·4096+19、5·4096+1、7·4096+320，因为 safetensors 末尾不页
对齐）、行宽 320/512/1024/4096、偏移覆盖 delta 0/1/4095 与"正好落在 EOF""落在最后
不足一页内""整页越过 EOF"；每行与缓冲读期望逐字节比对。深度钳制 256→64、64→64、
1→1 全对；强制失败（-EREMOTEIO）后立即 close 不炸；500 次 open/read/close 不炸。
第一轮跑出 25 FAIL 是**探针自己的缺陷**（一次失败把 reader 标成 dead ⇒ 后续 -EBADF
级联）+ 小文件算术下溢，修好后 **0 FAIL**。它算过：500 次全过只给出每周期约 0.6% 的
95% 上界，**不等于覆盖"挂起中的取消"**。

### 18.2 它否掉的三条优化（我们独立核实后接受）

- **event-scoped wait 不是优化**（它最强的一条修正）。我们把它引的代码跟**实际在跑的
  那份**逐行核对（`vllm/models/qwen4_exp/nvidia/ple_ssd.py` 的 `start_prefetch`，
  服务器用的是 site-packages 里的部署副本）：
  `wait_event(_previous_use)` → `_device_ids[:tokens].copy_(ngram_ids)` →
  `wait_stream(torch.cuda.current_stream())` → 侧流上 `_ids.copy_` + `_ids_ready.record`。
  `wait_stream` 只覆盖**当时已排进主流的活**，而 ids 生产者在调用点之前就已入队 ⇒
  换成"ids 之后记 event"表达的是**同一条依赖**，先验收益 ≈ 0。它进一步指出：ids 来自
  MTP draft，本身排在一步 GPU 工作的后段 ⇒ **那 8.9–9.5 ms 很可能是上游计算依赖，
  不是可拆的等待**。⇒ **我们 §15.2 里"消掉 ids 等待"这条的估价要从 +17% 大幅下调**。
- **GPU 热点行缓存被廉价否证**（它修正了自己第一轮的门槛）：每步 350–520 行时，
  即使行命中率 99%，**整批全中只有 3%（350 行）/0.54%（520 行）**；要让某一步不需要
  主机汇合，需要 **99.97–99.98%** 的行命中率。所以 512 MiB 缓存**消不掉每步的 D2H
  依赖**，除非 ids 能更早可用。它要求先做**原始 row-ID trace + 按批模拟**（10k 解码
  步，raw ID 约 14–42 MB），报"整批全中/整步全中比例、每批 miss 的中位/p95/p99"，
  若模拟显示大多是部分命中、且实现照样付主机汇合 ⇒ 杀掉该项目。
- **MTP×2 的 +23.8% 很可能是这个流量集的产物**：重复散文 + 贪心 + 前缀缓存命中改变了
  prefill 切分（我们已实测贪心输出在同一次运行内都不可复现 ⇒ 文本比对不可用作判据）。
  它坚持"提示词标签不等于接受率"，长推理/代码/多语言在局部同样可能高度可预测，所以
  低接受率类别必须**用标定集实测筛出、再到留出集评估**，不能靠改温度造假。

### 18.3 它给的下一步排序，与我们的分歧

它的排序：① 混合流量 MTP×1 vs MTP×2 战役 ② 标注清楚"改了什么"的 PLE 消融筛选
③ ids/preservation/copy/consumer 时间线 ④ 原始 row-ID trace + 按批模拟
⑤ 三档派发扰动（0/+1/+3 ms 每步，两个深度）。

我们的分歧只有一条：**②里的 MTP×2 结构视图消融只要 ~4 分钟起服 + 一轮流量，是清单
里最便宜的一项**，所以先做它、再做需要搭流量集的 ①。其余四条接受它的顺序。它也明确
反对"把事件替换当成优化"、并把真异步排除出前五。

混合流量战役的验收线（它的原话）：混合集解码增益 **≥10%**；配对 95% CI 下界最好 **>5%**；
任何类别不得出现可复现的 **>5%** 解码退化（否则改为按流量路由）；TTFT/尾延迟/内存/
输出质量不得有不可接受退化。口径：`R_d = Σ发出token / Σ解码时间`，`G = R_2/R_1 - 1`，
**按类别加权并分别报告**，不要平均逐请求百分比。深度是起服时定死的 ⇒ 一次初始化不能
同时比两个深度，ABBA 需要更多配置实例，**不要拿同一次初始化里的多个窗口冒充重复**。

派发扰动的设计要点（它给的判读）：`s = Δ稳态每步墙钟 / Δ实际注入延时`；95% 上界 <0.1 ⇒
在该区间内运营上算平；下界 >0.3 ⇒ 有实质敏感；下界 >0.7 ⇒ 大部分注入被暴露；区间横跨
平与实质 ⇒ 不确定；+1 平而 +3 实质 ⇒ 存在松弛阈值而非一条全局线性关系。19 ms/步下
s=0.3 在 +3 ms 时多 0.9 ms ≈ −4.5%，远大于 ±1% 噪声 ⇒ 可判；但要证"斜率<0.1"需要
**等价性置信界**，不是"检验不显著"就行。它强调：注入点必须在**常规 Triton 派发路径**
的真实 eager 启动之前、只对稳态解码步生效、排除 warmup/capture/编译/`warmup=True`，
**不得包 `CUDAGraph.replay`、不得改已捕获 kernel、不得加图断点**；预算按步给但要摊到
合格启动上，若无合格启动则该步没有扰动（不要偷偷挪地方）。它还提醒：**加 3 ms 的敏感度
不能反推"省掉今天的派发开销就能拿回同样的时间"**。

### 18.4 PLE 上限的口径修正（重要）

它坚持 trimmed 结构视图那一档**既改了模型输出、又改了内存/图状态**，所以 133 vs 113
**不是"PLE-free 天花板"**，只是净差筛选。要给 ids 依赖真正定价，需要**记录/回放式
oracle 实验**：先录一段 MTP×2 窗口的 row-ID/步形状/发出与 draft 行为/查表顺序，再在
同一工作负载下比较"照常规路径拿行"与"把同样的行提前变得可用"，尽量保住分配与图分段，
并列出残余差异。**若这个"匹配"的干预落在 3–5% 以内 ⇒ 杀掉昂贵的 PLE 重构。** 用零行
或 trimmed 权重时，生成 token 与投机接受率会变 ⇒ 计时归因必须冻结/回放 token 与形状
轨迹，否则比较把"实现改动"和"工作负载变化"混在一起。

### 18.5 已落地的决定

- prepared launch **保持默认关闭**（无收益证据；fail-closed 补齐后也不再为它花性能预算）。
- PLE reader **保持同步**；翻异步必须先满足它列的生命周期闸门（所有权契约、完成状态机
  区分"立即成功/挂起/立即失败"、manual-reset 事件与已置位槽位退役、取消后必须确立**终态
  完成**、**失败返回要在 `rows_read` 返回前排干**而不是等 `rows_close`、约 1000 轮有界
  cancel/complete/close 且计数证明确有真挂起被覆盖）。它明确警告：`CancelIoEx` 不等完成，
  取消可能以成功/`ERROR_OPERATION_ABORTED`/别的终态错误收场；**"再等一段有界时间然后照样
  释放"不算 drain**，确立不了终态就必须让被引用内存活着。Application Verifier 不是保留
  同步实现的前提，装它也不该为了这个决定去动正在跑的主机。
- 事件替换 **降级为研究项**（先验 ≈ 0）。
- GPU 行缓存 **降级**：先做 row-ID trace + 批量模拟。
- 顺序：MTP×2 结构视图消融筛选（最便宜）→ 混合流量 MTP 战役 → ids 时间线 → row-ID trace。

### 18.6 会诊本身的经验

两轮都命中，而且**双向**：它第一轮一处断言被我们的测量否证，第二轮一处建议（">90% 行
命中就够"）被它自己的算术修正。有效模式仍然是"它读源码 + 我们跑实验"的交叉验证。
一个坑值得记：worktree 里 `vllm/model_executor/layers/ple/` 是我们**拷进去的未跟踪目录**，
被我们自己的 `git clean -fd` 删掉了；真正在跑的 PLE 预取实现是
`models/qwen4_exp/nvidia/ple_ssd.py`（及其 site-packages 部署副本），**在 worktree 源码里
grep 不到它**。以后引代码去会诊，必须引**部署副本的实际行号**，否则评审会读到不在跑的
那份。

## 19. A 做完：结构视图消融 + 深度阶梯（今天全部同机、同宿主内存低压）

### 19.1 A：PLE-free 结构视图 @ MTP×2（三块，±1%）

起服参数与全量基线逐字节比对过，只差 model/served-name 与没有 PLE 的
`additional_config` 块。口径 `--tokens 900 --skip 3 --window 70`，宿主空闲 43.6–43.8 GiB。

| | tok/s（三块） | 中位 | 接受长度 | 平均接受率 | submit p50 | block p50 |
|---|---|---|---|---|---|---|
| 结构视图 PLE-free ×2 | 165.0 / 164.7 / 166.7 | **165.0** | 2.68–2.83 | 83.5–91.3% | **5.08 ms** | **10.85 ms** |
| 全量 + PLE ×2（同日重跑，新起服） | 144.4 / 149.5 / 152.2 | **149.5** | 2.67–2.83 | 83.3–89.8% | 17.34 ms | 0.10 ms |

接受长度与接受率**逐档对齐**（2.68–2.83 vs 2.67–2.83）⇒ **没有接受率混淆**，差的是实现。
净差 **+10.4%**（中位对中位；均值对均值 +11.0%）。按 §18.4 的口径，这是**筛选不是天花板**：
结构视图同时换掉了权重、内存与图状态（GPU KV 容量也从全量的档位变成
87,552 token / 21.38x 并发）。

### 19.2 步内部形状换了主人

全量 ×2：**submit p50 17.3 ms、block 0.10 ms** ⇒ CPU 提交当限速器，GPU 往返被完全遮住。
PLE-free：**submit p50 5.1 ms、block 10.85 ms** ⇒ CPU 快了，GPU 往返才第一次露出来。
PLE 自己的账（`VLLM_PLE_SSD_STATS`，同日同机）：wait **11.3 ms/步**（其中 ids
10.9 ms）、取行 0.21–0.38 ms ⇒ **17.3 ms 的 submit 里 11 ms 是 PLE 的 CPU 等待**，
即 submit 的 **63%**。这就是为什么"消掉 PLE"卖 +10%，而"只省派发"卖不动。

### 19.3 单流受限性质（三档深度各 480 个 nvidia-smi 采样）

| 配置 | GPU util 均值 | 峰值 |
|---|---|---|
| PLE-free ×2 | 45.2% | 99% |
| 全量 + PLE ×2 | 40.2% | 92% |
| 全量 + PLE ×3 | 39.9% | 93% |
| 全量 + PLE ×4 | 39.8% | 93% |

⇒ **单流解码不在 GPU 吞吐上**，GPU 六成时间是闲的；步时被 CPU 提交路径决定。
（§14.3 那句"~7 ms 是 GPU，~10 ms 是等"仍然成立，但今天能指认等的是谁。）

### 19.4 深度阶梯（同一台机器、起服参数只差 `num_speculative_tokens`，逐字节 diff 验证）

| depth | 三块 tok/s | 中位 | Δ(中位) | 接受长度（窗口） | 每位置接受率 | 步时 = 接受长度/中位 | submit p50 | PLE wait / ids | cache 命中 |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 144.4 / 149.5 / 152.2 | 149.5 | — | 2.67–2.83 | p1 0.88–0.93, p2 0.77–0.88 | 18.3 ms | 17.34 | 11.3 / 10.9 | 93.1–93.2% |
| 3 | 155.7 / 160.5 / 155.8 | 155.8 | **+4.2%** | 3.16–3.55 | 0.83–0.91 / 0.71–0.86 / 0.61–0.78 | 21.8 ms | 20.1 | 13.5 / 12.7 | 92.9–93.1% |
| 4 | 175.7 / 171.4 / 179.0 | **175.7** | **+17.5% vs ×2** | 3.75–4.49 | 0.82–0.94 / 0.72–0.90 / 0.63–0.85 / 0.57–0.79 | 22.7 ms | 21.8 | 14.3 / 13.9 | 94.0–94.1% |
| 5 | 起服即死 | — | — | — | — | — | — | — | — |

深度 5 不是性能问题，是**布局断言**：`qsa_cache.py:846` `AssertionError: QSA ring
capacity 12 must divide the attention block size 1616`（1616/12 不整除 ⇒ 环形容量随
深度长，块大小不跟着长）。⇒ 可用深度上限 **4**；想开 5 必须真去解 QSA 环形容量与
block size 的关系，不值当为 +5~6% 的推测收益去做。

**两件事同时成立**：① 深度越深，PLE ids 等待越长（+1.3 ms/档，正是 §18.2 里 gpt-6
"ids 生产者排在一步 GPU 工作后段、前面排的主流工作越多"预测的形状）；② 但每步换到
更多被接受的 token ⇒ **把这个固定的 CPU 停顿摊薄了**。所以加深 MTP 目前是"消掉 PLE
停顿"的免费近似：×4 用 +17.5% 买到的是同一个 11→14 ms 的等待被 4.0 个 token 分担。

### 19.5 生产含义与必须先做的验证

- 默认深度应该从 ×2 往 ×3/×4 挪（零代码，只改 `--num-speculative-tokens`）。
- **但**这条流量（"从 N 倒数 600 个整数"、贪心）接受率有水分，深度越深、接受率崩掉时
  浪费的 draft 越多 ⇒ §18.3 的**混合流量战役**从"该做"升级为"必须先做"，且现在比的是
  ×2 vs ×3 vs ×4 三条，不是两条。
- PLE 的 CPU 停顿现在有了价格（~10%）也有了身份（submit 的 63%）；它是否可拆，
  取决于 ids 时间线（§18.3 排 3，今天的数据把它往前推）。

### 19.6 新工具

`_dev/probe/_log_decode.py`：PowerShell 重定向的原生命令日志是**混合编码**（前缀 ANSI，
后面 UTF-16LE），`iconv` 直接吃会出乱码、`grep -a` 直接吃会漏掉全部稳态行。该助手在第一
个 NUL 处切开、分别解码、输出 UTF-8，可再传一个过滤子串。本轮所有日志判读都靠它。

## 20. 第三轮 gpt-6 会诊纪要（`astra_brief3.md` ↔ `astra_reply3.md`，8.3 / 22.1 KB）

它这轮的判断是**读源码**得出的，三条源码断言我们逐条核实为真：

1. **命中率统计口径是错的（我们的错）**：`ple_ssd.py:597` 用 `table.hits + table.reads`
   算百分比，而 `_stat_wait/_stat_ids/_stat_read/_stat_steps` 每次上报清零、
   `table.hits/table.reads` **不清零** ⇒ 打出来的 "% from cache" 是**进程启动以来的累计值**；
   且计数发生在 `unique = np.unique(flat)`（`ple_ssd.py:361`）之后 ⇒ 它是**每次 lookup 内
   去重后的行**，不是行的出现次数；prompt 预取也用同一张表。**所以 §19 表里的
   "93.1–94.1%" 不是邻近 200 步解码窗口的命中率**，×4 那一档"命中率随深度上升"也可能
   只是暖机/分母差异。
2. **QSA 环形容量公式**（`qsa_cache.py:844-846`）：`span = compress_ratio +
   num_speculative_tokens`，`capacity = compress_ratio * cdiv(span, compress_ratio)`，
   checkpoint 的 `indexer_compress_ratio = 4` ⇒ `C(d) = 4·⌈(4+d)/4⌉`。断言要求
   `block_size(1616) % C(d) == 0`，1616 = 16×101 的因子只有 1,2,4,8,16,101,202,404,808,1616。
   于是 **d=1–4 → 8 通过；d=5–8 → 12 不通过（×6/×7 与 ×5 同死）；d=9–12 → 16 过这条断言**
   （但过断言 ≠ 支持、接受率、内存、正确性）。它明确反对：不要删断言、不要把容量钳到 8
   （源码注释：被拒绝的 draft 行会覆盖后面还要用的已提交 key ⇒ 正确性危害）；不要为了
   绕过断言跳 ×9；改 block size 要把 attention/Mamba/QSA 对齐链整条推一遍（举例：保住
   1616 又塞进容量 12 ⇒ LCM 4848，三倍大）。⇒ **×4 是当前可用包络**。
3. **`block` 不是纯 GPU 时间**（`core.py:824`）：`block = t_update_start - t_submit_end`，
   中间夹着 `future.result()`、`_process_aborts_queue()`、`scheduler.update_from_output()`、
   `_attach_iteration_details()` ⇒ **不能把两个 p50 相加当延迟地板**。

### 20.1 它对我们判读的三处收回（我们都接受）

- `10.9 / 17.34 = 63%` 只是**量级比较**，不是账目划分：混了协调线程的均值与另一个作用域
  的 submit 中位数。
- **"GPU 占用 40% ≠ 六成空闲可用"**：能不能塞有用活取决于依赖空隙、串行 kernel、资源争用。
- `_c1_traffic.py` 的分母是**整个非流式 HTTP 请求**（含 prefill 与响应处理）⇒ 我们所有
  "tok/s" 都是**请求级吞吐**，不是隔离的解码吞吐；§19 里"接受长度 ÷ tok/s = 步时"是
  **推算的有效步时**，不是实测引擎步长。**这条直接动摇了 §19 表的绝对步时列。**
- 取行只有 0.21–0.38 ms/步：即便全暴露在 18.3 ms 里也只值 **1.2–2.1%** ⇒ **SSD 吞吐与
  worker 数调参再次降级**；且不得用主机缓存百分比去推 GPU 缓存命中率或整批全中概率。

### 20.2 深度 ×4 的账（它用我们的数重算）

| | ×2 | ×4 | 变化 |
|---|---|---|---|
| 有效步时 | 18.3 ms | 22.7 ms | +24.0% |
| 每步发出 token | 2.74 | 3.99 | +45.8% |
| 吞吐 | 149.5 | 175.7 | +17.5% |
| ids 等待 / 发出 token | 3.98 ms | 3.49 ms | −12.5% |

盈亏条件 `A4/A2 > T4/T2`：×4 要**每步多 24% 的 token 才打平**，要 +10% 吞吐则需 **+36.4%**
（即 3.39 / 3.73 token/步，对 A2≈2.74）。**这些阈值必须按流量类别重算。**
它把 ×4 说成"摊薄一笔随深度增长的依赖开销"，而不是我们说的"固定 CPU 停顿"——更准确。

### 20.3 只拆 CPU 等待能卖多少（它的预算先验，非置信区间）

`R_new ≈ 149.5 × 18.3/(18.3 − h)`：h=0.5 ms → +2.8%；1.0 → +5.8%；1.5 → +8.9%；
1.8 → +10.9%。**先验中心 ≈ 3–6%**（不是 10%）。在 ×4 上每省 1 ms 只值 **+4.6%**
（要 +5% 需省 1.08 ms，要 +10% 需 2.06 ms）⇒ **深度越深，同样的绝对节省买的百分比越小**，
所以 **oracle 必须在 ×4 上定价**，不是 ×2。既没有"16 ms 硬地板"的证据，也没有"保住 PLE
还能拿满那 10%"的依据。

### 20.4 oracle 的设计（它替我们拟的，三臂）

"预热缓存 + 忽略 ids"**不合法**：缓存只映射 id→行，它不知道这一步该按什么顺序取哪些行，
100% 命中也可能静默喂错嵌入。合法做法是把**该次 lookup 的、按序的、精确行载荷**交给原路径：

| 臂 | 载荷 | 放行条件 | 作用 |
|---|---|---|---|
| 常规参照 | 常规查行 | ids 真的就绪 | 衡量"预处理载荷"本身把基线挪了多少 |
| 晚到精确载荷 | 预先装好的精确行 | ids 真的就绪 | 两臂同样地拿掉"装行" |
| 早到精确载荷 | 同一批预先装好的行 | **侧向 staging 缓冲最早可安全复用时** | 给 ids 依赖定价 |

**因果比较是 late vs early**，参照臂只是校准。硬约束：早到的 H2D **不得仍排在原来的
`wait_stream(current_stream)` 与诊断 D2H 之后**，否则只是把等待挪了地方；必须保留
`_copy_ready`、`_previous_use`、消费者顺序、PLE 的全部 GPU 计算、以及 ids 生产者本身。
"同 prompt 同 seed 回放"**不够**——draft 轨迹一变，需要的行就变 ⇒ 要么**冻结 token/state
轨迹**（这是计时 oracle，不是自主生成正确性证明），要么**在线回放 + 逐位 id 校验**，任何
不一致整轮作废（不许只报一致的步）。列出残余差异：预装 vs 常规查行的锁开销、trace/载荷
内存对宿主压力、H2D 提交时机与协调调度、id 拷贝/校验诊断、保留或删除的流依赖、图分段与
分配与 staging 缓冲个数、强制轨迹 vs 自主解码、暖机与是否计入准备时间。
停手门槛（×4）：**95% 上界 <5% ⇒ 停**；点估 <5% 且区间跨 5% ⇒ 经济上可停但统计不确定；
5–10% 可复现 ⇒ 只做"有可信路径保住大部分 oracle 收益"的小实现；>10% 且下界 >5% ⇒ 值得做。
**弱 oracle 若不小心保留了汇合，不能用来杀重构**——必须先证明干预真的把行就绪与下游提交
提前了。

### 20.5 混合流量战役（它排第一）与晋级门槛

**三档**（×2/×3/×4），起服成本 13.5 分钟而非 9 分钟，"便宜的保险"。**预先声明**：主比较
×4 vs ×2，次比较 ×3 vs ×2，选择比较 ×4 vs ×3 ⇒ 防事后挑最好的噪声。**必须用固定请求表**，
不能"70 秒里塞进几条算几条"——现有脚本让快的配置拿到更多请求 ⇒ prompt 组成不同（它抓到
了我们工具的这个偏置 ✓）。第一轮约 60–120 条留出请求、六类、按 prompt 与采样设置配对，
含短输出与长输出、至少两个 prompt 长度档；确认阶段反序配置 ⇒ 6 次起服 = 27 分钟起服成本。
**同一初始化里的多个窗口不算初始化级重复。** bootstrap 按配对请求或块，不按单 token。

指标优先级：① 解码时长与 time-to-last-token；② TTFT 与 p95/p99 inter-chunk 间隔；
③ 发出 token、引擎步数、**实测**步长；④ 按**每发出 token** 计的 draft/接受/丢弃；⑤ 每
发出 token 的 GPU 忙时或能耗（若便宜可得）；⑥ KV/状态内存、余量、错误与质量检查。
它按我们的推算给了示意值：×2 丢 0.26 draft/步 = 0.10/发出 token，×4 丢 1.01/步 =
**0.25/发出 token**（示意，不是计数器替代品）。
它接受"单流下多做的活可能不占墙钟"，**但不接受因此不做账**——可能费能耗、拉长流式间隔、
在并发下变贵。若生产严格单流，不必为完备性专门开并发战役，但默认值要按此限定范围。

晋级矩阵：×4 全过 ⇒ 在校验过的服务区间内设默认；×4 在某些类挂而 ×3 过 ⇒ ×3 默认或按类
路由；×4 与 ×3 差 ~2% 内且不确定区间重叠 ⇒ 若 ×3 尾更好或每 token 工作更少则选 ×3；
增益只出现在重复/高接受流量 ⇒ ×2 保持通用默认、深度选择性开；均值涨但流式间隔或 TTFT 破
SLO ⇒ 不许凭 tok/s 晋级。

### 20.6 猜测 ids：验证式预取可以，消费猜测不行

合法版本：① 用现有历史预测候选 id；② 预取这些 id 的不可变行；③ 权威 id 到手后**逐位
精确校验**；④ 只消费权威 id 对应的行；⑤ 未命中/不匹配走原精确路径。这能藏住取行，但**若
主机仍必须先得知并校验真实 id 才能继续，它并不自动消掉 CPU 汇合**。"消费猜的行、事后再
纠正"被明确反对：作废并重放的清单包括 target 计算、KV/QSA/Mamba 状态、接受决策、输出发放。
正确性门槛**不用最终 token 串相等**，要求局部不变量：权威整数 id 与用于选行的 id 一致；
消费的 BF16 行字节与不可变 checkpoint 逐位一致；顺序、重复、请求身份、buffer 代际正确；
预测失手不得抵达消费者；注入预测失败必须走精确回退路径；覆盖拒绝/回滚、环形回绕、请求
复用、padding、EOS 历史、chunked prefill 边界。浮点不确定性**不是**整数 id 或行字节出错
的借口。性能上量"整次 lookup 预测成功率"与回退罚，不按单行命中率；一阶盈亏 `pH > (1-p)M + C`，
用实测值——**采样到的空闲占用率提供不了其中任何一个数**。

### 20.7 它最终的排序（与我们一致，仅一处微调）

① 混合 ×2/×3/×4 战役（带正确的延迟与步长账）② PLE 时间线 + 修好区间计数器
③ 在 ×4 上做有界匹配的"早到行"oracle ④ row-ID trace + 按批缓存模拟 ⑤ 深度布局审计
（便宜的源码审计 ≠ 实现）。派发优化与真异步仍在这些之下。
**我们的一处微调**：§20.1 第 1 条那个命中率口径是**5 行的小修**，我们想今天就落，不等
"完整测量改造"的那半天——因为它已经在污染每一个在跑的数字。

## 21. 命中率口径落地的代价：我们喂给 gpt-6 的分母是错的

修好口径（worktree `f33da54841`）之后重起一台 ×2 全量 + PLE，暖机一块、再量两块：
**147.1（暖机，丢弃）/ 146.5 / 151.1**，中位 **148.8**——与 §19 的 149.5 齐平，说明计时
账没被这次改动动到。真正变的是 PLE 那一行的含义：

```
PLE decode window: 200 lookups, 53.6 rows per lookup, 99.6% unique hits,
99.6% row hits, 98.5% all-hit lookups, miss p50/p95/p99 0/0/8, 0 lookahead lookups
```

| 窗口口径（每 200 步） | 实测范围 |
|---|---|
| 每次 lookup 的行数 | **48.0 / 53.6**（两种步形交替） |
| 去重行命中率 | 95.8–99.6% |
| 行出现次数命中率 | 与上几乎相同 |
| **整批无需读盘的 lookup 比例** | **83.5–98.5%** |
| 每次 lookup 缺行数 p50 / p95 / p99 | **0** / 0–16 / 8–40 |
| 窗口内 prompt 预取 lookup 数 | **0** |
| ids 等待 / 取行 | 10.7–10.9 ms / 0.33–0.67 ms |

三条推论，其中第一条是**纠错**：

1. **我们把分母报错了约 7 倍**。§19/§20 里"每步 350–520 行"是拿**累计**行计数除以 200 步
   得来的；真实值是**每次 lookup 约 50 行**。gpt-6 第三轮据此独立性算术推出"要 99.97–99.98%
   行命中才能让某步免掉主机汇合"——**那个结论建立在我们的错数上，按 ~50 行/lookup 重算，
   97% 行命中就给出约 22% 的整批全中，而实测整批全中是 83.5–98.5%**（行之间强相关，独立性
   假设本就不适用）。它要我们"按批模拟而不是用独立性算术"是对的，我们照做了。
2. **中位那一步根本不读盘**（每个窗口的 p50 缺行都是 0），尾部才需要几条 ⇒ 取行总成本
   只有 0.33–0.67 ms/步，**全部消掉在 18.3 ms 里也只值约 3%**。
3. 因此 **GPU 热点行缓存更没理由了**：它打的是 ≤0.67 ms 的那一段，而真正花钱的 10.7–10.9 ms
   是 **ids 汇合**，行从哪儿来它一点都不关心。§20 的降级现在有了正确的分母支撑。

### 21.1 `lazy` 加载的真实账单（这次才看清）

默认 `safetensors_load_strategy=None` = **memory-mapped lazy**（`config/load.py:70`）：起服
日志里的"Loading safetensors checkpoint shards 12/12 [00:43]"只是过表头，**真正 143 GiB 的
page-in 推迟到权重第一次被碰到时才付**。这次的账单：首个补全请求 **72.5 s / ~12 tok/s**，
curl 直接 60 s 超时；py-spy 显示 EngineCore 主线程仍在 `load_weights`（`fc_embedding.qweight`
→ fused_moe routed_experts），显存每分钟涨几百 MiB——**在推进，不是僵死**。第二次起服只用
了 4.5 分钟，因为上一次已经把大部分页拽进了 OS 缓存（注意：PLE 自己的读走 NO_BUFFERING，
**权重加载走的是普通缓冲读，OS 缓存有效**）。

⇒ 战役协议必须加两条：**暖机块并丢弃**（把这笔账排在测量窗口外），以及**测量块的步长要用
实测而不是接受长度÷请求级吞吐推算**。今天 ×2 三块"144.4→149.5→152.2 递增"的形态，现在有了
解释：那是 page-in 在跑，不是缓存在暖。

### 21.2 这轮踩到的三个工具坑（都已修/记）

1. **`_log_decode.py` 不认 UTF-16 BOM**，在第一个 NUL 处切会**错位**；而专用日志恰好是
   `FF FE` 开头的 UTF-16LE ⇒ 什么都不匹配。同时 Windows Python 的 stdout 默认 GBK，遇到进度
   条方块字符直接 `UnicodeEncodeError`。两处都已修（BOM 走 `utf-16`；stdout reconfigure）。
2. **起服脚本自己写日志**到 `_dev\out\logs\flashnext_<时间戳>.log`——**真日志在那里**，我们
   重定向的 `*_out.txt`/`*_err.txt` 只是 PowerShell 的中继。以后判读一律先看专用日志。
3. **上一条命令被工具超时连坐杀掉时，会打断 PowerShell 的日志中继管道，而 python 还活着并
   继续占着 64 GiB 显存** ⇒ 服务器变僵尸、请求全挂、日志不再增长。清场一律走
   `_dev/bin/_stop_vllm.ps1`，且**每条命令的耗时必须明显小于工具超时**。

### 21.3 本轮产出

- worktree `f33da54841`（PLE 窗口计数器）；主仓 `99af724b34` / `18ee06b3e9` /
  `67cdc45704` / `b5f5f2f4b4` / `667f9bfd6f` / `4f7f0d9a`（第四轮会诊存档）。
- `_dev/probe/_ple_eof_probe.c`（18 例 EOF 探针）、`_ple_overlap_probe.c`（OVERLAPPED 探针）。
- 日志：`_dev/out/logs/flashnext_20260930_{093700,102838,103808,105054,110608,111025}.log`、
  `_dev/out/{struct_mtp2,mtp3,mtp4,mtp2b,plefix,plefix2}_*.txt`。

---

## 22. 时间线插桩的结论：设备链 1.6 ms，宿主等待 11 ms

### 22.1 设计与开销

`VLLM_PLE_SSD_TRACE=1` 启用。每步 8 个预分配 CUDA event（环 1600 个，够 200 步），
**只在窗口关闭时读取**，所以计时不会在被计的路径上引入同步；每 200 步一次设备
同步。区间分层按 `tokens`（48 行一步与 53.6 行一步不是一回事，不许先合并）。

**开销实测 ≈ 0**：开插桩的三个正式测量块 **149.5 / 149.0 / 147.8**（中位 149.0），
与同日未插桩的 149.5 / 148.8 齐平。暖机门按 §21.1 执行（两块 145.5 / 146.9，
差 0.97%，中间无 `load_weights` 帧 ⇒ 判停，丢掉一块再测）。

### 22.2 解码步的区间（`tokens=3`，每窗口 n≈200，p50/p95 µs）

| 区间 | 含义 | p50 | p95 |
|---|---|---:|---:|
| `p->w` | 保存完成 → 侧流跨过 `_previous_use` 等待 | 318 | 336 |
| `w->j` | 跨过 → `wait_stream(current)` 完成（**排在主流已排队工作后的时长**） | **2.94** | 3.07 |
| `j->s` | 汇合 → D2H 开录 | 3.65 | 3.74 |
| `copy` | **D2H 本身** | **45** | 58 |
| `s->e` | D2H 开录 → 收录 | 3.68 | 3.97 |
| `e->h` | ids 落到宿主 → H2D 已排 | 1115 | 1570 |
| `h->c` | H2D 已排 → 消费者抵达 | 53.8 | 66.5 |
| **整条设备链** | 保存完成 → 消费者抵达 | **1548** | 2007 |

宿主侧同一批步：`gate`（等上一批行 H2D）**3.2 µs**、**ids 等待 11117 µs**、
`rows` 150 µs、`pending`（等待起点 → 交回主线程）11375 µs。
后续窗口重复同形状：链 p50 **1.55–1.95 ms**、ids **9.98–10.91 ms**。

### 22.3 判读：落在"宿主没及时看见"那一行

三个互相咬合的事实：

- `w->j = 2.94 µs` ⇒ 侧流在主流已排队工作上只等了 3 µs ⇒ **当时主流是空的**；
- `gate = 3.2 µs` ⇒ 上一步的行 H2D 早已完成 ⇒ 上一步的 GPU 工作**已经做完**；
- `h->c = 53.8 µs` ⇒ 消费者在 H2D 排好后 54 µs 就到 ⇒ 主流当时几乎没排队。

而设备自己从保存完成到消费者抵达只走 **1.55 ms**（D2H 只 45 µs），宿主的
`_ids_ready.synchronize()` 却花 **11.1 ms**。按 gpt-6 §20.4 的判读矩阵，这落在
"**D2H 结束 → 宿主醒来：GPU 已经完成，宿主没及时看见**"，而不是"上游计算依赖"。
**它同时削弱了 §19/§21 的"ids 生产者排在一步 GPU 工作后段"这个解释**：若生产者
真排在后段，`w->j` 与 `gate` 都不会是微秒级。

### 22.4 轮询臂：输了

把阻塞的 `Event.synchronize()` 换成 `query()` + `sleep(0.2 ms)`（`VLLM_PLE_SSD_IDS_POLL`，
默认关）去测"GIL 被等住"这一支：

| 臂 | 正式块 | 中位 | ids 等待 |
|---|---|---:|---:|
| A 阻塞等待（带 trace） | 149.5 / 149.0 / 147.8 | **149.0** | 9.98–10.91 ms |
| B 轮询等待（带 trace） | 148.0 / 145.0 | **146.5** | **11.42 ms** |

**B 比 A 低约 1.7%，且 ids 等待未缩短** ⇒ 若事件早已完成，轮询本该早早退出，所以
**事件本身确实晚了约 11 ms**，`query()` 也报晚 ⇒ "只是 GIL 抢不到"这一支被部分否证；
轮询臂不保留（代码留着，开关默认关）。暖机门在 B 臂同样执行过（W2 148.0 / W3 146.6，
差 0.95%，无加载帧 ⇒ 判停、丢掉 W3）。

### 22.5 顺带查清：ids 的 CPU→GPU→CPU 往返是可证明冗余的

`_compute_ngram_ids` 是**纯 CPU 计算**（输入全为宿主张量：`query_start_loc`、
`context_kv_metadata`、`prev_tokens`），`_device_ids` 只是 `copy_` 上设备再 `cpu()` 搬
回来，代码还专门断言搬回来的与 CPU 算的逐元一致 ⇒ 那趟往返**不产生任何信息**，只提供
排序和一个断言。

真正花掉 11 ms 的嫌疑因此转向**单缓冲带来的 WAR 序列化**：`wait_event(_previous_use)`
与 `wait_stream(current)` 共同保证"上一步的核已把 `_rows` 消费完"才能覆写同一块缓冲
⇒ 读盘被串行在 GPU 消费之后，而这条保护实际由 `wait_stream(current)` 提供。**下一件
该做的事是把这个串行拆掉（多缓冲轮转，约 25 行），而不是继续猜 CUDA 内部。**

### 22.6 本轮产出

- worktree `1e8f7b13f2`（时间线插桩）、`31c11d6ce4`（ids 轮询等待，默认关）；
  主仓 `7ac0f8f611`（`-PleTrace`）、`a0f4bd8324`（`-PleIdsPoll`）。
- `_dev/bin/_ps_parse_check.ps1`：不执行脚本只做 PowerShell 语法解析（以后每次改
  .ps1 都跑它）。本轮所有 .ps1 修改均 `parse OK`。
- 日志：`_dev/out/logs/flashnext_20260930_{113047,114630}.log`（A 臂 / B 臂）、
  `_dev/out/{trace,poll}_{out,err}.txt`。
- 停服确认：GPU1 `0 MiB`、`--- compute apps: (none)`。

---

## 23. 前缀测量与"等待前是否已就绪"：两个解释被劈开

### 23.1 它排的第一名：前缀锚点

gpt-6 第五轮把"派发→保存"的前缀测量排为第一名（一次重启 + 一块 ≈ 5.25 分钟），
并给了门：**~10–11 ms ⇒ reading 6 活；<2 ms 而 ids 仍 10–11 ms ⇒ reading 6 死；
2–9 ms ⇒ 归属不完整，需补两个标记**。

锚点落在 `model_runner.py` 的 `# Run model.` 之前（**一处覆盖 FULL / PIECEWISE /
eager 三条路**），与它的建议同向；图捕获期间跳过。槽位用 `_prefix_pending` 双端队列
配对，环回收，仍然不分配。site-packages 的补丁留 `.orig` 原件。

**结果：`model dispatch to preserve p50 36.86 µs`（三个窗口同一个值）**，而 ids 等待
仍 10–11 ms ⇒ 按它的门，**reading 6（"11 ms 是本步 PLE 之前的模型 GPU 工作"）被杀**。

但那个值本身可疑：三窗口 p50 完全相同，不像分布。在 breakable graph 下整步的核是在
派发时一次性入队的，所以这个区间测的很可能只是**两个 event 记录之间的固定开销**，
而不是任何有意义的工作量。**图模式下这个锚点不 informative**，需要 eager 对照才能判明。

### 23.2 轮询计数：第一次指向调度

第一次轮询臂（无 pre-query 仪器，142.3 tok/s）：

```
wait p50 10653.90 us, polls p50 20.0, us per poll 532.70,
seen true by main thread in 199 of 199 steps
```

按 0.2 ms 纯睡应该数到 ~53 次，实际 **20 次、每次 532 µs**（每轮多丢 ~330 µs 在被重新
调度上）；且主线程 199/199 看到事件已完成 ⇒ 当时看像**宿主调度**。

### 23.3 决定性判据：等待开始前先 query 一次

两行代码，两个臂都适用。阻塞臂（生产路径，暖机门后正式块 **148.5**）：

```
wait p50 10393.60 us, polls p50 0.0,
ready before waiting in 0 of 200 steps, seen true by main thread in 200 of 200 steps
```

**开始等待时事件在 200 步里 0 次已完成** ⇒ 那 10.4 ms 是**真实的 GPU 侧依赖**，不是
调度、不是 GIL、不是"宿主没及时看见"。到主线程到达查找点时它已完成（200/200）——
两句合起来把时间钉在**"从入队到 D2H 真正跑完"**这一段上。

§22.4 我当时写"事件本身确实晚了约 11 ms，query 也报晚"——这句现在**成立且被量化**；
而 §22 把它归入"GPU 已完成、宿主没看见"那一行，**现在看是错的**。

### 23.4 剩下的唯一候选，和测它的方法

侧流排在谁后面？前缀锚点看不见答案，因为那份工作在派发标记**之前**就已入队 ⇒
候选只剩 **上一个步骤尚未跑完的主流工作（GPU 落后 CPU 约一步）**。

现成的环就能测：`elapsed(上一步的 events[7]（消费者点）, 本步的 events[5]（ids 落地）)`
≈ 6 行。若 ≈ 10 ms ⇒ "GPU 落后一步"成立，那么可拆的方向是让 ids 生产者**不排在整步
之后**（host 侧推导 ids，或把它挪到单独流上早发），而不是拆缓冲轮转；
若 ≪ 10 ms ⇒ 时间住在侧流自己的交接里，另案。

### 23.5 本轮产出

- worktree `0af1ae53cc`（前缀锚点 + runner）、`7b81393e46`（轮询计数 + 主线程见证）、
  `1a5ae0d1ef`（等待前 query 判据）；site-packages 同步，`model_runner.py.orig` 留原件。
- 日志：`_dev/out/logs/flashnext_20260930_{123456,124505,125515}.log`（前缀臂 / 轮询臂 /
  阻塞臂）、`_dev/out/{prefix,poll2,pre}_{out,err}.txt`。
- 第五轮会诊存档：`consultations/astra_brief5.md` / `astra_reply5.md`（14.7 KB）。

---

## 24. eager 对照：那 10 ms 不是固有延迟，是入队节奏造出来的

### 24.1 两个模式的结构数字

同一套插桩、同一套参数，唯一差别是 `-Graphs`：

| 量 | 图模式（FULL，147 tok/s） | eager（`enforce_eager=True`，17.3 tok/s） |
|---|---:|---:|
| 前缀 `dispatch→preserve` p50 | 36.86 µs | **30.72 µs** |
| **ids 等待 p50** | **10393 µs** | **250 µs** |
| 跨步 `上一步消费者→本步 ids 落地` p50 | 16385 µs | 116180 µs |
| 整条设备链 p50 | 1548 µs | 3763 µs |
| `e->h` p50 | 1115 µs | 3306 µs |
| `ready before waiting` | 0 / 200 | 0 / 199 |
| `seen true by main thread` | 200 / 200 | 199 / 199 |

### 24.2 结论一：前缀锚点不可用

eager 下它也只有 30.7 µs，与图模式同量级。两种模式下这两个录制点的主流都是排空的
⇒ 两个 event 一入队就被执行 ⇒ **它测的是第二个 event 的录制开销，不是工作量**。gpt-6
指定的锚点位置在这里给不出它想要的量。**不要再把它当结论用**。

### 24.3 结论二：等待等的是"GPU 爬到本步 ids 生产者"，而代价来自入队节奏

`_start_layer_ple_prefetch` 只出现在 `model.py:516/528`，即 **model forward 内部**：
图捕获时它作为 eager 段留在原位，重放时按原位执行。所以等待的内容确实是"本步在
这个 PLE 层之前已经入队的主流工作"（gpt-6 的 reading 6 在这一点上是对的），但**它决
定性地不是固有延迟**：

- **eager**：宿主逐核入队、GPU 追得上 ⇒ 宿主走到 prefetch 时 GPU 已跑完前面的核 ⇒
  `wait_stream(current)` 几乎立即满足 ⇒ 等待 **250 µs**。
- **FULL 图**：整步核在一次重放里灌进队列 ⇒ 宿主立刻跑到 GPU 前面**一整步** ⇒ ids 要等
  GPU 爬到图里的生产者 ⇒ 等待 **10.4 ms**。

两者等的是同一件事，价钱差 40 倍，差别只在**宿主入队比 GPU 执行快了多少**。
`step gap` 在 eager 下是 116 ms（一步 ≈ 156 ms）、在图模式下是 16.4 ms（一步 ≈ 18.4 ms），
都是"一步减去前面那段"的形状 ⇒ 进一步支持这个解释。

### 24.4 由此重新排杠杆

- **"把 ids 生产者早发到单独流"这路死**：`_prepare_ngram_context` 在 `prepare_inputs`
  里、派发之前跑（`model_state.py:98-125`），但读的是 `num_computed_tokens.gpu` 与
  `all_token_ids.gpu`，这两者要上一步采样完才更新 ⇒ 早发的生产者仍排在同一条链上。
- **eager 不是出路**：17.3 tok/s vs 147 tok/s，差 8.5 倍。
- **剩下的主杠杆是 gpt-6 排第一的那条：宿主侧推导 ids**（它同时也把 D2H 省掉，读盘
  可以在提交时就开始、与整步重叠）。代价在它点出的正确性上：MTP 接受、请求打包、
  EOS 边界、与 GPU 路径用的确切 token 状态，需要跨普通解码与投机接受模式做**逐位比对**。
- 次要解释待验：图模式把宿主送到 GPU 前面一整步，同时也让 GPU 有六成时间空闲 ⇒
  **单流是入队节奏受限**，与 §21 的 util 读数一致。

### 24.5 本轮产出

- worktree `9bb0241401`（跳步区间）；日志 `_dev/out/logs/flashnext_20260930_{131613,132628}.log`
  （图臂 / eager 臂）、`_dev/out/{gap,eager}_{out,err}.txt`。
- eager 下 `timeout 140` 跑不完 70 秒窗口 ⇒ 控制实验要用短窗口小请求。

---

## 25. 第六轮会诊：原命题被证伪，但它换成了可以便宜验证的形式

### 25.1 一个不得不在意的发现：这一天的数字全在批队列路径上

`--async-scheduling` 在起服脚本里只在 `-AsyncSched` 时才加（`_flashnext_struct_serve.ps1:198-202`），
而我今天所有起服都没加它。可 `block` 这个字段**只在 `step_with_batch_queue` 里记录**
（`core.py:824`；普通 `step()` 只记 gap/submit/update，在 661/669/690），而日志里有
`block p50` ⇒ **异步调度是 vLLM 自己默认开的**（`async_scheduling is None` 走自动判定，
几个禁用条件都没命中）。

⇒ 今天所有吞吐、submit、block 都是**异步调度 + 批队列**下的读数。这也解释了早前那个
"`--async-scheduling` 无效"的观察：它本来就开着，加不加旗标自然看不出差别。

### 25.2 gpt-6 的四条源码断言，我们自己验了

| 断言 | 判定 | 证据 |
|---|---|---|
| 普通请求不做 draft D2H，宿主拿 `-1` 占位 | **真，但有条件** | `spec_decode/utils.py` 里占位分支上面那句注释明写"只在异步调度关闭时发生"；我们是异步 ⇒ 真 draft 值经 copy 流回宿主（`get_draft_tokens` 里有 `copy_event.synchronize()`） |
| 解码输入 ids 在 GPU 上拼装 | **真** | `input_batch.py:398+` = Triton 核 `_combine_sampled_and_draft_tokens_kernel`，从设备载 `last_sampled_tokens` / `draft_tokens` |
| 我们量的是批队列路径 | **真**，被自己日志证实 | `block p50` 只存在于 `step_with_batch_queue` |
| CPU `num_computed_tokens_np` 是乐观上界 | **真** | `states.py:61` 原话：`(upper bound on GPU value)` |

### 25.3 结论："宿主已经拿着 ids 全部输入"不成立，但死因不是它想的那个

两个独立的反例：

1. **精确的 `num_computed_tokens` 边界不在宿主上**（CPU 镜像明确只是上界）——ngram
   上下文的尾巴正好要靠这个边界；
2. **本步被验的 draft 候选值不在宿主的调度信息里**（调度器只有数量和占位）——即使
   它们后来会被拒绝，仍然影响本步的 ids。

但第 2 条在**我们的异步配置下其实有解**（真值经 copy 流回来了），真正缺的是把它
**连同版本/代际一起、在提交之前交出来**。所以 gpt-6 把提案从"推导已经知道的 ids"
改成"**先让一份完整、正确版本的输入快照更早可用，再推导**"，并要求把这个额外依赖
先定价再开工。

### 25.4 它给的排序与门槛（本轮照收）

| 序 | 动作 | 价值与代价 |
|---|---|---|
| 1 | **固定请求表、代码类加权、×2/×3/×4 战役** | 现有证据已是 ×4 +17.5%、×3 +4.2%，几乎无实现风险；3 次起服 + 每配置 3 块 = 23.25 分钟，加收敛块与丢弃块上限 ≈ 34.5 分钟 |
| 2 | **在 ×4 上有界早到行 oracle** | 不先建宿主镜像就能给剩余机会定价；一次起服、基线 3 块 + oracle 3 块 ≈ 11.5 分钟（过门之前） |
| 3 | **更早的快照导出 + 受保护的宿主侧解码 ids** | 只在正确性/来源检验与×4 oracle 有实质增益之后再做；成本是状态管道、生命周期测试、回退与反复验证，不只是一个哈希函数 |

门槛（×4，22.7 ms 步、接受 token 不变）：**省 0.445 ms = +2%；省 1.08 ms = +5%；
省 2.06 ms = +10%**。它不会为低于 ~2% 的 oracle 收益资助大规模状态镜像；≥5% 才值得。
另提醒：**submit 时间本身不是目标**——§21 的 PLE-free 已经演示了省下的提交时间会变成
输出端阻塞。

### 25.5 最便宜的定命实验（它的建议，不花重启）

在候选的早 prefetch 点装一个**不干预的可用性见证**（生产 PLE 路径不变），记录：
迭代/调用号与请求代际；宿主已处理到的输出版本；精确的 last-sampled 与 draft 值是
否存在（还是只有占位）；计算 token 位置是精确还是上界；打包与 padding；每个必需
字段首次可用的时刻。**不要**在那个点插 `.cpu()`/同步然后称之为"已经可用"。判读表：

| 结果 | 结论 |
|---|---|
| 提交时缺 draft 或上一步输出未处理 | 原命题假，需要导出/来源工程 |
| 候选完整但 token/上下文不符 | 宿主重建错了，在改行路径之前停 |
| 输入对但 ids 不同 | 哈希/EOS/溢出/布局实现错 |
| 输入与 ids 都对但可用时刻贴着现有 ids 到达 | 语义可行，时间收益未现 |
| 精确快照明显更早 | 去给行重叠定价，仍不证明端到端提速 |

### 25.6 行的缓冲：它接受的排序（单槽在串行使用下已足够正确）

它指出现在三重职责已经分开：`_copy_ready.synchronize()` 保护宿主行不被尚在跑的
H2D 覆写；`wait_event(_previous_use)` 保护 GPU 行在上一个消费者跑完前被覆写；
`wait_stream(current)` 只是把新 ids 排在 D2H 之前——**所以不必为了护行而保留那个宽泛
的主流等待**。它接受的方案是每槽每代际：`consumed[s, g-1]` → H2D → `ready[s, g]` →
消费者 → `consumed[s, g]`，并点了六个坑：将来任务必得在当前代际事件**已入队**之后
才算就绪；CUDA 流等待不保护宿主对 pinned 内存的写；`consumed` 要录在**最后一次 GPU
读暂存之后**；被等的上一个 `consumed` 必须已经录了/入了队；图重放用的是槽地址，
转 Python 引用不会更新已捕获的指针；捕获/dummy 执行不得造真 I/O 也不得吃掉生产代际。

### 25.7 本轮产出

- `consultations/astra_brief6.md` / `astra_reply6.md`（17.9 KB）。
- worktree `0af1ae53cc` / `7b81393e46` / `1a5ae0d1ef` / `9bb0241401`；主仓 `af21d31213` /
  `94d2a17c7d`。文档至此 1376 行。

---

## 26. 战役结果：深度增益是内容依赖的，混合代码流量上只有 +0.3%

固定 60 条留出请求（edit 14 / gen 12 / explain 8 / copy 8 / prose 10 / short 8，代码类占
70%），一次过表 = 一个块，顺序单流，`temperature=0`，输出上限 1024（多数请求以
`finish_reason=length` 结束，不让接受率被 EOS 早停污染）。三档深度各自起服一次，走同一条
暖机门（连续两块差 ≤2% 且中间无 `load_weights` 帧 ⇒ 丢掉最后一块），然后取三个计数块的中位。

| 深度 | 三个计数块 | 中位 | vs ×2 |
|---|---|---|---|
| ×2 | 119.2 / 119.5 / 117.4 | **119.2** | — |
| ×3 | 116.2 / 119.1 / 118.8 | **118.8** | **-0.4%** |
| ×4 | 117.2 / 119.6 / 122.1 | **119.6** | **+0.3%** |

分类中位相对 ×2：

| 类 | ×3 | ×4 |
|---|---|---|
| copy | +4.9% | **+10.5%** |
| short | +6.8% | **+10.8%** |
| gen | -0.6% | +4.5% |
| edit | +1.1% | +1.7% |
| explain | -0.9% | -0.9% |
| **prose** | -3.7% | **-6.8%**（≥5% 回归 ⇒ 触发不可晋级判据） |

**结论一：+17.5% 塌成 +0.3%。** 同一台机器、同一配置、同一深度，在合成重复流量上 ×4 给
+17.5%，在固定代码类加权表上只给 +0.3%。差别不是测量协议（两边都过暖机门、都单流、都
`temperature=0`），也不是请求长度（合成表里也有 10% 短请求），而是**内容**：分类账把它挑出来了
——`copy` 和 `short`（高度可预测）各 +10% 以上，`prose`（自由散文，接受率低）反向 -6.8%。
深度是一种**内容投机**，不是通用加速。

**结论二：×4 不能当默认。** gpt-6 的晋级判据里"没有任何重要类别出现 ≥5% 的可复现回归"这一条
被 prose 打破；而 ×3 的总体也没有过门槛。所以按它的矩阵走最后一格：**×2 保持默认，深度按内容
选择性开**（copy/short/gen 这类可预测输出可以吃到 +10%，长篇散文要留在 ×2）。

**结论三：这条结果反过来给 PLE 重构定价。** 战役要答的问题是"同样省 1 ms 在这份流量上值不值
5%"，而它顺带先答了更贵的那个：**×4 砍掉三分之一的 SSD 查表，混合流量只动 +0.3%**。若连查表
数量减少 33% 都换不来 5%，那"把 CPU 等待从 10.7 ms 里拿掉"在同一份流量上更不可能值 5% ⇒ 按
gpt-6 的资助规则（3–5% 之间 ⇒ 杀昂贵 PLE 重构），**PLE 重构现在不该被资助**。

**结论四：混合流量的钱在别处。** 同机同配置下，本表总体 119 tok/s 而长输出合成流量 147 tok/s，
差 19%——差在**每请求开销**（prefill + 响应处理），不是解码。若要抬这份流量的吞吐，先攻的是请求
生命周期开销，而不是解码步内的 PLE 等待。

## 27. 可用性见证的第一份读数（gpt-6 判据的第一格）

见证块只报存在与时刻，不读设备状态。第一次起服（×2）就打出：

```
PLE snapshot witness step=66100 reqs=1 padded=1 drafts_on_host=False computed_np_max=260
```

**在派发之前，本步的 draft 真值不在宿主上**（`draft_tokens_np is None`），即使异步调度开着——
因为 MTP 起草器本身就跑在这一步里面，真值要到步内才存在。所以 §25.1 那条"宿主重放前已拿着全部
输入"的强形式被否证之后，它的弱形式（"真 draft 值经 copy 流回宿主"）也**不足以支撑一个更早的
宿主侧 ids**：早期点根本没有本步的 draft。gpt-6 §6 判读表的第一格（draft 仍是占位 ⇒ 先修输入
可用性，再谈 PLE 重构）现在由实测填上了，而且修法比它设想的更硬：要么把起草器搬到宿主之前的位
置，要么就放弃"更早"这条路。

## 26/27 的产物

`_dev/probe/_battle_table.py`（请求表，源码由 `_gen_table.py` 从解析后的表快照重新生成，重包
前后解析结果逐字段一致）、`_battle_pass.py`（一次过表 = 一块，逐类速率 + 完成原因 + 错误标记）、
`_battle_summary.py`（分类中位与回归标记）、`-PleWitness` 开关与见证方法。

---

## 28. PLE 宿主路径不在关键路径上（同一份日志里的免费判决）

### 28.1 我原本想打的"PLE 整机开关"对照作废，原因两条

1. **做不成配平对照**：PLE-free 只能靠跳过 PLE 权重（`--language-model-only`）实现，那同时换
   了权重、输出、内存和图状态——正是 gpt-6 在 `astra_reply4` 里用来否证"133 vs 113 是 PLE-free
   天花板"的那三条；而 PLE 表不搬 SSD 就装不进显存。
2. **配置版替代（`-PleCacheMb 4096`）操作检查反向失败**：行需求量几乎不变（85.0/85.5/89.9 行
   /lookup），但命中率从 78/77/95% 掉到 39/39/46%，行装配从 2.4/2.6/0.9 ms 涨到 6.0/6.5/5.9 ms，
   设备链里只有 `e->h` 变长（1.3–5.0 ms → 7.7–8.1 ms），其余段逐字不动。同时宿主
   `avail_pagefile` 从 141.9 GiB 掉到 73.7 GiB（约 68 GiB 页文件被提交）⇒ 违反了"宿主低压才测"
   的纪律。此外两臂在同 prompt、`temperature=0` 下**生成并不相同**（copy `length 2/stop 6` vs
   `length 1/stop 7`，prose 4133 vs 3993 token）⇒ 跨重启输出会分叉，分类账无法逐请求配对。

结论：命中率差异可能只是 ids 流不同的后果，不能归因于缓存大小；这一臂不进任何结论。

### 28.2 免费的判决：同一臂内的 258 个解码窗口

`_trace_windows.py` 把每个 200 步窗口的 `rows / ids / chain / gap / pending` p50 拉出来做相关。
C512 ×4 臂（`flashnext_20260930_144630.log`，258 窗口）：

| 量 | p50 范围 | r(·, gap) |
|---|---|---|
| `rows`（行装配） | **220 – 8439 µs**（38 倍） | **-0.676** |
| `ids`（ids 等待） | 11253 – 14145 µs | +0.001 |
| `gap`（步长） | **19945 – 20560 µs**（±1.5%） | — |

行成本在一个 run 内摆 38 倍，步长只摆 1.5%，而且是**负相关**：宿主花 8.4 ms 装配行的那些窗口，
步子 20.0 ms；宿主只花 0.22 ms 的那些窗口，步子 20.3 ms。C4096 臂同形状（rows 1893–7015 µs，
gap 20008–20460 µs，r=-0.665）。

**判读**：宿主跑在 GPU 前面一整步，它的 PLE 活（ids 等待、行装配、SSD 读）**重叠在 GPU 的下一步
工作里**，所以既不进步长也不进吞吐。给这个结论背书的是 §26 那条端到端证据：×4 砍掉 1/3 查表，
混合流量只动 +0.3%。

### 28.3 于是资助判定变了

- **PLE 的宿主侧工作（ids 等待、行缓存、按批缓存、异步 reader、row-ID trace 优化）在当前流量上
  不可资助**：它不在关键路径上，省下来的时间会被 GPU 那一步吃掉。
- **PLE 的 GPU 侧工作仍在关键路径上**：结构视图（连层一起去掉）在 ×2 上给 +10.4%。这两件事不矛
  突——去掉层缩短的是 GPU 步子，去掉宿主等待什么也不缩短。
- 与"GPU util 40%"并存不矛盾：步子由 GPU 爬过**上一步剩下的 ~16 ms** 决定（§24 的入队节奏），
  利用率低但仍然是 GPU 在给步子定节奏。
- 因此下一批钱应该花在：(a) 步子里 GPU 那 20 ms 的构成与入队节奏；(b) 每请求开销（§26.4，119 vs
  147 那 19%）。**不再花在 PLE 宿主路径上。**

## 28.4 方法论教训（值得写下来的两条）

- **跨重启的分类账不是配对实验**：同 prompt + `temperature=0` 在两次起服之间也会分叉（MoE/原子
  归约非确定），所以"固定请求表"只固定了**输入**，没固定**输出流**。跨臂比较只能用聚合率，不能
  用逐请求。
- **一个配置开关可以在宿主侧吃掉几十 GiB 页文件而不改变步长**：`ple_ssd_cache_mb` 8 倍就把宿主推
  进了另一个内存状态。任何配置实验起服之后、测量之前，必须先读 `_hostmem.py` 并记下
  `avail_pagefile`。

---

## 29. "每请求开销"这 19% 是接受长度，不是生命周期

原本以为 119 vs 147 的差在 prefill 与响应处理。三个战役日志的 `_idle_share.py` 把它否证了：

| 日志 | 10 s 区间数 | 引擎空转 | 只有 prefill | 运行中解码速率 p10/p50/p90 |
|---|---|---|---|---|
| ×2 固定表 | 136 | **0 (0.0%)** | 0 | 97.9 / 116.1 / 129.5 |
| ×3 固定表 | 116 | **0 (0.0%)** | 0 | 91.0 / 116.2 / 133.4 |
| ×4 固定表 | 135 | **0 (0.7%)** | 0 | 91.1 / 118.6 / 135.7 |

⇒ 引擎在请求之间**不空转**，prefill 以 14–58 tok/s 低速穿插，从不独占一个区间。所以那 19% 不在请
求生命周期里。

剩下的解释是接受长度：同一 ×2 深度，固定表上平均接受长度 **2.31–2.40**，合成重复流量上
**2.68–2.83**（§19.1）⇒ 每步少 ~14.5% 的接受 token。反推步长：固定表 2.35/119 ≈ **19.7 ms**，合
成 2.75/147 ≈ **18.7 ms**（14.5% × 5.3% ≈ 19% ✓ 数目对得上）。

固定表上的逐位置接受率（×4 臂）：

| 位置 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 接受率 | 0.86–0.89 | 0.64–0.73 | 0.45–0.65 | 0.36–0.58 |

×2 在固定表上已经拿到 1.35 个接受 draft（接受长度 2.35），×4 多花两次起草换回 ~1.0 个 ⇒ **大致打
平**，这就是 §26 里 ×4 只值 +0.3% 的机制。起草吞吐 163.6 tok/s 对接受吞吐 56.4 tok/s，起草本身是
真 GPU 开销。

**剩下的杠杆按期望值/成本排序**（PLE 宿主路径已判死，见 §28）：

1. **那 20 ms 的 GPU 步子由谁决定**——我们一整天量的是宿主侧与 PLE 计时，而给步子定节奏的是
   GPU。结构视图已经证明其中约 10% 是 PLE 层的 GPU 活，剩下 90% 没有归因。这是唯一还没有数的大
   项，也是最便宜的下一枪（一次起服 + 一个短窗口 + torch profiler 的核级归因）。
2. **起草器质量**：接受长度 2.35 → 2.75 在固定表上就是 **+17%**，比今天所有杠杆都大，但要动引擎
   级起草路径。
3. **按类别路由深度**：算细账只有 +2.6%（copy 13.4%×10.5% + gen 25.5%×4.5% + short 0.9%×10.8%，
   加上避开 prose 的 -6.8%×18.4%），而且需要每请求深度控制——vLLM 目前没有这个入口。性价比低。

---

## 30. Launch A：步子由设备功定，稳态 93.4% 的墙钟在流窗口里

第七轮给的排序第一条是"在 FULL 重放外面加一对 event，外加图内若干相位对"。发现**这套东西已
经存在**：`StepTimingCollector`（`async_utils.py:48`）给每步记 `forward_start/end` 与
`drafter_start/end`，整块只在退出时做一次同步（与我 PLE 插桩同一设计原则）。但它此前**只被图捕
获期使用**，而且那三个 mark 只写在 `_dummy_run` 里 ⇒ 生产路径的 `propose`（`model_runner.py`
生产段）从没被标记 ⇒ 窗口静默无输出。补三行之后才有数（worktree `d9e51eec8f`）。

**观察者成本**：带窗口的 pass3 = 119.0 tok/s，对照战役 ×2 中位 119.2 ⇒ **0.2%**。

| 阶段 | pass | 窗口 span p50 | forward p50 | drafter p50 | busy |
|---|---|---|---|---|---|
| 暖机 | stw1 (105.3) | 21.5–21.9 ms | 19.2–19.6 | 2.36 | 93.5% |
| 暖机 | stw2 (116.2) | 20.5 / 18.4 ms | 18.1 / 15.7 | 2.36 | 90.6 / 76.7% |
| 稳态 | stw3 (119.0) | 18.6 / 17.2 ms | 16.2 / 14.8 | 2.36 | 93.5 / 93.4% |
| 稳态 | stw4 (120.3，丢弃) | 19.1 / 17.8 / 16.9 ms | 16.7 / 15.4 / 14.6 | 2.36 | 93.5 / 93.4 / 93.3% |

三条读数：

1. **步长跟着设备 span 走**：span 在窗口间摆 16.9↔19.1 ms，busy 稳在 93.3–93.5% ⇒ 步子不是被等
   待卡住的，是被设备功推着走的。
2. **约 6.5% 的恒定余量**（≈1.3 ms/步）是 span 之外的墙钟。按 gpt-6 的资助算术（20 ms 步：2% =
   0.40 ms、5% = 1.00 ms、10% = 2.00 ms），把这 1.3 ms 砍一半只值 **+3.3%**。
3. **drafter 恒定 2.36 ms ≈ 步子的 12%**，与内容无关；forward 占 73–80%。drafter 砍 25–50% 值
   +3–6%，但它产出的是把吞吐翻倍的那 1.35 个接受 draft（§29），所以只能从"同样两次起草更便宜"
   这个方向动，不能从深度动（深度已判为打平）。

**必须留着的保留意见**：`elapsed_time` 量的是**流窗口的长度**，不是核时长之和 ⇒ 93.4% 只说明这条
流有 93.4% 的时间处于"已入队未排空"的窗口里，窗口内部仍可能有空洞（我们已经在同一步子里量到过
`e->h` 1.3–5 ms 的内部空洞）。所以"不可省"这一格还没签字——需要核时长之和与事件 span 对得上才算。

## 31. Launch B 已装成开关（nsys 不在这台机上）

`nsys` 未安装（只有 Nsight Compute 2026.2.1），所以按 gpt-6 的 fallback 走 torch profiler：
`-Profiler` 开关用点号式 CLI，全部额外记录关死（`with_stack/record_shapes/with_memory/with_flops
=false`、`ignore_frontend=true`），`wait 5 / warmup 3 / active 30`，输出
`D:/code/vllm-windows/_dev/out/profile`。判据照它给的：**若核时长之和与 CUDA 事件 span 明显对不
上、或重放计时变化超过 2–3%，那一臂只能当归因筛子，不能当吞吐基线**。

## 32. 工具教训：`pi --print` 只在结束时写 stdout

后台起会诊时，`sleep` 必须超过 pi 的**完整**运行时间，否则命令一结束就把还没写完输出的进程连坐
杀掉（第七轮第一次：420 s、第二次轮询 840 s 都得 0 字节）。改成**前台等待**后一次成功：1122 s、
15.9 KB。LM Studio 在 `localhost:12346`，`gpt-6-astra` 在列。

### 32.1 补一条：解析检查要在提交之前跑

加 `-Profiler` 时我把一个 `}` 粘进了 `--additional-config` 那一行，`$PleSsd` 块的收尾坏掉，而
`_ps_parse_check.ps1` 是在 commit 之后才跑的 ⇒ 坏脚本进了历史（`c2e8f5ce7a` 修在
`7e3eff7e46`）。**顺序改成：改完 → 解析 → 再提交。**

---

## 33. Launch B 落空：torch profiler 在这台机上不吐核记录

起服带 `-Profiler`（点号式参数修好之后，`5f15c5d406`）→ 暖到门（pw1 106.7 / pw2 112.5 / pw3 113.5，
差 0.9%，无加载帧 ⇒ 丢 pw3）→ 在计数块里 `POST /start_profile`、5 秒后 `POST /stop_profile`。
排程按 `wait=5, warmup=3, active=30` 走通（日志确认），产物落
`_dev/out/profile/`：一份 `*.pt.trace.json.gz`（988 KB）加一份 `profiler_out_0.txt`。

**结果是零核事件。** 64481 个事件的类别分布：`cpu_op` 64413、`user_annotation` 60、`Trace` 1、
`None` 7，**`kernel` 0、`gpu_memcpy` 0**；`profiler_out_0.txt` 里也只有 CPU 列，没有 Self CUDA 列。
所以 gpt-6 排第二的"核级归因"在这台机上换不了工具就落不了地：CUPTI 在这条 Windows/CUDA 路径上没
有产出核记录。

**这一臂仍然给了两样有用的东西：**

1. **扰动是可测的**：门后吞吐 113.5 → 计数块 **109.5（-3.5%）**，正落在 gpt-6 划的"超过 2–3% 就不
   能当吞吐基线"那一格 ⇒ 判为归因筛子都算不上（它连归因对象都没有）。
2. **CPU 侧的视图是真的**：`ProfilerStep*` CPU total 683.1 ms / 30 = **22.77 ms/步**；FULL 重放的
   注解 `execute_context_0(0)_generation_1(3)` 自 CPU **14.33 ms/次（28 次）**——这显然含 CUPTI 追
   踪开销，不能当成"重放真的花 14 ms 宿主时间"（未插桩时 `submit p50 ≈ 22 ms`、且宿主跑在 GPU 前
   面一整步）。宿主侧 aten 里量级最大的是 `copy_` 22.8 ms/30 步 = 0.76 ms/步、`index` 0.44 ms/步、
   `slice` 0.46 ms/步、`as_strided` 13066 次调用。

## 34. 归因还剩两条路，需要拍板

- **装 Nsight Systems**（现在只有 Nsight Compute 2026.2.1）。这是 gpt-6 排第二的正解：`--cuda-graph-trace=node`
  + `--capture-range=cudaProfilerApi` + `--profiler-config.profiler=cuda`。代价是要下载几百 MB，而且是
  你的机器上的安装决定。
- **图内相位 event 对**（gpt-6 排第一那条的加强版）：在少数几个模块的 forward 上挂 event（目标模型
  进出、首层注意力、一个 MoE 块、PLE 层、采样器、起草器），捕获时把 record 烘进图里，重放会重新
  record，退出时一次读。**可行性已经由 PLE 插桩证明了**——`events[6]` 就是烘进图里的 record，它给了
  稳定可复现的每重放区间。代价：一次起服 + 暖机到门 + 若干窗口（~20 分钟），不需要任何下载。

---

## 35. A 路（nsys）在这台机上被权限挡住

好消息：**不用下载**——`Nsight Systems 2026.1.3` 已经装在
`C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3`（`target-windows-x64\nsys.exe`，
`--version` = 2026.1.3.425），并且支持 `--cuda-graph-trace=<granularity>` 与
`--capture-range=cudaProfilerApi`。于是起了 nsys 臂：`-NSys` 开关把引擎收回同进程
（`VLLM_ENABLE_V1_MULTIPROCESSING=0`）并加 `--profiler-config.profiler cuda`，让捕获只在
`/start_profile`→`cudaProfilerStart` 与 `/stop_profile` 之间打开（`78740fcc64`）。

- 暖机门：nw1 102.0 / nw2 111.1 / nw3 113.4 / nw4 114.4（差 0.9% ⇒ 丢 nw4）。
- 捕获块 **113.8**（对门后 114.4 只差 **-0.5%**）⇒ nsys 的扰动比 torch profiler 的 -3.5% 温和得多。
- 结果：**`Generated: No reports were generated`**，`_dev/out/nsys/` 空。硬杀停服也可能让 nsys 没
  机会落盘，但这不是主因（见下）。

**一分钟控制实验把它判掉了**：`_dev/probe/_cuda_belly.py`（200 次 2048² BF16 matmul + 20 次
pinned D2H）在 nsys 下正常跑完并生成 `belly.nsys-rep`（45 KB），然而

```
cuda_gpu_kern_sum → SKIPPED: sqlite does not contain CUDA kernel data
cuda_api_sum      → SKIPPED: sqlite does not contain CUDA trace data
```

⇒ **连 API 记录都没有。** 一个平凡 CUDA 程序都记不到，就不是 vLLM 接线的问题，也不是捕获范围的
问题：nsys 在这台机上没能把探针注入进目标进程。它开跑时明说 `WARNING: CPU sampling requires
administrative privileges, disabling.`，而往别的进程注入 DLL 恰好是需要特权的动作（SeDebug 一类）。
所以 torch profiler 与 nsys 在这里是**同一个盲点**：宿主上的 CUDA 归因工具全都被权限/策略挡在外面。

### 35.1 如果你愿意提权，A 还剩一次机会（命令给你）

在**管理员权限**的 PowerShell 里：

```powershell
$env:CUDA_VISIBLE_DEVICES = '1'
& 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe' `
  profile --trace cuda,nvtx --force-overwrite true `
  --output D:\code\vllm-windows\_dev\out\nsys\belly2 `
  D:\code\vllm-flashtest\.venv\Scripts\python.exe `
  D:\code\vllm-windows\_dev\probe\_cuda_belly.py --ops 200 --size 2048
& 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe' `
  stats --report cuda_gpu_kern_sum D:\code\vllm-windows\_dev\out\nsys\belly2.nsys-rep
```

若这次报出核名，A 就能继续（那时再把 vLLM 那臂用同一条路子起）；若仍然空，A 彻底出局。

### 35.2 只剩 B：图内相位 event 对

不需要任何外部工具、不需要提权，而且**可行性早已被我们自己的 PLE 插桩证明**：`events[6]` 是烘进
图里的 `cudaEventRecord`，每次重放都重新记录，给出稳定可复现的每重放区间。它能把时间**定位到相
位**（目标模型进出 / 首层注意力 / 一个 MoE 块 / PLE 层 / 采样器 / 起草器），但**给不出核名**——核
名只有 A 能给。所以顺序是：**B 先跑（定位），A 只在你能提权的前提下再跑（命名）**。

---

## 36. 外部归因在这台机上判死，原因在硬件：CMP SKU

用户在提权 PowerShell 里跑了 §35.1 的命令：**那条 CPU 采样告警不见了**（⇒ 确实提权了，ETW 类表进了
sqlite），但 `cuda_gpu_kern_sum` 仍然 `SKIPPED ... does not contain CUDA kernel data`。belly2.sqlite
里 30 张表只有 `PROCESSES`(420)、`TARGET_INFO_GPU`(8)、`TARGET_INFO_SYSTEM_ENV`(48) 与调度/ETW 枚举，
**一张 CUPTI activity 表都没有**。

我又拿另一个构建试了一次（Nsight Compute 自带的 `nsys.exe`，版本 **2026.2.1.0**，比独立安装的
2026.1.3 更新）：`belly3.nsys-rep` 正常生成，`cuda_gpu_kern_sum` 仍然"no CUDA kernel data"。⇒ **两
个 nsys 构建 × 提权与否 × 平凡 CUDA 程序，四次全空。**

然后 `--gpu-metrics-devices=help` 把原因摊开了：

```
GPU Metrics: None of the installed GPUs are supported:
        Ampere GA100 | NVIDIA Graphics Device PCI[0000:0c:00.0] - CMP SKU
        Ampere GA100 | NVIDIA Graphics Device PCI[0000:0b:00.0] - CMP SKU
```

**CMP SKU** 是 NVIDIA 的矿卡专线（GA100 核心的 CMP/IGMP 部分），与本仓项目目录名
`qwen3.8-flash-next-cmp170hx` 对得上。这类部分的驱动路径对 CUPTI activity 追踪受限，因此一条就同
时解释了三件事：nsys 记不到核（连 API 记录都没有）、torch profiler 记不到核、GPU metrics 直接报不
支持。**外部 CUDA 归因在这台机上不是配置问题，是硬件/固件问题。**

### 36.1 结论与唯一剩下的路

- **A（核名归因）出局**，**A′（GPU metrics）也出局**（外部采样同样不被支持）。核名要拿只能去那台
  参考机或 WSL 里试，但那不是本机的问题域。
- **只剩 B：图内 `cudaEventRecord` 相位对**。它不碰 CUPTI、不需要提权、不需要下载，而且已经在本机
  上被证明可行（`events[6]` 是烘进图里的 record，重放每次重新记录、区间稳定）。它能把 15–19 ms 的
  forward **定位到相位**，但给不出核名，也分不出相位内部"功 vs 空洞"——那需要核名或 GPU 计数器，
  而这两样在这台机上都拿不到。
- 因此资助判断只能用相位占比来做：某相位 ≥1.00 ms ⇒ 过 5% 线；≥0.40 ms ⇒ 过 2% 线（20 ms 步）。

---

## 37. 部署真相、一次我自造的假线索，以及 B 的落地

### 37.1 我编了一个不存在的 E: 盘，然后信了三十五分钟

读 `config.json` 时我用了一个脑子里冒出来的路径 `E:\models\Qwen3.8-Flash-Next-Base`，打不开，于是
顺着"盘掉了"这个故事做了整整一段磁盘取证：`Get-PSDrive`、`Get-Partition`、`Get-Volume`、`Get-Disk`、
WSL `/mnt/e`、D: 上找替身权重。**结论是这台机器从来没有过 E: 盘**，权重一直在 `D:\models\`，PLE 的
SSD 直读读的就是模型目录自己的 safetensors（没有独立的 `vllm_ple` 目录）。什么都没坏，什么都没被
堵住，我白白丢了三十五分钟。

教训写进纪律：**任何路径都要先从启动脚本/日志里取，不许凭记忆写**。凭记忆写出来的路径打不开时，
第一动作是怀疑自己的记录，而不是怀疑环境。

### 37.2 服务器 import 的是哪一份 vLLM（此前一直没钉死）

最近所有起服都传了 `-Venv D:\code\vllm-flashtest`，所以**运行副本**是
`D:\code\vllm-flashtest\.venv\Lib\site-packages\vllm`（引擎 `v0.1.dev21640+gb629bcc71`），它是
`D:\code\vllm-flashtest` 这个 worktree（branch `flash-next-win`，`d9e51eec8f`）的一份安装。运行副本
的 `model.py` 与 worktree 源码在归一化行尾之后**逐字节同内容**（各 1103 行，diff 0 hunk）⇒ 改源码再
拷进 venv 是安全的，也不存在"装了但源码没有"的暗改。

`D:\code\vllm-windows\.venv` 装的是路线 A 的 main（`0.29.1.dev0+g13e844c86.d20260926`，非 editable），
**这些起服一次也没用过它**。两份 `model.py` 内容真的不同：route-A 那份写 `hidden_states = hidden_states
+ self.ple(...)`，运行这份写 `hidden_states = self.ple(...)` ⇒ 拿 route-A 那份当参照一定会引错行号。

起服日志里的真实旋钮（17:42 那次战役）：`quantization=inc`、`cudagraph_mode=FULL_AND_PIECEWISE`、
compile `mode=NONE`、capture sizes `[1,2,3,4,6,8,12,16,24]`、`kv_cache_memory_bytes=15032385536`、
MTP `num_spec_tokens=2`、`ple_ssd_cache_mb=512` / `workers 16` / `io_depth 256` /
`prefetch_tokens 16384`、`language_model_only`、`max_num_seqs=4`、`max_model_len=4096`。

### 37.3 模型构成（决定 B 该采哪几层）

`D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`：**48 层**、hidden 2560、**512 experts / top-10**、
`hc_count 4`、MTP 1 层。`layer_types` = **36 × `linear_attention`（GatedDeltaNet）+ 12 ×
`qwen_sparse_attention`（在 3, 7, …, 47）**，每层都带 routed-expert MLP。
`ple_layer_ids=[2]` 的判据是 `layer_idx + 1 in ple_layer_ids` ⇒ **PLE 实际落在 layer 1**，不是 layer 2。
所以"每层一个 PLE"是错的假设；层与层不同类型，采样必须按类型分层。

### 37.4 B 的形状（已落地；门没开时完全惰性）

`vllm/utils/phase_events.py`：命名 event 对。`begin/end` 记在当前流上；捕获时 `cudaEventRecord` 被烘
成图节点，重放每步重新记录；worker 每步 `harvest()` 一次，**只读 query 已完成的那些对**，所以观察者
永远不吃设备同步。13 对/步 ⇒ 26 条 record ⇒ 预计 26–52 µs/步（0.15–0.3%），起服后照例量扰动。

相位表：`model`、`logits`、六个层跨度（**0、1、3、16、32、47** = 首层、PLE 层、首个 QSA、1/3 处、
2/3 处、末层 QSA），外加两层的内部相位（layer 1 的 `ple/attn/mlp`、layer 32 的 `attn/mlp`）。相位嵌套
成立：**层跨度 − (ple + attn + mlp) = 超连接的功**。

harvest 落点在 `sample_tokens` 里 `forward_end()` 之后——那是每步都跑的 Python，即使这步只是图重放
（重放时模型里的 Python 一行不跑，标记只靠烘进去的节点）。开关：`-PhaseEvents [-PhaseEventsLogEvery N]`
⇒ `VLLM_PHASE_EVENTS=1`。回滚：三个文件的 `*.preB.orig` 快照 + 删掉新文件。

判资助线的尺子不变：20 ms 的步子上 **≥1.00 ms 过 5% 线、≥0.40 ms 过 2% 线**。

---

## 38. B 出数：14.7 ms 的 forward 拆到相位，账能闭合

### 38.1 三个接线错误（诚实记账）

B 从"写完代码"到"出数"之间踩了三个坑，都不是测量学问题，是接线问题：

1. **穿模块树找不到观察者**：worker 只查顶层 `self.model`，而这里解析出的顶层是
   `Qwen4ExpForConditionalGeneration`，语言模型挂在 `language_model` 下面 ⇒ 改成进程级单例。
2. **按不存在的属性名登记相位**：`getattr(layer, "attn")` 在层上没有这个东西（只有
   `linear_attn` / `self_attn`）⇒ 注意力跨度从没登记，11 对而不是 13 对就是这么来的。
3. **harvest 挂在了 dummy 路径上**：锚点那段 `forward_end()` 属于 `_dummy_run`（772 行起），
   生产的是 `sample_tokens`（2101 行起，生产标记在 2246/2247/2264）⇒ 整个运行只 harvest 了一次。
   日志里那三行自证（plan / armed / harvest first）是把这三坑分辨出来的唯一原因。
   教训：**锚点文本长得像不代表位置对，必须确认它所在的函数在生产路径上。**

### 38.2 观察者成本：量不出来

计数三块 **119.6 / 120.4 / 120.3 ⇒ 中位 120.3**，对 ×2 基线中位 119.2 是 +0.9%。观察者（26 条
device record + 26 条 host query）不可能让解码变快，所以这就是跨重启的噪声/热态；连同 Launch A 的
0.2% 一起读：**这套观察者在本机的吞吐成本低于噪声底**。

（记法提醒：`_battle_summary.py` 只认以 `m1/m2/m3` 结尾的计数标签，别的标签会被当成 0 个计数块。这
次的 `e1/e2/e3` 因此只能手算中位。下次计数块请沿用 `m1/m2/m3`。）

### 38.3 尺度问题：所有读数被同一个因子放大，比值不受影响

事件对是在宿主落后于设备队列的位置上读的，所以每个跨度都被同一个因子放大。标定用 Launch A 的
`forward p50`：`model 134.711 / forward 14.737 = 9.14` ⇒ 除以它就是把相位换回"步"单位。**比值不依
赖这个因子**，所以资助判断优先用比值。

112 行表、每相位 ~28,000 个样本：

| 相位 | 原始 p50 (ms) | 换回步 (ms) | 说明 |
|---|---|---|---|
| model（整个 forward） | 134.711 | **14.737** | 标定锚点 |
| layer0 | 6.313 | 0.691 | 最重的层（首层要落地 HC 状态、还要发下一层的 PLE 预取） |
| layer1（PLE 所在层） | 3.548 | 0.388 | ple 0.0123 / attn 0.196 / mlp 0.163，残差 0.017 = 超连接 |
| layer16 | 2.814 | 0.308 | 典型 GDN 层 |
| layer32 | 2.832 | 0.310 | attn 71.9%，mlp 16.1%，残差 12% |
| layer3（QSA） | 2.384 | 0.261 | |
| layer47（末层 QSA） | 2.349 | 0.257 | |
| logits | 0.780 | 0.085 | 0.58% ⇒ 连 2% 线都不过 |
| layer32.mlp | 0.456 | 0.050 | W2G64 层的 MoE 很便宜 |
| layer1.mlp | 1.487 | 0.163 | W3 层的 MoE 贵 3 倍 |
| layer1.ple | 0.112 | 0.012 | **PLE 的 GPU 功只占 forward 的 0.08%** |

**按层型摊开能对回整步**：36 个 GDN 层 × 0.31 + 12 个 QSA 层 × 0.26 + layer0/layer1 ≈ **15.3 ms**，
对上 `model` 的 14.74 ms（差 4%）⇒ 层外面没有藏着大块功，这张表是完整的。

### 38.4 资助判定（尺子：≥1.00 ms 过 5% 线，≥0.40 ms 过 2% 线）

- **GatedDeltaNet 注意力是最大的功**：每层 ~0.22 ms × 36 = **8.0 ms，占 forward 的 54%**。砍 10%
  = 0.8 ms/步 = +4%；砍 20% = 1.6 ms = +8% ⇒ **过 5% 线，这是唯一值得开的新战场**。
- **MoE 的功按量化分配强烈分层**：W2G64 的层（12-35）mlp ≈ 0.05 ms，W3 的层（0-11、36-47）≈ 0.16 ms
  ——与 `build_info.json` 里 `backbone_routed_expert_allocation` 完全对得上。24 层贵、36 层便宜
  ⇒ 量化分配本身是杠杆，但那是权重制备的决定，不是核的事。
- **logits 判死**：0.085 ms = 0.58%，两条线都不过。
- **PLE 判死（两侧都判死）**：宿主路径 §28.4 已经判掉；GPU 侧现在量到 0.012 ms/步 = 0.08% ⇒ 连 2%
  线的一半都不到。**PLE 这条线到此关闭，不再资助。**
- **起草器仍是已知最大单项**：2.359 ms/步，占整步 13.7%，且不受本次标定影响（它是宿主标记测的）。
- **剩下解释不了的仍是步内余量**：forward 14.74 + drafter 2.36 = 17.1 ms，而步长 p50 ~20 ms ⇒ 还有
  ~2.9 ms/步（≈15%）不在任何相位跨度里。B 到这就到顶了：再往下要核名或 SM 计数器，而这两样在这台
  CMP 卡上都拿不到（§36）。

---

## 39. 日志普查：把「哪份在跑 / 参数一不一样 / 有没有显存占满」三问一次答掉

### 39.1 来源与工具

`_dev/out/logs/` 里有 **105 份**服务日志。它进了 `.gitignore`（§commit 424042f845），所以任何
`git clean -fd` 都动不了它——这次它替我们把现场保住了。新工具
`_dev/probe/_log_argv.py` 从每份日志里取三个锚点：

1. `non-default args: {...}` —— 真正进到 API server 的参数字典；
2. `Initializing a V1 LLM engine (<build>)` —— 应答的那个**构建串**，它直接暴露解释器
   是从哪份工作树 import 的；
3. 同一行上的 `enforce_eager=` / `cudagraph_mode` / `quantization=`。

启动脚本不是权威，** argv 和构建串才是**。

### 39.2 哪份在跑（版本普查）

105 份里 **60 份走到了引擎**：

| 构建串 | 起服次数 | 是哪份 |
|---|---|---|
| `v0.1.dev21640+gb629bcc71` | **55** | worktree `D:\code\vllm-flashtest`（branch `flash-next-win`）的 venv |
| `v0.29.1.dev0+g13e844c86.d20260926` | **2** | 主仓 `D:\code\vllm-windows` 的 venv（路线 A 编的 main），09-27 19:39/19:49，struct 视图 + eager |
| 构建串取不到 | 3 | 没走到 EngineCore 行 |

⇒ 全部测量臂都在 worktree 那份构建上；主仓那份**也起过服**（两次 struct 烟测），但从没进过任何测量臂。
深度普查顺手也出来了：**×2 20 次、×1 13 次、×3 2 次、×4 2 次**，其余是无 MTP 的 struct 臂。

### 39.3 上下文与 KV：这 60 次里参数一点没漂

- `max_model_len = 4096` —— **60 次全部如此，零例外**。
- `max_num_seqs = 4` —— 全部如此；`max_num_batched_tokens = 2048` 只有一例外
  （09-29 23:14 那次给到 4096）。
- `gpu_memory_utilization = 0.9` —— 全部如此。
- **async scheduling 只有 1 次**（09-29 23:09 那一个实验臂）⇒ 生产臂从来没开过它。
- KV 两种给法同时存在：`gpu_memory_utilization 0.9` 一直被传，但生产臂另外钉死
  `kv_cache_memory_bytes = 15032385536`（**14.0 GiB**，33 次）；09-29 的三个实验臂分别用
  `17179869184`（16 GiB）和 `12884901888`（12 GiB）。
  钉 bytes 会让 vLLM **跳过显存探测** ⇒ KV 大小是我们的决定，不是分配器试探出来的。
- 引擎自己打印的余量：`GPU KV cache size: 70,041 tokens, Maximum concurrency for 4,096 tokens
  per request: 17.10x`（生产档），另有 88,320/21.56x 和 49,444/12.07x 两档。
  `--max-num-seqs 4` ⇒ 我们只用掉 **4 / 17.1 ≈ 23%** 的 KV 余量。

### 39.4 「显存占满导致变慢」：在这 105 份日志里查无此事

扫全部 105 份找 `preempt` / `recompute`：**没有任何一条来自 Flash-Next 的起服**。唯一两条命中
是 09-21 的一次崩溃转储，里面是 `cmpl-bench-*` 请求 id、`num_spec_tokens=7` —— 那是**另一个项目**
（27B-W4A16 + dflash，71,680 上下文），不是这套。

⇒ KV/显存占满→抢占→变慢这条链在我们这边**没有留下任何痕迹**，而且按 §39.3 的余量它也不可能发生。
我们真踩过的「显存占满变慢」是另一种：**孤儿 EngineCore 占着整张卡**（§文档里那次「继续占着 64 GiB
显存 ⇒ 服务器变僵尸、请求全挂」），对付它的手段是 `_dev/bin/_stop_vllm.ps1` 加起服前读
`_dev/probe/_hostmem.py`。

### 39.5 两边启动参数是否一样：能声称的与不能声称的

**有据可查的一致项**（§13.8 / §17.3 的记录 + 我们采纳的 PLE 配置）：MTP×2、图模式、
`--max-num-seqs 4`，以及 PLE 那四个数（16 workers / 512 MiB 行缓存 / AIO depth 256 /
prefetch 16384）。

**我们所有 artefact 里都没有记录的**（因此**不能声称一致**）：他们的 `max_model_len`、他们怎么给
KV（bytes 还是 utilization）、他们开没开 async scheduling、他们的 compile mode 与 capture sizes。
这四项要变成事实，只有一条路：你允许我按同一条协议去量 GPU0 上那个引擎一次。

另外三个 artefact 属于**另一个项目**，不要拿来当这套的证据：`_dev/out/_ab_results/`、
`_dev/out/_ab_wsl.log`、`_dev/out/_argv_diff.txt`（它们的上下文是 71,680、drafter 是 dflash、
权重是 W4A16 27B）。

### 39.6 两处归位（我先前说过头的）

1. 我说过「`_dev/logs/` 整个没了、启动器丢了」。真相：真正的日志目录 `_dev/out/logs/` **从来没缺过**，
   `_dev/logs` 这个路径是我凭空写的（§37.1 那条教训又应验了一次）。唯一真的丢掉的是一个**未提交的
   冗余副本** `_flashnext_serve.ps1`；而 `_serve_bg.ps1` 的默认 `-ServeScript` 就是已跟踪的
   `_flashnext_struct_serve.ps1`，普查证明**全部战役跑的就是这份没丢的脚本**（`-DryRun` 现在能逐字
   复现生产日志里那条 argv）。
2. 我说过主仓那份 venv「从来没起过服」。它起过，两次（§39.2）⇒ 已更正。

顺带两个落地改动：`_flashnext_struct_serve.ps1` 的头注释归位（它同时是生产启动器），以及
`--kv-cache-memory` 写成全名 `--kv-cache-memory-bytes`（先前靠 argparse 前缀匹配侥幸成立，
一旦将来多出一个同前缀选项就会当场歧义）。

### 39.7 新坑：bash → `powershell.exe` 会吃掉 Windows 路径里的反斜杠

```
-Venv D:\code\vllm-flashtest      → 绑定成 [D:codevllm-flashtest]  ⇒ 5 个原生件全报 missing
-Venv 'D:\code\vllm-flashtest'    → 绑定成 [D:\code\vllm-flashtest] ⇒ missing count = 0
```

⇒ 从 MSYS bash 调 `powershell.exe -File`，**每个 Windows 路径都要加引号**（或
`MSYS_NO_PATHCONV=1`）。生产起服不受影响：`_run_b_arm.ps1` 里的参数是 PowerShell 数组字面量，
根本不经过 bash。这次的「humming 原生件缺失」纯是我这么调用造出来的假警报。

---

## 40. 他们的引擎终于可读：`start.sh` 链、live 指标、以及"显存占满"两边各是什么

参照物不再是一份文档转述，而是**正在跑的那台**：`~/code/qwen3.8-flash-next-cmp170hx/start.sh`
→ `bin/start.sh` → `vllm-native/bin/run_native.sh`（distro `Ubuntu-24.04`，HOME `/home/hong`）。
live 实例：pid 11753、`uptime 23:22:51`、日志 `vllm-native/logs/server.log`（唯一一份，未轮转）、
构建串 **`v0.29.1rc1.dev402+ga5a30471f.ple1`**、端口 9393（`ops/tools/forward8000.py` 转发到 :8000）。
本节目**只读**：读脚本、读日志、`GET /metrics`，没碰那个服务，也没碰 GPU0。

### 40.1 启动参数逐项对照（live 真值 vs 我们 33 次生产起服）

| 项 | 他们（live log） | 我们 | 判定 |
|---|---|---|---|
| `max_model_len` | **262,144** | **4,096** | **差 64×** |
| KV 给法 | 不 pin，`--gpu-memory-utilization 0.94` 让 vLLM profile ⇒ `GPU KV cache size 307,041 tokens / 262,144 并发 1.17x` | `--kv-cache-memory-bytes 15032385536`（14.0 GiB）⇒ `70,041 tokens / 4,096 并发 17.10x` | **机制不同**（我们跳过显存探测；那条 util 对我们是惰性参数） |
| `max_num_seqs` | 4 | 4 | 一致 |
| `max_num_batched_tokens` | 2048 | 2048 | 一致（我们仅一次实验臂给过 4096） |
| `gpu_memory_utilization` | 0.94 | 0.90 | 数值不同，对我们不生效 |
| MTP | ×2 | ×2 | 一致 |
| compile mode | `CompilationMode.NONE` | `NONE` | **一致**（原先列为"未知"，现已确定） |
| `cudagraph_mode` | `FULL_AND_PIECEWISE` | 同 | 一致 |
| `splitting_ops` / `ir_enable_torch_wrap` | `[]` / `False` | `[]` / `False` | **一致** |
| capture sizes | `[1..16,18..32,64,128,256,512,1024,2048]`，`max_cudagraph_capture_size=2048` | `[1,2,3,4,6,8,12,16,24]`，`max 24` | **不同**（他们也给 prefill 尺寸建图；纯 decode bs≤4 两边都有小图） |
| `quantization` | inc | inc | 一致 |
| PLE SSD | 16 workers / 512 MiB / depth 256 / prefetch 16384 | 同四个数 | 一致（我们就是抄来的） |
| `mamba_cache_mode` | align | align | 一致 |
| prefix caching / chunked prefill | 显式 on / on | 默认 on / on | 一致 |
| async scheduling | argv 里没有 | 仅 1 次实验臂 | 一致（生产都关） |
| `safetensors_load_strategy` | **lazy** | auto | **不同**（只影响加载，不影响稳态） |
| `reasoning_parser=qwen3` + `enable_auto_tool_choice` + `tool_call_parser=qwen3_xml` | 有 | **无** | **不同**：他们对外是带推理解析与工具调用的服务，我们的测量面没这两样 |
| `OMP_NUM_THREADS` / `MKL_NUM_THREADS` | **1** | 未设 | **不同**：他们把宿主 CPU 线程钉到 1 |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` | **1** | 未设（=关） | **我们树里有这个开关**（`vllm/compilation/breakable_cudagraph.py`、`config/vllm.py:116,777`），从没开过 |
| `VLLM_WSL2_ENABLE_PIN_MEMORY` | 1（"少了它 decode 掉 30–35%"，OPS.md） | 未设；`envs.py:323` 里有、默认 False | WSL2 专属路径，Windows 原生上大概率无意义 |
| `QSA_ALLOC_HEAL`(+384/256 MiB/1 s) | 1 | **我们代码里零命中** | **他们 fork 的补丁**，不是 env 能开关的东西 |
| 构建串 | `v0.29.1rc1.dev402+ga5a30471f.ple1` | `v0.1.dev21640+gb629bcc71` | **不同代码线** |

### 40.2 "显存占满导致变慢"：两边各自的事实

**他们**：暴露面是结构性的——KV 不 pin、吃掉 94% 显存、上下文 262,144 ⇒ `Maximum concurrency for
262,144 tokens per request: 1.17x`，即一条满长度请求几乎吃掉整个 KV 池。为此他们有三层防御：
`QSA_ALLOC_HEAL` 自愈线程（free<384 MiB 且 cached>256 MiB 时清 allocator）、`bin/drop_host_cache.sh`、
`start.sh` 默认回收 WSL 页缓存（注释写明"防整机发卡"）。

实测痕迹：`QXHEAL fired #1: free_before=0MiB cached_before=5576MiB` —— **分配器真的见底过一次**，
发生在 CUDA graph 捕获期（日志里那行就在 `Capturing CUDA graphs (PIECEWISE) 0/30` 中间）。此后 23 小时
**只此一次**；`vllm:num_preemptions_total = 0`，整份日志里 preempt/recompute 命中 **0**。
⇒ 那个"占满"确实发生过、也确实被治好了，但稳态没在抖。

**我们**：没有这个暴露面（KV 钉死；util 那条是惰性的；4096 上下文下 KV 余量 17.10x 而只允许 4 条序列
⇒ 用掉 23%）。105 份日志 preemption/recompute **零命中**。我们踩过的"显存占满变慢"是另一类：**孤儿
EngineCore 占着整张卡**（"继续占着 64 GiB 显存 ⇒ 服务器变僵尸、请求全挂"），手段是 `_stop_vllm.ps1`
加起服前读 `_hostmem.py`。另外把另一个项目的"启动彩票"教训拿来扫我们自己：12 次 eager 全是我们主动
`--enforce-eager` 的烟测臂，全部生产臂（×2 20、×1 13、×3 2、×4 2）都是 `FULL_AND_PIECEWISE`，
**没有一次静默落回 eager** ⇒ 119.2 基线未被该失败模式污染。

### 40.3 只读指标把"131.19 能不能跟 119.2 比"这件事改写了

`GET :9393/metrics`（累计 23 小时）：

| 计数器 | 值 | 推论 |
|---|---|---|
| `prompt_tokens_total` / 1,623 请求 | 186,831,626 | **平均每请求 ≈ 115,139 个 prompt token** |
| `generation_tokens_total` / 1,620 | 1,364,273 | 平均每请求 ≈ 842 个输出 token |
| `prefix_cache_hits/queries` | 178,468,800 / 187,220,967 | **命中率 95.3%** |
| `spec_decode` accepted/draft | 787,737 / 1,153,042 | 接受率 **68.3%**；draft 占全部生成 token 的 **57.7%** |
| `inter_token_latency_sum/count` | 11,367.36 s / 577,259 | 平均 **TPOT 19.69 ms**（跨全部请求，含并发与长 prefill 干扰，不是 c1 口径） |
| `iteration_tokens_sum/count` | 10,118,141 / 554,009 | 平均 18.3 token/步 —— 被 2048-token 的 chunked prefill 主导，不是 decode 口径 |
| `num_preemptions_total` | **0** | 23 小时零抢占（指标级，比 grep 更强） |

⇒ **他们的 131.19 是在"平均 11.5 万 token 的 prompt + 95.3% 前缀命中"的流量上测出来的**；我们的
119.2 是 60 条留出、短 prompt、几乎无共享前缀的固定表。**这两个数不是同一个工作量**，所以之前那句
"约 -9%"不能当结论用。

仍然可比的只有一件事：**小批解码的每步成本**（他们 17.2–17.6 ms vs 我们步长 p50 ~20 ms）。而接受率
这一维现在对上了：我们固定表接受长度 2.35 ⇒ 接受率 ≈ (2.35−1)/2 = **67.5%**，他们 **68.3%** ⇒ 几乎
同一个数 ⇒ 差距确实在步时，不在接受长度。

### 40.4 这一条我写错了，立刻归位：breakable cudagraph 我们**早就在用**

写完 40.1 的表之后去查这个开关，结果是**反的**：

- `vllm/config/vllm.py:77` 的 `DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES` 里明确列着
  `Qwen4ExpForCausalLM`、`Qwen4ExpForConditionalGeneration`、`Qwen4ExpMTP`；
- 于是 `_maybe_enable_breakable_cudagraph` 给我们**自动**置位，输出里有
  `Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1`（`vllm.py:781`，见 `_dev/out/c4096_out.txt`
  第 37 行、`_dev/out/eager_out.txt` 第 38 行、`_dev/out/gap_out.txt` 第 37 行）；
- 我们 105 份日志里 **47 份**打了 `Breakable CUDA graph enabled`，即每个走到图捕获的臂都在用；
- 反过来 `cudagraph_dispatcher.py:48` 的断言也要求它：`FULL_AND_PIECEWISE` + `mode=NONE` +
  `splitting_ops=[]` 同时成立时，**只有** breakable 开着才不会炸 ⇒ 那些臂能跑起来本身就是证据。

⇒ 他们的 `VLLM_USE_BREAKABLE_CUDAGRAPH=1` 对我们**是空操作**：`envs.py:783` 默认 `"0"`，显式设 `=1`
与自动置位走同一读取路径，不会有差别。**没有可测的东西。**

**真正新的旋钮只剩宿主 CPU 线程**：他们 `OMP_NUM_THREADS=1` 且 `MKL_NUM_THREADS=1`，我们从来没设过
（`-OmpThreads` 默认 0 ⇒ 不写 env）。这是唯一一个"他们做了、我们没做、一次重启就能对照"的差别。
他们还设了 `TOKENIZERS_PARALLELISM=false`、`HF_HUB_OFFLINE=1`、`CUDA_DEVICE_ORDER=PCI_BUS_ID`，
我们都没设；这三个不该跟线程数混在同一臂里。

不可采纳项：`QSA_ALLOC_HEAL`（我们树里零命中，是他们 `qsa.py` 的 heal 构建）、
`VLLM_WSL2_ENABLE_PIN_MEMORY`（WSL2 路径；我们 §15 已单独测过 pin 在两边都只值 ~1.17x）。

---

## 41. OMP 臂：零结果，以及第一次把步内宿主空隙拆到调用点

### 41.1 臂本身（跟 B 臂只差线程数）

`_dev/bin/_run_omp_arm.ps1` = B 臂逐字相同 + `-OmpThreads 1`（现在同时写 `MKL_NUM_THREADS`）。
起服 pid 32088 / EngineCore 5580，日志 `flashnext_20260930_220551.log`；**3 分钟就绪**（上一臂把权重
留在操作系统页缓存里，`lazy` mmap 直接命中），图捕获 5 s / 0.39 GiB。

暖机序列：104.5 → 113.8 → 116.8 → 117.5 ⇒ 第三、四块差 0.6% ≤2% 且中间无 `load_weights` ⇒ 门过，
按纪律丢掉最后一块 ⇒ 稳态估计 116.8。
计数三块 `omp-m1/m2/m3`：**119.1 / 119.9 / 119.4 ⇒ 中位 119.4**。

| 读数 | B 臂（不钉线程） | OMP 臂 |
|---|---|---|
| 计数中位 | 120.3 | **119.4** |
| forward p50 | ~14.74 | 14.78 / 14.77 |
| drafter p50 | 2.359 | 2.361–2.362 |
| step span p50 | ~17.1 | 17.138–17.154 |
| device busy | 93.3–93.5% | 93.1–93.2% |
| layer32 内 attn/mlp | 71.9 / 16.1 | 71.2 / 16.2 |
| layer1 内 ple/attn/mlp | 3.2 / 50.4 / 41.9 | 3.2 / 49.9 / 42.4 |
| logits/model | 0.0058 | 0.0057 |
| PLE-GPU | 0.0123 | 0.0121 |

⇒ **钉线程什么都没改**（吞吐 -0.7%，噪声内；步长、busy、每个相位比值差都在 1% 内）。
"宿主 CPU 线程竞争"这个解释就此出局。顺带：本臂也打了 `Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1`
⇒ §40.4 的更正在活臂上再次成立。

### 41.2 又一个"其实已经开着"：async scheduling

§1384–1391 早就写清楚了：`async_scheduling is None` 时 vLLM 自己走自动判定并打开，所以**今天所有
吞吐、submit、block 读数都在异步调度 + 批队列下取得**。§389 那次"`--async-scheduling` 量不出增益"
（102.6 vs 103.8）之所以量不出，是因为旗标本来就多余。⇒ 跟 breakable 一样，**没有可试的东西**。

### 41.3 py-spy：把步线程拆成三段（新工具 `_dev/probe/_pyspy_step_shares.py`）

采集（跟稳态负载放在同一条命令里，否则子进程会被杀掉）：
`py-spy record --idle --format raw --duration 60 --pid 5580`，产物 `_dev/out/spy_omp_arm.raw`
（47,992 样本 ≈ 8 个线程 × 6,000 ⇒ 100 Hz、60 s ✓）。

步程标记取 `execute_model` **或** `sample_tokens`（先前只认 `execute_model` 会漏掉一半以上）：

| 量 | 值 |
|---|---|
| 步线程样本 | 5,875（占全部 12.2%） |
| 每步落在步程里的宿主 CPU | **19.14 ms**（按窗口内 3,069 步换算）≈ 节律 19.55 ms |
| 其中阻塞在 `threading.wait` | **58.7%** |
| 这些等待的父帧 | **100% 是 `Future.result` (`concurrent/futures/_base.py:451`)** ⇒ **11.23 ms/步**，栈在 `ple_ssd` 里 |
| 剩下的真 CPU 活 | **19.14 − 11.23 = 7.91 ms/步** |

⇒ §14.3 那个形状**今天仍在**（当年 9.8–10.4 ms/步等 prefetch，现在 11.23 ms/步），只是它被 GPU 工作盖住了。

按子系统切这 7.91 ms（样本可同时命中多个子系统，故不相加）：

| 子系统 | ms/步（宿主 CPU） |
|---|---|
| attention 后端 | **2.78** |
| ├ 其中 gdn / linear attn | 1.74 |
| triton 派发 | 1.25 |
| qsa | 0.84 |
| torch op 派发 | 0.70 |
| h2d / uva 拷贝 | 0.47 |
| moe / humming | 0.30 |

拆到调用点（`--subsystem "attention backends"`）：

```
build_attn_metadata (vllm/v1/worker/gpu/attn_utils.py:495)   1.51 ms/步   后端 CPU 的 54.3%
run (triton/runtime/jit.py:744)                               0.40
build (gdn_attn.py:222)                                       0.27
build (qsa_cache.py:659)                                      0.19
```

### 41.4 定位变了，而 §28 的判决仍然成立

- **交叉验证站得住**：Triton 派发 1.25 ms/步 vs §17.4 用 py-spy 样本换算出的 1.14 ms/步——不同仪器、
  同一量级。
- GPU 每步闲 **~2.4 ms**：节律 19.55 − 设备跨度 17.14；busy 93.2%。
- 那 11.23 ms 的 Future 等待**被 GPU 工作盖住**，所以 §28.4 的判决不变（行成本与空隙负相关
  r = −0.676；GPU 侧 PLE 只有 0.012 ms）⇒ **别再去买"把行做快"**。
- 真正能卡住 GPU 的候选，现在有了名字：`build_attn_metadata` 1.51 + Triton 派发 1.25 ≈ **2.76 ms**，
  恰好覆盖那 2.4 ms 的空窗。

### 41.5 资助算术（按今天这条节律重算）

接受长度 2.35、节律 19.55 ms ⇒ 120 tok/s 量级：省 1 ms/步 ⇒ 2.35/0.01855 = 126.7 ⇒ **+5.9%**；
省 1.5 ms ⇒ **+8.3%**；把 2.4 ms 全收 ⇒ **~+14%**，而这正好等于跟他们 17.4 ms 步长的差。

**下一步（尚未动手）**：让 `build_attn_metadata` 不必每步全量重建（按 batch descriptor 缓存/复用），
env-gate + `.orig` 回滚 + B 当尺子。动手前先按 AGENTS.md §1 跑重复劳动检查。

**为什么这条路值得走**：它归因用的是 py-spy（本机可用），不需要核名也不需要 SM 计数器 ⇒
**绕开了 §36 的 CMP 硬件封锁**，是当前这台机器上唯一还能往前推的方向。

**必须记下的置信度**：以上全部来自**一次** 60 s 采集、**一个**臂。要当结论用，至少得复采一次；
而且 py-spy 给的是宿主 CPU 时间，不是墙钟，二者只在"该 CPU 排在关键路径上"时才等价。

### 41.6 复采：三次、两种配置，结构成立，绝对值带 ±15%

第二次起服（`_run_b_arm.ps1`，**不钉线程**的生产配置，pid 34108 / EngineCore 7460，日志
`flashnext_20260930_224144.log`，~4 分钟就绪，KV 打印 `70,041 tokens / 17.10x` 与历次逐字一致）里
取两次 60 s 采集，同一条命令内用两遍固定表负载压住（`--passes 2` ≈ 370 s）。
新工具 `_dev/probe/_step_gap_table.py` 把服务自己的步窗行转成"节律 − 设备跨度 = 空窗"表。

| | 首采（OMP 臂） | 复采 A | 复采 B |
|---|---|---|---|
| 步线程样本占全部 | 12.2% | 12.2% | 12.2% |
| `Future.result` 等待 | **11.23 ms/步** | **11.20** | **11.50** |
| 注意力后端占步样本 | 14.5% | 14.8% | 16.1% |
| `build_attn_metadata` 占后端 | **54.3%** | **49.5%** | **51.4%** |
| `build_attn_metadata` ms/步 | 1.51 | 1.43 | 1.71 |
| `qsa_cache.py:659` ms/步 | 0.19 | 0.19 | 0.20 |
| 每步"在步里"的 CPU | 19.14 | 19.41 | 20.66 |
| 真 CPU = 差值 | 7.91 | 8.19 | 9.14 |

**方法论：那条"每步 CPU"有一部分是在量节律，不是在量工作。** 步线程几乎整段都在步程里
（占比三次都是 12.2% ⇒ 每线程 6,000 样本里 ~750 个属于步线程），所以它的 CPU 必然≈节律；节律从
19.5 漂到 20.9，"每步 CPU"就跟着漂。**能信的是比值**：等待占比、后端占比、`build_attn_metadata`
占后端的比例，三次都在 1 个百分点以内。

| 窗口 | 平均节律 | 设备跨度 p50 | **空窗** | busy |
|---|---|---|---|---|
| 首采 | 19.47 ms | 17.156 | **2.35 ms/步** | 92.0% |
| 复采 A | 20.23 ms | 17.498 | **2.09 ms/步** | 92.5% |
| 复采 B | 20.85 ms | 17.290 | **2.53 ms/步** | 90.5% |

⇒ **设备跨度稳（17.16–17.50），漂的全在宿主侧**（节律 19.5–20.9）。空窗 **2.1–2.5 ms/步**，跨采集跨配置都稳。

**一条硬约束**：注意力后端的 CPU（2.78 / 2.88 / 3.32 ms/步）**比空窗大 0.6–0.8 ms** ⇒ 它不可能整体
在关键路径上，**上限就是那个空窗**：全部删掉也只值 2.1–2.5 ms/步（≈ +12% 到 +14%）。而
`build_attn_metadata` 单项 1.4–1.7 ms **稳稳装在空窗里面** ⇒ 把它大部分消掉，可信收益 **1.0–1.5 ms/步
≈ +5% 到 +8%**，过 5% 线。

**尚未确立的一环**：我们只证明了它的量级装得下空窗，**没有证明它就发生在空窗里**（采样给的是
CPU 时间，不是时间轴位置）。要补这一环，最便宜的办法是直接测它的墙钟：在部署副本里给
`build_attn_metadata` 前后加 `perf_counter_ns`，env-gate，每 N 步打一行 p50，一次重启读
`_step_gap_table.py` 的空窗跟它对表。**这一步做完才谈得上动手改缓存。**

### 41.7 本轮我犯的两个错（记下来免得重犯）

1. **复采 B 落在了空转引擎上**：第一遍负载 ~185 s，在 22:49:17 结束，而采集 B 到 22:49:41 才开始 ⇒
   那 47,992 样本全是空转等待，整份作废。教训：**采集窗口必须先跟负载时长对表**，
   要两次采集就用 `--passes 2` 这种明显盖住两段的负载。
2. `_pyspy_step_shares.py` 里 `duration = (end - start) / 1000` —— 时钟本来就是整秒，我多除了一次
   1000 ⇒ 整张表被放大 1000 倍（"3 steps over 0 s"、"19616 ms/step"）。这种数量级错误本该在输出
   "每步 19616 ms" 的那一刻就被拦住：**任何 ms/步 的读数如果超过整步长度，先怀疑自己的算术。**

## 42 宿主那 2.9 ms 找到了调用点，但三条"跟我们不一样"的假设全部死亡

给 `build_attn_metadata` 装墙钟（`VLLM_ATTN_BUILD_TIMING=1`，默认关；源码提交
`e925444baf`，另两次拆分提交未单列）之后，在同一 60 秒窗口里同时读引擎的步窗与计时行。
新工具 `_dev/probe/_attn_build_report.py` 把计时行按 batch 大小、builder 类、调用点三层
拆开，并自己从步窗算出"每步几次调用"，最后跟同窗口的空窗对表。

### 42.1 结论：贵的那次是模型侧，发生在入队之前（三次复现）

| 调用点 | 调用数 | ms/步 | p50 ms/调用 | 身份 |
|---|---|---|---|---|
| **`mamba_hybrid.py:313`** | ~2,900 | **2.84** | **2.92** | 模型的 model-state（混合 mamba 路径），在 forward 入队**之前** |
| `speculator.py:349` | ~2,900 | 0.40 | 0.42 | drafter，在入队之后 |

| batch | 调用数 | p50 ms/调用 | ms/步 |
|---|---|---|---|
| tokens=1 | ~2,900 | 0.39 | 0.39 |
| tokens=3 | ~2,880 | 2.82 | 2.73 |
| tokens≈75–122（prefill 块） | 各 1 次 | 2.6–5.0 | ~0 |

三个窗口（跨两次重启）：**构建 3.159 / 3.206 / 3.131 ms/步**；空窗 **2.39 / 3.06 / 2.97 ms/步**；
GDN 份额 55.9 / 56.5%。⇒ **模型侧那次是串行宿主时间**，drafter 那次只值 0.4 ms。

### 42.2 构成（按每步毫秒）

| builder | 调用数 | ms/步 | 份额 | 单次 |
|---|---|---|---|---|
| `GDNAttentionMetadataBuilder` | 8,700 | **1.77** | 56% | 0.61 ms |
| `QSAMetadataBuilder` | 14,500 | **0.84** | 26% | 0.17 ms |
| `PleShortConvAttentionMetadataBuilder` | 2,900 | **0.56** | 17.5% | 0.57 ms |

墙钟 2.9 ms ≫ py-spy 的 CPU ~1.4 ms ⇒ **约 1.2 ms 是这个线程"没在跑"**（等 GIL / 等设备）。
嫌疑点是 spec-decode 分支里三处 **"用 CPU 布尔掩码索引 GPU 张量"**（`gdn_attn.py:306/327/330/365`）：
`GPU[cpu_bool_mask]` 走 `masked_select`，而它必须知道输出多大 ⇒ **同步设备**。我们稳态解码
（全序列都在 spec）每步正好撞上其中两处。

### 42.3 天花板与资金算术（修正后的锚）

空窗 **2.2–3.1 ms/步** 是"消掉宿主工作最多能买到的东西"：全消 ≈ **+11% 到 +15%**。
只把 GDN+QSA 各砍一半 ≈ 省 1.3 ms/步 ≈ **+7%** ⇒ 过 5% 线。

**锚的更正（我自己的错）**：§41 末尾写"把 2.4 ms 全收 ≈ +14%，正好等于跟他们 17.4 ms 步长的差"——
那个 17.4 ms 是**我们自己合成基准里的 submit p50**（§17 第 733 行），而 §17 第 745 行早就写明
"**每步成本我们跟他们齐平**：我们 17.4 ms，他们 17.2–17.6 ms ⇒ 引擎侧没有差距"。
⇒ **正确表述：不存在需要追赶的步长差；能收的是我们自己引擎内部的宿主空窗。** 消掉它是反超，不是追赶。

### 42.4 三条"他们不一样"的假设，全部被否证

| 假设 | 检查 | 结果 |
|---|---|---|
| 他们有跨 KV 组元数据复用（#58762） | `a5a30471f` 作者日期 2026-09-19，早于 #58762 合并 09-28；他们的 runner `grep -c update_block_table` = **0** | **否**：两边都没有 |
| 他们跑 MRV1（旧 runner 有复用） | 他们日志只出现 `[model_runner.py:…]`，无 `gpu_model_runner.py` | **否**：也是 MRV2 |
| 他们的 mamba block table 算法不同 | 他们 `utils.py:1156-1171` 与我们 eager 路径**逐字相同**，且他们 `envs.py` **没有** `VLLM_MAMBA_ALIGN_FUSED` | **否**：同一套 5-op 代码 |

⇒ 三处低效**都是共享的**，所以都不解释引擎间差异；它们是我们**可以消掉以反超**的东西。
上游 #58851（draft、未合并）做的正是"跨组共享"，而我们的配置每组只有 ~1.5 个 GDN 组 ⇒
共享大约只能省 GDN 的 1/3 ≈ 0.57 ms/步 ≈ +2.8% ⇒ **不够过线**，所以我们要走另一条路
（这也回答了 §1 的"是否重复劳动"：不重复，因为路子不同且上游尚未合并）。

### 42.5 下一步顺序（先便宜后贵）

1. **免费的一条**：`-FusedAlign`（纯环境变量，把 5 个小 GPU 操作换成 1 次 Triton kernel）。
   **融合分支只有我们的树里有，他们没有** ⇒ 若有效即是差异化。~20 分钟，零代码。
2. **改代码的一条**：`gdn_attn.py` 里加"全序列都是 spec"的快路——把两处
   `block_table_tensor[cpu_mask, :k]` / `num_accepted_tokens[cpu_mask]` 换成等价的切片
   （掩码全真时 `masked_select` 与切片取值逐元素相同 ⇒ 语义不变），消掉两次设备同步。
   用 B 臂当尺子量。
3. 两条都有效则合计 ~+7%。**先测第 1 条，再动第 2 条。**

## 43 与 GPT-6.1 Sol 的第 8 轮会诊：先探针，后改码

简报 `consultations/sol_brief1.md`（11,679 字符，197 行），回复
`consultations/sol_reply1.md`（19,646 字节，约 14 分钟返回）。命令按 AGENTS.md 体例：
`nohup timeout 2400 pi --print --provider lmstudio --model gpt-6.1-sol --thinking high
--approve --append-system-prompt "read-only reviewer..." "$(cat consultations/sol_brief1.md)"`。
会诊期间两块卡的显存读数全程不变（GPU0 62,701 MiB / GPU1 0 MiB）⇒ 会诊不占 GPU。

### 43.1 它的排序（原样记录）

| 序 | 动作 | 成本 | 预期暴露收益 | 判决 |
|---|---|---|---|---|
| 0 | 先验部署件 + 前缀掩码等价性检查 | <5 min | 无 | **A、B 之前必做** |
| 1 | 事件边界日志 + 临时"宿主松弛探针" | 10–20 min + 一次重启 | 诊断 | **最便宜地判断 B 打的是不是节拍器** |
| 2 | A（融合对齐，若部署件真有） | ~15 min | 0.2–0.4 ms/步 ≈ +1~2% | 顺手做，单独不过线 |
| 3 | B（消掉两次 masked-select 同步） | ~35 min | 若同步暴露 0.8–1.5 ms/步 ≈ +4~8% | 主要候选 |

它算的 A 上限：`20/(20−0.42)−1 ≈ 2.1%` ⇒ **A 单独到不了 5% 线**，只能当 B+A 的零件。
跨组共享它按 0.57 ms ≈ 2.8% 判为不够线，只在更大改动里才可重提。

### 43.2 它的第 0 步抓到一个真错，但方向反了

它 `rg` 的是 `D:/code/vllm-windows/vllm/...`（上游克隆），说那里没有 `VLLM_MAMBA_ALIGN_FUSED`
⇒ 那是**另一棵树**。验我们自己运行的部署件：
`site-packages/.../v1/attention/backends/utils.py:1157 _mamba_align_gather_triton`、
`:1200 if HAS_TRITON and envs.VLLM_MAMBA_ALIGN_FUSED:`、`envs.py` 里 3 处命中，
且与 worktree 源码 md5 相同（`e230b30ff968`）⇒ **A 确实是纯环境变量实验**，第 0 步这半边过了。
**教训：这台机器上"树混淆"是真实风险**——同一个文件在三棵树里内容不同（上游克隆 / worktree 源码 /
site-packages 部署件），任何"代码不存在"的断言都必须先说清是哪棵树。

### 43.3 前缀掩码等价性测试（新工具 `_gdn_mask_equivalence.py`，GPU1 空闲、小张量）

| 用例 | 取值 | 观察 |
|---|---|---|
| 全真、无 padding | OK | 切片是 **view 且 `is_contiguous()=False`**（stride 6,1）；掩码选择返回连续 copy |
| 真前缀 + 尾部 padding | OK | 同上 |
| **中间一个假行** | **DIFF** | ⇒ 必须走慢路（正是它预言的拒绝场景） |
| 短前缀 + 2 行 padding | OK | 前缀条件成立 |
| 宽行、k<宽度 | OK | 切片 stride (10,1)，非连续 |

⇒ 安全条件不是 `mask.all()` 也不是 `num_spec_decodes == num_reqs`，而是**前缀掩码**：
`mask[:n] 全真 且 mask[n:] 全假`，且断言要对 `mask.numel()` 而不是 `num_reqs` 做。
⇒ **坑是实的**：切片给非连续 view，掩码选择给连续 copy ⇒ 消费方是否容忍步长决定要不要
`.contiguous()`；后者是一次 kernel 拷贝（~20–60 µs）**不再同步设备**，仍比同步便宜。

### 43.4 读 staging 块之后，成本模型改了

`gdn_attn.py:429-480`：FULL-graph 且 `num_prefills==0 and num_decodes==0` 时，每步把元数据
`copy_` 进持久缓冲再 `fill_` 尾部，共 **~7 次 copy_ + ~5 次 fill_ ≈ 12 次小张量 kernel 启动**，
外加 `async_tensor_h2d` 一次。⇒ 所以 GDN 每次调用 0.61 ms 里，**同步**（2 处 masked-select）与
**派发**（12 次启动）各占一块。B 只消同步，消不掉派发；派发那块与 §17 的 Triton/torch 派发
对得上。⇒ 也解释了为什么必须先做探针：**若同步其实是重叠的，B 的收益接近零**。

### 43.5 定下的顺序（按它的建议，加上我们自己的证据）

1. **D（把 accept length 从 2.35 提到 2.75，round 7 估 +17%）要先有一个实现成本估算**：
   它的盈亏线是"D 若 <75–120 分钟则优于 B"。我们已知**廉价形态的 D 已被否证**（MTP×3/×4
   在生产表上 per-class 不过线，×4 中位 +0.3%），所以 D 只剩引擎级改动 ⇒ 先花 20 分钟
   把"引擎级改 draft 到底要动什么"写清楚，再决定 D 还是 B。
2. **宿主松弛探针**：在模型侧调用点注入 0/0.5/1/2 ms 的 `time.sleep`（**释放 GIL**，避免抢
   别的线程的 CPU），拟合 `s = Δ节律 / Δ注入延迟`：`s≈0` ⇒ 这段被设备盖住；`s≈1` ⇒ 宿主就是
   节拍器，省多少就快多少；`0<s<1` ⇒ 只有部分暴露，转折点给出可用的重叠余量。
   配一路 CUDA event + `event.query()`（不同步）在 build 入口/返回处分类重叠。
   ~10–20 min + 一次重启。
3. **B**：前缀掩码快路（断言按 43.3），消费方若不容忍非连续则 `.contiguous()`（仍是拷贝不是同步）。
   按 43.6 的判据验收，用 B 臂当尺子。
4. **A**：在同一批重启里顺手测；单独不过线就不给独立资金。

### 43.6 验收判据（B 的通过条件不是"计时器变短"）

```
模型侧构建墙钟 下降
+ 设备跨度 基本不变
+ 空窗 按暴露量下降
+ 节律 同幅下降
+ 固定表中位吞吐 上升（accept length、内存、请求构成不变）
```

它的完整判定矩阵（13 行）已存 `sol_reply1.md`；要点：`M↓ 而 G、C、T 都不动` ⇒ 重叠或被吸收，
不许单独晋升；`M↓ 且 D↓ 同幅、G 不变` ⇒ 收益来自设备侧，节拍器假设未被证明；
`M 不变而 C↓、G↓、T↑` ⇒ 计时器范围或调用点覆盖搞错了，别记账到这个候选上。

## 44 第二处锚错误（由用户的追问逼出来）：§17.3 那句"齐平"是苹果比橘子

用户问："源码逐字一样，那差异从哪来？"——追问逼我去查 §17 那两个数的**定义**，结果：

- 我们的 **17.4 ms** 那一列是 **submit p50**，定义见第 622 行：`schedule + 提交 execute/sample`，
  即**宿主自己干活的那段时间窗**；
- 他们的 **17.2–17.6 ms** 是**整步周期 p50**（用他们自己的 tok/s ÷ tok/step 反推落在
  118.4/2.20=53.8 步/s ⇒ 18.6 ms，130.9/2.20=59.5 ⇒ 16.8 ms 之间，自洽）。

⇒ **第 745 行"每步成本我们跟他们齐平 ⇒ 引擎侧没有差距"不成立**：那是拿我们的**宿主干活窗**
去比他们的**整步周期**。正确的同量纲比较是**周期对周期**：

| | 周期（步长的定义：一步到下一步的间隔） | 设备干活 | 宿主额外 |
|---|---|---|---|
| 我们（合成 MTP×2 c1） | **19.0 ms**（52.7 步/s） | — | — |
| 我们（生产固定表，三次窗口） | **19.5–20.9 ms** | **17.2–17.5 ms** | **2.2–3.1 ms** |
| 他们（同机 A/B，MTP×2） | **17.2–17.6 ms** | 未知 | ≈ 0（周期≈设备） |

⇒ **我们每步周期比他们长 ~1.4–1.8 ms**，而我们**设备干活的时间只有 ~17.3 ms**，
比他们的周期只多 ~0.1–0.5 ms ⇒ **差的那 1.5–2 ms 几乎全在宿主侧**，正好等于我们自己的空窗。
⇒ 所以 §42.3 的正确表述应是：**存在一个约 1.5–2 ms/步的真实周期差；它不在我们比对过的那三个文件里；
它的大小与我们量到的宿主空窗同量级** ⇒ 消掉空窗大约就等于抹掉这个差，而不是"只追平"。

**教训（比上一条更一般）**：引用一个数之前先查它的**定义与量纲**。同一个词"每步成本"在这份文档里
同时指"宿主提交窗"和"整步周期"两种东西，而 §17.3 的结论跨了这两个量纲。

## 45 第三处更正：我们从没做过"同条件、同方法"的两引擎对照

用户第二次追问（"你是说你根本没在完全相同条件下用相同方法比对过两边？"）逼出了出处核查。
直接去参照引擎的项目里读原文，而不是引用我们文档里的转述：

- `~/code/qwen3.8-flash-next-cmp170hx/ops/OPS.md:1501` §9.25 标题是
  **「MTP 投机解码深度 1 vs 2：A/B 实测」**⇒ 里面的"两边都充分预热 + 宿主低压 + 同一批 prompt"
  指的是**他们引擎内部的 MTP=1 与 MTP=2 两种深度**，**不是 WSL 对 Windows 的跨引擎对照**。
- 因此我们 §17.3 写的"跟他们同机公平 A/B 对齐（OPS §9.25）"是**误标**：我们只是把我们的合成
  c1 数字**贴到**他们内部 A/B 的 MTP=2 那一列上。三次量纲/输入都不同：
  他们的步间延迟（客户端到达时间戳）vs 我们的 submit 窗（宿主插桩）；他们的 prompt 集 vs 我们的
  合成重复句式；他们的引擎配置（262144 上下文、KV 自适应、捕获到 2048）vs 我们的（4096、钉 14 GiB、
  捕获到 24）。

⇒ **诚实结论：跨引擎方向上我们既没有证明"我们更慢"，也没有证明"我们齐平"。**
唯一严格的东西是他们内部的 A/B，和我们内部的 A/B，各自独立。

### 45.1 他们那条测量协议教训，跟我们的发现独立重合

OPS §9.25.1：第一轮 MTP=2 测出 94.9/96.2/105.9 tok/s（步时 21.8–22.6 ms）"看起来更深反而更慢"，
**是假象**——数据取自①引擎刚启动（只跑过几个 prefill，无解码预热）②宿主内存高压
（Windows 可用 6.9 GB、`vmmemWSL` 57.8 GB、页面文件狂换）。balloon 缩回（可用 51.4 GB）后
同配置复测：**步时 22.2 → 17.87 ms，120.1 tok/s**。⇒ 他们定的硬规矩是"预热 ≥2500 解码 token +
`vmmemWSL < 32 GB`，否则同一配置可测出 ±25% 假差异"。
⇒ 这与我们独立发现的"宿主压力让率动 ~25%"以及我们的预热闸门**是同一件事的两条独立证据**。

### 45.2 找到了一件能做真对照的工具（只读取得，不碰他们的引擎）

`ops/bench/dec_bench.py`：纯标准库，流式 `/v1/completions`，`ignore_eos`，temperature 0.6，
先 `run(64,"warm1")` 再 4 次 `run(256,...)`，输出：**步间 p50/p95/max、纯解码 tok/s、
接受率（`spec_decode_num_accepted/drafted` 增量）、TTFT、引擎 TPOT（metrics 增量）、解码期磁盘读**。
默认硬编码 `M=http://127.0.0.1:9393`、`model=Qwen3.8-Flash-Next`、
prompt `请详细说明 vLLM 中 PagedAttention 的工作原理。`。

⇒ 一致性检验：他们 MTP=2 报 126 tok/s 且 tok/step 2.20 ⇒ 步间隔 = 2.2/126 = **17.5 ms**
与他们 p50 17.2–17.6 ms 吻合 ⇒ 说明它给的"步间 p50"在 spec decode 批量下发下确实≈步长。

**两条路（代价与触碰程度完全不同）：**

| 方案 | 做法 | 代价 | 触碰 |
|---|---|---|---|
| **B（零触碰）** | 取他们的脚本+同一 prompt，**只打我们自己的引擎**，跟他们已留档的 MTP=2 同方法数字对表 | ~25 min | 完全不碰他们引擎/GPU0 |
| **A（真同条件）** | 同一脚本同时打两边、同一批 prompt、都预热、宿主低压 | ~25 min + 需要用户点头 | **会给他们引擎加负载**、扰动其 23 小时稳态与 `/metrics` |

⇒ B 能答"同样的输入同样的方法，两台配置差多少"；A 才能答"同条件下两个引擎差多少"。

## 44 快路做了，机制有效，生产尺子不动——而且它是被同条件对照臂救回来的

### 44.1 头对头（同一夜、同一宿主状态、同一 9 遍协议）

| | 对照臂（开关**关**，pid 7768，日志 082524） | 快路臂（开关**开**，pid 16408，日志 075033） |
|---|---|---|
| 热身 ×4 | 107.5 / 116.2 / 117.3 / 118.5 | 104.5 / 112.3 / 116.2 / 114.3 |
| 计数 m1–m5 | **119.5 / 120.2 / 119.4 / 119.9 / 121.5** | 116.4 / 118.0 / 120.3 / 119.6 / 119.2 |
| 中位 / 均值 | **119.9 / 120.1** | 119.2 / 118.7 |
| 接受率 / 接受长度（整机 `/metrics`） | **66.7% / 2.33** | 66.5% / 2.33 |
| 宿主 | load 38→39%，avail 54.0→52.8 GiB，page-file avail 68.7→67.4 GiB | load 38%，avail 53.9，page-file 68.7 |

⇒ 接受率一模一样 ⇒ **改动的语义没坏**（这是 `/metrics` 给的硬证据，不是日志里的猜测）；
⇒ 吞吐中位 **−0.6%**、均值 **−1.2%** ⇒ **不过线，且不涨。**

### 44.2 机制核对：同规则取"日志最后 60 秒"，对照臂反而更快

| 对齐窗口 | 对照（关） | 快路（开） |
|---|---|---|
| 步频 | **51.0 步/s** | 50.0 步/s |
| 节律 | **19.61 ms** | 20.02 ms |
| 空窗 | **2.168 ms** | 2.221 ms |
| 设备跨度 | ~17.44 | ~17.80 |
| GDN ms/步 | 1.661 | **1.467** |
| tokens=3 p50/调用 | 2.734 | **2.425** |
| build/步合计 | 3.037 | **2.902** |

⇒ 快路确实把构建时间削掉 **0.135 ms/步**（GDN −0.20、tokens=3 p50 −0.31），
但节律反而**慢 0.41 ms**、空窗反而**大 0.05 ms**。
⇒ 所以 §43 之后那句"空窗降 0.75 ms、节律降 0.83 ms"**是位置错配 + 过夜宿主差造成的假象**。

### 44.3 结论：那两次同步藏在设备后面，消掉它换来的是额外派发

解释自洽：`GPU[cpu_mask]` 的同步发生在"设备本来就忙"的时候 ⇒ 它不占关键路径；
换成"切片 + `.contiguous()`"等于**多了 2 次小拷贝 kernel 的派发**（各 ~20–60 µs）
⇒ 隐藏的等待消失、可见的派发增加 ⇒ 净 −1%。
即评审矩阵第二行：**M↓ 而 G、C、T 不动 ⇒ 重叠或被吸收 ⇒ 不许单独晋升。**

**这条否证掉的东西很重要**：`build_attn_metadata`（模型侧 2.9 ms 墙钟）**不是节拍器**。
它那 2.9 ms 装在设备跨度 17.4 ms 里面，不是装在 2.2 ms 空窗里面。于是"消掉元数据构建
≈ 消掉空窗"这条资金论证作废。空窗 2.17 ms 里装的必定是**另一段不与设备重叠的宿主工作**
（候选：Triton/torch 派发、staging 的 ~12 次小张量启动、调度器、采样与输出回传）。

### 44.4 方法论教训（这条比结论本身值钱）

跨夜比较不可信：page-file 可用空间从昨晚 133.6 GiB 掉到今晨 68.7 GiB（你的机器过夜期间
吃了交换空间），而率随宿主压力本来就能动 ~25%。若只比"快路臂 vs 昨晚的臂"，我会得出
"空窗降、节律降、机制成立"的结论，而吞吐却说不通——两边都能自圆其说，其实都是错的。
**是"在同一夜再跑一个同条件对照臂"这一手把结论扳回来的。** 往后的规矩：
**跨日或宿主状态变了，就不许拿旧臂当尺子，必须在同一条件下重跑对照。**

### 44.5 现场与代码状态

本次判定为不晋升，实验开关、专用启动脚本和实现已删除；默认路径回到原代码。
源码提交 WT `fc66301a87`、脚本 WIN `17ab85336d`。
部署件 `.orig` 备份齐（`gdn_attn.py.orig` 新建、`envs.py.orig` 原有）。
服务已停，GPU1 `0 MiB`、无 compute app；GPU0 全程 62,701 MiB 未动。

## 46 第一次"同输入、同方法"的跨引擎对照（零触碰他们引擎），以及它自己的坑

工具 `_dev/probe/_dec_bench_windows.py`：只读取自他们 `ops/bench/dec_bench.py`，
prompt 与短句 prompt **逐字比对一致**，`temperature 0.6` + `ignore_eos` + 流式 + `/metrics`
四个计数器增量全部照抄；只改端点与模型名，并把预热加长（暖 3×1024 + 64 = **3072 个解码 token**，
因为两边都认定"冷引擎 + 宿主高压会测出 ±25% 假差异"）。
跑在**无任何观测器**的干净生产臂（新脚本 `_run_clean_prod_arm.ps1`：跟晋升基线同一组开关，
去掉 step timing / phase events / attn timing）。

### 46.1 对表他们留档的 MTP=2（同一 prompt、同一方法）

| 指标 | 他们（OPS §9.25.2） | 我们（本次） | 判读 |
|---|---|---|---|
| 步间 p50 | **17.2–17.6 ms** | **19.0 / 19.1 / 21.1 / 22.9**（暖身段 20.4–22.8） | 我们慢 ~1.4–5 ms/步 |
| 稳态纯解码 tok/s | **118.4–130.9**（中位 ≈126） | **101.8–115.9** | 我们慢 ~10–20% |
| tok/step | **2.20** | **2.08–2.25** | **齐平** |
| accepted/drafted | **51.6–64.9%** | **54.5–62.7%** | **落在他们带内** |
| 步间 p95 | 18.4–21.1 | 22.4–24.8 | 我们也差 2–4 ms |
| 单步 max | 他们 4 次里 1 次 **~1331 ms** | 我们最大 **23.5–32 ms** | 我们的尾巴更干净 |

**最要紧的一条对齐**：他们 p50 17.2–17.6 ≈ **我们三次窗口量到的设备跨度 17.2–17.5** ⇒
**设备工作齐平**；而我们的 p50 19–23 = 设备跨度 + 空窗 ⇒ **差距在宿主空窗，不在接受率**
（接受率与 tok/step 都在他们带内）。这与 §42–§45 的宿主侧调查方向一致。

### 46.2 但今天的幅度不能当真——踩到了两边都写过的同一个坑

起服时宿主 `avail 56.4 GiB / load 35%`；测量结束时 **`avail 33.6 GiB / load 61%`**，
可用 pagefile 从 135.1 GiB 掉到 48.7 GiB ⇒ **测量中途宿主进入内存高压**。
OPS §9.25.1 的原始案例就是这件事：同一配置步时 **22.2 ms → 17.87 ms**、95–106 tok/s → 120.1 tok/s，
只因为宿主内存恢复了。我们自己的 §12 也量到"率随宿主压力动 ~25%"。
⇒ 本次 p50 序列 22.8 → 20.4 → 18.0 → 21.1 → 19.0 → 19.1 → 22.9 **没有单调趋势、噪声很大**，
无法把"多少是引擎、多少是宿主压力"分开。
⇒ **规矩：跨引擎对照必须在两边都验证过"宿主低压"的窗口里做**（他们 `bin/drop_host_cache.sh` +
等 20–60 s；我们这边对应 `_hostmem.py` 读数必须高可用低负载，并在每块前后各读一次）。

### 46.3 我们自己度量的会话间离散（影响 5% 晋升线的侦测力）

同一组开关、无观测器：**今天固定表中位 ≈115.3**（pass1 114.7 / pass2 115.9，差 1.0% 过闸门；
pass0 91.5 是冷残留，`prose` 类 53.6 → 98.7/100.4 几乎翻倍 ⇒ 第一遍有逐类 JIT/缓存效应），
而历史基线是 **119.2** ⇒ **同配置换会话差 ~3.3%**。
⇒ 含义：单跑一次去判 5% 的效果，噪声底就有 3% 上下。**要么同会话配对复测，要么把同一改动
跨会话复现两次**，否则 5% 的线很容易被供应商噪声伪造或掩埋。

### 46.4 下一步

1. **在验证过的宿主低压窗口里重跑这份对照**（两边读数都要记），才把"慢 1.4–5 ms/步"当事实用。
2. 然后照 §43.5 的顺序做**宿主松弛探针**（它判的是"我们自己的空窗是不是节拍器"，与跨引擎对照无关，
   所以不受这条噪声影响——探针看的是 Δ节律/Δ注入延迟的**斜率**，同一会话内自比）。

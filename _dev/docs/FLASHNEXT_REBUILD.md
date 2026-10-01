# Windows Flash-Next：删除 venv 后如何重建

## 保存边界

**修复源码在 Git 中，venv 不是源码的唯一副本。**

| 内容 | 可重建的来源 |
| --- | --- |
| vLLM Python / CUDA / C++ 修复 | 仓库的 `vllm/`、`csrc/`、CMake 配置；从当前 checkout 构建 |
| Windows Python 运行时部署 | `_dev/bin/_patch_vllm_windows_runtime.py` 的 13 文件清单 |
| Humming Windows Python / C++ 移植 | `_dev/bin/_patch_humming_windows.py`，覆盖 8 个上游文件 |
| Humming 四个原生组件及 manifest | `_dev/bin/_humming_{device_info,nvrtc,cubinpatch,launcher}.ps1` |
| PLE buffered IOCP DLL | `vllm/models/qwen4_exp/nvidia/ple_ssd_io_win.c` + `_dev/bin/_ple_ssd_io_win_build.ps1` |
| 已验证依赖版本 | `_dev/requirements/flashnext-windows-constraints.txt` |
| 启服参数 | `start_qwen38_flash_next.ps1` + `_dev/bin/_flashnext_struct_serve.ps1` |
| 性能结论 / 测量代码 | `BENCH_FLASHNEXT_GPU1_VS_WSL2_GPU0.md`、`_dev/bench/`、`_dev/probe/` |

不提交 `.venv`、`.venv-flashnext`、`.flashnext-root`、`.venv-maintenance`、
DLL/PYD、JIT 缓存、`.orig` 和 `_dev/out`。`.orig` 只是本机回滚副本，
重建不需要从旧 venv 复制它。审计时才用现存 `.orig` 验证补丁能否还原当前安装。

模型权重在 `D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`，不是 Git 资产。
原始性能日志仍在被忽略的 `_dev/out`；Git 中保存结论和测量脚本，不保存全部原始日志。
如需保留严格的原始测量证据，应另外归档 `_dev/out`。

## 一键重建（不自动启停服务）

当前脚本以本机布局 `D:\code\vllm-windows` 为基准，需要 Python 3.12、uv、Git、
MSVC 14.44 / Windows SDK 10.0.26100、CUDA 13.3，以及下载依赖和构建依赖的网络访问。
约束文件固定已验证的关键版本，但不是带哈希的完整离线依赖锁。

```powershell
$repo = 'D:\code\vllm-windows'

# 默认只查看步骤，零构建、零服务操作；-DryRun 也支持。
& "$repo\_dev\bin\_setup_flashnext.ps1"

# 仅在明确停止 Flash-Next 之后执行；端口 9393 在监听时会拒绝重建。
& "$repo\_dev\bin\_setup_flashnext.ps1" -Execute

# 重建完成后另行启动，默认 GPU1 / :9393；不碰 GPU0 / WSL。
& "$repo\start_qwen38_flash_next.ps1"
```

重建顺序：

1. 用 `uv` provision `.venv-flashnext`；拒绝修改主 `.venv`，不自动删除旧环境。
2. 创建 `.flashnext-root/.venv` → `.venv-flashnext` junction，避免 checkout 的
   `vllm/` 遮蔽安装包；拒绝覆盖不符合预期的目录或 junction。
3. 从当前 checkout 构建并安装 vLLM，不从旧 site-packages 复制二进制。
4. 部署追踪的 vLLM Python 文件，应用可重复执行的 Humming 补丁。
5. 编译 Humming 的 device-info、NVRTC、cubin patcher、launcher，并更新 manifest。
6. 编译默认 buffered IOCP PLE helper 到 `_dev/out/ple_ssd_io`。
7. 只读校验部署文件。脚本不自动启动、停止或重启服务。

不要在服务运行时删除 venv、修改 site-packages、替换 DLL 或删除 junction。
删除环境前先明确停止服务。删除 `.flashnext-root` 时也不要递归跟随其 junction
去删除目标环境。重建流程本身不执行这些删除操作。

默认保留 ctx=262144、batch=2048、util=0.94、KV=14 GiB、MTP×2、CUDA Graphs、
PLE depth=256 / cache=512 MiB / workers=16 / prefetch=16384；`-Async` 是可选开关。

## 只读审计与 CPU 回归

```powershell
# 对比全部已安装 vLLM Python 源码，并在临时目录重放 Humming 两次。
# 不导入 torch/vLLM，不向 GPU 发负载，不修改运行环境。
& "$repo\.venv-flashnext\Scripts\python.exe" `
  "$repo\_dev\probe\_audit_flashnext_sources.py" --replay-humming

# 重建后的严格文本一致性检查（非部署操作）。
& "$repo\.venv-flashnext\Scripts\python.exe" `
  "$repo\_dev\bin\_patch_vllm_windows_runtime.py" --check

# 部署行为测试，只使用标准库和临时文件。
& "$repo\.venv-flashnext\Scripts\python.exe" -m unittest discover `
  -s "$repo\_dev\test" -p test_flashnext_deployment.py -v

# 编译到独立目录，绝不替换服务正在使用的 DLL。
& "$repo\_dev\bin\_ple_ssd_io_win_build.ps1" `
  -OutDir "$repo\_dev\out\ple_ssd_io_commit_check"
& "$repo\.venv-flashnext\Scripts\python.exe" `
  "$repo\_dev\probe\_ple_ssd_io_win_test.py" --synthetic --depth 256 `
  --dll "$repo\_dev\out\ple_ssd_io_commit_check\ple_ssd_io_win.dll"
```

源码审计会分别报告文本差异和去除注释、docstring、格式后的 Python AST 差异。
文本差异不一定是行为差异；严格部署检查仍会拒绝文本漂移。
审计 JSON 写入 `_dev/out/flashnext_source_audit.json`。其中
`installed_only_python` 必须核对来源，不能自动认定为无关文件。
本次其中的 55 个 FlashAttention Python 文件已逐字核对 pinned Git revision
`506341a143fcabd4bb79052a7605ada727d6b3f5`，包含 CMake 的 CuteDSL namespace 转换。
审计脚本在 `.deps/vllm-flash-attn-src` 存在时会自动重做这一核对。
`vllm/_version.py` 是 `setup.py` / setuptools-scm 生成的版本信息，不是独立修复源码。
初次审计仅有 PLE docstring 和 prepared-launch 格式差异，无 Python AST 行为差异；
本次重新部署后，严格的 13 文件文本校验也已通过。

本次整理验证了部署回归、8 文件 Humming 补丁双次重放、PowerShell 语法/编排、
独立 buffered/direct PLE DLL 的合成文件正确性（525 行、depth=256、页边界/
EOF、提交失败后的取消，以及短读完成包后的取消）。

2026-10-02 整理时的一次“在线端口拒绝”测试调用实际发现 9393 不再监听，
因此旧版脚本执行了**现有隔离环境**的 provision、vLLM 增量构建、运行时部署、
四个 Humming 组件构建及 PLE helper 构建，均成功；严格 13 文件校验通过。
服务日志末尾为 `exited with -1`（文件更新时间 2026-10-01 23:34），退出原因未定位。
没有调用停服命令，也没有自动拉起服务。这不是从空 venv 的完整重建证明。

为避免再次误执行，脚本现在默认只打印计划，实际构建必须显式传 `-Execute`；
端口查询失败会中止，不再静默放行。回归测试使用临时空仓库和临时 TCP listener，
不再以真实服务端口触发重建入口。**未删除原 venv，全新空环境重建仍待维护窗口验证。**

### 本机 lint 的已知限制

提交前执行了完整 pre-commit。Ruff、Markdown、typos、SPDX、导入限制等通过；
以下三项没有全绿，提交时仅按名称跳过，未使用 `--no-verify`：

- `mypy-3.10`：PLE / prepared-launch 的 38 个类型错误，与 HEAD 原文件的
  独立检查逐条相同（包括 Windows 缺少 POSIX 常量及已有 trace 动态字段）。
  对照输出留在 `_dev/out/mypy_flashnext_{baseline,current}.txt`。
- `check-torch-cuda-call`：PLE 原有 CUDA API 调用，当前改动未增加 CUDA API。
- `update-dockerfile-graph`：Windows 环境无法找到脚本 shebang 的 `/bin/bash`。

这些是保留的检查债务，不代表完整 CI 已通过；准备上游 PR 时需另行处理。

## 性能实验只作离线、按需入口

默认使用 buffered IOCP。Direct I/O 仅供 A/B，必须指定独立输出目录：

```powershell
& "$repo\_dev\bin\_ple_ssd_io_win_build.ps1" -Direct `
  -OutDir "$repo\_dev\out\ple_ssd_io_direct_probe"
```

`_dev/probe/_ple_cache_io_probe.ps1` 保存了原盘读计数实验；
`_dev/probe/_single_stream_report.py` 汇总已有原始测量文件。
前者会产生磁盘读负载，不属于只读源码审计，也不应随重建自动运行；
只有共享盘上的服务都空闲且确有测量需要时才运行。
后者不会重新跑 benchmark，但需要原始 `_dev/out` 文件。

单流最终结论与停止门槛见 `BENCH_FLASHNEXT_GPU1_VS_WSL2_GPU0.md` §6。
本次整理不再追逐约 1 ms 的性能收益，不重启服务、不清缓存、不更改驱动。

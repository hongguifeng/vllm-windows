# Project Maya 本机部署与性能记录

> **机器**：`HONG-PC` · Windows 11 26100 · 2× NVIDIA CMP 170HX 64G + 1× RTX 3090
> **模型**：GLM-5.3-Flash **Maya-M**（116 GB，IQ2_S/IQ3_XXS 量化）
> **Maya 版本**：v1.1.27（commit `4615f34`） · 仓库克隆于 `D:\code\project-maya`
> **记录日期**：2026-10-25
> **文档路径**：`D:\Maya-models\MAYA-LOCAL-REPORT.md`

---

## 0. 结论摘要（TL;DR）

| 问题 | 结论 |
|---|---|
| 这台机器能跑 Project Maya 吗？ | **能，而且超过官方同款硬件基准**。官方 `--check` 判定 “This PC can run Maya.”，三个 GPU 全部可用 |
| 实际速度 | **decode 50–67 tok/s**，**prefill 600–1600 tok/s**（随 prompt 长度变快） |
| 官方 2×CMP 170HX 64G 基准 | 61.3 tok/s decode / 514 tok/s prefill（Maya-M，全专家驻显存） |
| 显存是否够 | 116 GB 模型 **100% 驻显存**（12513 个专家槽位，命中率 100%，零 SSD 读取） |
| 上下文能开多大 | **实测 1M（1 048 576 tokens）可用**，321K tokens 的 prompt 实测通过 |
| 并发 | 4 并发聚合 **93.3 tok/s（×2.39）**；但 4 并发与 1M 上下文**不能同时** |
| 主要门槛 | Windows 页面文件（commit 记账）、需要停掉占用两张 CMP 的 vLLM |

**一句话**：硬件完全够，唯一的"坑"是 Windows 把显存分配算进 commit 记账，所以必须把页面文件从 101 GB 扩到 229 GB。

---

## 1. 硬件与软件环境

### 1.1 硬件

| 部件 | 规格 | 备注 |
|---|---|---|
| CPU | AMD Ryzen 7 5700X，8 核 / 16 线程，3.4 GHz，AVX2 | 比官方基准机的 28 核 Xeon 弱，但全专家驻显存时 CPU 基本不参与 |
| RAM | 88 GB（87.9），DDR4-2400，4 条混插（8+16+32+32 GB） | Maya 只要求 32 GB；实际 pin 走约 9.5 GB 做 RAM 层 |
| GPU 0 | RTX 3090，24 GB，CC 8.6，936 GB/s，PCIe Gen2 x16 | 驱动显示，桌面占 ~0.9 GB |
| GPU 1 | **CMP 170HX 64G**，64 GB HBM2e，CC 8.0，1769 GB/s，PCIe Gen2 x8 | 无显示输出（MCDM），纯计算卡 |
| GPU 2 | **CMP 170HX 64G**，64 GB HBM2e，CC 8.0，1769 GB/s，PCIe Gen2 x8 | 同上 |
| 合计显存 | **152 GB** | 官方 check 识别为 "152 GB of VRAM" |
| 系统盘 C: | Intel SSDSC2KW512G8，SATA，剩 35.7 GB | 页面文件主要落在这里（100 GB） |
| 数据盘 D: | **KIOXIA EXCERIA G2 NVMe 1.86 TB**，剩 185 GB | 模型 + pack 放这里；实测顺序读 3.8 GB/s（热缓存） |
| 页面文件盘 F: | TOSHIBA Q300 447 GB，剩 213 GB | 新增 128 GB 页面文件 |

### 1.2 软件

| 项 | 版本 / 路径 |
|---|---|
| OS | Windows 11 专业版 10.0.26100 |
| NVIDIA 驱动 | 610.88（KMD 610.88，CUDA UMD 13.3） |
| CUDA Toolkit | 13.3（`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3`） |
| C++ 编译器 | Visual Studio **2022** Build Tools 17.14（MSVC 19.44.35228） |
| Python | 3.11.0（Maya 自己在 `.venv` 里装 cmake / ninja） |
| Git | 已安装（克隆用） |

### 1.3 环境检查结论

官方 `START-MAYA.bat --check` 的判定：

```
[ok] using GPU 1 (CMP 170HX 64G) + GPU 2 (CMP 170HX 64G) + GPU 0 (RTX 3090): layers split across the 3
[ok] CUDA toolkit 13.3
[ok] NVIDIA driver 610.88
[ok] C++ compiler: Visual Studio Build Tools
[ok] Python 3.11.0
[ok] RAM: 88 GB
[ok] CPU: AMD Ryzen 7 5700X (AVX2, 16 threads)
[!]  Windows' page file is 101 GB; set it to at least 160 GB
This PC can run Maya.
```

**唯一不达标的是页面文件**，不是硬件。

### 1.4 工具链冒烟测试（额外验证）

用本机 nvcc + MSVC 分别编译 `sm_80`（CMP）与 `sm_86`（3090）：

```
nvcc -arch=sm_80 -c t.cu  -> SM80_OK  (obj 41715 bytes)
nvcc -arch=sm_86 -c t.cu  -> SM86_OK  (obj 41715 bytes)
```

`nvcc --list-gpu-arch` 含 `compute_80` / `compute_86`；`cuda_profiler_api.h`、`nvtx3/nvToolsExt.h` 均存在（README 里那个常见坑不存在）。

Maya 的 `CMakeLists.txt` 要求 **sm_80+**（tf32 mma、bf16 math），CMP 170HX 正好是 sm_80，卡在门槛上但满足。

---

## 2. 部署过程与耗时

### 2.1 模型下载

直连 HuggingFace 速度不可用，改用镜像 + 多连接分片：

| 源 | 单流实测 | 90 GB 预计 |
|---|---|---|
| `huggingface.co` 直连 | 0.48 MB/s | **53 小时** |
| `hf-mirror.com` 单流 | 1.6 MB/s | 16 小时 |
| `hf-mirror.com` + 6 路 range 并发 | **4.2 MB/s** | **6 小时** |
| 本次实际（8 路 range） | ~9 MB/s（峰） | **3 小时 24 分**（15:02 → 18:26） |

下载脚本：`D:\Maya-models\dl-maya.ps1`（8 路 range、`.partN` 断点续传、可重入）。

**校验（全部通过，与官方清单一致）**：

| 文件 | 大小 | sha256 前 16 位 | 结果 |
|---|---|---|---|
| `...Maya-M-IQ2_S-00001-of-00003.gguf` | 44.40 GiB | `3f740d1b1022b4a0` | ✅ |
| `...Maya-M-IQ2_S-00002-of-00003.gguf` | 44.54 GiB | `285951d2afa0cd98` | ✅ |
| `...Maya-M-IQ2_S-00003-of-00003.gguf` | 19.10 GiB | `ebf1ce713f712077` | ✅ |
| `vision/mmproj-GLM-5.3-Flash-F16.gguf` | 1.05 GiB | `3627575df16bd152` | ✅ |
| `vision/GLM-5.3-Flash-vocab.gguf` | 0.01 GiB | `8f53cb1bd2e631c1` | ✅ |

> 注意：视觉文件必须放在 `<models-dir>/Maya-M/vision/` 下，Maya 从模型目录读，不是 models 根目录。

### 2.2 构建（`--setup --no-start`，不占 GPU）

| 阶段 | 耗时 | 产物 |
|---|---|---|
| 引擎编译（sm_80 + sm_86） | **约 7 分钟**（18:31 → 18:38） | `build\maya.exe` 150.7 MB |
| 模型校验（109 GB sha256） | 约 2 分钟 | `.maya-verified.json` |
| pack 索引构建 | 约 2 分钟 | `D:\Maya-models\Maya-M\pack` **908 MB** |
| 视觉编码器编译（llama.cpp + ggml-cuda，276 个目标） | **约 17 分钟**（18:39 → 18:56） | `build-vision\bin\maya-vision.exe` 89 MB |

实际编译远比文档说的 "20–40 分钟" 快（CPU 8 核，只编 sm_80 + sm_86 两个架构）。

### 2.3 页面文件（Windows 特有门槛）

Windows 把**显存分配**和 Maya pin 的 RAM 一起计入 commit（RAM + 页面文件）。

| | commit 上限 | 已用 | 是否够 Maya（约 234 GB） |
|---|---|---|---|
| 改造前 | 201 GB（88 RAM + 113 页面文件） | 191 GB（vLLM 占着） | ❌ |
| 改造后 | **315.9 GB**（88 RAM + 100 C: + 128 F:） | 12.7 GB（空载） | ✅ |

具体做法：保留 `C:\pagefile.sys 102400 159744` 原样，新增 `F:\pagefile.sys 131072 131072`（128 GB）。
脚本：`D:\Maya-models\set-pagefile.ps1`（改 `PagingFiles` 注册表值；用 `New-CimInstance` 会撞 `InitialSize` 的 uint32 类型错误）。

> 页面文件在这里**基本不承担换页**——Maya 的内存层是 pinned 的、不会被换出；这 200+ GB 主要是给 Windows 记 commit 账。

### 2.4 与 vLLM 的互斥

部署时两块 CMP 正被 vLLM 占满（`vllm.entrypoints.openai.api_server`，Qwen3.8-Flash-Next-AutoRound-3bpw，端口 **9393/9394**，`--gpu-memory-utilization 0.96`，各占 63 GB / 65 GB）。

- 下载和编译**不需要**停 vLLM（这两步只用网络、磁盘、CPU）；
- 只有**加载模型那一刻**需要显存和 commit，所以必须在启动前停掉；
- 本次是改页面文件后重启，vLLM 随之结束。

---

## 3. 最终配置

`D:\code\project-maya\maya-maya-m.json`：

```json
{
  "exe": "D:\\code\\project-maya\\build\\maya.exe",
  "args": [
    "--glm-pack", "D:\\Maya-models\\Maya-M\\pack",
    "--max-context", "1048576",
    "--expert-cache", "auto",
    "--prefill", "auto",
    "--kv-streaming", "on",
    "--expert-profile", "D:\\code\\project-maya\\data\\expert-profile-glm.bin"
  ],
  "gpu": [1, 2, 0],
  "port": 8080,
  "vision": {
    "exe": "D:\\code\\project-maya\\build-vision\\bin\\maya-vision.exe",
    "mmproj": "D:\\Maya-models\\Maya-M\\vision\\mmproj-GLM-5.3-Flash-F16.gguf",
    "model": "D:\\Maya-models\\Maya-M\\vision\\GLM-5.3-Flash-vocab.gguf",
    "gpu": true
  }
}
```

配置备份：`maya-maya-m.json.good-1m`（1M）、`maya-maya-m.json.good-256k`（256K）。

### 3.1 层分配（引擎自动 split）

```
glm split: layers [0, 22) on CUDA0   (CMP 170HX #1)
           layers [22, 44) on CUDA1  (CMP 170HX #2)
           layers [44, 45) on CUDA2  (RTX 3090 + MTP 草稿层)
```

| 卡 | 层数 | 专家池 | 加载后占用 |
|---|---|---|---|
| CUDA0 (CMP#1) | 22 | 44.8 GB | 50.3–53.4 GB / 64 |
| CUDA1 (CMP#2) | 22 | 51.7 GB | 55.2–57.2 GB / 64 |
| CUDA2 (3090) | 1 + MTP | 5.7 GB | 8.5–8.9 GB / 24 |

引擎开始的预测：`a token 6.2 ms predicted (CUDA0 2.7 ms, CUDA1 2.7 ms, CUDA2 0.8 ms)` —— 即理论上 161 tok/s，实际 50–67（差距来自 MTP 草稿命中率、PCIe hop、实际内存带宽）。

---

## 4. 性能实测

### 4.1 加载与预热

| 项 | 数值 |
|---|---|
| 进程启动 → `/health` 就绪 | **约 100 秒** |
| 专家预热吞吐 | 0.59–0.87 GB/s |
| CUDA0 预热 | 5472 专家 / 47.4 GB / 56 s |
| CUDA1 预热 | 6336 专家 / 54.7 GB / 63 s |
| CUDA2 预热 | 576 专家 / 6.0 GB / 10 s |
| 显存命中率（稳态） | **100.00%** |
| RAM fetches / disk reads | **0.00 /tok** |

### 4.2 decode（写答案）

| 场景 | 实测 | 配置 |
|---|---|---|
| 首次请求（预热后立即） | 52.4 tok/s | 32K / parallel 1 |
| 300-token 长输出（数数） | **67.0 tok/s**（14.92 ms/token） | 32K / parallel 1 |
| 普通问答（256 tok 输出） | 45.9 / 47.5 tok/s | 128K / parallel 1 |
| 同上 | 41.7 / 53.0 tok/s | 256K / parallel 4 |
| 同上 | 43.5 tok/s（decode 50.8） | 512K / parallel 2 |
| 最终配置抽测 | **50.0 tok/s**（20.00 ms/token） | **1M / parallel 1** |

> MTP（NextN）草稿接受率实测 **50–86%**，直接决定 decode 高低：接受率高时 67 tok/s，低时 35–41。

### 4.3 prefill（读 prompt）

| prompt 长度 | 墙钟时间 | 吞吐 | 备注 |
|---|---|---|---|
| 3,142 tokens | 5.23 s | **601 tok/s** | 单 chunk 3200 |
| 30,024 tokens | 29.6 s | **~1,014 tok/s** | 2 chunk |
| 40,525 tokens | 41.2 s | **983 tok/s** | 3 chunk，RAM +0.0 GB |
| 150,022 tokens | 93.0 s | **1,613 tok/s** | RAM +0.3 GB |
| 321,023 tokens | 228.5 s | **1,405 tok/s** | 超过 256K 档，RAM −0.4 GB |

规律：**prompt 越长，单 token 越便宜**（chunk 摊销 PCIe hop）——3K 时 1.66 ms/token，150K 时 0.62 ms/token。

### 4.4 视觉编码（图片输入）

测试图：320×320 白底红圆（红 `#DC1414`），问题 "What shape and color is in this image?"：

- 回答：**"The image shows a red circle."** ✅
- 耗时 6.7 s（含首帧编码）
- 日志：`CUDA0 the vision encoder borrows the pool's tail (3.85 GB) ... 356 experts moved to colder slots ... 3.85 GB are back in the expert pool` —— 编码时临时借专家池，归还后无残留

### 4.5 并发（`--parallel`）

同一组 4 个 160-token 请求，greedy：

| 配置 | 单请求 | 2 并发（聚合） | 4 并发（聚合） | 加速比 |
|---|---|---|---|---|
| 1M + parallel 1 | 50.0 tok/s | — | — | — |
| 512K + parallel 2 | 43.5 | **65.7** | — | ×1.51 (2 路) |
| 256K + parallel 4 | 41.7–53.0 | 51.2 | **93.3** | **×2.39** (4 路) |
| 256K + parallel 4（顺序跑 4 个对照） | — | — | 39.1 | 1.00 |

- 4 并发时每个请求各自仍有 **~24 tok/s**（日志：`request 5..8 done: 151–160 tokens in 6.3–6.6 s`）
- **parallel 对单用户几乎无惩罚**：配置成 parallel 4 时，单独一个请求仍是 41.7–53 tok/s

### 4.6 上下文长度的成本

| 设置 | state/KV 显存（三卡合计） | 3090 上 prompt 通路 | 引擎报的剩余显存 | 备注 |
|---|---|---|---|---|
| 32 K | 0.67 GB | 2.83 GB | 5162 MiB | |
| 128 K | ~0.8 GB | 4.46 GB | 4363 MiB | |
| 256 K | ~0.8 GB | 6.14 GB | 4373 MiB | |
| **1 M** | ~0.8 GB | ~11 GB | 4704 MiB | RAM 占用从 32 GB 升到 48.6 GB |

**结论：上下文在 VRAM 里几乎免费**，因为 `--kv-streaming on` 把 KV 主体放 RAM（`the latent cache in 1.07 / 6.44 / 5.37 GB of pinned RAM`），而 GLM 的 KV 本身极省——321K tokens 实测 RAM **零增长**。

### 4.7 与官方基准对照

| | 官方（2×CMP 170HX 64G，28 核 Xeon，91 GB RAM） | 本机（2×CMP + 3090，8 核 5700X，88 GB） |
|---|---|---|
| decode | 61.3 tok/s | **50–67 tok/s** |
| prefill（8K） | 514 tok/s | **~1,000 tok/s** |
| 专家驻留 | 全部 VRAM | 全部 VRAM（100% 命中） |

多出来的 3090 主要帮在 prefill 和 MTP 草稿层上。

---

## 5. 失败的尝试与原因（重要，避免重复踩）

### 5.1 给 3090 加层：无收益

手动 `"layer_split": "21,42"`（3090 从 1 层 → 3 层）：

| 配置 | decode（同一测试） |
|---|---|
| auto（3090 = 1 层） | **45.9 / 47.5 tok/s** |
| 手动（3090 = 3 层） | 44.3 / 45.8 tok/s |

原因（日志可证）：每个 chunk 跨到 3090 要付 **~730 ms 的 PCIe 跳转**（Gen2，实测拷贝 1.6 GB/s），且它还要兼做 head + MTP。所以 3090 上**每层的边际成本高于 CMP**，引擎的 split 搜索算得对。已回退 auto。

### 5.2 1M 上下文 + 并发：撑不住

| 组合 | 结果 |
|---|---|
| 1M + parallel 2 | ❌ 引擎启动时报 `maya generate: glm fast: the warmed expert tables did not upload (out of memory)`，服务卡在 "still starting" |
| 1M + parallel 4 | ❌ 专家池被算成 **0 slots a layer**，prefill 退化成逐 token，`glm prefill: ... token by token`，且 `KV streaming: the residency maps did not reset` |
| 512K + parallel 2 | ✅ 正常 |
| 256K + parallel 4 | ✅ 正常 |

**规律**：上下文 × 并发数 共享同一份显存预算，乘积大致决定上限（1M×4、1M×2 超限；512K×2、256K×4 可用）。

### 5.3 长 prompt 之后的"掉速尾巴"

每次喂完超长 prompt，**紧接着那一轮** decode 会掉到 4–5 tok/s，日志特征：

```
glm stat: decode 219.42 ms/tok (4.56 tok/s) | vram hit 99.18% | ram fetches 2.75/tok | promotions 2.75/tok
```

原因：prompt 期间向专家池借了 2 GB 级别的激活缓冲（`borrowing 2029 MB of the expert pool while a prompt runs`），挤掉的专家要在随后几个 token 里搬回（promotions）。**下一轮立刻恢复**（实测 25.2 → 50.5 → 49.7 tok/s）。

缓解：`--prefill 2048`（借得更少，但 chunk 变多、prefill 变慢）。

---

## 6. 运行注意事项

1. **Windows 是实验性支持**：Maya 在 Linux 上开发和测速；本机是 Windows + CMP 卡（MCDM 设备），属于官方未验证组合，但实测可用。
2. **不要用 WSL2**：WSL2 的 GPU 驱动只 pin ~1 GB 内存，Maya 要 pin 几十 GB。本机是原生 Windows 跑的。
3. **Maya 会 pin 走几乎所有空闲 RAM**（留约 6 GB），运行期间其他程序会明显变卡；1M 上下文时 RAM 占用 48.6 GB。
4. **显存吃满会退化成"分页"**：Windows 下卡被塞满时驱动把显存换到内存，decode 会掉到个位数。若出现，调大 `--vram-reserve-mib`（如 1500–2048）或降低上下文/并发。
5. **与 vLLM 互斥**：两者都要几十 GB 显存，不能同时满载。想共存给 Maya 配 `"idle_unload_s": 900`（空闲 15 分钟归还显存与 RAM）。
6. **端口**：默认 8080；OpenAI 兼容 `/v1`，Anthropic 兼容 `/v1/messages`，另有 `/v1/responses`。
7. **日志**：`D:\code\project-maya\maya-maya-m.log`；诊断报告 `START-MAYA.bat --report`。

---

## 7. 常用命令速查

```powershell
# 启动（独立窗口，推荐日常用）
D:\code\project-maya\START-MAYA.bat
# 或
D:\code\project-maya\run-maya-maya-m.bat

# 停止
Stop-Process -Name maya -Force

# 只检查硬件
D:\code\project-maya\START-MAYA.bat --check

# 换上下文 / 并发（改完重启生效，约 100 秒）
#   1M 单用户      : --max-context 1048576 --parallel 1
#   512K + 2 并发  : --max-context 524288  --parallel 2
#   256K + 4 并发  : --max-context 262144  --parallel 4

# 调页面文件（需管理员 + 重启）
pwsh -File D:\Maya-models\set-pagefile.ps1

# 健康检查
curl http://127.0.0.1:8080/health
```

---

## 8. 文件清单

| 路径 | 内容 |
|---|---|
| `D:\code\project-maya` | 仓库（v1.1.27）+ `.venv` + `build\maya.exe` + `build-vision\` |
| `D:\code\project-maya\maya-maya-m.json` | 运行配置（1M 上下文） |
| `D:\code\project-maya\maya-maya-m.log` | 引擎与服务器日志 |
| `D:\Maya-models\Maya-M\` | 3 个 GGUF（108 GiB）+ `vision/` + `pack/`（908 MB） |
| `D:\Maya-models\dl-maya.ps1` | 多线程断点续传下载脚本 |
| `D:\Maya-models\set-pagefile.ps1` | 页面文件配置脚本（需管理员） |
| `D:\Maya-models\download.log` | 下载日志（3 小时 24 分） |
| `D:\Maya-models\bench-ab.ps1` | A/B 基准脚本（本报告第 9 章用的就是它）：TTFT 探针 + 3 次 512-token decode + 40K prefill（随机 tag 击穿缓存）+ 长 prompt 后 decode |
| `D:\Maya-models\maya-knob.ps1` | 改一个旋钮（GPU/上下文/env）+ 重启 + 等健康检查，用于可回滚实验 |
| `D:\Maya-models\ab-results.csv` | 第 9 章（单请求 A/B）的原始结果行 |
| `D:\Maya-models\bench-conc.ps1` | 并发基准脚本（第 10 章）：顺序 vs N 并发，聚合 tok/s 与加速比 |
| `D:\Maya-models\conc-results.csv` | 第 10 章并发的原始结果行 |
| `D:\Maya-models\bench-quick.ps1` | 快速 decode 基准（第 11 章）：3 × 256 token，中位数 |
| `D:\Maya-models\dl-parallel.ps1` | 通用并行断点续传下载器（带每段校验，第 11.1 节的 DFlash2 就是它下的） |
| `/home/hong/deep-swe/maya-deepswe7.json` | DeepSWE 10 题 job 配置（第 13 章） |
| `/home/hong/SWE-bench_Pro-os/v2/tooling/configs/host_maya_12346.yaml` | DeepSWE 用的 agent 配置（api_base 指 12346、timeout 3600） |
| `D:\Maya-models\MAYA-LOCAL-REPORT.md` | 本文档（Markdown 源） |
| `D:\Maya-models\MAYA-LOCAL-REPORT.docx` | 本文档的 Word 版 |

---

## 9. 优化实验与最终配置（2026-10-25 21:35–21:57）

> **本节取代第 3 节的初始配置**：最终保留 `gpu: [1, 2]`（**不用 RTX 3090**）。
> 第 4 章的数值是"优化前"的 3 卡基线，保留作为对照。

### 9.1 方法

- **decode**：`(输出 token 数 − 1) / (总墙钟 − TTFT)`，同一固定 prompt、greedy、`max_tokens=512`，取 3 次中位数；TTFT 用 `max_tokens=1` 的请求单独测。这样剔除了 prefill 与首 token 的影响。
- **prefill**：40,525 tokens 的 prompt，**开头加随机 tag 击穿前缀缓存**（否则会被 slot 缓存秒回，测量失真 —— 第一次实验就踩了这个坑）。
- **长 prompt 后的首个 decode**：紧跟在 40 K prompt 之后，测同一段文字，反映"专家池被借走再搬回"的代价。
- 每次只改一个变量，改完重启（约 83–90 秒回到 ready），跑同一套 bench。
- 脚本：`bench-ab.ps1`（测量）、`maya-knob.ps1`（改一个旋钮 + 重启 + 等健康检查）。

### 9.2 结果

| 配置 | decode | prefill (40 K) | 长 prompt 后首个 decode | 相对基线 |
|---|---:|---:|---:|---:|
| 基线：1 M / 3 卡 / MTP 开 | 55.8 | 965 | 35.6 | — |
| MTP 关（`STRATA_GLM_NO_MTP=1`） | 43.9 / 46.8 | 929 / 934 | 26.4 / 27.0 | **−16 ~ −21%** |
| `STRATA_GLM_SPEC_HEAD=2` | 39.0 | 943 | 28.1 | **−30%** |
| **2 卡（`gpu: [1,2]`）** | **71.8** | 1 070 | **50.7** | **+29%** |
| 2 卡 + `COMPLETE_POOL` + `RESIDENT_SKIP` | 72.7 | 988 | 50.4 | +30%（≈噪声） |
| **最终：2 卡，默认 env** | **71.9** | **1 086** | **50.3** | **+29%** |

三次重复的离散度：最终配置 71.6 / 71.9 / 71.9（很稳）；MTP 关时 43.9 / 36.3 / 51.5（离散大）。

### 9.3 三个结论

1. **砍掉 3090 是唯一的结构性收益：decode +29%，长 prompt 后恢复 +41%。**
   机制不是"少一次跳转"（decode 的跳转 64 KB ≈ 40 µs，可忽略），而是：
   - 少一个串行段（每 token 少走一张卡）；
   - 2 卡时 MTP 走"两卡处理相邻 token"的直连流水（`SETTINGS.md:54`），3 卡时靠 `SPEC_HEAD` 分组，反而不如 2 卡干净。
   引擎自带的 token 预测（`6.2 ms`）与实际（20 ms → 现在 13.9 ms）之间那部分固定开销，被这次减少的段数吃掉了相当一块。

2. **MTP 草稿是净收益（+19 ~ 27%），不能关。** 这直接否掉了"低接受率时草稿在亏钱"的猜测：即便接受率掉到 33%，MTP 仍然更快。

3. **`SPEC_HEAD=2` 明显更差（−30%）。** 默认 `h = n/2 = 1` 是对的：让第一张卡单独跑草稿位、另外两张收尾当前 token，比"两张跑草稿、一张收尾"均衡得多。

### 9.4 为什么 `COMPLETE_POOL` / `RESIDENT_SKIP` 没用

两个设置的前提都是"**每个专家都在显存里**"（`SETTINGS.md:50-51`）。实测 `vram 12384/12513` —— 始终差 **129 个槽位**（约 1 GB），条件不满足，所以引擎直接忽略。实测差值 72.7 vs 71.8 在噪声范围内，按"少一个变量"处理，最终配置里**没有保留**这两个变量。

### 9.5 最终配置

```json
"gpu": [1, 2],
"args": ["--glm-pack", "D:\\Maya-models\\Maya-M\\pack",
         "--max-context", "1048576",
         "--expert-cache", "auto", "--prefill", "auto",
         "--kv-streaming", "on", "--expert-profile", "..."],
"env": {}
```

```
glm split: layers [0, 24) on CUDA0, [24, 45) on CUDA1   (两张 CMP，各 ~52 GB 专家池)
显存：60.1 / 58.6 GB（共 118.7 / 128 GB），路由专家命中 100%
```

### 9.6 取舍（要知道的代价）

| 项 | 3 卡 | 2 卡（现） |
|---|---|---|
| decode（短 prompt） | 55.8 tok/s | **71.9 tok/s** |
| prefill 40 K | 965 tok/s | **1 086 tok/s** |
| prefill 150 K（单独抽测） | 93.0 s（1 613 tok/s） | 122.9 s（1 220 tok/s） |
| 3090 | 被占用（做头部+MTP） | **完全空闲 24 GB** |
| 4 并发余量 | 更多 | 更少（1 M 下已无并发余量） |

150 K 那个数字与 40 K 的结论相反（2 卡更慢），只抽测了一次、未做重复，**不宜当作定论**；存疑留待以后验证。

### 9.7 与外部建议的对照

| 建议 | 结果 |
|---|---|
| "砍掉 3090，少一段 + 少一次 730 ms 跳转，+5~15%" | **方向对，量级低估**：+29%。但收益来源不是跳转（decode 跳转可忽略），而是少一段串行 + 2 卡 MTP 直连流水 |
| "`COMPLETE_POOL`+`RESIDENT_SKIP` 治长 prompt 掉速" | **条件不满足**（差 129 槽），实测无差别 |
| "关掉 kv-streaming" | 未测（1 M 下需 +12.9 GB 显存，放不下）；实测 KV 已有 98.27% 命中，预期无收益 |
| "并发才是 PP 的正解" | 与 1 M 冲突（已实测 1 M+2 并发就 OOM）；要并发送必须降上下文 |
| "给 3090 加层" | 已实测：更慢，已排除 |
| "`--calibrate` 基本无感" | 同意（CPU lane 0.00/tok，disk 0.00） |

## 10. 2 卡 + 256K 并发实验（2026-10-25 22:10–22:18）

> 本节把第 9 章最终配置的上下文从 1 M 降到 256 K、并发开到 4，**其余不变**（2 卡、MTP、auto split）。
> 当前运行配置即本节最后一次实验：`gpu [1,2] / max-context 262144 / parallel 4 / env 空`。

### 10.1 上下文是"每请求各一份"，不是共享

`--max-context` 在引擎帮助里是 **"KV/state capacity"** —— 它是**每个会话状态**的容量：

- `--parallel 1`：就是一个对话最多 262 144 token（prompt + 生成合计）；
- `--parallel N`：引擎给 **N 个会话各一份独立状态**，默认每份都是完整的 `--max-context`（日志可证：`4 sequence states: the first for 262144 tokens (0.53 GB), 3 more for 262144 (1.93 GB in all)`）。

这正是之前 1 M + 2/4 并发 OOM 的原因 —— 不是"1 M 被瓜分"，而是 **N × 1 M** 的状态和 N 份 prompt 通路同时要显存。
想让"大主会话 + 小副会话"共存，用 `MAYA_GLM_SEQ_CTX`（默认等于 context）：第一个会话拿满，其余各拿这个值。

### 10.2 并发度量方法

`bench-conc.ps1`：4 个不同主题的 prompt、greedy、`max_tokens=256`；先跑"顺序"（一次一个）再跑"同时"（N 个一起），比较**聚合 tok/s**。single 为 2 次中位数。2 卡、256 K、parallel 4，三组配置各测一次。

### 10.3 结果

| 配置 | 单请求 | 2 并发 | 4 并发 | 4 并发加速比 |
|---|---:|---:|---:|---:|
| **默认（无 env）** | 68.4 / 63.4 | 83.0 / 79.0 | 87.6 / 86.4 | **×1.43 / ×1.45** |
| `MAYA_GLM_SPLIT_ROWS=2` | 69.7 | **53.1（×0.84）** | 91.6 | ×1.47 |
| `SPLIT_ROWS=2` + `EVEN_GROUPS=1` | 63.9 | 78.6（×1.33） | 89.2 | ×1.48 |

（同一行里两个数字是两次独立测量，反映重复性。）

### 10.4 结论

1. **2 卡下单请求 63–70 tok/s；4 并发聚合 86–92 tok/s**，每个请求各自仍有 ~22 tok/s，加速比约 **×1.45**。
2. 加速比上不去是结构性的：2 张卡 = 2 段流水，默认"每张卡处理一个请求"，所以并行的**上限就是 2 个**；第 3、4 个请求只能交错等待。
3. **`SPLIT_ROWS=2` 在 2 请求时是负优化（×0.84）**：`k ≤ 2` 时所有请求被塞进同一个 group，两张卡不再各跑一个（文档 `SETTINGS.md:86` 说的正是这个问题）；加 `EVEN_GROUPS=1` 能修回来（78.6），但也没超过默认的 83.0。
4. **4 请求时三者差别在噪声内**（86.4 / 91.6 / 89.2），收益不足以抵消 2 请求时的损失 —— **默认配置最优，未保留任何 env**。

### 10.5 2 卡 vs 3 卡的取舍

| | 3 卡（1 M，parallel 1） | 3 卡（256 K，parallel 4） | **2 卡（256 K，parallel 4）** |
|---|---:|---:|---:|
| 单请求 decode | 55.8 | 41.7–53.0 | **63–70** |
| 4 并发聚合 | — | **93.3（×2.39）** | 86–92（×1.45） |
| 长 prompt 后首个 decode | 35.6 | — | ~50（2 卡 1 M 时） |
| 3090 | 占用 | 占用 | **空闲 24 GB** |

**如果主要是自己用**：2 卡明显更好（单请求快 25–30%）。
**如果多人 / 多 agent 并发**：3 卡的聚合吞吐更高（93.3 vs 87.6，且加速比 ×2.39 vs ×1.45），但得接受单请求变慢一半。

### 10.6 更正：`parallel 1` 与 `parallel 4` 的单请求差异（2026-10-25 22:31–22:41）

**先更正一个错误结论。** 早前说过"1 M + parallel 1 比 256 K + parallel 4 快 12%"，那个数字不成立：当时把两个口径混在一起比了 —— 一边是"扣除 TTFT 的纯 decode"，另一边是"含 TTFT 的端到端"，而且用的还是两个不同的 prompt。MTP 草稿接受率对文本内容极敏感（实测 33%–86%），不同 prompt 之间本来就能差 30%。

**重测方法**：同一 prompt、同一输出长度（模型自然停在 254 token）、greedy，只比较**墙钟端到端 tok/s**；每种配置独立重启后连测 2–3 轮。

| 配置（2 卡） | 单请求墙钟 tok/s | 重复性 |
|---|---:|---|
| 1 M + `parallel 1` | **60.6** | 4.18–4.19 s / 254 tok（单次） |
| 256 K + `parallel 1` | **58.0 / 58.2**（median） | 45 / 65 / 58 —— **波动大** |
| 256 K + `parallel 4` | **50.6 / 51.3**（median） | 50.5–51.6 —— **非常稳** |

**结论**：

1. **提速跟 1 M 无关，是 `parallel 1` 的功劳。** 256 K 与 1 M 的状态仅差 0.5–0.8 GB（都在显存里），KV 又走流式；把上下文从 1 M 降到 256 K 本身没有带来任何增益。
2. **真正的差异 ≈ 13–15%**：`--parallel 4` 预分配 4 份序列状态（约 1.9 GB）+ 更宽的 row buffer，单个请求也走"批量为 1"的解码路径；`--parallel 1` 走单请求的推测解码路径。这与文档"单请求 ±5%"的说法不一致 —— 文档那个 ±5% 是在**不同硬件/不同请求形状**上测的，本机实测是 13% 左右。
3. **代价的另一面**：`parallel 1` 的重复性差（45–65 抖动），`parallel 4` 非常稳（±1%）。赶时间的交互体验上，稳的那档反而更可预期。
4. 取舍很直接：**自己用 → `parallel 1`（+13%，且能吃满 1 M 上下文）；多人/多 agent → `parallel 4`（单请求 −13%，换 4 并发聚合 ~87 tok/s）。**

> **读数注意**：10.3 那套"单请求 68.4 / 63.4"来自 `bench-conc.ps1`（4 个不同主题的 prompt），10.6 这套来自 `bench-ab.ps1`（MoE 路由那个 prompt）。**两套脚本的绝对值不可互相比较** —— MTP 接受率对文本内容敏感，同样是"256 K + parallel 4"，两个 prompt 能差 20% 以上。能跨配置比较的只有"同一脚本、同一 prompt"的行。

### 10.7 一张卡会怎样（2026-10-25 22:52–23:02）

3 卡 → 2 卡是**变快**，那 2 卡 → 1 卡呢？**断崖式变慢，因为跨过了"模型放不下"这条线。**

单卡（GPU 1，64 GiB）实测：

```
CUDA0 VRAM before the expert pool: 9.55 GB（dense 6.86 + state/KV 0.92 + other 1.71）
CUDA0 expert pool 53.37 GB, 114-222 slots/layer（6591 total）   ← 只有 ~53% 的专家在显存
CUDA0 a pinning cap: 30.00 GB more pin                          ← RAM 层被 pin 上限截断
CUDA0 RAM tier 38.59 GB pinned（4777 slots）
CUDA0 tiers warm: 11038 experts (95.7 GB)                       ← 剩下 1475 个只能在 SSD
maya serve: CUDA0: 699 MiB of VRAM free with everything loaded  ← 几乎贴边
```

| 卡数（同 prompt、256 K、parallel 1） | decode（墙钟） | prefill | 路由命中 | 关键代价 |
|---|---:|---:|---:|---|
| 3 卡 | 48.6 | 965（40 K） | 100% | 多一段流水 |
| **2 卡** | **58.0**（45–65 抖动） | 1 130（40 K） | 100% | — |
| **1 卡** | **20.0**（14.6 / 18.5 / 20.0） | **187**（4.6 K，24.6 s） | **81%** | CPU lane 63 专家/token |

单卡的引擎自述：

```
glm stat: decode 49.71 ms/tok (20.12 tok/s) | vram hit 81.18% | cpu lane 63.00/tok in 31.76 ms/tok
```

**每 token 有 63 个专家要 CPU 算、耗时 31.8 ms —— 占了 49.7 ms 的 64%。** GPU 本身没闲着，是"显存装不下的那 47% 专家"全压到 8 核 CPU 上去了（PCIe 拷贝反而不划算，引擎选了 CPU lane）。

结论：

- **2 卡是这台机器的甜点**：模型 108 GB 专家刚好全驻显存，又只有一段流水。
- **1 卡不是"再快一点"，而是 −66% 的断崖**（58 → 20 tok/s），prefill 更是从 1 130 掉到 187 tok/s（**−83%**）。
- 3 卡比 2 卡慢的原因完全不同：3 卡时模型**也**全在显存（100% 命中），慢是因为多了一段串行流水 —— 这是"能装下但切得太多"，和 1 卡的"装不下"是两种问题。

## 11. 只从 Maya 内部找速度（2026-10-25 23:00–23:50）

前提：硬件已经查过 —— CMP 解码时 **1485 MHz / 60 W**（上限 1695 MHz / 250 W），**不是锁频、不是过热、不是功耗墙**，带宽只用了个位数百分比。所以瓶颈在引擎的每 token 调度，不在卡。

### 11.1 结果（`bench-quick.ps1`：同一 prompt，3 × 256 token，取中位数）

| 实验 | 设置 | median tok/s | 结论 |
|---|---|---:|---|
| 基线 | 默认（MTP 3 草稿，自适应窗口） | **60.6–60.7** | — |
| E1 | `STRATA_GLM_MTP_DRAFT=6` + `MTP_ADAPT=0`（强制最宽窗口） | 57.8 | **更差**：多出来的草稿大概率被拒，验证窗口白算 |
| E2 | `STRATA_GLM_MTP_CHAIN=1`（草稿在设备端连成一条链） | 60.5 | **无差别**：说明链本来就默认开着（token_embd 是 Q8_0，显存也够） |
| E3 | `--dflash`（DFlash2 草稿器，block 8，读主干第 6/15/25/34/43 层，717 MB 在 CUDA1） | **61.0** | +0.7%，**噪声内** |
| E4 | E3 + `MAYA_GLM_DFLASH_CUM=0.75`（放宽累积置信度） | **61.0** | 仍无差别 |

DFlash2 的完整信息：`Anbeeld/GLM-5.3-Flash-DFlash2-GGUF` @ `f3569ac…`，`GLM-5.3-Flash-DFlash2-Q4_K_M.gguf`，697,017,248 字节，sha256 `f4158ffb…`（**已校验通过**），许可证 **CC BY-NC-ND 4.0（非商用）**。加载它时 MTP 块不加载。

### 11.2 结论：单请求 ~61 tok/s 就是这台机器的上限

| 观察 | 说明 |
|---|---|
| 引擎自己的预测 | 一个 token **6.2 ms**（2.7 + 2.7 + 0.8） |
| 实测 | **16.4 ms**（61 tok/s）—— 2.6 倍差距 |
| 每轮推测解码 | 3 个草稿 + 1 次验证，接受率 65–79%，约 **3 token / 50 ms** |
| 卡的状态 | 1485 MHz、60 W、带宽个位数占用 → **延迟受限**，不是算力/带宽受限 |

那 2.6 倍的差距来自"每层都要启动 + 同步"的固定成本（45 层 × 2 卡、Windows + MCDM 驱动），**Maya 现有的旋钮没有一个能改变它**：加宽窗口只是多验证废草稿，换草稿器（DFlash2）在 2 卡上收益归零，设备端链本来就是开的。

### 11.3 真正还能变快的只有两条

1. **并发**：4 个请求聚合 **86–92 tok/s**（已验证，×1.45）。单请求的上限动不了，但总吞吐可以叠。
2. **硬件**：换带宽/PCIe 更好的卡，或单卡放下整个模型（3090 64 GB 改版、A100 80 GB 之类）。本机 2×64 GB + Gen2 的物理形态决定了 61 tok/s 这条线。

> 另外：DFlash2 的 0.7 GB 文件现已不再使用（配置已回 MTP），`D:\Maya-models\GLM-5.3-Flash-DFlash2\` 可以删掉。

## 12. 性能瓶颈深度分析（2026-10-25 23:52–23:58）

用引擎自带的分阶段计时器（`STRATA_GLM_SPEC_PROF=1`，见 `glm_fast_path.cu:6572`）+ 上下文扫描，把 16.6 ms/token 拆开。

### 12.1 每 token 的账（短上下文，2 卡，`parallel 1`）

引擎输出（252 步）：

```
glm spec prof (252 steps, ms/step): wait tail token 8.83 | draft 1.54 | redo 0.53 |
                                   enqueue tail 0.89 | enqueue head 4.91 | emit 0.00
glm spec prof CUDA0 (342 calls): sync 2.775 | boundary 0.122 | embed 0.027 | enqueue layers 1.021 |
                                 DEVICE token 11.47 ms, gap before 0.86 ms
glm spec prof CUDA1 (253 calls): sync 0.002 | boundary 0.025 | embed 0.000 |
                                 DEVICE token 10.74 ms, gap before 0.00 ms
glm stat: decode 16.66 ms/tok (60.03 tok/s)
```

| 侧 | 阶段 | ms/step | 说明 |
|---|---|---:|---|
| host | wait tail token | 8.83 | 等尾卡算完当前 token（本质是 GPU 时间） |
| host | **enqueue head** | **4.91** | 主机提交头卡 24 层的 kernel |
| host | draft | 1.54 | MTP 草稿 |
| host | enqueue tail | 0.89 | 提交尾卡 |
| host | redo | 0.53 | 草稿被拒时重算 |
| host 合计 | | **16.70** | ≈ 实测 16.66 ms/token |
| device | CUDA0（头卡 24 层） | 11.47 | 两卡流水，取大者 |
| device | CUDA1（尾卡 21 层） | 10.74 | |

**结论：host 合计 16.7 ms ≈ device 的 11.5 ms，而总时长 16.66 ms ≈ host 合计 → 主机提交是临界路径，GPU 没被喂饱。**

旁证：解码时卡只有 **60 W / 250 W、1485 MHz / 1695 MHz**，显存带宽只用到个位数百分比。不是算力、不是带宽。

### 12.2 设备时间里的"真实下限"（估算）

每卡每 token 要读：dense 权重 3.3–3.5 GB（每 token 都读）+ 路由专家的 8/278 ≈ 1.5 GB，合计 **≈ 4.9 GB**。
在 1 769 GB/s 下约 **2.8 ms** —— 实测 11.5 ms，**4 倍于带宽下界**。也就是说 GPU 有约 75% 的时间不是在等数据，而是在启动/等待 kernel。

> 佐证：`enqueue head` 4.91 ms ÷ 24 层 ≈ **0.2 ms 一层**的纯主机提交时间。正常 CUDA launch 是 3–10 µs，这里是它的 20–60 倍 —— 典型 WDDM / MCDM 驱动路径的提交开销（两块 CMP 在 `nvidia-smi` 里正是 MCDM 设备）。

### 12.3 上下文的影响：注意力只在增量里

| 上下文 | decode | 变化 |
|---|---:|---|
| ~200 tokens | **16.66 ms/tok** | 基线 |
| ~8 K | ~20.9 ms/tok | +25% |
| ~27.6 K | **22.87 ms/tok** | **+37%** |

27 K 时的分析器采样：

```
DEVICE token 13.36 ms（短上下文 11.47）| boundary 1.274 ms（短上下文 0.122）
```

**增量来源**：设备 token +1.9 ms（注意力扫更长的 KV）+ boundary 从 0.12 → 1.27 ms（KV 流式的窗口边界）。host 提交那 6.5 ms 基本不随上下文变——它是固定开销。

### 12.4 为什么前面那些优化都没用（现在有机制解释）

| 尝试 | 为什么无效 |
|---|---|
| 加宽 MTP 窗口（3→6 草稿） | 这个引擎的推测解码是**流水线**（每 step 出 1 token，草稿是提前算下一步），不是"一次验证出多个 token"。多出来的草稿只在命中时省延迟，被拒就白算 → 实测 57.8 |
| DFlash2 换草稿器 | 只改变草稿质量，不改变**每 token 的提交次数** → 实测 61.0 |
| `SPEC_HEAD=2` | 把 head 从 1 卡变 2 卡：要提交**两组**层的 kernel，`enqueue` 翻倍 → 实测 −30% |
| 3 卡 | 多一个 boundary + 多一份提交 → 48.6 tok/s |
| 1 卡 | 少一次提交，但 47% 专家落到 CPU lane，每 token 白付 **31.8 ms** 的 CPU 计算 → 20 tok/s |
| `MTP_DRAFT` / `DFLASH_CUM` / `MTP_CHAIN` | 都不改变 host 提交量（chain 本来已开） |

### 12.5 什么才真正改变这个数

| 方向 | 机制 | 本机可行性 |
|---|---|---|
| **并发** | 一次提交喂 N 个请求，提交成本摊薄 N 倍 | ✅ 已验证：4 并发聚合 87 tok/s（×1.45），每个请求仍有 ~22 tok/s |
| **单卡装下整个模型** | 消除跨卡 boundary/sync/第二份提交 | ❌ 116 GB > 64 GB 单卡；单卡实测因 CPU lane 掉到 20 tok/s |
| **减少每层 kernel 数 / CUDA graphs** | 直接砍掉那 0.2 ms/层的提交 | ⚠️ Maya 只有 `MAYA_GLM_KDA_GRAPH`（单卡 + KDA 层），本模型是 DSA 层，覆盖不到 |
| **换到 Linux** | 驱动提交路径更轻（Maya 的主场） | ❌ 不支持 WSL2（要 pin 几十 GB 内存） |
| **降上下文** | 27 K → 200 可省 6.2 ms/token | ✅ 但代价是上下文 |

### 12.6 一句话总结

> **这台机器不是"算得慢"，是"提交得慢"：每个 token 约 16.7 ms 花在主机提交与串行依赖上，GPU 真正需要的内存时间只有约 2.8 ms。**
> 所有能减少"每 token 提交次数"的手段（并发、单卡、内核融合、Linux）才有效；所有只改"猜什么/猜多少"的手段（MTP 窗口、DFlash2、草稿链）都无效——实测也确实是这个结果。

## 13. DeepSWE 10 题实测（2026-10-26 02:12–07:23）

在完成部署与调优后，用仓库里现成的 DeepSWE harness 给 Maya-M 跑了一轮 agentic 评测。**结论：3/10（30%）**，目标测试通过率 **91.3%**。

### 13.1 配置

| 项 | 值 |
|---|---|
| 题集 | DeepSWE（datacurve）seed-0 的 10 题，TS/Go/Python/JS |
| harness | Harbor 0.24.0 + `deep_host_agent:CommitHostMiniSwe`（agent 在宿主机跑，结束时 `git add -A && git commit`，供 collect 钩子 `git diff base HEAD` 取补丁） |
| 端点 | `http://127.0.0.1:12346/v1`（本地 LLM 代理 → Maya） |
| 模型 | `openai/glm-5.3-flash`（Maya-M，2-bit 量化） |
| 协议 | 温度 0、`max_tokens=30000`、`reasoning_effort=high`、`step_limit=200`、**单并发** |
| Maya | 2 卡、`--max-context 262144`、`--parallel 1` |
| 超时 | 请求 `timeout=3600`、agent ×2（6 h）、verifier ×2（1 h） |
| 总耗时 | **5 小时 11 分**（平均 ~43 分钟/题） |

### 13.2 结果

| # | 题目 | reward | f2p | p2p | 工具调用 | 失败形态 |
|---|---|---:|---:|---:|---:|---|
| 1 | arcane-drift-detection-baselines | **1.0** | 82/82 | 2/2 | 128 | — |
| 2 | bandit-structured-nosec-directives | 0 | 68/69 | 281/282 | 115 | 近失 + 回归 |
| 3 | effect-sse-httpapi-streaming | 0 | 42/47 | 70/70 | 200 | 部分实现 |
| 4 | go-critic-doc-link-checker | 0 | 2/3 | 15/16 | 93 | 近失 + 回归 |
| 5 | koota-entity-snapshot-rollback | 0 | 83/84 | 47/47 | 90 | 近失（无回归） |
| 6 | kysely-window-grouping-helpers | 0 | 250/254 | 22/22 | 111 | 近失（无回归） |
| 7 | langchain-request-coalescing | **1.0** | 50/50 | 232/232 | 104 | — |
| 8 | meriyah-explicit-resource-declarations | 0 | 48/49 | 51 469/51 469 | 126 | 近失（无回归） |
| 9 | true-myth-iterable-collection-combinators | **1.0** | 96/96 | 561/561 | 70 | — |
| 10 | vitest-duration-sharding | 0 | **0/56** | 0/24 | 111 | **构建被改坏** |

**合计：3/10 = 30%；目标测试 721/790 = 91.3%；回归测试几乎全绿。**

### 13.3 与参考臂对照（同一套 10 题、同一 harness）

| 臂 | 分数 | 备注 |
|---|---:|---|
| DeepSeek flash（API） | **7/10** | 参考上限 |
| AutoRound 3bpw（Qwen3.8） | **4/10** | 其中 2 题因 262 K 上下文超限 |
| **Maya-M（本机 2-bit）** | **3/10** | 无一题因上下文超限；1 题因构建被改坏 |

### 13.4 失败解剖：不是"不会写"，是"差最后一口"

- **5 题是近失**（koota 83/84、kysely 250/254、meriyah 48/49、bandit 68/69、go-critic 2/3）——差 1–4 个边界用例，而 DeepSWE 要求 f2p 与 p2p 全过才记 1 分。
- **2 题带回归**（bandit 的 `test_058_region_unioned_across_statement_lines`、go-critic 的 1 个 p2p）——改功能时碰坏了既有行为，这是小比特量化模型最典型的一种退化。
- **1 题部分实现**（effect-sse 42/47）——对照 DeepSeek 是 **0/47**（完全没做），Maya 反而做得更多。
- **1 题构建被改坏**（vitest）——补丁 73 KB，但 `src/node/duration-history.ts` 有 `TS9010` 类型错误，导致 `pnpm build` 失败、测试根本没跑起来（f2p 0/56 且 p2p 0/24 就是这个信号）。这是"没验证自己的改动能不能编译"的失败，不是环境问题。
- **反面证据**：`meriyah` 的 **p2p 51 469/51 469 全过**、`kysely` 的 `250/254`、`kysely`/`bandit` 的大套回归全绿 —— 说明它**不是乱改**，而是"少做一点、不做破坏"。

### 13.5 可复现步骤（含踩过的坑）

```bash
# 1) 准备 agent 配置（在 host_xhigh_200.yaml 基础上改三处）
#    api_base=http://127.0.0.1:12346/v1, timeout=3600, 去掉 chat_template_kwargs
# 2) 启动
cd /home/hong/deep-swe
export PYTHONPATH=/home/hong/SWE-bench_Pro-os/v2/tooling:$PYTHONPATH
export OPENAI_API_BASE=http://127.0.0.1:12346/v1 OPENAI_API_KEY=sk-local-maya
harbor run -c maya-deepswe7.json --job-name maya-deepswe7 \
  --agent-timeout-multiplier 2 --verifier-timeout-multiplier 2
```

坑（都已修）：

| 坑 | 现象 | 修法 |
|---|---|---|
| litellm 默认 600 s 超时 | Maya 日志 `done (cancel): 9343 tokens in 588.5 s`，agent 卡在重试里 20+ 分钟 | `model_kwargs.timeout=3600` |
| Docker 地址池耗尽 | 10 题全在 11 秒内 `RuntimeError: all predefined address pools have been fully subnetted` | `docker network prune -f` |
| harbor 工具环境缺依赖 | `No module named 'minisweagent'` / `'mcp'` | `uv pip install --python ~/.local/share/uv/tools/harbor/bin/python mini-swe-agent==2.4.6 mcp` |
| 宿主侧 `agent/model.patch` 为 0 字节 | 看着像抓补丁失败 | 正常：应先 commit，真正的补丁在 `artifacts/logs/artifacts/model.patch` |
| 单并发 vs 4 并发 | 4 并发下单请求只有 13–15 tok/s，长生成必撞 600 s 超时 | 评测用**单并发**：55–70 tok/s，43 分钟/题 |

### 13.6 结论与置信度

- **Maya-M 在这套题上 ≈ 30%**，介于 AutoRound 3bpw（40%）与"明显更弱"之间；但 **91.3% 的目标测试通过率**说明差距不在"能不能实现"，而在**最后几个边界用例 + 不破坏既有行为**。
- **样本很小**：10 题、单次运行。3/10 的 95% 置信区间大约是 **7%–65%**，与 4/10、7/10 都没有统计意义上的可分性（McNemar 需要成对样本，这里也没有同题配对）。
- 若要提高可信度，最有价值的两件事：① 对 5 道"近失"题做**同题配对重跑**（看是稳定性问题还是能力问题）；② 把 `go-critic` 这种"3 个测试"的小题剔除，它们放大了单点失败的权重。

## 附录 A：关键日志摘录

**启动与层分配**

```
loading the model (the first start takes a minute or two) ...
glm split auto: CUDA0 (NVIDIA CMP 170HX 64G) memory 1769 GB/s, 62.12 GB free
glm split auto: CUDA1 (NVIDIA CMP 170HX 64G) memory 1769 GB/s, 62.12 GB free
glm split auto: CUDA2 (NVIDIA GeForce RTX 3090) memory 936 GB/s, 22.70 GB free
glm split auto: layers [0, 22) on CUDA0, [22, 44) on CUDA1, [44, 45) on CUDA2 - a token 6.2 ms predicted
glm fast: CUDA0 layers [0,22) expert pool 44.76 GB, 291-291 slots/layer (5529 total)
glm fast: CUDA1 layers [22,44) expert pool 51.66 GB, 291-291 slots/layer (6402 total)
glm fast: CUDA2 layers [44,45) expert pool 5.67 GB, 291-291 slots/layer, the draft layer 291 (582 total)
maya serve: CUDA0: 5162 MiB of VRAM free with everything loaded
```

**decode 统计行**

```
glm stat: decode 14.92 ms/tok (67.00 tok/s) | vram hit 100.00% | ram fetches 0.02/tok | disk 0.00 reads/tok
glm stat: decode 20.00 ms/tok (50.00 tok/s) | vram hit 100.00% | ram fetches 0.00/tok | disk 0.00 reads/tok
glm spec: 155 of 236 drafts accepted (65.7%)
```

**prefill 统计行**

```
glm prefill: 3142 tokens in 5229.5 ms (1.66 ms/token, 601 tok/s), 1 chunk of 3200 tokens |
  CUDA0 1916 ms, CUDA1 2281 ms (hop in 60 ms), CUDA2 583 ms (hop in 229 ms)
glm prefill: 16384 tokens in 14129.5 ms (0.86 ms/token, 1160 tok/s), 2 chunks of 8192 tokens |
  CUDA0 7552 ms, CUDA1 8316 ms (hop in 315 ms), CUDA2 1552 ms (hop in 726 ms)
```

**视觉编码借池**

```
glm fast: CUDA0 the vision encoder borrows the pool's tail (3.85 GB) while it encodes
glm fast: CUDA0 3.85 GB lent to the vision encoder (356 experts moved to colder slots, 63 dropped)
glm fast: CUDA0 the vision encoder's 3.85 GB are back in the expert pool
```

**1M + 并发失败**

```
maya generate: glm fast: the warmed expert tables did not upload (out of memory)
glm prefill: CUDA2 the expert pool (0 slots a layer) can lend no prompt chunk (--prefill auto) - token by token
maya generate: KV streaming: the residency maps did not reset
```

## 附录 B：测量方法与误差

- decode tok/s 取自引擎自报的 `glm stat: decode X ms/tok (Y tok/s)`，是纯解码速率；端到端 tok/s（含 prefill 与 TTFT）会低 10–30%，短输出时尤其明显。
- 每个场景只测 1–3 次；MTP 接受率对内容敏感（50–86%），同一配置不同 prompt 可差 20 tok/s 以上，**不能把单次数字当绝对值**。
- 并发测试用 4 个不同的 160-token greedy 请求，顺序组与并发组用同样 prompt 集合，比较聚合 tok/s。
- 显存/内存数字来自 `nvidia-smi`、`Win32_OperatingSystem` 与引擎自己的 tier 日志；Windows 下显存分配"只在触碰时计入"，引擎报的剩余值可能比 `nvidia-smi` 保守。

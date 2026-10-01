# 在本机 Windows + vLLM 上运行 Qwen3.8-27B-W4A16-AutoRound-fast

模型来源：WSL `~/testcode/qwen38-27b-rtx3090/models/Qwen3.8-27B-W4A16-AutoRound-fast`，
已拷贝到 **`D:\models\Qwen3.8-27B-W4A16-AutoRound-fast`**（15.81 GB，17 个文件）。

## 最短路径

```powershell
# 纯文本，16k 上下文。CUDA Graph 默认开启，gpu_memory_utilization 自动按空闲显存推导
& D:\code\vllm-windows\_dev\bin\_run.ps1 -Model D:\models\Qwen3.8-27B-W4A16-AutoRound-fast `
                               -TextOnly -MaxLen 16384 -MaxSeqs 8 -BatchedTokens 4096

# 起 OpenAI 兼容服务
& D:\code\vllm-windows\_dev\bin\_run.ps1 -Model D:\models\Qwen3.8-27B-W4A16-AutoRound-fast `
                               -Serve -TextOnly -MaxLen 16384

# 新模型出问题、想把失败面缩小时才加 -Eager（关掉 CUDA Graph / torch.compile）
```

`_dev/bin/_run.ps1` 会在 `-Model` 是本地目录时自动改用 `_dev/bin/_verify_27b.py`（而不是 0.5B 那个烟测脚本）。
**CUDA Graph 默认开启是刻意的**：这个模型上它是 **73.5 tok/s**，而 eager 只有 **15.7 tok/s**。

## 模型规格

| 项目 | 值 |
|---|---|
| 架构 | `Qwen3_5ForConditionalGeneration`（多模态，含视觉塔） |
| 层数 / hidden | 64 层稠密，hidden 5120 |
| 注意力 | 24 Q heads / 4 KV heads，head_dim 256 |
| 混合注意力 | `full_attention_interval=4`：每 4 层 1 层全注意力，其余为线性注意力（GDN） |
| MTP | `mtp_num_hidden_layers=1`（`mtp_draft_vocab_ids.pt` + `model_extra_tensors.safetensors`） |
| 量化 | `compressed-tensors` W4A16 pack-quantized，group_size 128，对称 int4 |
| 量化异常点 | `embed_tokens` 是 **int8**，`lm_head` 是 **int4**，视觉块保持 bf16（在 `ignore` 里） |
| 权重体积 | 加载后 13.91 GiB |

## 必须先打的一处补丁（已固化）

这个 checkpoint 把 `embed_tokens` 也量化了，而上游 vLLM 的 qwen3_5 模型代码没有给
`VocabParallelEmbedding` 传 `quant_config`，直接加载会失败：

```
ValueError: There is no module or parameter named 'embed_tokens.weight_packed'
in Qwen3_5Model. The available parameters belonging to embed_tokens
(VocabParallelEmbedding) are: {'embed_tokens.weight'}
```

**内核上游其实已经有了**（`CompressedTensorsEmbeddingWNA16Int`），只是模型代码没接上。
对应 WSL 项目里的 `patches/qwen3_5-embed-quant.patch`（1759 字节，2 个 hunk），
已原样适配到 0.29.1 并固化为幂等脚本：

```powershell
# 同时改安装包与仓库源码两份（只改一份的话重装后会被覆盖 / 不生效）
D:\code\vllm-windows\.venv\Scripts\python.exe D:\code\vllm-windows\_dev\patches\fix_qwen3_5_embed_quant.py `
    D:\code\vllm-windows\.venv\Lib\site-packages\vllm `
    D:\code\vllm-windows\vllm
```

改动内容（`qwen3_5.py` 与 `qwen3_5_mtp.py` 各一处）：

```python
self.embed_tokens = VocabParallelEmbedding(
    self.vocab_size,
    config.hidden_size,
    quant_config=self.quant_config,                 # mtp 里用 vllm_config.quant_config
    prefix=maybe_prefix(prefix, "embed_tokens"),
)
```

**两处都要改**：只改主模型的话，MTP / spec-decode 模式会在
`Qwen3_5MultiTokenPredictor` 处以同样的错误崩掉。

纯 Python 改动，**不需要重新编译**，改完直接重启即可。

## 实测结果（本机 CMP 170HX，64 GB）

| 模式 | max_model_len | 空闲显存 | 引擎初始化 | KV cache | 生成 |
|---|---|---|---|---|---|
| eager | 8192 | 35.6 GiB（LM Studio 占着） | 78.5 s | 19.29 GiB / 239,691 tok | 134 tok / 9.7 s |
| CUDA Graph | 16384 | 35.6 GiB | 194 s（首次含 torch.compile 53 s） | 18.33 GiB / 261,446 tok | 134 tok / 1.6 s |
| CUDA Graph | 16384 | **63.3 GiB（独占）** | 183.6 s（compile 有缓存） | **41.86 GiB / 530,356 tok** | **294 tok / 4.0 s ≈ 73.5 tok/s** |
| eager | 16384 | 63.3 GiB | 41.8 s（eager 跳过 compile） | 44.9 GiB / 572,226 tok | 134 tok / 8.5 s ≈ 15.7 tok/s |

模型权重固定占 13.91 GiB。CUDA Graph + 独占显存时，按 16384 tokens/请求计算可支持
**32–35 路并发**（16k 上下文共 53 万 tokens 的 KV 池）。

**eager 与 CUDA Graph 在这个模型上差 4.7 倍**（15.7 vs 73.5 tok/s），所以 `_dev/bin/_run.ps1` 默认开 CUDA Graph。

运行日志里应能看到这几行，用来确认各条路径真的走对了：

```
INFO [compressed_tensors_wNa16.py:137] Using MarlinLinearKernel for CompressedTensorsWNA16
INFO [qwen_gdn_linear_attn.py:519]     GDN decode kernel: cuda
INFO [flash_attn.py:897]               Using FlashAttention version 2
```

- `MarlinLinearKernel` → 自编译的 Marlin W4A16 内核可用
- `GDN decode kernel: cuda` → **融合**的线性注意力 decode 内核被启用（见下）
- `torch.compile` 在 Windows 上可用，缓存落在 `C:\Users\hong\.cache\vllm\torch_compile_cache`

## 两个容易踩的性能 / 行为点

### 1. 不要照抄 `--mamba-ssm-cache-dtype float16`

WSL 那个 3090 项目显式传 `float16`，但在本卡上这样会**关掉融合内核**：

```
Falling back to the Triton GDN decode path: the fused CUDA kernel requires a BF16
GDN model with ..., BF16 convolution cache, ..., and a GPU with compute capability 8.0+
```

**不传这个参数**（用 vLLM 默认）时日志是 `GDN decode kernel: cuda`。
`_dev/bin/_verify_27b.py` 用环境变量 `VMAMBADTYPE` 控制，留空即默认。

### 2. 该模型的 chat template 默认开启 thinking

生成长回答时会以 `<think>` 开头。按请求关掉：

```python
llm.chat(messages, chat_template_kwargs={"enable_thinking": False})
```

或等价地在 API 请求里传 `chat_template_kwargs`。WSL 项目用 `ENABLE_THINKING=false` 达到同样效果。

## 显存注意事项

`gpu_memory_utilization` 是相对**总显存**的比例，但启动门禁要求
`空闲显存 >= utilization * 总显存`，所以**必须按空闲显存来算**。
本机实测遇到过两次被别的进程挤占：

- LM Studio 的 `llama-server.exe` 占 28.4 GB + 89% GPU
- `gpu_burn.exe` 占 58 GB

如果安装过程中别的进程**释放**了显存（空闲显存变大），vLLM 会直接断言失败：

```
AssertionError: Error in memory profiling. Initial free memory 35.59 GiB,
current free memory 49.29 GiB.
```

这不是配置问题 —— 等对方彻底退出后重跑即可。**跑大模型前先 `nvidia-smi` 看一眼。**

## 把模型从 WSL 拷出来（可复用）

```bash
D:/models/_copy_from_wsl.sh \
  /home/hong/testcode/qwen38-27b-rtx3090/models/Qwen3.8-27B-W4A16-AutoRound-fast \
  /d/models/Qwen3.8-27B-W4A16-AutoRound-fast
```

WSL2 → Windows 两条路组合使用（脚本已实现）：

| 路径 | 速度 | 限制 |
|---|---|---|
| `//wsl.localhost/<Distro>/...`（virtiofs） | ~160 MB/s | 读不了 root-only 文件（如 mode 0600） |
| `wsl.exe -u root -e bash -c 'cat <file>'` | ~86 MB/s | 什么都能读 |

脚本先用 `wsl -u root find -printf` 拿到文件名 + 字节数，逐个先试 virtiofs、
失败回退 root 流式，**每个文件都比对字节数**，已存在且大小一致则跳过（可断点续传）。
本次 15.81 GB / 17 文件，**1 分 54 秒，0 失败**。

注意 `model-00006-of-00007.safetensors` 在源目录里是 `-rw------- root:root`，
所以 virtiofs 读不到，走了 root 流式通道。
另：`wsl -u root` **免密可用**，而 `sudo` 需要密码，走不通。

---

## 开启原生 MTP 投机解码（实测 ≈ +48% 解码）

### 为什么不能直接开

这个 checkpoint 的 MTP 头**不是**共享目标模型的 `lm_head`，而是一个**独立裁剪过的草稿头**：

| 文件 | 内容 |
|---|---|
| `model_extra_tensors.safetensors` | `mtp.draft_lm_head.weight_{packed,scale,shape}` |
| `mtp_draft_vocab_ids.pt` | 40960 个 int64 词表 id（min 0 / max 248076，目标词表 248320） |

上游 vLLM 0.29 的 `Qwen3_5MTP.load_weights` 会把 `mtp.*` 重映射成 `model.*`，然后去找
`Qwen3_5MultiTokenPredictor.draft_lm_head` —— 而该模块里没有这个子模块，于是直接失败：

```
ValueError: There is no module or parameter named 'draft_lm_head'
in Qwen3_5MultiTokenPredictor.
```

### 必须先打的第二处补丁（已固化）

移植 HyperQwen 的 `qwen3_5-mtp-draft-vocab`，做成幂等脚本：

```powershell
D:\code\vllm-windows\.venv\Scripts\python.exe D:\code\vllm-windows\_dev\patches\fix_qwen3_5_mtp_draft_vocab.py `
    D:\code\vllm-windows\.venv\Lib\site-packages\vllm D:\code\vllm-windows\vllm
```

三处改动都在 `model_executor/models/qwen3_5_mtp.py`：

1. `Qwen3_5MultiTokenPredictor.__init__` —— 发现 `mtp_draft_vocab_ids.pt` 时建一个
   40960 行的 `draft_lm_head`（`ParallelLMHead` + 同样的 `quant_config`）
2. `Qwen3_5MTP.__init__` —— 配套的 `draft_logits_processor`
3. `Qwen3_5MTP.load_weights` —— 草稿头被关掉时跳过 `draft_lm_head` 的权重

`compute_logits` 变成：草稿头只算 40960 行 → 用 `index_copy_` 放回 248320 的全词表，其余填 `-inf`。
**投机解码依然是精确的**（拒绝采样由目标模型负责），受影响的只有接受率。
设 `MTP_DRAFT_VOCAB=0` 可以退回共享的全词表 `lm_head`。

纯 Python 改动，**不需要重新编译**。

### 启动

```powershell
# 当前推荐启动器（已经是它了）
& D:\code\vllm-windows\start_server.ps1                    # 视觉塔卸载 + DFlash2，71680 + async（默认）
& D:\code\vllm-windows\start_server.ps1 -TextOnly          # 纯文本，省 0.86 GiB 塔重量
& D:\code\vllm-windows\start_server.ps1 -NoAsync           # 回到 --no-async-scheduling（decode 慢 ~10%）
& D:\code\vllm-windows\start_server.ps1 -DryRun            # 打印 argv 而不启动
```

> 本节原先推荐的 `_serve_mtp.py` / `_serve_mtp.ps1` 已于 2026-09-21 删除，功能并入根目录 `start_server.ps1`
& D:\code\vllm-windows\_dev\bin\_run.ps1 -Model D:\models\Qwen3.8-27B-W4A16-AutoRound-fast `
    -Serve -TextOnly -MaxLen 204800 -MaxSeqs 4 -BatchedTokens 4096 `
    -MTP -SpecTokens 1 -MemUtil 0.95
```

关键参数是 `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`：
`method=mtp` 会自动解析出 `Qwen3_5MTP` 草稿架构；`num_speculative_tokens` 必须是
`mtp_num_hidden_layers` 的倍数（这里 =1）。

> ⚠️ **服务不能放在会被工具/终端超时回收的后台里**。实测：PowerShell 后台任务默认 120 s 超时，
> 到点整个进程树被杀 —— 日志会停在 `torch.compile took ...` 之后，**没有 traceback、
> 没有 WER 事件、系统日志也没有 TDR**，非常像"模型跑不起来"，其实是被杀的。
> `DETACHED_PROCESS` 也躲不过（约 3 s 后同死）。跑服务要用真正独立的方式
> （nohup / 计划任务 / 独立终端窗口）。

### 实测结果（CMP 170HX 64 GB 独占，max_model_len=204800，CUDA Graph 开）

| 指标 | 值 |
|---|---|
| KV cache | **656,132 tokens**，200K/请求时并发 **3.20x** |
| 显存 | 权重 14.81 GiB，总计 **61 027 MiB**（`gpu_memory_utilization=0.95`） |
| 引擎初始化 | 权重 21.7 s + 草稿头 3.3 s；backbone compile 19.4 s，草稿 compile 5.2 s |
| **解码** | **~109 tok/s**（350 tok / 3.20 s；三次 106.8 / 109.0 / 109.8） |
| 对照（无 MTP） | 73.5 tok/s @ 16 384 → **约 +48%** |
| **草稿接受率** | **81.1%**，Mean acceptance length **1.81** |

首次请求会现场 JIT 三个 Triton 内核（`_compute_local_logits_stats_kernel`、`_rejection_kernel`、
`_resample_kernel`），有一次延迟尖峰，之后稳定。

日志里用来确认各条路径走对的行：

```
[compressed_tensors_wNa16.py:137] Using MarlinLinearKernel for CompressedTensorsWNA16
[qwen3_5_mtp.py:109]              MTP drafter uses a 40960-token draft head
[speculator.py:151]               Capturing model for speculator...
[metrics.py:120]                  SpecDecoding metrics: ... Avg Draft acceptance rate: 81.1%
```

### 另一个顺手修掉的小问题

zh-CN 主机上控制台默认 GBK，vLLM 的 ASCII art banner 会在 logging handler 里抛
`UnicodeEncodeError: 'gbk' codec can't encode character '\u2580'`，日志里因此出现一段
**假 traceback**（进程其实没事）。`_dev/bin/_run.ps1` / `_serve_mtp.py` 里都设了
`PYTHONIOENCODING=utf-8` + `PYTHONUTF8=1` 解决。

---

## 开启图片输入（多图）

这个 checkpoint 是完整的 `Qwen3_5ForConditionalGeneration`，视觉塔一直都在 —— 只是之前为了
省事用 `--language-model-only` 把它剥掉了。

### 事实核对

| 项 | 值 |
|---|---|
| `config.json` | `architectures=['Qwen3_5ForConditionalGeneration']`、`language_model_only: false` |
| `vision_config` | 27 层 / hidden 1152 / patch 16 / spatial_merge 2 / `deepstack_visual_indexes: []`（无 deepstack） |
| 权重 | 333 个 `model.visual.*` 张量 = **0.92 GB**，**BF16 未量化** |
| 为什么没被量化 | `quantization_config.ignore` 把 `model.visual.blocks.*` + `merger.*` 全列进去了 |
| vLLM 支持 | `registry.py:575` 注册了该架构，processor 继承 `Qwen3VLProcessingInfo` |
| 依赖 | PIL 12.3.0 / torchvision 0.26.0+cu130 已装（librosa、soundfile 缺，但本模型无音频分支） |
| 文本侧权重 | 14.87 GB（开视觉后 15.79 GB） |

### 两个必须显式设的旋钮

vLLM 的 `--limit-mm-per-prompt` **默认是 999**，而这个 checkpoint 的 `processor_config.json`
里 `size.longest_edge = 16777216`（4096×4096）。按

```
每张图的 token 数 = 像素数 / (patch_size 16 × merge_size 2)² = 像素数 / 1024
```

算，**一张满分辨率图就是 16384 token**，8 张能吃掉大半个 200K 上下文。所以两个都显式钉住：

| 参数 | 本仓库默认值 | 含义 |
|---|---|---|
| `--limit-mm-per-prompt {"image": N}` | 8 | 单个请求最多几张图 |
| `--mm-processor-kwargs {"max_pixels": M}` | 1310720 | 每张图上限约 1280 token；设 0 则退回 checkpoint 的 4096² |

`--max-pixels` 参考档位（宽高比保持，按总面积缩放）：

| max_pixels | 约等于 | token/图 |
|---|---|---|
| 262144 | 512×512 | 256 |
| 1310720 | 1280×1024 | 1280 |
| 4194304 | 2048×2048 | 4096 |
| 16777216 | 4096×4096（checkpoint 默认） | 16384 |

### 启动

```text
# 默认：视觉塔卸载到主机内存，71680 上下文，--async-scheduling
& D:\code\vllm-windows\start_server.ps1

# 调参
& D:\code\vllm-windows\start_server.ps1 -TextOnly               # 退回纯文本，省 0.86 GiB 塔重量
& D:\code\vllm-windows\start_server.ps1 -NoAsync                # WSL 的 --no-async-scheduling 路径
& D:\code\vllm-windows\start_server.ps1 -MaxLen 60928           # 更小的上下文（更多 KV headroom）
& D:\code\vllm-windows\start_server.ps1 -DryRun                 # 只打印 argv
```

### 验证

`_dev/test/_vision_smoke.py` 是**自证型**的：它现场合成一张纯色底 + 大号白色数字的图，用 `data:` URI
发出去（不走网络），要求模型报出图上的数字，并打印服务端统计的 `multimodal_tokens`。

```text
.venv\Scripts\python.exe _dev/test/_vision_smoke.py --port 8000
```

实测输出：`content='Yellow, circle, 1111'`、`multimodal_tokens {'image': 196}`。
返回非 0 = 服务端拒收图片（说明还带着 `--language-model-only`）。

> ✅ **视觉路径已实测通过**（2026-09-20）。WSL 靠 `vision-tower-cpu-offload.patch` 压塔，
> 0.29 已把它上游化成 `--cpu-offload-gb 1 --cpu-offload-params visual`
> （外加 `VLLM_WEIGHT_OFFLOADING_DISABLE_UVA=1`），**不需要移植补丁**。
> 卸载 0.86 GiB，对文本吞吐**零成本**（ITL 26.93 vs 纯文本 28.46 ms）。
>
> ✅ ~~带塔时上下文必须 ≤ 60928~~ —— **已于 2026-09-22 修复，该限制退役**。
> 当初那个 `device-side assert` 的根因是 **flashinfer 的 radix top-k 在 CUDA 图重放时
> 返回越界 candidate_id**（本形状 vocab 248320 / k=16 会走 `RadixTopKMultiCTA`，
> 一行由 11 个 CTA 靠全局内存软件屏障同步），塔和 async 都不是原因 —— 只是 async 的
> 并发窗口让这个竞态足够频繁才被抓到。现在默认走 `torch.topk`（代价 <1% step），
> 塔 + 71680 + async 跑满 C1–C8 × 4 趟、零断言。
> **唯一残留的守卫**：只有显式 `DFLASH2_TOPK_IMPL=flashinfer` 回退旧路径时，
> 根目录 `start_server.ps1` 才把上下文降回 60928（`-AllowUnstable` 可绕过）。
> 详见 `AB_WSL_VS_WINDOWS.md` §2.2c 与 `_dev/patches/fix_flashinfer_topk_graph_replay.py`。

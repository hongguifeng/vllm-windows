# Flash-Next：Windows 原生（GPU1, :9393）vs WSL2 原生（GPU0, :8000）对照基准（2026-10-01）

Qwen3.8-Flash-Next-AutoRound-3bpw-MTP / CMP 170HX ×2（一卡一引擎，互不争显存）。
本仓库的 Windows 原生服务跑在 **GPU1**（`start_qwen38_flash_next.ps1` 默认 `-Gpu 1`，`:9393`）；
对照方是 WSL2 里的原生 vLLM（仓库 `/home/hong/code/qwen3.8-flash-next-cmp170hx`，**GPU0**，`:8000`）。

> **当前结论以 §6 为准**：buffered IOCP 后单流 prefill 为 0.719 / 2.053 /
> 9.475 s（2K / 8K / 32K 中位数），decode chunk 间隔约 18–19 ms。
> §0–5 保留的是切换 buffered 之前的历史数据及当时假设，不是当前性能结论。
> 历史对照不是严格配平 A/B，也未定位剩余 decode 差距的关键路径。
> 环境重建见 [FLASHNEXT_REBUILD.md](FLASHNEXT_REBUILD.md)。

原始数据：`_dev/out/logs/bench_flashnext_gpu1_20261001.log`（本次全部 stdout + 采样汇总）。
WSL 侧的同一份记录：对方仓库 `ops/OPS.md §9.35`、`ops/measurements/perf-history.csv`。

---

## 0. 结论摘要

| 维度 | Windows / GPU1（:9393） | WSL2 / GPU0（:8000） | 倍数 |
| --- | --- | --- | --- |
| prefill 2048（best of 3） | **4.35 s**（471 tok/s） | 0.71 s（~2 900 tok/s） | 6.1× 慢 |
| prefill 8192（best of 3） | **15.78 s**（519 tok/s） | 2.17 s（~3 700 tok/s） | 7.2× 慢 |
| prefill 32768（best of 2–3） | **67.63 s**（485 tok/s） | 12.69 s（2 582 tok/s） | 5.3× 慢 |
| decode 步时 p50（MTP=2） | 21.8 ~ 24.3 ms | 17.3 ~ 17.7 ms | 1.3× 慢 |
| decode tok/s（256 tok 出） | 89.6 ~ 96.4 | 117.6 ~ 124.5 | 1.3× |
| TTFT | 75 ~ 84 ms | 62 ~ 66 ms | 1.2× |
| 并发 1（128 tok）聚合 / 单流解码 | 85.3 ~ 106.3 / 101 ~ 129 tok/s | 115.9 ~ 144.3 / 127 ~ 157 tok/s | 1.3× |
| 并发 4（128 tok）聚合 / 单流解码 | **63.9 ~ 80.6 / 17.4 ~ 21.8 tok/s** | 277.6 ~ 319.8 / 83 ~ 90 tok/s | **约 4× / 4.5×** |
| MTP 接受率 | 52.4 ~ 63.8 %（累计 65.3 %） | 55.8 ~ 62.5 %（累计 60 ~ 68 %） | 相当 |
| 忙时功耗 均值 / 峰值 | **122.5 W** / 230.3 W | 170 W / 320 W | — |
| 忙时 SM 时钟 均值 / 峰值 | 1 260 / 1 485 MHz | 1 423 / 1 485 MHz | — |
| 负载期 `util>10 %` 的采样占比 | **19 %** | 38 %（口径见 §3.5） | — |
| 温度峰值 | 42 °C | 45 °C | — |

三条可判读的结论：

1. **预填是 5~7 倍差距，而 GPU 大部分时间在等**（负载期只有 19 % 的采样点 `util>10 %`，忙时均值 122.5 W / 上限 220 W）
   ⇒ 瓶颈在 **Windows 侧的 PLE/SSD 读路径**（`ple_ssd_io_win.dll` + Windows I/O 栈），不是 GPU 算力、不是功耗墙。
   这与 WSL 侧的结论一致（那边预填瓶颈同样在 PLE 表 SSD 直读，只是 `O_DIRECT` + 原生 AIO 这条路径快得多）。
2. **解码只慢 1.3×**（21.8~24.3 ms vs 17.3~17.7 ms）⇒ 单步的 Windows/WDDM 提交开销是百微秒量级，
   **解释不了预填的 6 倍差**。也就是说 GPU1 这套的价值在解码侧，预填是它的短板。
3. **并发 4 几乎没有批处理收益**（聚合 63.9~80.6 tok/s，还不如单流的 85~106；WSL 侧同档能到 277.6~319.8）——
   这一条与功耗/时钟/硬件无关，最值得先查（见 §4）。

---

## 1. 测试条件

| 项 | Windows（GPU1，本仓库） | WSL2（GPU0，对照仓库） |
| --- | --- | --- |
| 起服 | `start_qwen38_flash_next.ps1` → `_dev/bin/_flashnext_struct_serve.ps1` | `./start.sh` → `vllm-native/bin/run_native.sh` |
| 端口 / 模型名 | `127.0.0.1:9393` / `qwen3.8-flash-next-full` | `127.0.0.1:8000` / `Qwen3.8-Flash-Next` |
| 权重 | `D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP` | `/home/hong/models/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`（同一份，D: ext4.vhdx 内） |
| `max-model-len` | 262 144 | 262 144 |
| `max-num-seqs` | 4 | 4 |
| `max-num-batched-tokens` | 2 048 | 2 048 |
| `gpu-memory-utilization` | 0.94 | 0.94 |
| KV cache | `--kv-cache-memory-bytes 15032385536`（14 GiB，手工 pin）+ `--mamba-cache-mode align` | 引擎自算（加载后 8.69 GiB，307 041 token） |
| 投机解码 | MTP，`num_speculative_tokens=2` | 同 |
| PLE SSD offload | `ple_ssd_offload` + `ple_ssd_io_win.dll`（depth 256 / cache 512 MB / workers 16 / prefetch 16384） | 同参数，走 `ple_ssd_io.so`（`O_DIRECT` + 原生 AIO） |
| 其它 | `--language-model-only`，CUDA Graph 开 | `FULL_AND_PIECEWISE` 图，`QSA_ALLOC_HEAL=1` |
| GPU 功耗/时钟 | 上限 **220 W**、HBM **1 728 MHz** 恒定、SM 上限 1 695 MHz | 同（两卡设置在 §3.5 里核对过一致） |
| 驱动 / WSL | KMD 610.88 | 同（WSL 侧 CUDA 走 `/dev/dxg`，即同一个 Windows 驱动） |

> 两侧的功耗上限都是 220 W（`power.default_limit=250`、`max=300`），
> 本次对照**不是**在功耗墙上做的差异 —— 见 §3.5 的实测功耗。

---

## 2. 方法与命令（可复现）

请求一律**从 WSL 侧发**（`.wslconfig` 是 `networkingMode=mirrored`，WSL 里 `127.0.0.1:9393` 直达 Windows 侧服务），
用的是对照仓库里那套已经在用的探针，**协议与它自己的验收完全一致**，所以数字可以直接互比。

```bash
cd /home/hong/code/qwen3.8-flash-next-cmp170hx

# ① 预填：每次请求都是全新 token id（避免前缀缓存假命中），取 best-of-3
cd ops/bench
BASE_URL=http://127.0.0.1:9393 QWEN_SERVED_NAME=qwen3.8-flash-next-full \
  python3 warmup.py --reps 3 2048 8192 32768

# ② 解码：流式剔除 TTFT，先预热 (64 tok) 再测 3×256 tok，步时用相邻 chunk 间隔的中位数
BASE_URL=http://127.0.0.1:9393 QWEN_SERVED_NAME=qwen3.8-flash-next-full \
  python3 dec_bench.py

# ③ 并发：1 与 4（openai /v1/chat/completions 流式，128 输出 token，2 轮）
cd ../..
vllm-native/opt/vllm/.venv/bin/python benchmarks/bench_server.py \
  --url http://127.0.0.1:9393 --model qwen3.8-flash-next-full \
  --label winsrv-gpu1-128 --output /tmp/final-winsrv-128.json \
  --concurrency 1 4 --tokens 128 --rounds 2

# ④ GPU 侧采样（4 Hz；注意是 -i 1，被测引擎在 GPU1）
nvidia-smi -i 1 --query-gpu=power.draw,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu,clocks_throttle_reasons.active \
  --format=csv,noheader -lms 250
```

**两条命令上的坑**（本次踩到）：

- `ops/bench/warmup.py --url` **不会自动补 `/v1/completions`**：传 `--url http://127.0.0.1:9393`
  会 POST 到根路径直接 404。要么写全 `--url http://127.0.0.1:9393/v1/completions`，
  要么像上面这样用 `BASE_URL=` + `QWEN_SERVED_NAME=`（`ops/bench/_cfg.py` 会读）。
- `ops/bench/dec_bench.py` 原来把模型名硬编码成 `Qwen3.8-Flash-Next`（只对得上 WSL 那套），
  已改成 `_cfg.model_name()`（环境变量优先）—— 这是本次为测 `qwen3.8-flash-next-full` 顺手改的。

**测量纪律**（不遵守就会拿到假数，两条本次都实际踩过，见 §5）：

1. 两个引擎**不能同时加载**：它们的 PLE 表都在同一块 NVMe（D:）上，一方加载/预填会把另一方拖慢 40~60 %。
   测一侧时另一侧必须空闲（`vllm:num_requests_running == 0`）。
2. 解码步时必须在"预热充分 + 宿主内存低压"下测：WSL 侧的 `vmmemWSL` 要 < 32 GB 且**已经收缩完**
   （刚丢完页缓存的那 1~2 分钟里不能测）。

---

## 3. 结果

### 3.1 预填（全新 prompt，best-of-N）

| 长度 | Windows/GPU1 rep0/1/2 | best | WSL2/GPU0 best | 倍数 |
| --- | --- | --- | --- | --- |
| 2 048 | 4.788 / **4.350** / 4.701 s | 4.350 s（471 tok/s） | 0.711 s（2 880 tok/s） | 6.1× |
| 8 192 | 15.924 / **15.780** / 19.844 s | 15.780 s（519 tok/s） | 2.172 s（3 771 tok/s） | 7.3× |
| 32 768 | 67.793 / 68.334 / **67.628** s | 67.628 s（485 tok/s） | 12.690 s（2 582 tok/s） | 5.3× |

Windows 侧三轮之间很稳（±1~3 %），除 8192 的第三轮 19.844 s（+26 %，宿主抖动），
所以不是"偶发慢"，而是稳定在这个量级。

### 3.2 解码（流式，剔除 TTFT）

| 轮次 | Windows/GPU1 步时 p50 | tok/s | 接受率 | WSL2/GPU0 步时 p50 | tok/s |
| --- | --- | --- | --- | --- | --- |
| warm1（64 tok，前序预热） | 24.3 ms | 81.1 | 58.3 % | 17.7 ms | 124.5 |
| run2（256 tok） | 24.2 ms | 89.6 | 58.1 % | 17.9 ms | 117.6 |
| run3（256 tok） | 21.8 ms | 96.0 | 52.4 % | 18.1 ms | 123.7 |
| run4（256 tok，短 prompt） | 23.6 ms | 96.4 | 63.8 % | 18.6 ms | 118.0 |

TTFT 75 / 80 / 84 ms（WSL 侧 62 / 65 / 66 ms）。

### 3.3 并发（128 输出 token，2 轮）

| 档位 | Windows/GPU1 聚合 tok/s | 单流解码 tok/s | TTFT | WSL2/GPU0 聚合 | 单流解码 | TTFT |
| --- | --- | --- | --- | --- | --- | --- |
| C=1 | 85.3 / 106.3 | 101.1 / 129.2 | 243 / 221 ms | 115.9 / 144.3 | 127.2 / 156.7 | 105 / 76 ms |
| C=4 | **80.6 / 63.9** | 21.8 / 17.4 | 456 / 437 ms | 319.8 / 277.6 | 90.7 / 83.4 | 166 / 150 ms |

C=4 时 Windows 侧**聚合吞吐反而低于 C=1**，且每流解码掉到 17~22 tok/s
⇒ 4 个请求没有并起来（或并起来后每步被拉长到接近 4 倍）。WSL 侧同档聚合 ~300 tok/s、每流 83~90 tok/s。

### 3.4 MTP 接受率

Windows 侧单轮 52.4~63.8 %、`/metrics` 累计 65.3 %（2 019 / 3 094）；
WSL 侧 55.8~62.5 %、累计 60~68 %。**两侧一致** ⇒ GPU1 上的 MTP 路径是"正确地"跑起来的，不是"跑得快但在胡说"。

### 3.5 GPU 侧采样（250 ms × 1 294 点 ≈ 324 s，覆盖整段负载）

| 指标 | Windows / GPU1 | WSL2 / GPU0（同期对照口径） |
| --- | --- | --- |
| `util > 10 %` 的采样占比 | **19 %** | 38 % |
| 忙时功耗 均值 / p95 / 峰值 | **122.5** / 217.4 / 230.3 W | 170 / 231 / 320 W |
| 忙时 SM 时钟 均值 / 峰值 | 1 260 / 1 485 MHz | 1 423 / 1 485 MHz |
| 忙时 util 均值 | 67 % | 75 % |
| `SW Power Cap`（`0x4`）命中 | 92 / 1 294（7 %） | 264 / 936（28 %） |
| 温度峰值 | 42 °C | 45 °C |

口径说明：两列的采样窗口**长度与负载构成不同**（GPU1 这段是 3 轮 2048/8192/32768 + 解码 + 并发；
WSL 那段是 2 轮 131072 + 短预填 + 解码），所以只有"忙时功耗/时钟"可以直接比，
"忙时占比"这一行只用来支持一个定性结论：**GPU1 在整段 324 s 的预填负载里，有 81 % 的采样点几乎空闲** —— 它在等 I/O。

两卡的静态设置本次核对过一致：功耗上限 **220 W**（`default 250 / max 300`）、
HBM **1 728 MHz 恒定**（= 硬件上限 `Max Clocks: Memory`）、SM 上限 1 695 MHz。

---

## 4. 判读与下一步（按性价比排序）

1. **先查 C=4 为什么没有批处理收益**（收益最大、代价最低、与硬件无关）：
   对比两侧的调度/缓存参数 —— `--kv-cache-memory-bytes 15032385536` 手工 pin + `--mamba-cache-mode align`
   是关键差异（WSL 侧是引擎自算的 8.69 GiB / 307 041 token）。
   复现方法：`_dev/probe/_watch_req.ps1` / `_scan_step.py` 从日志反推每步 batch，
   或直接看 `vllm:num_requests_running` 与每步耗时随 C 的变化。
2. **再查 PLE 读路径**（差距最大但改动也更重）：
   Windows 侧是 `ple_ssd_io_win.dll` + 常规文件读；WSL 侧是 `O_DIRECT` + 原生 AIO（depth 256 / 16 workers / 16384 tok 预读）。
   可用 `_dev/probe/_pcie_scale.py`、`_paging_watch.ps1` 看是"读带宽不够"还是"读与计算没有重叠"。
   判据：若把 `ple_ssd_workers` / `ple_ssd_prefetch_tokens` 调大能让 32768 预填时间显著下降 ⇒ 是并发度问题；
   若完全不敏感 ⇒ 是单次读路径的延迟问题。
3. **解码的 1.3×** 属于 Windows 固有开销（WDDM 提交 + 无 `breakable cudagraph` 之类的差异），
   除非有明确需求，优先级最低。

---

## 5. 本次踩到的两个测量陷阱（写下来避免重犯）

1. **贴着"丢宿主页缓存"开测**：WSL 侧 `bin/drop_host_cache.sh` 一执行，`vmmemWSL` 从 44 GB 往 12.7 GB 收缩，
   宿主在忙着换页，而这期间脚本**当场打印的仍是 44.1 GB**（收缩没完成，读数本身也误导）。
   在这个窗口里测短预填，会拿到 0.65 s → 1.09 s（+68 %）、8192 从 2.23 s → 3.04 s 的假数字。
   ⇒ 丢完缓存要等 `vmmemWSL` 稳定（< 32 GB）再测。
2. **另一个引擎正在加载权重**：本次第一轮 250 W 测量之所以"变慢 60 %"，真正原因是
   本仓库的 Windows 引擎当时正在从同一块 NVMe 读 143 GB 权重。
   两个引擎共用一块盘 ⇒ **永远分时测量**，测一侧时另一侧必须空闲。

---

## 6. Buffered IOCP 后的单流复核（2026-10-01，22:11–22:20）

后续优先级由用户明确为 **单流 prefill / decode 与 WSL 达到同一水平**；
并发优化暂停。本节所有生成负载只发往 Windows GPU1 / `:9393`，未向 WSL / GPU0
发送生成请求，未改服务启动参数、KV 预算或 MTP 深度，也未刷新全局文件缓存。
本节为当前 Windows 与 **历史 WSL** 的对照，不是同时期、完全同协议的 A/B。

### 6.1 Prefill：新输入、实际计算 token 数、零前缀命中

使用安装环境的 `vllm.exe bench serve`，工作目录 `C:\Users\hong`，random dataset，
每档 3 个新 prompt，concurrency=1，output=1，temperature=0，seed=`40200+输入长度`。
`num_warmups=0`，`ready_check_timeout_sec=0`。此版本虽打印 initial test 标题，
但 ready check 关闭时并不发出额外测试请求；已核查安装环境源码。

| 输入 | Windows TTFT 三轮（s） | 中位数（s） | 历史 WSL best-of-N（s） |
| ---: | --- | ---: | ---: |
| 2048 | 0.883 / 0.717 / 0.719 | 0.719 | 0.711 |
| 8192 | 2.053 / 2.036 / 2.095 | 2.053 | 2.172 |
| 32768 | 10.485 / 7.584 / 9.475 | 9.475 | 12.690 |

每档 `request_prefill_kv_computed_tokens_sum` 增量分别为 6144、24576、98304，
等于请求输入总数；`prefix_cache_hits_total` 和 `num_preemptions_total` 增量均为 0。
因此不是 KV 前缀缓存假提速。32K 仍有明显轮间波动，不能只报最快一轮。

结论：已达到历史 WSL 的同一量级。不能宣称 Windows 比 WSL 更快：历史 WSL
用随机 token-ID 输入及 best-of-N，本次用 bench random 输入及中位数；软件版本、
宿主压力、文件缓存和 Python PLE 行缓存状态也没有完全配平。

原始结果：`_dev/out/buffered_single_prefill_{2048,8192,32768}.{json,log}`；
每档的 `*_before.metrics` / `*_after.metrics` 保存了服务计数器。
汇总：`_dev/out/buffered_single_stream_report.txt`。

### 6.2 Buffered 提速不只是热缓存；合并读尚不足以解释

独立 DLL 探针覆盖整个 95.37 GiB shard，32769 行（含文件最后一行），depth=256，
按 offset 排序，在字节校验之前重复读取相同 offsets。Windows / WSL 服务均空闲。
用 `Win32_PerfRawData_PerfDisk_LogicalDisk` 的 D: 累积原始计数器差分观察盘读，
不是用进程逻辑 ReadFile 字节数代替盘读。D: 仍是共享盘，后台负载不能绝对排除。

| 路径 / 输入 | native pass 时间（s） | 整个探针窗口 D: 盘读（MiB / 次） |
| --- | --- | --- |
| buffered 新 seed 40101，3 passes | 0.270 / 0.123 / 0.124 | 133.734 / 31717 |
| buffered 相同 seed，新进程，3 passes | 0.125 / 0.124 / 0.129 | 0.004 / 1 |
| buffered 新 seed 40102，3 passes | 0.241 / 0.121 / 0.126 | 133.336 / 31712 |
| direct 相同 offsets，2 passes | 3.697 / 3.448 | 276.252 / 65539 |
| 空闲基线，2.18 s | — | 0 / 0 |

每次抽样覆盖约 3.53 万个不同 4 KiB 页。新 seed 的第一 pass 确有大量设备读，
仍比 direct 快约 13–15 倍；重复 pass 可观察到额外约 2 倍文件缓存收益。
各探针的 130 行字节抽查无 mismatch；另外 seed 40103 的整个 32769 行批次逐行
校验均无 mismatch（`_dev/out/buffered_all_sample_rows_checked.log`）。
不同 seed 本身不保证冷缓存，盘读计数才
支持“包含设备 miss”的判读；未清空系统缓存，所以不应称为严格全冷测试。

必须修正之前的解释：buffered 盘读次数仍接近一行一次，**不能把主要收益归因于
Cache Manager 合并或顺序预取**。已证明的是这个机器上的 buffered 路径显著更快，
尚未定位到 ReadFile 提交、完成等待、文件系统、驱动或 minifilter 中的具体一层。
不能从已有结果指定某个安全软件为根因。

物理内存可用约 46.4–46.5 GiB，探针期间基本稳定。PrivateBytes 是提交量，不等于
resident working set；此前的 66.7 GiB private memory 不能用来断言实际 RAM 占用。

探针支持 `--repeat`，重复计时后再做校验；mismatch 会返回非零。
测量脚本已追踪为 `_dev/probe/_ple_cache_io_probe.ps1`，汇总代码为
`_dev/probe/_single_stream_report.py`；原始日志仍是本地产物。
记录：`_dev/out/buffered_cache_disk_snapshots.json`、
`_dev/out/buffered_cache_probe_summary.log` 及 `buffered_fresh_*` / `direct_same_*` 日志。

### 6.3 单流 decode：使用 WSL 同款自然语言探针

运行 `_dev/probe/_dec_bench_windows.py --port 9393 --model qwen3.8-flash-next-win
--warm-tokens 1024`，同款中文 PagedAttention prompt，completions 流式，temperature=0.6，
ignore_eos，MTP×2；先 3×1024 暖机，再测短块。暖机后可用 RAM 约 46.5 GiB。

| 轮次 | completion | 接受率 | 流式 chunk 间隔 p50（ms） | 剔除 TTFT 的 tok/s |
| --- | ---: | ---: | ---: | ---: |
| warm2 | 1024 | 62.7% | 18.6 | 120.8 |
| warm3 | 1024 | 60.3% | 18.5 | 119.1 |
| run2 | 256 | 55.3% | 18.4 | 112.9 |
| run3 | 256 | 59.9% | 18.0 | 120.7 |
| run4 | 256 | 52.8% | 18.6 | 110.3 |
| short prompt | 256 | 59.4% | 19.0 | 116.7 |

历史 WSL 同类记录的 chunk 间隔约 17.3–18.6 ms、输出约 118–124 tok/s。
当前 Windows 已接近，但接受率、随机采样和宿主压力仍会移动吞吐。流式 chunk
间隔是客户端观察，不是 CUDA-event GPU 单步时长，MTP chunk 也不等于一个 token。

另外做了 45 s、100 Hz、`--idle --nonblocking` 的 py-spy 采样：主步线程的样本约
60.6% 在 Future.result 等待，attention backend 15.8%。期间计数器增量 1852 次引擎
iteration；等待栈驻留约 10.48 ms/iteration，metadata builder 栈约 3.43 ms/iteration。
这些是 **wall stack residency，不是真 CPU service time，更不是关键路径成本**；
不同子系统样本可重叠，不能相加。采样 run 的短块间隔约 18.5–19.0 ms，未见大幅扰动。

记录：`_dev/out/buffered_natural_decode.log`、
`_dev/out/buffered_natural_decode_profiled.log`、`_dev/out/spy_buffered_c1.raw`。

### 6.4 收尾判定：不为约 1 ms 强行继续优化（2026-10-01，22:55）

用户明确收益门槛：“如果只能再抠出 1ms，就不要太勉强了”。因此此处更新 §6.3
之后原拟进行 instrumented restart 的计划：**当前停止性能微优化，保留 buffered IOCP**。
未重启、未替换 live DLL、未改 MTP×2、KV 或图参数；服务仍运行于 GPU1 / `:9393`。

最后一轮无 profiler 的 C1 复测，用同一自然语言脚本、3×1024 解码暖机：

| 测量块 | chunk 间隔 p50（ms） | 纯解码 tok/s | MTP 接受率 |
| --- | ---: | ---: | ---: |
| run2 | 18.3 | 117.2 | 58.5% |
| run3 | 18.5 | 118.6 | 59.9% |
| run4 | 18.5 | 115.9 | 58.0% |
| short prompt | 18.9 | 119.5 | 61.8% |

与 §6.3 及采样复测的 18–19 ms 同形状。前后 RAM 可用 46.4→45.5 GiB，无 KV 抢占，
请求计数增量恰为 8，生成 token 增量 4160，符合探针的全部暖机和测量块，未见额外
请求混入。记录：`_dev/out/single_stream_stop_check*.{log,metrics,txt}`。

**收益定价，不是假定硬上限**：若固定接受长度 2.2，把周期 18.5 ms 降到 17.5 ms，
吞吐比值为 `18.5/17.5 - 1 = 5.7%`；512-token 输出约省 `512/2.2 × 1 ms = 0.23 s`。
既没有证明存在一个稳定、可回收的 1 ms 缺口，也没有证明理论最大优化空间只有 1 ms。
但现有证据不支持廉价且可靠的大幅提升，因此按用户门槛收尾。

重新读完 `FLASH_NEXT_ROUTE_A_BUILD.md` 的后期实验，发现必须避免重走已否证路线：

- §41：OMP/MKL 线程数钉死中位吞吐 −0.7%；异步调度和 breakable graphs 原已开启。
- 后部第二个 §44：GDN CPU-mask 快路在同夜对照中令构建时间下降约 0.135 ms/步，
  但吞吐 119.9→119.2 tok/s（−0.6%），接受长度同为 2.33；未晋升、代码已移除。
  因此不能把“metadata 构建较贵”当作“删除它即可省同等周期”的论据。
- §26：MTP×4 在合成重复文本收益大，在真实混合固定表仅 +0.3%，不改当前 ×2。
- §33–36：历史 torch profiler / 两个 Nsight 构建连小 CUDA 探针也没有 CUDA activity；
  在没有新的工具可用性证据时，不重复安装、提权或驱动级尝试。硬件/权限根因仍需
  独立证据，不能把文档中的猜测当作已定位事实。
- §38 的相位数字曾乘除 9.14 做标定，不能据此把“GDN 8 ms”当作可靠 kernel 预算。
  重用 event 对也可能混到不同 replay 的代次。breakable FULL 的 event span 可含 host
  等待，forward+drafter 并非完整步周期；不能把周期均值减 span 中位数叫精确 GPU idle。

外部只读收尾评审：`consultations/astra_brief10.md` / `astra_reply10.md`，建议停止。
本轮不新增复杂快路、kernel 重写、缓存清理、加深 MTP 或 instrumented restart。
未来仅在同口径、接受率及宿主状态配平后，观察到稳定约 **≥2 ms/步或 ≥10%** 的实际
差距，或出现已有验证的低风险简单改动时，再考虑重开。该门槛按用户收益偏好设定，
不是声称当前性能已达到硬件理论极限。

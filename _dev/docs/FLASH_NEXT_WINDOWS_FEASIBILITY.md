# Qwen3.8-Flash-Next 迁移到本仓库（Windows / CMP 170HX）可行性

调研日期：2026-09-27。参考实现：WSL 里的 `/home/hong/code/qwen3.8-flash-next-cmp170hx/`，
当前正跑在 docker 容器 `hong-pc`（镜像 `18gogogo/170hx1-qwen38nextf:sm80`，端口 9393→8000）里，
占用 GPU0 的 62.1 GiB。GPU1 空闲。

## 1. 模型与 WSL 侧基线

| 项目 | 值 |
|---|---|
| 架构 | `Qwen4ExpForConditionalGeneration`（`vllm/models/qwen4_exp`，本仓库已注册） |
| 层构成 | 48 层 = 36 `linear_attention`(GDN) + 12 `qwen_sparse_attention`(QSA)；MTP 1 层 `full_attention` |
| head_dim | 256（和 27B 同）；`hc_count=4` 超连接；`indexer_budget=2048` |
| PLE | 只有 1 层（layer id 2），`ngram_size=3`，`heads_per_ngram=8` → 16 头 × 20M 行 = 3.2 亿行，每行 160×bf16 = 320 B = **95.37 GiB**，独占 `model-00001-of-00011.safetensors` |
| 权重总量 | 143 GiB（13 个 shard：11 主 + 2 MTP），其中 GPU 常驻 **47.32 GiB**（容器实测） |
| CUDA graph | 捕获 28 s，占 **2.37 GiB**；`FULL_AND_PIECEWISE` + breakable graph |
| 起服耗时 | 加载 130 s + 图 28 s，整体 8–10 min |
| 量化 | `auto-round` 3bpw → vLLM 侧走 `inc`（`quant_method=inc`），MoE 是 `RoutedExperts` + 混合位宽 per-expert 覆盖 |
| 参考性能 | **pass2 / 参考机**（不是本机：15 GiB RAM + 31 GiB swap，企业级 SATA SSD）：**MTP=1**，c1 解码 110.99、c16 聚合 667.13、8K prefill 2,636。**pass3 / 本机 WSL2 原生**：**MTP=2**，c1 解码 **131.19**、c4 聚合 **305.42**、c8 304.07。两个参考的 MTP 深度不同，不能直接互比。Windows 原生已复测 MTP×2：在我们这套单流流量上 c1 **144.0** tok/s（接受长度 2.73、接受率 86%，每步成本跟他们 MTP×2 齐平），见 ROUTE_A §17；他们的 131.19 是他们负载下的数，不是靶子。**同日重测深度阶梯（ROUTE_A §19）**：×2 中位 **149.5**（三块 144.4/149.5/152.2，±5%）、×3 **155.8**、×4 **175.7**（对 ×2 **+17.5%**，零代码改动），深度 5 起服即被 QSA 布局断言挡死；GPU 占用始终 ~40%，步时被 CPU 提交路径当限速。

结论：**95.37 GiB 的 PLE 表不可能进显存，也不可能进 pinned host memory**。本机装了 88 GiB 物理内存
（`Win32_PhysicalMemory` 求和），Windows 空闲 48.2 GiB、commit 上限 239.9 GiB / 空闲 128.0 GiB；
95.37 GiB 直接大于总内存，而本项目已经因为 commit 逼近 130 GB 出过 BSoD。
所以 WSL 侧用 SSD 后端，本仓库也必须用 SSD 后端 —— 上游那条 `EngramConfig(cpu_offload=True)` 的
pinned-host + UVA 路线在本机不可用。

## 2. 三个硬阻塞（按严重程度）

### 2.1 当前构建的树根本没有 host/SSD 路线
`HEAD = v0.29.0`：
- `vllm/models/qwen4_exp/nvidia/ple_layer.py`（1256 行）里 `Qwen4ExpNGramEmbedding` 是 **GPU 常驻**的，
  文件头注释直接写着 "GPU-resident"；只有 FP8-on-device 一种方法。
- 没有 `ngram_embedding.py`，没有 `vllm/config/engram.py`，全仓库 `grep Engram` 零命中。
- 把补丁打在工作树上逐文件失败（`git apply --check`）：`test_ple.py`、`test_auto_round.py`、
  `breakable_cudagraph.py`、`mtp.py` 等都 does not apply —— v0.29.0→a5a30471f 之间 PLE 被拆分重构过。

也就是说：**要在 Windows 上跑这个模型，先把树升到带 Engram/ngram_embedding 的版本，或者把这套
host-offload 语义手工后移到 v0.29.0。** 后者约等于重写上游一个 feature，不划算。

### 2.2 补丁能干净落到 `origin/main`，但那条分支没有 Windows 适配
证据（已做过，不是推测）：

```
git worktree add --detach D:\code\vllm-windows origin/main   # 128 MB
cd D:\code\vllm-windows && git apply --check -v .../qwen38-ple-ssd.patch
→ 13 个文件全部 Checking 通过，零 error
```

而 `merge-base(origin/main, a5a30471f) = 71fc70d3ae`，那 23 个上游 commit **一个都没碰**补丁涉及的
8 个文件 —— 所以干净是必然的，不是运气。

问题是 `origin/main` 是"跟 upstream main"的那条线（`Merge branch 'vllm-project:main' into main` ×8 +
几个 README），它的 `requirements/cuda.txt` 没有任何 win32 分支：`torch==2.13.0`、`torchvision==0.28.0`、
`flashinfer-python==0.6.18.post1`、`tilelang==0.1.12`、`humming-kernels[cu13]==0.1.12`、
`nvidia-cutlass-dsl[cu13]==4.6.2`、`quack-kernels==0.6.4`、`instanttensor>=0.1.9`。
本仓库能跑的那条 `vllm-for-windows` 分支自己维护了一整套 win32 替身：
`torch 2.11.0+cu130`、`torchvision 0.26.0+cu130`、`SystemPanic/flashinfer-windows 0.6.11.post3`、
`tilelang 0.1.10`、`SystemPanic/humming-windows 0.1.15`、`cudnn-frontend <1.19`。

好消息：`torch-2.13.0+cu130-cp312-win_amd64.whl` 在 download.pytorch.org 的 cu130 索引里**确实存在**
（2.10/2.11/2.12/2.12.1/2.13/2.14 都有）。坏消息：flashinfer 0.6.18 / humming 0.1.12 / cutlass-dsl 4.6
这几样在 Windows 上没有对应端口，而 `indexer_qsa.py`（QSA 稀疏注意力）是依赖 flashinfer 的。
所以升到 `origin/main` 的真实成本不在打补丁，而在**依赖重新移植 + 全量重编（37 min）+ 把本轮之前那
1937 行 27B 优化后移**。

### 2.3 PLE 的 I/O 层是 Linux-only，而且本机把 Windows 的等价物也封掉了
`_dev/probe/_ple_winio_probe.py` 实测（今天跑过）：

```
os.pread          ABSENT    os.posix_fadvise  ABSENT
os.O_DIRECT       ABSENT    /proc/self/fd/3   ABSENT
```

这四项都能在 Python 侧用 seek+read / 不做 fadvise / 直接按路径 open 顶替 —— 代码里本来就有
`ple_ssd_workers` 的线程池 fallback，只差一个 `pread` 替身。

真正的问题在**绕过文件缓存**这一层。`ple_ssd_io.c` 靠 O_DIRECT 把每行扩成 4 KiB 对齐页读，
Windows 的等价物是 `FILE_FLAG_NO_BUFFERING`，本机**在所有盘、所有文件上一律失败**：

| 目标 | 结果 |
|---|---|
| D:\ 上的 27B safetensors | `ReadFile` → `ERROR_INVALID_PARAMETER (87)`，4096B@0 / 512B@0 / 4096B@4096 全失败 |
| D:\ 上新建的 probe.bin | 同上 87 |
| C:\ 上新建的 probe.bin | 同上 87 |
| `C:\Windows\System32\drivers\etc\hosts` | 同上 87 |
| 同一文件不带 NO_BUFFERING | `ok=1 got=4096`，正常 |
| IOCP + `FILE_FLAG_OVERLAPPED` 提交 | 先成功约 60 个，随后 `ERROR_INVALID_HANDLE (6)` |

对齐本身没做错（offset/size 都是 4096 的倍数，缓冲区来自 `VirtualAlloc`，64 KiB 粒度对齐），
同一段代码换掉 NO_BUFFERING 就能读 —— 所以这是环境级的拦截，不是用法错误。最可疑的是
**火绒**（`HipsDaemon` + `ahflt.sys`，分组 `FSFilter Activity Monitor`）在文件栈里；
Defender 实时防护是关的，但 `WdFilter` 仍是 boot-start。`fltmc` 需要管理员权限，没跑成，
所以"哪个 minifilter 干的"目前只是推断，不是结论。

后果：Windows 版只能走 **buffered** 线程读，靠系统文件缓存。实测缓存命中的随机读：

```
threads  1: 165,548 rows/s  697 MiB/s  p50 0.00 ms p90 0.01 ms p99 0.02 ms
threads  8: 142,048 rows/s  598 MiB/s  p50 0.04 ms p90 0.13 ms p99 0.27 ms
threads 32: 137,845 rows/s  580 MiB/s  p50 0.20 ms p90 0.54 ms p99 1.16 ms
```

注意这组数字是 **3 GiB 热文件的页缓存命中**，不是设备速度。95 GiB 的表配 64 GiB 内存，命中率
大概 2/3 封顶，冷的部分要付设备延迟 —— **这个还没测**，它是决定"值不值得迁"的关键数字。

### 2.4 反过来说，这块盘已经被证明能扛住这个负载
WSL 里的 `/` 是 `/dev/sdd` 1007 GiB（用 803 GiB），而 `D:\wsl\ext4.vhdx`（834 GiB 实占 / 896 GiB 稀疏）
就在**同一块 EXCERIA G2** 上。也就是说 pass3 那组 **131 tok/s / 305 tok/s**（本机、WSL2 原生、
PLE 走 `O_DIRECT`+AIO）是穿过 `ext4 → vhdx → NTFS → NVMe` 这么一层更深的栈跑出来的。原生 Windows
buffered 读只少掉 vhdx 这一层，不可能更差。**盘不是阻塞点**，剩下的风险全在软件侧。
（注意：pass2 的 110.99 / 667.13 是**参考机 + SATA SSD** 上测的，不能拿来为本机的盘背书。）

`_dev/probe/_ple_coldread_probe.py` 的冷读实测（先写一个 100 GiB 的 `FILE_FLAG_WRITE_THROUGH` 文件，
让它根本不进文件缓存，再随机读）：

```
4096 B 随机读 threads 1 :  13,392 reads/s   p50 0.063 ms  p90 0.081 ms  p99 0.115 ms
```

这一行是全场唯一可信的"冷"数字：63 µs 正是 NVMe 单队列随机读的延迟，13.4k IOPS 也对得上。
后面 4/8/16/32 线程冲到 15–16 万 reads/s、p50 掉到 7–20 µs，那是 Windows 缓存管理器把读扩展成大
extent 后命中了自己的 standby list —— **同一个文件被同一个脚本反复读，第二轮起就不是冷读**；
320 B 那一组（p50 3 µs）整组都被上一轮 4096 B 的读预热过，只能当污染样本看。要把并发测准，
得每个配置前重新 `--make` 一次（冷文件靠重写，不靠重读），一次 115 s。

即便如此结论已经够用：QD1 冷读 ~13k IOPS / 63 µs；解码每步最多 `batch×16` 行（c16 = 256 行），
一个 2048-token prefill 块 `2048×16 = 32,768` 行 ≈ 2,560 个 4 KiB 页 ≈ 10 MiB 设备工作量；
16 个读线程 + 补丁自带的 512 MiB 行缓存（≈110 万行）足以把它压到毫秒级。

## 3. 其它会踩到的点

- **容量**：64 GiB 卡上 47.32 GiB 权重 + 2.37 GiB 图 + 视觉塔，`--gpu-memory-utilization 0.96` 要 61.4 GiB。
  本项目的实测经验是 WDDM 还有约 1.4 GiB 驱动保留 + 3 GiB 不敢碰的顶部区域，
  真实可用约 60 GiB → KV 只剩 8–10 GiB，262,144 上下文多半保不住。
  `start_server.ps1` 那套 fit 公式是按 27B 的 ~78 KB/token 标定的，这个混合模型要重新标定。
- **TP=2 不是出路**：本 rig 上从没跑过 NCCL；而且 `Qwen4ExpPLESSDEmbedding.__init__` 明确拒绝
  `TP!=1 or DP!=1`。
- **GPU0 被 WSL 容器占着**（62.1 GiB）。Windows 侧只能上 GPU1，或者先把容器停掉。
- **数据**：143 GiB 模型现在只在 WSL ext4（`/dev/sdd`）里。`dd` 实测 WSL→`/mnt/d` 稳定 **231 MB/s**
  → 全程约 11 min，可行。但 `D:\wsl\ext4.vhdx` 和模型目标目录在同一块 NVMe 上，拷贝会和正在跑的
  WSL 服务抢盘 —— 那个服务恰好是 SSD-delivery-bound（8K prefill 有 36% 时间在等 PLE）。
  D: 现在剩 480 GiB，放完模型剩约 337 GiB，还要给 vhdx 留增长空间。
  WSL 侧只看得到 62 GiB 内存 / 16 GiB swap，页缓存装不下这张表，那边才要靠 O_DIRECT + 应用层行缓存。
- **spec decode 不一样**：WSL 用 MTP=1；本仓库的收益来自 DFlash2，而 Flash-Next 没有 DFlash2 drafter。
  MTP 路径依赖补丁里"checkpoint 层 0 量化项重映射到运行时层 48"。
- **可能能沿用的本地优化**：head_dim=256 → `prefill_attn_hd256.py` 相关；int8 Marlin 激活；
  GDN 的 FLA/mamba 补丁。但它们全是照 v0.29.0 的布局写的，换树就得重新对一遍。
- **没验证过的部分**：QSA 稀疏注意力、`hc_count=4` 超连接、INC 的 2/3bit `RoutedExperts` MoE 在
  sm_80 + Windows 上到底跑不跑得起来（`low_latency_gemm.py` 明确 gate 在 `(10,3)` 能力上，sm_80 走不到它）。

## 4. 建议的推进顺序（先花小钱证伪）

1. **先测冷盘随机读** —— 已测，见 §2.4：QD1 冷读 13k IOPS / p50 63 µs，而且这块盘正在跑同一个负载
   （WSL 容器）。一票否决的风险已经排除；还缺的是**并发**冷读数，办法是每个配置前重建冷文件。
2. **确认模型能不能进 Windows**：已完成。143 GiB 已落在
   `D:\models\Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`（元数据 sha256 全 MATCH，PLE shard 头尾 1 GiB
   MATCH，header offset 与数据区严格相等；校验见 `_dev/probe/_copy_verify.sh`）。
   结构烟测**已经跑过，结论见 §6**：现有构建连层都构造不出来（QSA 在 fork 树里不存在），
   所以这一条的答案是“否”，且原因不是 Windows。
3. 第 2 条的“便宜证伪”已经付过钱了，而且它否掉的是另一件事：烟测证明**路线 A 是前提而不是选项**。
   §7 用实测把这笔大钱拆成了可数的 16 个冲突文件。
   落地点：`git worktree add D:\code\vllm-windows origin/main` → `git apply qwen38-ple-ssd.patch`
   → 把 win32 依赖分支和 build 修复 forward-port 过去 → 再把 `ple_ssd.py` 的 reader 换成
   Windows buffered 线程实现（Linux AIO 那条 `_native` 路径直接不要）。
4. 起服参数按 60 GiB 可用重算：`--gpu-memory-utilization`、`--max-model-len`、graph capture 上限都要
   重标，不能照抄 WSL 的 0.96 / 262144。

## 5. 本轮留下的工件

- 分支 `wip/qwen38-flash-next`：两个 commit，`e0cdee4a68`（27B 优化，40 文件 / +1937）和
  `62c004312d`（`_dev` 脚手架，232 文件）。tag `pre-flash-next-20260927` → `62c004312d`。
  **两个 commit 都不含本轮新建的文件**；本轮的 `_dev/probe/_ple_winio_probe.py` 和
  `_dev/out/wsl_flash_next/` 保持未跟踪，`git status` 里能看到。
- `_dev/probe/_ple_coldread_probe.py`：造冷文件（write-through）+ buffered 随机读的 IOPS/延迟。
  ⚠ 同一个文件重复测第二轮就不冷了，要换新文件才有冷读意义。
- `_dev/probe/_ple_winio_probe.py`：Linux-only 调用清单 + 页扩展取行的正确性对比 +
  同步多线程/IOCP 两种读法的 IOPS 与延迟。重跑：
  `D:\code\vllm-windows\.venv\Scripts\python.exe D:\code\vllm-windows\_dev\probe\_ple_winio_probe.py`
  （`--file` 指到大于内存的文件即可测冷盘）。
- `_dev/out/wsl_flash_next/`：从容器里取出的 `ple_ssd.py` / `ple_ssd_io.c` / `ngram_embedding.py` /
  `model_state.py` / `mtp.py`，以及 `qwen38-ple-ssd.patch` 的本地副本。
- `D:\code\vllm-windows`：`origin/main` 的临时 worktree（128 MB），补丁在此干净通过，
  是第 4 节的落地位置。不需要时 `git worktree remove D:\code\vllm-windows`。
- `_dev/probe/_flashnext_trim.py`：造“PLE-free 视图”（hardlink + 改 `index.json` /
  `ple_layer_ids: []`），输出到 `D:\models\Qwen3.8-Flash-Next-struct`。
  视图已于 2026-10-03 删除（删的全是硬链接名字，AR 的 142.56 GiB 数据一字未动，只回收 48.8 MiB 独有元数据）；
  不可再生的两份 index（227,565 / 227,574 keys）与 compact `config.json` 存档在 `_dev/out/struct_view_meta/`。
  ⚠ 该脚本只按 `model-00001-of-00011.safetensors` 这个名字跳过 PLE 分片，若 PLE 仍在 rsync 传输中就会被当普通文件链进视图（沿用 rsync 的隐藏临时名），重建视图前先修这里。
- `_dev/bin/_flashnext_struct_serve.ps1`：烟测专用的起服脚本（GPU1 / eager / `--language-model-only`，
  `-DryRun`、`-Graphs`、`-WithVision`、`-Mtp` 四个开关）。日志到 `_dev/out/logs/flashnext_*.log`。
- `_dev/test/_flashnext_smoke.py`：`/health` 轮询 + 短/长 prefill + 两路并发。
- `_dev/probe/_copy_verify.sh`：拷贝完整性（元数据 sha256、所有 shard 尺寸、PLE shard 头尾 1 GiB）。
  从 WSL 跑：`wsl.exe -d Ubuntu -- bash /mnt/d/code/vllm-windows/_dev/probe/_copy_verify.sh`
  （git-bash 下记得 `MSYS_NO_PATHCONV=1`）。
- `_dev/out/wsl_flash_next/probe_inc_moe.patch`：给 site-packages 打的 probe-only 两处 hunk，
  原件在 `_dev/out/installed_backups/inc/`。回滚：
  `Copy-Item _dev\out\installed_backups\inc\config_parser.py.orig <site-packages>\inc\config_parser.py -Force`
  （ schemes 同理）。不设 `VLLM_PROBE_INC_HUMMING_LOWBIT` 时这些 hunk 不生效。
- `_dev/out/wsl_flash_next/winfixes/*.patch`：fork 的 5 个 Windows commit 的 `format-patch` 副本，
  §7 的计数就是从这里测出来的。
- `_dev/out/wsl_flash_next/container_{quant_humming,inc,inc_wna16_scheme,factory}.py`：容器
  `a5a30471f` 的 INC/humming 参考件（注意 `container_humming.py` 其实是 `vllm/utils/humming.py`）。

## 6. 结构烟测的实测结果（2026-09-27 19:39 – 19:51）

拷贝与视图：
- phase1（除 PLE shard 外的 47.6 GiB）rc=0 @19:38:45，phase2（95.37 GiB PLE shard）rc=0 @19:51:56，
  实测 ~160 MB/s（比 `dd` 的 231 MB/s 低，因为容器正在同一块 NVMe 上服务同一个模型）。
- `D:\models\Qwen3.8-Flash-Next-struct`：`_dev/probe/_flashnext_trim.py` 造的“PLE-free 视图”
  （hardlink 34 个文件、`index.json` 去掉 128 个 PLE key、`config.json` 里 `ple_layer_ids: []`）。

第一次 19:39（GPU1，enforce-eager，`--language-model-only`，4096 ctx）：

```
ValueError: MoeWNA16Method only supports int4 and int8 now.
```

这不是 Windows 的问题，是 INC 的两处缺口：
1. `inc/config_parser.py` 只把类名里含 `fusedmoe` 的层当 MoE，而 qwen4_exp 用
   `FusedMoEFactory` → `RoutedExperts`，类名里没有 `fusedmoe` → 专家的 per-layer config 挂不上 →
   `get_moe_method` 落到 `MoeWNA16Method`（只认 int4/int8）→ 抛错。补丁里带这个 hunk。
2. `origin/main`（和容器的基线 `a5a30471f`）的 `inc/schemes/inc_wna16_scheme.py` 有
   `CUDA_HUMMING_SUPPORTED_BITS = {2, 3, 5, 6, 7}`，CUDA 上的 2/3-bit 直接走 humming；
   fork（`13e844c86d`）完全没有这段路由，只能掉进 `_resolve_gptq_moe`。

为了把验证成本压到近零，我给 site-packages 打了 probe-only 的这两处 hunk
（`_dev/out/wsl_flash_next/probe_inc_moe.patch`，原件备份在 `_dev/out/installed_backups/`），
并且全部 gate 在 `VLLM_PROBE_INC_HUMMING_LOWBIT=1` —— 不设这个变量时 27B 的解析路径一字不变。

第二次 19:49：

```
INFO [fused_humming_moe.py:118] Using indexed gemm for humming moe
INFO [humming_utils.py:744]    Using HummingIndexedExperts Humming MoE backend.
ERROR ValueError: Invalid layer_type qwen_sparse_attention
```

- Windows 侧 `humming 0.1.15`（SystemPanic 的 Windows 端口）**能把 2/3-bit MoE 的 method 建起来**
  （容器里是 upstream `humming 0.1.12`）；kernel 真跑起来与否仍未证。
- 然后死在第 3 层：`git grep qwen_sparse_attention 13e844c86d` → **0 处**，`origin/main` 里有。
  fork 的 `vllm/models/qwen4_exp/` 与 `origin/main` 相差 23 文件 / +3489 −1758，`origin/main` 还多出
  `ngram_embedding.py`、`ops/ple.py`、`ops/qsa_indexer.py`。48 层里 12 层的 QSA 在现有构建上
  根本构造不出来 → **路线 A 是前提，不是选项**。
- 烟测只证明了 layer 0–2（`linear_attention` + MoE）能构造；GDN 全 36 层、QSA、hc_count=4 的
  前向、CUDA graph、marlin 2-bit 全部仍未验证。
- “用 `--load-format dummy` 更便宜地试结构”这条路也否掉了：专家参数 ≈ 512 × 3 × 640 × 2560 × 48
  ≈ 120 B，bf16 dummy 权重 ≈ 240 GiB，64 GiB 的卡放不下。

## 7. 路线 A 的工作量（实测计数，不是估计）

把 fork 的 5 个 Windows commit（`3d3a44e4a8`、`b137929e44`、`160f6b2b76`、`4eebec3745`、
`13e844c86d`）逐文件 `git apply --check` 到 `origin/main + qwen38-ple-ssd.patch` 上：
**65 个文件干净通过，16 个文件冲突**。要手工的 16 个：

| 文件 | 来自 | 性质 |
|---|---|---|
| `requirements/cuda.txt` | 3 个 commit 都撞 | win32 依赖 pin，最需要想清楚的一处 |
| `requirements/common.txt` | 3d3a44e4a8 | 同上 |
| `CMakeLists.txt` | 3d3a44e4a8 | MSVC/build 开关 |
| `cmake/external_projects/deepgemm.cmake` | 3d3a44e4a8 | 外部项目开关 |
| `cmake/external_projects/flashkda.cmake` | 13e844c86d | 同上 |
| `README.md` | 3d3a44e4a8 | 文档噪声 |
| `csrc/fs_io.cpp` | 4eebec3745 | win32 文件 I/O |
| `csrc/libtorch_stable/moe/marlin_moe_wna16/ops.cu` | 3d3a44e4a8 | Windows 编译修复 |
| `csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu` | 3d3a44e4a8 | 同上 |
| `csrc/libtorch_stable/quantization/marlin/marlin.cu` | 3d3a44e4a8 | 同上 |
| `csrc/libtorch_stable/activation_kernels.cu` | 13e844c86d | 同上 |
| `vllm/distributed/device_communicators/shm_broadcast.py` | 13e844c86d | win32 共享内存/信号 |
| `vllm/entrypoints/cli/serve.py` | 3d3a44e4a8 | 入口噪声，好解 |
| `vllm/entrypoints/grpc_server.py` | 3d3a44e4a8 | 同上 |
| `vllm/model_executor/warmup/kernel_warmup.py` | 4eebec3745 | win32 warmup |
| `vllm/v1/attention/backends/flashinfer.py` | 13e844c86d | flashinfer-windows 适配 |

模型代码（qwen4_exp / GDN / QSA / PLE / MTP）不需要移植 —— 它们本来就在 `origin/main` 里，
升级是把它们**拿过来**，真正要搬的只有上面这 16 处 win32 适配。

---

## 8. 2026-10-03：显存中毒的反事实、视觉默认与 KV/并发饱和（GPU1:9393/9394）

流水见 `.workbuddy/memory/2026-10-03.md`；这一节只留**能反复查阅的结论、数字和回滚路径**。

### 8.1 两种"满了"是两回事（最容易搞错的一条）

| | allocator 中毒 | KV 占用 100% |
|---|---|---|
| 看到的信号 | `free` 掉到 1xx MiB 且**不回弹**、短请求 TTFT 3–4 倍、引擎自报 prefill 掉到几百～一千 tok/s | `vllm:kv_cache_usage_perc = 1.0`、`Waiting: N`、`num_preemptions_total` 上升 |
| 归属 | torch/CUDA caching allocator（cached 块不还给驱动） | scheduler（块不够） |
| 处理 | `torch.cuda.empty_cache()` 主动回收（本项目做成后台线程） | 引擎自己排队 + 抢占重算，请求结束自动回 0 |
| 会不会不可恢复 | **会**：空转 90 s 不自愈，继续压会加深（155 → 103 MiB），196K 直接跑不完 | 不会 |

判据用 free + 短请求 TTFT + 引擎自报 prefill 吞吐，**不要用 nvidia-smi 的百分比**：`--gpu-memory-utilization 0.94` 本来就是"把自动 KV 灌到份额顶"，显示 92–98% 是设计如此。

### 8.2 healer：反事实是决定性的

同一端口、同一配置、同一条 131K+图 请求，只换 `-NoAllocHeal`：

| | healer 开 | healer 关 |
|---|---|---|
| 该次 prefill | 28.4 s | **50.5 s** |
| 收尾 free | 1403 MiB（触发 1 次：213 MiB / cached 3045 MiB） | **155 MiB，卡死；空转 90 s 不自愈** |
| 之后 8K/1-tok ×3 | 1.93 / 1.78 / 1.81 s | **6.65 / 6.55 / 6.54 s**（= 中毒原签名 6.96/7.07/7.05） |
| 再压 196K | 62.9 s 完成 | **>400 s 未完成**，free → 103 MiB，卡用满 64,621/64,724 MiB |
| 引擎自报 prefill | 13,000–19,900 tok/s | **818 → 1,636 tok/s** |

实现：`vllm/utils/mem_utils.py::start_alloc_heal_thread()`，由 `vllm/v1/engine/core.py` 在 `init engine (profile, create kv cache, warmup model) took ...` 之后启动（只在 engine core 进程、图捕获之后，避免抢捕获和让 API server 白建 CUDA context）。
阈值：`QSA_ALLOC_HEAL=1`、`QSA_ALLOC_HEAL_FREE_MB=384`、`QSA_ALLOC_HEAL_CACHED_MB=256`、`QSA_ALLOC_HEAL_COOLDOWN_S=1.0`，150 ms 轮询；语义直接照抄 WSL 侧已验证的那套。启动日志 `[mem_utils.py:75] alloc-heal thread started`，每次回收 `[mem_utils.py:65] alloc-heal fired #N`。

### 8.3 视觉：默认开 + `max_pixels` 上限是必须项

- 本 checkpoint `patch_size=16`、`merge=2` ⇒ factor 32 ⇒ **1 vision token = 32×32 px**；`max_pixels 1310720` ⇒ 上限 ≈**1280 tok/图**。
- checkpoint 默认 `size.longest_edge = 16777216`（4096×4096）⇒ **16384 tok/图**，而 `--limit-mm-per-prompt {"image":{"count":4}}` 允许 4 张 ⇒ 没有上限时 4 张图能吃掉 65k 上下文。**所以默认开视觉就必须带 max_pixels。**
- 账（vision + 上限）：weights+非torch 48.28、峰值激活 **0.58**、CUDAGraph 0.17 ⇒ **KV 10.93 GiB = 386,698 tok，262,144 并发 1.48x**，idle free 3213 MiB。
  开视觉相比纯文本只少 0.81 GiB KV；上限还把 profile 峰值激活从 2.07 GiB 压回 0.58 GiB（无上限那次 KV 只剩 9.45 GiB）。
- 验收（全部单流 + 4 并发，`/tokenize` 精确配额）：1024px 单图 1024 tok / 0.9 s；3×2048px 压后 prompt 3748 tok（不压 = 12288）且颜色形状全对；4 张 4975 tok 全对；5 张干净拒绝；4 并发带图各 1260 tok、4/4 正确；**满信封 261,972 tok / 57.7 s**；131K 预算 + 4 图 = 129,972 tok / 29 s；196K 两轮 69.10 / 62.85 s（视觉关 63.41/64.19 ⇒ 长上下文无惩罚）。
- 已知副作用：`Draft model Qwen4ExpMTP does not support external multimodal embeddings` —— 带图请求的 MTP 草稿只吃文本，投机收益打折；正确性不受影响。
- 开关：外层 `start_qwen38_flash_next.ps1` 默认开视觉，`-NoVision` 回纯文本，`-MaxPixels 0` 放开到 checkpoint 上限。**内层 `_flashnext_struct_serve.ps1` 与测量臂（`_serve_bg.ps1`）默认仍是纯文本**，历史 benchmark 口径没变。

### 8.4 并发与 KV：这版引擎把 prefill 串行化

- 4 条并发长 prefill **灌不满 KV**：4×95,200 → usage 峰值 0.368；4×130,300（超容量 1.36x）→ 0.451；抢占 0；TTFT 阶梯 21.6/41.7/61.7/80.6 s。原因是 `--max-num-batched-tokens 2048`（MTP 还把 `max_num_scheduled_tokens` 压到 2048）使同时只驻留 1–2 条。
- 要真占满必须**用 `min_tokens` 把块占住**（模型默认答 8 个 token 就 EOS 立刻释放）：2×191,200 + `min_tokens 6500` → usage **1.000**、抢占 +2、中途 canary 1.9 s → 54.9 s；再挤一条 130,000（需求 1.33x）→ usage 仍 1.000、抢占 +2、canary 98.0 s；带图 2×190,055 → usage 1.000、抢占 +3、带图 canary 67.2 s 仍正确、healer 零触发。
- 多轮：prompt 59,331→119,664→179,999→181,041，延迟 32.6→34.1→35.6→**85.3 s**；**最后一轮只加 28 tok 也要 85 s = 整段历史重算**；`Prefix cache hit rate` 仅 **4.5–8.8%**（健康多轮应 90%+）。**"多轮越打越慢"是前缀缓存在这个 hybrid 模型上不生效，与显存无关**；下一步该查 `--mamba-cache-mode align` × `enable_prefix_caching=True`。

### 8.5 防重入（因为真的撑爆过一次）

`_start_vllm_service.ps1` 在 `Start-Process` **之前**检查：①目标端口有 TCP 监听 → 拒绝；②`-GuardGpu` 那张卡 `memory.used ≥ 4096 MiB` → 拒绝并列 compute-app pid（列不出来说明持有者可能在 WSL）。命中即什么都不启动；`-SkipGuard` 逃生。`start_qwen38_27b.ps1` 走同一个 wrapper，自动受保护；`_dev/bin/_serve_bg.ps1` 自己 `Start-Process`，**还没有保护**。
事故复盘：07:47:39 GPU1 上已有 engine 在加载（**端口未绑，`/health` 探不到**），07:53:36 我又起了第二个 → 一张卡两个 engine。**起服前必须读 nvidia-smi，不能用 /health 判断有没有人在跑。**

### 8.6 回滚

- 撤 healer：`git checkout vllm/utils/mem_utils.py vllm/v1/engine/core.py`，或把 `_dev/out/installed_backups/alloc_heal/*.orig` 拷回 `.venv-flashnext/Lib/site-packages/vllm/`，然后重启。
- 撤视觉默认：起服时传 `-NoVision`（代码默认是开）。

### 8.7 MemUtil 扫描：KV 最多加到哪一档（2026-10-03，GPU1:9394，vision 默认 +
max_pixels=1310720，MTP x2，每档全新启动）

| 档位 | KV | KV tokens | 并发@262K | 空闲 free | 8K 基线 | 196K TTFT | 196K 后 8K | heal 次数 | 最低 free |
|-------|------|-----------|-----------|-----------|----------|-----------|------------|-----------|-----------|
| 0.94 | 10.93 GiB | 386,698 | 1.48x | 3,213 MiB | 1.94–2.31 s | 62.9 / 69.1 / 74.9 s | 1.78–2.03 s | 18 | 111 MiB |
| 0.96 | 12.20 GiB | 431,596 | 1.65x | 1,913 MiB | 2.08–2.32 s | 70.1 s | 1.98–2.26 s | 20 | 105 MiB |
| 0.97 | 12.84 GiB | 453,320 | 1.73x | 1,285 MiB | 2.20–2.41 s | 75.8 s | 1.84–2.33 s | 23 | **0 MiB**（瞬时） |
| 0.98 | 起不来 | — | — | — | — | — | — | — | — |

0.98 的报错原文（启动阶段就被拒，不是中毒）：
`ValueError: Free memory on device cuda:0 (62.21/63.61 GiB) on startup is less than
desired GPU memory utilization (0.98, 62.34 GiB).`
即硬上限 = 62.21 / 63.61 = **0.978**，**0.97 是能起来的最后一档**。
换算：每 +0.01 ≈ +0.64 GiB ≈ +22.6k tokens（+5% 容量）。

结论三条：

1. **中毒不由 KV 大小决定，由缓存块能否归还决定**。无 healer 时 0.94 已经实测中毒
   （8K 6.5–6.8 s、196K >400 s 跑不完）；有 healer 时 0.94 / 0.96 / 0.97 都不中毒
   （事后 8K 全部回到 1.8–2.4 s）。所以"加到多少会中毒"在无 healer 的世界里是
   "任何一档都会"，在有 healer 的世界里是"到分配器再也还不出东西为止"。
2. **往上加是拿余量换容量，0.96 花的不是速度是余量**：空闲 free 3,213 → 1,913 MiB，
   heal 强度 18（最低 111 MiB）→ 20（最低 105 MiB），换来 +44,898 tok（+11.6%，
   并发 1.48x → 1.65x）。而 0.94 自己三次启动的 196K 相差 12 s（19% 跨度），比
   0.96 与 0.94 之差还大——**当前分辨率分不出 0.96 变慢**；超出噪声的是 0.97
   （75.8 s 顶到 0.94 最差端、262K 期间 canary 从 1.9 s 被拖到 82.9 s、瞬时 free 归零）。
   判断有无退化应盯 **heal 触发次数与最低 free**，不是 TTFT（要分辨 2–5% 的 TTFT 差，
   需同协议各跑 9 个 8K）。
3. **0.97 的深层压力开始变形**：262K 单发 123.1 s，canary 868 tok 被拖 82.9 s，而
   kv_usage 峰值只有 0.59、0 抢占——即**不是 KV 满，而是分配/调度被挤**。慢的原因仍
   是假设（分配器无 free 可用 → 长 prefill 里频繁 empty_cache/cudaFree 卡流），要坐实
   需要 step 级时间分解。注意 262K 的两个数不是同一把尺子：0.94 的 57.7 s 来自 vision
   probe（含 1,024 tok 解码），0.97 的 123.1 s 来自饱和 probe（out=16），所以"慢一倍"只是线索。

**服务默认改为 0.96**（`start_qwen38_flash_next.ps1`）：0.96 相对 0.94 在速度上分辨不出，
却多出 11.6% 的 KV 容量，且仍留 1.9 GiB 空闲余量；0.97 只在确实需要多 66k tokens 时开，
且 healer 必须开。内层 `_dev/bin/_flashnext_struct_serve.ps1` 的默认也跟着改成 0.96，
因为 `_serve_bg.ps1` 根本不传 `-MemUtil`——测量臂因此与服务共用同一上限（旧日志里只剩
两份还带着 0.94，可比性损失很小）。**没跟着改的 0.9x 都属于别的模型**：`_dev/bench/_ab_serve_win.sh`
与 `_bench_27b.py` 是 27B-W4A16 那套（max-len 60,928、KV 钉死 6 GiB、激活余量不同），
`_dev/bin/_run.ps1` 按**空闲**显存推导比例（必须适配 free 而非总量），上游 CI 脚本仍是 0.9。

另：手设 `-KvGiB` 不要超过 **12.8 GiB**，它绕过同一套上限校验，只会更早撞墙。

- ⚠ **服务跑的是安装树**（`.flashnext-root/.venv/Lib/site-packages/vllm` 是 junction → `.venv-flashnext/Lib/site-packages/vllm`），不是仓库 `vllm/`。改完仓库文件必须拷进安装树；**重装/重建 vllm 后这两份要重新拷**。

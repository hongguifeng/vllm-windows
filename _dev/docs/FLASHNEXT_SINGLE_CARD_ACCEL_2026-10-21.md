# 单卡 AutoRound 3bpw 提速实验（移植 ElectroHobo W4A16 配方）2026-10-21

目标：把 `ElectroHobo/qwen38-flash-next-w4a16-cmp170hx-recipe` 的改进搬来加速
**单卡** `Qwen3.8-Flash-Next-AutoRound-3bpw-MTP`。P1（PLE 表进主机内存）因本机
RAM 只有 87.9 GiB、表是 BF16 95.37 GiB 而暂缓，本轮做 P2–P5。

测量口径：`_dev/probe/_tpot_bench.py`（非流式、token 数取自 `usage`、
spec 计数取自 `/metrics` 增量），每次 200/128 token、temperature 0、ignore_eos。
基线是生产 `:9393`（MTP-3 + adaptive_k 1–3，默认 block）。

## 0. 启动器开关（2026-10-21 晚，多实例 + DFlash）

`start_qwen38_flash_next.ps1` 与 `_dev/bin/_flashnext_struct_serve.ps1` 新增：

| 开关 | 默认 | 作用 |
| --- | --- | --- |
| `-Spec auto\|mtp\|dflash\|none` | `auto`（跟随 `-Mtp`，即生产行为不变） | 选投机方法；`dflash` 用外部 DFlash 2 drafter |
| `-Drafter <path>` | `D:\models\Qwen3.8-Flash-Next-DFlash2` | drafter 目录 |
| `-DflashDrafts <n>` | `7` | dflash 每步草稿数（忽略 `-MtpTokens`/`-AdaptiveK`） |
| `-KvOffloadDir <path>` | 空 = `D:\vllm-offload\<Port>` | **每个实例自己的 tier 目录** |

多实例 KV offload：tier 的 region 文件挂在 `VLLM_OFFLOAD_REGION_DIR` 下，启动时只清理
**自己目录**里的陈旧 region；不同端口的实例各占一个目录，互不可见也不可能互删。实测两个
实例同时开 16 GiB tier：`D:\vllm-offload\`（旧默认，生产在跑）与
`D:\vllm-offload\9402\`，各 15.98 GB，两个服务同时 `health=200`。想共享一份
content-addressed 缓存时显式传同一个 `-KvOffloadDir` 即可。

两个踩过的坑（已修）：

1. **`$spec` 与参数 `$Spec` 在 PowerShell 里是同一个变量**（大小写不敏感），赋值被
   `ValidateSet` 拦掉，导致命令行里出现 `--speculative-config dflash` 被 python 拒收。
   局部变量改名 `$specJson`。
2. **DFlash 会触发 flashinfer JIT**（`run_ninja`），而生产启动路径没把 MSVC/ninja 放进
   PATH，engine 以 `FileNotFoundError [WinError 2]` 退出。前端脚本现在 source
   `_dev/bin/_env.ps1` 并设 `VSLANG=1033`（否则中文编译器日志会在 JIT 里触发
   `UnicodeDecodeError`），同时把 venv 的 `Scripts` 加进 PATH。

---

## 1. 基线（更正之前的口径错误）

`_dev/probe/_bench_port.py` 数的是流式 chunk，不是 token，之前报的 42/108 tok/s 偏低。

| 臂 | c1 | c4 聚合 | c1 步时 | c4 步时 | 接受/步 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 生产 MTP-3 + adaptive_k | **106.8** | **287.3** | 23.1 ms | 34.9 ms | 1.49 / 2.09 |
| 冷 prefill（4096 tok，唯一 prompt） | 1.26 s ≈ **3,250 tok/s** | — | — | — | — |

## 2. P2｜DFlash 2 草稿模型

移植内容（与配方同源，48 行）：

| 文件 | 改动 |
| --- | --- |
| `vllm/models/qwen4_exp/nvidia/model.py` | `Qwen4ExpModel` 加 `EagleModelMixin`；新增 `_aux_hidden_state()`（层前状态、四路超连接平均）；forward 收集 `aux_hidden_state_layers` 并在有 drafter 时返回 `(hidden, aux)`；两个 LM 类加 `SupportsEagle3` |
| `vllm/model_executor/models/config.py` | 投机方法白名单加 `dflash` |
| `vllm/v1/worker/gpu/model_runner.py` | `get_mtp_target_hidden_states()` 返回 None 时不再切片（两处）；否则 DFlash 在 profiling 阶段崩 `'NoneType' object is not subscriptable` |

drafter：`D:\models\Qwen3.8-Flash-Next-DFlash2`（incoai，revision `a22a1744`，1150 MiB，
`target_layer_ids [5,15,25,35,45]` → 引擎取 6/16/26/36/46，日志已确认）。

结果（单卡）。**注意：早先一轮 DFlash/MTP 对比是在 GPU2 还挂着试验实例时测的，MTP 被压低
30–70%，已作废**；下表是两臂各自独占机器重测的干净数字（真实文本用仓内代码文件与中文文档）：

| 负载 | MTP-3 c1 | DFlash-7 c1 | MTP-3 c4 | DFlash-7 c4 |
| --- | ---: | ---: | ---: | ---: |
| 代码（英文 Python） | 134.7 | **157.6（+17%）** | **346.0** | 293.3（−15%） |
| 中文散文 | **105.9** | 86.3（−19%） | **249.1** | 169.7（−32%） |
| 重复填充文本 | 112.9 | 111（持平） | 317.6 | — |

接受率：DFlash 代码 c1 47.7%、中文散文 c1 仅 **19.2%**；MTP 同一负载 69.9% / 51.9%。
DFlash 每步产出更多 token（代码 c1 3.34 接受/步、4.27 tok/步），但 7 个草稿的验证代价在
多并发和低接受率负载上吃掉收益。

**结论：DFlash 只在「单用户 + 代码/英文」这一格里快 17%，其余全输给 MTP**，所以
**生产保持 MTP**；DFlash 留在 `_flashnext_arm.ps1 -Spec dflash` 后面当可选臂（例如单独占
一张卡服务代码场景）。它对我们这套的价值不如配方宣称的那么大，原因写在 P2 节前面的
假设里：配方是 TP2 + x16 P2P，宽验证几乎免费；单卡上验证 8 个 token 要真金白银。

## 2.5 为什么 DFlash 在我们这里没有配方/别的项目那种优势

三轮排查，每次只改一个变量、一次只跑一个实例：

**① 捕捉点不是原因。** inco.ai 的 SGLang 补丁直接对“延迟合并”的原始张量按 HC 平均；配方
版本先用 `mlp_hyper_connection.combine` 落定再平均。加 `VLLM_DFLASH_AUX_PRECOMBINE=1`
切换对比（代码文本、c1、7 草稿）：completion 35.1% vs 34.7%、chat 37.8% vs 39.6% —— 无差异。

**② 引擎侧不缺实现。** `DFlash2Qwen3ForCausalLM`、`candidate_selector`（含 triton
采样核）、`conv_group_size`/`conv_kernel_size`/`block_size`、`selector_top_k`（从
`draft_config` 读）都在，drafter 权重无缺失。

**③ 差在同草稿数下的接受率**（代码文本、c1、两臂独占机器）：

| 配置 | 接受率 | 接受/步 | tok/步 | 步时 | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| MTP-3（生产） | **69.9%** | 2.08 | 3.08 | 22.8 ms | **134.7** |
| DFlash-3 | 60.6% | 1.82 | 2.82 | 23.9 ms | 117.7 |
| DFlash-7 | 35.1% | 2.46 | 3.39 | 27.1 ms | 124.7 |
| DFlash-7 + chat 输入 | 37.8% | 2.64 | 3.57 | ~27 ms | 130.4 |

结论：**每步产出并不差（3.39 vs 3.08）**，输在两点：① 同宽度下接受率比 MTP 低 10 个点；
② 单卡上宽验证更贵（7 草稿步时 27.1 vs 22.8 ms）。配方的 43–80% 收益建立在 TP2 +
PCIe x16 + P2P 上——宽验证在他们那儿几乎免费（他们 README 把 x16 电容改写的 P2P 写在
前置条件里）；单卡没有 P2P，验证 8 个 token 要真金白银。

最可能的原因：**草稿头与 target 的配套关系**。MTP 头是随我们这份 Intel AutoRound 3bpw
checkpoint 一起打包的（对量化后的权重天然对齐）；DFlash2 是在 BF16 基座 / halt95 的
Merlin W4A16 权重上训练的，我们这里是另一套量化，隐藏态分布有偏移。若以后换 Merlin
权重或上双卡 P2P，值得重测。

## 2.6 测量纪律（吃过一次亏）

1. **两臂不能同时在线**：一个实例跑着时测另一个，MTP 的 c1 会被压低 25–70%。
2. **端口必须确认空闲再启动**：上一实例的 `--data-parallel-rpc-port` 没释放时，新实例会在
   `zmq` bind 上失败（`ZMQError: Address in use 30352`），而旧进程还在 `/health` 上应答 ——
   于是“新配置”测到的其实是旧进程，日志里也看不到 `Eagle3 auxiliary layers`。启动后必须
   核对：日志有 `Application startup complete` + `Eagle3 auxiliary layers`，且
   `num_draft_tokens/num_drafts == 配置的草稿数`。

## 3. P3｜草稿数按并发调度

`--speculative-config` 的 `num_speculative_tokens_per_batch_size` 在我们的引擎里已支持，
与 `adaptive_k` 互斥。实测生效：c1 走 7 个草稿、c4 走 5 个（日志/指标一致）。
对 DFlash 没有救回差距（见上）；对 MTP 留作后续 A/B 的开关。

## 4. P4｜便宜参数（本轮唯一明确的单卡提速）

| 配置 | c1 | c4 聚合 | c1 步时 | c4 步时 |
| --- | ---: | ---: | ---: | ---: |
| 生产：默认 block、batched 2048 | 106.8 | 287.3 | 23.1 ms | 34.9 ms |
| 固定 k=3、默认 block | 106.8 | 289.7 | 23.7 ms | 36.9 ms |
| **`--block-size 1680 --prefix-match-unit 16`**、固定 k=3 | **110.7** | **329.0** | 22.6 ms | 31.2 ms |
| 同上 + `--max-num-batched-tokens 8192` | 166.7* | 322.0 | 22.4 ms | 32.4 ms |

*该行 c1 的 166.7 来自一次 92% 接受率的重尾采样（填充 prompt 高度重复），不可复现，以步时为准。

- **`--block-size 1680 --prefix-match-unit 16` 是赢的**：c1 +3.7%、c4 **+13.7%**，步时
  34.9→31.2 ms。1680 可被 16 整除，也是 QSA ring（5–7 草稿时为 12）的合法块大小。
- `--max-num-batched-tokens 8192`（生产 2048、配方 3460）：步时无改善，冷 prefill 无改善
  （1.27 s vs 1.26 s @4096 tok）。spec 模式下引擎会把 `max_num_scheduled_tokens`
  钉在 spec 派生值上，所以这个旋钮作用有限。
- 冷 prefill 我们约 3,250 tok/s（不是之前误报的 8,041，那是前缀缓存命中）。

## 5. P5｜KV offload 到主机内存 tier（已完成 Windows 移植）

原实现用 POSIX 共享内存（`/dev/shm/vllm_offload_{id}.mmap` + `MADV_POPULATE_WRITE`

- `st_nlink` + “映射后立即 unlink”），Windows 上直接 `FileNotFoundError`。

改动只在 `vllm/v1/kv_offload/cpu/shared_offload_region.py`（其余 6 个文件无 POSIX 依赖）：

| 点 | Windows 处理 |
| --- | --- |
| 路径 | `%TEMP%\vllm_offload_{id}.mmap`（`tempfile.gettempdir()`） |
| 映射 | `mmap.mmap(fd, size)`（无 `MAP_SHARED`/`prot` 参数，句柄即共享） |
| 预触页 | 无 `madvise` → 直接用已有的逐页写回退路径 |
| 删除语义 | Windows 不能删已映射文件 → barrier 后保留，`cleanup()` 关映射后再删 |
| 陈旧文件 | 创建前先尝试删除同名文件（上一进程未清掉的大文件），删不掉就退回 joiner |
| 空间检查 | 改成对 `dirname(mmap_path)` 检查（tier 是真实磁盘文件，`%TEMP%` 得有空间） |

验证（`_dev/probe/_kv_offload_region_win_test.py`，CPU-only）：创建 32 KiB 区域、第二个句柄
join 后能看到第一个写的字节、写回双向可见、`cleanup()` 删除文件、陈旧文件被新一次创建替换 ——
`RESULT OK`。

端到端（GPU2，`--kv-transfer-config` 16 GiB tier，MTP+1680，`health=200`）：

```text
Created mmap file C:\Users\hong\AppData\Local\Temp\vllm_offload_<id>.mmap (17.16 GB)
vllm:kv_offload_total_bytes_total{transfer_type="GPU_to_CPU"} 6.92e8   # 692 MB 已下沉
vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"} 0.0      # 未触发回填
WARNING cudaHostRegister failed for rank=0 at 0.00 of 17.16 GB (code=2); the offload region stays pageable
```

- 功能通：带 tier 的服务正常回答（中文解释 KV cache offload，正确）。
- **回填路径已验证**（`_dev/probe/_kv_offload_reload_test.py`；GPU 池压到 36,764 token、
  tier 16 GiB、`max-model-len 32768`，三个 11.6k/19.8k/7.1k token 的不同 prompt 把池挤爆）：

  ```text
  tier after filling: GPU_to_CPU 2214.1 MB, CPU_to_GPU 0.0 MB
  pass 2（原 prompt 再问一次）: cached 11600 / 11612 token
  tier refill during pass 2: 507.4 MB        ← 命中来自主机 tier，不是重算
  RESULT OK (answer identical)
  ```

  驱逐下沉 2.2 GB、再次访问从 tier 取回 507 MB、`cached_tokens` 11600（99.9%），
  且贪心输出与驱逐前**逐字节相同**。
- **未 pin 住**：Windows 上 17 GB 的 `cudaHostRegister` 失败（code=2），区域保持可分页 →
  传输多一次拷贝，带宽不如 Linux pinned 版本。平台限制，不是移植缺陷（回填正确，只是慢些）。
- **残留文件**：torch 仍持有 buffer 引用，Windows 不能删已映射文件，所以每次停机都会在
  `%TEMP%` 留一份 `cpu_bytes_to_use` 大小的文件。由于每次 engine id 都不同，原先“同名清理”
  救不了它；本轮补了 sweep：创建时扫描 `vllm_offload_*.mmap` 逐个尝试删除，被别的进程映射的
  自然删不掉（安全）。单测已覆盖“外来陈旧文件被回收”。

结论：**P5 移植完成，驱逐与回填都验证过**，可作容量特性使用。上生产前只剩两个决策：
`KV_OFFLOAD_GB` 开多大（16 GiB 约合数十万 token 的额外容量，占 `%TEMP%` 所在盘），以及能否接受
可分页带来的传输开销（未 pin）。

## 6. 下一步建议

1. 先把 **P4 的 1680 + prefix-match-unit 16** 加进生产启动参数（需要停机重启一次，
   建议同时复测 adaptive_k 与固定 k=3，因为生产用的是 adaptive_k）。
2. P2 若还要救：先验证 aux 隐藏态与 inco.ai 训练输入的一致性，再考虑只在 c1/c2
   低并发启用 DFlash（c1 目前略胜 MTP）。
3. P5 可以做，但按“长上下文重载”价值单独评估。

## 7. 已落地到生产（2026-10-21 01:13 重启）

`adaptive_k` 与固定 k=3 在 1680 下一并复测：c1 **114.7 vs 110.7**、c4 324.3 vs 329.0，
固定 k 没有优势，因此生产保留 adaptive_k，只加两个新参数。

改动：`start_qwen38_flash_next.ps1` 新增 `-BlockSize`（默认 1680）与 `-PrefixMatchUnit`
（默认 16），透传到 `_dev/bin/_flashnext_struct_serve.ps1` 的两个新参数；
`-BlockSize 0 -PrefixMatchUnit 0` 可回到引擎自选值。

重启后实测（`health=200`，冷启动 242 s）：

| 指标 | 改前 | 改后 | 变化 |
| --- | ---: | ---: | ---: |
| c1 | 106.8 | **112.9** | +5.7% |
| c4 聚合 | 287.3 | **317.6** | +10.6% |
| c4 单请求延迟 | 1.61 s | 1.56 s | −3% |
| KV 池 | 419,430 | 415,306 | −1%（块粒度） |
| 贪心输出 | 51 token | 51 token，逐字节相同 | — |

原始日志：`_dev/out/logs/qwen38-flash-next-win_20261021_011356.stdout.log`。

### 7.1 生产跑的是快照，不是仓库树（踩过两次）

生产服务用的是 `.flashnext-root\.venv\Lib\site-packages\vllm`（2026-10-01 装进去的拷贝），
改仓库 `vllm/` 下的文件对它**没有影响**。第一次带 tier 重启就是这样失败的：

```text
FileNotFoundError: [Errno 2] No such file or directory: '/dev/shm/vllm_offload_...mmap'
```

所以凡是要让生产生效的 `vllm/` 改动，都得同步到快照。工具：
`_dev\bin\_sync_snapshot_overlay.ps1`

```powershell
& _dev\bin\_sync_snapshot_overlay.ps1 -DryRun          # 看哪些文件会变
& _dev\bin\_sync_snapshot_overlay.ps1                  # 默认 core 组（KV tier 的 region）
& _dev\bin\_sync_snapshot_overlay.ps1 -Groups all      # core + dflash + tp2
& _dev\bin\_sync_snapshot_overlay.ps1 -Restore         # 回滚到上一份备份
```

- 按“改动动机”分组：`core`（KV tier region）、`dflash`（Eagle3 捕获 + 白名单 + runner guard）、
  `tp2`（TP-aware PLE，TP=1 时是 no-op）；
- 每次替换前把快照里的原文件备份到 `_dev\out\overlay_backup\<时间戳>\`，`-Restore` 可回滚；
- 文件按 SHA256 比对，已经一致就跳过；
- 改完要重启服务才生效（Python 已经在内存里加载了旧模块）。

### 7.2 当前生产配置（2026-10-21 03:02 重启）

```text
--block-size 1680 --prefix-match-unit 16
--kv-transfer-config {"kv_connector":"OffloadingConnector","kv_role":"kv_both",
                      "kv_connector_extra_config":{"cpu_bytes_to_use":17179869184}}
--speculative-config {"method":"mtp","num_speculative_tokens":3,
                      "adaptive_k":{"min":1,"max":3,"log_interval":200},
                      "disable_eagle_block_drop":true}
```

tier 落在 `D:\vllm-offload\vllm_offload_<engine_id>.mmap`（`VLLM_OFFLOAD_REGION_DIR` 可改）。
占 **16 GiB 磁盘**（实测 D: 372.50→356.47 GB），**不占 16 GiB 提交内存**：区域没 pin 住
（`cudaHostRegister` code=2），访问过的页只进可回收的文件缓存。KV 池仍是 415,306 token。

两个启动入口的默认值已对齐：`start_qwen38_flash_next.ps1` 与内层
`_dev/bin/_flashnext_struct_serve.ps1` 的 `-KvOffloadGb` 都是 **16**，所以直接调内层脚本
（例如 `_serve_bg.ps1` 的默认 `-ServeScript`）也会带 tier；`-KvOffloadGb 0` 关掉。

## 8. 原始日志索引

原始日志：`_dev/out/logs/arm_dflash7_*`、`arm_dflash_sched_*`、`arm_mtp1680_*`、
`arm_mtp_k3_*`、`arm_p4_*`、`arm_kvoff_*`。

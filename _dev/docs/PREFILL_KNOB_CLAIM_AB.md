# Flash-Next：网上流传的四个"调优档位"在本机的 A/B（2026-10-07）

外部说法：Qwen3.8-Flash-Next 把 prefill 从 "3k+ t/s" 提到 "5k+ t/s"，改动是
prefill batch 2048→4096、MTP 草稿 1→2、PLE row cache 512 MiB→8192 MiB、PLE AIO depth 256→512。
本记录是这四项在 **本仓库、本模型、本硬件** 上的实测拆分。

装置：GPU1（CMP 170HX 64G，空闲卡）起独立实例 `:9494`，生产实例（GPU0 `:9393`）全程不动；
同一个 Uncensored AutoRound 3bpw+MTP checkpoint；每档 warmup 一条（独立 seed，不计分），
测量 3 条（seed=40200+len，跨臂同一批 prompt）、`--random-output-len 1`、C=1、temperature=0、
`/v1/completions`；`request_prompt_tokens_sum` / `request_prefill_kv_computed_tokens_sum` 增量
恒等于 6144/24576/98304（6 轮时 258048），`prefix_cache_hits_total` 与 preemption 增量恒为 0
⇒ 零前缀命中、无抢占。脚本 `_dev/bench/_prefill_arm.sh`，结果 `_dev/out/prefill_<arm>_*.json`。
每个测量窗口都对着生产日志核对过"有没有并发请求"（下表最后一列）。

## 1. 冷启动、3 轮、配平条件（中位 TTFT 秒 / 输入 token ÷ 中位 TTFT）

| 臂 | 配置 | 2048 | 8192 | 32768 | 生产并发 |
| --- | --- | --- | --- | --- | --- |
| A | 生产档：batch 2048、MTP×3、cache 512 MiB、depth 256 | 0.721 / 2840 | 2.061 / 3974 | 9.300 / 3523 | 无 |
| A′ | 与 A 完全相同，重启复测 | 0.734 / 2789 | 2.063 / 3971 | 9.263 / 3538 | 无 |
| E | 与 A 相同，但第 4 次加载引擎 | 0.673 / 3041 | 2.030 / 4035 | 8.917 / 3675 | 无 |
| B | **只改** batch 2048→4096（其余同 A） | 0.697 / 2937 | 2.186 / 3747 | 8.784 / 3730 | 无 |
| C | 整套：4096 + MTP×2 + 8192 MiB + depth 512 | 0.745 / 2748 | 2.271 / 3608 | **7.497 / 4371** | 无 |
| D | **只改** PLE：cache 8192 MiB + depth 512 | 0.693 / 2954 | 2.057 / 3983 | **7.885 / 4156** | 无 |

A↔A′ 差 ≤1.8% ⇒ 同一条件下没有漂移；B/C/D 与基线的差才可归因。
E 比 A/A′ 快 3–6%（同一份 95.37 GiB PLE 表被反复读，宿主文件缓存越来越热）⇒ **换第 N 次加载引擎本身就是一个变量**。

## 2. 判读

- **"5k+" 在本机没有复现。** 8k/32k 的基线就是 3.5–4.0k t/s（对应对方说的 "3k+"），
  四个改动的任何一种都没把 8k 推到 5k；唯一过 4.4k 的是热态的 2048（见 §3）。
  5k 要求 8192 的 TTFT 降到 1.64 s（-20%），实测 2.03–2.27 s。
- **batch 2048→4096 单改 = 噪声内**：+3.4% / -5.6% / +5.9%，都低于 10% 判定线；
  3 轮里最长与最快差 30%（32k：9.81 vs 7.10 s），3 个样本本来就读不出 5% 的效应。
- **真正的杠杆是 PLE 两项**（cache 512 MiB→8192 MiB + depth 256→512）：
  32k 的 TTFT -15.2%（D）/ -19.4%（C 整套），且 D、C 两臂独立复现。
  8k 上两者无效（D 2.057 ≈ 基线 2.061）。
- **整套改动把 8k 弄慢了 10%**（C 2.271 vs 基线 2.061）——MTP×2/adaptive k≤2 与 4096 分块在
  这个长度上是负收益；所以"照抄那一套"在 8k 上是亏的。
- 机制**未直接测到**：默认日志里没有 PLE 命中率/行等待（`VLLM_PLE_SSD_STATS` 没开），
  只有计时证据。`_dev/docs/FLASH_NEXT_ROUTE_A_BUILD.md` §28 说 PLE 宿主侧工作不在关键路径上，
  那是**解码窗口**的结论，长 prefill 不在此覆盖范围内。

## 3. 热态（同一引擎的第二次测量，6 轮、seed 51000，两臂条件配平）

| 臂 | 配置 | 2048 | 8192 | 32768 |
| --- | --- | --- | --- | --- |
| D2 | cache 8192 MiB + depth 512 | 0.461 / 4447 | 2.103 / 3896 | 8.192 / 4000 |
| E2 | 生产档 | 0.448 / 4570 | 2.018 / 4060 | 9.544 / 3433 ← **窗口内有生产请求（19:03:53）** |

热态 2048 两臂都到 4.4–4.6k t/s（行缓存/文件缓存已热），说明**跨请求复用**才是大行缓存真正买到的东西；
E2 的 32k 数字被一次并发的生产 prefill 污染（共用同一块 NVMe），不能当作配平读数。

## 4. 内存与操作安全（8192 MiB row cache）

`row_bytes = 320`（ngram 行 = 160 维 bf16），`cache_limit = cache_mb*1MiB // (320+128)`：
512 MiB → 1.20 M 行（占全表 320 M 行的 0.37%）；8192 MiB → 19.2 M 行（6.0%）。
prompt 预取上限 `min(16384, cache_limit // (2*16))` 在两个尺寸下**都等于 16384**（未被夹），
所以这一臂不存在"顺手把预取视界也改了"的交互。
整轮过程中 free physical 41.8–45.7 GiB、commit 余量 63.4–73.5 GiB、页文件用量 ~7 GiB
⇒ 没有触发 §28.1 那次 68 GiB 提交事故，8192 MiB 在这台机器上可运营。

## 5. 采纳建议

`-PleCacheMb 8192 -PleDepth 512`（保持 batch 2048、MTP×3）：32k prefill 实测快 15–19%，
8k 无回退，内存有余量。**不建议**改 batch→4096（读数在噪声内，且 §12.1 记录它会换数值路径）。

**已落为默认值**（2026-10-07）：

- `start_qwen38_flash_next.ps1` → `-PleDepth 512`、`-PleCacheMb 8192`（内层 `_dev/bin/_flashnext_struct_serve.ps1`
  的 128 MiB / 256 保持不动，烟测与探针臂仍然用它，外层总是显式传值，所以生产臂是确定的）。
- 对方 WSL 仓库 `~/code/qwen3.8-flash-next-cmp170hx/config/engine.env` →
  `QWEN_SSD_CACHE_MB:=8192`、`QWEN_SSD_DEPTH:=512`（`./start.sh --params` 已确认真生效）。
- 在跑的 GPU0 生产实例（`:9393`）仍是旧的 512 MiB / depth 256，**要下次重启才吃到新默认**。
Windows 侧写进 `start_qwen38_flash_next.ps1` 的默认值，WSL 侧写进 `config/engine.env`
（`QWEN_SSD_CACHE_MB` / `QWEN_SSD_DEPTH`）。

# 170HX 基线基准（2026-09-20）

CMP 170HX / Qwen3.8-27B-W4A16-AutoRound-fast / vLLM 0.29 Windows 源码构建。
协议复刻自 WSL 项目 `bench/run_benchmarks.sh`，便于与 RTX 3090 侧直接对比。
原始日志：`_dev/out/_bench_base/*.log`；汇总表：`_dev/out/_bench_base/SUMMARY.md`；复现脚本：`_dev/bench/_bench_suite.py`。

---

## 1. 测试条件

| 项 | 值 |
|---|---|
| GPU | NVIDIA Graphics Device（CMP 170HX, GA100/sm_80）, 65,536 MiB |
| 模型 | `Qwen3.8-27B-W4A16-AutoRound-fast`（W4A16，文本侧 14.87 GB） |
| 服务 | `_serve_mtp.py`：MTP 投机 k=1，CUDA Graph 开，`--language-model-only` |
| `max_model_len` | 204,800 |
| `--max-num-seqs` | **4** |
| `--max-num-batched-tokens` | 4096 |
| `--gpu-memory-utilization` | 0.95 |
| KV cache | 849 blocks × 800 = **679,200 token**（200K 上下文下 3.32x 并发余量） |
| 显存 | 用 62,041 / 空 3,114 MiB；KV 池实际占用 ≈0% |
| 测量工具 | `vllm bench serve`（OpenAI 后端），每档独立 `--seed`（避免前缀缓存假命中） |
| 采样 | temperature=0，`--percentile-metrics ttft,tpot,itl,e2el` |

**口径说明**

- `tok/step` 取自 Prometheus 计数差 `1 + Δaccepted / Δdrafted`（日志自报的
  `Acceptance length` 作为回退），二者一致（差 ≤0.01）。
- `decode tok/s` = `C × 1000 / medTPOT`，是"每流间隔 × 并发数"的**上限估算**，
  不是吞吐实测值；实测吞吐看 `e2e tok/s`。
- **C 超过服务端 `--max-num-seqs` 的档位会被拆成多批**，此时 `meanTTFT` 反映排队
  而非并发延迟。本服务 `max-num-seqs=4`，所以 C=8 实为 4+4 两批。

---

## 2. 并发阶梯（真实 prompt，out=1024，T=0）

数据集：`_dev/bench/_prompts_real.jsonl`，8 条 prompt，输入合计 1,130 token。

| C | e2e tok/s | decode tok/s* | medTPOT ms | meanTTFT ms | p99TTFT ms | tok/step | medITL ms | p99ITL ms | dur s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **108.4** | 110.5 | 9.05 | 126.5 | 163.8 | 1.780 | 16.22 | 18.16 | 72.2 |
| 2 | **194.3** | 212.3 | 9.42 | 170.1 | 207.2 | 1.790 | 16.82 | 19.63 | 40.1 |
| 4 | **352.6** | 392.9 | 10.18 | 219.6 | 275.3 | 1.810 | 18.17 | 20.95 | 22.1 |
| 8 † | 353.3 | (779.0) | 10.27 | 5591.8 | 11665.6 | 1.800 | 18.15 | 21.38 | 22.1 |

† 受 `max-num-seqs=4` 限制，实为 4+4 两批 —— 与 C=4 总时长几乎相同（22.1 s）即为佐证。
该行的 `decode` 列（779.0）**高估**，实际跑的是 4 条并发；`meanTTFT` 是排队时间。

**要点**

- C=1→2→4 的 e2e 扩展 108.4 → 194.3 → 352.6，接近线性（1.79x / 1.81x）。
- 同期 medTPOT 只从 9.05 涨到 10.18 ms（**+12%**）—— 4 并发下几乎不减速。
  原因：decode 的成本是"每步读一遍 ~14.8 GB 权重"，与 batch 内序列数无关，并发只是摊薄它。
- C=1 的 meanTTFT 126.5 ms 对应 1,130 token 的输入，折算 prefill ≈ 8.9k tok/s（短 prompt，受调度开销主导，非稳态值）。

---

## 3. Prefill 阶梯（random dataset，output=1）

| 输入长度 | C | n | prefill tok/s | meanTTFT ms | p99TTFT ms | dur s |
|---|---|---|---|---|---|---|
| 4,096 | 1 | 8 | **2,238** | 1,830.2 | 1,848.4 | 14.6 |
| 16,384 | 1 | 4 | **2,111** | 7,763.2 | 7,777.6 | 31.1 |
| 65,536 | 1 | 2 | **1,724** | 38,022.1 | 38,033.7 | 76.0 |
| 102,400 | 1 | 1 | **1,515** | 67,568.1 | 67,568.1 | 67.6 |

**要点**

- 4k → 100k 衰减 2,238 → 1,515 tok/s（−32%），符合 attention 的 O(n²) 项在长序列下变重。
- TTFT 与输入长度基本线性：1,830 ms（4k）→ 67,568 ms（100k），即 ~0.66 ms/token。
- 单并发下 2.2k tok/s 是这张卡的稳态 prefill 上限量级。

---

## 4. 长上下文（1 × 100k 输入 / 256 输出）

| 项目 | meanTTFT ms | p99TTFT ms | medTPOT ms | medITL ms | p99ITL ms | dur s |
|---|---|---|---|---|---|---|
| 1×100k/256 | 65,465.4 | 65,465.4 | 45.88 | 91.79 | 92.98 | 77.2 |

tok/step = 1.990；prefill = 1,296 tok/s

**要点**

- 100k 输入下 medTPOT 45.88 ms（对比短 prompt 的 9.05 ms）—— 长上下文 decode 约慢 5x，
  这是 KV 读取量随上下文线性增长的必然结果。
- 无 KV offload 也能直接吃下 100k（KV 池 679k token，峰值占用远低于上限）。
  这是 64 GB 显存相对 24 GB 卡的结构性优势。

---

## 5. Warmup（random 256 in / 256 out，16 prompts）

| 项目 | C | e2e tok/s | decode tok/s | medTPOT ms | meanTTFT ms | p99TTFT ms | tok/step | medITL ms | p99ITL ms | dur s |
|---|---|---|---|---|---|---|---|---|---|---|
| random 256/256 | 8 † | 323.2 | 741.4 | 10.79 | 2589.7 | 3857.2 | 1.820 | 18.05 | 127.11 | 12.7 |

† 同样受 `max-num-seqs=4` 限制（16 prompts / 并发 8 → 4 批）。p99ITL 127 ms 是批间边界
（一批结束、下一批 prefill 插队），不是稳定态指标。

---

## 6. 与 RTX 3090（HyperQwen）对照

参照来源：WSL 项目 `~/testcode/qwen38-27b-rtx3090/README.md`（vLLM 0.28.0，
同一 checkpoint）与 `bench/results/step_watch.log`（2026-09-19 实测）。
**两侧服务配置不同**，见下方差异表。

| 指标 | 170HX（本次实测） | RTX 3090 / HyperQwen | 说明 |
|---|---|---|---|
| 单流 decode (C=1) | 108.4 e2e / 110.5 上限 | **127**（MTP 路径 121） | 170HX ≈ 3090 的 85–91% |
| step 中位 | **16.2 ms**（medITL） | 23.7 ms（step_watch） | 170HX **快 32%** |
| tok/step | 1.78 – 1.81（MTP k=1） | **4.51**（DFlash2） | 3090 投机效率高 2.5x |
| prefill 4k | **2,238 tok/s** | —（1k in：batch 1,810 / single 1,440） | 170HX 不弱，量级相当或更好 |
| prefill 长序列 | 1,515 tok/s @100k | — | 无 offload 直吃 100k |
| 并发上限 | 4（**配置**限制） | 8 slots（single）/ 64（batch） | 硬件支持更高，受 `max-num-seqs` 约束 |
| 4 并发 e2e | **352.6 tok/s** | batch 64 并发 1,035 tok/s | 前者 4 并发，后者 64 并发，非同尺度 |
| 上下文 | 100k 实测通过，上限 200k | 150k（profile D）/ 240k（profile E） | 3090 靠 KV offload 换上下文 |
| 显存约束 | 64 GB，KV 池 679k token | 24 GB，靠 KV offload 16 GB | 170HX 无 offload 需求 |

### 关键差异（对比时必须带上）

| 维度 | 170HX | 3090 |
|---|---|---|
| 投机方法 | **MTP k=1**（模型自带单层 MTP） | **DFlash2**（7 drafts/pass，15 tokens/step）或 MTP（profile D） |
| 上下文换显存 | 不需要 | 16 GB KV offload 到 CPU |
| vLLM 版本 | 0.29（Windows 源码构建） | 0.28.0（Docker 镜像） |
| 推理框架 | 本地源码 + MSVC 构建 | `ghcr.io/syv-ai/hyperqwen` 容器 |

### 怎么读这组对比

1. **硬件 170HX 更快，软件 3090 更成熟。** step 时间 16.2 vs 23.7 ms 说明前者的单步成本
   低 32%（W4A16 权重更小 + 带宽更高）；但 3090 靠 DFlash2 把 tok/step 做到 4.51，
   单流净吞吐反而更高。**170HX 的瓶颈在投机效率，不在算力。**
2. **提高 `num_speculative_tokens` 是最直接的抓手。** 当前 k=1，tok/step 1.78；
   若能提到 2–3 并维持可接受接受率，单流吞吐有望显著抬升（接受率会用
   `vllm:spec_decode_num_accepted_tokens_total` 直接看到）。
3. **并发吞吐受配置而非硬件限制。** KV 池 679k token 只用了 ~0%，`max-num-seqs=4`
   是人为约束。提到 16/32 后 64 GB 卡的聚合吞吐才有可比性（3090 batch 是 64 并发）。
4. **prefill 与长上下文是 170HX 的强项。** 2,238 tok/s（4k）与 100k 直吃无 offload，
   这两项上 3090 需要 int8 激活或 KV offload 才追得上。

---

## 7. 复现方式

```powershell
# 服务（另开一个独立终端，勿放在会被超时回收的后台里）
# 注：记录当时用的是 _serve_mtp.py（200K 上下文 + MTP + CUDA Graph），
#     该脚本已于 2026-09-21 删除并并入根目录 `start_server.ps1`。等价替代：
#       & D:\code\vllm-windows\start_server.ps1 -TextOnly -Spec mtp
# 本次基准的绝对数字随脚本迁移可能不完全可比，趋势结论不变。

# 全量套件（约 10 分钟）
& .\.venv\Scripts\python.exe _dev/bench/_bench_suite.py

# 只跑某几段
& .\.venv\Scripts\python.exe _dev/bench/_bench_suite.py --only cohort,prefill

# 不跑 GPU，仅从已保存日志重建 SUMMARY.md
& .\.venv\Scripts\python.exe _dev/bench/_bench_suite.py --reparse
```

依赖：`pandas`（custom 数据集需要，`uv pip install --python .venv/Scripts/python.exe pandas`）。
缺它会报误导性的 `ImportError: Please install vllm[bench]`。

---

_数据采集：2026-09-20 11:49–12:01（主套件 7.0 min + cohort 补跑 3.5 min）。_

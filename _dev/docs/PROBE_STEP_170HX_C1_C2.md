# step_probe 单/双并发实测（CMP 170HX · Windows · MTP）

用 WSL 项目 `bench/step_probe.py` 的同一份脚本（纯标准库、走 OpenAI chat API、逐 step 计时），
测当前 Windows 服务的单并发与双并发。脚本副本：`_dev/probe/_step_probe.py`
（原始参照 `_step_probe_ref.py` 已删 —— 它当时与 `_dev/probe/_step_probe.py` 逐字节相同）。

## 测试条件

| 项 | 值 |
|---|---|
| 服务 | `http://127.0.0.1:8000`，`Qwen3.8-27B-W4A16-AutoRound-fast` |
| 引擎 | vLLM 0.29 (Windows 源码构建)，W4A16 Marlin，CUDA Graph 开 |
| 投机解码 | MTP 草稿头（截断词表 40960），`num_speculative_tokens=1` |
| 上下文 | `max_model_len=204800`，KV cache 656,132 tokens |
| 显存 | 61,027 MiB / 65,536 MiB（权重 14.81 GiB，`--gpu-memory-utilization 0.95`） |
| 负载 | 独占，无 peer（探针「负载」列 = 无） |
| 请求 | 内置默认 prompt（英文+「怎么理解这段话」），353 输入 token |
| 输出 | `-n 1000 -i`（ignore_eos，固定长度，便于跨机对比） |
| 轮次 | `-r 2` |

## 汇总

| 轮 | C | e2e (tok/s) | 单流dec (tok/s) | TTFT (s) | tok/step | step中位 (ms) | step均值 (ms) | 倍数 | offload | 负载 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1/2 | 1 | 88.0 | 89.6 | 0.21 | 1.52 | 17.0 | 17.0 | 0.67x | +0 | 无 |
| 2/2 | 1 | 90.9 | 92.4 | 0.19 | 1.57 | 17.0 | 17.1 | 0.67x | +0 | 无 |
| 1/2 | 2 | 165.9 | 90.1 | 0.39 | 1.60 | 17.9 | 17.7 | 0.70x | +0 | 无 |
| 2/2 | 2 | 172.9 | 89.1 | 0.34 | 1.59 | 18.0 | 17.8 | 0.71x | +0 | 无 |

step 分位：c=1 为 p50 17.0 / p90 18.0 / p99 22.0 ms；c=2 为 p50 17.9 / p90 18.6 / p99 23.0 ms。

## 结论

### 1. 单步成本 17.0 ms，比 3090 基线（25.5 ms）快 33%

`倍数` 列 0.67x 是**好事**（该列 = step中位 / 3090 的 25.5 ms 基线，只有 ≥1.3x 才算异常）。
W4A16 权重仅 ~14.8 GiB，170HX 的 HBM2e 带宽更高，每步读一遍权重的成本更低。
四个数据点的 step 中位方差为 0（17.0/17.0/17.9/18.0），说明服务非常稳定。

### 2. 双并发几乎不减速 —— 这就是这张卡在 W4A16 下的甜点区

- 单流 decode：**89.6 → 90.1 tok/s**（基本持平，甚至略升）
- step 中位：**17.0 → 17.9 ms**（+5.9%）
- e2e：**88 → 166~173 tok/s**，`e2e / 单流dec` = 1.84~1.94 ≈ C=2

一次前向读一遍权重（~14.8 GiB）是 decode 的主要成本，它**与 batch 内序列数无关**。
C=2 时两条序列摊同一份权重读取，所以 per-stream 速度不掉、总吞吐近似翻倍。
该卡显存里躺着 656k token 的 KV cache，200K 上下文可跑 3.2 路并发，理论上还能继续往上堆 C。

### 3. MTP 是唯一的加速来源，加速比 ≈ tok/step

tok/step 实测 1.52~1.60，即 MTP 草稿平均每步拿走 ~1.6 个 token（服务端全局计数
`1+accepted/drafted` = 1.52~1.59，一致）。加速比 = tok/step / 1.0（无投机时每步 1 token），
所以本配置相对无 MTP 约有 1.5~1.6 倍解码提升。

注意 tok/step 由**内容**决定，不是故障指标：中文长文 1.5~1.6，英文基准可到 2.2~2.5
（本机 `-i` 强制续写中文，属偏悲观的一侧）。

### 4. 延迟分布健康，无插队

p99（22~23 ms）略高于 p50（17~18 ms），来自首次请求的 Triton 内核 JIT 与采样抖动；
`步均值` 与 `步中位` 几乎相等（17.0/17.0、17.9/17.7），说明没有 peer 抢占 batch，
也没有 KV 驱逐流量（`offload` 列 +0 MB）。

## 复现命令

```bash
cd /d/code/vllm-windows
# 单并发
./.venv/Scripts/python.exe _dev/probe/_step_probe.py --port 8000 \
    --model Qwen3.8-27B-W4A16-AutoRound-fast -c 1 -n 1000 -i -r 2
# 双并发
./.venv/Scripts/python.exe _dev/probe/_step_probe.py --port 8000 \
    --model Qwen3.8-27B-W4A16-AutoRound-fast -c 2 -n 1000 -i -r 2
```

脚本默认 `--port 18020 --model qwen3.8-27b`（WSL 项目的端口/模型名），
在 Windows 侧必须显式覆盖这两个参数。`--host` 默认 127.0.0.1 无需改。

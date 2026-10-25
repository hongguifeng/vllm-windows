# Qwen3.8-Flash-Next：量化与推理预算对 SWE-bench 能力的影响

> 对象：Qwen3.8-Flash-Next 的 EXL3 5.05bpw 与 AutoRound 3bpw 两套权重，
> 外加 DeepSeek flash 作外部参照。测试集为 SWE-bench Verified 的 HARD-26 与 S-40，
> 评测器为 mini-swe-agent 2.4.6 + swebench 5.0.2。
> 记录底稿（AutoRound 两臂的原始数据）：`agent_bench/RESULTS_flash_next_censored_vs_uncensored.md`。
> 后续（测试集饱和后换用 SWE-Bench Pro V2 / DeepSWE）：
> `agent_bench/RESULTS_pro_v2_deepswe_2026-10-23.md`。

## 1. 结论速览

1. **同旧预算时统计等价，放开预算后 EXL3 5.05bpw 点估计反超**：1536/60 下 31/66 vs 38/66
   （p=0.14）；同样的 xhigh 30000/200 下 **64/66 vs 61/66**（discordant 4/1，p=0.38）。
   方向与"位数更高"一致，但 n=66 还不足以确认。
2. **之前所有低分基本都是预算假象**：只把 cap 放到 30000、步数放到 200（medium），
   同一份 3bpw 权重就从 38/66 升到 52/66；再把 effort 提到 xhigh，到 **61/66（92.4%）**。
3. **effort 与预算两个旋钮独立有效**：只放预算 38→52（p=0.0005）；在充足预算下再把
   medium 换成 xhigh，52→61（p=0.035）。把 xhigh 加在旧预算上反而更差（35/66）。
4. **真正的能力边界在"多文件/完整修复"**：放开预算后仍失败的题，几乎都是
   "改对文件但只改一半"或"全覆盖但逻辑不对"，与量化精度无关。
5. **DeepSeek flash 在旧预算下更强（55/66），新预算下与 Qwen 打平（59 vs 61，p=0.69）**；
   它的瓶颈不是预算，加预算只 +4（p=0.22）。

主矩阵（resolved / total）：

| 配置 | HARD-26 | S-40 | 合并 66 |
|---|---:|---:|---:|
| AutoRound 3bpw censored（A，medium 1536/60） | 16/26 | 22/40 | 38/66（57.6%） |
| AutoRound 3bpw uncensored（B，medium 1536/60） | 10/26 | 20/40 | 30/66（45.5%） |
| **EXL3 5.05bpw**（medium 1536/60） | 12/26 | 19/40 | 31/66（47.0%） |
| **EXL3 5.05bpw，xhigh 30000/200** | **26/26** | **38/40** | **64/66（97.0%）** |
| Qwen 3bpw，medium 30000/200 | 22/26 | 30/40 | 52/66（78.8%） |
| Qwen 3bpw，xhigh 30000/60 | 13/26 | 22/40 | 35/66（53.0%） |
| **Qwen 3bpw，xhigh 30000/200** | **25/26** | **36/40** | **61/66（92.4%）** |
| DeepSeek flash，medium 1536/60 | 24/26 | 31/40 | 55/66（83.3%） |
| DeepSeek flash，xhigh 30000/200 | 24/26 | 35/40 | 59/66（89.4%） |

## 2. 背景与问题

2026-10-13/14 的记录用同一份协议（medium、`max_tokens=1536`、step limit 60）比较了
Qwen3.8-Flash-Next 的两个 AutoRound 3bpw 变体（censored A / uncensored B），
合并 66 题分别为 38/66 与 30/66（p=0.115）。之后需要回答三个问题：

- 换用 recipe 的 **EXL3 5.05bpw** 权重，是否比 3bpw AutoRound 更强？
- 如果差不多或更差，差在哪——量化、服务栈，还是 agent 协议？
- 这套子集对**外部强模型**（DeepSeek）还有多少区分度？

## 3. 实验设计

### 3.1 模型与端点

| 臂 | 权重 | 端点 / 硬件 | 服务栈 |
|---|---|---|---|
| A | AutoRound 3bpw censored（+MTP） | Windows `:9393`，1×CMP 170HX | Windows 原生 vLLM fork |
| B | AutoRound 3bpw uncensored（+MTP） | Windows `:9394`，1×CMP 170HX | 同上 |
| EXL3 | `turboderp/Qwen3.8-Flash-Next-exl3`（5.05bpw） | WSL `:8000`，2×CMP 170HX | recipe 容器，TP=2 + MTP k=4，`qwen3_coder` 解析器 |
| Qwen 3bpw | 同 A（AutoRound 3bpw 权重） | `:9393`（MTP）跑 HARD-26，`:9402`（DFlash `num_speculative_tokens=7`）跑 S-40 | Windows 原生 + 投机解码 |
| DeepSeek | `deepseek-flash`（托管） | `https://api.deepseek.com` | OpenAI 兼容 API，原生 function calling |

EXL3 服务启动：`MIN_GPU_MIB=24000 CUDA_VISIBLE_DEVICES=1,2 P2P=off GPU_UTIL=0.95 SPEC_N=4 bash start.sh`
（served name `qwen3.8-flash-next`，KV 1,020,688 tokens）。

### 3.2 测试集

| 子集 | 文件 | 规模 | 说明 |
|---|---|---:|---|
| HARD-26 | `subset_hard25.json` | 26 | 从 Verified 中挑的难例 |
| S-40 | `subset_s40.json` | 40 | 结构性难例（多文件/大仓库） |
| 合并 | — | 66 | 记录的主分析口径 |

### 3.3 协议与变量

- 客户端：mini-swe-agent 2.4.6，`temperature=0`，swebench 5.0.2，litellm 1.105.0；
- 评估：`swebench.harness.run_evaluation --max_workers 4 --timeout 2400`；
- 记录语义：`resolved/total` 为端到端，空补丁不计入 completed；
- 变量矩阵：

| 配置名 | reasoning effort | max_tokens | step limit |
|---|---|---:|---:|
| 记录基线 | medium | 1,536 | 60 |
| cap 修复 | medium | 4,096 | 60 |
| 2×2 | medium / xhigh | 1,536 / 30,000 | 60 / 200 |

## 4. 两套量化到底差在哪

### 4.1 配置

| | EXL3 | AutoRound |
|---|---|---|
| 算法 | trellis/码本量化（TCQ 类），128×128 块 | block-wise 重建 + 可学习舍入（AutoRound 0.15.0） |
| 存储格式 | `trellis` I16 + `suh`/`svh` F16 + `mul1` 标量 | GPTQ 兼容打包：`qweight`/`qzeros` I32 + `scales` F16 |
| 平均位宽 | 5.05 | ~3.15 |
| 分组 | 每行/列各一个 FP16 scale | 每 128 个输入共享一组 scale/zero，对称 |
| 逐层分配 | 5 bit × 75,277；6 bit × 167（head、vision）；7 bit × 307（linear_attn qkv/z） | 3 bit × 36,888（主模型 experts）；4 bit × 1,536（MTP experts）；8 bit × 183（linear_attn）；FP16 × 625（小张量） |
| 校准 | 250 行 × 2048 列 | iters 50，lr 0.002，gradient_accumulate 8 |

### 4.2 同一层的实际占用（`layers.0.mlp.experts.0.gate_proj`，2560×640）

| | 张量 | 字节 | bpw |
|---|---|---:|---:|
| EXL3 | trellis [160,40,80] I16 + suh [2560] + svh [640] + mul1 | 1,030,404 | **5.03** |
| AutoRound | qweight [240,640] I32 + qzeros [20,60] + scales [20,640] | 644,800 | **3.15** |

EXL3 是 AutoRound 的 1.60 倍字节。两者都主动保护线性注意力（EXL3 7 bit，AutoRound 8 bit）；
多出来的位主要花在 MoE experts 上。

### 4.3 PLE（n-gram embedding）表

| | AutoRound | EXL3 |
|---|---|---|
| 存储 | BF16，320 B/行 | 6-bit trellis，122 B/行 |
| 大小（320M 行） | 95.37 GiB | 36.36 GiB |
| 行布局 | 128 shard × 2,500,012 行 | head_offsets 与 AutoRound 逐项相同 |

把两边同 id 的行解码后对比：行是**对齐**的（同 id cos≈0.86，异 id ≈0.05），但 EXL3 行的
范数是 BF16 表的 ~1/127，且方向残差约 13%——远差于 6-bit 标量量化应给的 cos 0.99994。
说明两个 checkpoint 的 PLE 参数化不同（或解码假设与转换器约定不一致），
**这次对比不能当作 PLE 量化误差来引用**，也无法据此断言 BF16 PLE 是 AutoRound 的优势：
记录中的 A、B 都保留 BF16 PLE，B 却低于 EXL3。

## 5. 结果

### 5.1 同协议主对照（medium，1536/60）

| 臂 | HARD-26 | S-40 | 合并 | 空补丁 |
|---|---:|---:|---:|---|
| A | 16/26（61.5%） | 22/40（55.0%） | 38/66（57.6%） | 6 + 7 |
| B | 10/26（38.5%） | 20/40（50.0%） | 30/66（45.5%） | 8 + 3 |
| EXL3 | 12/26（46.2%） | 19/40（47.5%） | 31/66（47.0%） | 9 + 4 |

- EXL3 vs A：HARD discordant 7/3，p=0.34；S-40 5/2，p=0.45；合并 12/5，**p=0.14**。
- EXL3 vs B：S-40 4/3，p=1.00。
- 参考：记录自己的 A vs B（同模型、只差去审查）合并差 12.1 pp，p=0.115——
  这套测量在 n=66 的臂间噪声底就有 ~12 pp；要 80% 功效分辨 10 pp 需 ≈177 个配对实例。

### 5.2 cap 修复实验（把"没提交"的题换成 4096 重跑）

| 对象 | 实例 | @4096 解出 | 仍空 |
|---|---:|---:|---:|
| EXL3 的 9 个空补丁 | 9 | 6 | 3（撞 60 步上限） |
| A 的 6 个空补丁 | 6 | 3 | 3（astropy-13398、sphinx-9229、sphinx-9461） |

修复后 HARD-26：A 19/26 vs EXL3 18/26，差距基本消失。4 个两边都空的题上：astropy-13398
（EXL3 ✓ / A ∅）、django-11885（∅ / ✓）、pytest-5787（✓/✓）、sphinx-9229（∅/∅）——打平。

### 5.3 预算 × effort 的 2×2（同一份 3bpw 权重）

| 配置 | HARD-26 | S-40 | 合并 66 |
|---|---:|---:|---:|
| medium，1536/60 | 16/26（61.5%） | 22/40（55.0%） | 38/66（57.6%） |
| medium，30000/200 | 22/26（84.6%） | 30/40（75.0%） | 52/66（78.8%） |
| xhigh，30000/60 | 13/26（50.0%） | 22/40（55.0%） | 35/66（53.0%） |
| xhigh，30000/200 | 25/26（96.2%） | 36/40（90.0%） | 61/66（92.4%） |

- 只放预算（1536/60 → 30000/200，medium）：38→52，discordant 1/15，**p=0.0005**；
- 只加 effort（30000/200，medium → xhigh）：52→61，discordant 3/12，**p=0.035**；
- effort 加在旧预算上更差（xhigh 30000/60 = 35 < medium 1536/60 = 38）：xhigh 每回合更长，
  在 60 步内被截断的回合更多。

同一预算（xhigh 30000/200）下的量化对照：

| 权重 | HARD-26 | S-40 | 合并 66 | 用时（HARD / S-40） |
|---|---:|---:|---:|---|
| EXL3 5.05bpw | **26/26** | **38/40** | **64/66（97.0%）** | 10,590 s / 11,194 s |
| AutoRound 3bpw | 25/26 | 36/40 | 61/66（92.4%） | 8,030 s / 12,986 s |

discordant 4/1，p=0.38——EXL3 多解出 4 个（Qwen 只多 1 个），方向与"位数高 1.6 倍更好"
一致，但样本量不足以显著。EXL3 这一轮把 HARD-26 全部解出（含此前所有配置都失败的
sphinx-9229），S-40 只剩 django-10097、django-13195。

代价（墙钟）：medium 30000/200 为 HARD 3,382 s / S-40 3,976 s；
AutoRound xhigh 30000/200 为 HARD 8,030 s / S-40 12,986 s；
EXL3 xhigh 30000/200 为 HARD 10,590 s / S-40 11,194 s（两个子集并行、共享 TP2 服务）。

### 5.4 外部参照：DeepSeek flash

| DeepSeek | HARD-26 | S-40 | 合并 66 |
|---|---:|---:|---:|
| medium，1536/60 | 24/26（92.3%） | 31/40（77.5%） | 55/66（83.3%） |
| xhigh，30000/200 | 24/26（92.3%） | 35/40（87.5%） | 59/66（89.4%） |

- vs A（1536/60）：HARD 1/9，p=0.022；S-40 2/11，p=0.023；合并 3/20，p=0.0005；
- vs EXL3：合并 2/26，p<1e-4；
- 加预算只 +4（p=0.22）——它已在自身天花板附近，失败不是预算型；
- xhigh/200 下 vs Qwen：**61 vs 59，discordant 2/4，p=0.69**，统计打平。

### 5.5 汇总

| 配置 | resolved（66） | 与 A 的差 | 显著性 |
|---|---:|---:|---|
| A 基线 | 38（57.6%） | — | — |
| EXL3 5bpw（1536/60） | 31（47.0%） | −10.6 pp | p=0.14 |
| AutoRound xhigh 30000/60 | 35（53.0%） | −4.6 pp | — |
| AutoRound medium 30000/200 | 52（78.8%） | +21.2 pp | p=0.0005 |
| AutoRound xhigh 30000/200 | 61（92.4%） | +34.8 pp | p<1e-4 |
| **EXL3 5bpw xhigh 30000/200** | **64（97.0%）** | **+39.4 pp** | **p<1e-4** |
| DeepSeek medium | 55（83.3%） | +25.8 pp | p=0.0005 |
| DeepSeek xhigh/200 | 59（89.4%） | +31.8 pp | p<1e-4 |

## 6. 轨迹级分析

轨迹目录：EXL3 `runEH`/`runES40`，A `runH9393`/`runS40A`，Qwen 配置 `runM9393_200`/`runXH9393_200`，
DeepSeek `runDS`/`runDS40`。分析脚本 `_dev/probe/_swe_traj_compare.py`
（解析 tool_calls、reasoning、preds patch、gold patch）。

### 6.1 过程统计（中位数）

| 指标 | A(HARD) | EXL3 1536/60(HARD) | EXL3 xhigh/200(HARD) | DeepSeek(HARD) | AutoRound xhigh/200(HARD) |
|---|---:|---:|---:|---:|---:|
| 步数 | 30 | 21 | 47.5 | 20 | 55.5 |
| 工具调用 | 30.5 | 23 | 61.5 | 33.5（并行 2 条/步） | 72.5 |
| 思考字符/回合 | 16.8k | 16.1k | 54.1k | 15.8k | 65.7k |
| 首次编辑位置 | 第 9 个命令 | 第 6.5 个 | 第 19 个 | 第 10.5 个 | 第 21.5 个 |
| 跑过测试 | 22/26 | 19/26 | 26/26 | 25/26 | 25/26 |
| 补丁长度 | 1832 | 1292 | 4676 | 3688 | 4580 |
| 改动文件数 | 1.0 | 1.0 | 2.0 | 1.5 | 2.0 |
| gold 文件覆盖率 | 0.68 | 0.58 | **1.00** | 0.93 | 0.95 |

S-40 上同一趋势：A gold 0.64、EXL3 0.66、DeepSeek 0.82。
两边都解出的题上 gold 覆盖率几乎相同（HARD 0.96/0.96，S-40 0.90/0.92）——
**收敛到同一个修复时质量相当**，差异在"是否收敛"。

### 6.2 失败模式

同协议下两边共同失败：HARD-26 有 7 个，S-40 有 16 个。

| 类型 | HARD-26 | S-40 | 判据 |
|---|---:|---:|---|
| cap 可改善（已证明） | 4 | 0 | 4096 重跑至少一边解出 |
| cap 可改善（高置信，未重跑） | 0 | 3 | django-12406/15563/16032：2–14 步内连续两次 `reasoning=1536`、无 tool call |
| 步数上限（cap 无效） | 2 | 1 | sphinx-9229、sphinx-9461、django-11734 |
| 能力/完整性不足 | 1 | 12 | 有补丁但只覆盖部分 gold 文件，或全覆盖但逻辑不对 |

"能力/完整性"类的形态：

- **多文件修复只做一半**（12 个里 10 个 gold 是多文件）：django-13512（2 文件只改 1）、
  django-14315（A/B 各找到一半）、django-16256、sphinx-7462、seaborn-3187、pylint-4604；
- **补丁字节级相同**：django-13512、django-14170 两臂补丁 MD5 相同——不是量化噪声，
  是模型对该 PR 的理解停在同一处（django-14170 两边都改了同一个错文件，gold 0/2）；
- **全覆盖仍失败**：django-14376（B 2/2）、matplotlib-24870（B 2/2）、django-13195（A 3/3）；
- **大规模替换**：sympy-13091 gold 改 21 个文件，A 0、B 1/21。

放开预算后（xhigh/200）的残余失败：AutoRound 3bpw 还有 sphinx-9229（单条响应 >30k token
仍无 tool call）、django-10097、django-14170、pylint-4604、sympy-22080；EXL3 5bpw 只剩
**django-10097、django-13195**——前者正是"两臂补丁字节相同"的题之一，后者是 A 覆盖了
3/3 gold 文件却仍不过的题，属于两端都硬的能力边界。

### 6.3 个例

- **django-13128**（A 解出、EXL3 未）：gold 把 `as_sql` 的类型判断搬进 `_resolve_output_field`
  并删旧分支；EXL3 只加了 `_resolve_output_field`，旧分支保留。
- **sympy-13852**（A 解出、EXL3 未）：gold 要给 `polylog` 加 `s==2` 特值；EXL3 只改了
  docstring 和 `_expand_func`。
- **django-13121**（EXL3 解出、A 未）：A 跑满 60 步，最后几步退化成 `echo hello` / `echo ping`
  空转；EXL3 30 步解出。
- **astropy-14369**（A 解出、EXL3 空）：EXL3 在第 4 步后连续两次 1536-token 纯思考、无 tool call，
  判 `RepeatedFormatError`；该状态在 cap 4096 下可正常继续并解出。

## 7. 结论与建议

1. **量化差距只有在预算不设限时才显形**：1536/60 下 EXL3 5.05bpw 与 AutoRound 3bpw
   看不出差别（31 vs 38，p=0.14）；同样的 xhigh 30000/200 下 **64/66 vs 61/66**
   （discordant 4/1，p=0.38）——方向与"位数高 1.6 倍更好"一致，但仍未显著，
   需要更多配对样本才能确认。
2. **预算和 effort 才是主导变量**：1536/60 下的分数（47–61%）显著低估了模型能力；
   AutoRound 3bpw 到 92.4%，EXL3 5bpw 到 97.0%。要比较模型或量化，先把预算放到不截断的水平。
3. **effort 需要匹配预算**：xhigh 在 60 步下反而更差；200 步下才兑现（medium 78.8% → xhigh 92.4%）。
4. **真正的能力边界**在"多文件/完整修复"和极少数格式崩溃个例上，这类题提 cap、提精度都不解决；
   提示层可以针对（提交前跑相关测试、要求列出所有受影响文件），但需要单独验证。
5. **子集分辨力**：HARD-26 对 DeepSeek 已饱和（92%），S-40 仍有区分度（77–90%）；
   若继续做本地模型对比，应优先扩 S-40 类样本并把预算固定在不截断的水平。

## 8. 复现

```bash
# EXL3 服务（WSL）
wsl -d Ubuntu -- bash -lc 'cd ~/qwen38-recipe && MIN_GPU_MIB=24000 \
  CUDA_VISIBLE_DEVICES=1,2 P2P=off GPU_UTIL=0.95 SPEC_N=4 BOOT_CHECK=1 \
  READY_TIMEOUT=3600 bash start.sh'

# 本地 vLLM 臂（示例：xhigh 30000/200）
cd /home/hong/bench
setsid nohup .venv/bin/python run_mini.py --subset subset_hard25.json \
  --output runXH9393_200 --model-name openai/Qwen3.8-Flash-Next \
  --api-base http://localhost:9393/v1 --workers 2 --enable-thinking \
  --reasoning-effort xhigh --max-tokens 30000 --step-limit 200 \
  >> runXH9393_200.sup.log 2>&1 < /dev/null & sleep 6

# DeepSeek（API 模式把 reasoning_effort 作为顶层字段）
BENCH_API_KEY=sk-... setsid nohup .venv/bin/python \
  /mnt/d/code/vllm-windows/agent_bench/run_mini.py \
  --subset subset_hard25.json --output runDSxH_hard \
  --model-name openai/deepseek-flash --api-base https://api.deepseek.com/v1 \
  --mode api --workers 2 --enable-thinking --reasoning-effort xhigh \
  --max-tokens 30000 --step-limit 200 >> runDSxH_hard.sup.log 2>&1 < /dev/null & sleep 6

# 评估
.venv/bin/python -m swebench.harness.run_evaluation \
  --dataset_name subset_hard25.json --predictions_path runXH9393_200/preds.json \
  --run_id runXH9393s200 --max_workers 4 --timeout 2400 --report_dir /home/hong/bench

# 配对比较
.venv/bin/python mcnemar_compare.py \
  --pair "hard: A vs xh200" openai__Qwen3.8-Flash-Next.runH9393.json \
  openai__Qwen3.8-Flash-Next.runXH9393s200.json \
  --pair "s40: A vs xh200" openai__Qwen3.8-Flash-Next.runS40A.json \
  openai__Qwen3.8-Flash-Next-DFlash7.runXH9402s200.json \
  --union "hard: A vs xh200,s40: A vs xh200"

# 轨迹分析
.venv/bin/python /mnt/d/code/vllm-windows/_dev/probe/_swe_traj_compare.py \
  --a-run runH9393 --b-run runEH \
  --a-report openai__Qwen3.8-Flash-Next.runH9393.json \
  --b-report openai__qwen3.8-flash-next.runEH.json --subset subset_hard25.json
```

## 9. 产物清单

| 配置 | 报告（`~/bench/`） | 轨迹目录 |
|---|---|---|
| A / B 基线 | `openai__Qwen3.8-Flash-Next.runH9393.json`、`.runH9394.json`、`.runS40A.json`、`.runS40B.json` | `runH9393`、`runH9394`、`runS40A`、`runS40B` |
| EXL3 5bpw | `openai__qwen3.8-flash-next.runEH.json`、`.runES40.json` | `runEH`、`runES40` |
| cap 4096 修复 | `openai__qwen3.8-flash-next.runEH4k2.json`、`openai__Qwen3.8-Flash-Next.runH4k.json` | `runEH4k`、`runH4k` |
| Qwen xhigh 30000/60 | `openai__Qwen3.8-Flash-Next.runXH9393k.json`、`openai__Qwen3.8-Flash-Next-DFlash7.runXH9402k.json` | `runXH9393_30k`、`runXH9402_30k` |
| Qwen xhigh 30000/200 | `openai__Qwen3.8-Flash-Next.runXH9393s200.json`、`openai__Qwen3.8-Flash-Next-DFlash7.runXH9402s200.json` | `runXH9393_200`、`runXH9402_200` |
| Qwen medium 30000/200 | `openai__Qwen3.8-Flash-Next.runM9393s200.json`、`openai__Qwen3.8-Flash-Next-DFlash7.runM9402s200.json` | `runM9393_200`、`runM9402_200` |
| EXL3 xhigh 30000/200 | `openai__qwen3.8-flash-next.runEXHh.json`、`.runEXHs.json` | `runEXH_hard`、`runEXH_s40` |
| DeepSeek | `openai__deepseek-flash.runDS.json`、`.runDS40.json`、`.runDSxHh.json`、`.runDSxHs.json` | `runDS`、`runDS40`、`runDSxH_hard`、`runDSxH_s40` |
| 辅助脚本 | `_dev/probe/_swe_traj_compare.py`（轨迹）、`_dev/probe/_ple_precision_compare.py`（PLE）、`_dev/probe/_deepseek_probe.py`（API） | — |

## 10. Caveats

- 每个配置只跑一次；n=66，能分辨的差距约 15–20 pp 以上，单点差异不能当结论。
- 服务栈不同：EXL3 在 WSL recipe（TP2 + MTP k=4、`qwen3_coder`），A/B 在 Windows 原生单卡
  （`qwen3_xml`）；chat template 两者 MD5 相同，但批量组成/数值路径不同会被 agent 轨迹放大。
- Qwen 的两个 30k 配置分别用了 MTP（`:9393`）和 DFlash7（`:9402`）投机解码；同一份权重，
  T=0 下理论上不影响输出，但两个子集不在同一端口上跑，跨子集比较需注意。
- DeepSeek 是托管的不同模型，仅作外部参照，不能用来推断量化质量。
- PLE 的精度对比结论未闭环（两 checkpoints 参数化不同）；如需定论，需要做 PLE 消融
  （同权重换 BF16 PLE 表）而不是行级比较。

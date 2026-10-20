# EXL3 5.05bpw vs AutoRound 3bpw：SWE-bench Verified 能力对照（2026-10-20）

## 问题

把 recipe 的 **EXL3 5.05bpw** 权重（`turboderp/Qwen3.8-Flash-Next-exl3`）在 WSL
recipe 服务上跑 SWE-bench Verified，和本仓库 2026-10-13/14 记录的
**AutoRound 3bpw** 两臂（censored A / uncensored B，Windows 原生 :9393/:9394）做同协议对比：
位数高 1.7 倍的 EXL3 是否真的更强？如果不是，差在哪。

记录底稿：`agent_bench/RESULTS_flash_next_censored_vs_uncensored.md`。

## 协议（与记录完全一致）

- mini-swe-agent 2.4.6、swebench 5.0.2、litellm 1.105.0
- `temperature=0`、`max_tokens=1536`、`enable_thinking=true`、`reasoning_effort=medium`
- step limit 60、workers 2、history thinking 保留
- 测试集：HARD-26（`subset_hard25.json`，26 题）、S-40（`subset_s40.json`，40 题）
- 评估：`swebench.harness.run_evaluation --max_workers 4 --timeout 2400`

服务端：

| 臂 | 端点 | 权重 | 硬件/并行 |
|---|---|---|---|
| A（记录） | Windows `:9393` | AutoRound 3bpw censored | 1×CMP 170HX |
| B（记录） | Windows `:9394` | AutoRound 3bpw uncensored | 1×CMP 170HX |
| **EXL3（本次）** | WSL `:8000` | EXL3 5.05bpw | 2×CMP 170HX，TP=2 + MTP k=4 |

EXL3 服务：`MIN_GPU_MIB=24000 CUDA_VISIBLE_DEVICES=1,2 P2P=off GPU_UTIL=0.95 SPEC_N=4 bash start.sh`，
served name `qwen3.8-flash-next`，KV 1,020,688 tokens。

## 结果

### HARD-26（cap 1536）

| 臂 | resolved | 有提交（非空补丁） | 空补丁 | 推理耗时 |
|---|---:|---:|---:|---:|
| A：AutoRound 3bpw censored | **16/26（61.5%）** | 20 | 6 | 2,204 s |
| B：AutoRound 3bpw uncensored | 10/26（38.5%） | 18 | 8 | — |
| **EXL3 5bpw** | **12/26（46.2%）** | 17 | 9 | 2,497 s |

EXL3 vs A：discordant 7/3，exact **p = 0.34**；EXL3 落在 A 与 B 之间。

### S-40（cap 1536）

| 臂 | resolved | 完成 | 空补丁 |
|---|---:|---:|---:|
| A（AutoRound censored） | **22/40（55.0%）** | 33 | 7 |
| B（AutoRound uncensored） | 20/40（50.0%） | 37 | 3 |
| **EXL3 5bpw** | **19/40（47.5%）** | 36 | 4 |

EXL3 vs A：5/2，p = 0.45；vs B：4/3，**p = 1.00**。

### 合并（HARD-26 + S-40 = 66，记录的主分析口径）

| 臂 | resolved | 差距 | McNemar |
|---|---:|---|---|
| A | 38/66（57.6%） | — | — |
| B | 30/66（45.5%） | — | — |
| **EXL3** | **31/66（47.0%）** | 对 A −10.6 pp（discordant 12/5） | **p = 0.14** |

按记录的功效公式，80% 功效分辨该差距需 **≈177** 个配对实例；现有 66 个，**不显著**。
记录自己的 A vs B（同模型、只差去审查）差值 12.1 pp、p=0.115——这套测量的臂间噪声底就有 ~12 pp。

## cap 4096 对照（把"没提交"的题重跑）

| 对象 | 实例数 | @4096 resolved | 仍空 |
|---|---:|---:|---:|
| EXL3 的 9 个 1536 空补丁 | 9 | **6** | 3（撞 60 步上限） |
| A 的 6 个 1536 空补丁 | 6 | **3** | 3（astropy-13398、sphinx-9229、sphinx-9461） |

- 修复后 HARD-26：A 19/26（73.1%）vs EXL3 18/26（69.2%），差距基本消失。
- 4 个两边都空的实例上，4096 结果：astropy-13398（EXL3 ✓ / A ∅）、
  django-11885（∅ / ✓）、pytest-5787（✓/✓）、sphinx-9229（∅/∅）——基本打平。
- **结论**：1536 对两臂都低估，EXL3 被低估得更多（早死 9 个 vs 6 个）。

## 解题过程分析（轨迹级）

轨迹目录：EXL3 `~/bench/runEH`、`runES40`；记录 A `~/bench/runH9393`、`runS40A`。
分析脚本：`_dev/probe/_swe_traj_compare.py`（解析 tool_calls、reasoning、patch、gold patch）。

### 过程统计（中位数/均值）

| 指标 | A(HARD) | EXL3(HARD) | A(S-40) | EXL3(S-40) |
|---|---:|---:|---:|---:|
| 步数 | 30 | **21** | 22 | **19** |
| 工具调用 | 30.5 | 23 | 23.5 | 20.5 |
| 思考字符/回合 | 16.8k | 16.1k | 12.7k | 13.4k |
| 首次编辑（第几个命令） | 9 | **6.5** | 7 | **5.5** |
| 有编辑的回合 | 23/26 | 20/26 | 38/40 | 38/40 |
| 跑过测试的回合 | 22/26 | 19/26 | 32/40 | 33/40 |
| 补丁长度中位 | 1832 | 1292 | 1092 | 1650 |
| 命中 gold 文件 | 20/26 | 17/26 | 32/40 | **34/40** |
| gold 文件覆盖率 | 0.68 | 0.58 | 0.64 | 0.66 |
| 格式错误总数（其中 cap） | 46（35） | 63（49） | — | — |

两边都解出的题（HARD 9 个 / S-40 17 个）上 gold 文件覆盖率几乎相同
（0.96/0.96；0.90/0.92），补丁大小同量级——**收敛到同一个修复时质量相当**。

### 失败模式分解

两边共同失败：HARD-26 **7** 个，S-40 **16** 个。

| 类型 | HARD-26 | S-40 | 证据 |
|---|---:|---:|---|
| cap 可改善（已证明） | 4 | 0 | 4096 重跑至少一边解出 |
| cap 可改善（高置信，未重跑） | 0 | 3 | django-12406/15563/16032：2–14 步内连续两次 `reasoning=1536`、无 tool call |
| 步数上限（cap 无效） | 2 | 1 | sphinx-9229（58/59 步）、sphinx-9461（A 60 步）、django-11734（B 57 步） |
| 能力/完整性不足 | 1 | 12 | 有补丁但只覆盖部分 gold 文件，或全覆盖但逻辑不对 |

其中"能力/完整性"类的典型形态：

- **多文件修复只做一半**（12 个里 10 个 gold 是多文件）：django-13512（2 文件只改 1）、
  django-14315（A/B 各找到一半）、django-16256、sphinx-7462、seaborn-3187、pylint-4604 等。
- **补丁字节级相同**：django-13512、django-14170 两臂补丁 MD5 相同——不是量化噪声，
  是模型对该 PR 的理解停在同一处（django-14170 两边都改了同一个错文件，gold 0/2）。
- **全覆盖仍失败**（逻辑不对）：django-14376（B 覆盖 2/2）、matplotlib-24870（B 2/2）、
  django-13195（A 覆盖 3/3）。
- **大规模替换**：sympy-13091 gold 改 21 个文件，A 0、B 1/21。

个例：

- **django-13128**（A 解出、EXL3 未）：gold 把 `as_sql` 的类型判断搬进
  `_resolve_output_field` 并删旧分支；EXL3 只加了 `_resolve_output_field`，旧分支保留。
- **sympy-13852**（A 解出、EXL3 未）：gold 要给 `polylog` 加 `s==2` 特值；EXL3 只改
  docstring 和 `_expand_func`，漏了特值。
- **django-13121**（EXL3 解出、A 未）：A 跑满 60 步，最后几步退化成
  `echo hello` / `echo ping` 空转；EXL3 30 步解出。

## 补充：DeepSeek API（deepseek-flash）对照（2026-10-20 晚）

同一套子集、同一协议（cap 1536、`reasoning_effort=medium`、温度 0、step limit 60），
端点换成 `https://api.deepseek.com`（`deepseek-flash`）。`run_mini.py` 新增
`--mode api --api-key`：API 模式把 `reasoning_effort` 作为顶层字段发送，不再发
vLLM 专用的 `chat_template_kwargs`。

| 子集 | DeepSeek | AutoRound A | EXL3 | McNemar vs A | vs EXL3 |
|---|---:|---:|---:|---|---|
| HARD-26 | **24/26（92.3%）** | 16/26（61.5%） | 12/26（46.2%） | 1/9，p=0.022 | 0/12，p=0.0005 |
| S-40 | **31/40（77.5%）** | 22/40（55.0%） | 19/40（47.5%） | 2/11，p=0.023 | 2/14，p=0.004 |
| 合并 66 | **55/66（83.3%）** | 38/66（57.6%） | 31/66（47.0%） | 3/20，p=0.0005 | 2/26，p<1e-4 |

过程统计（中位数）：HARD-26 步数 20（A 30）、工具调用 33.5（A 30.5，DeepSeek 一次响应
并行发 2 条命令）、思考字符 15.8k（A 16.8k）、**gold 文件覆盖率 0.93**（A 0.68）、
补丁长度 3688（A 1832）；S-40 步数 16.5、gold 覆盖率 0.82（A 0.64）。
空补丁只有 HARD 1 个、S-40 2 个（都是 1536 思考溢出），远少于 Qwen 两臂。

注意：这是**托管的不同模型**（不是 Qwen 的量化版本），不能拿来推断量化质量；
并行工具调用也让它在 60 步预算内做了更多事。它的价值是给这套子集一个"外部强模型"
参照：HARD-26 对它明显偏简单（92%），S-40 仍有区分度（77.5%）。

产物：`~/bench/openai__deepseek-flash.runDS.json`、`runDS40.json`，轨迹 `runDS/`、`runDS40/`。

## 结论

1. **EXL3 5.05bpw 在本测试上与 AutoRound 3bpw 统计等价**：三个口径的 p 值 0.14–1.00，
   点估计略低于 A（censored）、基本等于 B（uncensored）；"位数高就该更好"没有被数据支持，
   但也没有被证伪（n=66 只够分辨 ~20 pp 以上的差距）。
2. **HARD-26 的差距一半以上是 1536 的思考溢出**：EXL3 有 8 个回合在 2–14 步内
   因连续两次 1536-token 思考、没有 tool call 而直接死亡；4096 下 9 个空补丁恢复 6 个。
   A 也有同样失败模式（6 个空补丁里 4 个），只是少 3 个。
3. **真正的能力差距集中在"多文件/完整修复"**：S-40 的 16 个共同失败里 12 个是
   "改对文件但只改一半"或"全覆盖但逻辑不对"，其中 2 个两臂补丁完全相同——
   这类不会因为提 cap 或提精度而改变。
4. **行为差异**：EXL3 更早动手（首次编辑 5.5–6.5 vs 7–9 个命令）、更早收工
   （19–21 vs 22–30 步）、补丁更小；A 探索更久、测试更多、补丁更大，偶尔磨到步数上限。
   EXL3 的失败是"局部收敛"，A 的失败是"磨"。

## 复现

```bash
# EXL3 服务（WSL）
wsl -d Ubuntu -- bash -lc 'cd ~/qwen38-recipe && MIN_GPU_MIB=24000 \
  CUDA_VISIBLE_DEVICES=1,2 P2P=off GPU_UTIL=0.95 SPEC_N=4 BOOT_CHECK=1 \
  READY_TIMEOUT=3600 bash start.sh'

# HARD-26 / S-40 推理
cd /home/hong/bench
setsid nohup .venv/bin/python run_mini.py --subset subset_hard25.json \
  --output runEH --model-name openai/qwen3.8-flash-next \
  --api-base http://localhost:8000/v1 --workers 2 --enable-thinking \
  --reasoning-effort medium --max-tokens 1536 --step-limit 60 \
  >> runEH.sup.log 2>&1 < /dev/null & sleep 6

# 评估
.venv/bin/python -m swebench.harness.run_evaluation \
  --dataset_name subset_hard25.json --predictions_path runEH/preds.json \
  --run_id runEH --max_workers 4 --timeout 2400 --report_dir /home/hong/bench

# cap 4096 对照（空补丁实例）
#   EXL3: runEH4k / runEH4k2（9 个 HARD 空补丁，runEH4k2 是完整评估）
#   A   : runH4k（6 个 HARD 空补丁，:9393 在 CMP 卡上以 -Gpu 1 启动）

# DeepSeek API 对照（HARD-26 / S-40）
cd /home/hong/bench
BENCH_API_KEY=sk-... setsid nohup .venv/bin/python \
  /mnt/d/code/vllm-windows/agent_bench/run_mini.py \
  --subset subset_hard25.json --output runDS \
  --model-name openai/deepseek-flash --api-base https://api.deepseek.com/v1 \
  --mode api --workers 4 --enable-thinking --reasoning-effort medium \
  --max-tokens 1536 --step-limit 60 >> runDS.sup.log 2>&1 < /dev/null & sleep 6

# 轨迹分析
.venv/bin/python /mnt/d/code/vllm-windows/_dev/probe/_swe_traj_compare.py \
  --a-run runH9393 --b-run runEH \
  --a-report openai__Qwen3.8-Flash-Next.runH9393.json \
  --b-report openai__qwen3.8-flash-next.runEH.json --subset subset_hard25.json
```

## 产物

| 文件 | 内容 |
|---|---|
| `~/bench/openai__qwen3.8-flash-next.runEH.json` | EXL3 HARD-26 @1536 |
| `~/bench/openai__qwen3.8-flash-next.runES40.json` | EXL3 S-40 @1536 |
| `~/bench/openai__qwen3.8-flash-next.runEH4k2.json` | EXL3 HARD 空补丁 @4096（6/9） |
| `~/bench/openai__Qwen3.8-Flash-Next.runH4k.json` | A HARD 空补丁 @4096（3/6） |
| `~/bench/openai__deepseek-flash.runDS.json`、`runDS40.json` | DeepSeek HARD-26 / S-40 |
| `~/bench/runEH`、`runES40`、`runEH4k`、`runH4k`、`runDS`、`runDS40` | 轨迹与 preds |
| `_dev/probe/_swe_traj_compare.py` | 轨迹对比脚本 |

## Caveats

- 每臂单次运行，n=66；记录本身也说明需要 ~177 配对实例才有 80% 功效。
- 服务栈不同：EXL3 是 WSL recipe（TP2+MTP k=4、`qwen3_coder` 工具解析器），
  记录两臂是 Windows 原生单卡（`qwen3_xml`）；chat template 两者 MD5 相同。
- 分辨率与硬件无关，但批量组成/数值路径不同会让 agent 轨迹被单 token 差异放大。
- S-40 的 3 个"高置信 cap 类"（django-12406/15563/16032）尚未在 4096 下重跑验证。

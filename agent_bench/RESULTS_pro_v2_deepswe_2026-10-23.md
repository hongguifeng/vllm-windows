# SWE-Bench Pro V2 与 DeepSWE：新一轮能力对照（2026-10-23）

> 背景：SWE-bench Verified 的 HARD-26 / S-40 在放开预算后已饱和
> （EXL3 5bpw xhigh/30000/200 拿到 26/26 与 38/40），无法再区分模型。本轮换用两个
> 更新、更难、规模可控的测试集：**SWE-Bench Pro V2**（20 题子集）和 **DeepSWE**
> （seed 0 的 10 题子集），用 DeepSeek API 与本地量化权重做对照。

## 1. 结论速览

| 测试集 | 子集 | DeepSeek flash | EXL3 5.05bpw | AutoRound 3bpw |
|---|---|---|---:|---:|
| **SWE-Bench Pro V2** | 20 题（A/B 各 10） | **20/20（100%）** | **17/20（85%）** | **18/20（90%）** |
| **DeepSWE** | 10 题（seed 0） | **7/10（70%）** | — | **4/10（40%）** |

> **2026-10-26 追加**：本机自持的 **Maya-M**（GLM-5.3-Flash 2-bit 量化，2×CMP 170HX）在**同一套 DeepSWE 10 题**上跑到 **3/10**，但目标测试 **721/790（91.3%）**通过，且**没有一题因 262k 上下文超限**。详见 §9。

- Pro V2 对 DeepSeek 已接近天花板（20/20，42 分钟）；AutoRound 18/20，EXL3 17/20，
  其中 EXL3 有 1 题是评测侧补丁捕获 bug 导致的假失败（`openlibrary-03095f`，详见 §3.1），
  修正后两者基本持平。
- DeepSWE 区分度明显更好：DeepSeek 7/10，AutoRound 4/10；AutoRound 的 6 个失败里
  2 个是 262k 上下文超限（协议），4 个是真实能力差异（回归/近失/未实现）。
- 两个测试集的镜像都不走 Docker Hub（Pro V2 在 GHCR、DeepSWE 在 ECR public），
  本机到 Docker Hub 的拉取会挂起，这也是 SWE-bench-Live 最终没能跑起来的原因。

## 2. 测试集与接入方式

### 2.1 SWE-Bench Pro V2

- 规模：642 题 / 11 个仓库，另有 **HARD-51** 子集；本次取 **20 题**：
  按仓库分层抽 10 个 HARD-51 + 10 个中等难度（每仓库 1–4 题），拆成 A/B 两组各 10 题，
  两个端口并发。
- 任务格式：Harbor task 目录（`task.toml` + `instruction.md` + `environment/` + `tests/` + `solution/`）；
  镜像 `ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>`。
- 评测：Harbor + `host_mini_swe:HostMiniSweAgent`（mini-swe-agent 跑在宿主机，容器只执行 bash；
  这样无需在容器里装 agent，也避开 PyTorch/PyPI 依赖）。
- 协议：`reasoning_effort=xhigh`、`max_tokens=30000`、`step_limit=200`、温度 0、
  每卡并发 ≤2（KV 约束）。

### 2.2 DeepSWE

- 规模：113 题（TS/Go/Python/JS/Rust），长周期"原始工程任务"，Harbor 任务格式；
  本次用 `--n-tasks 10 --sample-seed 0` 取确定性 10 题。
- 镜像：`public.ecr.aws/d3j8x8q7/swe-bench-202605:<ext_id>-v1.1`（ECR public）。
- 评测：Pier（Harbor 的 fork）+ `mini-swe-agent`（容器内安装），
  需额外装 `mcp`（见 §5），协议同为 xhigh / 30000 / 200。

## 3. 结果

### 3.1 SWE-Bench Pro V2（20 题）

| 臂 | 模型/端点 | 结果 | 用时 | 备注 |
|---|---|---|---:|---|
| DeepSeek | `deepseek-flash`，xhigh/30000/200，`-n 3` | **20/20（100%）** | 42 min | 无异常，20 个 trial 全 reward=1 |
| AutoRound | `Qwen3.8-Flash-Next`（:9393）与 `-DFlash7`（:9402），每卡 `-n 2` | **18/20（90%）** | 20:33–22:40（中途遇磁盘掉线，重启后补跑 5 题） | 失败 2 题：`ansible-b748ed`、`ansible-c1f2df` |
| EXL3 | `qwen3.8-flash-next`，xhigh/30000/200，每轮 `-n 3` | **17/20（85%）** | 14 题 107 min + 补跑 46 min | 失败 3 题见下（其中 1 题疑似评测 bug） |

AutoRound 的 2 个失败均在 ansible 仓库，而且都是**单测试近失**：

| 任务 | AutoRound 产出 | 验证结果 | 原因 |
|---|---|---|---|
| `ansible-c1f2df`（F5 message routing） | 34.5 KB patch | **1 个测试失败**：`test_create_generic_route` | 与 EXL3 是同一道题、同一类边界行为（create/remove/update route），AutoRound 只差这一个用例；DeepSeek 全过。 |
| `ansible-b748ed`（multipart 预处理） | 10.0 KB patch | **1 个测试失败**：`test_prepare_multipart` | 近失：改动了 `prepare_multipart` 的编码路径，但某个输入形态的边界没对齐；其余测试全过。 |

后者与 EXL3 是同一道 F5 message-routing 题，两者都失败，说明该题对 3bpw 量化模型是共同难点。
其余 18 题（含全部 openlibrary、qutebrowser、flipt、gravitational）均通过。

EXL3 失败的 3 题及原因：

| 任务 | EXL3 产出 | 验证结果 | 原因 |
|---|---|---|---|
| `ansible-40ade1`（mount_facts 重写） | 61.5 KB patch，65 次工具调用，`Submitted` | **27/48 测试失败** | 改得过大、方向偏：把 `mount_facts` 的多套解析路径一起重写，`test_gen_mounts_by_sources`、`test_duplicate_source` 等行为与参考实现不一致。同一题 DeepSeek 只用 27 KB 就全过。 |
| `ansible-c1f2df`（F5 message routing） | 40.9 KB patch，58 次调用，`Submitted` | **3/6 测试失败** | 典型的"近失"：`test_create_generic_route`、`test_remove_generic_route`、`test_update_generic_peer` 三个行为用例没过；公共逻辑改了，但边界行为没对齐。DeepSeek 14.9 KB 全过。 |
| `openlibrary-03095f`（Solr 更新器） | **0 字节 patch**（但轨迹里实际改了 3 文件 +215/−10 并 **git commit** 成功） | 测试补丁应用失败 | **两个问题叠加**：(1) 我们的 `HostMiniSweAgent` 用 `git add -A && git diff --cached` 捕获补丁，agent 一旦自己 commit，捕获结果为空；(2) agent 还新建了 `scripts/tests/test_new_solr_updater.py`，与隐藏 test patch 里的同名文件冲突，验证器报 `already exists in working directory`，直接判失败。DeepSeek 同一题只用 2.4 KB patch 全过。 |

因此 `openlibrary-03095f` 这一题**不能算作 EXL3 的能力失败**，是我们评测侧的补丁捕获/文件冲突问题；
需要用一个 commit-aware 的捕获方式重跑才能得到真实结果（修正后 EXL3 的 Pro V2 成绩可能是 18/20 而不是 17/20）。

其余 17 题（含全部 4 个 openlibrary、2 个 qutebrowser、flipt、gravitational 等）均通过；
补跑的 6 题（4 个 openlibrary + 2 个 qutebrowser）**6/6 全解**。

### 3.2 DeepSWE（10 题，DeepSeek API）

Job：`dswe-ds3`，19:13:42 → 20:12:41，**59 分钟**，10/10 trial 完成，0 报错。

| # | 任务 | 语言 | 参考解 | 结果 | f2p |
|---|---|---|---:|---:|---|
| 1 | arcane-drift-detection-baselines | Go | 14 文件 +1083 | ✅ | 82/82 |
| 2 | bandit-structured-nosec-directives | Python | 7 文件 +629/−24 | ❌ | 65/69 |
| 3 | effect-sse-httpapi-streaming | TS | 7 文件 +829/−42 | ❌ | 0/47 |
| 4 | go-critic-doc-link-checker | Go | 4 文件 +835 | ✅ | 3/3 |
| 5 | koota-entity-snapshot-rollback | TS | 6 文件 +571 | ✅ | 84/84 |
| 6 | kysely-window-grouping-helpers | TS | 15 文件 +827 | ✅ | 254/254 |
| 7 | langchain-request-coalescing | Python | 4 文件 +820 | ✅ | 50/50 |
| 8 | meriyah-explicit-resource-declarations | TS | 6 文件 +573/−6 | ❌ | 45/49 |
| 9 | true-myth-iterable-collection-combinators | TS | 4 文件 +455 | ✅ | 96/96 |
| 10 | vitest-duration-sharding | TS | 10 文件 +837 | ✅ | 56/56 |

- 精确通过 7/10（70%）；fine-grained f2p 总通过率 **88.6%**（73.5/79 测试加权）。
- 失败形态分两类：**近失**（bandit 差 4 个测试、meriyah 差 4 个测试）与
  **完全没实现**（effect SSE：0/47，agent 没有产出相关代码）。
- 全部 10 题均为 `feature_request`，参考解中位 7 文件 / +800 行，不是"找一个 bug"型任务。

### 3.2.1 DeepSWE · AutoRound（双卡并发，结果与失败原因）

同一套 10 题（seed 0），AutoRound 3bpw 双卡并发跑（`:9393` MTP + `:9402` DFlash7，
xhigh/30000/200，每卡 2 并发）。已完成的 trial 及失败原因：

| 任务 | 结果 | f2p | p2p | 失败原因 |
|---|---|---:|---:|---|
| arcane-drift-detection-baselines | ✅ | 82/82 | 2/2 | — |
| go-critic-doc-link-checker | ✅ | 3/3 | 16/16 | — |
| true-myth-iterable-collection-combinators | ✅ | 96/96 | 561/561 | — |
| bandit-structured-nosec-directives | ❌ | **69/69** | **281/282** | **回归**：目标测试全过，但改动破坏了 1 个 pass-to-pass 测试，按规则 reward=0（DeepSeek 是 f2p 65/69，另一种失败形态） |
| koota-entity-snapshot-rollback | ❌ | 80/84 | 47/47 | **近失**：4 个目标测试未过（DeepSeek 全过） |
| kysely-window-grouping-helpers | ❌ | **0/254** | 22/22 | **未实现**：agent 没有产出相关改动（0/254 f2p），DeepSeek 全过 |
| langchain-request-coalescing | ❌ | **49/50** | 232/232 | **单测试近失**：50 个目标测试只差 1 个；DeepSeek 全过 |
| effect-sse-httpapi-streaming | ❌ | — | — | **上下文超限**：30k 输出时 232,145 + 30,000 > 262,144；把 `max_tokens` 降到 16000 重跑，又撞 246,145 + 16,000 = 262,145（多 1 token）——历史会涨到填满剩余预算，调 max_tokens 只是挪墙 |
| meriyah-explicit-resource-declarations | ❌ | — | — | 首轮 Harbor **artifact collision**（`/logs/artifacts/model.patch` 路径重叠）；重跑（30k 与 16k 两次）都撞 **262k 上下文上限**（16k 时同样是 246,145 + 16,000 = 262,145） |
| vitest-duration-sharding | ✅ | 56/56 | 24/24 | — |

**最终 AutoRound 4/10（40%）**：arcane ✅、go-critic ✅、true-myth ✅、vitest ✅；
bandit ❌（回归）、koota ❌（近失）、kysely ❌（未实现）、langchain ❌（近失）、
effect ❌（262k 上下文）、meriyah ❌（262k 上下文，首轮还叠加 artifact collision）。

对比 DeepSeek 的 7/10：差 3 题，其中 2 题（effect、meriyah）AutoRound 是上下文超限、
DeepSeek 也都没过（effect 0/47、meriyah 45/49），所以按"两边都算未解"口径，
差距主要来自 koota、kysely、langchain 三题——这三题 DeepSeek 全过、AutoRound 分别是
近失（80/84）、未实现（0/254）、近失（49/50）；而 bandit 是 AutoRound 反向领先的形态
（f2p 69/69 但破坏 p2p，DeepSeek f2p 只有 65/69）。

注意：`effect` 与 `meriyah` 的两次失败都是 **262k 上下文上限**（协议限制），
不是模型没做出来；其余失败（bandit 回归、koota/langchain 近失、kysely 未实现）是真实能力差异。

### 3.3 每题用时（从每个 trial 的 `result.json` 提取）

每个 trial 记录四个阶段：`environment_setup`（拉镜像/起容器）、`agent_setup`（装 agent）、
`agent_execution`（模型交互）、`verifier`（跑测试）。汇总（秒）：

| 臂 | 题数 | 解出 | agent 中位 | agent 均值 | agent 范围 | verifier 中位 | 环境中位 |
|---|---:|---:|---:|---:|---|---:|---:|
| ProV2 · DeepSeek | 20 | 20 | **178** | 316 | 29–1131 | 16 | 3 |
| ProV2 · AutoRound | 20 | 18 | **685** | 848 | 222–2165 | 15 | 3 |
| ProV2 · EXL3 | 20 | 17 | **1162** | 1319 | 307–2980 | 16 | 3 |
| DeepSWE · DeepSeek | 10 | 7 | **592** | 672 | 260–1710 | 43 | 153 |
| DeepSWE · AutoRound | 8（有结果） | 4 | **1319** | 1345 | 794–1985 | 50 | 4 |

同一个 agent 循环下，DeepSeek 每题的模型时间约为 AutoRound 的 1/4、EXL3 的 1/6；
DeepSWE 的容器启动（153 s 中位）明显比 Pro V2（3 s，镜像已缓存）慢。

Pro V2 逐题 agent 用时（✅/❌ 为结果）：

| 任务 | DeepSeek | AutoRound | EXL3 |
|---|---:|---:|---:|
| `ansible-0ea40e` | ✅ 50s | ✅ 681s | ✅ 648s |
| `ansible-0fd887` | ✅ 116s | ✅ 379s | ✅ 436s |
| `ansible-106909` | ✅ 98s | ✅ 564s | ✅ 1130s |
| `ansible-11c177` | ✅ 31s | ✅ 333s | ✅ 359s |
| `ansible-40ade1` | ✅ 870s | ✅ 1098s | ❌ 2980s |
| `ansible-b748ed` | ✅ 212s | ❌ 688s | ✅ 1457s |
| `ansible-c1f2df` | ✅ 165s | ❌ 539s | ❌ 1184s |
| `ansible-d33bed` | ✅ 271s | ✅ 1581s | ✅ 2531s |
| `flipt-2ca5df` | ✅ 1084s | ✅ 1671s | ✅ 2485s |
| `flipt-3d5a34` | ✅ 1131s | ✅ 2165s | ✅ 2494s |
| `teleport-1316e6` | ✅ 496s | ✅ 1172s | ✅ 1151s |
| `openlibrary-00bec1` | ✅ 89s | ✅ 312s | ✅ 306s |
| `openlibrary-03095f` | ✅ 82s | ✅ 405s | ❌ 615s |
| `openlibrary-08ac40` | ✅ 161s | ✅ 626s | ✅ 1029s |
| `openlibrary-09865f` | ✅ 131s | ✅ 462s | ✅ 789s |
| `openlibrary-0dc5b2` | ✅ 211s | ✅ 795s | ✅ 1173s |
| `openlibrary-322d7a` | ✅ 590s | ✅ 1560s | ✅ 2721s |
| `openlibrary-53d376` | ✅ 305s | ✅ 849s | ✅ 1351s |
| `qutebrowser-01d1d1` | ✅ 191s | ✅ 856s | ✅ 1200s |
| `qutebrowser-0833b5` | ✅ 29s | ✅ 222s | ✅ 332s |

DeepSWE 逐题用时：

| 任务 | agent | verifier | 结果 |
|---|---:|---:|---|
| arcane-drift-detection-baselines | 791s | 196s | ✅ |
| bandit-structured-nosec-directives | 544s | 31s | ❌ |
| effect-sse-httpapi-streaming | 1710s | 35s | ❌ |
| go-critic-doc-link-checker | 636s | 39s | ✅ |
| koota-entity-snapshot-rollback | 333s | 23s | ✅ |
| kysely-window-grouping-helpers | 548s | 57s | ✅ |
| langchain-request-coalescing | 661s | 48s | ✅ |
| meriyah-explicit-resource-declarations | 776s | 96s | ❌ |
| true-myth-iterable-collection-combinators | 260s | 30s | ✅ |
| vitest-duration-sharding | 461s | 65s | ✅ |

墙钟总计：Pro V2 DeepSeek 42 min（`-n 3`）、AutoRound 20:33–22:40（含 21:41 磁盘事故与
重启后补跑）、EXL3 16:47–19:59（含中断与补跑）、DeepSWE 59 min（`-n 3`）。
原始数据：`/home/hong/bench/timings.json`（由 `_dev/probe/_timings.py` 生成）。

## 4. 分析

1. **Pro V2 对强模型已偏饱和**：DeepSeek 20/20，AutoRound 18/20，EXL3 17/20。
   这套 20 题子集仍能拉开 DeepSeek 与本地 3–5bpw 量化的差距，但同家族量化之间差异很小：
   AutoRound 18 vs EXL3 17，而 EXL3 的 17 里还有 1 题是评测 bug（修正后 18/20），
   两者在 Pro V2 上实质持平。
2. **共同难点**：`ansible-c1f2df`（F5 message routing 的边界行为）EXL3 与 AutoRound 都失败，
   说明它更像"量化模型的共同能力边界"而非某个权重的偶然失误。
3. **DeepSWE 的区分度更好**：DeepSeek 70% 的成绩没有天花板效应，且失败集中在"多文件 + 行为测试"
   的真实工程改造上；f2p 88.6% 说明多数失败是"差一点"，适合做细粒度对比。
4. **Pro V2 的失败集中在 ansible / openlibrary**：这些题参考解 4–6 文件、几百行，
   要求同时改公共 API 与内部实现；与之前 S-40 上"多文件修复只做一半"的失败模式一致。
5. **预算仍然是前置条件**：两个测试集都用 xhigh + 30k + step 200；若回到 1536/60，
   预计会重新出现大量"思考截断/步数上限"死亡，分数不可比。

## 5. 工程记录（踩坑与修法）

- **Docker Hub 不可用**：`starryzhang/*` 镜像拉取会挂起（0 字节/秒），SWE-bench-Live
  （500 题 verified 子集已建好）因此无法评测；GHCR 与 ECR public 正常。
- **Harbor 需要 buildx**：`docker compose down --rmi local` / egress sidecar 构建报
  `unknown flag: --file`，原因是 `docker buildx` 插件缺失；把 buildx 二进制放到
  `~/.docker/cli-plugins/docker-buildx` 后解决。
- **容器内 agent 安装依赖 PyPI**：不稳定时改用 Harbor 的 `host_mini_swe:HostMiniSweAgent`，
  让 LLM 循环跑在宿主机、容器只执行命令，彻底避开这个问题。
- **补丁捕获有盲区**：`HostMiniSweAgent` 用 `git add -A && git diff --cached` 收补丁，
  agent 若自己 `git commit`，抓到的就是空补丁（`openlibrary-03095f` 即此情况）。
  修法：在 `setup()` 里记下 base commit，结束后用 `git diff --cached <base>` 捕获
  （能同时覆盖已提交和未提交改动）；并避免 agent 改写/新建与隐藏 test patch 同名的文件。
- **Responses API 需要 `mcp`**：litellm 在 tools + Responses 路径下会 import `mcp`，
  Pier 的默认安装没有它 → 全部 trial 出现 `model.patch` 0 字节。用
  `--ak extra_python_packages=["mcp"]` 解决；Harbor 侧则用自定义 agent 在 install 命令里
  追加 `--with mcp`。
- **容器访问模型端点**：recipe 默认把 8000 发布在 `127.0.0.1`，容器经 docker bridge
  访问不到；改 `HOST=0.0.0.0` 重新发布（或 host agent 模式直接走 `127.0.0.1`）。
- **KV 约束**：单卡 170HX 上 recipe KV 约 1.02M tokens；每卡并发限制 2，避免长上下文
  互相抢占导致 preemption。

## 6. 复现

```bash
# Pro V2（A 组，:9393；B 组把 -i 列表换成另一半、端口改 9402）
cd /home/hong/SWE-bench_Pro-os
OPENAI_API_KEY=EMPTY OPENAI_API_BASE=http://127.0.0.1:9393/v1 \
MSWEA_API_KEY=EMPTY MSWEA_COST_TRACKING=ignore_errors \
PYTHONPATH=/home/hong/bench/.venv/lib/python3.12/site-packages:v2/tooling \
harbor run -p datasets/pro20 $(cat datasets/pro20/idsA.txt | sed 's/^/-i /') \
  -e docker -a host_mini_swe:HostMiniSweAgent -m openai/Qwen3.8-Flash-Next \
  --ak config_file=v2/tooling/configs/host_xhigh_200.yaml --ak step_limit=200 \
  -n 2 --job-name pro20-ar1 -y

# DeepSWE（seed 0 的 10 题）
cd /home/hong/deep-swe
OPENAI_API_KEY=sk-... OPENAI_API_BASE=https://api.deepseek.com/v1 \
MSWEA_API_KEY=sk-... MSWEA_COST_TRACKING=ignore_errors \
pier run -p tasks -e docker -a mini-swe-agent -m openai/deepseek-flash \
  --ak config_file=/home/hong/deep-swe/pier_ds_200.yaml \
  --ak 'extra_python_packages=["mcp"]' \
  --ae OPENAI_API_KEY=sk-... --ae OPENAI_API_BASE=https://api.deepseek.com/v1 \
  --n-tasks 10 --sample-seed 0 -n 3 --job-name dswe-ds3 -y
```

## 7. 产物

| 内容 | 路径 |
|---|---|
| Pro V2 · DeepSeek 20 题 | `~/SWE-bench_Pro-os/jobs/pro20-ds/` |
| Pro V2 · EXL3 前 14 题 | `~/SWE-bench_Pro-os/jobs/pro20-exl3/` |
| Pro V2 · EXL3 补跑 6 题 | `~/SWE-bench_Pro-os/jobs/pro20-exl3-rest2/` |
| Pro V2 · AutoRound A/B 组（首轮） | `~/SWE-bench_Pro-os/jobs/pro20-ar1/`、`pro20-ar2/` |
| Pro V2 · AutoRound 补跑（重启后 5 题） | `~/SWE-bench_Pro-os/jobs/pro20-ar1b/`、`pro20-ar2b/` |
| DeepSWE · DeepSeek 10 题 | `~/deep-swe/jobs/dswe-ds3/` |
| DeepSWE · AutoRound（双卡并发 + 补跑） | `~/deep-swe/jobs/dswe-ar-h1/`、`dswe-ar-h2/`、`dswe-ar-e1/`、`dswe-ar-e3/`、`dswe-ar-e4/` |
| 20 题子集与镜像清单 | `~/SWE-bench_Pro-os/datasets/pro20/ids.txt` |
| 任务脚本 | `agent_bench/make_pro_subset.py`、`_dev/probe/_pull_pro20.sh`、`_dev/probe/pro_agent.py`、`_dev/probe/_deepswe_summary.py` |

## 8. Caveats

- 每个配置单次运行；Pro V2 20 题与 DeepSWE 10 题都是小子集，比例差 1–2 题不构成结论。
- Pro V2 的 20 题按仓库分层挑选，偏向 ansible/openlibrary/qutebrowser，不代表 642 题整体。
- AutoRound 在 21:41 遭遇宿主机 **D: 盘掉线**（KIOXIA SSD 从系统消失，Event ID 51），
  两个模型服务与 WSL 同时挂掉；重启后磁盘恢复，补跑了剩余 5 题，结果完整（18/20）。
- DeepSWE 的 AutoRound 结果有两个失败是 **262k 上下文超限**（effect、meriyah），
  属于协议限制而非能力；且 meriyah 首轮还叠加了 Harbor 的 artifact collision。
  把 `max_tokens` 从 30000 降到 16000 重跑**没有解决**：历史会增长到填满剩余预算
  （16k 时停在 246,145 + 16,000 = 262,145，比上限多 1 token）。要跑完必须做历史缩减
  （strip thinking / 截断旧 observation / 摘要压缩），本次未实施，按协议失败记录。
- DeepSWE AutoRound 期间还有两次人为/环境中断：清理容器时误删模型服务、代理 502、
  以及一次为"重平衡"杀掉了在跑的 e2（langchain/meriyah 各浪费约 7 分钟），均已重跑。
- EXL3 的 Pro V2 一轮曾被误杀（清理容器时删了模型服务），已补跑；前 14 题中 3 题因中断
  重跑过，不影响已记录结果；其中 `openlibrary-03095f` 的 0 字节补丁是评测侧捕获 bug，
  用 commit-aware 捕获重跑后 EXL3 应为 18/20。

## 9. 追加：Maya-M（GLM-5.3-Flash 2-bit，本机 2×CMP 170HX）DeepSWE 10 题（2026-10-26）

> 新臂：**Project Maya v1.1.27 + Maya-M**（自持 GGUF 量化，IQ2_S/IQ3_XXS，116 GB），
> 两张 CMP 170HX 64G 各占一半层、`--max-context 262144`、`--parallel 1`。
> 走本地 LLM 代理 `http://127.0.0.1:12346/v1`（OpenAI 兼容，`/v1/models` 报 `glm-5.3-flash`、`n_ctx 262144`）。
> 与前面几轮**同题集（seed 0 的 10 题）、同 harness（Harbor + host 侧 mini-swe-agent）、同协议**
> （温度 0、`max_tokens=30000`、`reasoning_effort=high`、`step_limit=200`）。
> 唯一差异：为了稳定，本臂用**单并发**（每请求 55–70 tok/s），总耗时 **5 h 11 min**（≈43 min/题）。

### 9.1 配置

| 项 | 值 |
|---|---|
| 模型 | Maya-M（GLM-5.3-Flash，2-bit 量化），OpenAI 兼容端点 |
| agent | `deep_host_agent:CommitHostMiniSwe`（宿主机侧跑 LLM 循环；结束时 `git add -A && git commit`，供 collect 钩子 `git diff base HEAD` 取补丁） |
| agent 配置 | `v2/tooling/configs/host_maya_12346.yaml`：`api_base=http://127.0.0.1:12346/v1`、`timeout=3600`、去掉 Qwen 专属 `chat_template_kwargs` |
| 协议 | 温度 0、`max_tokens=30000`、`reasoning_effort=high`、`step_limit=200` |
| 并发 | `n_concurrent_trials=1`（`--parallel 1`） |
| 超时 | 请求 3600 s；`--agent-timeout-multiplier 2`、`--verifier-timeout-multiplier 2` |
| 结果 | **3/10（30%）**；f2p **721/790 = 91.3%**；0 errored / 0 上下文超限 |

### 9.2 结果（10 题，job `maya-deepswe7`）

| # | 任务 | reward | f2p | p2p | 工具调用 | 失败形态 |
|---|---|---:|---:|---:|---:|---|
| 1 | arcane-drift-detection-baselines | **1** | 82/82 | 2/2 | 128 | — |
| 2 | bandit-structured-nosec-directives | 0 | 68/69 | 281/282 | 115 | 近失 + 回归 |
| 3 | effect-sse-httpapi-streaming | 0 | 42/47 | 70/70 | 200 | 部分实现 |
| 4 | go-critic-doc-link-checker | 0 | 2/3 | 15/16 | 93 | 近失 + 回归 |
| 5 | koota-entity-snapshot-rollback | 0 | 83/84 | 47/47 | 90 | 近失 |
| 6 | kysely-window-grouping-helpers | 0 | 250/254 | 22/22 | 111 | 近失 |
| 7 | langchain-request-coalescing | **1** | 50/50 | 232/232 | 104 | — |
| 8 | meriyah-explicit-resource-declarations | 0 | 48/49 | 51 469/51 469 | 126 | 近失 |
| 9 | true-myth-iterable-collection-combinators | **1** | 96/96 | 561/561 | 70 | — |
| 10 | vitest-duration-sharding | 0 | **0/56** | 0/24 | 111 | **构建被改坏** |

### 9.3 三个臂的对照（同题集）

| 臂 | 分数 | 是否撞 262k 上下文墙 | 备注 |
|---|---:|---|---|
| DeepSeek flash（API） | **7/10** | 否 | 参考上限 |
| AutoRound 3bpw（Qwen3.8-Flash-Next） | **4/10** | 是（effect、meriyah） | 转 2 并发后加深了上下文压力 |
| **Maya-M 2-bit（本机 2×CMP 170HX）** | **3/10** | **否** | 单并发 55–70 tok/s；91.3% 目标测试通过 |

### 9.4 失败解剖：不是"不会写"，是"差最后一口"

- **5 题近失**（1–4 个边界用例）：koota 83/84、kysely 250/254、meriyah 48/49、bandit 68/69、go-critic 2/3。
  DeepSWE 的规则是 f2p 与 p2p 全过才记 1，所以 0.99 的分也记 0。
- **2 题带回归**：bandit 挂 `test_058_region_unioned_across_statement_lines`（f2p 还差 1 个）、
  go-critic 挂 1 个 p2p。**改功能时碰坏既有行为**，是小比特量化模型最典型的一种退化。
- **1 题部分实现**：effect-sse 42/47。对照 DeepSeek 是 **0/47**（完全没做），Maya 反而做得更多。
- **1 题构建被改坏**：vitest 的补丁 73,317 B，但 `src/node/duration-history.ts` 有 `TS9010`
  类型错误 → `pnpm build`（rollup + unplugin-isolated-decl）失败、`dist/cli.js` 不存在、
  测试根本没跑起来。**`f2p 0/56` 且 `p2p 0/24` 就是这个信号**（不是"实现错"，是"没编译过"）。
- **反面证据**：meriyah 的 **51,469 个 p2p 全过**、kysely 250/254、bandit 281/282 ——
  说明它**没有乱改**，只是少做一点。

### 9.5 本轮新踩的坑（与 §5 合并参考）

| 坑 | 现象 | 修法 |
|---|---|---|
| litellm 默认 600 s 请求超时 | Maya 侧 `glm multi: request N done (cancel): 9343 tokens in 588.5 s`；agent 卡在重试退避里 **24 分钟不前进**（轨迹 mtime 冻结、Maya 里只剩 2–3 running） | `model_kwargs.timeout=3600`（`host_maya_12346.yaml`） |
| Docker 地址池耗尽 | 启动后 10 题在 **11 秒内全部** `RuntimeError`：`all predefined address pools have been fully subnetted` | `docker network prune -f`（历史 run 的 compose network 残留了 30 个，清到 3 个） |
| harbor 的 uv 工具环境缺依赖 | `No module named 'host_mini_swe'` / `'minisweagent'` / `'mcp'` | `PYTHONPATH=v2/tooling`；`uv pip install --python ~/.local/share/uv/tools/harbor/bin/python mini-swe-agent==2.4.6 mcp` |
| 宿主侧 `agent/model.patch` 恒为 0 B | 看着像"抓补丁失败" | 正常：`CommitHostMiniSwe` 先 commit，`git diff --cached` 自然为空；**真正的补丁在 `artifacts/logs/artifacts/model.patch`** |
| 并发与长生成的冲突 | 4 并发下每请求仅 13–15 tok/s，9 K token 的步骤要 10 min，必撞 600 s 超时 | 评测改**单并发**（55–70 tok/s），代价是总量 5 h |

### 9.6 Caveats

- **单次运行、10 题**：3/10 的 95% 置信区间约 **7%–65%**，与 4/10、7/10 都不可分；
  也没有做同题配对（McNemar）检验。
- 本臂是 **2-bit 量化**（Maya-M，IQ2_S/IQ3_XXS），与 DeepSeek API、AutoRound 3bpw 不是严格同条件比较。
- 单并发是**评测选择**，不是模型限制；换 `--parallel 2` 能把 10 题压到约 2.5–3 h，但长生成会接近 600 s 线（已用 3600 s 超时规避）。
- `go-critic` 只有 3 个 f2p 测试，单点失败会被放大；若只看 f2p 通过率，应把这题按"3 个样本"加权。

### 9.7 与 3bpw 臂（AutoRound / Qwen3.8-Flash-Next）逐题对照

两臂**同题集、同 harness 家族、同协议**（温度 0、`max_tokens=30000`、`step_limit=200`），差异只有两处：
3bpw 臂是 `xhigh` + **每卡 2 并发**；本臂是 `high` + **单并发**。下表两列都取自各自 job 目录里的
`verifier/reward.json`（3bpw 的数据从 `jobs/dswe-ar-*/` 重新抽取，与 §3.2.1 一致）。

| 任务 | 3bpw reward | 3bpw f2p | 3bpw p2p | Maya reward | Maya f2p | Maya p2p | 差异 |
|---|---:|---:|---:|---:|---:|---:|---|
| arcane-drift-detection-baselines | **1** | 82/82 | 2/2 | **1** | 82/82 | 2/2 | 平 |
| true-myth-iterable-collection-combinators | **1** | 96/96 | 561/561 | **1** | 96/96 | 561/561 | 平 |
| langchain-request-coalescing | 0 | 49/50 | 232/232 | **1** | **50/50** | 232/232 | **Maya 多 1 个测试 → 过** |
| bandit-structured-nosec-directives | 0 | **69/69** | 281/282 | 0 | 68/69 | 281/282 | 都因**回归**挂；AR 目标全过 |
| koota-entity-snapshot-rollback | 0 | 80/84 | 47/47 | 0 | **83/84** | 47/47 | 都近失；Maya 少差 3 个 |
| kysely-window-grouping-helpers | 0 | **0/254** | 22/22 | 0 | **250/254** | 22/22 | **同分，但实现量差 250 个测试** |
| go-critic-doc-link-checker | **1** | 3/3 | 16/16 | 0 | 2/3 | 15/16 | AR 过；Maya 近失 + 回归 |
| vitest-duration-sharding | **1** | 56/56 | 24/24 | 0 | **0/56** | 0/24 | AR 过；Maya **构建被改坏**（TS9010） |
| effect-sse-httpapi-streaming | — | — | — | 0 | 42/47 | 70/70 | AR **无结果**（262k 超限）；Maya 有分 |
| meriyah-explicit-resource-declarations | — | — | — | 0 | 48/49 | 51 469/51 469 | AR **无结果**（262k 超限 + artifact collision）；Maya 有分 |

**① 分数：4/10 vs 3/10，统计上不可分。**
两臂都出结果的 8 题里，Maya 独赢 1 题（langchain）、3bpw 独赢 2 题（go-critic、vitest），
同过 2 题、同挂 3 题。配对 McNemar 的不一致对是 **b=1 / c=2**，精确双侧 **p = 1.0** ——
10 题的样本量下，这个 1 题之差就是噪声。3bpw 多的那 1 分，还正好来自 **2 题上下文超限** 的免除。

**② 实现广度：Maya 明显更高，但被两件事抵消。**
8 道共同题的 f2p 合计：**3bpw 435/694（62.7%）** vs **Maya 631/694（90.9%）**。
差距几乎全部来自 `kysely`：3bpw 是 **0/254**（完全没实现），Maya 是 **250/254**。
同时 Maya 在 `koota`（+3）、`langchain`（+1，且因此从 0 变成 1）上也更好；
只在小样本题上更差（`go-critic` −1、`bandit` −1）。

**③ 反向的例外：`vitest` 是 Maya 唯一"整题归零"的失败。**
3bpw 56/56 全过，Maya 的补丁 73 KB 但有 `TS9010` 类型错误 → `pnpm build` 失败 → `f2p 0/56` 且 `p2p 0/24`。
**这不是"实现得少"，是"没验证自己的改动能编译"**，也是两臂最本质的差别：
3bpw 臂在这个任务上跑通了构建，Maya 没有。

**④ 上下文：Maya 更省。**
同样的 256 K 上限、同样的 30 K 输出预算，3bpw 在 `effect-sse`/`meriyah` 上分别涨到
`232,145+30,000` 和 `246,145+16,000` 撞墙；Maya 这两题的 prompt 峰值只到 **185 K / 140 K** 就收工
（`step_limit=200` 先到），因此**两题都有分**。原因可能是 Maya 每步输出更短、或更早收敛，
这一点值得用轨迹长度做单变量对照，本轮未做。

**⑤ 一句话结论**：在这套 10 题上，**2-bit 的 Maya-M 与 3-bit 的 AutoRound 打平（3 vs 4，p=1.0）**；
Maya 的**功能实现更完整**（f2p 90.9% vs 62.7%，主要靠 kysely），但**工程自检更弱**（vitest 编译都不过），
且 3bpw 的 2 个失败是协议超限而非能力。要分出高下，需要 ① 把 262 K 墙去掉重跑 3bpw 那两题，
② 对 5 道近失题做配对重跑（每臂 2–3 次）。

### 9.8 产物

| 内容 | 路径 |
|---|---|
| DeepSWE · Maya-M 10 题（本文件第 9 节） | `~/deep-swe/jobs/maya-deepswe7/` |
| 题目录（每题含 `agent/`、`artifacts/`、`verifier/`） | `~/deep-swe/jobs/maya-deepswe7/instance_*` → 现为 `<task>__<suffix>/` |
| job 配置 | `~/deep-swe/maya-deepswe7.json` |
| agent 配置 | `~/SWE-bench_Pro-os/v2/tooling/configs/host_maya_12346.yaml` |
| 早前 4 并发试跑（作废，供对照） | `~/deep-swe/jobs/maya-deepswe4/`、`maya-deepswe5/`（`maya-deepswe6` 为解析配置试跑） |

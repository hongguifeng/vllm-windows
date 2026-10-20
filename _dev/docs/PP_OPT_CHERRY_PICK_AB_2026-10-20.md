# PP 优化补丁落地与 A/B（2026-10-20）

分支 **`pp-opt`**（从 `flash-next-win` @ `965b61d55c` 拉出）。目标：把
`FORK_CMP170HX_IMPROVEMENTS.md` §5 Tier 3 的 PP scheduling 补丁落到本树，
先跑 CPU 单测，再做 PP=2 的 c=1/c=4 A/B。

## 1. 落地情况

| 分支上的 commit | 原始 commit | 开关 | 结果 |
|---|---|---|---|
| `8fa8fb1ede` | `6a0dd75b52` | 无（correctness） | ✅ 已落 |
| `2a0226b0fb` | `5fa416150a` | `VLLM_PP_SPREAD_DECODES` | ✅ 已落 |
| `a312e8fb19` | `666f36dff7` | `VLLM_PP_PACKED_HOP` / `..._HOP_NO_METADATA` | ✅ 已落（改了 4 处类型/测试写法过 mypy） |
| `ef514f3562` | `3de0edb3f6` | `VLLM_PP_SPLIT_DRAFT_EVENT` | ✅ 已落 |
| — | `49753f11bc` | `VLLM_PP_DRAFT_TAIL_STAGE` | ❌ **未落**（原因见下） |
| — | `378c37b009` | （上者的 follow-up） | ❌ 一并不落 |

`49753f11bc` 跳过原因（有实证）：

1. 它依赖 morrow 分支的 **DFlash2** speculator 字段 `candidate_sampler` / `top_k`，
   本树的 `DFlash2Speculator` 没有（mypy 直接报 4 处 `attr-defined`）。
2. 它改的 `vllm/ampere_thin_gemm/thin_gemm.py` 在 HEAD 里**根本没有跟踪**（工作区那份是
   之前部署留下的未跟踪文件），commit 里还带 `mark_drafter_aux_fc_folded` / `aux_fc_folded`
   这类本树不存在的符号。
3. 它触发仓库的 `check-torch-cuda-call` 钩子（2 处 `torch.cuda` 调用）。
4. **我们的 drafter 是 Qwen4Exp MTP，不是 DFlash2**，`VLLM_PP_DRAFT_TAIL_STAGE` 本来也用不上。

冲突处理方式：`envs.py` 取 ours 再手工加本 commit 的变量/lambda（morrow 的 GLM5 env 块不引入）；
`model_runner.py` 手工合并 `remote_tail` 分支，去掉本树没有的 `aux_fc_folded` 两行。
所有开关默认 off，PP=1 生产路径位级不变。

## 2. CPU 单测

```powershell
$env:PYTHONPATH='D:\code\vllm-windows'
.flashnext-root\.venv\Scripts\python.exe -m pytest `
  tests/v1/core/test_pp_spread_decodes.py `
  tests/v1/worker/test_pp_deferred_block_tables.py `
  tests/v1/worker/test_pp_hop.py `
  tests/v1/worker/test_pp_split_draft_event.py `
  tests/v1/worker/test_pp_utils.py -q
```

**28 passed**（18.25 s）。新增的 4 个测试文件全绿。

## 3. A/B 协议

- PP=2 + CUDA graph，8192 ctx，两个 170HX 都在 CPU x8 槽（Gen2 x8），
  `CUDA_VISIBLE_DEVICES=1,2`、`VLLM_PLE_ALLOW_PP=1`。
- 同一晚、背靠背重启；每臂只改环境变量。
- c=1：`_longgen.py`（数数 prompt，256 tok，MTP）。
- c=4：`_conc4_decode.py`（4 请求 ×128 tok，短 prompt，唯一 salt，读 `usage.completion_tokens`；
  比 `_bench_port.py` 更适合 MTP，因为后者把多 token 的 chunk 当成 1 个 token 计数）。
- 先做 no-MTP 的 c=4：每 chunk = 1 token，调度对照最干净。

## 4. 结果

### 4.1 no-MTP c=4（干净对照）

| 臂 | 6 次 aggregate（tok/s） | 中位 | 每请求中位 |
|---|---|---|---|
| flags off | 79.7 / 83.0 / 82.5 / 84.0 / 84.7 / 86.6 | 83.5 | 21.1 |
| `SPREAD_DECODES=1` | 86.2 / 87.2 / 88.4 / 86.6 / 86.4 / 87.9 | **86.9** | 21.8 |

**+4.1%（中位）/ +4.4%（均值）**。6 次里 spread 全部 ≥86.2、baseline 全部 ≤86.6，
方向稳定但幅度很小。

### 4.2 MTP c=4（生产口径）

| 臂 | aggregate 样本（tok/s） | 中位 |
|---|---|---|
| flags off（两轮共 14 个） | 101.3-222.2，**双峰**（~110 与 ~200+） | ~149 |
| `SPREAD_DECODES=1`（8 个） | 101.3-129.4，**全部在慢档** | ~123 |

MTP 下 baseline 会间歇进入 ~200+ 的快档，而 spread 一次都没进。
**生产口径下 spread 没有收益，反而压掉了快档。** 推断：batch=4 的 MTP drafter/GEMM
形状比 batch=2 更高效，spread 是拿 batch 效率换流水重叠，在 MTP 下不划算。

### 4.3 c=1

`SPREAD_DECODES` 在 c=1 是 no-op（cap = ceil(1/2) = 1）。实测 off 67-81、spread 70-73，
均在噪声内；`SPLIT_DRAFT_EVENT` 在 c=1 也没有可测收益（68.8 vs 70-78，偏慢但在噪声内）。

### 4.4 另外两个 flag

| 开关 | 结果 |
|---|---|
| `VLLM_PP_PACKED_HOP=1` | **起服务即挂**：graph capture 后反复 `No available shared memory broadcast block found in 60 seconds`，两个 worker 卡死；去掉它（spread+split）能正常起。 |
| `VLLM_PP_SPLIT_DRAFT_EVENT=1` | 与 spread 同开：c=4 中位 ~115、c=1 ~68.8；单独贡献无法从 MTP 噪声里切出来，**没有正收益证据**。 |
| `VLLM_PP_HOP_NO_METADATA` | 依赖 PACKED_HOP，未单独测。 |

## 5. 结论与建议

1. **4/5 补丁已落在 `pp-opt`，28 个 CPU 测试全绿**；第 5 个（DFlash2 draft tail）不适用于本树。
2. **没有一个 flag 值得在生产 MTP 配置里打开**：
   - `SPREAD_DECODES` 只在 no-MTP c=4 上 +4%，MTP 下反而压掉快档；
   - `SPLIT_DRAFT_EVENT` 无可测收益；
   - `PACKED_HOP` 起服务即死锁；
   - `DRAFT_TAIL` 是 DFlash2 专用。
3. 如果还要继续，优先级是：
   - 修 `PACKED_HOP` 的死锁（唯一可能直接砍每步 latency 的改动）；
   - 给 `SPREAD_DECODES` 加折中策略（例如只在 decoding 数 > 2×pp_size 时 cap，
     避免把 batch=4 拆成 batch=2）；
   - `SPLIT_DRAFT_EVENT` 单独做一轮 MTP c=1 的 20 次重复，确认它是否真的偏慢。

## 6. 复现

```powershell
git switch pp-opt
$env:PYTHONPATH='D:\code\vllm-windows'
$env:VLLM_PLE_ALLOW_PP='1'
$env:CUDA_VISIBLE_DEVICES='1,2'
# off 臂：不设任何 VLLM_PP_*
# spread 臂：
$env:VLLM_PP_SPREAD_DECODES='1'
& _dev\probe\_pp_tp_smoke.ps1 -Mode pp -Size 2 -MaxLen 8192 -Mtp -LogPrefix ppopt
python _dev\probe\_conc4_decode.py --port 9401 --runs 6
```

分支 `pp-opt` 保留；工作区已切回 `flash-next-win`，所有 flag 默认 off。

# _dev —— 本机开发脚手架（源码进 Git，产物留本地）

Flash-Next 的环境重建与“删除 venv 不丢修复”说明见
[FLASHNEXT_REBUILD.md](docs/FLASHNEXT_REBUILD.md)。
`bin/`、`probe/`、`test/`、`requirements/` 和结论文档是追踪的源码；
`out/`、虚拟环境、DLL/PYD 和缓存是忽略的生成产物。

这一棵是 2026-09-22 从仓库根收拢进来的：原先 60 多个 `_*` 脚本、8 份自建文档和
一堆输出目录全摊在根上，和 vLLM 自己的 `CMakeLists.txt` / `setup.py` 混在一起。
现在根上只留 `_dev/` 一个入口。

**约定：所有脚本都用绝对路径定位，可以从任意 cwd 调用。**

## 目录

| 目录 | 放什么 | 关键文件 |
| --- | --- | --- |
| `bin/` | 日常运维入口（构建 / 冒烟 / 起服的回归脚本） | `_run.ps1`、`_build.ps1`、`_build_nccl.ps1`、`_env.ps1`（MSVC/CUDA/NCCL 环境，被其它脚本 dot-source）、`_verify*.py`、`_deps.ps1`、`_serve_cli_matrix.ps1` |
| `bench/` | 性能基准与对照 | `_bench_suite.py`（主力）、`_ab_bench.py`、`_ab_serve_win.sh`、`_itl_client.py`、`_bench_27b.py`、`_prompts_real.jsonl` |
| `test/` | 功能自测（打已启动的服务） | `_test_longctx.py`、`_test_token_reserve.py`、`_test_tools.py`、`_vision_smoke.py`（448px 小图，验塔能不能跑）、`_vision_pressure.py`（`--px/--count` 发满预算大图，验"跑图后会不会把显存顶过界"；**压视觉必须加 `--unique`**，否则 `--count N` 是同一张图×N，全走 MM 缓存压不到编码器）、`_smoke_batch2.py` |
| `probe/` | 故障排查 / 复现（一次性实验，大多带 `--apply` / `--undo` 注入） | `_repro_async_tower.ps1`、`_repro_topk_*.py`、`_clamp_exp.py`、`_idcheck_exp.py`、`_patch_draft_eager.py`、`_probe_*.py`、`_selftest_*.py`、`_step_probe.py`、**显存/掉速那一套**：`_paging_watch.ps1`（边发请求边采 `Local/Non Local/Shared` + `dmon -s t`，自动发现引擎 pid）、`_procgpumem.ps1`、`_gpumem_counters.ps1`、`_scan_req.py` / `_scan_step.py`（从日志反推 tok/s 与 ms/步）、`_watch_req.ps1`、`_pcie_scale.py`、`_uva_offload_cache_probe.py`（离线证明卸载的权重副本跑一次 forward 就落进 allocator cache 不再释放；纯 torch，不起服）、`_longctx_vision.py`（**验收探针**：单请求同时吃满长上下文 + 一张满预算图，按 `/tokenize` 精确配额后校验 needle 与图内容双答对）、`_check_serve_defaults.ps1`（**改完 `start_server.ps1` 参数块就跑这个**：ParseFile + 五个 DryRun 探测，核对默认档仍是已验证的 71680/5.8e9/ceil8/release、且 `-MaxGraphCapture` 随 `-MaxSeqs` 派生；每个探测走独立子进程）、`_scan_ctrlchars.py`（扫出文档里"被字符串转义吃掉反斜杠"留下的控制字符——`D:\code\vllm-windows\...` 会变成带 `\v`/`\b` 控制符的隐形坏路径；`--fix` 就地修，跳过 `out/`）、**WSL 侧对照**：`_wsl_wait_ready.sh`（轮询容器 health + 采起服显存曲线）、`_wsl_vision_ab.ps1`（WSL 上跑"纯文本 → 读图 → 纯文本"）、`_mmap_portability_probe.py`（判 `--kv-offloading-size` 能否移植到 Windows：逐项验 `mmap.MAP_SHARED`/`madvise`/映射期 `unlink`/`torch.frombuffer`，5 秒出清单）、**flash-next / 9393-9394 那一套（2026-10-03）**：`_flashnext_vision.py`（长上下文 + 一张图的验收，`--max-ctx` 跟服务上限走，按 `/tokenize` 精确配额 ⇒ 测得到 262,144 满信封，而 bench 协议天然测不了）、`_flashnext_multi_vision.py`（多图 + 判读 `--mm-processor-kwargs max_pixels` 是否真生效：不压 4096 tok/图 vs 压后 ~1280）、`_flashnext_kv_saturation.py`（**故意灌满 KV**：`--tokens-list` / `--images` / `--out-tokens`（会同时写成 `min_tokens`，否则模型答 8 tok 就 EOS 立刻放块、永远灌不满）/ `--canary` 在占满时插一条短请求量排队代价；采 `vllm:kv_cache_usage_perc`、抢占与 heal 日志增量）、`_flashnext_multiturn_grow.py`（多轮把历史累积到占满，顺带暴露 prefix cache 命中率只有个位数百分比） |
| `patches/` | 幂等 python 补丁（起服前要跑） | `port_wsl_patches.py`、`fix_winloop_import.py`、`fix_responses_toolcall_none_name.py`、`fix_flashinfer_topk_graph_replay.py`、`fix_offload_release.py`（把 `--cpu-offload-*` 每次 forward 上传的塔副本用完就还给驱动；env 门控，根目录 `start_server.ps1 -ReleaseOffloadCopy` 才生效 —— 不还的话 879 MiB 一进 allocator 就永久占着卡）等；`wsl/` 存 WSL 移植的 .patch |
| `docs/` | 自建结论文档（非上游 docs） | `AB_WSL_VS_WINDOWS.md`、`RUN_QWEN38_27B_W4A16.md`、`BENCH_170HX_BASELINE.md`、`BENCH_3090.md`、`BENCH_FLASHNEXT_GPU1_VS_WSL2_GPU0.md`（Flash-Next 的 Windows/GPU1 vs WSL2/GPU0 对照）、`BUILD_WINDOWS_CMP170HX.md`、`PROBE_STEP_170HX_C1_C2.md`、`WSL_*.md` |
| `out/` | 一切产物：日志、基准集、A/B 结果 | `logs/serve_<ts>.log`、`_bench_base*/`、`_ab_results/`、`_bench_repro/`、`_aot_dump/` |

## 常用调用

```powershell
& D:\code\vllm-windows\start_server.ps1                       # 起服（默认 71680 + async）；脚本在**仓库根**，不在 _dev/bin
& D:\code\vllm-windows\start_server.ps1 -DryRun               # 只打 argv（零副作用，不杀任何进程）
& D:\code\vllm-windows\_dev\bin\_run.ps1               # 0.5B 端到端冒烟
& D:\code\vllm-windows\_dev\probe\_repro_async_tower.ps1 -Boots 3
Get-Content D:\code\vllm-windows\_dev\out\logs\serve_*.log -Wait -Tail 40
```

> 起服脚本 `start_server.ps1` 2026-09-22 从 `_dev/bin/_serve.ps1` 移到仓库根并改名（先叫 `Serve.ps1`，同日又改成现在这个），每次跑不用先 cd。旧名一共出现在 15 个文件里，用 `_dev/probe/_fix_serve_refs.py --apply` 一把刷掉。
> 它内部所有路径都是绝对的，可以从任意 cwd 调用；`_env.ps1`、运行日志位置都没变。

## 改路径时的注意点

- ps1 脚本用 `$REPO`（仓库根）+ `$DEV`（本目录）两个变量，`$DEV` 由 `_env.ps1` 导出；
  `start_server.ps1` / `_run.ps1` / `_verify.ps1` 另外自带一份，因为它们不一定先 dot-source。
  ⚠️ `start_server.ps1` 已不在 `_dev/bin`，别用 `$PSScriptRoot` 猜它旁边是谁。
- py / sh 脚本在文件头算 `HERE`（自己所在目录）、`DEV = HERE/..`、`REPO = DEV/..`。
  别再把 `REPO` 当成"脚本所在目录"。
- 输出一律走 `_dev/out`，`_bench_suite.py` 仍支持 `VBENCH_OUT` 覆盖。
- 补丁脚本要传**包目录**作参数，两份都打（site-packages + 仓库 `vllm/`）。
- 改完 ps1 必跑一次语法解析：
  `[System.Management.Automation.Language.Parser]::ParseFile($path, [ref]$null, [ref]$errs)`

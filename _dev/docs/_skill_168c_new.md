### 16.8c 想在 24 GiB 卡上同时要"长上下文 + 视觉"：先算账，再选"腾非 KV"还是"压 KV"

**先算账，别先调参。** 卡侧账（本项目 27B W4A16 + DFlash2 + 塔，24 GiB 3090，全部实测或离线量过）：

| 项 | 怎么拿 | MiB |
|---|---|---|
| 主模型权重（W4；含视觉塔 879、mtp 312） | 离线读 safetensors header 按 dtype×shape 求和 | 15,060 |
| drafter 权重（DFlash2 W4A16） | 同上 | 1,221 |
| CUDA graphs（`max_cudagraph_capture_size` 64） | 日志 `Graph capturing finished ... took X GiB` | 1,330 |
| 工作区 / 杂项 | 上面几项之外由 `Local` 反推 | ~110 |
| **底账小计（塔还没上卡）** | `Local` 实测 21,727.7 MiB @pin 4.2e9 | **17,721** |
| 视觉塔的上传副本 | 首张图之后**永久** +808~879，除非用完就还（见路线 1） | +879 |
| 视觉编码器自己的工作区 | 首张图之后（1/2/8 张满预算图 = +808/+352/+120，**饱和**） | +~400 |

⇒ pin 6.0e9 时 `Local` 实测 **24,170.5**，已经贴死卡（24,576）⇒ 掉到 104.6 ms/步（§16.9）；
pin 4.2e9 时 `Local 21,727.7` ⇒ 26.6 ms/步。**可动的只有上面加粗那两块 + KV。**

⛔ **算卡侧账时不要把 `Non Local` 加进去**：它是 host-pinned 内存，不是卡内存。
证据：WSL 那个 16 GiB `/dev/shm` KV 池在计数器里显示成 `Non Local 16,484`，
`Local + Non Local` = 40,511 ≫ 24,576，而它 p50 26.5 ms/步完全健康。**判超卡只看 `Local`。**

**KV 容量不是 pin 的线性函数 —— 用拟合，别用除法。** 三个实测锚点（同模型、bf16、`-MaxSeqs 8`）：

| pin / `-MaxLen` | 日志里的容量 |
|---|---|
| 4.2e9 / 48,000 | 48,168 |
| 6.0e9 / 60,928 | 72,899 |
| 6.0e9 / 71,680 | 75,181 |

`tokens ≈ (pin − 1.091e9) / 81,859 + 0.2122 × MaxLen`（三点都在 20 tok 以内）。
⇒ **70000 tok 要 pin 5.7~5.8e9**，不是 `70000 × 78 KB = 5.5e9` 那么乐观；
`79,807 B/token`（= 6e9/75,181）只对那一个 pin 成立，4.2e9 那个实际是 87,195 B/token。
`start_server.ps1` 的 `expectKv` 已按这个公式算，不再写死 75,181。

**路线 1（先走这条，零速度代价）＝ 腾非 KV：**

- **视觉副本用完就还。** `--cpu-offload-gb/--cpu-offload-params` 的非 UVA 分支**每次 forward** 都
  `state_dict().to(device)`，而 vLLM 只在 capture（`v1/worker/gpu/model_runner.py`）和 teardown/sleep
  调 `empty_cache()`，推理循环里从不调 ⇒ 那 879 MiB 一进 allocator 就再也不还给驱动
  （离线复现：`_dev/probe/_uva_offload_cache_probe.py`，reserved 0 → 98 MiB → +0 → +0）。
  补丁 `_dev/patches/fix_offload_release.py` + 起服开关 `-ReleaseOffloadCopy` 在 forward 返回前
  `del device_state; torch.cuda.empty_cache()`；代价 = 下次读图多一次 cudaMalloc（塔本来就每次 re-upload，
  PCIe 流量不变）。
- **降图捕获上界** `-MaxGraphCapture 8`：列表生成式是 `[1,2,4] + range(8,256,8) + range(256,N+1,16)`，
  而 `-MaxSeqs 8` 根本排不出 >8 的 decode 批 ⇒ 16~64 那几档只服务 16~64 token 的混合 prefill 步。
  降下来能还回一部分 1,330（看日志那行实际降到多少，别只看预期）。
- 两块合起来 ≈ 1.3~2.2 GiB ⇒ **bf16 的 70000 + 视觉成立，而且完全不碰 dtype。**

**路线 2（激进，有速度代价）＝ 压 KV。** 先读 `config.json` 的 `layer_types` 判断有多少层真在用 KV：
本模型 64 层 = **48 层 GDN（`linear_attention`）+ 16 层 full_attention**，`head_dim=256 / 24 heads / 4 kv`
⇒ 池里 **82% 在那 16 层 full attention** 上，可以量化；48 层 GDN 走 mamba state，不受 KV dtype 影响。

`--kv-cache-dtype` 与 backend **必须配对**（0.29 合法值见 `config/cache.py` 的 `CacheDType`）：

| dtype | 必需 backend | 依据 |
|---|---|---|
| `bfloat16` | `FLASH_ATTN` | 基线 |
| `int8_per_token_head` / `int4_` / `fp8_per_token_head` | **`TRITON_ATTN`** | 量化器在 `v1/attention/backends/triton_attn.py`，`_is_per_token_head_quant` |
| `turboquant_k3v4_nc` / `turboquant_4bit_nc` / `turboquant_k8v4` | **`TURBOQUANT`** | `TurboQuantAttentionBackend`；prefill 走标准 SDPA，decode 走 Triton（FlyDSL 那条是 AMD gfx950 专用） |

**⚠️ fp8 KV 在本项目的 sm86 卡上不要碰**：`FLASH_ATTN` 拒绝（要 FA3/SM90+）、`TRITON_ATTN` 拒绝（要 SM89+），
只剩 FlashInfer 一条路，而 WSL 侧 `#34` 记录过 3090 上 fp8+MTP+prefix cache 在 28–34k 出现**确定性 Xid-31 MMU write-fault**。

**int8 档的实测代价（WSL gotcha 40，同模型）**：池容量 **2×**（136,429 vs bf16 69,758 tokens @ 5.2 GiB）
= ~40 KB/token ⇒ 71680 tok 只需 ~2.8 GiB；但 **60k 时 -34% decode / -44% prefill**，
chat 长度只 +2.5% e2e；prefill 112k 文档 251 s vs FLASH_ATTN ~112 s。**长上下文才是它的代价所在，别只看容量。**

**⚠️ 压了 dtype 必须同时降 `--kv-cache-memory-bytes`** —— 这个 pin 按**字节**算，留 6e9 就是花同样的显存
换 2× 容量，一点没省。

**本项目 `start_server.ps1` 已就绪的开关**：`-KvDtype` / `-AttnBackend`（留空自动配对）、
`-MaxGraphCapture` / `-ReleaseOffloadCopy`。`int8` 档无需任何新补丁 —— `spec-decode-attn`（split-KV verify kernel）
与 `prefill-attn-int8`（gate 正是 `24 heads / 4 kv / 256`）都已在 `port_wsl_patches.py` 的 CARRIED_B 里应用，
且脚本本就默认 `VLLM_SPEC_DECODE_ATTN=1`。

```powershell
# 首选：bf16 的 70000 + 视觉，零速度代价（先停掉 WSL 容器腾出卡）
.\start_server.ps1 -Force -MaxLen 70000 -KvBytes 5800000000 -MaxGraphCapture 8 -ReleaseOffloadCopy
# 备选：压 KV 换容量，接受长上下文变慢
.\start_server.ps1 -Force -MaxLen 71680 -KvDtype int8_per_token_head -KvBytes 3200000000
```

**⚠️ 起服守卫有过单位 bug（2026-09-22 已修）**：`$free` 是 MiB（`nvidia-smi --nounits`）却被直接和
GiB 的 `$need` 比 —— `24000 < 25.3` 永远为假，所以这个守卫**从来没生效过**，6 GiB/71680 那次 thrash
起服时一句警告都没有。现在两边都折成 GiB；卡侧非 KV 基数按 **17.8 GiB（塔未上卡）+ 0.9 GiB（塔副本）**，
`-ReleaseOffloadCopy` 时只校验稳态那一份，并单独提示读图瞬间的峰值。


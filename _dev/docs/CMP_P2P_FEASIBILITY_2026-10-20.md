# 两张 CMP 170HX 能不能开 PCIe P2P（2026-10-20 实测）

**结论：这台机器上不能，而且不是"改个配置"就能解决的。** 三层证据独立指向同一个答案，其中一层是卡级
（NVML 报 `GPU_NOT_SUPPORTED`），不是 Windows 层面能绕过的。

复现：`.\.venv\Scripts\python.exe _dev\probe\_p2p_capability_probe.py --bench`

---

## 1. 三层证据

| 观察面 | 结果 | 说明 |
|---|---|---|
| CUDA 运行时 | `cudaDeviceCanAccessPeer` **6/6 全 0**；`cudaDeviceEnablePeerAccess` **6/6 返回 217** `cudaErrorPeerAccessUnsupported` | 应用层能问到的最终答案。用 torch 的 `can_device_access_peer` 复核，同样全 0 |
| NVML（能说出原因） | CMP↔CMP：READ/WRITE = **`GPU_NOT_SUPPORTED`**；3090↔CMP：**`NOT_SUPPORTED`** | 关键区别：两张 CMP 之间被判定为**卡本身不支持**，不是"芯片组/拓扑不支持"（`CHIPSET_NOT_SUPPORTED`），也不是"被注册表关掉"（`DISABLED_BY_REGKEY`） |
| 拓扑 API | `nvmlDeviceGetTopologyCommonAncestor` rc=2（`NVML_ERROR_NOT_SUPPORTED`） | Windows 上拓扑矩阵不可用，这就是 `nvidia-smi topo -m` 直接报 "Failed to run topology matrix" 的原因；Linux 上判断 P2P 的第一步在这里做不了 |

注册表侧也没有可拨的开关：`HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Parameters` 下只有
`DmaRemappingCompatible=3`、`WDDMv22EnableSmoothTransition=0` 等，**没有任何 P2P 相关键**。

> 诚实边界：Windows 上无法区分 `GPU_NOT_SUPPORTED` 是「挖矿 SKU 的 P2P 能力位被熔断」还是「MCDM
> 驱动模型本身不提供 P2P」。两者都要到裸机 Linux 上才能分辨。但无论哪一种，Windows 侧都没有对应
> 的解锁路径。

## 2. 链路与拓扑事实

| 项 | 值 | 来源 |
|---|---|---|
| 卡 | 2× CMP 170HX 64G（GA100，sm_80，63.6 GiB）+ 1× RTX 3090（对照，x4） | `nvidia-smi` |
| 驱动 | 610.88 / CUDA UMD 13.3 / NVML 13.610.88 | `nvidia-smi`、`nvmlSystemGetDriverVersion` |
| 驱动模型 | **CMP = MCDM**（compute-only），3090 = WDDM | `nvidia-smi --query-gpu=driver_model.current` |
| PCI 位置 | `0B:00.0` / `0C:00.0`，父节点 `00:03.1` / `00:03.2`（AMD Matisse GPP bridge，CPU 直连两个 x8 槽） | PnP `DEVPKEY_Device_Parent` |
| 链路 | `nvidia-smi` 报 **Gen2 x8**（两卡一致）；Windows PnP 报 Gen1 x8 | 两处读数不一致，见下 |
| 实测带宽 | **3.22 GB/s** @256 MiB（pinned 中转，双向重叠） | `[实测]` |
| 虚拟化 | VBS 运行中（`VirtualizationBasedSecurityStatus=2`，hypervisor present） | `Win32_DeviceGuard` |

链路代次以实测为准：Gen2 x8 = 4 GB/s 线速，实到 3.22 GB/s 即 80%，符合；Gen1 x8 只有 2 GB/s，被
实测排除。Windows PnP 的 "Gen1 / max Gen1" 是陈旧读数（`DEVPKEY_PciDevice_*LinkSpeed`）。另外
VBS/hypervisor 在跑意味着 IOMMU 与 DMA remapping 由 hypervisor 掌管，P2P 需要的 DMA 直通没有任何
配置入口。

## 3. 没有 P2P 的现状代价（这才是"开不了"要付的账）

`cuda:{1} -> cuda:{2}`（两张 CMP），pinned 中转 = 显式 D2H + H2D 异步拷贝，即 NCCL SHM/CE transport
走的同一条路：

| 尺寸 | `copy_` 直连 | pinned 中转 | 中转有效带宽 |
|---|---|---|---|
| 4 KiB | 1.19 ms | **0.033 ms** | 0.12 GB/s |
| 64 KiB | 0.25 ms | 0.054 ms | 1.21 GB/s |
| 1 MiB | 0.93 ms | 0.355 ms | 2.96 GB/s |
| 16 MiB | 7.82 ms | 6.59 ms | 2.55 GB/s |
| 256 MiB | 99.7 ms | 83.3 ms | **3.22 GB/s** |

两点值得记住：

1. **torch 的直接跨卡 `copy_` 反而更慢**（4 KiB 要 1.19 ms）—— 它在无 peer access 时的每调用开销
   远高于自己走 pinned 中转。所以"少写代码直接 copy"在这台机器上是反向优化。
2. **小消息的中转成本 ~33 µs**，这是本条路径的物理下限，和 NCCL 报的 4 KiB send/recv 0.10 ms 同量级
   （`_dev/docs/PP_VS_TP_WINDOWS.md` §2.5：20 MiB 6.64 ms ≈ 3.0 GB/s，all_reduce 16 MiB 8.69 ms）。

## 4. 就算解锁了 P2P，值不值

不值，理由是算术而非猜测：

- **带宽上限**：Gen2 x8 = 4 GB/s。现在中转已经跑到 3.22 GB/s，P2P 最好也就 +25%，拿不到数量级。
  延迟侧空间大一些（4 KiB 的 33 µs 有望降到 10 µs 量级），但它换不来吞吐。
- **TP 的死结不在 P2P**：`2048 × 2560 × 2 B ≈ 10 MiB` 的 hidden state，TP=2 每层过一次线、48 层
  ≈ 480 MiB/chunk。按 4 GB/s 算 ≈ **120 ms/chunk**；PP=2 只有 1 个边界、10 MiB ≈ **3 ms**（实测口径
  20 MiB 6.64 ms）。P2P 把 120 ms 变成 120 ms —— 比例不变，**P2P 救不了 TP**。
- **PP 根本不需要 P2P**：它要的是 10 MiB 边界和 5 KiB decode 消息，`SHM/CE` 已经够（实测边界 6.6 ms /
  0.1 ms）。所以本机的正解仍然是 PP，P2P 对当前方案零收益。

## 5. 如果一定要试：唯一路径与判读矩阵

只有**裸机 Linux** 一条路（WSL2 不行：它走 dxgkrnl 的 GPU 半虚拟化，peer access 必然为 0，既有记录
也印证了这点）。代价是独占机器 + 装系统 + 掉卡风险，需要的三件事：

1. `cmpunlocker --p2p --no-iommu` 之类 Linux 驱动级解锁（挖矿 SKU 的惯用手段，属未支持 hack）；
2. IOMMU 关掉（`amd_iommu=off`，或至少 `iommu=pt` + ACS 覆盖）；
3. ACS 覆盖（`pcie_acs_override=downstream,multifunction`），因为两张卡在**两个不同 root port**
   （`00:03.1` / `00:03.2`）上，P2P 要穿过 AMD 根复合体。

判读矩阵（按顺序，任一步失败即终止）：

| 观察 | 判定 |
|---|---|
| `nvidia-smi topo -m` 显示 `SYS`，或 `topo -p2p r` 报 `Chipset Not Supported` | 平台层不支持，结束 |
| 拓扑 `PHB` 且 p2p OK，但 CUDA 仍返回 217 | 卡/VBIOS 层没开（回到第 1 步） |
| 都 OK，但加负载时主机挂死/掉卡 | 跨 root complex P2P 在 AMD 上的已知失败模式（调研 W8 已记录过一次"跨 root bus DMA 挂死"） |

## 6. 建议

- **不要再为 P2P 花时间**，保持 PP=2 + NCCL `SHM/CE`：`NCCL_P2P_DISABLE=1`、`--disable-custom-all-reduce`。
- 真要压跨卡成本，方向是**减字节**（PP 而非 TP、小消息合并、pinned buffer 复用）——4 KiB 33 µs 已是这条
  路的底线。
- 如果将来上裸机 Linux，先按 §5 的矩阵走一遍再谈 TP；在 Windows 上这份文档就是终点。

---

复现命令：

```powershell
cd D:\code\vllm-windows
.\.venv\Scripts\python.exe _dev\probe\_p2p_capability_probe.py --bench
```

探针同时打印 CUDA 与 NVML 两套判据；`--bench` 换尺寸扫中转成本。注意：探针在失败的
`cudaDeviceEnablePeerAccess` 后会调 `cudaGetLastError()` 清掉 sticky error，否则后续无关的 kernel
launch 会误报 "peer access is not supported"。

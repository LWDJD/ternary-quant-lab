# Radeon GPU Profiler (RGP) 使用手册 — 本机实操版

> 与官方文档的区别：**这份是从零试错打通的实操记录**，包含官网上没有写、但会直接把流程卡死的坑。
> 适用：本机（AMD Ryzen AI 9 H 365 / Radeon 880M，gfx1150，RDNA 3.5，Windows 11）。
> 最后更新：2026-09-25

---

## 0. 速查（TL;DR）

**抓一次 Vulkan 计算负载的 profile**，两步：

```powershell
# 终端 A（必须以非受限令牌 / 提权方式运行！）
cd D:\Project\openhanako\workbench\rdts\RadeonDeveloperToolSuite-2026-05-28-1806
New-Item -ItemType Directory -Force D:\captures | Out-Null
.\RadeonDeveloperPanelCLI.exe -m profiling -p llama -o D:\captures\moe.rgp `
    --rgp-auto-capture=dispatch:500:100 --rgp-capture-mode dispatch `
    --rgp-counter-collection --rgp-sqtt-buffer-size maximum
```

```powershell
# 终端 B（等 A 打出 "Waiting for GPU application to connect..." 之后再启动）
cd D:\Project\openhanako\workbench\prism-vulkan
.\llama-bench.exe -m "<模型路径>" -ngl 99 -p 0 -n 128 -r 1
```

**读结果**：`Start-Process .\RadeonGPUProfiler.exe` → File → Open → `D:\captures\moe.rgp`。

**唯一的硬性前置**：终端 A 必须脱离沙箱的受限令牌路径。否则一律报 `Failed to create router.`（详见第 3 节）。

### 0.1 ⚠️ 但在用 RGP 之前：先试 ggml-vulkan 自带计时

【2026-09-26 新增】如果目标是“一个 token 的时间花在哪些 op 上”，**不要上 RGP**。ggml-vulkan 本身就带逐 op 计时与逐 pipeline 资源统计：

```powershell
$env:GGML_VK_PERF_LOGGER="1"; $env:GGML_VK_PIPELINE_STATS="1"
llama-bench.exe -m <模型> -ngl 99 -p 0 -n 4 -r 1
```

| 变量 | 输出 |
|---|---|
| `GGML_VK_PERF_LOGGER` | 逐 op：调用次数 / 单次 µs / 累计 µs / GFLOPS |
| `GGML_VK_PIPELINE_STATS` | 逐 pipeline：`numUsedVgprs` / `numUsedSgprs` / `ldsUsageSizeInBytes` / `scratchMemUsageInBytes` |
| `GGML_VK_PERF_LOGGER_CONCURRENT` / `..._FREQUENCY` | 并发模式 / 打印频率 |

**优势**：不用提权（普通 shell 就能跑）、不用 GUI、不用截图读图，而且拿到的是渲染好的数值。
**唯一陷阱**：**第一张表包含预热与着色器编译，数字会失真 30~1000 倍**（会看到“一次 matvec 9273 µs”这种假数），必须看后续几张。

RGP 只在需要**逐 dispatch 的硬件计数器、占用率、停電分布**时才用。

---

## 1. 工具套件构成

**位置**：`D:\Project\openhanako\workbench\rdts\RadeonDeveloperToolSuite-2026-05-28-1806\`
**版本**：RDP v3.5.0 / RGP v2.7.0.32 / RGA 2.14.2.8（2026-05-28 构建）
**注意**：zip 解压即用，**没有安装脚本**，也不需要安装。

| 可执行文件 | 作用 | 本机可用性 |
|---|---|---|
| **`RadeonDeveloperPanelCLI.exe`** | **无界面的抓帧工具（主角）** | ✅ 可用（需非受限令牌） |
| `RadeonGPUProfiler.exe` | 读 `.rgp` 的 GUI | ✅ |
| `RadeonDeveloperPanel.exe` | 有界面的 Panel | ✅（需人工点击） |
| `RadeonDeveloperService.exe` | RDS 托盘常驻程序 | ⚠️ **不要手动启动**（见 3.2） |
| `RadeonDeveloperServiceCLI.exe` | headless RDS | ❌ **只是个转发客户端**，单独跑必报 `DD_RESULT_NET_CONNECTION_REFUSED` |
| `rga.exe` / `RadeonGPUAnalyzer.exe` | 离线着色器静态分析 | ⚠️ 能跑但输出常为空，见 7.3 |
| `RadeonMemoryVisualizer.exe` | 显存 trace | 未测 |
| `RadeonRaytracingAnalyzer.exe` | 光追分析 | 无关 |
| `rgd.exe` | 崩溃分析 CLI | 未测 |
| `rtda.exe` | **只是个 URL 下载器**，与 trace 无关 | 无用 |
| `utils\` | 着色器工具链：`glslc`、`spirv-as/dis`、`dxc`、`amdllpc`、`vulkan_backend.exe` | 备用 |

---

## 2. 前置条件清单（缺一不可）

| # | 条件 | 说明 |
|---|---|---|
| 1 | **AMD 官方 Adrenalin 驱动** | 本机 26.8.1（`26.10.41.01-260811a-203304C`）。公开驱动已含 developer mode 特性 |
| 2 | **非受限令牌 / 提权上下文运行 Panel** | **最容易踩的坑**，见 3.1 |
| 3 | **目标程序必须后启动** | 官方原文：`must NOT already be running`。先配好 Panel，再启动被测程序 |
| 4 | **不要预先启动 RDS** | 官方原文：`For Local connections, starting Radeon Developer Service is optional` |
| 5 | 清理残留进程 | 之前残留的 Panel/RDS 实例可能占住 router |

**不需要**（本机实测确认）：

- ❌ 不需要跑 `scripts\AddUserToGroup.bat` —— 官方明确它只服务 **DirectX 12** 的 System Activity，对 Vulkan 无影响
- ❌ 不需要关心端口 27300 —— 本地通信走**命名管道**，不是 socket。官方原文：`communication between the two uses pipes, not sockets and ports`
- ❌ 不需要安装驱动插件 —— 套件自包含

---

## 3. 踩过的坑（按报错对照）

### 3.1 `Failed to create router.` —— 沙箱受限令牌

**症状**：

```
[ERROR] [Default] (PID:0) Failed to create router.
Failed to initialize capture context
Failed to initialize capture API
```

**根因**：**在受限令牌（restricted token）下运行**。Panel 需要创建一个 IPC 对象（router），受限令牌下建不出来。

**验证方法**：

```powershell
([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
```

**注意**：这条输出 `False` 不一定说明"没提权"，也可能是受限令牌把管理员身份剥掉了。**判据是 capture API 能否初始化，不是这条命令。**

**解决**：换非受限路径运行。在 Hana 里即 `sandbox_permissions="require_escalated"`（此路径下 `Capture API initialized successfully` 立刻通过）。在普通交互式 PowerShell 里直接跑也没问题。

### 3.2 手动启动 RDS 反而有害

`RadeonDeveloperService.exe` 能常驻，但：

- 本地场景下**不需要它**，Panel 会自己处理
- 手动启动的实例可能持有 router，导致 Panel 建不出自己的
- 提权启动的实例，非提权会话**杀不掉**（`Access is denied`），只能重启电脑清理

**做法**：**永远不要预先启动 RDS。** 如果怀疑有残留：

```powershell
Get-Process RadeonDeveloperService,RadeonDeveloperPanel,RadeonDeveloperPanelCLI -ErrorAction SilentlyContinue
# 有就杀掉；杀不掉（提权实例）就重启
```

### 3.3 `RadeonDeveloperServiceCLI.exe` 报 `NET_CONNECTION_REFUSED`

```
Failed to create router context: DD_RESULT_NET_CONNECTION_REFUSED
```

它是**客户端**，要连托盘版 RDS 的 router。托盘版没起来它就必失败。**别用它。**

### 3.4 `Failed to connect to router.`（remote 模式）

用 `--remote-host` 时会走 socket 路径，本地没有监听端口，故失败。**本地不要用 remote 模式。**

### 3.5 没产出 `.rgp`，日志里看到提前 `Disconnected`

**症状**：日志出现 `Connecting to process: ...` 后紧跟 `[Status] Disconnected`。

**可能原因**：

1. 目标程序在 Panel 就绪**之前**启动了 → 严格按第 0 节的顺序
2. **客户端超时**：官方原文 `Connections to applications will timeout after a brief period of no API calls`。llama-bench 加载模型有 40 秒没有任何 GPU 调用，可能超时断开。**GUI 版有 "Disable client timeout" 开关，CLI 版没有** → 遇到这个就用 GUI。

### 3.6 有 Panel 进程但迟迟不产生数据

检查目标程序是否被 blocklist 拦了。CLI 内置默认 blocklist：

```powershell
.\RadeonDeveloperPanelCLI.exe --list-blocklist
```

---

## 4. Panel CLI 参数速查

```
-o, --output <path>            输出文件；默认 <mode>_<timestamp>.<ext>
-m, --mode <mode>              profiling | raytracing | memory | crash | clocks   （默认 profiling）
-p, --process <name>           只连接进程名包含该串的程序
--verbose                      详细日志（排障必开）
--system-info                  打印 OS/驱动/CPU/GPU 后退出
--list-blocklist               打印生效的 blocklist 后退出
--list-experiments             打印可用 driver experiments 后退出
```

**Profiling（RGP）专用**：

```
--rgp-auto-capture <spec>      'frame[:N]' 或 'dispatch[:start[:count]]'（start 必须 ≥1）
--rgp-auto-capture-delay <ms>  dispatch 自动抓取前的延迟
--rgp-capture-mode <mode>      default | frame | draw | dispatch
--rgp-render-op-count <n>      draw/dispatch 模式下抓取的 render op 数
--rgp-counter-collection       采硬件计数器
--rgp-instruction-tracing      指令级追踪
--rgp-shader-instrumentation   着色器插桩
--rgp-sqtt-buffer-size <size>  minimum | low | default | high | maximum
```

**对纯计算负载（无 swapchain）**：只能用 `--rgp-capture-mode dispatch` + `--rgp-auto-capture=dispatch:...`。

### 4.1 参数怎么选

| 参数 | 建议 | 理由 |
|---|---|---|
| `-p` | 目标进程名的唯一前缀，如 `llama` | 过滤，避免连上别的程序 |
| `--rgp-auto-capture` | `dispatch:500:100` 起步 | 前 500 次 dispatch 是加载+预热；取接下来的 100 次落在稳态 |
| `--rgp-sqtt-buffer-size` | `maximum` | 抓不到完整时间线比抓多更糟 |
| `--rgp-counter-collection` | 开 | APU 上部分计数器不可用，但能拿到的都有用 |
| `--rgp-instruction-tracing` | 先不开 | 数据量暴增；需要逐指令停顿分析时再开 |
| 目标程序的 `-n` | `128` 左右 | 太短抓不到稳态；太长抓帧耗时 |

**dispatch 起始值怎么定**：一次 token 大约有几百到两千次 dispatch。`500:100` 是经验值。若打开后发现抓到的都是小 op，把 start 调大或调小重抓——**管道打通后重抓只要一条命令**。

---

## 5. 抓帧完整流程（含验收点）

### 步骤 1：先验证链路通不通

```powershell
cd D:\Project\openhanako\workbench\rdts\RadeonDeveloperToolSuite-2026-05-28-1806
Get-Process RadeonDeveloperService -ErrorAction SilentlyContinue   # 必须为空
.\RadeonDeveloperPanelCLI.exe --system-info --verbose
```

**验收点**：必须打出 `Capture API initialized successfully` 及系统信息。
**打不出来就别往下走**，先解决第 3 节的问题。

顺带：`--system-info` 是拿系统/驱动/显存布局的快捷方式（比 WMI 稳，且不需要额外权限）。

### 步骤 2：挂上 Panel（终端 A）

```powershell
New-Item -ItemType Directory -Force D:\captures | Out-Null
.\RadeonDeveloperPanelCLI.exe -m profiling -p llama -o D:\captures\<名字>.rgp `
    --rgp-auto-capture=dispatch:500:100 --rgp-capture-mode dispatch `
    --rgp-counter-collection --rgp-sqtt-buffer-size maximum | Tee-Object D:\captures\panel.log
```

**验收点**：日志出现

```
Capture API initialized successfully
Profiling (RGP) feature enabled (auto-capture enabled)
Waiting for GPU application to connect...
Process filter: llama
```

看到 "Waiting for GPU application to connect..." 才启动下一步。

### 步骤 3：启动被测程序（终端 B）

```powershell
cd D:\Project\openhanako\workbench\prism-vulkan
.\llama-bench.exe -m "<模型路径>" -ngl 99 -p 0 -n 128 -r 1
```

### 步骤 4：确认产出

Panel 日志应出现：`Connecting to process: ...` → `[Status] Capturing` → `Dumping trace data` → `Processing counter data` → `Trace saved to: ...`。

**验收点**：`D:\captures\<名字>.rgp` 存在且 **> 1 MB**（2.5 MB 左右是 100 个 dispatch 的典型大小）。

---

## 6. 怎么读结果

### 6.1 视图地图

| 视图 | 位置 | 看什么 |
|---|---|---|
| **Profile summary** | 左侧栏 | 总时长、GPU idle%、是否 GPU bound |
| **System activity** | OVERVIEW 上部 | queue submission / command buffer 数、时间线 |
| **Pipelines** | OVERVIEW → Pipelines | 按着色器分组：谁占时间、Occupancy、VGPR、SGPR、**Scratch**、**Wave mode** |
| **Most expensive events** | 左侧栏 | 按耗时排序的事件表 + 百分位直方图 |
| **Barriers** | 左侧栏 | barriers/dispatch、drain time |
| **Wavefront occupancy** | EVENTS → 选一个事件 → 下方详情标签 | **最有价值**：wavefront 数、线程数、occupancy 实际/上限、资源占用 |
| **Instruction timing** | 同一组标签 | 逐指令停顿（**部分 RDNA3 无数据**） |
| ~~Memory Chart~~ | — | **RDNA APU 上为空**，别浪费时间 |

### 6.2 各列的含义（排障时最有用的几列）

| 列 / 指标 | 含义 | 怎么用 |
|---|---|---|
| **GPU idle %** | GPU 空转比例 | 高 → 启动/同步开销；**低（<1%）→ 时间真的花在执行里** |
| **Occupancy** | 占用率。**注意单位不统一**：Pipelines 汇总表里显示成 `12–12`（形如 min–max），事件详情里显示成 `4/16`（形如 实际/上限）。**同一个 pipeline 两处对不上**，用之前先在 GUI 里悬停列头读 tooltip 确认口径 |
| **VGPRs** | 每线程向量寄存器 | 决定最大 wave 数：约为 `1536 / VGPRs`（有上限截断）；也可反查是否被寄存器限制 |
| **SGPRs** | 标量寄存器 | 同上 |
| **Scratch mem** | 寄存器溢出量 | 非 "No" → 有 spill，是严重问题 |
| **Wave mode** | `wave64` / `No` | **wave64 在 RDNA 上占两个 wave 槽，最大在飞 wave 数减半** |
| **LDS** | 每 workgroup 共享内存字节数 | 也参与限制最大 wave 数 |
| **Drain Time** | barrier 前的排空耗时 | 占比大 → 同步开销是主因 |

### 6.3 判断流程（我们实际用过的推理顺序）

1. `GPU idle` 高不高？→ 高就是启动/同步问题，低了往下走
2. `Scratch` 有没有溢出？`Occupancy` 实际 vs 上限？→ 排除寄存器/占用率
3. workgroup 网格够不够大？→ 排除并行度
4. `Barriers` 的 drain 占比？→ 排除同步
5. **`Wavefront occupancy` 里内存计数器活跃度** → 低 = 不是带宽墙
6. 结合 5 和 occupancy 偏低 → **迟发在飞 wave 数不足的访存延迟暴露**

### 6.4 本机已知限制（官方 release notes）

| 限制 | 影响 |
|---|---|
| `Cache and ray tracing counter data collection is not currently supported on RDNA based APUs` | **Memory Chart / cache 计数在 Radeon APU 上拿不到** |
| `On some RDNA 3 hardware, detailed instruction timing data will not be available` | Instruction timing 可能是空的 → 改用 Wavefront occupancy |
| Crash Analysis / Hardware Crash Analysis 不支持 APU | 与本用途无关 |
| 一次只能抓一个 AMD GPU | 本机单 GPU，无影响 |

---

## 7. `.rgp` 文件与其它工具

### 7.1 文件格式（已解出）

```
AMD_RDF  v3
[512 字节头] + [payload] + [块索引]
头里含指向各块的 (offset, size)
```

块类型（从索引读出）：

```
AsicInfo / ApiInfo / ClockCalibration / CodeObject×N / COLoadEvent / PsoCorrelation /
SqttData / SpmSession / SpmCounterData×N / QueueInfo / QueueEvent / TraceConfig /
SystemInfo / DriverOverrides / TraceProcessInfo / DerivedSpm*
```

**`SqttData`（逐 wave 时序）和 `SpmCounterData`（硬件计数器）是 AMD 私有 schema。**
**不要尝试逆向——用 RGP GUI 读。** 唯一可程序化处理的是外层容器和块索引。

### 7.2 导出

RGP GUI 支持导出**管道二进制**（给 RGA 做 ISA 分析）。文档里**没有**导出通用 CSV 的口径；逐 dispatch 数据目前只能靠 GUI 查看或截图。

### 7.3 RGA（`rga.exe`）离线分析

```powershell
.\rga.exe -s vk-spv-offline -c gfx1150 --isa out.txt -a out.csv <shader.spv>
.\rga.exe -l                 # 列出支持的 ASIC
```

- ✅ 能识别 `gfx1150 (RDNA3.5)`，编译成功
- ⚠️ 实测**输出文件为空**，要用好还得完整复现 llama.cpp 的着色器 define 组合
- ⚠️ RGA 自己声明：`The Vulkan offline mode is independent of the installed driver and may not provide assembly code and resource usage that reflect the real-life case` —— **它用自己的编译器，不等于驱动的实际编译结果**

**结论**：做运行时瓶颈定位，RGP 是对的工具；RGA 只适合做理论占用率/寄存器用量的量级参考。

---

## 8. 复用模板

抓任何模型：

```powershell
# ---- 变量 ----
$MODEL  = "D:\AI\lmstudio\models\...\xxx.gguf"
$OUT    = "D:\captures\$(Get-Date -Format yyyyMMdd-HHmm)-$(Split-Path $MODEL -Leaf).rgp"
$FILTER = "llama"          # 目标进程名前缀
$START  = 500              # 跳过预热 dispatch
$COUNT  = 100              # 抓取数量
$GEN    = 128              # 目标生成 token 数

# ---- 终端 A（非受限令牌 / 提权）----
cd D:\Project\openhanako\workbench\rdts\RadeonDeveloperToolSuite-2026-05-28-1806
New-Item -ItemType Directory -Force D:\captures | Out-Null
.\RadeonDeveloperPanelCLI.exe --system-info          # 先验收：必须打出 Capture API initialized successfully
.\RadeonDeveloperPanelCLI.exe -m profiling -p $FILTER -o $OUT `
    --rgp-auto-capture="dispatch:${START}:${COUNT}" --rgp-capture-mode dispatch `
    --rgp-counter-collection --rgp-sqtt-buffer-size maximum
# 看到 "Waiting for GPU application to connect..." 后，另开终端 B

# ---- 终端 B ----
cd D:\Project\openhanako\workbench\prism-vulkan
.\llama-bench.exe -m $MODEL -ngl 99 -p 0 -n $GEN -r 1

# ---- 读结果 ----
Start-Process D:\Project\openhanako\workbench\rdts\RadeonDeveloperToolSuite-2026-05-28-1806\RadeonGPUProfiler.exe
```

**对照实验的建议**：同一个 `.rgp` 里能同时看到多个形状的 dispatch，所以**一次抓帧就能对比不同矩阵形状**（这正是发现"同字节数差 1.7 倍"的方式）。想比较两个模型时，用同一个 `$START/$COUNT` 各抓一次。

---

## 9. 排障速查表

| 症状 | 根因 | 处理 |
|---|---|---|
| `Failed to create router.` | 受限令牌 / 沙箱上下文 | 换非受限令牌运行（3.1） |
| `Failed to initialize capture API` | 同上 | 同上 |
| `DD_RESULT_NET_CONNECTION_REFUSED` | 用了 `RadeonDeveloperServiceCLI.exe` | 别用它 |
| `Failed to connect to router.` | 本地用了 `--remote-host` | 本地不要用 remote 模式 |
| 杀不掉 RadeonDeveloperService | 提权实例 | 重启电脑 |
| `--system-info` 成功但抓不到 `.rgp` | 目标程序启动过早 / 超时 | 严格按顺序；改用 GUI 并关掉 client timeout |
| `.rgp` 只有几百 KB | 抓到的 dispatch 太少 | 加大 `--rgp-auto-capture` 的 count |
| GUI 里 Memory Chart 空白 | RDNA APU 不支持 cache 计数器 | 正常，改看 Wavefront occupancy |
| Instruction timing 空白 | 部分 RDNA3 无此数据 | 改看 Wavefront occupancy |
| RGA 输出文件为空 | 离线模式信息不足 | 放弃 RGA，改用 RGP |

---

## 10. 一句话

**RGP 在这台机器上唯一真正的门槛是"不能跑在受限令牌下"**——越过之后，`RadeonDeveloperPanelCLI` 的 `dispatch` 级自动抓帧是一条命令的事，产出 2.5 MB 的 `.rgp`，用 GUI 打开就能拿到逐 dispatch 的耗时、占用率、寄存器、LDS 和停顿分布。**其余全是文档误导**（端口、DX12 那个 bat、手动启服务）。

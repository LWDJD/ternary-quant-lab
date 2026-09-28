# MoE 单流推理瓶颈：完整调查记录与续接指南

> 目的：本文档是上下文压缩后的唯一续接凭证。**自包含、可复现、不依赖对话历史**。
> 目标场景：**单发（batch=1、单序列）**，要的是从根源修复，而不是靠并发绕过。
> 最后更新：2026-09-25

---

## 0. 任务定义

**待解决的问题**：单流跑 `Qwen3.6-35B-A3B`（MoE，IQ2_M）时 decode 只有 **26~28 t/s**，而同一块 GPU 在 npl=32 并发下能到 **65 t/s**。差距 2.4 倍。

**已确认性质**：不是带宽、不是并行度、不是启动开销、不是占用率被资源卡死。
**已确认根因**：**在飞的 wave 数不足，导致访存延迟暴露**（详见第 6 节）。

**验收标准**：把专家矩阵乘的达成带宽从 24~41 GB/s 推向 97 GB/s 上限，单流 decode 目标 **45~55 t/s**。

---

## 1. 硬件环境

| 项 | 值 |
|---|---|
| CPU | AMD Ryzen AI 9 H 365 w/ Radeon 880M（Strix Point，10C/20T，4×Zen5 + 6×Zen5c，基频 2.0 GHz） |
| iGPU | Radeon 880M，**gfx1150，RDNA 3.5**，12 CU，时钟 600–2900 MHz |
| 内存 | LPDDR5X-8000，**128-bit**，理论 ~128 GB/s |
| OS | Windows 11 Home China，build 26100，WDDM 3.2 |
| 驱动 | AMD Adrenalin **26.8.1**，`26.10.41.01-260811a-203304C`，2026-08-11 |
| Vulkan | API 1.4.349 |

### 1.1 内存布局（关键，影响容量判断）

```
dxdiag:
  Memory:                32768 MB RAM        ← 总容量 32 GB
  Available OS Memory:   16170 MB RAM        ← 系统只看到 16 GB
  Dedicated Memory:      16209 MB            ← 16 GB 划给核显（BIOS carve-out）
  Shared Memory:          8084 MB
  Display Memory:        24294 MB

Vulkan heap（bwtest 输出）:
  heap 0 :  7.30 GiB            （非 DEVICE_LOCAL，= 共享池）
  heap 1 : 14.60 GiB  DEVICE_LOCAL（= carve-out）
  合计 21.9 GiB = llama.cpp 报告的 22420 MiB
```

**含意**：
- `DEVICE_LOCAL` 堆只有 14.60 GiB。10.85 GiB 的 MoE 装得下；4-bit 的 35B（~18 GB）装不下。
- **CPU 侧只有 16 GB 可见内存** —— 这是 `-ngl 0` 崩溃的原因之一（第 8.2 节）。

### 1.2 驱动/工具报的带宽数字是错的

AMD 工具报 `Bus bit width: 256 / Bandwidth: 251 GB/s`。**判定为错**，理由：

- AMD 官方 Ryzen AI 9 365 规格：LPDDR5x-8000
- TechPowerUp：Dual-channel（DDR5-5600 时 89.6 GB/s → 对应 128-bit）
- Gorgon Point（后继）报道："still **128-bit** memory bus"
- 工具自己的数算：1960 MHz × 4 ops × **16 字节（128-bit）** = 125 GB/s，正好对应 LPDDR5X-8000
- **实测反证**：纯流式读上限 97 GB/s。若真能到 251 GB/s，一个 100 万线程只做读取的 kernel 不该只有 97（39%）

**结论：理论峰值 ≈ 128 GB/s，实测可达 ≈ 97 GB/s（76%）。**

---

## 2. 软件栈

| 项 | 值 |
|---|---|
| llama.cpp | **PrismML fork**，build `prism-b10735-842b188`（`842b18804`），分支 `prism`（prism-v7） |
| 二进制 | `llama-prism-b10735-842b188-bin-win-vulkan-x64.zip`，解压于 `D:\Project\openhanako\workbench\prism-vulkan\` |
| Vulkan 设备能力 | `uma:1 fp16:1 bf16:1 fp4:0 warp size:64 shared memory:32768 int dot:1 matrix cores:KHR_coopmat`（**无 coopmat2**） |
| CPU 后端 | `ggml-cpu-cascadelake.dll`（AVX-512 + VNNI 路径） |

**本机可用工具链**（都已验证）：Vulkan SDK 1.4.357.0（`glslc`、`glslangValidator` 在 PATH）、VS2022 BuildTools（`vcvars64.bat` 在 `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\`）、g++ 15.2（`C:\msys64\ucrt64\bin\`）、Vulkan `vulkan-1.lib` 在 SDK 的 `Lib\`。

**重要：不要和 stock llama.cpp 混用 `ggml-*.dll`。** 主仓库不支持 PQ2_0(142)/PTQ1_0(143)，类型 id 超出上游 `GGML_TYPE_COUNT=43`。

---

## 3. 被测模型

| 路径 | 架构 | 量化 | 大小 | 参数 |
|---|---|---|---|---|
| `D:\AI\lmstudio\models\HauhauCS\Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive\...-IQ2_M.gguf` | qwen35moe | IQ2_M 2.7bpw | 10.85 GiB | 34.66B |
| `D:\AI\lmstudio\models\prism-ml\Ternary-Bonsai-2-27B-gguf\Ternary-Bonsai-2-27B-PTQ1_0.gguf` | qwen35（dense） | 三值 1.75bpw | 5.5384 GiB | 26.9B |
| 同目录 `...-PQ2_0.gguf` | qwen35 | 三值 2.13bpw | 6.7114 GiB | 26.9B |
| `D:\AI\lmstudio\models\openbmb\MiniCPM5-2B-GGUF\MiniCPM5-2B-Q8_0.gguf` | llama | Q8_0 | 2.4958 GiB | 2.52B |
| `D:\AI\lmstudio\models\openbmb\MiniCPM5-1B-GGUF\MiniCPM5-1B-F16.gguf` | llama | F16 | 2.0178 GiB | 1.08B |

### 3.1 MoE 模型维度（GGUF metadata 原文）

```
general.architecture             = qwen35moe
qwen35moe.block_count            = 40
qwen35moe.embedding_length       = 2048
qwen35moe.expert_count           = 256        ← 共 256 个专家
qwen35moe.expert_used_count      = 8          ← top-8 激活
qwen35moe.expert_feed_forward_length = 512
qwen35moe.expert_shared_feed_forward_length = 512
qwen35moe.attention.head_count   = 16
qwen35moe.attention.head_count_kv= 2
qwen35moe.attention.key_length   = 256
qwen35moe.attention.value_length = 256
qwen35moe.full_attention_interval= 4          ← 每 4 层 1 层全注意力，其余 30 层是 SSM/GDN
qwen35moe.ssm.inner_size         = 4096
qwen35moe.ssm.state_size         = 128
qwen35moe.ssm.group_count        = 16
qwen35moe.ssm.time_step_rank     = 32
qwen35moe.context_length         = 262144
```

**解读**：40 层中只有 10 层是全注意力，30 层是线性注意力（SSM/GDN）。专家 FFN 中间维 512。

### 3.2 ⚠️【2026-09-26 更正】实际量化类型不是 IQ2_M

文件名叫 IQ2_M，但**张量里没有一个 IQ2_M**。实际类型直方图（733 个张量）：

| 类型 | 个数 | 出现在 |
|---|---:|---|
| **IQ2_S** | **375** | ffn_gate/up_exps、ffn_*_shexp、ssm_out、ssm_alpha/beta、attn_q/k/gate、部分 ffn_down |
| **IQ3_S** | 16 | token_embd、attn_output、blk.0~9 的 ffn_down |
| Q4_K | 40 | attn_qkv |
| Q5_K | 1 | output（lm_head） |
| F32 | 301 | 各种 norm / bias / ssm_a / ssm_conv1d / ffn_gate_inp |

**含义**："IQ2_M" 只是量化配方的名字，不是 ggml 类型。`ggml-vulkan.cpp` 里对 `GGML_TYPE_IQ2_M` 零匹配。
**热 shader 是 `mul_mat_vec_iq2_s.comp`，不是第 7 节分析的 `mul_mat_vecq.comp`。**

关键张量形状（ne0=K，ne1=N，ne2=专家数）：

```
ffn_gate_exps.weight   [2048, 512, 256]   IQ2_S   → K=2048, N=512
ffn_up_exps.weight     [2048, 512, 256]   IQ2_S
ffn_down_exps.weight   [512, 2048, 256]   IQ3_S/IQ2_S → K=512, N=2048
attn_qkv.weight        [2048, 8192]       Q4_K
output.weight          [2048, 248320]     Q5_K    → 输出维度 248320
```

配合 `rm_iq = 4`（即 `NUM_ROWS = 4`），dispatch 网格 X = N/NUM_ROWS，**两组形状完全对上了**：

- `vkCmdDispatch(128, 8, 1)` = ffn_gate/up（512/4 = 128）
- `vkCmdDispatch(512, 8, 1)` = ffn_down（2048/4 = 512）

---

## 4. 全部实测数据

### 4.1 原始带宽（自建 Vulkan 微基准，见第 9 节路径）

纯读 `D:\Project\openhanako\workbench\bwtest\bw.exe read_uN.spv read N`，1 GiB buffer：

| UNROLL | 1024 线程 | 4096 | 16384 | 65536 | 262144 | 1048576 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16.5 | 53.6 | 89.5 | 92.7 | 92.6 | **97.1** |
| 2 | 31.6 | 86.2 | 94.3 | 89.6 | **98.7** | 94.2 |
| 4 | 59.5 | 87.1 | **97.6** | 92.8 | 94.6 | 91.5 |
| 8 | **92.6** | **101.0** | 93.5 | 92.2 | 94.5 | 93.2 |
| 16 | 89.2 | 97.5 | 91.9 | 89.4 | 89.0 | 94.7 |

- **纯读上限 ≈ 97~101 GB/s**（= 理论峰值的 76~79%）
- 纯写：稳定段 80~93（1024 线程那档出现 110，疑为写缓冲记账偏差）
- 拷贝（读+写合计）：82~94
- **饱和极易**：UNROLL=8 时 1024 线程就到 92.6 GB/s；UNROLL=1 需约 16384 线程

### 4.2 GEMV 达成带宽对照（Vulkan，`-ngl 99`）

| 模型 | 量化 | bpw | 大小 | pp512 | tg128 | decode 达成 | 占 97 GB/s |
|---|---|---:|---:|---:|---:|---:|---:|
| MiniCPM5-1B | F16 | 16 | 2.02 GiB | 2090.88 | 41.92 | **90.6 GB/s** | **93%** |
| MiniCPM5-2B | Q8_0 | 8.5 | 2.50 GiB | 1021.70 | 29.92 | 80.2 GB/s | 83% |
| Bonsai 27B | PQ2_0 | 2.13 | 6.70 GiB | 90.61 | 9.30 | 67.0 GB/s | 69% |
| Bonsai 27B | PTQ1_0 | 1.75 | 5.53 GiB | 48.56 | 10.56 | 62.8 GB/s | 65% |
| **MoE 35B-A3B** | **IQ2_M** | **2.7** | **10.85 GiB** | **350.99** | **26.50** | **24.4 GB/s** | **25%** |

换算：`GB/s = GiB × tok/s × 1.0737`。
**规律：bpw越低，能拉到的带宽越低，单调。**

### 4.3 批处理扫描（MoE，`-c 32768`，npp 512 / ntg 128）

| npl | S_PP t/s | **S_TG t/s** | 每 token ms | 达成带宽 | 占上限 |
|---:|---:|---:|---:|---:|---:|
| 1 | 216.20 | 24.12 | 41.5 | 24.4 GB/s | 25% |
| 2 | 268.97 | 32.88 | 30.4 | — | — |
| 4 | 323.02 | 40.78 | 24.5 | — | — |
| 8 | 340.59 | 41.47 | 24.1 | — | — |
| 16 | 349.40 | 52.26 | 19.1 | — | — |
| **32** | 349.38 | **65.23** | **15.4** | **65.8 GB/s** | **68%** |

**npl 32 复现三次：65.11 / 65.23 / 65.11。**
`-npl 64` 见 4.5（被 `-c` 污染，未取到干净值）。

同法测 dense Bonsai PTQ1_0：npl 1/2/4/8 → 9.13 / 15.85 / 15.48 / 16.28（**卡在 16**）。
对照：MoE 能爬到 65，dense 只能到 16 —— 这才是 MoE 在这台机器上的真正优势。

### 4.4 `-c` 只在并发下有效

| npl 32，-c | S_PP | S_TG |
|---:|---:|---:|
| 32768 | 341.32 | **65.11** |
| 65536 | 264.79 | **50.79** |
| 32768 | 340.74 | **64.87** |

A/B/A 复现，切换即恢复 → **确定是 `-c` 的影响，不是时钟漂移**（prefill 和 decode 同等下降 ~21%）。

显存数对不上原因：`-c 32768` KV 702 MiB、`-c 65536` KV 1342 MiB，合计 11704 / 12376 MiB，**都远低于 14.60 GiB 的 DEVICE_LOCAL 堆，没有越界**。机制未查明。

**单发不受 `-c` 影响**（npl 1，-c = 8192/32768/65536/131072 → 24.32 / 24.26 / 24.24 / 24.32，全平）。

### 4.5 上下文长度不影响 decode（【2026-09-26】重新验证，结论维持）

原文用 batched-bench（npp 64/1024/4096 → 24.42/27.50/25.92），因为口径粗糙而不可靠。
**用 llama-bench 交错 A/B/A/B 重测后，原结论成立：**

```powershell
# 必须交错跑，不能按上下文递增顺序跑
foreach($p in 0,2048,0,2048){ llama-bench.exe -m <MoE> -ngl 99 -p $p -n 128 -r 3 }
```

| `-p` | tg128 |
|---:|---:|
| 0 | 27.77 ± 0.04 |
| 2048 | 27.81 ± 0.07 |
| 0 | 27.75 ± 0.15 |
| 2048 | 27.84 ± 0.06 |

**四个值完全一致（27.75~27.84）→ 上下文长度确实不影响 decode。**

> ⚠️ **踩过的坑（方法论，很重要）**
>
> 我先按 `-p 0 / 512 / 2048 / 8192` 递增顺序跑了一次，得到 33.36 / 33.73 / **27.12** / 27.71，
> 看上去是个“512→2048 掉 20%”的台阶。
> **但交错重测后四个值全部相同——那个台阶是会话级别的漂移，不是上下文效应。**
>
> 教训：**实测数字随机器状态（温度/其他进程/显存分配）可整个会话地漂移 ~20%。**
> 因此：
> 1. 跨会话的绝对数**不可直接比**（同一个 `-p 0 -r 3` 组合在两次会话里得过 33.36 和 27.28）
> 2. 任何对比必须**在同一会话内交错跑 A/B/A**，看第二遍 A 能不能回到第一遍
> 3. 递增顺序的扫描会被漂移污染成假台阶

### 4.6 `-ngl` 异常（dense PTQ1_0，可复现）

| ngl | pp512 | tg128 |
|---:|---:|---:|
| 99 | 48.56（复测两次） | 10.56 |
| 64 | **59.46** | 8.53 |
| 63 | **59.87** | 6.36 |

挪 1~2 层到 CPU，**prefill 稳定 +23%**，decode -19%。`-ngl 64` 支配 `-ngl 63`。机制未查明（怀疑某层的 Vulkan 实现特别差，见 8.1）。

### 4.7 CPU 对照

| 模型 | 后端 | pp512 | tg128 | 达成带宽 |
|---|---|---:|---:|---:|
| Bonsai PTQ1_0 | CPU | 37.36 | **0.37** | 2.2 GB/s |
| Bonsai PQ2_0 | CPU | 55.07 | 5.45 | 39.3 GB/s |
| MiniCPM5-2B Q8_0 | CPU | 741.23 | 26.65 | 71.4 GB/s |
| MoE 35B-A3B | CPU | — | **崩溃** | — |

**CPU 侧 prefill 不弱**（37~55 t/s，是 GPU 的 61~77%）。**但 decode 差很多倍。**

---

## 5. 已排除的假设（逐条附证据）

| # | 假设 | 状态 | 证据 |
|---|---|---|---|
| 1 | 内存带宽物理上限 | **排除** | 原始读上限 97 GB/s，MoE 单流只用 25% |
| 2 | 并行度不足 / workgroup 太少 | **排除** | 专家 dispatch 网格 1024~4096 个 workgroup |
| 3 | 图切分（graph split）开销 | **排除** | GPU 路径只有 **2** 个 split |
| 4 | attention / SSM 随上下文变慢 | **排除** | 上下文 192→4224，速度不变（4.5） |
| 5 | GPU 空转 / 启动开销 | **排除** | RGP：**GPU idle 0.46%**，"application is GPU bound" |
| 6 | 占用率被资源卡死 / 寄存器溢出 | **排除** | 头号 pipeline **Scratch "No"**，VGPR 88（允许 ~16 wave） |
| 7 | barrier / drain | **排除** | **0.65 barriers/dispatch**；有 drain 的仅约 5 个事件，合计约 3% |
| 8 | kernel 内访存带宽打满 | **排除** | RGP：**内存单元大部分时间活跃度低** |
| 9 | 归约开销大 | **推翻（曾误判）** | 每 workgroup = 1 个 wavefront（64 线程），归约是 subgroup 级，不走 barrier |
| 10 | 工作单元只有 8 个 | **推翻（曾误判）** | 读 shader 源码后确认网格是 (行/NUM_ROWS, 专家数, 1) |

---

## 6. 根因（已确定）

**在飞 wave 数不足 → 访存延迟暴露 → 带宽上不去。**

证据链：

1. **每个 workgroup = 1 个 wavefront = 64 线程**（RGP：1024 wavefronts / 65536 threads / avg 64）。子组内没有更多并行。
2. **占用率 4/16**（RGP Wavefront occupancy：`VGPR 82(88), SGPR 105(128), LDS 8192, Occupancy 4/16`）。资源本身允许更多（VGPR 允许 ~16），实际只到 4。
3. **内存单元大部分时间闲置** → 不是带宽墙。
4. **达成 41 GB/s（上限 97）** → 一半带宽没用上。
5. 每个线程只发得出**少数几次 load**（K 在 64 线程间切分，K=2048 → 2 次迭代；K=512 → 只 1 次且半数线程无活）。

**Little's Law：在途字节 ÷ 延迟 = 带宽。在途字节太少 → 延迟盖不住 → 带宽上不去。**

**注意这与用户最早的直觉一致**（"并发请求不足以支撑这么大的读取量"），但机制不是"读取通道数"，而是**在飞的 wave 数 / 每线程 load 深度**。

### 6.1 受控对照：同样 2.83 MB，速度差 1.7 倍

RGP 抓到的两组 dispatch 形状（Y=8 = `expert_used_count`，X=512 = `expert_feed_forward_length`）：

| dispatch | 含义 | 数据量 | 实测耗时 | 达成带宽 | 占 97 GB/s |
|---|---|---:|---:|---:|---:|
| `vkCmdDispatch(128, 8, 1)` | K=2048 | **2.83 MB** | ~69 µs | **41 GB/s** | 42% |
| `vkCmdDispatch(512, 8, 1)` | K=512 | **2.83 MB** | ~117 µs | **24 GB/s** | 25% |

计算：`8 专家 × 行数 × K × 0.3375 B`（IQ2_M 2.7bpw → 0.3375 字节/参数）。

**两者字节数完全相同、专家数相同、模型相同，唯一差别是矩阵方向。K=512 那档慢 1.7 倍。**

对应 shader 里的守卫（见 7.1）：`K=512, K_PER_ITER=16, BLOCK_SIZE=64` → `col_stride = 1024 > 512` → **只有 `tid < 32` 的线程有活干**。

> ⚠️ **待验证**：这两档同时存在两个变量（半线程空转 + workgroup 数不同：1024 vs 4096），
> 不能简单归因到 `BLOCK_SIZE`。需要单独复现（改 spec constant 跑 A/B）才能定死。

---

## 7. 根因定位的代码依据

### 7.1 关键 shader（已读源码）

文件：`ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq.comp` 与 `mul_mat_vec_base.glsl`（分支 `prism`）。

> ⚠️【2026-09-26 更正】**这是错的 shader。** `mul_mat_vecq.comp` 是 Q8_1 激活 + MMQ 变体，而本模型的热专家路径是 `mul_mat_vec_iq2_s.comp`（参数 `b` 为 F32 时走的）。两者都有 K 切分，但机制不同：
>
> - `mul_mat_vecq.comp`：用 `col_stride = K_PER_ITER * BLOCK_SIZE` 切 K；K=512 时 `col_stride=1024 > 1024` 才会让半数线程不进入
> - **`mul_mat_vec_iq2_s.comp`（实际）：用按超块的分配** —
>   ```glsl
>   const uint num_blocks_per_row = p.ncols / QUANT_K;   // QUANT_K=256
>   const uint blocks_per_wg = gl_WorkGroupSize.x/16;    // BLOCK_SIZE/16
>   const uint itid = tid % 16;  const uint ix = tid / 16;
>   for (uint i = ix; i < num_blocks_per_row; i += blocks_per_wg) calc_superblock(...);
>   ```
>   `BLOCK_SIZE=64` → `blocks_per_wg=4`，`ix ∈ {0,1,2,3}`。
>   **K=512 → `num_blocks_per_row=2`，ix=2,3 的 32 个线程完全不动**（它们算出的部分和为 0，不影响正确性）。
>
> 结论（“K=512 半数线程空转”）仍然成立，但机制是 `blocks_per_wg` 而不是 `col_stride`。

```glsl
// 网格：X = 行/NUM_ROWS（Z 为行维扩展），Y = 专家号
const uint first_row = NUM_ROWS * (gl_WorkGroupID.x + gl_NumWorkGroups.x * gl_WorkGroupID.z);
#ifdef MUL_MAT_ID
    const uint expert_i0 = gl_WorkGroupID.y;
    expert_id = data_ids[expert_i0 + p.expert_i1 * p.nbi1];
#endif

// K 维切给 BLOCK_SIZE 个线程
const uint col_stride = K_PER_ITER * BLOCK_SIZE;
uint num_iters = p.ncols / col_stride;
if (num_iters * col_stride + K_PER_ITER * tid < p.ncols) num_iters++;   // ← K 小时半数线程不进入
```

- `K_PER_ITER`：Q2_0/PQ2_0/quant_k 为 **16**，legacy/MXFP4 为 8，IQ1_S/M 为 32
- `BLOCK_SIZE` / `NUM_ROWS` / `NUM_COLS` 都是 **spec constant**（host 侧选择）
- 归约：`USE_SUBGROUP_ADD` 走 subgroupAdd，否则走 shared memory（本例 LDS 8192 B）
- **不存在** `mul_mat_vecq_id.comp`（404）→ MUL_MAT_ID 复用同一个 shader，靠宏切换

### 7.2 RGP 抓到的实际资源占用

选中事件 `11 vkCmdDispatch(128, 8, 1)`：

```
Total wavefronts           = 1,024
Total threads              = 65,536
Avg threads per wavefront  = 64
CS: VGPR 82 (88)  SGPR 105 (128)  LDS 8192  Occupancy 4/16
```

`LDS 8192 B` 的来源（【2026-09-26 更正】）：**不是 `tmpsh`，而是 IQ2_S 的查找表。**

```glsl
types.glsl:1501:  shared uvec2 iq2s_grid[1024];   // 1024 × 8 B = 8192 B
```

`mul_mat_vec_iq2_s.comp` 的 `main()` 开头调 `init_iq_shmem(gl_WorkGroupSize)`，把这张 8 KB 表拷进共享内存，之后每次 dequant 都查它。

**这一条很重要**：8 KB 是**每个 workgroup 的固定 LDS 开销，与 `BLOCK_SIZE` 无关**。它限住了常驻 workgroup 数，因此：

- 把 `BLOCK_SIZE` 减半（wave32）不会减少 8 KB → 常驻 workgroup 数不变，但在飞线程减半 → **变慢**（实测 −10%）
- 之前从 LDS 反推 `NUM_COLS × NUM_ROWS = 32` 的推理作废

### 7.3 Pipeline 汇总（30 条，头 3 条占 74%）

| hash | Duration | 占比 | 事件数 | 均值 | Occupancy | VGPR | SGPR | Scratch | Wave mode |
|---|---:|---:|---:|---:|---|---|---|---|---|
| `0x13B781EC3D26A6AF` | 731.434 µs | **28.49%** | 9 | 83.175 µs | 12–12 | 82–82 | 105–105 | No | (空) |
| `0xB3E17291018454EB` | 684.047 µs | ~26.6% | | | | | | | |
| `0xCD627A4610FDF93B` | 485.099 µs | ~18.9% | | | | | | | |
| `0xABA9ACEDDA4B6ED9` | 90.110 µs | | | | | 16–16 | 41–41 | **57–57** | |
| …（其余 26 条 ≤ 71 µs，最小的 18~20 µs） | | | | | | | | | |

- 头 3 条 = gate / up / down 三组专家投影
- 有一条 pipeline 的 **Wave mode = `wave64`**，其余为 "No"
- 有一条 `Scratch 57`（寄存器溢出），但只占 90 µs，非主因

**Profile 总量：2,566.915 µs，5 个 queue submission / 5 个 command buffer。**

---

## 8. 附带发现的独立缺陷

### 8.1 PTQ1_0 缺 x86 向量化 vec_dot（CPU decode 0.37 t/s）

- CPU 上 PTQ1_0 decode **0.37 t/s**（2.2 GB/s），比同一 CPU 上的 PQ2_0（5.45 t/s）慢 15 倍
- 只有 decode 塌，prefill 没塌（37.36 t/s，是 GPU 的 77%）→ **mat-vec 路径缺实现**，回退到"反量化成 F32 + 通用浮点乘"
- 分叉提交记录佐证：已给 HIP 补了 `rocm: vectorized HIP PTQ1_0 vec_dot (#211)`，给 PQ2_0 补了 x86（`#263`），但 **PTQ1_0 的 x86 没人做**
- **修复**：参照 PQ2_0 的实现，给 PTQ1_0 写 AVX-512 VNNI 的 `vec_dot`。预期 0.37 → 5 t/s 量级

### 8.2 MoE 的 `-ngl 0` 直接崩溃

- 退出码 `0xC0000409`（STATUS_STACK_BUFFER_OVERRUN）
- 日志：`graph splits = 844 (bs=32) / 61 (bs=1)`，且 `Vulkan0 compute buffer size = 366.25 MiB`
- **`-ngl 0` 并非真 CPU**：仍创建 Vulkan0 缓冲，图在 CPU/Vulkan 之间来回切 61~844 次
- 叠加系统只有 16 GB 可见内存（模型 10.85 GiB + KV + 重打包缓冲）→ 与实际内存压力共同作用
- 结论：**该模型的 CPU 回退路径目前不可用**

### 8.3 文档与实测冲突：PQ2_0 在 Vulkan 上其实可用

- `Bonsai-demo/BACKEND-SUPPORT.md` 的审计基线是 `prism-b10709`，标 Vulkan × PQ2_0 = ❌
- 但 **b10735 实测 PQ2_0 正常跑在 Vulkan 上**：65/65 层在 Vulkan0，`loaded 402 Hadamard-folded weight(s)`，pp512 90.61 / tg128 9.30
- **PQ2_0 的 prefill 是 PTQ1_0 的 1.87 倍**（90.61 vs 48.56），decode 略慢 12%
- 所以选型结论：**长 prompt 场景用 PQ2_0，长输出场景用 PTQ1_0**

### 8.4 Bonsai 2 的其它实测

| -c | S_PP | S_TG |
|---:|---:|---:|
| 8192 | 216.65 | 24.32 |
| 32768 | 220.82 | 24.26 |
| 65536 | 215.56 | 24.24 |
| 131072 | 212.73 | 24.32 |

（上表其实是 MoE 的；Bonsai 相关数据见 4.2 / 4.6。）

---

## 9. 复现方法（全部命令）

### 9.1 自建原始带宽探针

代码：`D:\Project\openhanako\workbench\bwtest\`（`main.cpp` / `read.comp` / `write.comp` / `copy.comp` / `build.bat`）

```bat
cd /d D:\Project\openhanako\workbench\bwtest && build.bat
bw.exe read_u8.spv  read  8      :: 纯读，UNROLL=8
bw.exe write_u1.spv write 1
bw.exe copy_u8.spv  copy  8
```
`UNROLL` 由编译期决定，第 3 个参数只用于字节记账。也可用 `sandbox_permissions="require_escalated"` 之外的普通 shell 跑（不涉及 IPC）。

### 9.2 模型 benchmark

```powershell
cd D:\Project\openhanako\workbench\prism-vulkan
# 单流
.\llama-bench.exe -m "<模型路径>" -ngl 99 -r 2
# 批处理扫描
.\llama-batched-bench.exe -m "<模型路径>" -ngl 99 -c 32768 -npp 512 -ntg 128 -npl 1,2,4,8,16,32
```

### 9.3 RGP 抓帧（**硬获经验，务必按此走**）

**前置条件（缺一不可）**：

1. **必须以非受限令牌 / 提权方式运行**。沙箱的受限令牌路径会报 `Failed to create router.`。
   - 普通 shell（受限令牌）→ 失败
   - `sandbox_permissions="require_escalated"`（非受限令牌）→ **成功**
2. **不要手动启动 Radeon Developer Service**。文档原文：`For Local connections, starting Radeon Developer Service is optional.`
   手动启服务反而可能引发 router 冲突。
3. **端口 27300 无关**。文档：`communication between the two uses pipes, not sockets and ports`。
4. **`AddUserToGroup.bat` 对 Vulkan 无用**。文档限定它只服务于 DirectX 12 的 System Activity。组已加/注册表已设，都不影响本问题。
5. **目标程序必须在 panel 配置好之后再启动**。文档：`must NOT already be running`。

**抓帧命令（终端 A，提权）**：

```powershell
cd D:\Project\openhanako\workbench\rdts\RadeonDeveloperToolSuite-2026-05-28-1806
New-Item -ItemType Directory -Force D:\captures | Out-Null
.\RadeonDeveloperPanelCLI.exe -m profiling -p llama -o D:\captures\moe.rgp `
    --rgp-auto-capture=dispatch:500:100 --rgp-capture-mode dispatch `
    --rgp-counter-collection --rgp-sqtt-buffer-size maximum
```

**终端 B（等 A 就绪后）**：

```powershell
cd D:\Project\openhanako\workbench\prism-vulkan
.\llama-bench.exe -m "<MoE 模型路径>" -ngl 99 -p 0 -n 128 -r 1
```

**已产出**：`D:\captures\moe.rgp`（2,525,889 字节）。

**`.rgp` 格式**：`AMD_RDF` v3 = 512 字节头 + payload（2,507,201）+ 块索引（18,176）。
块索引列出：`AsicInfo / ApiInfo / ClockCalibration / CodeObject×24 / COLoadEvent / PsoCorrelation / SqttData / SpmSession / SpmCounterData×N / QueueInfo×2 / QueueEvent×5 / TraceConfig / SystemInfo / DriverOverrides / TraceProcessInfo / DerivedSpm*`。
**SQTT / SPM 是 AMD 私有 schema，需用 RGP GUI 读，不要尝试逆向。**

**在 GUI 里要看的位置**：

| 视图 | 位置 | 用途 |
|---|---|---|
| Pipelines | OVERVIEW → Pipelines | 按着色器分组，找头号 pipeline |
| Wavefront occupancy | EVENTS → 选一个长耗时的 `vkCmdDispatch` → 下方详情标签 | **停顿/占用率/资源** |
| Instruction timing | 同上标签页 | 逐指令停顿（**部分 RDNA3 硬件无此数据**） |
| Barriers | 左侧栏 | barrier / drain 开销 |
| ~~Memory Chart~~ | — | **RDNA APU 不支持 cache 计数器，会是空的，别看** |

已知问题（官方 release notes）：
- `Cache and ray tracing counter data collection is not currently supported on RDNA based APUs`
- `On some RDNA 3 hardware, detailed instruction timing data will not be available`

### 9.4 读 GGUF metadata

可用 PowerShell 直接解析（GGUF v3：`magic(4) + version(u32) + tensor_count(u64) + kv_count(u64)`，随后 KV 对；遇 `tokenizer.*` 键即可停止以避开巨大的词表数组）。

---

## 10. 单流修复方案（按成本排序）

> 用户主场景是**单发**，所以 `--parallel` 不适用。以下按改动成本从低到高。

> ⚠️【2026-09-26 实测更新】**本表 #2 / #3 / #4 已被实测否证**，详见第 15 节。
> 在源码里加了两个环境开关（同一二进制内 A/B）后实测（单流 MoE，tg128）：
>
> | 配置 | tg128 | 对比基线 |
> |---|---:|---:|
> | **基线**（BLOCK_SIZE=64, NUM_ROWS=4） | **27.72 ± 0.02** | — |
> | NUM_ROWS=8 | 23.40 ± 0.04 | **−16%** |
> | NUM_ROWS=16 | 26.49 ± 0.04 | −4% |
> | NUM_ROWS=2 | 26.02 ± 4.51 | −6%（不稳） |
> | **wave32**（BLOCK_SIZE=32） | 25.01 ± 0.28 | **−10%** |
>
> **基线最优。** 原因见 15.4：这些参数只作用于占 27% 的专家 matvec，且对稠密 matvec 与小算子无效。

| # | 手段 | 改哪 | 预期 | 证据强度 |
|---|---|---|---|---|
| 1 | **推测解码** | 配置 + 一个同词表草稿模型 | **~1.4x**（26 → 36 t/s 量级） | 由批处理曲线外推，未实测 |
| 2 | **`BLOCK_SIZE` 随 K 自适应** | `ggml-vulkan.cpp` 选 spec constant 的逻辑，**不动 shader 逻辑** | 消除 K=512 时半数线程空转，把 24 GB/s 那一档推向 41 | 源码 + 受控对照，**待 A/B 验证** |
| 3 | **改用 wave32** | shader 的 subgroup 设置 | wave64 在 RDNA 上占两个 wave 槽，换 wave32 理论上翻倍 wave 容量 | RGP 显示只有一条 pipeline 是 wave64 |
| 4 | **提高每线程 load 深度** | shader 循环结构（多累加器 / 预取） | 直接增加在途字节 | 机制推理，需 shader 实验 |
| 5 | **调 `NUM_ROWS` 增加 workgroup 数** | host 侧 spec constant | 对 `(128,8,1)` 那一档：1024 wave 只够铺满一轮多 | 推理，与 #2 耦合 |
| 6 | **融合多专家/多层为一次 dispatch** | 结构性改动 | 根本解法，但成本最高 | — |

**目标数值**：把专家矩阵乘从 24~41 GB/s 推向 97 GB/s 上限 → 单流 decode **45~55 t/s**。
**保守预期**：先拿 #1（~1.4x）与 #2（+20~30%），合计 26 → 40~47 t/s。

### 10.1 建议的下一步（按顺序）

1. **验证 #2**：在 `ggml-vulkan.cpp` 里找 `BLOCK_SIZE` 的选择逻辑，改成按 `ncols` 自适应，编译后跑 A/B（同一模型，`(512,8,1)` 形状那一档的耗时）。**这是唯一还没被解释的受控对照**。
2. **验证 #3**：确认哪些 pipeline 是 wave64，尝试把专家 matvec 改成 wave32，看 occupancy 是否从 4/16 上升。
3. **补测 `npl 64`**：在 `-c 32768` 下取干净值，确认聚合上限是否 >65。
4. **`-ngl` 异常**：扫 `ngl 65/68/72/80/90/96`，定位是哪一个层的 Vulkan 实现拖慢 prefill。
5. **若要做推测解码**：需要一个与 MoE 同词表的草稿模型；注意 AGENTS.md 说 Bonsai 2 暂无官方 DSpark drafter。

---

## 11. 未决问题

1. **`-ngl 64` 让 prefill +23% 的机制**。怀疑某层（可能是 GDN 层）的 Vulkan 实现特别差。分叉提交 `vulkan: decline GATED_DELTA_NET raw gates instead of computing them wrong (#239)` 是线索。
2. **`-c` 在高并发下造成 21% 差异的机制**。显存未越界，口径对不上。可能是 KV 排布步长变化导致 DRAM 行命中率下降。
3. **`BLOCK_SIZE` 与 workgroup 数哪个是 K=512 那档变慢的主因**。两个变量耦合，需单独 A/B。
4. **MoE 只用到 25% 带宽，而非专家部分（SSM/attention/router）效率如何**。专家占 74% 时间，但整体只到 25%，说明其它部分也不高效或我的 active-param 估算有偏差。需要补一次针对非专家 op 的 RGP 抓帧。
5. **`wave64` 具体是哪条 pipeline**。Pipelines 表里只有一条显示 `wave64`，但没定位到 hash。

---

## 12. 关键文件清单

| 路径 | 内容 |
|---|---|
| `D:\Project\openhanako\workbench\bwtest\` | 自建 Vulkan 原始带宽探针（`main.cpp` / `*.comp` / `build.bat` / `bw.exe` / `*.spv`） |
| `D:\Project\openhanako\workbench\prism-vulkan\` | PrismML fork 的 Windows Vulkan 二进制（b10735） |
| `D:\Project\openhanako\workbench\rdts\RadeonDeveloperToolSuite-2026-05-28-1806\` | AMD 工具套件（RDP CLI / RGP GUI / RGA CLI） |
| `D:\captures\moe.rgp` | MoE 的 RGP 抓帧（dispatch 500–600，含计数器 + SQTT） |
| `D:\Project\openhanako\workbench\dxdiag.txt` | 显存/内存布局原始输出 |

### 12.1 文档参考来源（外部）

- `PrismML-Eng/llama.cpp` 分支 `prism`，build `prism-b10735-842b188`
- `PrismML-Eng/Bonsai-demo`：`BACKEND-SUPPORT.md`、`MODEL-FORMATS.md`、`AGENTS.md`
- `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq.comp`、`mul_mat_vec_base.glsl`
- RGP 本地帮助：`rdts\RadeonDeveloperToolSuite-2026-05-28-1806\help\{rdp,rgp}\_sources\*.txt`
- MoQE 论文（arXiv 2310.02410）：MoE 专家层比 dense FFN 更抗低比特量化
- Nota AI 博客：MoE 量化的主要风险是 routing instability

---

## 13. 一句话版本

**单流慢是因为每个专家矩阵乘 dispatch 只有 1024 个 wavefront、每 workgroup 仅 1 个 wave、每线程只有几次 load，在飞请求太少盖不住访存延迟——不是带宽，不是并行度，不是占用率被卡，是并发深度不足。修复方向是提高每线程 load 深度、按 K 自适应 `BLOCK_SIZE`、以及用推测解码制造人为 batch。**

---

## 14. 跨模型适用性：这两条 kernel 优化对 Bonsai-2-27B 成立吗

### 14.1 Bonsai 2 27B 的真实维度（GGUF metadata 原文）

```
general.architecture              = qwen35            （dense，64 层）
qwen35.embedding_length           = 5120              ← hidden
qwen35.feed_forward_length        = 17408             ← MLP 中间维
qwen35.attention.head_count       = 24
qwen35.attention.head_count_kv    = 4
qwen35.attention.key_length       = 256
qwen35.attention.value_length     = 256
qwen35.ssm.inner_size             = 6144
qwen35.ssm.time_step_rank         = 48
qwen35.full_attention_interval    = 4                 ← 16 层全注意力 + 48 层 SSM
prism.hadamard.block_size         = 1024
prism.hadamard.sign_widths        = 5120,6144,17408
prism.hadamard.transform          = normalized-sylvester-walsh-hadamard
```

### 14.2 结论一：`BLOCK_SIZE` 自适应对 Bonsai **不适用（空操作）**

触发条件：`K < K_PER_ITER × BLOCK_SIZE = 16 × 64 = 1024`。

Bonsai 全部 matvec 的 K 维：

| op | K | num_iters = K/1024 | 每线程迭代数 |
|---|---:|---:|---:|
| attn QKV / MLP gate+up / SSM in_proj / lm_head | 5120 | 5 | 5 |
| attn O / SSM out_proj | 6144 | 6 | 6 |
| MLP down | 17408 | 17 | 17 |

**最小 K = 5120，是 `col_stride` 的 5 倍。所有线程都有活，每线程 5~17 轮迭代。**
所以这条优化对 Bonsai 是无收益的——它是 MoE 专属的（MoE 的 `expert_feed_forward_length` 只有 512 < 1024，才触发半数线程空转）。

### 14.3 结论二：wave32 理论上适用，但**收益预计明显更小**

- **机制**：RDNA 上 wave64 占两个 wave 槽 → 最大在飞 wave 数减半 → 盖延迟能力减半。
- **适用性**：Bonsai 与 MoE 用的是同一个 `mul_mat_vecq` shader 家族，若该 shader 被编译成 wave64，则两边同时受益。**这一条很可能通用。**
- **但 Bonsai 不是并发饥饿型的**：它已达 62.8~67.0 GB/s（占上限 65~69%），而 MoE 只有 24.4 GB/s（25%）。wave32 的边际收益会小得多。
- **验证方式**：对 Bonsai 抓一次 RGP，看 `Occupancy` 是否也远低于上限（MoE 是 4/16）。若 Bonsai 的 occupancy 本来就接近上限，则说明它不缺并发。

### 14.4 更根本的一点：这两条都不打 Bonsai 的痛点

Bonsai 的 gap 是 **65~69% vs F16 的 93%**，缺口约 25~30 个百分点，性质与 MoE 的 25% 完全不同。

**最可能的短板是三值解包（密集 trit）的指令成本，不是并发深度。** 证据：

- 社区 benchmark 里，PQ2_0 在**每一台** GPU 上都比 PTQ1_0 多拉 20~62% 带宽 → 这是**格式差异**，与并发无关
- 本站实测：同样上下文下 PTQ1_0 62.8 GB/s、PQ2_0 67.0 GB/s（+6.7%），且 PQ2_0 的 prefill 是 PTQ1_0 的 **1.87 倍**（90.61 vs 48.56）

**对 Bonsai 真正有效的手段（按证据强度）**：

| 手段 | 依据 |
|---|---|
| **改用 PQ2_0** | 实测 prefill +87%，decode 略慢 12%；长 prompt 场景直接换 |
| 扫 `-ngl` 定位 prefill 异常层 | 实测 `-ngl 64` 让 prefill **+23%**（48.56→59.46），机制未查明，是 Bonsai 专属线索 |
| 减少解包指令 | 格式固有成本，需 shader 级实验 |
| wave32 | 通用但边际收益小，需先抓 RGP 确认 occupancy |

### 14.5 一句话

**`BLOCK_SIZE` 自适应是 MoE 专属（Bonsai 的 K 全是 5120+，无空转线程）；wave32 通用但对 Bonsai 只是小补。Bonsai 的短板在三值解包的指令成本，优先换 PQ2_0，而不是套用 MoE 的这两条。**

---

## 15. 【2026-09-26 新增】逐 op 实测：时间到底花在哪

### 15.1 方法：**先试 ggml-vulkan 自带的计时，不要直接上 RGP**

```powershell
$env:GGML_VK_PERF_LOGGER="1"; $env:GGML_VK_PIPELINE_STATS="1"
llama-bench.exe -m <MoE> -ngl 99 -p 0 -n 4 -r 1
```

每个 graph 评估后打一张 `Vulkan Timings` 表，含逐 op 调用次数、单次耗时、累计、GFLOPS。
**比 RGP 便宜得多，也精确得多**（不用提权、不用 GUI、不用截图读图）。相关开关：

| 环境变量 | 作用 |
|---|---|
| `GGML_VK_PERF_LOGGER` | 逐 op 累计耗时表 |
| `GGML_VK_PIPELINE_STATS` | 每个 pipeline 的 VGPR/SGPR/LDS/scratch |
| `GGML_VK_PERF_LOGGER_CONCURRENT` | 并发模式 |
| `GGML_VK_PERF_LOGGER_FREQUENCY` | 打印频率 |

> ⚠️ **第一张表包含预热与着色器编译，数字失真 30~1000 倍**（会出现“一次 matvec 9273 µs”这种假数）。**必须看后续几张。**

### 15.2 一个 token 的真实构成（稳态，总计 38.06 ms）

| 类别 | 累计 ms | 占比 |
|---|---:|---:|
| **稠密 matvec（非专家）** | ~16.2 | **43%** |
| **专家 matvec（MUL_MAT_ID）** | ~10.3 | **27%** |
| 小算子（CPY / GET_ROWS / SCALE / ADD / MUL / …） | ~7.2 | 19%（**已拆解，见 15.9**） |
| SSM（GATED_DELTA_NET + SSM_CONV） | 2.2 | 5.7% |
| Norm / RoPE / 杂项 | ~2.1 | ~5% |
| **Flash Attention** | **0.245** | **0.6%** |

关键条目（稳态单次耗时，取多张表的一致值）：

| op | 调用数 | 单次 µs | 累计 µs |
|---|---:|---:|---:|
| MUL_MAT_ID_VEC iq2_s m=512 n=8 k=2048（gate/up 专家） | 80 | 62~106 | ~4950~8470 |
| MUL_MAT_ID_VEC iq2_s m=2048 n=8 k=512（down 专家） | 37 | 128~135 | ~4720 |
| MUL_MAT_VEC q5_K m=248320 k=2048（**lm_head**） | **1** | **~3000** | 3000 |
| MUL_MAT_VEC q4_K m=8192 k=2048（attn_qkv） | 30 | 90~127 | ~3500 |
| GATED_DELTA_NET | 30 | 68 | 2046 |
| MUL_MAT_VEC iq2_s m=4096 k=2048（attn_gate） | 30 | 57~79 | ~1800 |
| MUL_MAT_VEC iq2_s m=2048 k=4096（ssm_out） | 30 | ~58 | ~1730 |
| CPY | 60 | 25~36 | ~1590 |
| GET_ROWS | 61 | 24 | ~1495 |
| SCALE | 60 | 25 | ~1490 |
| MUL_MAT_VEC f32 m=256 k=2048（router） | 40 | 30 | ~1220 |
| MUL_MAT_VEC iq2_s m=2048 k=512 | 38 | 19~28 | ~715 |
| FLASH_ATTN_EXT dst(256,16) q(256,1,16) k(256,256,2) | 10 | **24** | **245** |

### 15.3 达成带宽：按 op 形状差 6 倍

| op | m（输出行数） | 量化 | 达成带宽 |
|---|---:|---|---:|
| lm_head | 248320 | Q5_K | **~115 GB/s** |
| attn_qkv | 8192 | Q4_K | ~81~94 GB/s |
| router | 256 | F32 | ~69 GB/s |
| attn_gate / ssm_out | 4096 / 2048 | IQ2_S | ~45 GB/s |
| 专家 gate/up | 512×8 专家 | IQ2_S | ~43 GB/s |
| **专家 down** | 2048×8 专家 | IQ2_S | **~20 GB/s** |
| iq3_s m=2048 k=512 | 2048 | IQ3_S | ~21 GB/s |

**规律：效率随输出行数 m 单调上升。** m 大的（lm_head、attn_qkv）已接近上限；m 小的（专家、SSM 投影）只有 20~45 GB/s。

### 15.4 本节推翻的旧结论

| 旧结论 | 实测 |
|---|---|
| 专家占 74% 时间 | **专家占 27%**（74% 是 100 次 dispatch 采样窗口的偏差） |
| 专家达成 24 GB/s | down 确实 ~20，但 gate/up ~43，专家整体 20~43 |
| Flash attention 可能是大头 | **0.6%，可忽略**（与“上下文不影响 tg”一致） |
| 瓶颈是“在飞 wave 数不足” | 更准确：**小 m 的 GEMV 并行度不足**，且大头在稠密 matvec，不在专家 |
| 第 10 节 #2/#3/#4 会有效 | **实测全部无效或变慢** |

### 15.5 修正后的下一步

1. **不要再调专家 matvec 的 spec constant** — 天花板 27%，且实测无效。
2. **lm_head 一次 dispatch 就占 8%**（3000 µs / 350 MB）。已达 115 GB/s，接近上限，短期无空间。
3. **attn_qkv（30 次，~9%）** 同理，已接近上限。
4. **最可疑的一块：小算子的固定开销**。CPY/SCALE/GET_ROWS 三个加起来 ~12%（181 次 × ~25 µs），而 `ADD`/`MUL`/`SIGMOID` 只要 2~3 µs。为什么它们要 25 µs，需要单独定位（带宽？延迟？同步？）。
5. **减少 dispatch 数量（融合）** 仍是唯一能同时改善多项的结构性手段，与批处理实测 2.7 倍的方向一致。

### 15.6 构建环境（可复现）

源码：`D:\Project\openhanako\workbench\prism-llama.cpp`（HEAD `e311ed3`）
构建产物：`prism-llama.cpp\build-vs\bin\Release\`（`llama-bench.exe` / `llama-batched-bench.exe` / `ggml-vulkan.dll`）
构建脚本：`D:\Project\openhanako\workbench\vk-build.bat`（`configure-vs` / `build-vs`）

两个环境开关（本次为 A/B 加的，不设时行为与 stock 完全一致）：

| 变量 | 作用 |
|---|---|
| `GGML_VK_DMMV_SUBGROUP_SIZE` | 覆盖 matvec 的 subgroup size（即 `BLOCK_SIZE`） |
| `GGML_VK_DMMV_NUM_ROWS` | 覆盖 `rm_stdq/rm_kq/rm_iq`（即 `NUM_ROWS`） |

**两个构建坑**（命中过，不要再踩）：

- **Ninja 生成器 configure 卡死在 `Detecting C compiler ABI info`**（`mspdbsrv` 命名管道）。改用 `Visual Studio 17 2022` 生成器，28 秒过。
- **MSBuild 在一次性沙箱命令里卡死在 `Checking File Globs`**，在 tty 会话里正常。编译必须走 tty。
- `cl` / `vcvars64` / `glslc` 本身都没问题（用 `probe-cl.bat` 验证过）。

### 15.7 NUM_ROWS 完整扫描（每个几何旋钮的默认值都已经是最优）

`-p 0 -n 128 -r 5`：

| NUM_ROWS | tg128 | 对比默认 |
|---:|---:|---:|
| **4（默认）** | **33.51 ± 0.42** | — |
| 2 | 29.30 ± 0.11 | −13% |
| 1 | 20.07 ± 0.05 | **−40%** |
| 8 | — | −16%（见 15.4 的 -r 2 数据） |

**结论：减小 NUM_ROWS（增加 workgroup 数）反而变差。**

之前从“效率随 m 上升”推出“增加 workgroup 数会有效”是误读：
大 m 的 op 之所以效率高，是因为**总活儿多**，不是因为切得碎。把 NUM_ROWS 降下来只是让每个 workgroup 重新读一遍激活、并放大每 workgroup 的固定开销。

**至此三个几何旋钮全部扫完，默认值均为最优：**

| 旋钮 | 测过的值 | 最优 |
|---|---|---|
| `BLOCK_SIZE`（= subgroup size） | 32 / 64 | **64** |
| `NUM_ROWS` | 1 / 2 / 4 / 8 / 16 | **4** |
| subgroup size | 32 / 64 | **64** |

**这意味着“BLOCK_SIZE 自适应”和“改 wave32”不是“没实现好”，而是这个 kernel 的几何已经调到了最优。** 想再快只能改算法或改 dispatch 结构，不能改这三个数。

### 15.8 已作废：长上下文额外的 20%

【当节结论已推翻】曾以为长上下文会附加 ~20% 开销。
**交错重测否决了这个假设——上下文对 decode 无影响，详见 4.5。**
那次看上去的“台阶”是会话级漂移（见 4.5 的坑点）。存档以免重走。

因此 15.2 的剖析就是全部：单发 MoE 的 38 ms 已经全部归属清楚，不存在“还不清楚的一块”。

---

## 15.9 【2026-09-26】dispatch 数量：已试，**不是杠杆**

先试了两个方向，都被实测否掉：

### 15.9.1 减少 submit 次数 —— 无效

默认 `max_nodes_per_submit = 100`（一个 token 的 graph 约 1400 个节点）。
交错 A/B/A/B/A 测五个点：

| `GGML_VK_MAX_NODES_PER_SUBMIT` | tg128 |
|---|---:|
| 100（默认） | 27.75 / 27.71 / 27.64 |
| 1000 | 27.75 |
| 10000 | 27.82 |

**全部落在 ±0.1 噪声内。submit 边界不是瓶颈。**
（相关旋钮：`GGML_VK_MAX_NODES_PER_SUBMIT`，在 `ggml-vulkan.cpp:6492`）

### 15.9.2 那些“小算子”其实是大搬运，不是固定开销

把 perf logger 的兑底分支也补上形状标注后，真相出来了：

```
CPY d=f32(524288) s0=f32(128,128,32):            30 x 48.2 us = 1447 us
GET_ROWS d=f32(524288) s0=f32(524288) s1=i32(1): 30 x 47.5 us = 1426 us
CPY d=f32(24576) s0=f32(3,8192):                  30 x 4.1  us = 122  us
GET_ROWS d=f32(24576):                            30 x 3.8  us = 115  us
```

`524288 × 4 B = 2.1 MB` 每次，**30 次对应 30 个 SSM 层**。搬运量 4.2 MB/48 µs = **87 GB/s，已经贴着上限**——它们不是空转，是在搬 GDN 的递归状态（每层 2 MB，每层每 token 搬两遍）。

**所以“减少 dispatch 数量”暂时无法拿到收益**：真正多的那些算子每个都有实际数据要搬，而且已经接近带宽上限。想减少它们的代价，得改 llama.cpp 的 GDN 图结构（状态原地更新），不属于 kernel 参数能碰的范围。

### 15.9.3 一个附带发现：偶发 2 ms 卡顿

本次剖析里 `MUL_MAT_VEC q4_K m=512 n=1 k=2048` 单次从 ~11 µs 跳到 **218.95 µs（20 倍）**，
另一张表里 `RMS_NORM_MUL RMS_NORM(128,32,1,1)` 也出现过 ~76 µs（常态 6 µs）。
**每张 graph 的总时长在 36.1~39.4 ms 之间波动，说明存在偶发的毫秒级卡顿**（推测是异步提交的 fence 或临时分配）。还没定位。

---

## 15.10 【2026-09-26】真正的头号问题：**IQ 格式矩阵乘只跑到 ~45 GB/s**

### 15.10.1 同一形状、同一 kernel 家族，只有格式不同

| 形状 | 格式 | grid LDS | 单次 µs | 达成带宽 |
|---|---|---:|---:|---:|
| m=2048 n=8 k=512 | **IQ3_S** | 2 KB | 138.0 | **28 GB/s** |
| m=2048 n=8 k=512 | **IQ2_S** | 8 KB | 142.4 | **19 GB/s** |
| m=8192 n=1 k=2048 | **IQ2_S** | 8 KB | 111.8 | **48 GB/s** |
| m=4096 n=1 k=2048 | IQ2_S | 8 KB | 60.1 | 45 GB/s |
| m=2048 n=1 k=4096 | IQ2_S | 8 KB | 59.0 | 45 GB/s |
| m=8192 n=1 k=2048 | **Q4_K** | **0** | 124.2 | **76 GB/s** |
| m=248320 n=1 k=2048 | **Q5_K** | **0** | 2984 | **117 GB/s** |
| m=256 n=1 k=2048 | **F32** | **0** | 31.4 | **67 GB/s** |

**结论：IQ 格式封顶在 45~48 GB/s，而 LDS=0 的格式是 67~117 GB/s。差 1.6~2.5 倍。**

`GGML_VK_PIPELINE_STATS` 已经直接给出证据：

```
pipeline stats for mul_mat_vec_q4_k_q8_1_f32:
  ldsUsageSizeInBytes: 0          ← Q4_K，达成 76 GB/s
```

而 IQ2_S 是 8192（`shared uvec2 iq2s_grid[1024]`）。
**最一致的解释：那 8 KB 查找表限住了常驻 workgroup 数。**
这也解释了为什么三个几何旋钮全无效：它们都不改变这 8 KB。

### 15.10.2 这笔账有多大

汇总所有 IQ 格式 matvec 的耗时：

| op | 累计 µs |
|---|---:|
| MUL_MAT_ID_VEC iq2_s m=512 n=8 k=2048（gate/up 专家） | 4975 |
| MUL_MAT_ID_MUL iq2_s m=2048 n=8 k=512（down 专家） | 5267 |
| MUL_MAT_VEC iq2_s m=4096 k=2048 | 1803 |
| MUL_MAT_VEC iq2_s m=2048 k=4096 | 1769 |
| MUL_MAT_VEC iq2_s m=512 k=2048 | 1297 |
| MUL_MAT_VEC iq2_s m=8192 k=2048 | 1118 |
| MUL_MAT_VEC iq2_s m=2048 k=512 | 727 |
| MUL_MAT_ADD iq3_s m=2048 k=4096 | 656 |
| MUL_MAT_VEC iq2_s m=32 k=2048 | 467 |
| 其余 iq3_s | 468 |
| **合计** | **~18 550 µs = 49%** |

**半个 token 花在 IQ 格式的矩阵乘上，而它们只跑到 ~45 GB/s。**

若能把它们推到 90 GB/s：

- IQ 部分 18.6 ms → ~9.3 ms
- 总时长 37.7 → ~28.4 ms
- **单流 27.7 → 36.8 t/s，+33%**

### 15.10.3 下一步的具体做法

把 `iq2s_grid` 从 LDS 里拿出来——它只需要被动态索引一次。三个层次：

| 方案 | 成本 | 风险 |
|---|---|---|
| a. 直接把 `shared` 改成 `const` 数组直接索引 | 低 | **可能更慢**：glslc 会把 8 KB 常量数组降级成 Private（scratch）内存 |
| b. 改成 UBO / SSBO 绑定（host 上传一次） | 中：要加 descriptor + host 侧代码 | 常量缓存路径，预期接近 LDS 速度 |
| c. 压缩表：每个字节的值域实际只有 {0x08,0x19,0x2b}，即一个 2-bit 编码，8 KB → 2 KB | 高：要改 dequant 数学 | 能用 |

**先做 a 做快速否证（改一行 + 重编 + 剖析，约 15 分钟）：**
若 a 变慢，说明常量路径不行，直接上 b；若 a 就变快了，说明 LDS 确实是主因。

> 另外记一条：`mul_mat_vec_q4_k` 的 LDS 是 0，是因为它的降维走 `USE_SUBGROUP_ADD_NO_SHMEM`，连 `tmpsh` 都不分配。IQ 的 8 KB 完全来自查找表。

### 15.10.4 【结果】方案 a 成功：**+18.8%，已验证**

完整实验记录、patch 与二进制备份见 `D:\Project\openhanako\workbench\iq-lds\`（`CHANGELOG.md` / `patches\` / `bin\` / `logs\`）。

**改动**（`types.glsl` 的 `#if defined(DATA_A_IQ2_S)` 块，共 6 行）：

```glsl
// 删掉 shared 数组与拷贝循环，改为：
#define iq2s_grid iq2s_grid_const
void init_iq_shmem(uvec3 wgsize) { }   // 不再拷到 LDS
```

**实测（同一会话交错 A/B/A，两个二进制并存）**：

| | tg128 |
|---|---:|
| 基线（LDS 8192） | 33.63 ± 0.03 |
| **方案 a（LDS 0）** | **39.87 ± 0.18** |
| 基线（LDS 8192） | 33.54 ± 0.02 |

**+18.8%。正确性：PPL 逐位一致（双版均 4.5671 ± 0.25628），不是“算错所以快”。**

实现细节：LDS 8192 → **0**，`scratchMemUsageInBytes` 仍为 **0**（glslc 未降级为 Private），VGPR 88 未变。
**那 8 KB 到底被安置在哪里尚未查明**（不在 LDS、不在 scratch、不在 VGPR），是下一步可做的事。

**推广到 `iq3s_grid` 已试，无可测收益**（它只有 2 KB，且 IQ3_S 只占 token 的 ~3%）。
**方案 b 判定被 a 覆盖**（b 的目标是“把表从 LDS 拿出来”，a 已经做到 LDS=0）。
**方案 c 已设计未实施**（8 KB → 2 KB，但在 LDS 这个维度不可能优于 a）。

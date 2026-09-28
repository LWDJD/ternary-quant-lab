# PrismML 三值格式（PTQ1_0）：把 MoE 模型做成"Bonsai 2 式"三值模型

> **目标**：把 `qwen35moe` 架构的 Qwen3.6-35B-A3B 做成 PrismML 的 PTQ1_0（1.75 bpw 三值），
> 即"MoE 版的 Bonsai 2"。
>
> 最后更新：2026-09-27
> 工作目录：`D:\Project\openhanako\workbench\prism-pack\`

---

## 0. 一句话现状

**流水线跑通了、旋转器定量验证正确、速度内存问题已解决。
但"能用的质量"尚未达成**（最好的版本能正确回答"法国首都"，但会陷入重复）。

---

## 1. 结论汇总

### 1.1 已验证成立（附证据）

| 事实 | 证据 |
|---|---|
| **方案本身成立** | PrismML 的 Ternary-Bonsai-2-27B PTQ1_0（5.54 GB）在**本机本 runtime** 上产出连贯高质量文本，10.42 t/s |
| **`qwen35moe` + PTQ1_0 全链路可用** | 加载零错误；Vulkan 实际调用 `mul_mat_vec_id_ptq1_0_f32`（MoE 专家）；fork 自报 `1.75 bpw ternary (group 128)`；tg8 31.70 t/s |
| **我的旋转器正确** | 旋转后写 F16（不量化）→ **PPL 71.98 vs 原版 71.91**（MiniCPM5-1B，差 0.15%） |
| **显式符号实现正确** | 加入 ±1 符号后，旋转+F16 仍为 **PPL 71.98** |
| **元数据契约被加载器消费** | 把 `prism.hadamard.block_size` 原地改成非法的 3 → 报 `invalid prism.hadamard.block_size: 3` |
| **打包格式正确** | 与 fork 自身 `llama-quantize --outtype PTQ1_0` 的产物：**每块 scale 逐字节相同、95.5% 的 qs/qh 字节相同** |
| **运行时变换的形态** | `src/llama-impl.h:57`：`n = rot->ne[0]`，激活 reshape 成 `(n, N/n)`，做 `H_norm`，带 `GGML_HINT_SRC0_IS_HADAMARD` |

### 1.2 已被否证的假设（负结论）

| 假设 | 结论 | 证据 |
|---|---|---|
| **不旋转的三值化可行** | ❌ | 四组交叉：MiniCPM(**F16 源**)→PTQ1_0 乱码；MoE(IQ2_M)→PTQ1_0 乱码；MoE(Q4_K_M)→**TQ1_0** 乱码；而 BitCPM 8B(**三值训练**) 连贯、Bonsai 2(**带旋转**) 连贯 |
| **换格式能解决** | ❌ | PTQ1_0(g128) 与 TQ1_0(g256) 未旋转时都是乱码——格式不是变量，旋转才是 |
| **`d = amax` 是正确的量化尺度** | ❌ | 旋转后组是近高斯；`amax≈3.5σ` → 仅 **16.6%** 的值能超过 `0.5·amax`，83% 被压成 0。真品 Bonsai 2 非零比例 **67.2%**，反推尺度约 **0.75σ** |
| **`quantize_ptq1_0` 可用于稠密权重** | ❌ | 源码自述 "lossless for checkpoints that are **already ternary** at group 128" |
| **imatrix 能提升三值质量** | ❌ | Hadamard 是正交变换，会把各输入通道的方差**搅成平均值**→ 旋转后每个通道重要性近似相同。这正是 PrismML 注释 "an imatrix has no role" 的含义 |
| MiniCPM5-1B 可作低比特试验台 | ❌ | 用**厂商自带量化器**：Q2_0(3.47bpw) PPL 2.36e6、TQ2_0(3.35bpw) 7.86e6、PTQ1_0(3.16bpw) 8.71e6；Q4_0(4.88bpw) 才 82.93。**该模型在 3.5 bpw 以下就崩** |
| **加线程能加速** | ❌（在修复 churn 之前） | 旧代码 6 线程 205 s vs 串行 167 s（更慢）；修复后 8 线程 41.7 s vs 85.5 s（2.05×） |

---

## 2. 质量实测

### 2.1 MiniCPM5-1B（`llama` 架构，验证旋转器用）

| 变体 | PPL（corpus-small, c=512） |
|---|---:|
| 原版 F16 | **71.91** |
| 旋转 + F16 | **72.02** |
| 旋转 + 符号 + F16 | **71.98** |

**结论：旋转器与符号实现在数学上无损。**

### 2.2 Qwen3.6-35B-A3B（目标模型）

| 版本 | bpw | 大小 | PPL | 文本 |
|---|---:|---:|---:|---|
| IQ2_M（基线，现用） | 2.69 | 7.60 GB | **4.5593** | 正常 |
| Q2_K（常规方法，无需 imatrix） | 2.98 | 12.33 GB | **37.49** | — |
| **本实现 block 512** | **1.75** | **7.39 GB** | **66.37** | 词沙拉 ✗ |
| **本实现 block 1024** | **1.75** | **12.14 GB** | 未测（显存不足） | **"The capital of France is Paris." ✓ 随后重复** |

**要点**：
- 未旋转的三值是 **~10⁶**；本实现 **66** —— **四个数量级的改善**，证明旋转/符号/拟合 scale 全部起效
- **block 1024 明显优于 512**（正确事实性回答 vs 词沙拉），但代价 **+4.75 GB**（`ffn_down_exps` 等 80 个张量被迫保持源精度）
- 66 仍远高于基线 4.56，且文本会陷入重复——**尚不可用**

---

## 3. 方法（完整规格）

### 3.1 旋转

whitepaper（`Bonsai-demo/bonsai-2-27b-whitepaper.pdf` §2.4）：

```
R = (1/√n) · H_n · S          n = block size（Bonsai 2 用 1024）
f(x) = W(R·x)
```

运行时（`llama-impl.h:57`）对激活做：先 `x ⊙ s`，再分块归一化 Walsh–Hadamard。
反推折叠式（H、S 均为对合矩阵）：

```
W' = W · S · H_norm          先按输入维乘符号，再乘归一化 Walsh
```

**精确可逆，零误差。**

### 3.2 运行时构造的 H

`src/llama-model.cpp:2047-2058` 逐位构造：

```cpp
scale = 1/sqrt(block_size);
data[row*bs + col] = (parity(row & col) & 1) ? -scale : scale;
```

即 `H_norm[i][j] = (-1)^popcount(i&j)/√n`（Sylvester 序）。`fwht_norm()` 与之逐位相等。

### 3.3 符号（必需，非装饰）

Hadamard 第一行全为 +1，恒等符号下 `x'[0] = n·mean(组)` → 权重每组的非零均值变成**巨大离群值**，单独顶起整组 scale。随机 ±1 后 `x'[0] ≈ 0`。

**约束**：符号按**宽度**（= 张量 `ne[0]`）索引，**同宽张量必须共用同一条向量**——运行时按宽度缓存符号张量，且对"同一激活+同一 rot"的变换在图里做去重（`hadamard_memo`）。

**当前实现**：每个宽度一条由 `sha256(seed:width)` 决定的确定性随机 ±1 向量。
**未验证的改进**：PrismML 的符号可能不是随机的，而是**优化过的**（SpinQuant 那类做法）。这是一个真实的质量杠杆。

### 3.4 量化尺度（与 fork 参考实现**故意不同**）

fork 的 `quantize_row_ptq1_0_ref` 用 `d = max|x|`，只适合已三值的检查点。
本实现改用**最小二乘三值拟合**（交替迭代）：

```
t = clamp(round(x/d), -1, +1)
d = |<x,t>| / <t,t>
```

收敛实测（旋转后 128 组）：

| iters | 相对误差 | 非零比例 |
|---:|---:|---:|
| 1 | 58.64% | 35.0% |
| 2 | 48.45% | 45.9% |
| **3** | **45.12%** | **50.7%** |
| 6 | 43.61% | 53.7% |
| 10 | 43.54% | 53.8% |

（真品 Bonsai 2：非零 67.2%）**取 3 次，到 10 次仅差 1.6 个百分点。**

### 3.5 块布局（`ggml-common.h`）

```c
#define QK_PTQ1_0 128
typedef struct {
    uint8_t   qs[(128 - 4*128/64)/5];   // 24 B，每字节 5 个 trit → 120 个
    uint8_t   qh[128/64];               //  2 B，每字节 4 个 trit →   8 个
    ggml_half d;                        // 尺度 —— 在末尾，不在开头
} block_ptq1_0;                         // 28 B / 128 权重 = 1.75 bpw
```

`qs` 两级：`c=16` 吃元素 0..79 → `qs[0:16]`；`c=8` 吃 80..119 → `qs[16:24]`。
解包时 `uint8_t` 的**按 256 取模**就是三进制位提取本身，不能省。

### 3.6 GGUF 元数据契约

```
prism.hadamard.version      u32  1
prism.hadamard.block_size   u32  512 或 1024
prism.hadamard.transform    str  "normalized-sylvester-walsh-hadamard"
prism.hadamard.axis         str  "input-last-dimension"
prism.hadamard.sign_mode    str  "explicit"
prism.hadamard.sign_widths  arr  [宽度...]
prism.hadamard.sign_values  arr  [±1 ...]（按 widths 顺序拼接）
prism.hadamard.weight_names arr  [已折叠张量名]
```

**block 只能全局单值**（运行时用 `rot->ne[0]`）。MoE 专家 down 投影的输入维是 **512**，
所以 block 1024 下这些张量**无法折叠**，只能保持源精度 → **+4.75 GB**。这是当前最大的结构性约束。

### 3.7 折叠/保留策略（block 512 时）

- **折叠 + 三值化**：`output` / `attn_q/k/v/qkv/gate/output` / `ffn_gate/up/down` / `ffn_*_exps` / `ffn_*_shexp`（341 个）
- **保持源精度**：所有 norm、`ssm_a`、`ssm_dt.bias`、`ssm_conv1d`、`ssm_alpha/beta`（whitepaper Table 2 明列）、router、`ssm_out`、`token_embd`
- **block 1024 时额外保留**：`ffn_down_exps`、`ffn_down_shexp`（80 个，ne0=512）

---

## 4. 代码

```
prism-pack\
├─ pack_gguf.py       主工具：GGUF → 旋转 → PTQ1_0，单趟流式，可断点复用
├─ ptq1_0.py          PTQ1_0 打包/解包 + 向量化三值量化（含最小二乘拟合）
├─ prism_pack.py      折叠数学 + 自检（对 runtime 的 R 逐位验证）
├─ patch_kv.py        原地改 KV（探测加载器是否消费元数据）
├─ test_roundtrip.py  GGUF 读写往返自检
└─ logs\              全部原始日志
```

### 用法

```powershell
# 正式产物（block 512 + 8 线程）
python pack_gguf.py <in.gguf> <out.gguf> --block 512 --iters 3 --jobs 8

# block 1024（ffn_down_* 自动保持源精度，+4.75 GB）
python pack_gguf.py <in.gguf> <out.gguf> --block 1024 --iters 3 --jobs 8

# 只旋转不量化 —— 数学上必须与原模型等价，用于验证旋转器
python pack_gguf.py <in.gguf> <out.gguf> --block 512 --no-quant

# 让匹配的张量保持 F16 —— 二分定位是哪个张量被三值化后毁掉模型
python pack_gguf.py <in.gguf> <out.gguf> --keep-f16 "blk\.\d+\.ffn_.*"

# 断点复用：跳过已算好的 <dst>.parts（改了 block 或 iters 后必须重算）
python pack_gguf.py <in.gguf> <out.gguf> --block 1024 --jobs 8 --reuse
```

### 环境

- Python：`C:\Users\15081\AppData\Local\Programs\Python\Python313\python.exe`（自带 numpy + pyyaml；默认的 `python` 缺 pyyaml）
- gguf-py：`prism-tip\gguf-py`（以 `sys.path` 注入，**不改动该目录**）
- 运行时：`prism-tip\build-vs\bin\Release\`（`llama-completion` / `llama-perplexity` / `llama-imatrix` / `llama-quantize`）

### 性能与内存（重要工程记录）

**原始版本 35B 要 ~85 分钟、峰值内存 7.4 GB（2 GB 的 MiniCPM）。根因是我自己的 bug**：
分块只切了中间结果，却把每块 **`parts.append()` 累加**、最后 `np.concatenate` →
单个张量同时存在 **源 2 GB + 累加 2 GB + 拼接 2 GB + 临时 1.4 GB**。
GB 级分配引发大量缺页中断 → **CPU 只有 5%（1/20 核）**，而且 **numpy 分配器的内部锁被争用 → 加线程反而更慢**。

**修复**：
1. **只累加打包后的字节**（比 f32 小 20 倍），源按行切片、边解量化边折叠
2. `lroundf`（`floor(abs+0.5)*sign` = 4 个临时数组）→ `np.rint`（1 op）
3. FWHT 蝶形改成就地运算（每级 1 份拷贝而非 2 份）
4. 拟合迭代 6 → 3

**结果**：

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| MiniCPM 串行 | 173 s | 85.5 s |
| MiniCPM 8 线程 | 205 s（更慢） | **41.7 s（2.05×）** |
| **35B 全流程** | **~85 分钟** | **~21 分钟** |
| 内存峰值 | 7398 MB | **2592 MB** |

线程数扫描（20 逻辑核）：1→85.5s，4→47.7s，**8→41.7s**，12→41.1s，16→41.0s（饱和于 8）。

---

## 5. 踩过的坑

1. **块布局里 `d` 在末尾**，不在开头。按 `d, qs, qh` 写会通篇差 2 字节。
2. **`qs` 解码必须按 uint8 截断**（`(qs*3^n) & 0xFF`）。
3. **`lroundf` 是"四舍五入远离零"**，`np.rint` 是"取偶"。（修复 churn 后已改用 `np.rint`，只影响精确 .5 的边界。）
4. **`t.shape` 是 ggml 的 ne 顺序（`ne0 = shape[0]`）**；`data.shape` 是转置视图，量化类型的最后一位是**字节数**（Q6_K 的 2048 宽行 = 1680 字节）。
5. **纯 Python 逐块量化器是性能杀手** → tensordot + base-3 权重矩阵，52M 元素/秒。
6. **`GGUFWriter.add_array(key, val)` 只有两个参数**，类型自动推断。
7. **不要复制 `GGUFReader` 的 `GGUF.*` 伪字段**，否则写出重复键。
8. **`llama-completion` 默认不打印加载细节**；PowerShell 的 `2>` 会丢失原生进程日志并产出 UTF-16。要看日志用 `Start-Process -RedirectStandardError`。
9. **`gguf-py` 遇到重复键会 KeyError**，排查时用纯 Python 解析器更省事。
10. **`llama-quantize` 拒绝从 K-quant/IQ 重新量化**，需 `--allow-requantize`；`IQ2_XXS` 等还**强制要求 imatrix**。
11. **`llama-completion`/`perplexity` 必须显式给 `-c`**，否则用模型默认的 262144 上下文，KV cache 直接 OOM。
12. **`ProcessPoolExecutor` 在本沙箱被拒**（WinError 5）→ 用 `ThreadPoolExecutor`（numpy 逐元素运算会释放 GIL）。
13. **`Select-Object -First N` 会提前终止上游管道并杀死原生进程** —— 用 `Tee-Object` 写日志后另行读取，不要在同一管道里 `-First`。（这曾导致两次"作业无声死亡、产物为零"。）
14. **重跑前清理 `<dst>.parts`**：它只依赖张量+block+iters，改了其中任一项必须重算。

---

## 6. 接下来可以选择的路

### (a) 冲"可用质量"：混合精度配方

**依据**：block 1024 已经能正确回答事实性问题（block 512 是词沙拉），说明**保留更多高精度张量能换回质量**。
`--keep-f16` 开关已就绪，可以像做二分那样逐类保留，成本是一次 21 分钟的重打包。

候选顺序（按敏感度猜测）：
1. `token_embd` + `output`（词表相关，已默认保留 `token_embd`）
2. `attn_*`（block 512 时保留它把 PPL 从 1.77e6 降到 8.4e4——但那是 MiniCPM 的无效数据）
3. `ffn_*_shexp`（共享专家，每条 token 都过）
4. 首尾若干层整体保留

**目标**：找到"每 +1 GB 换多少 PPL"的曲线，看是否能在 ~9-10 GB 内把 PPL 压到 ~10 以内。

### (b) 优化符号（而不是随机）

PrismML 的 `sign_mode = explicit` 不一定等价于"随机"。SpinQuant 那类工作会**学习**旋转/符号以最小化量化误差。
当前实现是确定性随机。可以试：
- 用一小段校准数据**搜索**每个宽度的符号（贪心/坐标下降）
- 或者用更大的 block（1024）配合符号，让旋转更充分

**这是目前唯一还没动过的、原理上站得住的杠杆。**

### (c) 换更好的源

| 源 | 大小 | 可行性 |
|---|---:|---|
| Q4_K_M（现用） | 19.71 GB | ✅ 本地已有 |
| Q8_0 | ~37 GB | ⚠️ 需腾出磁盘（D: 29 GB / C: 22 GB） |
| **f16（原始）** | **~65 GB** | ❌ 需腾出 ~70 GB |

**f16 是唯一能对标 PrismML 质量的源**（他们是从全精度做的）。若你要认真追这个目标，先解决磁盘。

### (d) 在"有参照的稠密模型"上验证流水线

**最强的判断实验**：找一个 **7-8B 的稠密、白名单架构**（`llama`/`qwen3`）模型，
用同一套流水线三值化，和它的 Q4_0/Q8_0 对照。

- 若稠密模型上能接近 PrismML 的保留率 → **问题在 MoE 的结构特性**
- 若稠密模型上也差很多 → **我的流水线还缺东西**（大概率是符号优化）

本地没有合适的稠密模型（K2-Horizon 是 `k2-horizon` 架构、不在白名单），需要下载一个 ~15 GB 的 f16。

### (e) 接受现有产物

- **block 512，7.39 GB**（1.75 bpw，PPL 66，词沙拉）——能加载能跑，质量不可用
- **block 1024，12.14 GB**（1.75 bpw，能正确回答事实性问题但会重复）——比 IQ2_M（7.60 GB）**更大且更差**

**从纯工程角度，(e) 没有价值**——除非你的目标是"验证这条技术路线可行"，那它已经达成。

### (f) 我建议的顺序

1. **(d)** 用稠密模型做一次对照——它能一次性告诉你"是我的问题还是 MoE 的问题"，成本一个下载 + 一次打包
2. 若指向我的流水线 → **(b)** 优化符号
3. 若指向 MoE → **(a)** 混合精度，找到可用配方
4. **(c)** 只在确认值得之后再做（需要 70 GB 空间）

---

## 7. 已知的未解问题

1. **PrismML 的符号是怎么来的** —— 随机？优化？包在哪一步？（他们的打包器未公开）
2. **block 512 vs 1024 的质量差从何而来** —— 是混合范围（旋转更充分）还是"多保留了 80 个高精度张量"？**这两个变量没分离**。
3. **MoE 三值是否有结构性障碍** —— 专家稀疏激活，量化误差不被平均；PrismML 只发布过稠密三值模型，**无先例**。
4. **`ssm_out` 折叠** —— 需 `gdn_v_grouped` + `perm_*`，且源 GGUF 里 `ssm_out` 可能已被 `_reorder_v_heads` 重排过。
5. **`token_embd` 折叠** —— 需列进 `inverse_weight_names`（非 tied 情况下合法）。
6. **12.14 GB 模型的 PPL** —— 显存/内存不足，未能测出。

---

## 8. 相关文档

- `IQ-LDS-实验记录与交接.md` —— GeForce 之外的 Vulkan IQ 优化（另一条线）
- `MoE单流瓶颈-完整调查记录.md` —— 单流 decode 带宽利用率调查
- `RGP-使用手册.md` —— AMD RGP 抓帧流程
- 外部：`Bonsai-demo/bonsai-2-27b-whitepaper.pdf`（方法学）、`docs.prismml.com/download/formats`（后端支持矩阵）

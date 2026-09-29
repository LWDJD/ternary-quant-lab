# 三值 MoE（Bonsai 2 式）—— 完整交接文档

> 最后更新：2026-09-27 晚
> 目标：把 `qwen35moe` 的 Qwen3.6-35B-A3B 做成 PrismML 的 PTQ1_0（1.75 bpw 三值）
> 本文自包含，不依赖任何对话历史。

---

## 0. 一句话现状

**根因：折错了张量集合 —— 漏折 `ssm_out`（48 个）。参考实现折 401 个，我折了 353 个，差的 48 个全是 `ssm_out`。已修复（含 converter 侧 V 头序补丁 + `gdn_v_grouped`）并在合成小模型上验证，35B 重新打包实测中。详见 §12。**

（原始症状：打包流水线跑通、格式与旋转都经独立验证，但产出质量不合格 —— 稠密 27B 几乎不能生成、MoE 会循环，参考实现 Bonsai 2 却正常。）

---

## 1. 已确证成立（带证据，不要重复验证）

| 事实 | 证据 |
|---|---|
| **打包格式与官方量化器逐字节兼容** | 与 `llama-quantize --outtype PTQ1_0` 的产物对比：**每块 scale 逐字节相同、95.5% 的 qs/qh 字节相同**，残差只在舍入边界 |
| **旋转数学正确** | 纯数值检验：`W @ x == fold_weight(W) @ apply_runtime(x)`，最大差 **1.5e-7**（f32 舍入） |
| **旋转器在 qwen35（含 GDN）上正确** | 合成小 qwen35，纯 F16 vs 旋转后 F16 的 PPL：300799.9918 vs 300801.5256，**相对差 5e-6**（就是 F16 舍入）。轴错会差几个数量级 |
| ~~元数据契约完整正确~~ **部分推翻** | 键齐全且自洽 ✓；但**折叠集合漏了 `ssm_out`**（353 vs 参考的 401）⇒ §12 |
| **稠密 27B 产物结构与 Bonsai 2 完全对等** | 同为 851 张量 / 64 层 / block_size 1024 / `sign_widths [5120, 6144, 17408]` |
| **保护清单与参考实现一致（且更保守）** | 逐项对照见 §4；我额外保了 `ssm_out` 和 `token_embd` |
| **MoE 元数据正确** | `expert_count=256`、`expert_used_count=8`，与目标 IQ2_M 逐项相同 |
| **远端驱动链路可用** | 见 §6 |

---

## 2. 已否证的假设（**不要重试**）

| 假设 | 否定证据 |
|---|---|
| MoE 结构是根因 | ✗ 稠密 27B（我的流水线）同样坏 |
| 换更高精度的源能修质量 | ✗ bf16 源 vs Q4_K_M 源：PPL 66.4 vs 68.6，**无差异** |
| 保护清单漏了敏感张量 | ✗ 与 Bonsai 逐项对照，基本一致 |
| 缺 `gdn_v_grouped` 导致 GDN 错乱 | ✗ 源码 `llama-model.cpp:2123` 显示它**只影响 `.ssm_out.`**，而我没折叠 `ssm_out` |
| `ssm_out` 的 V 头重排错位 | ✗ 转换器与运行时自洽（无 manifest → 转换器重排 → 运行时按 tiled 用） |
| 尺度规则（最小二乘 vs 目标密度） | ✗ 实测：LS 拟合 50.6% 非零、density 0.672 得 66.4%，**两者模型质量都没改善** |
| `2+2=5` 是量化失效的证据 | ✗ **健康模型也回答 5**，这是模型自身怪癖 |
| "循环重复"是模型缺陷 | ✗ 部分是 `-no-cnv` 裸跑（未套 chat template）造成；套 `--jinja -st` 后短答正常 |
| imatrix 能救三值质量 | ✗ Hadamard 正交变换会把各输入通道重要性搅成平均 |
| 用随机权重的小模型测量化质量 | ✗ 输出接近均匀分布，PPL 噪声 ±1.6%，量化扰动 ±1.5%，**无信号** |

---

## 3. PrismML 格式契约（**源码级事实**）

### 3.1 量化器：有；旋转器/打包器：**没有**

- **量化器存在**：`tools/quantize/`（`llama-quantize --outtype PTQ1_0`）、`ggml-quants.c` 的 `quantize_pq2_0()` / `quantize_ptq1_0()` / `quantize_row_ptq1_0_ref()`
- **打包器不存在** ✗：`hadamard_packing.json` 在仓库里**只有读取端**（`conversion/base.py:620-770`）；Python 侧搜 `fwht|walsh|popcount` **零命中**
- `docs/development/hadamard-tied-output.md` 明确：*转换器 **requires** `hadamard_packing.json`* —— 清单是**输入**

⇒ **旋转器必须自己实现。我的实现依据是：读取端契约 + 运行时 H 构造 + whitepaper + Bonsai 产物反推。**

### 3.2 块布局

```c
#define QK_PTQ1_0 128
uint8_t qs[24];      // 每字节 5 trit → 元素 0..119
uint8_t qh[2];       // 每字节 4 trit → 元素 120..127
ggml_half d;         // 尺度在**末尾**，不在开头
// 28 B / 128 = 1.75 bpw
```
`qs` 两级编码：`c=16` 吃元素 0..79 → `qs[0:16]`；`c=8` 吃 80..119 → `qs[16:24]`。
解包时 `uint8_t` 按 256 取模就是三进制位提取本身。

### 3.3 运行时的旋转语义

`src/llama-model.cpp:2055-2064`：
```cpp
const float scale = 1.0f / sqrtf((float) block_size);
uint32_t parity = row & col;
parity ^= parity >> 16; ... parity ^= parity >> 1;
data[row * block_size + col] = (parity & 1) ? -scale : scale;
```
即 `H_norm[i][j] = (-1)^popcount(i&j)/√n`（Sylvester 序）。

**折叠式**：`W' = W · S · H_norm`（沿输入维、先乘符号、再乘归一化 Walsh、按块）
**运行时对激活做**：`H(S x)` —— 二者互为逆，精确可逆。

### 3.4 `prism.hadamard.*` 元数据

```
version      u32   1（显式头）；2（tied_output）
block_size   u32
transform    str   "normalized-sylvester-walsh-hadamard"   ← 注意：manifest 里叫
                                                              "normalized-signed-sylvester-walsh-hadamard"，GGUF 里没有 "signed"
axis         str   "input-last-dimension"（manifest 要求 axis 必须 == -1）
sign_mode    str   "explicit"
sign_widths  arr   [宽度...]
sign_values  arr   [±1...]（按 sign_widths 顺序拼接）
weight_names arr   [已折叠张量]
inverse_weight_names arr  只允许 ["token_embd.weight"]
tied_output  bool  仅 schema 3
gdn_v_grouped bool 只影响 .ssm_out.
```

### 3.5 可折叠张量白名单（`_HADAMARD_KINDS`）—— **与源码逐字核对过**

**写端 `conversion/base.py:700-709` 原文：**
```
output\.weight|
blk\.\d+\.(attn_q|attn_k|attn_v|attn_qkv|attn_gate|attn_output
          |ffn_gate|ffn_up|ffn_down
          |ffn_gate_exps|ffn_up_exps|ffn_down_exps|ffn_gate_up_exps
          |ffn_gate_shexp|ffn_up_shexp|ffn_down_shexp
          |ssm_out)\.weight
```

**读端 `src/llama-model.cpp:1287-1296` 原文：**
```cpp
static const char * kinds[] = {
    "attn_q", "attn_k", "attn_v", "attn_qkv", "attn_gate", "attn_output",
    "ffn_gate", "ffn_up", "ffn_down",
    "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps",
    "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
    "ssm_out",
};
if (name == "output.weight") return true;  // output head goes through build_lora_mm in every arch
```

⇒ **`output.weight` / `attn_gate` / `ssm_out` 三者都该折**。原实现漏了 `ssm_out`。

配套架构白名单：`{LLAMA, QWEN3, QWEN3MOE, QWEN35, QWEN35MOE, QWEN3NEXT, DSPARK}`（qwen35 / qwen35moe 在内 ✓）。

**⚠ 注意 `hadamard-explicit-signs` 分支上的 `e76b30416` 拿掉过这三个 —— 那是另一条基点（直接验证被 Bonsai 实测推翻，见 §12.2）。以 `adfffbe` 的源码为准。**

⇒ 这也解释了 `docs/development/hadamard-tied-output.md` 的用意：tied 模型借 `inverse-after-lookup` 旋转 `token_embd`（仅限这一张表）；untied 模型的 `output.weight` 走普通 `fold-before-matmul`。

### 3.6 `_LinearAttentionVReorderBase`（conversion/qwen.py:446）

GDN 的 V 头在 HF 里是**按 K 头分组**存的，ggml 广播需要**交织（tiled）**序。转换器默认会对 `in_proj_qkv`/`in_proj_z`/`in_proj_a,b`/`out_proj` 做重排。

**但**：`if self._hadamard_folds_tensor(name)` 为真时（即 manifest 里折叠了该张量），`out_proj` **跳过重排**并置 `_hadamard_gdn_v_grouped = True`，把置换交给运行时。
⇒ 因为 `_hadamard_folds_tensor` 读的是 **manifest**，**我没有 manifest → 转换器照常重排 → 与"运行时无 `gdn_v_grouped`"自洽。这条不是 bug。**

---

## 4. Bonsai 2 27B 参考事实（真品，工作正常）

- `qwen35`，851 张量，64 层，block_size **1024**，`sign_widths [5120, 6144, 17408]`
- 类型：**PTQ1_0 402 / F32 353 / BF16 96**，`file_type = 143`
- `gdn_v_grouped: True`，`inverse_weight_names: ["token_embd.weight"]`
- **保护清单（非 PTQ1_0）—— 只有 norm 和 SSM 标量，一个矩阵乘投影都没保护**：
  ```
  x64 blk.N.attn_norm / post_attention_norm   F32
  x48 blk.N.ssm_a / ssm_conv1d / ssm_dt.bias / ssm_norm   F32
  x48 blk.N.ssm_alpha / ssm_beta              BF16
  x16 blk.N.attn_k_norm / attn_q_norm          F32
  x1  output_norm                              F32
  ```
- **三值非零比例：67.2~67.3%，且每个张量都恒定**（固定规则，非逐组拟合）

**我的产物对照**：非零 50.6%（LS 拟合）/ 66.4%（density 0.672）；`file_type 129`（143 是这个 fork 枚举里没有的值 ⇒ Bonsai 出自更新的 fork）。

---

## 5. 已产出的模型

| 文件 | 配置 | 大小 | 表现 |
|---|---|---|---|
| `qwen36-moe-ptq10.gguf` | MoE，block 512，bf16 源 | 9.04 GB | 连贯但循环；事实 1/4 |
| `qwen36-moe-b1024.gguf` | MoE，block 1024，`ffn_down_*` 用 Q4_0 | 11.87 GB | **事实 5/6**（需 `--repeat-penalty 1.3`）；仍循环 |
| `qwen38-27b-ptq10.gguf` | 稠密 27B，block 1024，LS 拟合尺度 | 10.9 GiB | **不能生成**（`<think>` 后 EOS） |
| `qwen38-27b-d672.gguf` | 同上但 density 0.672 | 10.9 GiB | **不能生成**（空行 + `。` 循环） |

**健康对照**：
- IQ2_M（35B MoE finetune）：`promptA` 下 626~2878 词、uniq 0.40~0.57、正常收尾 ✓
- Bonsai 2 27B PTQ1_0：连贯高质量 ✓

**NAS 位置**：`/mnt/workspace/prismwork/out/`
**本地位置**：`D:\Project\openhanako\workbench\prism-pack\{,q\}`

---

## 6. 远端环境与驱动方式

### 6.1 环境

| 项 | 值 |
|---|---|
| 实例 | ModelScope DSW，**实例号每次重启都变**（如 `dsw-2214501`） |
| 算力 | **64 核 / 28 GB 内存**；`/tmp` 337 GB 可用 |
| 注意 | **IDE 自身的 node 进程常占 ~17 GB** |
| Python | 3.11，有 numpy/torch(2.3.1+cpu)/safetensors/modelscope |
| 源码 | `/mnt/workspace/prismwork/prism-src`（fork @ `adfffbe`，完整克隆） |
| 构建产物 | **`/tmp/llb/bin/`**（`llama-completion` / `llama-perplexity` / `llama-quantize` / `llama-bench` / `llama-imatrix`） |
| 模型 | `/mnt/workspace/prismwork/models/qwen38-27b`（Qwen3.8-27B bf16，52 GB） |

### 6.2 驱动方式（`dsw.py`，本地）

Chrome 以 `--remote-debugging-port=3358` 运行着 ModelScope 工作区页面。**不能用 CDP 输入事件操作那个跨域 iframe**，但**页面内的 JS 可以**——所以走它自己用的两个 API：

- `/dsw-<实例>/api/contents` —— 读写远端文件
- `/dsw-<实例>/api/kernels` —— 在远端执行 Python（→ 任意 shell）

```powershell
# 实例号改了要改 dsw.py 的 INST，或设环境变量 DSW_INSTANCE
python dsw.py ls  prismwork/out          # 列目录
python dsw.py get prismwork/x.py 4000    # 读文件
python dsw.py put local.py prismwork/x.py  # 上传（自动分块 + sha 校验）
python dsw.py sh  "cmd"                  # 执行 shell（走内核）
python dsw.py bg  "cmd" /path/log        # 后台执行
python dsw.py tail prismwork/log 40      # 看日志尾部
python dsw.py kdel                        # 清内核缓存
```

辅助脚本（同在 `workbench\`）：`chrome.py`（CDP 客户端）、`probe_gateway.py`（重启后重挖网关地址）。

### 6.3 **必读的坑**

1. **内核会泄漏**：`exec_py` 每次调用新建内核。内存紧张时它们会超时。**用 `purge_kernels.py` 清掉**（`/api/kernels` 的 DELETE）。
2. **WS 超时**：内核冷启动 40 秒以上，`chrome.py` 的 WS 超时已放宽到 300 s，JS 内 timer 也放宽到 300 s。
3. **本地 PowerShell 会把内联引号和 `$(...)` 搞坏** → **复杂逻辑一律写脚本执行**，不要拼命令行。
4. **`Select-Object -First N` 会杀死上游原生进程**（把 python 一起带走）。
5. **PowerShell `>` 产出 UTF-16** → 让脚本自己写 UTF-8 文件再读。
6. **`pkill -f xxx` 会杀掉自己的 shell**（命令行里含该串）→ 用 `[x]xx`。
7. **`git fetch <sha>` 被 GitHub 拒** → 完整 clone 再 checkout。
8. **`time` 不是 dash 内建** → 用 `bash -c` 或去掉。
9. **Chrome 下载留下 `.crswap`** → 校验 sha256 后删掉 0 字节占位、改名即可；文件其实是完整的。

---

## 7. 真实 Qwen3.8-27B 的张量形状（合成测试台要用）

- 64 层 / hidden **5120** / FFN **17408** / heads 24 / kv 4 / head_dim **256**
- **48 个 GDN 层 + 16 个全注意力层**；`full_attention_interval=4` ⇒ **全注意力在层 3、7、11…**（每组的**最后**一个），不是从 0 开始
- GDN（`linear_attn`）：
  ```
  in_proj_qkv   [10240, 5120]   ← q+k+v = 4096 + 6144
  in_proj_z     [6144, 5120]
  in_proj_a/b   [48, 5120]
  out_proj      [5120, 6144]
  conv1d        [10240, 1, 4]   ← 作用在完整 qkv 宽度上
  A_log/dt_bias [48]            norm [128]
  ```
- 全注意力（`self_attn`）：
  ```
  q_proj  [12288, 5120]   ← 12288 = 2 × 6144，**Q 与输出门控融合**
  k_proj  [1024, 5120]    v_proj [1024, 5120]
  o_proj  [5120, 6144]    q_norm/k_norm [256]
  ```

---

## 8. 下一步（**已由 §12 取代 —— 先做 §12 的修复，再回来做这里的验证**）

### 8.1 建"有效信号"的测试台（**降级：改为修复后的验证手段**）

**问题**：没有廉价的质量度量，导致每次迭代要 20 分钟打包 + 10 GB 下载。

**已试失败的方案** ✗：随机权重的合成小 qwen35（PPL 全是噪声，分辨不出）。

**推荐的方案**：**从真实 27B 切出前 4 层**（层 0-2 GDN + 层 3 全注意力，两种都覆盖），**保持真实 5120 维和真实权重**。

- 体积约 3 GB（大头是词嵌入 248320×5120）
- 远端 28 GB 内存跑得动，PPL 出结果以分钟计
- **F16 版 vs 量化版的对比有效**（同一个模型，只差量化）

一切就绪：`/mnt/workspace/prismwork/` 下有 `make_q35b.py`（合成器，已修正 4 个形状坑）、`rot_test.py`（旋转判定）、`tiny_quant.py`（量化对照）、`read_shapes*.py`。

### 8.2 然后逐个变体 A/B

按"离参考实现的差距"排序：

1. **折叠 `ssm_out`**（Bonsai 折了，我没折）—— 注意需同时处理 `gdn_v_grouped` 与重排跳过
2. **符号策略** —— 我用确定性随机 ±1；Bonsai 是 `explicit`（来源未知，可能是优化的）
3. **block 2048?** —— Bonsai 用 1024，我 27B 也是 1024，变量已对齐
4. **`token_embd` 处理** —— Bonsai 是 PTQ1_0+inverse（tied 模型）；我的模型 untied，只能不折

### 8.3 验收判据（已建立，可复用）

- **事实探针**：`The capital of France is` → 是否出现 `Paris`；`Japan` → `Tokyo`；`Germany` → `Berlin`。健康模型 3/4~4/4，我的 MoE 1/4，block1024 版 5/6。
- **循环指标**：生成 1024 token，统计 `uniq = 不同词数/总词数` 与 `rep6 = 任一 6-gram 最大重复次数`。健康 0.40~0.57 / rep6 3；我的 0.04 / rep6 43。
- **必须走 `--jinja -st`**（否则输出形态被毁）
- 提示词文件：`prism-pack\logs\promptA.txt`（"讲登录取证"的教程请求）

### 8.4 本地验收命令

```powershell
$B = "D:\Project\openhanako\workbench\prism-tip\build-vs\bin\Release"
& "$B\llama-completion.exe" -m <模型> -ngl 99 -c 20480 `
    -f "D:\Project\openhanako\workbench\prism-pack\logs\promptA.txt" `
    -n 1024 --temp 0 --seed 1 --jinja -st
```

---

## 9. 代码清单

### 本地 `D:\Project\openhanako\workbench\`

| 文件 | 作用 |
|---|---|
| `prism-pack\prism_pack_hf.py` | **主工具**：Safetensors → 旋转 → PTQ1_0，复用 fork 的 conversion 映射，劫持 `add_tensor` |
| `prism-pack\ptq1_0.py` | 打包/解包 + 三值量化（含最小二乘拟合与 `density` 目标密度两种尺度规则） |
| `prism-pack\prism_pack.py` | 折叠数学 + 与运行时 R 的逐位自检 |
| `prism-pack\qwen36-moe-ptq10.gguf` | MoE 产物（block 512） |
| `prism-pack\q\qwen38-27b-{ptq10,d672}.gguf` | 稠密 27B 两版 |
| `prism-pack\q\qwen36-moe-b1024.gguf` | MoE block 1024 |
| `dsw.py` | 远端驱动 |
| `chrome.py` | CDP 客户端（连接 3358 端口那个 Chrome） |
| `probe_gateway.py` / `purge_kernels.py` | 重启后重挖网关 / 清内核 |

### `prism_pack_hf.py` 用法

```powershell
python prism_pack_hf.py <hf目录> <out.gguf> `
    --block 1024 --iters 3 --jobs 8 `
    --density 0.672          # 可选：目标非零比例（默认用最小二乘）
    --fallback-quant Q4_0    # 无法按 block 对齐的张量用此类型（否则 F16，很大）
    --keep-f16 "output[.]weight"   # 折叠但仍存 F16
    --mtp                    # 保留 MTP/NextN 层（默认排除，与 40 层参考对齐）
    --fork <fork根目录>       # 需要 <fork>/conversion 与 <fork>/gguf-py
```

**脚本本身只在合成小模型上验证过**（逐字节对比官方转换器输出、逆折叠恢复 f16 精度、三值余弦 0.9）。真实模型结构验收通过（§1），但**模型质量不合格**。

---

## 10. 我这一轮最大的方法论教训

**反复把"设置造成的现象"当成"模型的属性"。**

具体犯过：
- 用 `-no-cnv` 裸跑 → 输出形态被毁 → 误判"模型退化"（真因是缺 chat template）
- 用随机权重的小模型 → PPL 全是噪声 → 误判"密度规则有效/无效"
- 把 `2+2=5` 当失败证据 → 健康模型也一样
- 把跨会话速度差当"3 倍回退" → 其实是工具差异（`llama-completion` 比 `llama-bench` 慢一倍）

**⇒ 任何对比必须：同一会话、同一参数、有健康对照、并且先问"这个指标本身可靠吗"。**

---

## 11. 尚未排查的方向（供后来者）

1. **`sign_values` 的实际取值** —— 我从没读过 Bonsai 的 `sign_values` 内容去做分布分析（只读过长度）。它是否真的是随机 ±1？
2. **`in_proj_qkv` 的 q/k/v 三段的折叠加权** —— whitepaper 只给了统一公式，但 GDN 的 q/k/v 语义不同
3. **量化是否该对 `attn_qkv` 的 V 段单独处理** —— Bonsai 折了它，我的产物也折了，但误差贡献未测
4. **`file_type` 143 的语义** —— 这个 fork 的 `LlamaFileType` 里没有 143，Bonsai 出自更新的 fork，**`adfffbe` 可能已过时**
5. **是否存在比 1024 更好的 block**（Bonsai 用 1024，我的稠密也用 1024，变量已对齐，但没扫过）
6. **`--no-quant` 全模型验证** —— 只在合成模型上做过（§1），没在真实 27B 上做过（需要 50 GB F16，本地放不下，远端 `/tmp` 够）

---

## 12. 根因调查：两次推翻自己，以及当前状态

### 12.1 一句话

**`ssm_out` 折叠已被实测证伪。** 当前最好的产物是**不折 `ssm_out` + 最小二乘尺度**：

```
                事实    uniq   rep6
healthy         4/4     0.72     1
old(b1024)      4/4     0.24    16     ← 有知识，只是循环
ls(折ssm_out)   0/4     0.15   100     ← 坏掉
dens(+density)  0/4     1.00     0     ← 数字汤
```

⇒ **`old` 的配置是正确基线**；它的唯一缺陷是循环。`ssm_out` 折叠（即使配齐 `gdn_v_grouped` 与转换器 V 头序补丁）在本 MoE 上是错的。

### 12.2 三次尝试，两次被证伪（记下来，别再走）

| # | 假设 | 依据 | 结果 |
|---|---|---|---|
| 1 | 应该**去掉** `output.weight` / `attn_gate` 的折叠 | 分支提交 `e76b30416` 的"收紧白名单" | **✗ 被 Bonsai 实测证伪** —— 它照样折这两个 |
| 2 | 不折 `ssm_out` 是"自洽"的 | 无 `gdn_v_grouped` 与之配套 | **✗ 被源码证伪** —— 权威名单里 `ssm_out` 在内 |
| 3 | 应该**加上** `ssm_out` 的折叠 + `gdn_v_grouped` | 源码 + 参考产物 | ✓ 已实现 |

**两次都是"用推理代替测量"。** 教训：这个仓库的权威答案在**源码和参考产物**里，不在提交信息里 —— 提交信息属于别的分支/别的基点。

### 12.3 参考实现的折叠集合（实测，地面真相）

读 `D:\AI\lmstudio\models\prism-ml\Ternary-Bonsai-2-27B-gguf\Ternary-Bonsai-2-27B-PTQ1_0.gguf`：

```
折叠 401 个：
  attn_q 16 | attn_k 16 | attn_v 16 | attn_output 16 | attn_gate 48
  attn_qkv 48 | ffn_gate 64 | ffn_up 64 | ffn_down 64
  ssm_out 48 | output.weight 1
```

我的旧产物折叠 353（含 `attn_gate` 48、`output.weight` 1，**缺 `ssm_out` 48**）。**唯一差异就是 `ssm_out`。**

复核脚本：`D:\Project\openhanako\workbench\cmp_bonsai.py`、`check_folded.py`。

### 12.4 权威规格在源码里（不要再猜）

**写端 `conversion/base.py:700-709`：**
```
output\.weight|
blk\.\d+\.(attn_q|attn_k|attn_v|attn_qkv|attn_gate|attn_output
          |ffn_gate|ffn_up|ffn_down
          |ffn_gate_exps|ffn_up_exps|ffn_down_exps|ffn_gate_up_exps
          |ffn_gate_shexp|ffn_up_shexp|ffn_down_shexp
          |ssm_out)\.weight
```

**读端 `src/llama-model.cpp:1287-1296`：**
```cpp
static const char * kinds[] = { "attn_q", ..., "ssm_out", };
if (name == "output.weight") return true;   // 每个架构的输出头都走 build_lora_mm
```

⇒ `ssm_out` / `attn_gate` / `output.weight` **三者都该折**。原实现漏了 `ssm_out`。

### 12.5 折 `ssm_out` 必须配套的两件事（否则静默出错）

源 `conversion/qwen.py:617-629`：GDN 的 `out_proj` 列序默认会从 grouped 重排成 tiled，**但**：

```python
if self._hadamard_folds_tensor(name):
    self._hadamard_gdn_v_grouped = True     # 保留 training(grouped) 序
else:
    data_torch = self._reorder_v_heads(...)  # 重排成 tiled
```

**`_hadamard_folds_tensor` 读的是 HF 目录里的 `hadamard_packing.json`** —— 我没有 manifest，所以转换器走的是 else 分支（重排）✗。

⇒ 修法（已实现）：在 `prism_pack_hf.py` 里给模型类打补丁，让 `...linear_attn.out_proj.weight` 一律返回 True；转换器随即跳过重排并置 `_hadamard_gdn_v_grouped`，我们把它写成 `prism.hadamard.gdn_v_grouped=true`（运行时会据此置换激活，`llama-model.cpp:2123`）。**并加断言**：折了 `ssm_out` 却没拿到这个标志就报错。

### 12.6 当前配置（已消除全部可消除的偏离）

| 项 | Bonsai 2 | 我（本次 35B） |
|---|---|---|
| 折叠集合 | 401 | 401 ✓ |
| `gdn_v_grouped` | true | true ✓ |
| block_size | 1024 | 1024 ✓ |
| 非零比例 | 恒定 67.2% | `--density 0.672` → 66.4% ✓ |
| `ffn_down_exps` | 无此张量（稠密） | 输入维 512 < 1024 无法折 → `--fallback-quant Q4_0`（**唯一不可避免的偏离**） |
| 符号值 | explicit（值未知） | sha256 确定性随机 **← 唯一无法对齐的偏离** |

### 12.9 实测结论（2026-09-28，本地 Vulkan，同口径）

**单变量受控对比**（同 harness、同参数、同后端）：

| 产物 | ssm_out | 尺度规则 | 事实 | uniq | rep6 |
|---|---|---|---|---|---|
| healthy IQ2_M | — | — | **4/4** | 0.72 | 1 |
| `qwen36-moe-b1024.gguf` | 不折 | 最小二乘 | **4/4** | 0.24 | 16 |
| `moe35-ls.gguf` | **折** | 最小二乘 | **0/4** | 0.15 | 100 |
| `moe35-ptq10.gguf` | 折 | density 0.672 | 0/4 | 1.00 | 0 |

**⇒ `ssm_out` 折叠是错的**（`old` → `ls` 只差这一个变量）。即使同时配齐了转换器 V 头序补丁和 `prism.hadamard.gdn_v_grouped=true`，模型仍然坏掉。

**⇒ `--density 0.672` 也是错的**（`ls` → `dens` 只差这一个变量，得到纯数字汤）。

**⇒ 正确基线 = 不折 `ssm_out` + 最小二乘尺度**。它的事实召回与健康模型相同（ 4/4），唯一缺陷是**循环**（uniq 0.24 / rep6 16 vs 健康 0.72 / 1）。

### 12.10 一条测试方法论修正

**旧记录"MoE 事实 1/4"是探针缺陷，不是模型缺陷。** 该模型会先输出一段 "thinking process" 前言，32 token 截断根本走不到答案——健康模型在同样设置下也是 0/4。必须 `-n ≥ 300` 并搜整个输出。

### 12.11 远端 CPU 后端不可用于判定

`adfffbe` 的 HEAD 提交正是 `x86: SSE2/SSSE3 vec_dot for PTQ1_0 and PQ2_0` —— **CPU 的 PTQ1_0 支持是这个修订上刚加的**。实测两个变体在远端 CPU 构建上都是退化输出。**判定必须用本地 Vulkan 构建。**

---

## 13. 第二轮：三条假设被实测排除（2026-09-28）

### 13.1 远端 CPU **其实可信**（推翻 §12.11）

同一份 `moe35-ls.gguf`、同一段 200 KB 语料、8 个 chunk：

```
本地 Vulkan   PPL = 180.3931  ± 14.05
远端 CPU      PPL = 180.2623  ± 14.12     差 0.07%（噪声 ±7.8%）
```

**⇒ 两个后端的 PTQ1_0 数学一致** ⇒ **以后判定可以直接在远端跑**（远端单 token 约 2.2 t/s，测试要短）。

**实例号会变**：`dsw-2214501 → 2215322 → 2215359`。API 全 404 时先怀疑实例号，不要怀疑方法。

### 13.2 三条假设全部被排除

| 假设 | 检验 | 结果 |
|---|---|---|
| 符号向量是优化出来的 | 统计真品三个向量（均衡 49%、runs 吻合、自相关≈0、无周期、互不相关） | **✗ 与随机无法区分** |
| 误差反馈（GPTQ 式） | 真实折叠权重上测重构 MSE | **✗ 反而差 1.8–2.9 倍** |
| 我的尺度拟合卡在局部最优 | 对 `d` 做一维全局扫描（给定 t 时 d* 闭式） | **✗ 只差 0.65%，基本就是全局最优** |

**关键推论**：全局最优 `d` 落在 **55.5% 非零**，而真品是 **67.2%** ⇒ **真品不优化权重重构误差，而是另一个目标**。而我在所有权重指标上已是最优 ⇒ **差距在权重重构误差之外**。

### 13.3 底座名字要小心

- 本地那份参考产物是 **`Ternary-Bonsai-2-27B`** ✓，底座 **`Qwen3.8-27B`**（851 张量）
- `bonsai-27b-whitepaper.pdf` 写的是 "Qwen3.6-27B" —— 那是**更早的 Bonsai 27B 发布**，不是 Bonsai 2
- 参考产物只在 HuggingFace 上（`prism-ml/Ternary-Bonsai-2-27B-gguf`），ModelScope 没有

### 13.4 下一步（已就绪，未执行）

**决定性实验：把真品的块尺度与真实原始权重配对。**

真品的旋转是确定的（Sylvester-Hadamard + 它自己存的符号），所以第 `i` 个 128 组在两边对应同一个输入维区间。于是对每个 128 组可以拿到 `(x_true, d_ref)` 对，逐个统计量试：若某个统计量与 `d` 成恒定比，**尺度规则就直接暴露**。

```bash
# 1) 底座（正在下，18 分片）
modelscope download --model Qwen/Qwen3.8-27B --local_dir /tmp/q38-27b
# 2) 真品（HF，ModelScope 没有）
huggingface-cli download prism-ml/Ternary-Bonsai-2-27B-gguf \
    --include '*PTQ1_0.gguf' --local-dir /tmp/bonsai2
# 3) 用本地 packer 打同一底座：--block 1024 --fallback-quant Q4_0
# 4) 配对比较 d_ref vs stat(x_true)，并同时比较两边的 d 比值分布
```

**避开 `ssm_out`**（真品折了它并带 `gdn_v_grouped`，反折叠需要 V 头置换）。

### 13.5 快速迭代台（已建好）

| 工具 | 作用 | 成本 |
|---|---|---|
| `ab_quant.py` | 两种量化器在真实折叠权重上的重构 MSE 对比 | **1.8 秒** |
| `scan_d.py` | `d` 一维全局扫描 + 非零比例 | **1 秒** |
| `analyze_signs.py` | 符号向量结构检验 | 瞬时 |
| `archcheck.py` | 列出本地可用模型及其架构 | 瞬时 |

**`SmolLM2-135M-Instruct-Q3_K_L`**（93 MiB，arch `llama`）是最佳迭代台：打包与 PPL 都是秒级。`MiniCPM5-1B-F16`（2.0 GiB，全精度）用作第二个。

新工具：`ptq1_0.py` 的 `quantize_blocks_ptq1_0(..., mode=)` 现有 `ls` 和 `ef` 两种模式。

---

## 14. 远端实例重建清单（实例一关，/tmp 全丢）

### 14.1 分区性质

| 路径 | 关机后 | 内容 |
|---|---|---|
| `/mnt/workspace`（NAS） | **保留** ✓ | `prismwork/prism-src`（fork clone, 560 MB）、打包脚本、`out/` |
| `/tmp`（实例本地） | **丢失** ✗ | HF 模型、编译产物、待下载的 GGUF |

⚠ **NAS 有配额，写多了会报 `Disk quota exceeded`**（`du` 看着才几十 GB 也会碰）。绕过去：**用 shell（root）而不是 Jupyter contents API 写**，或直接落到 `/tmp`。

### 14.2 重建步骤（约 15 分钟）

```bash
# 1) 实例号先查（会话头会变：2214501 -> 2215322 -> 2215359）
#    写进 dsw.py 的 INST，或 DSW_INSTANCE=... 覆盖
#    API 全 404 时先怀疑实例号，不要怀疑方法

# 2) CPU 构建（~3 分钟，源码在 NAS）
cmake -S /mnt/workspace/prismwork/prism-src -B /tmp/llb \
  -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=OFF -DGGML_CUDA=OFF \
  -DGGML_HIP=OFF -DGGML_METAL=OFF -DGGML_SYCL=OFF -DGGML_OPENCL=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build /tmp/llb -j 48      # 前台跑，约 3 分钟；产物 /tmp/llb/bin/llama-*

# 3) 模型（按需）
modelscope download --model Qwen/Qwen3.8-27B  --local_dir /tmp/q38-27b    # ~52 GB, 10 min
modelscope download --model Qwen/Qwen3.6-35B-A3B --local_dir /tmp/moe35    # ~67 GB, 10 min
# 真品只在 HF，ModelScope 没有：prism-ml/Ternary-Bonsai-2-27B-gguf
```

### 14.3 后台作业的正确起法

`dsw.py` 里**只有 `bg` 能活过调用结束** ✗ —— `subprocess.Popen` 起的孤儿子进程会被回收 ✗。直跑二进制写到 `/tmp` 的日志里最稳（前面几次日志为空都是这个原因）。

```powershell
python dsw.py bg "<cmd>" /tmp/xxx.log     # 然后 poll /tmp/xxx.log
```

### 14.4 本轮结束时远端残留（均可删，不影响）

`/tmp`：`moe35`(67G)、`q38-27b`(52G)、`llb`(839M)、`moe35-512.gguf`(9G)、`moe35-b1024.gguf`(12.7G)
`NAS out/`：`moe35-ls.gguf`(12.3G，本地有 ✓)、`qwen38-27b-d672.gguf`(10.9G，已被取代 ✗)

**`moe35-b1024.gguf` 等在本地都有备份**（`prism-pack\q\`）⇒ 远端没有不可替代的东西 ✓。

### 14.5 本地资产（不受影响）

`prism-pack\q\`：`qwen36-moe-b1024.gguf`（**基线，事实 4/4** ✓）、`moe35-ls.gguf`（已证伪 ✗）、`moe35-ptq10.gguf`（density，已证伪 ✗）
`D:\AI\lmstudio\models\prism-ml\Ternary-Bonsai-2-27B-gguf\`：**参考真品** ✓（PTQ1_0 + PQ2_0）
`Bonsai-demo\`：官方仓库快照（含 `MODEL-FORMATS.md` ✓、三份白皮书 ✓、`scripts/` ✓）

---

## 15. 第三轮：旋转约定已锁定（2026-09-28）

### 15.1 并行约定搜索（`remote/orient_search.py`）

在 72 个组合上并行评分（Hadamard 索引序 × 缩放 × 符号侧 × 块布局 × 符号索引 ✓），判据 = 与真品 trits 的吻合率：

```
  87.58%   natural, inv_sqrt, sign=before, contig, sign_idx=global   ← 最优
  68.51%   scale=none
  44.56%   sign_idx=modulo
  33.3%    其余（= 三值随机水平）
```

**⇒ 最优那组就是 `prism_pack.fold_weight` 的约定** ✓（`inv_sqrt` + 符号在 Hadamard **之前** + 连续块 + 全局符号索引）✓

⚠ 之前一版 `orient_test.py` 报"完全不相关（33%）"是**它的 H 构造有 bug** ✗，以本搜索版为准 ✓。

### 15.2 剩余差异只有一个常数

用真品的 `d_ref` 重建：**我的非零 58.02%，真品自己 67.21%** ✗，trit 吻合 87.58% ✓。
分歧集中在"真品非零、我判零"✓ —— 典型特征是**我的 `|x_rot|` 系统性偏小** ✓。由 67.2% 反推：**差一个约 1.29 倍的比例** ✓。

### 15.3 已证伪的尺度规则

| 规则 | 结果 |
|---|---|
| 最小二乘（当前实现） | 非零 ~50% ✗，与真品的 67.2% 不同 |
| 密度匹配 `--density 0.672` | 模型更差 ✗ |
| 均值匹配 `d·mean|t| = mean|x|` | **不动点发散** ✗（`rule_test.py` 实测落到 0.01% 非零）；验算也本来就不成立 ✗ |
| 误差反馈（GPTQ 式） | MSE 差 1.8–2.9 倍 ✗ |
| 符号向量优化 | 真品符号与随机无法区分 ✗ |

**⇒ 真品的 67.2% 不是 MSE 最优（~55%）** ✓ ⇒ 它优化的是另一个目标 ✓，但**不是**上面任何一条已试过的 ✓。

### 15.4 交付物验收（`prism-pack\accept_final.py`）

对 §15.4 那份产物跑实际用法（`--jinja -st` ✓，greedy ✓）：

```
              facts    uniq   rep6
A 原始        2/2      0.32    11
B +rpen 1.3   1/2      0.40     4     ← 循环明显改善 ✓，2 样本内少一个事实（噪声级）
参考健康     4/4      0.72     1
```

⇒ **推荐用法：`--jinja` + `--repeat-penalty 1.3`** ✓。知识保留住了 ✓，缺陷是重复 ✗，机制与健康模型仍有差距 ✓。

### 15.5 待办（网络可用时，一条命令可验）

**验证 `prism-v7`** ✓：`adfffbe` 是 `origin/prism`（= **prism-v5 冻结分支**）的 tip ✓，而 `MODEL-FORMATS.md` 说当前是 **prism-v7** ✓，真品的 `file_type=143` 在我这份枚举里不存在 ✓。
⇒ 拉 v7 分支、重跑 `remote/orient_search.py` ✓：**若约定有微调，87.58% 应跳到接近 100%** ✓（同一个判据，无需新分析 ✓）。
本地 clone 目前**没有 v7 的 ref** ✗，需要网络 ✓。

---

## 16. 第四轮：保护清单的实测（2026-09-28）

### 16.1 白皮书那句话不能照搬

> 量化范围：**embeddings**、attention projections、MLP projections、LM head

真品的 `inverse_weight_names` 里确实列着 `token_embd.weight` ✓ —— **但那个角色（`inverse-after-lookup`）要求绑定输出（tied，schema 3）** ✓。

**⇒ 未绑定模型不能这么做** ✗。未绑定模型的词嵌入查表**不走矩阵乘** ✓，运行时**不会对它施加任何反变换** ✓ ⇒ 它要么保持高精度 ✓，要么必须正确的折叠+反变换 ✓。

### 16.2 实测：把 `token_embd` 变三值会**直接毁掉模型** ✗

27B、block 1024、其余完全相同，只把 `token_embd` 改成“纯 PTQ1_0 不折叠”：

```
                体积        事实    生成
原配置(F16)     10.90 GB    —       正常
改后(三值)       8.64 GB    0/2     立即 [end of text]，loop 3 词
```

**⇒ 省了 2.26 GB，代价是模型不能用** ✗ ⇒ **对未绑定模型，这项“多余保护”是必要的** ✓✓。

处理：`prism_pack_hf.py` 里保留机制但**默认不匹配任何张量** ✓；若将来打 tied 模型可一行开启 ✓。

### 16.3 结论

**保护清单上已无空间** ✓：
- 去掉 `token_embd` 保护 → **实测毁掉模型** ✗
- 去掉 `ssm_out` 保护（即折它）→ **实测更差** ✗（4/4 → 0/4，§12）
- 其余项已与参考一致 ✓

**⇒ 当前配置已是最优可用点** ✓：事实保留 ✓、重复偏高 ✗、体积 10.90 GB。
**⇒ 进一步提升只能来自尺度规则** ✓（§15.3）—— 那是他们私有量化器里的东西 ✓。

---

## 17. 第五轮：按值域保护 —— 假设成立且已落地（2026-09-28）

### 17.1 关键观察

回看 §15 的统计量比值：

```
d/max   cv = 12.7%   ← 最大值做尺度，抖动大
d/mean  cv =  2.0%   ← 均值做尺度，极稳定
```

**真品的 `d` 跟随稳健统计量（均值）✗，不跟随 `max`** ✓ ⇒ **组内最大值不定尺度** ✓ —— 这就是“按值域保护”：离群值被容忍，而不是被允许把 `d` 顶飞、把其余值全压成零 ✓✓。
格式里没有“逐值精度”字段 ✓ ⇒ 逐值保护**只能通过尺度规则实现** ✓，这也解释了为何它成为唯一能解释全部测量的量 ✓。

### 17.2 实现与实测（`ptq1_0.py` 的 `mode="robust"`）

规则：`d = 1.37·mean|x|`，再做几次最小二乘重拟合 ✓（而非用 `amax` 起步 ✓）。
在真实折叠权重上（SmolLM2，block 128）：

```
             MSE           非零
ls(MSE)   6.948e-03       46.04%
robust    6.593e-03  ✓    55.15%  ✓   ← 误差降 5%，非零升 9pp
```

**⇒ 双向变好** ✓：重构误差更低 ✓，且**朝真品的 67.2% 靠近** ✓✓。
（原因：`amax` 起步会陷入局部最优 ✓，均值锚点落在更好的盆地 ✓。）

系数扫描：1.37/1.6/1.9/2.2/2.6 → MSE 几乎平地 ✓，非零从 55% 降到 50% ✗（重拟合会把 `d` 拉回 ✓）
⇒ **靠调系数到不了 67%** ✗，但 1.37 已是最优点 ✓。

### 17.3 下一步（一行接线）

`ptq1_0.py` 已支持 `mode="robust"` ✓，但**打包器 CLI 还没暴露** ✗ ⇒ 加个 `--scale robust` ✓，然后用它重打 MoE ✓ 并验收 ✓。

### 17.4 同时被否证的（同一轮）

**把 `token_embd` 量化成三值（不折叠）会直接毁掉未绑定模型** ✗：27B 实测 0/2 事实、生成立即 `[end of text]`（省 2.26 GB 但不可用 ✓）。参见 §16 ✓。

**⇒ 结论：位置维度的保护已无空隙 ✓；值域维度的保护刚被证实有空间 ✓✓。**

---

## 18. 第六轮：规则被解开（2026-09-28）

### 18.1 一个方法论修正：两个块大小不是一回事

**旋转块 = 1024**（`prism.hadamard.block_size` ✓）
**量化块 = 128**（`QK_PTQ1_0` ✓）

我此前的评分脚本用 128 去折叠 ✗ ⇒ 一致率落到随机水平（33.3% ✓）。改成 1024 后立刻到 **87.5%** ✓✓。
**教训**：评分脚本必须和打包器用同一个折叠块 ✓，否则会得出“规则完全错了”的假结论 ✗。

自校验形状（正确时应当长这样）：
```
<|x|>nz/d = 0.964    <|x|>zero/d = 0.231    corr(|x|, |d·t|) = 0.651
```

### 18.2 真品的尺度：`d = 1.40 · mean|x|`（直接测出来的）

```
tensor                    n   d/mean|x|    cv%   d/max|x|    cv%
blk.0.ffn_gate.weight    320    1.3985   8.96     0.3961  14.34
blk.0.ffn_up.weight      320    1.3971   9.00     0.3955  14.32
blk.0.ssm_out.weight     384    1.4113  17.68     0.4054  21.82
```

跨张量一致 ✓，且**比 max 稳定** ✓（9% vs 14% ✓）。§17 的“均值匹配不动点”被否 ✗（那个预测 1.488 ✓）。

### 18.3 真品的判决：阈值在 **0.38·d**，不是 0.5·d

用真品自己的 `d` 扫阈值（7680 个样本 ✓）：

```
 theta    agree   nonzero
 0.360   90.49%   69.43%
 0.380   90.38%   67.72%   ← 非零命中 67.2%，一致率≈峰值
 0.400   90.36%   66.09%
 0.500   87.77%   58.11%
```

**两个独立观测量指向同一点** ✓✓（非零比例 ✓ + trit 一致率 ✓）。

### 18.4 两个旋钮是正交的（关键结构）

扫 (a, θ) 二维，`d = a·mean|x|`、判决在 `θ·d`：

```
 a     theta  nonzero   agree
1.30   0.40   67.92%   90.47%
1.40   0.38   67.20%   90.43%
1.50   0.36   66.74%   90.47%
```

**一致率只取决于乘积 `a·θ`** ✓✓ —— 所有落在 `a·θ ≈ 0.53` 的组合都得 90.4% ✓，而 `a` 单独变化毫无影响 ✓。
⇒ **`θ` 决定哪些 trit 存活** ✓；**`a` 只决定它们的重建幅度** ✓（trit 里不含幅度 ✓）。所以两者必须**分开**标定 ✓。

### 18.5 直接证据：他们开源的编码器不是产出模型的工具

| | d | 判决 | 非零 | 对真品的一致率 |
|---|---|---|---|---|
| 开源 `quantize_row_ptq1_0_ref` | `amax` | 0.5·d | **16.6%** | **≈33%（随机）** |
| 真品实测 | **1.40·mean|x|** | **0.38·d** | **67.2%** | — |

**⇒ 拿他们自己开源的编码器去比他们自己发布的模型，一致率是随机水平** ✓✓。这是“配方未开源”的硬证据 ✓，不再是推测 ✓。

### 18.6 我此前错在哪

最小二乘的 `d ≈ 1.67·mean|x|` ✓ 配 θ=0.5 ✓ ⇒ 阈值 **0.83·mean|x|** ✗
⇒ 只保留 **51%** ✗（真品 67% ✓），**且重建值系统性偏大** ✗。两个错误同向叠加 ✓。

### 18.7 已落地

```powershell
python prism-pack\prism_pack_hf.py <hf> <out.gguf> --block 1024 --scale ref --theta 0.38 ...
```
`ptq1_0.py` 的 `mode="ref"`：`d = 1.40·mean|x|`，`trit = 0 if |x| < 0.38·d else sign(x)` ✓。

### 18.8 剩下的缺口（已定位完毕）

**分箱诊断（这才是真相）**：
```
     |x|/d range        n  agreement
       0.70-1.00     1337     99.63%
       1.00-1.50     1313     99.92%
       1.50-99.00     759    100.00%
       0.34-0.42      523     53.92%   ← 只有阈值带内掉

sign agreement (over reference non-zero slots): 100.00%  ✓✓✓
zero-pattern agreement (over reference zero slots): 84.55%
```

**符号一致率 100.00%** ✓✓ ⇒ **折叠逐位正确，无任何残差** ✓（此前把 9.6% 归给折叠是错的 ✗）。不一致**全部**集中在零模式 ✓，且**只在边界带内** ✓。

**零模式的形状是秩序规则**：
```
                        blocks    mean      sd   min   max
blk_0.ffn_gate.weight    10240   86.05    0.34    86    95
blk_0.ffn_up.weight      10240   86.05    0.34    86    99
blk_0.ssm_out.weight     12288   86.05    0.37    86    95

observed sd 0.35  vs  binomial 5.31  ->  ratio 0.067
```

**每块非零个数锁在 86（sd 0.35）** ✓ ⇒ 不是“低于阈值就清零” ✗，而是“**每块取模最大的 86 个**” ✓。**86/128 = 67.19%** ✓ —— 那个神秘恒定常数就是这么来的 ✓。
（这一条解释了：为何比例恒定 ✓、为何全局阈值匹配不上 ✓、为何不一致只在边界 ✓。）

**两种规则等价**：秩序 k=86 与阈值 0.38·d 都给 **90.44%** ✓ ⇒ 残差不存于规则形式 ✓，而是边界处约 3 个/块的并列翻转 ✓ —— **全部是小值 ✓，对模型伤害远小于我此前那版的系统性错误** ✓。

## 19. 第七轮：配置 A 在正确规则下重测 —— 依然更差，但指向运行时（2026-09-28）

### 19.1 为什么重测

真品元数据是 `gdn_v_grouped=true` **且** `ssm_out` 被折叠（401 个里含 48 个 `ssm_out`）✓；我此前是 `false` + 不折 ✗。旧否证（4/4 → 0/4 ✓）是在**错误的尺度规则**下做的 ✗ ⇒ 前提已变 ✓。

### 19.2 实现 `--config-a`

我的 fork（`adfffbe`）**已经有** v7 的 per-tensor 实现 ✓（`conversion/qwen.py:568` 的 `_hadamard_folds_tensor()` ✓、`.out_proj` 的 grouped 分支 ✓、`self._hadamard_gdn_v_grouped = True` ✓）。
缺的只是触发器：`hadamard_folded_names()` 本该读 `hadamard_packing.json` ✗（只有他们的私有量化器能产出 ✗）。
`--config-a` 用猴子补丁把 `linear_attn.out_proj.weight` 加进去 ✓ —— 因为我**逐层都折** ✓，按种类匹配在这个场景下是对的 ✓（不是 fork 警告的那个 bug ✓）。同时关掉 `SKIP_FOLD` ✓；原有的断言会拦住“折了 ssm_out 却没保持 grouped 序”这种静默错误 ✓。

### 19.3 结果

```
配置 A（折 ssm_out + gdn_v_grouped=true） :  facts 0/4   rep6 106   uniq 0.21   ✗
不折 ssm_out（gdn_v_grouped=false）        :  facts 2/4   rep6  65   uniq 0.43
```

打包侧完全符合预期 ✓（这说明是模型行为问题，不是配置没生效 ✓）：
```
converter kept the grouped V order for ssm_out (gdn_v_grouped)  ✓
folded 291, kept 362, fallback 80  (93.39 GB -> 11.17 GB)  in 898s
prism.hadamard: block=1024 folded=291 widths=[2048, 4096] gdn_v_grouped=True  ✓
assert 未触发 ✓；291 - 261 = 30 = 线性注意力层数 ✓（40 层 / interval 4）✓
```

### 19.4 推论（重要）

**真品用配置 A 且工作正常 ✓，我用同一配置却更差 ✗ ⇒ 差异不在配置选择，而在我的运行时装它不对** ✓。
最可能：`hadamard-v7` 那个分支修的 **“grouped-order scoping”** ✗ —— 我的是 v5 冻结分支 ✗。
⇒ **下一步：用 `prism-v7` 重建后重测配置 A** ✓（判据现成 ✓，只换运行时 ✓）。

### 19.5 本轮工程坑

- **NAS 有配额**：文件系统报 979T 空闲 ✗，但写 12 GB 会 `cp` 失败 ✗ ⇒ 把 `models/`（Qwen3.8-27B，52 GB，可重下 ✓）移到 `/tmp` ✓ 后立刻能写 ✓
- **`echo '<b64>' | base64 -d` 在本实例会写出 0 字节** ✗ ⇒ 用 `printf '%s' '<b64>' | base64 -d` ✓
- **`dsw.py put`（Jupyter contents API）404** ✗ ⇒ 只走 base64-over-shell，单文件 ≲ 20 KB
- 远端 `grep -oE "a|b"` 的 `|` 被 `/bin/sh` 拆开 ✗ ⇒ 不用分支模式
- 大日志无换行 ✗ ⇒ `tr -cd '[:print:]\n' | tail -c N`

### 19.6 产物（持久区 `/mnt/workspace/prismwork/out/`）

| 文件 | 大小 | 说明 |
|---|---|---|
| `moe35-ref-rule.gguf` | 12,743,572,512 | `--scale ref --theta 0.38`，不折 ssm_out |
| `moe35-configA.gguf` | 12,295,307,136 | 同上 + `--config-a` |
| `moe35-ls.gguf` | 12,295,307,136 | 早期最小二乘版 |
| `qwen38-27b-d672.gguf` | 10,900,923,328 | 稠密 27B 实验 |

## 20. 第八轮：两个测量口径错误，与重新武装清单（2026-09-28）

### 20.1 我犯了两个同类的错（都是“尺子”而非“对象”）

| # | 错误 | 后果 | 教训 |
|---|---|---|---|
| 1 | 验收脚本没洗掉 `Loading model...` 的退格转圈字符 | `rep6` 被污染成 65/106，不可用 | 处理模型输出前先清控制字符 ✓ |
| 2 | 拿“我的产物字节”去比“真品字节” | 得到 0.00%，我一度当成重大发现 | **符号向量不同 ⇒ 折叠结果完全不同 ⇒ trit 无关** ✗ |

**第 2 条最危险** ✓：它差点让我把“口径不一致”当成“编码器有 bug” ✗。
**铁律**：跨实现比 trit，**必须用同一个符号向量** ✓。`rule_fit.py` / `find_theta.py` 全程用真品的符号向量 ✓，所以 90.4% 有效 ✓；`block_agree.py` 用了我自己的随机向量 ✗，所以 0% 无意义 ✗。

### 20.2 已排除的错误假设（含自己制造的那个）

- **“编码器与解码器不互逆”** —— **已实测否证** ✓✓：`roundtrip2.py` 把真品的块解码出 trit、再用同一 `d` 反编，**400/400 逐字节复现** ✓。存储层是忠实的 ✓。
- 配置 A（折 `ssm_out`）✗（v5/v7 双运行时都更差）
- 运行时版本 v5 vs v7 ✗（指标逐位相同）
- 块大小 1024 vs 512 ✗（都坏；512 的 rep6 略好：106 → 20）

### 20.3 已实现但本轮未跑完的决定性测试

`--signs-from <dir>` ✓ 已加进打包器 ✓（读 `sign_<width>.npy` 而不是自己随机生成 ✓）。

```powershell
python3 pphf2.py /tmp/models/qwen38-27b /tmp/d27ref.gguf \
    --block 1024 --scale ref --theta 0.38 --signs-from /mnt/workspace/prismwork \
    --fork /mnt/workspace/prismwork/prism-src
python3 block_agree.py /tmp/d27ref.gguf     # 期望回到 ~90%
```

- 回到 **~90%** ⇒ **整条链路（折叠+规则+编码）全部正确** ✓ ⇒ 问题确定在 MoE 特有部分或运行时用法 ✓
- 仍为 **~0%** ⇒ **确有一处未看见的真差异** ✓，那次就是真发现 ✓

**真品的符号向量已存 NAS** ✓：`/mnt/workspace/prismwork/sign_{5120,6144,17408}.npy` ✓（从 GGUF 元数据提的 ✓）—— `/tmp` 丢了也不影响 ✓。

### 20.4 重启后重新武装清单（/tmp 会丢，NAS 永久）

1. 建 CPU 构建：`cmake -S .../prism-src -B /tmp/llb -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_METAL=OFF -DGGML_SYCL=OFF -DGGML_OPENCL=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF` + `--build /tmp/llb -j 48`（~3 分钟 ✓）
2. 重下 27B：`modelscope download --model Qwen/Qwen3.8-27B --local_dir /tmp/models/qwen38-27b`
3. 重下真品：`curl -sL -o /tmp/bonsai2-ptq10.gguf https://hf-mirror.com/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PTQ1_0.gguf`
4. MoE 权重（若要重打）：`modelscope download --model Qwen/Qwen3.6-35B-A3B --local_dir /tmp/moe35`
5. 验实例号：`python probe_gateway.py`（实例号每次重启必变 ✓）

### 20.5 当前验证矩阵

| 层 | 状态 | 证据 |
|---|---|---|
| 尺度/判决规则 | **已验证** ✓ | 对真品 90.4%（=1.66895 下的上界），且扫参已到最优 |
| 折叠数学 | **已验证** ✓ | 符号一致率 100.00%；F16 等价 |
| 字节编码 | **已验证** ✓ | roundtrip 400/400 逐字节 |
| 内容选择（哪些张量进三值） | **未验证** ✗ | MoE 无参服物；`fallback 80` = 512 宽的 `ffn_down_exps`/`_shexp` |
| 模型行为 | **坏** ✗ | 事实 0~2/4，循环明显 |

**⇒ 三层已清 ⇒ 怀疑只剩“内容选择”与运行时用法** ✓。若 20.3 回到 90% ✓，“内容选择”就是唯一方向 ✓。

## 21. 第九轮：90% 不够 —— 残差定量定位（2026-09-28 夜）

### 21.1 决定性实验与结果

用真品的符号向量重打稠密 27B（`--signs-from /mnt/workspace/prismwork` ✓）：

```
trit 一致率     ~90.4%    （rule_fit 独立测得）
字节一致率       57.40%    （实测）
整块全同率        0.00%    （实测）
预测 P(字节同) ≈ 0.904^5 = 0.604      实测 0.574  ✓✓
```

三数自洽 ✓ ⇒ **不存在隐藏的编码错误** ✓；且符号向量一致后，字节一致率从 **0% → 57%** ✓⇒ 上一次的 0% 确实是口径问题 ✓。

**但这份 27B 产物仍然坏** ✓（事实 0/4 ✓、rep6 10 ✓）。

### 21.2 关键定量结论：90% 远远不够

trit 存的是 `±d`，而 `d = 1.40·mean|x|`。**边界处翻转一个 trit ＝ 把一个本该接近 0 的小值换成满幅值**。一个 128-trit 块里约 13 个翻转 ⇒ **大批权重被替换成大幅错值** ✗。
**⇒ 90.4% trit 一致不能被当作“基本对了”** ✗，它对应的是一个完全不可用的模型 ✓。

### 21.3 残差在哪里（精确画像）

```
符号一致率        100.00%   ✓   数值正负全对
零模式一致率       84.55%   ✗   分歧全在“哪些值被清零”
每块保留个数       86，sd 0.35  ✓   连个数都完全一致
```

**个数对、正负对，只有“选哪 86 个”有 15% 分歧** ✓。

两个候选，且它们做出的预测不同：
1. **他们排序依据不是 `|x|`** ✗，而是与之 ~90% 相关的量；
2. **我的折叠值在边界附近有微小误差** ✗（Hadamard 累加顺序/精度），不足以翻转正负 ✓，但足以重排边界处的近似并列值 ✓。

**第 2 条与观测完全吻合** ✓⇒ **下一步：让折叠值与真品逐位一致** ✓。

### 21.3b 为什么只能是精度

`edge_check.py` 的分箱：远离阈值**处处 100%** ✓，边界带 **53.9%** ✓（≈二选一的随机水平 ✓）。

- 远离处全对 ⇒ **我的折叠值域排序是对的** ✓（变换选对了 ✓）
- 边界带随机 ⇒ **那里全是近似并列值** ✓，我的值与真品差一点点 ✓，谁被保留就跟着变 ✓
- **⇒ 任何“更好的阈值/秩序规则”都修不了这里** ✗ —— 本来就分不开
- **⇒ 唯一能修的是把 Hadamard 的累加顺序与精度做成和真品逐位一致** ✓（很可能是 f32 的某种固定累加顺序 ✓）

### 21.3d **一个未解决的矛盾（交接重点）**

推理链：
1. 折叠自洽 ✓（`--no-quant` 下旋转后 F16 与纯 F16 等价，已验 ✓）
2. 规则与真品近似 ✓（90.4% ✓，分歧集中在边界近似并列值 ✓）
3. ⇒ 一个自洽、规则近似相同的模型**应当能用** ✓
4. **但它实测是坏的** ✗（MoE 0~2/4 ✓，稠密 27B 0/4 ✓）

**⇒ 前提里一定有洞** ✓。最可能的那条：

> **“旋转后 F16 等价”只验证了逆变换的数学** ✓，
> **没验证运行时对具体架构的图路径，是否真的在每一条该走的地方都施加了那条逆变换** ✗。

这类错是**静默的** ✓：模型能加载 ✓、能跑 ✓、只是算错 ✓ —— 与观测到的“事实掉光但文本结构还在”完全吻合 ✓。

**依据**：`conversion/base.py` 自己注释说，允许清单的前提是“只有图上经过 `build_lora_mm` 的地方才施加激活变换” ✓。**在清单里只代表架构名匹配** ✗，不代表该模型所有目标矩阵乘都走了那条路径 ✓。

### 21.3e 下一轮第一个实验（可证伪）

**给运行时的 Hadamard 变换加一个计数器** ✓：
- 统计实际执行次数 ✓，对比“应执行次数”＝折叠张量数 × 每次前向的 token 数 ✓
- 若对不上 ⇒ **整件事立刻解释** ✓✓（某些层被静默跳过 ✓）
- 若对得上 ⇒ 排掉这条路 ✓，回到内容选择 ✓

次要备选：对比 `qwen35` 与 `qwen3`（后者更简单、无 GDN）在同一套流程下的行为 ✓ —— 若 `qwen3` 正常而 `qwen35` 不正常 ⇒ 差异就在 GDN 路径 ✓。

（废除一条无效指标：`corr(|x|, |d·t|) = 0.651` 无意义 ✗ —— 真品的重建幅度恒为 `d` ✓，该相关系数不含信息 ✓。）

### 21.3c 下一轮可执行起点

1. 读出他们 C 实现里的 FWHT 累加方式（`ggml-cpu/quants.c` 的 PTQ1_0 向量路径 ✓、`ggml-vulkan` 的 FWHT 着色器 ✓）
2. 把 `prism_pack.fold_weight` 改成同顺序同精度
3. 用 `edge_check.py` 验收：边界带应从 53.9% 明显上升，远离处保持 100%
4. 边界带稳定后，再用 `blk_cmp.py` 看字节一致率是否从 57% 向更高处走
5. 最后才看模型行为 ✓

### 21.4 本轮的度量教训（第三个）

**“整块字节全同率”在 90% trit 一致下必然是 0，没有分辨力** ✗。有分辨力的是：
trit 一致率 ✓、字节一致率 ✓、以及 `edge_check.py` 的分箱曲线 ✓。

已完成的三层验证（§20.5）不受影响 ✓。

## 22. 第十轮：验收基准校准 —— 推翻了 rep6 判据（2026-09-29）

### 22.1 把真品放进同一套验收

一直没做的事：**验证判据本身** ✓。用同一个 `accept_ref.py`、同一提示、同样贪婪采样跑真品：

```
真品 Bonsai 2 27B :  事实 4/4  PASS ✓     rep6  9  FAIL ✗     uniq 0.57 PASS
我的稠密 27B      :  事实 0/4  FAIL ✗     rep6 10             uniq 0.42
我的 MoE(ref规则) :  事实 2/4  FAIL ✗     rep6 65             uniq 0.43
```

### 22.2 结论一：`rep6 ≤ 3` 是错的判据，废除

**真品自己也是 rep6 = 9** ✗，而我的 27B 是 **10** ✓ —— 几乎相同 ✓。
⇒ 贪婪采样下复读是这类模型的常态 ✓，与质量无直接关系 ✓。
**此前所有基于 rep6 的“模型坏了”判断都要打折** ✓（§19.3、§21 的行为判定 ✓）。

### 22.3 结论二：真正的差距只有一个——事实性

```
真品  4/4    我的 27B  0/4    我的 MoE  2/4
```

**⇒ 我的模型与真品一样流畅** ✓，差的是“知道的东西” ✗ ⇒ **不是“模型塌了”** ✗，是**知识路径被损坏** ✓。

嫌疑由此收敛到承载知识的路径：
- 词表侧：`token_embd` 已保 F16 ✓
- **输出侧：`output.weight` 被折叠并三值化 ✗**（27B 上是 **1.27 B 参数**，知识出口最大的一块 ✓）
- MoE 专家（仅 MoE 模型）✓

**对照**：真品元数据同样含 `output.weight` ✓ ⇒ 它也做了同样选择 ✓ ⇒ **问题大概不在“折没折”，而在“我折得对不对”** ✓。

### 22.4 验收脚本的两处待修

1. 生成文本里混入了模板宏残渣（`add text files using globbing pattern >` ✓）—— 提示构造需修 ✓
2. `rep6` 不再作为 pass/fail 项 ✓，改用：**事实探针（主）+ uniq（辅）+ 目测连贯性** ✓

### 22.5 下一步

**靶子：`output.weight`** ✓。比对它的折叠与量化与真品是否一致（用 §5 的 `blk_cmp.py` / `edge_check.py` ✓，符号向量已对齐 ✓）。
若它也吻合 ⇒ 知识损失另有出处（MoE 专家 / 或真品本身就在此处弱 ✓）⇒ 再做一次分层探针。

## 23. 第十一轮：同口径全模型对比 —— 无结构性损坏（2026-09-29）

### 23.1 度量修正（第三次踩同一个坑）

**“整块字节全同率”在 90% trit 一致下必然是 0** ✗（`0.9^128 ≈ 10⁻⁶` ✓）。
**有效指标：逐字节一致率** ✓ —— 每字节 5 个 trit ✓ ⇒ 预期 `0.9^5 = 59%` ✓；
若某张量沿错轴折叠 ⇒ 会崩到 ~0.39%（随机）✓。**一次换对，结论立刻可读。**

### 23.2 结果（我的 27B vs 真品，符号向量已对齐）

```
加权逐字节一致率    63.02%     预期 59%   实测略高 ✓
低于 50% 的张量        3 个      最低 47.71%（blk.55.attn_k）
低于 20% / 5% / 1%    0 个  ✓✓
最佳：output.weight  84.93%
```

### 23.3 三条结论

1. **没有任何张量是坏的** ✓✓：最差 47.7% vs 随机 0.39% ✓，353 个均在 48~85% ✓
   ⇒ **“某个张量折错轴”这类结构性损坏正式排除** ✓
2. **`output.weight` 反而是一致率最高的（84.93%）** ✓
   ⇒ **§22.5 猜“知识出口受损”是错的** ✗，收回 ✓
3. **失败来自 9% 的聚合，不是任何单点** ✓
   ⇒ §21.2 用**单层**估算（+14%）去否定“90% 不够”是错的 ✗——**把复利当单利算了** ✓
   ⇒ **“90% 不够”的结论成立** ✓（我在 §21.2 推翻它反而是错的 ✓）

### 23.4 精确靶子

要让边界判决逐一相同 ✓，而边界处全是近似并列值 ✓ ⇒ 二选一：
- **让折叠值逐位复现他们** ✓（Hadamard 的累加顺序与精度 ✓，见 §21.3c 的清单 ✓）
- **或找到他们的并列打破规则** ✓（可能根本不是一个纯阈值 ✓）

**验收入口**：`block_agree.py`（逐字节率 ✓，不要用整块全同率 ✗）+ `edge_check.py` 的分箱曲线 ✓。

## 24. 第十二轮：一个逻辑结论，排掉一整类假设（2026-09-29）

### 24.1 边界间距

```
第 86/87 名间距（相对 d）：中位数 0.652%   p10 0.097%   p90 2.197%
fp32 舍入步长：6e-6 %
```

**要看低分位而不是中位数** ✓：翻转只发生在 gap 小于我误差处 ✓，
**p10 = 0.097% 正好对应约 10% 的块** ✓ —— **就是实测的 10%** ✓。
⇒ 反推：我的折叠与真品差 **≈ 0.1%** ✓。

（教训：光看“中位数比步长大十万倍”会推出“精度无关” ✗——差的结论 ✓。）

### 24.2 精度不是变量（实测）

```
fp32 (baseline)          89.00%
bf16 input               89.00%    ← 无改善
bf16 input+output        89.08%
fp16 input               89.00%
```

### 24.3 **有条件**的逻辑结论（已修正）

> **前提：他们的判据是 `|x|` 的单调函数** ✓
> 在此前提下，若我的 `|x|` 与真品**相同** ✓，则任何单调判据（阈值 θ·d ✓ 或取前 86 名 ✓）
> 都会与真品**完全一致** ✓，至多在分界那**一个**元素上分歧 ✓。
> ⇒ 实测 10% 分歧 ⇒ **在该前提下，我的 `|x|` 与真品不同** ✓（量级 ~0.1% ✓）。

**⚠ 前提未验证** ✗。若判据**非单调** ✓（按列重要度加权排序 ✓、误差反馈 ✓、
联合优化 ✓），**值相同也能产生 10% 分歧** ✓。

**⇒ 正确表述是一个二选一** ✓：
1. 我的折叠值与真品差 ~0.1%
2. 他们的判据不是 `|x|` 的单调函数

**本轮已排除第 1 条下的所有精度机制** ✓（§24.2 入参舍入 ✓、§24.5 逐级舍入 ✓）
⇒ **天平倒向第 2 条** ✓。

（自洽性检验仍成立：我手上两条规则彼此只差 0.06% ✓，而各自与真品差 10% ✗
⇒ 差异不可能来自规则形式 ✓。）

> 若我的 `|x|` 与真品**相同** ✓，则**任何**块内对 `|x|` 单调的判据
> （阈值 θ·d ✓ 或取前 86 名 ✓）都会与真品**完全一致** ✓，
> 至多在分界那**一个**元素上分歧 ✓。
>
> **⇒ 实测 10% 分歧 ⇒ 逻辑上只能说明我的 `|x|` 与真品不同** ✗ ——
> **且差异大到能重排约 10% 的边界判决** ✓（量级 ~0.1% ✓）。

**这排掉了一整类假设** ✓：

| 候选 | 状态 |
|---|---|
| 尺度/判决规则的形式 | **排除** ✗ |
| 输入精度 bf16/fp16 舍入 | **排除** ✗（实测无改善）|
| 逐阶 fp32 舍入 | **排除** ✗（~1e-6，小五个量级）|
| 折叠公式（符号序、H 变体）| 已测 ✗（换序掉到 32%）|

### 24.5 逐级蝶形舍入（实测，无效）

```
stage rounding    agreement
fp32                  89.00%
bf16                  89.00%   ← 无任何变化
fp16                  89.00%
```

工具：`remote/stage_prec.py` ✓。
**⇒ “降低精度”这一族全部排除** ✓：入参舍入 ✗、逐级舍入 ✗、逐阶 fp32 累积 ✗。

### 24.4 下一轮唯一剩下的方向

**他们的 Hadamard 在逐阶中间步骤上损失了精度** ✗（不是只在输入上）——
例如每级蝶形过一遍 fp16/bf16 ✓，或累加用低精度中间量 ✓。
这类误差结构与“整块入参舍入”不同 ✓，**不会被 §24.2 的测试捕捉到** ✓。

**实验：逐级蝶形做 fp16/bf16 舍入，看一致率能否越过 90%** ✓。
判据：`prec_check.py` 扩展成逐阶舍入的多个变体 ✓。

## 25. 第十三轮：结构性排除 —— **0/1280**（2026-09-29）

### 25.1 更锐利的判据

不再看一致率 ✓，改问一个二值问题 ✓：

> **真品每块保留的 86 个，是否恰好等于我按 `|x|` 排序的前 86 个？**

### 25.2 结果

```
blocks examined        1280
精确前缀          0  (0.00%)   ← 一个都没有
每块保留数        mean 86.03   min 86   max 90
对称差           mean 13.69
```

### 25.3 推论（比一致率强得多）

> **凡“他们的判决是我这批 `|x|` 的单调函数”的假设，全部死亡** ✗：
> 阈值偏移 ✓、名次差几 ✓、近似并列扰动 ✓，**都要求至少部分块能切出来** ✗。
> **实测 0** ✓。

⇒ **他们的判决依赖的量，不是我的折叠 `|x|`** ✓✓。

这也修正了 §24.3 的二选一：不是“值差 0.1% **或** 判据非单调” ✓，
而是 **判据确实不建在我的 `|x|` 上** ✓。

### 25.4 为什么这解释了闭源

最自然的候选：**重要性加权判据**（AWQ 式，如按 `|x| · sqrt(E[x²])` 排序）✓：

| 观测 | 吻合？ |
|---|---|
| 不是我的 `|x|` 的截断 | ✓（0/1280）|
| 分歧集中在边界附近 | ✓（两者强相关）|
| 块内位置无结构 | ✓（权重差异盖过重要度差异）|
| 每块恰好 86 个 | ✓（秩规则）|
| **天然需闭源** | ✓✓ **需要校准数据 + 流程，不是一个公式** ✓|

⇒ 白皮书“mathematically grounded framework”的真实含义：
**不是一个能猜出来的常数** ✗，**而是一个**需要真实激活才能定的**判据** ✓。

### 25.5 下一轮唯一的方向

**需要一次校准跑：拿少量文本过模型，统计每列 `E[x²]`** ✓，
然后用 `|W| · sqrt(E[x²])` 排序 ✓ 取前 86 ✓ → 与真品比前缀率 ✓。
判据现成：`prefix_check.py` ✓（把 `|x|` 换成重要度即可 ✓）。
成本：一次短校准（分钟级）+ 一次模型加载 ✓。

## 26. 第十四轮：联立两个观测，又掉一支（2026-09-29）

### 26.1 把数字合起来算

```
对称差 mean 13.69  ⇒ 约 7 次交换/块
⇒ 若成因是“值扰动” ⇒ 扰动幅度 ≈ 7 × 0.65% ≈ 4.5% 的 d

但分歧中 4.3% 落在我的 rank 48–63
⇒ 4.5% 的扰动最多把元素挪过 ~7 个名次
⇒ rank 48 不可能跳进前 86   ✗
```

### 26.2 结论

**“我的值差一点”这一支被自己的数据否掉** ✗（任何合理幅度的扰动解释不了 rank 48 处的分歧）。
**“折叠算子不同”又被符号一致率 100% 排除** ✗。

⇒ **只剩一条** ✓：

> **他们的判据用的是另一个量** ✓，
> **不是“我这批值上任何接近的排序”** ✓。

**连带影响** ✓：§25.5 把误差反馈放回候选池是不对的 ✗ —— 它仍在 `|x|` 附近排序 ✓，同样救不了 rank 48 ✓。

### 26.3 教训（比单次度量错误更重要）

前几轮我一直**分开解释单个数字** ✗，没有把它们**联立** ✓。
一联立就掉了一个候选 ✓ —— 这种失误直接拖长搜索 ✓，应作为习惯改正。

### 26.4 候选池现状

| 候选 | 状态 |
|---|---|
| 同值 + 单调判据（任意形式）| **排除** ✗（0/1280）|
| 值扰动 | **排除** ✗（§26.1）|
| 折叠算子不同 | **排除** ✗（符号 100%）|
| 误差反馈 | **排除** ✗（仍在 `|x|` 附近排序）|
| 精度机制 | **排除** ✗ |
| **依赖外部数据的判据**（激活重要度等）| **唯一剩下** ✓ |

### 26.5 下一步（方向已唯一，成本明确）

做一次**校准跑** ✓：给 llama.cpp 加一个激活统计钩子 ✓，拿少量文本过原始模型 ✓，
收集折叠后各列的 `E[x²]` ✓，按 `|x| · sqrt(E[x²])` 排序取前 86 ✓，
用 `prefix_check.py` 量前缀率 ✓。

- 前缀率→接近 100% ⇒ **判据锁定，配方可自行复现** ✓✓
- 仍近 0 ⇒ 连“重要度加权”也不是 ✓ ⇒ **说明缺的不只是数据，而是我一整类没想到的东西** ✓
  （这就是真正的分界线，要如实告诉用户 ✓）

## 27. 第十五轮：机制线索 —— 硬地板 86 与离散值网格（2026-09-29）

### 27.1 被忽略很久的线索

真品每块非零数：**min 恰好 86** ✓、max 99 ✓、sd 0.35 ✓、均值 86.05 ✓。

- **“永不低于 86”** ⇒ 机制**保证**至少 86 ✓
- **“偶尔更多”** ⇒ 边界上有**并列** ✓
- **并列需要粗网格** ✓（fp32 连续值几乎不碰头 ✓，bf16 会 ✓）

⇒ 机制很可能是：**取第 86 大的值为阈值，保留所有 ≥ 它的** ✓ —— 计数 = 86 + 边界并列数 ✓，
同时解释地板与溢出 ✓。

### 27.2 测试盲区（我自己的）

第一版把舍入加在**入参**上 ✗ —— 折叠在 fp32 里算完，输出的值仍然连续 ⇒ 当然零并列 ✓。
**舍入必须加在折叠**之后** ✓**。换位置后立刻从 0 并列变成有并列 ✓。
而且前几轮一直在看**一致率**这一个数 ✗，**计数分布的形状里一直藏着机制信息** ✓。

### 27.3 结果（640 块）

```
                    块中有超出比例    平均超出
真品                    3.1%          0.037
fp16 折叠后舍入          1.2%          0.014
bf16 折叠后舍入          8.6%          0.097
fp32                    0.0%          0
```

**签名是真的** ✓，**且性质与“离散值网格”完全一致** ✓（网格越粗、并列越多 ✓）。
**但真品的网格落在 fp16 与 bf16 之间** ✗（约 9–10 位尾数 ✓，非标准 ✓）。

### 27.4 诚实的限定

**这不是锁定** ✗——是“机制方向对了” ✓、具体精度未对上 ✓。
**更重要的是：它只解释很小一块** ✗——超出量最大仅 +4 ✓、中位为 0 ✓ ⇒
**即使完全复现，也只修掉 10% 分歧的一小部分** ✓。
⇒ **它是关于机制的线索，不是分歧的主要来源** ✓。

### 27.5 对 §26 的修正（重要）

§26 写“trit 不是折叠权重的纯函数” ✗ —— **修正：它仍然是权重的函数** ✓，
**只是值网格比我以为的粗** ✓，“不是纯函数”这个判断**过早，收回** ✓。
§26.4 的候选池应改为“**仍有候选，但都是权重层面的细节**” ✓。

### 27.6 下一轮判据

扫网格精度（尾数位数）✓，**要求同时命中**：超出比例 **3.1%** + 平均超出 **0.037** ✓。
工具：`remote/tie_check.py` ✓（把 `to_fp16/to_bf16` 换成可配尾数位数的舍入 ✓）。

### 27.7 网格扫描结果

```
mantissa   tie rate  mean excess
       8      4.38%       0.0500
       9      2.19%       0.0234
      10      1.25%       0.0141
      16      0.00%       0.0000
      23      0.00%       0.0000

真品目标：    3.12%       0.0375
```

**真品落在 8 位与 9 位之间** ✗（≈8.6 位 ✓）—— **无标准格式对应** ✗。

**但很可能是我的分布略有偏差** ✓：并列率同时取决于**网格**与**我折叠值在分界附近的密度** ✓，
密度稍不同，并列率就整体平移 ✓。⇒ **这个数不能当结论用** ✓。

**量级提醒** ✓：真品平均超出仅 0.037 ✓、中位 0 ✓、最大 +4 ✓ ⇒
**即使网格完全对上，也只修掉 10% 分歧的极小一部分** ✗。
⇒ **机制线索，不是分歧来源** ✓。

工具：`remote/grid_scan.py` ✓。

### 18.9 v7 分支（网络已打通）

本地 git 可用姿势：`-c http.sslBackend=openssl -c http.proxy=http://127.0.0.1:44343` ✓

| 分支 | 内容 |
|---|---|
| **`prism-v7`** | `c1abda3`，**v7 正式分支** |
| `hadamard-v7` | 修 dead F16 fusion gate / contract validation / **grouped-order scoping** |
| `hadamard-explicit-signs` | 白名单补 `output.weight` + `attn_gate`（此前会拒绝加载 ✓）；逆变换移到设备侧（PCIe 往返导致解码 3× 损失 ✓） |
| `hadamard-gguf-contract` | Vulkan 仅在设备支持 fp16 时创建 F16 FWHT 流水线 |
| `feat/pq1_0-g64` | PQ1_0 group-64 变体 |

**`ggml-quants.c` 在 v5 与 v7 逐字节相同** ✓（同 SHA ✓）⇒ 编码器跨版本未变 ✓，配方差异不在 codec 里 ✓。
**`hadamard_packing.json` 整个仓库只有读、没有写** ✓✓（`conversion/base.py:626,637` ✓）—— 这是“隐藏件”的确切指认 ✓。

### 15.6 当前最好的可用产物

**`prism-pack\q\qwen36-moe-b1024.gguf`**（11.87 GiB）—— block 1024 + 最小二乘 + `ffn_down_exps` 走 Q4_0 兑底：
**事实保留** ✓（4 事实探针最高 4/4 ✓）、会循环 ✗ ⇒ 配 `--repeat-penalty 1.3` 可用 ✓。详见 §15.4 验收数据 ✓。

### 15.5 剩余空间（下一步的候选）

旋转已定 ✓ ⇒ 值得试的只剩：

1. **那 1.29 倍常数的来源** —— 候选：符号向量的实际取值含义 ✓、额外的逐块归一化 ✓、真品的 `file_type=143`（更新的 fork，约定可能小改 ✓）
2. **换到最新 fork**（`prism-v7` 代 ✓，见 `MODEL-FORMATS.md` ✓）后重建——`adfffbe` 是 `origin/prism`（= prism-v5 冻结分支）的 tip ✓，可能不是 Bonsai 2 用的那一支 ✓
3. 输出感知的校准（需要校准数据 ✓，与"无训练"不矛盾 ✓）

**工具**：`remote/orient_search.py`（并行约定搜索 ✓）、`rule_test.py`（规则对比 ✓）、`remote/scale_rule.py`（统计量比值 ✓）、`ab_quant.py`、`scan_d.py`。
**迭代台**：`SmolLM2-135M`（93 MiB，arch `llama`）✓。

### 12.8 顺带确证：CHECK A 通过

§1 那条最重要结论（旋转器正确）**不是同义反复**：
```
/tmp/q35-rot.gguf    prism.hadamard.weight_names len=25   ← 真折了 25 个
/tmp/q35-plain.gguf  无任何 hadamard 键
```
修复后小模型复核：折叠 27（+2 条 `ssm_out`）、`gdn_v_grouped=True`、round-trip 相对差 1.2e-4（随机模型噪声级）。

## 28. 第十六轮：PPL 分水岭 —— 不是"差 10%"，是差 309 倍（2026-09-29）

`
我的包 (d27r2.gguf, 用真品符号向量)  PPL = 1290.55
真品 (Bonsai 2 27B, 同一语料)        PPL =    4.18
                                      --------
                                      309 倍
`

### 28.1 三条结论

1. **"90% trit 一致"完全不能代表接近** ✗ —— 早先"边界翻转是小值、代价小"的直觉是**错的** ✓。
2. **之前"模型坏了"的观察是真的** ✓ —— PPL 给了独立印证 ✓（不是采样假象 ✓）。
3. **309 倍解释不了"10% 权重错"** ✗⇒ **必有系统性错误** ✓，且**它不表现在 trit 一致率上** ✓。

### 28.2 立即可测的候选

真品元数据 gdn_v_grouped = true ✓，我的包是 alse ✗；64 层里 30 层是 GDN。
配置 A（	rue）只在 MoE 上测过 ✗，而 MoE 基线本就坏 ✗ ⇒ **那个对照无信息量** ✓。
**⇒ 27B 才是该测的地方，且现在有 PPL 这个干净判据** ✓（不受采样干扰 ✓）。

### 28.3 教训

**我一直在用"一致率"当接近度** ✗ —— **而一致率与结果之间可以差 309 倍** ✓。
**⇒ 今后判定接近度：只认 PPL / 事实探针，不认比特一致率** ✓。

## 29. 第十七轮：找到恒定幅度偏差 2.45%（2026-09-29）

### 29.1 检查清单上的洞

此前验证了 trit ✓、验证了折叠 ✓，**却始终没验证过写入文件的 `d`** ✗ ——
而 "`d = 1.40·mean|x|`" 一直被当作已验证（其实只在 4 行 × 5 块上量过比值）。

### 29.2 测量（1.88 亿个块）

`
mine/reference d:
  median 1.02455    mean 1.02458    std 0.02287
  p1 0.97003   p25 1.01040   p75 1.03841   p99 1.08328

per-tensor medians: mean 1.02552  std 0.00727  min 1.00895  max 1.04486
`

**紧密聚簇 ⇒ 恒定系数错，不是散布** ✓ ⇒ 真值为 **1.40 / 1.02455 = 1.36645** ✓。

### 29.3 为什么这解释 309 倍

`d` 是幅度：**每个非零权重的绝对值都系统性偏大 2.45%** ✗，穿过 64 层放大 ✓，
**而 trit 一致率完全看不出来** ✓✓。（幅度误差在一致率上是隐形的，这是关键。）

### 29.4 已改

`_SCALE_MULT["ref"]` 由 1.40 改为 **1.36645** ✓，正在重打一份 27B，PPL 待测。

### 29.5 早先那个 1.3985 是采样噪声

只用了 4 行 × 5 块（20 块）✓，块间 std 0.0229 ⇒ 噪声足以给出 2.3% 偏差 ✓。
**1.88 亿块的测量才是权威** ✓。

## 30. 第十八轮：d 与 trit 必须联合自洽（2026-09-29）

### 30.1 一个操作失误与它的确认

第一次 d 修正后 PPL 与上一版**到小数点后四位完全相同** ✗ —— 原因是我把
「编译 + 推送 + 启动」写在同一条命令里，PS 解析失败 ⇒ **推送没执行** ✗，而后单独启动的打包
用的是旧文件 ✗。重推后 PPL **确实变了** ✓ ⇒ 确认了这一点 ✓，也说明这类"并列多步命令"
必须拆开执行 ✓。

### 30.2 但方向是反的

`
真品                                 PPL =    4.18
d = 1.40·mean|x|  (trit 90% 同)      PPL = 1290.55
d = 1.366·mean|x| (同一批 trit)      PPL = 2244.52   ← 缩小幅度，差一倍
`

### 30.3 推论

**`d` 与 trit 必须联合自洽** ✓：真品的 `d` 是配合**真品的 trit** 最优的 ✓，
我拿**我的 trit**（差 10%）去配他们的 `d`，不是最优组合 ✓。
且 trit 个数锁死 86 ⇒ `d` 只影响幅度 ⇒ **实测说：对这批 trit，幅度应更大** ✓。

⇒ **正确方向：按我的 trit 拟合 `d`**（最小二乘 ✓），**而不是照抄真品的 `d`** ✓。

### 30.4 主次

**1290 / 2245 / 4.18** ✓ —— 所有变体都是灾难级 ✗。
**`d` 的调整是二阶效应** ✗，**主问题仍在别处** ✓。
下一步：`--scale ls`（幅度更大 ✓）看 PPL 走向 ✓。

## 31. 第十九轮：一刀切开 —— 错误全在量化侧；类型差异只剩 ssm_out（2026-09-29）

### 31.1 --no-quant 对照（判定性）

`
真品 (ternary)                 PPL =    4.18
我的 --no-quant (旋转 + F16)    PPL =    3.46   ← 比真品还好 ✓
我的量化版                      PPL = 1290 / 2245   ✗
`

**⇒ 折叠 ✓、符号向量 ✓、运行时对 GGUF 的处理 ✓ 全部正确** ✓✓，
**错误 100% 在量化这一步** ✓✓。这一刀把问题对半切开 ✓，也让"另一半已无问题"成为结论 ✓。

（F16 版 53.8 GB ✓、内存 28 GB ✓ ⇒ 靠 mmap 跑 ✓，用 `--chunks 4` 限时 ✓。）

### 31.2 类型差异（逐张量比对）

`
ssm_out      mine F16(1)   ref PTQ1_0(143)   ← 唯一实质差异
ssm_alpha    mine F16(1)   ref BF16(30)      ← 精度等价，无害
ssm_beta     mine F16(1)   ref BF16(30)      ← 同上
`

**实质差异只有 `ssm_out`** ✓：真品折了 ✓，我没折 ✗（= 配置 A/B 之别 ✓）。

**但必须立刻说清** ✗：**F16 对照里 `ssm_out` 也是 F16** ✓ ⇒ **"保持 F16"本身不毁模型** ✗
（它是**高精度**方向 ✓）。⇒ **这个差异解释不了 1290 倍** ✗。

### 31.3 现在最尖锐的矛盾

**trit 90% 对 ✓、`d` 只差 2.45%** ✗ ⇒ **两者都只该造成"略差"** ✓，**实测却是 370 倍** ✗✓。

⇒ 候选只剩两个（都属"静默且灾难性"一类）✓：
1. **`d` 在运行时被读成了别的东西** ✗
2. **元数据某处与实际不符** ✗ ⇒ 运行时会施加一个打包时**没施加**的旋转 ✓

### 31.4 操作教训（第二次同类）

「编译 + 推送 + 启动」写进一条命令 ✗ ⇒ PS 解析失败 ⇒ **推送没执行** ✗ 而启动执行了 ✓，
导致一次假阴性 ✓。**多步命令必须拆开执行** ✓。

## 32. 第二十轮：量级错了 —— 逐权重独立量化不可能达到 1.21 倍（2026-09-29）

### 32.1 闭环测量

`
blk.0.ffn_gate.weight    RMSE 45.61%   非零 67.99%
blk.0.ffn_up.weight      RMSE 45.64%   非零 67.95%
blk.0.ffn_down.weight    RMSE 45.59%   非零 68.06%

平均重建 RMSE 45.61%      朴素三值理论预期 ~46%
`

**⇒ 打包器就是一个逐块独立三值量化器** ✓，**RMSE 与理论值完全一致** ✓。

### 32.2 量级对比（这是关键）

`
真品：1.75 bit 三元 ⇒ PPL 从我的 F16 基线 3.46 升到 4.18（**1.21 倍，几乎无损**）
我  ：同比特宽度     ⇒ 45.6% 权重 RMSE ⇒ PPL 1290
`

**⇒ 逐权重独立量化器**做不到 1.21 倍** ✓✓：它必然给出 45.6% 量级的失真 ✗，
穿过 64 层即灾难 ✓。

**⇒ ⇒ 真品的配方是结构性不同的** ✓✓ —— **不是“选哪 86 个”** ✗，
**而是某种让三元近乎无损的机制** ✓。

**⇒ ⇒ ⇒ 前几轮在“哪些值存活”上的全部努力，优化的是细节** ✗ —— 缺的是另一个量级的东西 ✓。

### 32.3 标准答案，且我早期错误排除过

**误差反馈 / 迭代补偿（GPTQ 式）** ✓：**用真实激活分配误差** ✓，
**最小化输出误差而非逐权重误差** ✓。

**45.6% 的权重 RMSE 可以对应很小的输出误差** ✓ —— 只要误差落在激活的零空间里 ✓。

**早期以 MSE 为由否定它** ✗ —— **而真品的规则本身不是 MSE 最优** ✓ ⇒ **那条否证无效** ✓（§26.2 已自纠 ✓）。

### 32.4 材料已就位

`/tmp/imat.dat`（13.6 MB ✓）**就是真实激活统计** ✓ —— **CATQ/GPTQ 式方法的直接输入** ✓。
⇒ 下一轮：用 imatrix 作为 Hessian 对角，实现逐层输出误差最小化的三值量化 ✓，
再用 PPL 判定 ✓（唯一可信判据 ✓）。

### 18.9 v7 分支（网络已打通）

## 33. 第二十一轮：目标函数错了 —— ls 比"实测真规则"好 7.4 倍（2026-09-29）

### 33.1 客观排序（第一次有了 PPL 排序）

`
真品（dense 27B，三元）              PPL    4.18
b1024           (MoE, --scale ls )   PPL   26.52     ← 我最好的一版 ✓
moe35-ref-rule  (MoE, --scale ref)   PPL  197.39     ✗
dense 27B       (--scale ref)        PPL 1290.0      ✗
`

**同模型对比（都是 MoE）**：`ls` **26.52** vs `ref` **197.39** ⇒
**我花几周"量出来的真规则"比最朴素的最小二乘差了 7.4 倍** ✗✓。

### 33.2 为什么

- `ls` 最小化**权重重建误差** ✓（MSE 最优 ✓）
- `ref` 匹配**真品的 trit 模式** ✗

**两者不是一回事** ✓：**匹配 trit 模式 ≠ 最小化输出误差** ✗。

**⇒ 我一直在优化"和真品像不像"** ✗，**而该优化的是"模型还能不能用"** ✓。
**⇒ 所有"规则改进"无效的原因就在此** ✓：**改进的是相似度指标，与 PPL 不同向** ✓。

**⇒ 这正是"显而易见但注意不到"** ✓：**目标函数错了** ✗，**它就在出发点那一行** ✓。

### 33.3 副作用：一个口径错误

我曾把 MoE 的 1290 与 dense 的 4.18 直接比 ✗ —— **不同模型不可比** ✓。
同模型下 MoE 只有 26.5 ✓，**恰好对应"开头通顺、后面漂移"** ✓
⇒ **b1024 确实是我最接近的一版** ✓✓。

### 33.4 回到正轨后的旋钮

1. **保 F16 的范围**（注意力/SSM ✓）—— 对照在跑 ✓
2. **尺度/密度** —— 目标改为**降低输出误差** ✓，不是匹配真品 ✓
3. **真正对症的方向**：**逐层输出误差最小化（GPTQ 式，用 imatrix ✓）** ✓
   —— 材料已就位 ✓（`/tmp/imat.dat` ✓），**这才是"逐权重独立"之外的那条路** ✓

### 33.5 b1024 复刻的元数据 delta

复刻体积 **12,743,572,512** vs 原版 **12,743,572,544** ⇒ **差 32 字节** ✓ ⇒
**张量数据大概率逐字节相同** ✓，**差一个元数据字段** ✓（待对出 ✓）。


## 34. 第二十二轮：找到了 —— 注意力/SSM 张量必须保 F16（2026-09-29）

### 34.1 决定性对照

```
b1024    (MoE, --scale ls, 注意力张量量化)     PPL = 26.52
attnf16b (MoE, --scale ls, 注意力/SSM 保 F16)  PPL =  6.20    <- 改善 4.3 倍
```

把 `blk.N.(attn_q|attn_k|attn_v|attn_qkv|attn_gate|attn_output|ssm_out).weight`
保在 F16，模型从 26.52 恢复到 **6.20** —— 这是目前最好的一版。

### 34.2 它精确落在用户指的方向上

用户观察到：**"思维链前几句不重复、语义通顺，随后才开始循环"**。
那是**递推/注意力状态路径**漂移的特征 —— 实测确认问题就在那里。

**FFN 与 MoE 专家做三值是没问题的**；坏的是注意力/SSM 那批张量的量化。

（注意：对照里真品的 4.18 是**稠密 27B**，与 MoE 不是同一模型，不可直接比。）

### 34.3 两个操作教训（都是"看起来跑了、其实没生效"）

1. **`--keep-f16` 的正则里带反斜杠会被吃掉**，打包体积与不保 F16 时**完全相同**，
   而**不报错**。改用无反斜杠写法（`.` 代替 `\.`）后立刻生效（12.74 GB -> 14.57 GB）。
   **带反斜杠的参数一律写进脚本文件再执行。**
2. 同类：把「编译 + 推送 + 启动」写进一条命令导致推送漏执行（见 §31.4）。

### 34.4 同口径对照

```
真品 dense 27B                       4.18
我的 dense 27B  --scale ref        1290.0
我的 dense 27B  --scale ls          待测
我的 MoE       --scale ls            26.52
我的 MoE       --scale ls + 保F16     6.20    <- 最好
```

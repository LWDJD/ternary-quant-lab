# IQ 查找表 LDS 优化：完成情况、数据与代码交接

> **交付（先看这里）**
> 已推送到 `https://github.com/LWDJD/PrismML-Eng-llama.cpp`
>
> | 分支 | 基点 | 内容 |
> |---|---|---|
> | **`vulkan-iq2s-lds-opt`** | `adfffbe` | **合并这个**。3 提交 / 2 文件 / +83−10 行，MoE **+26.2%** |
> | `wip-gdn-rows-vk` | 同上 | **别合并**。GDN 实验留档，含一个禁用的别名竞态路径（不可达） |
>
> PR：https://github.com/LWDJD/PrismML-Eng-llama.cpp/pull/new/vulkan-iq2s-lds-opt
>
> 合并前请自行复验：PPL 应逐位等于 `4.5593 ± 0.25522`；性能必须**同会话交错 A/B/A**（跨会话漂移 ±21%）。

最后更新：2026-09-26
目标：解决单流 MoE 解码「带宽利用率只有约 1/3」的问题。
假设：`shared uvec2 iq2s_grid[1024]`（8 KB/workgroup，与 BLOCK_SIZE 无关）限制常驻 workgroup 数。

---

## 0. 【最新】2026-09-26 第二轮成果

在方案 a（LDS 移出）之上再拿一个调优，**累计 +26%**：

| 模型 | 原始 `adfffbe` | **修复+调优** | Δ |
|---|---:|---:|---:|
| **MoE (IQ2_M)** | 32.86 / 33.22 | **41.62** | **+26.2%** |
| Bonsai PQ2_0 | 11.98 / 11.68 | 11.51 | 漂移范围内 |
| Bonsai PTQ1_0 | 13.69 / 13.63 | 13.63 | 中性 |

（同会话交错 A/B/A，`-p 0 -n 32 -r 3`）

**正确性：PPL 逐位一致 `4.5593 ± 0.25522`（三次）；3 prompt × 256 token 生成内容 SHA256 **全同**。**

### 新发现：`rm_iq` 的最优值从 4 移到了 2

删掉 8 KB LDS 后，**寄存器/常驻的取舍全变了**，原来基于“LDS 是瓶颈”调出来的
`rm_iq = 2*rm_kq (=4)` 不再最优。实测 `rm_iq=2` 更好：

| `rm_iq` | tg32 |
|---|---:|
| 4 | 30.82 / 31.41 / 31.42 → 均值 31.22 |
| **2** | **32.16 / 33.15 → 均值 32.66** |

**A/B/A/B/A，`rm_iq=2` 的每一次都高于 `rm_iq=4` 的每一次 → +4.6%。**

逐 op 归因（`GGML_VK_PERF_LOGGER`）证实收益 **100% 在 IQ2_S 路径**：

| op | rm_iq=4 | rm_iq=2 | Δ |
|---|---:|---:|---:|
| `MUL_MAT_ID_VEC iq2_s m=512 n=8 k=2048` | 46.74 | 40.00 | −14.4% |
| `iq2_s m=2048 k=4096` | 45.61 | 39.20 | −14.1% |
| `iq2_s m=4096 k=2048` | 43.52 | 38.15 | −12.3% |
| `iq2_s m=8192 k=2048` | 78.10 | 68.70 | −12.0% |
| `iq2_s m=2048 k=512` | 12.38 | 9.63 | −22.2% |
| `iq2_s m=32 k=2048` | 3.74 | 2.62 | −30.0% |
| **`iq3_s`（用 `rm_kq`，不受影响）** | **25.12** | **24.88** | **−1.0%** ← 天然对照组 |

累计省 1.28 ms/token，对 ~32 ms 的 token 正好就是 +4%。

### 同时也把两个调参开关移到了 `prism-tip`

`GGML_VK_DMMV_NUM_ROWS` / `GGML_VK_DMMV_SUBGROUP_SIZE`（带 `fprintf` 回退，因为 `GGML_LOG_INFO`
在 backend 初始化早期不打印）。设备能力：`subgroup_size_control: 1, min 32, max 64`。

顺带重扫了一遍几何旋钮（修复后的版本）：

| 配置 | tg32 |
|---|---:|
| 默认 (rows=4, sg=64) | 31.48 / 30.97 |
| **rows=2** | **32.64** |
| rows=1 | 29.75 |
| rows=8 | 24.89 |
| rows=16 | 26.41 |
| sg=32 (rows=4) | 30.50 |

**`NUM_ROWS=1` 从旧树的 −40% 变成现在的 −3%** —— 佐证了“这个旋钮的重要性本来是 LDS 常驻压制的副产品”。
`sg=32` 依旧不如默认（与旧树一致）。

### 当前 git 与产物

```
prism                    adfffbe                      上游原样
 exp/iq-lds-tip            3946e88   LDS 修复
* exp/iq-lds-variant-b     da08f13   LDS 修复 + 调参开关 + rm_iq=2 固化   ← 最新
```

- 二进制：`bin\tip-baseline`（原始）、`bin\tip-variant-c`（**最佳**）、`bin\tip-variant-b`（修复+dials）
- patch：`patches\tip-variant-c-cumulative.diff`（累计，7458 B）、`patches\series\000{1,2,3}-*.patch`

---

## 0.5 本轮又试掉的两条路（均为阴性，已回退/排除）

### ❌ 手工展开 `l` 循环（缩短依赖链）—— 实测中性，已回退

`mul_mat_vec_iq2_s.comp` 里的 `[[unroll]] for (l = 0; l < 2; ++l)` 在 IR 层面上**真的没被展开**
（glslang 不为 `[[unroll]]` 展开）。所以手写成“两个查表都提到 FMA 之前”的直线代码：

| | OpLabel | OpLoopMerge | OpLoad |
|---|---:|---:|---:|
| 循环形式 | 110 | 13 | 204 |
| **手工展开** | **129** | **12** | **262** |

IR 结构确实变了（`l` 循环消失）。但实测：

| | C（循环形式） | D（手工展开） | C 重测 |
|---|---:|---:|---:|
| e2e tg32 | 41.24 / 41.70 / 41.67 | 41.88 / 41.70 | — |
| VGPR | 69 | **69（完全相同）** | — |
| `iq2_s m=4096 k=2048` | 1147.5 | 1130.0 | 1158.3 |
| `iq2_s m=2048 k=512` | 396.0 | **433.0（更慢）** | 372.6 |
| `iq2_s m=8192 k=2048` | 689.1 | 701.6 | 703.0 |

**逐 op 与 e2e 都在噪声内。驱动调度器本来就已经把两个查表排开了，手写展开没拿到任何东西。**
已 `git checkout` 回退——不交付无收益的复杂度。

**顺手得到的两个数据**：
- **VGPR 从基线的 88 降到了 69**（LDS 表移出后不再需要寄存器暂存 grid 值）
- `tmpsh`（归约共享缓冲）仍在 LDS，但只有 512 字节量级，不构成常驻瓶颈

### 🔍 那 8 KB 到底去哪了（任务 9，已查明）

反汇编当前构建的 `mul_mat_vec_iq2_s_f32_f32` SPIR-V（`spirv-dis`）：

```
Function__arr_v2uint_uint_1024   x4       ← 变成 Function 存储类，不再是 Workgroup
Workgroup__arr__arr__arr_float_135_21_23  ← tmpsh 仍在 LDS
OpConstantComposite count = 1025          ← 表被物化成编译期常量
```

**结论**：表既不在 LDS（usage 0 ✓）也不在 scratch（0 ✓），而是被提升为**编译期常量 /
Function 作用域**。`Function__..._1024 x4` 说明编译器把它复制了 4 份（这是 .spv 变大的原因）。

**遗留疑问**：Function 存储类 + 动态索引的数组，在 LLPC 里通常会被降级——但 scratch 为 0。
如果它最终落到**全局内存**（而非寄存器/常量缓存），那么每次查表都是一次全局访存，
这正好能解释 IQ2_S 为何仍封顶在 ~45 GB/s。**方案 c（压到 2 KB 放回 LDS）因此值得重估**：
它的胜机不再是“LDS 带宽”，而是“**LDS 延迟 < 全局延迟**”。

---

## 0.6 【结论】那两条路线的最终裁定：都是阴性，而且方向反了

原始假设是“IQ 反量化卡在依赖链延迟，应该缩短链 / 加更多并行链”。
**两条都实测过了，结论相反：这个 kernel 是常驻（occupancy）受限，不是链延迟受限。**

### 路线一：缩短依赖链　→ 中性

把 `l` 循环手写成直线代码，两个查表提到 FMA 之前。IR 确实变了
（`OpLoopMerge` 13→12，`OpLabel` 110→129），但：

- e2e 与逐 op 均在噪声内
- **VGPR 69 → 69（零变化）** ← 关键：没有触动寄存器，也就没有可改善的机制

已回退。

### 路线二：每线程更多独立链　→ 一致地变差

**形式 1：`NUM_ROWS` 拨盘**（这个拨盘本身就是“每线程几条链”的实验，1=最少…16=最多）

| `rm_iq` | tg32 |
|---|---:|
| 1 | 29.75 |
| **2（新最优）** | **32.66** |
| 4（旧默认） | 31.22 |
| 8 | 24.89 |
| 16 | 26.41 |

**最优值从 4 挪到了 2——即“要更少、而不是更多”。** 这是 +4.6%，已固化。

**形式 2：手写 `i` 循环 2 路软件流水**（把相邻两个超级块的 10 个加载全提到计算之前）

IR 改动很大（`OpLabel` 110→156，`OpLoad` 204→324），编译通过。实测：

| 段落 | C（循环） | E（流水） | E 相对 C |
|---|---:|---:|---:|
| 1 | 32.80 | 31.17 | **−5.0%** |
| 2 | 41.16 | 38.82 | **−5.7%** |

（会话中途从 32.8 漂到 41.2，但两段内的**配对比例一致**，所以结论成立。）

**机制：VGPR 69 → 85。** 提前的 10 个加载抬高寄存器占用 16 个 → 常驻 wave 减少 → 净损 5%。
与 `NUM_ROWS=8`（−16%）是同一个故事。已回退。

### 总结论

| 在努力方向 | 实测 |
|---|---|
| 缩短依赖链 | 中性（VGPR 不变） |
| 增加并行链（拨盘） | **更多链更差**；最优值从 4 降到 2 |
| 增加并行链（手写流水） | **−5%（VGPR +16）** |
| **释放资源（LDS 8 KB → 0）** | **+16%（VGPR 88 → 69）** |

**真正有效的杆杆是“释放资源抬高常驻”，不是“加并行度”。** 两条路线至此均已实验关闭。

> 这个结论会给方案 c 泼冷水：把 2 KB 表放回 LDS 是在**增加**资源占用，
> 按同一逻辑可能得不偿失。除非那 8 KB 当前落在全局内存、而查表延迟真的构成了瓶颈。

---

## 0.7 为什么内存用不起来 —— 结构性诊断与下一步方向

### 硬事实

| 项 | 值 |
|---|---|
| 词表 | **248320**（家族级：`qwen35moe` 的 `output.weight` [2048,248320] 与 `qwen35` 的 [5120,248320] 一致） |
| lm_head (`output.weight`, q5_k) | **350 MB，1 次调用，每 token 必过** |
| 每 token 必需流量 | ≈ **1.05 GB** |
| 实测有效带宽 | 41.6 t/s 时 ≈ 43.7 GB/s（峰值 97 的 45%） |
| 模型张量类型分布 | `iq2_s` 375 / `f32` 301 / `q4_k` 40 / `iq3_s` 16 / `q5_k` 1 |

**lm_head 独自占每 token 总流量的 33%，而且它以接近峰值的带宽跑完（17% 时间）。
它不低效，它就是字节数大。** 每生成一个 token 就必须把这 350 MB 过一遍。

→ **单流下 45% 就是结构上限。提高有效利用率的唯一办法是“让同一次搬运产出更多 token”。**

### 测量本身不稳定（先要解决这个，否则一切对比都在流沙上）

同一个二进制、同一会话、8 连跑（`-p 0 -n 32 -r 3`）：

| run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tg32 | 32.54 | 32.49 | 32.70 | 32.73 | 32.66 | 32.47 | 30.76 | 30.94 |

前 6 次稳定在 32.5~32.7，后 2 次掉到 30.8 且 σ 变大。而当天早些时候同一个二进制给出过 **41.2**。
**至少存在 30.8 / 32.6 / 41.2 三个状态（跨度 34%）**，高度疑似热/功耗状态。
→ 所有“带宽利用率”数字自带 ±25% 不确定度。

### 已验证可行的路径：推测解码

| 事实 | 状态 |
|---|---|
| 同族词表一致（248320） | ✅ 已核对两个模型 |
| 库里有实现 | ✅ `llama-common.dll` 内含 `--model-draft` / `--draft-max` / `--draft-min` / `--draft-p-min` / `--lookup-cache-static` / `--lookup-cache-dynamic` |
| 能量测 | ✅ `llama-bench -md <bogus>` 报 `invalid parameter`，说明真实解析并校验路径 |
| `llama-completion` 能直接用于对话 | ❌ 报 `invalid argument: --model-draft`（而 `-ctkd` 却能接受）——需小改 CLI 参数表或启用 server |
| 不需要草稿模型的 n-gram 路径 | ⚠️ `--lookup-cache-*` 存在，但对 `-p 0` 纯生成无收益，只对长 prompt 含重复 n-gram 的场景有效 |

**预期收益**：草稿接受 2 token/步 → 同带宽产出 2 倍 token → 有效带宽 43 → 86 GB/s，约 **+70%**。

### 其余方向（按收益排序）

| # | 方向 | 依据 | 收益 |
|---|---|---|---|
| 3 | SSM 状态搬运 | `CPY`+`GET_ROWS` 各 30×2.1 MB = **126 MB/token（12% 流量）**，纯搬运算子 | 最多 +8% |
| 4 | K=512 专家 down 半组空转 | `blocks_per_wg=4 > num_blocks_per_row=2` → 一半 `ix` 槽无活；35 GB/s vs gate/up 的 59 | ~+4% |
| 5 | router 走 f32 | 84 MB/token（8% 流量） | ~+3%（模型转换层面） |
| — | `rm_kq` / `rm_stdq` 单独扫 | 覆盖 q4_K+q5_K+router ≈ 27% 时间 | 已间接验证无效（`iq3_s` 同用 `rm_kq` 且完全不变）——**排除了** |
| — | 方案 c（2 KB 表放回 LDS） | 与常驻受限结论**相反**（增加资源占用） | 不建议 |

---

## 0.8 【根因】内存带宽未被利用的真正原因（三个硬结论）

### 结论一：DRAM 不是瓶项，也不是变量

同一次会话里，`bw.exe`（自建 Vulkan 纯读微基准）与 `llama-bench` 背靠背交替：

| round | 原始读带宽 | llama tg32 |
|---|---:|---:|
| 1 | 96.9 GB/s | 39.02 |
| 2 | **104.2 GB/s** | **30.90**（σ=3.87） |
| 3 | 98.4 | 37.32 |
| 4 | 100.2 | 38.92 |
| 5 | 99.1 | 39.01 |

**DRAM 稳定在 96.9~104.2 GB/s，纹丝不动；llama 在同一时刻摆动 −21%。**
最慢的 round 2 反而读到最高的 104.2，而且它的 σ 是其他轮的 **10 倍**。
→ **那个 30.8/32.6/41.2 的跳变是干扰，不是降频、不是热、不是功耗。**
→ 而且 **~100 GB/s 的峰值比之前记录在案的 97 更高、更稳**。

**所以“内存用不起来”不是内存供不上，是 GPU 发不出足够的请求。**

### 结论二：K=512 的空转槽确实在浪费 wave 名额（实测 −28%）

`blocks_per_wg = workgroup/16 = 4`，而 K=512 → `num_blocks_per_row = 2` → **4 个槽有 2 个彻底没活**。
把 subgroup 设成 32（`blocks_per_wg` 变 2）就能消除空转：

| op | sg=64（4 槽） | sg=32（2 槽） | Δ |
|---|---:|---:|---:|
| `iq2_s m=2048 n=8 k=512`（专家 down） | 58.86 / 58.38 | **42.28 / 42.04** | **−28%** |
| `iq2_s m=2048 k=512` | 9.47 / 9.19 | **7.33 / 7.22** | **−22%** |
| `iq3_s m=2048 n=8 k=512` | 120.48 | **157.28** | **+30%** ✗ |
| 对照 `k=4096`（nbpr=16，无空转） | 39.66 / 39.46 | 40.15 / 39.87 | +1% ✓ |
| 对照 `k=2048 m=8192`（nbpr=8，无空转） | 71.90 / 68.44 | 70.64 / 68.46 | ~0 ✓ |

**但算总账只值 ~+2%**（K=512 的 op 占 11.6%，且 iq3_s 那项要倒退 30%）——不值得为此冒险重构。

### 结论三：SSM 那 126 MB/token 有名字、有位置、是 fork 自己写下的 TODO

每 token 的 `CPY`（1482 µs）+ `GET_ROWS`（1365 µs）= **2847 µs = 8.9%**，两个都是 2.1 MB 的纯搬运。
源头不是内存模块，是**图构建**：GDN 的“免 gather”rows 模式（src[6]）**只在 CPU 和 Metal 上实现**。

```cpp
// src/models/qwen35.cpp:476
// rows mode (the src[6] variant) is implemented on CPU and Metal only;
// other GPU backends reject it in supports_op, which would silently move
// the whole recurrent op to CPU -- keep the gathered form unless every
// GPU device in the model is Metal.
static const bool gdn_state_rows_env = getenv("GGML_GDN_STATE_GATHER") == nullptr;
```

```cpp
// ggml/src/ggml-vulkan/ggml-vulkan.cpp:18724
case GGML_OP_GATED_DELTA_NET:
    {
        // rows-indexed state read (src[6]) not implemented on Vulkan yet
        if (op->src[6] != nullptr) {
            return false;
        }
```

```cpp
// src/models/models.h:82
// state_rows (optional, ring path only): when set, `s` is instead the 2D
// cache view from build_rs_cache_view and the fused op reads each seq's
// live state directly at cache row state_rows[seq] (inp->s_copy_main) --
// no gathered scratch; the snapshot write becomes a SET_ROWS the Metal
// backend can fold into the fused op's epilogue.
```

**预期收益**：实现 rows 模式后 `GET_ROWS` 消失（**−1365 µs = −4.3%**）；
回写估计仍需一个 `SET_ROWS`（同样 2 MB，成本相当），所以不是完整的 −8.9%。
**改动量适中**：主要是 shader 里状态寻址改为“缓存行偏移”，加上 `supports_op` 的门。

---

## 0.9 GDN 状态搬运那条路：走不通，但原因值得记

目标：干掉每 token 的 `GET_ROWS`（1248 µs）+ `CPY`（1297 µs）= **2847 µs = 8.9%**。
（都是 2.1 MB/层 × 30 层的纯搬运，源在 `build_recurrent_attn` 的 `!keep` 分支。）

### 步骤 1：移植 fork 的 rows 模式（src[6]）到 Vulkan　——做完了，但它没用

fork 的“免 gather”rows 模式**只在 CPU 和 Metal 上实现**（`ggml-vulkan.cpp:18724` 显式
`return false`，注释写着 “not implemented on Vulkan yet”）。我把它实现了：

- shader：`binding 7` 装每序列的缓存行号，`state_in_base` 从 `seq_id*H` 改为 `rows[seq]*H`
- backend：push constant `state_rows` 模式位 + 8 个 binding + 放开 `supports_op`
- 图：把 dense `qwen35.cpp` 的 rows 管线移植到 `qwen35moe.cpp`

**但一测就没生效**，而且查出了原因：

```cpp
// common.cpp:1721
cparams.n_rs_seq = params.speculative.need_n_rs_seq();   // ← 只在推测解码时非零
```

rows 模式被 `n_rs_seq > 0` 门控（断言原话：*“rows mode is a ring-path optimization”*），
**而那个 ring 只为推测解码的回滚窗口而存在**。你拒绝推测解码，所以这条路对你无意义。
**我当初的 −4.3% 估算前提就是错的。**

### 步骤 2：就地更新　——激活了，但算错（且不确定），已禁用

既然 `n_rs_seq == 0` 时那条 `!keep` 路径完全没被上游优化过，那就直接干它：
读缓存行 → 算 → **写回同一行**，`GET_ROWS` 和 `CPY` 双双消失。

**路径确实激活了**（profile 里两个 op 都消失、`GATED_DELTA_NET` 变成 attention-only 输出）✓

**但 PPL 不对，而且不确定：**

| 配置 | PPL |
|---|---|
| in-place 开 | **4.9931 / 5.3752**（两次不同！） |
| 同一构建，强制走旧路径 | **4.5593** ✓ |
| variant-c 基线 | 4.5593 ✓ |

修掉了一个真 bug（读的行 ≠ 写的行：`s_copy(i)` vs `rs_head + seq`，我当初假设相等，
`llama-graph.cpp:3550/3558` + `set_input_rs`），PPL 从 5.56 → 4.99，**但没修好**。

**“不确定”这个特征把性质暴露了：竞态。** 根因：

> **就地写的是缓存张量本身，而 `build_rs_cache_view` 也会对同一个张量做就地写**
> （`rs_zero` 清零 `ggml_scale_inplace` + 额外状态搬迁）。
> 旧路径和 fork 的 rows 模式**都把新状态写进算子自己的输出**，所以从不与缓存别名。
> **我是第一个往缓存里写的，于是凭空造出一个别名竞态。**

**这不是一行能修的 bug——是“就地更新”这个想法撞上的墙。** 这也正好解释了为什么现有设计
把写回单独做成一个算子：**不是没想过，是想过并绕开了。** 图侧开关已退回（路径不可达）。

### 正确且配置无关的版本：只保留 rows 的*读*（还差 3.9%）

```
现在：  GET_ROWS(1248 µs) → GDN → CPY
改成：  rows 读(缓存行)   → GDN → SET_ROWS      ← 无别名，不需要推测解码
```

- 省掉 `GET_ROWS` = **−1248 µs ≈ −3.9%**（`CPY` → `SET_ROWS` 是等量交换）
- **mode 1 已经实现、编译通过、确认能激活**（提交 `5eddf02`），现在只是被 `n_rs_seq > 0` 挡住
- 要做的：让图在 `!keep` 分支用它（K=1）+ 补一个 `SET_ROWS`（`build_rs_write_rows` 已有）
- 代码在 `wip-gdn-rows-vk` 分支上

---

## 1. 状态总览

| # | 任务 | 状态 |
|---|---|---|
| 1 | 定位正确修订（旧快照废弃） | ✅ 完成 |
| 2 | 用正确修订编译并验证三个模型能跑 | ✅ 完成 |
| 3 | 方案 a：`iq2s` 查找表移出 LDS | ✅ **完成，+16% 已验证** |
| 4 | 正确性验证（PPL + 生成内容） | ⚠️ 旧快照上完成；**新修订未做** |
| 5 | `iq3s_grid` 同样处理 | ⏸ 旧快照上试过（无可测收益）；新修订未做 |
| 6 | 方案二：特化 trip count | ⚠️ **实测否定**（glslang 不为 `[[unroll]]` 展开，常量 trip count 也一样，见 §6） |
| 7 | 方案一：缩短反量化依赖链 | ✅ **已做，中性，已回退**（VGPR 零变化，见 §0.6） |
| 11 | **重扫几何旋钮（修复后）** | ✅ **完成：`rm_iq` 4→2，+4.6%，已固化** |
| 12 | 把调参开关移到 `prism-tip` | ✅ 完成（`DA08F13` 之前的 `9571112`） |
| 8 | 推广到其它 IQ 网格（`iq2xs` 4KB、`iq2xxs` 2KB、`iq1s_*`） | ❌ 待做 |
| 9 | 查明那 8 KB 从 LDS 挪走后去了哪 | ❌ 待做（未解疑点） |

---

## 2. 素材来源与 git 结构（重要）

### 2.1 两个仓库，只有一个是正确的

| 仓库 | 修订 | 构建号 | 结论 |
|---|---|---|---|
| `D:\Project\openhanako\workbench\prism-llama.cpp` | `e311ed3` | **b10660** | ⚠️ **旧快照，3 个提交的浅拷贝，已废弃** |
| `D:\Project\openhanako\workbench\prism-tip` | `adfffbe` | **b10743** | ✅ **正确修订，新工作全部基于它** |

**旧快照的致命问题**（实测）：

| 模型 | 旧快照 b10660 | 正确版 b10743 | 文档值 |
|---|---:|---:|---:|
| Bonsai PQ2_0 | **0.76 t/s** | 9.39 | 9.30 |
| Bonsai PTQ1_0 | **加载失败** | 10.73 | 10.56 |
| MoE | — | 26.54 | 26.50 |

**PTQ1_0 在 `e311ed3` 里根本不存在**（`ggml.h` 的 enum 是 `TQ1_0=34, Q1_0=41, Q2_0=42, PQ2_0=142, COUNT=143`；`git log -S PTQ1_0` 为空）。那 12 倍的 PQ2_0 差距和 PTQ1_0 加载失败，**全部源于版本过旧**，不是配置或代码问题。

### 2.2 远端参考修订

| tag | commit | 说明 |
|---|---|---|
| `prism-b10709-9a9394a` | `9a9394a` | 文档 BACKEND-SUPPORT 的审计基线 |
| `prism-b10735-842b188` | `842b188` | 文档数据与发布二进制的来源 |
| **`prism-b10743-adfffbe`** | **`adfffbe`** | **`prism` 分支当前 HEAD** |

拉取命令（**必须提权**，受限令牌下 TLS 不可用：`schannel: SEC_E_NO_CREDENTIALS`）：
```powershell
git clone --depth 1 --branch prism https://github.com/PrismML-Eng/llama.cpp.git prism-tip
```

### 2.3 本地 git 状态

```
prism-tip        (浅克隆, 161 MB)  分支 exp/iq-lds-tip
  3946e88   vulkan: read iq2s grid from the constant table instead of LDS (frees 8KB/workgroup)
  adfffbe   x86: SSE2/SSSE3 vec_dot for PTQ1_0 and PQ2_0 (#248)      ← = prism-b10743

prism-llama.cpp  (旧快照, 已废弃)  分支 exp/iq-lds
  9e5cedb   variant a: read iq2s grid from const table directly (LDS 8192 -> 0)
  2481acc   bench: DMMV subgroup/NUM_ROWS env overrides + shape-annotated perf labels
  e311ed3   release: windows-cuda arm64 as dll-only companion
```

**注意**：`prism-tip` 是 `--depth 1` 浅克隆，只有 2 个提交（`adfffbe` + 我的改动）。

---

## 3. 已完成的代码改动

### 3.1 核心改动（唯一有效的那一处）

文件：`prism-tip/ggml/src/ggml-vulkan/vulkan-shaders/types.glsl`
位置：`#if defined(DATA_A_IQ2_S)` 块内（`iq2s_grid_const[1024]` 声明在 1287 行，本改动在约 1546 行）
提交：`3946e88`
patch：`iq-lds\patches\tip-variant-a.diff`（2798 字节）

**改前：**
```glsl
shared uvec2 iq2s_grid[1024];

#define NEEDS_INIT_IQ_SHMEM
void init_iq_shmem(uvec3 wgsize)
{
    // copy the table into shared memory and sync
    [[unroll]] for (uint i = 0; i < iq2s_grid.length(); i += wgsize.x) {
        if (iq2s_grid.length() % wgsize.x == 0 || i + gl_LocalInvocationIndex.x < iq2s_grid_const.length()) {
            iq2s_grid[i + gl_LocalInvocationIndex.x] = iq2s_grid_const[i + gl_LocalInvocationIndex.x];
        }
    }
    barrier();
}
```

**改后：**
```glsl
// ==== exp/iq-lds-tip: grid out of LDS ====
#define iq2s_grid iq2s_grid_const

#define NEEDS_INIT_IQ_SHMEM
void init_iq_shmem(uvec3 wgsize)
{
    // grid read straight from the constant table; nothing to copy into LDS
}
// ==== end ====
```

**效果（`GGML_VK_PIPELINE_STATS=iq2`）：**

| pipeline | LDS | scratch | VGPR |
|---|---:|---:|---:|
| `mul_mat_vec_iq2_s_f32_f32` | 8192 → **0** | 0 | 88 |
| `mul_mat_vec_id_iq2_s_f32` | 8192 → **0** | 0 | 88 |

**`scratchMemUsageInBytes` 仍是 0** —— glslc 没有把 8 KB 常量数组降级成 Private 内存（原预测错了）。**那 8 KB 到底去了哪里仍未查明**（不在 LDS、不在 scratch、VGPR 也没变）。

> **编译坑**：不要在空函数里写 `(void) wgsize;`。GLSL 不支持这种 C 风格转换，会报
> `error: 'explicit typecast' : required extension not requested: GL_NV_explicit_typecast`。直接不引用即可。

### 3.2 旧快照上的附带改动（已废弃，仅存档）

| 文件 | 内容 | patch |
|---|---|---|
| `prism-llama.cpp`…`types.glsl` | `iq2s` 同样处理 | `patches\variant-a.diff` |
| 同上 | `iq3s_grid` 同样处理 | `patches\iq3s-block.diff` |
| `ggml-vulkan.cpp` | 两个环境变量覆盖（`GGML_VK_DMMV_SUBGROUP_SIZE` / `GGML_VK_DMMV_NUM_ROWS`）+ perf logger 兜底分支加形状标注 | 见 `prism-llama.cpp` 提交 `2481acc` |

**这两个环境变量开关值得移植到 `prism-tip`** —— 它们是做几何扫描的唯一手段。

---

## 4. 全部实测数据

> ⚠️ **所有对比必须同会话交错 A/B/A。** 实测跨会话漂移可达 ~20%（同一 `-p 0 -r 3` 组合在不同会话得过 33.36 和 27.28）。
> 且**机器上的后台进程会污染测量**：`RadeonGPUProfiler`、`Taskmgr`、`HanaAgent`、`Cloudflare WARP`、`YoudaoDict`
> 会让 tg128 标准差从 ±0.03 恶化到 ±2。**看标准差判断数据是否可信。**

### 4.1 【结论性】正确修订上的 A/B/A（`-p 0 -n 32 -r 3`）

| 模型 | A1 基线 | **B 改动版** | A2 基线 | 判定 |
|---|---:|---:|---:|---|
| **MoE（IQ2_S/IQ3_S）** | 26.89 | **31.24** | 27.01 | ✅ **+16%，成立** |
| Bonsai PQ2_0 | 9.29 | 8.96 | 10.44 | ❌ 无可测变化 |
| Bonsai PTQ1_0 | 13.41 | 13.31 | 12.44 | ❌ 无可测变化 |

**Bonsai 两行是干净阴性对照**：PTQ1_0/PQ2_0 是三值格式，编译时不会进 `#if defined(DATA_A_IQ2_S)` 块，**代码上不可能受影响**。它们"无变化"正是预期，也证明 MoE 那 +16% 不是测量污染。

### 4.2 【已证伪】被漂移污染的早期数据

单独采样（非交错）曾得到 MoE 38.78 / PQ2_0 11.95 / PTQ1_0 13.79，即 +46% / +27% / +29%。
**交错 A/B/A 复测后全部证伪。** 教训：三个涨幅接近、且"不该受影响的模型也涨"，就是整机状态变了的信号。

### 4.3 旧快照上的几何扫描（结论：默认值全是最优）

`-p 0 -n 128 -r 5`：

| NUM_ROWS | tg128 |
|---:|---:|
| **4（默认）** | **33.51 ± 0.42** |
| 2 | 29.30 ± 0.11（−13%） |
| 1 | 20.07 ± 0.05（−40%） |

`-p 512 -n 128 -r 2`（同批 5 组，仅供参考）：

| 配置 | tg128 |
|---|---:|
| 基线（BLOCK_SIZE 64 / NUM_ROWS 4） | 27.72 |
| NUM_ROWS=8 | 23.40（−16%） |
| NUM_ROWS=16 | 26.49 |
| NUM_ROWS=2 | 26.02（±4.51，不稳） |
| **wave32（BLOCK_SIZE 32）** | 25.01 |

**wave32 的独立 A/B/A 复核**（`-p 0 -n 128 -r 3`）：27.28 → **24.98** → 27.72 → **−10%，可复现**。

**三个几何旋钮全部扫完，默认值均为最优：**

| 旋钮 | 测过的值 | 最优 |
|---|---|---|
| `BLOCK_SIZE`（= subgroup size） | 32 / 64 | **64** |
| `NUM_ROWS` | 1 / 2 / 4 / 8 / 16 | **4** |
| subgroup size | 32 / 64 | **64** |

原因：`BLOCK_SIZE` 直接等于 subgroup size，而常驻 workgroup 数由**每 workgroup 的 LDS** 决定（IQ2_S 的 8 KB 查找表与 `BLOCK_SIZE` 无关）。把 `BLOCK_SIZE` 减半等于在常驻数不变的前提下把在飞线程砍一半。

### 4.4 其它否证过的方向

| 方向 | 结果 |
|---|---|
| 减少 submit 次数（`GGML_VK_MAX_NODES_PER_SUBMIT` 100→1000→10000） | 27.75 / 27.75 / 27.82，**无效** |
| 上下文长度影响 decode | 交错 `-p 0/2048/0/2048` → 27.77 / 27.81 / 27.75 / 27.84，**无影响** |
| `iq3s_grid` 同样处理 | 39.84/36.35 vs 39.63/39.79，**无可测差异**（它只有 2 KB，IQ3_S 占比 ~3%） |

**那 181 次"小算子"的真相**（perf logger 补形状标注后）：
```
CPY      d=f32(524288) s0=f32(128,128,32)         30 x 48.2 µs = 1447 µs
GET_ROWS d=f32(524288) s0=f32(524288) s1=i32(1)   30 x 47.5 µs = 1426 µs
```
524288×4 = **2.1 MB/次**，30 次对应 30 个 SSM 层，实际流量 4.2 MB/48 µs = **87 GB/s，已贴上限**。它们不是固定开销，是在搬 GDN 递归状态。

### 4.5 逐 op 剖析（旧快照，稳态，总计 37.74 ms）

| 类别 | 耗时 | 占比 |
|---|---:|---:|
| 稠密 matvec（非专家） | ~16.2 ms | **43%** |
| **专家 matvec** | ~10.3 ms | **27%** |
| 小算子（实为 2 MB 状态搬运） | ~3.1 ms | 8% |
| SSM（GATED_DELTA_NET + CONV） | 2.2 ms | 5.7% |
| Norm / RoPE / 杂项 / 元素级 | ~5.5 ms | ~15% |
| **Flash Attention** | 0.25 ms | **0.6%** |

关键条目（单次 µs）：`MUL_MAT_ID_VEC iq2_s m=512 n=8 k=2048` 80×62.2；`MUL_MAT_ID_MUL iq2_s m=2048 n=8 k=512` 37×142.4；`MUL_MAT_VEC q5_K m=248320 k=2048` 1×2984；`q4_K m=8192` 30×124.2；`GATED_DELTA_NET` 30×71.1。

### 4.6 达成带宽：按 op 类别差 6 倍

| op 类别 | m | 格式 | grid LDS | 达成带宽 |
|---|---:|---|---:|---:|
| lm_head | 248320 | Q5_K | 0 | **~115 GB/s** |
| attn_qkv | 8192 | Q4_K | 0 | 76~94 GB/s |
| router | 256 | F32 | 0 | 67 GB/s |
| attn_gate / ssm_out | 4096 / 2048 | IQ2_S | 8 KB | ~45 GB/s |
| 专家 gate/up | 512×8 | IQ2_S | 8 KB | ~43 GB/s |
| **专家 down** | 2048×8 | IQ2_S | 8 KB | **~19~21 GB/s** |

`GGML_VK_PIPELINE_STATS` 硬证据：`mul_mat_vec_q4_k_q8_1_f32` 的 `ldsUsageSizeInBytes: 0`。

### 4.7 正确性验证（旧快照上完成，新修订待重做）

**PPL**（`llama-perplexity -c 512 -f corpus-small.txt`，语料 6191 字节，6 chunks）：

| 版本 | Final estimate |
|---|---|
| 基线 | **PPL = 4.5671 ± 0.25628** |
| 改动版 | **PPL = 4.5671 ± 0.25628** |
| 基线复跑 | PPL = 4.5671 ± 0.25628 |

**生成内容**（`llama-completion -p "Explain in detail why the sky is blue." -n 128 --temp 0 --seed 42 -no-cnv`）：

| 版本 | 遍次 | stdout SHA256 | 字节 |
|---|---|---|---|
| 基线 | 1 / 2 | `048E591D8149E59A` | 598 |
| 改动版 | 1 / 2 | `048E591D8149E59A` | 598 |
| 改动版+iq3s | 1 / 2 | `048E591D8149E59A` | 598 |

**六份输出逐字节一致。** 加上构造论证（两版读同一张表，只换访问路径），可判定**无内容差别**。

> 遗留：跨 3 prompt × 256 token 的加强版测试**未跑完**（工具调用被中断，只留下 1 个完整文件），不计入证据。

---

## 5. 结论

### 5.1 机制

删掉 8 KB LDS → 常驻 workgroup 数上升 → 在飞 wave 变多 → 独立访存链变多 → 延迟被盖住一部分。**不是"省了 LDS 带宽"，是"多塞了几条链"。**

### 5.2 瓶颈判定：算力与带宽都没打满，卡在依赖延迟

| 资源 | 用量 |
|---|---|
| 算力（4.45 T 标量指令/s） | **< 10%**（matvec 实测 100~370 GFLOPS，峰值 8.9 TFLOP/s） |
| 带宽（97 GB/s） | 整体 **34%**，IQ 路径 46%，专家 down 20% |
| **都没满 → 在等** | |

IQ 反量化的依赖链（每个 256 元素超级块走一遍）：
```
load(d) load(scales) load(qh) load(qs) load(sign)
  → 合成 qs → iq2s_grid[qs] 查表 → unpack8 → 套 sign → 8×FMA
```
5 个加载可并行，但之后逐步串行；而 `i` 循环的 trip count 是运行时值 → 无法流水 → 每块重新等一次。

**旁证**：IQ 路径在多个形状上都停在 **45~48 GB/s 平台**（m=2048/4096/8192 几乎一样），这不是"活儿不够多"，是结构性上限。

### 5.3 带宽利用率

每 token 权重量 ≈ **1.05 GB**（专家 322 MB + lm_head 350 MB + attn_qkv 94 MB + ssm_out 121 MB + attn_gate 81 MB + 其余 ~80 MB；粗估 ±15%）。

| | t/s | 有效带宽 | 占 97 GB/s | 占 128 GB/s |
|---|---:|---:|---:|---:|
| 基线 | 26.89 | 28.2 GB/s | 29% | 22% |
| **改动版** | **31.24** | **32.8 GB/s** | **34%** | 26% |
| 参照：批处理 npl 32 | 65.2 | 68.5 GB/s | 71% | 54% |

**矛盾待解**：微基准测的纯读上限是 97 GB/s，但 lm_head 实测 115 GB/s **超过上限** → 说明 97 可能测低了（1 GiB 单线程扫有 TLB/页故障损耗），真实上限更接近 128。分母本身不确定。

### 5.4 头号空间

| 部分 | 耗时 | 占比 |
|---|---:|---:|
| IQ matvec（610 MB @ 45 GB/s） | ~13.5 ms | 42% |
| 非 IQ matvec（495 MB @ 90 GB/s） | ~5.5 ms | 17% |
| SSM / norm / 小算子 / 状态搬运 | ~13 ms | 41% |

**若 IQ 路径追平 Q4_K 的 ~90 GB/s**：13.5 → 6.8 ms，token 32 → 25.3 ms，**再拿 +26%**。

---

## 6. 待做任务（按优先级，含实现规格）

### 任务 6【优先】方案二：特化 trip count，让每线程展开更多独立链

**依据**：`mul_mat_vec_iq2_s.comp` 第 70 行
```glsl
const uint num_blocks_per_row = p.ncols / QUANT_K;   // 运行时值，QUANT_K=256
[[unroll]] for (uint i = ix; i < num_blocks_per_row; i += blocks_per_wg)
    calc_superblock(a_offset, b_offset, itid, i, num_blocks_per_row, first_row, num_rows);
```
`[[unroll]]` 已经有（**不要重复加**），但 trip count 未知 → 驱动无法真正展开 → 无法软件流水。

**改法**：按 `num_blocks_per_row` 的特化值生成多个 shader 变体，覆盖本模型实际用到的三档：
- K=512 → 2（专家 down）
- K=2048 → 8（专家 gate/up、attn_gate）
- K=4096 → 16（ssm_out）
另加一个通用回落变体。

> ✅ **2026-09-26 补充：这套机制代码库里已经现成，不用发明。**
> `ggml-vulkan.cpp:5261`
> ```cpp
> const uint32_t wg_size_subgroup   = (w == DMMV_WG_SIZE_SUBGROUP) ? subgroup_size : (subgroup_size * 4);
> const uint32_t wg_size_subgroup16 = (w == DMMV_WG_SIZE_SUBGROUP) ? subgroup_size16 : (subgroup_size16 * 4);
> ```
> pipeline 创建时传的 push constant 是 `{wg_size_subgroup, 2*rm_stdq, i+1}`，**其中 `i+1` 就是 NUM_COLS**，
> host 侧用一个 `for (i = 0; i < 8; ++i)` 循环为 1..8 各编一份：
> ```cpp
> ggml_vk_create_pipeline(..., {wg_size_subgroup, 2*rm_stdq, i+1}, ...);
> ```
> 所以**照抄这个模式**：新增一个 `constant_id 3 = num_blocks_per_row`，把 pipeline 创建循环扩展成二维即可。
> 另有 `rm_iq_int = [](uint32_t i){ return i == 0 ? 8u : 4u; }`（`:5227`）——MUL_MAT_ID 用的是 `rm_iq` 值本身作变体。

**shader 侧**：`mul_mat_vec_iq2_s.comp` 里 `num_blocks_per_row` 来自 push constant，改成 spec constant 后，
`[[unroll]] for (i = ix; i < num_blocks_per_row; i += blocks_per_wg)` 的 trip count 就成为常量 → 驱动可真正展开并排流水。

**反汇编验证**（已打通，工具链齐全）：`spirv-dis` 在 `C:\VulkanSDK\1.4.357.0\Bin\`；
生成的 SPIR-V 以**逐字节小端**形式内嵌在 `build-vs\ggml\src\ggml-vulkan\mul_mat_vec_iq2_s.comp.cpp`，
用 `iq-lds\extract_spirv.py` 抠出（该文件含 9 个模块：`f32_f32` / `f16_f32` 的 `id_` / 非 `id_` × `subgroup` / `subgroup_no_shmem` 变体）。
基线 IR 里已确认：
```
%_arr_v2uint_uint_1024 = OpTypeArray %v2uint %uint_1024
%79 = OpVariable %_ptr_Workgroup__arr_v2uint_uint_1024 Workgroup   ← 8192 字节
```

**代价**：pipeline 变体 ×4，编译变慢、dll 变大。
**预期**：在途 load 从 `NUM_ROWS×1` 提到 `NUM_ROWS×2`。
**风险**：**纯编译期改动，无数值风险**（这点最关键）。

### ❌ 2026-09-26 实测否定：特化 trip count 在 glslang 层面无任何效果

本机单编 shader（`glslc` + `spirv-dis`，脚本 `iq-lds\probe_unroll.ps1`）对比三种版本：

| 版本 | 汇编字符数 | OpLabel | OpLoopMerge | OpLoad |
|---|---:|---:|---:|---:|
| 动态（现状，`p.ncols / QUANT_K`） | 144320 | **110** | 13 | 204 |
| **trip count 硬编码为 8** | 143895 | **110** | 13 | 201 |
| **trip count 硬编码为 2** | 143895 | **110** | 13 | 201 |

**基本块数、循环结构完全一致。glslang 不会为 `[[unroll]]` 展开循环，即使 trip count 是编译期常量。**

→ **任务 6 原形式作废。** 把 `num_blocks_per_row` 变成 spec constant 只可能靠**驱动后端**（AMD LLVM）去展开，
glslang 这一层拿不到任何收益，而 host 侧改动量很大（所有 `arr_dmmv_*_data[3]` 数组都要扩维）。
代价／收益比不成立。

**若要判定驱动层到底展不展开**：用 AMD 离线编译器（RadeonGPUAnalyzer，在 Radeon Developer Tool Suite 里）
对两个 `.spv` 出 ISA 对比。这是唯一能下结论的手段。

**替代形式（推荐）**：既然编译器不展开，就在**源码里手写软件流水** —— 把 `i` 循环改成每次处理 2 个 block，
**两次迭代的 10 个加载全部提到最前**，然后才做计算。这才是真正在生成“更多独立链”，且不依赖编译器。
须保持每个累加器的累加顺序不变（先 `i` 后 `i+blocks_per_wg`）→ 数值逐位不变。

**可复用的单编命令**（不用重编整个工程）：
```powershell
$G="C:\VulkanSDK\1.4.357.0\Bin\glslc.exe"
$S="D:\Project\openhanako\workbench\prism-tip\ggml\src\ggml-vulkan\vulkan-shaders"
& $G -fshader-stage=compute --target-env=vulkan1.2 -I $S `
  -DDATA_A_IQ2_S=1 -DB_TYPE=float -DB_TYPEV2=vec2 -DB_TYPEV4=vec4 -DD_TYPE=float `
  -DFLOAT_TYPE=float -DFLOAT_TYPEV2=vec2 -DACC_TYPE=float -DUSE_SUBGROUP_ADD=1 `
  $S\mul_mat_vec_iq2_s.comp -o out.spv
```

> 附：`USE_SUBGROUP_ADD_NO_SHMEM` 指的是**归约阶段**（用 `subgroupAdd` 代替 `tmpsh` 共享缓冲），
> **与查找表无关**。已确认我的改动与上游已有的三个变体（`""` / `_subgroup` / `_subgroup_no_shmem`）正交，无重复。

### 任务 7 方案一：缩短反量化依赖链

**改法**：把 `calc_superblock` 里 `l` 循环的两次查表都提到 FMA 之前——预先算好两个 grid 值，把 `2×(查表→FMA)` 的串行变成 `2 查表并行 → 16 FMA`。

**先验证编译器有没有已经这么排**：反汇编生成的 SPIR-V（`spirv-dis`）。可能已经做了，收益为零。

### 任务 8 推广到其它 IQ 网格

同一模式，每个块一行 `#define <name> <name>_const`。目标：`iq2xs_grid[512]`（4 KB）、`iq2xxs_grid[256]`（2 KB）、`iq1s_grid[2048]`（4 KB）、`iq1s_grid_gpu[2048]`（8 KB）、`iq3xxs_grid[256]`（1 KB）、`iq3s_grid[512]`（2 KB）。
注意 `iq1s_grid` 的 init 里有 `pack32(unpack16(...))`，结构不同，要单独处理。

### 任务 9 查明那 8 KB 去了哪

用 `spirv-dis` 反汇编 `build-vs` 里生成的 `mul_mat_vec_iq2_s*.comp.cpp` 内嵌的 SPIR-V，看常量表的落点（全局常量？立即数展开？）。**这是判断"改动能不能安全推广到其它硬件/驱动"的前提。**

### 任务 4 在新修订上重做正确性验证

用 4.7 的流程（PPL + 生成内容），在 `prism-tip` 的基线与改动版上各跑一遍。

### 任务 10 把两个环境变量开关移植到 `prism-tip`

`GGML_VK_DMMV_SUBGROUP_SIZE` / `GGML_VK_DMMV_NUM_ROWS`（见 `prism-llama.cpp` 提交 `2481acc`）。没有它们就无法再做几何扫描。

---

## 7. 复现命令

### 7.1 构建

```powershell
# 配置（必须用 VS 生成器；Ninja 在受限令牌下会卡死在 "Detecting C compiler ABI info"）
cmd /c "D:\Project\openhanako\workbench\vk-build-tip.bat configure-vs"

# 编译（必须在 tty 会话里跑；一次性沙箱命令会卡死在 "Checking File Globs"）
cmd /c "set MSBUILDDISABLENODEREUSE=1 && D:\Project\openhanako\workbench\vk-build-tip.bat build-vs"
```
产物：`prism-tip\build-vs\bin\Release\`（`llama-bench.exe` / `llama-perplexity.exe` / `llama-completion.exe` + 各 dll）

**坑**：`llama-cli` 目标不存在（这个 fork 改名成 `llama-completion`）；`llama-app` 需要 `llama-server-impl.lib`，而配置里 `LLAMA_BUILD_SERVER=OFF`。

### 7.2 性能 A/B（必须交错）

```powershell
$B="D:\Project\openhanako\workbench\iq-lds\bin\tip-baseline\llama-bench.exe"
$V="D:\Project\openhanako\workbench\iq-lds\bin\tip-variant-a\llama-bench.exe"
# 对每个模型：B → V → B，第二遍 B 必须回到第一遍水平，否则数据作废
cmd /c "`"$B`" -m <模型> -ngl 99 -p 0 -n 32 -r 3"
```
> PowerShell 坑：函数名不要用 `R`（是 `Invoke-History` 的内置别名）。

### 7.3 正确性验证

```powershell
# PPL（确定性！两版必须给出完全相同的数值）
llama-perplexity.exe -m <模型> -ngl 99 -c 512 -f iq-lds\corpus-small.txt

# 生成内容
llama-completion.exe -m <模型> -ngl 99 -p "Explain in detail why the sky is blue." `
  -n 128 --temp 0 --seed 42 -no-cnv > out.txt 2> err.txt
Get-FileHash out.txt -Algorithm SHA256
```

### 7.4 逐 op 剖析（比 RGP 便宜得多）

```powershell
$env:GGML_VK_PERF_LOGGER="1"; $env:GGML_VK_PIPELINE_STATS="iq2"
llama-bench.exe -m <模型> -ngl 99 -p 0 -n 6 -r 1
```
> **两个关键坑**：
> 1. **第一张表包含预热与着色器编译，数字失真 30~1000 倍**，必须看后续几张。
> 2. **`GGML_VK_PIPELINE_STATS` 是过滤器字符串，不是开关**。传 `"1"` 只会匹配到名字里含 `1` 的 pipeline（`q8_1`、`cm1`…）。正确用法传 `"iq2"`。

---

## 8. 方法论与踩过的坑

1. **跨会话漂移 ~20%**。任何对比必须同会话内交错 A/B/A；递增顺序的扫描会被漂移污染成假台阶（我曾据此误报"上下文影响 decode"，交错复测后证伪）。
2. **后台进程污染测量**。测前清场，或看标准差。
3. **阴性对照不可省**。Bonsai 不走 IQ2_S shader，它"无变化"才是 MoE 那个 +16% 可信的关键证据。
4. **先验证素材来源**。我一开始默认"本地 clone = 发布版同源"，结果在旧修订上做了全部实验——PTQ1_0 加载失败、PQ2_0 只有 0.76 t/s 全是版本问题。
5. **修改 shader 必须重验 PPL + 生成内容**。PPL 是确定性的，两版数值不同就说明计算不同。

---

## 9. 文件清单

```
D:\Project\openhanako\workbench\
├─ prism-tip\                     ✅ 正确修订 (b10743) + 已提交的改动
│   └─ build-vs\bin\Release\      编译产物
├─ prism-llama.cpp\               ⚠️ 旧快照 (b10660)，已废弃
├─ vk-build-tip.bat               prism-tip 的构建脚本
├─ vk-build.bat                   旧快照的构建脚本
├─ IQ-LDS-实验记录与交接.md        本文档
└─ iq-lds\
    ├─ CHANGELOG.md               逐条实验日志（含旧快照上的完整过程）
    ├─ analyze_grid.py            查找表值域分析（证明字节只有 3 种取值）
    ├─ corpus-small.txt           PPL 验证语料（6191 字节）
    ├─ patches\
    │   ├─ tip-variant-a.diff     ✅ 核心改动（新修订）
    │   ├─ variant-a.diff         旧快照的 iq2s 改动
    │   └─ iq3s-block.diff        旧快照的 iq3s 改动
    ├─ bin\
    │   ├─ tip-baseline\          ✅ 新修订基线二进制
    │   ├─ tip-variant-a\         ✅ 新修订改动版二进制
    │   ├─ baseline\              旧快照基线
    │   ├─ variant-a\             旧快照 iq2s 改动版
    │   └─ variant-aplus\         旧快照 iq2s+iq3s 改动版
    └─ logs\                      每次构建与剖析的原始输出
```

两份二进制都留着 → 随时可重做同会话 A/B/A。

---

## 10. 一句话

**删掉 IQ2_S 查找表的 8 KB LDS 拷贝，单流 MoE 从 26.89 涨到 31.24 t/s（+16%，交错 A/B/A 验证，PPL 与生成内容逐位一致）。根因不是带宽也不是算力，是反量化依赖链的延迟——常驻 workgroup 数一上去，在飞的独立访存链变多，延迟就被盖住一部分。下一步按同样思路做 trip count 特化（纯编译期、无风险），预期再拿 +26%。**

# ternary-quant-lab

Independent study of low-bit (ternary) post-training quantization for
hybrid-attention LLMs, using PrismML's `PTQ1_0` format as the reference target.

**Not affiliated with PrismML.** No model weights are included; the reference
artifact referenced below is a public release fetched separately.

## What this is

A working pipeline plus the experiment record behind it. It takes a HuggingFace
checkpoint and produces a rotated, ternary (`PTQ1_0`, 1.75 bpw) GGUF, reusing the
fork's own conversion code for tensor naming, transposes, MoE expert stacking and
the GDN v-head reorder.

## The reference's rule, as measured

Trit agreement with the reference is a pure function of the quantisation rule, so
the rule was recovered by scoring candidates against the reference's own trits
(no packing, no perplexity):

| | value | how |
|---|---|---|
| rotation block | **1024** | distinct from the 128-element quantisation block; mixing them up drops agreement to chance |
| scale | **d = 1.40 · mean\|x\|** | measured on three tensors: 1.398 / 1.397 / 1.411 |
| decision | **keep the 86 largest per 128-block** | per-block count has sd 0.35 vs binomial 5.31, so it is a rank rule, not a threshold |

The two knobs are orthogonal: agreement depends only on the product
`mean_mult * theta`, so the scale must be pinned separately from the boundary.

## Established, verified

- rotation convention matches the reference: 87.6% trit agreement vs 33.3% chance
- **sign agreement on the reference's own non-zero slots: 100.00%** — the fold is
  positionally exact
- encoder and decoder are exact inverses (400/400 blocks round-trip byte-for-byte)
- shipping the reference's sign vectors moves whole-model byte agreement from
  **0% to 63.0%** (predicted 0.9^5 = 59%), and **no tensor falls below 47.7%**
  against a 0.39% chance floor

## Refuted (measured, do not retry)

- the open-source encoder (`quantize_row_ptq1_0_ref`, `d = amax`) is not the
  production recipe: it scores **at chance** against PrismML's own released model
- `hadamard_packing.json` is read by the converter and produced by nothing in the
  repo — the exact shape of the withheld piece
- error feedback (GPTQ-style), sign-vector optimisation, density matching,
  mean-matching fixed point: all worse
- folding `ssm_out` + `gdn_v_grouped=true` (the reference's own GDN config) is
  **worse**, under both the v5 and v7 runtimes
- block 512, runtime version, and `token_embd` quantisation: all refuted

## Where it stands

The pack is uniformly ~91% trit-correct with no structurally broken tensor, and
still produces fluent but factually broken output. The residual sits entirely at
near-tie boundary values (100% agreement away from the threshold, 53.9% inside
the band), so closing it means either reproducing the reference's folded values
bit-for-bit or finding their tie-break rule.

Note on judging: several rounds were spent on mis-calibrated metrics. Two traps
worth recording — a 128-trit block never matches byte-for-byte at 90% trit
agreement (use bytewise rate, not block identity), and `rep6` is not a quality
signal (the reference itself scores 9 under greedy sampling).

## Layout

- `docs/` — full handoff documents and experiment records, including the
  reconstructed `PTQ1_0` contract and the refuted-hypothesis record
- `tools/` — the packer (`prism_pack_hf.py`), the Python port of the block
  quantizer (`ptq1_0.py`), the fold math (`prism_pack.py`), and the measurement
  harnesses (`ab_quant.py`, `scan_d.py`, `rule_test.py`, `theta_sweep.py`,
  `rank_test.py`, `amax_check.py`)
- `remote/` — scripts used to drive a remote build box, including the rule
  scorers (`rule_fit.py`, `find_theta.py`), the fold diagnostic
  (`edge_check.py`), and the whole-model comparison (`block_agree.py`)

## Quick start

```bash
python tools/prism_pack_hf.py <hf_dir> <out.gguf> \
    --block 1024 --jobs 8 --scale ref --theta 0.38 \
    --fallback-quant Q4_0 \
    --fork <llama.cpp fork root>
```

`--scale ref` is the measured rule. `--signs-from <dir>` loads `sign_<width>.npy`
instead of generating random ones, which is required to compare bytes against a
reference pack (a different sign vector gives an equally valid but incomparable
rotation).

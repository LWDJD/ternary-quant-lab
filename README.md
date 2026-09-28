# ternary-quant-lab

Independent study of low-bit (ternary) post-training quantization for
hybrid-attention LLMs, using PrismML's `PTQ1_0` format as the reference target.

**Not affiliated with PrismML.** No model weights are included; the reference
artifacts referenced here are public releases fetched separately.

## What this is

A working pipeline plus the experiment record behind it. The pipeline takes a
HuggingFace checkpoint and produces a rotated, ternary (`PTQ1_0`, 1.75 bpw) GGUF,
reusing the fork's own conversion code for tensor naming, transposes, MoE expert
stacking and the GDN v-head reorder.

## Established results

| Finding | Evidence |
|---|---|
| Rotation convention matches the reference | 87.6% trit agreement over 72 candidate conventions in parallel; chance is 33.3% |
| The reference's scale is **not** least-squares | it holds a constant 67.2% non-zero; the MSE optimum is ~55% |
| The reference's scale tracks `mean|x|` (cv 2.0%), not `max|x|` (cv 12.7%) | value-range protection: outliers are tolerated, not allowed to set the scale |
| Outlier-robust scale beats max-initialised least squares | MSE 0.949x, non-zero 55.2% vs 46.0% |
| Error feedback (GPTQ-style) is worse | 1.8-2.9x MSE |
| Optimising the sign vectors is a dead end | the reference's own sign vectors are statistically random |
| Protection by position has no room left | quantising `token_embd` for an untied model destroys it (0/2 facts); folding `ssm_out` made it worse (4/4 -> 0/4) |

Remaining gap to the reference: a ~1.29x constant factor plus the exact scale
rule, both inside their unpublished quantizer.

## Layout

- `docs/` — full handoff documents and experiment records, including the
  reconstructed `PTQ1_0` contract and the refuted hypotheses
- `tools/` — the packer (`prism_pack_hf.py`), the exact Python port of the
  block quantizer (`ptq1_0.py`), the fold math (`prism_pack.py`), and the
  measurement harnesses (`ab_quant.py`, `scan_d.py`, `rule_test.py`)
- `remote/` — scripts used to drive a remote build box

## Quick start

```bash
python tools/prism_pack_hf.py <hf_dir> <out.gguf> \
    --block 1024 --jobs 8 --scale robust \
    --fallback-quant Q4_0 \
    --fork <llama.cpp fork root>
```

`--scale ls` (default) is the max-initialised alternating least-squares fit;
`--scale robust` is the outlier-robust rule described above.

## Judging a result

Same flags on both models, `--jinja -st`, `--no-display-prompt`, greedy:

- factual probes at `-n 300` (a thinking preamble needs the room)
- repetition: `uniq` (distinct/total words) and `rep6` (max repeats of any 6-gram);
  a healthy model sits near `uniq 0.72 / rep6 1`

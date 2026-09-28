#!/usr/bin/env python3
"""
PrismML Hadamard packing, GGUF -> GGUF, single streaming pass.

    python pack_gguf.py <in.gguf> <out.gguf> [--block 512] [--dry-run] [--source f16]

For every foldable weight it does

    dequantize -> W' = W @ S @ H_norm (blockwise along ne[0]) -> PTQ1_0

and writes the result plus the `prism.hadamard.*` contract that
prism-tip/src/llama-model.cpp expects.  Non-foldable tensors are copied
verbatim.  No intermediate file, so peak memory is one tensor at a time.

Why the rotation is mandatory: the PTQ1_0 group scale is just max|w| over the
128 weights, so an unrotated group with one outlier collapses every other
weight to trit 0.  Measured: MiniCPM5-1B from a clean F16 source, and
Qwen3.6-35B-A3B from IQ2_M, both produce garbage with no rotation, while
PrismML's own rotated Bonsai 2 27B produces clean text on the same runtime.

v1 simplifications (deliberate, correctness first):
  * sign_mode = identity  (no sign table)
  * block = 512, not the whitepaper's 1024: the MoE expert down-projection has
    ne[0] = 512, and block_size is global, so 512 is the largest value that
    divides every foldable width (2048, 4096, 512).
  * ssm_out is not folded (keeps the source layout, avoids the GDN V-head
    permutation question).  Kept at source precision, ~0.09 GB.
  * token_embd is not folded either (would need the inverse-after-lookup role).
    Kept at source precision.
"""

import argparse
import hashlib
import os
import re
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prism-tip", "gguf-py"))

from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType   # noqa: E402
from gguf.quants import dequantize                                             # noqa: E402

import ptq1_0 as Q                                                            # noqa: E402
from prism_pack import fwht_norm                                              # noqa: E402

# same regex conversion/base.py enforces, and the same list llama-model.cpp checks
FOLDABLE = re.compile(
    r"output\.weight|"
    r"blk\.\d+\.("
    r"attn_q|attn_k|attn_v|attn_qkv|attn_gate|attn_output"
    r"|ffn_gate|ffn_up|ffn_down"
    r"|ffn_gate_exps|ffn_up_exps|ffn_down_exps|ffn_gate_up_exps"
    r"|ffn_gate_shexp|ffn_up_shexp|ffn_down_shexp"
    r"|ssm_out"
    r")\.weight"
)

# v1: not folded yet, copied at source precision
SKIP_FOLD = re.compile(r"blk\.\d+\.ssm_out\.weight")

# helper KV that must not be duplicated / must be recomputed.
# GGUFReader also materialises header fields as pseudo-entries named "GGUF.*";
# copying those produced a file with duplicate keys, which made the loader read
# the prism.hadamard block as absent (weights rotated, no activation transform).
DROP_KV = {"general.file_type", "general.architecture", "general.quantization_version"}
DROP_PREFIX = ("GGUF.",)


def foldable_names(reader, block, verbose=True):
    """Names to fold, after checking the block divides ne[0].

    t.shape is in ggml ne order (ne[0] first) -- that is what the runtime sees
    as weight->ne[0].  The numpy array gguf-py hands back is the transposed view,
    so the ne[0] axis is its LAST axis.  Both must stay consistent.
    """
    out, skipped = [], []
    for t in reader.tensors:
        if not FOLDABLE.fullmatch(t.name):
            continue
        ne0 = int(t.shape[0])
        if SKIP_FOLD.fullmatch(t.name):
            skipped.append((t.name, ne0, "v1: ssm_out not folded"))
            continue
        if ne0 % block:
            skipped.append((t.name, ne0, f"ne[0]={ne0} not divisible by block={block}"))
            continue
        out.append(t.name)
    if verbose:
        print(f"  foldable: {len(out)}   skipped: {len(skipped)}")
        for n, ne0, why in skipped:
            print(f"    skip {n:44s} ne0={ne0:<6d} {why}")
    return out


def dequant(t) -> np.ndarray:
    """Tensor -> float32 in ggml order (ne[0] fastest), shape preserved."""
    a = np.asarray(t.data)
    if t.tensor_type == GGMLQuantizationType.F32:
        return a.astype(np.float32)
    if t.tensor_type == GGMLQuantizationType.F16:
        return a.astype(np.float32)
    if t.tensor_type == GGMLQuantizationType.BF16:
        u = a.view(np.uint16).astype(np.uint32) << 16
        return u.view(np.float32)
    return dequantize(a, t.tensor_type).astype(np.float32)


def sign_vector(width: int, seed: int = 0x5EED) -> np.ndarray:
    """Deterministic +/-1 vector for a given width.

    Required, not cosmetic.  The Hadamard matrix's first row is all ones, so with
    identity signs the first rotated coordinate is n * mean(group); real weight
    groups have a nonzero mean, which makes that one coordinate a massive outlier
    that then dominates the group scale and crushes the other n-1 weights to zero.
    Random +/-1 removes exactly that, and matches PrismML's own models, which ship
    sign_mode = explicit.

    One vector per width, shared by every tensor of that width: the runtime keys
    its sign tensors by width, and tensors reading the same activation must apply
    the same transform or the graph's transform memo would be wrong.
    """
    h = hashlib.sha256(f"{seed:#x}:{width}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    return rng.choice([-1.0, 1.0], size=width).astype(np.float32)


def fold_only(x_flat: np.ndarray, ne0: int, block: int, signs: np.ndarray) -> np.ndarray:
    """x_flat is float32 in ggml order -> rotated float32.

    Matches the runtime exactly: activation <- H_norm @ (signs * activation), so the
    stored weight must satisfy  W' @ (H_norm @ (s * x)) == W @ x  =>  W' = W @ S @ H_norm.
    """
    assert ne0 % block == 0, (ne0, block)
    v = (x_flat.reshape(-1, ne0) * signs).reshape(-1, block)
    return fwht_norm(v).reshape(-1)


def fold_and_pack(x_flat: np.ndarray, ne0: int, block: int, signs: np.ndarray) -> bytes:
    """x_flat is float32 in ggml order. Signed-Hadamard fold then PTQ1_0."""
    return Q.quantize_row_ptq1_0(fold_only(x_flat, ne0, block, signs))


def part_path(outdir: str, name: str) -> str:
    return os.path.join(outdir, "w%s.bin" % hashlib.sha1(name.encode()).hexdigest()[:16])


def _fold_chunk(job):
    """Worker: fold+quantize a slice of tensors, spooling each result to disk.

    Memory is bounded per tensor: the source stays a file-backed view and is
    dequantized a few hundred rows at a time, and only the PACKED payload is
    accumulated (about 20x smaller than f32).  The earlier version accumulated the
    folded f32 chunks and concatenated them, which made one 2 GB tensor peak at
    ~7.4 GB -- that churn, not bandwidth, was why it crawled and why threads lost.
    """
    src, names, block, sign_tab, no_quant, keep_re, outdir, iters, r = job

    from gguf import GGUFReader, GGMLQuantizationType
    from gguf.quants import dequantize
    import ptq1_0 as Q

    if r is None:
        r = GGUFReader(src)
    tb = {t.name: t for t in r.tensors}
    RAW = (GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.BF16)
    out = []
    for name in names:
        t = tb[name]
        ne = [int(v) for v in t.shape]
        ne0 = ne[0]
        view = np.asarray(t.data)
        view = view.reshape(-1, view.shape[-1])
        nrows = view.shape[0]
        # ~128 MiB of float32 working set per chunk
        rows_per_chunk = max(1, int((128 << 20) // max(1, ne0 * 4)))
        keep16 = no_quant or (keep_re is not None and re.fullmatch(keep_re, name))
        packed = bytearray()
        for c0 in range(0, nrows, rows_per_chunk):
            sub = view[c0:min(c0 + rows_per_chunk, nrows), :]
            if t.tensor_type in RAW:
                x = sub.astype(np.float32).reshape(-1)
            else:
                x = dequantize(sub, t.tensor_type).astype(np.float32).reshape(-1)
            xf = fold_only(x, ne0, block, sign_tab[ne0])
            del x
            if keep16:
                packed += xf.astype(np.float16).tobytes()
            else:
                packed += Q.quantize_row_ptq1_0(xf, iters)
            del xf
        path = part_path(outdir, name)
        with open(path, "wb") as f:
            f.write(packed)
        if keep16:
            out.append((name, path, "f16", tuple(reversed(ne)), 0))
        else:
            row_bytes = ne0 // 128 * 28
            dshape = [int(v) for v in t.data.shape]
            out.append((name, path, "ptq", tuple(dshape[:-1]) + (row_bytes,), len(packed)))
        del packed
    return out



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-quant", action="store_true",
                    help="write folded tensors as F16 instead of PTQ1_0; the result must be "
                         "mathematically identical to the source, which isolates the rotator "
                         "from the quantizer")
    ap.add_argument("--keep-f16", default=None,
                    help="regex of folded tensors to still fold but store as F16, to bisect "
                         "which tensor's ternarization breaks the model")
    ap.add_argument("--jobs", type=int, default=1,
                    help="threads for the fold+quantize stage; tensors are independent")
    ap.add_argument("--iters", type=int, default=3,
                    help="least-squares fit iterations per 128-group")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse cached per-tensor results in <dst>.parts (skips the fold)")
    args = ap.parse_args()

    if args.block & (args.block - 1):
        print("block must be a power of two")
        return 1

    print(f"[read] {args.src}")
    r = GGUFReader(args.src)
    arch = str(r.fields["general.architecture"].contents()) if "general.architecture" in r.fields else "?"
    print(f"  arch={arch}  tensors={len(r.tensors)}  kvs={len(r.fields)}")

    folded = foldable_names(r, args.block)

    # one sign vector per folded width; the runtime keys its sign tensors by width
    # and every tensor of that width must share it
    ne0_of = {t.name: int(t.shape[0]) for t in r.tensors}
    fold_widths = sorted({ne0_of[n] for n in folded})
    for wd in fold_widths:
        if wd % args.block:
            print(f"  ! sign width {wd} not a multiple of block {args.block}")
            return 1
    signs = {wd: sign_vector(wd) for wd in fold_widths}
    print(f"  folded widths: {fold_widths}")

    if args.dry_run:
        tot = have = 0
        for t in r.tensors:
            n = int(np.prod([int(v) for v in t.shape]))
            tot += n
            if t.name in folded:
                have += n * 28 / 128 / 4      # packed bytes as float-equivalent
        print(f"\n  dry run: would fold {len(folded)} tensors")
        print(f"  parameters {tot/1e9:.2f} B")
        return 0

    w = GGUFWriter(args.dst, arch)
    w.add_quantization_version(2)
    w.add_file_type(int(GGMLQuantizationType.PTQ1_0))   # placeholder, fixed below

    # ---- metadata: copy everything we do not recompute ----
    copied = 0
    written = set()
    for key, field in r.fields.items():
        if key in DROP_KV or key.startswith(DROP_PREFIX) or key in written:
            continue
        try:
            types = list(field.types)
        except Exception:
            types = []
        if not types:
            continue
        val = field.contents()
        vt = GGUFValueType(types[0])
        st = GGUFValueType(types[1]) if len(types) > 1 else None
        try:
            w.add_key_value(key, val, vt, st)
            written.add(key)
            copied += 1
        except Exception as e:
            print(f"  ! kv {key}: {e}")
    print(f"  copied {copied} kvs")

    # ---- the hadamard contract ----
    w.add_uint32("general.file_type", 1 if args.no_quant else 129)   # MOSTLY_F16 / MOSTLY_PTQ1_0
    w.add_uint32("prism.hadamard.version", 1)
    w.add_uint32("prism.hadamard.block_size", args.block)
    w.add_string("prism.hadamard.transform", "normalized-sylvester-walsh-hadamard")
    w.add_string("prism.hadamard.axis", "input-last-dimension")
    w.add_string("prism.hadamard.sign_mode", "explicit")
    w.add_array("prism.hadamard.sign_widths", [int(v) for v in fold_widths])
    w.add_array("prism.hadamard.sign_values",
                [int(v) for wd in fold_widths for v in signs[wd]])
    w.add_array("prism.hadamard.weight_names", folded)
    print(f"  prism.hadamard: block={args.block} sign_mode=explicit "
          f"folded={len(folded)} widths={fold_widths}")

    # ---- fold + quantize: parallel over tensors (independent, order-free) ----
    tmpdir = args.dst + ".parts"
    os.makedirs(tmpdir, exist_ok=True)
    t_start = time.time()
    payload = {}
    keep_of = {n: (args.no_quant or (args.keep_f16 is not None and re.fullmatch(args.keep_f16, n)))
               for n in folded}
    if args.reuse:
        for n in folded:
            p = part_path(tmpdir, n)
            if not os.path.exists(p):
                continue
            t = next(x for x in r.tensors if x.name == n)
            ne = [int(v) for v in t.shape]
            if keep_of[n]:
                payload[n] = (p, "f16", tuple(reversed(ne)), 0)
            else:
                rb = ne[0] // 128 * 28
                payload[n] = (p, "ptq", tuple(int(v) for v in t.data.shape[:-1]) + (rb,),
                              os.path.getsize(p))
        print(f"  reuse: {len(payload)}/{len(folded)} cached")
    todo = [n for n in folded if n not in payload]
    if args.jobs > 1 and len(todo) > 1:
        import concurrent.futures as cf
        chunks = [todo[i::args.jobs] for i in range(args.jobs)]
        jobs = [(args.src, c, args.block, signs, args.no_quant, args.keep_f16, tmpdir, args.iters, r)
                for c in chunks]
        print(f"  folding {len(todo)} tensors over {args.jobs} threads")
        # threads, not processes: numpy releases the GIL for elementwise work, and
        # process spawning is restricted in this sandbox (WinError 5).  They only
        # help now that the per-tensor allocation churn is gone -- with GB-scale
        # temporaries they measured slower than serial.
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            for k, res in enumerate(ex.map(_fold_chunk, jobs)):
                for nm, path, kind, shape, ln in res:
                    payload[nm] = (path, kind, shape, ln)
                print(f"  worker {k+1}/{len(jobs)} done ({len(payload)} total)")
    elif todo:
        print(f"  folding {len(todo)} tensors serially")
        for nm, path, kind, shape, ln in _fold_chunk(
                (args.src, todo, args.block, signs, args.no_quant, args.keep_f16,
                 tmpdir, args.iters, r)):
            payload[nm] = (path, kind, shape, ln)
    print(f"  folded in {time.time() - t_start:.0f}s")

    # ---- tensors ----
    for i, t in enumerate(r.tensors):
        ne = [int(v) for v in t.shape]
        dshape = [int(v) for v in t.data.shape]
        if t.name in payload:
            path, kind, shape, ln = payload[t.name]
            if kind == "f16":
                arr = np.fromfile(path, dtype=np.float16).reshape(shape)
                w.add_tensor(t.name, arr, shape, GGMLQuantizationType.F16)
                tag = f"FOLD+F16 ne0={ne[0]}"
            else:
                arr = np.fromfile(path, dtype=np.uint8)
                w.add_tensor(t.name, arr, shape, GGMLQuantizationType.PTQ1_0)
                tag = f"FOLD+PTQ1_0 ne0={ne[0]} -> {ln} B"
        else:
            w.add_tensor(t.name, np.asarray(t.data), tuple(dshape), t.tensor_type)
            tag = f"copy {t.tensor_type.name}"
        if i < 3 or i % 100 == 0:
            print(f"  [{i+1}/{len(r.tensors)}] {t.name:44s} {tag}")

    print("[write]", args.dst)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("done:", f"{os.path.getsize(args.dst)/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

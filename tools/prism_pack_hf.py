#!/usr/bin/env python3
"""Pack a HuggingFace Safetensors checkpoint straight into a PrismML PTQ1_0 GGUF.

Why this exists
---------------
`pack_gguf.py` starts from an existing GGUF, so going from a full-precision
checkpoint to a rotated ternary model needs an intermediate GGUF of tens of GB.
This script instead drives the fork's own HF -> GGUF conversion
(`conversion/`) and intercepts the tensor stream right before it reaches the
GGUF writer.

Every piece of the hard part is therefore the fork's code, not ours: tensor
names, transposes, MoE expert stacking, and the GDN v-head reorder.  This file
only inserts the signed-Hadamard fold and the ternary quantization.

Orientation
-----------
`GGUFWriter.add_tensor` receives arrays in PyTorch orientation -- a Linear weight
arrives as [out, in] -- because the writer reverses the shape when it records
ggml `ne`.  So the *input* dimension is the LAST axis, which is exactly the
convention `prism_pack.fold_weight` already uses.  Verified against
`Qwen2MoeModel.modify_tensors`' own comment ("PyTorch (A,B,C) -> GGUF writes
[C,B,A]") and empirically on a tiny qwen3moe checkpoint: HF [32, 64] -> GGUF
ne=[64, 32].

Usage
-----
    python prism_pack_hf.py <hf_dir> <out.gguf> [--block 512] [--iters 3]
                            [--no-quant] [--jobs N] [--keep-f16 REGEX]

    --no-quant   fold but store F16 -- mathematically must reproduce the source,
                 so the result isolates the rotator from the quantizer
    --keep-f16   regex of tensors to fold but still store as F16

Requires torch (the conversion classes use it) plus numpy and pyyaml.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_FORK = HERE.parent / "prism-tip"

# Tensors that get folded (rotated along the input dim).
#
# `token_embd` is deliberately absent: it is untied here and cannot be folded.
# MATCHES THE REFERENCE exactly: Bonsai 2 folds 401 tensors and the set is
# identical -- including attn_gate (48) and output.weight (1).
#
# `ssm_out` IS on the fork's allowlist and the reference folds it, but folding it
# here is measured to BREAK the model.  Controlled A/B on the 35B MoE, identical
# harness, single variable:
#     no ssm_out fold, least-squares scale : facts 4/4, uniq 0.24, rep6  16
#     ssm_out folded,  least-squares scale : facts 0/4, uniq 0.15, rep6 100
# Both were tried and both also came out degenerate on the remote CPU build.
# So it is SKIP_FOLD'd below, and GDN_FOLD_SUFFIX is NOT applied to the converter
# -- skipping that reorder while NOT folding would break out_proj's column order.
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
# ssm_out stays at source precision (F16).  Needed if the fold is ever enabled
# on purpose; see GDN_FOLD_SUFFIX.
SKIP_FOLD = re.compile(r"blk\.\d+\.ssm_out\.weight")

# Quantize-but-do-NOT-rotate.  DISABLED BY DEFAULT -- measured, not assumed.
#
# The whitepaper lists embeddings among the quantized tensors, and the reference
# does store token_embd as PTQ1_0.  But the reference role for it is
# `inverse-after-lookup`, which requires a TIED output (schema 3).  These models are
# untied, and an untied embedding lookup never goes through a matmul, so the runtime
# applies no activation transform to that table.
#
# Measured on the 27B (block 1024, everything else identical): making token_embd
# plain unrotated ternary produced immediate end-of-text on every prompt -- 0/2 facts
# and a 3-word generation.  So the extra F16 protection is NECESSARY here, and the
# whitepaper's "embeddings are quantized" does not transfer to an untied model.
#
# Kept (matching nothing) so the mechanism is one line away if a tied model is ever
# packed, or if the fold-with-inverse path becomes available.
UNFOLDED_QUANT = re.compile(r"(?!)never")

# HF-side suffix of the GDN out projection.  LEFT DISABLED: applying it tells the
# converter to keep the grouped V order (and so to emit gdn_v_grouped), which is
# only correct if ssm_out is actually folded.  Since it is not folded, the
# converter must do its normal grouped->tiled column reorder.
GDN_FOLD_SUFFIX = ".__gdn_v_grouped_disabled__"

SEED = 0x5EED


def bootstrap(fork_root: Path) -> None:
    """Make the fork's `conversion` package and its vendored gguf-py importable."""
    for p in (str(fork_root), str(fork_root / "gguf-py"), str(HERE)):
        if p not in sys.path:
            sys.path.insert(0, p)


def sign_vector(width: int, seed: int = SEED) -> np.ndarray:
    """One deterministic random +/-1 vector per width.

    The runtime keys its sign tensors by width, so every tensor sharing a width
    must share the vector.  The signs matter: with an identity sign the first
    Hadamard row is all +1, which turns the group mean into a huge outlier that
    alone drives the group's scale.
    """
    import hashlib

    h = hashlib.sha256(f"{seed}:{width}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    return np.where(rng.integers(0, 2, size=width) == 0, -1.0, 1.0).astype(np.float32)


def main() -> int:
    # Without this the per-tensor INFO lines from conversion/ are dropped (the
    # default level is WARNING), leaving a 30-minute job with no visible progress.
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="directory containing config.json + *.safetensors")
    ap.add_argument("dst", help="output GGUF path")
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--iters", type=int, default=3, help="least-squares fit iterations")
    ap.add_argument("--no-quant", action="store_true",
                    help="fold but store F16; the result must equal the source exactly")
    ap.add_argument("--keep-f16", default=None,
                    help="regex of folded tensors to still fold but store as F16")
    ap.add_argument("--jobs", type=int, default=1,
                    help="threads used inside each tensor's fold+quantize "
                         "(the outer conversion loop is inherently serial)")
    ap.add_argument("--fork", default=str(DEFAULT_FORK),
                    help="llama.cpp fork root holding conversion/ and gguf-py/")
    ap.add_argument("--density", type=float, default=None, metavar="P",
                    help="fraction of trits to keep non-zero, e.g. 0.672 to match "
                         "PrismML's own packer.  Omit to use the least-squares scale, "
                         "which zeroes ~50.6%% -- about 17 points more than the reference.")
    ap.add_argument("--scale", default="ls", choices=("ls", "robust"),
                    help="ls = max-initialised alternating least squares (default).  "
                         "robust = outlier-robust: d = 1.37*mean|x| then refit.  The "
                         "reference's d tracks mean|x| (cv 2.0%%) not max|x| (cv 12.7%%), "
                         "and on real folded weights robust measures 0.949x MSE and "
                         "55.2%% non-zero vs 46.0%% -- better AND closer to its 67.2%%.")
    ap.add_argument("--fallback-quant", default=None, metavar="TYPE",
                    help="GGML type name (e.g. Q4_0) for tensors the fold set targets "
                         "but that cannot be folded at this block size (ffn_down_* at "
                         "block 1024).  Without it they become F16, which is huge.")
    ap.add_argument("--mtp", action="store_true",
                    help="also export the MTP/NextN draft block as an extra layer "
                         "(default: excluded, matching the 40-block reference GGUF)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.block & (args.block - 1):
        print("block must be a power of two")
        return 1
    if not args.no_quant and args.block % 128:
        # the PTQ1_0 packing group is 128 elements; a fold block that does not
        # align with it would mix groups across the boundary
        print(f"block must be a multiple of 128 for PTQ1_0 (got {args.block})")
        return 1

    bootstrap(Path(args.fork))

    import gguf
    from conversion import ModelBase, ModelType, get_model_architecture, get_model_class

    import ptq1_0 as Q
    from prism_pack import fold_weight

    dir_model = Path(args.src)
    print(f"[hf]  {dir_model}")

    hparams = ModelBase.load_hparams(dir_model, False)
    arch = get_model_architecture(hparams, ModelType.TEXT)
    cls = get_model_class(arch, mmproj=False)
    print(f"  arch={arch}  class={cls.__name__}")

    # The official converter exports the MTP / NextN speculative-draft block as an
    # extra layer (block 40 for this checkpoint, since the config says
    # mtp_num_hidden_layers = 1).  The reference target GGUF has exactly 40 blocks
    # and no nextn metadata, and the runtime path being validated does not use
    # speculative decoding -- so exclude it by default and keep the block layout
    # identical to the model we know loads.  Pass --mtp to keep it.
    if getattr(cls, "supports_mtp_export", False):
        cls.no_mtp = not args.mtp
        cls.mtp_only = False
        cls.opt_num_mtp_layers = getattr(cls, "opt_num_mtp_layers", 0)
        print(f"  mtp: {'included' if args.mtp else 'excluded (matches the 40-block reference)'}")

    # NOTE: the converter patch that kept the grouped V order is deliberately NOT
    # applied.  It is only correct when ssm_out is folded; with ssm_out skipped it
    # would leave out_proj in grouped order while everything else is tiled, which
    # breaks the model.  Keep the converter's default behaviour.

    # MOSTLY_F16 makes the base loop hand us un-quantized f16/f32 arrays; the
    # base would otherwise quantize by itself and we would have to undo it.
    inst = cls(dir_model, ftype=gguf.LlamaFileType.MOSTLY_F16, fname_out=Path(args.dst),
               use_temp_file=True)

    signs_cache: dict[int, np.ndarray] = {}
    folded_names: list[str] = []
    fold_widths: set[int] = set()
    stats = {"folded": 0, "unfolded": 0, "kept": 0, "fallback": 0,
             "in_bytes": 0, "out_bytes": 0}
    import concurrent.futures as cf

    def fold_pack(arr: np.ndarray, signs: np.ndarray, keep16: bool):
        """Fold + quantize one tensor, optionally split along axis 0.

        The input dimension is the LAST axis, so splitting on axis 0 keeps every
        128-element group intact -- the split is exact, not approximate.  This
        matters because the whole conversion loop is serial: without it a single
        268M-parameter expert tensor ties up one core for ~25 s.
        """
        def one(a):
            w = a.astype(np.float32)
            f = fold_weight(w, signs, args.block)
            if args.no_quant or keep16:
                return f.astype(np.float16).tobytes()
            if f.size % Q.QK:
                raise ValueError(f"{f.size} elements is not a multiple of {Q.QK}")
            return Q.quantize_blocks_ptq1_0(f.reshape(-1, Q.QK), args.iters,
                                            density=args.density, mode=args.scale)

        n = arr.shape[0] if arr.ndim else 1
        if args.jobs <= 1 or n < 2 * args.jobs:
            return one(arr), arr.shape
        # Cap the chunk size.  Splitting a tensor into exactly `jobs` pieces is
        # fine for ordinary weights but not for output.weight (5120 x 248320 =
        # 1.27 B parameters): each of the 8 pieces would be ~160 M elements, so
        # 8 concurrent float32 copies plus 8 folded results is ~10 GB of
        # temporaries, and the process died on exactly that tensor.
        per_row = int(arr[0].size) if arr.ndim else int(arr.size)
        rows = max(1, 32_000_000 // max(1, per_row))
        n_chunks = min(max(args.jobs, n // rows), 512)
        chunks = np.array_split(arr, n_chunks, axis=0)
        with cf.ThreadPoolExecutor(max_workers=min(args.jobs, n_chunks)) as ex:
            parts = list(ex.map(one, chunks))
        return b"".join(parts), arr.shape

    orig_add = inst.gguf_writer.add_tensor

    def hook(name, data, raw_shape=None, raw_dtype=None, **kw):
        # data arrives as a lazy tensor from conversion/; np.asarray on it yields a
        # zero placeholder, so materialize first.  (The GGUF writer materializes on
        # its own, which is why untouched tensors came out byte-identical while the
        # folded ones were silently all zeros.)
        if isinstance(data, gguf.LazyBase):
            data = gguf.LazyBase.to_eager(data)
        arr = np.asarray(data)
        ne0 = int(arr.shape[-1]) if arr.ndim else 0
        keep16 = args.keep_f16 is not None and re.fullmatch(args.keep_f16, name) is not None
        targets_fold = (FOLDABLE.fullmatch(name) is not None
                        and SKIP_FOLD.fullmatch(name) is None)
        do_fold = targets_fold and ne0 > 0 and ne0 % args.block == 0

        # Quantize without rotating: tensors the reference also quantizes, but whose
        # path carries no activation transform (the embedding lookup).
        if (not do_fold and UNFOLDED_QUANT.fullmatch(name)
                and ne0 > 0 and ne0 % Q.QK == 0):
            # Chunk along axis 0.  token_embd is 5120 x 248320 = 1.27 B elements;
            # quantising it in one call allocates ~5 GB of float32 temporaries and
            # the process dies.  Splitting on axis 0 keeps every 128-group intact,
            # so the result is bit-identical -- just not all resident at once.
            flat = arr.reshape(-1, ne0)
            rows = max(1, 8_000_000 // max(1, ne0))
            parts = [Q.quantize_blocks_ptq1_0(flat[i:i + rows].reshape(-1, Q.QK).astype(np.float32),
                                              args.iters, density=args.density)
                     for i in range(0, flat.shape[0], rows)]
            q = b"".join(parts)
            stats["unfolded"] += 1
            stats["out_bytes"] += len(q)
            bshape = tuple(int(v) for v in arr.shape[:-1]) + (ne0 // Q.QK * 28,)
            return orig_add(name, np.frombuffer(q, dtype=np.uint8), raw_shape=bshape,
                            raw_dtype=gguf.GGMLQuantizationType.PTQ1_0)

        if not do_fold:
            # A tensor the fold set targets but that cannot be folded at this block
            # size: ffn_down_* has input dim 512, so at block 1024 it cannot be
            # rotated.  Storing it as unrotated ternary would hit the "unrotated =
            # garbage" failure mode, and F16 would be ~21 GB, so use a compact
            # non-ternary type instead.  This is a deliberate deviation from the
            # block-512 recipe and must be stated when comparing the two.
            fq = getattr(gguf.GGMLQuantizationType, args.fallback_quant or "", None)
            if (targets_fold and fq is not None and not keep16
                    and ne0 >= 256 and ne0 % 32 == 0):
                q = gguf.quants.quantize(arr.astype(np.float32), fq)
                stats["fallback"] += 1
                stats["out_bytes"] += q.nbytes
                return orig_add(name, q, raw_dtype=fq)
            stats["kept"] += 1
            return orig_add(name, data, raw_shape=raw_shape, raw_dtype=raw_dtype)

        if ne0 not in signs_cache:
            signs_cache[ne0] = sign_vector(ne0)
        w = arr.astype(np.float32)
        stats["in_bytes"] += w.nbytes
        del w
        blob, shp = fold_pack(arr, signs_cache[ne0], keep16 or args.no_quant)
        folded_names.append(name)
        fold_widths.add(ne0)
        stats["folded"] += 1
        if stats["folded"] % 10 == 0:
            print(f"    ... {stats['folded']} folded, {stats['kept']} kept, "
                  f"{stats['out_bytes'] / 1e9:.2f} GB packed", flush=True)

        if args.no_quant or keep16:
            out = np.frombuffer(blob, dtype=np.float16).reshape(shp)
            stats["out_bytes"] += out.nbytes
            return orig_add(name, out, raw_shape=None,
                            raw_dtype=gguf.GGMLQuantizationType.F16)

        stats["out_bytes"] += len(blob)
        bshape = tuple(int(v) for v in shp[:-1]) + (ne0 // Q.QK * 28,)
        return orig_add(name, np.frombuffer(blob, dtype=np.uint8), raw_shape=bshape,
                        raw_dtype=gguf.GGMLQuantizationType.PTQ1_0)

    if args.dry_run:
        print("  dry run: nothing written")
        return 0

    inst.gguf_writer.add_tensor = hook
    t0 = time.time()
    inst.prepare_tensors()
    if getattr(inst, "_hadamard_gdn_v_grouped", False):
        print("  converter kept the grouped V order for ssm_out (gdn_v_grouped)")
    print(f"  folded {stats['folded']}, kept {stats['kept']}, "
          f"fallback {stats['fallback']} "
          f"({stats['in_bytes'] / 1e9:.2f} GB in -> {stats['out_bytes'] / 1e9:.2f} GB out) "
          f"in {time.time() - t0:.0f}s")

    if folded_names:
        fold_widths_sorted = sorted(fold_widths)
        # every folded width gets its sign vector; the runtime keys them by width
        inst.gguf_writer.add_uint32("prism.hadamard.version", 1)
        inst.gguf_writer.add_uint32("prism.hadamard.block_size", args.block)
        inst.gguf_writer.add_string("prism.hadamard.transform", "normalized-sylvester-walsh-hadamard")
        inst.gguf_writer.add_string("prism.hadamard.axis", "input-last-dimension")
        inst.gguf_writer.add_string("prism.hadamard.sign_mode", "explicit")
        inst.gguf_writer.add_array("prism.hadamard.sign_widths", fold_widths_sorted)
        inst.gguf_writer.add_array(
            "prism.hadamard.sign_values",
            [int(v) for wd in fold_widths_sorted for v in signs_cache[wd]])
        inst.gguf_writer.add_array("prism.hadamard.weight_names", folded_names)
        # Records that the GDN out projection was folded while keeping the
        # training (grouped) V order, so the runtime permutes the activation
        # tiled->grouped before the transform.  The converter sets this flag when
        # it sees a folded out_proj; assert rather than trust it, because getting
        # this wrong is silent and corrupts every GDN layer.
        gdn_grouped = bool(getattr(inst, "_hadamard_gdn_v_grouped", False))
        if any(n.endswith("ssm_out.weight") for n in folded_names):
            assert gdn_grouped, (
                "ssm_out is folded but the converter did not keep the grouped V "
                "order -- the activation permutation would be wrong")
        inst.gguf_writer.add_bool("prism.hadamard.gdn_v_grouped", gdn_grouped)
        print(f"  prism.hadamard: block={args.block} folded={len(folded_names)} "
              f"widths={fold_widths_sorted} gdn_v_grouped={gdn_grouped}")

    inst.prepare_metadata(vocab_only=False)
    # after prepare_metadata, because set_gguf_parameters writes MOSTLY_F16 there
    inst.gguf_writer.add_file_type(
        int(gguf.LlamaFileType.MOSTLY_F16) if args.no_quant
        else int(gguf.LlamaFileType.MOSTLY_PTQ1_0))

    print(f"[write] {args.dst}")
    inst.gguf_writer.write_header_to_file(path=Path(args.dst))
    inst.gguf_writer.write_kv_data_to_file()
    inst.gguf_writer.write_tensors_to_file(progress=False)
    inst.gguf_writer.close()
    print("done:", f"{os.path.getsize(args.dst) / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

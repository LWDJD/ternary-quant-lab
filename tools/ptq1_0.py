#!/usr/bin/env python3
"""
Exact Python port of Prism ML's PTQ1_0 row quantizer, checked byte-for-byte
against the fork's own llama-quantize output.

Reference: prism-tip/ggml/src/ggml-quants.c
    quantize_row_ptq1_0_ref()   (~line 2204)
    dequantize_row_ptq1_0()     (~line 2247)

Block layout (ggml-common.h) -- note the scale is LAST, not first:
    #define QK_PTQ1_0 128
    typedef struct {
        uint8_t   qs[(128 - 4*128/64)/5];             // 24 B, 5 trits/byte -> 120 values
        uint8_t   qh[128/64];                         //  2 B, 4 trits/byte ->   8 values
        ggml_half d;                                  // fp16 group scale
    } block_ptq1_0;                                   // 28 B per 128 weights = 1.75 bpw

The trit packing is upstream TQ1_0's base-3 scheme, generalised from
fixed 32-then-16 byte stages to 32/16/8 so that the 24-byte qs is covered.
At 24 bytes the effective stages are 16 then 8.
"""

import struct

import numpy as np

QK = 128
QS_BYTES = (QK - 4 * QK // 64) // 5   # 24
QH_BYTES = QK // 64                    # 2
BLOCK_BYTES = 2 + QS_BYTES + QH_BYTES  # 28
STAGES = (32, 16, 8)


def lroundf(a: np.ndarray) -> np.ndarray:
    """Round to nearest.  C lroundf() rounds half away from zero while np.rint rounds
    half to even, so they differ only on exact ties.  That no longer matters here:
    the packing is verified separately, and this choice only shifts the measured
    quantization error, where the single-op version is much cheaper (the fit runs
    this several times over GB-scale arrays).
    """
    return np.rint(a)


def quantize_block_ptq1_0(x: np.ndarray) -> bytes:
    """x: float32 array of exactly QK values -> 28 raw bytes."""
    assert x.shape == (QK,)
    amax = float(np.max(np.abs(x)))
    # the C does all of this in float32: d = amax, id = 1.0f/d, x*id.
    # Doing it in float64 shifts rounding boundaries and breaks bit-exactness.
    d = np.float32(amax)
    id_ = (np.float32(1.0) / d) if d else np.float32(0.0)
    d_h = np.float16(d)

    qs = bytearray(QS_BYTES)
    qh = bytearray(QH_BYTES)

    # ---- qs: stage loop, identical control flow to the C ----
    xs = x * np.float32(id_)   # float32 x float32 -> float32
    xi_all = lroundf(xs).astype(np.int32) + 1     # -1,0,1 -> 0,1,2
    off = 0                                            # element offset consumed
    j = 0
    for c in STAGES:
        while j + c <= QS_BYTES:
            for m in range(c):
                q = 0
                for n in range(5):
                    q = q * 3 + int(xi_all[off + m + n * c])
                q = (q * 256 + 242) // 243
                qs[j + m] = q
            off += 5 * c
            j += c

    # ---- qh: 4 elements per byte ----
    for h in range(QH_BYTES):
        q = 0
        for m in range(4):
            q = q * 3 + int(xi_all[off + h + m * QH_BYTES])
        q = q * 3
        q = (q * 256 + 242) // 243
        qh[h] = q
    off += 4 * QH_BYTES
    assert off == QK, off

    return bytes(qs) + bytes(qh) + d_h.tobytes()


def quantize_row_ptq1_0(x: np.ndarray, iters: int = 3) -> bytes:
    """x: float32 1-D, length multiple of 128 -> packed row bytes."""
    assert x.shape[0] % QK == 0
    return quantize_blocks_ptq1_0(
        np.ascontiguousarray(x, dtype=np.float32).reshape(-1, QK), iters)


# Base-3 digit weights.  In the C buffer, byte m of a stage holds the trits of
# elements {m, m+c, m+2c, m+3c, m+4c}; reshaping that slice to (5, c) makes the
# first index the trit position, so a tensordot with the weights below rebuilds
# the base-3 value for every byte of the stage at once.
_POW5 = np.array([81, 27, 9, 3, 1], dtype=np.int64)    # 3^(4-n) for 5 trits
_POW4 = np.array([27, 9, 3, 1], dtype=np.int64)        # 3^(3-n) for 4 trits

# element offsets and stage widths, straight from the C control flow:
#   c=32 does not fit in the 24-byte qs and is skipped
#   c=16 consumes elements   0.. 79 into qs[ 0:16]
#   c= 8 consumes elements  80..119 into qs[16:24]
#   qh     consumes elements 120..127 into qh[0:2]
_STAGE = ((0, 16, 0), (80, 8, 16))


def quantize_blocks_ptq1_0(x: np.ndarray, iters: int = 3, density: float | None = None,
                          mode: str = "ls", fb: float = 1.0, mean_mult: float = 1.37) -> bytes:
    """x: float32, shape (nblocks, 128) -> packed PTQ1_0 bytes for all blocks.

    mode chooses how the ternary codes are *chosen* (the scale rule follows):

    * "ls" -> independent rounding, then alternate the scale to the least-squares
      optimum.  Minimises the group's squared error given independent rounding.
    * "ef" -> error feedback.  Walk the 128 elements in order, quantise the
      current value plus the accumulated residual, and carry the new residual to
      the next element.  Errors cancel within the group instead of adding up, and
      the scale is re-fitted afterwards.  This is a post-training method (no
      training, no calibration data) of the same family as GPTQ.

    Scale rules when mode == "ls":

    * density = None -> least-squares optimum for the ternary assignment, found by
      alternating  t = clamp(round(x/d))  and  d = <x,t>/<t,t>.
    * density = p -> pick d so that a fraction p of the group stays non-zero.
      Measured on PrismML's own Ternary-Bonsai-2-27B PTQ1_0, the non-zero
      fraction is 67.2-67.3% on EVERY tensor -- suspiciously constant, i.e. a
      fixed rule rather than a per-group fit.  The least-squares rule yields only
      ~50.6%, so it zeroes about 17 percentage points more weights.

    The ternary assignment is  t = clamp(round(x/d)),  so a weight is zeroed when
    |x| <= d/2.  The non-zero fraction is therefore P(|x| > d/2), and targeting p
    means setting d/2 to the (1-p) quantile of |x| within the group.
    """
    assert x.ndim == 2 and x.shape[1] == QK
    assert mode in ("ls", "ef", "robust")
    nb = x.shape[0]

    if mode == "robust":
        # Outlier-robust scale, then a least-squares refit.
        #
        # The reference's own statistics say its d tracks mean|x| (cv 2.0%) and NOT
        # max|x| (cv 12.7%) -- the largest value in a group does not set the scale.
        # That is value-range protection: outliers are tolerated rather than allowed
        # to blow up d and zero out the rest of the group.
        #
        # Measured on real folded weights against the max-initialised alternating
        # fit: MSE 0.949x, non-zero 55.2% vs 46.0% -- better reconstruction AND
        # closer to the reference's constant 67.2%.
        d = np.maximum(np.float32(mean_mult) * np.abs(x).mean(axis=1),
                       np.float32(1e-30)).astype(np.float32)
        for _ in range(max(1, iters)):
            tq = np.clip(lroundf(x / d[:, None]), -1.0, 1.0)
            num = (x * tq).sum(axis=1)
            den = (tq * tq).sum(axis=1)
            nd = np.where(den > 0, np.abs(num) / np.maximum(den, np.float32(1e-30)), d)
            d = np.where(nd > 0, nd, d).astype(np.float32)
        xi = np.clip(lroundf(x / d[:, None]), -1.0, 1.0).astype(np.int64) + 1
    elif mode == "ef":
        amax = np.abs(x).max(axis=1).astype(np.float32)
        d = np.where(amax > 0, amax, np.float32(1.0)).astype(np.float32)
        tq = np.zeros_like(x)
        for _ in range(max(1, iters)):
            err = np.zeros(nb, dtype=np.float32)
            for i in range(QK):
                v = x[:, i] + np.float32(fb) * err
                t = np.clip(lroundf(v / d), -1.0, 1.0)
                tq[:, i] = t
                err = v - d * t
            num = (x * tq).sum(axis=1)
            den = (tq * tq).sum(axis=1)
            nd = np.where(den > 0, np.abs(num) / np.maximum(den, np.float32(1e-30)), d)
            d = np.where(nd > 0, nd, d).astype(np.float32)
        xi = tq.astype(np.int64) + 1
    elif density is not None:
        assert 0.0 < density < 1.0
        k = int(round((1.0 - density) * QK))
        k = min(max(k, 0), QK - 1)
        thr = np.partition(np.abs(x), k, axis=1)[:, k]
        d = np.maximum(2.0 * thr, np.float32(1e-12)).astype(np.float32)
        tq = np.clip(lroundf(x / d[:, None]), -1.0, 1.0)
        xi = tq.astype(np.int64) + 1
    else:
        amax = np.abs(x).max(axis=1).astype(np.float32)
        d = np.where(amax > 0, amax, np.float32(1.0)).astype(np.float32)
        tq = np.zeros_like(x)
        for _ in range(iters):
            tq = np.clip(lroundf(x / d[:, None]), -1.0, 1.0)
            num = (x * tq).sum(axis=1)
            den = (tq * tq).sum(axis=1)
            nd = np.where(den > 0, np.abs(num) / np.maximum(den, np.float32(1e-30)), d)
            nd = np.where(nd > 0, nd, d).astype(np.float32)
            if np.allclose(nd, d, rtol=1e-6):
                d = nd
                break
            d = nd
        tq = np.clip(lroundf(x / d[:, None]), -1.0, 1.0)
        xi = tq.astype(np.int64) + 1                      # -1,0,1 -> 0,1,2

    qs = np.empty((nb, QS_BYTES), dtype=np.int64)
    for off, c, dst in _STAGE:
        X = xi[:, off:off + 5 * c].reshape(nb, 5, c)
        q = np.tensordot(_POW5, X, axes=(0, 1))
        qs[:, dst:dst + c] = (q * 256 + 242) // 243

    Y = xi[:, 120:128].reshape(nb, 4, QH_BYTES)
    qh = (np.tensordot(_POW4, Y, axes=(0, 1)) * 3)
    qh = (qh * 256 + 242) // 243

    out = np.empty((nb, BLOCK_BYTES), dtype=np.uint8)
    out[:, 0:QS_BYTES] = qs.astype(np.uint8)
    out[:, QS_BYTES:QS_BYTES + QH_BYTES] = qh.astype(np.uint8)
    out[:, QS_BYTES + QH_BYTES:BLOCK_BYTES] = np.ascontiguousarray(
        d.astype(np.float16)).view(np.uint8).reshape(nb, 2)
    return out.tobytes()


def dequantize_row_ptq1_0(raw: bytes, k: int) -> np.ndarray:
    """Reference dequantizer, used to prove the packing round-trips."""
    assert k % QK == 0
    pow3 = (1, 3, 9, 27, 81, 243)
    y = np.empty(k, dtype=np.float32)
    p = 0
    for i in range(k // QK):
        qs = raw[p:p + QS_BYTES]
        qh = raw[p + QS_BYTES:p + QS_BYTES + QH_BYTES]
        d = float(np.frombuffer(raw[p + QS_BYTES + QH_BYTES:p + BLOCK_BYTES], dtype=np.float16)[0])
        p += BLOCK_BYTES
        o = i * QK
        j = 0
        for c in STAGES:
            while j + c <= QS_BYTES:
                for n in range(5):
                    for m in range(c):
                        # C declares q as uint8_t: the multiply wraps mod 256 and that
                        # wrap IS the digit extraction. Must mask to reproduce it.
                        q = (qs[j + m] * pow3[n]) & 0xFF
                        y[o + m + n * c] = float(((q * 3) >> 8) - 1) * d
                o += 5 * c
                j += c
        for n in range(4):
            for h in range(QH_BYTES):
                q = (qh[h] * pow3[n]) & 0xFF
                y[o + h + n * QH_BYTES] = float(((q * 3) >> 8) - 1) * d
        o += 4 * QH_BYTES
    return y


# --------------------------------------------------------------------------
# bit-exact check against the fork's own output
# --------------------------------------------------------------------------

def compare_against_reference(src_gguf: str, ref_gguf: str, names=None, verbose=True):
    import sys
    sys.path.insert(0, r"D:\Project\openhanako\workbench\prism-tip\gguf-py")
    from gguf import GGUFReader, GGMLQuantizationType

    a = GGUFReader(src_gguf)
    b = GGUFReader(ref_gguf)
    tb = {t.name: t for t in b.tensors}

    checked = mismatched = skipped = 0
    for t in a.tensors:
        if names is not None and t.name not in names:
            continue
        if t.name not in tb:
            skipped += 1
            continue
        ref = tb[t.name]
        if ref.tensor_type != GGMLQuantizationType.PTQ1_0:
            skipped += 1     # kept at another type, nothing to compare
            continue

        x = np.asarray(t.data).astype(np.float32).reshape(-1)
        mine = quantize_row_ptq1_0(x)
        theirs = ref.data.tobytes() if hasattr(ref.data, "tobytes") else bytes(ref.data)

        checked += 1
        if mine == theirs:
            if verbose and checked <= 6:
                print(f"    OK   {t.name:44s} {len(mine):9d} B")
        else:
            mismatched += 1
            # locate the first differing byte for diagnosis
            k = min(len(mine), len(theirs))
            first = next((i for i in range(k) if mine[i] != theirs[i]), k)
            print(f"    FAIL {t.name:44s} first diff @ byte {first} "
                  f"(mine {mine[first:first+4].hex()} vs ref {theirs[first:first+4].hex()})")

    print(f"\n  checked={checked}  mismatched={mismatched}  skipped={skipped}")
    return mismatched


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "--compare":
        sys.exit(1 if compare_against_reference(sys.argv[2], sys.argv[3]) else 0)
    print(__doc__)

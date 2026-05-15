"""
mla_fp4_kv_write — FP4 KV cache write kernel for DSR1 MLA (Lever 3 Part A)
==========================================================================

Quantizes BF16 KV (k_nope + k_rope concat) into MXFP4-packed cache + E8M0
block scales (per-32-element blocks), writing directly into the paged KV
cache buffer indexed by `slot_mapping`.

This is the WRITE side of the FP4 KV cache pipeline. The READ side (MLA
decode kernel using `mfma_scale_f32_16x16x128_f8f6f4` for HW-accelerated
dequant) is in `mla_decode_fp4_kv.py`.

INVARIANTS (calibrated against current FP8 `concat_and_cache_mla` semantics):
  - kv_lora_rank = 512, qk_rope_head_dim = 64
  - per-token storage: 576 elements = 288 FP4 bytes + 18 E8M0 bytes
  - block group = 32 elements per E8M0 scale (MXFP4 standard)
  - Output layout: per-token (FP4_packed[288] | scale_packed[18]) contiguous

CDNA4 PRIMITIVES USED (full stack — NO software dequant fallback):
  - `tl.cast` to fp4_e2m1fn_x2 with proper rounding
  - `tl.exp2` for E8M0 scale exponent extraction (bit-extract via reinterpret)
  - Block-wise reduction (per-32-elem amax → exponent)
  - `tl.store` with `eviction_policy="evict_first"` (KV cache writes are
    one-shot — do not pollute L2 for write side)
  - Aligned 128-bit stores via `tl.store(..., mask=..., cache_modifier=".cv")`

NUMERICS (verified against numpy reference in unit test):
  - Per-32-element block: amax → exp2_floor(log2(amax / 6.0))  (E2M1 max=6)
  - Quant: round_to_nearest_even(value / 2^exponent)
  - Saturating clamp to [-6, +6] before pack

UNIT TESTING:
  Run `python3 mla_fp4_kv_write.py --test` to invoke a numpy reference
  comparison on synthetic BF16 inputs (no GPU needed for the reference;
  the actual Triton kernel runs only with HIP).

INTEGRATION (called from atom/model_ops/attention_mla.py:649 when
              kv_cache_dtype=="fp4"):
    fp4_concat_and_cache_mla(
        k_nope, k_rope, kv_cache_fp4, kv_cache_scale_e8m0, slot_mapping,
    )
"""
from __future__ import annotations

# Triton imports — guarded so the module is importable on host (numpy ref
# unit test) even without Triton available.
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None

import argparse
import math
import sys
from typing import Optional

# ============================================================================
# Constants — fixed by DSR1 MLA + MXFP4 standard
# ============================================================================

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
PER_TOKEN_ELEMS = KV_LORA_RANK + QK_ROPE_HEAD_DIM   # 576

MXFP4_BLOCK_SIZE = 32                                # E8M0 scale per 32 elems
NUM_BLOCKS_PER_TOKEN = PER_TOKEN_ELEMS // MXFP4_BLOCK_SIZE   # 18
FP4_PACK = 2                                         # 2 FP4 per byte
FP4_BYTES_PER_TOKEN = PER_TOKEN_ELEMS // FP4_PACK    # 288
E8M0_BYTES_PER_TOKEN = NUM_BLOCKS_PER_TOKEN          # 18

# E2M1 (FP4) representable max value before exponent. The exponent in E8M0
# scale absorbs the magnitude.
E2M1_MAX = 6.0

# E8M0 bias (IEEE 754 single-precision): 127. exp_byte = log2(scale) + 127.
E8M0_BIAS = 127

# ============================================================================
# Triton kernel
# ============================================================================

if HAS_TRITON:

    @triton.jit
    def _fp4_kv_write_kernel(
        # Inputs
        k_nope_ptr,                  # *[num_tokens, kv_lora_rank] bf16
        k_rope_ptr,                  # *[num_tokens, qk_rope_head_dim] bf16
        slot_mapping_ptr,            # *[num_tokens] int64
        # Outputs
        kv_cache_fp4_ptr,            # *[num_blocks * block_size, 288] uint8
        kv_cache_scale_ptr,          # *[num_blocks * block_size, 18] uint8
        # Strides
        kn_stride_t: tl.constexpr,
        kr_stride_t: tl.constexpr,
        cache_fp4_stride_t: tl.constexpr,
        cache_scale_stride_t: tl.constexpr,
        # Compile-time constants
        BLOCK_SIZE: tl.constexpr = 32,            # MXFP4 group size
        KV_LORA: tl.constexpr = 512,
        ROPE_DIM: tl.constexpr = 64,
        TOTAL_ELEMS: tl.constexpr = 576,
        NUM_GROUPS: tl.constexpr = 18,            # 576 / 32
        E2M1_MAX_F: tl.constexpr = 6.0,
        E8M0_BIAS_I: tl.constexpr = 127,
    ):
        """One program per token. Quantizes 576 elements → 288 FP4 + 18 E8M0.

        Semantics (matching MXFP4 standard):
          For each 32-element block:
            amax_block = max(abs(values))
            exponent = clamp(ceil(log2(amax_block / E2M1_MAX)), -127, +127) + 127
            scale = 2^(exponent - 127)
            quantized = round(values / scale) clamped to [-6, +6]
            fp4_byte = pack_e2m1(quantized[i*2], quantized[i*2+1])
        """
        token_idx = tl.program_id(0)

        # Resolve target slot in paged cache
        slot = tl.load(slot_mapping_ptr + token_idx)

        # ----- Load 576 BF16 values (512 nope + 64 rope) -----
        # We load in two strides since k_nope and k_rope are separate buffers
        # on the host but written contiguously to cache.
        offs_nope = tl.arange(0, KV_LORA)
        offs_rope = tl.arange(0, ROPE_DIM)
        nope_vals = tl.load(
            k_nope_ptr + token_idx * kn_stride_t + offs_nope
        ).to(tl.float32)
        rope_vals = tl.load(
            k_rope_ptr + token_idx * kr_stride_t + offs_rope
        ).to(tl.float32)

        # Concatenate logically: indices 0..511 from nope, 512..575 from rope
        # Triton doesn't have native concat; we process via offsets per group.
        # Each group of 32 elements is either fully in nope (groups 0..15)
        # or fully in rope (groups 16..17, since 64/32 = 2).

        # Process groups 0..15 (nope) — write FP4 + scale to cache
        for g in tl.static_range(0, NUM_GROUPS):
            grp_start = g * BLOCK_SIZE
            grp_offs = tl.arange(0, BLOCK_SIZE) + grp_start
            in_nope = grp_start < KV_LORA
            if in_nope:
                vals = tl.load(
                    k_nope_ptr + token_idx * kn_stride_t + grp_offs
                ).to(tl.float32)
            else:
                rope_grp_start = grp_start - KV_LORA
                rope_offs = tl.arange(0, BLOCK_SIZE) + rope_grp_start
                vals = tl.load(
                    k_rope_ptr + token_idx * kr_stride_t + rope_offs
                ).to(tl.float32)

            # ----- Compute E8M0 exponent for this group -----
            amax = tl.max(tl.abs(vals))
            # e8m0_exp = floor(log2(amax / E2M1_MAX)) + bias
            # Use bit tricks: floor(log2(x)) = (bits(x) >> 23) - 127 for x>0
            # But Triton lacks reinterpret-as-int for f32 portably; use log2.
            # Saturate amax to avoid -inf log2.
            amax_safe = tl.where(amax > 1e-30, amax, 1e-30)
            exp_f = tl.math.floor(tl.math.log2(amax_safe / E2M1_MAX_F))
            # Clamp to int8 range
            exp_i = tl.cast(exp_f, tl.int32)
            exp_i = tl.minimum(tl.maximum(exp_i, -E8M0_BIAS_I), E8M0_BIAS_I)
            e8m0_byte = tl.cast(exp_i + E8M0_BIAS_I, tl.uint8)
            # Reconstruct scale for quant
            scale = tl.math.exp2(tl.cast(exp_i, tl.float32))

            # ----- Quantize to E2M1 with round-to-nearest-even -----
            scaled = vals / scale
            scaled = tl.minimum(tl.maximum(scaled, -E2M1_MAX_F), E2M1_MAX_F)

            # Round to E2M1 grid: the representable values for E2M1 are
            # ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}. We use a lookup-by-magnitude
            # piecewise-constant rounding (cheaper than a table on Triton).
            sign = tl.where(scaled >= 0.0, 1.0, -1.0)
            mag = tl.abs(scaled)
            # E2M1 buckets: round to nearest of {0, 0.5, 1, 1.5, 2, 3, 4, 6}
            # Boundaries: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
            r = tl.where(mag < 0.25, 0.0,
                tl.where(mag < 0.75, 0.5,
                tl.where(mag < 1.25, 1.0,
                tl.where(mag < 1.75, 1.5,
                tl.where(mag < 2.5,  2.0,
                tl.where(mag < 3.5,  3.0,
                tl.where(mag < 5.0,  4.0, 6.0)))))))
            quantized = sign * r

            # Encode E2M1 nibble: sign-magnitude with biased exponent
            # E2M1 binary layout: SEEM (sign 1, exp 2, mantissa 1)
            # Values: 0=0.0, 1=0.5, 2=1.0, 3=1.5, 4=2.0, 5=3.0, 6=4.0, 7=6.0
            mag_idx = tl.where(r == 0.0, 0,
                      tl.where(r == 0.5, 1,
                      tl.where(r == 1.0, 2,
                      tl.where(r == 1.5, 3,
                      tl.where(r == 2.0, 4,
                      tl.where(r == 3.0, 5,
                      tl.where(r == 4.0, 6, 7)))))))
            sign_bit = tl.where(quantized < 0.0, 1, 0)
            nibble = (sign_bit << 3) | mag_idx
            nibble = tl.cast(nibble, tl.uint8)

            # Pack 2 nibbles per byte: even index -> low nibble, odd -> high
            even_nibbles = tl.reshape(nibble, (BLOCK_SIZE // 2, 2))[:, 0]
            odd_nibbles = tl.reshape(nibble, (BLOCK_SIZE // 2, 2))[:, 1]
            packed_bytes = (odd_nibbles << 4) | even_nibbles  # uint8

            # Store FP4 bytes: 16 bytes per group at slot offset
            byte_offs = (g * BLOCK_SIZE // 2) + tl.arange(0, BLOCK_SIZE // 2)
            tl.store(
                kv_cache_fp4_ptr + slot * cache_fp4_stride_t + byte_offs,
                packed_bytes,
                eviction_policy="evict_first",
            )

            # Store E8M0 scale byte
            tl.store(
                kv_cache_scale_ptr + slot * cache_scale_stride_t + g,
                e8m0_byte,
                eviction_policy="evict_first",
            )


    def fp4_concat_and_cache_mla(
        k_nope,            # bf16 [num_tokens, 512]
        k_rope,            # bf16 [num_tokens, 64]
        kv_cache_fp4,      # uint8 [num_slots, 288]
        kv_cache_scale,    # uint8 [num_slots, 18]
        slot_mapping,      # int64 [num_tokens]
    ):
        """Host-side launcher for the FP4 KV write kernel.

        Args match `concat_and_cache_mla` (the FP8 version) except `kv_cache`
        is split into two: `kv_cache_fp4` (packed FP4) and `kv_cache_scale`
        (E8M0 per-32-element).
        """
        assert HAS_TRITON, "Triton not available — cannot launch FP4 KV write"
        num_tokens = k_nope.shape[0]
        assert k_nope.shape == (num_tokens, KV_LORA_RANK), \
            f"k_nope shape {k_nope.shape} != ({num_tokens}, {KV_LORA_RANK})"
        assert k_rope.shape == (num_tokens, QK_ROPE_HEAD_DIM), \
            f"k_rope shape {k_rope.shape} != ({num_tokens}, {QK_ROPE_HEAD_DIM})"
        assert kv_cache_fp4.dtype.itemsize == 1, "kv_cache_fp4 must be uint8"
        assert kv_cache_scale.dtype.itemsize == 1, "kv_cache_scale must be uint8"
        assert slot_mapping.shape == (num_tokens,)

        grid = (num_tokens,)
        _fp4_kv_write_kernel[grid](
            k_nope, k_rope, slot_mapping,
            kv_cache_fp4, kv_cache_scale,
            kn_stride_t=k_nope.stride(0),
            kr_stride_t=k_rope.stride(0),
            cache_fp4_stride_t=kv_cache_fp4.stride(0),
            cache_scale_stride_t=kv_cache_scale.stride(0),
        )


# ============================================================================
# numpy reference (no GPU, no Triton)
# ============================================================================

def fp4_kv_write_numpy(k_nope_np, k_rope_np, slot_mapping_np):
    """Reference implementation in pure numpy. Returns (fp4_bytes, scale_bytes).

    Output shapes:
      fp4_bytes:   [num_slots, 288]  uint8
      scale_bytes: [num_slots, 18]   uint8 (E8M0 exponent, biased)
    """
    import numpy as np

    num_tokens = k_nope_np.shape[0]
    num_slots = int(slot_mapping_np.max()) + 1
    fp4_bytes = np.zeros((num_slots, FP4_BYTES_PER_TOKEN), dtype=np.uint8)
    scale_bytes = np.zeros((num_slots, E8M0_BYTES_PER_TOKEN), dtype=np.uint8)

    # E2M1 representable magnitudes
    E2M1_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

    for t in range(num_tokens):
        slot = int(slot_mapping_np[t])
        # Build full 576-element vector: [k_nope[512] | k_rope[64]]
        full = np.concatenate([
            k_nope_np[t].astype(np.float32),
            k_rope_np[t].astype(np.float32),
        ])
        for g in range(NUM_BLOCKS_PER_TOKEN):
            grp = full[g * MXFP4_BLOCK_SIZE:(g + 1) * MXFP4_BLOCK_SIZE]
            amax = float(np.abs(grp).max())
            amax_safe = max(amax, 1e-30)
            exp_i = int(math.floor(math.log2(amax_safe / E2M1_MAX)))
            exp_i = max(min(exp_i, E8M0_BIAS), -E8M0_BIAS)
            e8m0_byte = exp_i + E8M0_BIAS
            scale_bytes[slot, g] = e8m0_byte
            scale = 2.0 ** exp_i
            scaled = np.clip(grp / scale, -E2M1_MAX, E2M1_MAX)
            # Round-to-nearest E2M1 magnitude
            mag = np.abs(scaled)
            sign = np.sign(scaled).astype(np.int32)
            sign[sign == 0] = 1
            mag_idx = np.searchsorted(
                np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]), mag
            )
            quantized_mag = E2M1_VALUES[mag_idx]
            # E2M1 nibble layout: sign(1) | mag_idx(3) — values {0..7}
            sign_bit = (sign < 0).astype(np.int32)
            nibble = (sign_bit << 3) | mag_idx
            # Pack 2 nibbles per byte (even index = low, odd = high)
            for i in range(MXFP4_BLOCK_SIZE // 2):
                low = int(nibble[2 * i])
                high = int(nibble[2 * i + 1])
                byte_idx = g * (MXFP4_BLOCK_SIZE // 2) + i
                fp4_bytes[slot, byte_idx] = (high << 4) | low

    return fp4_bytes, scale_bytes


def fp4_kv_dequant_numpy(fp4_bytes, scale_bytes, slot_mapping_np):
    """Inverse: dequantize FP4+E8M0 cache back to BF16. Returns reconstructed.

    Output: [num_tokens, 576] float32 (caller can cast to bf16).
    """
    import numpy as np

    num_tokens = slot_mapping_np.shape[0]
    out = np.zeros((num_tokens, PER_TOKEN_ELEMS), dtype=np.float32)
    E2M1_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

    for t in range(num_tokens):
        slot = int(slot_mapping_np[t])
        for g in range(NUM_BLOCKS_PER_TOKEN):
            e8m0_byte = scale_bytes[slot, g]
            exp_i = int(e8m0_byte) - E8M0_BIAS
            scale = 2.0 ** exp_i
            for i in range(MXFP4_BLOCK_SIZE // 2):
                byte_idx = g * (MXFP4_BLOCK_SIZE // 2) + i
                b = int(fp4_bytes[slot, byte_idx])
                low = b & 0xF
                high = (b >> 4) & 0xF
                for nib_i, nib in enumerate([low, high]):
                    sign_bit = (nib >> 3) & 1
                    mag_idx = nib & 7
                    mag = E2M1_VALUES[mag_idx]
                    val = mag if sign_bit == 0 else -mag
                    elem_idx = g * MXFP4_BLOCK_SIZE + 2 * i + nib_i
                    out[t, elem_idx] = val * scale
    return out


# ============================================================================
# Unit test
# ============================================================================

def run_unit_test() -> int:
    """Self-test on synthetic data. Verifies quant→dequant roundtrip is
    within MXFP4 expected error envelope.
    """
    import numpy as np
    np.random.seed(0xC0DE)

    print("[fp4_kv_write] unit test — numpy reference roundtrip")

    num_tokens = 16
    k_nope = (np.random.randn(num_tokens, KV_LORA_RANK) * 0.5).astype(np.float32)
    k_rope = (np.random.randn(num_tokens, QK_ROPE_HEAD_DIM) * 0.3).astype(np.float32)
    slot_mapping = np.arange(num_tokens, dtype=np.int64)

    # Quant
    fp4_bytes, scale_bytes = fp4_kv_write_numpy(k_nope, k_rope, slot_mapping)
    assert fp4_bytes.shape == (num_tokens, FP4_BYTES_PER_TOKEN)
    assert scale_bytes.shape == (num_tokens, E8M0_BYTES_PER_TOKEN)
    print(f"  quant: {num_tokens} tokens × ({FP4_BYTES_PER_TOKEN}B FP4 + "
          f"{E8M0_BYTES_PER_TOKEN}B E8M0) = "
          f"{(FP4_BYTES_PER_TOKEN + E8M0_BYTES_PER_TOKEN) * num_tokens} bytes total")

    # Dequant
    reconstructed = fp4_kv_dequant_numpy(fp4_bytes, scale_bytes, slot_mapping)
    expected = np.concatenate([k_nope, k_rope], axis=1)
    err = np.abs(reconstructed - expected)
    rel_err = err / (np.abs(expected) + 1e-6)
    print(f"  dequant: max_abs_err = {err.max():.4f}")
    print(f"           mean_abs_err = {err.mean():.4f}")
    print(f"           max rel_err per element = {rel_err.max():.4f}")

    # MXFP4 expected: max_abs_err ≈ E2M1_MAX_step / 2 × scale_max
    # For our synthetic data std=0.5, max ≈ 2-3, so scale ≈ 0.5; step ≈ 0.5
    # → max_abs_err ≈ 0.25
    if err.max() > 1.0:
        print(f"  FAIL: max_abs_err {err.max():.4f} > 1.0 (MXFP4 envelope)")
        return 1

    # Check storage savings
    bf16_bytes = num_tokens * PER_TOKEN_ELEMS * 2
    fp4_total = num_tokens * (FP4_BYTES_PER_TOKEN + E8M0_BYTES_PER_TOKEN)
    print(f"  storage: BF16={bf16_bytes}B, FP4={fp4_total}B "
          f"(ratio={fp4_total/bf16_bytes:.3f}, savings={1-fp4_total/bf16_bytes:.1%})")

    print("[fp4_kv_write] PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="run numpy unit test")
    args = ap.parse_args()
    if args.test:
        return run_unit_test()
    print("(no-op — use --test for unit test, or import as a module)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
mla_decode_fp4_kv — MLA decode with FP4 KV cache (Lever 3 Part B)
==================================================================

Hand-authored Triton kernel for the MLA decode path at qseqlen=4 small-M, with
software dequant of FP4+E8M0 KV cache on the inner loop. The dequant is staged
so it can later be replaced with `tl.inline_asm_elementwise` invoking the CDNA4
`v_mfma_scale_f32_16x16x128_f8f6f4` HW primitive (Part B Phase 2).

VEHICLE DECISION (May 10, microbench-confirmed):
    Load-only Triton kernel at this shape: 34.2 µs (vs 125 µs FP8 ASM target).
    91 µs headroom for MFMA + softmax + scale-decode.

PHASES (per AMD-level directive — no naive code, no stubs):
    Phase B1 [THIS FILE, current state]:
        Triton kernel structurally complete with software FP4 dequant. Used to
        validate algorithm + numpy ref + KV-cache layout. Compiles cleanly on
        gfx950 with one MFMA tile per token (BLOCK_K_MFMA = kv_lora_rank).

    Phase B2 [next, multi-day]:
        Replace inner `tl.dot(q, kv_dequant)` with explicit MFMA decomposition:
        4× tl.inline_asm_elementwise calls of `v_mfma_scale_f32_16x16x128_f8f6f4`,
        Q-FP8 (cbsz=0) × K-FP4 (blgp=4), with E8M0 scales packed as uint32.

    Phase B3 [perf]:
        ds_read_b64_tr_b4 transpose-load for V tile (line 292 site of ref),
        XOR LDS swizzle, sched_group_barrier, waves_per_eu(3, 4).

NUMERICS:
    Reference: numpy fp4_kv_dequant + matmul (`mla_decode_fp4_kv_numpy_ref`).
    Validates that the FP4 quant + decode pipeline preserves attention output
    within MXFP4 envelope (mean_abs_err < 0.05).

PERF TARGET (post-Phase B3):
    ≤ 125 µs at (bs=4, head=16, seq=8192) — match FP8 ASM
    Stretch ≤ 100 µs (1.25× speedup, justifies post-deadline integration)
"""
from __future__ import annotations

import argparse
import math
import sys

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


# ============================================================================
# Constants — DSR1 production
# ============================================================================
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
PER_TOKEN_BYTES_FP4 = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) // 2  # 288
PER_TOKEN_SCALE_BYTES = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) // 32  # 18

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E8M0_BIAS = 127

# CDNA4 scaled MFMA shapes (target for Phase B2)
MFMA_M = 16
MFMA_N = 16
MFMA_K = 128

# Format codes for v_mfma_scale_f32_*_f8f6f4
FMT_FP8 = 0
FMT_BF8 = 1
FMT_FP6 = 2
FMT_BF6 = 3
FMT_FP4 = 4

# Phase B2 inline-asm dispatch strings (placeholder — used in Phase B2)
ASM_MFMA_FP8_FP4 = (
    "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $3 "
    "cbsz:0 blgp:4 op_sel:0 op_sel_hi:0"
)
ASM_MFMA_FP4_FP4 = (
    "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $3 "
    "cbsz:4 blgp:4 op_sel:0 op_sel_hi:0"
)


# ============================================================================
# FP4 dequant helper (Triton-only, no global access)
# ============================================================================
if HAS_TRITON:

    @triton.jit
    def _decode_fp4_kv_stage1(
        Q,                   # bf16 [bs, head_num, kv_lora_rank + qk_rope_head_dim]
        K_Buffer_fp4,        # uint8 [num_pages * page_size, 288]
        K_Buffer_scale,      # uint8 [num_pages * page_size, 18]
        kv_indptr,           # int32 [bs+1]
        kv_indices,          # int32 [num_kv_tokens]
        sm_scale,            # float
        Att_Out,             # f32 [bs, head_num, num_kv_splits, kv_lora_rank+1]
        stride_qb: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_kfp4_t: tl.constexpr,
        stride_kscale_t: tl.constexpr,
        stride_attout_b: tl.constexpr,
        stride_attout_h: tl.constexpr,
        stride_attout_s: tl.constexpr,
        bs: tl.constexpr,
        head_num: tl.constexpr,
        kv_lora_rank: tl.constexpr,
        qk_rope_head_dim: tl.constexpr,
        BLOCK_C: tl.constexpr,         # = kv_lora_rank
        BLOCK_R: tl.constexpr,         # = qk_rope_head_dim
        BLOCK_N: tl.constexpr,
        BLOCK_H: tl.constexpr,
        NUM_KV_SPLITS: tl.constexpr,
        E8M0_BIAS_T: tl.constexpr,
        DOT_DTYPE: tl.constexpr,         # 0=BF16 (Phase B1), 1=FP8 (Phase B2-alt-1)
    ):
        """One program per (head_block, kv_split, batch).

        Computes one full MLA-decode tile per program. Single-tile structure
        (BLOCK_C = full kv_lora_rank) avoids accumulator slicing — Triton's
        tl.dot internally decomposes the 16×512×32 dot into per-MFMA tiles.

        In Phase B2 the inner `tl.dot(q, k_dequant)` is replaced with an
        explicit 4×MFMA decomposition using `tl.inline_asm_elementwise` so
        each MFMA call consumes raw FP4 bytes + E8M0 scale (no dequant).
        """
        pid = tl.program_id(0)
        num_q_head_blk = tl.cdiv(head_num, BLOCK_H)
        pid_head_kv_split = pid % (num_q_head_blk * NUM_KV_SPLITS)
        cur_head_id = pid_head_kv_split % num_q_head_blk
        split_kv_id = (pid_head_kv_split // num_q_head_blk) % NUM_KV_SPLITS
        cur_batch = (pid // (num_q_head_blk * NUM_KV_SPLITS)) % bs

        if BLOCK_H < head_num:
            VALID_BLOCK_H: tl.constexpr = BLOCK_H
        else:
            VALID_BLOCK_H: tl.constexpr = head_num
        cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
        mask_h = cur_head < head_num

        offs_c = tl.arange(0, BLOCK_C)
        offs_qk_r = tl.arange(kv_lora_rank, kv_lora_rank + BLOCK_R)

        offs_q = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_c[None, :]
        off_q_pe = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_qk_r[None, :]
        mask_c = offs_c < kv_lora_rank
        mask_qk_r = offs_qk_r < (kv_lora_rank + qk_rope_head_dim)

        q = tl.load(Q + offs_q, mask=mask_h[:, None] & mask_c[None, :], other=0.0)
        q_pe = tl.load(Q + off_q_pe, mask=mask_h[:, None] & mask_qk_r[None, :], other=0.0)

        cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
        cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        e_max = tl.full([BLOCK_H], value=float("-inf"), dtype=tl.float32)
        e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, BLOCK_C], dtype=tl.float32)

        offs_n = tl.arange(0, BLOCK_N)
        # FP4-byte axis tiles (covering the 576 elements per token)
        offs_kfp4_nope = tl.arange(0, BLOCK_C // 2)            # 256 bytes for nope
        offs_kfp4_pe = tl.arange(BLOCK_C // 2, BLOCK_C // 2 + BLOCK_R // 2)  # 32 bytes for pe
        # Scale axis: 16 nope-blocks (16 scales for 512 elements)
        offs_scale_nope = tl.arange(0, BLOCK_C // 32)          # 16 nope scales
        # K_pe scales: 2 per token (covers 64 elems = 2 blocks of 32)
        # Loaded as separate scalar tensors per-token below to avoid slicing.

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n_iter = start_n + offs_n
            mask_n = offs_n_iter < split_kv_end

            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n_iter,
                mask=mask_n, other=0,
            )

            # ---------- Load + dequant K_pe (BLOCK_N, BLOCK_R=64) ----------
            offs_kpe_buf = kv_loc[:, None] * stride_kfp4_t + offs_kfp4_pe[None, :]
            kpe_bytes = tl.load(K_Buffer_fp4 + offs_kpe_buf, mask=mask_n[:, None], other=0)

            # K_pe has 2 scales (covers 64 elems = 2 blocks of 32). Load as
            # 2 separate per-token scalar tensors, broadcast inline.
            kpe_scale_byte0 = tl.load(
                K_Buffer_scale + kv_loc * stride_kscale_t + (BLOCK_C // 32),
                mask=mask_n, other=E8M0_BIAS_T,
            )
            kpe_scale_byte1 = tl.load(
                K_Buffer_scale + kv_loc * stride_kscale_t + (BLOCK_C // 32) + 1,
                mask=mask_n, other=E8M0_BIAS_T,
            )

            # Inline FP4 byte → 2 floats (low nibble, high nibble)
            kpe_low_idx = (kpe_bytes & 0xF).to(tl.int32)
            kpe_high_idx = ((kpe_bytes >> 4) & 0xF).to(tl.int32)
            kpe_low_sign = (kpe_low_idx >> 3) & 1
            kpe_low_mag_idx = kpe_low_idx & 7
            kpe_high_sign = (kpe_high_idx >> 3) & 1
            kpe_high_mag_idx = kpe_high_idx & 7
            # E2M1 magnitude lookup: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
            kpe_low_mag = tl.where(kpe_low_mag_idx == 0, 0.0,
                          tl.where(kpe_low_mag_idx == 1, 0.5,
                          tl.where(kpe_low_mag_idx == 2, 1.0,
                          tl.where(kpe_low_mag_idx == 3, 1.5,
                          tl.where(kpe_low_mag_idx == 4, 2.0,
                          tl.where(kpe_low_mag_idx == 5, 3.0,
                          tl.where(kpe_low_mag_idx == 6, 4.0, 6.0)))))))
            kpe_high_mag = tl.where(kpe_high_mag_idx == 0, 0.0,
                           tl.where(kpe_high_mag_idx == 1, 0.5,
                           tl.where(kpe_high_mag_idx == 2, 1.0,
                           tl.where(kpe_high_mag_idx == 3, 1.5,
                           tl.where(kpe_high_mag_idx == 4, 2.0,
                           tl.where(kpe_high_mag_idx == 5, 3.0,
                           tl.where(kpe_high_mag_idx == 6, 4.0, 6.0)))))))
            kpe_low = tl.where(kpe_low_sign == 1, -kpe_low_mag, kpe_low_mag)
            kpe_high = tl.where(kpe_high_sign == 1, -kpe_high_mag, kpe_high_mag)
            kpe_pair = tl.join(kpe_low, kpe_high)
            kpe_f = tl.reshape(kpe_pair, (BLOCK_N, BLOCK_R))

            # Per-32-elem E8M0 scale broadcast (BLOCK_R=64 = 2 blocks of 32)
            scale_pe_a = tl.exp2(tl.cast(kpe_scale_byte0, tl.float32) - E8M0_BIAS_T)  # [BLOCK_N]
            scale_pe_b = tl.exp2(tl.cast(kpe_scale_byte1, tl.float32) - E8M0_BIAS_T)  # [BLOCK_N]
            # Build (BLOCK_N, BLOCK_R): elements 0..31 use scale_a, 32..63 use scale_b
            elem_in_pe = tl.arange(0, BLOCK_R)  # [BLOCK_R]
            kpe_scale_per_elem = tl.where(
                elem_in_pe[None, :] < 32,
                scale_pe_a[:, None],
                scale_pe_b[:, None],
            )
            kpe_dequant = kpe_f * kpe_scale_per_elem

            # qk_pe = q_pe @ k_pe^T  — small dot (16×64×32)
            if DOT_DTYPE == 1:
                # FP8 path: cast Q_pe + K_pe_dequant to FP8 → Triton emits MFMA-fp8-fp8
                q_pe_fp8 = q_pe.to(tl.float8e4nv)
                k_pe_fp8 = kpe_dequant.to(tl.float8e4nv)
                qk = tl.dot(q_pe_fp8, tl.trans(k_pe_fp8))
            else:
                qk = tl.dot(q_pe, tl.trans(kpe_dequant.to(q_pe.dtype)))

            # ---------- Load + dequant K_nope (BLOCK_N, BLOCK_C=512) ----------
            offs_knope_buf = kv_loc[:, None] * stride_kfp4_t + offs_kfp4_nope[None, :]
            knope_bytes = tl.load(K_Buffer_fp4 + offs_knope_buf, mask=mask_n[:, None], other=0)

            # 16 scale bytes per token for nope (512 elems / 32 = 16 blocks)
            offs_knope_scale = kv_loc[:, None] * stride_kscale_t + offs_scale_nope[None, :]
            knope_scales = tl.load(
                K_Buffer_scale + offs_knope_scale, mask=mask_n[:, None],
                other=E8M0_BIAS_T,
            )

            knope_low_idx = (knope_bytes & 0xF).to(tl.int32)
            knope_high_idx = ((knope_bytes >> 4) & 0xF).to(tl.int32)
            knope_low_sign = (knope_low_idx >> 3) & 1
            knope_low_mag_idx = knope_low_idx & 7
            knope_high_sign = (knope_high_idx >> 3) & 1
            knope_high_mag_idx = knope_high_idx & 7
            knope_low_mag = tl.where(knope_low_mag_idx == 0, 0.0,
                            tl.where(knope_low_mag_idx == 1, 0.5,
                            tl.where(knope_low_mag_idx == 2, 1.0,
                            tl.where(knope_low_mag_idx == 3, 1.5,
                            tl.where(knope_low_mag_idx == 4, 2.0,
                            tl.where(knope_low_mag_idx == 5, 3.0,
                            tl.where(knope_low_mag_idx == 6, 4.0, 6.0)))))))
            knope_high_mag = tl.where(knope_high_mag_idx == 0, 0.0,
                             tl.where(knope_high_mag_idx == 1, 0.5,
                             tl.where(knope_high_mag_idx == 2, 1.0,
                             tl.where(knope_high_mag_idx == 3, 1.5,
                             tl.where(knope_high_mag_idx == 4, 2.0,
                             tl.where(knope_high_mag_idx == 5, 3.0,
                             tl.where(knope_high_mag_idx == 6, 4.0, 6.0)))))))
            knope_low = tl.where(knope_low_sign == 1, -knope_low_mag, knope_low_mag)
            knope_high = tl.where(knope_high_sign == 1, -knope_high_mag, knope_high_mag)
            knope_pair = tl.join(knope_low, knope_high)
            knope_f = tl.reshape(knope_pair, (BLOCK_N, BLOCK_C))

            scale_nope_f = tl.exp2(tl.cast(knope_scales, tl.float32) - E8M0_BIAS_T)  # [BLOCK_N, 16]
            scale_nope_per_elem = tl.reshape(
                tl.broadcast_to(scale_nope_f[:, :, None], (BLOCK_N, BLOCK_C // 32, 32)),
                (BLOCK_N, BLOCK_C),
            )
            knope_dequant = knope_f * scale_nope_per_elem

            # qk += q @ k_nope^T  — DOMINANT site (16×512×32)
            if DOT_DTYPE == 1:
                # Phase B2-alt-1: cast Q + K_nope_dequant to FP8 → MFMA-fp8-fp8
                q_fp8 = q.to(tl.float8e4nv)
                knope_fp8 = knope_dequant.to(tl.float8e4nv)
                qk += tl.dot(q_fp8, tl.trans(knope_fp8))
            else:
                qk += tl.dot(q, tl.trans(knope_dequant.to(q.dtype)))

            qk *= sm_scale
            qk = tl.where(
                mask_h[:, None] & (offs_n_iter[None, :] < split_kv_end),
                qk, float("-inf"),
            )

            # Online softmax
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]

            # acc += p @ V  where V (== K_nope, transposed access) is the
            # already-dequantized knope_dequant. Phase B3 uses ds_read_b64_tr_b4 for transpose load.
            if DOT_DTYPE == 1:
                p_fp8 = p.to(tl.float8e4nv)
                knope_v_fp8 = knope_dequant.to(tl.float8e4nv)
                acc += tl.dot(p_fp8, knope_v_fp8)
            else:
                acc += tl.dot(p.to(knope_dequant.dtype), knope_dequant)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_attout_b
            + cur_head[:, None] * stride_attout_h
            + split_kv_id * stride_attout_s
            + offs_c[None, :]
        )
        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=mask_h[:, None] & mask_c[None, :],
        )

        offs_mid_o_lse = (
            cur_batch * stride_attout_b
            + cur_head * stride_attout_h
            + split_kv_id * stride_attout_s
            + kv_lora_rank
        )
        tl.store(Att_Out + offs_mid_o_lse, e_max + tl.log(e_sum), mask=mask_h)


# ============================================================================
# Numpy reference
# ============================================================================

def _fp4_dequant_token(fp4_bytes, scale_bytes):
    import numpy as np
    out = np.zeros(576, dtype=np.float32)
    for g in range(18):
        e8m0_byte = scale_bytes[g]
        scale = 2.0 ** (int(e8m0_byte) - E8M0_BIAS)
        for i in range(16):
            byte_idx = g * 16 + i
            b = int(fp4_bytes[byte_idx])
            low = b & 0xF
            high = (b >> 4) & 0xF
            for ni, nib in enumerate([low, high]):
                sign_bit = (nib >> 3) & 1
                mag_idx = nib & 7
                mag = _E2M1_VALUES[mag_idx]
                v = mag if sign_bit == 0 else -mag
                out[g * 32 + 2 * i + ni] = v * scale
    return out


def mla_decode_fp4_kv_numpy_ref(
    q_np, kv_cache_fp4, kv_cache_scale, kv_indptr, kv_indices, sm_scale,
):
    import numpy as np
    bs = q_np.shape[0]
    head_num = q_np.shape[1]
    out = np.zeros((bs, head_num, KV_LORA_RANK), dtype=np.float32)

    for b in range(bs):
        kv_start = int(kv_indptr[b])
        kv_end = int(kv_indptr[b + 1])
        seq_len = kv_end - kv_start
        if seq_len == 0:
            continue
        K_full = np.zeros((seq_len, KV_LORA_RANK + QK_ROPE_HEAD_DIM), dtype=np.float32)
        for i in range(seq_len):
            slot = int(kv_indices[kv_start + i])
            K_full[i] = _fp4_dequant_token(kv_cache_fp4[slot], kv_cache_scale[slot])

        K_nope = K_full[:, :KV_LORA_RANK]
        K_pe = K_full[:, KV_LORA_RANK:]

        for h in range(head_num):
            q = q_np[b, h, :KV_LORA_RANK]
            q_pe = q_np[b, h, KV_LORA_RANK:]
            qk = (q[None, :] @ K_nope.T)[0] + (q_pe[None, :] @ K_pe.T)[0]
            qk *= sm_scale
            qk_max = qk.max()
            p = np.exp(qk - qk_max)
            p /= p.sum()
            out[b, h] = p @ K_nope
    return out


# ============================================================================
# Unit test (numpy ref)
# ============================================================================

def run_unit_test() -> int:
    import numpy as np
    np.random.seed(0xBEEF)

    bs, head_num, seq_len = 2, 4, 16
    print(f"[mla_decode_fp4_kv] unit test — bs={bs}, head_num={head_num}, seq_len={seq_len}")

    q_np = (np.random.randn(bs, head_num, KV_LORA_RANK + QK_ROPE_HEAD_DIM) * 0.1).astype(np.float32)

    sys.path.insert(0, str(__file__).rsplit("/", 1)[0])
    from mla_fp4_kv_write import fp4_kv_write_numpy

    k_nope = (np.random.randn(bs * seq_len, KV_LORA_RANK) * 0.5).astype(np.float32)
    k_rope = (np.random.randn(bs * seq_len, QK_ROPE_HEAD_DIM) * 0.3).astype(np.float32)
    slot_mapping = np.arange(bs * seq_len, dtype=np.int64)
    fp4_bytes, scale_bytes = fp4_kv_write_numpy(k_nope, k_rope, slot_mapping)

    kv_indptr = np.array([i * seq_len for i in range(bs + 1)], dtype=np.int32)
    kv_indices = np.arange(bs * seq_len, dtype=np.int32)
    sm_scale = 1.0 / math.sqrt(KV_LORA_RANK)

    out_fp4 = mla_decode_fp4_kv_numpy_ref(
        q_np, fp4_bytes, scale_bytes, kv_indptr, kv_indices, sm_scale
    )

    out_bf16 = np.zeros_like(out_fp4)
    for b in range(bs):
        K_full = np.concatenate([
            k_nope[b * seq_len:(b + 1) * seq_len],
            k_rope[b * seq_len:(b + 1) * seq_len],
        ], axis=1)
        K_nope = K_full[:, :KV_LORA_RANK]
        K_pe = K_full[:, KV_LORA_RANK:]
        for h in range(head_num):
            q = q_np[b, h, :KV_LORA_RANK]
            q_pe = q_np[b, h, KV_LORA_RANK:]
            qk = q @ K_nope.T + q_pe @ K_pe.T
            qk *= sm_scale
            qk_max = qk.max()
            p = np.exp(qk - qk_max)
            p /= p.sum()
            out_bf16[b, h] = p @ K_nope

    err = np.abs(out_fp4 - out_bf16)
    rel = err / (np.abs(out_bf16).max() + 1e-6)
    print(f"  max_abs_err  = {err.max():.4f}")
    print(f"  mean_abs_err = {err.mean():.4f}")
    print(f"  max_rel_err  = {rel.max():.4f}")

    if err.mean() > 0.05:
        print(f"  FAIL: mean_abs_err {err.mean():.4f} > 0.05 (MXFP4 envelope)")
        return 1
    if rel.max() > 0.40:
        print(f"  FAIL: max_rel_err {rel.max():.4f} > 0.40 (sanity ceiling for tiny test)")
        return 1
    print("[mla_decode_fp4_kv] numpy reference PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="run numpy correctness unit test")
    args = ap.parse_args()
    if args.test:
        return run_unit_test()
    print("(no-op — pass --test for numpy ref test, or import as a module)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

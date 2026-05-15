"""
mla_decode_fp4_kv_stage2 — logsumexp combine across NUM_KV_SPLITS partial outputs

Pairs with `_decode_fp4_kv_stage1` in `mla_decode_fp4_kv.py`. Stage 1 produces
partial outputs at shape [bs, head, NUM_KV_SPLITS, kv_lora_rank+1] where the
last column is `e_max + log(e_sum)` per split. Stage 2 combines these via
numerically stable logsumexp into the final output [bs, head, kv_lora_rank].

Reference: aiter/ops/triton/mla_decode_rope.py:_fwd_grouped_kernel_stage2.
"""
from __future__ import annotations

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


if HAS_TRITON:

    @triton.jit
    def _decode_fp4_kv_stage2(
        Att_Out,             # f32 [bs, head, num_kv_splits, kv_lora+1]
        Final_Out,           # bf16 [bs, head, kv_lora]
        stride_att_b: tl.constexpr,
        stride_att_h: tl.constexpr,
        stride_att_s: tl.constexpr,
        stride_fin_b: tl.constexpr,
        stride_fin_h: tl.constexpr,
        bs: tl.constexpr,
        head_num: tl.constexpr,
        kv_lora_rank: tl.constexpr,
        NUM_KV_SPLITS: tl.constexpr,
        BLOCK_C: tl.constexpr,         # = kv_lora_rank
    ):
        """One program per (batch, head). Reduce across NUM_KV_SPLITS axis.

        For each (b, h):
            lse_max = max(lse[s] for s in 0..NUM_KV_SPLITS)
            den     = sum(exp(lse[s] - lse_max))
            num[c]  = sum(exp(lse[s] - lse_max) * partial_out[s, c])
            out[c]  = num[c] / den
        """
        cur_batch = tl.program_id(0)
        cur_head = tl.program_id(1)

        offs_c = tl.arange(0, BLOCK_C)
        mask_c = offs_c < kv_lora_rank

        # First pass: find lse_max
        lse_max = tl.full((), float("-inf"), dtype=tl.float32)
        for s in tl.static_range(0, NUM_KV_SPLITS):
            lse_off = (
                cur_batch * stride_att_b
                + cur_head * stride_att_h
                + s * stride_att_s
                + kv_lora_rank
            )
            lse = tl.load(Att_Out + lse_off)
            lse_max = tl.maximum(lse_max, lse)

        # Second pass: combine
        num = tl.zeros((BLOCK_C,), dtype=tl.float32)
        den = tl.zeros((), dtype=tl.float32)
        for s in tl.static_range(0, NUM_KV_SPLITS):
            base = (
                cur_batch * stride_att_b
                + cur_head * stride_att_h
                + s * stride_att_s
            )
            partial = tl.load(Att_Out + base + offs_c, mask=mask_c, other=0.0)
            lse = tl.load(Att_Out + base + kv_lora_rank)
            w = tl.exp(lse - lse_max)
            num += w * partial
            den += w

        out = num / den
        out_off = (
            cur_batch * stride_fin_b
            + cur_head * stride_fin_h
            + offs_c
        )
        tl.store(Final_Out + out_off, out.to(Final_Out.dtype.element_ty), mask=mask_c)


    def launch_stage2(att_out, final_out, num_kv_splits):
        """Host-side launcher."""
        bs, head_num = final_out.shape[0], final_out.shape[1]
        kv_lora_rank = final_out.shape[2]
        # Round BLOCK_C up to next power of 2 (Triton requirement)
        BLOCK_C = 1
        while BLOCK_C < kv_lora_rank:
            BLOCK_C *= 2
        grid = (bs, head_num)
        _decode_fp4_kv_stage2[grid](
            att_out, final_out,
            stride_att_b=att_out.stride(0),
            stride_att_h=att_out.stride(1),
            stride_att_s=att_out.stride(2),
            stride_fin_b=final_out.stride(0),
            stride_fin_h=final_out.stride(1),
            bs=bs, head_num=head_num,
            kv_lora_rank=kv_lora_rank,
            NUM_KV_SPLITS=num_kv_splits,
            BLOCK_C=BLOCK_C,
            num_warps=4,
            num_stages=1,
        )

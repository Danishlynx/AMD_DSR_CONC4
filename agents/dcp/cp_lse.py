"""DCP=4 Stage 4 — LSE-renormalized cross-rank attention combine.

Ported from vLLM upstream `vllm/v1/attention/ops/common.py:1-256` (PR #23734).
AMD/CDNA4 portable: pure Triton, no MFMA, gfx950-supported ops only.

We use the **all-reduce variant** (`cp_lse_ag_out_ar`), not reduce-scatter.

Why AR vs RS for ATOM:
  ATOM at TP=4 already gives each rank 8 heads (TP-sharded). DCP shards KV
  (sequence dim), NOT heads. After local-shard attention each rank has
  [B, 8, D] for its KV-shard partial. To combine, we LSE-renormalize each
  rank's local output then ALL-REDUCE across DCP ranks. Output shape
  unchanged at [B, 8, D]. No head-resharding required.
  vLLM's RS variant assumes heads ALSO sharded by DCP (additional 1/N).
  That's a different architectural choice.

Install path: /app/aiter-test/aiter/ops/triton/cp_lse.py
Caller: /app/ATOM/atom/model_ops/attention_mla.py (post-MLA-decode wire)
"""

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except ImportError:
    _TRITON_OK = False


@triton.jit
def _correct_attn_cp_out_kernel(
    outputs_ptr,
    new_output_ptr,
    lses_ptr,
    vlse_ptr,
    outputs_stride_B,
    outputs_stride_H,
    outputs_stride_D,
    lses_stride_N,
    lses_stride_B,
    lses_stride_H,
    lse_idx,
    HEAD_DIM: tl.constexpr,
    N_ROUNDED: tl.constexpr,
    IS_BASE_E: tl.constexpr,
):
    """LSE-renormalize each rank's partial attention output.

    Args:
        outputs_ptr  : [B, H, D] local rank's attention output (read & write — corrected in place ok)
        new_output_ptr: [B, H, D] write target (can alias outputs_ptr)
        lses_ptr     : [N, B, H] all-gathered LSEs across CP ranks
        vlse_ptr     : [B, H]   final LSE write target
        lse_idx      : this rank's index into N (cp_rank)
    """
    batch_idx = tl.program_id(axis=0).to(tl.int64)
    head_idx = tl.program_id(axis=1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)
    num_n_offsets = tl.arange(0, N_ROUNDED)

    lse_offsets = (
        num_n_offsets * lses_stride_N
        + batch_idx * lses_stride_B
        + head_idx * lses_stride_H
    )

    lse = tl.load(lses_ptr + lse_offsets)
    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0, lse_max)
    lse -= lse_max
    if IS_BASE_E:
        lse_exp = tl.exp(lse)
        lse_acc = tl.sum(lse_exp, axis=0)
        lse = tl.log(lse_acc)
    else:
        lse_exp = tl.exp2(lse)
        lse_acc = tl.sum(lse_exp, axis=0)
        lse = tl.log2(lse_acc)
    lse += lse_max

    lse_offsets_out = batch_idx * lses_stride_B + head_idx * lses_stride_H
    tl.store(vlse_ptr + lse_offsets_out, lse)

    output_offsets = (
        batch_idx * outputs_stride_B
        + head_idx * outputs_stride_H
        + d_offsets * outputs_stride_D
    )

    lse_offset = (
        lse_idx * lses_stride_N + batch_idx * lses_stride_B + head_idx * lses_stride_H
    )
    lse_tmp = tl.load(lses_ptr + lse_offset)
    lse_finally = lse_tmp - lse
    lse_finally = tl.where(
        (lse_finally != lse_finally) | (lse_finally == float("inf")),
        -float("inf"),
        lse_finally,
    )
    factor = tl.exp(lse_finally) if IS_BASE_E else tl.exp2(lse_finally)
    output = tl.load(outputs_ptr + output_offsets)
    output = output * factor
    tl.store(new_output_ptr + output_offsets, output)


class CPTritonContext:
    """Reuse compiled kernel across calls (Triton JIT is expensive)."""
    def __init__(self):
        self.inner_kernel = None

    def call_kernel(self, kernel, grid, *regular_args, **const_args):
        if self.inner_kernel is None:
            self.inner_kernel = kernel[grid](*regular_args, **const_args)
        else:
            self.inner_kernel[grid](*regular_args)


_default_ctx = CPTritonContext()


def correct_attn_out(
    out: torch.Tensor,
    lses: torch.Tensor,
    cp_rank: int,
    ctx: CPTritonContext = None,
    is_lse_base_on_e: bool = True,
):
    """LSE-renormalize the local rank's partial attention output.

    Args:
        out: [B, H, D] (or [B, 1, H, D] — auto-squeezed)
        lses: [N, B, H] (or [N, B, H, 1] — auto-squeezed)
        cp_rank: this rank's index into N

    Returns: (corrected_out: [B, H, D], final_lse: [B, H])
    """
    if ctx is None:
        ctx = _default_ctx

    if out.ndim == 4 and out.shape[1] == 1:
        out = out.squeeze(1)
    assert out.ndim == 3, f"expected out [B,H,D], got {tuple(out.shape)}"

    if lses.ndim == 4 and lses.shape[-1] == 1:
        lses = lses.squeeze(-1)
    if lses.ndim == 4 and lses.shape[1] == 1:
        lses = lses.squeeze(1)
    assert lses.ndim == 3, f"expected lses [N,B,H], got {tuple(lses.shape)}"

    B, H, D = out.shape
    N = lses.shape[0]

    o_sB, o_sH, o_sD = out.stride()
    l_sN, l_sB, l_sH = lses.stride()

    lse = torch.empty_strided(
        (B, H), (l_sB, l_sH), device=lses.device, dtype=lses.dtype
    )

    grid = (B, H, 1)
    regular_args = (
        out, out, lses, lse,
        o_sB, o_sH, o_sD,
        l_sN, l_sB, l_sH,
        cp_rank,
    )
    const_args = {"HEAD_DIM": D, "N_ROUNDED": N, "IS_BASE_E": is_lse_base_on_e}
    ctx.call_kernel(_correct_attn_cp_out_kernel, grid, *regular_args, **const_args)
    return out, lse


def cp_lse_ag_out_ar(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group,
    ctx: CPTritonContext = None,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
):
    """All-gather LSEs, LSE-renormalize local out, then ALL-REDUCE across CP ranks.

    Use this variant when heads are NOT sharded by CP (only KV is sharded).
    Output shape unchanged: [B, H, D].

    Args:
        cp_attn_out: [B, H, D] local rank's partial attention output
        cp_attn_lse: [B, H]   local rank's per-position LSE
        cp_group: GroupCoordinator (DCP group accessor)

    Returns: combined attention output [B, H, D] (and optional lse [B, H])
    """
    if cp_group.world_size == 1:
        if return_lse:
            return cp_attn_out, cp_attn_lse
        return cp_attn_out

    if ctx is None:
        ctx = _default_ctx

    cp_attn_lse = cp_attn_lse.contiguous()
    # All-gather LSEs along a new leading rank-dim → [N, B, H]
    lses = cp_group.all_gather(cp_attn_lse, dim=0).reshape(
        (cp_group.world_size,) + cp_attn_lse.shape
    )
    out, lse = correct_attn_out(
        cp_attn_out,
        lses,
        cp_group.rank_in_group,
        ctx,
        is_lse_base_on_e=is_lse_base_on_e,
    )
    # All-reduce sums the corrected partials across CP ranks
    out = cp_group.all_reduce(out)

    if return_lse:
        return out, lse
    return out

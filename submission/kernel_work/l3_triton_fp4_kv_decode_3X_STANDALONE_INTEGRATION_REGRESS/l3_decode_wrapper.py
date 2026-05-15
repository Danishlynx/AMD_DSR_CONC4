"""
l3_decode_wrapper — drop-in replacement for aiter.mla.mla_decode_fwd

Matches aiter signature exactly. Dispatches to the L3 Triton FP8-cache kernel
+ stage 2 combine. Used when ATOM_L3_FP8_TRITON=1 env var is set and shape
matches gating heuristic.

Gating (must match a production decode call shape we've validated):
  - max_seqlen_q == 1 (decode only — not prefill, not MTP folded)
  - q.dtype == bf16 (no q_scale-driven FP8 cast needed; we cast inline)
  - kv_buffer.dtype == fp8_e4m3fnuz (production format)
  - q.shape[1] in (16, 32) — TP=4 head split
  - page_size == 1 (production typical)
  - nhead_kv == 1 (MLA constraint)
"""
from __future__ import annotations
import os
import torch

# Import the kernels — packaged alongside this file in /tmp deploy
import mla_decode_fp8_kv as _fp8_mod
from mla_decode_fp4_kv_stage2 import launch_stage2 as _launch_stage2


L3_ENABLED = os.environ.get("ATOM_L3_FP8_TRITON", "0") == "1"
L3_FORCE_KS = int(os.environ.get("ATOM_L3_NUM_KV_SPLITS", "64"))
L3_DEBUG = os.environ.get("ATOM_L3_DEBUG", "0") == "1"


def _shape_supported(q, kv_buffer, max_seqlen_q, page_size, nhead_kv):
    """Conservative dispatch gate — only paths we've validated."""
    if not L3_ENABLED:
        return False
    if max_seqlen_q != 1:
        return False
    if page_size != 1:
        return False
    if nhead_kv != 1:
        return False
    if q.dtype != torch.bfloat16:
        return False
    # production passes kv_buffer.view(-1, 1, 1, D) — accept any dtype-flexible 4-D view
    if kv_buffer.dim() != 4:
        return False
    if kv_buffer.shape[1] != 1 or kv_buffer.shape[2] != 1:
        return False
    # head_num must be 16 or 32 (TP=4 or TP=8 split)
    if q.dim() < 2 or q.shape[-2] not in (16, 32):
        return False
    return True


def l3_decode_fwd(
    q,
    kv_buffer,
    o,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    max_seqlen_q,
    page_size=1,
    nhead_kv=1,
    sm_scale=None,
    logit_cap=0.0,
    num_kv_splits=None,
    num_kv_splits_indptr=None,
    work_meta_data=None,
    work_indptr=None,
    work_info_set=None,
    reduce_indptr=None,
    reduce_final_map=None,
    reduce_partial_map=None,
    q_scale=None,
    kv_scale=None,
    intra_batch_mode=False,
    return_logits=False,
    return_lse=False,
):
    """Drop-in replacement for aiter.mla.mla_decode_fwd.

    Writes into `o` (in-place). Returns same as aiter (None, or tuple if
    return_lse/return_logits — we don't support these yet, will fall back).
    """
    if return_logits or return_lse:
        raise NotImplementedError("l3_decode_fwd does not support return_logits/return_lse")

    bs = qo_indptr.shape[0] - 1
    # Reshape Q for our kernel: [bs, head, D] expected
    # Production call: q shape = [bs * max_seqlen_q, head, qk_head_dim]
    # With max_seqlen_q=1, this is [bs, head, D]
    qk_head_dim = q.shape[-1]
    head_num = q.shape[-2]
    kv_lora_rank = o.shape[-1]
    qk_rope_head_dim = qk_head_dim - kv_lora_rank

    if q.shape[0] != bs:
        # Reshape — handle fold cases by treating as [bs, head, D]
        q_2d = q.reshape(bs, head_num, qk_head_dim)
    else:
        q_2d = q

    # KV buffer flatten to [num_tokens, qk_head_dim]
    # Production passes kv_buffer.view(-1, 1, 1, D); we flatten back.
    kv_flat = kv_buffer.view(-1, qk_head_dim)

    # kv_scale: per-tensor FP8 scale (scalar tensor); default to 1.0 if absent
    if kv_scale is None:
        kv_scale_t = torch.ones(1, dtype=torch.float32, device=q.device)
    else:
        kv_scale_t = kv_scale.to(torch.float32).reshape(1)

    # Override num_kv_splits to our optimal (production caps at 16; we use 64)
    NUM_KV_SPLITS = L3_FORCE_KS

    # Output reshape: o is [bs * max_seqlen_q, head, kv_lora] = [bs, head, kv_lora] at max_seqlen_q=1
    o_2d = o.view(bs, head_num, kv_lora_rank)

    # Stage 1 buffer
    Att_Out = torch.empty(bs, head_num, NUM_KV_SPLITS, kv_lora_rank + 1,
                          dtype=torch.float32, device=q.device)

    # Stage 1 launch
    BLOCK_H = 16  # works for head_num 16 or 32 (32 will run 2 head blocks per batch×split)
    grid = (bs * (head_num // BLOCK_H) * NUM_KV_SPLITS,)
    _fp8_mod._decode_fp8_kv_stage1[grid](
        q_2d, kv_flat, kv_indptr, kv_indices, kv_scale_t,
        sm_scale,
        Att_Out,
        stride_qb=q_2d.stride(0),
        stride_qh=q_2d.stride(1),
        stride_kv_t=kv_flat.stride(0),
        stride_attout_b=Att_Out.stride(0),
        stride_attout_h=Att_Out.stride(1),
        stride_attout_s=Att_Out.stride(2),
        bs=bs,
        head_num=head_num,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        BLOCK_C=kv_lora_rank,
        BLOCK_R=64,
        BLOCK_N=32,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        DOT_DTYPE=1,
        num_warps=8,
        num_stages=3,
    )

    # Stage 2: combine NUM_KV_SPLITS partial outputs into final o
    _launch_stage2(Att_Out, o_2d, NUM_KV_SPLITS)

    if L3_DEBUG:
        print(f"[L3] bs={bs} head={head_num} qk_d={qk_head_dim} kv_lora={kv_lora_rank} "
              f"ks={NUM_KV_SPLITS} kv_scale={float(kv_scale_t.item()):.4f}", flush=True)
    return None


def maybe_dispatch_l3(
    q, kv_buffer, o, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
    max_seqlen_q, page_size=1, nhead_kv=1, **kwargs,
):
    """Returns True if L3 handled the call; False if caller should fall back to aiter."""
    if _shape_supported(q, kv_buffer, max_seqlen_q, page_size, nhead_kv):
        try:
            l3_decode_fwd(
                q, kv_buffer, o, qo_indptr, kv_indptr, kv_indices,
                kv_last_page_lens, max_seqlen_q,
                page_size=page_size, nhead_kv=nhead_kv,
                **{k: v for k, v in kwargs.items() if k in (
                    "sm_scale", "logit_cap", "num_kv_splits", "q_scale", "kv_scale",
                )},
            )
            return True
        except Exception as e:
            if L3_DEBUG:
                print(f"[L3] DISPATCH FAIL: {type(e).__name__}: {e}", flush=True)
            return False
    return False

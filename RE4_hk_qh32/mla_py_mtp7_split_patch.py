#!/usr/bin/env python3
"""Patch /app/aiter-test/aiter/mla.py to add MTP=7 python-split shim.

At nhead=32, sq>4, fp8/fp8, we intercept mla_decode_fwd and split into sq=1 calls.
Each sq=1 call rebuilds its own metadata via get_mla_metadata_v1. This avoids
the ASM kernel's `qo_len <= 4` check that blocks MTP=7 natively.

Idempotent: checks for marker comment before patching.
"""
import sys, re

target = "/app/aiter-test/aiter/mla.py"
marker = "# MTP7_PYTHON_SPLIT_SHIM"

with open(target, 'r') as f:
    src = f.read()

if marker in src:
    print("patch already applied, skipping")
    sys.exit(0)

# Find the function signature, insert shim RIGHT after device = q.device at line 183.
# Insert a check: if nhead == 32 and max_seqlen_q in (5,6,7,8) and fp8/fp8 and persistent_mode:
#   split into sq=1 calls.
# We need access to get_mla_metadata_v1 and get_mla_metadata_info_v1 which are in aiter.

anchor = "    device = q.device\n    assert logit_cap <= 0"
if anchor not in src:
    raise RuntimeError("anchor not found in mla.py")

shim = '''    device = q.device
    # MTP7_PYTHON_SPLIT_SHIM: at sq>4 with nhead==32 fp8/fp8, native ASM rejects qo_len>4.
    # We split into max_seqlen_q parallel sq=1 calls with fresh metadata per position.
    # Cost: ~40-80us launch overhead x max_seqlen_q, typically +0.3-0.6ms/step.
    # Gain: unlocks MTP=N where N = max_seqlen_q - 1, giving ~N/4x throughput.
    if (max_seqlen_q in (5, 6, 7, 8)
        and q.size(1) == 32
        and q.dtype == dtypes.fp8
        and kv_buffer.dtype == dtypes.fp8):
        import os as _os
        if _os.environ.get("ATOM_MTP7_PYTHON_SPLIT", "0") != "0":
            return _mtp7_python_split_dispatch(
                q, kv_buffer, o, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
                max_seqlen_q, page_size, nhead_kv, sm_scale, logit_cap,
                num_kv_splits, num_kv_splits_indptr,
                work_meta_data, work_indptr, work_info_set,
                reduce_indptr, reduce_final_map, reduce_partial_map,
                q_scale, kv_scale, intra_batch_mode, return_logits, return_lse,
            )
    assert logit_cap <= 0'''

new_src = src.replace(anchor, shim)

# Insert helper function _mtp7_python_split_dispatch at top of file after imports.
helper = '''

def _mtp7_python_split_dispatch(
    q, kv_buffer, o, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
    max_seqlen_q, page_size, nhead_kv, sm_scale, logit_cap,
    num_kv_splits, num_kv_splits_indptr,
    work_meta_data, work_indptr, work_info_set,
    reduce_indptr, reduce_final_map, reduce_partial_map,
    q_scale, kv_scale, intra_batch_mode, return_logits, return_lse,
):
    """Split a sq=N decode call (N ∈ {5,6,7,8}) into N independent sq=1 calls.
    Each sq=1 call gets fresh metadata. Output is assembled back into `o`.
    """
    import torch
    from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
    device = q.device
    total_q = q.size(0)
    nhead = q.size(1)
    head_dim = q.size(2)
    v_head_dim = o.size(2)
    batch_size = qo_indptr.size(0) - 1
    # q shape [total_q = bs * sq, nhead, head_dim]; per-position reshape.
    q_bs = q.view(batch_size, max_seqlen_q, nhead, head_dim)
    o_bs = o.view(batch_size, max_seqlen_q, nhead, v_head_dim)
    # Compute sq=1 metadata ONCE (shared structure; only cu_seqlens_q differs).
    # Simpler: rebuild per call.
    kv_seqlens = (kv_indptr[1:] - kv_indptr[:-1]).tolist()
    kv_last_page_lens_sq1 = torch.ones(batch_size, dtype=torch.int32, device=device)
    for pos in range(max_seqlen_q):
        q_pos = q_bs[:, pos, :, :].contiguous().view(batch_size, nhead, head_dim)
        o_pos_buf = torch.empty_like(o_bs[:, pos, :, :].contiguous().view(batch_size, nhead, v_head_dim))
        cu_seqlens_q_sq1 = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        # Build metadata for sq=1 on this batch.
        sz = get_mla_metadata_info_v1(
            batch_size, 1, nhead,
            torch.bfloat16, q.dtype, is_sparse=False, fast_mode=True,
        )
        (wmd_sz, wmd_dt), (widx_sz, widx_dt), (wis_sz, wis_dt), \\
            (ri_sz, ri_dt), (rfm_sz, rfm_dt), (rpm_sz, rpm_dt) = sz
        wmd_ = torch.empty(wmd_sz, dtype=wmd_dt, device=device)
        widx_ = torch.empty(widx_sz, dtype=widx_dt, device=device)
        wis_ = torch.empty(wis_sz, dtype=wis_dt, device=device)
        ri_ = torch.empty(ri_sz, dtype=ri_dt, device=device)
        rfm_ = torch.empty(rfm_sz, dtype=rfm_dt, device=device)
        rpm_ = torch.empty(rpm_sz, dtype=rpm_dt, device=device)
        get_mla_metadata_v1(
            cu_seqlens_q_sq1, kv_indptr, kv_last_page_lens_sq1,
            nhead, 1, True,
            wmd_, wis_, widx_, ri_, rfm_, rpm_,
            page_size=page_size,
            dtype_q=q.dtype, dtype_kv=kv_buffer.dtype,
            kv_granularity=16,
            max_seqlen_qo=1,
            uni_seqlen_qo=1,
            fast_mode=1,
            max_split_per_batch=16,
        )
        mla_decode_fwd(
            q_pos, kv_buffer, o_pos_buf,
            cu_seqlens_q_sq1, kv_indptr, kv_indices, kv_last_page_lens_sq1,
            1,
            page_size=page_size,
            nhead_kv=nhead_kv,
            sm_scale=sm_scale,
            work_meta_data=wmd_, work_indptr=widx_, work_info_set=wis_,
            reduce_indptr=ri_, reduce_final_map=rfm_, reduce_partial_map=rpm_,
            q_scale=q_scale, kv_scale=kv_scale,
        )
        o_bs[:, pos, :, :] = o_pos_buf.view(batch_size, nhead, v_head_dim)
    return None, None
'''

# Insert helper before the `def mla_decode_fwd(` definition.
fn_anchor = "def mla_decode_fwd("
if fn_anchor not in new_src:
    raise RuntimeError("def mla_decode_fwd not found")
new_src = new_src.replace(fn_anchor, helper + "\n\n" + fn_anchor, 1)

with open(target, 'w') as f:
    f.write(new_src)

print(f"patched {target} with python split shim")

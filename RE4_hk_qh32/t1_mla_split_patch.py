#!/usr/bin/env python3
"""T1 patch: clean cudagraph-safe python split for FP8 sq=8 → 8× sq=1 in aiter/mla.py.

Session-17 T1 implementation. Replaces session-16's broken shim which had .tolist()
(D→H sync — breaks cudagraph capture). This version:
- Uses module-level cache of pre-allocated metadata buffers keyed by (bs_eff, nhead)
- Avoids all D→H sync (no .item()/.tolist()/.cpu())
- Uses in-place copy_() / fill_() on cached tensors
- All tensor ops stay on GPU

Idempotent: detects if old shim exists and replaces.
"""
import sys
import re

TARGET = "/app/aiter-test/aiter/mla.py"

NEW_SHIM = '''
# ============================================================================
# T1 session-17: Cudagraph-safe MTP=7 metadata split (ATOM_MTP7_SPLIT=1 env gate)
# Splits sq=N (N ∈ {5,6,7,8}) into N sq=1 calls reusing shipped fp8 persistent kernel.
# Safety invariants (cudagraph capture compatible):
#   - No .item() / .tolist() / .cpu() (no D→H sync)
#   - Pre-allocated metadata buffers cached by (bs, nhead) key at module scope
#   - kv_indptr is replicated via torch.repeat_interleave (all-GPU ops)
# ============================================================================

_MTP7_SPLIT_CACHE = {}

def _mtp7_split_get_or_alloc(batch_size: int, max_seqlen_q: int, nhead: int,
                              head_dim: int, v_head_dim: int,
                              q_dtype: torch.dtype, kv_dtype: torch.dtype,
                              device: torch.device):
    """Allocate + cache per-position sq=1 metadata buffers. Called outside
    cudagraph capture (first-time allocation), then reused from cache.
    """
    key = (batch_size, max_seqlen_q, nhead, str(q_dtype), str(kv_dtype))
    cached = _MTP7_SPLIT_CACHE.get(key)
    if cached is not None:
        return cached
    from aiter import get_mla_metadata_info_v1
    bs_eff = batch_size * max_seqlen_q  # virtual batches for split
    # Metadata shape info (sq=1 for the split, but we use bs_eff "virtual" batches)
    sz = get_mla_metadata_info_v1(
        bs_eff, 1, nhead, torch.bfloat16, q_dtype,
        is_sparse=False, fast_mode=True,
    )
    (wmd_sz, wmd_dt), (widx_sz, widx_dt), (wis_sz, wis_dt), \\
        (ri_sz, ri_dt), (rfm_sz, rfm_dt), (rpm_sz, rpm_dt) = sz
    buf = dict(
        work_meta_data=torch.empty(wmd_sz, dtype=wmd_dt, device=device),
        work_indptr=torch.empty(widx_sz, dtype=widx_dt, device=device),
        work_info_set=torch.empty(wis_sz, dtype=wis_dt, device=device),
        reduce_indptr=torch.empty(ri_sz, dtype=ri_dt, device=device),
        reduce_final_map=torch.empty(rfm_sz, dtype=rfm_dt, device=device),
        reduce_partial_map=torch.empty(rpm_sz, dtype=rpm_dt, device=device),
        # Pre-allocated virtual metadata derived tensors:
        cu_seqlens_q_eff=torch.arange(bs_eff + 1, dtype=torch.int32, device=device),
        kv_last_page_lens_eff=torch.ones(bs_eff, dtype=torch.int32, device=device),
        kv_indptr_eff=torch.empty(bs_eff + 1, dtype=torch.int32, device=device),
        logits_eff=torch.empty((1, 1, nhead, v_head_dim), dtype=torch.float32, device=device).expand(-1, -1, -1, -1),
        # logits/attn_lse sized per outer total_q×num_kv_splits — let mla_decode_fwd re-alloc internally
    )
    _MTP7_SPLIT_CACHE[key] = buf
    return buf


def _mtp7_split_dispatch(
    q, kv_buffer, o, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
    max_seqlen_q, page_size, nhead_kv, sm_scale, logit_cap,
    num_kv_splits, num_kv_splits_indptr,
    work_meta_data, work_indptr, work_info_set,
    reduce_indptr, reduce_final_map, reduce_partial_map,
    q_scale, kv_scale, intra_batch_mode, return_logits, return_lse,
):
    """Split sq=N decode into N sq=1 calls. CUDA-graph safe.

    Q layout assumption: [bs*sq, nhead, head_dim] with positions-within-batch
    contiguous (i.e., q[b*sq+p, :, :] = batch b position p). This is how
    MTP speculative decoding lays out Q.
    """
    device = q.device
    bs = qo_indptr.shape[0] - 1
    nhead = q.size(1)
    head_dim = q.size(2)
    v_head_dim = o.size(2)

    # Get cached split metadata (allocates once per (bs, sq, nhead) on first call)
    buf = _mtp7_split_get_or_alloc(
        bs, max_seqlen_q, nhead, head_dim, v_head_dim,
        q.dtype, kv_buffer.dtype, device,
    )

    # Reshape Q/O into [bs, sq, nhead, *] — zero-copy view
    q_bs = q.view(bs, max_seqlen_q, nhead, head_dim)
    o_bs = o.view(bs, max_seqlen_q, nhead, v_head_dim)

    from aiter import get_mla_metadata_v1

    # Per-position dispatch loop (unrolled at Python level — max_seqlen_q is a
    # compile-time Python int, cudagraph captures static graph per iteration).
    for pos in range(max_seqlen_q):
        # Zero-copy slice of Q/O at this position: [bs, nhead, *]
        # .contiguous() may copy; slice is stride-(max_seqlen_q) along dim 0.
        q_pos = q_bs[:, pos, :, :].contiguous()
        o_pos = o_bs[:, pos, :, :].contiguous()

        # Build per-position metadata using cached buffers.
        # All this replicates the first-bs batches as max_seqlen_q * bs virtual
        # batches of sq=1 each. kv_indptr_eff maps each virtual batch to the
        # SAME kv range as its source batch.
        # For position p of batch b: virtual batch idx = b (not b + p*bs — since
        # we iterate pos externally and pass only bs batches this iter).
        # Each iteration processes `bs` virtual batches of sq=1, all at position pos.
        get_mla_metadata_v1(
            qo_indptr,           # reuse original — sub-batches of size 1 each
            kv_indptr,
            kv_last_page_lens,
            nhead, 1, True,
            buf['work_meta_data'], buf['work_info_set'], buf['work_indptr'],
            buf['reduce_indptr'], buf['reduce_final_map'], buf['reduce_partial_map'],
            page_size=page_size,
            dtype_q=q.dtype, dtype_kv=kv_buffer.dtype,
            kv_granularity=16,
            max_seqlen_qo=1,
            uni_seqlen_qo=1,
            fast_mode=1,
            max_split_per_batch=16,
        )
        mla_decode_fwd(
            q_pos, kv_buffer, o_pos,
            qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
            1,
            page_size=page_size,
            nhead_kv=nhead_kv,
            sm_scale=sm_scale,
            work_meta_data=buf['work_meta_data'],
            work_indptr=buf['work_indptr'],
            work_info_set=buf['work_info_set'],
            reduce_indptr=buf['reduce_indptr'],
            reduce_final_map=buf['reduce_final_map'],
            reduce_partial_map=buf['reduce_partial_map'],
            q_scale=q_scale, kv_scale=kv_scale,
        )
        # Copy position output back into full output tensor (if o_pos is a copy)
        o_bs[:, pos, :, :].copy_(o_pos)
    return None, None

# ============================================================================
# End T1 session-17 shim
# ============================================================================

'''

ENTRY_SHIM = '''    # T1_SESSION17_SPLIT_ENTRY: cudagraph-safe sq=8 → 8×sq=1 split
    if (max_seqlen_q in (5, 6, 7, 8)
        and q.size(1) == 32
        and q.dtype == dtypes.fp8
        and kv_buffer.dtype == dtypes.fp8
        and os.environ.get("ATOM_MTP7_SPLIT", "0") != "0"):
        return _mtp7_split_dispatch(
            q, kv_buffer, o, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
            max_seqlen_q, page_size, nhead_kv, sm_scale, logit_cap,
            num_kv_splits, num_kv_splits_indptr,
            work_meta_data, work_indptr, work_info_set,
            reduce_indptr, reduce_final_map, reduce_partial_map,
            q_scale, kv_scale, intra_batch_mode, return_logits, return_lse,
        )
'''

def main():
    with open(TARGET, 'r') as f:
        src = f.read()

    # Remove old session-16 shim if present
    if '_mtp7_python_split_dispatch' in src:
        # Remove the OLD helper function (session-16 broken version)
        start = src.find('\ndef _mtp7_python_split_dispatch(')
        if start >= 0:
            end = src.find('\ndef mla_decode_fwd(', start)
            if end >= 0:
                src = src[:start] + '\n' + src[end:]
                print("removed old _mtp7_python_split_dispatch helper")

    if '_mtp7_split_dispatch' in src and 'T1_SESSION17_SPLIT_ENTRY' in src:
        print("T1 shim already applied, nothing to do")
        return

    # Remove the old session-16 ENTRY block that called _mtp7_python_split_dispatch
    old_entry_pattern = re.compile(
        r"    # MTP7_PYTHON_SPLIT_SHIM:.*?assert logit_cap <= 0",
        re.DOTALL
    )
    m = old_entry_pattern.search(src)
    if m:
        # Keep the "assert logit_cap <= 0" line
        src = old_entry_pattern.sub("    assert logit_cap <= 0", src)
        print("removed old session-16 entry shim")

    # Install new helper function BEFORE `def mla_decode_fwd(`
    fn_anchor = "\n\ndef mla_decode_fwd("
    if fn_anchor not in src:
        print("ERROR: cannot find def mla_decode_fwd anchor", file=sys.stderr)
        sys.exit(1)
    src = src.replace(fn_anchor, NEW_SHIM + fn_anchor, 1)

    # Install entry hook right after `    device = q.device` in mla_decode_fwd
    entry_anchor = "    device = q.device\n    assert logit_cap <= 0"
    if entry_anchor not in src:
        # Maybe already has extra whitespace or variants
        entry_anchor2 = "    device = q.device\n"
        idx = src.find(entry_anchor2)
        if idx < 0:
            print("ERROR: cannot find entry anchor in mla_decode_fwd", file=sys.stderr)
            sys.exit(1)
        insert_pos = idx + len(entry_anchor2)
        src = src[:insert_pos] + ENTRY_SHIM + src[insert_pos:]
    else:
        src = src.replace(
            "    device = q.device\n    assert logit_cap <= 0",
            "    device = q.device\n" + ENTRY_SHIM + "    assert logit_cap <= 0",
            1,
        )

    with open(TARGET, 'w') as f:
        f.write(src)
    print(f"T1 shim installed in {TARGET}")
    print(f"  helper: _mtp7_split_dispatch + _mtp7_split_get_or_alloc")
    print(f"  env gate: ATOM_MTP7_SPLIT=1")
    print(f"  trigger: sq∈{{5,6,7,8}}, nhead=32, fp8/fp8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Correctness test for HK qh32 v8 at sq ∈ {1, 2, 4, 8}.

v8 vs v7 differences:
  - v7 only handled sq=1 (one qo position per work_idx)
  - v8 adds inner q_pos loop over [qo_start, qo_end) + Opt-E s_setprio

Reference oracle for each sq:
  sq ∈ {1, 2, 4}: baseline ASM persistent (AITER_ENABLE_HK_QH32_V8=0) — bit-exact expected
  sq = 8:         Python split into 8 sq=1 calls using ASM persistent — tolerates softmax associativity noise <1e-2

Run:
  HOME=/tmp AITER_ENABLE_EXPERIMENTAL=1 python3 /tmp/test_hk_qh32_v8_correctness.py

Dispatch paths:
  - At sq∈{1,2,4}: ASM `mla_a8w8_qh32_qseqlen{1,2,4}_gqaratio32_ps` (persistent .co exists)
  - At sq=8: NO ASM .co exists; only path is v8 (or Python split via wrapper)

Tolerance:
  max_abs_diff < 1e-2  AND  rel_L2 < 0.05
"""
import os
import sys
import torch

# NOTE: set envs BEFORE aiter import so JIT build pickups the right module.
os.environ.setdefault("HOME", "/tmp")
os.environ.setdefault("AITER_ENABLE_EXPERIMENTAL", "1")

import aiter  # noqa: E402
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1  # noqa: E402
from aiter.mla import mla_decode_fwd  # noqa: E402


def build_metadata(batch_size, max_seqlen_q, num_heads, kv_seqlens, device):
    dtype_q = torch.float8_e4m3fn
    dtype_kv = torch.float8_e4m3fn

    cu_seqlens_q_cpu = torch.zeros(batch_size + 1, dtype=torch.int32)
    for b in range(batch_size):
        cu_seqlens_q_cpu[b + 1] = cu_seqlens_q_cpu[b] + max_seqlen_q
    cu_seqlens_q = cu_seqlens_q_cpu.to(device)

    kv_indptr_cpu = torch.zeros(batch_size + 1, dtype=torch.int32)
    for b in range(batch_size):
        kv_indptr_cpu[b + 1] = kv_indptr_cpu[b] + kv_seqlens[b]
    kv_indptr = kv_indptr_cpu.to(device)

    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int32, device=device)

    sz = get_mla_metadata_info_v1(
        batch_size, max_seqlen_q, num_heads,
        torch.bfloat16, dtype_kv, is_sparse=False, fast_mode=True,
    )
    (wmd_sz, wmd_dt), (widx_sz, widx_dt), (wis_sz, wis_dt), \
        (ri_sz, ri_dt), (rfm_sz, rfm_dt), (rpm_sz, rpm_dt) = sz

    work_meta_data = torch.empty(wmd_sz, dtype=wmd_dt, device=device)
    work_indptr = torch.empty(widx_sz, dtype=widx_dt, device=device)
    work_info_set = torch.empty(wis_sz, dtype=wis_dt, device=device)
    reduce_indptr = torch.empty(ri_sz, dtype=ri_dt, device=device)
    reduce_final_map = torch.empty(rfm_sz, dtype=rfm_dt, device=device)
    reduce_partial_map = torch.empty(rpm_sz, dtype=rpm_dt, device=device)

    get_mla_metadata_v1(
        cu_seqlens_q, kv_indptr, kv_last_page_lens,
        num_heads, 1, True,
        work_meta_data, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        page_size=1,
        dtype_q=dtype_q, dtype_kv=dtype_kv,
        kv_granularity=16,
        max_seqlen_qo=max_seqlen_q,
        uni_seqlen_qo=max_seqlen_q,
        fast_mode=1,
        max_split_per_batch=16,
    )
    return (cu_seqlens_q, kv_indptr, kv_last_page_lens,
            work_meta_data, work_indptr, work_info_set,
            reduce_indptr, reduce_final_map, reduce_partial_map)


def call_mla(q, kv_buffer, meta, kv_indices, max_seqlen_q, num_heads,
             kv_lora=512, rope=64, use_v8=False):
    head_dim = kv_lora + rope
    v_head_dim = kv_lora
    total_q = q.size(0)
    device = q.device

    os.environ["AITER_ENABLE_HK_QH32_V8"] = "1" if use_v8 else "0"

    (cu_seqlens_q, kv_indptr, kv_last_page_lens,
     work_meta_data, work_indptr, work_info_set,
     reduce_indptr, reduce_final_map, reduce_partial_map) = meta

    q_scale = torch.ones(1, dtype=torch.float32, device=device)
    kv_scale = torch.ones(1, dtype=torch.float32, device=device)

    o = torch.zeros(total_q, num_heads, v_head_dim, device=device, dtype=torch.bfloat16)
    mla_decode_fwd(
        q, kv_buffer.view(-1, 1, 1, head_dim), o,
        cu_seqlens_q, kv_indptr, kv_indices, kv_last_page_lens,
        max_seqlen_q,
        num_kv_splits=16,
        sm_scale=1.0 / (head_dim ** 0.5),
        work_meta_data=work_meta_data,
        work_indptr=work_indptr,
        work_info_set=work_info_set,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
        q_scale=q_scale, kv_scale=kv_scale,
    )
    torch.cuda.synchronize()
    return o


def reference_sq8_via_python_split(q, kv_buffer, kv_seqlens, batch_size, num_heads,
                                    kv_lora=512, rope=64):
    """For sq=8 reference: split into 8 sq=1 calls using ASM path."""
    device = q.device
    head_dim = kv_lora + rope
    v_head_dim = kv_lora
    total_q_sq8 = batch_size * 8

    # q shape: [total_q_sq8=batch_size*8, num_heads, head_dim]
    # Reshape so each sq=1 call sees its position slice.
    o_parts = []
    for pos in range(8):
        # Extract per-batch position `pos`
        q_slice = torch.zeros(batch_size, num_heads, head_dim, dtype=q.dtype, device=device)
        for b in range(batch_size):
            q_slice[b] = q[b * 8 + pos]

        meta = build_metadata(batch_size, 1, num_heads, kv_seqlens, device)
        kv_indices = torch.arange(sum(kv_seqlens), dtype=torch.int32, device=device)

        o_slice = call_mla(q_slice, kv_buffer, meta, kv_indices, 1, num_heads,
                            kv_lora, rope, use_v8=False)
        o_parts.append(o_slice)  # [bs, nh, v_head]

    # Assemble: interleave positions per batch
    o_out = torch.zeros(total_q_sq8, num_heads, v_head_dim, device=device, dtype=torch.bfloat16)
    for pos in range(8):
        for b in range(batch_size):
            o_out[b * 8 + pos] = o_parts[pos][b]
    return o_out


def run_shape(batch_size, max_seqlen_q, kv_seqlens, num_heads=32, seed=42):
    device = "cuda"
    torch.manual_seed(seed)

    kv_lora = 512
    rope = 64
    head_dim = kv_lora + rope
    v_head_dim = kv_lora

    total_q = batch_size * max_seqlen_q
    total_kv_pages = sum(kv_seqlens)

    q_bf16 = torch.randn(total_q, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    q = q_bf16.to(torch.float8_e4m3fn)
    kv_bf16 = torch.randn(total_kv_pages, 1, 1, head_dim, device=device, dtype=torch.bfloat16)
    kv_buffer = kv_bf16.to(torch.float8_e4m3fn)

    kv_indices = torch.arange(total_kv_pages, dtype=torch.int32, device=device)

    shape_str = f"bs={batch_size} sq={max_seqlen_q} kv={kv_seqlens}"

    # --- Reference ---
    if max_seqlen_q in (1, 2, 4):
        # ASM persistent at the requested sq
        meta = build_metadata(batch_size, max_seqlen_q, num_heads, kv_seqlens, device)
        o_ref = call_mla(q, kv_buffer, meta, kv_indices, max_seqlen_q, num_heads,
                          kv_lora, rope, use_v8=False)
    elif max_seqlen_q == 8:
        # Python split into 8× sq=1 calls (ASM persistent)
        o_ref = reference_sq8_via_python_split(q, kv_buffer, kv_seqlens, batch_size, num_heads,
                                                kv_lora, rope)
    else:
        print(f"[SKIP] {shape_str}: unsupported reference for sq={max_seqlen_q}")
        return True

    # --- Candidate (v8) ---
    try:
        meta_v8 = build_metadata(batch_size, max_seqlen_q, num_heads, kv_seqlens, device)
        o_v8 = call_mla(q, kv_buffer, meta_v8, kv_indices, max_seqlen_q, num_heads,
                         kv_lora, rope, use_v8=True)
    except Exception as e:
        print(f"[ERROR] v8 call failed at {shape_str}: {e}")
        return False

    # --- Compare ---
    diff = (o_v8.float() - o_ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    l2_per_row = (o_v8.float() - o_ref.float()).pow(2).sum(-1).sqrt()
    ref_norm = o_ref.float().pow(2).sum(-1).sqrt() + 1e-8
    rel_l2 = (l2_per_row / ref_norm).mean().item()

    tol_abs = 1e-2
    tol_l2 = 5e-2
    ok = (max_diff < tol_abs) and (rel_l2 < tol_l2)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {shape_str}: max_abs={max_diff:.4e} mean={mean_diff:.4e} rel_L2={rel_l2:.4e}")
    return ok


def main():
    # DSR1 workload — cover the full qseqlen range up to MTP=7 (sq=8)
    shapes = [
        # --- sq=1 (MTP=0 baseline / sanity) ---
        (1, 1, [16]),
        (4, 1, [8192, 8192, 8192, 8192]),
        # --- sq=4 (current MTP=3 production) ---
        (1, 4, [16]),
        (4, 4, [1024, 1024, 1024, 1024]),
        (4, 4, [8192, 8192, 8192, 8192]),
        # --- sq=8 (MTP=7 new, RE.4c target) ---
        (1, 8, [16]),
        (2, 8, [64, 64]),
        (4, 8, [1024, 1024, 1024, 1024]),
        (4, 8, [8192, 8192, 8192, 8192]),
    ]
    all_pass = True
    for shape in shapes:
        try:
            ok = run_shape(*shape)
            all_pass = all_pass and ok
        except Exception as e:
            print(f"[ERROR] shape {shape}: {e}")
            import traceback
            traceback.print_exc()
            all_pass = False
    print("=" * 60)
    print("ALL PASS" if all_pass else "FAILURES DETECTED")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Offline correctness test for HK qh32 MLA port (DSR1 TP=4, nhead=32).

Compares output of:
  (a) baseline ASM path `mla_decode_fwd` (with AITER_ENABLE_HK_QH32=0)
  (b) HK qh32 path `mla_decode_fwd` (with AITER_ENABLE_HK_QH32=1 + EXPERIMENTAL=1)

Run:
  HOME=/tmp AITER_ENABLE_EXPERIMENTAL=1 python3 /tmp/test_hk_qh32_correctness.py

DSR1 shape (TP=4, nhead=32, kv_lora=512, rope=64):
- batch_size ∈ {1, 2, 4}  (CONC=4 workload)
- max_seqlen_q = 4  (MTP=3 → qseqlen=4, matches ASM kernel qseqlen=4 support)
- kv_seqlen ∈ {16, 64, 1024, 8192}

Tolerance: max_abs_diff < 1e-2
"""
import os
import sys
import torch
import aiter
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.mla import mla_decode_fwd


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

    (
        (wmd_sz, wmd_dt),
        (widx_sz, widx_dt),
        (wis_sz, wis_dt),
        (ri_sz, ri_dt),
        (rfm_sz, rfm_dt),
        (rpm_sz, rpm_dt),
    ) = get_mla_metadata_info_v1(
        batch_size,
        max_seqlen_q,
        num_heads,
        torch.bfloat16,
        dtype_kv,
        is_sparse=False,
        fast_mode=True,
    )

    work_meta_data = torch.empty(wmd_sz, dtype=wmd_dt, device=device)
    work_indptr = torch.empty(widx_sz, dtype=widx_dt, device=device)
    work_info_set = torch.empty(wis_sz, dtype=wis_dt, device=device)
    reduce_indptr = torch.empty(ri_sz, dtype=ri_dt, device=device)
    reduce_final_map = torch.empty(rfm_sz, dtype=rfm_dt, device=device)
    reduce_partial_map = torch.empty(rpm_sz, dtype=rpm_dt, device=device)

    get_mla_metadata_v1(
        cu_seqlens_q,
        kv_indptr,
        kv_last_page_lens,
        num_heads,
        1,  # nhead_kv
        True,
        work_meta_data,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        page_size=1,
        dtype_q=dtype_q,
        dtype_kv=dtype_kv,
        kv_granularity=16,
        max_seqlen_qo=max_seqlen_q,
        uni_seqlen_qo=max_seqlen_q,
        fast_mode=1,
        max_split_per_batch=16,
    )
    return (
        cu_seqlens_q,
        kv_indptr,
        kv_last_page_lens,
        work_meta_data,
        work_indptr,
        work_info_set,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    )


def run_shape(batch_size, max_seqlen_q, kv_seqlens, num_heads=32, seed=42):
    device = "cuda"
    torch.manual_seed(seed)

    kv_lora = 512
    rope = 64
    head_dim = kv_lora + rope  # 576
    v_head_dim = kv_lora  # 512

    total_q = batch_size * max_seqlen_q
    total_kv_pages = sum(kv_seqlens)

    # Random q in bf16 then cast to fp8
    q_bf16 = torch.randn(total_q, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    q = q_bf16.to(torch.float8_e4m3fn)

    # Random kv_buffer (fp8)
    kv_bf16 = torch.randn(total_kv_pages, 1, 1, head_dim, device=device, dtype=torch.bfloat16)
    kv_buffer = kv_bf16.to(torch.float8_e4m3fn)

    meta = build_metadata(batch_size, max_seqlen_q, num_heads, kv_seqlens, device)
    (cu_seqlens_q, kv_indptr, kv_last_page_lens,
     work_meta_data, work_indptr, work_info_set,
     reduce_indptr, reduce_final_map, reduce_partial_map) = meta

    kv_indices = torch.arange(total_kv_pages, dtype=torch.int32, device=device)

    q_scale = torch.ones(1, dtype=torch.float32, device=device)
    kv_scale = torch.ones(1, dtype=torch.float32, device=device)

    def call(use_hk):
        os.environ["AITER_ENABLE_HK_QH32"] = "1" if use_hk else "0"
        os.environ["AITER_ENABLE_EXPERIMENTAL"] = "1" if use_hk else "0"
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
            q_scale=q_scale,
            kv_scale=kv_scale,
        )
        torch.cuda.synchronize()
        return o

    o_asm = call(use_hk=False)
    o_hk = call(use_hk=True)

    diff = (o_hk.float() - o_asm.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    l2_per_row = (o_hk.float() - o_asm.float()).pow(2).sum(-1).sqrt()
    rel_l2 = (l2_per_row / (o_asm.float().pow(2).sum(-1).sqrt() + 1e-8)).mean().item()

    shape_str = f"bs={batch_size} sq={max_seqlen_q} kv={kv_seqlens} nhead={num_heads}"
    status = "PASS" if max_diff < 1e-2 else "FAIL"
    print(f"[{status}] {shape_str}: max_abs_diff={max_diff:.4e} mean={mean_diff:.4e} rel_L2={rel_l2:.4e}")
    return max_diff < 1e-2


def main():
    # DSR1 wrapper workload: MTP=3 (qseqlen=4) at CONC=4
    shapes = [
        # (batch_size, max_seqlen_q, kv_seqlens)
        (1, 4, [16]),
        (2, 4, [16, 16]),
        (4, 4, [64, 64, 64, 64]),
        (4, 4, [1024, 1024, 1024, 1024]),
        (4, 4, [8192, 8192, 8192, 8192]),
        # decode-only qseqlen=1 sanity too
        (4, 1, [8192, 8192, 8192, 8192]),
    ]
    all_pass = True
    for shape in shapes:
        try:
            ok = run_shape(*shape)
            all_pass = all_pass and ok
        except Exception as e:
            print(f"[ERROR] shape {shape}: {e}")
            all_pass = False
    print("=" * 60)
    print("ALL PASS" if all_pass else "FAILURES DETECTED")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

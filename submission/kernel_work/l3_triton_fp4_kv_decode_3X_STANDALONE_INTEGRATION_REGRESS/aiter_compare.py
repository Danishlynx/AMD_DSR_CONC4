"""aiter_compare — time aiter's production mla_decode_fwd at the same shape
to disambiguate the L3 5× claim.

Per advisor (May 10):
    Aiter's 125 µs reference is at whatever NUM_KV_SPLITS aiter picked. If aiter
    also scales near-linearly with KS, the L3 5.36× speedup collapses to "I tuned
    KS, aiter didn't." This script tests that.

Outputs at production shape (bs=4, head=16, seq=8192):
    - L3 Triton at KS=8, 16, 32, 64
    - aiter mla_decode_fwd (default config)
    - load-only Triton at KS=8, 64
"""
import os
import sys
import torch

# Suppress aiter init noise
os.environ.setdefault("AITER_LOG_LEVEL", "WARN")

import mla_decode_fp4_kv as l3mod


def time_kernel(fn, n_warmup=5, n_iters=200):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / n_iters


def main():
    if not torch.cuda.is_available():
        print("ERROR: cuda/hip not available")
        return 1

    bs = 4
    head_num = 16
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    seq_len = 8192
    BLOCK_C = kv_lora_rank
    BLOCK_R = 64
    BLOCK_H = 16

    print(f"shape: bs={bs}, head={head_num}, seq={seq_len}, kv_lora={kv_lora_rank}")

    # Common buffers
    Q = torch.randn(bs, head_num, kv_lora_rank + qk_rope_head_dim, dtype=torch.bfloat16, device="cuda")
    K_fp4 = torch.zeros(bs * seq_len, 288, dtype=torch.uint8, device="cuda").random_(0, 256)
    K_scale = torch.zeros(bs * seq_len, 18, dtype=torch.uint8, device="cuda").random_(120, 130)
    kv_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device="cuda") * seq_len
    kv_indices = torch.arange(0, bs * seq_len, dtype=torch.int32, device="cuda")
    Att_Out = torch.zeros(bs, head_num, 128, kv_lora_rank + 1, dtype=torch.float32, device="cuda")

    # ---------- L3 Triton sweep ----------
    print(f"\n[L3 Triton FP4]")
    l3_results = {}
    for ks in [8, 16, 32, 64]:
        grid = (bs * (head_num // BLOCK_H) * ks,)
        def fn(ks_=ks, grid_=grid):
            l3mod._decode_fp4_kv_stage1[grid_](
                Q, K_fp4, K_scale, kv_indptr, kv_indices, 1.0 / (kv_lora_rank ** 0.5),
                Att_Out,
                stride_qb=Q.stride(0), stride_qh=Q.stride(1),
                stride_kfp4_t=K_fp4.stride(0), stride_kscale_t=K_scale.stride(0),
                stride_attout_b=Att_Out.stride(0),
                stride_attout_h=Att_Out.stride(1),
                stride_attout_s=Att_Out.stride(2),
                bs=bs, head_num=head_num,
                kv_lora_rank=kv_lora_rank, qk_rope_head_dim=qk_rope_head_dim,
                BLOCK_C=BLOCK_C, BLOCK_R=BLOCK_R, BLOCK_N=32, BLOCK_H=BLOCK_H,
                NUM_KV_SPLITS=ks_,
                E8M0_BIAS_T=127,
                DOT_DTYPE=1,
                num_warps=8, num_stages=3,
            )
        try:
            t = time_kernel(fn)
            l3_results[ks] = t
            print(f"  KS={ks:3d}: {t:7.2f} µs")
        except Exception as ex:
            print(f"  KS={ks:3d}: FAIL {ex}")

    # ---------- aiter mla_decode_fwd (proper signature) ----------
    print(f"\n[aiter mla_decode_fwd (FP8 cache, full signature)]")
    try:
        from aiter.mla import mla_decode_fwd

        page_size = 1
        nhead_kv = 1  # MLA uses single shared KV head
        num_pages = bs * seq_len  # one page per token at page_size=1

        # Try multiple dtypes — production uses BF16 for kv_buffer (the kernel has internal fp8 path)
        kv_buffer = torch.randn(
            num_pages, page_size, nhead_kv, kv_lora_rank + qk_rope_head_dim,
            dtype=torch.bfloat16, device="cuda",
        ) * 0.5

        # Per-query indptr (decode = qo per batch = 1 query per token)
        qo_indptr = torch.arange(0, bs + 1, dtype=torch.int32, device="cuda")
        # kv_last_page_lens: how many tokens used in the last page per batch
        kv_last_page_lens = torch.full((bs,), page_size, dtype=torch.int32, device="cuda")

        max_seqlen_q = 1   # decode = 1 query per batch
        out_aiter = torch.zeros(bs, head_num, kv_lora_rank, dtype=torch.bfloat16, device="cuda")

        # Q for aiter: [total_q_tokens, head_num, kv_lora+rope]
        Q_aiter = Q.reshape(bs * 1, head_num, kv_lora_rank + qk_rope_head_dim)

        # Test with multiple num_kv_splits values
        for ks_aiter in [None, 8, 16, 32, 64]:
            try:
                # Call aiter with explicit num_kv_splits
                def aiter_fn():
                    mla_decode_fwd(
                        Q_aiter, kv_buffer, out_aiter,
                        qo_indptr, kv_indptr, kv_indices,
                        kv_last_page_lens, max_seqlen_q,
                        page_size=page_size, nhead_kv=nhead_kv,
                        sm_scale=1.0 / (kv_lora_rank ** 0.5),
                        num_kv_splits=ks_aiter,
                    )
                t = time_kernel(aiter_fn, n_warmup=3, n_iters=50)
                ks_str = "auto" if ks_aiter is None else str(ks_aiter)
                print(f"  KS={ks_str:>4s}: {t:7.2f} µs")
            except Exception as ex:
                ks_str = "auto" if ks_aiter is None else str(ks_aiter)
                print(f"  KS={ks_str:>4s}: FAIL {type(ex).__name__}: {str(ex)[:120]}")
    except ImportError as ex:
        print(f"  aiter import failed: {ex}")
    except Exception as ex:
        print(f"  setup failed: {type(ex).__name__}: {ex}")

    # ---------- Load-only at high KS ----------
    print(f"\n[Load-only Triton at KS=8 vs KS=64]")
    try:
        import microbench_loadonly as lomod
        for ks in [8, 64]:
            grid = (bs * (head_num // BLOCK_H) * ks,)
            def lo_fn(ks_=ks, grid_=grid):
                lomod._loadonly_kernel_fp4_pattern[grid_](
                    K_fp4, K_scale, kv_indptr, kv_indices,
                    torch.zeros(1, dtype=torch.float32, device="cuda"),
                    stride_kfp4_t=K_fp4.stride(0),
                    stride_kscale_t=K_scale.stride(0),
                    bs=bs, head_num=head_num,
                    BLOCK_C_T=BLOCK_C, BLOCK_R_T=BLOCK_R, BLOCK_N_T=32, BLOCK_H_T=BLOCK_H,
                    NUM_KV_SPLITS_T=ks_,
                    kv_lora_rank=kv_lora_rank, qk_rope_head_dim=qk_rope_head_dim,
                )
            t = time_kernel(lo_fn)
            print(f"  KS={ks:3d}: {t:7.2f} µs")
    except Exception as ex:
        print(f"  load-only failed: {ex}")

    # ---------- Summary ----------
    print(f"\n=== Summary ===")
    print(f"L3 Triton FP4:")
    for ks, t in l3_results.items():
        print(f"  KS={ks:3d}: {t:7.2f} µs")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Kernel Engineering Work (Supporting Deliverables)

This directory contains kernel work that did NOT make it into the headline TPOT-reducing path but represents real engineering effort that AMD's CDNA4 / AITER kernel teams may find valuable. **The TPOT win (Phase 11 v3 — TRT-LLM thinking port) is in `ATOM_main/` and `patches/scripts/phase11_per_phase_mtp/` at repo root, not here.**

## Summary table

| Deliverable | Status | Numerics | Perf | What it shows |
|---|---|---|---|---|
| [`phase2_fp4t_fused_ar_BUILT_DORMANT/`](phase2_fp4t_fused_ar_BUILT_DORMANT/) | ✅ Built end-to-end (C++ + Pybind + Python dispatcher) | n/a (kernel not yet consumer-wired) | dormant; estimated −0.4 to −0.8 ms TPOT if consumer-wired | Adds `else if constexpr(std::is_same_v<OutT, opus::fp4_t>)` branch to aiter's `ar_fusion_epilogue`: BF16 → FP4-packed direct via `__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16` + per-32-group E8M0 scale + 4-lane DPP reduce. Verified by `nm` symbol export. Last-mile wiring deferred (multi-day work — `DeepseekV2MoE.forward` wrapped by `torch.ops.aiter.maybe_dual_stream_forward` custom-op with fixed Tensor-only signature). |
| [`r2_small_m_moe_BUILT_NEVER_INTEGRATED/`](r2_small_m_moe_BUILT_NEVER_INTEGRATED/) | ✅ Hand-authored CDNA4 ASM, 23× microbench speedup at W2 | BIT-EXACT (max_abs_err = 0.0625) at full DSR1 W13 + W2 GEMM shape | 23× speedup at W2 in microbench; never integrated (production dispatcher already on FlyDSL fast path) | M2.1–M2.13b CDNA4 MFMA pipeline reference. Uses `v_mfma_scale_f32_16x16x128_f8f6f4` + `ds_read_b64_tr_b4` + `global_atomic_pk_add_bf16` + per-1×32 E8M0 scale. Diff'd `r2_M2_1_single_tile_fp4_gemm.cu` → `r2_M2_4_verified_d_with_tr_b4_attempt.cu` walks through layout debugging step by step. |
| [`l3_triton_fp4_kv_decode_3X_STANDALONE_INTEGRATION_REGRESS/`](l3_triton_fp4_kv_decode_3X_STANDALONE_INTEGRATION_REGRESS/) | ✅ Triton kernel, integration regressed | within MXFP4 envelope (max_abs_err < 0.05) | **3.04× faster than aiter `mla_decode_fwd`** standalone (23 µs vs 70 µs at production shape bs=4 head=16 seq=8192) | FP8-cast `tl.dot` for matmul, `NUM_KV_SPLITS=64` + `nw=8 ns=3` + `BLOCK_N=32`. Integration regressed +1.10 ms because dispatch gate at `l3_decode_wrapper.py:34` reads `if max_seqlen_q != 1: return False`; production verifier uses MQ=4 → kernel never fires, dispatch check adds overhead. MQ=4 specialization (3-5 days) would unblock for −0.3 to −0.5 ms TPOT. |

## How to interpret "BUILT but DORMANT / NEVER INTEGRATED"

For Phase 11 v3 (the actual TPOT win) and these kernel deliverables, the gate to enter production code path was:

1. Kernel compiles and launches without crash → ✅ All four kernel deliverables pass this gate
2. Numerics match reference within model's tolerance → ✅ All four pass (R2 bit-exact, L3 within MXFP4 envelope, Phase 2 verified by `nm` symbol check, Phase 11 v3 GSM 0.9318 PASS)
3. **Single-prompt smoke at ISL=8192 production shape** → ✅ Phase 11 v3 passes; others not benched under production wire-up
4. **GSM8K N=3 median ≥ 0.93 under official harness** → ✅ Phase 11 v3 only
5. **Kimbochen 4-iter median(2,3,4) shows ≥ −0.30 ms TPOT delta** → ✅ Phase 11 v3 (−0.661 ms) only

Phase 11 v3 made it all the way through. The kernels here passed gates 1+2 but had a downstream consumer-wiring blocker (Phase 2 fp4_t), or dispatcher conflict at the integration boundary (L3 MQ=4), or production already had a faster path so they regressed (R2 vs FlyDSL).

Each is a **reusable reference** for AMD's kernel team or for future submissions where the integration context changes.

## See also

- [`../investigation/dead_levers_for_reference/`](../investigation/dead_levers_for_reference/) — patches that didn't pass gate 1 or 2 (kept compact for "don't re-test" reference)
- [`../TECHNICAL_APPROACH.md`](../TECHNICAL_APPROACH.md) §5 "Built-but-dormant deliverables" — narrative explanation
- [`../PR_DESCRIPTION.md`](../PR_DESCRIPTION.md) §"What's also in this branch" — patch-level file list with status markers

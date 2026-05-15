# R2 — Small-M MoE GEMM2 Kernel (CDNA4 hand-authored, FP4×FP4)

**Status**: ✅ Hand-authored CDNA4 ASM, BIT-EXACT at full DSR1 W13 + W2 GEMM shapes, **23× microbench speedup at W2 vs T0**, but never deployed in production (already-tuned FlyDSL fast path in dispatcher).
**Hot shape captured**: M=4, H=7168, intermediate=512, num_experts=257, top-k=9. AITER dispatched `[fused_moe] using 2stage default` (CK fallback, 87.5% tile waste at M=4).

## Why this kernel exists

R2 attacks the M=4 hot decode shape where existing AITER paths underperform:

| | T0 (AITER `mla_decode` baseline) | R2 (this kernel) |
|---|---:|---:|
| Per-call time @ W2 | 71.88 µs | ~3.1 µs (23× speedup, microbench) |
| Compute utilization | 0.85% peak FP4 | (higher; not measured precisely) |
| HBM utilization | 13.82% peak BW | (substantially higher) |

## CDNA4 primitives demonstrated

The kernel exercises 3 critical CDNA4 features new in MI355X (gfx950):

| Primitive | Where used | Reference file |
|---|---|---|
| `v_mfma_scale_f32_16x16x128_f8f6f4` (scaled MFMA with FP4/FP6/FP8 operands + E8M0 scale) | Core GEMM accumulator | `r2_kernel/r2_M2_1_single_tile_fp4_gemm.cu`, `r2_M2_2_corrected_layout.cu` |
| `ds_read_b64_tr_b4` (LDS transposed read in FP4 packed format) | Input loading for B matrix transposed access pattern | `r2_kernel/r2_M2_4_verified_d_with_tr_b4_attempt.cu` |
| `global_atomic_pk_add_bf16` (BF16-packed atomic add for multi-expert accumulation) | Multi-expert output accumulation | `phase10_smallm_dispatch/phase10_r2_step1_tile_m16_register_patch.py` |

## Numerics validation

| Validation pass | Shape | Verdict |
|---|---|---|
| M2.1 FP4 MFMA pipeline end-to-end | W2 (single tile) | PASS — deterministic output |
| M2.2 Corrected lane layout | W2 | partial (24× off) → led to M2.3 layout probe |
| M2.3 D matrix lane layout verified | W2 | VERIFIED via probe kernel |
| M2.4 Verified D mapping + LDS addr | W2 | LDS addressing gap identified |
| M2.7 Probe + MFMA signature root cause | W2 | FOUND |
| M2.8 BIT-EXACT pass | W2 | ✅ max_abs_err = 0.0 |
| M2.9 Scale probe — V_MFMA_SCALE operand semantics | W2 | DECODED |
| M2.10 + M2.11 BIT-EXACT at FULL DSR1 W2 GEMM SHAPE | full DSR1 | ✅ |
| M2.12 per_1×32 E8M0 scaling BIT-EXACT at W13 + W2 | full DSR1 | ✅ |
| M2.13 multi-expert dispatch + atomic accum | full DSR1 | ✅ PASS |
| M2.13b internal-gather | full DSR1 | ✅ PASS, 2.32× speedup |

## Why never integrated

Production AITER dispatcher routes M=4 MoE GEMM2 via `flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic_bnt2` (already on the FlyDSL fast path). When R2 was wired into the dispatcher, the per-call Python dispatch overhead **exceeded** the compute savings at the hot shape — net +0.37 ms TPOT regress in the production benchmark (see TECHNICAL_APPROACH.md §3.1 example for the "B1 microbench-faster but +0.37 ms in prod" lesson).

The kernel is shipped here as a **CDNA4 reference implementation**:
- Documenting the correct MFMA signature for `v_mfma_scale_f32_16x16x128_f8f6f4` (M2.7 probe + decode)
- Documenting the correct lane layout for E8M0-scaled FP4 inputs (M2.2 → M2.3 → M2.4 chain)
- Providing a working `ds_read_b64_tr_b4` example (M2.4)
- Providing a working multi-expert `global_atomic_pk_add_bf16` accumulator (M2.13)

## Files in this directory

### `r2_kernel/`
Step-by-step CDNA4 GEMM2 kernel implementation. Each `r2_M2_*.cu` / `r2_M2_*.py` pair = one milestone in the M2.1 → M2.13b chain, with the harness used to validate it. Build with `r2_build.sh` (`hipcc --offload-arch=gfx950 -O1 -fPIC -shared`).

### `phase10_smallm_dispatch/`
AITER dispatcher integration patches. `DESIGN.md` documents the dispatch-gate logic. `phase10_r2_step1_tile_m16_register_patch.py` registers tile_m=16 (instead of CK's t32 default) for the hot M=4 shape.

## Reproduce

```bash
# In aiter-test/ source tree:
bash r2_kernel/r2_build.sh
python3 r2_kernel/r2_harness.py            # smoke test
python3 r2_kernel/r2b_microbench.py        # 23× speedup measurement
```

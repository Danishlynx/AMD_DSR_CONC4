# Phase 10 R2 — Small-M MoE GEMM Kernel (Apr 29 ~13:25 IST)

## Context

All 6 plan v7 levers (L0/L1/L2/L3/L4 v1/L5) tested = ALL DEAD or regress. Plan v7 is exhausted. Tier 2 reserves R1 (cross-layer expert prefetch) and R2 (small-M MoE GEMM kernel) are the only forward path. Per plan v7 analysis, R2 is the "highest-leverage AMD-reference contribution" — the kernel doesn't exist on CDNA4 today, AMD's docs explicitly flag the small-M MoE gap.

## Critical discovery (Apr 29 13:25)

The plan v7 description of R2 said "write new ASM kernel `flydsl_moe2_smallm_t4x128x256_atomic.s`" implying multi-week from-scratch ASM work.

**Reality check via container source audit:** FlyDSL kernels are produced by an MLIR-based JIT codegen pipeline at `/app/aiter-test/aiter/aot/flydsl/moe.py` — they're parameterized by `tile_m`, `tile_n`, `tile_k`. The registered configs in `get_flydsl_stage1_kernels` (at moe_kernels.py:61) and `get_flydsl_stage2_kernels` (line 134) currently use:

```python
tile_ms = [32, 64, 128]
```

That's IT. No tile_m below 32. At our M=4 decode case, the 32-row MFMA tile-M wastes 28/32 = 87.5% of slots — the exact issue plan v7 identified.

**Adding smaller tile_m is a config registration + AOT recompile**, NOT new ASM. Much smaller scope than expected.

## CDNA4 MFMA constraint

The smallest CDNA4 V_MFMA_F32_*X*X*_F8F6F4 tile is 16x16x128 (per ISA p.83+98). So:
- tile_m=4 still wastes 12/16 inside the MFMA itself
- tile_m=16 wastes 0 in the MFMA (well-aligned to 16x16x128)
- tile_m=32 wastes 28/32 at M=4 (current baseline)

So tile_m=16 is the natural target. M=4, M=8 still align cleanly into a single 16x16x128 MFMA wave. Saves 50% of compute slots vs current tile_m=32 at M=4.

## Strategy

**Step 1 (this session):** Register `tile_m=16` configs for stage 1 + stage 2 in `get_flydsl_stage1_kernels` + `get_flydsl_stage2_kernels`.
**Step 2 (this session):** Trigger AOT precompile via `aiter/aot/flydsl/moe.py main()` to produce the new kernels. Verify they appear in `/app/aiter-test/aiter/jit/flydsl_cache/`.
**Step 3 (this session):** Find the dispatcher that selects flydsl tile_m at runtime. Modify to prefer tile_m=16 at M ≤ 8 (DSR1 decode).
**Step 4 (this session):** Boot, run N=3 GSM8K + 3-iter perf. Expected: TPOT drop because half the MFMA waste is gone.

If tile_m=16 lands well: extend to also register tile_m=4 (despite MFMA waste) to see if launch overhead reduction outweighs MFMA waste at our small workload.

## Numerics safety

The new kernel uses the SAME MFMA ops + SAME FP4 layout + SAME E8M0 scale layout as tile_m=32. Only the workgroup tile dimension changes. Numerics should be bit-equivalent (or within MFMA-tile-block-aggregation noise, ~1e-3 BF16 atol). GSM8K canary mandatory but expected pass clean.

## Mergeability

PR to ROCm/aiter: "moe_kernels: register tile_m=16 stage1+stage2 configs for small-M decode (gfx950)". One-file change to `aiter/ops/flydsl/moe_kernels.py:71` (`tile_ms = [16, 32, 64, 128]`) + corresponding entry in stage2. AOT recompile at install time picks them up. Backwards-compatible (existing tile_m=32+ paths unchanged).

## Files to modify

| # | File | Change |
|---|---|---|
| 1 | `/app/aiter-test/aiter/ops/flydsl/moe_kernels.py` line 71 | `tile_ms = [16, 32, 64, 128]`. Then re-trigger `_register_all_configs()`. |
| 2 | Stage 2 ALREADY has `tile_m=16` registered (line 142). NO change needed for stage 2. |
| 3 | Dispatcher: `aiter/fused_moe.py` calls `_flydsl_stage1_wrapper` with `kernelName` string set in `metadata.stage1.kernelName`. Need to find the dispatch logic that sets `metadata.stage1` per (M, N, K). Likely in `aiter/fused_moe.py` near top, look for `dispatch` or `select_metadata`. Modify to prefer t16x128x256 variant at M ≤ 8. |
| 4 | AOT recompile trigger: `cd /app/aiter-test && python -m aiter.aot.flydsl.moe --csv aiter/configs/dsv3_fp4_tuned_fmoe.csv` or rely on first-boot lazy compile (will add ~5-10 min to first boot). |

## Status (Apr 29 ~13:45 IST)

**Step 1 DONE.** `tile_ms = [16, 32, 64, 128]` patch applied to `moe_kernels.py:71` in container. Backup at `.pre_phase10_r2_step1`. py_compile clean. `_register_all_configs()` will pick up tile_m=16 stage1 variants at next module import.

**MAJOR FINDING during Step 2 trace:** dispatcher is CSV-driven (`get_2stage_cfgs` reads `tuned_fmoe.csv` keyed by `(cu_num, token, model_dim, inter_dim, expert, topk, ...)`). Existing dsv3 CSV row for `(256, 2, 7168, 256, 257, 9, fp4, per_1x32)` ALREADY uses `flydsl_moe1_afp4_wfp4_bf16_t16x128x256_w3_kb7_bnt0_go_fp4` — meaning **t16 stage1 kernel for inter_dim=256 has been compiled and works**. But token=4, 8, 16 rows all use t32 variants. Likely because the original tuning sweep didn't include t16 in the tile-search list at those token counts.

**Measured `us1` (stage 1 latency) from existing CSV:**
- token=1, t32: 13.31 µs (winner — chosen by tuner)
- token=2, **t16**: 16.35 µs (winner — chosen by tuner; t16 actually selected)
- token=4, t32: 21.79 µs
- token=8, t32: 31.39 µs
- token=16, t32: 52.76 µs

The token=2 case proves t16 wins at very small M but NOT necessarily at token=4, 8, 16 — kernel-launch overhead and LDS efficiency depend on workload size. Need a real measurement.

**Resume next session — Step 2:**
1. Run `gemm_moe_tune.py` for ONLY our DSR1 decode shapes:
   - keys: `(256, token∈{4,8,16}, 7168, 256, 257, 9, ActivationType.Silu, bfloat16, fp4, fp4, per_1x32, 1, 0)`
   - Should pick up tile_m=16 variants now that registry includes them
   - ~5-10 min per shape × 3 shapes = 15-30 min
2. Compare us_total before/after on these 3 rows in the CSV
3. If t16 variants win: keep new CSV (already merged in-place). Rebench.
4. If t16 doesn't win: revert CSV to original, conclude tile_m at M=4..16 is genuinely tile_m=32 limited, move to R1 (expert prefetch).
5. (Bonus) re-run tune for inter_dim=512 shapes (the 5 rows starting at row 6 in our grep output) — same possibility.

**Important: AOT precompile may be needed first.** Before running gemm_moe_tune.py, the t16 kernels at our specific (model_dim=7168, inter_dim=256, E=257, topk=9) shape might not exist in the JIT cache. Trigger via:
```bash
cd /app/aiter-test
python -m aiter.aot.flydsl.moe --csv aiter/configs/model_configs/dsv3_fp4_untuned_fmoe.csv 2>&1 | tee /tmp/aot_compile_t16.log
```
Or accept the lazy compile penalty on first inference.

**v6e v1 patches still in-place env-gated NULL-OP** (envs.py + layernorm.py + deepseek_v2.py) — will keep until R2 lands.

## Risk register for R2 (small-M MoE GEMM)

| Risk | Mitigation |
|---|---|
| Codegen at tile_m=16 hits register pressure or LDS bank conflicts producing slower kernel | Profile first; if slow, drop tile_m=16 variant before benching |
| AOT compile fails for tile_m=16 (some FlyDSL guard rejects it) | Fall back to lazy first-boot compile; if still fails, file ROCm/aiter issue |
| Dispatcher hard to locate / modify | Trace `metadata.stage1.kernelName` set-site via grep + pdb |
| Numerics drift on small tile (block-aggregation order) | N=3 GSM8K canary mandatory before perf bench |
| GSM8K margin only +0.0018 from L4 v6e v1, baseline at 0.9348 — any R2 numerics drift kills gate | Run baseline N=3 first to confirm gold+v6c reproduces; then R2 N=3 |

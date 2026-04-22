# v9 HK qh32 design — 16x16x128 MFMA upgrade + LDS relayout

## Context

v8 (session-15) = v7 + Opt-E s_setprio coverage. Correctness-preserving, minor perf gain from MFMA/VALU dual-issue hints. Does NOT solve the -35% gap vs ASM at sq=4.

**Root cause of -35% gap** (per v7 source analysis + HK mma.cuh inspection):
- v7 uses HK's `mfma1616x32` (4× cycles per K-unit)
- ASM uses hand-tuned `mfma_scale_16x16x128_f8f6f4` (1× cycles per K-unit, 4× density on fp8)
- Per NoPE iter: v7 issues 2× mfma_16x16x32 covering 64 K-cols. v9 would issue 1× mfma_16x16x128 covering 128 K-cols → 4 vs 2 iters for 512 K-cols.

## Confirmed availability (on MI355X gfx950 ROCm 7.2.2)

- HipKittens `mfma1616128` binding at `3rdparty/HipKittens/include/ops/warp/register/tile/mma.cuh:119` calls `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4`
- HK register-tile type `rt_16x128_s` defined at `types/register/rt_shape.cuh:37,47` and `types/types.cuh:69`
- This is the CDNA4 block-scaled MFMA. Pure FP8 mode: pass scale=0 (1.0 semantic) to all scale args.

## Traits changes v8 → v9

```cpp
kBlockK          = 128   // was 32, now = MFMA K-dim natively
// NoPE iteration count: kKvLoraRank / (kBlockK * 2) where * 2 accounts for kv_0 + kv_1 halves
//   v7: 512 / (32*2) = 8 iters × 2 mfma each = 16 mfma calls
//   v9: 512 / (128*2) = 2 iters × 2 mfma each = 4 mfma calls  (4× fewer)
// RoPE iter count: kQkRopeHeadDim / (kBlockK * 2) — 64 / 256 = 0.25 — CAN'T FIT.
//   Alternative: process RoPE with kBlockK=32 (keep old tile) — mixed tile widths
//   Or pad RoPE to kBlockK=128 (wastes compute on padding)
```

**RoPE challenge**: kQkRopeHeadDim=64 is only half of kBlockK=128 at proposed width. Two sub-options:
- (a) **Mixed-width**: keep 16x16x32 for RoPE pass (it's short, only 1 NoPE-equivalent iter anyway). NoPE uses 16x16x128.
- (b) **Fuse NoPE+RoPE**: concatenate K's NoPE (512) and RoPE (64) into single stream of 576; pad 64 to 128 with zeros; MFMA over 5×128 tiles (4 pure NoPE, 1 mixed NoPE-pad-RoPE). Messy.

**Recommended**: (a) mixed-width. RoPE is only 64 K-cols so gain from upgrading is minimal; keep 32-wide MFMA there.

## VGPR budget estimation

v7 kv_0/kv_1 each = 4 VGPRs per lane (rt_16x32 = 16×32/64 = 8 fp8 = 2 DWORDs×2 slots). 

v9 kv_0 = rt_16x128 per lane = 16 × 128 / 64 = 32 fp8 = 8 DWORDs = **16 VGPR-halves = 8 VGPRs**. ×2 for kv_0+kv_1 = **16 VGPRs total** (vs 8 in v7).

oaccu unchanged (128 VGPRs). p_comp = 8 fp32 = 2 VGPRs (unchanged). p_mfma = 2 fp8 = 0.5 VGPRs (unchanged).

Total per-lane VGPR: ~172 → ~180 (still under 256 budget at kOccupancy=1).

## LDS layout changes in buffer_managers.cuh

**v7 KvManagerV2** packs K as 9 blocks × 32 rows × 64 cols (9×32×64 = 18432 elements × 1 byte = 18 KiB). Per `load_k_to_gpr<kRowOffset=0, kColOffset=0>`: reads 16×32 sub-tile.

**v9 KvManagerV2**: need `load_k_to_gpr_wide<kRowOffset, kColOffset=128>` that reads 16×128 sub-tile. Options:
- Multi-issue 4× `ds_read_b64` per lane (4 × 16B = 64B per lane) — fits in 8 VGPRs
- Single `ds_read_b128` wide (16B per lane × multiple phases) — simpler but needs LDS swizzle for bank conflicts

## num_pv_iter changes

Currently `kNumPVIter = kVoHeadDim / (kBlockK * 2) = 512 / 64 = 8` at kBlockK=32.
At kBlockK=128: `512 / 256 = 2` — PV loop goes from 8 iters to 2 iters. Each iter does 4× more oaccu accumulation work.

## Correctness strategy

1. Start by keeping v7's LDS layout (9 blocks × 32×64) UNCHANGED
2. Add `load_k_to_gpr_wide<kRowOffset, kColOffset>` that reads 16×128 by issuing 4× ds_read_b64 chaining 4 column slices (0, 32, 64, 96 at the same row-block)
3. New kernel `kn_mla_v32_fwd_decode_h32_fp8_fp8_v9` with halved nope_iter and mma1616128 calls
4. Test at sq=1 vs v7 (bit-exact required)
5. Test at sq=4 vs v7 (bit-exact required — same architecture as v8)
6. Bench vs v7/v8

Expected bench: if MFMA halves, TPOT could improve by 15-20% on MLA portion (~18% of wall) → ~-3% TPOT overall. If scheduling overhead also reduces, possibly -5-8% TPOT.

## Files (deferred for v9 session)

| File | Status |
|---|---|
| `v9_h32.cuh` | To write (est. 850 LOC, v7-derivative with MFMA upgrades) |
| `buffer_managers_v9_additions.cuh` | To write (KvManagerV2 extensions) |
| `test_hk_qh32_v9_correctness.py` | Adapt v8 test |
| `hk_decode_fwd_v9_append.diff` | Add v9 dispatch branch |
| `compile_hk_qh32_v9.sh` | Add v9 to JIT build |

## Rollback safety

v9 gated via `AITER_ENABLE_HK_QH32_V9=1` env. Coexists with v7 and v8 symbols. No binary-level displacement.

## Time estimate

- LDS wide-read extension: 4-6h
- Kernel rewrite with mma1616128 calls: 4-6h
- Correctness debug: 8-16h (MFMA lane-mapping nuances at 128-wide)
- Bench + iterate: 2-4h
- **Total: 1.5-2 days**

## Priority vs v8

**v8 is safe and deployable**. If v8 bench matches v7 (bit-exact + similar perf), the sq=4 production path has no regression risk. v9 is the actual perf uplift path and is the RIGHT next investment after v8 bench confirms safety.

# RE.4a — HK qh32 v7 wrapper bench: FAILS vs ASM at qseqlen=4

## Setup
- Server: `reproducer_best` container, INT4 AR + HK qh32 enabled
- Launch script: `/tmp/p0_launch_hk_qh32.sh` (adds `AITER_ENABLE_HK_QH32=1` + `AITER_ENABLE_EXPERIMENTAL=1`)
- Bench: 3× kimbochen bench + `--use-chat-template --ignore-eos --num-prompts 40`
- Time: Apr 22 02:42-02:47 UTC

## Results

| Run | Total Thr (tok/s) | Thr/GPU (÷4) | Mean TPOT | Median TPOT |
|---|---:|---:|---:|---:|
| 1 | ~3560 | ~890 | 10.07 ms | 10.44 ms |
| 2 | 3547 | 887 | 9.58 ms | 9.78 ms |
| 3 | 3441 | 860 | 9.83 ms | 10.18 ms |
| avg | ~3516 | ~879 | 9.83 ms | 10.13 ms |

## Delta vs RE.1 baseline (INT4 AR only, no HK)
- Thr/GPU: 1360 → 879 = **-35% (DISASTER)**
- TPOT mean: 6.15 → 9.83 = **+60%**

## Why HK qh32 lost

HK v7 kernel is **correct** (passes bit-exact vs ASM at qseqlen=4 per `RE4a_correctness_RESULT.md`) but **structurally slower** than hand-tuned ASM:
- ASM kernel `mla_a8w8_qh32_qseqlen4_gqaratio32_ps` is persistent, fully hand-optimized, ~4ms per kernel call
- HK kernel is generic CK/Triton-style, ~6-8ms per kernel call on same shape
- ASM has optimized MFMA scheduling, occupancy, LDS layout that HK's algebraic approach can't match

This matches Apr 19 session-8 log: "E-08-05 lucky 2/4 (165.35, 159.87, 150.23 interact — 2 of 3 runs fail 165 gate)" — HK gave interact ~160-165 which in wrapper-equivalent = ~860-890 thr/GPU. **Apr 19 was not fixable because the kernel isn't BUGGY — it's just slow.**

## Conclusion

**HK qh32 at qseqlen=4 is NOT a win.** ASM path is ~35% faster. Keep ASM for all qseqlen=4 workloads (MTP=3 default).

## The real prize remains: RE.4b (qseqlen=8 for MTP=7 unlock)

ASM persistent kernel CRASHES at qseqlen=8 (fold trick fails at gqa_ratio=32 * qseqlen=8 = 256, metadata invariant broken). HK is the ONLY viable path for qseqlen=8. Even if HK is 35% slower per kernel call, if it unlocks MTP=7:
- MTP=3 (current): ~2.1 tokens/step at ~6ms TPOT = 2.86 ms/token
- MTP=7 with HK: ~3.5 tokens/step at ~10ms TPOT = 2.86 ms/token (break-even on per-kernel)
- BUT the per-token-MTP=7 *avoids the extra iteration overhead* that keeps accumulating at MTP=3 for more tokens — so real gain is MTP=7 reducing *decode step count* by ~30-40%

HK at qseqlen=8 should give ~3.5 tok/step at ~10ms = effective 2.86 ms/token. Combined with INT4 AR: estimated 1500+ thr/GPU. **This is RE.4b, the multi-day kernel work.**

## Blockers for RE.4b
- `get_mla_metadata_v1` may not produce valid work_info at qseqlen=8 (our qseqlen=8 smoke test crashed at metadata stage, not kernel)
- Need to inspect `/app/aiter-test/csrc/kernels/mla/metadata/v1_2_device.cuh` for qseqlen-dependent invariants
- May require custom metadata build OR extending v1_2 with qseqlen=8 path

## Rollback
```bash
# Remove HK env from launch script
sed -i '/AITER_ENABLE_HK_QH32/d; /AITER_ENABLE_EXPERIMENTAL=1/d' /tmp/p0_launch_profiled.sh
# Restart
docker restart reproducer_best
# Launch clean
docker exec -d reproducer_best bash /tmp/p0_launch_profiled.sh
```

Baseline (RE.1) restored.

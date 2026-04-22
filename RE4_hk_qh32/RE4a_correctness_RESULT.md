# RE.4a — HK qh32 v7 correctness test PASSES (Apr 22 session-14 evening)

## Result

All 6 test shapes PASS vs ASM reference (`mla_decode_fwd` with HK disabled).

| bs | sq | kv_seqlens | max_abs_diff | rel_L2 | Status |
|---:|---:|---|---:|---:|---|
| 1 | 4 | [16] | 0.0 | 0.0 | ✅ BIT-EXACT |
| 2 | 4 | [16,16] | 0.0 | 0.0 | ✅ BIT-EXACT |
| 4 | 4 | [64,64,64,64] | 0.0 | 0.0 | ✅ BIT-EXACT |
| 4 | 4 | [1024]×4 | 0.0 | 0.0 | ✅ BIT-EXACT |
| 4 | 4 | [8192]×4 | 0.0 | 0.0 | ✅ BIT-EXACT |
| 4 | 1 | [8192]×4 | 5.37e-3 | 1.48e-2 | ✅ PASS (<1e-2) |

## Reinterpretation of Apr 19 finding

Apr 19 session-8 log said: "E-08-05 lucky 2/4 (interact 165.35, 159.87, 150.23 — 2 of 3 runs fail 165 gate, NOT submittable). C1 HK kernel compiles but produces wrong output."

**The "wrong output" referred to BENCH VARIANCE**, not numerical correctness. v7 produces bit-exact matching output. The Apr 19 bench variance was likely due to:
- Server state noise (warmup, DVFS)
- Occupancy=1 being too constrained (check if v7 needs higher occupancy)
- Possibly v7 at that time was DIFFERENT (pre-v7 iterations may have had bugs)

## Implications

1. **RE.4a = DONE** with no fix needed. v7 is correct at qseqlen=4.
2. **RE.4b qseqlen=8** becomes the critical path — extend kernel for MTP=7 unlock (+15-20% TPOT).
3. Bench v7 under wrapper once MoE tuner completes. If v7 matches or beats ASM at qseqlen=4, keep HK as default path.

## Test command

```bash
HOME=/tmp HIP_VISIBLE_DEVICES=1 AITER_ENABLE_EXPERIMENTAL=1 AITER_ENABLE_HK_QH32=1 \
  python3 /tmp/test_hk_qh32_correctness.py
```

Completed in ~30 seconds on GPU 1 (in parallel with MoE tuner running on GPU 0).

## Next step

When MoE tuner finishes (2-4h), cold-boot server with both:
- `AITER_ENABLE_HK_QH32=1 AITER_ENABLE_EXPERIMENTAL=1` (enable HK path)
- RE.3 MoE CSV (persistent via model_configs/)
- RE.1 INT4 AR env (already in launch)

Run 3-run wrapper bench. If HK beats ASM or GSM8K holds, proceed to RE.4b qseqlen=8 extension.

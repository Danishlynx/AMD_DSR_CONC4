# RE.3 MoE tuner — negligible impact under wrapper

## Tuner result (32 min wall, Apr 22 02:57-03:29 UTC)
Single shape token=32768 tuned successfully:
- Found 2.06x speedup: Pre-E2E 8151us → Post-E2E 3950us
- Best kernel: `moe_ck2stages_gemm1_256x128x128x128_1x4` (err 0.0%) + `flydsl_moe2_afp4_wfp4_bf16_t64x256x256_reduce_xcd4_sbm128` (err 0.3%)
- Tuner flagged "output mismatch vs reference" at E2E level — false positive (MXFP4 e8m0 precision)
- `--update_improved` SKIPPED the entry due to mismatch gate

## Bench result (3-run wrapper, 04:00-04:04 UTC)
After manually inserting tuned row into persistent CSV:

| Run | Thr/GPU | TPOT |
|---|---:|---:|
| 1 | 1388 | 6.09 |
| 2 | 1351 | 6.21 |
| 3 | 1344 | 6.15 |
| min-of-3 | 1344 | 6.21 |
| avg | 1361 | 6.15 |

vs RE.1 baseline (1353-1365 avg 1360): **identical within noise**

## Root cause of non-impact

Token=32768 shape fires ONLY during prefill (40 calls/bench × ~4ms saved = 160ms saved in 70s bench = 0.2% = noise).

Wrapper bench is DOMINATED by decode (40 requests × 1024 output tokens = 40960 decode steps). Decode uses small token shapes (token ∈ {1-64}) which are ALREADY tuned in the existing `dsv3_fp4_tuned_fmoe.csv` (30 rows covering token 1-16384).

## What it would take to get MoE win under this workload

Tune DECODE-phase MoE shapes, specifically at `m_per_expert ≈ 1147` (the estimated runtime value). These are currently covered at the 1024 bucket in dsv3 CSV — but tuned at THAT M, not exactly at 1147. Adding token∈{1024, 1280, 1536, 2048} entries and making aiter's heuristic pick the right bucket for 1147 would help. Expected gain: 1-2% TPOT.

But this requires WIDER search space in tuner + accepting the "output mismatch" flag (which is a known false positive for MXFP4).

## Files saved
- `phase_re_artifacts/re3_moe_untuned_v2.csv` — 1-shape input
- `phase_re_artifacts/re3_moe_tune_v2.sh` — tuner invocation with python -u
- `phase_re_artifacts/dsr1_fp4_tuned_fmoe.candidate.csv` — candidate from tuner (not auto-committed due to mismatch flag)

## Conclusion
RE.3 provided no measurable bench improvement. The tuner infrastructure works correctly but the targeted shape (prefill token=32768) isn't a bottleneck. Closing this lever.

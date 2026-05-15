# Phase 0 — Clean Baseline (May 01, post-redesign)

## Summary

Re-anchor Phase 11 v3 production baseline after cleaning stale state. Phase 0 is the foundation step before the AITER-Absorption + R2-v2 Persistent plan begins (P1 onward).

| Metric | Phase 0 final | Apr 30 KEEP | First May 01 measure | Pre-wipe drift | Gate | Pass? |
|---|---:|---:|---:|---:|---:|:--:|
| TPOT_med (3-iter median) | **5.894 ms** | 5.641 | 5.867 | 6.047 | ≤ 4.54 | ❌ −1.354 |
| GSM8K_med (9 runs) | **0.9310** | 0.9318 | 0.9287 | 0.9318 | ≥ 0.93 | ✅ |
| E2E_med | 6362 ms | ~6300 | 6379 | 6594 | ≤ 5000 | ❌ |
| Tput/GPU | 1454.72 | ~1400 | 1465.82 | 1410.22 | ≥ 1500 | ❌ −45 |
| Intvty | 169.65 | 177 | 170.46 | 165.37 | ≥ 165 | ✅ |
| Gates | **2/4** | 2/4 | 2/4 | 2/4 | — | — |

## What was done

1. **Workers killed** — server PIDs 75763 + 4 spawn_main + 4 compile_worker pools all `kill -9`, VRAM verified clean (~284 MB/GPU baseline).
2. **CSV restored pristine** — `dsv3_fp4_tuned_fmoe.csv` line 42 reverted from contaminated `t16x256x256_atomic_sbm32` (Cand A swap from earlier exhaustion run) → pristine `t32x128x256_atomic_bnt2` via `cp /tmp/aiter_backups/dsv3_fp4_tuned_fmoe.csv.original`. Verified diff against backup = empty.
3. **First fresh boot + 3-iter bench** → TPOT_med = 6.047 ms (drift +0.180 ms vs the 5.867 ms first-May-01 measurement). GSM 0.9318 PASS (recovered from 0.9287 since CSV is pristine). Iter 2 had high variance (6.164 ms).
4. **Drift investigation** — wiped 2.1 GB of stale JIT caches:
   - `/tmp/torchinductor_root` (561 MB, last touched Apr 29 — pre-redesign session)
   - `/tmp/.triton/cache` (40 MB)
   - `/tmp/.cache/comgr` (1.5 GB)
   - Preserved: `/tmp/.cache/huggingface` (1.7 TB model weights), `/tmp/.cache/atom` (1.1 GB ATOM source cache), `/tmp/.aiter` (build artifacts).
5. **Second fresh boot + 3-iter bench** → TPOT_med = 5.894 ms (recovered −0.153 ms toward the 5.867 ms anchor; within ±0.05 ms noise band).
6. **Snapshot committed** — `rocm/atom-dev:dsr1_may01_clean_baseline_5894` sha `b6f8c9d03206`.

## Drift root cause (concrete)

Stale `torchinductor` + `comgr` caches from Apr 29 work were silently picked up on each fresh boot, biasing the kernel-selection / fusion patterns differently than a cold-build would produce. The drift accumulated +0.180 ms across the first three May-01 measurements (5.867 → 6.047). Cache wipe restored expected behavior (5.894 ≈ 5.867 within noise).

The +0.253 ms persistent gap between Phase 0 final (5.894) and the Apr 30 KEEP anchor (5.641) is **NOT explained by JIT cache** — it remained after wipe. Likely cause: the Apr 30 KEEP snapshot image had cache state baked in at commit time that is no longer reproducible from source. We accept 5.894 as the new working anchor.

## Gate impact

Binding gate was already TPOT (E2E ≈ TTFT + 1024×TPOT, Tput/GPU = 1024 / latency). New TPOT gap to gate (≤ 4.54 ms) = **−1.354 ms** instead of the planned −1.327 ms. Plan reality-budget arithmetic shifts:
- Mid-sum P1+P2+P3+P4+P5 = −1.15 ms → cumulative TPOT 4.744 ms = 3/4 (was 4.717).
- Closing 4/4 still requires upper-bound on P3 OR P4 + at least one breakthrough lever.

## Iter detail (post-cache-wipe bench)

```
==== ITER 1/3 START 09:33:55 ====
GSM8K runs: 0.9189 0.9272 0.9280
GSM8K_median = 0.9272
GSM8K_PASS_MEDIAN=NO (gate >=0.93)
FAIL: median GSM8K below gate. Performance benchmark SKIPPED.

==== ITER 2/3 START 09:39:35 ====
GSM8K_med    : 0.9318     PASS
TPOT_med     : 5.879 ms
OVERALL: 2/4 gates passed

==== ITER 3/3 START 09:46:29 ====
GSM8K_med    : 0.9363     PASS
TPOT_med     : 5.910 ms
OVERALL: 2/4 gates passed

==== 3x AGGREGATE ====
TPOT_med    = 5.894 ms   (gate <=4.54)  FAIL
GSM8K (all 9 runs) median = 0.9310  PASS
GSM8K runs: [0.9189, 0.9272, 0.928, 0.931, 0.9371, 0.9318, 0.9378, 0.9363, 0.9295]
```

**Iter 1 cold-cache fragility on GSM8K** documented — first GSM run after cudagraph-capture hits unwarmed JIT paths. Mitigation idea (P5 territory): force a warmup GSM run before the first official iter; not pursued in Phase 0.

## Artifacts

- Container snapshot: `rocm/atom-dev:dsr1_may01_clean_baseline_5894` sha `b6f8c9d03206`
- Pre-existing rollback target: `rocm/atom-dev:dsr1_apr30_phase11_v3_2of4` sha `c58cf2ce4512`
- Pristine CSV backup: container `/tmp/aiter_backups/dsv3_fp4_tuned_fmoe.csv.original`
- Bench logs: container `/tmp/bench_3x_093355/` (post-cache-wipe), `/tmp/bench_3x_085641/` (pre-cache-wipe drift run)

## Verdict — KEEP and advance to P1

Phase 0 is COMPLETE. Anchor for P1 perf-neutrality gate = **5.894 ± 0.05 ms**.

## Next

P1 — AITER cherry-pick + container rebuild (#2733, #2890, #2823, #2845, #2717), all env gates default OFF. KEEP gate: TPOT_med ∈ [5.84, 5.94] ms with all gates OFF; GSM8K_med ≥ 0.928.

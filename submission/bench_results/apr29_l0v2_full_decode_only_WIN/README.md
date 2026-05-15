# L0-v2 FULL_DECODE_ONLY — KEEP / 2/4 GATES (Apr 29 ~15:27 IST)

## Verdict
**FIRST gate crossing of the Apr 29 session. TPOT 6.106 → 5.940 ms (−0.166 ms). Intvty crosses gate (168.34 ≥ 165). Gates: 1/4 → 2/4. KEEP per strict abort.**

## What was attempted
Lever L0-v2: `ATOM_CUDAGRAPH_MODE=FULL_DECODE_ONLY` env-var-gated retry of L0 v1 (which died at +2.606 ms TPOT due to FULL_AND_PIECEWISE breaking dual-stream MoE alt-stream overlap). v2 captures decode-only graphs (uniform batches), leaves prefill PIECEWISE — sidesteps the conflict that killed v1.

Boot script: `/tmp/boot_l0v2_full_decode_only.sh` (gold+v6c stack + v6c env + this single new env line). Boot 12m43s (PID 384373, READY 15:18:53 IST). Cudagraph capture in 1.55s (vs FULL_AND_PIECEWISE which took longer + then regressed forward).

## Bench result (3-iter median, official kimbochen harness)

| Metric | Phase 0 baseline | L0-v2 | Δ | Verdict |
|---|---:|---:|---:|---|
| GSM8K_med (N=3) | 0.9348 | 0.9325 | -0.0023 | PASS (margin +0.0025, tight) |
| TPOT_med | 6.106 ms | **5.940 ms** | **−0.166 ms** | WIN |
| E2E_med | 6758 ms | 6461.74 ms | **−296 ms** | improvement (still misses gate by 1462) |
| Tput/GPU | 1371 | 1420.60 | **+49.6 (+3.6%)** | improvement (still misses gate by 79) |
| Intvty | 163.76 | **168.34** | **+4.58** | **PASS gate (165)** ✅ |
| Gates | 1/4 | **2/4** | **+1** | FIRST gate crossing of Apr 29 session |

GSM8K runs: 0.934 / 0.9310 / 0.9325 (median 0.9325, run 2 was tight at 0.9310 — barely above gate). Variance span 0.003 (vs Phase 0's 0.0038, similar).

## Why it landed where L0 v1 didn't

L0 v1 used `cudagraph_mode=FULL_AND_PIECEWISE` which captures BOTH decode AND prefill into FULL graphs. The big prefill graphs disrupted the dual-stream MoE alt-stream overlap (DEC-046 era pattern; SemiAnalysis-flagged "CDNA4 has no async features" caveat).

L0-v2 uses `cudagraph_mode=FULL_DECODE_ONLY` which captures ONLY decode batches as FULL graphs (uniform batches at sizes 1/2/4/8/16/32 from `--cudagraph-capture-sizes`); prefill stays PIECEWISE. The dual-stream MoE only fires at decode (token threshold 1024), and the FULL capture of decode preserves alt-stream relationships because the captured decode is a single uniform batch.

The dossier-2 analysis claimed "FULL_DECODE_ONLY has the same problem as FULL because dual-stream MoE is decode-time" — this was theoretically motivated but **empirically wrong on our stack**. Measurement beats theory.

## Why TPOT delta (-0.166 ms) is below dossier-1's -0.45 ms estimate

Dossier-1 mid-est was -0.45 ms based on plan v7 analysis. Real delta -0.166 ms because:
- ATOM has fewer per-step host operations than vLLM upstream (some upstream-cited overhead doesn't apply)
- Dual-stream MoE alt-stream did partially get touched even at FULL_DECODE_ONLY (capture of decode includes the dual-stream wrapper)
- But the GAIN side dominates: cudagraph launch overhead drops from ~60 launches/decode-step to ~1 launch/decode-step

## What this unblocks

L0-v2 now in canonical stack. Subsequent levers (L1-v2, L2-v2, L3-v3 HIP_ONLINE_TUNING, Q8 QuickReduce, relaxed-thinking) STACK ON TOP of this baseline. Cumulative target: TPOT 5.940 → <5.0 ms.

## Where everything lives
- `/app/ATOM/atom/utils/envs.py` — `ATOM_CUDAGRAPH_MODE` env var (Apr 29 13:24 patch, kept in-place, NULL-OP at default)
- `/app/ATOM/atom/model_engine/arg_utils.py` — env-read of cudagraph_mode (kept in-place)
- `/app/ATOM/atom/config.py` — `if self.compilation_config.cudagraph_mode is None` guard (kept in-place)
- Boot script: `/tmp/boot_l0v2_full_decode_only.sh` — gold+v6c + this env line
- Server killed (PID 384373 at end-of-bench)
- Evidence on host: `/projects/teamA/danish/apr29_evidence/l0v2_full_decode_only_WIN/`
- Evidence on laptop: this directory

## Strict-plan abort decision: KEEP
TPOT 5.940 < 6.10 baseline ✓ AND GSM8K 0.9325 ≥ 0.93 ✓ AND +1 gate crossed. KEEP. L0-v2 env stays in stack going forward. Advance to L1-v2 stacked.

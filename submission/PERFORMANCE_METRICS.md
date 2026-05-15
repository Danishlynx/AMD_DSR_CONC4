# Performance Metrics â€” DSR1 / MI355X CONC=4 Submission

**AMD Submission Item 4: Performance metrics documentation (throughput per GPU)**

**Snapshot**: `rocm/atom-dev:dsr1_apr30_phase11_v3_2of4` (sha `c58cf2ce4512`)
**Date measured**: 2026-04-30 (CONC=4), 2026-04-27 (CONC=32 reference)
**Hardware**: 4Ã— AMD Instinct MI355X (gfx950, CDNA4), TP=4 single-replica, MTP=3, FP8 KV cache
**Harness**: `kimbochen/dsr1_benchmark.cpp` â€” ISL=8192, OSL=1024, num_prompts=40, max-concurrency=4, num_warmups=8, 4-iter median(2,3,4); GSM8K via lm_eval N=3 median
**Activation**: `ATOM_ENABLE_PER_PHASE_RELAXED_MTP=1` + `ATOM_CUDAGRAPH_MODE=FULL_DECODE_ONLY` + `ATOM_ENABLE_RELAXED_MTP=1` (stack documented in `docs/Daily Updates/SERVER.md`)

---

## Headline: CONC=4 (Apr 30, official kimbochen harness)

| Metric | Value | Gate | Status |
|---|---:|---:|:---:|
| **GSM8K (N=3 median)** | **0.9318** | â‰¥ 0.93 | âœ… **PASS** (margin +0.0018) |
| Median TPOT | **5.641 ms** | â€” | (drives Interactivity â†‘, E2E â†“) |
| **Throughput per GPU** | **1449 tok/s** | â‰¥ 1500 | âŒ FAIL (off by 51 = âˆ’3.4%) |
| **Interactivity (tok/s/user)** | **177.26** | â‰¥ 165 | âœ… **PASS** (margin +12 = +7%) |
| Median E2E latency | **6210 ms** | â‰¤ 5000 | âŒ FAIL (off by 1210 = +24%) |
| **GATES** | | | **2/4** âœ…âœ…âŒâŒ |

Delta vs L0-v2 baseline (Apr 30 same-server measurement):

| Metric | L0-v2 baseline | This submission | Î” |
|---|---:|---:|---:|
| GSM8K (N=3 median) | 0.9386 | 0.9318 | âˆ’0.0068 |
| TPOT median | 6.302 ms | **5.641 ms** | **âˆ’0.661 ms** |
| Throughput per GPU | 1387 tok/s | **1449 tok/s** | **+62** |
| Interactivity | 158.68 FAIL | **177.26 PASS** | **+18.58** *(+1 gate crossed)* |
| E2E median | 6723 ms | 6210 ms | âˆ’513 ms |
| **Gates** | 1/4 | **2/4** | **+1** |

---

## CONC=32 reference (Apr 27, A27 baseline â€” Phase 11 v3 not yet applied)

The same submission stack **minus the Phase 11 v3 sampler kernel** was independently measured at CONC=32 on Apr 27:

| Metric | Value | Gate | Status |
|---|---:|---:|:---:|
| **GSM8K (single run)** | **0.9431** | â‰¥ 0.93 | âœ… **PASS** |
| Median TPOT | 17.80 ms | â€” | â€” |
| **Throughput per GPU** | **3831 tok/s** | â‰¥ 3900 | âŒ FAIL (off by 69 = âˆ’1.8%) |
| **Interactivity** | **56.17** | â‰¥ 50 | âœ… **PASS** (margin +6.17 = +12%) |
| Median E2E latency | **19044 ms** | â‰¤ 18000 | âŒ FAIL (off by 1044 = +5.8%) |
| **GATES** | | | **2/4** âœ…âœ…âŒâŒ |

> **Caveat**: Phase 11 v3 was added 3 days *after* this measurement. The lever is concurrency-agnostic (sampler change), so it would likely improve CONC=32 too â€” but that improvement is **projected, not measured**.
>
> **Why this matters for follow-up**: failing gates at CONC=32 miss by only **5.8%** (E2E) and **1.8%** (Tput) â€” vs much larger gaps at CONC=4. **CONC=32 4/4 is likely reachable before CONC=4 4/4** as additional levers stack.

---

## TPOT-reduction trajectory (full campaign, 5 weeks)

| Layer | Lever | Type | TPOT Î” | Cumulative |
|---|---|---|---:|---:|
| Baseline | Vanilla TP=8 MTP=3 fp8 KV | â€” | â€” | ~7.88 ms (TP=4) |
| #1 | TP=8 â†’ TP=4 single-replica | config | â€” | 7.88 ms |
| #2 | `ATOM_ENABLE_RELAXED_MTP=1` *(stock ATOM flag â€” was OFF)* | env flag | **âˆ’2.29 ms** | 5.59 ms |
| #3 | `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4` | env flag | minimal | 5.59 ms |
| #4 | RCCL_MSCCLPP knobs (ROCm 7.1+) | env flags | **âˆ’0.18 ms** | 5.41 ms |
| #5 | `rocm-smi --resetperfdeterminism` (SCLK boost) | platform | **âˆ’0.08 ms** | 5.33 ms |
| #6 | **`--cudagraph-capture-sizes [1,2,4,8,16,32]`** | CLI flag | **âˆ’1.41 ms** â­ | 4.84 ms *(warm informal-bench)* |
| #7 | 8-curl warmup pattern (vs 5 large prompts) | bench discipline | within #6 | 4.84 ms |
| #8 | `RELAXED_TOP_N` 8 â†’ 9 | sampler | small | 4.84 ms |
| #9 | `ATOM_MSCG_K` unset (was 2 â€” silent regression removed) | config | **âˆ’0.05 ms** | 4.84 ms |
| â€” | *Apr 27: official-harness re-baseline (N=3)* | â€” | â€” | **6.171 ms** |
| #10 | `ATOM_CUDAGRAPH_MODE=FULL_DECODE_ONLY` (L0-v2) | env flag | **âˆ’0.166 ms** | 6.005 ms |
| #11 | **Phase 11 v3 â€” TRT-LLM thinking port** â­ *(this PR)* | **first-party Triton kernel + plumbing** | **âˆ’0.661 ms** | **5.641 ms** |

### Summary by category

| Category | Î” TPOT | Levers |
|---|---:|---|
| Stock ATOM env flags (were OFF by default) | **âˆ’2.29 ms** | `ATOM_ENABLE_RELAXED_MTP` |
| CLI / cudagraph configuration | **âˆ’1.58 ms** | `--cudagraph-capture-sizes` + `FULL_DECODE_ONLY` |
| **First-party source-level kernel + plumbing (this PR)** | **âˆ’0.661 ms** | **Phase 11 v3** |
| Comm + platform knobs | âˆ’0.26 ms | RCCL_MSCCLPP + perf-determinism + MSCG_K unset |
| **Net** | **âˆ’2.24 ms / âˆ’28%** | (over 5 weeks) |

---

## What this PR specifically delivers

Of the **âˆ’2.24 ms** total reduction across the campaign, this PR's first-party source-level contribution is **âˆ’0.661 ms** via the Phase 11 v3 TRT-LLM thinking port. **That âˆ’0.661 ms crosses Interactivity from 158.68 (FAIL) â†’ 177.26 (PASS) â€” the +1 gate this PR delivers.**

The other levers in the trajectory are either pre-existing flags AMD/ATOM already ship (which were just toggled ON), or CLI configurations â€” those are documented for reproducibility but are not source modifications in this PR.

**Files modified by this PR** (in ATOM_main):
- `atom/utils/envs.py` â€” adds `ATOM_ENABLE_PER_PHASE_RELAXED_MTP` env flag (+2 lines)
- `atom/model_engine/model_runner.py` â€” `spec_phase` int8 tensor allocation + setter registration + prefill-slot reset (+18 lines)
- `atom/model_ops/rejection_sampler.py` â€” `rejection_phased_sample_kernel` Triton kernel + module globals + env-gated dispatcher (+~190 lines)

**Total**: ~210 LOC added, 0 removed, 3 files. Default OFF (`ATOM_ENABLE_PER_PHASE_RELAXED_MTP=0`) â†’ bit-identical to upstream.

---

## Raw evidence

| Artifact | Location |
|---|---|
| Kimbochen 4-iter JSON (CONC=4 Apr 30) | `bench_results/apr30_phase11_per_phase_mtp_v3_KEEP_2of4/evidence.tgz` |
| GSM8K N=3 logs (CONC=4 Apr 30) | (same evidence.tgz) |
| Boot log | (same evidence.tgz) |
| CONC=32 reference data (Apr 26 informal-bench multi-CONC) | `bench_results/apr26/conc32_warm_run{1,2}.json` |
| CONC=32 official harness data (Apr 27) | `docs/Daily Updates/MASTER.md` Â§"A27 CONC=32 reference" |
| Apr 29 L0-v2 evidence (L0-v2 lever) | `bench_results/apr29_l0v2_full_decode_only_WIN/` |
| Phase 0 anchor reference | `bench_results/may01_phase0_clean_baseline/` |

---

## See also

- [`README.md`](README.md) â€” top-level overview + repo structure
- [`PR_DESCRIPTION.md`](PR_DESCRIPTION.md) â€” full PR description (item 2)
- [`TECHNICAL_APPROACH.md`](TECHNICAL_APPROACH.md) â€” profiling methodology + decision tree (item 5)
- [`docs/Daily Updates/REPRODUCE.md`](docs/Daily%20Updates/REPRODUCE.md) â€” single-command reproduction recipe
- [`docs/Daily Updates/OFFICIAL_HARNESS.md`](docs/Daily%20Updates/OFFICIAL_HARNESS.md) â€” kimbochen harness contract

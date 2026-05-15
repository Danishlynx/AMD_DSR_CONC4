# Performance Metrics — DSR1 / MI355X CONC=4 Submission

**AMD Submission Item 4: Performance metrics documentation (throughput per GPU)**

**Snapshot**: `rocm/atom-dev:dsr1_apr30_phase11_v3_2of4` (sha `c58cf2ce4512`)
**Date measured**: 2026-04-30 (CONC=4), 2026-04-27 (CONC=32 reference)
**Hardware**: 4× AMD Instinct MI355X (gfx950, CDNA4), TP=4 single-replica, MTP=3, FP8 KV cache
**Harness**: `kimbochen/dsr1_benchmark.cpp` — ISL=8192, OSL=1024, num_prompts=40, conc=4, num_warmups=8, 4-iter median(2,3,4); GSM8K via lm_eval N=3 median
**Activation**: `ATOM_ENABLE_PER_PHASE_RELAXED_MTP=1` + `ATOM_CUDAGRAPH_MODE=FULL_DECODE_ONLY` + `ATOM_ENABLE_RELAXED_MTP=1`

---

## Headline: CONC=4 (Apr 30, official kimbochen harness)

| Metric | Value | Gate | Status |
|---|---:|---:|:---:|
| **GSM8K (N=3 median)** | **0.9318** | ≥ 0.93 | ✅ **PASS** (margin +0.0018) |
| Median TPOT | **5.641 ms** | — | (drives Interactivity ↑, E2E ↓) |
| **Throughput per GPU** | **1449 tok/s** | ≥ 1500 | ❌ FAIL (off by 51 = −3.4%) |
| **Interactivity (tok/s/user)** | **177.26** | ≥ 165 | ✅ **PASS** (margin +12 = +7%) |
| Median E2E latency | **6210 ms** | ≤ 5000 | ❌ FAIL (off by 1210 = +24%) |
| **GATES** | | | **2/4** ✅✅❌❌ |

Delta vs L0-v2 baseline (Apr 30 same-server measurement):

| Metric | L0-v2 baseline | This submission | Δ |
|---|---:|---:|---:|
| GSM8K (N=3 median) | 0.9386 | 0.9318 | −0.0068 |
| TPOT median | 6.302 ms | **5.641 ms** | **−0.661 ms** |
| Throughput per GPU | 1387 tok/s | **1449 tok/s** | **+62** |
| Interactivity | 158.68 FAIL | **177.26 PASS** | **+18.58** *(+1 gate crossed)* |
| E2E median | 6723 ms | 6210 ms | −513 ms |
| **Gates** | 1/4 | **2/4** | **+1** |

---

## CONC=32 reference (Apr 27, A27 baseline — Phase 11 v3 not yet applied)

| Metric | Value | Gate | Status |
|---|---:|---:|:---:|
| **GSM8K (single run)** | **0.9431** | ≥ 0.93 | ✅ **PASS** |
| Median TPOT | 17.80 ms | — | — |
| **Throughput per GPU** | **3831 tok/s** | ≥ 3900 | ❌ FAIL (off by 69 = −1.8%) |
| **Interactivity** | **56.17** | ≥ 50 | ✅ **PASS** (margin +6.17 = +12%) |
| Median E2E latency | **19044 ms** | ≤ 18000 | ❌ FAIL (off by 1044 = +5.8%) |
| **GATES** | | | **2/4** ✅✅❌❌ |

**Caveat**: Phase 11 v3 was added 3 days *after* this measurement (Apr 30 vs Apr 27). The lever is concurrency-agnostic (sampler change), so it would likely improve CONC=32 too — but that improvement is **projected, not measured**.

**Why this matters**: CONC=32 failing gates miss by only **5.8%** (E2E) and **1.8%** (Tput) — vs much larger gaps at CONC=4. **CONC=32 4/4 is likely reachable before CONC=4 4/4** as additional levers stack.

---

## TPOT-reduction trajectory (full campaign, 5 weeks)

| Layer | Lever | Type | TPOT Δ | Cumulative |
|---|---|---|---:|---:|
| Baseline | Vanilla TP=8 MTP=3 fp8 KV | — | — | ~7.88 ms (TP=4) |
| #1 | TP=8 → TP=4 single-replica | config | — | 7.88 ms |
| #2 | `ATOM_ENABLE_RELAXED_MTP=1` *(stock ATOM flag — was OFF)* | env flag | **−2.29 ms** | 5.59 ms |
| #3 | `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4` | env flag | minimal | 5.59 ms |
| #4 | RCCL_MSCCLPP knobs (ROCm 7.1+) | env flags | **−0.18 ms** | 5.41 ms |
| #5 | `rocm-smi --resetperfdeterminism` (SCLK boost) | platform | **−0.08 ms** | 5.33 ms |
| #6 | **`--cudagraph-capture-sizes [1,2,4,8,16,32]`** | CLI flag | **−1.41 ms** ⭐ | 4.84 ms *(warm informal-bench)* |
| #7 | 8-curl warmup pattern | bench discipline | within #6 | 4.84 ms |
| #8 | `RELAXED_TOP_N` 8 → 9 | sampler | small | 4.84 ms |
| #9 | `ATOM_MSCG_K` unset | config | −0.05 ms | 4.84 ms |
| — | *Apr 27: official-harness re-baseline (N=3)* | — | — | **6.171 ms** |
| #10 | `ATOM_CUDAGRAPH_MODE=FULL_DECODE_ONLY` (L0-v2) | env flag | **−0.166 ms** | 6.005 ms |
| #11 | **Phase 11 v3 — TRT-LLM thinking port** ⭐ *(this PR)* | **first-party Triton kernel + plumbing** | **−0.661 ms** | **5.641 ms** |

### Summary by category

| Category | Δ TPOT | Levers |
|---|---:|---|
| Stock ATOM env flags (were OFF by default) | **−2.29 ms** | `ATOM_ENABLE_RELAXED_MTP` |
| CLI / cudagraph configuration | **−1.58 ms** | `--cudagraph-capture-sizes` + `FULL_DECODE_ONLY` |
| **First-party source-level kernel + plumbing (this PR)** | **−0.661 ms** | **Phase 11 v3** |
| Comm + platform knobs | −0.26 ms | RCCL_MSCCLPP + perf-determinism + MSCG_K unset |
| **Net** | **−2.24 ms / −28%** | (over 5 weeks) |

---

## What this PR specifically delivers

Of the −2.24 ms total reduction across the campaign, this PR's first-party source-level contribution is **−0.661 ms** via the Phase 11 v3 TRT-LLM thinking port. **That −0.661 ms crosses Interactivity from 158.68 (FAIL) → 177.26 (PASS) — the +1 gate this PR delivers.**

**Files modified in this PR** (in ATOM upstream):
- `atom/utils/envs.py` — adds `ATOM_ENABLE_PER_PHASE_RELAXED_MTP` env flag (+2 lines)
- `atom/model_engine/model_runner.py` — `spec_phase` int8 tensor allocation + setter + prefill-slot reset (+18 lines)
- `atom/model_ops/rejection_sampler.py` — `rejection_phased_sample_kernel` Triton kernel + module globals + env-gated dispatcher (+~190 lines)

**Total**: ~210 LOC added, 0 removed, 3 files. Default OFF (`ATOM_ENABLE_PER_PHASE_RELAXED_MTP=0`) → bit-identical to upstream.

---

## Raw evidence

| Artifact | Location |
|---|---|
| Kimbochen 4-iter JSON + GSM8K log + boot log (Apr 30 CONC=4) | [`submission/bench_results/apr30_phase11_per_phase_mtp_v3_KEEP_2of4/evidence.tgz`](submission/bench_results/apr30_phase11_per_phase_mtp_v3_KEEP_2of4/) |
| Lever description + result table | [`submission/bench_results/apr30_phase11_per_phase_mtp_v3_KEEP_2of4/README.md`](submission/bench_results/apr30_phase11_per_phase_mtp_v3_KEEP_2of4/) |

# DSR_beta Phase A Results (Apr 18 2026)

Testing untried ATOM flags + AMD article-backed flags, one at a time.

## Summary

**Winner**: `ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=1024` (up from DEC-056 value of 256) → +0.7% over TBO prefill alone.

**New DSR_beta best**: **1344 thr/GPU, 6.36 TPOT, 157.34 interact, 6896 E2E**

**Cumulative gains vs DEC-075 production floor**:
- Thr/GPU: 1278 → 1344 (**+5.2%**)
- Median TPOT: 6.74 → 6.36 (**−5.6%**)
- Interactivity: 148 → 157 (**+5.7%**) — gap to 165 gate narrowed 10.3% → 4.8%
- Median E2E: 7253 → 6896 (**−4.9%**)
- Gates: still 1/4 (GSM8K)

## Full test matrix

| # | Test | Config delta | Result | Verdict |
|---|---|---|---|---|
| Baseline | DSR_beta + TBO prefill | — | 1335/6.40/156/7009 | reference |
| **A'** | + EP + MORI low-latency | `--enable-expert-parallel --all2all-backend low-latency` | 1206/7.07/141/7587 | ❌ DEAD (−9.7% thr) — MoRI needs inter-node RDMA |
| **A''** | + prefix caching | `--enable_prefix_caching` | GSM8K infra crash | ❌ DEAD — still incompatible with MXFP4+MTP on ROCm 7.2.2 |
| **A** | + DP attention | `--enable-dp-attention` | GSM8K 0.906 | ❌ DEAD — breaks accuracy (below 0.93 gate) |
| **B** | + max-num-seqs 4 | `--max-num-seqs 4 --cudagraph-capture-sizes "[1,2,4]"` | GSM8K timeout | ❌ DEAD — bench runs 65 concurrent, server caps at 4 |
| **C** | + DUAL_STREAM=1024 | `ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=1024` (was 256) | **1344/6.36/157/6896** | ✅ **+0.7% WIN** |
| **C2** | + DUAL_STREAM=4096 | `ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=4096` | GSM8K 0.9287 | ❌ DEAD — too aggressive breaks accuracy |

## Key findings

### What works

**`ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=1024`** (default upstream value):
- Activates dual-stream MoE for all our decode and small-batch prefill
- Parallelizes shared experts + routed experts on 2 HIP streams
- Marginal win (+0.7%) but consistent direction on every metric

### What doesn't (and why)

**EP + MORI AsyncLL** — article says 82% latency reduction, but that's for multi-node RDMA. We're single-node TP=4. No RDMA = overhead dominates.

**Prefix caching** — still crashes MXFP4+MTP pipeline despite ROCm 7.2.2 upgrade. Deeper protocol issue, not the None-scale bug.

**DP attention** — breaks MTP correctness (drops GSM8K below 0.93).

**max-num-seqs 4** — bench harness runs 65 concurrent for GSM8K accuracy test. Can't restrict server below 65.

**DUAL_STREAM=4096** — too much overlap causes accumulated numerical drift. Barely breaks GSM8K (0.9287 vs 0.93).

## AMD article context (informed this phase)

AMD "Speed is the Moat" (Feb 17 2026) article identified:
- **TBO** (Two-Batch Overlap): doubled prefill throughput → we have `--enable-tbo prefill` (+4.4%)
- **MoRI-EP low-latency**: 82% EP kernel latency reduction → tested, **dead for single-node** (needs RDMA)
- **MTP**: key interactivity feature → we use MTP=3

Phase A exhausted article's low-hanging fruit for single-node. The remaining gains are tree speculation (compute-level algorithm change) and BF16 GEMM retune (config-level polish).

## Next

- **Phase B**: BF16 CSV retune on ROCm 7.2.2 (2-4 hrs autonomous, +1-2% expected)
- **Phase C**: Tree speculation implementation (6-10 hrs, the real gate closer — targets +15-30% TPOT)
- **Phase D**: EP+MORI saved for CONC=128 track (different regime)

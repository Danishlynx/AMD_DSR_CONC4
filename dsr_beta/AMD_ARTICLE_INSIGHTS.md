# AMD "Speed is the Moat" (Feb 17 2026) — Insights for our DSR1 track

Source: https://www.amd.com/en/developer/resources/technical-articles/2026/speed-is-the-moat

## Top 3 actionable insights

### 1. MoRI low-latency REQUIRES EP to deliver — we tested it WRONG

**Article**:
- *"Adaptive Kernel Selection: High-throughput kernels are used for prefill and high-concurrency decode, while **low-latency kernels are activated for low-concurrency scenarios**"*
- *"Expert Parallelism (MORI-EP): Purpose-built for large scale MoE models like DeepSeek-R1. Recent kernel-level optimizations have reduced latency by **up to 82%**, driving HBM, XGMI, and RDMA communication overheads close to their theoretical upbound."*

**Our previous test**: `--all2all-backend low-latency` at TP-sharded MoE (no EP) → neutral/−1%. Logical — there's no EP all2all to overlap.

**Correction**: MORI AsyncLL is for **EP + low-concurrency**. CONC=4 IS low concurrency. The untested combo:
```
--enable-expert-parallel --all2all-backend low-latency
```

**Caveat**: EP crashed on ROCm 7.1.1 (gfx950 kernel bugs). New ROCm 7.2.2 + aiter main may have fixes.

### 2. TBO prefill = validated main lever (already in our stack)

- Article's *"approximately doubling prefill throughput through parallelism restructuring"* IS TBO.
- Our `--enable-tbo prefill` delivered +4.4%.
- Article's claim of "1.08x-1.2x single-node uplift" matches our 1.044x.
- Confirmed: we have the right lever. No action.

### 3. H1 Q1 features we may not be using

| Feature | Article | Our status | Action |
|---|---|---|---|
| Chunked Prefill | Q1 2026 | Never tested | Try `--enable-chunked-prefill` if flag exists |
| Prefix Cache | Q1 2026 | DEAD on MXFP4 (old stack) | Retry on DSR_beta — may be fixed |
| Sparse/Radix Attention | Q1 2026 | For Qwen, not DSR1 | skip |
| FP4 | Q1 2026 | ✓ we're on MXFP4 | no-op |

## Updated Tier 1 test order (post-article)

| # | Test | Article-backed? | Priority |
|---|---|---|---|
| **A'** | `--enable-expert-parallel` + `--all2all-backend low-latency` | ⭐ **82% EP latency claim** | **TOP** |
| A'' | `--enable-prefix-caching` (retry on new stack) | Q1 feature, may be fixed | High |
| A''' | Chunked prefill (if flag exists) | Q1 feature | High |
| A | `--enable-dp-attention` | Implied via "communication bubbles" | Medium |
| B | `--max-num-seqs 4` | — | Low |
| C | `ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=1024` | — | Low |

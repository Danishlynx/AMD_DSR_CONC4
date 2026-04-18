# AMD Phase 2 Hackathon — DSR1 Track — DSR_beta Snapshot

This repo is a **backup + reproducibility package** for Danish's DeepSeek-R1 (DSR1) track submission.

## Branch structure

| Branch | Content | Purpose |
|---|---|---|
| **`main`** | DEC-075 production floor (1278/6.74/148/7253) | Stable fallback — proven reproducible on ROCm 7.1.1 stack |
| **`dsr_beta_snapshot`** (this branch) | **DSR_beta WIN (1335/6.40/156/7009)** | Latest best — ROCm 7.2.2 + latest aiter/ATOM/flydsl + TBO prefill |

## Current best result (this branch — DSR_beta + TBO prefill, 2026-04-18)

Measured via the official harness `./dsr1_benchmark perf`:

| Metric | Value | Gate | Pass? |
|---|---|---|---|
| Thr/GPU (÷4) | **1335** | ≥ 1500 | ❌ −11% |
| Median TPOT | **6.40 ms** | — | — |
| Mean TPOT | **6.18 ms** | — | — |
| Interactivity | **156.24** | ≥ 165 | ❌ −5.5% |
| Median E2E | **7009 ms** | ≤ 5000 | ❌ +40% |
| Median ITL | **16.18 ms** | — | — (MTP=3 burst ✓) |
| GSM8K | **0.9386** | ≥ 0.93 | ✅ |
| **Gates** | **1/4** | — | GSM8K only |

Total throughput: **5339 tok/s** at ISL=8192, OSL=1024, CONC=4.

## Gains vs production DEC-075 floor (`main` branch)

| Metric | DEC-075 (main) | DSR_beta (this branch) | Δ |
|---|---|---|---|
| Thr/GPU | 1278 | **1335** | **+4.4%** |
| Median TPOT | 6.74 | **6.40** | **−5.0%** |
| Interactivity | 148 | **156** | **+5.4%** |
| Median E2E | 7253 | **7009** | **−3.4%** |
| GSM8K | ≥0.93 | 0.9386 | stable |

Interactivity gap to gate narrowed from 10.3% → 5.5%.

## Stack (this branch)

- **Base image**: `rocm/atom-dev@sha256:52c5195a712b5d3a0993d5e63de9b8ffc13a77d0c4b2f31d40afe9e62c12ab5f` (tag `rocm/atom-dev:latest`, 2026-04-17)
- **ROCm**: 7.2.2 (vs prod 7.1.1)
- **PyTorch**: 2.10.0+rocm7.2.2.git40d237bf (vs prod 2.9)
- **aiter**: commit `73ad0023e15e9735b3af95b3357b99cf7f801bf1` on main (v0.1.12.post1+)
- **ATOM**: commit `f8453e3fc0f65191fb2034602dc9a2066a78020b` on main (v0.1.3.dev90+, includes TBO)
- **flydsl**: 0.1.3.1 (vs prod 0.1.2)
- **triton**: 3.5.1
- **Model**: `/projects/teamA/danish/models_merged/DSR1-drafter-FP4` — DEC-075 merged checkpoint (layer 61 MoE swapped from MoEFP4 variant for FP4 drafter fast path)
- **Hardware**: 4× AMD Instinct MI355X, TP=4 single-replica, port **8890**
- **KV cache**: FP8
- **Speculation**: native DeepSeek MTP (k=3), relaxed acceptance (top_n=8, delta=0.5)
- **⭐ Key flag**: `--enable-tbo prefill` (new in ATOM Apr 16) — the one delta from DEC-075 that delivered the win

## Local patches (mergeable against upstream)

Full diff: [dsr_beta/patches/dsr_beta_local_mods.diff](dsr_beta/patches/dsr_beta_local_mods.diff)

| File | Change | Rationale |
|---|---|---|
| `atom/model_ops/rejection_sampler.py` | `RELAXED_TOP_N = 10 → 8`, `RELAXED_DELTA = 0.6 → 0.5` | Sweep showed this point maxes accept-rate while passing GSM8K gate (DEC-073 tuning) |
| `atom/model_ops/attention_mla.py` | `num_kv_splits=16 → None` (auto-tune) | Manual value 16 was suboptimal for TP=4 SR shape (Session 6A intervention) |

## Reproduction

Three scripts in [dsr_beta/scripts/](dsr_beta/scripts/):

```bash
# 1. Create DSR_beta container with pinned image + apply local patches
bash dsr_beta/scripts/dsr_beta_setup.sh

# 2. Launch server with the winning recipe (--enable-tbo prefill)
bash dsr_beta/scripts/dsr_beta_launch.sh

# 3. Bench
bash dsr_beta/scripts/dsr_beta_bench.sh
```

Full step-by-step recipe: [dsr_beta/REPRODUCTION.md](dsr_beta/REPRODUCTION.md)

## Experiments log (all 4 DSR_beta runs on Apr 18)

| # | Config | Thr/GPU (÷4) | Median TPOT | Interact | E2E | Verdict |
|---|---|---|---|---|---|---|
| 1 | Baseline (upgrade only, no TBO) | 1315 | 6.56 | 152.46 | 7150 | +2.9% vs DEC-075 |
| **2** | **+ TBO prefill** | **1335** | **6.40** | **156.24** | **7009** | **+4.4% vs DEC-075 (BEST)** |
| 3 | + TBO all | 939 | 9.53 | 104.96 | 10099 | −30% regression |
| 4 | + MORI low-latency on top of (2) | 1322 | 6.64 | 150.52 | 7201 | −1% (overhead at TP-sharded MoE) |

## Experiments deferred (regressed at CONC=4, may work at higher CONC)

Archive of CONC-conditional dead levers — retry in CONC=32/128 tracks:

- `--enable-tbo all` — designed for larger batches (CONC=32+)
- `--all2all-backend low-latency` (MORI AsyncLL) — needs EP to have work to overlap; retry at CONC=128 + EP=8
- `--enable-expert-parallel` — crashed on ROCm 7.1.1, retry on this new stack
- QuickReduce INT4 — min 16 MB tensor, valid at CONC=128 prefill
- DP×TP combos — retry on ROCm 7.2.2

Permanently dead (no retry): `--num-speculative-tokens 4` (MLA qseqlen=5 kernel missing on gfx950).

## DEC lineage (historical — on main branch)

| DEC | Lever | Result |
|---|---|---|
| DEC-056 | DUAL_STREAM=256 | floor 1209/6.89 |
| DEC-066 | BF16 CSV 9 rows | 1221/6.73/148.6 |
| DEC-071 | BF16 decode tune (97 rows) | 1267/6.96/143.8 |
| DEC-072 | BF16 prefill tune | DEAD — GSM8K 0.865 crash |
| DEC-073 | Relaxed MTP (8, 0.5) | 1270/6.80/147.1/7318 |
| DEC-074 | Naive tree spec | ABANDONED — kernel refactor regressed |
| DEC-075 | **Drafter FP4 transplant (merged checkpoint)** | **1278/6.74/148/7253 (main branch floor)** |
| **Apr 18** | **DSR_beta stack + TBO prefill** | **1335/6.40/156/7009 (this branch)** |

## Open candidates (pending work)

1. **Other new ATOM flags** — deep-dive into upstream arg_utils.py since our pin for undocumented flags
2. **BF16 GEMM CSV retune** on ROCm 7.2.2 — old DEC-071 tune incompatible (hipBLASLt solidx renumbered between 7.1.1 → 7.2.2). Expected +1-2% after retune.
3. **Tree speculation** — EAGLE-2 style on the DSR_beta stack. The real gate closer (current kernel floor blocks E2E at 4.52 TPOT).
4. **EP=8 + MORI** stack — reserved for CONC=128 track.

See [docs/Current_plan.md](docs/Current_plan.md) for live status.

## Repository layout

```
.
├── README.md                                 ← you are here
├── dsr_beta/                                 ← DSR_beta snapshot (this branch's key addition)
│   ├── REPRODUCTION.md                       ← full reproduction recipe with image digest
│   ├── patches/dsr_beta_local_mods.diff      ← the 2 local patches
│   ├── scripts/
│   │   ├── dsr_beta_setup.sh                 ← pull image + create container + apply patches
│   │   ├── dsr_beta_launch.sh                ← launch server (winning recipe)
│   │   └── dsr_beta_bench.sh                 ← bench runner
│   └── bench_results/
│       └── dsr_beta_tbo_prefill_WIN.json     ← the 1335/6.40/156 bench JSON
├── ATOM_main/                                ← modified ATOM source tree (DEC-075 production state)
├── aiter_configs/                            ← 97-row BF16 tune (DEC-071, ROCm 7.1.1 — NOT compatible with 7.2.2)
├── bench_results/                            ← raw test_*.json from official harness (DEC-075)
├── session_logs/                             ← chronological engineering logs
├── docs/
│   ├── SERVER_MAP.md
│   ├── BRIEF_FOR_KIMI_OPUS.md
│   ├── Current_plan.md                       ← live plan state
│   ├── MASTER_FINDINGS.md                    ← canonical findings
│   ├── daily_log.md                          ← chronological DEC record
│   ├── Danish.md                             ← strategic context
│   └── best_reproduce.md                     ← DEC-075 production reproduction
├── scripts/                                  ← DEC-075 helpers (merge_dec075_v5.py, parse_trace.py)
└── patches/                                  ← (future) clean diffs for upstream PRs
```

## Legal / usage

Private snapshot, backup only. Do not redistribute without Danish's permission. AMD ATOM code is licensed per ROCm/ATOM repository's original license.

---

Last update: 2026-04-18 DSR_beta + TBO prefill = 1335/6.40/156/7009/0.9386 (1/4 gates, interact gap 5.5%).

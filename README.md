# AMD Phase 2 Hackathon — DSR1 Track Submission

**Branch**: `main` (clean submission)
**Author**: Danish Lynx (`danishlynx@gmail.com`)
**Best snapshot**: `rocm/atom-dev:dsr1_apr30_phase11_v3_2of4` (sha `c58cf2ce4512`)
**Harness**: official `kimbochen/dsr1_benchmark.cpp` — ISL=8192, OSL=1024, num_prompts=40, max-concurrency=4, num_warmups=8, 4-iter median(2,3,4), GSM8K N=3 median ≥ 0.93

---

## TL;DR

**CONC=4** (Apr 30, official kimbochen harness, with Phase 11 v3 applied):

| Metric | Stock baseline (L0-v2) | **This submission** | Δ | Gate |
|---|---:|---:|---:|---|
| GSM8K (N=3 median) | 0.9386 | **0.9318 PASS** | −0.0068 | ≥ 0.93 ✅ |
| **TPOT median** | 6.302 ms | **5.641 ms** | **−0.661 ms** | drives E2E + Intvty |
| **Throughput per GPU** | 1387 | **1449** | **+62** | ≥ 1500 ❌ (off 51) |
| **Interactivity (tok/s/user)** | 158.68 FAIL | **177.26 PASS** | **+18.58** | ≥ 165 ✅ |
| Median E2E latency | 6723 ms | 6210 ms | −513 ms | ≤ 5000 ❌ (off 1210) |
| **Gates** | **1/4** | **2/4** | **+1** | |

**CONC=32** (Apr 27, official kimbochen harness — measured on the A27 baseline stack that Phase 11 v3 builds on top of):

| Metric | A27 baseline @ CONC=32 | Gate | Status |
|---|---:|---:|:---:|
| GSM8K (single run) | **0.9431** | ≥ 0.93 | ✅ PASS |
| **Interactivity (tok/s/user)** | **56.17** | ≥ 50 | ✅ PASS (+12% margin) |
| Median E2E latency | 19044 ms | ≤ 18000 | ❌ FAIL (−5.8%) |
| Throughput per GPU | 3831 | ≥ 3900 | ❌ FAIL (−1.8%) |
| TPOT median | 17.80 ms | — | — |
| **Gates** | **2/4** | | |

*Same stack as Apr 30 CONC=4 minus the Phase 11 v3 sampler kernel; Phase 11 v3 is concurrency-agnostic and would likely improve CONC=32 too, but we did not explicitly re-bench CONC=32 after Apr 30. **At CONC=32 the missing gates are within striking distance (E2E −5.8%, Tput −1.8%)**, suggesting CONC=32 4/4 may actually be reachable before CONC=4 4/4 as further levers stack.*

**Headline lever**: port of TRT-LLM's `use_relaxed_acceptance_for_thinking: true` (with `relaxed_topk=10`, `relaxed_delta=0.6` — TRT-LLM's published values) to ATOM/AITER for DeepSeek-R1 on MI355X. Per-phase Triton sampler tracks each sequence's `<think>...</think>` reasoning phase on a GPU-resident `int8[max_num_seqs]` tensor and applies relaxed acceptance **only inside thinking**, never stricter than the baseline elsewhere.

---

## Repository layout (this submission)

The repo is organized in three tiers, **TPOT-reduction first** (the headline win), then **supporting engineering** (kernel deliverables that didn't make it into the production hot path but are real engineering work), then **investigation log** (DEAD-lever reference for AMD so they don't re-test).

```
AMD_DSR_CNCC4/
│
├── PR_DESCRIPTION.md                          ← AMD submission item 2 (PR description)
├── TECHNICAL_APPROACH.md                      ← AMD submission item 5 (technical approach)
├── README.md                                  ← this file
│
├── ─────────── TIER 1: THE TPOT-REDUCING WIN ───────────
│
├── ATOM_main/                                 ← ATOM source with Phase 11 v3 patches applied
│   └── atom/
│       ├── model_engine/model_runner.py       (spec_phase tensor allocation + reset)
│       ├── model_ops/rejection_sampler.py     (rejection_phased_sample_kernel + dispatcher)
│       ├── utils/block_convert.py             (Phase 1 keystone: cudagraph-safe Triton grid)
│       └── utils/envs.py                      (ATOM_ENABLE_PER_PHASE_RELAXED_MTP env flag)
├── patches/scripts/
│   ├── phase11_per_phase_mtp/                 (v1/v2/v3 patch iteration trail — THE WIN)
│   │   ├── v3_top8_outside_thinking_fix.py    ← the KEEP / 2/4-gates patch
│   │   ├── v1_initial.py / v2_triton_type_fix.py
│   │   └── v3_1_per_pos_DEAD.py / v3_1b_top_n_12_DEAD.py (variants ruled out)
│   └── phase1_l0_cudagraph_mode_patch.py      (L0-v2 FULL_DECODE_ONLY env-flag patch)
├── bench_results/
│   ├── apr30_phase11_per_phase_mtp_v3_KEEP_2of4/  ⭐ HEADLINE EVIDENCE PACK
│   ├── apr29_l0v2_full_decode_only_WIN/           (L0-v2 supporting lever)
│   ├── apr26/                                     (multi-CONC reference data)
│   └── may01_phase0_clean_baseline/               (Phase 0 anchor)
│
├── ─────────── TIER 2: KERNEL ENGINEERING (supporting) ───────────
│
├── kernel_work/                               ← Built kernel deliverables NOT in production hot path
│   ├── README.md                              (summary table — status of each)
│   ├── phase2_fp4t_fused_ar_BUILT_DORMANT/         (aiter fused AR+RMSNorm+FP4 quant kernel — built, dormant)
│   ├── r2_small_m_moe_BUILT_NEVER_INTEGRATED/      (CDNA4 hand-authored FP4 MoE GEMM2, BIT-EXACT, 23× microbench)
│   └── l3_triton_fp4_kv_decode_3X_STANDALONE_INTEGRATION_REGRESS/  (Triton FP4 KV kernel, 3.04× faster standalone)
│
├── ─────────── TIER 3: INVESTIGATION LOG (for AMD reference) ───────────
│
├── investigation/                             ← Compact reference of DEAD levers (so AMD doesn't re-test)
│   ├── README.md                              (≈30-row table of what was tried and ruled out)
│   └── dead_levers_for_reference/             (preserved patch scripts for representative DEADs)
│
├── ─────────── Documentation + repro infra ───────────
│
├── docs/Daily Updates/
│   ├── MASTER.md                              (full engineering log + bench history)
│   ├── REPRODUCE.md                           (canonical reproduction recipe)
│   ├── OFFICIAL_HARNESS.md                    (kimbochen harness contract)
│   ├── Plan.md                                (master execution plan)
│   └── SERVER.md                              (launch-script variants)
├── scripts/                                   (boot scripts for TP=4/TP=8, multi-CONC)
├── agents/                                    (paired-bench harness + N=3 official wrappers)
└── aiter_configs/                             (hand-tuned FlyDSL + hipBLASLt CSVs needed for repro)
```

**Priority ordering**: AMD reviewers should focus on **Tier 1** for the verified 2/4-gates result. **Tier 2** is supporting engineering — kernels that built cleanly but didn't enter production for reasons documented in each subdir's README. **Tier 3** is "what was tried and ruled out" — kept compact so reviewers can verify scope without re-running experiments.

---

## Quick reproduction

Full step-by-step recipe in [`docs/Daily Updates/REPRODUCE.md`](docs/Daily%20Updates/REPRODUCE.md).

```bash
# 1) Pull canonical snapshot
docker pull rocm/atom-dev:dsr1_apr30_phase11_v3_2of4
docker run -d --name dsr1_repro \
  --ipc=host --shm-size=32g --network=host \
  --device=/dev/kfd --device=/dev/dri \
  -v /docker/huggingface/:/tmp/.cache/huggingface \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  rocm/atom-dev:dsr1_apr30_phase11_v3_2of4 sleep infinity

# 2) Reset perf-determinism (required for SCLK 2396 MHz boost)
docker exec dsr1_repro rocm-smi --resetperfdeterminism -d 0 -d 1 -d 2 -d 3

# 3) Boot with the lever enabled (ATOM_ENABLE_PER_PHASE_RELAXED_MTP=1)
docker exec -d dsr1_repro bash /tmp/boot_phase11_per_phase_mtp.sh
# Wait ~13 min for cudagraph capture; tail /tmp/*.log for "Application startup complete"

# 4) Warmup with 8 small curls (CRITICAL — hits decode cudagraph batches [1,2,4,8])
docker exec dsr1_repro bash -c '
for i in 1 2 3 4 5 6 7 8; do
  curl -s http://0.0.0.0:8890/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"amd/DeepSeek-R1-0528-MXFP4\",\"prompt\":\"Hello world '"$i"'\",\"max_tokens\":50,\"temperature\":0}" > /dev/null
done && echo warmup_done'

# 5) Official kimbochen 4-iter bench; take median(iter2, iter3, iter4)
docker exec dsr1_repro bash /tmp/dsr1_benchmark_4iter.sh

# 6) GSM8K N=3 (separate eval; median >= 0.93)
docker exec dsr1_repro bash /tmp/run_gsm8k_n3.sh
```

**Expected**: TPOT_med 5.641 ms ± 0.05, Tput/GPU 1449 ± 20, GSM8K_med 0.9318 ± 0.005, Interactivity 177.26, **2/4 gates**.

---

## What the lever does (mechanism)

DSR1-R1 emits explicit `<think>...</think>` reasoning blocks (token IDs `128798` open / `128799` close). These two phases have **different logit-distribution shapes**:

| Phase | Logit shape | Optimal acceptance |
|---|---|---|
| Inside `<think>` (reasoning) | wider, low-margin | top-N=10, δ=0.6 (TRT-LLM published) |
| Outside thinking (final answer) | sharper, correctness-bound | top-N=8, δ=0.6 (= baseline) |

ATOM's stock `RELAXED_TOP_N=8` is applied **globally**. Phase 11 v3 tracks each sequence's phase on a GPU `int8[max_num_seqs]` tensor and dispatches the wider top-10 acceptance **only inside thinking**, leaving answer-phase acceptance at the proven baseline. Net effect: never stricter than baseline anywhere → no GSM8K regression on the answer phase, but more accepted draft tokens during reasoning → fewer total decode forwards → lower TPOT.

The kernel is **cudagraph-safe**: the phase tensor lives at fixed GPU storage; the Triton kernel uses only `tl.load`/`tl.store`; no Python `setattr` in the forward path. Captured cleanly under `ATOM_CUDAGRAPH_MODE=FULL_DECODE_ONLY` for batch sizes `[1, 2, 4, 8, 16, 32]`.

### Why v3 succeeded where v1 and v2 didn't

| Variant | Outside-thinking top-N | Result | Issue |
|---|---:|---|---|
| v1 | int8/int32 type mismatch | Triton crash | Fixed by explicit cast in kernel |
| v2 | top-N=1 (strict greedy) | +0.86 ms regress | Stricter than baseline → accept-rate fell |
| **v3** | **top-N=8 (= baseline)** | **−0.661 ms WIN** | Matches baseline outside thinking, only relaxes inside |

---

## How to enable / disable

```bash
# Default (this PR is dormant — bit-identical to stock ATOM):
unset ATOM_ENABLE_PER_PHASE_RELAXED_MTP

# Activated (the 2/4-gates configuration):
export ATOM_ENABLE_PER_PHASE_RELAXED_MTP=1
```

Also requires `ATOM_CUDAGRAPH_MODE=FULL_DECODE_ONLY` (L0-v2 lever) and stock `ATOM_ENABLE_RELAXED_MTP=1` (both pre-existing flags). Full env-flag stack documented in [`docs/Daily Updates/SERVER.md`](docs/Daily%20Updates/SERVER.md).

---

## Files changed vs upstream

All changes are env-gated; default behavior (env unset) is bit-identical to upstream.

| File | Change |
|---|---|
| `ATOM_main/atom/utils/envs.py` | Add `ATOM_ENABLE_PER_PHASE_RELAXED_MTP` (default `0`) |
| `ATOM_main/atom/model_engine/model_runner.py` | Allocate `self.spec_phase = torch.zeros(max_num_seqs, int8, cuda)`; register on rejection_sampler module; reset prefill-slot phases on new request (Python-side, outside captured graph) |
| `ATOM_main/atom/model_ops/rejection_sampler.py` | Module-level `_spec_phase_tensor` + `set_spec_phase_tensor()` setter. New `rejection_phased_sample_kernel` Triton kernel: dual top-N branches (8/10), per-sequence phase lookup, commit-token scan for `<think>`/`</think>` IDs. Env-gated dispatch. |
| `ATOM_main/atom/utils/block_convert.py:142-215` | Cudagraph-safe Triton grid: `cdiv(n_cols, blocks_per_tile)` instead of `cdiv(max_num_blocks, blocks_per_tile)`. Phase 1 keystone (neutral perf, unblocks downstream). |

**Total LOC delta**: ~210 lines added, ~30 modified. No upstream files renamed or removed.

---

## Cumulative TPOT progress (chronological)

```
Start (vanilla TP=8 MTP=3 fp8 KV):                  ~7.88 ms  (1/4 gates)
       |
+ TP=8 -> TP=4 SR config                             7.88 ms
       |
+ ATOM_ENABLE_RELAXED_MTP=1 (stock flag, was OFF)    5.59 ms  <- -2.29 ms
       |
+ VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4           5.59 ms  (Tput +6.8%)
       |
+ RCCL_MSCCLPP knobs                                 5.93 ms  <- -0.18 ms
       |
+ rocm-smi --resetperfdeterminism (SCLK boost)       5.85 ms
       |
+ --cudagraph-capture-sizes [1,2,4,8,16,32]          4.84 ms  <- -1.41 ms (warm, single biggest unlock)
       |
+ ATOM_CUDAGRAPH_MODE=FULL_DECODE_ONLY               5.94 ms  <- (on later official harness baseline)
       |
+ Phase 11 v3 (TRT-LLM thinking port)                5.641 ms <- -0.661 ms ⭐ this PR's win
       |
2/4 gates achieved (GSM + Interactivity)
```

**Net official-harness TPOT progress: 7.88 → 5.641 ms (−2.24 ms, −28%) over 5 weeks.**

---

## Documentation map

| Document | Purpose |
|---|---|
| [`PR_DESCRIPTION.md`](PR_DESCRIPTION.md) | AMD submission **item 2** — pull-request description with file-by-file changes |
| [`TECHNICAL_APPROACH.md`](TECHNICAL_APPROACH.md) | AMD submission **item 5** — full technical approach: profiling methodology, bottleneck attribution, decision tree, lever inventory, architectural ceiling |
| [`docs/Daily Updates/MASTER.md`](docs/Daily%20Updates/MASTER.md) | Full engineering log (Apr 10 → May 04) with bench history and findings |
| [`docs/Daily Updates/REPRODUCE.md`](docs/Daily%20Updates/REPRODUCE.md) | Canonical step-by-step reproduction recipe |
| [`docs/Daily Updates/OFFICIAL_HARNESS.md`](docs/Daily%20Updates/OFFICIAL_HARNESS.md) | Kimbochen harness contract and measurement discipline |
| [`docs/Daily Updates/Plan.md`](docs/Daily%20Updates/Plan.md) | Master execution plan |
| [`docs/Daily Updates/SERVER.md`](docs/Daily%20Updates/SERVER.md) | Launch-script variants |
| [`bench_results/apr30_phase11_per_phase_mtp_v3_KEEP_2of4/README.md`](bench_results/apr30_phase11_per_phase_mtp_v3_KEEP_2of4/) | Headline evidence pack — kimbochen JSON + GSM8K log + boot log |

---

## Stack

- **Base image**: `rocm/atom-dev:dsr1_session17_re1_best_3of4_apr23_0627` (sha `2286b9de5107`)
- **ROCm**: 7.2.2 / HIP runtime + LLVM 21
- **PyTorch**: 2.10.0+rocm7.2.2.git40d237bf
- **aiter**: HEAD on this branch (with Phase 2 fp4_t fused AR+RMSNorm+quant kernel built — see [`PR_DESCRIPTION.md`](PR_DESCRIPTION.md) §"What's also in this branch")
- **ATOM**: HEAD on this branch (with Phase 11 v3 patches)
- **flydsl**: 0.1.3.1
- **triton**: 3.5.1
- **Model**: `amd/DeepSeek-R1-0528-MXFP4` (HF cached at `/tmp/.cache/huggingface/hub`)
- **Hardware**: 4× AMD Instinct MI355X (gfx950, CDNA4) per TP=4 run

---

## Acknowledgments

- **TRT-LLM team** for publishing `use_relaxed_acceptance_for_thinking` + values `relaxed_topk=10` / `relaxed_delta=0.6` — the design referenced by Phase 11 v3.
- **AMD AITER team** for the FlyDSL FP4 MoE GEMM fast-path, the persistent MLA kernel infrastructure, and the QuickReduce INT4 codec.
- **AMD ATOM team** for the `RELAXED_MTP` infrastructure and the `FULL_DECODE_ONLY` cudagraph mode.

---

## Contact

- Author: Danish Lynx (`danishlynx@gmail.com`)
- Repository: `Danishlynx/AMD_DSR_CNCC4`
- Submission target: `ai_dev_contests@amd.com`

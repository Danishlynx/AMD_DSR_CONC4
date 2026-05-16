# Technical Approach -- DSR1 MXFP4 on MI355X CONC=4

**Goal**: close 4 official gates at CONC=4 (TPOT <= 4.601 ms, E2E <= 5000 ms, Tput >= 1660 tok/s/GPU, GSM8K >= 0.93) on `amd/DeepSeek-R1-0528-MXFP4` running on 4x MI355X with TP=4 single-replica MTP=3.

**Constraint**: no weight modifications.

**Outcome**: **2/4 gates** (GSM + Interactivity) at TPOT 5.641 ms / Tput 1449 / GSM 0.9318 / Intvty 177.26 under the official kimbochen N=3 harness. The TPOT-reducing lever delivered in this submission is **Phase 11 v3 -- per-phase relaxed MTP (TRT-LLM thinking port)**, contributing **-0.661 ms TPOT** measured under the canonical harness.

---

## 1. Measurement discipline (read this first)

Two different measurement regimes were used during the campaign. The per-lever TPOT delta numbers below are explicitly tagged with which regime produced them -- they are **NOT** directly comparable across regimes.

| Era | Bench | Parameters | Authoritative? |
|---|---|---|:---:|
| Apr 14-26 | Informal dev-bench | `--random-range-ratio=0.8`, 1 warmup, varying num_prompts | (WARN) **NO** |
| **Apr 27 onward** | **Official `kimbochen/dsr1_benchmark.cpp`** | `--random-range-ratio=1.0`, num_warmups=8, num_prompts=40, chat-template, 4-iter median(2,3,4) x N=3 boots | PASS **YES** |

The Apr 26 informal bench measured "TPOT 4.84 ms / 3/4 gates" on the locked stack. The same stack re-measured Apr 27 under the official kimbochen N=3 harness measured **TPOT 6.171 ms / 1/4 gates** ("A27 baseline"). The informal numbers were partly mirage from the looser bench parameters (different `--random-range-ratio`, fewer warmups).

**Going forward in this document, all authoritative TPOT delta claims are from the official kimbochen N=3 harness, measured Apr 27 onward.**

Three measurement traps were also discovered during the campaign and corrected:

| Trap | Discovery | Rule |
|---|---|---|
| Single-shot GSM8K | A26 (RELAXED_TOP_N=9) measured 0.9378 single-shot but N=3 median was 0.9287 (FAIL) -- the 0.9378 was an outlier | **N=3 GSM median >= 0.93 -- never single-shot.** |
| Informal vs official harness | "3/4 gates" on Apr 26 informal bench (`--random-range-ratio=0.8`, 1 warmup) became "1/4 gates" under official kimbochen (chat-template, 8 warmups, 40 prompts) | **Only `kimbochen/dsr1_benchmark.cpp` produces authoritative numbers.** |
| Microbench != integration | Microbench-faster kernels often regressed under live production workload (per-call Python dispatch overhead exceeded compute savings) | **No lever is "done" until the official harness median says so.** |

---

## 2. Profiling -- bottleneck identification

A torch.profiler chrome trace captured 20 decode steps at the production CONC=4 shape (4 reqs x 4 tokens/step = bs=16 per verifier forward, MTP_K=3). Key measurements:

| Metric | Value |
|---|---:|
| Capture window | 274.2 ms (20 decode steps) |
| GPU active total | 18.3 ms (sum of 2,100 GPU kernel events) |
| **GPU busy fraction** | **6.7%** (the GPU is 93% idle) |
| `hipEventSynchronize` events | 60 x ~891 us = **53.4 ms** (25.5% of decode wall) |
| `hipGraphLaunch` events | 19 x ~493 us = 9.4 ms |

**Decisive implication**: at CONC=4 / TP=4 / MTP=3, the bottleneck is **per-step CPU orchestration + synchronization barriers**, NOT GPU compute. Even an infinitely-fast GPU would cap TPOT improvement at ~-0.9 ms -- still missing the 4.601 gate. Levers must attack synchronization or decode-step count, not raw kernel speed.

**Per-decode-step decomposition @ TPOT 6.02 ms** (May 13 measurement on the locked stack):

| Component | Time | % of TPOT |
|---|---:|---:|
| Real GPU kernel execution | ~915 us | 15% |
| `hipEventSynchronize` (3x per step) | ~2,670 us | 44% |
| Python orchestration + scheduling | ~2,440 us | 41% |

This is why **a sampler-level lever (Phase 11 v3) that reduces decode-step count** is more effective than chasing kernel-speed improvements at this concurrency.

---

## 3. The lever -- why TRT-LLM's thinking-port works

DSR1-R1 emits explicit `<think>...</think>` reasoning blocks. Speculative decoding's acceptance rate depends on the **shape of the logit distribution at each position** -- and the reasoning phase has fundamentally different logit shape from the answer phase:

| Phase | Logit shape | Best speculative acceptance |
|---|---|---|
| Inside `<think>...</think>` | wider, low-margin (multiple plausible continuations) | top-N=**10**, delta=0.6 (TRT-LLM published values) |
| Outside thinking (final answer) | sharper, correctness-bound | top-N=**8**, delta=0.6 *(= ATOM stock `RELAXED_TOP_N`)* |

The stock ATOM `RELAXED_TOP_N=8` is applied **globally**. By tracking each sequence's phase on a GPU-resident `int8[max_num_seqs]` tensor and dispatching the wider top-10 acceptance **only inside thinking** -- leaving answer-phase acceptance at the proven baseline -- the lever is **never stricter than baseline anywhere**. Net effect: more accepted draft tokens during reasoning -> fewer total decode forwards -> **lower TPOT, no GSM regression on the answer phase**.

This is the same design TRT-LLM ships in their LLM API as `use_relaxed_acceptance_for_thinking: true` + `relaxed_topk: 10` + `relaxed_delta: 0.6`. This PR ports the design to ATOM/AITER's existing Triton rejection_sampler infrastructure.

### Why v3 succeeded where v1 and v2 didn't

| Variant | Outside-thinking top-N | Result (official kimbochen N=3) | Cause |
|---|---:|---|---|
| v1 | (int8/int32 type mismatch in Triton kernel arg) | Triton crash | Fixed in v2 by explicit cast |
| v2 | top-N=1 (strict greedy) | **+0.86 ms TPOT regress** | Stricter than baseline -> accept rate dropped -> more forwards |
| **v3** | **top-N=8 (= baseline)** | **-0.661 ms TPOT WIN** | Matches baseline outside thinking, relaxes only inside |

The v2 -> v3 fix is the key insight: **the lever should never be stricter than the existing baseline**. Wider relaxation is only safe inside the verbose, low-margin reasoning phase. Outside thinking the model is in a correctness-bound regime where top-8 is the proven safe choice.

---

## 4. Cudagraph safety

The lever is captured cleanly under `ATOM_CUDAGRAPH_MODE=FULL_DECODE_ONLY` for batch sizes `[1, 2, 4, 8, 16, 32]` in 1.5 s after weights loaded. **No HSA exceptions, no assertion errors, no torch.compile recompiles.**

Three properties ensure capture-safety:

1. **Fixed-storage GPU phase tensor**: `self.spec_phase = torch.zeros(max_num_seqs, int8, cuda)` is allocated once at runner init and never reallocated. The captured graph references a stable GPU address.
2. **Triton-only access to the phase tensor**: the kernel uses `tl.load` / `tl.store` exclusively. No `setattr`, no host-side D2H copies, no Python branching inside the forward path.
3. **Phase reset happens outside the captured graph**: when a new request begins (prefill), the Python-side `self.spec_phase[:num_prefill_seqs].zero_()` runs **before** the captured decode graph fires. Decode-only batches leave the phase untouched.

---

## 5. Files changed (ATOM upstream)

All changes are env-gated; default behavior (env unset) is **bit-identical** to upstream.

| File | Change |
|---|---|
| `atom/utils/envs.py` | Adds `ATOM_ENABLE_PER_PHASE_RELAXED_MTP` (default `0`) -- registers the env flag |
| `atom/model_engine/model_runner.py` | Allocates `self.spec_phase` int8 tensor at init; registers via setter on `rejection_sampler` module (module-level setter, not Python `setattr` in forward path); resets prefill-slot phases on new request (Python op outside the captured cudagraph) |
| `atom/model_ops/rejection_sampler.py` | Module-level `_spec_phase_tensor` + `set_spec_phase_tensor()` setter. New `rejection_phased_sample_kernel` Triton kernel: dual top-N branches (strict=8 / relaxed=10) selected per-sequence by phase, `tl.load`/`tl.store` phase scan, commit-token scan for `<think>`/`</think>` IDs to advance phase. Env-gated dispatch from `rejection_sample()` |

**Total**: ~210 LOC added, ~30 modified, 3 files. **0 lines deleted from upstream.**

Patch scripts at [`submission/patches/scripts/phase11_per_phase_mtp/`](submission/patches/scripts/phase11_per_phase_mtp/) implement these changes as idempotent string-replacement scripts (apply order: v1 -> v2 -> v3).

---

## 6. Authoritative TPOT reductions under the official kimbochen N=3 harness

Two levers were validated under the canonical harness; **both ship in this submission's feature branch**:

| # | Lever | Type | TPOT before | TPOT after | delta | Measured |
|---|---|---|---:|---:|---:|---|
| 1 | `ATOM_CUDAGRAPH_MODE=FULL_DECODE_ONLY` (L0-v2) | env flag | 6.106 ms | 5.940 ms | **-0.166 ms** | Apr 29 |
| 2 | **Phase 11 v3 -- TRT-LLM thinking port** *(this PR)* | **first-party Triton kernel** | 6.302 ms | **5.641 ms** | **-0.661 ms** | Apr 30 |

Baselines differ across days (6.106 vs 6.302) due to +/-0.25 ms cross-boot noise. Both deltas were validated against their own day's same-server baseline.

**Net authoritative reduction from A27 baseline (Apr 27 official harness, 6.171 ms) to final state (Apr 30, 5.641 ms): -0.530 ms.**

---

## 7. Informal-bench era levers (context, NOT authoritative)

Apr 14-26: these env flags / CLI configs were turned ON during the informal-bench era of the campaign and **ARE part of the production stack** (the boot script enables them all). However, their **per-lever TPOT delta numbers from that era are NOT authoritative under the canonical harness**:

| Lever | Type | Informal-bench delta |
|---|---|---:|
| `ATOM_ENABLE_RELAXED_MTP=1` *(stock ATOM, was OFF)* | env flag | (historical -2.29 ms) |
| `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4` | env flag | (minimal TPOT, +6.8% thr) |
| RCCL_MSCCLPP knobs (ROCm 7.1+) | env flags | (historical -0.18 ms) |
| `rocm-smi --resetperfdeterminism` (SCLK boost) | platform | (historical -0.08 ms) |
| `--cudagraph-capture-sizes [1,2,4,8,16,32]` | CLI flag | (historical -1.41 ms) |
| 8-curl warmup pattern | bench discipline | absorbed |
| `ATOM_MSCG_K` unset | config | (historical -0.05 ms) |

When the full informal-bench-era stack was re-measured Apr 27 under the official kimbochen N=3 harness, the result was **TPOT 6.171 ms (A27 baseline)**, not the 4.84 ms the informal bench had reported. **The historical TPOT delta figures above are informal-bench artifacts**, not authoritative under the canonical harness.

The honest framing: these levers contribute to the production stack (the boot script turns them all ON), but the only TPOT reductions independently validated under the canonical kimbochen N=3 harness in this campaign are the two listed in sect 6.

---

## 8. Measured result

**CONC=4 (Apr 30, official kimbochen N=3 harness, Phase 11 v3 enabled):**

| Metric | Baseline (L0-v2) | Phase 11 v3 | delta |
|---|---:|---:|---:|
| GSM8K (N=3 median) | 0.9386 | **0.9318 PASS** | -0.0068 |
| TPOT median | 6.302 ms | **5.641 ms** | **-0.661 ms** |
| Tput / GPU | 1387 | 1449 | +62 |
| Interactivity | 158.68 FAIL | **177.26 PASS** | **+18.58** |
| E2E median | 6723 ms | 6210 ms | -513 ms |
| **Gates** | **1/4** | **2/4** | **+1** |

**CONC=32 (Apr 27, A27 baseline -- pre-Phase-11-v3):** same stack minus Phase 11 v3 also scored **2/4 gates** (GSM 0.9431 PASS, Interactivity 56.17 PASS; E2E off 5.8%, Tput off 1.8%). Phase 11 v3 wasn't re-benched at CONC=32; improvement there is projected, not measured.

---

## 9. What this PR delivers

**Phase 11 v3 contributes -0.661 ms TPOT** measured under the official kimbochen N=3 harness -- the larger of the two authoritative reductions in this campaign (the other being L0-v2 at -0.166 ms). **That -0.661 ms crosses Interactivity from 158.68 (FAIL) -> 177.26 (PASS), the +1 gate this PR delivers**, taking the submission from 1/4 to 2/4 gates.

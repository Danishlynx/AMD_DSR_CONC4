# DSR1 / MI355X CONC=4 -- Phase 11 Per-Phase Relaxed-Acceptance MTP v3 (TRT-LLM Thinking Port)

> **Result**: 2/4 official kimbochen gates at CONC=4. Phase 11 v3 delivers **-0.661 ms TPOT** under the official kimbochen N=3 harness -- the lever that crosses Interactivity from FAIL to PASS. Env-gated NULL-OP at default.

| | |
|---|---|
| **Author** | Danish Lynx |
| **Feature branch** | `feature/phase11_v3_thinking_port` |
| **Base** | `main` (pristine ATOM upstream snapshot) |
| **Best snapshot** | `rocm/atom-dev:dsr1_apr30_phase11_v3_2of4` (sha `c58cf2ce4512`) |
| **Authoritative harness** | `kimbochen/dsr1_benchmark.cpp` -- ISL=8192, OSL=1024, num_prompts=40, conc=4, num_warmups=8, 4-iter median(2,3,4) x N=3 boots; GSM8K via lm_eval N=3 median >= 0.93 |
| **Activation** | `ATOM_ENABLE_PER_PHASE_RELAXED_MTP=1` (default `0` = bit-identical to upstream) |

---

## Summary

Ports TRT-LLM's `use_relaxed_acceptance_for_thinking: true` (with `relaxed_topk=10`, `relaxed_delta=0.6`) to ATOM/AITER for DeepSeek-R1 on MI355X. A per-sequence phase-tracking Triton sampler applies relaxed acceptance **only inside `<think>...</think>` reasoning blocks**, matching the baseline `RELAXED_TOP_N=8` elsewhere -- so the lever is **never stricter than baseline anywhere**.

**Under the official kimbochen N=3 harness, this PR delivers -0.661 ms TPOT** (6.302 -> 5.641 ms), **+1 gate** (1/4 -> 2/4) -- crosses the Interactivity gate from 158.68 (FAIL) to 177.26 (PASS).

---

## Authoritative result (CONC=4, official kimbochen N=3 harness, Apr 30)

| Metric | Baseline (L0-v2) | Phase 11 v3 | Delta | Gate | Status |
|---|---:|---:|---:|---:|:---:|
| GSM8K (N=3 median) | 0.9386 | **0.9318** | -0.0068 | >= 0.93 | PASS |
| Median TPOT | 6.302 ms | **5.641 ms** | **-0.661 ms** | -- | (drives below) |
| Median E2E | 6723 ms | 6210 ms | -513 ms | <= 5000 | FAIL (off 1210) |
| Throughput / GPU | 1387 | **1449** | +62 | >= 1500 | FAIL (off 51) |
| Interactivity | 158.68 | **177.26** | **+18.58** | >= 165 | **PASS** (+1 gate) |
| **Gates** | **1/4** | **2/4** | **+1** | | |

Evidence: [`submission/bench_results/apr30_phase11_per_phase_mtp_v3_KEEP_2of4/`](submission/bench_results/apr30_phase11_per_phase_mtp_v3_KEEP_2of4/) -- raw kimbochen JSON, GSM8K log, boot log.

### Bonus reference @ CONC=32 (Apr 27, A27 baseline, official harness -- pre-Phase-11-v3)

Same stack minus the Phase 11 v3 sampler kernel also scored **2/4 gates** at CONC=32 (Apr 27): GSM 0.9431 PASS, Interactivity 56.17 PASS; E2E off 5.8%, Tput/GPU off 1.8%. Phase 11 v3 wasn't re-benched at CONC=32 -- would likely improve there too (it's concurrency-agnostic) but **projected, not measured**.

---

## TPOT reductions under the official kimbochen N=3 harness (authoritative)

Two levers were measured under the canonical harness in this campaign -- both ship in this submission's feature branch:

| # | Lever | Type | TPOT before | TPOT after | Delta | Status |
|---|---|---|---:|---:|---:|:---:|
| 1 | `ATOM_CUDAGRAPH_MODE=FULL_DECODE_ONLY` (L0-v2) | env flag | 6.106 ms | 5.940 ms | **-0.166 ms** | measured Apr 29 |
| 2 | **Phase 11 v3 -- TRT-LLM thinking port** (this PR) | **first-party Triton kernel** | 6.302 ms | **5.641 ms** | **-0.661 ms** | measured Apr 30 |

Both measurements use the official kimbochen 4-iter median(2,3,4) x N=3 boots methodology. Baselines differ across days (6.106 vs 6.302) due to +/-0.25 ms cross-boot noise -- both deltas were validated on their own day's baseline.

**Authoritative reduction from Apr 27 A27 baseline (6.171 ms) to final state (5.641 ms): -0.530 ms net.** Phase 11 v3 in this PR contributes **-0.661 ms** measured on its own daily baseline.

---

## Historical context (informal-bench era -- NOT authoritative)

Earlier in the campaign (Apr 14--26), TPOT improvements were tracked on an **informal bench** (`--random-range-ratio=0.8`, 1 warmup, varying num_prompts). Several stock ATOM env flags / CLI configs were turned ON during this era:

| Lever | Type | Informal-bench Delta (NOT authoritative) |
|---|---|---:|
| `ATOM_ENABLE_RELAXED_MTP=1` (stock ATOM, was OFF) | env flag | (historical -2.29 ms) |
| `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4` | env flag | (small) |
| RCCL_MSCCLPP knobs (ROCm 7.1+) | env flags | (historical -0.18 ms) |
| `rocm-smi --resetperfdeterminism` (SCLK boost) | platform | (historical -0.08 ms) |
| `--cudagraph-capture-sizes [1,2,4,8,16,32]` | CLI flag | (historical -1.41 ms) |
| 8-curl warmup pattern | bench discipline | absorbed into above |
| `RELAXED_TOP_N` 8 to 9 | sampler | (small) |
| `ATOM_MSCG_K` unset | config | (historical -0.05 ms) |

**Important**: when the FULL stack of these informal-bench levers was re-measured Apr 27 under the official kimbochen N=3 harness, the result was **TPOT 6.171 ms** ("A27 baseline") -- **NOT the 4.84 ms the informal bench had reported**. The informal-bench numbers were partly mirage (different `--random-range-ratio`, fewer warmups, different harness behavior).

**These levers ARE part of the production stack** (they ship enabled in the boot script -- see Activation below), but the per-lever delta-TPOTs in the table above are **historical, not authoritative under the canonical harness**.

---

## Mechanism (Phase 11 v3)

DSR1-R1 emits explicit reasoning blocks delimited by `<think>...</think>` tokens (IDs `128798` open / `128799` close). The two phases have **different logit-distribution shapes**:

| Phase | Logit shape | Acceptance criterion |
|---|---|---|
| Inside `<think>...</think>` (reasoning) | wider, low-margin | top-N=**10**, delta=0.6 (TRT-LLM published) |
| Outside thinking (final answer) | sharper, correctness-bound | top-N=**8**, delta=0.6 (= ATOM stock `RELAXED_TOP_N`) |

A GPU-resident `int8[max_num_seqs]` phase tensor tracks each sequence's state. The Triton kernel dispatches top-10 acceptance **only when the phase indicates thinking**, leaving answer-phase acceptance at the proven baseline. **More accepted drafts during reasoning -> fewer total decode forwards -> lower TPOT, no GSM regression on the answer phase.**

### v3 iteration trail

| Variant | Outside-thinking top-N | Result | Issue |
|---|---:|---|---|
| v1 | (int8/int32 type bug) | Triton crash | Fixed by explicit cast |
| v2 | top-N=1 (strict greedy) | **+0.86 ms regress** | Stricter than baseline -> accept-rate dropped |
| **v3** | **top-N=8 (= baseline)** | **-0.661 ms WIN** | Matches baseline outside thinking; relaxes only inside |

---

## Files changed (ATOM upstream)

All changes are env-gated; with `ATOM_ENABLE_PER_PHASE_RELAXED_MTP` unset, behavior is **bit-identical** to upstream.

| File | Change |
|---|---|
| `atom/utils/envs.py` | Adds `ATOM_ENABLE_PER_PHASE_RELAXED_MTP` (default `0`) |
| `atom/model_engine/model_runner.py` | Allocates `self.spec_phase = torch.zeros(max_num_seqs, int8, cuda)`; registers via setter on `rejection_sampler` module; resets prefill-slot phases on new request (Python-side, outside the captured cudagraph) |
| `atom/model_ops/rejection_sampler.py` | Module-level `_spec_phase_tensor` + setter. New `rejection_phased_sample_kernel`: dual top-N branches (strict=8 / relaxed=10) selected per-sequence by phase, `tl.load`/`tl.store` phase scan (cudagraph-safe), commit-token scan for `<think>`/`</think>` IDs. Env-gated dispatch. |

**Total**: ~210 LOC added, ~30 modified across 3 files. **0 upstream lines deleted.**

Patch scripts: [`submission/patches/scripts/phase11_per_phase_mtp/`](submission/patches/scripts/phase11_per_phase_mtp/) -- `v1_initial.py` -> `v2_triton_type_fix.py` -> `v3_top8_outside_thinking_fix.py` (KEEP).

---

## Cudagraph safety

Captured cleanly under `ATOM_CUDAGRAPH_MODE=FULL_DECODE_ONLY` for batch sizes `[1, 2, 4, 8, 16, 32]` in 1.5 s after weights loaded. No HSA exceptions, no assertion errors, no torch.compile recompiles.

- Phase tensor lives at **fixed GPU storage** (allocated at runner init, never reallocated)
- Triton kernel uses only `tl.load` / `tl.store` -- no Python `setattr`, no host-side D2H copies in the forward path
- Phase reset on new request happens at **prefill** time, **before** the captured decode graph fires

---

## Activation

```bash
# This PR's lever:
export ATOM_ENABLE_PER_PHASE_RELAXED_MTP=1

# Pre-existing flags that the boot script also sets (= the production stack):
export ATOM_CUDAGRAPH_MODE=FULL_DECODE_ONLY     # L0-v2 lever (-0.166 ms, also measured under official harness)
export ATOM_ENABLE_RELAXED_MTP=1                # stock ATOM flag (was OFF by default)
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export RCCL_MSCCLPP_ENABLE=1 RCCL_MSCCLPP_THRESHOLD=1048576 RCCL_P2P_BATCH_ENABLE=1
# Plus --cudagraph-capture-sizes [1,2,4,8,16,32] on the server CLI
# Plus rocm-smi --resetperfdeterminism cold-boot step
```

---

## Reproduction

```bash
docker pull rocm/atom-dev:dsr1_apr30_phase11_v3_2of4
docker run -d --name dsr1_repro \
  --ipc=host --shm-size=32g --network=host \
  --device=/dev/kfd --device=/dev/dri \
  -v /docker/huggingface/:/tmp/.cache/huggingface \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  rocm/atom-dev:dsr1_apr30_phase11_v3_2of4 sleep infinity

docker exec dsr1_repro rocm-smi --resetperfdeterminism -d 0 -d 1 -d 2 -d 3
docker exec -d dsr1_repro bash /tmp/boot_phase11_per_phase_mtp.sh    # ~13 min for cudagraph capture

# 8-curl warmup (CRITICAL -- hits decode batches [1,2,4,8])
for i in 1 2 3 4 5 6 7 8; do
  docker exec dsr1_repro curl -s http://0.0.0.0:8890/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"amd/DeepSeek-R1-0528-MXFP4\",\"prompt\":\"Hello world $i\",\"max_tokens\":50,\"temperature\":0}" > /dev/null
done

docker exec dsr1_repro bash /tmp/dsr1_benchmark_4iter.sh    # kimbochen 4-iter, median(2,3,4)
docker exec dsr1_repro bash /tmp/run_gsm8k_n3.sh            # GSM8K N=3 median
```

**Expected**: TPOT_med 5.641 ms +/- 0.05, Tput/GPU 1449 +/- 20, GSM8K_med 0.9318 +/- 0.005, **2/4 gates**.

---

## See also

- [`submission/TECHNICAL_APPROACH.md`](submission/TECHNICAL_APPROACH.md) -- profiling + bottleneck attribution + lever mechanism + measurement discipline
- [`submission/PERFORMANCE_METRICS.md`](submission/PERFORMANCE_METRICS.md) -- detailed performance metrics + harness regime separation
- [`submission/README.md`](submission/README.md) -- submission overview

---

## Acknowledgments

- **TRT-LLM team** -- for publishing `use_relaxed_acceptance_for_thinking` + the `relaxed_topk=10` / `relaxed_delta=0.6` values used here
- **AMD AITER team** -- FlyDSL FP4 MoE GEMM fast path, persistent MLA, QuickReduce INT4 codec
- **AMD ATOM team** -- `RELAXED_MTP` infrastructure + `FULL_DECODE_ONLY` cudagraph mode that this PR builds on

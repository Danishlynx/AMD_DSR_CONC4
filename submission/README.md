# AMD Phase 2 Hackathon — DSR1 Track Submission

**Author**: Danish Lynx (`danishlynx@gmail.com`)
**Result**: **2/4 gates** at CONC=4 via Phase 11 per-phase relaxed-MTP v3 (TRT-LLM thinking port)
**Snapshot**: `rocm/atom-dev:dsr1_apr30_phase11_v3_2of4` (sha `c58cf2ce4512`)
**Authoritative harness**: `kimbochen/dsr1_benchmark.cpp` — official kimbochen 4-iter median(2,3,4) × N=3 boots, GSM8K N=3 median

## Result (CONC=4, official kimbochen N=3 harness, Apr 30)

| Metric | Value | Gate | Status |
|---|---:|---:|:---:|
| GSM8K (N=3 median) | **0.9318** | ≥ 0.93 | ✅ PASS |
| Median TPOT | **5.641 ms** | — | (drives below) |
| Throughput / GPU | **1449** | ≥ 1500 | ❌ off 51 |
| Interactivity | **177.26** | ≥ 165 | ✅ PASS |
| Median E2E | 6210 ms | ≤ 5000 | ❌ off 1210 |
| **Gates** | | | **2/4** |

This PR's lever — **Phase 11 v3 (TRT-LLM thinking port)** — delivers **−0.661 ms TPOT** measured under the official kimbochen N=3 harness on Apr 30. That −0.661 ms crosses Interactivity from 158.68 (FAIL) → 177.26 (PASS), the +1 gate this PR contributes. Env-gated NULL-OP at default: `ATOM_ENABLE_PER_PHASE_RELAXED_MTP=1` to activate.

## Repository layout

```
submission/
├── README.md                                ← this file
├── PR_DESCRIPTION.md                        ← AMD item 2 (PR description)
├── TECHNICAL_APPROACH.md                    ← AMD item 5 (technical approach)
├── PERFORMANCE_METRICS.md                   ← AMD item 4 (perf metrics)
├── patches/scripts/
│   ├── phase11_per_phase_mtp/               ← THE TPOT-reducing kernel + dispatcher
│   │   ├── v1_initial.py / v2_triton_type_fix.py / v3_top8_outside_thinking_fix.py   ← v1/v2/v3 trail (v3 = KEEP)
│   │   ├── v3_1_per_pos_DEAD.py / v3_1b_top_n_12_DEAD.py                             ← variants ruled out
│   │   └── README.md
│   └── phase1_l0_cudagraph_mode_patch.py    ← supporting: L0-v2 FULL_DECODE_ONLY (−0.166 ms, also measured under official harness)
└── bench_results/
    └── apr30_phase11_per_phase_mtp_v3_KEEP_2of4/
        ├── README.md
        └── evidence.tgz                     ← raw kimbochen JSONs, GSM8K log, boot log
```

Repo root (`atom/`, `csrc/`, `docs/`, `tests/`, etc.) is **pristine ATOM upstream**, unchanged from the fork base on `main`.

## Mechanism

DSR1-R1 emits explicit `<think>...</think>` reasoning blocks (token IDs `128798` / `128799`). Per-sequence Triton sampler tracks each sequence's reasoning phase on a GPU-resident `int8[max_num_seqs]` tensor. **Inside** thinking: top-N=10, δ=0.6 (TRT-LLM published). **Outside** thinking: top-N=8, δ=0.6 (= ATOM stock `RELAXED_TOP_N`). Net: never stricter than baseline anywhere; relaxes more inside the reasoning phase only → more accepted draft tokens during reasoning → fewer total decode forwards → lower TPOT. Cudagraph-safe (fixed phase tensor, `tl.load`/`tl.store` only).

## TPOT reductions measured under the official kimbochen N=3 harness

Two levers were validated under the canonical harness; **both ship in this submission's feature branch**:

| Lever | TPOT before | TPOT after | Δ | Measured |
|---|---:|---:|---:|---|
| `ATOM_CUDAGRAPH_MODE=FULL_DECODE_ONLY` (L0-v2) | 6.106 ms | 5.940 ms | **−0.166 ms** | Apr 29 |
| **Phase 11 v3 — TRT-LLM thinking port** *(this PR)* ⭐ | 6.302 ms | **5.641 ms** | **−0.661 ms** | Apr 30 |

**A27 baseline (Apr 27, official harness): 6.171 ms** → final state 5.641 ms = **−0.530 ms net authoritative reduction**.

Other env flags / CLI configs in the production stack (e.g., `ATOM_ENABLE_RELAXED_MTP=1`, `--cudagraph-capture-sizes [1,2,4,8,16,32]`, RCCL_MSCCLPP knobs) were turned ON during the Apr 14–26 informal-bench era of the campaign; **their per-lever ΔTPOT numbers from that era are historical, not authoritative under the canonical harness** — see [`PERFORMANCE_METRICS.md`](PERFORMANCE_METRICS.md) for the regime-separated breakdown.

## Activation

```bash
export ATOM_ENABLE_PER_PHASE_RELAXED_MTP=1   # this lever (-0.661 ms, authoritative)
export ATOM_CUDAGRAPH_MODE=FULL_DECODE_ONLY  # L0-v2 (-0.166 ms, authoritative)
export ATOM_ENABLE_RELAXED_MTP=1             # stock ATOM flag (was OFF by default)
```

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

# 8-curl warmup (CRITICAL)
for i in 1 2 3 4 5 6 7 8; do
  docker exec dsr1_repro curl -s http://0.0.0.0:8890/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"amd/DeepSeek-R1-0528-MXFP4\",\"prompt\":\"Hello world $i\",\"max_tokens\":50,\"temperature\":0}" > /dev/null
done

docker exec dsr1_repro bash /tmp/dsr1_benchmark_4iter.sh    # kimbochen 4-iter, median(2,3,4)
docker exec dsr1_repro bash /tmp/run_gsm8k_n3.sh            # GSM8K N=3 median
```

Expected: TPOT_med 5.641 ms ± 0.05, Tput/GPU 1449 ± 20, GSM8K_med 0.9318 ± 0.005, **2/4 gates**.

## Files changed vs upstream ATOM (this PR's contribution)

| File | Change |
|---|---|
| `atom/utils/envs.py` | Add `ATOM_ENABLE_PER_PHASE_RELAXED_MTP` env flag (+2 lines) |
| `atom/model_engine/model_runner.py` | Allocate `self.spec_phase` int8 tensor + setter + prefill reset (+18 lines) |
| `atom/model_ops/rejection_sampler.py` | `rejection_phased_sample_kernel` Triton kernel + env-gated dispatcher (+~190 lines) |

~210 LOC added, 0 removed, 3 files. Default OFF → bit-identical to upstream.

## Contact

- Author: Danish Lynx (`danishlynx@gmail.com`)
- Repository: `Danishlynx/AMD_DSR_CONC4`
- Submission target: `ai_dev_contests@amd.com`

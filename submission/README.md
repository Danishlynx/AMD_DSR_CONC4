# AMD Phase 2 Hackathon — DSR1 Track Submission

**Author**: Danish Lynx (`danishlynx@gmail.com`)
**Result**: **2/4 gates** at CONC=4 via Phase 11 per-phase relaxed-MTP v3 (TRT-LLM thinking port)
**Snapshot**: `rocm/atom-dev:dsr1_apr30_phase11_v3_2of4` (sha `c58cf2ce4512`)

## TL;DR

| Metric | Value | Gate | Status |
|---|---:|---:|:---:|
| GSM8K (N=3 median) | **0.9318** | ≥ 0.93 | ✅ PASS |
| Median TPOT | **5.641 ms** | — | (drives below) |
| Throughput / GPU | **1449** | ≥ 1500 | ❌ off 51 |
| Interactivity | **177.26** | ≥ 165 | ✅ PASS |
| Median E2E | 6210 ms | ≤ 5000 | ❌ off 1210 |
| **Gates** | | | **2/4** |

The headline lever — **Phase 11 v3 (TRT-LLM thinking port)** — delivers **−0.661 ms TPOT** (6.302 → 5.641 ms), crossing Interactivity from FAIL → PASS. Env-gated NULL-OP at default: `ATOM_ENABLE_PER_PHASE_RELAXED_MTP=1` to activate.

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
│   └── phase1_l0_cudagraph_mode_patch.py    ← supporting: L0-v2 FULL_DECODE_ONLY (−0.166 ms)
└── bench_results/
    └── apr30_phase11_per_phase_mtp_v3_KEEP_2of4/
        ├── README.md                        ← lever description + result table
        └── evidence.tgz                     ← raw kimbochen JSONs, GSM8K log, boot log
```

The rest of the repo (at repo root: `atom/`, `csrc/`, `docs/`, `tests/`, etc.) is **pristine ATOM upstream**, unchanged from the fork base on `main`.

## Mechanism (one paragraph)

DSR1-R1 emits explicit `<think>...</think>` reasoning blocks (token IDs `128798` / `128799`). Per-sequence Triton sampler tracks each sequence's reasoning phase on a GPU-resident `int8[max_num_seqs]` tensor. **Inside** thinking: top-N=10, δ=0.6 (TRT-LLM published). **Outside** thinking: top-N=8, δ=0.6 (= ATOM stock `RELAXED_TOP_N`). Net: never stricter than baseline anywhere, but relaxes more inside the reasoning phase → more accepted draft tokens during reasoning → fewer total decode forwards → lower TPOT. Cudagraph-safe (fixed phase tensor, `tl.load`/`tl.store` only).

## Activation

```bash
export ATOM_ENABLE_PER_PHASE_RELAXED_MTP=1   # this lever
export ATOM_CUDAGRAPH_MODE=FULL_DECODE_ONLY  # required (L0-v2)
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

# 8-curl warmup (CRITICAL — hits decode batches [1,2,4,8])
for i in 1 2 3 4 5 6 7 8; do
  docker exec dsr1_repro curl -s http://0.0.0.0:8890/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"amd/DeepSeek-R1-0528-MXFP4\",\"prompt\":\"Hello world $i\",\"max_tokens\":50,\"temperature\":0}" > /dev/null
done

docker exec dsr1_repro bash /tmp/dsr1_benchmark_4iter.sh    # kimbochen 4-iter, median(2,3,4)
docker exec dsr1_repro bash /tmp/run_gsm8k_n3.sh            # GSM8K N=3 median
```

Expected: TPOT_med 5.641 ms ± 0.05, Tput/GPU 1449 ± 20, GSM8K_med 0.9318 ± 0.005, **2/4 gates**.

## Files changed vs upstream ATOM

| File | Change |
|---|---|
| `atom/utils/envs.py` | Add `ATOM_ENABLE_PER_PHASE_RELAXED_MTP` env flag (+2 lines) |
| `atom/model_engine/model_runner.py` | Allocate `self.spec_phase` int8 tensor + setter + prefill reset (+18 lines) |
| `atom/model_ops/rejection_sampler.py` | `rejection_phased_sample_kernel` Triton kernel + env-gated dispatcher (+~190 lines) |

Total: ~210 LOC added, ~30 modified, 3 files. **0 lines removed.** Default OFF → bit-identical to upstream.

## Contact

- Author: Danish Lynx (`danishlynx@gmail.com`)
- Repository: `Danishlynx/AMD_DSR_CONC4`
- Submission target: `ai_dev_contests@amd.com`

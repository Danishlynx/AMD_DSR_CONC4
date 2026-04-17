# AMD Phase 2 Hackathon — DSR1 Track — DEC-073 Snapshot

This repo is a **backup + reproducibility package** for Danish's DeepSeek-R1 (DSR1) track submission.

## Result (DEC-073 floor)

Measured via the official harness `./dsr1_benchmark perf`:

| Metric | Value | Gate | Pass? |
|---|---|---|---|
| Thr/GPU (÷4) | **1257** | ≥ 1500 | ❌ −17% |
| Median TPOT | **6.77 ms** | — | — |
| Interactivity | **147.8** | ≥ 165 | ❌ −10% |
| Median E2E | **7390 ms** | ≤ 5000 | ❌ +48% |
| GSM8K | **0.9348** | ≥ 0.93 | ✅ |
| **Gates** | **1/4** | — | GSM8K only |

Total throughput: **5027 tok/s** at ISL=8192, OSL=1024, CONC=4 — about **+30% above AMD's public baseline** of 3871 tok/s at same shape.

## Stack

- **ATOM** commit `108a70e` + 3 local patches (in `ATOM_main/`)
- **aiter** commit `f8c1d76bd` + 97-row BF16 decode tune (`aiter_configs/`)
- **flydsl** 0.1.2
- **Model**: `amd/DeepSeek-R1-0528-MXFP4` (unmodified, as shipped)
- **Hardware**: 4× AMD Instinct MI355X, TP=4 single-replica
- **KV cache**: FP8
- **Speculative decoding**: native DeepSeek MTP (k=3), relaxed acceptance (top_n=8, delta=0.5)

## Changes made (mergeable against upstream)

### ATOM (3 files in [ATOM_main/atom/](ATOM_main/atom/))
| File | Change | Rationale |
|---|---|---|
| `model_ops/rejection_sampler.py` | Hardcode `RELAXED_TOP_N = 8`, `RELAXED_DELTA = 0.5` | Sweep showed this point maxes accept-rate while passing GSM8K gate |
| `model_ops/attention_mla.py` | `num_kv_splits=None` (auto-tune) | Manual value 16 was suboptimal for TP=4 SR shape |
| `spec_decode/eagle.py` | Phase 4A v4 drafter HIP graph scaffolding | Null perf on our workload; kept as harmless infrastructure for downstream tree-spec work |

### aiter ([aiter_configs/dsv3_bf16_tuned_gemm.csv](aiter_configs/dsv3_bf16_tuned_gemm.csv))
97 tuned decode-shape rows (M=1/4/16 LM head, M=16 MLA projections) tuned via gradlib with errRatio=0.05. Extends the upstream file.

## Reproduction

```bash
# Inside container danish_atom_main with ATOM patches + aiter CSV applied:
export HOME=/tmp AITER_BUILD_DIR=/tmp/.aiter_cache TRITON_CACHE_DIR=/tmp/.triton_cache
export HIP_FORCE_DEV_KERNARG=1 NCCL_MIN_NCHANNELS=16
export ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=256 ATOM_ENABLE_RELAXED_MTP=1
export HF_HOME=/projects/teamA/hf_cache HUGGINGFACE_HUB_CACHE=/projects/teamA/hf_cache/hub
export HIP_VISIBLE_DEVICES=0,1,2,3

cd /workspace/ATOM_main && \
python3 -m atom.entrypoints.openai_server \
  --model amd/DeepSeek-R1-0528-MXFP4 --server-port 8888 -tp 4 \
  --kv_cache_dtype fp8 --method mtp --num-speculative-tokens 3 \
  --max-model-len 10240 --gpu-memory-utilization 0.85

# In separate shell:
cd /workspace/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x
source specific_conc_var.sh
./dsr1_benchmark perf
```

See [docs/best_reproduce.md](docs/best_reproduce.md) for full details.

## Repository layout

```
.
├── README.md                   ← you are here
├── ATOM_main/                  ← full modified ATOM source tree with .git history + .bak files
├── aiter_configs/
│   └── dsv3_bf16_tuned_gemm.csv ← tuned 97-row CSV
├── bench_results/              ← raw test_*.json from the official harness (proof of numbers)
├── session_logs/               ← chronological engineering session logs from earlier sprints
├── docs/                       ← strategic and planning documents
│   ├── SERVER_MAP.md           ← infrastructure overview (server + containers + filesystem)
│   ├── BRIEF_FOR_KIMI_OPUS.md  ← separation-of-concerns for Kimi track
│   ├── Current_plan.md         ← active plan state
│   ├── MASTER_FINDINGS.md      ← canonical project findings
│   ├── daily_log.md            ← chronological DEC record
│   ├── Danish.md               ← strategic context
│   └── best_reproduce.md       ← full reproduction instructions for DEC-073
├── patches/                    ← (future) clean diffs vs upstream for PR submission
└── repro/                      ← (future) repro scripts
```

## What's in `ATOM_main/` specifically

- Modified source files (the 3 above)
- Full `.bak_*` history (each backup named after the attempt, e.g. `rejection_sampler.py.bak_before_tree_spec_0641`) — useful forensic record of what was tried
- `atom/spec_decode/tree_spec.py` — a 120-line EAGLE-2-style tree topology builder added by a prior session, **not currently wired** into eagle.py (candidate for future tree-speculation work)

## DEC lineage

| DEC | Lever | Result |
|---|---|---|
| DEC-056 | DUAL_STREAM=256 | floor 1209/6.89 |
| DEC-058 | +9-row BF16 CSV tune + NCCL=16 | 1202/7.19 |
| DEC-064 | Relaxed MTP (7, 0.4) | 1253/7.06 |
| DEC-066 | +new tuned CSV (9 rows total) | 1221/6.73/148.6 |
| DEC-069 | Phase 4A v4 drafter HIP graph | NULL (DEC-057 profile proved no Python gap) |
| DEC-071 | BF16 decode tune (88 new rows → 97 total) | 1267/6.96/143.8 |
| DEC-072 | BF16 prefill tune | DEAD — GSM8K 0.865 crash, reverted |
| **DEC-073** | **Relaxed MTP (8, 0.5)** | **1270/6.80/147.1/7318/0.934** ← current floor |
| DEC-074 | Naive tree spec (top-2 at last pos) | ABANDONED — kernel refactor regressed to 0.807 |

## Probes explored and ruled out (post-DEC-073)

- **EP + TP=4** (`--enable-expert-parallel`): GSM8K drops to 0.9287, fails gate
- **MTP=4** (`--num-speculative-tokens 4`): AITER asserts `qo_len ≤ 4` for FP8 MLA — hard kernel constraint
- **BF16 KV cache** (`--kv_cache_dtype bf16`): −4% throughput, +6% TPOT — regresses
- **Weight modification** (transplant or hand-requant): ruled out by competitor's policy (stay on stock model)

## Open candidate — real tree speculation

- `atom/spec_decode/tree_spec.py` (prior session artifact) plus `aiter/op_tests/triton_tests/utils/mla_extend_ref.py` (SGLang-origin 460-line MLA Triton kernel with full `custom_mask` support) provide the infrastructure for a real tree-spec implementation with BF=2 branching at depth 0 (topology: 1 root + 2 branches × 3 depth = 7 leaves, `qo_len=7`)
- Not yet implemented in our stack; feasibility analysis ongoing
- See [docs/Current_plan.md](docs/Current_plan.md) for status

## Legal / usage

Private snapshot, backup only. Do not redistribute without Danish's permission. AMD ATOM code is licensed per ROCm/atom repository's original license.

---

Generated 2026-04-18 as a disaster-recovery snapshot before potential AMD IT cleanup of the hackathon server.

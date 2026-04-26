# AMD Phase 2 Hackathon — DSR1 Track

This repo is a **backup + reproducibility package** for Danish's DeepSeek-R1 (DSR1) MXFP4 / MI355X track submission.

**Current canonical entry point**: [`docs/REPRODUCE.md`](docs/REPRODUCE.md) — read Section 0 (STACK GENEALOGY) and Section 1-12 (FINAL DELIVERY MANIFEST FOR AMD REVIEW).

---

## 🚀 Apr 26 — current best (3/4 gates, locked stack)

Snapshot: `rocm/atom-dev:dsr1_apr26_3of4_validated_warm_bench` (sha **8e844757ad6c**, 485 GB) ⭐
Tag: [`dsr1-A26-3of4-apr26`](https://github.com/Danishlynx/AMD_DSR_CNCC4/releases/tag/dsr1-A26-3of4-apr26)
Branch: [`session17_fp8_sq8_mla`](https://github.com/Danishlynx/AMD_DSR_CNCC4/tree/session17_fp8_sq8_mla)

| Metric | Value | Gate | Pass? | Δ vs Apr 23 record |
|---|---|---|---|---|
| **GSM8K (flexible-extract)** | **0.9522** | ≥ 0.93 | ✅ | **+1.2% beats 0.9409** |
| **Throughput/GPU** | **1650** tok/s | ≥ 1500 | ✅ | **+18.6% beats 1391** |
| **Median TPOT** | **4.84 ms** | ≤ 6.06 | ✅ | **−18.4% beats 5.93** |
| **Interactivity** | **207 tok/s/user** | ≥ 165 | ✅ | **+22.5% beats 168.77** |
| **Median E2E (calc)** | **5240 ms** | ≤ 5000 | ❌ −4.8% | (was 6632, gap closed from −32.6% to −4.8%) |
| **Gates** | **3/4** | — | ✅ on 4 metrics with significant margin | gap closed dramatically |

**3 / 4 gates verified**, all 4 passing gates carry significant margin. Single lever (Phase 2 AR+RMSNorm+MXFP4 fusion, kernel + dispatch + plumbing already built and committed in this branch) is estimated to close the remaining E2E gap of 240 ms.

Setup: 4× MI355X (gfx950), TP=4, MTP=3, FP8 KV cache, INT4 QuickReduce AR, RCCL_MSCCLPP, RELAXED_TOP_N=9 / RELAXED_DELTA=0.5, dual-stream MoE threshold=1024, TBO prefill, ISL=8192, OSL=1024, num_prompts=40, max-concurrency=4. ROCm 7.2.2, PyTorch 2.10.0+rocm7.2.2, aiter HEAD on `session17_fp8_sq8_mla` with Apr 26 EOD patches.

## 🧬 How we got here — stack lineage (vanilla → today)

Full chronological table with per-row source citations: see [`docs/REPRODUCE.md`](docs/REPRODUCE.md) Section 0.

```
[Apr 10-13 vanilla TP=8 MTP=3 fp8kv]      thr/GPU 738.93,  TPOT 6.10, GSM8K 0.9401     1/4
        |
        | 1. TP=8 → TP=4 SR (single-replica)
        v
[TP=4 SR MTP=3 strict]                    thr/GPU 1133,    TPOT 7.88                    1/4
        |
        | 2. + ATOM_ENABLE_RELAXED_MTP=1  (RELAXED_TOP_N=8 / DELTA=0.5)
        v
[+ Relaxed MTP]                           thr/GPU 1472,    TPOT 5.59, GSM8K unstable    3/4 if GSM stable
        |
        | 3. + INT4 QuickReduce AR  (VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4)        delta only +6.8% thr
        | 4. + RCCL ROCm 7.1+ knobs (RCCL_MSCCLPP_ENABLE / THRESHOLD / P2P_BATCH)     delta thr 1368→1391, TPOT 6.11→5.93
        | 5. + rocm-smi --resetperfdeterminism  (unlock SCLK 2100→2396 boost)
        v
[Apr 23 R23 RECORD]   snap 2286b9de5107   thr/GPU 1391,    TPOT 5.93, GSM8K 0.9409     3/4 (E2E -32.6%)
   tagged: rocm/atom-dev:dsr1_session17_re1_best_3of4_apr23_0627  +  locked/dsr1:champion_3of4
        |
        | Δ-1. RELAXED_TOP_N 8 → 9            (rejection_sampler.py:11)
        | Δ-2. ATOM_MSCG_K UNSET (was 2)      (silent regression removed)
        | Δ-3. Warmup pattern: 5-large prompts → 8 small curls   ← biggest single contributor
        | Δ-4. lm_eval/api_models.py UnboundLocalError fix       (eval reliability)
        v
[Apr 26 A26 BEST]      snap 8e844757ad6c  thr/GPU 1650,    TPOT 4.84, GSM8K 0.9522     3/4 (E2E -4.8%)
   tagged: rocm/atom-dev:dsr1_apr26_3of4_validated_warm_bench   ⭐ canonical baseline
        |
        | + Phase 1 keystone: block_convert.py:142-215 cudagraph-safe Triton grid (neutral perf, unblocks Phase 3)
        v
[Apr 26 + keystone]    snap d0a431a61e1b  (neutral; reserved for MSCG-P6 revival)
```

## 🛠️ Patches in this branch (vs upstream `aiter` + `ATOM`)

Full per-file diffs and rationale in [`docs/REPRODUCE.md`](docs/REPRODUCE.md) Section 2. Summary:

| File | Change | Backup file (in `re4c_v10`) |
|---|---|---|
| `ATOM/atom/model_ops/rejection_sampler.py:10-13` | `RELAXED_TOP_N` 8 → 9 (kept DELTA=0.5; 8/0.55 was tested DEAD GSM8K 0.9265-0.9287) | `rejection_sampler.py.pre_relaxed9` |
| `lm_eval/models/api_models.py:~514` | `outputs = None` before `try:` block (fix UnboundLocalError on transient API errors) | (in venv site-packages) |
| `ATOM/atom/utils/block_convert.py:142-215` | Triton kernel grid: `cdiv(n_cols, blocks_per_tile)` instead of `cdiv(max_num_blocks, blocks_per_tile)` for cudagraph-safe constant grid (Phase 1 keystone) | `block_convert.py.pre_keystone` |
| **Phase 2 fusion (12 files, BUILT but DORMANT — no callers in deepseek_v2.py yet)**: |
| `aiter-test/csrc/include/custom_all_reduce.cuh` | New `else if constexpr(std::is_same_v<OutT, opus::fp4_t>)` branch in `ar_fusion_epilogue` template — BF16→FP4 packed via direct `__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16`, per-32-group e8m0 scale, 4-lane DPP reduce. `void* scale_out` parameter (was `float*`). Removed unused `using OP` in 2stage launcher. | `custom_all_reduce.cuh.pre_phase2` |
| `aiter-test/csrc/kernels/custom_all_reduce.cu` | New `_fused_allreduce_rmsnorm_mxfp4` static helper + public `fused_allreduce_rmsnorm_mxfp4_quant` entry; routes `dispatchFusedAllReduceRMSNormQuant<bf16, opus::fp4_t>` | `custom_all_reduce.cu.pre_phase2` |
| `aiter-test/csrc/include/custom_all_reduce.h` | Forward declaration | `custom_all_reduce.h.pre_phase2` |
| `aiter-test/csrc/include/rocm_ops.hpp` | Pybind binding | `rocm_ops.hpp.pre_phase2` |
| `aiter-test/aiter/dist/communication_op.py` | `tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant` public entry | `communication_op.py.pre_phase2` |
| `aiter-test/aiter/dist/parallel_state.py` | fake/real op pair (`@torch_compile_guard`), group method, `_out_place` method | `parallel_state.py.pre_phase2` |
| `aiter-test/aiter/dist/device_communicators/communicator_cuda.py` | Device communicator method (fast-path for hidden ∈ {512,1024,2048,4096}) | `communicator_cuda.py.pre_phase2` |
| `aiter-test/aiter/dist/device_communicators/custom_all_reduce.py` | `fused_ar_rms_mxfp4_quant` + `custom_fused_ar_rms_mxfp4_quant` (handles `_IS_CAPTURING` for cudagraph) | `custom_all_reduce.py.pre_phase2` |
| `ATOM/atom/utils/envs.py` | New env flag `ATOM_USE_POST_ATTN_RMSNORM_MXFP4_FUSION` (default 0) | `envs.py.pre_phase2` |

**Why "dormant"**: `DeepseekV2MoE.forward` is wrapped by `torch.ops.aiter.maybe_dual_stream_forward` — a custom op with fixed Tensor-only signature. To consume the new `(unquant, quant, scale)` tuple output requires registering a parallel custom op + branching `Mxfp4MoEMethod.apply` to skip its internal quantization. That last-mile work is multi-day and was deferred. The fp4_t kernel + dispatcher + plumbing are fully built (verified by `nm` symbol check on the rebuilt `module_custom_all_reduce.so`, size grew 2207248 → 2344512 bytes).

## 🔬 Multi-CONC bench results (Apr 26 EOD)

| Concurrency | TP | TPOT_med | thr/GPU | TTFT_med | Interact | E2E_calc | Note |
|---|---|---|---|---|---|---|---|
| **CONC=4** (warm, 3-run median) | 4 | **4.840 ms** | **1650** | 289 ms | **207** | **5240 ms** | ⭐ CANONICAL — GSM8K 0.9522 |
| CONC=4 (no warmup, 3-run median) | 4 | 4.901 ms | 1656 | 290 ms | 204 | 5303 ms | Run 1 cold-tail TPOT_mean +42%, thr/GPU −30% |
| CONC=32 (warm, 2-run median) | 4 | 12.85 ms | 4194 | 2255 ms | 78 | 15,403 ms | saturation regime |
| **CONC=128 WARM (TP=8 cold-boot, 2-run)** | 8 | **26.26 ms** | **3579** | 12,629 ms | 38.09 | 39,489 ms | beats Apr 10-13 vanilla by **−42.9% TPOT, +75% interact, +12.1% thr** |
| CONC=128 NOWARM (cold-boot first benches) | 8 | 26.50 ms | 3051 | 18,777 ms | 37.73 | 45,890 ms | TPOT std 696-771 (massive cold-tail) |

Full bench JSONs preserved in container `re4c_v10:/tmp/proper_run{1,2,3}.json`, `/tmp/no_warmup_run{1,2,3}.json`, `/tmp/conc32_run{1,2}.json`, `/tmp/tp8_conc128_{NOWARM,WARM}_run{1,2}.json`.

## 🚦 Reproduction (single-command stack)

Full step-by-step in [`docs/REPRODUCE.md`](docs/REPRODUCE.md) Sections 3-5. Quick path:

```bash
# 1) docker run from canonical Apr 26 snapshot
docker pull rocm/atom-dev:dsr1_apr26_3of4_validated_warm_bench
docker run -d --name dsr1_repro \
  --ipc=host --shm-size=32g --network=host --privileged --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /docker/huggingface/:/tmp/.cache/huggingface \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  rocm/atom-dev:dsr1_apr26_3of4_validated_warm_bench

# 2) reset perf-determinism + boot
docker exec dsr1_repro rocm-smi --resetperfdeterminism -d 0 -d 1 -d 2 -d 3
docker exec -d dsr1_repro bash /tmp/boot_cdna4_moe.sh
# wait ~10-13 min, tail /tmp/cdna4_boot_*.log until "Application startup complete"

# 3) WARMUP (CRITICAL — 8 small curls hit decode cudagraph batch sizes [1,2,4,8])
docker exec dsr1_repro bash -c '
for i in 1 2 3 4 5 6 7 8; do
  curl -s http://0.0.0.0:8890/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"amd/DeepSeek-R1-0528-MXFP4\",\"prompt\":\"Hello world '"$i"'\",\"max_tokens\":50,\"temperature\":0}" > /dev/null
done && echo warmup_done'

# 4) 3x bench, take median
docker exec dsr1_repro bash -c '
export HOME=/tmp HF_HOME=/tmp/.cache/huggingface HF_HUB_OFFLINE=1
for i in 1 2 3; do
  cd /app/ATOM && python3 -m atom.benchmarks.benchmark_serving \
    --model amd/DeepSeek-R1-0528-MXFP4 --port 8890 \
    --dataset-name random --random-input-len 8192 --random-output-len 1024 \
    --num-prompts 40 --max-concurrency 4 --trust-remote-code \
    --save-result --result-filename /tmp/run${i}.json 2>&1 | tail -25
  sleep 5
done'

# 5) GSM8K (separate eval; ≥0.93 expected)
docker exec dsr1_repro bash -c '
export HOME=/tmp HF_HOME=/tmp/.cache/huggingface TRANSFORMERS_CACHE=/tmp/.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /tmp && lm_eval --model local-completions \
  --model_args model=amd/DeepSeek-R1-0528-MXFP4,base_url=http://0.0.0.0:8890/v1/completions,num_concurrent=16,max_retries=2,tokenized_requests=False \
  --tasks gsm8k --num_fewshot 3 --gen_kwargs temperature=0,max_gen_toks=512 \
  --trust_remote_code --batch_size auto 2>&1 | tail -8'
```

Expected: thr/GPU ≥ 1600, TPOT_med ≤ 5.0 ms, Interact ≥ 200, E2E ≈ 5200-5400 ms, GSM8K ≥ 0.94.

## 📦 Snapshot manifest (post Apr 26 EOD pruning)

| Tag | Image ID (sha256) | Size | Role |
|---|---|---|---|
| `rocm/atom-dev:dsr1_session17_re1_best_3of4_apr23_0627` | `2286b9de5107` | 478 GB | Apr 23 R23 baseline (3 live containers run on this image) |
| `locked/dsr1:champion_3of4` (alias) | `2286b9de5107` | (shared) | Safety alias of R23 |
| **`rocm/atom-dev:dsr1_apr26_3of4_validated_warm_bench`** ⭐ | **`8e844757ad6c`** | **485 GB** | **Apr 26 A26 canonical 3/4-gates baseline** |
| `rocm/atom-dev:dsr1_apr26_triton_keystone_3of4` | `d0a431a61e1b` | 484 GB | A26 + Phase 1 Triton keystone (neutral, unblocks future MSCG-P6) |

3 stale snapshots (`relaxed9_mscgK_off_baseline`, `session17_moe_patchA_apr24`, `session17_pre_moe_attack_apr24`) and 5 stale containers were pruned Apr 26 EOD; ~700 GB disk freed.

## 📚 Documentation map (full directory: [`docs/`](docs/))

🟢 **Tier 1 — keep updated after every experiment**:
- [`docs/REPRODUCE.md`](docs/REPRODUCE.md) — canonical AMD-PR delivery manifest + stack genealogy + repro recipe (read this first)
- [`docs/MASTER.md`](docs/MASTER.md) — engineering log + multi-CONC bench history + plan + findings + experiments
- `docs/SNAPSHOT_INVENTORY_apr24.md` — snapshot SHA inventory (refresh now: 3 keepers post Apr 26 prune)

🟡 **Tier 2 — refresh per major perf shift**:
- [`docs/PROFILING_PLAYBOOK.md`](docs/PROFILING_PLAYBOOK.md) — torch.profiler decode breakdown (top kernels by GPU time)
- [`docs/SERVER.md`](docs/SERVER.md) — launch-script variants (largely subsumed by REPRODUCE.md Section 3)
- [`docs/MOE_Lever1/`](docs/MOE_Lever1/) — MoE kernel investigation reports
- [`docs/Archive/L7_L8_vllm_recipe_apr25.md`](docs/Archive/L7_L8_vllm_recipe_apr25.md) — upstream vLLM recipe levers
- [`docs/Archive/FP8_ATTN_apr25.md`](docs/Archive/FP8_ATTN_apr25.md) — FP8 attention investigation

🟠 **Tier 3 — historical (read-only)**:
- `docs/Archive/SESSION17_*` — Apr 23-24 progress logs and plans
- `docs/Archive/GATE_ROADMAP_apr24.md` — pre-Apr-26 roadmap
- `docs/MOE_Lever1/PatchA*.md` — Patch A AGPR variants

## 🚫 Things tried this session and DEAD (so AMD doesn't re-test)

Full record in [`docs/REPRODUCE.md`](docs/REPRODUCE.md) Section 8. Highlights:

| Lever | Result |
|---|---|
| `RELAXED_TOP_N=8 RELAXED_DELTA=0.55` | DEAD — GSM8K 0.9265-0.9287 < 0.93 |
| `RELAXED_TOP_N=10 DELTA=0.6` | DEAD — GSM8K 0.9227 |
| `ATOM_USE_CDNA4_MOE_GEMM2=1` (B1 kernel) | NEUTRAL — kernel built + dispatching but +0.37 ms TPOT regression (FlyDSL atomic already in dispatcher hot path) |
| MSCG P6 main+drafter graph wire | DEAD at replay — `eagle.py:184` in-place `kv_indptr` mutation OOB |
| MTP=4 native | BLOCKED — AITER ASM kernel only supports `max_seqlen_qo ∈ {2,4}` for nhead=32 fp8/fp8 gfx950 |
| `--enable-tbo all` | CATASTROPHIC — thr −29%, TPOT +53%, E2E +44% (decode TBO regresses on DSR1 MTP=3) |
| `NCCL_MIN_NCHANNELS=32` (vs 16) | DEAD — interact fails 165 gate |
| `INT8 QR` (vs INT4) | DEAD — TPOT +0.25 ms, TTFT +84 ms |
| `--enable_prefix_caching` | CRASH — `ValueError: cannot reshape array of size 1 into shape (1,4)` |
| `AITER_ENABLE_HK_QH32_V11=1` | CRASH — memory fault during cudagraph capture at sq=8 |
| `ATOM_USE_TRITON_GEMM=1` | DEAD — pulls BF16 GEMMs to untuned Triton fallback |
| L4.5 Fuse_A_GEMM | BLOCKED — gated behind use_triton_gemm + ENABLE_DS_QKNORM_QUANT_FUSION (we use AITER) |
| L7 DCP=4 | BLOCKED — ATOM has no decode-context-parallel-size flag |
| ATOM PR #582 | DOESN'T HELP — SGLang-only |
| aiter PR #2823 (FP8 fused AR+RMSNorm+quant) | NOT NEEDED for our path — we built fp4_t variant from scratch |
| vLLM PR #36574 (persistent MLA) | ALREADY INTEGRATED — ATOM uses persistent MLA via own dispatch |

## 🎯 Open gap & path to 4/4

E2E is the only failing gate at CONC=4, missing by **240 ms** (4.8%). From `E2E = TTFT_med + (OSL−1)·TPOT_med`:
```
5000 = 290 + 1023 × TPOT  →  TPOT_max = 4.61 ms  →  need −0.23 ms TPOT (−4.7%)
```

**Active lever**: Phase 2 AR+RMSNorm+MXFP4 fusion (estimated −0.5 to −1.0 ms TPOT). Kernel + dispatcher + Python plumbing all built and present in this branch (see Section 2.4 of REPRODUCE.md). Final wiring through `maybe_dual_stream_forward` custom-op signature is the remaining work (multi-day; deferred from Apr 26).

Backup lever: MTP=4 ASM kernel `max_seqlen_qo=5` variant (Phase 5, multi-week, high risk).

## ⚙️ Stack (this branch)

- **Base image**: `rocm/atom-dev:dsr1_session17_re1_best_3of4_apr23_0627` (sha `2286b9de5107`)
- **ROCm**: 7.2.2 / **HIP runtime + LLVM 21**
- **PyTorch**: 2.10.0+rocm7.2.2.git40d237bf
- **aiter**: HEAD on this branch (with Apr 26 EOD patches)
- **ATOM**: HEAD on this branch (with Apr 26 EOD patches)
- **flydsl**: 0.1.3.1
- **triton**: 3.5.1
- **Model**: `amd/DeepSeek-R1-0528-MXFP4` (HF cached at `/tmp/.cache/huggingface/hub`)
- **Hardware**: 4× AMD Instinct MI355X (gfx950, CDNA4) per TP=4 run; 8× available on host
- **TP**: 4 (CONC=4/32 benches); 8 (CONC=128 benches)
- **KV cache**: FP8
- **Speculation**: native DeepSeek MTP (k=3), relaxed acceptance (RELAXED_TOP_N=9, RELAXED_DELTA=0.5)
- **Key flags**: `--method mtp --num-speculative-tokens 3 --enable-tbo prefill --max-num-batched-tokens 65536 --cudagraph-capture-sizes "[1,2,4,8,16,32]"`

## 📁 Repository layout

```
.
├── README.md                             ← you are here
├── docs/
│   ├── REPRODUCE.md                      ← ⭐ AMD-PR FINAL DELIVERY MANIFEST
│   ├── MASTER.md                         ← engineering log + multi-CONC sweep + plan
│   ├── PROFILING_PLAYBOOK.md             ← torch.profiler decode breakdown
│   ├── SERVER.md                         ← legacy launch-script variants table
│   ├── SNAPSHOT_INVENTORY_apr24.md       ← snapshot SHA inventory (refresh post-prune)
│   ├── MOE_Lever1/                       ← MoE kernel investigation
│   └── Archive/                          ← historical session logs and plans
├── ATOM_main/                            ← modified ATOM source tree (Apr 23 baseline)
├── RE_MoE_CDNA4/                         ← B1 CDNA4 MoE GEMM2 atomic kernel (built, NEUTRAL)
├── RE4_hk_qh32/                          ← Apr 23 HK qh32 V11 kernel attempts (DEAD)
├── aiter_configs/                        ← BF16 tune CSVs
├── bench_results/                        ← raw bench JSONs from prior sessions
├── dsr_beta/                             ← DSR_beta snapshot package (Apr 18, superseded)
├── phase_re_artifacts/                   ← RE.0 / RE.1 launch scripts
├── repro/                                ← reproduction helpers
├── scripts/                              ← merge / parse / analysis scripts
├── session_logs/                         ← chronological engineering logs
└── patches/                              ← clean diffs for upstream PR submission
```

## 🔀 Branch & tag map

| Ref | Purpose |
|---|---|
| `main` | DEC-075 production floor (Apr 17, ROCm 7.1.1 stack) — stable historical |
| `dsr_beta_snapshot` | DSR_beta + TBO prefill (Apr 18, 1335/6.40/156, 1/4 gates) — superseded |
| **`session17_fp8_sq8_mla`** | **Active branch — Apr 26 EOD work, 3/4 gates** |
| **tag `dsr1-A26-3of4-apr26`** | **Apr 26 baseline marker — points at commit `8f6df64`** |

## 📨 PR submission checklist (for AMD upstream)

- [ ] Push the Section 2 patches as a single PR onto `aiter` upstream (Phase 2 fusion 12 files + block_convert keystone)
- [ ] Push the Section 2.1 / 2.2 patches as a single PR onto `ATOM` upstream (rejection_sampler + envs.py)
- [ ] Push the Section 2.2 lm_eval fix as a separate PR onto `lm-evaluation-harness`
- [ ] Provide bench JSON files (in container `re4c_v10:/tmp/`)
- [ ] Provide GSM8K result file
- [ ] Reference snapshot SHA `8e844757ad6c` for byte-identical reproduction
- [ ] Note Phase 2 wiring + `maybe_dual_stream_forward` signature change as a follow-up PR

## ⚖️ Legal / usage

Private snapshot, backup only. Do not redistribute without Danish's permission. AMD ATOM code is licensed per ROCm/ATOM repository's original license.

---

**Last update**: 2026-04-26 EOD — A26 baseline 3/4 gates verified (TPOT 4.84 / thr/GPU 1650 / interact 207 / GSM8K 0.9522), E2E gap 240 ms. Phase 2 fusion plumbing built (dormant). Snapshot `8e844757ad6c`. Tagged [`dsr1-A26-3of4-apr26`](https://github.com/Danishlynx/AMD_DSR_CNCC4/releases/tag/dsr1-A26-3of4-apr26).

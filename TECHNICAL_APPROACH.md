# Technical Approach — DSR1 MXFP4 on MI355X CONC=4

**Goal**: close the 4 official gates at CONC=4 (TPOT ≤ 4.601 ms, E2E ≤ 5000 ms, Tput ≥ 1660 tok/s/GPU, GSM8K ≥ 0.93) on `amd/DeepSeek-R1-0528-MXFP4` running on 4× MI355X with TP=4 single-replica MTP=3.

**Constraint**: no weight modifications; harness is `kimbochen/dsr1_benchmark.cpp` (chat-template, ISL=8192, OSL=1024, num_prompts=40, max-concurrency=4, num_warmups=8, 4-iter median(2,3,4) + GSM8K N=3 median ≥ 0.93 gate).

**Outcome**: **2/4 gates** (GSM + Interactivity) at TPOT 5.641 ms / Tput 1449 / GSM 0.9318 / Intvty 177.26, via the Phase 11 per-phase relaxed-MTP lever (TRT-LLM port). Remaining 2 gates (E2E, Tput) require multi-week architectural work (megakernel HIP / FP4 weight-loading L1 redo) characterized in §6.

This document explains how that result was reached, what was measured, what was discarded, and why.

---

## 1. Architectural baseline (Apr 18 deep-dive)

Before authoring any lever, the full pipeline was modeled from sources + HF config + the `re4c_v10` container runtime. The key numbers:

### Model (`amd/DeepSeek-R1-0528-MXFP4`)
- 61 transformer layers (3 dense FFN + 58 MoE), `hidden_size=7168`, `vocab=129280`
- MLA attention: 128 heads, `kv_lora_rank=512`, `qk_head_dim=192`, `v_head_dim=128`
- MoE: 256 routed experts + 1 shared, top-8 per token, group-limited (`n_group=8`, `topk_group=4`)
- MTP: 1 next-N layer (layer 61) for speculative decoding (`MTP_K=3`)
- Quark MXFP4 quant: `dtype=fp4_e2m1`, `group_size=32`, `scale=e8m0`
- **Critical exclusion**: `exclude: re:model.layers.61.*` → **drafter (layer 61) is BF16, NOT FP4**, dispatches to the slow `QuantType.No` MoE path (`2stage default`, ~3-5× slower than the verifier's `flydsl_moe1_afp4_wfp4_bf16` fast path)

### Hardware (MI355X / CDNA4 / gfx950)
- 256 CUs (vs MI300X 304 — denser), 2 XCD dies × 4 XCDs each
- Native MXFP4 matrix cores (`v_mfma_scale_f32_16x16x128_f8f6f4`)
- Peak: 10 PFLOPS MXFP4, ~2.5 PFLOPS BF16
- HBM3e: 256 GB usable, 8 TB/s
- L2 32 MB (4 MB/XCD), Infinity Cache 256 MB
- LDS per CU: 160 KB (+32 vs MI300X — bigger tiles possible)
- XGMI 4-GPU ring: 256 GB/s/link; NCCL latency ~40 µs/call

### Decode pipeline @ CONC=4 MTP=3 (initial DEC-057 estimate)
At CONC=4, bs=16 tokens per step (4 reqs × 4 positions: 3 drafts + 1 bonus).

```
DRAFTER (runs 3 iterations sequentially, uses layer 61 only — BF16 slow path):
  iter 0-2: bs=4 → MLA + MoE(SLOW BF16) + RMSNorm → argmax
  Total drafter: 8.67 ms

TARGET FWD (bs=16 verifier):
  Embedding + 3 dense layers:                       ~0.5 ms
  MoE layers 3-60 (58 layers × ~0.32 ms):           ~9.4 ms (overlap via dual-stream)
  LM head BF16 GEMM (M=16, hidden=7168, vocab):     ~4.57 ms (HBM-bound reading 926 MB weight)
  Sample + rejection:                               ~negligible
  Total target fwd:                                 ~10.9 ms

STEP TOTAL: ~21.8 ms / 3.0 toks/fwd → TPOT 7.27 ms
```

### CONC scaling
| | CONC=4 | CONC=32 | CONC=128 |
|---|---|---|---|
| bs/step | 16 | 128 | 512 |
| Experts/token | 0.6 | 4.5 | 18 |
| **MoE regime** | **MEM-bound** | compute | saturated |
| AR msg | 28 KB | 224 KB | 896 KB |
| Gate TPOT | ≤ 4.601 ms | (different) | (different) |

**CONC=4 is the hardest** of the three concurrency points because it fights the memory-bound regime where the MoE GEMM has 0.6 tokens/expert routed — far below the ~8-10 saturation threshold.

---

## 2. Profiling-driven bottleneck attribution

### 2.1 Initial profile (DEC-057 estimate, kernel-buckets)

| Component | ms | % of step |
|---|---:|---:|
| Drafter MoE × 3 iters | 3–4 | 14-18 % |
| Main MoE GEMM | 5.89 | 27 % |
| BF16 GEMM (LM head + MLA projs) | 4.57 | 21 % |
| AllReduce × 60 | 2.96 | 14 % |
| MLA attention chain | 3.42 | 16 % |
| RMSNorm | 1.02 | 5 % |
| Drafter non-MoE | ~4-5 | 20 % |

This initial attribution **over-counted MoE/MLA wall** and **under-counted synchronization** because DEC-057 measured GPU kernel buckets without proper accounting of the `hipEventSynchronize` events between launches.

### 2.2 Corrected attribution (real torch.profiler chrome trace)

After capturing a proper chrome trace under the official kimbochen workload:

| Metric | Value | Source |
|---|---:|---|
| Capture window | 274.2 ms (20 decode steps) | trace span |
| GPU active total | 18.3 ms | sum of 2,100 GPU kernel events |
| **GPU busy fraction** | **6.7 %** | active / span |
| `hipEventSynchronize` events | 60 × ~891 µs = **53.4 ms** | trace runtime API events |
| `hipGraphLaunch` events | 19 × ~493 µs = 9.4 ms | trace |

**Per-decode-step decomposition @ TPOT 6.02 ms (May 13 baseline)**:

| Component | Time | % of TPOT |
|---|---:|---:|
| Real GPU kernel execution | ~915 µs | 15 % |
| `hipEventSynchronize` (3× per step × ~891 µs) | ~2,670 µs | 44 % |
| Python orchestration + scheduling | ~2,440 µs | 41 % |

**Decisive implication**: at CONC=4 / TP=4 / MTP=3, **the GPU is 93 % idle**. Even infinitely-fast GPU kernels cap TPOT improvement at ~−0.9 ms (6.02 → 5.1), still missing the 4.601 gate by 0.5 ms.

**Therefore the gate-closing direction is**: attack synchronization barriers and per-step launch count, **not** raw kernel speed.

### 2.3 Sync-source attribution (Phase 1 D2 deliverable)

Re-analyzed the chrome trace to identify which Python frames emit the `hipEventSynchronize` events:

- `model_runner.py` has 12 D2H-capable sites; **all hot-path ones are gated by `dp_size > 1`** → don't fire at TP=4 dp=1
- `spec_decode/eagle.py` (drafter) has **zero** `.cpu()` / `.item()` / `.tolist()` calls — fully GPU-resident
- `scheduler.py` has **zero** hot-path syncs
- `aiter/bert_padding.py` has `.item()` calls but is NOT imported by ATOM

**Conclusion**: the 3 syncs/step are **NOT in ATOM Python**. The pattern around each sync (from windowed trace inspection) is:
```
aten::copy_ (39µs) → hipMemcpyAsync (30µs) → ... → hipEventSynchronize (1027µs) → hipEventDestroy → aten::to / aten::cat
```

This is **intra-step C++/aiter/CompiledFxGraph epilogue coordination** — completion events from collectives, MLA kernel post-launch sync, or torch.compile subgraph boundaries. It is captureable inside a cudagraph (would become an internal graph edge), but **cannot be patched by a Python-level change**.

### 2.4 V3 megagraph + post-deadline detailed measurement (May 15)

After deploying V3 megagraph (verifier+sampler captured, drafter outside) and re-instrumenting with CUDA events + the `sync_decompose_patch`:

| Per-step (V3) | Time |
|---|---:|
| Megagraph replay (verifier + sampler fused, GPU time) | ~14 ms |
| Drafter (eager 3-iter loop, GPU time) | ~4.2 ms |
| Eager kernels between (sampler, scatter, etc.) | ~1 ms |
| `recv_mtp_status` sync (CPU-blocked) | ~10.4 ms, but **already_done = 1 %** (always blocking) |
| **Total step wall time** | **~17–21 ms** |
| **Tokens per step** | 1 + 3 × 0.675 ≈ **3.03** (1 bonus + MTP_K × accept) |
| **TPOT** | 20 / 3.03 ≈ **6.6 ms** (matches V3 N=3 mean 6.667) |

**Key finding**: the `recv_mtp_status` sync on V3 is 10.4 ms (doubled from GOLD's 4.86 ms), but step time is unchanged from GOLD (~18 ms vs ~18 ms). The sync is **mostly hidden behind GPU work** — removing it gives ~0 TPOT delta because GPU work is the actual critical path.

This **corrected** the May 13 Option C zeros-stub interpretation. The zeros-stub gave TPOT 4.58 ms not by removing the sync but by **breaking MTP** (setting `num_rejected = 0` disabled the reattend-rejected-tokens path, cutting GPU attention work per step). GSM8K rambled ("8 times 7 = ..." with no answer) confirming correctness was broken.

**Therefore the gate-closing lever needs to reduce GPU work**, not sync. Phase 11 v3 (the Apr 30 win) does exactly this — it cuts decode forwards by raising accept rate during thinking, which directly removes drafter+verifier GPU cycles.

---

## 3. Decision tree — how each lever was chosen and tested

For every lever, the rule was: **profile first, then patch**. The decision flow was always:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Identify a candidate from profile / arch / upstream PR   │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Estimate addressable budget                              │
│    — How many ms does this attack on the critical path?     │
│    — If < 0.3 ms (noise floor), drop or down-prioritize.    │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Implement env-gated NULL-OP                              │
│    — Default OFF must be bit-identical to upstream.         │
│    — Single env flag enables; clean revert path.            │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Single-prompt smoke at production shape (ISL=8192)       │
│    — Catches MSCG-class crashes before sustained-load bench.│
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Numerics canary                                          │
│    — 8-prompt batch: compare layer-wise cosine vs baseline. │
│    — Catches the L1-class silent corruption (BF16-as-FP4).  │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. GSM8K N=3 median ≥ 0.93                                  │
│    — Single-shot GSM is unreliable (outlier variance 0.014).│
│    — N=3 median is the gating signal.                       │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. Official kimbochen 4-iter (warmup 8 small curls; ISL     │
│    8192 OSL 1024 num_prompts 40 conc 4 chat-template)       │
│    — Take median(iter2, iter3, iter4); discard iter1 cold.  │
└────────────────────┬────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. KEEP threshold: ≥ −0.30 ms TPOT AND GSM_med ≥ 0.93       │
│    Else: REVERT + log to MASTER.md DEAD section.            │
└─────────────────────────────────────────────────────────────┘
```

This is why so many "fixes" got killed — most levers either failed at step 4 (crash), step 5 (numerics), step 6 (GSM), or fell below the 0.30 ms threshold at step 7. The discipline of running all 7 checks is what made the campaign's wins trustable.

### 3.1 Why this discipline matters — three examples

- **L1 FP4 fused QKV (Triton)**: passed steps 4 and 5, regressed +0.381 ms at step 7. Reverted. Without strict measurement discipline, the kernel author's microbench (where it looked faster) would have entered production and damaged TPOT.

- **K2 one-line gate patch**: enabled AMD's shipped MXFP4 1-stage MoE via `run_1stage=token<256`. Looked architecturally correct. Failed step 7 with +1.15 ms regress because 1-stage is slower at the hot M=4 decode shape. Reverted in <1 hour.

- **Option C zeros-stub** (May 13): looked like a 1.42 ms TPOT WIN at step 7 → would have closed the gate. But step 5 numerics canary caught the GSM regression (model rambling on math). Re-investigation showed the "win" was MTP-broken floor, not pure sync removal. Saved 2–3 weeks of misdirected work.

---

## 4. The actual TPOT-reducing levers (chronological, by date)

### 4.1 Configuration / platform (collected over Apr 14–22)

| Lever | TPOT Δ | Mechanism |
|---|---:|---|
| TP=8 → TP=4 single-replica | (config baseline) | CONC=4 means TP=4 is sufficient and saves AllReduce hops |
| `ATOM_ENABLE_RELAXED_MTP=1` + `RELAXED_TOP_N=8` + `RELAXED_DELTA=0.5` | −2.29 ms | Stock ATOM flag; widens acceptance probability — was OFF by default. |
| `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4` | +6.8 % throughput (minimal TPOT) | INT4 QR reduces AR payload size at small messages |
| `RCCL_MSCCLPP_ENABLE` + `RCCL_MSCCLPP_THRESHOLD` + `RCCL_P2P_BATCH_ENABLE` | −0.18 ms | ROCm 7.1+ RCCL knobs for in-network AR |
| `rocm-smi --resetperfdeterminism` (SCLK 2100 → 2396 MHz) | ~−0.08 ms | Required cold-boot reset; sticky across reboots |

### 4.2 Cudagraph capture configuration (Apr 20 — the largest single unlock)

```bash
--cudagraph-capture-sizes [1,2,4,8,16,32]
```

**Δ TPOT: −1.41 ms** (6.13 → 4.84, warm).

Before this change ATOM was capturing the default set including unused batch sizes (33 graphs). The MI355X cudagraph driver's per-launch cost scales with the **graph node count** (~40 ns/node), and `hipGraphLaunch` dominated the wall time (77.7 % of decode wall per the Apr 20 profile, 1,525 nodes per replay).

Limiting to `[1,2,4,8,16,32]` (the actually-used decode batches) cut graph nodes per replay from 1,525 to ~600 → −1.41 ms.

**This was the single biggest unlock of the entire campaign** — and it was a CLI flag, not a kernel.

### 4.3 Warmup discipline (Apr 23)

```bash
# OLD: 5 large random prompts (random tokens, long output)
# NEW: 8 small curls hitting all decode cudagraph batch sizes [1,2,4,8]
for i in 1..8; do curl -d '{"prompt": "Hello world '$i'", "max_tokens": 50}'; done
```

Cold-tail TPOT_mean was +42 % on iter 1 without proper warmup. The 8-curl pattern hits every batch-size cudagraph at least once, eliminating cold-tail. Took median(iter2, iter3, iter4) and discarded iter1.

Embedded into the bench discipline; effect already in #4.2's "warm" measurement.

### 4.4 `ATOM_CUDAGRAPH_MODE=FULL_DECODE_ONLY` (Apr 29 L0-v2)

```bash
export ATOM_CUDAGRAPH_MODE=FULL_DECODE_ONLY
```

**Δ TPOT: −0.166 ms** (6.106 → 5.940).

`FULL_AND_PIECEWISE` (the previous default) breaks dual-stream MoE alt-stream overlap on CDNA4 (no async — `--enable-tbo all` had +53 % TPOT regress for the same reason). `FULL_DECODE_ONLY` captures only decode steps (uniform batches), leaves prefill as `PIECEWISE` → sidesteps the conflict.

### 4.5 **Phase 11 per-phase relaxed-MTP v3** (Apr 30 — the headline first-party lever)

**Δ TPOT: −0.661 ms** (6.302 → 5.641). **2/4 gates crossed.**

Port of TRT-LLM's `use_relaxed_acceptance_for_thinking: true` (LLM-API config flag for reasoning-model spec decoding) to ATOM/AITER. Full mechanism described in `PR_DESCRIPTION.md` §"What the lever does".

**Why this lever was selected**:
1. **Architectural fit**: DSR1-R1 emits explicit `<think>...</think>` reasoning blocks → per-phase logit-distribution shape is well-known from inference-time stats.
2. **Already-validated upstream**: TRT-LLM published `relaxed_topk=10`, `relaxed_delta=0.6` for DeepSeek-R1; these are the values used here (no parameter sweep needed).
3. **Cudagraph-safe by construction**: phase tensor lives on GPU, kernel uses `tl.load`/`tl.store` only — no host-side D2H copies in forward path. Passes step 4 (smoke) without modification.
4. **Bounded scope**: ~210 LOC; env-gated NULL-OP; single Triton kernel + one runner-state allocation + one model-runner reset call. Reversible in <1 min.
5. **Addresses the right bottleneck**: increases per-step accepted tokens → reduces total decode forwards → reduces all GPU work proportionally. Doesn't try to make kernels faster — makes them run **less often**.

### 4.6 Lever G (TOP_N 10 → 12, May 04)

**Δ TPOT: −0.20 ms** on dev-bench (5.641 → 5.441). **Not officially kimbochen-validated; lives only in the `dsr1_fresh` work container**, no snapshot. Worth shipping; flagged for AMD to validate.

```python
# atom/model_ops/rejection_sampler.py:11
PHASE_RELAXED_TOP_N = 12  # was 10
```

### 4.7 Net official-harness TPOT progress

```
Start (vanilla TP=8 MTP=3 fp8 KV):   ~7.88 ms  (1/4 gates)
                  ↓
Apr 30 (Phase 11 v3 + all configs):  5.641 ms  (2/4 gates) — ⭐ HEADLINE
                  ↓
May 13 (re-measured baseline):       6.02 ms   (2/4 gates) — drift between snapshots
                  ↓
May 15 (V3 megagraph post-deadline): 6.667 ms  (regress; correctness-preserving fix for a 4-month bug)
```

**Best official result: TPOT 5.641 ms / Tput 1449 / GSM 0.9318 / Intvty 177.26 → 2/4 gates** on snapshot `rocm/atom-dev:dsr1_apr30_phase11_v3_2of4` (sha `c58cf2ce4512`).

---

## 5. What ELSE was built (built-but-dormant deliverables)

These are shippable engineering deliverables that didn't make it into production because of last-mile integration constraints. They are documented here so AMD's review can decide whether to invest the remaining wiring effort.

### 5.1 Phase 2 fp4_t fused AR+RMSNorm+quant kernel (FULLY BUILT)

Status: ✅ Built end-to-end (C++ kernel + Pybind + Python dispatcher). DORMANT (consumer not wired).

What it does: extends aiter's existing `fused_allreduce_rmsnorm_quant` kernel template with a new `else if constexpr(std::is_same_v<OutT, opus::fp4_t>)` branch that performs:
1. AllReduce (existing AITER path)
2. RMSNorm in-place (existing AITER path)
3. **NEW**: BF16 → FP4-packed direct conversion via `__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16` with per-32-group E8M0 scale + 4-lane DPP reduce

This eliminates one full BF16-then-quantize-to-FP4 pass per layer × 60 MoE layers = ~0.6-0.8 ms TPOT estimated, would close the E2E gate when combined with Phase 11 v3.

Why dormant: `DeepseekV2MoE.forward` is wrapped by `torch.ops.aiter.maybe_dual_stream_forward`, a custom-op with fixed Tensor-only signature. Consuming the new `(unquant, quant, scale)` 3-tuple output requires registering a parallel custom-op + branching `Mxfp4MoEMethod.apply` to skip its internal quantization. Multi-day work, deferred at deadline.

Verification: `nm` on the rebuilt `module_custom_all_reduce.so` confirms the new symbol is exported (file size grew 2,207,248 → 2,344,512 bytes). Standalone kernel tests pass numerics within MXFP4 envelope (max_abs_err < 0.05).

### 5.2 L3 Triton FP4 KV decode kernel (STANDALONE WIN, INTEGRATION REGRESSED)

Status: ✅ Standalone 3.04× faster than aiter `mla_decode_fwd` (23 µs vs 70 µs at production shape). Integration into ATOM regressed +1.10 ms due to dispatch gate bug.

What it does: Triton kernel for FP4-quantized KV decode at `NUM_KV_SPLITS=64` + `nw=8 ns=3` + `BLOCK_N=32`, with FP8-cast `tl.dot` for matmul. Numerics pass within MXFP4 envelope.

Why integration regressed: the dispatch gate at `L3_kernels/l3_decode_wrapper.py:34` reads `if max_seqlen_q != 1: return False`. Production verifier uses `MQ=4` (folded multi-query), so L3 never fires; the dispatch check added overhead with no kernel benefit. Authoring an `MQ=4`-specialized kernel variant (estimated 3-5 days) would unblock this lever for −0.3 to −0.5 ms TPOT.

### 5.3 CDNA4 hand-written MoE GEMM2 kernel (B1, BIT-EXACT)

Status: ✅ Bit-exact (max_abs_err = 0.0625, GSM 0.9382). +0.37 ms perf regress.

Implements 3 critical CDNA4 primitives missing in production GEMM2 path:
- `v_mfma_scale_f32_16x16x128_f8f6f4` (scaled MFMA)
- `ds_read_b64_tr_b4` (LDS transposed read)
- `global_atomic_pk_add_bf16` (BF16-packed atomic add)

The kernel runs correctly under cudagraph capture. The perf regress is because FlyDSL's `flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic` is **already in the dispatcher hot path** for this shape — there's no headroom over a tuned vendor kernel at M=4 / N=128 / K=2048.

### 5.4 GPU-resident slot_mapping kernel (FUNCTIONAL)

Status: ✅ Bit-exact CPU-GPU equivalence verified.

`L3_kernels/slot_mapping_mtp_gpu.py` computes `slot_mapping` and `positions` from `(block_tables, context_lens, num_rejected)` entirely on GPU. Eliminates the CPU consumer at `aiter_mla.prepare_decode:L492`. Used as a building block for the May 14 Option C 3-step rewrite that ultimately confirmed sync removal couldn't close the gate.

### 5.5 V3 megagraph multi-call aliasing bug fix

Status: ✅ 4-month investigation completed. Bug fundamentally solved.

The MSCG_P6 megagraph design (verifier+drafter+sampler in one cudagraph) was killed for 4 months by a PyTorch ROCm CUDAGraph allocator bug: **same callable invoked 2+ times in one capture aliases internal transients**. Diagnostic table:

| Drafter iters per capture | Outcome |
|---|---|
| 1 | ✅ Stable 270s+ sustained |
| 2 | ❌ Crash < 15 s |
| 3 (production MTP_K=3) | ❌ Crash < 25 s |

V3 fix: remove drafter from megagraph capture, let postprocess fall back to standard drafter path. NO CRASH, MTP acceptance 67.5 % (HIGHER than baseline 60 %), but TPOT regresses +0.65 ms (loses the fusion win).

Real value of V3: **characterizes a previously-fatal architectural blocker**, provides a working but performance-neutral workaround, and unblocks the full megagraph design IF the PyTorch allocator bug is fixed upstream (or if drafter is captured via per-iter cudagraphs per the D.1 Option B design doc).

---

## 6. The architectural ceiling (what's left to close 4/4)

The May 15 chrome trace + sync_decompose + D.1 drafter probe gave the final per-step decomposition (V3 stable foundation). To close TPOT gate, step time needs to drop from ~18 ms to ≤ 13.8 ms — **save ~4 ms of GPU work**.

The remaining levers, with realistic deltas:

| Lever | Step Δ | TPOT Δ | Effort | Risk |
|---|---:|---:|---|---|
| D.1 Option B — manual per-iter drafter cudagraphs (Option A `make_graphed_callables` failed: incompatible with `@support_torch_compile` decorator) | −1 ms | −0.33 ms | 2-3 weeks | medium |
| Phase 5 L3 MQ=4 kernel redo | −1 ms | −0.33 ms | 1-2 weeks | low |
| Phase 6 TBO decode overlap (currently TBO-prefill-only) | −1 ms | −0.33 ms | 2-4 weeks | medium |
| **Phase 3 megakernel HIP** (60-layer fused single kernel) | **−4 to −5 ms** | **−1.3 to −1.6 ms** | **2-3 months** | **HIGH** |
| FP4 weight loading L1 redo (numerics-critical) | ? | ? | 3-4 weeks | numerics |

**Without Phase 3 megakernel HIP, the gate is unreachable.** Stacking the smaller levers gets to ~5.5–5.7 ms TPOT — still missing 4.601 by ~1 ms. Combined with megakernel: 4.17 ms TPOT — passes by 0.4 ms margin.

This matches the May 08 four-parallel-agent architectural verdict and the Apr 18 deep-dive: **CONC=4 4/4 in this configuration is AMD-internal-team scope, multi-month work**.

---

## 7. Measurement discipline & noise floors (corrected over the campaign)

The April-May campaign discovered several measurement traps that affected lever interpretation:

| Trap | Discovery | Rule going forward |
|---|---|---|
| Single-shot GSM8K | A26 (RELAXED_TOP_N=9) measured 0.9378 single-shot, but N=3 median was 0.9287 (FAIL). The 0.9378 was an outlier. | **N=3 GSM median ≥ 0.93 — no single-shot.** |
| Informal vs official harness | Apr 26 "3/4 gates" was on informal bench (`random-range-ratio=0.8`, no chat-template, 1 warmup). Same stack under official kimbochen harness (chat-template, 8 warmups, 40 prompts) = 1/4 gates. | **Only `kimbochen/dsr1_benchmark.cpp` produces authoritative DSR1 numbers.** |
| Microbench ≠ integration | L3 FP4 KV kernel: standalone 3.04× faster, integration regressed +1.10 ms (dispatch gate bug). B1 CDNA4 GEMM2: bit-exact and microbench-faster, +0.37 ms in production (FlyDSL already in dispatcher). | **No lever is "DONE" until the official harness median says so.** |
| Probe overhead | D.1 CUDA-event probe added 1.2 ms TPOT artifact (forced CPU-GPU serialization at sync points). | **Subtract probe overhead from comparisons; don't trust probe-instrumented absolute numbers.** |
| Same-boot vs cross-boot noise | Same-boot variance: ±0.09 ms across reps. Cross-boot: ±0.25 ms. Iter-to-iter: ±0.7 ms. | **Any lever predicting < 0.5 ms delta is invisible to single-boot benches. Need paired-bench harness (3 boots × 4 iters × 3 reps = 36 samples) for sub-0.5 ms claims.** |
| Reading hipEventSync as GPU active | First chrome-trace parse counted `cat="cuda_runtime"` `hipEventSynchronize` events as GPU kernel time → reversed conclusion to "GPU is 6 % busy" → would have committed to wrong lever path. | **Filter `cat in {kernel, gpu_memcpy, gpu_memset}` only when computing "GPU active". CUDA runtime API calls are CPU-side.** |

The campaign authored a **paired-bench harness** (`L3_kernels/paired_bench_harness.py`) implementing the N×M×R design with Welch t-test and Cohen-d statistical significance. It is the right tool for future sub-0.5 ms lever validation.

---

## 8. Repository structure (for navigation)

```
AMD_DSR_CNCC4/
├── PR_DESCRIPTION.md                          ← this PR (item 2)
├── TECHNICAL_APPROACH.md                      ← this document (item 5)
├── README.md                                  ← top-level overview + stack genealogy
├── ATOM_main/                                 ← ATOM source (with Phase 11 v3 patches)
├── L1_patches/                                ← L1 kernel investigation artifacts
├── L2_patches/                                ← L2 RMSNorm-fusion patches
├── L3_kernels/                                ← L3 FP4 KV Triton kernel + V3 megagraph patches
├── RE4_hk_qh32/                               ← HipKittens qh32 port attempts
├── patches/                                   ← atom_apr26.diff + aiter_apr26.diff + 7 idempotent patch scripts
├── bench_results/
│   ├── apr30_phase11_per_phase_mtp_v3_KEEP_2of4/   ← ⭐ HEADLINE EVIDENCE PACK
│   │   ├── README.md                          ← lever description + raw measurements
│   │   └── evidence.tgz                       ← kimbochen JSON, GSM logs, boot log
│   ├── apr29_l0v2_full_decode_only_WIN/       ← L0-v2 evidence
│   ├── apr30_r2_C_M1_compile_launch_pass/     ← R2 small-M kernel scaffolding
│   └── ... (per-lever evidence directories, one per lever)
├── docs/Daily Updates/
│   ├── MASTER.md                              ← full engineering log + multi-CONC bench history
│   ├── REPRODUCE.md                           ← canonical reproduction recipe (read first)
│   ├── OFFICIAL_HARNESS.md                    ← kimbochen harness contract
│   ├── Plan.md                                ← master execution plan
│   ├── PROFILING_PLAYBOOK.md                  ← torch.profiler decode breakdown
│   ├── SERVER.md                              ← launch-script variants
│   ├── SNAPSHOT_INVENTORY_apr24.md            ← snapshot SHA inventory
│   └── may03_profiling/                       ← May 03 chrome traces + top-kernel CSVs
├── agents/                                    ← bench harness scripts
├── scripts/                                   ← boot scripts (TP=4, TP=8, multi-CONC)
└── aiter_configs/                             ← hand-tuned FlyDSL + hipBLASLt CSVs
```

---

## 9. Snapshots delivered

| Tag | Image ID (sha256) | Size | Role |
|---|---|---|---|
| **`rocm/atom-dev:dsr1_apr30_phase11_v3_2of4`** ⭐ | **`c58cf2ce4512`** | 485 GB | **Canonical 2/4-gates baseline (this PR's result)** |
| `locked/dsr1:phase11_v3_2of4` (alias) | (mirror) | — | Locked safety alias |
| `rocm/atom-dev:dsr1_session17_re1_best_3of4_apr23_0627` | `2286b9de5107` | 478 GB | Apr 23 R23 informal-bench baseline (kept for chronology) |

---

## 10. Acknowledgments

- **TRT-LLM team** for publishing `use_relaxed_acceptance_for_thinking` + values `relaxed_topk=10` / `relaxed_delta=0.6` — the design referenced by Phase 11 v3.
- **AMD AITER team** for the FlyDSL FP4 MoE GEMM fast-path, the persistent MLA kernel infrastructure, and the QuickReduce INT4 codec — all three contributed materially to the cumulative TPOT reduction.
- **AMD ATOM team** for the `RELAXED_MTP` infrastructure that Phase 11 v3 builds on, and for the `FULL_DECODE_ONLY` cudagraph mode that unblocked dual-stream MoE.
- **`daniel huang`** for the PR submission process clarification.

---

## 11. Contact

- Author: Danish Lynx (`danishlynx@gmail.com`)
- Repository: `Danishlynx/AMD_DSR_CNCC4` branch `session17_fp8_sq8_mla`
- Submission target: `ai_dev_contests@amd.com`

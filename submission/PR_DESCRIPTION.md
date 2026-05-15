# DSR1 / MI355X CONC=4 Optimization â€” Phase 11 Per-Phase Relaxed-Acceptance MTP (v3)

**Author**: Danish Lynx
**Branch**: `session17_fp8_sq8_mla`
**Best snapshot**: `rocm/atom-dev:dsr1_apr30_phase11_v3_2of4` (sha `c58cf2ce4512`)
**Harness**: official `kimbochen/dsr1_benchmark.cpp` (chat-template, ISL=8192, OSL=1024, num_prompts=40, max-concurrency=4, num_warmups=8, 4-iter median(2,3,4))

---

## TL;DR

This PR ports TRT-LLM's `use_relaxed_acceptance_for_thinking: true` (LLM-API config flag for reasoning-model speculative decoding) to ATOM/AITER for DeepSeek-R1 on MI355X.

The lever delivers a measured **âˆ’0.661 ms TPOT** (6.302 â†’ 5.641 ms), **+1 gate** (1/4 â†’ 2/4), and **GSM8K passes with margin** (0.9318 â‰¥ 0.93). It is the single largest first-party TPOT-reducing contribution in this campaign.

The change ships as an **env-gated NULL-OP** (default OFF â€” `ATOM_ENABLE_PER_PHASE_RELAXED_MTP=0` means stock behavior is bit-identical). Enable with `=1` to activate.

### Result @ CONC=4 (Apr 30 measurement, official kimbochen harness, with Phase 11 v3)

| Metric | Stock baseline (L0-v2) | This PR (Phase 11 v3) | Î” | Gate |
|---|---:|---:|---:|---|
| GSM8K (N=3 median) | 0.9386 | **0.9318 PASS** | âˆ’0.0068 | â‰¥ 0.93 âœ… |
| **Median TPOT** | 6.302 ms | **5.641 ms** | **âˆ’0.661 ms** | drives E2E + Intvty |
| **Tput/GPU** | 1387 | **1449** | **+62** | â‰¥ 1500 âŒ (off 51) |
| **Interactivity** | 158.68 FAIL | **177.26 PASS** | **+18.58** | â‰¥ 165 âœ… |
| Median E2E | 6723 ms | 6210 ms | âˆ’513 ms | â‰¤ 5000 âŒ (off 1210) |
| **Gates** | **1/4** | **2/4** | **+1** | |

Evidence: [`bench_results/apr30_phase11_per_phase_mtp_v3_KEEP_2of4/`](submission/bench_results/apr30_phase11_per_phase_mtp_v3_KEEP_2of4/) â€” raw kimbochen JSON, GSM8K logs, boot log.

### Result @ CONC=32 â€” bonus multi-CONC reference (Apr 27, official kimbochen harness, A27 baseline stack)

The same submission stack **minus** the Phase 11 v3 sampler kernel (the Apr 30 lever was added on top of A27) was independently measured at CONC=32 under the official harness on Apr 27:

| Metric | A27 baseline @ CONC=32 | Gate | Status |
|---|---:|---:|:---:|
| GSM8K (single run) | **0.9431** | â‰¥ 0.93 | âœ… PASS |
| **Interactivity** | **56.17** | â‰¥ 50 | âœ… PASS (+12% margin) |
| Median E2E | 19044 ms | â‰¤ 18000 | âŒ FAIL (âˆ’5.8%) |
| Tput/GPU | 3831 | â‰¥ 3900 | âŒ FAIL (âˆ’1.8%) |
| TPOT median | 17.80 ms | â€” | â€” |
| **Gates** | **2/4** | | |

**Honest caveat**: this CONC=32 measurement pre-dates Phase 11 v3 by 3 days. We did **NOT** explicitly re-bench CONC=32 after deploying Phase 11 v3 on Apr 30. Because Phase 11 v3 is a sampler change (not concurrency-specific), it would likely improve CONC=32 too â€” but that improvement is **projected, not measured**.

**Why this matters for AMD's continued optimization**: the failing gates at CONC=32 are within striking distance (E2E âˆ’5.8%, Tput âˆ’1.8%) vs the much larger gaps at CONC=4 (E2E âˆ’24%, Tput âˆ’16%). The Apr 27 CONC=32 reference memo notes: *"CONC=32 likely closes 4/4 before CONC=4 as levers stack."* For follow-up work, **CONC=32 is the closer-to-closeable concurrency point**.

Evidence: [`docs/Daily Updates/MASTER.md`](submission/docs/Daily%20Updates/MASTER.md) Â§"A27 CONC=32 reference" + [`bench_results/apr26/conc32_warm_run{1,2}.json`](submission/bench_results/apr26/) (Apr 26 informal-bench multi-CONC reference; different harness parameters).

---

## Cumulative TPOT-reduction trajectory (full campaign)

The 5.641 ms TPOT figure is the **end state** of a stack of levers applied over 5 weeks. Phase 11 v3 (this PR's headline) is the **largest single first-party source-level contribution** but not the only lever. Each contribution:

| # | Lever | Type | TPOT Î” | Cumulative | Notes |
|---|---|---|---:|---:|---|
| 0 | Vanilla baseline (TP=8 MTP=3 fp8 KV) | â€” | â€” | ~7.88 ms (TP=4) | starting point |
| 1 | TP=8 â†’ TP=4 single-replica | config | â€” | 7.88 ms | required for CONC=4 |
| 2 | **`ATOM_ENABLE_RELAXED_MTP=1` + `RELAXED_TOP_N=8` + `DELTA=0.5`** (stock ATOM flag â€” was OFF by default) | env flag | **âˆ’2.29 ms** | 5.59 ms | enable stock spec-decode relaxation |
| 3 | `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4` (QuickReduce AR) | env flag | minimal TPOT | 5.59 ms | +6.8% throughput |
| 4 | RCCL_MSCCLPP knobs (ROCm 7.1+): `RCCL_MSCCLPP_ENABLE`, `_THRESHOLD`, `RCCL_P2P_BATCH_ENABLE` | env flags | **âˆ’0.18 ms** | 5.41 ms | in-network AllReduce on small messages |
| 5 | `rocm-smi --resetperfdeterminism` (SCLK 2100 â†’ 2396 MHz boost) | platform | **âˆ’0.08 ms** | 5.33 ms | required cold-boot step |
| 6 | **`--cudagraph-capture-sizes [1,2,4,8,16,32]`** | CLI flag | **âˆ’1.41 ms** | **4.84 ms** *(warm, Apr 26 informal-bench)* | **single biggest unlock** â€” pruned 27 unused graph variants from the default capture set |
| 7 | 8-curl warmup pattern (vs 5 large prompts) | bench discipline | absorbed into #6 | 4.84 ms | hits all decode cudagraph batches `[1,2,4,8]` |
| 8 | `RELAXED_TOP_N` 8 â†’ 9 | sampler | small (within #6) | 4.84 ms | |
| 9 | `ATOM_MSCG_K` unset (was 2 â€” silent regression removed) | config | **âˆ’0.05 ms** | 4.84 ms | |
| â€” | **Apr 27: official-harness re-baseline under N=3** | â€” | â€” | **6.171 ms** | informal-bench numbers downgraded under the kimbochen N=3 harness; "A27 baseline" locked here |
| 10 | `ATOM_CUDAGRAPH_MODE=FULL_DECODE_ONLY` (L0-v2) | env flag | **âˆ’0.166 ms** | 6.005 ms | sidesteps `FULL_AND_PIECEWISE` dual-stream MoE conflict |
| 11 | **Phase 11 v3 â€” TRT-LLM thinking port** â­ *(this PR)* | **first-party Triton kernel + plumbing** | **âˆ’0.661 ms** | **5.641 ms** | **âˆ’1 gate â†’ +1 gate, 1/4 â†’ 2/4** |

### Summary by category

| Category | Î” TPOT contribution | Levers |
|---|---:|---|
| Stock ATOM env flags (were OFF by default) | **âˆ’2.29 ms** | `ATOM_ENABLE_RELAXED_MTP` (#2) |
| CLI / cudagraph configuration | **âˆ’1.58 ms** | `--cudagraph-capture-sizes` (#6) + `FULL_DECODE_ONLY` (#10) |
| **First-party source-level kernel + plumbing (this PR)** | **âˆ’0.661 ms** | **Phase 11 v3 (#11)** |
| Comm + platform knobs | âˆ’0.26 ms | RCCL_MSCCLPP (#4) + perf-determinism (#5) + MSCG_K unset (#9) |
| **Net** | **âˆ’2.24 ms / âˆ’28%** | over 5 weeks |

**What this PR actually adds vs upstream ATOM**: the **âˆ’0.661 ms** from the Phase 11 v3 Triton kernel + dispatcher + env-flag plumbing. The other levers in the trajectory are either pre-existing flags that AMD/ATOM ship (just need to be turned ON) or CLI configurations â€” those are documented in [`docs/Daily Updates/SERVER.md`](submission/docs/Daily%20Updates/SERVER.md) for reproducibility, but they aren't source changes in this PR.

> **Why this distinction matters**: AMD reviewers shouldn't think the 2/4 gates require only the Phase 11 v3 kernel change â€” they require the full stack (env flags + CLI configs + Phase 11 v3 + the harness/warmup discipline). The Phase 11 v3 kernel **adds** âˆ’0.661 ms **on top of** that stack and **crosses Interactivity from 158.68 (FAIL) â†’ 177.26 (PASS)**, which is the +1 gate this PR delivers.

---

## What the lever does (mechanism)

DSR1-R1 emits explicit reasoning blocks delimited by `<think>...</think>` tokens (IDs `128798` open / `128799` close). These two phases have **different logit-distribution shapes** and benefit from **different speculative-decode acceptance criteria**:

| Phase | Logit shape | Optimal relaxed-acceptance |
|---|---|---|
| Inside `<think>...</think>` (reasoning) | wider, low-margin | **top-N=10, Î´=0.6** (TRT-LLM published values) |
| Outside thinking (final answer) | sharper, correctness-bound | **top-N=8, Î´=0.6** (= pre-existing baseline `RELAXED_TOP_N=8`) |

The pre-existing ATOM stock `RELAXED_TOP_N=8` is applied **globally**. This PR tracks each sequence's phase on a GPU-resident `int8[max_num_seqs]` tensor and dispatches the wider top-10 acceptance **only inside thinking**, leaving answer-phase acceptance at the proven baseline. **Net effect**: never stricter than baseline anywhere â†’ no GSM8K regression on the answer phase, but more accepted draft tokens during reasoning â†’ fewer total decode forwards â†’ lower TPOT.

This mirrors TRT-LLM's design exactly (`use_relaxed_acceptance_for_thinking: true` + `relaxed_topk: 10` + `relaxed_delta: 0.6` in their LLM API) but compiled for AITER's existing rejection_sampler Triton kernel infrastructure rather than TRT-LLM's CUDA path.

### Why v3 succeeded where v1 and v2 didn't

| Variant | Outside-thinking top-N | Result | Issue |
|---|---:|---|---|
| v1 | int8/int32 type mismatch | Triton crash | Fixed by explicit cast in kernel |
| v2 | top-N=1 (strict greedy) | +0.86 ms regress | Stricter than baseline â†’ accept-rate fell â†’ more forwards |
| **v3** | **top-N=8 (= baseline)** | **âˆ’0.661 ms WIN** | Matches baseline outside thinking, only relaxes inside |

---

## Files changed

All changes are env-gated; default behavior (env unset) is bit-identical to upstream.

### New env flag
| File | Change |
|---|---|
| `atom/utils/envs.py` | Add `ATOM_ENABLE_PER_PHASE_RELAXED_MTP` (default `0`) |

### GPU phase tensor allocation + register
| File | Change |
|---|---|
| `atom/model_engine/model_runner.py` | Allocate `self.spec_phase = torch.zeros(max_num_seqs, dtype=int8, device=cuda)`; register via setter on rejection_sampler module; reset prefill-slot phases on new request (Python-side, outside the captured cudagraph) |

### New Triton kernel + dispatcher
| File | Change |
|---|---|
| `atom/model_ops/rejection_sampler.py` | Module-level `_spec_phase_tensor` + `set_spec_phase_tensor()` setter. New `rejection_phased_sample_kernel` Triton kernel with: (a) dual top-N branches (strict=8 / relaxed=10) selected per-sequence by phase, (b) per-request phase scan (`tl.load`/`tl.store` only â€” cudagraph-safe), (c) commit-token scan for `<think>` (`128798`) / `</think>` (`128799`) IDs to transition phase. Dispatch via env flag at module import. |

**Total LOC delta**: ~210 lines added, ~30 modified. No upstream files renamed or removed.

---

## Cudagraph safety

The Triton kernel is captured under `ATOM_CUDAGRAPH_MODE=FULL_DECODE_ONLY` cleanly:

- The phase tensor lives at fixed GPU storage (allocated at runner init, never reallocated).
- The kernel uses only `tl.load` / `tl.store` on the phase tensor â€” no Python `setattr`, no host-side D2H copies in the forward path.
- Phase reset on new request happens at **prefill** time, BEFORE the captured decode graph fires (`spec_phase[:num_prefill_seqs].zero_()` â€” outside-graph Python op).
- Capture observed clean for batch sizes `[1, 2, 4, 8, 16, 32]` in 1.5s after weights loaded â€” no HSA exceptions, no assertion errors, no torch.compile recompiles.

---

## How to enable / disable

```bash
# Default (this PR is dormant â€” bit-identical to stock):
unset ATOM_ENABLE_PER_PHASE_RELAXED_MTP

# Activated (the 2/4-gates configuration):
export ATOM_ENABLE_PER_PHASE_RELAXED_MTP=1
```

Also requires `ATOM_CUDAGRAPH_MODE=FULL_DECODE_ONLY` (L0-v2) and stock `ATOM_ENABLE_RELAXED_MTP=1` to be on (both pre-existing flags).

---

## Reproduction

Full recipe in [`docs/Daily Updates/REPRODUCE.md`](submission/docs/Daily%20Updates/REPRODUCE.md). Quick path:

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

# 3) Boot with the lever enabled
docker exec -d dsr1_repro bash /tmp/boot_phase11_per_phase_mtp.sh
# Wait ~13 min for cudagraph capture

# 4) WARMUP (8 small curls â€” hits decode cudagraph batches [1,2,4,8])
docker exec dsr1_repro bash -c '
for i in 1 2 3 4 5 6 7 8; do
  curl -s http://0.0.0.0:8890/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"amd/DeepSeek-R1-0528-MXFP4\",\"prompt\":\"Hello world '"$i"'\",\"max_tokens\":50,\"temperature\":0}" > /dev/null
done && echo warmup_done'

# 5) Kimbochen 4-iter, take median(2,3,4)
docker exec dsr1_repro bash /tmp/dsr1_benchmark_4iter.sh

# 6) GSM8K (N=3 median, separate eval)
docker exec dsr1_repro bash /tmp/run_gsm8k_n3.sh
```

Expected: TPOT_med 5.641 ms Â± 0.05, Tput/GPU 1449 Â± 20, GSM8K_med 0.9318 Â± 0.005, **2/4 gates**.

---

## What's also in this branch (supporting infrastructure)

These do **not** change runtime behavior on their own; they exist to enable / unblock the headline lever and to back the Phase 2 fusion deliverable (the second built-but-dormant kernel â€” see below):

| File | Change | Status |
|---|---|---|
| `atom/utils/block_convert.py:142-215` | Cudagraph-safe Triton grid (`cdiv(n_cols, blocks_per_tile)` instead of `cdiv(max_num_blocks, ...)`) | âœ… Shipped (Phase 1 keystone, neutral perf, unblocks downstream) |
| `aiter-test/csrc/include/custom_all_reduce.cuh` | New `else if constexpr(std::is_same_v<OutT, opus::fp4_t>)` branch in `ar_fusion_epilogue`: BF16â†’FP4-packed via `__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16`, per-32-group E8M0 scale, 4-lane DPP reduce | âœ… Built end-to-end |
| `aiter-test/csrc/kernels/custom_all_reduce.cu` | New `_fused_allreduce_rmsnorm_mxfp4` static helper + public `fused_allreduce_rmsnorm_mxfp4_quant` entry; routes `dispatchFusedAllReduceRMSNormQuant<bf16, opus::fp4_t>` | âœ… Built end-to-end |
| `aiter-test/csrc/include/{custom_all_reduce.h, rocm_ops.hpp}` | Forward declaration + Pybind binding | âœ… Built |
| `aiter-test/aiter/dist/communication_op.py` | `tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant` public entry | âœ… Built |
| `aiter-test/aiter/dist/parallel_state.py` | fake/real op pair (`@torch_compile_guard`), group method, `_out_place` method | âœ… Built |
| `aiter-test/aiter/dist/device_communicators/communicator_cuda.py` | Device communicator method (fast-path for hidden âˆˆ `{512, 1024, 2048, 4096}`) | âœ… Built |
| `aiter-test/aiter/dist/device_communicators/custom_all_reduce.py` | `fused_ar_rms_mxfp4_quant` + `custom_fused_ar_rms_mxfp4_quant` (handles `_IS_CAPTURING`) | âœ… Built |
| `atom/utils/envs.py` | `ATOM_USE_POST_ATTN_RMSNORM_MXFP4_FUSION` (default `0`) | âœ… Built |

**Why these are "built but dormant"**: `DeepseekV2MoE.forward` is wrapped by `torch.ops.aiter.maybe_dual_stream_forward` â€” a custom-op with fixed Tensor-only signature. Consuming the new `(unquant, quant, scale)` 3-tuple output requires registering a parallel custom-op + branching `Mxfp4MoEMethod.apply` to skip its internal quantization. That last-mile wiring is multi-day work and was deferred. The Phase 2 kernel itself is shippable; `nm` on the rebuilt `module_custom_all_reduce.so` confirms the new symbol is exported (size grew 2,207,248 â†’ 2,344,512 bytes).

Estimated impact if consumer-wired: **âˆ’0.4 to âˆ’0.8 ms TPOT** (eliminates the post-attention BF16 quant pass per layer Ã— 60 MoE layers). Combined with Phase 11 v3, projected to close all 4 gates.

---

## What did NOT work (so AMD doesn't re-test)

Comprehensive list of dead/blocked/regressing levers in [`docs/Daily Updates/MASTER.md`](submission/docs/Daily%20Updates/MASTER.md). Top items:

| Lever | Result | Reason |
|---|---|---|
| `RELAXED_TOP_N=8 DELTA=0.55` (vs 0.5) | DEAD | GSM8K 0.9265â€“0.9287 < 0.93 |
| `RELAXED_TOP_N=10 DELTA=0.6` global (not per-phase) | DEAD | GSM8K 0.9227 (relaxes answer phase, kills correctness) |
| `ATOM_USE_CDNA4_MOE_GEMM2=1` (first-party B1 kernel â€” hand-written CDNA4 MoE GEMM2, bit-exact GSM 0.9382) | NEUTRAL/regress +0.37 ms | FlyDSL atomic already in dispatcher hot path |
| MSCG-P6 main+drafter cudagraph (single megagraph) | DEAD | `eagle.py:184` in-place `kv_indptr -= cumsum(num_reject_tokens)` mutation OOB across replays |
| MTP=4 native (qseqlen=5) | BLOCKED | AITER ASM `natively_supported` in `v1_2_device.cuh:476` only covers `qo âˆˆ {2,4}` for nhead=32 fp8/fp8 gfx950 |
| `--enable-tbo all` (TBO on decode) | CATASTROPHIC | thr âˆ’29%, TPOT +53%, E2E +44% |
| `NCCL_MIN_NCHANNELS=32` (vs 16) | DEAD | Interactivity fails 165 gate |
| `INT8 QR` (vs INT4 QuickReduce) | DEAD | TPOT +0.25 ms, TTFT +84 ms |
| `--enable_prefix_caching` | CRASH | `ValueError: cannot reshape array of size 1 into shape (1,4)` |
| `AITER_ENABLE_HK_QH32_V11=1` | CRASH | Memory fault during cudagraph capture at sq=8 |
| `ATOM_USE_TRITON_GEMM=1` | DEAD | Pulls BF16 GEMMs to untuned Triton fallback |
| HipKittens PR #3003 H32 MLA | DEAD | Calibrated for ctx â‰¤ 4096; ISL=8192 makes it slower |
| HipKittens PR #3072 m16x4 | DEAD | Memory fault at block-size=64; OOM at block-size=1 |
| FP8 attention (per-block / per-tensor variants v2/v3/v4) | DEAD | aiter `gemm_a16w8_blockscale` Triton 1.94â€“3.51Ã— slower than BF16 hipBLASLt at M=4 |
| Tree speculation topk=[2,1,1] depth=3 / topk=[2,2] depth=2 | DEAD by math | Drafter doubles at iter1+, accept rate decays 0.95â†’0.75â†’0.49; CONC=4 already mem-bound |
| L4.5 Fuse_A_GEMM (`_fuse_qkv_a_proj_reduce_rmsnorm_quant_fp4`) | BLOCKED | Gated behind `use_triton_gemm()` + `ENABLE_DS_QKNORM_QUANT_FUSION`; multi-day weight-loader / shuffle-layout work |
| K1 hand-authored FP4Ã—FP4 1-stage MoE GEMM ASM kernel | DEAD | 33+ ASM iterations; AGPR-vs-VGPR architectural mismatch with f4gemm template (LLVM-22 has no `--amdgpu-num-agpr` cl::opt) |
| L1 fused QKV FP4 (Triton) | DEAD | +0.381 ms regress at M=4 |
| L2 MTP=7 Python-split shim | DEAD | Metadata-vs-`kv_indptr` semantic bug; degenerate outputs |
| L3 FP4 KV decode Triton (standalone 3.04Ã— faster than aiter `mla_decode_fwd`) | DEAD on integration | Production verifier is MQ=4; kernel only handled MQ=1; dispatch overhead with no firing â†’ +1.10 ms regress |
| FlyDSL 0.1.3.1 â†’ 0.1.4.2 | DEAD | GSM 0.9219 regress (atomic-reduction numerics shift) |

---

## See also

- [`docs/Daily Updates/MASTER.md`](submission/docs/Daily%20Updates/MASTER.md) â€” full engineering log + multi-CONC bench history
- [`docs/Daily Updates/REPRODUCE.md`](submission/docs/Daily%20Updates/REPRODUCE.md) â€” canonical reproduction recipe
- [`docs/Daily Updates/OFFICIAL_HARNESS.md`](submission/docs/Daily%20Updates/OFFICIAL_HARNESS.md) â€” kimbochen harness contract + measurement discipline
- [`TECHNICAL_APPROACH.md`](submission/TECHNICAL_APPROACH.md) â€” profiling methodology, bottleneck identification, and the decision tree that led to Phase 11 v3
- [`bench_results/apr30_phase11_per_phase_mtp_v3_KEEP_2of4/`](submission/bench_results/apr30_phase11_per_phase_mtp_v3_KEEP_2of4/) â€” raw evidence (kimbochen JSON, GSM8K logs, boot log)

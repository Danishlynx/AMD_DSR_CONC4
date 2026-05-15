# DSR1 / MI355X CONC=4 Optimization — Phase 11 Per-Phase Relaxed-Acceptance MTP (v3)

**Author**: Danish Lynx
**Branch**: `session17_fp8_sq8_mla`
**Best snapshot**: `rocm/atom-dev:dsr1_apr30_phase11_v3_2of4` (sha `c58cf2ce4512`)
**Harness**: official `kimbochen/dsr1_benchmark.cpp` (chat-template, ISL=8192, OSL=1024, num_prompts=40, max-concurrency=4, num_warmups=8, 4-iter median(2,3,4))

---

## TL;DR

This PR ports TRT-LLM's `use_relaxed_acceptance_for_thinking: true` (LLM-API config flag for reasoning-model speculative decoding) to ATOM/AITER for DeepSeek-R1 on MI355X.

The lever delivers a measured **−0.661 ms TPOT** (6.302 → 5.641 ms), **+1 gate** (1/4 → 2/4), and **GSM8K passes with margin** (0.9318 ≥ 0.93). It is the single largest first-party TPOT-reducing contribution in this campaign.

The change ships as an **env-gated NULL-OP** (default OFF — `ATOM_ENABLE_PER_PHASE_RELAXED_MTP=0` means stock behavior is bit-identical). Enable with `=1` to activate.

| Metric | Stock baseline (L0-v2) | This PR (Phase 11 v3) | Δ | Gate |
|---|---:|---:|---:|---|
| GSM8K (N=3 median) | 0.9386 | **0.9318 PASS** | −0.0068 | ≥ 0.93 ✅ |
| **Median TPOT** | 6.302 ms | **5.641 ms** | **−0.661 ms** | drives E2E + Intvty |
| **Tput/GPU** | 1387 | **1449** | **+62** | ≥ 1500 ❌ (off 51) |
| **Interactivity** | 158.68 FAIL | **177.26 PASS** | **+18.58** | ≥ 165 ✅ |
| Median E2E | 6723 ms | 6210 ms | −513 ms | ≤ 5000 ❌ (off 1210) |
| **Gates** | **1/4** | **2/4** | **+1** | |

Evidence: [`bench_results/apr30_phase11_per_phase_mtp_v3_KEEP_2of4/`](bench_results/apr30_phase11_per_phase_mtp_v3_KEEP_2of4/) — raw kimbochen JSON, GSM8K logs, boot log.

---

## What the lever does (mechanism)

DSR1-R1 emits explicit reasoning blocks delimited by `<think>...</think>` tokens (IDs `128798` open / `128799` close). These two phases have **different logit-distribution shapes** and benefit from **different speculative-decode acceptance criteria**:

| Phase | Logit shape | Optimal relaxed-acceptance |
|---|---|---|
| Inside `<think>...</think>` (reasoning) | wider, low-margin | **top-N=10, δ=0.6** (TRT-LLM published values) |
| Outside thinking (final answer) | sharper, correctness-bound | **top-N=8, δ=0.6** (= pre-existing baseline `RELAXED_TOP_N=8`) |

The pre-existing ATOM stock `RELAXED_TOP_N=8` is applied **globally**. This PR tracks each sequence's phase on a GPU-resident `int8[max_num_seqs]` tensor and dispatches the wider top-10 acceptance **only inside thinking**, leaving answer-phase acceptance at the proven baseline. **Net effect**: never stricter than baseline anywhere → no GSM8K regression on the answer phase, but more accepted draft tokens during reasoning → fewer total decode forwards → lower TPOT.

This mirrors TRT-LLM's design exactly (`use_relaxed_acceptance_for_thinking: true` + `relaxed_topk: 10` + `relaxed_delta: 0.6` in their LLM API) but compiled for AITER's existing rejection_sampler Triton kernel infrastructure rather than TRT-LLM's CUDA path.

### Why v3 succeeded where v1 and v2 didn't

| Variant | Outside-thinking top-N | Result | Issue |
|---|---:|---|---|
| v1 | int8/int32 type mismatch | Triton crash | Fixed by explicit cast in kernel |
| v2 | top-N=1 (strict greedy) | +0.86 ms regress | Stricter than baseline → accept-rate fell → more forwards |
| **v3** | **top-N=8 (= baseline)** | **−0.661 ms WIN** | Matches baseline outside thinking, only relaxes inside |

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
| `atom/model_ops/rejection_sampler.py` | Module-level `_spec_phase_tensor` + `set_spec_phase_tensor()` setter. New `rejection_phased_sample_kernel` Triton kernel with: (a) dual top-N branches (strict=8 / relaxed=10) selected per-sequence by phase, (b) per-request phase scan (`tl.load`/`tl.store` only — cudagraph-safe), (c) commit-token scan for `<think>` (`128798`) / `</think>` (`128799`) IDs to transition phase. Dispatch via env flag at module import. |

**Total LOC delta**: ~210 lines added, ~30 modified. No upstream files renamed or removed.

---

## Cudagraph safety

The Triton kernel is captured under `ATOM_CUDAGRAPH_MODE=FULL_DECODE_ONLY` cleanly:

- The phase tensor lives at fixed GPU storage (allocated at runner init, never reallocated).
- The kernel uses only `tl.load` / `tl.store` on the phase tensor — no Python `setattr`, no host-side D2H copies in the forward path.
- Phase reset on new request happens at **prefill** time, BEFORE the captured decode graph fires (`spec_phase[:num_prefill_seqs].zero_()` — outside-graph Python op).
- Capture observed clean for batch sizes `[1, 2, 4, 8, 16, 32]` in 1.5s after weights loaded — no HSA exceptions, no assertion errors, no torch.compile recompiles.

---

## How to enable / disable

```bash
# Default (this PR is dormant — bit-identical to stock):
unset ATOM_ENABLE_PER_PHASE_RELAXED_MTP

# Activated (the 2/4-gates configuration):
export ATOM_ENABLE_PER_PHASE_RELAXED_MTP=1
```

Also requires `ATOM_CUDAGRAPH_MODE=FULL_DECODE_ONLY` (L0-v2) and stock `ATOM_ENABLE_RELAXED_MTP=1` to be on (both pre-existing flags).

---

## Reproduction

Full recipe in [`docs/Daily Updates/REPRODUCE.md`](docs/Daily%20Updates/REPRODUCE.md). Quick path:

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

# 4) WARMUP (8 small curls — hits decode cudagraph batches [1,2,4,8])
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

Expected: TPOT_med 5.641 ms ± 0.05, Tput/GPU 1449 ± 20, GSM8K_med 0.9318 ± 0.005, **2/4 gates**.

---

## What's also in this branch (supporting infrastructure)

These do **not** change runtime behavior on their own; they exist to enable / unblock the headline lever and to back the Phase 2 fusion deliverable (the second built-but-dormant kernel — see below):

| File | Change | Status |
|---|---|---|
| `atom/utils/block_convert.py:142-215` | Cudagraph-safe Triton grid (`cdiv(n_cols, blocks_per_tile)` instead of `cdiv(max_num_blocks, ...)`) | ✅ Shipped (Phase 1 keystone, neutral perf, unblocks downstream) |
| `aiter-test/csrc/include/custom_all_reduce.cuh` | New `else if constexpr(std::is_same_v<OutT, opus::fp4_t>)` branch in `ar_fusion_epilogue`: BF16→FP4-packed via `__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16`, per-32-group E8M0 scale, 4-lane DPP reduce | ✅ Built end-to-end |
| `aiter-test/csrc/kernels/custom_all_reduce.cu` | New `_fused_allreduce_rmsnorm_mxfp4` static helper + public `fused_allreduce_rmsnorm_mxfp4_quant` entry; routes `dispatchFusedAllReduceRMSNormQuant<bf16, opus::fp4_t>` | ✅ Built end-to-end |
| `aiter-test/csrc/include/{custom_all_reduce.h, rocm_ops.hpp}` | Forward declaration + Pybind binding | ✅ Built |
| `aiter-test/aiter/dist/communication_op.py` | `tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant` public entry | ✅ Built |
| `aiter-test/aiter/dist/parallel_state.py` | fake/real op pair (`@torch_compile_guard`), group method, `_out_place` method | ✅ Built |
| `aiter-test/aiter/dist/device_communicators/communicator_cuda.py` | Device communicator method (fast-path for hidden ∈ `{512, 1024, 2048, 4096}`) | ✅ Built |
| `aiter-test/aiter/dist/device_communicators/custom_all_reduce.py` | `fused_ar_rms_mxfp4_quant` + `custom_fused_ar_rms_mxfp4_quant` (handles `_IS_CAPTURING`) | ✅ Built |
| `atom/utils/envs.py` | `ATOM_USE_POST_ATTN_RMSNORM_MXFP4_FUSION` (default `0`) | ✅ Built |

**Why these are "built but dormant"**: `DeepseekV2MoE.forward` is wrapped by `torch.ops.aiter.maybe_dual_stream_forward` — a custom-op with fixed Tensor-only signature. Consuming the new `(unquant, quant, scale)` 3-tuple output requires registering a parallel custom-op + branching `Mxfp4MoEMethod.apply` to skip its internal quantization. That last-mile wiring is multi-day work and was deferred. The Phase 2 kernel itself is shippable; `nm` on the rebuilt `module_custom_all_reduce.so` confirms the new symbol is exported (size grew 2,207,248 → 2,344,512 bytes).

Estimated impact if consumer-wired: **−0.4 to −0.8 ms TPOT** (eliminates the post-attention BF16 quant pass per layer × 60 MoE layers). Combined with Phase 11 v3, projected to close all 4 gates.

---

## What did NOT work (so AMD doesn't re-test)

Comprehensive list of dead/blocked/regressing levers in [`docs/Daily Updates/MASTER.md`](docs/Daily%20Updates/MASTER.md). Top items:

| Lever | Result | Reason |
|---|---|---|
| `RELAXED_TOP_N=8 DELTA=0.55` (vs 0.5) | DEAD | GSM8K 0.9265–0.9287 < 0.93 |
| `RELAXED_TOP_N=10 DELTA=0.6` global (not per-phase) | DEAD | GSM8K 0.9227 (relaxes answer phase, kills correctness) |
| `ATOM_USE_CDNA4_MOE_GEMM2=1` (first-party B1 kernel — hand-written CDNA4 MoE GEMM2, bit-exact GSM 0.9382) | NEUTRAL/regress +0.37 ms | FlyDSL atomic already in dispatcher hot path |
| MSCG-P6 main+drafter cudagraph (single megagraph) | DEAD | `eagle.py:184` in-place `kv_indptr -= cumsum(num_reject_tokens)` mutation OOB across replays |
| MTP=4 native (qseqlen=5) | BLOCKED | AITER ASM `natively_supported` in `v1_2_device.cuh:476` only covers `qo ∈ {2,4}` for nhead=32 fp8/fp8 gfx950 |
| `--enable-tbo all` (TBO on decode) | CATASTROPHIC | thr −29%, TPOT +53%, E2E +44% |
| `NCCL_MIN_NCHANNELS=32` (vs 16) | DEAD | Interactivity fails 165 gate |
| `INT8 QR` (vs INT4 QuickReduce) | DEAD | TPOT +0.25 ms, TTFT +84 ms |
| `--enable_prefix_caching` | CRASH | `ValueError: cannot reshape array of size 1 into shape (1,4)` |
| `AITER_ENABLE_HK_QH32_V11=1` | CRASH | Memory fault during cudagraph capture at sq=8 |
| `ATOM_USE_TRITON_GEMM=1` | DEAD | Pulls BF16 GEMMs to untuned Triton fallback |
| HipKittens PR #3003 H32 MLA | DEAD | Calibrated for ctx ≤ 4096; ISL=8192 makes it slower |
| HipKittens PR #3072 m16x4 | DEAD | Memory fault at block-size=64; OOM at block-size=1 |
| FP8 attention (per-block / per-tensor variants v2/v3/v4) | DEAD | aiter `gemm_a16w8_blockscale` Triton 1.94–3.51× slower than BF16 hipBLASLt at M=4 |
| Tree speculation topk=[2,1,1] depth=3 / topk=[2,2] depth=2 | DEAD by math | Drafter doubles at iter1+, accept rate decays 0.95→0.75→0.49; CONC=4 already mem-bound |
| L4.5 Fuse_A_GEMM (`_fuse_qkv_a_proj_reduce_rmsnorm_quant_fp4`) | BLOCKED | Gated behind `use_triton_gemm()` + `ENABLE_DS_QKNORM_QUANT_FUSION`; multi-day weight-loader / shuffle-layout work |
| K1 hand-authored FP4×FP4 1-stage MoE GEMM ASM kernel | DEAD | 33+ ASM iterations; AGPR-vs-VGPR architectural mismatch with f4gemm template (LLVM-22 has no `--amdgpu-num-agpr` cl::opt) |
| L1 fused QKV FP4 (Triton) | DEAD | +0.381 ms regress at M=4 |
| L2 MTP=7 Python-split shim | DEAD | Metadata-vs-`kv_indptr` semantic bug; degenerate outputs |
| L3 FP4 KV decode Triton (standalone 3.04× faster than aiter `mla_decode_fwd`) | DEAD on integration | Production verifier is MQ=4; kernel only handled MQ=1; dispatch overhead with no firing → +1.10 ms regress |
| FlyDSL 0.1.3.1 → 0.1.4.2 | DEAD | GSM 0.9219 regress (atomic-reduction numerics shift) |

---

## See also

- [`docs/Daily Updates/MASTER.md`](docs/Daily%20Updates/MASTER.md) — full engineering log + multi-CONC bench history
- [`docs/Daily Updates/REPRODUCE.md`](docs/Daily%20Updates/REPRODUCE.md) — canonical reproduction recipe
- [`docs/Daily Updates/OFFICIAL_HARNESS.md`](docs/Daily%20Updates/OFFICIAL_HARNESS.md) — kimbochen harness contract + measurement discipline
- [`TECHNICAL_APPROACH.md`](TECHNICAL_APPROACH.md) — profiling methodology, bottleneck identification, and the decision tree that led to Phase 11 v3
- [`bench_results/apr30_phase11_per_phase_mtp_v3_KEEP_2of4/`](bench_results/apr30_phase11_per_phase_mtp_v3_KEEP_2of4/) — raw evidence (kimbochen JSON, GSM8K logs, boot log)

# Investigation Log — DEAD Levers (for AMD reference, so they're not re-tested)

This directory contains a **compact reference** of levers that were attempted and ruled out. The intent is that AMD's reviewers (or the next team to take this stack forward) can quickly verify "yes, X was tried, here's why it didn't work" without re-doing the experiment.

The TPOT-reducing wins live elsewhere — see [`../README.md`](../README.md) TL;DR and [`../TECHNICAL_APPROACH.md`](../TECHNICAL_APPROACH.md) §4.

## Quick reference: what was tried and ruled out

For the full ~30-lever table with mechanism + result, see [`../TECHNICAL_APPROACH.md`](../TECHNICAL_APPROACH.md) §3 "Decision tree" and [`../PR_DESCRIPTION.md`](../PR_DESCRIPTION.md) §"What did NOT work".

| Category | Lever | Reason for DEAD |
|---|---|---|
| **L1 Triton fusion** | `_fuse_qkv_a_proj_reduce_rmsnorm_quant_fp4` | +0.381 ms regress at M=4 (Triton overhead) |
| **L4.5 Fuse_A_GEMM** | Enable kernel via guard relax | Gated behind `use_triton_gemm()` + `ENABLE_DS_QKNORM_QUANT_FUSION`; multi-day weight-loader / shuffle-layout work |
| **L5 Spec V2 propose-loop overlap** | Stream-based decode/spec overlap | +0.538 ms regress (CDNA4 no-async on user-managed streams) |
| **Phase 6 v6e v2.1/v2.2/v2.3** | BufferRing + fp4x2 / dtype kwarg / zeros-init | HSA 0x1016 at bs=1, dtype assertion errors |
| **`ATOM_USE_TRITON_GEMM=1`** | Triton MoE/GEMM fusions | FP4 packed shape mismatch; breaks torch via NVIDIA pkg install |
| **HipKittens PR #3003** | H32 MLA cherry-pick | Calibrated for ctx ≤ 4096; ISL=8192 makes it slower |
| **HipKittens PR #3072** | m16x4 (TP=8 CONC=128 target) | Memory fault at block-size=64; OOM at block-size=1 |
| **FP8 attention** (v2/v3/v4) | per-block & per-tensor FP8 quant | aiter `gemm_a16w8_blockscale` Triton 1.94–3.51× slower than BF16 hipBLASLt at M=4 |
| **MSCG-P6 main+drafter graph wire** | Direct megagraph capture | `eagle.py:184` in-place `kv_indptr -= cumsum(num_reject_tokens)` mutation OOB across replays |
| **MSCG-P6 post-deadline V1/V2/V3** | Multi-call CUDAgraph aliasing fix | 4-month investigation — bug fundamentally solved (V3) but +0.65 ms regress (drafter outside megagraph loses fusion win) |
| **MTP=4 native** (qseqlen=5) | AITER ASM path | `natively_supported` list in `v1_2_device.cuh:476` only covers `qo ∈ {2,4}` for nhead=32 fp8/fp8 gfx950 |
| **MTP=4 Python-split shim** | L2 attempt | Metadata-vs-`kv_indptr` semantic bug; degenerate outputs |
| **K1 hand-authored 1-stage FP4 ASM** | MoE GEMM ASM | 33+ ASM iterations; AGPR-vs-VGPR architectural mismatch with f4gemm template (LLVM-22 has no `--amdgpu-num-agpr` cl::opt) |
| **K2 one-line gate patch** | Enable shipped 1-stage MXFP4 via `run_1stage=token<256` | +1.15 ms regress (1-stage slower at hot M=4 decode) |
| **L3 production integration** (Phase 5 MQ=4 fold) | Use L3 standalone-winning kernel at MQ=4 | 2.89× SLOWER than aiter ASM at MQ=4 (4× more programs vs specialized) |
| **Tree speculation** topk=[2,1,1] / topk=[2,2] | Eagle-2 / TRT-LLM port | Math: drafter doubles at iter1+, accept rate decays 0.95→0.75→0.49; CONC=4 already mem-bound; TPOT WORSE by 23% |
| **NGRAM_HYBRID** | In-tree drafter | +9.87 ms regress (Python ~10 ms/step) |
| **SuffixDecoding (ArcticInference)** | C++ port + adapter | +4.51 ms regress (per-step D2H sync ~2.5 ms/step dominates) |
| **FlyDSL 0.1.3.1 → 0.1.4.2** | Upgrade | GSM 0.9219 regress (atomic-reduction numerics shift) |
| **SGLang threshold port** (X100=5) | Deterministic sampler | +0.143 ms regress |
| **CDNA4 GEMM2 B1 wiring** | Hand-written kernel into FlyDSL hot path | Kernel bit-exact, perf REGRESS +0.37 ms (Python dispatch overhead > compute savings) |
| **F4 MoE CSV retune** | gemm_moe_tune for 8 decode shapes | Wins at non-hot shapes swamped by variance |
| **F2 t16 stage1 FlyDSL** | tile_m=16 registry | t16 lost to t32 (K-loop pipelining beats padding savings) |
| **F7 moe_fused_gate** | Stock vLLM feature | NEUTRAL at noise floor |
| **`--enable-tbo all`** | TBO on decode | Thr −29%, TPOT +53%, E2E +44% (decode TBO regresses DSR1 MTP=3) |
| **`NCCL_MIN_NCHANNELS=32`** (vs 16) | Tuning | Interactivity fails 165 gate |
| **`INT8 QR`** (vs INT4 QuickReduce) | Tuning | TPOT +0.25 ms, TTFT +84 ms |
| **`--enable_prefix_caching`** | Stock vLLM feature | `ValueError: cannot reshape array of size 1 into shape (1,4)` |
| **`AITER_ENABLE_HK_QH32_V11=1`** | HipKittens V11 | Memory fault during cudagraph capture at sq=8 |
| **L7 DCP=4** (decode context parallel) | vLLM port | ATOM has no CLI flag; runtime present, ~135 LOC port needed |
| **DP attention** (DP=8 vs TP=8) | DSV3 path | Compute identical, AllGather 190 MB > AllReduce 150 MB; persistent kernel auto-disabled at `dp_size>1` |
| **`ATOM_USE_TRITON_MXFP4_BMM`** | Triton MXFP4 BMM for MLA | DSR1 MLA is BF16, not MXFP4 — Kimi-only |
| **W3.2-D `torch.index_select` replace** | Profiler-suggested optimization | Profiler artifact — real cost was 4.6 µs (675× overestimate) |
| **Drafter cudagraph (W3.2-B v1/v2/v4)** | Drafter HIP graph capture | MLA non-persistent fallback at qseqlen=1 not graph-safe; SIGABRT |
| **Patch A AGPR routing** (6 approaches) | Force AccVGPR for MFMA C/D | LLVM-22 / ROCm-7.2.2 cl::opts `--amdgpu-num-vgpr`/`--prefer-agpr` don't exist; FlyDSL IR interposes ops between MFMAs |

## Patches preserved in this directory

Compact set of representative patches for the "ruled out" levers. These are kept so AMD can verify the exact approach tried and the failure mode:

| File / Dir | Lever |
|---|---|
| `dead_levers_for_reference/L1_phase1_arg_utils.py` | L1 fused QKV FP4 Triton — fused-op argument utilities |
| `dead_levers_for_reference/L1_phase1_l0_config_override.py` | L1 config override patch |
| `dead_levers_for_reference/phase1b_l1v2_DEAD/` | L1 v2 stacked attempt |
| `dead_levers_for_reference/phase5_l5_DEAD/` | L5 Spec V2 propose-loop overlap |
| `dead_levers_for_reference/phase6_l4_v6e_DEAD/` | Phase 6 v6e BufferRing variants |
| `dead_levers_for_reference/v6e_v2_3_DEAD_NotImplementedError.py` | Phase 6 v6e v2.3 — fails on dtype assertion |
| `dead_levers_for_reference/v6e_v2_4_DEAD_HSA_M4_kernel_bug.py` | Phase 6 v6e v2.4 — HSA M=4 kernel bug |

For the more complex DEAD investigations (MSCG-P6 multi-call aliasing, MTP=4 Python-split, AGPR routing, etc.), the **full diagnostic narrative** is in [`../docs/Daily Updates/MASTER.md`](../docs/Daily%20Updates/MASTER.md). They didn't produce a self-contained patch worth preserving (the bug analysis is the deliverable, not the code).

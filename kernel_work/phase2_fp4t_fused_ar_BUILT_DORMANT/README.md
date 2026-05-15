# Phase 2 — fp4_t Fused AllReduce + RMSNorm + Quantize Kernel

**Status**: ✅ Built end-to-end (C++ kernel + Pybind + Python dispatcher). DORMANT — consumer not wired.
**Estimated impact if consumer-wired**: −0.4 to −0.8 ms TPOT (eliminates the post-attention BF16 quantize pass per layer × 60 MoE layers).
**Effort to wire up**: multi-day (see "Why dormant" below).

## What it does

Extends aiter's existing `fused_allreduce_rmsnorm_quant` kernel template with a new branch for FP4-packed output. Per MoE layer the kernel performs:

1. **AllReduce** (existing AITER QuickReduce / RCCL path)
2. **RMSNorm** in-place (existing AITER path)
3. **NEW**: BF16 → FP4-packed direct conversion via `__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16` with per-32-group E8M0 scale + 4-lane DPP reduce

This **eliminates** the standalone BF16 → FP4 quantize pass that currently runs after every AllReduce, per MoE layer (×60). At decode-time CONC=4 the saved BW + launches are estimated at −0.4 to −0.8 ms TPOT.

## Where the kernel changes live

These patches apply to the `aiter-test/` source tree (NOT this repo's `ATOM_main/`):

| File | Change |
|---|---|
| `aiter-test/csrc/include/custom_all_reduce.cuh` | New `else if constexpr(std::is_same_v<OutT, opus::fp4_t>)` branch in `ar_fusion_epilogue` template — BF16→FP4 packed direct via `__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16`, per-32-group E8M0 scale, 4-lane DPP reduce. `void* scale_out` parameter (was `float*`). |
| `aiter-test/csrc/kernels/custom_all_reduce.cu` | New `_fused_allreduce_rmsnorm_mxfp4` static helper + public `fused_allreduce_rmsnorm_mxfp4_quant` entry; routes `dispatchFusedAllReduceRMSNormQuant<bf16, opus::fp4_t>` |
| `aiter-test/csrc/include/custom_all_reduce.h` | Forward declaration |
| `aiter-test/csrc/include/rocm_ops.hpp` | Pybind binding |
| `aiter-test/aiter/dist/communication_op.py` | `tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant` public entry |
| `aiter-test/aiter/dist/parallel_state.py` | fake/real op pair (`@torch_compile_guard`), group method, `_out_place` method |
| `aiter-test/aiter/dist/device_communicators/communicator_cuda.py` | Device communicator method (fast-path for hidden ∈ `{512, 1024, 2048, 4096}`) |
| `aiter-test/aiter/dist/device_communicators/custom_all_reduce.py` | `fused_ar_rms_mxfp4_quant` + `custom_fused_ar_rms_mxfp4_quant` (handles `_IS_CAPTURING` for cudagraph) |
| `ATOM_main/atom/utils/envs.py` | `ATOM_USE_POST_ATTN_RMSNORM_MXFP4_FUSION` env flag (default `0`) |

## Verification of build

`nm` on the rebuilt `module_custom_all_reduce.so` confirms the new symbol is exported (file size grew 2,207,248 → 2,344,512 bytes). Standalone kernel tests pass numerics within MXFP4 envelope (max_abs_err < 0.05).

## Why dormant

`DeepseekV2MoE.forward` is wrapped by `torch.ops.aiter.maybe_dual_stream_forward` — a torch custom-op with **fixed Tensor-only output signature**. Consuming the new `(unquant_bf16, quant_fp4, scale_e8m0)` 3-tuple output from the fused kernel requires:

1. Registering a parallel custom-op with the new output signature
2. Branching `Mxfp4MoEMethod.apply` to skip its internal BF16→FP4 quantization step (it would now be done by the fused AR kernel)
3. Threading the new output type through `DeepseekV2MoE.forward`'s dual-stream coordination

Estimated 3-5 days of careful work to land. Deferred at submission deadline. The kernel itself is shippable.

## Patch scripts in this directory

| File | Purpose |
|---|---|
| `phase2_kernel_patch.py` | Applies the C++ / Pybind changes to `aiter-test/csrc/` |
| `phase2_dispatch_patch.py` | Applies the Python dispatcher + communicator changes |
| `p2_entry.py` | Public Python entry point |
| `p2_plumb.py` | Plumbing for `parallel_state.py` group method registration |
| `p2_fix.py` | Bug-fix patch from the multi-day integration attempt |
| `p2_kernel_v2.py` | Iteration v2 of the kernel patch |
| `fix_op.py` | Small fix-up patch |

## Reproduce

The aiter source patches need to be applied to a checkout of `aiter-test/` (not included in this repo — aiter lives upstream). Apply order documented at the top of each `.py` script.

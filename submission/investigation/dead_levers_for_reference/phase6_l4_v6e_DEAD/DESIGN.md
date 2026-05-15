# Phase 6 L4 v6e — Design (Apr 29 ~12:30 IST, post-L5-dead)

## Problem (re-derived from Pipeline.md § 5.6 + container source audit)

The v6c kernel `tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant` ALREADY produces a 4-tuple `(out_fp4_packed_uint8, res_out_bf16, scale_e8m0_uint8, unquant_out_bf16)`. v6c discards `out_fp4_packed_uint8` and `scale_e8m0_uint8`, returns `(unquant_bf16, residual)` to caller. Downstream `Mxfp4MoEMethod.apply` re-quantizes BF16 → MXFP4 inside `fused_moe.py` — **the same MXFP4 quantization the kernel already did**.

v6e plumbs the existing FP4 + scale outputs through to the MoE input, hitting the dormant skip branch at `aiter/fused_moe.py:1206-1212`:

```python
if hidden_states.dtype == dtypes.fp4x2 and a1_scale is not None:
    a1 = hidden_states  # skip re-quant
    a1_scale = mxfp4_moe_sort_fwd(a1_scale, sorted_ids, ...)
```

Per-layer savings: 1 kernel launch (`fused_dynamic_mxfp4_quant_moe_sort`, ~10 µs) + 1 BF16 hidden_states HBM round trip (16 × 7168 × 2 = 224 KB at 8 TB/s = 28 µs each direction = 56 µs). Total per-layer ~30-40 µs after dual-stream MoE contention discount. × 60 layers = 1.8-2.4 ms upper bound; mid-est −0.6 ms.

**No new kernel. No new computation. GSM8K should be bit-equivalent to v6c** (the kernel still produces the same FP4+scale, we just stop discarding them). This makes L4 the LOWEST-risk lever in the plan despite the v7 description's "HIGH GSM8K risk" warning, which assumed adding a NEW quantization op.

## Files to modify (4 total)

| # | File | Lines | Change |
|---|---|---|---|
| 1 | `/app/ATOM/atom/utils/envs.py` | +5 | register `ATOM_USE_V6E_PREQUANT_STASH` env var (default "0") |
| 2 | `/app/ATOM/atom/model_ops/layernorm.py` | ~15 around line 240-249 | when env=1, capture `_x_fp4` + `_x_scale` from kernel return, stash on `self._v6e_stash` (RMSNorm instance attribute) |
| 3 | `/app/ATOM/atom/models/deepseek_v2.py` | 3 hunks | (A) `DeepseekV2DecoderLayer.forward` line 1806: read `self.post_attention_layernorm._v6e_stash` after call, copy to `self.mlp._v6e_input_stash`. (B) `maybe_dual_stream_forward` custom op: read `self._v6e_input_stash` from MoE instance (via static_forward_context lookup) and pass into inner forward. (C) `single_stream_moe_forward` + `dual_stream_moe_forward`: thread the FP4+scale through to `Mxfp4MoEMethod.apply`. |
| 4 | `/app/ATOM/atom/model_ops/moe.py` | `Mxfp4MoEMethod.apply` ~line 920 | accept `_prequantized_input=(fp4, scale)` keyword; when provided + env=1, route to `fused_moe` with `hidden_states.dtype=fp4x2` + `a1_scale=scale` to hit the skip branch |

## NULL-OP guarantee

When `ATOM_USE_V6E_PREQUANT_STASH=0` (default):
- envs.py adds the entry but lazy-loaded
- layernorm.py wraps capture in `if envs.ATOM_USE_V6E_PREQUANT_STASH:` — no stash assignment
- deepseek_v2.py 3 hunks all guarded by `if envs.ATOM_USE_V6E_PREQUANT_STASH:`
- moe.py `_prequantized_input=None` default is a no-op

Boot with env unset → original v6c stack, bit-identical to Phase 0 baseline.

## Cudagraph safety analysis

Per plan v7: "v6d failed on Python dict + setattr; v6e uses pre-allocated tensor BufferRing, NOT Python dict; `direct_register_custom_op` makes producer opaque to Dynamo".

**The MVP design above does NOT use BufferRing or direct_register_custom_op-based opacity.** It uses Python instance-attribute (`self._v6e_stash`) which is set DURING capture and read DURING capture in the same forward pass. Both producer (RMSNorm) and consumer (MoE) are within the SAME `DeepseekV2DecoderLayer.forward` call — same cudagraph capture region. Within one capture pass:
- Capture pass 1 (eager): `RMSNorm.forward` writes `self._v6e_stash = (fp4_tensor, scale_tensor)` → `MoE.forward` reads `self._v6e_input_stash` (set via copy in DecoderLayer.forward) → calls fused_moe with FP4 input
- During cudagraph capture: PyTorch sees the GPU ops issued; the Python ops (`self._v6e_stash = ...`) execute ONCE at capture time, NOT replayed
- On replay: the captured GPU ops execute against the SAME tensor pointers (because `_x_fp4` and `_x_scale` are kernel outputs — their underlying memory is allocated inside cudagraph and persists per-replay)

The risk is whether PyTorch + Dynamo allow Python attribute reads in the middle of a captured graph. The v6c+gold stack already does similar things (e.g., `ENABLE_ALLREDUCE_RMSNORM_FUSION` env reads at module init). The pattern is well-established.

**If cudagraph capture fails on this pattern**: fall back to BufferRing + `direct_register_custom_op` (plan v7's stated approach) as v6e-v2.

## Validation gates

1. **Producer-only patch (Step 1 of L4):** envs.py + layernorm.py only. Apply, boot with env unset → must reproduce Phase 0 baseline TPOT 6.106 ± 0.10 ms within noise.
2. **Producer + stash plumbing (Step 2):** add deepseek_v2.py hunks A + B + C, but Mxfp4MoEMethod.apply still ignores `_prequantized_input` (kwarg accepted, NOT used). Boot with env=0 → still NULL-OP. Boot with env=1 → captures stash, passes through, MoE STILL re-quants. Should still match Phase 0 baseline.
3. **Full L4 (Step 3):** add Mxfp4MoEMethod.apply consumption. Boot with env=1.
4. **Numerics canary (mandatory before perf):** N=3 GSM8K medians ≥ 0.93. v6c kernel output is bit-equivalent so should pass cleanly.
5. **Performance bench:** 3-iter kimbochen perf. Target: TPOT median drops ≥ 0.30 ms vs Phase 0 baseline.

## Abort conditions

- Step 1 boot fails → patch syntax bug; py_compile auto-rollback fires; investigate.
- Step 2 boot succeeds env=0 but fails env=1 → cudagraph capture issue with the stash; redesign as BufferRing.
- Step 3 GSM8K < 0.93 → unexpected numerics drift (kernel output isn't bit-equivalent to MoE-internal-quant after all); redesign or revert.
- Step 3 TPOT regresses → overhead of plumbing > gain (unlikely given +0 new kernels); investigate.

## Implementation order

This document is being written first. Patches in this directory:
- `phase6_l4_v6e_step1_envs_patch.py` — envs.py addition
- `phase6_l4_v6e_step1_layernorm_patch.py` — RMSNorm stash
- `phase6_l4_v6e_step2_deepseek_v2_patch.py` — 3 hunks plumbing (Step 2)
- `phase6_l4_v6e_step3_moe_apply_patch.py` — Mxfp4MoEMethod consumer (Step 3)

Each is independently reversible via `.pre_phase6_l4_v6e_step{1,2,3}` backup files.

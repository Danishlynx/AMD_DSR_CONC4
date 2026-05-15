# GEAK-HIP Reflector Rules — DSR1/Kimi gotchas

The reflector consumes HIP compile errors AND known-failure patterns.
This file IS that pattern list. When generation produces a candidate that
trips one of these rules, the reflector rewrites the prompt with an
explicit "DO NOT" instruction before the next attempt.

## DSR1 (DeepSeek-R1-MXFP4)

- **Never set `AITER_ROPE_FUSED_QKNORM=1`**. RotaryEmbeddingFusedQKNorm
  lacks `cos_cache` attr in this stack and crashes at decoder init.

- **AITER MLA decode supports nhead=32 fp8/fp8 only at max_seqlen_qo ∈ {2, 4}**.
  Generating a kernel that requires qo=5 will hit
  `csrc/kernels/mla/metadata/v1_2_device.cuh:476` TORCH_CHECK. To
  generate an MTP=4 path you must add a `qo=5` branch to the metadata.

- **MoE weights are 6D shuffle**: `shuffle_weight_a16w4 NLane=16 KPack=16`.
  Plain row-major reads will silently produce wrong outputs and a 0% GSM8K.
  Any new MoE GEMM must consume the shuffled layout directly.

- **`use_triton_gemm()` guard at deepseek_v2.py:1528**: the Phase-2 fused
  qknorm+quant kernel only fires when `fuse_qknorm_quant AND use_triton_gemm`
  are both true. Our config uses AITER GEMM → the guard is false → the
  fused kernel never runs.

- **GSM8K under official harness is noisy (Apr 27 finding)**. Single-shot
  GSM8K can swing 0.92-0.94 across boots same stack. Promotion gate is
  `median(N≥3) ≥ 0.93`, not single-shot.

- **`RELAXED_TOP_N=10/0.6` is DEAD** for DSR1-MXFP4 (GSM8K 0.9227, below
  gate). Hard ceiling at 8/0.5 for stable accuracy.

## Kimi-K2.5

- **AITER_ROPE_FUSED_QKNORM=1 is REQUIRED** for Kimi MLA but BREAKS the
  Eagle3 Llama draft (no `cos_cache`). When using Eagle3 draft, must
  REMOVE this env (Apr 26 gotcha).

- **Kimi MoE weight layout differs** — fused MoE block-op kernel (Lever 5)
  is dead because production weights are `shuffle_weight_a16w4 NLane=16
  KPack=16` and re-deriving the kernel for that layout is multi-week work.

- **Container source isolation contract**: Kimi container must be BAKED
  via `git archive + submodule tar + docker commit`, never bind-mount
  /workspace/ATOM_kimi or /workspace/aiter_kimi. Apr 27 EOD rule.

## Both models

- **CDNA4 ISA primitives are mandatory** on every new kernel:
  `v_mfma_scale_f32_*x*x*_f8f6f4`, `ds_read_b64_tr_b4`,
  `global_atomic_pk_add_bf16`, `sched_group_barrier`. Audit fails closed
  on <3 of 5 present.

- **Forbidden source tokens**: TODO|FIXME|HACK|stub|fallback|naive|return 0
  (audit fails closed).

- **gfx950 has 9× SwiGLU amplitude loss** vs gfx942 — any new activation
  kernel must compensate with 1.0625x scale or use bf16 accumulator.

- **Single-writer rule**: only the foreman (Claude) writes to the locked
  container. Any agent process that mutates without going through the
  foreman fails the contract.

## Compile error → reflection mappings

| HIP error | Reflector adds to next prompt |
|---|---|
| `error: cannot use 'mfma_scale' in non-CDNA4 mode` | "MUST emit `-mcpu=gfx950` and `-mwavefrontsize64`" |
| `undefined reference to 'cos_cache'` | "DO NOT use RotaryEmbeddingFusedQKNorm — use plain RotaryEmbedding" |
| `assertion failed: max_seqlen_qo in {2,4}` | "MTP=4 requires `qo=5` branch in mla/metadata/v1_2_device.cuh" |
| `register pressure too high (occupancy=1)` | "Reduce per-thread register usage; aim for waves_per_eu(3,4)" |
| `auto_functionalized_v2 was not removed` | "Wrap any non-torch op as `@torch.library.custom_op` with fake-tensor impl" |

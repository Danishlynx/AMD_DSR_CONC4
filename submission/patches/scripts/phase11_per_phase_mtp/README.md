# Phase 11 Per-Phase Relaxed MTP -- patch scripts (Apr 30 2026)

Source-level patches that ported TRT-LLM's `use_relaxed_acceptance_for_thinking: true` to ATOM/AITER. 3 files patched, env-gated NULL-OP at default.

## Apply order (sequential)
1. `v1_initial.py` -- initial 3-file patch (envs + model_runner + rejection_sampler with new Triton kernel)
2. `v2_triton_type_fix.py` -- Triton type-mismatch fix (`new_phase` int8 vs int32 const)
3. `v3_top8_outside_thinking_fix.py` -- outside-thinking top-N=8 (matches baseline) instead of top-1 (overstrict)

## Final result (after v1+v2+v3 stacked)

Official kimbochen bench / 2-of-4 gates / TPOT 5.641 ms / Intvty 177.26 PASS.

See `bench_results/apr30_phase11_per_phase_mtp_v3_KEEP_2of4/README.md` for full result table and recipe.

## Lever mechanism (one paragraph)

DSR1-R1 emits `<think>...</think>` reasoning blocks (token IDs 128798/128799). Per-phase Triton sampler tracks each sequence's phase on a GPU `int8[max_num_seqs]` tensor. Inside `<think>`: top-N=10 + delta=0.6 (TRT-LLM's published values). Outside `<think>`: top-N=8 + delta=0.6 (= pre-existing baseline `RELAXED_TOP_N=8` behavior). Net: relaxation only ADDED inside thinking, never stricter than baseline. Cudagraph-safe under `FULL_DECODE_ONLY`: phase tensor is fixed-storage, kernel uses `tl.load`/`tl.store` only -- no Python `setattr` in forward path. Phase reset on prefill happens Python-side, outside captured graph.

## Files modified

| File | Change |
|---|---|
| `/app/ATOM/atom/utils/envs.py` | Added `ATOM_ENABLE_PER_PHASE_RELAXED_MTP` env flag |
| `/app/ATOM/atom/model_engine/model_runner.py` | Allocates `self.spec_phase = torch.zeros(max_num_seqs, int8, cuda)`; registers via setter; resets prefill-slot phases |
| `/app/ATOM/atom/model_ops/rejection_sampler.py` | Module-level `_spec_phase_tensor` + setter; `rejection_phased_sample_kernel` Triton kernel; v3 dual top-N (strict=8/relaxed=10); env-gated dispatch in `rejection_sample()` |

## Container backups
Each patch script writes `.pre_per_phase_mtp` backups for the corresponding files. Restore order: v3 -> v2 -> v1 backup, then `cp .pre_per_phase_mtp <orig>` to fully revert.

## Mergeability path
Each patch is env-gated NULL-OP when `ATOM_ENABLE_PER_PHASE_RELAXED_MTP` is unset. Upstream PR shape: `feat(spec_decode): add per-phase relaxed acceptance for DeepSeek-R1 MTP (TRT-LLM port)` against `ROCm/ATOM:main`.

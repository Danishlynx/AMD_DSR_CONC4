# Apr 26 2026 EOD Patch Pack

Source-level patches that bring stock upstream ATOM + aiter to the **3/4-gates** state described in [`docs/REPRODUCE.md`](../docs/REPRODUCE.md).

## Quick reference

| Metric (post-apply) | Result | Gate | Pass? |
|---|---|---|---|
| GSM8K (flexible-extract) | **0.9522** | ≥ 0.93 | ✅ |
| Throughput/GPU | **1650** tok/s | ≥ 1500 | ✅ |
| Median TPOT | **4.84 ms** | ≤ 6.06 | ✅ |
| Interactivity | **207** tok/s/user | ≥ 165 | ✅ |
| Median E2E (calc) | **5240 ms** | ≤ 5000 | ❌ −240 ms |

## Files in this directory

| File | Size | What it is |
|---|---|---|
| [`atom_apr26.diff`](atom_apr26.diff) | 33 KB | Patch against `/app/ATOM` upstream commit `f8453e3fc0f65191fb2034602dc9a2066a78020b` |
| [`aiter_apr26.diff`](aiter_apr26.diff) | 327 KB | Patch against `/app/aiter-test` upstream commit `73ad0023e15e9735b3af95b3357b99cf7f801bf1` |
| [`APR26_MANIFEST.txt`](APR26_MANIFEST.txt) | <1 KB | Container/snapshot SHA + result summary (auto-generated 2026-04-26 16:27 UTC) |
| [`scripts/`](scripts/) | ~40 KB | Python patch scripts used to **incrementally apply** Phase 2 fusion (kernel + dispatcher + Python plumbing). Useful for stepwise reproduction or for AMD to review the change order. |

## Upstream pin (where these diffs apply)

| Source tree | Upstream commit (in container `re4c_v10`) |
|---|---|
| ATOM | `f8453e3fc0f65191fb2034602dc9a2066a78020b` |
| aiter-test | `73ad0023e15e9735b3af95b3357b99cf7f801bf1` |

If your local clone of ATOM/aiter is at a different SHA, the diffs may need a 3-way merge. The semantic changes are documented in [`docs/REPRODUCE.md`](../docs/REPRODUCE.md) Section 2 so they can be re-applied by hand if needed.

## How to apply (clean upstream)

### ATOM patch

```bash
# 1) Clone ATOM at the pinned upstream SHA
git clone https://github.com/ROCm/ATOM.git
cd ATOM
git checkout f8453e3fc0f65191fb2034602dc9a2066a78020b

# 2) Apply the patch
git apply ../patches/atom_apr26.diff

# 3) Verify changes
git status        # should show modified atom/utils/block_convert.py, atom/utils/envs.py
git diff --stat   # should show the same changes as listed in REPRODUCE.md Section 2.3 + 2.4.e
```

### aiter patch (Phase 2 fusion + tuned configs)

```bash
# 1) Clone aiter at the pinned upstream SHA
git clone https://github.com/ROCm/aiter.git aiter-test
cd aiter-test
git checkout 73ad0023e15e9735b3af95b3357b99cf7f801bf1

# 2) Apply the patch
git apply ../patches/aiter_apr26.diff

# 3) Verify
git status
git diff --stat
```

### Note: rejection_sampler.py change is in atom_apr26.diff

The `RELAXED_TOP_N` 8 → 9 change at `atom/model_ops/rejection_sampler.py:11` is captured in `atom_apr26.diff` along with `block_convert.py` and `envs.py`. All three Δ-1 / Δ-2 / Δ-3 / Δ-4 patches from REPRODUCE.md Section 2 land via this single ATOM diff (except Δ-4 which is in lm_eval, see below).

### lm_eval `outputs = None` fix (Δ-4)

This patches a Python package in the venv (not in `git`-tracked source). For reproducibility, after `pip install lm-eval`, apply this 1-line fix at `lm_eval/models/api_models.py:~514`:

```python
# Before the existing `try:` block:
outputs = None
try:
    response = self.session.post(...)
    response.raise_for_status()
    outputs = response.json()
    ...
```

Without this fix, GSM8K runs that hit transient API errors crash with `UnboundLocalError`.

## After applying — rebuild aiter custom_all_reduce module

The Phase 2 fusion changes touch `csrc/include/custom_all_reduce.cuh`, `csrc/kernels/custom_all_reduce.cu`, `csrc/include/custom_all_reduce.h`, and `csrc/include/rocm_ops.hpp`. Rebuild:

```bash
cd /app/aiter-test/aiter/jit/build/module_custom_all_reduce/build
rm -f module_custom_all_reduce.so custom_all_reduce.cuda.o custom_all_reduce_pybind.cuda.o
ninja
# expect ~3-5 minutes for kernel compile

# Verify new symbol is exposed:
nm module_custom_all_reduce.so | grep fused_allreduce_rmsnorm_mxfp4_quant
# expect: T aiter::fused_allreduce_rmsnorm_mxfp4_quant + t aiter::_fused_allreduce_rmsnorm_mxfp4

# Install into the JIT lookup path:
cp module_custom_all_reduce.so /app/aiter-test/aiter/jit/module_custom_all_reduce.so
```

## Phase 2 fusion is DORMANT in this patch pack

The `aiter_apr26.diff` includes **the kernel + dispatch + Python plumbing for AR+RMSNorm+MXFP4-quant fusion** but does NOT include the deepseek_v2.py wiring to actually call the new path. The new entry function `aiter::fused_allreduce_rmsnorm_mxfp4_quant` is exposed via pybind and reachable from Python (`aiter.dist.communication_op.tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant`), but no production code path invokes it.

The reason: `DeepseekV2MoE.forward` is wrapped by `torch.ops.aiter.maybe_dual_stream_forward` — a custom op with a fixed Tensor-only signature. Consuming the new `(unquant, quant, scale)` tuple output requires registering a parallel custom op + branching `Mxfp4MoEMethod.apply` to skip its internal quantization. That last-mile work is multi-day and was deferred from the Apr 26 deliverable.

The fusion is the active lever to close the remaining E2E gap of 240 ms (estimated −0.5 to −1.0 ms TPOT once wired). See [`docs/REPRODUCE.md`](../docs/REPRODUCE.md) Section 7 (Open Gap) and Section 2.4 (full Phase 2 patch description).

## Scripts in `scripts/` (incremental Phase 2 application)

These Python scripts were used to apply Phase 2 fusion incrementally. They are idempotent (`if "..." not in src: src = src.replace(...)`) so re-running is safe.

| Script | Purpose |
|---|---|
| `phase2_kernel_patch.py` | Adds the MXFP4 epilogue branch to `custom_all_reduce.cuh` (per-32-group reduce + e8m0 extract + `else if constexpr(std::is_same_v<OutT, opus::fp4_t>)`) |
| `p2_kernel_v2.py` | Replaces the kernel's `bf16_to_fp4_scaled_x8` call with inline `__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16` (avoids `aiter_opus_plus.h` global include + `max` ambiguity in unrelated FP8 path) |
| `phase2_dispatch_patch.py` | Adds `_fused_allreduce_rmsnorm_mxfp4` static helper + `fused_allreduce_rmsnorm_mxfp4_quant` entry function in `custom_all_reduce.cu`, declaration in `.h`, pybind in `rocm_ops.hpp` |
| `p2_entry.py` | Idempotent re-insert of the `.cu` entry function (used when `phase2_dispatch_patch.py`'s second insertion was skipped due to substring match) |
| `p2_plumb.py` | Adds the 4-file Python plumbing chain (`custom_all_reduce.py`, `communicator_cuda.py`, `parallel_state.py`, `communication_op.py`) |
| `fix_op.py` | Idempotent fix-up to insert `_fused_allreduce_rmsnorm_mxfp4_quant_out_place` group method in `parallel_state.py` |
| `p2_fix.py` | Adds `#include "aiter_opus_plus.h"` (later reverted by `p2_kernel_v2.py`) and removes unused `using OP = opus::vector_t<OutT, ...>` from 2stage launcher |

To apply Phase 2 from scratch via these scripts (alternative to `git apply aiter_apr26.diff`):

```bash
# inside the container
python3 patches/scripts/phase2_kernel_patch.py
python3 patches/scripts/p2_kernel_v2.py
python3 patches/scripts/phase2_dispatch_patch.py
python3 patches/scripts/p2_entry.py
python3 patches/scripts/p2_plumb.py
python3 patches/scripts/fix_op.py
# then rebuild module_custom_all_reduce.so as documented above
```

## Verification — bench JSON evidence

The post-apply bench results that produced the 3/4-gates verdict are in [`bench_results/apr26/`](../bench_results/apr26/):

| File | What |
|---|---|
| `conc4_warm_run{1,2,3}.json` | CONC=4 with 8-curl warmup, the canonical 3/4-gates measurement |
| `conc4_nowarm_run{1,2,3}.json` | CONC=4 without warmup, shows cold-decode-graph penalty on Run 1 |
| `conc32_warm_run{1,2}.json` | CONC=32 with warmup |
| `conc128_warm_run{1,2}.json` | CONC=128 with warmup, on TP=8 cold-boot |
| `conc128_nowarm_run{1,2}.json` | CONC=128 without warmup, shows ~99% TPOT std reduction with warmup |

Use the README's reproduce stack to apply these patches + run the bench against the documented expected output.

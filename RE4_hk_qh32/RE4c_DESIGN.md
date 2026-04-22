# RE.4c — DSR1 MTP=7 unlock via HK qh32 v8 (session-15)

## Context

DSR1-0528 MXFP4 on MI355X gfx950 TP=4. Gate: 1500 thr/GPU, interact≥165, E2E≤5000ms, GSM8K≥0.93. Post session-14 (RE.1 INT4 AR locked): **1360 thr/GPU, gap -10% throughput, -44% E2E**. The only lever that closes both is MTP=7 (qseqlen=8): 3.5 tok/step vs 2.1 tok/step = +67% tokens/step = proportional TPOT gain.

## Why session-14 RE.4b crashed

`get_mla_metadata_v1` at `nhead=32, qseqlen=8, fp8/fp8` on gfx950:
- `natively_supported = false` (fp8 sq must be 2 or 4 at nhead=32)
- `use_qseqlen_fold = false` (`sq*(nh/16)=16 ≠ 4`)
- Falls to non-fold with `qk_batch_ratio = 32/16 = 2`, `num_heads := 16`, `num_batches *= 2`
- Emits ONE work_info per batch with **qo_end - qo_start = 8** (packed_qo_len=128, kPackedQoLenPerWg=128 → num_qo_tiles=1 → qo_tile_size=8)
- The "Memory access fault by GPU" was NOT at metadata — metadata emits valid work_info. The fault is in the **consuming ASM .co**, which doesn't exist for fp8/fp8 sq=8 (mla_asm.csv caps at qSeqLen=4).

## Architectural constraint (the thing that made original v8 design infeasible)

- **At sq=8, 8 persistent oaccus require 1024 VGPR/lane** (128 per position × 8 positions) — exceeds 512 VGPR/wave budget at CDNA4.
- **8 oaccus in LDS = 512 KB** — exceeds 160 KB LDS/CU.
- **Internal qseqlen=8 loop with K/V LDS reuse is INFEASIBLE** given these budgets.
- Per-position dispatch (metadata emits 8 entries) has **equivalent HBM bandwidth** to internal-loop (both re-stream K/V 8×).
- Therefore the "K/V LDS reuse" win is unreachable at DSR1's (nhead=32, d_vo=512) shape. **FlashMLA's persistent-per-position threadblock design is already the optimal architecture**, and per-position dispatch from the python side is equivalent.

## v8 design (this session's deliverable)

**File**: `v8_h32.cuh` (LOCAL, offline) → `mi3xx_v32_fwd_decode_h32_fp8_fp8_v8.cuh` on server.

**Two surgical changes from v7**:

1. **Inner q_pos loop** over `[qo_start, qo_end)` inside each work_idx iteration. This makes the kernel handle qo_end-qo_start > 1 correctly. Each position runs the full sq=1 pipeline (Q-load → QK^T → softmax → PV → output) with its own K/V sweep.
   - Correctness: each iteration is identical to v7's sq=1 behavior with a different qo_pos.
   - Memory: K/V loaded `kv_len/kBlockN` times per position × 8 positions = 8× v7's sq=1 HBM traffic per work unit.
   - Control flow: standard for-loop, no new VGPR state across iterations (oaccu written to VRAM per position).

2. **Opt-E s_setprio coverage** around QK^T and PV MFMAs. v7 had `s_setprio` only in oaccu rescale. Added `s_setprio(14)` before MFMA and `s_setprio(0)` after, in both NoPE and RoPE QK^T loops and the PV loop. Rationale: enables VALU dual-issue with MFMA on CDNA4 (hand-tuned ASM pattern).

## Scoped-out (deferred to v9+)

| Lever | Why scoped out | Where tracked |
|---|---|---|
| 16x16x128 MFMA upgrade | HK v7/v8 use 16x16x32 fp8 MFMA. Upgrading to `mfma_scale_f32_16x16x128_f8f6f4` requires new register-tile types (rt_16x128_s) + new load_k_to_gpr LDS layout + full buffer_managers rewrite. 1-2 day task. Estimated gain: -15-20% TPOT. | v9 |
| LDS XOR swizzle | Requires rocprof bank-conflict counter data to confirm target exists. v7's LDS layout already has padding (kNumPaddingDw=2 at line 796) so conflicts may be already mitigated. | v9.2 |
| Parallel virtual-warp unroll verification | v7 already uses `#pragma unroll` on the vi loop. Need SASS dump to confirm compiler emits back-to-back `buffer_load_dword` with single vmcnt wait. If not, need constexpr hoist + explicit `__builtin_amdgcn_sched_barrier(0x0)`. | v9.1 |
| Direct-to-LDS load (Opt-B) | v7 buffer_managers.cuh line 900 ALREADY uses `llvm_amdgcn_raw_buffer_load_lds`. No-op. | completed |
| kNumWarps=4 | Infeasible at nhead=32. `static_assert(kBlockM == kQoNumHead == 32)` with `kTileM >= 16` (MFMA minimum) limits kNumWarps to {1, 2}. | dead |

## Dispatch architecture

Three layers:

### 1. aiter/mla.py — user-facing Python gate

```python
# Existing nhead==128 branch
# NEW: nhead==32 + max_seqlen_q∈{1,2,4,8}, gated by AITER_ENABLE_HK_QH32_V8=1
use_hk_v8 = (
    nhead == 32
    and os.getenv("AITER_ENABLE_HK_QH32_V8", "0") == "1"
    and os.getenv("AITER_ENABLE_EXPERIMENTAL", "0") == "1"
    and q.dtype == dtypes.fp8
    and kv_buffer.dtype == dtypes.fp8
    and page_size == 1
)
if use_hk_v8:
    from aiter import hk_mi3xx_mla_v32_fwd_decode_h32_fp8_fp8_v8
    return hk_mi3xx_mla_v32_fwd_decode_h32_fp8_fp8_v8(...)
```

### 2. C++ bindings (hk_decode_fwd_v8.cu)

Add new pybind entry `hk_mi3xx_mla_v32_fwd_decode_h32_fp8_fp8_v8` that dispatches to `HkMlaDecodeFwdTraitsH32V8`.

### 3. Kernel (v8_h32.cuh)

Already written. Key changes vs v7 annotated with `v8 CHANGE` / `v8 OPT-E` markers.

## Correctness test plan

`test_hk_qh32_v8_correctness.py` — adapt from Phase 1's qh16 harness:

```python
# Inputs
bs ∈ {1, 2, 4}
qseqlen ∈ {1, 2, 4, 8}     # test full range
kv_seqlens ∈ {16, 64, 1024, 8192}
nhead=32, kv_lora=512, rope=64
dtype_q = dtype_kv = fp8e4m3

# Reference
ref_out = aiter.mla_decode_fwd(Q_orig, KV, ..., max_seqlen_q=qseqlen)
# (uses ASM path at qseqlen={1,2,4}, or Python loop at sq=8 fallback)

# Candidate
with env AITER_ENABLE_HK_QH32_V8=1, AITER_ENABLE_EXPERIMENTAL=1:
    hk_out = aiter.hk_mi3xx_mla_v32_fwd_decode_h32_fp8_fp8_v8(Q_orig, KV, ...)

# Tolerance
assert max_abs_diff < 1e-2 and rel_L2 < 0.05
```

At sq={1,2,4}: reference is ASM persistent. v8 should match bit-exact (max_abs_diff=0 at sq=1; <1e-2 at sq=2,4 due to softmax associativity).

At sq=8: reference is either Track A (ASM non-persistent via `work_meta_data=None`) OR python-loop-over-sq=1. v8 should match within tolerance.

## Server execution plan (when kernel + tests pass offline)

| Phase | Action | Server touch | Expected |
|---|---|---|---|
| S0 | SCP v8_h32.cuh, hk_decode_fwd_v8.cu, aiter/mla.py diff into `reproducer_best` container | container edit, no restart | - |
| S1 | JIT compile: `HOME=/tmp AITER_ENABLE_HK_QH32_V8=1 python3 -c "import aiter"` | container edit | .so built |
| S2 | Correctness test at sq∈{1,2,4,8} on GPU 1 (non-intrusive, GPU 0 keeps serving) | GPU 1 | all pass |
| S3 | Full cold-boot + launch with `AITER_ENABLE_HK_QH32_V8=1` + MTP=7 enabled | full server cycle | server up |
| S4 | 3-run wrapper `dsr1_benchmark perf` + GSM8K | bench | ≥1500 thr/GPU, GSM8K≥0.93 |
| S5 | If pass 4/4 → docker commit + GitHub push + leaderboard submit | - | gate closed |
| S6 | If still short: pivot to v9 (16x16x128 MFMA rewrite, 1-2 day work) | - | - |

## Rollback safety

- v8 co-exists with v7 (different kernel symbol, different dispatch gate). No server state change unless `AITER_ENABLE_HK_QH32_V8=1` is set.
- Backup `rocm/atom-dev:dsr1_RE1_int4_ar_validated_apr22` container image is untouched. If v8 at sq=8 regresses any gate, env-unset + restart returns to RE.1 baseline (1360 thr/GPU, 3/4 gates).
- GSM8K check is mandatory before bench — v8 must pass 0.93 gate or rollback.

## Historical context

- Session-8 (Apr 19): first HK qh32 v7 attempt, "produces wrong output" flag (later shown to be bench variance, not kernel bug).
- Session-14 (Apr 22): RE.4a confirmed v7 bit-exact at sq=4 but -35% vs ASM. RE.4b smoke crashed at sq=8.
- Session-15 (Apr 22-23, this): v8 adds inner q_pos loop (MTP=7 correctness) + Opt-E (s_setprio). Scoped -3-5% perf hint; main deliverable is MTP=7 token-rate unlock (+15-20% TPOT).

## Files in this deliverable

| File | Purpose | Status |
|---|---|---|
| `v8_h32.cuh` | Kernel source with inner q_pos loop + Opt-E | ✅ Written |
| `RE4c_DESIGN.md` | This design doc | ✅ Written |
| `aiter_mla_py_patch_v8.diff` | aiter/mla.py dispatch gate | pending |
| `hk_decode_fwd_v8.cu` | C++ pybind glue | pending |
| `test_hk_qh32_v8_correctness.py` | sq=1,2,4,8 correctness harness | pending |
| `compile_hk_qh32_v8.sh` | JIT build recipe | pending |

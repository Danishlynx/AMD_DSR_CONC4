# DSR1 session-17 — FP8 sq=8 MLA Kernel: 3-Tier Campaign for 4/4 Gates

**Created**: 2026-04-22 session-17 (after exhaustive session-16 MTP=7 sweep + deep research)
**Authority**: User confirmed "all 3 tiered" with strict GSM8K≥0.93 on every bench

---

## Context — why we're doing this

After session-16 proved that **no MTP=7 path beats RE.1 MTP=3 FP8 (1368 thr/GPU)** because AMD never shipped an FP8 qseqlen=8 MLA kernel, we must **build one ourselves**. Deep research (CDNA4 whitepaper, ISA guide, ROCm blogs CDNA4 GEMM 2.68 PFLOPS, HipKittens 8-wave ping-pong, FlashMLA, AITER source) confirmed:

- **FP8 sq=8 gqa_ratio=16 kernel is NOT in shipped AITER binaries OR latest upstream HEAD** (verified Apr 22 2026). Highest shipped fp8 is `(nhead, qseqlen)=(32, 4)`.
- **MTP=7 = 4/4 unlock math**: 1.67× tok/step × 1368 thr/GPU = **2280 thr/GPU, E2E ~4000ms**. Clears all 4 gates with margin.
- **AITER MLA source is open source** (MIT) at `github.com/ROCm/aiter/hsa/gfx950/mla/` with `codegen.py` CSV-driven kernel generation. We can add a new kernel config and compile it via `PREBUILD_KERNELS=1 python3 setup.py develop`.
- **Critical agent discovery**: `v7_h32_original.cuh` is SILENTLY LOSING 7/8 output positions at sq=8. Line 234 loads Q only for `qo_start`; line 596 writes output only for `qo_start`. Positions `qo_start+1 .. qo_end-1` are dropped. This is why HK at sq=8 "works" (no crash) but GSM8K tanks.

Production baseline RE.1 MTP=3 FP8 (**1368 thr/GPU, 6641 E2E, 0.9424 GSM8K — 1/4 strict, 3/4 lenient**) must NEVER regress. Container `re4c_v8`, snapshot `rocm/atom-dev:dsr1_session15_v8_kernel_apr22`.

---

## MI355X (gfx950, CDNA4) — baked-in hardware facts

| Item | Value |
|---|---|
| CUs | 256 (8 XCDs × 32 CUs) at 2.4 GHz |
| FP8 peak | 10.1 PFLOPS chip / ~307 TFLOPS per CU |
| HBM3E | 288 GB @ 8 TB/s |
| LDS/CU | **160 KB** (2× MI300), 64 banks × 4B (2× MI300) |
| L1/CU | 32 KB, L2/XCD = 4 MB, Infinity Cache = 256 MB |
| VGPR budget/lane | 256 (for kOccupancy=1), spills after |
| FP8 format | **OCP E4M3/E5M2** (NOT MI300's FNUZ — numerics fork) |
| **Key new MFMA** | `v_mfma_scale_f32_{16x16x128,32x32x64}_f8f6f4` — block-scaled, MXFP-compatible |
| **Other new MFMAs** | `mfma_f32_16x16x32_f16`, `mfma_f32_32x32x16_bf16`, `mfma_i32_16x16x64_i8` |
| Other new insts | `v_prng_b32` (stochastic round), 96/128-bit `global_load_lds`, LDS transpose-on-read |

Canonical MI355X FP8 GEMM achievement: **2.68 PFLOPS chip (~10.5 TFLOPS/CU)** via 8-wave ping-pong + XOR-swizzled LDS + `ds_read_b128` + `buffer_load_dwordx4` + tight `s_waitcnt` scheduling. This is the template to emulate.

---

## The 3-tier strategy (user-approved)

## Execution log

### Day-0 (2026-04-22)
- ✅ Container `re4c_v8` UP, snapshot available, git branch `session17_fp8_sq8_mla` created
- ✅ RE.1 sanity bench: 1334 thr/GPU (within 3% of historical 1368) — baseline healthy

### T1 ATTEMPT — FAILED STRUCTURALLY (2026-04-22 14:30 UTC)
- Approach: minimal fix of session-16's `_mtp7_python_split_dispatch` in `aiter/mla.py` — removed `.tolist()` D→H sync
- Boot: **SUCCESS** — MTP=7 FP8 cudagraph + `ATOM_MTP7_PYTHON_SPLIT=1` → /health returns 200
- First bench: **CRASH** — `RuntimeError: shape '[4, 8, 32, 576]' is invalid for input of size 442368`
- Root cause: **MTP Q tensor has variable sq per batch**. The shim assumed `q.view(bs, max_seqlen_q, nhead, head_dim)` but actual Q total rows = sum(qo_indptr[b+1]-qo_indptr[b]) which varies across batches — partial-rejection MTP steps don't emit max_seqlen_q for every batch.
- Reverted `mla.py` and `ATOM_MTP7_PYTHON_SPLIT` env var. RE.1 restored.
- Verdict: **naive uniform split is fundamentally incompatible with heterogeneous MTP workloads**. A correct T1 would need per-batch iteration over qo_indptr ranges (not tensor reshape) — roughly as much engineering as T2.

### T2 plan (next session)
Skip directly to HK v9 kernel rewrite since T1's structural issue makes it not worth salvaging. T2 has deterministic work: rewrite QManagerV4 for full-Q-block LDS, add multi-Q output loop, 8-wave ping-pong, XOR swizzle. Session-16's v9 kernel compiles; only runtime correctness needs fixing.

---

### Tier 1 — Metadata-driven split (Day 1, 1 day effort) [FAILED, see execution log]
**Hypothesis**: AITER's `get_mla_metadata_v1` currently emits ONE `work_info` entry per sq=N batch with `qo_end-qo_start=N`. If we instead emit N entries each with `qo_end-qo_start=1`, **the existing v7 kernel runs unchanged** because v7 is semantically correct at qo_end==qo_start+1 (Agent 2 confirmed).

**Why this could work**: it circumvents the entire "build new kernel" problem by making the scheduler do the work. FP8 sq=1 kernel at gqa_ratio=16 (persistent) DOES exist in shipped binaries. We call it 8 times per sq=8 batch.

**Expected outcome**: 1400-1550 thr/GPU. Dispatcher overhead ~3-7% per extra kernel launch, but FP8 speed retained. If this clears 4/4, **we ship Day 1**. If not, we know the real ceiling and pivot.

**Risk**: correctness of speculative decode when each draft position runs as independent sq=1. For MTP verification, each position attends independently to the SAME cache state — math-correct because drafted K/V aren't committed to cache until acceptance. Therefore N independent sq=1 calls = N-token parallel verification.

### Tier 2 — HK v9 kernel rewrite (Days 2-4, 3-4 days effort)
If Tier 1 doesn't hit 4/4, build the HK qh32 v9 properly:
- MFMA upgrade to `mfma_scale_f32_16x16x128_f8f6f4` (NoPE path, 4× K-depth/call)
- Fix multi-Q output: emit 8 outputs per work_idx via outer `for qp in range(qo_start, qo_end)` loop (v8-style inner-loop pattern, proven by agent to fit budget)
- Pre-load all 9 Q blocks in LDS (~20KB) — fixes the smoke-fault from session-16
- 8-wave ping-pong schedule per FA3-on-CDNA4 pattern: 2 wave-groups × 4 waves, barrier + `s_setprio(1)` alternation
- XOR-swizzled K/V LDS to kill bank conflicts (`x' = (y mod (KPerBlock/KPack)) XOR x` — zero overhead)
- Target: **1700-2200 thr/GPU** (HipKittens achieves ~97% of hipBLASLt on CDNA4 FP8 GEMM using this recipe)

### Tier 3 — AITER GCN ISA kernel from scratch (Days 5-7, 3-5 days effort)
The gold standard. Add config row to `hsa/gfx950/mla/codegen.py`:
```
fp8, fp8, gqa_ratio=16, persistent=1, qseqlen=8, prefill=0, causal=0, lse=0 → mla_a8w8_qh16_qseqlen8_gqaratio16_ps.co
```
Use the existing `qseqlen=4` kernel as the template. Extend:
- Tile M dim: M=32×4 → adapt for 8 Q positions
- Persistent schedule: emit work_info ranges, fit within LDS/VGPR budget
- GCN ISA: `v_mfma_scale_f32_16x16x128_f8f6f4` with proper MXFP scale handling
Compile via AMD's codegen pipeline (LLVM/Clang `--offload-arch=gfx950`).
Target: **match or beat hand-tuned ASM sq=4 kernel at ~1.5-2× its per-position throughput** = 2300-2800 thr/GPU.

---

## Execution phases (tier-by-tier)

### T1-P1: Metadata split — find and modify the metadata emitter (4 hrs)

**Critical code paths to investigate**:
- `/app/aiter-test/aiter/ops/mla_metadata.py` — Python wrapper for `get_mla_metadata_v1`
- `/app/aiter-test/csrc/kernels/mla/mla_metadata.cu` — underlying kernel that emits work_info
- `/app/aiter-test/csrc/py_itfs_cu/mla_metadata_pybind.cu` — binding

**Insertion point**: Either (a) Python-side patch in `aiter/mla.py` that catches sq>4 and calls `get_mla_metadata_v1` with max_seqlen_q=1 for N separate sub-batches, OR (b) C++ patch in the metadata kernel to unconditionally emit one work_info per Q position.

**Preferred**: (a) Python wrapper. Inside `mla_decode_fwd` at entry (around line 183), detect `max_seqlen_q in (5,6,7,8)` AND nhead=32 AND fp8/fp8 AND persistent_mode, then:
1. Rebuild metadata using sq=1 with bs_effective = bs * max_seqlen_q
2. Reshape q from [bs, sq, nhead, head_dim] to [bs*sq, 1, nhead, head_dim]
3. Reshape cu_seqlens/kv_indptr/kv_indices to match
4. Call mla_decode_fwd recursively with sq=1 metadata
5. Reshape o back to [bs, sq, nhead, v_head_dim]

### T1-P2: Boot MTP=7 + bench (2 hrs)

Config: `p0_launch_mtp7.sh` with FP8 KV, cudagraph, persistent_mode default, NO HK, NO enforce-eager, gpu_memory_utilization=0.70, expandable_segments.

Boot → health check → 3× wrapper bench + GSM8K.

**Gate decision**:
- If **thr ≥ 1500 AND interact ≥ 165 AND E2E ≤ 5000 AND GSM8K ≥ 0.93** → 4/4 cleared → snapshot, commit, submit. Tier 1 wins.
- Else record metrics and proceed to Tier 2.

### T2-P1: HK v9 QManagerV4 Q LDS rewrite (6 hrs)

Current `QManagerV4::load_q_to_gpr` rotates blocks through a 2176-byte LDS ring buffer. Rewrite to pre-load all 9 Q blocks (~19584 bytes per warp → 9 KB × 2 warps = 18 KB additional Q LDS). Confirms fit: 18 KB Q + 38 KB KV + ~20 KB Vt + 8 KB O = ~84 KB / 160 KB budget — plenty of room.

New `QManagerV4Full::load_q_to_gpr_full` method that:
- Does all 9 `vram_2_lds<0..512>` calls back-to-back
- Drains with single `s_waitcnt vmcnt(0)`
- Returns without any `lds_2_gpr` — GPR load happens later from wide-read helpers

Then `lds_2_gpr_wide<GPR_START, kColBase>` reads 128 cols (2 adjacent 64-col blocks) into 8 VGPRs per lane using the XOR-swizzled address formula.

### T2-P2: Kernel multi-Q output loop rewrite (8 hrs)

Adopt v8's proven inner-loop pattern (from agent 2 analysis):
```cpp
for (uint32_t qp = qo_start; qp < qo_end; ++qp) {
    // Reset LDS pointers per position
    p_lds_kv_curr = p_lds_q + kSzLdsQ;
    p_lds_kv_next = p_lds_kv_curr + kSzLdsKv;

    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();

    // Reload Q for this position (all heads, this Q row)
    q_manager.load_q_to_gpr<k_q_nope_begin, k_q_rope_begin>(params.query, warp_idx, qp, p_lds_q);

    // ... existing NoPE + RoPE + PV + output ...

    // Output goes to row 'qp' (not 'qo_start')
    o_manager.output_to_vram<oaccu_base, col_offset>(
        params.final_output.raw_ptr, warp_idx, qp, p_lds_o);
}
```

This re-streams K/V from HBM 8× which is the correctness-safe penalty. Amortization: each K/V tile serves 8 Q rows within the K-loop iteration, so HBM reads = 1× baseline × 8 outer / 8 inner = neutral.

### T2-P3: 8-wave ping-pong schedule (6 hrs)

Current kNumWarps=2 with kVirtualPerReal=4 uses 2 real × 4 virtual = 8 virtual positions. Re-use for ping-pong:
- Waves 0-3 (SIMD group A): do Q@K^T + softmax
- Waves 4-7 (SIMD group B): prefetch next K/V chunk
- Swap via `s_barrier` + `s_setprio(1)` on compute waves

Barrier pattern per iteration:
```cpp
if (wave_group_B) {
    __builtin_amdgcn_s_barrier();  // stall until A done
}
// A computes, B loads
__builtin_amdgcn_s_setprio(wave_group_A ? 1 : 0);
```

HipKittens' public blog shows this pattern achieving 2680 TFLOPS on CDNA4 FP8 GEMM (97% of hipBLASLt). Apply to MLA.

### T2-P4: XOR swizzle for LDS bank conflicts (3 hrs)

Current KV LDS layout has 4-phase `ds_read_b64` conflicts. Apply canonical swizzle:
```cpp
uint32_t swizzled_offset(uint32_t row, uint32_t col, uint32_t col_stride) {
    uint32_t pair = (row >> 1) & 7;
    uint32_t perm = pair ^ (((pair >> 1) ^ (pair >> 2)) & 1);
    return (row * col_stride + col) ^ (perm << 4);
}
```
Zero storage overhead vs 12.5-25% for padding.

### T2-P5: Smoke → Sweep → Bench → Gate (6 hrs)

Correctness ladder:
- Smoke: sq=1 bs=1 kv=16 → max_abs_diff < 1e-2 vs v7
- Smoke: sq=8 bs=1 kv=16 → max_abs_diff < 1e-2 vs Python-split reference
- Sweep: bs ∈ {1,2,4} × sq ∈ {1,2,4,8} × kv ∈ {16, 1024, 8192} = 36 shapes
- Wrapper bench 3× runs
- GSM8K full run
- Gate check: all 4 thresholds

### T3-P1: AITER codegen setup (8 hrs)

Clone `ROCm/aiter` upstream at HEAD. Read `hsa/gfx950/mla/codegen.py` + any existing `.s` sources under `hsa/gfx950/mla/src/` (to be discovered). If pure CSV-to-assembly, the template kernel is emitted from Python; we add a row and regenerate.

If kernel is hand-written GCN ISA in separate files, find the sq=4 `.s` file and duplicate as sq=8 variant. Modify:
- Tile M dim double
- Persistent schedule to cover 8 Q positions per work
- Output stores to 8 rows not 4

Build: `export GPU_ARCHS=gfx950; PREBUILD_KERNELS=1 python3 setup.py develop` inside container.

### T3-P2: Integrate new .co + dispatch (4 hrs)

Add `.co` file to `/app/aiter-test/hsa/gfx950/mla/`. Register in `asm_mla.cu` (the dispatch table). Modify:
- Line 308 check: add `else if (max_seqlen_q == 8) { config_max_seqlen_q = 8; sub_Q = 128; }` for gqa_ratio=16 fp8.
- `get_heuristic_kernel_mla()` returns new kernel name for (fp8, fp8, 16, persistent, qseqlen=8).

### T3-P3: Smoke → Bench → Submit (6 hrs)

Same validation ladder as T2-P5. If T3 gives ≥1700 thr/GPU AND GSM8K ≥ 0.93, ship as the final submission.

---

## Verification for every tier (strict 4/4 gate check)

After each tier's bench:
```bash
# 3× wrapper benches with kimbochen script
for i in 1 2 3; do
  docker exec re4c_v8 python3 /tmp/bmk/benchmark_serving.py \
    --model amd/DeepSeek-R1-0528-MXFP4 --backend vllm --base-url http://0.0.0.0:8890 \
    --dataset-name random --random-input-len 8192 --random-output-len 1024 \
    --num-prompts 40 --max-concurrency 4 --ignore-eos --save-result \
    --num-warmups 8 --use-chat-template \
    --result-filename tier{X}_run${i}.json
done

# GSM8K — MANDATORY every tier (user requirement)
docker exec re4c_v8 bash -c '
  cd /app/ATOM && HOME=/tmp python3 -m atom.benchmarks.dsr1_benchmark gsm8k \
    --model amd/DeepSeek-R1-0528-MXFP4 --base-url http://0.0.0.0:8890 \
    --num-shots 8 --save-results /tmp/tier{X}_gsm8k.json
'
```

All 4 gates (thr≥1500, interact≥165, E2E≤5000, GSM8K≥0.93) must pass before committing.

---

## Critical files

### Container `re4c_v8`
| Path | Purpose | Tier |
|---|---|---|
| `/app/aiter-test/aiter/mla.py` | Dispatch: inject T1 split shim at entry | T1 |
| `/app/aiter-test/csrc/kernels/mla/mla_metadata.cu` | Metadata kernel (alternative T1 path) | T1 |
| `/app/aiter-test/csrc/kernels/mla/hk/mi3xx_v32_fwd_decode_h32_fp8_fp8_v9.cuh` | HK v9 kernel (already drafted session-16) | T2 |
| `/app/aiter-test/csrc/kernels/mla/hk/hk_mla_buffer_managers.cuh` | QManagerV4Full + load_k_wide_to_gpr | T2 |
| `/app/aiter-test/csrc/kernels/mla/hk_decode_fwd.cu` | HK dispatcher + env gate | T2 |
| `/app/aiter-test/hsa/gfx950/mla/codegen.py` | ASM kernel codegen — ADD sq=8 row | T3 |
| `/app/aiter-test/csrc/py_itfs_cu/asm_mla.cu` | Line 308 dispatch patch for sq=8 | T3 |

### Local repo
| Path | Purpose |
|---|---|
| `dsr1-hackathon-dec073/RE4_hk_qh32/v9_h32.cuh` | Final HK v9 kernel |
| `dsr1-hackathon-dec073/RE4_hk_qh32/buffer_managers_v9.patch.cuh` | Q LDS rewrite + wide loaders |
| `dsr1-hackathon-dec073/RE4_hk_qh32/aiter_mla_py_mtp7_split.patch` | T1 dispatch shim |
| `dsr1-hackathon-dec073/RE4_hk_qh32/asm_mla_codegen_add_sq8.patch` | T3 codegen patch |
| `dsr1-hackathon-dec073/RE4_hk_qh32/mla_a8w8_qh16_qseqlen8_gqaratio16_ps.s` | T3 ASM source |
| `dsr1-hackathon-dec073/RE4_hk_qh32/TIER_RESULTS.md` | bench results per tier |

---

## Key existing utilities to REUSE (not rewrite)

| Function | Path | Why |
|---|---|---|
| `get_mla_metadata_v1` | `aiter.ops.mla_metadata` | Python wrapper — call with sq=1 for T1 split |
| `mla_decode_stage1_asm_fwd` | `aiter` binding | The FP8 sq=1 persistent kernel we split into |
| v7's `mla_main` lambda | `v7_h32_original.cuh:274-627` | Agent 2: works correctly at qo_end==qo_start+1; reuse unchanged in T1 |
| v8's `qp_rel` inner loop | `v8_h32.cuh:219-238` | Agent 2: proven multi-Q pattern for T2 kernel |
| HipKittens `mfma1616128` binding | `3rdparty/HipKittens/include/ops/warp/register/tile/mma.cuh:119` | T2 dispatch via `rt_16x128_s` tiles |
| `amd_matrix_instruction_calculator` | `github.com/ROCm/amd_matrix_instruction_calculator` | T2/T3 register layout verification |
| `rocprof-compute --roof-only` | ROCm 7.x builtin | T2/T3 roofline analysis |

---

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| T1 python dispatch overhead eats MTP=7 gain | Pre-allocate metadata buffers; measure vs theoretical; fallback to T2 |
| T2 Q LDS rewrite breaks sq={1,4} | Keep v7/v8 coexisting via env gate `AITER_ENABLE_HK_QH32_V9` |
| T2 8-wave ping-pong regresses by LDS contention | Profile with rocprof-compute; fall back to 4-wave if SQ_VALU_MFMA_BUSY < 60% |
| T3 codegen path has hidden AMD-internal tooling | If `codegen.py` relies on closed scripts, fall back to disassembling sq=4 `.co` and hand-modifying (objcopy + rebuild) |
| GSM8K drops below 0.93 on any tier | STOP and debug correctness — user-mandated strict gate |
| Session-15 v8 inner loop proven correct but slow | Use v8 pattern as T2 fallback if full rewrite blocks |

---

## Rollback

Every tier is env-gated. Toggle OFF to revert to RE.1:
```bash
unset AITER_MTP7_METADATA_SPLIT        # T1 off
unset AITER_ENABLE_HK_QH32_V9           # T2 off
# T3 dispatch disabled by reverting asm_mla.cu patch + removing .co from hsa/gfx950/mla/
```
RE.1 baseline = MTP=3 FP8 @ 1368 thr/GPU, proven every session. Snapshot image: `rocm/atom-dev:dsr1_session15_v8_kernel_apr22`.

---

## Honest estimates + non-negotiables

**Effort**: T1 1 day · T2 3-4 days · T3 3-5 days · Total 7-10 days if pursuing all tiers. T1 alone may close 4/4; measure first, expand if needed.

**Never-skip rules**:
1. Correctness before perf: every tier must pass GSM8K ≥ 0.93 AND 12+ correctness sweep shapes before considering a tier "done"
2. Pin GPUs 0-3 only (Kimi team owns 4-7)
3. Every patch has `.preT{N}` backup before write
4. Docker snapshot at every green gate
5. Git commit + push at every green gate
6. AMD-kernel-engineer rigor: every optimization has hardware rationale (MFMA cycles, LDS banks, VGPR, wave occupancy)
7. Research-backed: reference `amd_matrix_instruction_calculator` for register layouts, `rocprof-compute` roofline for bottleneck identification

**Projected outcome per tier**:
- T1 metadata split: 1400-1550 thr/GPU, E2E ~4500-5500ms — MAYBE 4/4
- T2 HK v9 proper: 1700-2200 thr/GPU, E2E ~3500-4500ms — LIKELY 4/4
- T3 ASM native: 2300-2800 thr/GPU, E2E ~2800-3500ms — **safely 4/4 with margin**

Ship the first tier to hit 4/4. Stack later tiers only if first ships and we want dominant submission.

---

## Day-0 opening (first 2 hours after approval)

1. Verify container state: `docker ps re4c_v8` UP, snapshot available
2. Git branch: `session17_fp8_sq8_mla` from current HEAD
3. Pull upstream ATOM PR #582 for diff study (DSR1 FP4 MLA weight quant — may inform T2 MLA design)
4. Pull upstream AITER MLA codegen to understand T3 build pipeline
5. Re-confirm RE.1 3-run bench = 1368 thr/GPU (baseline health)
6. Begin T1-P1: locate metadata emission code path

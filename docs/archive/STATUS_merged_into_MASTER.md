# DSR1 CONC=4 — STATUS (single source of truth for current state)

**Last updated**: 2026-04-22 session-14 noon UTC (wrapper-vs-direct bench regression discovered, Q3.3 applied)

## 🚨🚨🚨 SESSION-14 CRITICAL FINDING (Apr 22 ~12:00 UTC) — "gold 1500" was NEVER submittable

Our "gold P0 3/4 gates" claim from Apr 20 is BROKEN. The 1500-1614 thr/GPU numbers in `P0_run{1,2,3}.json` and `P0_reverify_run{1,2,3}.json` came from **direct bench** (`python3 -m atom.benchmarks.benchmark_serving`), NOT from the competition wrapper (`dsr1_benchmark`). Confirmed by filename convention (wrapper always writes `result_isl<ISL>_osl<OSL>_conc<CONC>.json`; our files are `/tmp/P0_run1.json` which matches the direct-bench command documented at `best_reproduce.md:117-128`).

**Competition flow is locked** (`dsr1_benchmark.cpp:1124-1134`):
1. `run_accuracy_test_gsm8k` — `lm_eval --num_concurrent=65 --num_fewshot=3` (~10 min heavy load)
2. `validate_accuracy` — must be ≥ 0.93 GSM8K to continue
3. `run_benchmark_serving` — clones `github.com/kimbochen/bench_serving`, runs its `benchmark_serving.py`
4. `tput_per_gpu = total_token_throughput / 8.0` (hardcoded for TP=8; we modify to /4.0 for TP=4)

**Only allowed wrapper modification**: `/8.0 → /4.0` (per Danish directive).

**Today's wrapper baseline (same P0 config, wrapper flow)**: 1289-1327 thr/GPU — **below the 1500 gate**.
**Today's direct-bench baseline (same P0 config)**: 1465-1614 thr/GPU — matches "gold" claim.

**The gap is ~12-15% induced by GSM8K-before-perf** (GPU DVFS state trough, HIP dispatch dict pollution, Python scheduler state, allocator fragmentation — server is not cold when perf starts).

**Implication for Q3/Q4 plan**: Every lever must be re-validated under wrapper flow, not direct bench. A lever that helps direct-bench but not wrapper does NOT count for submission. Yesterday's 3/4 was never wrapper-validated, so our true starting gate count may be 1/4 or 2/4, not 3/4.

**Q3.3 status**: `moe.py` + `deepseek_v2.py` shared-experts fusion patches APPLIED to `reproducer_best` container (spawned from gold image). Server cold-booted, smoke test passed. Direct-bench 3 runs: 1465/1524/1614 thr/GPU, TPOT 5.22-5.31. Wrapper bench LAUNCHED at 12:06:51 UTC, result pending (~12 min).

Memory: [`memory/feedback_wrapper_divide_by_4_only.md`](../../../.claude/projects/c--Users-danis-OneDrive-Desktop-AMD/memory/feedback_wrapper_divide_by_4_only.md)

---

## 🎯 SESSION-13 (Apr 21 evening) — multi-day commit + lab container

**Authority** (Danish, evening): "you will not stop until all 4 gates accomplished, the only order, everything has to be done even if complex or broken, patch it fix it do it"

**MTP=7 paths exhausted** (see session-12 entry below). All qseqlen=8 cudagraph paths crash:
- FP8 + 7-patch surgery (asm_mla.cu fold extension): boot success, smoke test pass, **CRASH at first GSM8K inference** under load
- BF16 KV + cudagraph capture: **CRASH at first capture shape** (bs=4 max_q_len=8) regardless of capture sizes
- Eager mode at qlen=8: WORKS (GSM8K 0.9287 within noise of 0.93) but 4× slower TPOT — net E2E worse

**Pivoted to multi-day Q-series plan** at `C:\Users\danis\.claude\plans\fizzy-toasting-teacup.md`:
- Q3 host-side stack (20-28h): TRITON_GEMM, shared-experts (#24097), LN+FP8 fusion (#25693), RoPE+KV fusion (#26383). Stack ceiling ~5300ms E2E (still 3/4)
- Q4 HK qh32 qseqlen=8 native kernel port (3-5 days): only mathematical hope of 4/4

**Containers**:
- `dsr_beta_q3_lab` ← spawned from gold P0 image `rocm/atom-dev:dsr1_P0_3of4_gates_apr20`. P0 booting now. AITER `73ad002`, ATOM `f8453e3` matching gold. Port 8892:8890.
- `danish_atom_dsr_beta` ← original session-13 working container. All PB session-12 reverts confirmed clean. Idle.

**Session-13 finding**: ATOM has `is_rocm_aiter_fusion_shared_expert_enabled` already plumbed (5 call sites in topK.py). `FusedMoEModularKernel(prepare_finalize, shared_experts=...)` accepts shared_experts directly. DSR1 model has `self.shared_experts` at line 879. Q3.3 may be lighter than expected (uncomment + remove explicit shared_experts call sites + adjust topk).

Memory: [`memory/project_dsr1_session13_qplan_apr21.md`](../../../.claude/projects/c--Users-danis-OneDrive-Desktop-AMD/memory/project_dsr1_session13_qplan_apr21.md)

---

## 🚨 SESSION-12 PIVOT (Apr 21 ~05:30 UTC) — read this first

Previous P5 HK qseqlen=5 surgery walked back. Slot B `--enforce-eager --num-speculative-tokens 4 + AITER_ENABLE_HK_QH32` crashed with "Memory access fault, Reason: Unknown" during warmup BEFORE cudagraph capture. Falsifies "kernel works in eager, only cudagraph fails" hypothesis.

**Replaced with**: 1-line patch to `/app/ATOM/atom/model_ops/attention_mla.py:569` mirroring vLLM PR #39616 (merged YESTERDAY, +76% tok/s on MI355X with Kimi-K2.5+Eagle3 spec=7). Disable persistent metadata for qseqlen > 4; AITER kernel computes its own non-persistent metadata internally.

**MTP map per AITER #2720**:
- Working FP8 qseqlen: {1, 2, 3, 4, 8} → spec ∈ {0, 1, 2, 3, 7}
- **DEAD silent-corrupt**: {5, 6, 7} → spec ∈ {4, 5, 6} (do NOT bench these, GSM8K silently tanks)
- **The only viable spec > 3 is spec=7 (qseqlen=8)**

**Patch applied** Apr 21 05:30 UTC. AITER pin `73ad002` (PR #2727 in HEAD), ATOM pin `f8453e3`. GPUs reduced to 4 (0-3) at ~05:25 UTC after Kimi reclaimed 4-7.

**Next**: PB boot eager MTP=7 → PC cudagraph + bench → PD GSM8K → PE commit if 4/4.

Memory: [`memory/project_dsr1_session12_mtp7_pivot.md`](../../../.claude/projects/c--Users-danis-OneDrive-Desktop-AMD/memory/project_dsr1_session12_mtp7_pivot.md)

---

**Session-10 baseline (P0 lock):**

**Mission**: pass 4/4 CONC=4 gates at `amd/DeepSeek-R1-0528-MXFP4` on 4× MI355X, TP=4 single-replica.

**Strategic constraint (from Daniel Huang Apr 20)**: this is also a mergability game. Acting as AMD engineer. Patches must NOT overlap with AMD's in-progress work. Stock model only — no merged/transplanted checkpoints.

Danish directive (standing): *"you have unlimited time and all resources, just make me reach all those 4/4 gates. you are ordered to not stop before reaching 4/4 gates, nothing else applies this is the only directive"* + session-10 lock-in: *"you wont stop until all 4/4 gates are accomplished. you will do proper research and engineering work into this and not gambling. kernel level engineering like an AMD engineer with 15 years of experience. never prematurely declare anything dead."*

## 🎯 CURRENT CAMPAIGN: P0-P8 Kernel Engineering (session-10 Apr 20)

Current phase: **P0 ✅ COMPLETE — 3/4 GATES CROSSED** (massive, +2 gates from pure hygiene)

**Master plan**: [Current_plan.md](Current_plan.md) + [`../../.claude/plans/fizzy-toasting-teacup.md`](file:///C:/Users/danis/.claude/plans/fizzy-toasting-teacup.md)
**Validated bottleneck**: [Bottleneck.md](Bottleneck.md) — hipGraphLaunch 77.7% wall, 1525 nodes/graph
**P0 result**: [dsr_beta/bench_results/P0_clean_floor.json](../dsr_beta/bench_results/P0_clean_floor.json)

## P0 NEW BASELINE (Apr 20 session-10, replaces 1351/6.66/150/7221/0.934)

| Metric | min-of-3 | Gate | Status |
|---|---:|---:|---|
| Thr/GPU | **1500.11** | ≥1500 | ✅ PASS (+11% vs 1351) |
| Interactivity | **185.04** | ≥165 | ✅ PASS (+23% vs 150) |
| Median E2E | **5762.86** | ≤5000 | ❌ FAIL (762 over, −20% vs 7221) |
| GSM8K flex | **0.9318** | ≥0.93 | ✅ PASS |
| **GATES** | | | **3/4** |

Single unlock: `--cudagraph-capture-sizes "[1,2,4,8,16,32]"` (default was 33 variants expanding to 512, we only use ≤4 at CONC=4 — dispatch-dict + device-mem pressure reduction)

## Phase table (updated)

| Phase | Status | Target TPOT | Target Gates |
|---|---|---:|---:|
| **P0 hygiene** | ✅ DONE | 5.40 ms actual | **3/4 actual** |
| P1 fusions (TRITON_GEMM) | NEXT | 5.20 ms | 3/4 |
| P2 shared-expert fusion activation | pending | 5.00 ms | 3/4 |
| P3 persistent MLA/host-overhead backport | pending | 4.85 ms | 3/4 |
| P4 drafter graph isol | pending (tbox 2d) | — | — |
| P5 HK MLA v2 qh32 | pending | 4.70 ms | 3/4 |
| P7 MTP=4 + HK qseqlen=5 | pending | 3.85 ms | **4/4** target |
| P8 MTP=5 + HK qseqlen=6 | pending | 3.30 ms | 4/4 safe |

---

## CANONICAL FLOOR (committable, STOCK MODEL — Apr 20 2026)

**Model**: `amd/DeepSeek-R1-0528-MXFP4` (NO merged checkpoints — InferenceX official model)

| Metric | Value | Gate | Status |
|---|---|---|---|
| **Thr/GPU** (÷4) | **1351** | ≥1500 | ❌ −9.9% |
| Total throughput | 5403.96 tok/s | — | — |
| **Median TPOT** | **6.66 ms** | — | (need ≤4.52 for E2E gate) |
| Median TTFT | 370.15 ms | — | — |
| Median ITL | 16.23 ms | — | — |
| **Interactivity** | **150.23** | ≥165 | ❌ −9.0% |
| **Median E2E** | **7221.33 ms** | ≤5000 | ❌ +44% |
| **GSM8K** | **0.934** | ≥0.93 | ✅ |
| **Gates** | **1/4** | 4/4 | GSM8K only |

**Workload**: ISL=8192, OSL=1024, CONC=4, num_prompts=40 (matches InferenceX `--num-prompts $((CONC * 10))`)

**Result file**: `/projects/teamA/danish/experiments/stock_floor_MTP3_TBO_QR_canonical.json`

**Reproduction recipe**: see `best_reproduce.md` (full launch command + env vars + bench command)

### Why merged DSR1-drafter-FP4 was dropped
- Daniel Huang Apr 20 mergability rule: "follow AMD progress on these two models, because if some overlaps, it might not be merged"
- InferenceX official benchmark uses canonical `amd/DeepSeek-R1-0528-MXFP4`
- Empirical: stock vs merged delta = 0.7% (1351 vs 1361 = within run-to-run variance)
- Reproducibility: stock = single canonical artifact; merged = custom transplant recipe

## 🧪 Session-8 aspirational run — E-08-05 "2/4 gates" was LUCK, not submittable

`E-08-05` (merged + MTP=3 + TBO prefill + 53-row filtered CSV + QUICK_REDUCE FP + max-num-batched-tokens=65536) delivered **interactivity 165.35 ✅, TPOT 6.05, GSM8K 0.9333** — Run 1 cleared the interact gate for the first time at TP=4 SR.

**Min-of-3 stability (DEFINITIVE)**:

| Run | Thr/GPU | TPOT | E2E | Interact | GSM8K | Gate |
|---|---|---|---|---|---|---|
| E-08-05 (Run 1) | 1304 | 6.05 | 6592 | **165.35** | 0.9333 | 2/4 ✅ |
| E-08-05b (Run 2) | 1350 | 6.25 | 6867 | **159.87** | 0.9363 | 1/4 ❌ |
| E-08-05c (Run 3) | 1351 | 6.66 | 7221 | **150.23** | 0.934 | 1/4 ❌ |
| **min-of-3** | — | — | — | **150.23** | — | **1/4** |

Run-to-run interactivity spread is ~10% (150–165). Run 1's 0.2% margin over the 165 gate was noise. **Min-of-3 fails. Not submittable as 2/4.** We need structural TPOT margin from a real kernel, not lucky runs.

→ The only path with positive-math projection is C1: custom HK qh32 kernel lifting the qseqlen=4 cap so MTP=4+ runs.

---

## Session-8 state (Apr 19 late evening → Apr 20 overnight) — C1 HK kernel v1→v2→v3 iterations

Danish directive 2026-04-19: **"timing is not the constraint, build it, I want AMD optimized kernels"**.

### What's been done

1. **B2 P-EAGLE position-only gamble**: applied, benched, 30% accept / 1.9 tok/step / **−31% thr regression** → reverted via `.pre_lever_b2`. Training-free init gives near-zero accept at t+2/t+3 as research predicted.
2. **C2 short-patch analysis**: all variants dead in our stack:
   - (a) Top-K rescoring at i=2 = 0% lift (RELAXED_TOP_N=8 already absorbs)
   - (b) Dual-chain bs×=2 verify = flat/neg (+5-15% accept × 2× verify cost)
   - (c) True tree = needs per-query attention mask = C1 kernel work
3. **C3 MTP=4+** blocked on C1 (no qseqlen>4 kernel; `hsa/codegen.py` is CSV compiler not kernel generator).
4. **🎯 HK discovery**: HipKittens MLA already in-tree at `/app/aiter-test/csrc/kernels/mla/hk/` (2646 LOC). FP8 + DeepSeek MLA shape + runtime max_seqlen_q baked in. Blocker: `static_assert(nhead==128)`.
5. **C1 patches deployed** (backups `.pre_c1`):
   - NEW `/app/aiter-test/csrc/kernels/mla/hk/mi3xx_v32_fwd_decode_h32_fp8_fp8.cuh` — h32 traits + wrapper reusing h128 kernel body via template (kBlockM=32, kNumWarps=2, kTileM=16)
   - NEW `num_head==32` dispatch branch in `hk_decode_fwd.cu`
   - `aiter/jit/optCompilerConfig.json` — h32 header added to `module_hk_mla` srcs
   - `aiter/mla.py:330-437` — `use_hk` gated on new `AITER_ENABLE_HK_QH32` + native-supported extended for qh32 qseqlen=5-8
   - `atom/config.py:882` — MTP cap lifted 4→8
6. **JIT compile** ✅ **SUCCEEDED** in 34.3s under standalone test. Template instantiated cleanly at kNumWarps=2. `module_hk_mla.so` built.
7. **First boot attempt HUNG**: MTP silently collapsed to MTP-1 (no `max_q_len=4` captures), pgrep showed 2 of 4 workers alive, log flooded with "No SHM broadcast block" timeout (40+ times). Likely HK qh32 crashed silently on rank 2/3 during MTP-3 drafter capture.
8. **Container restart** cleared 330 zombie python3 + 282GB VRAM leak. All patches survived.
9. **Control boot** without `AITER_ENABLE_HK_QH32` launched to isolate cause. In progress at session close.

### Critical JIT cache workaround

`/root/.aiter` is read-only in the container overlay FS (even for uid=0). Use `HOME=/tmp` env override for all invocations.

---

## 🔬 C1 HK qh32 kernel iterations (E-08-06 series, Apr 19→20 overnight)

### v1 — compiled + booted + garbage output
- **Kernel**: standalone h32 body (~860 LOC) with virtual-warp loops at Q load, K async_load, V load+transpose+store
- **Compile**: 465KB .so after fixing 2 bugs: (a) removed duplicate symbol definitions (HkMlaDecodeFwdParams, pack_4f32_to_fp8, max_8, PvGemmEpilogueType — already pulled in via h128 header include in hk_decode_fwd.cu); (b) reverted `kOccupancy: 4 → 1` to restore VGPR budget so `pack_4f32_to_fp8<fp8_e4m3>` template substitution resolves at GPR 121
- **Boot**: server up, `/health` OK, `max_q_len=4` captures present → MTP=3 actively dispatches HK path
- **Single request test** (`"What is 2+2?"`): output = `"firc,●●irc.●●. bbb \n \n.\nrc##1，●●"` — **GARBAGE**
- **Root cause**: Q load virtual-warp loop overflows `gl_q<q_t, -1, kNumTilesM=2, kTileM=16, 576>` buffer — at h32, `kNumTilesM = kBlockM/kTileM = 32/16 = 2` (vs 8 at h128). Writing at `virtual_warp_idx ∈ {2, 4, 6}` clobbered out-of-bounds memory

### v2 — reverted Q + K virtual-warp loops, kept V
- **Fix**: Q load + K initial async_load both single-call with real `warp_idx`. V store_transposed_v_to_lds virtual-warp loop kept (LDS access, correct distribution of 8 warp slots over 2×4 iterations)
- **Compile**: SUCCESS
- **Boot**: OK, `max_q_len=4`
- **Single request test**: output = `"ggy the 1, questionnaire 1. ttsett1chioాన1# The\nWell,"` — **STILL GARBAGE**
- **Root cause**: inconsistency between K staging and V read. K async_load fills 2-warp-sized LDS (real warp_idx 0,1 → 2 LDS slots). V store writes to 8-warp-virtual positions (4 iterations × 2 real warps = 8 LDS slots in a different layout). V load reads from K staging LDS — at virtual_warp positions {2,3,4,5,6,7} there is **no data** (K never wrote there) → reads uninitialized → garbage

### v3 — outer K virtual-warp applied — STILL GARBAGE
- **Fix**: re-apply virtual-warp loop to outer initial K async_load
- **Boot**: server up 40+ min, max_q_len=4 captures, MTP=3 active, /health OK
- **Single request test**: `"What is 2+2?"` → output `"1SPJ.輕易.#的快sey角和的快角和角和角和角和角和oun NorthwesternQuiz Ver 000的快     000. Z"` = **GARBAGE** (TPOT_s=0.0077 — kernel runs fast, just wrong)
- **Real root cause**: v3-fix-script comment correctly anticipated — INNER K prefetch sites at lines 288 + 314 still use real warp_idx. Each tile iter prefetches NEXT tile's K with only 2 of 8 LDS slots → next iter's V load reads garbage from vwarp slots 2-7
- **Note**: my earlier kSzLdsKv-overflow hypothesis was WRONG — `kNumSubBlocks = kNumRows / kNumRowsPerSubBlock = 32/4 = 8` is INDEPENDENT of kNumWarps; LDS is correctly sized

### v4 — full-tile virtual-warp K prefetch (replaces chunked) — STILL GARBAGE
- **Fix v4** (`/tmp/fix_v4.py` on server): drop chunked `async_load_k_tile` per-iter prefetch entirely; replace with single full-tile `async_load_k` virtual-warp loop at top of `mla_main` lambda. Trades chunked overlap with NoPE GEMM for correctness
- **Boot run-1**: HIP OOM crash — VRAM zombies from v3 pkill (89% occupied)
- **Container restart** cleared GPUs 0-3 to 0%. v4 patch survived restart (827 lines, 3 "v4:" markers verified)
- **Boot run-2**: server up clean, 0 errors, all 4 workers init success, max_q_len=4 captures, /health OK
- **Single request test**: `"What is 2+2?"` → output `"bb00:kkkqg\nb\nbbbbbb00\n1C  \n\n5. Z2\n    (Z, and 2"` = **STILL GARBAGE**
- **.so timestamp 04:14 > source 03:42** → v4 IS being executed; output genuinely from new code
- **All warp_idx sites now virtual-warp-looped**: Q (single-call OK at kNumTilesM=2), outer K loop ✓, inner K full-tile loop ✓ (v4), V load+transpose+store loop ✓, output (single-call OK at kQoNumHead=32 = 2-warp natural coverage)

### Conclusion: virtual-warp simulation hits structural wall

After 4 patches covering EVERY warp_idx site, output is still garbage. The HK kernel was designed around 8 warps cooperating in fixed lockstep on the LDS layout. Bolting virtual-warp loops onto every site doesn't recover correctness. Plausible remaining issues:
- Hardcoded constants in load_v_to_gpr line 1000: `col = (lane%16)*8 + warp_idx/2 * 128` — `*128` is per-warp-pair col-tile width that depends on layout density, not just warp count
- `kNumColsPerWarp = kNumCols/kNumWarps` (line 597) = 8 at h128, would be 32 at h32 — but other sites hardcode `*8` and don't scale with formula
- Implicit `s_barrier` semantics that assume 8-warp participation
- `s_waitcnt lgkmcnt(N)` tuned for specific 8-warp dependency chains

### v5 — native 2-warp buffer manager redesign (COMMITTED, multi-day)

- Write `KvManagerV2_H32` + `VtManagerV1_H32` classes natively for 2-warp LDS layout
- Native math: `kNumColsPerWarp = kNumCols / kNumWarps = 64/2 = 32` (vs hardcoded 8 in v1)
- Each warp covers (16 rows × 256 cols) split-by-cols, OR (16 rows × all 512 cols with 4 inner col-tile iters)
- Replace ALL hardcoded `*8`, `*128` with formulas based on kNumWarps
- ~400-600 LOC across 2 new manager classes + minor kernel changes
- Estimated 1-2 days careful coding + correctness verification + per-tile MFMA testing
- Structurally correct — no virtual-warp simulation, no hardcoded 8-warp constants
- Likely FASTER than virtual-warp version (no 4× serial overhead per LDS access)

---

## Active plan

Plan file: `C:\Users\danis\.claude\plans\fizzy-toasting-teacup.md`

### Next steps (resume checklist)

1. Check `/tmp/atom-control.log` for `max_q_len=4` captures (control boot result)
2. If control passes → baseline MTP-3 still works; HK integration is the bug. Debug per-rank:
   - (a) Drafter tensor shape mismatch vs HK kernel expectations
   - (b) JIT cache lock contention (4 ranks compile `module_hk_mla` simultaneously)
   - (c) `work_info_set` metadata from ATOM's `prepare_mtp_decode` incompatible with HK kernel
3. If control also fails → env drift vs session-7 floor; deeper investigation
4. If HK proves unviable after debug → revert `.pre_c1` backups + submit floor `1361/6.35/157/6842/0.934` as final committable entry (1/4 gates)

### Gate math projections

- **Floor**: 1361/6.35/157/6842/0.934 → 1/4 gates
- **C1+C3 MTP=4 land, +33% tok/step**: TPOT 4.77, E2E ~4880, interact ~210 → **4/4 gates** if GSM8K holds
- **No other path** identified with positive expected math (all shorter levers exhausted session-7)

---

## Floor reproduction recipe

### Stack (DSR_beta config)

| Component | Value |
|---|---|
| Docker image | `rocm/atom-dev@sha256:52c5195a712b5d3a0993d5e63de9b8ffc13a77d0c4b2f31d40afe9e62c12ab5f` |
| ROCm | 7.2.2 |
| PyTorch | 2.10.0+rocm7.2.2.git40d237bf |
| aiter | commit `73ad0023e15e9735b3af95b3357b99cf7f801bf1` (main) |
| ATOM | commit `f8453e3fc0f65191fb2034602dc9a2066a78020b` (main) |
| flydsl | 0.1.3.1 |
| triton | 3.5.1 |
| Container | `danish_atom_dsr_beta` port 8890 |
| Model | `/projects/teamA/danish/models_merged/DSR1-drafter-FP4` (DEC-075 drafter FP4 transplant) |

### Required local patches (3)

1. `rejection_sampler.py`: `RELAXED_TOP_N=8, RELAXED_DELTA=0.5` (was 10, 0.6)
2. `attention_mla.py`: `num_kv_splits=None` (was 16)
3. **Phase 3 sync-fuse** — `model_runner.py`: merge `send_mtp_status_to_cpu_async` rejected+bonus tensors into single stacked tensor. Patch: `dsr_beta/scripts/phase3_patch.py`

### Launch (floor config, mtp_k=3)

```bash
~/bin/docker exec -e HOME=/tmp \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e NCCL_MIN_NCHANNELS=16 \
  -e ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=1024 \
  -e ATOM_ENABLE_RELAXED_MTP=1 \
  -e MODEL=amd/DeepSeek-R1-0528-MXFP4 -e PORT=8890 -e TP=4 \
  danish_atom_dsr_beta bash -c '
    cd /projects/teamA/danish/repos/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x
    bash launch_atom_server.sh --enable-tbo prefill --num-speculative-tokens 3
  '
```

**Boot verify markers**:
- `Capturing bs=4, max_q_len=4` → mtp_k=3 captured correctly ✓
- `flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w3_fq` at bs=4 → drafter FP4 fast path ✓
- If `max_q_len=2` only → MTP silently collapsed to MTP-1 (BAD signal)

### Bench

```bash
# From inside danish_atom_dsr_beta:
cd /projects/teamA/danish/repos/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x
./dsr1_benchmark perf          # TPOT, thr, interact, E2E + GSM8K in one run
./dsr1_benchmark acc           # GSM8K standalone (use min-of-3 for stability)
```

---

## Honest lever inventory (Apr 19 session-8 close)

| Lever | Status | Notes |
|---|---|---|
| B drafter HIP graph (v1→v6) | ❌ ALL CRASHED session-7 | Fundamentally incompatible with MoE+NCCL on gfx950 |
| C prefix cache (v1→v4) | ❌ ALL CRASHED session-7 | Kernel has byte-level FP8 + layout + stride assumptions |
| A1 hipBLASLt retune | ❌ BLOCKED session-7 | Tuner solidx non-round-trip + aiter JIT destroyed pristine CSV |
| B2 P-EAGLE position-only gamble | ❌ −31% thr session-8 | Training-free = near-zero accept t+2/t+3. Reverted |
| B1 drafter FP4 transplant | ✅ ALREADY DEPLOYED DEC-075 | Baked into floor |
| **C1 HipKittens qh32 port** | 🚧 session-8 JIT ✅, first boot hung | Control boot in progress |
| C2 tree spec | ⚠️ PROVED DEAD | All variants no-op or net-neg (needs C1's kernel mask) |
| C3 MTP=4+ | ⏳ BLOCKED on C1 | No qseqlen>4 kernel exists yet |
| Patch #4 MLA flatten | ✅ ALREADY IN MAIN | git-blame Oct-Dec 2025 |
| TP=8 (parked for higher CONC) | ⚠️ 2/4 gates (interact ✓ first time) | Launch-latency bound at CONC=4; won't fix 4/4 alone |

---

## Rules in force

1. **Autonomous mode** — no permission asking
2. **CONC=4 only** until 4/4 gates pass (CONC=32/128 gated)
3. **GitHub push ONLY on new record**
4. **"Infeasible never terminal"** — 3 ranked paths with file:line blockers, start cheapest
5. **Pause before server boot** (12-15 min cold boot locks all 4 GPUs)
6. **Pre-measure** every intervention: target ms + mechanism + expected delta + pass/fail gate + post-measure
7. **Always optimized**, never naive
8. **Timing not a constraint for C1** (Danish auth'd)

---

## Fallback if C1 proves unviable

- Revert `.pre_c1` backups on: `hk_decode_fwd.cu`, `optCompilerConfig.json`, `mla.py`, `atom/config.py`
- h128 path untouched throughout (proven + bit-identical)
- Submit floor `1361/6.35/157/6842/0.934` as final committable entry (1/4 gates)

## Key related docs

- **FINDINGS.md** — canonical DECs + dead/alive lever decisions
- **HISTORY.md** — chronological session narratives
- **INFRA.md** — server/hardware/container/filesystem reference
- **BRIEF_FOR_KIMI_OPUS.md** — cross-agent handoff for Kimi track

## Memory pointers (for Claude sessions)

- `project_forged_plan_apr18_evening.md` — session-8 state (read first on resume)
- `project_c1_port_design.md` — full C1 port design + tracking checklist
- `project_c1_hipkittens_mla_archaeology.md` — HK code archaeology
- `.claude/plans/fizzy-toasting-teacup.md` — active plan

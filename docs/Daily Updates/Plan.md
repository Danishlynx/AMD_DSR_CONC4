# DSR1 path to gates — MoE-fusion-driven plan (May 03 v3, May 04 status update)

**Mirror of**: `C:\Users\danis\.claude\plans\for-tyheplan-you-made-validated-oasis.md`
**Status (May 04)**: Phase A Days 1-2 EXECUTED. F1+F4+F7+F2 all NO-WIN at variance floor (architectural insights kept, perf=0). **Lever G TOP_N=10→12 = PROVISIONAL WIN −0.200 ms dev-bench, kimbochen-pending.** Forward = kimbochen-confirm G + Phase C C1 stream split + Lever A router cache + Lever D silu requant skip.

> 🚫 **MAY 04 KIMBOCHEN-ONLY DIRECTIVE**: lever WIN promotion requires `dsr1_benchmark perf` 4-iter median(2,3,4). Dev-bench numbers are guidance, not promotion proof. The only authoritative DSR1 baseline is GOLD `dsr1_apr30_phase11_v3_2of4` (sha c58cf2ce4512) kimbochen 4-iter: **TPOT 5.641 / GSM PASS / Intvty 178 / 2-of-4**.

## Context

Track 1 Day 0 complete: Phase 1a thresholds reverted to (10, 0.6), TPOT recovered to **5.617 ms** (informal mode), Intvty 178 ✅, GSM 0.94 ✅, **2/4 gates**. Need −1.55 ms TPOT to clear E2E gate at TP=4 / div=4 scoring.

**Prior v2 plan invalidated by deep verification**: PR #24097 fusion was already active. AITER mhc kernels target sinkhorn-MoE (DSR1 uses `noaux_tc`). AITER #2961 is FP8 PTPC (DSR1 is MXFP4). vLLM #41217 sparse MLA is V3.2-only (DSR1 is V3 — verified HF config has no `indexer`). Most upstream cherry-picks dropped.

**This v3 plan attacks the biggest verified bucket: MoE GEMM at 27% wall = 1.66 ms/iter.** Profile (May 03 M1 trace) shows MoE-stage1 0.75 ms + stage2 0.38 ms + sort/topk/quant ~0.5 ms. Per-layer: 5-7 separate kernels × 58 MoE layers = 290-406 launches/iter. Kernel-launch overhead alone is ~5 µs/layer × 58 = ~0.30 ms latent room.

**Deep MoE kernel scan discovered**: `aiter.fmoe_g1u1` (= `module_moe_fmoe_asm` prebuilt) is **REGISTERED for our exact tuple** at `aiter/fused_moe.py:522` (gfx950, Silu, per_1x32, bf16, fp4x2, fp4x2, isG1U1). This single-kernel ASM path eliminates inter-stage HBM round-trip + 4-6 kernel launches. Currently disabled because tuner CSV has `run_1stage=0`.

## Architecture pre-flight (read before each phase)

**Per implementation step, the architectural questions to answer first:**
1. What ATOM/AITER call site is being modified? (file:line)
2. What kernel/kernels does this affect at decode dispatch?
3. What does the M1 profile say about that kernel's % wall?
4. Is there a cudagraph capture concern (state mutation inside captured graph)?
5. What's the GSM8K risk (numerical change vs pure scheduling)?
6. What's the rollback path if regression?

## Phase A — Cheap MoE wins (Days 1-4, dispatcher/CSV changes only)

| Day | Step | File | Change | Δ TPOT (expected) | Validation |
|---|---|---|---|---:|---|
| 1 AM | **F1: Force `run_1stage=1`** | source-bundled `/app/aiter-test/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv` (live `/tmp/aiter_configs/` is empty so source CSV is read) | Set col `run_1stage=1` for our M=4-32 inter=256/512 rows. Dispatcher will use `aiter.fmoe_g1u1` ASM single-kernel path. | −0.05 to −0.20 ms | Boot log must show `[fused_moe] using 1stage` for our shape. 3-iter GSM ≥ 0.9325. 4-iter dev-bench TPOT vs 5.617 baseline. |
| 1 PM | **F4: CSV write to live mount** | `/tmp/aiter_configs/dsv3_fp4_tuned_fmoe.csv` | Run `gemm_moe_tune.py` writing to live mount. | −0.05 to −0.10 ms | checkAllclose vs torch ref. GSM ≥ 0.9325. |
| 2 AM | **F7: `moe_fused_gate` integration** | `/app/ATOM/atom/model_ops/topK.py` | Replace separate topk_softmax + group selection with `aiter.moe_fused_gate`. | −0.02 to −0.05 ms | Smoke + GSM canary. |
| 2 PM | **F5: `doweight_stage1=1` (tkw1)** | `/tmp/aiter_configs/dsv3_fp4_tuned_fmoe.csv` | `aiter.fmoe_g1u1_tkw1` at line 526 — verify MXFP4 support. | −0.02 to −0.08 ms | If tkw1 doesn't support MXFP4: drop. |
| 3 | **Phase A consolidate + 4-iter kimbochen perf** | — | Lock CSV. `dsr1_benchmark perf` 4-iter. | cumulative −0.10 to −0.30 ms | If GSM ≥ 0.93 + TPOT ≤ 5.40 → advance B. |
| 4 | **Phase A bench + decision** | — | Compare to baseline. | — | — |

## Phase B — Kernel author (Days 5-11)

| Day | Step | File | Change | Δ TPOT | Validation |
|---|---|---|---|---:|---|
| 5-6 | **F2 design: tile_m=16 FlyDSL** | `/app/aiter-test/aiter/ops/flydsl/moe_kernels.py:717` | Add `16` to `tile_ms = [32, 64, 128]`. Verify FlyDSL kernel template supports tile_m=16. JIT compile candidates. | (compile prep) | Compile success. Test scaffold microbench. |
| 7 | **F2 tune** | `/tmp/aiter_configs/dsv3_fp4_tuned_fmoe.csv` | `gemm_moe_tune.py` with t16 candidates at M=4. checkAllclose. | −0.10 to −0.20 ms | GSM canary + bench. |
| 8 | **F2 measure** | — | 4-iter dev-bench. | — | — |
| 9-10 | **F3: `moe_op_mxfp4_silu_fused` (Triton)** | ATOM dispatcher TBD | Wire env `ATOM_USE_TRITON_MXFP4_SILU_FUSED=1`. | −0.05 to −0.15 ms | Drop if Triton slower (history). |
| 11 | **Phase B consolidate** | — | Kimbochen perf 4-iter. | cumulative −0.10 to −0.25 ms additional | A+B target: −0.20 to −0.55 ms |

## Phase C — Non-MoE Track 1 levers (Days 12-14)

| Day | Step | File | Change | Δ TPOT |
|---|---|---|---|---:|
| 12 | **C1: MLA q_proj ‖ concat_and_cache stream split** | `aiter_mla.py:51-65` + `envs.py` | Reuse `dual_stream_moe_forward` pattern. Env-gated `ATOM_MLA_OVERLAP=1`. | −0.03 to −0.08 ms |
| 13 | **C2: ATOM_SPEC_V2_OVERLAP=1** | `boot_phase11_v3_memfix.sh` | Already wired at `eagle.py:299`. 5-min env probe. | −0.02 to −0.05 ms |
| 14 | **C3: R2-C dispatcher wire** | R2-C kernel + AITER `fused_moe.py:213` | Build R2-C `.so` inside dsr1_prof. Wire env-gated dispatcher. | −0.05 to −0.10 ms |

## Phase D — Final scoring + snapshot (Day 15)

- Run unmodified `dsr1_benchmark perf` 4-iter (no `BASELINE_GSM8K_METRIC` override)
- Record JSONs to `bench_results/may03_phase1/phaseD/`
- `docker commit dsr1_prof rocm/atom-dev:dsr1_may03_phaseD_<TAG>` if results improve
- Push evidence + git commit + tag
- Update existing Daily Updates docs per backup.md

## Stop conditions per phase

- **Phase A stop**: cumulative ≥ −0.20 ms AND GSM_med ≥ 0.9325 → advance B. < −0.05 ms or GSM regress → revert.
- **Phase B stop**: cumulative (A+B) ≥ −0.40 ms → ship 3/4 likely. < −0.20 ms → revert to A locked.
- **Phase C stop**: cumulative (A+B+C) ≥ −0.50 ms → final scoring. Otherwise lock at best.

## Out of scope (verified dropped)

- Eagle3 spec decode — out of scope per user
- AITER mhc cherry-pick (#2978/#2963) — sinkhorn ≠ noaux_tc
- AITER #2961 — FP8 ≠ MXFP4, Qwen-only
- vLLM #41217 sparse MLA — V3.2 ≠ V3 R1-0528
- Tree-spec at CONC=4 — DEC-074 no-op
- BF16 hipBLASLt re-tune — DEC-049/072 broke MTP
- MTP=4 at TP=4 — kernel-closed
- Megakernel layer fusion (P6) — 4-6 weeks
- Apr 20 P0-P8 — wrong bottleneck
- F6 `moe_op_e2e` Triton — slower than CK on MI355X

## Honest probability budget

| Phase | Wall | Cumulative ΔTPOT | Outcome | P(gates) |
|---|---|---:|---|---:|
| Phase A | 4 days | −0.10 to −0.15 ms | TPOT ~5.45-5.50 | P(2/4) 95%, P(3/4) 20% |
| Phase A+B | 11 days | −0.20 to −0.35 ms | TPOT ~5.25-5.40 | P(3/4) 35%, P(4/4) 5% |
| A+B+C | 14 days | −0.25 to −0.50 ms | TPOT ~5.10-5.35 | P(3/4) 50%, P(4/4) 12% |

**Realistic ship**: 3/4 gates by mid-May. P(4/4) ≈ 12% without Eagle3.
**Top-10 ($10K) guaranteed** at any path from "ship Apr 30 baseline" onward.

---

## Execution log (live)

### Phase A Day 1 AM — F1 architectural pre-flight

- ✅ Q1 site: `/app/aiter-test/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv` (live `/tmp/aiter_configs/` empty)
- ✅ Q2 dispatch: `MOEMetadata` returns `(stage1_func, None, ...)` when `run_1stage=True` → `aiter.fmoe_g1u1` ASM (= `module_moe_fmoe_asm.so`)
- ✅ Q3 wall%: 27% (moe_gemm1 13.34 + moe_gemm2 6.81 + sort/topk/quant ~7%)
- ✅ Q4 cudagraph: ASM .so prebuilt — graph-safe (no Python attribute mutation)
- ✅ Q5 GSM: reduction order shift → 3-iter canary mandatory
- ✅ Q6 rollback: cp from `.preTrack1_F1_run1stage` backup, instant

**Pre-flight verdict: PROCEED**. Edit target = source-bundled CSV (`model_configs/`).

### Phase A Day 1 AM — F1 EDIT APPLIED (May 03 ~15:30 UTC)

- Backup created: `/app/aiter-test/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv.preTrack1_F1_run1stage`
- Backup md5: `f3dc44c9d41c3ace577277f6ce314300`
- Post-edit md5: `36d0b916dbda8ad9ced5a049f636d392`
- **19 decode-regime rows flipped run_1stage 0→1** at (cu_num=256, M ∈ {1,2,4,8,16,32}, expert=257, topk=9)
- 0 prefill rows touched (M ≥ 64 left unchanged)
- Confounding edits reverted for clean F1 isolation:
  - Lever #11 max_split_per_batch=32 → 16 (reverted from `aiter_mla.py.preTrack1_split32`)
  - Lever #12 ATOM_USE_V6E_PREQUANT_STASH=1 → removed from boot script
- Thresholds confirmed (10, 0.6) baseline
- Boot dispatched: `/tmp/phase11_memfix_boot_153340.log`, monitor `bqzig3pvc`

Validation criteria for F1:
1. Boot log must show `[fused_moe] using 1stage` for our shape (NOT `2stage default`)
2. Smoke test: curl /v1/completions returns coherent output
3. 3-iter GSM canary median ≥ 0.9325 (informal mode)
4. 4-iter dev-bench TPOT vs Day 0 baseline 5.617 ms — accept if ≤ 5.617 + 0.05 ms
5. Revert if regression > +0.05 ms or GSM < 0.9325

### F1 RESULT — DEAD (architectural block)

**Boot log proof of dispatch (boot_160505 log)**:
```
[aiter] [fused_moe] using 1stage (kernelName1='flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w3', 
  kernelName2='flydsl_moe2_afp4_wfp4_bf16_t32x256x256_atomic_persist') 
  for (256, 32, 7168, 512, 257, 9, ..., 'QuantType.per_1x32', True, False)
```
✅ Dispatcher fired `using 1stage` exactly at the M boundary we set (M=32 flipped, M=64 stayed 2stage).

**Boot crash root cause** (during cudagraph capture):
```
RuntimeError: fmoe_g1u1 failed: [AITER] /app/aiter-test/csrc/py_itfs_cu/asm_fmoe.cu:304 
  get_heuristic_kernel not find kernel gfx950flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w3
```

**Architectural finding**: `aiter.fmoe_g1u1` (the 1stage ASM dispatcher) requires kernels registered in **ASM-bundled heuristic registry** (`module_moe_fmoe_asm.so`). Our CSV rows reference **FlyDSL** kernel names (`flydsl_moe1_*`). These are two separate kernel registries — `fmoe_g1u1` cannot use FlyDSL kernel names. To use the 1stage path, we'd need an ASM kernel bundled for our exact shape (cu_num=256, M=32, model_dim=7168, inter_dim=512, expert=257, topk=9, FP4×FP4). **No such ASM kernel exists in `module_moe_fmoe_asm.so` for our shape** (the registration table at `fused_moe.py:522` declares the entry function but the heuristic kernel registry doesn't have shape-matched kernels).

**Lever F1 status**: DEAD without authoring new ASM kernels for our shape (multi-week kernel work, not in Phase A scope — overlaps with Phase B F2 t16 author).

**Auto-merger gotcha discovered**: `aiter` boot-time CSV merger globs ALL `dsv3_fp4_tuned_fmoe.csv*` files in `model_configs/` directory (including backup suffixes `.preTrack1_*`). When duplicates detected, merger auto-resolves "lowest us" and writes back, **overwriting backup files**. Backup files MUST be moved OUT of `model_configs/` directory before booting. Apr 14 RE.3 auto-wipe finally explained.

**Recovery**: CSV restored from `/projects/teamA/danish/repos/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv` (md5 `eea36e73...`, 47 rows). Different version than original pristine but valid schema. Baseline reboot in flight to confirm.

### Phase A Day 1 — F1 FINAL VERDICT: viable mechanism, NEUTRAL-or-NEGATIVE perf at our shape

**F1 v3 (M=32 only)**: TPOT 5.7196 ms vs baseline 5.7141 = +0.005 ms (NEUTRAL, within noise).
**F1 v4 (M=4,8,16,32)**: TPOT ~6.99 ms vs baseline 5.7141 = **+1.27 ms REGRESSION**.

**Root cause**: ASM kernel registry's only matching tile is 32x512. At M=4 padding waste = 87.5%; at M=8 = 75%; at M=16 = 50%. The kernel-launch-overhead saving (~0.30 ms total) is dwarfed by 6× compute waste from zero-padding small M to t32.

**Architectural finding**: F1 1stage path is **structurally limited at small-M decode regime**. The ASM registry has FP4 Silu kernels ONLY at tile_M=32 — smaller-M variants (t16, t8, t4) don't exist for our (cu_num=256, FP4×FP4, expert=257) shape. To make F1 profitable would require **authoring new ASM kernels with smaller M-tiles** — multi-week work that overlaps with Phase B F2 (FlyDSL t16 author) but at ASM target.

**Final F1 status**: NEUTRAL lever. CSV reverted to git-restored pristine (md5 `7ca7f861b300628922a9e926912d2cc3`). **F1 banked zero perf delta but proved 3 architectural insights**:
1. AITER CSV merger globs backup files → backups must be moved out of `model_configs/` directory
2. `fmoe_g1u1` ASM dispatcher requires C++ Itanium-mangled kernel names (NOT plain or FlyDSL names)
3. The 1stage path's tile rigidity makes it unprofitable at our small-M regime without new kernel work

### Phase A Day 1 NOT DEAD: ASM registry inspection found FP4 Silu kernels

Deeper debug per AMD principal engineer protocol "stay on task, dump SASS, debug deeper":
- `module_moe_fmoe_asm.so` exposes **674 gfx950 kernels** total
- **68 FP4 kernels** registered with `pertokenMXfp4` quant scheme
- Pattern: `fmoe_bf16_pertokenMXfp4_g1u1_<vs/novs>_silu_<1tg/2tg>_<ps?>_<tile>`

Sample registered FP4 Silu kernels (gfx950):
```
fmoe_bf16_pertokenMXfp4_g1u1_vs_silu_2tg_32x256
fmoe_bf16_pertokenMXfp4_g1u1_novs_silu_2tg_32x256
fmoe_bf16_pertokenMXfp4_g1u1_novs_silu_1tg_32x512
fmoe_bf16_pertokenMXfp4_g1u1_vs_silu_1tg_ps_32x512  ← strongest candidate (vs, ps, 32x512)
fmoe_bf16_pertokenMXfp4_g1u1_vs_silu_1tg_32x512
fmoe_bf16_pertokenMXfp4_g1u1_novs_silu_1tg_ps_32x512
fmoe_bf16_pertokenMXfp4_g1u1_vs_silu_2tg_ps_32x256
fmoe_bf16_pertokenMXfp4_g1u1_novs_silu_2tg_ps_2tg_32x256
```

### F1 v2 — write CSV row with ASM kernel name + run_1stage=1

After baseline reboot validates pristine state, attempt:
- Add new CSV rows for our shape with kernelName1 = `fmoe_bf16_pertokenMXfp4_g1u1_vs_silu_1tg_ps_32x512`
- Set run_1stage=1
- Boot, dispatch will pick our new row, fmoe_g1u1 ASM lookup will resolve correctly
- 32x512 tile matches our inter_dim=512; M-tile 32 pads our M=4-32 (zero-cost padding inside kernel)

**Expected effort**: 1 day. **Expected delta**: −0.05 to −0.20 ms TPOT (eliminates 4-6 kernel launches per layer, single-kernel ASM forward).
**Risk**: First time running this kernel in DSR1 production — need GSM canary + checkAllclose-equivalent validation.

### F1 v2 RESULT — NEUTRAL/regress (architectural finding: 1stage path tile-rigid for small M)

- F1 v3 (M=32 only flipped): TPOT 5.7196 ms vs baseline 5.7141 = +0.005 ms (NEUTRAL, within noise).
- F1 v4 (M=4,8,16,32 flipped): TPOT ~6.99 ms vs baseline 5.7141 = **+1.27 ms REGRESSION**.

**Root cause**: ASM kernel registry's only matching FP4 Silu tile is 32x512. Padding waste:
- M=4 → 87.5% waste
- M=8 → 75% waste
- M=16 → 50% waste

The kernel-launch-overhead saving (~0.30 ms total) is dwarfed by 6× compute waste from zero-padding small M to t32.

**Architectural conclusion**: F1 1stage path is **structurally limited at small-M decode regime**. ASM registry has FP4 Silu kernels ONLY at tile_M=32. Smaller-M variants (t16, t8, t4) don't exist for our (cu_num=256, FP4×FP4, expert=257) shape. Profitable F1 would require authoring NEW ASM kernels with smaller M-tiles — multi-week, overlaps with Phase B F2 at ASM target. CSV reverted to git-restored pristine (md5 `7ca7f861b300628922a9e926912d2cc3`).

### Phase A Day 1 PM — F4 `gemm_moe_tune.py` re-retune RESULT — NEUTRAL/regress

Tuner output: 5/8 wins on M=8/16/32 × inter=256/512 (best: M=32 inter=512 -77.02 µs). 3 regress on M=4/M=8 inter=256. AITER merger kept lowest-us per shape; new winners FIRED at decode (boot log confirmed). But TPOT median(iter2,3,4) = **6.1477 ms = +0.43 ms vs baseline 5.7141**.

**Root cause**: M=4 hot-path unchanged (old beat new there). Bench variance ±0.5 ms (5.56→6.61 range across iters) swamps kernel-level deltas. M=16/32 inter=512 wins (-29 to -77 µs/call × 58 layers = potential -0.94 to -1.7 ms cumulative) didn't surface because M=4 dominates decode shape distribution at CONC=4 + MTP=3 verify.

**Architectural lesson**: don't re-run gemm_moe_tune.py expecting Phase A wins for hot-path-already-tuned regimes. CSV reverted via `.preTrack1_F4_181956`. CSV `us` column is informational, not authoritative for current bench (tuner non-deterministic across runs).

### Phase A Day 2 — F7 `moe_fused_gate` DEFERRED (sub-noise-floor)

Architectural deep-read: `aiter.biased_grouped_topk` already dispatches to `biased_grouped_topk_hip` for token_num ≤ 54k (always at CONC=4). `aiter.moe_fused_gate` is alternative kernel with extra `n_share_experts_fusion` param. Shared-expert fusion already active upstream (Q3.3 patch). F7 = swap `hip` for `fused_gate` impl. Ceiling −0.02 to −0.05 ms / Risk: GSM regression on numerics swap. Given F4 noise band ±0.5 ms and F7 ceiling below noise floor, F7 deferred.

### Phase B Days 5-7 — F2 tile_m=16 FlyDSL stage1 author RESULT — NO WIN

- F2 v1: added `16` to stage1 FP4 registry at `aiter/ops/flydsl/moe_kernels.py:72`. 240→272 kernels. JIT compile clean. Tuner picked tile_m=32 at both M=4 inter=256 and inter=512.
- F2 v2: extended k_batch enumeration to t16 candidates at line 87. 272→304 kernels. All compile. Tuner STILL picked tile_m=32.
- `_fp4` suffix mystery: tuner ALREADY enumerates `_fp4` variants at `gemm_moe_tune.py:2342-2351`; pristine CSV's M=4 us values (33.60 / 47.80) are STALE baselines from prior tuner runs at different bench/thermal state.

**Architectural finding**: t16 vs t32 use the SAME MFMA primitive `mfma_f32_16x16x32_f8f6f4`. t32 internally iterates 2x in M direction → better K-loop pipelining (more LDS read cycles overlapped with MFMA). At M=4 K=7168 HBM-bound regime, padding-waste advantage of t16 is offset by t32's pipelining advantage. **For real M=4 wins need either**: (1) new ASM 1stage kernel author at smaller M-tile (multi-week, dead per F1), (2) smaller MFMA primitive (out of scope), (3) reducing MoE call count (Phase B F3 attempt), or (4) acceptance amortization (Eagle3-class out of scope).

**State**: t16 registry expansion KEPT (architectural consistency with stage2 — t16 was already in stage2 registry). kb extension REVERTED. CSV pristine.

### Phase A Day 1 final — Lever G `PHASE_RELAXED_TOP_N 10→12` (PROVISIONAL WIN)

Outside-the-box: Phase 1a memory only tested STRICTER direction (9, 0.55) which regressed +0.51 ms; PERMISSIVE direction (10→12) was UNTRIED.

Single-line edit at `/app/ATOM/atom/model_ops/rejection_sampler.py:22`:
```python
PHASE_RELAXED_TOP_N = 12  # G: 10->12 (more permissive accept inside thinking)
```
PHASE_STRICT_TOP_N=8 unchanged → outside-`<think>` GSM accuracy preserved.

**Dev-bench 4-iter median(iter2,3,4)**: TPOT **5.5141 ms = −0.200 ms vs 5.7141 dev-bench baseline**.
**GSM canary 3-iter median**: **0.9401 PASS** (margin +0.0076 over 0.9325 gate).
**Backup**: `/tmp/rejection_sampler.py.preTrack1_G_top12_020248`. md5 1e8ae536→fbe06529.

**Mechanism**: wider top-N inside `<think>` captures more near-miss draft tokens target validates → MTP acceptance ~69% → ~72-75%. STRICT path (final-answer emission) unchanged.

**G v2 (TOP_N=14) REGRESSED** — saturated at 12; 14 admits too many drafts target rejects, drafter waste outweighs acceptance gain.

**STATUS**: provisional. Per kimbochen-only directive, **lever G is NOT promoted until measured under `dsr1_benchmark perf` 4-iter median(2,3,4) on `dsr1_fresh`**. Dev-bench numbers are guidance only.

### Forward levers (May 04, post-Phase-A/B exhausted)

1. **Kimbochen-confirm Lever G** (TOP_N=12) — `dsr1_benchmark perf` 4-iter on `dsr1_fresh`. Snapshot if median(2,3,4) shows ≥0.10 ms TPOT improvement + GSM_med ≥ 0.93.
2. **Phase C C1 — MLA q_proj ‖ concat_and_cache stream split** (architectural call-graph). Reuse `dual_stream_moe_forward` pattern at `deepseek_v2.py:912`. Env-gated `ATOM_MLA_OVERLAP=1`. Predicted −0.03 to −0.08 ms.
3. **Lever A — MoE router-decision cache** across decode steps (algorithmic, compounds with G).
4. **Lever D — skip silu requant via bf16×fp4 MFMA** (kernel-level, requires CDNA4 instruction availability check).
5. **Phase B F3** — `moe_op_mxfp4_silu_fused` Triton dispatcher (drop if regress, Triton historically slower than CK on MI355X).
6. **Phase C C2** — `ATOM_SPEC_V2_OVERLAP=1` env flip (already wired at `eagle.py:299`). −0.02 to −0.05 ms.
7. **Phase C C3** — R2-C `_r2_smallm_stage2_wrapper` env-gated dispatcher wire. −0.05 to −0.10 ms (single-shape lever).

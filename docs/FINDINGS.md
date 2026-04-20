# MASTER_FINDINGS — AMD Phase 2 Hackathon
## Last Updated: 2026-04-20 session-10 — P0-P8 KERNEL CAMPAIGN LOCKED, bottleneck VALIDATED

## 🎯 SESSION-10 BREAKTHROUGH (Apr 20): BOTTLENECK DEFINITIVELY IDENTIFIED

**After 10 sessions of debugging, the REAL bottleneck is identified and VALIDATED** (V1/V4/V5 overlap parsers, not hypothesis):

- **`hipGraphLaunch` = 77.7% of wall time** at CONC=4 (915 calls × 63 µs each, torch.profiler)
- **NOT overlapped with GPU work**: V5 decode-window GPU util 2.2% (opposite of Kimi where hipEventSync was 99.4% overlapped)
- **Root cause**: each decode-step HIP graph contains ~1525 nodes (61 transformer layers × ~25 kernels/layer)
- **Per-node submission cost ~40 ns already optimal** — the problem is NODE COUNT, not per-node cost

### What this overturns
- ❌ **HipKittens MLA C1 port (sessions 8/9 work)**: targets kernel that is 4.3% of 2.3% wall = 0.1% wall. Wrong lever in isolation.
- ❌ **F0a hipEventSync fix**: DSR1 hipEventSync is only 3.1% wall (not Kimi's 88%). Wrong pattern.
- ❌ **Compact-experts MoE**: 0.5% wall. Too small.
- ❌ **`hipGraphInstantiateFlagDeviceLaunch`**: symbol exists on ROCm 7.2.2 but returns `hipErrorNotSupported` functionally. Dead lever.

### What unlocked
- ✅ **Node-count reduction** via kernel fusion is the primary lever
- ✅ **HK MLA C1 port CONTINUES as Phase P5** but with NEW root-cause approach (fix AITER metadata builder `kPackedQoLenPerWg=128` hardcode at `csrc/kernels/mla/metadata/v1_1_device.cuh:662` instead of Python preprocessor)
- ✅ **MTP=4/5 via HK qseqlen=5/6 kernels** is the E2E gate unlock
- ✅ **vLLM #27224** (host overhead between decode steps) is a direct backport target

### Current P0-P8 campaign
See [Current_plan.md](Current_plan.md) for phase table. Master plan at [`../../.claude/plans/fizzy-toasting-teacup.md`](file:///C:/Users/danis/.claude/plans/fizzy-toasting-teacup.md).

Projected trajectory: floor 1351/6.66/150/7221/0.934 (1/4) → P3 2/4 → P7 3/4 → P8 **4/4** at ~2280 Thr/GPU, 3.95 ms TPOT, 4350 ms E2E, 0.93+ GSM8K.

Bottleneck details: [Bottleneck.md](Bottleneck.md). Campaign memory: [`memory/project_dsr1_kernel_campaign_apr20.md`](../../../.claude/projects/c--Users-danis-OneDrive-Desktop-AMD/memory/project_dsr1_kernel_campaign_apr20.md).

---

## 🔒 CURRENT CONC=4 FLOOR: stock-canonical (Apr 20 2026)
**Model**: `amd/DeepSeek-R1-0528-MXFP4` (HuggingFace canonical, NO merged checkpoints)

**Config**: TP=4 SR + MTP=3 + TBO prefill + 53-row filtered BF16 CSV + relaxed MTP (8, 0.5) + DUAL_STREAM=1024 + NCCL=16 + max-batched-tokens=65536 + QUICK_REDUCE FP + CAST_BF16_TO_FP16 + AITER_VSKIP=0 + HIP_FORCE_DEV_KERNARG=1

**Numbers**: 1351 thr/GPU (÷4), 6.66 ms median TPOT, 150.23 interact, 7221 ms median E2E, **0.934 GSM8K**. **1/4 gates**.

**Result file**: `/projects/teamA/danish/experiments/stock_floor_MTP3_TBO_QR_canonical.json`

**Repro**: `dsr1-hackathon-dec073/docs/best_reproduce.md` (full launch + env vars + bench command).

### Why merged DSR1-drafter-FP4 dropped (Apr 20)
- Daniel Huang mergability rule: AMD reference benchmark uses canonical model
- Empirical: stock vs merged delta 0.7% (within run-to-run variance)
- Reproducibility: stock = single artifact; merged = custom transplant recipe

### Historical floor lineage (replaced by stock canonical)
- DEC-073 floor: `1270/6.80/147.1/7318/0.934` (Apr 18, merged DSR1-drafter-FP4, ROCm 7.1.1 stack)
- "1361 floor": `1361/6.35/157.55/6842/0.934` (Apr 18 evening, merged + Phase 3 sync-fuse)
- **Apr 20 stock canonical**: `1351/6.66/150.23/7221/0.934` (replaces above as committable floor)

## 📊 Push-session DEC lineage (Apr 17-18)

| DEC | Change | Result |
|---|---|---|
| DEC-066 | pre-push floor (9-row BF16 CSV) | 1221/6.73/148.6/7663/0.9378 |
| DEC-069 | Phase 4A v4 drafter HIP graph (post-crash fix w/ graph_capture ctx) | NULL — DEC-057 already proved Python gap ≈ 0 |
| DEC-071 | BF16 decode tune full sweep (97 rows) | 1267/6.96/143.8/7495/0.9303 (marginal +3.8% thr) |
| DEC-072 | BF16 prefill tune (M=1024-8192) | **DEAD — GSM8K 0.865 crash from numerical drift** |
| **DEC-073** | **Relaxed MTP (8, 0.5)** | **1270/6.80/147.1/7318/0.934 (CURRENT BEST)** |

## 🚨 FINAL PUSH MODE — Apr 17 night → Apr 18 night (SINGLE MISSION)

**User declaration**: Block 3 / Kimi Block 2 / May 15 horizon all DROPPED. Pass all 4 CONC=4 gates by Apr 18 night or submit at sub-rank.

### Remaining levers after DEC-073

| # | Lever | Expected | Status |
|---|---|---|---|
| Tree spec top-2 @ i=2 (next) | DEC-074 | TPOT −0.5 ms → 6.30 | IN PROGRESS (tree-spec code next) |
| Drafter MoE BF16→MXFP4 requant | DEC-075 | TPOT −0.5 ms | pending |
| ATOM #421 wire-in | DEC-076 | TPOT −0.2 ms | pending |
| rocprof + custom kernel | backup | unknown | pending |

### Locked dead levers (DO NOT retry)

- Phase 4A drafter cudagraph / Phase 4B async scheduling (Python gap ≈ 0)
- BF16 PREFILL CSV tune (GSM8K crashed at 0.865, DEC-072)
- AITER #2727 simple cherry-pick (a16w16 BF16 only, we use a8w8 FP8)
- AITER #2620 full cherry-pick (API drift to flydsl 0.1.3.1)
- ATOM #421 simple cherry-pick (Qwen-only dispatch; wire-in still on table for DEC-076)
- QuickReduce INT4, TP=2 SR, TP=4×DP=2, AITER v0.1.12, prefix caching, MTP-MoEFP4 model
- Env regressions: GPU_MAX_HW_QUEUES=5, OMP=1, triple-fusion env vars with relaxed MTP

### Realistic outcome

- **Tree spec lands**: 2/4 gates (interact + GSM8K). Submit sub-rank.
- **Tree spec + drafter requant**: possibly 3/4.
- **4/4 at CONC=4**: < 20% probability. E2E gate requires structural shift beyond what we have in 10 hours.

### Rollback path to DEC-073 if tree spec breaks

1. Restore rejection_sampler.py from `.bak_before_8_0.5_*`
2. Re-apply (8, 0.5) manually if needed
3. CSV at `/tmp/dsv3_bf16_tuned_gemm.csv.DEC071_0512` → restore if dedup cycle broke it
4. Relaunch server with DEC-073 config

## Last Updated: 2026-04-17 LATE NIGHT — FINAL PUSH MODE (Block 3 + Kimi + May 15 horizon DROPPED)

## 🚨 FINAL PUSH — Apr 17 night → Apr 18 night (SINGLE MISSION)

**User declaration**: Block 3 / Kimi Block 2 / May 15 horizon all DROPPED. Pass all 4 CONC=4 gates by Apr 18 night or submit at sub-rank.

### Current state (04:11 UTC Apr 17)
- BF16 decode CSV tuner RUNNING (~45 min left, ETA 04:55 UTC). All 200+ decode shapes untuned per DEC-057 profile.
- Phase 4A v4 shipped DEC-069 with NULL result (0 ms cut). Patch stays as harmless infra.
- Next 24 hrs will apply full lever stack.

### Full lever stack being applied (10 levers, nothing dropped)

| # | Lever | Target ms | Expected | Time | Prob |
|---|---|---|---|---|---|
| 1 | BF16 DECODE tune | 4.57 | −1.5 | 45 min | 99% |
| 2 | rocprof HW counter profile | diagnostic | unlocks #10 | 30 min | 100% |
| 3 | BF16 PREFILL tune (M=1024-8192) | TTFT 377 | −150 ms | 30 min | 95% |
| 4 | AITER PR #2620 fused mxfp4 quant moe sort | 2.13 | −0.3 | 45 min | 85% |
| 5 | AITER PR #2727 MI350 MLA ps shapes | 3.42 | −0.1 | 30 min | 80% |
| 6 | ATOM PR #421 gated_rmsnorm_quant | 1.02 | −0.2 | 60 min | 75% |
| 7 | Relaxed MTP (8, 0.5) | toks/fwd | TPOT −0.2 | 30 min | 70% |
| 8 | QuickReduce non-INT4 | 2.96 | 0 to −0.3 | 30 min | 40% |
| 9 | Minimal tree spec (top-2 at i=2) | toks/fwd | +0.1-0.3 | 4-6 hrs | 30% |
| 10 | Custom kernel rocprof-informed | ? | ? | 6+ hrs | 20% |

**Realistic stacked**: TPOT 4.43 + TTFT 227 → E2E 4763 ✅ + interact 226 ✅ + thr 1620 ✅ + GSM8K ≥ 0.93 ✅ → **4/4 gates**. Probability 50-60%.

### Binding gate math
- E2E ≤ 5000 → TPOT ≤ 4.52 ms (from 6.73 current) → need −33%
- Thr ≥ 1500 → TPOT ≤ 5.63 ms
- Interact ≥ 165 → TPOT ≤ 6.06 ms
- GSM8K ≥ 0.93 (already passed)

### Active files
- Plan: `C:\Users\danis\.claude\plans\fizzy-toasting-teacup.md`
- Memory: `memory/project_final_push_apr17_18.md` (push mission) + `project_wall_clock_budget_hard.md` (ground truth) + `project_sota_apr17_intel.md` (PR list) + `feedback_pre_measure_or_dont_ship.md` (rule)
- Desktop: `Current_plan.md` + `Best_atom_dsr_cncc4/best_reproduce.md` + this file + `daily_log.md`

### If 4/4 missed
Submit best-effort at each CONC for sub-rank score. User accepts this is final iteration.

## Historical note (below is pre-final-push content, superseded)

## Last Updated: 2026-04-17 Day 3 — CONC=4 floor LOCKED at DEC-066, Phase 4A null, plan reset to measurement-driven sprint

## 🔒 CONC=4 LOCKED FLOOR: DEC-066 (1/4 gates)
- Thr/GPU: **1221** (gate ≥1500, ❌ −19%)
- Median TPOT: **6.73 ms** → Interact **148.6** (gate ≥165, ❌ −10%)
- Median E2E: **7663 ms** (gate ≤5000, ❌ +53%)
- GSM8K: **0.9378** (gate ≥0.93, ✅)
- Config: TP=4 SR, ATOM 108a70e, aiter f8c1d76bd, flydsl 0.1.2, relaxed MTP (7, 0.4), DUAL_STREAM=256, NCCL_MIN_NCHANNELS=16, 9-row BF16 tuned CSV. Repro in `Best_atom_dsr_cncc4/best_reproduce.md`.

## 🚫 Phase 4A v4 drafter HIP graph: NULL result (DEC-069, 2026-04-17)
- Patch correct: graph captured, replay stable, accept rate preserved (65.73% vs 62.5% DEC-066), GSM8K 0.9401
- TPOT delta: +1.3% (noise). **Zero TPOT cut.**
- Root cause: DEC-057 had already proven Python/CPU gap is ≈0 (kernel time = step time within 0.3%). Phase 4A optimized Python launch overhead that didn't exist.
- Plan-level miss: proposed intervention without checking prediction against measured budget. Third occurrence of same anti-pattern per memory (feedback_profile_before_intervene.md).
- Phase 4B async scheduling DROPPED by same root cause.
- Patch stays in eagle.py as harmless infra (may help Block 3 tree spec).

## 🎯 Active plan: `C:\Users\danis\.claude\plans\fizzy-toasting-teacup.md`
- Measurement-driven sprint (Apr 17-24 Block 1, Apr 25-May 4 Kimi Block 2, May 5-15 Block 3 tree spec)
- Rule: no intervention ships without (target ms + mechanism + expected delta + pass/fail gate + post-measurement). Enforced from DEC-070 onward.
- Apr 18 AM: CONC=32 baseline. Apr 18 PM: full BF16 tune sweep (Lever #1, target 4.57 ms, expected −1.5 ms TPOT).
- Apr 19: AITER PRs #2620 (fused mxfp4 quant moe sort) + #2727 (MI350 MLA ps mode).
- Apr 20: ATOM PR #421 (gated_rmsnorm_quant) + QuickReduce test.
- Apr 21: CONC=128 baseline.
- Apr 22-23: GSM8K stability + lock. Apr 24: submit.

## Last Updated: 2026-04-16 Day 2 — Fresh profile, infrastructure separation, all-in CONC=4 plan

## 🏗️ INFRASTRUCTURE — Track Separation (2026-04-16)

**Two competition tracks run in parallel on the same 8-GPU MI355X node, fully isolated:**

| Track | Container | GPUs | Port | ATOM source | aiter source | Model | Status |
|---|---|---|---|---|---|---|---|
| **DSR1 Track 1** ($350K) | **`danish_atom_main`** ⭐ | **0-3** (card1/9/17/25, renderD128/136/144/152) | internal | `/projects/teamA/danish/repos/ATOM_main` (108a70e + local mods) | `/projects/teamA/danish/repos/aiter` (f8c1d76bd + re-export patch) | `amd/DeepSeek-R1-0528-MXFP4` | **ACTIVE — Day 2 execution** |
| **Kimi Track 2** ($650K) | **`danish_kimi`** | **4-7** (card33/41/49/57, renderD160/168/176/184) | 8889 | `/projects/teamA/danish/kimi/ATOM_kimi` (fresh clone) | `/projects/teamA/danish/kimi/aiter_kimi` (fresh clone) | `amd/Kimi-K2.5-MXFP4` | **Ready for separate Opus** |

**Zero shared mutable state between tracks.** Each has its own ATOM clone, aiter clone, and GPU set. HF model cache (`/projects/teamA/hf_cache`) is shared read-only. Bench scripts (`amdgpu_bounty_optimization/`) are shared read-only.

**`danish_atom`** (old pristine fallback) is parked — crashes on DSR1 MoE drafter. Do NOT use for benches.

### Disk cleanup done (2026-04-16)
- Deleted ~1.2 TB of `gpucore.*` crash dumps from `/projects/teamA/danish/repos/ATOM_main/`
- Deleted `/projects/teamA/danish/competition_sglang/` (SGLang Day 1 attempt, dead)
- Disk free on `/dev/md0`: ~20 TB

### How to access each track
```bash
# DSR1 (this Opus)
~/bin/docker exec -it danish_atom_main bash

# Kimi (separate Opus instance)
~/bin/docker exec -it danish_kimi bash
```

---

## 🔬 Day 2 FRESH PROFILE — Real Bottleneck Picture (DEC-057, 2026-04-16 03:56 UTC)

**Replaces stale DEC-055.** Captured at EXACT DEC-056 floor config: DUAL_STREAM=256, OSL=1024, TP=4 SR, MTP=3, relaxed (5, 0.3). Via `profile_offline.py`. Total decode: 21.8 ms/step (matches bench 21.73 within 0.3%).

| Category | ms/step | % | vs DEC-055 | Key finding |
|---|---|---|---|---|
| **MoE GEMM (FlyDSL)** | **5.89** | **26.2%** | was 57% | DUAL_STREAM overlap + OSL=1024 rebalanced |
| **BF16 GEMM (UNTUNED!)** | **4.57** | **20.3%** | was 13% | **ALL decode shapes on slow torch default — LM head, MLA projections** |
| **AllReduce/NCCL** | **2.96** | **13.2%** | was 5.5% | **5.4× bigger than DEC-055 — hidden by short output** |
| MLA attention | 2.26 | 10.1% | 12% | Kernel = `qh32_qseqlen4` — **NOT padded 32→128** |
| MoE routing | 1.39 | 6.2% | 4.5% | MoeSorting + topk + quant |
| MLA reduce | 1.16 | 5.2% | — | 17236 calls at 17.8 μs |
| RMSNorm | 1.02 | 4.5% | 2% | Larger at OSL=1024 |
| Quant/dequant | 0.74 | 3.3% | — | FP4 quant for MoE input |
| Other | 1.74 | 7.8% | 8% | — |

**Critical corrections:**
1. **BF16 GEMM is 20% not 13%** — LM head `M=16 N=32320 K=7168` alone = 7.2% of all decode, running on untuned `torch solution:0`
2. **AllReduce is 13% not 5.5%** — NCCL generic kernel = 530 ms (8.8%). DEC-055 understated because `--output-length 32`
3. **MLA is NOT padded 32→128** — the 2×16 dispatch analysis is WRONG for ATOM commit 108a70e. `qh32` kernel handles 32 heads natively
4. **The 1.5 ms "stranded cost" is NOT Python overhead** — it's AllReduce + routing + quant that DEC-053 analytical solve missed

**Active plan**: BF16 tune (Phase 1) + AllReduce optimization (Phase 2) + Tree Spec port (Phase 3). See `C:\Users\danis\.claude\plans\fizzy-toasting-teacup.md`.

---

## 🚨 CANONICAL DSR1 CONTAINER: `danish_atom_main` 🚨

**ALL DSR1 benches run in `danish_atom_main` — never in `danish_atom`.** Container details:

| Property | Value |
|---|---|
| ATOM source | editable `/projects/teamA/danish/repos/ATOM_main` at commit `108a70e` |
| Local mods | `(5, 0.3)` relaxed MTP hardcoded in `rejection_sampler.py`, `num_kv_splits=None` in `attention_mla.py`, Iter 6 decode-metadata patches |
| aiter | `/projects/teamA/danish/repos/aiter` at `f8c1d76bd` with LOCAL re-export patch (`concat_and_cache_mla, fused_qk_rope_concat_and_cache_mla` appended to `__init__.py`) |
| flydsl | 0.1.2 pip installed, `is_flydsl_available()=True` |
| v917 MoE patch | **DEAD** — reverted. Three crashes Day 1 (ZeroDivisionError, 512-thread launch check, API drift). `fused_moe.py` restored from `.bak_before_v917_1743`. |

**Pre-flight verify before every launch** (all must pass):
```bash
python3 -c 'import atom; print(atom.__file__)'
# → /projects/teamA/danish/repos/ATOM_main/atom/__init__.py
python3 -c 'from atom.model_ops import rejection_sampler as r; print(r.RELAXED_TOP_N, r.RELAXED_DELTA)'
# → 5 0.3
python3 -c 'import aiter; print(hasattr(aiter, "concat_and_cache_mla"))'
# → True
```

## 🎯 CURRENT CONC=4 COMMITTABLE FLOOR (DEC-056, 2026-04-15 evening)

**Container**: `danish_atom_main` (canonical; see above)
**Config**: TP=4 SR + hardcoded relaxed MTP `(5, 0.3)` + `ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=256`

**Numbers via official `./dsr1_benchmark perf`**:

| Gate | Value | Target | Status |
|---|---|---|---|
| **Thr/GPU** | **1209** | ≥1500 | ❌ −19% |
| **Median TPOT** | **6.89 ms** | — | — |
| **Interactivity** | **145** | ≥165 | ❌ −12% |
| **Median E2E** | **7464 ms** | ≤5000 | ❌ −49% |
| **GSM8K** | **0.9363** | ≥0.93 | ✅ |

**1/4 gates passing.** New floor replaces the 1191/7.40 baseline. Full repro command + raw output in `daily_log.md` DEC-056 section. **This is the Day 3 fallback floor** if SGLang pivot and tree spec both fail.

**Key finding**: `ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD` at default 16384 was preventing MoE stream overlap from firing at our bs=16 effective decode batch. Lowering to 256 activated overlap and saved −0.51 ms median TPOT (−6.9%), which translates to −508 ms median E2E (−6.4%) and +10 tok/s/user interactivity (+7.4%) on a single env var change.

**DEC-055 (v917 MoE port) is DEAD** (−3% regression). **GPU_MAX_HW_QUEUES=5 is DEAD** (−4% regression, Compass warning confirmed). Do not re-test either.

## 🚨🚨 Day 1 afternoon findings (DEC-050..054)

**DEC-051**: Chat template isolated as 100% of harness gap. Drafter accept rate drops from 84% → 57% on chat-wrapped random prompts. Depth-3 accepts crash from 72% → 32%. Position-0 rejects go from 2.7% → 20.1%.

**DEC-052**: Threshold tuning (5, 0.3) vs (3, 0.2) — GSM8K min 0.9371 vs 0.9333 (better margin), but perf essentially flat (1191 vs 1169 thr/GPU). Looser thresholds don't fix "drafter predicts wrong distribution". Locked (5, 0.3) as new floor for GSM8K margin.

**DEC-053**: MTP=1 vs MTP=3 test on chat-template. MTP=1 hits 84% accept (vs MTP=3's 60%) but is 7% WORSE overall (1111 vs 1191). Solved for m and d: **main_fwd ≈ 10.9 ms, drafter_fwd ≈ 3.1 ms**. Drafter is cheap (25% of main). **Real bottleneck = main model forward pass at ~11 ms, not drafter.**

**DEC-054**: 2026-04-14 profile (TP=8 ISL=128 strict) is STALE for our config. Fresh profile at TP=4 SR ISL=8192 MTP=3 relaxed needed BEFORE any kernel-level optimization. Profile in flight at `/projects/teamA/danish/repos/trace/day1_tp4sr_isl8192_real/`.

## 🚨🚨🚨 DEC-050 — DEC-047 floor was measured with wrong harness (2026-04-15)

## 🚨🚨🚨 DEC-050 — DEC-047 floor was measured with wrong harness (2026-04-15)

**`./dsr1_benchmark perf` is the official scoring tool.** `atom.benchmarks.benchmark_serving` is the internal dev bench. **They give ~22% different numbers on the same server.**

Side-by-side on the same live server with hardcoded relaxed MTP (3, 0.2):

| Harness | Total thr | Thr/GPU (÷4) | Median TPOT | Interact | Median E2E |
|---|---|---|---|---|---|
| Internal `atom.benchmarks.benchmark_serving` | 5833 | **1458** | **5.47** | **183** | ~5905 |
| Official `./dsr1_benchmark perf` | 4561 | **1140** | **7.77** | **129** | **8364** |

**Yesterday's "DEC-047 floor" (1470/5.54/180.5) was via internal bench. Real official-harness gate status at CONC=4 is 1/4 passing (only GSM8K), not 2/4.** All future committable numbers come from the official tool only. See `feedback_bench_harness_matters.md` in memory. **Harness gap itself is the biggest single CONC=4 optimization target** — close it → +22% for free.

Also confirmed: tool's `tput_per_gpu = total/8.0` is hardcoded and wrong per rules. Ziguan Discord 2026-04-15 07:10: "total_token/s/4" at TP=4. Compute per-GPU manually.

## 🚨🚨🚨 HARD TIMELINE (Danish 2026-04-15)

**30-day budget Apr 15 → May 15 = 3 × 10-day blocks:**
- **Block 1: DSR1 baseline** — Apr 15-24 (10 days)
- **Block 2: Kimi K2.5 baseline** — Apr 25 - May 4 (10 days)
- **Block 3: Exceed by 28%** — May 5-15 (10 days)

**Inside Block 1: 3 days per CONC, no exceptions.**
- CONC=4: Apr 15-17 (TODAY = Day 1)
- CONC=32: Apr 18-20
- CONC=128: Apr 21-23
- Buffer + first submit: Apr 24

**CONC=4 RULE:** TP=4 SR or TP=2 SR ONLY. **NEVER TP=8.** TP=8 cannot reach 1500 tok/s/GPU at CONC=4 mathematically — `/num_GPUs_used = 8` divisor caps it at ~580/GPU even with perfect kernels. TP=4 SR gets `/4` (currently 1470 with relaxed MTP, −2% noise from gate). TP=2 SR gets `/2` (theoretical 2200-2900/GPU but interact + E2E likely fail; previously crashed in DEC-044, retry pending).

**No multi-day kernel ports in Block 1.** Tree speculation (4-6 day port) is deferred to Block 3.

**Lock and move on.** If a CONC plateaus by end of its 3-day window, lock the floor as committable, move to next CONC. Come back in Block 3.

## Last Updated: 2026-04-13 END OF SESSION 6A (post-deep-research, execution plan locked Apr 14-28)

## 🎯 ACTIVE EXECUTION PLAN (Apr 14 onward)

**READ FIRST**: `C:\Users\danis\.claude\plans\fizzy-toasting-teacup.md` ("DSR1 Path-to-Baselines Execution Plan")

**Supporting memory files (in priority order)**:
1. `project_dsr1_quickstart_card.md` — 2-min state snapshot (READ THIS FIRST after context compact)
2. `project_dsr1_research_findings_session6a.md` — 20 concrete findings with source URLs
3. `project_dsr1_tp4_single_replica_alive.md` — TP=4 measurements from Session 6A

**Current phase**: Phase 0 (pre-execution setup) — not yet started. Execute in order: Phase 0 → 1 → 2 → 3 → 4 (if TP=4 multi-config alive) → 5 (custom kernels only if needed) → 6 (polish) → 7 (submit).

**Blocker for Phase 4**: need Daniel Discord reply confirming `num_GPUs_used = 4` reporting is allowed. Phases 0-3 and 5-7 are independent of this answer and can proceed without it.

## 🔬 Session 6A Research — concrete configuration gaps (2026-04-13 evening)

Three parallel research agents found WHY our 738/2345/3555 is half of baseline. The answer is configuration, not kernel quality. Ranked by impact:

### Top 7 actionable findings for Phase 1 tomorrow

1. **WRONG MODEL**: swap `amd/DeepSeek-R1-0528-MXFP4` → `amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4` (has quantized MTP layer weights). ATOM PR #411 publishes 29758 tok/s system (= 3720/GPU) at CONC=128, same workload, with our exact recipe. **Model swap is +4.6% for free.**
2. **`GPU_MAX_HW_QUEUES=5`** is a hidden prerequisite for dual-stream MoE to actually overlap (ATOM PR #499 body verbatim). Our dual-stream has been firing but NOT overlapping.
3. **Missing CLI flags**: `--async-scheduling --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' --no-enable-prefix-caching`. ATOM docs call FULL_AND_PIECEWISE "the most performant mode for most models". We run default PIECEWISE.
4. **`ATOM_USE_TRITON_GEMM=1 + ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION=1`** pair unlocks the auto-disabled fusion that our logs warn about.
5. **`ATOM_USE_TRITON_MXFP4_BMM=1`** — untested, literally targets our MXFP4 BMM in MLA attention.
6. **`ATOM_ENABLE_RELAXED_MTP=1`** — merged via PR #411, requires MTP-MoEFP4 model, increases MTP acceptance 81% → 86%.
7. **Pull ATOM main** (past commit 38d0d7f374 #547 stream-parallel decode metadata) — 5 merged PRs ahead of our pinned 108a70e.

### Architectural findings

8. **MORI-EP WORKS on single-node 8-GPU** via `IntraNode` kernel using XGMI peer-to-peer (NOT multi-node only — this was my earlier mistaken assumption). Published MI355X EP8 bandwidth: 345 GB/s dispatch / 420 GB/s combine. Exact DSR1 command in ATOM PR #515: `MORI_SHMEM_MODE=ISOLATION MORI_SHMEM_HEAP_SIZE=6G ... --enable-dp-attention --enable-expert-parallel`. **Phase 2 of plan tests this.**
9. **SGLang + MORI PD disaggregation is DEAD** for single-node MI355X FP4 (SGLang issues #18006 + #21942 confirm broken upstream). **DROP from plan — saves 2-3 days.**
10. **vLLM env vars** off by default, never tested by us: `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1`, `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1`, `VLLM_ALL2ALL_BACKEND=mori`, `VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1`, `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1`. Tested in Phase 3 via atom-vllm plugin path.

### Hard-rule additions from research

11. **NEVER enable `--enable-tbo` on DSR1** — ATOM PR #515 measured -14 to -24% regression.
12. **GLM-5 recipe warning**: DP-attn + EP + fp8 KV may not mix when gqa=8. DSR1 has gqa=1 so probably fine, but start without fp8 KV when combining DP-attn + EP and re-add after baseline.
13. **Unverified claims** (treat as NOT load-bearing): "MORI-EP 82% MoE latency reduction" not found in any published source; "AITER sampling op 1.6× thr" has no corresponding env var in mainline vLLM.

### Conservative Phase 1 impact projection

| CONC | Current | + Tier 1 (+15-25%) | + MORI-EP (+10%) | AMD Gate | Projected Gap |
|---|---|---|---|---|---|
| 4 thr | 738 | ~870 | ~950 | 1500 | -37% (needs TP=4 divisor trick for CONC=4) |
| 32 thr | 2345 | ~2770 | ~3050 | 3900 | -22% |
| 128 thr | 3555 | ~4200 | ~4620 | 6000 | -23% |

**Best case after full plan**: 6-8 of 9 gates passing. CONC=4 closed via TP=4 multi-config + kernel work. CONC=128 is the last hold-out and may need compute-comm overlap kernel (Phase 5 kernel work).

---

## SESSION 6A HEADLINE — TP=4 single replica gives +27-52% per-GPU throughput (DEC-021 was wrong about TP=4 in general)

**The single biggest strategic finding of the entire DSR1 effort.** Session 5 DEC-021 declared "all TP<8 × DP variants dead" — that was right about TP=4 × DP=2 (multi-replica with DP, genuinely dead) but WRONG about TP=4 single replica (4 GPUs used, 4 idle, num_GPUs_used=4 in scoring formula). The benchmark binary `./dsr1_benchmark perf` divides by 8 hardcoded regardless of actual TP, hiding the TP=4 advantage. **The competition rules formula uses num_GPUs_you_used = 1, 2, ..., 8 — if you use 4 GPUs, divide by 4.**

### Session 6A measured TP=4 single replica (canonical workloads)

| CONC | TP=8 BEST BASE thr/GPU | **TP=4 single thr/GPU at num_GPUs=4** | Δ | TP=4 TPOT median | TP=4 Interactivity | TP=4 E2E | Status |
|---|---|---|---|---|---|---|---|
| 4 (40 prompts) | 738.93 | **1124.7** | **+52.2%** | 7.86 ms (+30%) | 127 ❌ broke | ~8424 ms ❌ | thr improved, gates broken |
| 32 (320 prompts) | 2345.57 | **3084.6** | **+31.5%** | 23.36 ms (+49%) | 42.8 ❌ broke | 24310 ms ❌ broke | thr improved, gates broken |
| 128 (1280 prompts) | 3555.19 | **4543.0** | **+27.8%** | 65.09 ms (+56%) | 15.4 ❌ worse | 67289 ms ❌ worse | thr improved, gates worse |

**Net at TP=4 single replica RAW**: 0/9 gates passing (was 3/9 at TP=8 because TP=4 broke previously-passing interactivity/E2E gates). TP=4 alone is a NET REGRESSION on gate count. **Must pair TP=4 with TPOT-cut interventions** to recover interactivity.

### Required TPOT cuts on TP=4 to pass gates

| CONC | TP=4 TPOT now | TPOT needed for gates | Required cut | Feasibility |
|---|---|---|---|---|
| 4 | 7.86 ms | ≤4.5 ms (165 interact + 5000 E2E) | **−43%** | tight, plausible with full Tier 1+2 stack |
| 32 | 23.36 ms | ≤14 ms (50 interact + 18000 E2E) | **−40%** | plausible with Tier 1+2 stack |
| 128 | 65.09 ms | ≤18 ms (48 interact + 22000 E2E) | **−72%** | **NOT FEASIBLE** — must use TP=8 + different attack |

### Multi-config submission strategy (DEC-025, Daniel approved DEC-022 Session 5)

| CONC | Submission config | Why |
|---|---|---|
| **4** | TP=4 single + Tier 1 interventions | +52% throughput baseline + plausible TPOT cuts |
| **32** | TP=4 single + Tier 1 interventions | +31% throughput baseline + plausible TPOT cuts |
| **128** | TP=8 + PD disagg OR custom kernel work | TP=4 TPOT degradation impossible at CONC=128 |

### NEW MENTAL MODEL: configuration first, custom kernels last (DEC-026)

**AMD ships AITER. We have the same kernels they have at the kernel layer.** The 2× gap from 738 to the 1500 baseline is not a kernel-quality gap — it's a configuration gap. AMD's recipe is hiding in flags and architecture, not in custom kernels. Sweep all configuration moves (TP, EP, DP, scheduler, framework, multi-step, prefix caching, AITER op toggles, multi-config submission) BEFORE writing custom kernel patches. Custom kernels are the **scoring bonus** on top of configuration, not the qualification path. See memory file `feedback_configuration_first_kernels_last.md` for the full rule.

### Pre-execution check (Day 0 of next session)

**Discord Daniel** to confirm `num_GPUs_used = 4` reporting is allowed per the rules formula. The rules text says yes (`num_GPUs_you_used = 1, 2, ..., 8`), the bounty binary says no (always divides by 8). The whole TP=4 strategy depends on the rules text winning. Without confirmation, fall back to TP=8-only with smaller per-CONC gains.

### Intervention #1 (Session 6A) — num_kv_splits=16 → None — RESULT: FLAT (kept as upstream cleanup)

Patched ATOM `attention_mla.py:592` from `num_kv_splits=16` to `num_kv_splits=None` (let AITER auto-tune). Hypothesis: AITER's `get_meta_param()` heuristic picks i=8 at CONC=32 and i=2 at CONC=128 vs the hardcoded 16 (manually verified by hand calculation of the heuristic). Expected: -3-10% TPOT at higher CONCs.

**Actual measurement at canonical workloads (Session 6A)**:
- CONC=4: 738 → 749 thr/GPU (+1.4%), TPOT 6.07 → 5.92 ms (-2.5%) — flat
- CONC=32: 2345 → 2364 thr/GPU (+0.8%), TPOT 15.65 → 15.38 ms (-1.7%) — flat
- CONC=128: 3555 → 3576 thr/GPU (+0.6%), TPOT 41.61 → 41.66 ms (+0.1%) — flat

**All within noise. Net ~+1% across CONCs. No regression anywhere.** Most likely cause of the flat result: ATOM uses persistent mode (passes `work_meta_data`), and per the WebFetch agent's reading of AITER source line 269, persistent mode internally overrides `num_kv_splits` to `cu_num` regardless of what the caller passes. So our `None` is functionally equivalent to `16`. The patch is harmless cleanup (uses the documented-correct API per AITER's "for experts only!!!" comment) but doesn't change kernel behavior in our config. **Keeping the patch as upstream-mergeable cleanup, not committing as a perf win.**

### Intervention #2 (Session 6A) — q→FP8 cast — INVALID AS PLANNED

The Intervention Plan v2 said to uncomment the `q → q.to(dtypes.fp8)` block at `attention_mla.py:479-481`. Direct source read of the actual file shows **no such commented block exists** at those lines. WebFetch agent hallucinated the source. Lines 470-525 are inside `_forward_prefill_mla()` (line 449) and contain sparse-attention handling + the prefill `mla_decode_fwd` call site. There's no q→FP8 cast (commented or uncommented) anywhere in the function. **Skip Intervention #2.** Possibly there's a different code path in a different ATOM version that has this cast — but it's not in our checkout.

### Engineering model deliverables built in Session 6A (memory files)

Foundational reading completed via direct source-read (WebFetch from `ROCm/ATOM` and `ROCm/aiter` GitHub mains) instead of running blind experiments:

1. **`project_dsr1_latency_budget.md`** — Wall-clock TPOT decomposition. CONC=4 first cut: 5.97 ms = ~7.6 ms/forward GPU kernel time (66%, upper bound) + ~3.9 ms/forward non-kernel residual (34%, lower bound for Python + drafter + comm). The 34% residual is structurally consistent with MTP drafter running in Python OUTSIDE the cudagraph (confirmed by source).

2. **`project_atom_execution_flow.md`** — ATOM source code trace: `EngineCore.busy_loop()` → `_process_engine_step()` (line 164) → `runner_mgr.call_func("forward")` → `ModelRunner.forward()` (line 1717) → `run_model()` (decides cudagraph replay vs eager at line 1619) → `postprocess()` (Python, OUTSIDE cudagraph, line 1694). MTP `propose_draft_token_ids()` runs in Python on next batch cycle. **Biggest CONC=4 lever: get drafter into cudagraph (4-5 day patch).** Also: layer 0 input_norm AllReduce is NOT fused (deepseek_v2.py:1695 gates on `layer_idx > 0`), MTP drafter exists at `model_runner.py:1745`.

3. **`project_aiter_kernel_map.md`** — AITER kernel dispatch table for DSR1. `mla_decode_fwd` signature, `get_meta_param()` heuristic (manually computed for our shapes), `fused_allreduce_rmsnorm` dispatch path, FlyDSL stage1+stage2 wrapper paths. Also: vLLM uses persistent mode via `get_mla_metadata_v1()` — DIFFERENT code path than ATOM's hardcoded num_kv_splits=16.

4. **`project_framework_comparison_dsr1.md`** — ATOM vs SGLang vs vLLM matrix. Key findings: ATOM has fused_allreduce_rmsnorm kernel (saves ~5-10% TPOT), SGLang uses scheduling-overlap instead (no fused kernel), SGLang has `mooncake/`, `mori/`, `nixl/` PD disaggregation backends that ATOM doesn't have. **For DSR1 staying on ATOM is correct UNLESS PD disagg becomes the only path to CONC=128 6000 gate.**

5. **`project_dsr1_tp4_single_replica_alive.md`** — TP=4 measurements + reproduction commands + pattern analysis (NEW Session 6A). The headline document for the multi-config strategy.

6. **`project_dsr1_intervention_path_v2.md`** — 14-day execution plan for Apr 14-28 (NEW Session 6A). Day-by-day priority list.

7. **`feedback_configuration_first_kernels_last.md`** — The strategic mental model rule (NEW Session 6A).

8. **`feedback_build_model_before_optimizing.md`** — Session 5 lesson, still load-bearing for future sessions.



## STRATEGIC REFRAME (Session 5 research — 2026-04-13)

Three research agents returned intel that reframes the whole competition posture. Read this before planning any further DSR1 work.

### 1. Our DSR1 numbers likely dominate the field

AMD's OWN published DSR1 best per-GPU numbers (from `ROCm/ATOM/recipes/DeepSeek-R1.md`):

| AMD Published | Per-GPU |
|---|---|
| CONC=128 FP8 TP=8 MI300X ISL=1024 | 534 |
| CONC=256 FP8 TP=8 MI300X ISL=1024 | 755 |
| **CONC=128 FP8+MTP3 TP=8 MI300X ISL=1024** | **864 (their best)** |

**Our BEST BASE**: CONC=128 MI355X MXFP4 ISL=8192 **= 3555 tok/s/GPU**, 4.1× higher.

Caveat: apples-to-oranges (MI300X vs MI355X, FP8 vs MXFP4, ISL=1024 vs 8192). But anchors one fact: **the 6000 tok/s/GPU CONC=128 target is an AMD-internal stretch goal, not a number competitors are publicly hitting**. Track 1 is capped at 10 finalists. Each finalist gets guaranteed $10k + shot at grand prize pool.

**Cannot verify our actual rank without opening leaderboard URLs in a browser** (Gradio spaces at `daniehua/dsr1-fp4-isl8192-osl1024-conc{4,32,128}.hf.space` — scraping is blocked, need real browser session).

**Strategic implication**: if a browser check confirms we're top-3-5, further DSR1 chasing is diminishing returns. **Remaining competition time is better spent on Kimi K2.5 (Track 2, $650k) + mergeable upstream contributions.**

### 2. Kimi K2.5 is a multi-day pivot, NOT a model swap

Kimi K2.5 vs DSR1-0528 deltas (both are `DeepseekV3ForCausalLM`):

| Dim | DSR1 | Kimi K2.5 | Impact |
|---|---|---|---|
| `n_routed_experts` | 256 | **384** | FlyDSL tuned CSV won't match, needs re-tune |
| `num_attention_heads` | 128 | **64** | gqa_ratio halves, TP=8 → 8 heads/rank |
| MTP head | Yes | **No** (uses EAGLE3 via vLLM) | Lose biggest decode optimization |
| `rope_theta` | 10000 | **50000**, YaRN-32 | RoPE cache rebuild |
| Multimodal | — | MoonViT 400M vision tower | Extra non-quantized compute |

**AMD's published Kimi recipe**: `vllm/vllm-openai-rocm:v0.17.0` (not 0.15 not 0.18), ROCm 7.1.0, `VLLM_ROCM_USE_AITER=1`, TP=4, `--enforce-eager`. vLLM 0.15 is BROKEN for Kimi; needs backports from vLLM PRs #33320 and #34501.

**AMD's own Kimi K2-Thinking published ceiling** (4× MI355 MXFP4 ISL=1024/OSL=1024): CONC=128 = 837 tok/s/GPU. Much less explored than DSR1. **Track 2 $-per-engineering-hour is much higher.**

**Kimi effort estimate**: 1-3 days minimum (vLLM 0.17 pull + launch + GSM8K + FlyDSL 384-expert re-tune + EAGLE3 wiring + sweep + robustness). **Not a "while DSR1 runs" task.** Week 2 work with proper prep.

### 3. MI355X hardware realities (from research agent)

- **Realistic HBM3e sustained BW: 6.5-7.0 TB/s** (theoretical 8.0). **Up to 20% BW loss on naive row-major workgroup placement** — chiplet-aware scheduling is an untouched ~20% lever.
- **LDS: 160KB per CU** on CDNA4 (vs 64KB CDNA3). Read BW doubled. Big staging headroom that existing AITER kernels may not fully exploit.
- **Matrix cores**: FP8 ~20 PF (2× MI300X), MXFP4 ~40 PF dense / ~80 PF w/ sparsity.
- **Infinity Fabric**: 7 links × ~153 GB/s bidir per GPU → ~1.075 TB/s aggregate. TP=8 ring-allreduce bounded at ~150 GB/s.
- **256 active CUs** (8 XCDs × 32) vs 304 on MI300X.

**The `decode_qlen ∈ {2,4}` persistent kernel limitation — EXPLAINED**:
> At gqa_ratio=32 the wave holds 32 Q heads per CU. LDS budget for K/V tiles is the binding constraint — only qlen 2 or 4 leave enough banks free for the fp8 scale vectors without bank conflicts. The ASM kernel hard-codes double-buffered K/V tile into 160KB LDS. **Recompile with wider LDS staging (feasible on 160KB) could lift this.**

**This is a real upstream AITER PR opportunity** — a ~2-3 day kernel engineering task (outside our budget) but **filing the issue with the LDS-bank-conflict analysis is a 30-min, mergeable contribution** that AMD's kernel team would take seriously. It goes straight into our competition submission narrative as a "found a real bug, gave the fix roadmap" line item.

### 4. Theoretical TPOT floor for our workload

- DSR1-MXFP4 effective HBM reads per token ≈ 10-12 GB per GPU (sparse MoE, 9 of 256 experts active)
- At 6.5 TB/s sustained → **~1.54 ms theoretical floor per forward**
- With MTP=3 accept rate ~1.89× → **~0.82 ms effective TPOT floor**
- Current CONC=128 TPOT: 41.61 ms. **51× above theoretical floor.** Most of the gap is scheduler/prefill overhead, not kernel time. Fresh profile will confirm.

## SESSION 5 RESULTS (2026-04-13): TP=4/TP=2 × DP exhausted, no gate improvement

Every TP<8 variant was tested. All failed:

| Variant | Config | Result |
|---|---|---|
| Path A | TP=4 DP=2 bf16 no-MTP | GSM8K=0.9409 ✓, **memory access fault at CONC=128 40%** (nhead=32 OOB M>4 bug) |
| Path A capped | + `--max-num-seqs 16` | `AssertionError: graph_bs[0] <= max_num_seqs` at launch — ATOM's own sanity check |
| Path A-fp8 | TP=4 DP=2 fp8 MTP=1 | `decode_qlen=2,4 gqa_ratio=32` kernel assertion (EAGLE draft runs qlen=1, no kernel) |
| Path A-fp8-mtp3 | TP=4 DP=2 fp8 MTP=3 | Same assertion at cudagraph capture |
| Path A' | TP=2 DP=4 bf16 MTP=3 | Launched, GSM8K=**0.9045 FAIL** (~5% below 0.93 gate) |
| Path A' no-MTP | TP=2 DP=4 bf16 no-MTP | GSM8K=**0.9386 PASS** ✓ / Throughput=**2750.38** (**-22.6% vs BEST BASE 3555**) / E2E 53630ms (**+23% worse**) / Interactivity 23.58 / **NET LOSS** |

**Path A' no-MTP final interpretation**: Removing MTP fixed accuracy (0.9045 → 0.9386), confirming MTP was the accuracy killer at TP=2+DP=4+BF16. But without MTP we lose speculative decoding, and the DP=4 arithmetic multiplier does NOT materialize in practice because (a) TP=2 per-replica throughput is much worse than TP=8 (smaller compute per rank, higher comm overhead fraction), (b) DP=4 sync barrier is larger than predicted at CONC=128, (c) no speculative acceleration means no 1.89× MTP accept-rate gain. **Net: 22% worse than BEST BASE.** Definitive confirmation that TP<8 × DP is not viable for DSR1 on gfx950.

**Cherry-pick attempt**: ATOM commit `4911f42 disable persistent mla for fp8 kvcache` — rejected with conflict. Commit targets `attention_mla_sparse.py` which our HEAD deleted (sparse MLA refactored out). Not applicable to dense MLA path.

**Daniel Discord reply**: **multi-config submission confirmed ACCEPTED** ("we think its accepted" 2026-04-13 08:21). Validated the strategy but moot — no TP<8 variant works.

**Conclusion**: DSR1 DP scaling is blocked at the AITER kernel layer on gfx950 for our required configs. Our TP=8 + FlyDSL + dualstream BEST BASE is the final DSR1 config. Remaining time goes to Tier 1 cheap wins + Kimi Week 2 pivot + upstream contributions.



## Current Best Configuration
- **Track 1 (DSR1) — BEST BASE as of Session 4**: **ATOM main 108a70e + AITER main a35b45ad9 + flydsl 0.1.2 pip installed, TP=8 + MTP=3 + FP8 KV, `--max-model-len 10240`, `ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=16384`**.

| CONC | Throughput/GPU | Median TPOT | Median E2E | Interactivity | GSM8K | Gates passing |
|---|---|---|---|---|---|---|
| 4 | **738.93** | 5.97 ms | 6324 ms | **167.37 ✅** | 0.9378 | 1 of 3 (interactivity) |
| 32 | **2345.57** | 15.65 ms | **16507 ms ✅** | **63.92 ✅** | 0.9416 | 2 of 3 (E2E, interactivity) |
| 128 | **3555.19** | 41.61 ms | 43637 ms | 24.03 | 0.9424 | 0 of 3 |
| Gates | 3 of 9 | | | | | started at 0 of 9 |

Container: `danish_atom_main`. Session-over-session: **CONC=4 +30.6%**, CONC=32 +17.1%, CONC=128 +15.0%.

**SUBMISSION STRATEGY (committed 2026-04-13 after Daniel confirmed multi-config accepted on Discord)**:
- CONC=4 & CONC=32: BEST BASE above (TP=8 + MTP=3 + FP8-KV + flydsl + dualstream)
- CONC=128: TP=4 + DP=2 variant TBD (Path A: no-MTP BF16-KV retest at CONC=128 specifically — never tested before, the 341 tok/s/GPU Session 3 number was CONC=4 where DP barrier dominates)
- Daniel's exact words: "we think its accepted" (2026-04-13 08:21 Discord DM). Soft but committed by team lead.

## RESULT-004: FlyDSL pip install unlocks native tuned MoE kernels (2026-04-12 Session 4)

**Patch**: `pip install --force-reinstall "flydsl==0.1.2"` inside the container. One command. No code changes.

**Mechanism**: AITER's `fused_moe.py:838` calls `is_flydsl_available()` before dispatching MoE. When the flydsl Python package is missing (default container state), it returns False and AITER falls back to `ck_tile::MoeFlatmmKernel` (the 15.68% bottleneck in our authoritative profile). Installing flydsl==0.1.2 (AITER's exact pinned version) flips `is_flydsl_available()` to True, and the pre-existing `dsv3_fp4_tuned_fmoe.csv` tune file auto-routes DSR1 shapes to FlyDSL stage1/stage2 kernels that AMD had already pre-tuned (46 rows of `flydsl_moe1_*` / `flydsl_moe2_*` configs for shape `(7168, 256, 257, 9, per_1x32)`).

| CONC | Pre-FlyDSL (dualstream only) | **FlyDSL + dualstream** | Δ | Notes |
|---|---|---|---|---|
| 4 | 728 / 160.40 interact | **738.93 / 167.37** ✅ | +1.5% thr / **+4.3% interact** | **FIRST CONC=4 INTERACTIVITY GATE PASS OF ENTIRE HACKATHON** |
| 32 | 2270 / 62.87 / 16785 E2E | **2345.57 / 63.92 / 16507** ✅✅ | +3.3% thr | E2E gate already passed, interactivity improved further |
| 128 | 3280 / 22.14 / 47531 | **3555.19 / 24.03 / 43637** | **+8.4% thr**, -8.2% E2E | Biggest FlyDSL win — large-batch MoE where FlyDSL shines |

**Accuracy**: GSM8K stable in 0.938-0.945 range across four runs (0.9378, 0.9386, 0.9416, 0.9424). Initial 0.9447 → 0.9378 drop was run-to-run variance, not regression. Still ≥ 0.93 gate with comfortable margin.

**Context**: AITER already had `_flydsl_stage1_wrapper` / `_flydsl_stage2_wrapper` plumbed at `fused_moe.py:640-710`. `dsv3_fp4_tuned_fmoe.csv` already had 46 flydsl-winning rows. The entire infrastructure was pre-built — likely informed by Phase 1 MoE leaderboard submissions where FlyDSL for DSR1 shapes was the winning approach. The only missing piece was the pip dependency. Most Phase 2 teams will never notice the `flydsl unavailable` log line. **This is a hidden cliff-edge optimization.**

## RESULT-003: Dual-stream threshold raise (2026-04-12 Session 4)
`ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=16384` (default 1024) enables dual-stream MoE path during prefill. Our ISL=8192 prefill was excluded by default because 8192 > 1024.

| CONC | Throughput/GPU | Prior baseline | Δ |
|---|---|---|---|
| 4 | 728 | 743 | -2% (within noise) |
| 32 | **2270** | 2156 | **+5.3%** |
| 128 | **3280** | 3092 | **+6.1%** |

**Cost**: one env var, zero code, zero accuracy impact (GSM8K 0.9447 unchanged). **Keep in BEST BASE.** Source commented-out `# and not get_forward_context().context.is_prefill` at [atom/models/deepseek_v2.py:780](atom/models/deepseek_v2.py#L780) suggests AMD was uncertain; empirically our workload benefits.

## PATCH-005: BF16→FP8 quant_spec override for MLA o_proj — PARKED (2026-04-12 Session 4)

**Goal**: Convert MLA `o_proj` (largest contributor to the 17.41% BF16 GEMM bottleneck) from BF16 to per_Token FP8 by overriding `base_quant_config` in [deepseek_v2.py:1321](atom/models/deepseek_v2.py#L1321) and installing a custom `weight_loader_process` that quantizes BF16→FP8 with per-row scales at load time.

**Design**: new file `atom/mla_fp8_patch.py` (~70 lines) with `build_fp8_override_qc` (shallow-copies QuantizationConfig, strips o_proj from exclude_layers, prepends a per_Token FP8 LayerQuantConfig pattern spec) + `install_bf16_to_fp8_loader` (closure that on BF16 input computes `max_abs / 448` per-row, quantizes, writes both `self.weight.data` and `self.weight_scale.data` in-place into pre-born FP8 Parameter storage). Two-line surgical edit to `DeepseekV2MLAAttention.__init__` gated by `ATOM_MLA_FP8=1`.

**Result**: could not make it work. Server launched, weights loaded, **cudagraph capture reached `bs=128`** then crashed with the *same* `increment_version expects each element of the iterable to be a tensor` error as PATCH-004 v3/v4 at [torch/_functorch/_aot_autograd/runtime_wrappers.py:322](torch/_functorch/_aot_autograd/runtime_wrappers.py#L322). Failure is inside `submod_2` runtime — AOT autograd's captured tensor references become invalid for at least one Parameter in the compiled graph's arg list.

**Why the design hypothesis was wrong**: We thought "born FP8 at construction time" would sidestep the post-load mutation that broke PATCH-004. It doesn't. The standard `per_Token` FP8 path in `process_weights_after_loading` itself does something that invalidates AOT autograd's captured Parameter refs — likely `shuffle_weights(self.weight)` replacing `.data` storage, even though the Parameter object identity is preserved. Every working FP8 layer in ATOM (MLP `gate_up_proj`, `down_proj`) uses `per_1x128` (block-128 scaling), never `per_Token`. The `per_Token` path is an under-exercised branch.

**What to try if we come back**:
1. Switch the override from `per_Token` to `per_1x128` FP8. Matches the tested MLP path, requires computing block-128 weight scales instead of per-row, and trusts the same `process_weights_after_loading` code the MLP already exercises safely.
2. Alternatively: patch `atom/utils/cuda_piecewise_backend.py` or `atom/utils/backends.py` to invalidate/recompile the submod_2 graph if any captured Parameter's `.data` identity changes after load. Heavier surgery but addresses the root cause.
3. Alternatively: pre-quantize the state dict offline (write an FP8 copy of the DSR1 weights to disk), then the load path gets FP8 tensors directly with no BF16→FP8 conversion hook needed.

**Cost so far**: ~6 hours across two sessions debugging PATCH-004 + PATCH-005. Banked zero throughput. Expected ceiling ~5-7% overall if it had landed.

**Decision**: park, move on. The parked patch file and edits have been reverted; backups in `linear.py.bak` and `deepseek_v2.py.bak`. All findings captured here for a future attempt.

- **Track 2 (Kimi K2.5)**: Not yet started. Plan: same container (ATOM main has `kimi_k25.py`), after DSR1 knob sweep completes.

## AUTHORITATIVE KERNEL PROFILE — ATOM main TP=8 MTP=3 FP8-KV CONC=4 ISL=8192 OSL=1024 (2026-04-12)
**Captured**: 32 real requests, 4 warmups, torch.profiler, all 8 ranks parsed and summed (74213 ms total GPU time across the node, ~9277 ms per rank, <2% variance rank-to-rank — rock-solid symmetric data). **This replaces the stale pin profile.**

### Top kernels grouped by area
| Area | % | Kernels (time ms across all 8 ranks) | Phase 1 coverage |
|---|---|---|---|
| **1. BF16 GEMM (decode dense)** | **17.41%** | `aiter::bf16gemm_fp32bf16_tn_32x64_splitk_clean` (12920) | ❌ NOT Phase 1 — NOVEL TERRITORY |
| **2. MoE (ck_tile Flatmm)** | **15.68%** | `ck_tile::MoeFlatmmKernel` (11637) | ⚠️ Phase 1 has MoE but this is NEWER ck_tile variant — check if Danish #1 targets it |
| **3. All-reduce (total)** | **15.16%** | `reduce_scatter_cross_device_store` (7133, 9.61%) + NCCL generic (2747, 3.70%) + `cross_device_reduce_2stage` (1373, 1.85%) | ❌ NOT Phase 1 — NOVEL TERRITORY |
| **4. MLA (total)** | **13.75%** | `mla_a8w8_qh16_qseqlen2_gqaratio16_ps` (6503, 8.76%) + `kn_mla_reduce_v1_ps` (3702, 4.99%) | ✅ Phase 1 MLA (Danish #8) |
| **5. RMSNorm (total)** | **8.71%** | `local_device_load_rmsnorm` (3336, 4.50%) + `add_rmsnorm_quant 256` (1728, 2.33%) + `add_rmsnorm_quant 64` (1668, 2.25%) + `quant_256_32` (278, 0.38%) | ❌ Fusion target (novel) |
| **6. Batched GEMM a8w8** | **5.09%** | `_batched_gemm_a8w8_a_per_token_group_prequant` (3778) | ⚠️ Phase 1 GEMM adjacent |
| **7. Fill + act_and_mul** | **4.24%** | `vectorized_elementwise_kernel FillFunctor` (1575, 2.12%) + `act_and_mul_kernel silu` (1572, 2.12%) | Fusion target |
| **8. MoE sorting** | **3.71%** | `ck_tile::MoeSortingKernel` (2750) | Fusion with MoE |
| **9. MoE MX GEMM (older ck path)** | **2.60%** | `kernel_moe_mxgemm_2lds` (1931) | ✅ Phase 1 MoE (Danish #1) — but minority of MoE time |
| **10. Fused QK RoPE + KV write** | **2.25%** | `fuse_qk_rope_concat_and_cache_mla` (1668) | Already fused, good |
| **11. Grouped topk** | **2.17%** | `grouped_topk_opt_sort_kernel` (1611) | Routing |
| **12. MLA prefill (FMHA)** | **1.05%** | `fmha_fwd_hd192_hd128_bf16_causal_group` (780) | Prefill only, small |

### Big picture
- **~50% of runtime is in kernels NOT covered by Phase 1** (BF16 GEMM 17.4% + All-reduce 15.2% + RMSNorm 8.7% + Fill/elementwise 4.2%). **This is where novel work wins.**
- **~32% is Phase 1 territory** (MoE total 22% + MLA total 13.75% + MoE MX GEMM 2.6%). Solid kernel-integration wins if Danish's kernels drop in cleanly.

### Critical uncertainties resolved (2026-04-12)
1. **Danish's Phase 1 MoE kernel targets FlyDSL**, NOT `ck_tile::MoeFlatmmKernel`.
   - Source: `Phase1_kernal_Results/MOE/Danish.py` + MOE.md
   - Uses `aiter.ops.flydsl.moe_kernels.flydsl_moe_stage1`/`stage2` with `tile_m=32 tile_n=256 tile_k=128`
   - ATOM main's current MoE dispatch routes through `ck_tile::MoeFlatmmKernel` (15.68%, 11637ms) — a DIFFERENT kernel path
   - The older `ck::kernel_moe_mxgemm_2lds` path (Phase 1's ck baseline target) is only 2.60% (1931ms) on ATOM main
   - **Implication**: direct kernel swap will NOT work. Must either (a) patch ATOM main to dispatch to FlyDSL instead of ck_tile, (b) port FlyDSL ideas into a ck_tile replacement, or (c) monkey-patch AITER's `get_2stage_cfgs` like Danish's Phase 1 v917 code did. Option (a) or (c) is most likely winnable. Real win ceiling: ~15% of runtime (11637ms → ~7000ms with FlyDSL's 40% kernel improvement from Phase 1, assuming similar speedup on ck_tile shapes).
2. **Danish's Phase 1 GEMM kernel (9.29µs, #1) targets SMALL-M MXFP4 GEMM**, NOT our Phase 2 bottleneck.
   - Source: `Phase1_kernal_Results/GEmm MM/Danish.py` + Gemm.md
   - Benchmark shapes: M ∈ {4,16,32,64,256}, N ∈ {2112,2880,3072,4096,7168}, K ∈ {512,1536,2048,7168}
   - Covers "MoE expert-sized GEMMs" where each expert has small-M workload
   - Our 17.41% BF16 GEMM (`bf16gemm_fp32bf16_tn_32x64_splitk_clean`) is **BF16**, not MXFP4 — Phase 1 GEMM does not apply
   - **Implication**: Phase 1 GEMM is ~2.6% of our runtime at best (the small ck_moe_mxgemm path). The real 17% BF16 GEMM bottleneck is **novel territory** — no Phase 1 kernel targets it.
3. **Danish's Phase 1 MLA kernel (31.9µs, #8) has precision risk**.
   - Source: `Phase1_kernal_Results/MIXED MLA/MIXED_MLA.md`
   - Uses `persistent_mode=2` (pg2) with "~4% mismatch risk" and "~50% LB pass rate" per Phase 1 notes
   - **Phase 2 requires GSM8K ≥ 0.93 robustly** (we've seen 0.9386-0.9409 variance on current baseline; cannot afford to lose more)
   - Danny/LunNova's Phase 1 "precision-safe" Triton split-K approach (28.6-29.4µs, rank 5-7) is a better template IF we integrate MLA
   - **Implication**: Don't drop-in Danish's MLA kernel. Either keep AITER's current MLA (already the decode-optimized `mla_a8w8_qh16_qseqlen2_gqaratio16_ps` which is already fast at 8.76%) or port Danny/LunNova's precision-safe Triton split-K design.
4. **BF16 GEMM 17.41% call site** — still unresolved. Count is 1173784 across all ranks for 32 requests / 8 ranks = 4585 calls/request. At 61 layers × 1 call/layer/token × ~1024 decode tokens ≈ 62464 calls/request — off by 13×. More likely 1 call/layer × several projections. Candidates: MLA o_proj (61 × ~75 = ~4575, matches!), or dense layers 0-2 FFN (3 × 3 projections × 1024 tokens = 9216 — doesn't match as cleanly). **Most likely: MLA o_proj (output projection)** run in BF16 because the absorbed MLA path keeps o_proj as BF16 for numerical stability.

### Phase 1 kernel → Phase 2 reality mapping
| Phase 1 kernel | Phase 1 target | Phase 2 hot path | Drop-in win? |
|---|---|---|---|
| Danish MoE FlyDSL 69.9µs #1 | `_fused_moe` ck path | `ck_tile::MoeFlatmmKernel` (15.68%) | NO — dispatch patch required. Real win ~15% if FlyDSL beats ck_tile. |
| Danish GEMM MXFP4 9.29µs #1 | small-M MXFP4 GEMM | BF16 GEMM 17.41% (wrong dtype) + ck_moe_mxgemm 2.60% (small) | NO — wrong bottleneck. Contributes ~2.6% at most. |
| Danish MLA 31.9µs #8 (pg2) | AITER MLA decode | `mla_a8w8_qh16_qseqlen2_gqaratio16_ps` (8.76%) | NO — precision risk. Current kernel already fast. |

### Does the same reality apply to SGLang/vLLM? (partially YES)
Most of our bottlenecks are in AITER primitives, which ALL three frameworks share on ROCm:
- **BF16 GEMM 17.41%** → `aiter::bf16gemm_fp32bf16_tn_32x64_splitk_clean` is an AITER kernel. Same on SGLang (uses AITER) and vLLM (with `VLLM_ROCM_USE_AITER=1`). **Same bottleneck everywhere.**
- **All-reduce 15.16%** → `aiter::reduce_scatter_cross_device_store` + `cross_device_reduce_2stage`. AITER-provided. Same on all three frameworks. **Same bottleneck everywhere.**
- **RMSNorm 8.71%** → `aiter::local_device_load_rmsnorm` + `add_rmsnorm_quant_kernel`. AITER-provided. Same everywhere.
- **MoE** → DIFFERS. ATOM main uses `ck_tile::MoeFlatmmKernel`, SGLang uses AITER's `fused_moe` (closer to older ck path), vLLM uses `VLLM_ROCM_USE_AITER_MOE` flag routing. Each framework dispatches differently, even though they share the underlying kernels.
- **MLA** → DIFFERS significantly. ATOM uses `mla_a8w8_qh16_qseqlen2_gqaratio16_ps` (AITER ASM), SGLang can use Triton MLA (`--attention-backend triton`) or AITER, vLLM has `VLLM_ROCM_USE_AITER_MLA` flag.
- **Framework overhead** (scheduler, CUDA graph, Python dispatch) → DIFFERS. ATOM and SGLang have different scheduler costs at CONC=4.

**Net result**: ~45% of our bottleneck list (BF16 GEMM + AllReduce + RMSNorm + act_mul = 45.52%) is AITER-shared and would be identical if we switched frameworks. MoE and MLA would differ. So switching frameworks would NOT give us a free 50% improvement — most of the heavy lifting is in AITER kernels all three frameworks depend on. **Confirms ATOM-only decision: framework change ≠ kernel change.**

### Revised optimization priority (bulletproof)
**Tier 1 — novel work, biggest wins**:
1. BF16 GEMM 17.41% → investigate + write faster kernel (or convert to MXFP4 if possible)
2. All-reduce fusion 15.16% → extend `fused_allreduce_rmsnorm` to cover more paths

**Tier 2 — Phase 1 kernel integration**:
3. MoE 18.28% total (ck_tile Flatmm + ck MX GEMM) → Danish #1 MoE kernel (verify target path first)
4. MLA 13.75% → Danish #8 MLA kernel

**Tier 3 — smaller fusion wins**:
5. RMSNorm standalone 4.50% → fuse with surrounding ops
6. act_and_mul + fill 4.24% → eliminate via fusion

## CANONICAL RESULTS TABLE — DSR1 CONC=4 (data is gold, keep this updated after every experiment)
Format: `ISL=8192 OSL=1024 CONC=4 num_prompts=40` — always this workload unless noted.

| # | Date | Config | Thru/GPU | TPOT (ms) | TTFT (ms) | Interact | E2E (ms) | GSM8K | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-04-10 | ATOM pin TP=8 MTP=1 FP8-KV | 566.65 | 7.72 | 306.93 | 129.52 | 8189.87 | 0.9447 | initial baseline |
| 2 | 2026-04-11 | ATOM pin TP=8 MTP=3 FP8-KV | **668** | 6.80 | ~400 | 147.15 | 7332 | 0.9469 | best pin config |
| 3 | 2026-04-12 | ATOM main TP=4 MTP=3 FP8-KV (single) | 531.83 | 8.47 | 399.97 | 118.01 | 9148.54 | 0.9431 | 4 GPUs idle, hurts |
| 4 | 2026-04-12 | ATOM main TP=4 DP=2 MTP=3 BF16-KV | crash/0.0159 | — | — | — | — | 0.0159 | MTP+DP+BF16-KV broken (garbage output) |
| 5 | 2026-04-12 | ATOM main TP=4 DP=2 MTP=3 FP8-KV | CRASH | — | — | — | — | — | AITER gqa_ratio=32 needs persistent_mode, DP+fp8 disables it |
| 6 | 2026-04-12 | ATOM main TP=4 DP=2 no-MTP BF16-KV | 341.46 | 12.39 | 748.69 | 80.72 | 13494.38 | 0.9454 | DP sync barrier + no MTP tax |
| 7 | 2026-04-12 | **ATOM main TP=8 MTP=3 FP8-KV** | **738.93** | **6.10** | **254.61** | **163.92** | **6463.40** | **0.9401** | **NEW BASELINE** +10.6% vs pin |
| 8 | 2026-04-12 | main TP=8 MTP=3 FP8 `--enable_prefix_caching` | CRASH | — | — | — | — | — | `AttributeError: 'NoneType'.dim()` in `aiter/ops/triton/gather_kv_b_proj.py:29`. MXFP4 path passes `kv_proj_scale=None`, AITER's `gather_kv_b_proj` doesn't handle None. **NOT FIXED — deferred** (need real MXFP4 scale from ATOM side, not AITER side). File upstream when we understand MXFP4 weight layout better. |
| 9 | 2026-04-12 | main TP=8 MTP=3 FP8 `--max-num-batched-tokens 32000` | 735.18 | 6.41 | 253.79 | 156.04 | 6869.55 | 0.9431 | NEUTRAL-TO-WORSE. Interactivity -4.8%, TPOT +5.1%. Scheduler overhead > theoretical prefill batching at CONC=4. **Drop knob.** |
| 10 | 2026-04-12 | main TP=8 MTP=3 FP8 `--enable_prefix_caching` (patched AITER `gather_kv_b_proj` with ones-scale) | — | — | — | — | — | **0.7695** ❌ | Patch unblocks the crash but accuracy drops 18% (0.9401→0.7695). Confirms MXFP4 weights do NOT bake outer row-scale into weight tensor. Real fix: ATOM must pass correct scale tensor at `attention_mla.py:680` call site in prefix-cache path. **NOT FIXED — deferred**, needs deeper MXFP4 weight loader investigation. AITER patch reverted. |

## Gate status at current baseline (row 7, ATOM main TP=8 MTP=3 FP8-KV CONC=4)
| Gate | Required | Current | Pass? | Gap |
|---|---|---|---|---|
| Throughput/GPU | 1500 | 738.93 | ❌ | -50.7% (see FINDING-005 — aspirational) |
| Interactivity | 165 | 163.92 | ❌ | -0.66% (one knob away) |
| E2E ≤ 5000ms | 5000 | 6463 | ❌ | +29% |
| GSM8K ≥ 0.93 | 0.93 | 0.9401 | ✓ | robust (stderr ±0.0065) |

## Framework commitment (decision finalized 2026-04-12)
- **ATOM-only** for both tracks. Reason: mergeability (Rule 4.2). SGLang and vLLM dropped from plan.
- **ATOM main 108a70e** is primary. ATOM pin 33e0aac is fallback only.
- Container: `danish_atom_main` (active), `danish_atom` (fallback), `danish_sglang`/`danish_vllm` (stopped, unused).

## Baseline Numbers
### DSR1 on ATOM TP=8 (2026-04-10)
| CONC | mean_TTFT | median_TPOT | Throughput/GPU | Interactivity | E2E Latency | Pass? |
|------|-----------|-------------|----------------|---------------|-------------|-------|
| 4    | 306.93ms  | 7.72ms      | 566.65 tok/s   | 129.52 tok/s  | 8189.87ms   | NO    |
| 32   | 727.22ms  | 17.35ms     | 2003.70 tok/s  | 57.64 tok/s   | 18309.51ms  | PARTIAL (interactivity passes, E2E 309ms over, throughput 51% of target) |
| 128  | 2054.98ms | 45.69ms     | 3092.39 tok/s  | 21.89 tok/s   | 47914.57ms  | NO (all fail — E2E 2x over, interactivity half of target) |
GSM8K accuracy: 0.9447 (flexible-extract), 0.9393 (strict-match) — PASSED

### DSR1 on SGLang TP=8 — BLOCKED
- Model `amd/DeepSeek-R1-0528-mtp-mxfp4` returns 404 on HuggingFace (even with HF token auth)
- Setup was done correctly: AITER installed, sgl-kernel built (v0.4.1), SGLang installed (v0.5.10.post1)
- Awaiting Daniel's response on Discord about the model availability

### Kimi on vLLM TP=8 — BLOCKED
- Pre-built image `vllm/vllm-openai-rocm:v0.15.1` crashes with weight shape mismatch (`param_data.shape != loaded_weight.shape`)
- Building vLLM from source: AITER JIT cache incompatible (GLIBCXX_3.4.31 mismatch — AITER compiled on Ubuntu 24.04, vLLM image is Ubuntu 22.04)
- Kimi model `amd/Kimi-K2.5-MXFP4` downloaded successfully (~500GB cached in /projects/teamA/hf_cache/)
- Awaiting Daniel's guidance on correct vLLM version/setup

## Key Observations

### CRITICAL: TP=8 cannot pass CONC=4 throughput threshold
The throughput/GPU formula divides total throughput by num_GPUs:
- At TP=8, MTP=3: `4 × 9216 / (307 + 1024 × 6.80) / 8 = 634 tok/s/GPU` (target: 1500)
- To hit 1500 at TP=8: need TPOT = 2.70ms — **impossible** with current hardware
- **At TP=5: 1015 tok/s/GPU** — closer, kernel gains could push past 1500
- **At TP=4: 1268 tok/s/GPU** — feasible with kernel gains, BUT accuracy fails (0.928)
- **TP reduction is REQUIRED to qualify for the prize at CONC=4**

### TP experiments status
- **TP=4**: accuracy 0.928 (fails 0.93 by 0.002). AITER MLA kernel bug at TP=4 (ROCm/aiter Issue #1455) confirmed. Patched ATOM selector to disable MLA ASM (`ATOM_DISABLE_AITER_MLA=1`) but standard AiterBackend ALSO crashes with memory access faults at TP=4. The TP=4 problem is deeper than just the MLA kernel — the whole attention path has issues. Need to investigate further: try TP=4 without MTP, try Expert Parallelism alone, or look at newer AITER commits that may fix the bug.
- **TP=5**: TESTED — CRASHES. Custom allreduce only supports world_size [2,4,6,8]. AITER fallback path has bug (missing world_size attribute). Not viable on current AITER.
- **TP=2**: UNTESTED — would work (vocab/allreduce both OK) but TPOT would be much worse (fewer GPUs = more work per GPU). Only viable if TPOT improvement is dramatic.
- **TP=6**: crashes (vocab not divisible)
- **Expert Parallelism**: `--enable-expert-parallel` distributes experts across GPUs differently. Could give throughput benefits without accuracy loss of TP reduction.
- **TP=4 throughput data (MTP=1, no accuracy gate)**: Total throughput 3413 tok/s, TPOT 9.96ms. Throughput/GPU = 3413/4 = **853 tok/s/GPU** (vs TP=8's 566). 51% better per-GPU score but TPOT is 29% worse. With MTP=3 + kernels, projected ~1200 tok/s/GPU. Still needs accuracy fix (0.928→0.93) to be viable.
- **vLLM may handle TP=4 better**: vLLM has `VLLM_ROCM_USE_AITER_MLA=0` env var to disable buggy MLA kernel. ATOM has no equivalent. Consider testing DeepSeek-R1 on vLLM at TP=4 with MLA disabled.
- **AITER updated to latest HEAD (0.1.12.post2)** — still crashes at TP=4 with memory access faults during inference. The MLA ASM kernel is fundamentally incompatible with TP=4 on this hardware/ATOM combination. Server starts, processes some requests, then crashes.
- **TP=4 on ATOM status (updated 2026-04-12)**:
  - TP=4 + MTP=1: **WORKS** (no crash after AppArmor fix), accuracy 0.928 (fails 0.93)
  - TP=4 + MTP=3: **CRASHES** (memory access fault on GPUs 2-5). MLA ASM kernel path breaks at higher qo_len with TP=4's 32 heads/GPU.
  - So TP=4 is stable only with MTP=1. The 0.002 accuracy gap is the real blocker.
- **Only viable TP values for AITER MLA:** TP=1 (128 heads) or TP=8 (16 heads). TP=2 (64 heads) and TP=4 (32 heads) crash.
- **vLLM with `VLLM_ROCM_USE_AITER=0` + `--enforce-eager`** could run TP=4 using Triton fallback (1.2-1.6x slower but correct). This is on vLLM, not ATOM.
- **Strategy: Maximize TP=8 score** through MTP=3, kernel integration, and remaining knobs. Compete hardest on CONC=32/128 where throughput gap is smaller.

## Accuracy Requirement: Must Be ROBUST, Not Borderline
- Rule 4.2: "Your code must be mergeable into AMD repositories"
- AMD won't merge code that passes accuracy 1 time in 10 due to variance
- Need **consistent** accuracy ≥ 0.93 across multiple runs, not borderline passes
- **Targets: 0.935+ for safe margin** (0.005 above threshold)
- If TP=4 gives 0.930 on one run, must verify 3+ more runs before considering it viable
- The real fix needs to be solid enough to merge upstream

## Discord intel (2026-04-12)

### SGLang model mystery SOLVED
- Daniel confirmed: **"mtp version is amd/DeepSeek-R1-0528-MXFP4"** (same model as ATOM)
- SGLang config has WRONG model name (`amd/DeepSeek-R1-0528-mtp-mxfp4` — typo with `-mtp-`)
- Fix: Change MODEL in SGLang specific_conc_var.sh to `amd/DeepSeek-R1-0528-MXFP4`
- **SGLang DSR1 is now UNBLOCKED**

### Kimi vLLM
- **Josu confirmed: vLLM v0.19.0 works with Kimi** (not v0.15.1)
- He has issues with server stalls on Kimi (sampled token GPU→CPU stall)
- Daniel: "vllm compatibility issues have to be investigated by yourselves, that's part of the game"

### Docker/AppArmor fixes (Maharshi)
- Relaxed restrictions rolled out
- New wrapper: `docker-teamXYZ-unrestricted` allows `--ipc=host --network=host`
- Usage: `docker-teamA-unrestricted run --ipc=host --network=host ...`
- **divc13 had AppArmor blocking Unix sockets for TP>1** — might be related to our TP=4 crashes!
- Worth retesting TP=4 after AppArmor relaxation

### MTP sweep results (complete)
- **MTP=3 optimal for CONC=4**: TPOT 7.72→6.80ms (-11.9%)
- **MTP=2 slightly better for CONC=32**: TPOT 17.35→16.37ms
- **No MTP effect at CONC=128**: all values ~45-46ms
- MTP=4: crashes (MLA ASM kernel qo_len<=4 limit in current AITER)
- MTP=5: not supported (ATOM config max=4)
- **Decision: MTP=3 for all CONC** (biggest win at hardest level)

### Other observations
- **CONC=32 passes interactivity and E2E** with MTP=3 (59.82 ≥ 50, 17687ms ≤ 18000)
- **CONC=128 is hardest**: needs 55% TPOT reduction regardless of TP
- **Benchmark binary hardcodes /8 for throughput/GPU** — must modify if using TP<8
- **Profile is stable across concurrencies** — same kernel distribution at CONC=4/32/128

## Experiments Log
### [EXP-001] ATOM TP=8 Baseline CONC=4 (2026-04-10)
- Config: ATOM, TP=8, FP8 KV cache, MTP, max-model-len 10240
- Results: Throughput 566.65 tok/s/GPU, Interactivity 129.52, E2E 8189ms, TPOT 7.72ms, TTFT 306.93ms
- Verdict: FAILS all 3 thresholds (throughput, interactivity, E2E)
- Action: Need TP=4 and kernel optimizations

### [EXP-002] ATOM TP=8 Baseline CONC=32 (2026-04-10)
- Config: Same as EXP-001, CONC=32, NUM_PROMPTS=320
- Results: Throughput 2003.70 tok/s/GPU, Interactivity 57.64, E2E 18309ms, TPOT 17.35ms, TTFT 727.22ms
- Verdict: Interactivity PASSES (57.64 >= 50). E2E barely fails (309ms over). Throughput at 51% of target.
- Action: TP=4 would ~double throughput to ~4000, passing the 3900 target

### [EXP-003] ATOM TP=8 Baseline CONC=128 (2026-04-11)
- Config: Same as EXP-001, CONC=128, NUM_PROMPTS=1280
- Results: Throughput 3092.39 tok/s/GPU, Interactivity 21.89, E2E 47914ms, TPOT 45.69ms, TTFT 2054.98ms
- Verdict: ALL FAIL. E2E 2x over (47.9s vs 22s max). Interactivity half of target (21.89 vs 48).
- Action: Need major TPOT reduction at high concurrency

### [EXP-004] ATOM TP=4 CONC=4 (2026-04-11)
- Config: ATOM, TP=4, FP8 KV cache, MTP, max-model-len 10240
- Results: GSM8K 0.9287 (run 1), 0.9280 (run 2) — below 0.93 threshold
- Verdict: ACCURACY FAILS. Performance not tested. Gap is small (~0.002).
- Action: Revisit with accuracy recovery tricks. Not ruling out permanently.

### [EXP-005] ATOM TP=6 CONC=4 (2026-04-11)
- Config: ATOM, TP=6
- Results: CRASH — `assert num_embeddings % self.tp_size == 0` (vocab 129280 not divisible by 6)
- Verdict: TP=6 not possible. Valid TPs: 1, 2, 4, 5, 8 (factors of 129280)
- Action: Could try TP=5 (129280/5=25856 works mathematically)

### [EXP-006] ATOM TP=8 + QuickReduce Q4 + env vars — ALL CONC (2026-04-11)
- Config: Added AITER_QUICK_REDUCE_QUANTIZATION=Q4, PYTHON_GIL=0, AMD_DIRECT_DISPATCH=1, HIPBLASLT_ALLOW_FLUSH_DENORM=1
- Results vs baseline:

| CONC | Baseline TPOT | QuickReduce TPOT | Baseline Throughput | QR Throughput | Baseline E2E | QR E2E |
|------|--------------|-----------------|--------------------|--------------|-----------|----|
| 4    | 7.72ms       | 7.69ms (-0.4%)  | 566.65             | 562.35 (-0.8%)| 8189ms   | 8299ms (+1.3%) |
| 32   | 17.35ms      | 17.35ms (0%)    | 2003.70            | 2005.24 (+0.08%)| 18309ms | 18259ms (-0.3%) |
| 128  | 45.69ms      | 45.61ms (-0.2%) | 3092.39            | 3096.96 (+0.1%)| 47914ms | 47812ms (-0.2%) |

- Verdict: **NO IMPROVEMENT across all 3 concurrencies.** <1% difference everywhere.
- Conclusion: GPU-to-GPU all-reduce communication is NOT the bottleneck in ATOM TP=8 for DSR1. QuickReduce and these env vars are definitively ruled out.
- Action: Move to next experiment. Need to profile actual kernel-level bottlenecks.

### [EXP-007] ATOM TP=8 + --num-speculative-tokens 3 — ALL CONC (2026-04-11)
- Config: Added `--num-speculative-tokens 3` (default was 1 draft layer)

| CONC | Baseline TPOT | MTP=3 TPOT | Baseline Throughput | MTP=3 Throughput | Baseline E2E | MTP=3 E2E |
|------|--------------|-----------|--------------------|-----------------|-----------|----|
| 4 | 7.72ms | **6.80ms (-11.9%)** | 566 | **668 (+18%)** | 8189ms | **7332ms (-10.5%)** |
| 32 | 17.35ms | **16.72ms (-3.6%)** | 2003 | **2156 (+7.6%)** | 18309ms | **17687ms (-3.4%) — PASSES ≤18000!** |
| 128 | 45.69ms | 45.95ms (+0.6%) | 3092 | 3192 (+3.2%) | 47914ms | 48394ms (+1.0%) |

- Accuracy: 0.9469 (CONC=4), 0.9378 (CONC=32), 0.9447 (CONC=128) — ALL PASS
- Verdict: **BIG WIN at CONC=4 (-11.9% TPOT), moderate at CONC=32 (-3.6%), NO EFFECT at CONC=128.**
- CONC=32 now PASSES both E2E (≤18000) and Interactivity (≥50)!
- MTP benefit decreases with concurrency — at CONC=128 batch is already saturated.
- Action: Try MTP=5, then move to other knobs. MTP=3 is new baseline for low concurrency.

## Blocking Issues
- **SGLang DSR1**: Model `amd/DeepSeek-R1-0528-mtp-mxfp4` returns 404 on HuggingFace — doesn't exist yet. Asked Daniel on Discord.
- **vLLM Kimi**: Pre-built image has weight shape mismatch. Source build has GLIBCXX ABI mismatch (Ubuntu 24.04 JIT cache vs 22.04 container). Fix: clear JIT cache and rebuild AITER inside vLLM container. Asked Daniel on Discord.
- **AITER JIT cache shared across containers**: JIT `.so` files compiled in one container (Ubuntu 24.04) crash in another (Ubuntu 22.04). Must clear cache when switching containers.
- AppArmor restrictions on server — use `/usr/bin/docker` workaround or `~/bin/docker` with `/dev/dri/*` syntax

## Open Questions
- Daniel: When will `amd/DeepSeek-R1-0528-mtp-mxfp4` be published on HuggingFace? (SGLang blocker)
- Daniel: What vLLM version/commit works with `amd/Kimi-K2.5-MXFP4`? (Kimi blocker)
- Can we use ATOM for Kimi K2.5 track? (rules say "AMD ATOM or vLLM" — yes, worth trying)
- TP=5: Does ATOM support it? (vocab 129280 divisible by 5)

## Optimization Roadmap (step by step, in order)

### Step 1: Profile on BASE — DONE (2026-04-11)
- Profiled with `--torch-profiler-dir /workspace/trace`, 5 requests at CONC=1
- Total kernel time on rank 0: 3120.5 ms

**Kernel breakdown (top 10):**

| % | Time | Kernel | Category |
|---|------|--------|----------|
| 20.6% | 644ms | reduce_scatter_cross_device_store | All-reduce communication |
| 17.2% | 538ms | bf16gemm_fp32bf16_tn_32x64 | BF16 GEMM (dense layers) |
| 9.4% | 295ms | MoeFlatmmKernel | MoE GEMM |
| 5.8% | 180ms | batched_gemm_a8w8 | FP8 batched GEMM |
| 5.3% | 164ms | local_device_load_rmsnorm | RMSNorm + device load |
| 4.6% | 145ms | mla_a8w8 decode | MLA decode attention |
| 4.5% | 140ms | bf16gemm_fp32bf16_tn_48x64 | BF16 GEMM |
| 4.4% | 139ms | MoeSortingKernel | MoE token sorting |
| 3.2% | 101ms | mla_reduce_v1 | MLA reduce |
| 2.9% | 92ms | cross_device_reduce_1stage | All-reduce |

**Full profiling at ISL=8192, OSL=1024 (real workload, all CONC):**

| Kernel Category | CONC=4 | CONC=32 | CONC=128 | Your kernel? |
|----------------|--------|---------|----------|-------------|
| BF16 GEMM (decode dense) | 11.3% | 17.4% | 17.4% | Partial |
| MoE total (CK+sorting+MXFP4) | 17.8% | 20.8% | 21.5% | **YES — #1 Phase 1** |
| All-reduce total | 17.7% | 16.0% | 15.6% | No (framework) |
| MLA total (decode+reduce) | 13.4% | 13.7% | 13.5% | YES |
| FP8 batched GEMM | 4.9% | 4.8% | 4.9% | YES |
| RMSNorm | 4.2% | 4.1% | 4.2% | Fusion target |

**Key insights:**
- Profile is STABLE across concurrencies — same optimizations help everywhere
- Your Phase 1 kernels target 39.9% of total time (MoE + MLA + FP8 GEMM)
- BF16 GEMM at 17.4% is dense layer projections — not MXFP4, investigate why not FP4
- All-reduce at 15.6% is fundamental TP=8 cost — QuickReduce didn't help, need compute-communication overlap
- MLA is 13.5% — bigger than short-prompt profile showed, worth optimizing
- Initial short-prompt profile was MISLEADING — always profile with real workload

### Step 2: Quick knob tests on BASE — IN PROGRESS
- Test each config knob on base setup, one at a time, all 3 CONC
- ~~QuickReduce Q4~~ → NO EFFECT (tested all CONC)
- ~~PYTHON_GIL=0, AMD_DIRECT_DISPATCH=1, HIPBLASLT_ALLOW_FLUSH_DENORM=1~~ → NO EFFECT
- `--num-speculative-tokens 3` → TESTING NOW
- `--gpu-memory-utilization 0.95` → PENDING
- `--max-num-batched-tokens 16384/24000` → PENDING
- FULL_AND_PIECEWISE CUDA graph mode → PENDING
- TP=5 → PENDING

### Step 3: Record BEST BASE numbers at all 3 CONC — PENDING
- After Step 2, whatever config works best = our "BEST BASE"
- Run all 3 CONC and record — this is the "before" for kernel integration
- These numbers are what we compare against when plugging kernels

### Step 4: Plug kernel #1 (MoE) into BEST BASE — PENDING
- Your Phase 1 #1 kernel (69.9μs)
- Wire into aiter/ops/moe.py dispatch
- Benchmark all 3 CONC → measure exact TPOT improvement
- Verify accuracy >= 0.93

### Step 5: Plug kernel #2 (GEMM) on top — PENDING
- Your Phase 1 #1 kernel (9.29μs)
- 62% of compute is MXFP4 GEMM — biggest potential
- Wire into aiter/ops/gemm.py dispatch
- Benchmark all 3 CONC → measure exact improvement
- Verify accuracy >= 0.93

### Step 6: Plug kernel #3 (MLA) on top — PENDING
- Best available: Danny/LunNova (28.6μs) or Josu (21.2μs, pg2 risk)
- Wire into aiter/ops/mla.py dispatch
- Direct TPOT reduction — 40% of score
- Benchmark all 3 CONC → measure exact improvement
- Verify accuracy >= 0.93

### Step 7: Re-tune knobs on optimized kernel setup — PENDING
- Some configs might behave differently with faster kernels
- Re-test promising knobs from Step 2
- Find final BEST config
- Submit to leaderboard

---

## ATOM Available Flags (confirmed from --help)
```
--model MODEL
--trust-remote-code
--tensor-parallel-size / -tp
--data-parallel-size
--enforce-eager
--enable_prefix_caching
--port / --server-port
--kv_cache_dtype {bf16,fp8}
--block-size BLOCK_SIZE
--max-model-len MAX_MODEL_LEN
--cudagraph-capture-sizes CUDAGRAPH_CAPTURE_SIZES
--level LEVEL                          (0-3, default 3)
--load_dummy
--enable-expert-parallel
--torch-profiler-dir TORCH_PROFILER_DIR
--enable-dp-attention
--method {mtp}
--num-speculative-tokens NUM_SPECULATIVE_TOKENS
--max-num-batched-tokens MAX_NUM_BATCHED_TOKENS
--max-num-seqs MAX_NUM_SEQS
--gpu-memory-utilization GPU_MEMORY_UTILIZATION
--scheduler-delay-factor SCHEDULER_DELAY_FACTOR
--host HOST
```
NOTE: No --mark-trace, no FULL_AND_PIECEWISE flag, no --num-continuous-decode-steps. These are SGLang/vLLM features not in ATOM.

## Optimization Levers — Status
- ~~**QuickReduce Q4**~~: TESTED — NO EFFECT (<1% across all CONC). All-reduce is not the bottleneck.
- ~~**Env vars (PYTHON_GIL=0, AMD_DIRECT_DISPATCH=1, HIPBLASLT_ALLOW_FLUSH_DENORM=1)**~~: TESTED — NO EFFECT.
- ~~**TP=4**~~: TESTED — ACCURACY FAILS (0.928 < 0.93). Not viable without accuracy recovery.
- ~~**TP=6**~~: TESTED — CRASHES (vocab not divisible by 6).
- **`--num-speculative-tokens`**: FULLY SWEPT (1,2,3). MTP=4 crashes (MLA ASM qo_len<=4), MTP=5 not supported (max=4).
  - **MTP=3 is optimal for CONC=4**: TPOT 7.72→6.80ms (-11.9%), Interactivity 129→147 (+13.6%)
  - **MTP=2 is slightly better for CONC=32**: TPOT 17.35→16.37ms vs MTP=3 16.72ms
  - **No MTP effect at CONC=128**: all values ~45-46ms TPOT
  - **Decision: Use MTP=3** (biggest win at hardest concurrency level)
- **gpu-memory-utilization 0.95**: NOT TESTED. More memory for KV cache.
- **Max batched tokens**: NOT TESTED. Try `--max-num-batched-tokens 16384, 24000, 32000`.
- **TP=5**: NOT TESTED. Vocab 129280/5=25856 works. Would give 1.6x throughput/GPU if accuracy holds.
- **Chunked prefill size**: NOT TESTED. Check if ATOM has equivalent to SGLang's setting.
- **CUDA graph batch sizes**: NOT TESTED. Try different capture sizes.
- **Prefix caching**: NOT TESTED. Try `--enable-prefix-caching`.
- **Profiling**: NOT DONE. Need to run ATOM trace profiler to find actual kernel bottlenecks.
- **Danish's Phase 1 kernels**: NOT STARTED. MLA (31.9μs), MoE (69.9μs), GEMM (9.29μs) — integrate into AITER.

## Code Modifications (patches applied to ATOM/AITER)

### RESULT-002: ATOM main TP=8 beats ATOM pin TP=8 by 11% across the board (2026-04-12)
**Config**: ATOM main (108a70e) + AITER main (a35b45ad9), TP=8, MTP=3, FP8 KV, MXFP4 weights, `--max-model-len 10240`. Exact same flags as ATOM pin baseline — ONLY the code version differs.

| Metric | ATOM pin 33e0aac | ATOM main 108a70e | Delta |
|---|---|---|---|
| Throughput/GPU | 668 | **738.93** | **+10.6%** |
| TPOT median | 6.80 ms | **6.10 ms** | **-10.3%** |
| Interactivity | 147.15 | **163.92** | **+11.4%** (1.08 off 165 target!) |
| TTFT median | ~400 ms | **254.61** ms | -36% |
| E2E median | 7332 ms | **6463 ms** | -11.8% |
| GSM8K | 0.9469 | 0.9401 | passes robust (both) |

**Why this is the new baseline**: ~120 commits of perf improvements between pin (33e0aac) and main (108a70e). Key contributors:
- `bac90b3` feat: replace triton fused_rms_fp8_group_quant with HIP kernel (#507)
- `5fd265c` support dual stream in prepare decode (#499)
- `a6ad84d` [plugin][MLA] optimize MLA metadata build and remove D2D copy (#387)
- `be22816` fix(eagle): skip attn_metadata update for non-16-head models (#484)
- `26bb804` fix deepseek tp 4 mtp mla metadata error (#460) — also unblocks TP=4

**Implication**: ATOM main = primary working baseline going forward. `danish_atom_main` container is active. Pin stays as fallback only. All future knob sweeps on main.

**Gap analysis for CONC=4**: interactivity passes in 1 tiny tweak, throughput 738 vs 1500 target is still ~50% gap (fundamentally limited per FINDING-005 / research report). Realistic ceiling with all knobs + kernels still ~1000 tok/s/GPU. Accept and focus scoring on CONC=32/128.

### FINDING-005: CONC=4 throughput/GPU=1500 is aspirational, not achievable (2026-04-12)
**Evidence from deep research** (Session 3 late):
1. AMD's own ATOM recipe [`recipes/DeepSeek-R1.md`](https://github.com/ROCm/ATOM/blob/main/recipes/DeepSeek-R1.md) publishes **zero numbers below CONC=128** for DSR1-MXFP4. Best published: 1,732 tok/s/GPU at CONC=128 / ISL=OSL=1024 (our workload is 8× longer prefill + 32× less batching — much harder).
2. **Our 668-738 tok/s/GPU IS the canonical recipe verbatim** — no secret flag exists that AMD publishes. We are running exactly what AMD recommends.
3. AMD's [MI355X distributed inference article](https://www.amd.com/en/developer/resources/technical-articles/2026/distributed-inference-performance-on-instinct-mi355x-gpu.html) charts CONC 4-128 but quotes headline wins only at CONC 64-128. Low concurrency is "competitive" not "crushing."
4. Cross-framework confirmation that DP+MTP is broken on MI355X: [SGLang #21942](https://github.com/sgl-project/sglang/issues/21942), [SGLang #20404](https://github.com/sgl-project/sglang/issues/20404). Not just ATOM.

**Realistic ceiling (stacked wins)**:
- ATOM main (current): **738** tok/s/GPU
- + `--enable-prefix-caching`: +5-15% on accuracy test portion
- + `--max-num-batched-tokens 32000`: +2-5% TTFT
- + MTP sweep at ISL=8192 (MTP=2 may beat MTP=3 at long prefill): +5-10%
- + Danish's Phase 1 kernels (MoE #1, GEMM #1): +10-25% on TPOT
- **Absolute ceiling: ~1000-1100 tok/s/GPU.** Target 1000 as real goal, not 1500.

**Implication**: Stop chasing CONC=4 thru 1500. Optimize for interactivity (163.92→165+, trivial) and focus scoring on CONC=32 (closer to thresholds) + CONC=128 (DP=2 territory) + accuracy robustness.

### BREAKTHROUGH-001: ATOM main + AITER main unblocks TP=4 (2026-04-12, Session 3 late)
- **What we did:** Cloned ATOM `main` (commit 108a70e) into `/projects/teamA/danish/repos/ATOM_main`, created `danish_atom_main` container, `pip install -e .` from the new checkout. Then checked out AITER `main` (commit a35b45ad9) in `/workspace/aiter` and reinstalled. Nuked all JIT `.so` files to force recompile against new source. First launch JIT-compiled ~10 modules (~15 min), subsequent launches fast.
- **Why it works:** ATOM main has commit `26bb804 fix deepseek tp 4 mtp mla metadata error (#460)` which is the direct fix. Plus PR `#484 skip attn_metadata update for non-16-head models`. Plus `_MLA_MIN_HEADS` + `padded_num_heads = max(num_heads, _MLA_MIN_HEADS)` head-repeat mechanism in `attention_mla.py` that extends nhead to the minimum supported, repeating Q heads, fully bypassing the broken gfx950 ASM nhead=32 path.
- **What I missed initially:** `feat/mla-head-repeat-nhead-lt-16` branch and all the commits 33e0aac→main containing TP=4 fixes. I should have fetched main and checked for fixes on day one. Two sessions of PATCH-001/002/003 were wasted fighting a bug that had already been fixed upstream. Lesson for every future session: **always check upstream HEAD for your bug class before diving into patches**.

### RESULT-001: ATOM main TP=4 + MTP=3 — WORKS but not enough alone (2026-04-12)
- **Config:** ATOM main (108a70e), AITER main (a35b45ad9), TP=4, MTP=3 (`--num-speculative-tokens 3`), FP8 KV cache, MXFP4 weights
- **GSM8K: 0.9431** (flexible-extract) / 0.9386 (strict-match) — **PASSES 0.93 robustly** (margin 0.011, stderr ±0.0064)
- **CONC=4 perf:**
  - Total throughput: 4254 tok/s
  - **Throughput/GPU: 531.83** (benchmark divides by 8 full-node GPUs) — **below 1500 target, and WORSE than ATOM pin TP=8's 668**
  - TPOT median: 8.47 ms (vs TP=8's 6.80 ms — 25% slower per token)
  - TTFT median: 399.97 ms
  - E2E median: 9148 ms (need ≤5000)
  - Interactivity: 118 (need 165)
- **Why TP=4 alone is worse:** The benchmark reports `total_throughput / 8`, so with only one TP=4 replica running, 4 GPUs sit idle — we get ~half the throughput we should. Also TPOT regressed because TP=4 has less compute parallelism per layer + the head-repeat mechanism does extra work repeating Q heads.
- **Path forward:** **DP=2 × TP=4** — run two TP=4 replicas in parallel (8 GPUs saturated), should ~2x the throughput/GPU because numerator doubles while denominator stays 8.

### FINDING-004: Benchmark normalizes by full node GPU count (not TP size) (2026-04-12)
- The competition `dsr1_benchmark perf` divides total throughput by 8 (full node), not by the TP size.
- **Implication:** Running TP=4 with only one replica is strictly worse than TP=8 — you halve your throughput because 4 GPUs are idle.
- **Implication:** To beat TP=8 you must use all 8 GPUs somehow:
  - `TP=4 × DP=2` (2 replicas, 4 GPUs each)
  - `TP=8` with fast kernels (if TPOT can be reduced enough)
  - Larger batch sizes per replica at TP=4 to amortize the 4 idle GPUs (doesn't help if they're truly idle)
- Earlier assumption "TP=4 automatically doubles throughput/GPU score" was WRONG. It only doubles if you also spin up a second replica.

### PATCH-003: ATOM site-packages q→FP8 conversion — CORRECT LOCATION (2026-04-12)
- **File:** `/opt/venv/lib/python3.12/site-packages/atom/model_ops/attention_mla.py` (lines 513-515)
- **Backup:** same path with `.bak` suffix
- **What:** Uncommented the `q = q.to(dtypes.fp8)` cast inside `_forward_decode`, right before the second `mla_decode_fwd` call (line 537). Added a `[PATCH-003]` print to verify engagement.
- **Why PATCH-002 failed:** We patched `/workspace/ATOM/...` but ATOM is installed into `/opt/venv/.../site-packages/atom/...`. The site-packages copy is what Python actually loads. Two copies, patched the wrong one.
- **Verified at runtime:** `[PATCH-003] q cast to fp8, shape=torch.Size([4, 32, 576]) dtype=torch.float8_e4m3fn` printed during warmup. Server launched cleanly. Confirms native gfx950 `nhead=32` path IS engaging.
- **Result:** SERVER CRASHES ANYWAY during real workload with `shape=[65, 32, 576]` (M=65). Memory access fault on 4 GPUs simultaneously after 4 successful calls at M=65. See FINDING-003 below.
- **Status:** Patch is correct and live; the crash is downstream in the AITER kernel itself, not in ATOM.

### FINDING-003: AITER gfx950 MLA native path has a real kernel bug for `nhead=32` with M>4 (2026-04-12)
- **Path exists:** `/workspace/aiter/aiter/mla.py` lines 287-304 have a dedicated branch for `gfx950 + nhead=32 + fp8 q + fp8 kv + max_seqlen_q=4` — exactly our TP=4 + MTP=3 case.
- **Path engages:** PATCH-003 confirms q reaches the kernel as FP8, shape `[4, 32, 576]` during warmup (M=4, 1 sequence) → no crash.
- **Path crashes on real batches:** During accuracy test, shape becomes `[65, 32, 576]` (M=65 ≈ 16 active sequences × 4 MTP tokens). After ~4 successful calls at M=65, 4 GPUs fault simultaneously with "Memory access fault by GPU node-2/3/4/5". Classic OOB write pattern — kernel writes OOB, crash hits on later read.
- **Conclusion:** The gfx950 nhead=32 kernel is only safe for `M == max_seqlen_q` (single sequence). For batched workloads it corrupts memory. This is a real AITER bug, not ATOM's fault. **File upstream issue** — this itself is a mergeable contribution (bug report + fix or guard).
- **Implication:** NO configuration of ATOM + AITER currently supports TP=4 for DSR1 with a real workload:
  - nhead=128 TP=8 → works (baseline, can't hit CONC=4 throughput)
  - nhead=32 TP=4 native → crashes on M>4 (this finding)
  - nhead=32 TP=4 simulated-as-16 → crashes on compressed KV (prior finding)
  - nhead=64 TP=2 → OOM (model too big for 2 GPUs)
- **Action:** Pivot to SGLang TP=4 + `--attention-backend triton` (bypasses AITER MLA entirely) or vLLM v0.19.0 TP=4 + `VLLM_ROCM_USE_AITER_MLA=0`.

### PATCH-002: ATOM attention_mla.py q→FP8 conversion (2026-04-12)
- **File:** `/workspace/ATOM/atom/model_ops/attention_mla.py`
- **Backup:** `/workspace/ATOM/atom/model_ops/attention_mla.py.bak`
- **What:** Uncommented lines 513-516 that convert q tensor to FP8 when kv_cache_dtype is fp8
- **Why:** AITER mla.py lines 291-297 have a NATIVE fast path for `gfx950 + nhead=32 + fp8 q + fp8 kv + max_seqlen_q=4` — this is exactly our TP=4 + MTP=3 case! But the path requires q to be FP8. ATOM commented out the q→FP8 conversion, so this path never triggers, causing fallback to persistent_mode which crashes.
- **Expected:** TP=4 + MTP=3 should now use the native 32-head path and work without crashes.
- **Risk:** q_scale was set in the commented code but not in our patch (we only did `q = q.to(dtypes.fp8)`). The q_scale is already passed via `self._q_scale`. Might cause issues if the scale isn't right.
- **Restore:** `cp /workspace/ATOM/atom/model_ops/attention_mla.py.bak /workspace/ATOM/atom/model_ops/attention_mla.py`

### PATCH-001: ATOM MLA backend selector (2026-04-11)
- **File:** `/workspace/ATOM/atom/utils/selector.py`
- **Backup:** `/workspace/ATOM/atom/utils/selector.py.bak`
- **What:** Added env var `ATOM_DISABLE_AITER_MLA=1` to bypass AITER MLA ASM backend and use standard AiterBackend instead
- **Why:** AITER MLA ASM kernel has known bug at TP=4 (ROCm/aiter Issue #1455) causing accuracy drop to 0.928. Disabling it should recover accuracy.
- **Risk:** Standard AiterBackend may not support MLA-specific optimizations (compressed KV cache, absorbed projections). Could cause errors or performance regression. If it crashes, restore from `.bak`.
- **Restore:** `cp /workspace/ATOM/atom/utils/selector.py.bak /workspace/ATOM/atom/utils/selector.py`

## Kernel Integration Status
- MLA: not started
- MoE: not started
- GEMM: not started

## Server Environment
- Node: mia1-p02-g55 (8x MI355X, 288GB HBM3e each)
- Time slot: 6AM-6PM IST daily
- Workspace: /projects/teamA/danish/
- Docker containers: danish_atom (running), SGLang + vLLM (not yet)
- Pinned commits: ATOM 33e0aac, AITER cbbdc50


---

# Decision History (merged from decision_log.md, 2026-04-13 Session 6A consolidation)

This section contains all DEC-N entries from the original `decision_log.md` file. Format: structured Context/Options/Decision/Rationale/Outcome per decision. Most recent decisions at top.

---

## FAILED INTERVENTIONS LOG — Session 6B Day 2 (2026-04-14, all day)

**Read this before attempting any of the listed interventions — they've been tried and reverted or found harmful.**

### DEC-047 — Relaxed MTP `(3, 0.2)` is the CONC=4 sweet spot — 2026-04-14 late
**Context:** After DEC-046's `(5, 0.3)` failed 1/3 runs, tightened `rejection_sampler.py:11-14` to `(TOP_N=3, DELTA=0.2)`.
**Decision:** **COMMITTABLE FLOOR.** 3/3 GSM8K runs pass: 0.9371 / 0.9356 / 0.9333 (mean 0.9353, min 0.9333, +0.33 pp margin). Warm bench at TP=4 SR CONC=4: **thr/GPU 1470, median TPOT 5.54 ms, interact 180.5, mean E2E ~6213 ms.** Speed essentially identical to (10, 0.6)'s 1472/GPU.
**Rationale:** Tightening thresholds barely reduces accept rate on random-text bench (86% → 85%) but dramatically improves accuracy on structured GSM8K reasoning. Best of both worlds: speed preserved, GSM8K stable. Backup saved at `rejection_sampler.py.BAK_3_0p2_STABLE`.
**Gate status at (3, 0.2):** thr 1470/1500 (−2% noise), interact 180.5/165 (✅ +9.4%), E2E 6213/5000 (❌ −24%), GSM8K 0.9333/0.93 (✅ +0.33 pp thin).
**Outcome:** This is the committable CONC=4 floor. **Next:** try (2, 0.1) for safer GSM8K margin (target min ≥0.935). Then measure CONC=32 + CONC=128 with the winning stack. E2E only closes via further TPOT reduction (kernel work).

### DEC-046 — Relaxed MTP threshold `(5, 0.3)` still marginal on GSM8K — 2026-04-14 late
**Context:** DEC-045 default `(10, 0.6)` gave +30% thr but GSM8K unstable at 0.916-0.933. Tightened `rejection_sampler.py` to `(TOP_N=5, DELTA=0.3)` to push accuracy up.
**Decision:** STILL MARGINAL. 3 GSM8K runs: 0.9393 / 0.9356 / **0.9272 (fail)**. 2/3 pass, min -0.28 pp below gate.
**Rationale:** Less aggressive than (10, 0.6) — mean GSM8K moved 0.924 → 0.934 — but min still dips below 0.93. Single submission = coin flip.
**Outcome:** Must tighten further to `(3, 0.2)`. Track via 3-run stability test before locking.

### DEC-045 — **BIG WIN**: `ATOM_ENABLE_RELAXED_MTP=1` on TP=4 SR at CONC=4 — 2026-04-14 evening
**Context:** PR #411 env flag found via agent research. `atom/model_ops/rejection_sampler.py:10-16` reveals mechanism: top-10 predictions + 60% probability threshold instead of strict argmax-match rejection sampling. Prior Day 1 daily_log incorrectly claimed this required MTP-MoEFP4 model — FALSE, it works with plain MXFP4.
**Decision:** **ADOPTED with caveats.** TP=4 SR + RELAXED_MTP=1 at CONC=4 warm bench:
- Thr/GPU: 1133 → **1472** (+30%) — passes 1500 gate by -1.9% (noise distance)
- Median TPOT: 7.88 → 5.59 ms (-29%)
- Interact: 127 → **178.9** — **PASSES 165 gate by +8.4%** ✅
- Mean E2E: 8040 → 6170 ms (-23%) — still fails 5000 gate by -23%
- MTP accept rate: 54% → **86%**
- Avg toks/fwd: 2.63 → 3.58 (near MTP=3 ceiling of 4.0)
- Accept-depth-3: 18% → 64% (drafter nails all 3 tokens most of the time)

**Rationale:** Relaxed MTP is the biggest single lever discovered in the competition. Mechanism is first-party AMD (PR #411, rules-permitted per section 4.4 "ATOM-specific optimizations allowed"). SGLang has equivalent `--speculative-accept-threshold-single=0.001`.

**Caveat — GSM8K unstable at default `(10, 0.6)` thresholds:** 2 runs delivered 0.9158 (FAIL) and 0.9333 (PASS). True accuracy ~0.92-0.94, on the knife edge of the 0.93 gate. Single-submission scoring = coin flip. Not committable until thresholds are tightened. See DEC-046 for tuning iteration.

**Outcome:** Relaxed MTP is the centerpiece of the CONC=4 path. Must find `(TOP_N, DELTA)` pair where min-of-3 GSM8K runs ≥ 0.935 AND thr/GPU ≥ 1300. Then run `./dsr1_benchmark perf` for official 9-gate measurement.

### DEC-044 — TP=2 single replica crashes with GPU memory access fault — 2026-04-14 late afternoon
**Context:** Rules formula `tput_per_gpu = total_token_throughput / num_GPUs_used` (where num_GPUs_used ∈ [1..8]) suggested TP=2 SR would give huge per-GPU divisor bonus (÷2 instead of ÷4 or ÷8). Tested launch with `-tp 2 --gpu-memory-utilization 0.85 HIP_VISIBLE_DEVICES=0,1`.
**Decision:** DROPPED PERMANENTLY. Booted cleanly — DSR1 fits in 2× MI355X HBM, weights loaded, cudagraphs captured successfully. Crashed **during warmup bench** with:
- `Memory access fault by GPU node-2 ... Reason: Write access to a read-only page.`
- `Memory access fault by GPU node-3 ... Reason: Unknown.`
**Rationale:** Structural bug in ATOM at TP=2 with DSR1's 128-head MLA. Happens during LM head forward at M×64640×K7168 shape. Likely MLA or LM head sharding incorrect at world_size=2, OR MTP drafter LM head sharding incompatible.
**Outcome:** TP=2 SR dead. TP=1 won't fit (370 GB FP4 > 288 GB HBM). **Smallest viable TP = 4.** Multi-config submission uses TP=4 SR for CONC=4 and TP=8 or TP=4 SR for CONC=32/128.

### DEC-043 — BF16 GEMM tuning reverted after merge dedup corruption — 2026-04-14 afternoon
**Context:** Captured decode-weighted profile at TP=4 CONC=4 ISL=8192 OSL=256, identified untuned BF16 MLA projection shapes as a target. Built 65-shape input CSV, ran `gradlib/gemm_tuner.py --libtype hipblaslt`, produced 29+34=63 tuned rows.
**Decision:** REVERTED via `git checkout` on the 3 source CSVs (`bf16_tuned_gemm.csv`, `dsv3_bf16_tuned_gemm.csv`, `kimik2_bf16_tuned_gemm.csv`).
**Rationale:** On server startup the multi-rank aiter merge logic detected 42 duplicate shape entries (our new rows overlapped with existing dsv3/kimik2 entries), auto-resolved by keeping lowest-`us` per shape, and wrote back deduped files. Some of the hipBLASLt solutions it kept benchmarked well in isolated tuning but badly in production — warm bench gave **-20% thr (739 → 591), +179% P99 TPOT (8.21 → 22.90 ms).** Also caused a server startup race via the merge lock file.
**Outcome:** Mid-M wins exist (e.g. M=192×2112×7168 hipblaslt 19.45µs vs asm 21.85µs, +12%), but integration is broken. Retry requires: single-rank atomic merge before multi-rank launch, OR direct manual row editing with pre-dedup. Not a priority given DEC-045's +30% from relaxed MTP. Rule learned: **never `pkill -9 -f "multiprocessing.spawn"`** while gradlib tuner is alive — it kills the spawn worker child.

### DEC-042 — `--block-size 1` regresses our ATOM + MTP=3 stack — 2026-04-14 (iter 12)
**Context:** Both compass report AND second research report claimed `--block-size 1` is MANDATORY for AITER MLA optimized path. ATOM default is 16 (`atom/model_engine/arg_utils.py:34`). Predicted 0-risk config win. Tested stacked on BEST BASE TP=8 MTP=3 at CONC=4.
**Decision:** REVERTED. 739 → 577 thr/GPU (-22%), TPOT 5.76 → 7.47 ms (+30%), P99 TPOT 8.21 → 24.25 ms (+195%), TTFT 270 → 307 ms.
**Rationale:** Reports likely describe SGLang or older ATOM MLA code path. Our ATOM version's AITER MLA path handles block_size=16 correctly — smaller blocks likely 16× the paged-attention metadata overhead per decode step without a compensating kernel benefit. P99 tripled = allocator/metadata churn, not compute regression.
**Outcome:** Stop trusting "universal AITER MLA recommendations" from external reports. Our ATOM+MTP=3+native openai_server path is a specific equilibrium; report-recommended flags have now failed 4/4 times (iter10 max_split, Test1 AMD env, Test2 NCCL prio, iter12 block-size). Future flag tests must be justified by reading OUR source code, not external docs.

### DEC-041 — `TORCH_NCCL_HIGH_PRIORITY=1` regresses our stack — 2026-04-14
**Context:** Compass report said it's a standard AMD RCCL optimization. Tested stacked on BEST BASE TP=8 MTP=3 at CONC=4.
**Decision:** REVERTED. -10% thr (739→666), +81% P99 TPOT (8.21→14.86).
**Rationale:** Making NCCL streams high-priority probably competes with ATOM's dual-stream MoE path which also claims high priority. Generic AMD tuning knobs disturb our well-tuned equilibrium.
**Outcome:** Do not test again without restructuring dual-stream config simultaneously.

### DEC-040 — TP=4 single replica fails all 3 CONC=4 gates alone — 2026-04-14
**Context:** Session 6A found TP=4 SR gives 1124 thr/GPU at CONC=4 via num_GPUs_used=4 divisor (+52% vs TP=8's 738). Re-measured today to verify interact + E2E which Session 6A never recorded.
**Decision:** TP=4 SR gives 1105 thr/GPU (confirmed) BUT TPOT 7.60 ms (+32% vs TP=8's 5.76), interact 132 (fail 165), E2E 8.2 s (fail 5.0 s). **FAILS ALL 3 CONC=4 GATES ALONE.**
**Rationale:** +50% thr/GPU from divisor is offset by -32% TPOT per-token. Per-rank work doubles at TP=4, slowing decode. To pass all 3 gates, TPOT must drop to ~5.0 ms (-34% via kernel work). More tractable than TP=8's -52% TPOT requirement, so TP=4 SR remains the likely CONC=4 starting config WITH kernel wins on top.
**Outcome:** Do not ship TP=4 SR alone. It's a starting config for kernel optimization, not a standalone solution.

### DEC-039 — AMD quickstart env vars `OMP_NUM_THREADS=1 AMDGCN_USE_BUFFER_OPS=1` regress our stack — 2026-04-14
**Context:** AMD's official DSR1 ATOM quickstart script sets these. Tested stacked on BEST BASE (TP=8 MTP=3) to see if we should adopt them.
**Decision:** REVERTED. -20% thr (739→593), +163% P99 TPOT (8.21→21.56). Massive regression.
**Rationale:** AMD's quickstart gets 468 thr/GPU with these env vars (and no MTP). Our BEST BASE gets 739 without them (and with MTP=3). **AMD's env vars are tuned for their no-MTP config.** `OMP_NUM_THREADS=1` chokes Python async scheduler at low CONC, creating P99 outliers. `AMDGCN_USE_BUFFER_OPS=1` untested in isolation (possibly fine, possibly not).
**Outcome:** Do not auto-apply AMD's documented env vars. Each must be individually A/B tested stacked on our MTP=3 path. Do not trust AMD's quickstart as tuned for our use case.

### DEC-038 — MLA `max_split_per_batch=1` regression at benchmark ISL — 2026-04-14 (iter 10)
**Context:** Profile captured at ISL=128 OSL=128 showed `kn_mla_reduce_v1_ps` scaling ratio 0.82× (faster at high bs) — fingerprint of split-k oversplit. Patched `atom/model_ops/attentions/aiter_mla.py:165` from `"max_split_per_batch": 16` to `1`.
**Decision:** REVERTED. At real benchmark ISL=8192: TPOT 5.76 → 9.92 ms (+63%), thr 739 → 438/GPU (-41%). Hard fail.
**Rationale:** Short-context profile misled. At ISL=128 KV cache fits in MI355X L2 (32 MB); at ISL=8192 KV blows L2 and split-k becomes a cache-streaming coalescing pattern (multiple workgroups cooperatively walk K). Forcing split=1 at long context destroys cache efficiency.
**Outcome:** Rule written: **always profile at the real benchmark ISL** (`feedback_profile_at_benchmark_isl.md`). Never trust short-context profile for long-context interventions. Do not retry `max_split=1` without re-profiling at ISL=8192 first.

### DEC-037 — Phase 1 FlyDSL port to DSR1 is REDUNDANT — 2026-04-14
**Context:** Danish won Phase 1 by replacing AITER stage1/stage2 with FlyDSL kernels (tile 32×256×128). Investigated whether to port that pattern to DSR1.
**Decision:** REDUNDANT. AITER main has ALREADY absorbed the FlyDSL path. Startup logs confirm the 2stage selector picks `flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w3` for DSR1 shapes. The HIP kernel binary name is `moe_gemm1_0` (which is why the profile confused me earlier). Our current MoE path IS the FlyDSL path.
**Rationale:** DSR1 shape `(cu_num=256, token=N, model_dim=7168, inter_dim=256, expert=257, topk=9, fp4x2, per_1x32)` is the actual selector key (inter_dim=256 because 2048 is TP-sharded across 8 ranks, expert=257 = 256 routed + 1 shared, topk=9 = 8 routed + 1 shared). FlyDSL kernels with those names exist in `dsv3_fp4_tuned_fmoe.csv` and are picked.
**Outcome:** Phase 1 FlyDSL work is upstream in AITER main. Do not port. The 22% of decode spent in `moe_gemm1_0` is the FlyDSL floor at tile_m=32 for small-M case — further reduction requires either smaller tile compiled variants or fundamentally different MoE algorithm.

### DEC-036 — CONC=4 kernel budget built from clean decode-only profile — 2026-04-14
**Context:** After M3 drafter cudagraph NEUTRAL revert, Danish called out gambling without data. Captured first clean per-kernel profile via `atom/examples/profile_offline.py`.
**Decision:** BUILT. Category budget: MoE GEMM 22%, MLA BF16 projections 24%, MLA attention core 16%, AllReduce chain 14%, MoE routing 13%, other 11%. Memory file: `project_dsr1_conc4_kernel_budget.md`.
**Rationale:** All prior "biggest bottleneck" claims were speculation. Required to prevent another M3-style failure. Per-kernel μs/call + scaling ratios (vs CONC=32) confirm CONC=4 is fixed-overhead bound, not compute-bound.
**Outcome:** Kernel budget becomes the authoritative document for intervention planning. Any future patch must cite a specific row.

### DEC-035 — M3 drafter cudagraph NEUTRAL, reverted — 2026-04-14
**Context:** Predicted -22% TPOT from capturing MTP drafter forward into cudagraph based on "~2 ms Python overhead" napkin math.
**Decision:** REVERTED. Measurement: 5.81 → 5.78 ms TPOT (within noise). Dispatch check diagnostic confirmed replay fires on every iter ≥1, but delta was zero because actual drafter Python overhead is <0.1 ms (not 2 ms).
**Rationale:** `torch.compile(backend="eager")` on the drafter was already dynamo-traced — minimal Python dispatch between ops. The overhead I thought I was removing was already ~zero.
**Outcome:** Rule written: `feedback_profile_before_intervene.md`. Never predict a delta without a kernel-level wall-clock budget. Do not retry drafter cudagraph without new data showing Python overhead materially exists.

### DEC-034 — Execute the plan in memory, stop re-planning — 2026-04-14
**Context:** Danish called out "you make new plans everyday, what can we even achieve then?" after multiple strategic pivots mid-execution.
**Decision:** Two-stage plan (`project_dsr1_two_stage_plan.md`) is authoritative. Only re-plan when measurement invalidates a specific assumption. No strategic discussions without explicit user ask.
**Rationale:** Plan churn = zero shipped work. Re-litigating strategy every session burns engineering hours on meta-discussion.
**Outcome:** Feedback memory written: `feedback_stop_replanning_execute.md`.

### DEC-033 — MTP-MoEFP4 checkpoint is a TRITON TRAP, do NOT swap — 2026-04-14
**Context:** Session 6A research agent suggested swapping to `amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4` checkpoint for "correct config". Danish vetoed.
**Decision:** DO NOT SWAP. That checkpoint routes MoE hot path through Triton kernels ~1.5× slower than AITER CK+asm `f4gemm_bf16_per1x32Fp4_BpreShuffle_*` on MI355X. Same concern applies to `ATOM_USE_TRITON_GEMM=1` and `ATOM_USE_TRITON_MXFP4_BMM=1`.
**Rationale:** Our plain `amd/DeepSeek-R1-0528-MXFP4` checkpoint is the fast path because it keeps MoE on CK+asm. Triton on MI355X is ~40% slower for this workload.
**Outcome:** Feedback memory: `feedback_mtp_moefp4_triton_trap.md`. Supersedes Session 6A research finding #1.

### DEC-032 — MORI-EP PARKED until upstream ATOM fix — 2026-04-14
**Context:** Tried `--enable-dp-attention --enable-expert-parallel` with MORI-EP runtime. Hit two upstream ATOM bugs.
**Decision:** PARKED.
**Rationale:** (1) ATOM PR #389 regression: `alt_stream=None` in `dual_stream_moe_forward` for MTP drafter layer 61, patched locally with `if alt_stream is not None else torch.cuda.Stream()` in `deepseek_v2.py:93`. (2) After alt_stream fix, MTP drafter accept rate collapsed from 63% → 3-8% because DP-attention reshapes hidden states into the drafter with layout the drafter block doesn't handle. `atom/plugin/attention.py:726,1132` has explicit `TODO: support mtp` markers confirming this code path is unimplemented. Combined losses at CONC=128: -18% thr vs BEST BASE (MTP loss exceeds MORI-EP gain).
**Outcome:** Document upstream bugs. Retry only after AMD lands MTP+DP-attention compatibility fix. Do not retry with MTP enabled without that fix.

---

## DEC-031 — Pivot from env var sweeps to structural kernel work — 2026-04-14 Session 6B Day 2
**Context:** Day 2 morning (sprint Day 1) ran a clean env var sweep on TP=4 single replica native ATOM: iter 0 (baseline reproduction: 1099 thr/GPU CONC=4, confirms Session 6A's 1124 within 3%), iter 1 (+AITER_USE_FLYDSL_MOE family → 1136 thr/GPU CONC=4, +3.4%, KEPT), iter 2 (+HSA_ENABLE_SDMA=0 → 1116 CONC=4 -1.8%, 2773 CONC=32 -10% vs 3084 record, DROPPED), iter 3 (+--max-num-batched-tokens 131072 +--max-num-seqs 4096 +--block-size 1 from AMD's Dec 8 2025 DSR1 vLLM recipe → 1105 CONC=4 noise, 2910 CONC=32 +5% vs iter 2 but still -5.6% vs Session 6A record, MILD WIN at CONC=32 only). Danish then questioned the strategy: "if we dont know the blocker how can we solve it?" Built the full per-CONC bottleneck analysis — see memory file `project_dsr1_bottleneck_analysis_day2.md` for the full gate math + TPOT decomposition + kernel ROI table.
**Options:**
1. Continue env var sweeps (iter 4+) on TP=4 + other untested knobs (RCCL MSCCLPP, AITER_MXFP4_MOE_SF, CK_BLOCK_GEMM, etc.)
2. Test TP=2 and TP=1 single-replica (iter 4/5, cheap) THEN stop env var work entirely and commit Days 2-7 to structural kernel interventions
3. Stop all config work immediately and jump to kernel investigation now
**Decision:** Option 2. Finish TP=2/TP=1 this afternoon (~20 min), then stop env var sweeps. Commit Days 2-7 entirely to MTP-drafter-into-cudagraph + MORI-EP + MLA-decode-kernel-port in that order.
**Rationale:**
- Bottleneck analysis shows the gates require structural TPOT cuts (CONC=4 −55%, CONC=32 −43%, CONC=128 −50%) that config knobs cannot deliver. The primary blockers are:
  - **CONC=4**: Python overhead ~2 ms fixed (33% of TPOT) from MTP drafter + spec-decode postprocess running in eager Python outside the main cudagraph — touches `atom/model_engine/model_runner.py:1745`
  - **CONC=32**: MoE expert GEMMs at batch 32 with top-9 routing (42% of TPOT, ~6.5 ms) — 9 experts each seeing ~3-4 tokens per step, CK's `moe_ck2stages_gemm1/2` inefficient at those shapes
  - **CONC=128**: MoE compute + dispatch (50% of TPOT) + AllReduce (14% of TPOT, ~6 ms) — MORI-EP is AMD's published 82% dispatch latency lever
- Env var sweep ceiling math: Even perfectly stacking all remaining config knobs, CONC=4 TPOT floor is ~5.5 ms (from 6.10) → thr ~830/GPU. Gate is 1500 → gap 44%. Unreachable without structural kernel work.
- Kernel intervention math:
  - MTP drafter into cudagraph: −2 ms ALL CONCs (biggest relative effect at CONC=4 where Python is 33%)
  - MORI-EP: −4 ms at CONC=128, −1 ms at CONC=32, negligible at CONC=4
  - MLA decode kernel port: −0.15/0.3/0.8 ms at CONC=4/32/128
  - Stacked best-case cuts each CONC's TPOT by ~35% at CONC=4, ~18% at CONC=32, ~22% at CONC=128. Still short of full gates at CONC=32 and CONC=128, but the realistic best we can do in 10 days.
- Danish's Session 6B Day 1 engineering rule ("we have to do engineering") explicitly forbids continuing to twiddle without a bottleneck model. We have the model now; the rule says STOP twiddling.
- Every env var iteration costs 4-8 min wall clock. Budget for Days 2-7 (5 days × 8 hrs = 40 hours of sprint time) is better spent on 3 kernel patches at 1-2 days each.
**Outcome:** Today's afternoon (remaining): iter 4 TP=2 single replica (AITER qh64 support unknown) and iter 5 TP=1 single replica (AITER qh128 known supported). Both single-CONC=4 tests, ~6 min each. Whatever wins at CONC=4 gets added to the multi-config table.
Day 2 (Apr 15): start MTP drafter cudagraph work. Read `atom/model_engine/model_runner.py`, `atom/spec_decode/eagle.py`, ATOM's cudagraph capture path. Write a diff that moves `drafter.propose` into the captured graph.
Day 3 (Apr 16): land or abort MTP cudagraph patch. If abort, Day 3 afternoon starts MLA decode kernel work instead.
Day 4 (Apr 17): MORI-EP via container swap to `rocm/atom-dev:vllm-latest` running native `atom.entrypoints.openai_server` inside it (`/app/mori` is preinstalled in that image — unlike `danish_atom_main` which has the apt block).
Day 5 (Apr 18): MLA decode kernel port (Danny/LunNova precision-safe variant).
Day 6 (Apr 19): stack all wins, full 3-CONC re-measurement, gate count.
Day 7 (Apr 20): optional MTP=5+ AITER patch or finalize multi-config table.
Day 8 (Apr 21): GSM8K robustness + repro script.
Day 9 (Apr 22): PR draft against ROCm/ATOM + screenshots + tech writeup.
Day 10 (Apr 23): submit DSR1.
Realistic sprint outcome: 4-6 of 9 gates PASS, top-3 to top-5 DSR1 sub-rank, grand-prize probability ~10-15% (requires 9/9 per Rule 4.2).

**New canonical results row (Day 2 afternoon iter 1 best-so-far at TP=4):**

| Config | CONC=4 thr/GPU | CONC=32 thr/GPU | CONC=128 thr/GPU | GSM8K | Notes |
|---|---|---|---|---|---|
| BEST BASE TP=8 (today's repro) | 757.31 | 2345.57 (Session 6A record) | 3555.19 (Session 6A record) | 0.9462 | TP=8 champion for CONC=32/128 |
| TP=4 SR (iter 0 today) | 1099.01 (+45%) | (not measured) | (not measured) | 0.9424 | Session 6A 1124 reproduced within 3% |
| TP=4 SR + FlyDSL force (iter 1 today) | **1136.13** (+50%) | (not measured) | (not measured) | — | KEPT as TP=4 CONC=4 candidate |
| TP=4 SR + FlyDSL + SDMA=0 (iter 2 today) | 1116.05 | 2773.21 | (aborted) | — | DROPPED — SDMA=0 hurts |
| TP=4 SR + FlyDSL + 131072/4096/block1 (iter 3 today) | 1105.22 | **2910.50** | (pending) | — | CONC=32 candidate — still fails interact 50 gate (40.4) |


## DEC-030 — 10/10/10 sprint schedule commitment — 2026-04-13 Session 6B Day 1 (user directive)
**Context:** User explicitly set the remaining time budget: 10 days DSR1 sprint (Apr 14 → Apr 23), 10 days Kimi K2.5 sprint (Apr 24 → May 3), 10 days polish window for both tracks (May 4 → May 13), final submissions May 15. This compresses the original 14-day DSR1 plan and eliminates slack. Rule directive from user: "we have to do engineering" — every day has a specific deliverable or we revise.
**Decision:** Commit to the 10/10/10 structure. DSR1 lock by Apr 23 EOD, no later. Kimi starts Apr 24 regardless of DSR1 state (if DSR1 isn't fully optimized by Apr 23, it goes to the polish window).
**Rationale:**
- 20 days of sprint (10+10) is enough to establish baseline + stack Tier 1 wins + attempt 1-2 custom kernel interventions per track, but not enough for both custom kernel work AND framework bakeoffs
- 10 days of polish at the end means any gate we miss by <15% during the sprint becomes a realistic final-push target
- The alternative (spending 20 days on DSR1 alone, risking Kimi entirely) loses the $650K Kimi prize pool — ~2× the DSR1 prize pool
- Honest gate probability per track (sprint-only): DSR1 grand prize (9/9) ~10-15%, top-3 ~50-60%, top-10 ~85%. Kimi similar but with more FlyDSL lever known.
**Outcome:** 10-day DSR1 sprint structure: Day 1 TP=4 + env var sweep, Day 2 MORI-EP attempt, Day 3 consolidate, Days 4-5 MTP drafter into cudagraph kernel work, Days 6-7 MLA decode kernel port, Day 8 accuracy+repro lock, Day 9 PR draft + screenshots, Day 10 submit. Every day has a pass/fail deliverable; if Day N fails, Day N+1 pivots. No multi-day rabbit holes.


## DEC-029 — TP=4 single replica multi-config unblocked by rules text — 2026-04-13 Session 6B Day 1
**Context:** Session 6A DEC-024 established that TP=4 single replica works and gives +27-52% per-GPU throughput via the `num_GPUs_used=4` divisor in the leaderboard scoring formula. But we flagged it as blocked pending Daniel's Discord confirmation that `num_GPUs_used=4` reporting is rules-compliant, because the `dsr1_benchmark` binary hardcoded ÷8 regardless of TP. Tonight (Session 6B Day 1) the user pasted the full bounty rules text from danielhua23/amdgpu_bounty_optimization README, which we re-read carefully.
**Options:**
1. Wait for Daniel's Discord reply before executing any TP=4 strategy (blocks at least Day 1 of the 10-day DSR1 sprint)
2. Interpret the rules text as authoritative over the binary, execute TP=4 immediately
**Decision:** Option 2. TP=4 multi-config is unblocked by the rules text and does not require Daniel's Discord reply.
**Rationale:** Direct rules quotes establish this unambiguously:
> "the maximum supported configuration is TP/EP = 8. However, developers may choose smaller TP and EP sizes, as long as the model fits, and the following criteria must still be satisfied."
> "Token Throughput per GPU = concurrency × (input_length + output_length) / (mean_TTFT + output_length × mean_TPOT) / num_GPUs_you_used, num_GPUs_you_used = 1,2,...,8. (note: Since we provide a single node with 8× MI355, the maximum supported configuration is TP/EP = 8. However, developers may choose smaller TP and EP sizes, as long as the model fits)"
The `num_GPUs_you_used` variable explicitly ranges 1-8. The rules are authoritative and the binary is stale. A Discord confirmation would be nice-to-have but is not a blocker — we execute TP=4 starting Day 1 of the sprint and adjust only if Daniel explicitly contradicts the rules text.
**Outcome:** Phase 4 TP=4 single replica multi-config promoted to Day 1 first action. Session 6A measured numbers (1124/3084/4543 thr/GPU) are the Day 1 verification target. DEC-024 remains valid; this entry unblocks execution. DEC-021's "TP<8 dead" conclusion is further eroded — it only applies to TP<8 × DP multi-replica variants, not TP=4 single replica.


## DEC-028 — Phase 3 atom-vllm plugin mode DROPPED for DSR1 — 2026-04-13 Session 6B Day 1
**Context:** Phase 3 plan was to swap to `rocm/atom-dev:vllm-latest` container and use ATOM-as-vLLM-plugin (`vllm serve` with the ATOM platform plugin auto-registering) to gain access to three knobs not available in native ATOM: `--async-scheduling`, `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'`, and vLLM's mature cache manager. Container swap was executed successfully; plugin activated correctly; GSM8K ran on plugin mode at 0.9500 (better than BEST BASE's 0.9378); but two independent blockers emerged.
**Options:**
1. Keep fighting Phase 3: patch the ATOM plugin's MLA wrapper to construct `mla_modules` for the MTP drafter path, accept that the submission framework is vLLM (gray zone for DSR1 per rules)
2. Accept that plugin mode isn't viable for DSR1 this session, drop Phase 3, focus the remaining sprint time on native ATOM improvements
**Decision:** Option 2. Drop Phase 3 for DSR1 entirely. Keep `danish_atom_vllm_main` container alive for potential Kimi K2.5 work (vLLM IS an allowed framework for Track 2).
**Rationale:**
- **Blocker 1: plugin-mode MTP is unimplemented for DeepSeek.** Source evidence: `/app/ATOM/atom/plugin/attention.py:726` contains `# TODO: support mtp and sparse` and line 1132 contains `# TODO: support mtp` — ATOM developers' own TODO markers. ROCm/ATOM PR search confirms PR #544 "Support GLM-5 MTP for vLLM Pluggin" is DRAFT (opened Apr 11 2026) targeting branch `plugin_sparse_mla` which depends on open/unmerged PR #399. Both target GLM-5, not DeepSeek. Latest commit to `atom/plugin/attention_mla.py` is Mar 26 (a6ad84d). AMD is building plugin-mode MTP model-by-model starting with GLM-5; DeepSeek is not on the near-term roadmap.
- Launching `vllm serve ... --speculative-config '{"model":"amd/DeepSeek-R1-0528-MXFP4","method":"deepseek_mtp","num_speculative_tokens":3}'` crashes all 8 workers at model load with `AttributeError: 'NoneType' object has no attribute 'q_lora_rank'` in `/app/ATOM/atom/model_ops/attention_mla.py:145` — confirming the crash is the TODO materializing as an unhandled None case, not a version mismatch.
- **Blocker 2: vLLM is not a listed DSR1 framework per the bounty rules.** Rules §"Track 1" says verbatim: "Framework: AMD ATOM or SGLang". Rules §4.3: "For DeepSeek R1, the code must be mergeable into AMD ATOM or SGLang." The atom-vllm plugin runs under `vllm serve`, which is a gray-zone framework for DSR1 even with the plugin active. For Track 2 Kimi K2.5, the rule says "AMD ATOM or vLLM" so plugin mode IS allowed there.
- Phase 3 no-MTP baseline measured: CONC=4 −37%, CONC=32 −26%, CONC=128 −18% vs BEST BASE (with MTP). The −18% at CONC=128 is the tightest gap (smallest relative effect because MTP's speculative batch multiplier is less pronounced at high CONC), but still a net loss that no async-scheduling + cudagraph tweak can recover.
- Phase 3b with MTP was the only path to beat BEST BASE, and that path crashes.
- Cherry-picking PR #399 + PR #544 and adapting them for DeepSeek would be 2-4 days of plugin infrastructure work at the wrong time in the sprint (the 10-day DSR1 window is too tight to risk multi-day infrastructure work on a framework that's gray-zone for submission).
**Outcome:** Plugin mode is parked for DSR1. Native ATOM with `atom.entrypoints.openai_server` + MTP=3 is the only DSR1 submission path. `danish_atom_vllm_main` container stays alive for Kimi K2.5 sprint (Apr 24+) where vLLM is an allowed framework and async-scheduling + FULL_AND_PIECEWISE might give real wins. Wasted time: ~2 hours on the container swap, model verification, Phase 3a no-MTP benchmarks, and Phase 3b crash investigation. Knowledge gained that pays back: rules re-read (DEC-029), six new landmines documented, `/app/mori` preinstalled in vllm image (potential MORI-EP unblock for native ATOM via image reuse), and a clear understanding that plugin-mode MTP for DSR1 is an AMD roadmap item we can't accelerate.

**Landmines documented for future sessions (never re-hit):**
1. `ATOM_USE_TRITON_GEMM=1` on gfx950 forces `Mxfp4MoEMethod.use_triton=True` via `atom/model_ops/moe.py:644-651` → requires `triton_kernels` package which conflicts with AITER's ROCm-patched Triton → brick. NEVER set in any non-`rocm/atom-dev:vllm-latest` container. And even in that image, `triton_kernels` is not preinstalled, so still don't set.
2. `amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4` model hits the same triton_kernels trap at layer 0. ALSO the bounty rules pin `amd/DeepSeek-R1-0528-MXFP4` per `COMPETITION_QUICKSTART_EN.md`. PR #411's 3720 tok/s/GPU is AMD's research path, not the bounty path. Session 6A research finding #1 was wrong about swapping to MTP-MoEFP4.
3. ATOM main a6fe785 broke `Mxfp4MoEMethod` dispatch for the main body vs 108a70e — one of PRs #503/#531/#538/#547 is the culprit. Stay on 108a70e.
4. `--async-scheduling`, `--compilation-config`, `--no-enable-prefix-caching` are vLLM-only CLI flags. Not accepted by `atom.entrypoints.openai_server` (argparse error). They only work in atom-vllm plugin mode via `vllm serve`.
5. `mori` install in `danish_atom_main` is blocked by broken apt state (`rocm-hip` version conflict blocks `libpci-dev`+`libibverbs-dev`). Workaround: use `rocm/atom-dev:vllm-latest` image which has `/app/mori` preinstalled.
6. `VLLM_ROCM_USE_AITER_*` env vars are no-ops in atom-vllm plugin mode because `ATOMPlatform.get_attn_backend_cls()` returns `AiterMLABackend` directly, bypassing vLLM's own ROCm attention path that those env vars control. Source: `docs/vllm_plugin_backend_guide.md` section 3. Session 6A research listed them as Phase 3 wins — wrong for plugin mode.
7. Plugin-mode ATOM disables its own cudagraph logic (`enforce_eager=True`, `use_cudagraph=False`) and delegates to vLLM's cudagraph driver. Per `docs/vllm_plugin_backend_guide.md` section 2.2. The vLLM side has FULL_AND_PIECEWISE which is better than native ATOM's PIECEWISE, but only matters if the rest of the plugin path works, which for MTP it doesn't.


## DEC-027 — Phase 1 Tier 1 AITER-path stack is net-negative, PARKED — 2026-04-13 Session 6B Day 1
**Context:** Session 6B Day 1 tested the research-backed Tier 1 env var stack from Session 6A findings on top of BEST BASE. Stack: `GPU_MAX_HW_QUEUES=5`, `ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=131072` (up from 16384), `ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION=1`, `--gpu-memory-utilization 0.95` (up from 0.92), `--max-num-batched-tokens 32768`, `--cudagraph-capture-sizes "[1..512]"`. Also dropped `ATOM_USE_TRITON_GEMM=1`, `ATOM_USE_TRITON_MXFP4_BMM=1`, `ATOM_ENABLE_RELAXED_MTP=1` at shakedown (landmines — see Session 6B Day 1 daily log).
**Options:**
1. Lock Phase 1 stack as new BEST BASE if any net win
2. Single-knob bisect the 5 surviving knobs to find which regressed CONC=4/32 (est. 20 min)
3. Park Phase 1 and jump to Phase 3 (atom-vllm plugin container) which has higher expected upside
**Decision:** Option 3. Park Phase 1 stack. Skip the single-knob bisect. Move to Phase 3 container swap.
**Rationale:**
- Phase 1a results vs today's reproduced BEST BASE 757 / 2345 / 3555:
  - CONC=4: 703.46 thr/GPU (−7.1%), TPOT 6.10→6.56 ms, interactivity 164→152 fails 165 gate
  - CONC=32: 2261.0 thr/GPU (−3.6%), TPOT 15.65→15.95 ms, interactivity 63.9→62.7 still passes 50 gate
  - CONC=128: 3588.85 thr/GPU (+0.95%), TPOT 41.61→41.04 ms, interactivity 24.0→24.4 still fails 48 gate
  - GSM8K: 0.9462 (up from 0.9378, accuracy improved not degraded)
- Phase 1b diagnostic dropped `ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION=1` → CONC=4 unchanged at 698.04 thr, 6.57 ms TPOT. RMSNORM fusion ruled OUT as the regressor. One of the 5 remaining knobs is, but 20 min of bisect to recover 7% is lower ROI than testing the atom-vllm plugin path.
- Phase 3 plugin mode gives 3 genuinely new levers: `--async-scheduling`, `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'`, and vLLM's mature cache manager. These cannot be tested in `danish_atom_main` native ATOM server.
- Session 6A research was wrong about `VLLM_ROCM_USE_AITER_*` env vars being Phase 3 wins — they're no-ops in plugin mode because `ATOMPlatform.get_attn_backend_cls()` returns `AiterMLABackend` directly, bypassing vLLM's ROCm attention path.
**Outcome:** Phase 1 stack parked in parking lot. Session 6B Day 2 starts with container swap to `rocm/atom-dev:vllm-latest` and Phase 3 recipe from `/projects/teamA/danish/repos/ATOM_main/recipes/atom_vllm/DeepSeek-R1.md`. BEST BASE remains TP=8 + MTP=3 + FP8-KV + flydsl + dualstream (16384) — unchanged, but new floor reading at CONC=4 is 757.31 thr/GPU (+2.5% vs recorded 738.93), attributed to container rebuild + transformers 4.57.6 numerics.

**Landmines recorded for future sessions (never re-hit):**
1. `ATOM_USE_TRITON_GEMM=1` on gfx950 forces `Mxfp4MoEMethod.use_triton=True` → `has_triton_kernels()` assert → would require upstream triton install → brick AITER. Source: `atom/model_ops/moe.py:644-651` in ATOM 108a70e. Only safe in `rocm/atom-dev:vllm-latest` (pre-resolved triton matrix).
2. `amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4` model hits same triton_kernels trap at layer 0. ALSO the bounty rules pin `amd/DeepSeek-R1-0528-MXFP4` per `COMPETITION_QUICKSTART_EN.md` — **PR #411's 3720 tok/s/GPU at CONC=128 is AMD's research path, NOT bounty path.** Session 6A research finding #1 was wrong.
3. ATOM main a6fe785 broke `Mxfp4MoEMethod` dispatch for main body (not just MTP) vs 108a70e. Stay on 108a70e.
4. `--async-scheduling`, `--compilation-config`, `--no-enable-prefix-caching` are vLLM-only CLI flags. Not accepted by `atom.entrypoints.openai_server` (argparse error). They only work in atom-vllm plugin mode via `vllm serve`.
5. `mori` install blocked by broken apt state in `danish_atom_main` (`rocm-hip 2.27.7.70101-38~22.04` conflicts with `2.27.7-29e1567b`).
6. `VLLM_ROCM_USE_AITER_*` env vars are no-ops in atom-vllm plugin mode — they control vLLM's own ROCm attention path, which is bypassed when the ATOM plugin returns its own attention backend.

**New canonical results row:**

| Config | CONC=4 thr/GPU | CONC=32 thr/GPU | CONC=128 thr/GPU | GSM8K | Notes |
|---|---|---|---|---|---|
| BEST BASE (today's repro, 2026-04-13) | **757.31** | 2345.57 (from 6A record) | 3555.19 (from 6A record) | 0.9462 | New floor after container rebuild |
| Phase 1a Tier 1 stack | 703.46 (−7.1%) | 2261.0 (−3.6%) | 3588.85 (+0.95%) | 0.9462 | Parked as net-negative |
| Phase 1b minus RMSNORM fusion | 698.04 | (not measured) | (not measured) | (not measured) | Rules out RMSNORM as regressor |


## DEC-026 — DSR1 strategy: configuration-first, custom kernels last — 2026-04-13 Session 6A
**Context:** Tonight's discovery that TP=4 single replica gives +27-52% per-GPU throughput as a 1-line config change exposed a fundamental misunderstanding of the optimization problem. AMD ships AITER (their own kernel library), so we have the same kernels they have. The 2× gap from our 738 to the 1500 baseline cannot be a kernel-quality gap — it's a configuration gap. AMD's recipe to hit baselines is hiding in flags, parallelism modes, framework choice, and submission strategy, not in custom kernels they're holding back.
**Options:**
1. Continue the previous "kernel-first" plan (MTP-into-cudagraph, AR-comm overlap, MTP=5+ AITER patch, Phase 1 MLA port — multi-week kernel work)
2. Reverse the order: sweep ALL untested configuration moves first, only write custom kernels for the specific gap remaining
**Decision:** Option 2. Configuration first. Custom kernels are the SCORING BONUS on top, not the qualification path.
**Rationale:**
- TP=4 single replica trick (+52% per-GPU, 1-line change) is bigger than any custom kernel could realistically deliver
- AMD MLPerf §7 quotes "23% improvement from AITER alone" — the kernel layer they have is published and we have it
- Configuration moves are 1-2 hours each; kernel work is days each; ROI per hour is 10-50× higher
- Untested configuration moves remaining: --enable-expert-parallel at TP=4 and TP=8, --enable-dp-attention, ATOM_USE_TRITON_GEMM combo, MTP sweep at TP=4, prefix caching with AITER scale fix, SGLang + MORI PD disaggregation, multi-step scheduling
- The previous "kernel-first" plan was upside-down: 5 sessions chasing kernel optimizations while a 1-line config change was hidden in plain sight
**Outcome:** All Tier 2 custom kernel work deferred to Days 6-10 (post-Tier-1) of the new 14-day plan. Tier 1 configuration sweeps execute Days 1-5. See `project_dsr1_intervention_path_v2.md` in memory.

## DEC-025 — Multi-config submission strategy locked: TP=4 for CONC=4/32, TP=8 for CONC=128 — 2026-04-13 Session 6A
**Context:** Session 6A measured TP=4 single replica at all 3 CONCs with canonical workloads. Throughput per-GPU at num_GPUs=4 reporting beats TP=8 by +27.8% (CONC=128) to +52.2% (CONC=4). But TPOT degrades 30-56% at TP=4 because of less per-rank compute parallelism. The TPOT degradation breaks interactivity and E2E gates at every CONC. Required TPOT cuts for gates: -43% at CONC=4 (plausible), -40% at CONC=32 (plausible), **-72% at CONC=128 (NOT FEASIBLE)**.
**Options:**
1. TP=8 only for all 3 CONCs (current BEST BASE, 3/9 gates passing, no per-GPU divisor advantage)
2. TP=4 single replica only for all 3 CONCs (0/9 gates passing raw, breaks interactivity at all CONCs)
3. **Multi-config: TP=4 for CONC=4/32 (where TPOT cuts can recover gates), TP=8 for CONC=128 (where TP=4 is unreachable)**
**Decision:** Option 3. Multi-config submission. Daniel approved multi-config in DEC-022 (Session 5).
**Rationale:**
- TP=4 single replica + TPOT cuts can plausibly hit CONC=4 (1500/165/5000) and CONC=32 (3900/50/18000) gates
- TP=4 single replica at CONC=128 would need TPOT 65 → 18 ms = -72% which is beyond any realistic intervention stack
- TP=8 at CONC=128 currently 3555 thr/GPU vs gate 6000 (-41%) — closer than TP=4's interactivity disaster
- CONC=128 still requires its own attack (PD disagg via SGLang+MORI is the only known architectural path)
**Outcome:** Multi-config submission committed. Tier 1 + Tier 2 interventions targeted at TP=4 for CONC=4/32 and TP=8 for CONC=128. Different config per CONC.

## DEC-024 — TP=4 single replica is ALIVE for DSR1 — corrects DEC-021 — 2026-04-13 Session 6A
**Context:** Session 5 DEC-021 declared "all TP<8 × DP variants for DSR1 dead on gfx950" after testing 6 TP=4 × DP=2 variants that all crashed. **DEC-021 conflated TP=4 × DP=2 (multi-replica with data parallelism — genuinely dead) with TP=4 single replica (4 GPUs used, 4 idle, num_GPUs_used=4 in scoring formula).** Different code paths. Tonight Session 6A measured TP=4 single replica + MTP=3 + FP8-KV at all 3 CONCs with canonical workloads — **it works at every CONC, no crashes, MTP firing at full 55-56% accept rate, GSM8K passes**.

The reason we missed this for 5 sessions: the `dsr1_benchmark perf` binary in the bounty repo divides by 8 hardcoded regardless of actual TP. TP=4 single replica reported "531.83 thr/GPU at CONC=4" in Session 3 — looked WORSE than TP=8's 668. We dismissed the path. **The competition rules say `num_GPUs_you_used = 1, 2, ..., 8` — if you use 4, divide by 4, not 8.** Using the rules formula instead of the binary gives 1124 thr/GPU at TP=4 single CONC=4 (+52% over TP=8 BEST BASE).
**Options:**
1. Keep DEC-021 ("all TP<8 dead") and continue with TP=8 only
2. Recognize the conflation, restore TP=4 single replica as a viable path
**Decision:** Option 2. DEC-024 SUPERSEDES DEC-021 for the specific case of TP=4 single replica.

DEC-021 is still correct that TP=4 × DP=2 (and all TP<8 × DP variants) are dead due to gfx950 kernel layer bugs. But TP=4 with NO DP, single replica, is alive.
**Rationale:**
- Session 6A measured TP=4 single replica running successfully at CONC=4 (40 prompts), CONC=32 (320 prompts), CONC=128 (1280 prompts) with canonical workloads
- MTP stats from server logs confirm: `Average toks/fwd: 2.67, Acceptance rate: 55-56%, Distribution: {0: 19%, 1: 28%, 2: 19%, 3: 33%}` — same as TP=8 baseline
- No crashes, no accuracy regression visible in logs
- The DEC-021 crash class (decode_qlen=2,4 kernel assertion + nhead=32 OOB at M>4) only fires under DP+fp8 persistent-mode interactions, not at TP=4 single replica
**Outcome:** TP=4 single replica is the new strategic cornerstone. Multi-config submission (DEC-025) uses TP=4 for CONC=4/32. **DEC-021 partially superseded.** PRE-EXECUTION CHECK: confirm with Daniel via Discord that `num_GPUs_used = 4` reporting is allowed (rules text says yes, binary says no).

## DEC-023 — Strategic reframe: 6000 CONC=128 target is AMD internal stretch, not competitor floor — 2026-04-13 Session 5
**Context:** Research agent found AMD's OWN published DSR1 per-GPU best is only 864 tok/s/GPU (CONC=128 FP8+MTP3 MI300X ISL=1024, from `ROCm/ATOM/recipes/DeepSeek-R1.md`). Our BEST BASE on MI355X MXFP4 ISL=8192 is 3555 tok/s/GPU — 4.1× higher. Caveat: apples-to-oranges (different hardware, dtype, context length), but the comparison anchors the key finding: the 6000 tok/s/GPU CONC=128 target we've been chasing is an AMD internal stretch goal, not a number competitors are publicly hitting.
**Options:**
1. Continue chasing 6000 at all costs (5+ hours of risky engineering per variant)
2. Recognize 3555 is likely top-3-5 of 10 finalists and redirect effort to (a) Tier 1 cheap wins, (b) upstream contributions, (c) Kimi K2.5 Week 2 pivot
**Decision:** Option 2. Stop chasing DSR1 throughput headroom. Lock BEST BASE as final DSR1 config.
**Rationale:**
- Our 3555 is 4× the public reference. Even with aggressive apples-to-oranges discount, we're in top-tier range.
- Track 1 is capped at 10 finalists; each gets guaranteed $10k. We're almost certainly already in the money zone.
- Every hour spent on DSR1 DP experiments this session produced zero gains (all 5 variants failed).
- Kimi K2.5 has $650k prize + much less explored ceiling (AMD published only 837 tok/s/GPU). Better $-per-effort ratio.
**Outcome:** Committed. Session 5 remaining time shifts to Tier 1 DSR1 quick wins + Kimi Week 2 planning + upstream AITER issue filing.
**Note:** Actual leaderboard rank cannot be verified without opening `daniehua-dsr1-fp4-isl8192-osl1024-conc{4,32,128}.hf.space` in a real browser — Gradio spaces are not scrapable. **User should check these URLs directly.**

## DEC-022 — Kimi K2.5 is Week 2+ pivot, NOT a "while DSR1 runs" session-5 fallback — 2026-04-13 Session 5
**Context:** Earlier in the session I estimated Kimi as "30-min model swap" because ATOM main has `kimi_k25.py`. Research agent pulled the actual Kimi K2.5 `config.json` and AMD's published recipe. Reality is different.
**Kimi vs DSR1 deltas:**
- `n_routed_experts = 384` (DSR1: 256) — FlyDSL CSV won't match, needs full re-tune
- `num_attention_heads = 64` (DSR1: 128) — halves gqa_ratio, different MLA kernel path
- **No MTP head** — uses EAGLE3 via vLLM, ATOM may not support it
- `rope_theta = 50000` (DSR1: 10000) with YaRN-32 — RoPE cache rebuild
- Multimodal: MoonViT 400M vision tower requires `--mm-encoder-tp-mode data`
- AMD recipe: **vLLM v0.17.0** (not 0.15, not 0.18), ROCm 7.1.0, `VLLM_ROCM_USE_AITER=1`, TP=4, `--enforce-eager`
- vLLM 0.15 is BROKEN for Kimi; needs backports from vLLM PRs #33320 and #34501
**Options:**
1. Start Kimi as today's Session 5 fallback if Path A' fails (original plan)
2. Recognize Kimi is a 1-3 day project and defer to Week 2 with proper prep
**Decision:** Option 2. Kimi gets its own session.
**Rationale:**
- Switching frameworks mid-session (ATOM → vLLM) requires container switch, image pull, dependency debug
- EAGLE3 wiring is 4-8 hours of its own work, and ATOM vs vLLM speculative paths are totally different
- Running an unprepared Kimi baseline at 5pm IST is a high-failure path that eats hours without delivering anything
- Track 2 is the $650k prize — worth doing RIGHT not fast
**Outcome:** Draft Week 2 Kimi battle plan this session. Execute Kimi in Session 6+.

## DEC-021 — Park ALL TP<8 × DP variants for DSR1 (kernel layer blocker) — 2026-04-13 Session 5
**Context:** Session 5 tested every reasonable TP<8 × DP combination at CONC=128 for DSR1. Results:
- Path A (TP=4 DP=2 bf16 no-MTP): GSM8K passed, memory fault at CONC=128 40% under load (nhead=32 M>4 OOB bug)
- Path A capped (--max-num-seqs 16): ATOM assertion rejects at launch
- Path A-fp8 (TP=4 DP=2 fp8 MTP=1): `decode_qlen=2,4` kernel assertion
- Path A-fp8-mtp3 (TP=4 DP=2 fp8 MTP=3): same assertion
- Path A' (TP=2 DP=4 bf16 MTP=3): GSM8K=0.9045 FAIL (~5% under gate)
- Path A' no-MTP: [last test in session]

Cherry-pick of `4911f42 disable persistent mla for fp8 kvcache` rejected — commit targets `atom/plugin/attention_mla_sparse.py` which our HEAD has deleted.

Root cause of `decode_qlen=2,4` per Session 5 hardware research: LDS bank-conflict optimization. At gqa_ratio=32, 32 Q heads per wave, only qlen ∈ {2,4} leaves enough LDS banks for fp8 scale vectors. Recompile with wider LDS staging (feasible on CDNA4's 160KB LDS) could lift this but is a 2-3 day kernel engineering task, outside competition budget.
**Options:**
1. Keep debugging TP<8 variants (chase source-level ATOM patches, AITER recompile, etc.)
2. Accept TP=8 BEST BASE as final DSR1 config and move on
**Decision:** Option 2. DSR1 DP scaling is a hard kernel-layer blocker on gfx950 for our config.
**Rationale:**
- All 5 tested variants failed independently; the failure modes span 3 distinct kernel limitations
- Daniel's multi-config submission "accepted" answer is moot if no alternate config produces valid results
- Fixing the LDS bank conflict is a real upstream contribution but takes AMD kernel engineer days — we can file the issue (30 min mergeable work) but cannot ship the fix
**Outcome:** Committed. TP=8 + FlyDSL + dualstream is the final DSR1 submission config. File upstream AITER issue. Move remaining session time to Tier 1 wins, Kimi Week 2 prep, upstream contributions.

## DEC-020 — Park PATCH-005 (BF16→FP8 MLA o_proj), pivot to next lever — 2026-04-12 Session 4
**Context:** Spent ~6 hours across PATCH-004 and PATCH-005 attempting to quantize MLA o_proj BF16→FP8 to eliminate the 17.41% BF16 GEMM bottleneck. PATCH-005 (born-FP8 at construction via quant_spec override) hit the same `increment_version expects each element of the iterable to be a tensor` AOT autograd crash as PATCH-004 during cudagraph `bs=128` capture.
**Options:**
1. Keep iterating on PATCH-005 — try per_1x128 scheme instead of per_Token, or dig into `cuda_piecewise_backend.py` for invalidation
2. Park and move to next lever
**Decision:** Park. Revert all edits, document 3 retry paths in MASTER_FINDINGS, move on.
**Rationale:**
- Expected ceiling was ~5-7% overall throughput if it had landed. Already 2× over effort budget (Danish.md says "half-day" for ATOM source patch).
- Dual-stream and FlyDSL are untried cheaper levers with higher expected gains.
- Three concrete retry paths are documented so a future session can resume without starting from zero.
**Outcome:** Reverted successfully. Moved to dual-stream threshold test → confirmed +5-6% at CONC=32/128. Then FlyDSL pip install → +8.4% CONC=128, first CONC=4 interactivity gate pass. Park decision validated.

## DEC-019 — FlyDSL via pip install flydsl==0.1.2 (no monkey-patch needed) — 2026-04-12 Session 4
**Context:** Danish's Phase 1 #1 MoE kernel used FlyDSL with v917 monkey-patch (append wrapper to fused_moe.py overriding `get_2stage_cfgs`). I expected Phase 2 to need the same surgery. Recon showed:
- Current AITER already has `_flydsl_stage1_wrapper` / `_flydsl_stage2_wrapper` natively plumbed at `fused_moe.py:640-710`
- `dsv3_fp4_tuned_fmoe.csv` already contains 46 rows with `flydsl_moe1_*` / `flydsl_moe2_*` kernel names for our exact DSR1 shape (`7168, 256, 257, 9, per_1x32`)
- `is_flydsl_available()` returns False because the Python `flydsl` package isn't pip-installed in the container (AITER internally imports `flydsl.compiler`)
- When False, AITER falls back to `ck_tile::MoeFlatmmKernel` — the 15.68% bottleneck in our profile
**Options:**
1. Write a full v917-style monkey-patch (hours of work)
2. `pip install flydsl==0.1.2` to flip `is_flydsl_available()` to True and let AITER's existing native path serve the tuned CSV config (seconds of work)
**Decision:** Option 2. One-line fix.
**Rationale:**
- The entire infrastructure is already built. AMD integrated the FlyDSL path natively after Phase 1 (likely informed by Phase 1 MoE leaderboard submissions).
- Version must be exactly 0.1.2 (AITER pins this; 0.1.3 raises `ImportError: Unsupported flydsl version`).
- Zero code changes to touch, trivial to revert, and the tuned CSV does all the dispatch work.
**Outcome:** Applied successfully. First CONC=4 interactivity gate pass of hackathon (167.37 > 165 vs baseline 160.40). CONC=32 +3.3% throughput. CONC=128 +8.4% throughput. GSM8K stable in 0.938-0.945 range. Locked into BEST BASE.
**Implication for submission**: `pip install flydsl==0.1.2` must appear in our reproduction script. Most Phase 2 teams will never notice the `flydsl unavailable` log line; this is a hidden cliff-edge optimization that gates a real multi-percent win.

## DEC-018 — Raise ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD from 1024 → 16384 — 2026-04-12 Session 4
**Context:** ATOM has a dual-stream MoE path where shared experts run on an alt CUDA stream in parallel with routed experts. Default env var `ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=1024` gates the path on `num_tokens ∈ (0, threshold]`. Our ISL=8192 prefill has `num_tokens > 1024`, so prefill was excluded by default. Decode (at small token counts) was already using dual-stream.
**Options:**
1. Keep default 1024 (prefill runs single-stream)
2. Raise to 16384 to cover prefill too
3. Set to 0 (disable entirely — sanity check)
**Decision:** Raise to 16384.
**Rationale:**
- Source file `atom/models/deepseek_v2.py:780` has a commented-out `# and not get_forward_context().context.is_prefill` — AMD flirted with guarding prefill out but left it unguarded. The commented code suggests uncertainty, not a definitive "this is bad for prefill" signal.
- Cheap to test, zero code changes (env var only), easy to revert.
**Outcome:** CONC=4 -2% (within ±3% noise, effectively flat), CONC=32 **+5.3% throughput**, CONC=128 **+6.1% throughput**. GSM8K 0.9447 unchanged. Clean win at CONC=32/128, neutral at CONC=4. Kept in BEST BASE.
**Notes for submission**: include the env var in the reproduction script. Consider upstreaming a PR that raises the default from 1024 since our workload is a likely common ISL for inference benchmarks.

## DEC-001 — Start with ATOM for Track 1 (DSR1) — 2026-04-10
**Context:** Need to choose between ATOM and SGLang for DeepSeek-R1 optimization.
**Options:**
1. ATOM — AMD's own framework, allows AMD-specific optimizations
2. SGLang — community framework, must be vendor-agnostic
**Decision:** Start with ATOM, also baseline SGLang for comparison.
**Rationale:**
- ATOM allows AMD-specific kernel optimizations (Danish's Phase 1 kernels are AMD-specific)
- Code mergeability is easier for ATOM (AMD controls the repo)
- ATOM gives ~1.2x throughput over vanilla vLLM per AMD benchmarks
- TP is flexible in ATOM (can try TP=4), while SGLang quickstart locks TP=8
- SGLang quickstart says "ATOM might be the better choice" for DSR1
**Outcome:** Pending — awaiting baseline numbers from both.

## DEC-002 — Baseline all 3 backends before optimizing — 2026-04-10
**Context:** Should we jump into optimizing one backend or measure all first?
**Options:**
1. Pick ATOM and start optimizing immediately
2. Baseline all 3 backends (ATOM, SGLang, vLLM) first, then choose best
**Decision:** Baseline all 3 first.
**Rationale:** We don't know which backend performs best out-of-the-box on MI355X. The fastest default baseline gives us the best starting point. Spending 1 day on baselines saves us from optimizing the wrong framework.
**Outcome:** Pending — ATOM baseline in progress, SGLang and vLLM not yet started.

## DEC-003 — Use /usr/bin/docker workaround for AppArmor — 2026-04-10
**Context:** Docker wrapper in ~/bin/ has AppArmor restrictions blocking /opt/rocm/ access inside containers.
**Options:**
1. Use ~/bin/docker wrapper (Maharshi's recommendation) — blocked by AppArmor
2. Use /usr/bin/docker directly with GnSight's workaround flags
**Decision:** Use /usr/bin/docker with GnSight's workaround.
**Rationale:** GnSight confirmed this works. The wrapper blocks access to ROCm libraries inside containers. Maharshi acknowledged the issue and will fix, but we can't wait.
**Outcome:** Working — ATOM container launched successfully, GPUs visible, PyTorch imports fine.

## DEC-004 — Pin ATOM and AITER to known-good commits — 2026-04-10
**Context:** Daniel's quickstart notes these commits work together. Both repos are fast-moving.
**Options:**
1. Use latest HEAD of both repos
2. Pin to danielhua23's reference commits
**Decision:** Pin to ATOM 33e0aac and AITER cbbdc50.
**Rationale:** Daniel explicitly said "they work for me at these commits." Using HEAD risks breaking changes. We can update later if needed.
**Outcome:** Working — both installed successfully, ATOM imports clean, server runs, accuracy 0.9447.

## DEC-005 — TP=4 is top priority for Session 2 — 2026-04-10
**Context:** Baseline numbers show throughput/GPU is the biggest gap (566 vs 1500 at CONC=4, 2003 vs 3900 at CONC=32).
**Options:**
1. Optimize kernels first (MLA, MoE, GEMM) to reduce TPOT
2. Try TP=4 first to double throughput/GPU via the scoring formula
3. Tune ATOM flags (batching, memory, etc.)
**Decision:** Try TP=4 first.
**Rationale:** The scoring formula divides by num_GPUs_used. TP=4 means dividing by 4 instead of 8, literally doubling throughput/GPU if latency doesn't degrade too much. DSR1 model ~155GB fits on 4x 288GB GPUs. At CONC=32, TP=4 would push throughput from ~2003 to ~4007, passing the 3900 target. This is the single highest-ROI experiment.
**Outcome:** FAILED — TP=4 drops GSM8K accuracy to 0.928 (two runs), below 0.93 threshold. The MXFP4 quantization loses precision when sharded across fewer GPUs. TP=4 is not viable unless accuracy can be recovered. Next: try TP=6 or focus on kernel optimizations with TP=8.

## DEC-006 — MTP speculative tokens 1→3 is the biggest config win — 2026-04-11
**Context:** Default ATOM MTP uses 1 draft layer. Research Report says 3 is optimal for 1.5-1.8x speedup.
**Options:**
1. Keep default MTP (1 draft token)
2. Set `--num-speculative-tokens 3`
3. Try higher values (5+)
**Decision:** Set `--num-speculative-tokens 3`, then sweep other values.
**Rationale:** Profiling showed MTP acceptance rate was 89% — excellent. But only 1 draft token was being used. Going to 3 draft tokens means each forward pass generates ~2.4 tokens on average instead of ~1.89.
**Outcome:** Full sweep complete (MTP=1,2,3 tested at all CONC. MTP=4 crashes, MTP=5 not supported):
- MTP=3 best for CONC=4: TPOT 7.72→6.80ms (-11.9%), Interactivity 129→147 (+13.6%)
- MTP=2 slightly better for CONC=32: TPOT 17.35→16.37ms (vs MTP=3 16.72ms)
- No MTP effect at CONC=128: all values ~45-46ms
- Decision: Use MTP=3 (biggest win at hardest concurrency)

## DEC-008 — TP=8 cannot pass CONC=4 throughput, must reduce TP — 2026-04-11
**Context:** Throughput/GPU = CONC × 9216 / (TTFT + 1024×TPOT) / num_GPUs. At TP=8, even with TPOT=0ms, throughput maxes out at ~566-668 tok/s/GPU at CONC=4. Target is 1500.
**Options:**
1. Stay TP=8, optimize TPOT only — CANNOT reach 1500 (math proves it)
2. TP=5 — divides by 5 not 8 = 1.6x throughput. Vocab divisible. Accuracy unknown.
3. TP=4 — divides by 4 not 8 = 2x throughput. Accuracy fails (0.928). Needs recovery.
4. Expert Parallelism — different GPU distribution. Unknown effect on throughput formula.
**Decision:** Test TP=5 immediately. Also explore TP=4 accuracy recovery paths.
**Rationale:** The math is undeniable. No amount of kernel optimization at TP=8 can pass CONC=4 throughput threshold. TP reduction or EP is required to qualify for prize.
**Outcome:** TP=5 crashes (AITER custom allreduce world_size=5 unsupported). TP=4+DP=2 identified as the real solution.

## DEC-009 — TP=4 × DP=2 with --enable-dp-attention is the winning config — 2026-04-11
**Context:** TP=8 can't pass CONC=4 throughput (math impossible). TP=5 crashes. TP=4 accuracy fails. Research found TP=4×DP=2 solution.
**Options:**
1. TP=4 alone (accuracy fails)
2. TP=4 + DP=2 + --enable-dp-attention (all 8 GPUs, divides by 4, DP attention preserves accuracy)
3. TP=4 + DP=2 + --enable-expert-parallel (adds EP on top)
**Decision:** Try TP=4 + DP=2 + --enable-dp-attention + MTP=3 immediately.
**Rationale:** AMD's own blog confirms "DP2/TP4/EP4 is ~45% better on throughput compared to DP1/TP8/EP8." DP attention avoids KV cache duplication which may fix the accuracy issue. All 8 GPUs stay utilized. Throughput divides by 4 not 8 = potential 2x throughput/GPU score.
**Outcome:** Server launched but crashed during accuracy test (timeout errors + server crash). DP=2 configuration may not be stable on this ATOM/AITER version, or needs longer warmup. Need to investigate further — check ATOM docs for DP=2 requirements, try without --enable-dp-attention, or try with --enable-expert-parallel separately.

## DEC-010 — TP=4 crashes were AppArmor, not ATOM bugs — 2026-04-12
**Context:** Yesterday we thought TP=4 was fundamentally broken due to AITER MLA kernel crashes (memory access faults). We spent hours trying fixes.
**New information:** Maharshi rolled out AppArmor relaxation. divc13 reported AppArmor was blocking Unix sockets for TP>1. We retested TP=4 after the rollout.
**Decision:** Retest TP=4 with default AITER MLA backend (no patch).
**Rationale:** The AppArmor fix specifically addressed Unix socket blocking for multi-GPU communication. This could have been causing the memory access faults.
**Outcome:** CONFIRMED — ATOM TP=4 no longer crashes. Full accuracy test completes normally. The "crashes" were AppArmor blocking inter-process comm, NOT AITER MLA kernel bugs. BUT accuracy is still 0.928 (below 0.93 threshold) — that's a separate issue (MXFP4 precision at TP=4).

## DEC-011 — Accuracy must be ROBUST, not borderline — 2026-04-12
**Context:** TP=4 accuracy is 0.928. Gap is only 0.002. Natural variance in GSM8K evaluation is ±0.005.
**Options:**
1. Aim for borderline pass (0.930 on lucky run)
2. Aim for robust margin (≥0.935 consistently)
**Decision:** Robust margin only.
**Rationale:** Competition Rule 4.2 says code must be mergeable into AMD repositories. AMD won't merge code that passes accuracy 1 time in 10 due to variance. A borderline pass might work once but fail their reproduction attempts.
**Outcome:** Pending — need experiments that push TP=4 to 0.935+ consistently (MTP tuning, different framework, kernel fixes).

## DEC-017 — Measurement protocol: CONC=4 for knob filtering, full sweep for final baseline only — 2026-04-12
**Context:** Each full CONC=4/32/128 experiment cycle is ~25 min. With 10+ knobs to test, that's 4+ hours per sweep. We need faster iteration without losing reliability.
**Decision:** 
- Knob filtering (steps 2-8): CONC=4 only, single run, ~5 min each. Keep knobs that give ≥2% improvement, drop hurts.
- Final "BEST BASE" (step 9): full CONC=4/32/128, 2 runs each, average. Locks in pre-kernel baseline.
- Exception: high-concurrency-specific knobs (like DP=2) tested at CONC=128 only.
**Rationale:** CONC=4 is our weakest metric (interactivity 163.92 vs 165 target — one knob away), TPOT is most sensitive to optimization at long decode, and knob effects strongly correlate across concurrencies. Single runs are fine for >5% deltas (ATOM main vs pin was 10.6%, well above noise). 2-5% variance only matters for borderline gains.
**Outcome:** Committed. Protocol documented in Optimization.md top section.

## DEC-016 — Realistic CONC=4 throughput target: ~1000 tok/s/GPU, not 1500 — 2026-04-12
**Context:** After 2 sessions of TP=4 / DP=2 / SGLang / vLLM experimentation, no config gets close to 1500 tok/s/GPU. Time to check whether the target is physically achievable.
**Evidence (FINDING-005)**:
1. AMD's own ATOM recipe ([recipes/DeepSeek-R1.md](https://github.com/ROCm/ATOM/blob/main/recipes/DeepSeek-R1.md)) publishes ZERO numbers below CONC=128 for DSR1-MXFP4. Best: 1,732 tok/s/GPU at CONC=128 / ISL=1024 — and our workload is 8× longer prefill with 32× less batching.
2. Our current 738 tok/s/GPU at CONC=4 IS the canonical recipe verbatim (TP=8 + MTP=3 + FP8 KV). No secret flag exists.
3. AMD's MI355X distributed inference article quotes headline wins only at CONC 64-128. Low concurrency is "competitive" not "crushing."
4. Cross-framework evidence that DP+MTP (the obvious fix) is broken on MI355X: SGLang #21942, #20404.
**Options:**
1. Keep chasing 1500 → guaranteed time waste, no config reachable
2. Accept ~1000 as real ceiling, optimize toward that
3. Focus scoring on CONC=32/128 where we have more headroom
**Decision:** Option 2 + 3. Target ~1000 tok/s/GPU at CONC=4, not 1500.
**Rationale:** Competition scoring (sub-rank based) means partial-credit is real. 738 tok/s/GPU at CONC=4 already beats teams that can't get their server to launch. Focus on closing interactivity gap at CONC=4 (163.92→165+, trivial), passing CONC=32 throughput (currently 2156 vs 3900), and accuracy robustness. Don't waste cycles on unreachable targets.
**Outcome:** Target updated in Optimization.md. Measurement table gate status reflects realistic deltas.

## DEC-015 — ATOM-only, drop SGLang and vLLM — 2026-04-12
**Context:** Tested SGLang TP=4 Triton (crashed on `deepseek_v2.py forward_absorb_fused_mla_rope_prepare` with nhead=32 dim mismatch, same class of bug as ATOM pin had). Research confirmed vLLM has AITER MLA support but requires community-gated PR for mergeability.
**Options:**
1. Multi-framework strategy — try ATOM, SGLang, vLLM each and submit winner
2. ATOM-only — commit to one framework for both tracks
**Decision:** ATOM-only. Both tracks (DSR1 + Kimi K2.5) go through `danish_atom_main`.
**Rationale:**
- **Mergeability (Rule 4.2)** — ATOM is AMD's own repo. One PR to `ROCm/atom` lands clean. vLLM/SGLang PRs are community-gated, slower, require vendor-neutral code (no AMD-only logic, proper fallback paths), and may be rejected by upstream reviewers who don't care about ROCm.
- **Kimi K2.5 support verified**: `kimi_k25.py` exists in ATOM main. Quickstart dir is named `kimik25-fp4-vllm-mi355x` but that's just a directory name — ATOM should work. Will confirm with Daniel.
- **Consistent optimization pipeline**: all knobs, all env vars, all fusions, all kernels target one framework. Don't waste time rediscovering them per framework.
- **Merge story in submission**: "here's my Phase 1 kernel, integrated into AITER, tested in ATOM, PR ready" — clean narrative vs "here's my kernel with 3 different integration patches for 3 frameworks."
**Outcome:** `danish_sglang` and `danish_vllm` containers stopped. Research dropped. All future work on `danish_atom_main`. Documented in Optimization.md "DO NOT TRY" table.

## DEC-014 — Winning config must use all 8 GPUs (DP=2 × TP=4) — 2026-04-12 (UPDATED by DEC-015)
Original decision: "Use DP=2 × TP=4 to saturate all 8 GPUs." Outcome: empirically failed. DP=2 broken with MTP+FP8-KV (crash), broken with MTP+BF16-KV (accuracy 0.0159 garbage), suboptimal without MTP (341 thru/GPU, WORST of all configs). DP=2 is only viable at high CONC — test at CONC=128 only, never CONC=4. See SGLang #21942, #20404 for cross-framework confirmation DP+MTP is broken on MI355X.

## DEC-013 — REVERSE DEC-012: ATOM main has TP=4 fix, use it — 2026-04-12 (late)
**Context:** DEC-012 concluded ATOM TP=4 was dead based on the pinned commit `33e0aac`. But I never checked `origin/main` for upstream fixes. After fetching, found commit `26bb804 fix deepseek tp 4 mtp mla metadata error (#460)` plus `#484` plus a full head-repeat mechanism (`_MLA_MIN_HEADS` + `padded_num_heads`) in `attention_mla.py`. The fix has been on main for weeks.
**Options:**
1. Stick with pinned commit and the pivot to SGLang/vLLM (DEC-012)
2. Update ATOM to main in a fresh container, test TP=4 there
**Decision:** Update ATOM to main. Keep `danish_atom` untouched as a fallback; use a new `danish_atom_main` container for the test.
**Rationale:** The fix exists. Testing it is ~30 minutes of JIT recompile. If it works, we unblock the entire TP=4 path *and* get Kimi K2.5 support for Track 2 as a bonus. If it doesn't, we lose nothing — original container is fine.
**Outcome:** BREAKTHROUGH. Server launches clean at TP=4 + MTP=3. GSM8K 0.9431 — passes 0.93 robustly. No crashes on real workload. CONC=4 throughput/GPU 531 (still below 1500 but that's a different problem — see DEC-014). **The two sessions we spent on PATCH-001/002/003 were diagnosing a bug that was already fixed upstream.** Lesson logged in MASTER_FINDINGS: always fetch `origin/main` before patching.

## DEC-014 — Winning config must use all 8 GPUs (DP=2 × TP=4) — 2026-04-12 (late)
**Context:** ATOM main TP=4 passes accuracy but perf is *worse* than TP=8 (531 vs 668 tok/s/GPU). The benchmark reports `total_throughput / 8` (full node GPUs), so a single TP=4 replica leaves 4 GPUs idle and takes a throughput hit.
**Options:**
1. Pure TP=4 (4 GPUs idle) — loses to TP=8
2. `DP=2 × TP=4` — 2 replicas, 8 GPUs saturated, ~2x throughput
3. `DP=4 × TP=2` — 4 replicas, but 2 GPUs per replica can't fit DSR1 (~155GB model)
4. Expert parallelism (EP) variants
**Decision:** `DP=2 × TP=4 + MTP=3`. Launch with `-tp 4 -dp 2`.
**Rationale:** This is the only config that uses all 8 GPUs *and* has TP small enough to beat TP=8's TPOT. DP=2 doubles the numerator of the throughput formula while the denominator stays at 8 — net ~2x the per-GPU score. Accuracy should match TP=4 alone since DP is just replication. MTP=3 stays because we already validated it's the best setting.
**Outcome:** Pending — next experiment. Expected throughput/GPU ~1000-1100, still possibly below 1500 at CONC=4 but a legitimate starting point for further optimization (kernel integration, larger batch sizes, etc).

## DEC-012 — ATOM TP=4 is dead by evidence, pivot to SGLang/vLLM — 2026-04-12 (SUPERSEDED by DEC-013)
**Context:** PATCH-003 successfully engaged AITER's gfx950 `nhead=32 + fp8 + max_seqlen_q=4` native fast path (verified by runtime print). Server launched clean. But during the real accuracy workload, shape became `[65, 32, 576]` (M=65 ≈ 16 sequences × 4 MTP tokens) and 4 GPUs memory-access-faulted simultaneously after ~4 successful calls at that shape.
**Options:**
1. Keep debugging ATOM — maybe another knob, max_num_seqs=1, force-single-sequence mode
2. Patch the AITER gfx950 ASM kernel — beyond hackathon scope (hand-written assembly)
3. Pivot to SGLang TP=4 with `--attention-backend triton` (bypasses AITER MLA entirely)
4. Pivot to vLLM v0.19.0 TP=4 with `VLLM_ROCM_USE_AITER_MLA=0` (keeps AITER MoE)
**Decision:** Pivot. SGLang Triton first (cleanest), vLLM as backup.
**Rationale:** Every AITER MLA code path for nhead=32 has now been tested and fails:
- Native gfx950 path: crashes on M>4 (this finding)
- Simulated nhead=16 path: crashes on compressed MLA KV cache
- Standard AiterBackend: can't handle MLA compressed KV
- Only nhead=128 (TP=8) works, and math proves that can't pass CONC=4 throughput
This is not a bug we can fix from Python — it's in AMD's hand-written ASM. Time-box: we already spent 2 sessions on ATOM TP=4. The finding itself (AITER kernel bug report) is a mergeable contribution. Move forward with SGLang.
**Outcome:** Pending — SGLang TP=4 + Triton setup is next experiment. Expected to work because Triton MLA doesn't have head-count constraints.

## DEC-007 — Profile with real workload, not short prompts — 2026-04-11
**Context:** Initial profile with 5 short requests showed MLA at 0.8% — misleading.
**Options:**
1. Trust short-prompt profile
2. Re-profile with ISL=8192, OSL=1024 at all concurrencies
**Decision:** Re-profile with real workload.
**Rationale:** Short prompts don't exercise the prefill or KV cache paths that dominate at ISL=8192.
**Outcome:** Real profile showed MLA at 13.5% (not 0.8%), MoE at 21.5%, BF16 GEMM at 17.4%. Completely different picture. Profile is stable across CONC=4/32/128. Lesson: always profile with actual benchmark workload.

---

## 2026-04-19 — Session 7 consolidated findings (post-research-report sprint)

### Gates status at session close

| Metric | Gate | Floor | Session-7 bench | Pass |
|---|---|---|---|---|
| thr/GPU | ≥1500 | 1361 | 1341 | ❌ −9% |
| median TPOT | — | 6.35 ms | 6.47 ms | — |
| interactivity | ≥165 | 157.55 | 154.63 | ❌ −3% |
| E2E median | ≤5000 ms | 6842 | 7009 | ❌ +40% |
| GSM8K | ≥0.93 | 0.934 | 0.9356 | ✅ |
| **Total** | 4 | **1/4** | **1/4** | |

**No new gates passed this session**. Bench numbers are within noise of the locked floor.

### TP=8 data point (parked)

Tested floor config at `-tp 8` across all 8 GPUs: **842 thr/GPU, 5.11 TPOT, 195.78 interact, 5511 E2E, 0.9303 GSM8K → 2/4 gates**. First config ever to pass interactivity. But **thr crashes −44%** because total throughput only grows +24% while divisor doubles — CONC=4 is launch-latency-bound, not compute-bound. TP=8 is a CONC=32/128 track play, NOT CONC=4.

### Research-report code-citation verification

Cross-checked the Apr 18 research tear-down against live code via Explore agent:

| Claim | Status |
|---|---|
| `aiter/mla.py:330-362` gfx950+qh32+fp8+max_seqlen_q=4 dispatch gate | ✅ CONFIRMED |
| `aiter/mla.py:380` use_qseqlen_fold for 48/64/96/128 heads | ✅ CONFIRMED |
| `atom/model_ops/attention_mla.py:680` MLA metadata build (research said 680) | ⚠️ Line drifted — actual at 569-590 |
| `atom/model_engine/model_runner.py:1741` capture_cudagraph | ⚠️ Line drifted — actual at 1905 |
| `atom/model_ops/rejection_sampler.py` `RELAXED_TOP_N=8, RELAXED_DELTA=0.5` | ✅ CONFIRMED |
| SGLang `eagle_draft_cuda_graph_runner.py` | ❌ Not on server filesystem; **cloned** to `/tmp/sglang_ref` for B3 port |
| `gather_kv_b_proj.py:29` `per_row_scale` assertion | ✅ CONFIRMED |

### Dead-on-arrival findings

- **Patch #4 MLA flatten** (SGLang commit 1ad8a0d, Apr 17 2026): equivalent optimization already merged into native `atom/model_ops/attention_mla.py` in October-December 2025 (git blame: `a73f7bca`, `f58c89aa`, `958f0e6e`, `20165596`). 0 ms to port.
- **Lever C prefix-cache ones-scale hack**: confirmed breaks GSM8K to 0.77 (research).
- **Lever C real fix (scale-format guard + weight_preshuffle=False)**: both v1/v2 crashed with Memory access fault on 2nd prefill batch. `gather_kv_b_proj` Triton kernel has multiple FP8-quant baked assumptions beyond scale format. Kernel rewrite required — out of sprint scope.

### Sustained-load class of failure (new)

Lever B v5 (drafter HIP graph with lazy capture) **survived** 236k tokens + 404 successful lazy captures across bs ∈ {1..65} × step_type ∈ {0,1}, then crashed on bench #2 during layer-61 drafter MoE forward. Not a design bug in the lazy-capture approach — an allocator/pool state cliff hit after hundreds of shape-specific captures. This is a NEW failure class to track: graph-based optimizations must account for allocator behavior at production scale, not just correctness at first-call scale.

### gfx950 qseqlen=4 ceiling — structural, needs kernel surgery

Research + code confirm: `mla_a8w8_qh32_qseqlen4_gqaratio16_ps` hard-asserts `max_seqlen_q == 4` at `aiter/mla.py:330-362` for the qh32 FP8 KV persistent-mode path. Lifting this requires a custom HIP kernel using HipKittens patterns (8-wave ping-pong, 4-wave interleave, chiplet swizzling) estimated at ~600-800 LOC. Until this lands, we cannot:
- Use MTP≥4 (qseqlen=5+)
- Use tree speculation with >4 total verified positions per sequence
- Use qseqlen-extended EAGLE variants

### Artifacts produced this session

- `dsr_beta/scripts/lever_b_drafter_graph.py` — Lever B v5 patch (apply/revert/verify, 3 fixes merged)
- `dsr_beta/scripts/drafter_fp4_transplant.py` — Phase B1 transplant reproduction
- `dsr_beta/scripts/README_TRANSPLANT.md` — transplant recipe + audit doc
- `dsr_beta/scripts/run_hipblaslt_retune.sh` — Phase A1 tuner wrapper
- `/tmp/sglang_ref/python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py` (server) — SGLang 432-LOC ref for B3

### Plan for remaining sprint window

Per Danish's "never prematurely dead, always optimized" directive:

1. **Phase A1 in flight** — hipBLASLt retune on 112 prefill/decode BF16 shapes. Tuner running with 4-GPU parallelism. Target: cut the "not found tuned config" warnings that currently dispatch to `torch solution:0` (default heuristic), which can be 5-20% slower than tuned on prefill shapes. **First new-perf lever with credible landing probability.**
2. **Phase B3** — SGLang `EagleDraftCudaGraphRunner` port to ATOM's `EagleProposer`. Replaces Lever B v5 with pre-allocated boot-time buffers (typed `EagleDraftInputBuffers` dataclass) and different capture strategy that should sidestep v5's allocator cliff.
3. **Phase B2** — P-EAGLE K=3 port (vLLM PR #32887). Parallel drafting at q_seqlen=4 fits the gfx950 ceiling exactly. Accuracy risk (DeepSeek MTP wasn't trained for mask tokens) — must gate on GSM8K ≥0.93 min-of-3.
4. **Phase C1** — Custom HIP MLA kernel escape-hatch, only if B2/B3 don't land 4/4. Multi-day kernel work in HipKittens vocabulary.

**Rules in force**: autonomous mode, CONC=4 only until 4/4, GitHub ONLY on new record, always optimized never naive, never prematurely declare dead.


## 2026-04-19 — Session 8 late evening findings (B2 tested, C1 HK port initiated)

### Gates status at session-8 close

Unchanged from session-7. **Floor 1361/6.35/157/6842/0.934 → 1/4 gates**. Zero benchmarks this session.

### B2 P-EAGLE position-only gamble — TESTED + REVERTED

First actual bench of the training-free P-EAGLE K=3 parallel drafter:
- **30.45% accept rate** (vs chain MTP-3's ~66%)
- **1.9 tokens/step accepted** (vs chain's 3.0)
- **−31% throughput regression**

Root cause confirms research prediction: DeepSeek MTP was trained causally (predict t+1 from hidden at t) with ZERO mask-token exposure. Position-only init repeats base token + hidden across all K+1 positions, differentiated only by RoPE. Drafter's head cannot classify positions t+2/t+3 without learned `emb(mask)` + `h_shared`. Near-zero accept at later positions.

Reverted via `.pre_lever_b2` backup.

### C2 tree-spec short-patch proved DEAD in our stack

Plan claimed "top-2 at depth i=2 gives 4 verification positions → fits qseqlen=4 natively". **Math error** — chain MTP=3 already uses qseqlen=4 (1 base + 3 drafts). Adding a 4th draft → qseqlen=5 → crashes `mla_a8w8_qh32_qseqlen4_gqaratio32_ps`.

Every real C2 variant evaluated:

| Variant | LOC | Expected delta vs floor | Verdict |
|---|---|---|---|
| (a) Top-K rescoring at i=2 via drafter logprob | ~30 | +0% — RELAXED_TOP_N=8 at `rejection_sampler.py:11` already absorbs plausible drafter candidates | No-op |
| (b) Dual-chain bs×=2 verify, shared drafter | ~150 | Flat/neg: +5-15% accept × 2× verify cost = −20-30% net thr | Kills perf |
| (c) True tree with custom per-query attention mask | >1000 | +15-20% accept | = C1 (needs kernel mask) |

**AITER MLA kernel has NO per-query attention mask support.** Shared-prefix tree is fiction at kernel level — shared prefix only means duplicated KV entries + separate reads = 2 independent bs=1 decodes of compute.

### C3 MTP=4+ explicitly blocked on C1

`atom/config.py:882` hardcoded `if num_speculative_tokens > 4: raise ValueError`. Lifting requires qseqlen=5-8 kernel. No such kernel exists on gfx950. `hsa/codegen.py` (206 LOC) is a CSV→C++-header compiler, NOT a kernel generator — reads `mla_asm.csv` which has qseqlen ∈ {0, 2, 4} and emits `asm_mla_configs.hpp`. Adding qseqlen=8 needs new `.co` kernel blobs (hand-written GPU assembly) OR new HIP kernel source.

### 🎯 BIG DISCOVERY: HipKittens MLA already in tree

At `/app/aiter-test/csrc/kernels/mla/hk/`:
- `hk_mla_buffer_managers.cuh` (1546 LOC)
- `hk_mla_softmax.cuh` (272 LOC)
- `hk_mla_utils.cuh` (16 LOC)
- `mi3xx_v32_fwd_decode_h128_fp8_fp8.cuh` (812 LOC)

Python binding `aiter.hk_mla_decode_fwd` at `aiter/ops/attention.py:1294`. Module `module_hk_mla`, JIT-compiled via `@compile_ops`.

Dispatch gate at `aiter/mla.py:429-437`: `use_hk = (nhead==128 and q.fp8 and kv.fp8 and page_size==1 and AITER_ENABLE_EXPERIMENTAL)`. Currently **never taken** at TP=4 (we have nhead=32).

Key properties of the HK kernel that matter:
- `HkMlaDecodeFwdTraits` template parameterized on `kQoNumHead_`
- FP8 E4M3 baked in for both q and KV
- DeepSeek MLA shape baked in: `kKvLoraRank=512, kQkRopeHeadDim=64, kQkHeadDim=576, kVoHeadDim=512`
- `max_seqlen_q` is RUNTIME param (driven by work_info_set, NOT baked into traits)
- Blocker: `static_assert(kBlockM == kQoNumHead, "Only supports nhead=128!")` at line 36
- VGPR constants (k_o_sz=128, k_q_nope_sz=32, etc.) sized by **kTileM** (=16), not kBlockM directly → stay same at kQoNumHead=32 if kTileM preserved at 16 with kNumWarps=2

### C1 port status at session-8 close

Danish authorized full port: "timing not constraint, build AMD optimized kernels".

**Patches deployed** (backups `.pre_c1`):
| File | Change |
|---|---|
| `/app/aiter-test/csrc/kernels/mla/hk/mi3xx_v32_fwd_decode_h32_fp8_fp8.cuh` | NEW 9KB — h32 traits (kBlockM=32, kNumWarps=2, kTileM=16) + wrapper reusing h128 kernel body via template |
| `/app/aiter-test/csrc/kernels/mla/hk_decode_fwd.cu` | Added `num_head==32` dispatch branch |
| `/app/aiter-test/aiter/jit/optCompilerConfig.json` | h32 header added to `module_hk_mla` srcs |
| `/app/aiter-test/aiter/mla.py:330-437` | `use_hk` gated on new `AITER_ENABLE_HK_QH32` + native-supported extended for qh32 qseqlen=5-8 |
| `/app/ATOM/atom/config.py:882` | MTP cap lifted 4→8 |

**JIT compile SUCCEEDED in 34.3s** under standalone dummy-tensor test with `AITER_ENABLE_EXPERIMENTAL=1 HOME=/tmp`. Template instantiated cleanly at kNumWarps=2. Buffer managers compiled via `if constexpr(T::kNumWarps > 4)` branches taking the 2-warp path. `module_hk_mla.so` written.

Critical: JIT cache requires writable dir. `/root/.aiter` is read-only in container overlay FS (even for uid=0). Workaround: `HOME=/tmp` env override.

**First boot attempt HUNG**:
- Weights loaded, dynamo compile passed
- Capture phase: ONLY `max_q_len=2` captures at bs=256→1 → canary: MTP silently collapsed to MTP-1
- `max_q_len=4` count = 0 (expected non-zero for MTP-3 main verification)
- Uvicorn up at 8890, `/health` OK
- Log flooded with `[aiter] No available shared memory broadcast block found in 60.0 seconds` (40+ occurrences)
- pgrep: **2 of 4** TP=4 workers alive
- Interpretation: HK qh32 crashed silently on rank 2/3 during MTP-3 drafter capture. Engine silently downgraded to MTP-1. Ranks 0/1 stuck waiting for broadcast acks from dead ranks.

**Container restart required**:
- pkill -9 left 330 zombie python3 processes + 282 GB leaked VRAM per GPU
- ROCm driver did not GC after 90s wait
- `docker restart danish_atom_dsr_beta` cleared zombies, VRAM back to 297 MB idle
- All C1 patches survived restart (verified via grep)

**Control boot** (without `AITER_ENABLE_HK_QH32`) launched at session close to isolate cause. Expected outcome:
- If max_q_len=4 captures appear → baseline MTP-3 works; HK integration is the bug
- If max_q_len=4 still missing → something else broke (env drift)

### Time budget session-8

- 2h HK archaeology (reading 2646 LOC of kernel + buffer managers)
- 1h design spec + memory commits
- 30min draft + deploy h32 kernel files
- 5min JIT compile (SUCCEEDED)
- 15min first boot (HUNG)
- 2min container restart
- 12min control boot (in progress at close)

### New memory files this session

- `project_c1_hipkittens_mla_archaeology.md` — HK code map, blockers, port scope
- `project_c1_port_design.md` — full port design spec + tracking checklist + boot-hang findings

### Resume checklist for session-9

1. Check `/tmp/atom-control.log` for `max_q_len=4` captures (control boot result)
2. If control works: HK is the bug. Debug per-rank stderr, launch with rank-split logging. Hypotheses:
   - (a) Drafter tensor shape mismatch vs HK kernel expectations
   - (b) JIT cache lock contention when all 4 ranks compile `module_hk_mla` simultaneously
   - (c) `work_info_set` metadata from ATOM's `prepare_mtp_decode` incompatible with HK kernel
3. If control also fails: something more fundamental — investigate env drift vs session-7 floor
4. If HK proves unviable after debugging: revert `.pre_c1` backups + submit floor as final committable entry (1/4 gates)

### Rules reminder

Autonomous, CONC=4 only until 4/4, GitHub ONLY on new record, timing not a constraint (Danish auth'd for C1), server-boot pause rule still applies.

---

## 🧪 2026-04-19 late evening → 04-20 overnight — E-08-05 "2/4" was run-to-run variance + C1 HK kernel v1→v2→v3 LDS-layout debug

### E-08-05 2/4 was not submittable — min-of-3 stability failed

| Run | Interact | Gate |
|---|---|---|
| E-08-05 | 165.35 | ✅ |
| E-08-05b (identical config, 7 min later) | 159.87 | ❌ |
| E-08-05c (same) | 150.23 | ❌ |
| **min-of-3** | **150.23** | **❌** |

Run-to-run interactivity spread ~10%. Run 1's 0.2% margin was noise. Cannot hold 165 gate with env-tuning alone. **Structural path required** → C1 HK kernel port to unlock MTP=4+.

### C1 HK qh32 kernel — v1/v2/v3 iteration log (cumulative root-cause analysis)

The h128 HipKittens MLA kernel at `csrc/kernels/mla/hk/mi3xx_v32_fwd_decode_fp8_fp8.cuh` uses `kNumWarps=8, kBlockM=128, kTileM=16`. For DSR1 qh32 we need `kBlockM=32`. At `kNumWarps=8` this would make `kNumTilesM = kBlockM/kTileM = 32/16 = 2` tiles per warp group, but the kernel's buffer managers assume 8 warps write to 8 LDS slots. Two options: (A) reduce `kNumWarps` to 2 and use a virtual-warp loop to simulate 8-warp LDS access, (B) redesign buffer managers natively for 2 warps.

We attempted (A) across 3 iterations before confirming (B) is likely required.

#### v1 — virtual-warp loops at ALL 3 sites (Q, K, V) — GARBAGE OUTPUT

At `kNumWarps=2, kVirtualPerReal=4`, each of 2 real warps iterates 4 times to cover 8 virtual warp positions. Applied to: Q load, K async_load, V load+transpose+store.

Compile required 2 fixes: strip duplicate symbol defs (HkMlaDecodeFwdParams, pack_4f32_to_fp8, max_8, PvGemmEpilogueType come from h128 header already included in hk_decode_fwd.cu), and revert `kOccupancy: 4 → 1` to restore VGPR budget for `pack_4f32_to_fp8<fp8_e4m3>` template substitution at GPR 121.

Boot: server up, `max_q_len=4` captures, HK path active. Single request → garbage text.

**Root cause**: Q load applies virtual-warp at vwarp ∈ {0..7}, but `gl_q<q_t, -1, kNumTilesM=2, kTileM=16, 576>` only has `kNumTilesM=2` slots at h32 (vs 8 at h128). Writes at vwarp ≥ 2 overflow the kNumTilesM dimension.

#### v2 — reverted Q + K virtual-warp loops, kept V — STILL GARBAGE

Fix: Q load + K initial async_load → single call with real warp_idx. V store virtual-warp loop kept.

**Root cause**: inconsistency between K staging and V staging LDS layouts. K fill at real warp_idx populates only 2 LDS slots. V store at virtual warp_idx writes to 8 LDS slots in a different partition. V load reads from K staging — at vwarp positions {2..7} there is no K data → reads uninitialized LDS → garbage.

Key insight: the LDS is a single contiguous region used by both the K→V path. Both producer and consumer must agree on the warp-to-slot mapping. Either both use real warp_idx (2 slots) or both use virtual warp_idx (8 slots). Mixing breaks.

#### v3 (IN FLIGHT) — virtual-warp loop for K and V, single call for Q and output

Fix: K async_load now uses same virtual-warp loop as V store. Both producer and consumer reference identical 8-virtual-warp LDS slot layout. Q load and output write remain single-call (Q buffer has kNumTilesM=2 constraint; output gl_o has no kNumWarps dim, real warp coverage matches output shape).

Caveat flagged in fix-v3: `kSzLdsKv = kNumBytesPerBlock * kNumBlocks` uses `kNumSubBlocks = kNumWarps = 2` at h32. Allocated LDS is sized for 2 warp slots. If v3 writes at 8 virtual positions overflow the 2-slot allocation, all 8-warp-layout writes stomp each other and each other's neighbor regions.

### v4 plan (if v3 still garbage) — override LDS allocation

Force `kSzLdsKv = 2112 * 9` (or equivalent 8-warp formula) in h32 traits, so LDS reserves space for 8 virtual-warp slots. Check against gfx950 160KB LDS budget — need to verify `kBlockM=32` doesn't push us over.

### v5 plan (last resort) — native 2-warp redesign

If even v4 fails (e.g. LDS budget exceeded or non-trivial indexing in store_transposed_v_to_lds assumes 8-warp dense grid), write new buffer manager classes `KvManagerV2_H32` + `VtManagerV2_H32` that natively use 2-warp LDS layout. Per-thread reshape inside one warp pair covers the same (row_blk, col_blk) tile grid as the 8-warp version. Estimated 400-600 LOC, multi-day but structurally correct regardless of LDS sizing.

### Key LDS layout observations from buffer_managers.cuh (archaeology)

In `VtManagerV1::store_transposed_v_to_lds`:
```
row_blk = (warp_idx % 2) * 4 + lane_idx / 16
col_blk = (lane_idx % 16) + warp_idx / 2 * 16
```

At `kNumWarps=8`:
- `warp_idx % 2` ∈ {0, 1} → row_grp ∈ {0, 4}
- `warp_idx / 2` ∈ {0..3} → col_blk offsets {0, 16, 32, 48}
- 2×4 = 8 unique (row_grp, col_blk) tuples, one per warp

Virtual-warp simulation at `kNumWarps=2, kVirtualPerReal=4`:
- Real warp 0 iterations (vwarp ∈ {0,2,4,6}): vwarp%2=0 always → row_grp=0; vwarp/2 ∈ {0,1,2,3} → col_blk {0,16,32,48}. Covers all 4 col_blks at row_grp 0.
- Real warp 1 iterations (vwarp ∈ {1,3,5,7}): vwarp%2=1 always → row_grp=1; vwarp/2 ∈ {0,1,2,3} → col_blk {0,16,32,48}. Covers all 4 col_blks at row_grp 1.
- Total: full 2×4 tile coverage, no overlap (real warp 0 only writes row_grp 0; real warp 1 only writes row_grp 1).

So the virtual-warp loop **logically** covers the correct grid. The failure has to be in LDS sizing or a secondary index that's also a function of warp_idx.

### Memory pointers for session-9 resume

- `project_RESUME_POINT_apr19_c1_kernel.md` — complete state snapshot with v1/v2/v3 chronology
- `project_c1_v1_compiles_wrong_numerics.md` — v1 first-milestone diagnosis
- `project_c1_port_design.md` — original port design + tracking checklist
- `project_forged_plan_apr18_evening.md` — ACTIVE plan pointer
- `.claude/plans/fizzy-toasting-teacup.md` — active plan
- Artifact trees:
  - `/projects/teamA/danish/c1_hk_port/` — working dir, kernel copies by version
  - `/app/aiter-test/csrc/kernels/mla/hk/mi3xx_v32_fwd_decode_h32_fp8_fp8.cuh` — ACTIVE
  - `/c/tmp/c1_hk_port/fix_v2.py` + `fix_v3.py` + `edit_h32_kernel.py` — editor scripts
  - `.pre_c1` backups on: `hk_decode_fwd.cu`, `optCompilerConfig.json`, `mla.py`, `atom/config.py`

---

## 🎯 2026-04-20 STOCK PIVOT + HK KERNEL v5 BREAKTHROUGH

### Strategic shift (Daniel Huang)
- DSR1 + Kimi sampled from [InferenceX](https://inferencex.semianalysis.com/inference)
- Mergability rule: act as AMD engineer, don't overlap with their in-progress work
- Stock model only (no merged DSR1-drafter-FP4)
- DSR1 ALLOWED MTP, Kimi NOT ALLOWED MTP, can't finetune MTP into Kimi

### v5 BREAKTHROUGH (one-line fix)
- KvManagerV2 line 791: `kNumRowsPerSubBlock = kNumRows / T::kNumWarps` evaluates to 16 at h32 (vs 4 at h128)
- **Hardcoded to constant 4** → equivalent at h128, unblocks h32 with v3/v4 virt-warp infrastructure
- Patch `/tmp/fix_v5.py` on server, backup `.pre_v5`
- **PROOF of correctness**: v5+nospec (qseqlen=1) produces FULLY COHERENT R1 reasoning (3 runs all "Okay, the user asked..."). TPOT 7.3 ms.
- **Bug isolated**: qseqlen=4 (MTP-3 verification) path still garbage. Targeted by v6+

### v6 patch (s_barrier between work_idx iterations)
- Hypothesis: at qseqlen=4 multiple work_idx iterations contaminate without inter-iter sync
- Patch: `__builtin_amdgcn_s_waitcnt(0); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0);` at top of work_idx loop
- Kernel 836 lines (+9 from v5)
- TO TEST on stock model

### MERGABILITY WIN: ROCm/aiter Issue #1468
- "Aiter MLA only supports 16 or 128 number of heads" (Nov 2025, exact our config)
- Assigned @ruanjm @zufayu, NO PR, NO progress in 5 months
- Our HK qh32 port directly closes this issue → PR will reference `Closes #1468`
- All patches env-gated/additive (default behavior unchanged)

### Updated memory pointers for resume
- `project_c1_hk_v1_v2_v3_iterations.md` — full v1→v6 trace
- `project_apr20_stock_pivot_v5_v6.md` — NEW (Apr 20 session)
- `MEMORY.md` — index updated

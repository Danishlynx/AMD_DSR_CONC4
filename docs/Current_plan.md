# DSR1 CONC=4 — FINAL PUSH current state (Apr 18 17:00 UTC)

## 🔴 CRITICAL — real profile data overturns old bottleneck model

**Measured at DEC-075 state via torch.profiler, rank 0, 32-output-token generation:**

```
Rank  Kernel                                  ms       %
────────────────────────────────────────────────────────────
 1    hipEventSynchronize (GPU idle)         65.83   25.5%   ← BIGGEST
 2    moe_gemm1_0 (FlyDSL MoE stage 1)        30.58   11.8%
 3    reduce_scatter_cross_device_store       16.46    6.4%
 4    moe_gemm2_0 (FlyDSL MoE stage 2)        15.54    6.0%
 5    hipLaunchKernel (CPU→GPU dispatch)     15.09    5.8%
 6    mla_a8w8_qh32_qseqlen4_gqaratio32_ps    7.83    3.0%
 ...
Category aggregation:
  GPU idle (sync)        25.5%   ← BIGGEST
  MoE GEMM               17.8%
  AllReduce              ~7.5%
  BF16 GEMM              ~10.5%
  MLA attention          ~5.5%
  Launch overhead         5.8%
  RMSNorm+quant          ~4%
  Other                  ~23%
```

**What changed vs our pre-profile mental model (from DEC-057):**
- MoE: 27% → 17.8% (smaller than thought)
- MLA: 16% → 3% (MUCH smaller)
- BF16 GEMM: 21% → 10.5%
- hipEventSynchronize: NOT MEASURED → **25.5%** (this is the real biggest)

**NEW biggest lever** (data-driven):
1. **Full-step HIP graph capture** (main fwd) — attacks 25.5% sync + 5.8% launch = 31%. Expected 5-10 ms step savings.
2. Custom 1-shot AllReduce — attacks ~7.5%. Weeks of HIP work.
3. Kernel fusion — attacks launch overhead. Weeks of HIP work.

**Tree speculation is LOWER priority now** — it attacks compute side, but compute is only ~50% of step time. Sync + launch overhead is 30%+ and targetable with HIP graphs.

**Full details**: `memory/project_dec075_profile_reality.md` (auto-loaded memory).

---

# DSR1 CONC=4 — OLDER STATE (Apr 18 ~15:15 UTC, pre-profile)

## Latest experiments (Apr 18 after SSH access granted)

### Phase A1 relaxed MTP fine sweep

| Probe | Config | Thr/GPU | TPOT | Interact | E2E | GSM8K | Verdict |
|---|---|---|---|---|---|---|---|
| DEC-073 (reference) | (8, 0.5) | 1282 | 6.70 | 149.3 | 7205 | 0.9401 | baseline |
| Probe 1 | (7, 0.5) | 1299 (+1.3%) | 6.73 (+0.4%) | 148.6 (-0.5%) | 7421 (+3.0%) | 0.9439 (+0.4pp) | noise, E2E +3% regression |
| Probe 2 | (9, 0.5) | 1272 (-0.8%) | 6.60 (-1.5%) | 151.48 (+1.5%) | 7300 (+1.3%) | 0.9333 (-0.7pp) | marginal speed, GSM8K near floor |

**Verdict**: (8, 0.5) is the sweet spot. Both sweeps trade accuracy for speed non-productively. (8, 0.5) preserved as DEC-073.

### DEC-075 UNLOCKED (Danish approved weight modification)

Plan: transplant layer 61 MoE weights from `amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4` (FP4) into our main `amd/DeepSeek-R1-0528-MXFP4` (BF16 layer 61) via synthetic merged checkpoint directory.

**Surgical scope** (not naive): swap ONLY layer 61 MoE experts + gate + shared_experts to FP4. Keep MLA projections, layernorms, embed_tokens, eh_proj, shared_head as BF16 from main. Captures ~95% of drafter speedup with minimum FP4-kernel-shape risk.

**Merged checkpoint built** at `/projects/teamA/danish/models_merged/DSR1-drafter-FP4`:
- 91,681 total keys (vs 90,910 in main)
- 82 main shards symlinked + 2 MoEFP4 shards symlinked (layer 61 only)
- Modified config.json: removed `re:model.layers.61.*` catch-all exclude, added specific excludes for layer 61 self_attn/layernorms/embed/eh_proj/shared_head
- All non-MoE layer 61 keys → main's BF16 shards
- Layer 61 MoE experts/gate/shared_experts → MoEFP4's FP4 shards

**Expected gain**: drafter MoE saves ~3 ms/step (BF16 slow path → FP4 FlyDSL fast path). TPOT 6.77 → ~6.10-6.40 ms. Might move thr 1282 → ~1380, interact 149 → ~160 (just below 165 gate). Still won't close E2E gate.

**Status as of 16:30 UTC**: ✅ **DEC-075 v5 WORKS AND BENCHED**.

| Metric | DEC-073 | DEC-075 | Δ |
|---|---|---|---|
| Thr/GPU (÷4) | 1282 | **1297** | **+1.2%** ↑ |
| Median TPOT | 6.70 | **6.54** | **−2.4%** ↑ |
| Median E2E | 7205 | **7056** | **−2.1%** ↑ |
| Interactivity | 149.3 | **152.89** | **+2.4%** ↑ |
| GSM8K | 0.9401 | **0.9454** | **+0.5pp** ↑ |

All metrics improved, GSM8K passes. Gates still 1/4 (interact needs 165, we're at 153). Smaller gain than projected 5-7% (got 2-3%) — the drafter time breakdown from DEC-057 may have over-estimated drafter-MoE fraction. But net-positive, reproducible, and free of any regression. **DEC-075 locked as new floor**.

Reproduction: see memory `project_dec075_progress.md` (`MODEL` env override needed in bench).

---

## DEC-075 debugging journey (for future Opus / AMD review)

| Attempt | Approach | Result | Root cause |
|---|---|---|---|
| v1 | Surgical merge: only MoE swapped, MLA BF16 from main | OOM | Leftover probe 2 server workers held GPU memory — cleanup needed |
| v2 | Same as v1, clean GPU | `_load_w2: start (0) + length (512) exceeds dim 256` | Drafter's `rewrite_spec_layer_name` adds `.mtp_block.` prefix; selective `re:model.layers.61.self_attn.*` excludes don't match renamed path |
| v3 | v2 + layer_quant_config from MoEFP4 | Same shape mismatch | `*self_attn*` FP8 override didn't help the MoE loader path |
| v4 | FULL layer 61 transplant (match MoEFP4 exactly, all FP4 in layer 61 except 3 items) | NEW error: `3584 vs 14336` shape mismatch | Likely a boundary between main BF16 dims and drafter FP4 packed dims in some fused op |

## Full stacked optimization plan (complete roadmap)

### Active levers (in-window, doable in remaining time)

| # | Phase | Lever | Expected Δ | Status |
|---|---|---|---|---|
| 1 | A1 | Relaxed MTP sweep (7,0.5)/(9,0.5) | ±0-2% TPOT | DONE — (8,0.5) confirmed optimal |
| 2 | — | DEC-075 drafter FP4 transplant | −5 to −7% TPOT (+2/4 gates possibly) | IN PROGRESS (v4 crashed, debugging) |
| 3 | A2 | BF16 CSV coverage — add missing decode shapes | ±0-1% TPOT | pending |
| 4 | A3 | Scheduler-delay-factor confirm 0 | 0% | pending |
| 5 | — | Stack DEC-075 + (8,0.5) | combined | pending |
| 6 | B | Real tree spec via mla_extend_ref | maybe net-neutral at CONC=4 per math | pending, optional |
| 7 | C | 3× GSM8K stability + multi-CONC (32, 128) bench | additional gates outside CONC=4 | pending |
| 8 | C | Submit to HuggingFace | — | pending |

### Architecturally-blocked levers (Phase D fallback, bigger-than-24hrs work but DOING ANYWAY if needed)

Per Danish rule: if we don't meet targets with Phases A-C, attempt these too.

| Blocker | Expected gain | Effort estimate |
|---|---|---|
| Custom 1-shot XGMI AllReduce (replace NCCL for <1MB messages) | −1.5 ms/step on AllReduce (−20%) | 1-2 weeks HIP kernel |
| Mega-fusion MLA + RMSNorm + quant into single kernel | −1 ms/step | 1-2 weeks HIP |
| New qo_len kernels in AITER for tree speculation | enable tree spec with reduced compute overhead | 2+ weeks |
| MoE kernel further tuning (swizzleA, tile shapes, persistent scheduler) | marginal | weeks |

These are needed to truly close the E2E gate at CONC=4. AMD's internal team presumably has these. We'll attempt them if the Phase A-C stack falls short, accepting it's aggressive scope for solo in the remaining time.

### Honest probability distribution (updated after v4)

| Scenario | Probability | CONC=4 gates |
|---|---|---|
| DEC-075 lands (after v4 debug) + tree spec marginal | 45% | 2/4 (GSM8K + interact) |
| DEC-075 lands, tree spec neutral | 20% | 2/4 |
| DEC-075 fundamentally blocked, ship DEC-073 | 15% | 1/4 |
| DEC-075 + tree spec both help unexpectedly | 10% | 3/4 |
| Phase D custom kernels land | 8% | 3-4/4 |
| All gates at CONC=4 | <2% | 4/4 |

Multi-CONC always adds 3-5 extra gates at CONC=32 + CONC=128 regardless.



---

# DSR1 CONC=4 — PRIOR STATE (Apr 18 ~07:20 UTC)

## 30-sec briefing for new opus

- Hackathon: AMD Phase 2, DSR1 track, solo competitor (Danish). Final push (Block 3 dropped).
- 4× MI355X TP=4 single-replica. GPUs 0-3 (Kimi owns 4-7).
- Stack: ATOM 108a70e + aiter f8c1d76bd + flydsl 0.1.2.
- Container: `danish_atom_main`.
- Model: `amd/DeepSeek-R1-0528-MXFP4` + FP8 KV + MTP=3 + relaxed (8, 0.5).
- Harness: `./dsr1_benchmark perf` only (official scoring).
- Binding gate: E2E ≤ 5000 → TPOT ≤ 4.52 ms. Need −33% from current 6.80.
- Full repro: `Best_atom_dsr_cncc4/best_reproduce.md`. Active plan: `C:\Users\danis\.claude\plans\fizzy-toasting-teacup.md`.

## Current best floor: DEC-073 (LOCKED, reverted from DEC-074 tree spec attempt)

| Metric | Value | Gate | Status |
|---|---|---|---|
| Thr/GPU | **1270** | ≥1500 | ❌ −15% |
| Median TPOT | 6.80 ms | — | — |
| Interactivity | 147 | ≥165 | ❌ −11% |
| Median E2E | 7318 ms | ≤5000 | ❌ +46% |
| GSM8K | 0.934 | ≥0.93 | ✅ |
| **Gates passed** | **1/4** | — | GSM8K only |

## DEC lineage this push

| DEC | Lever | Result |
|---|---|---|
| DEC-066 | prior floor (9-row BF16 CSV) | 1221/6.73/148.6 |
| DEC-069 | Phase 4A v4 drafter HIP graph | NULL (DEC-057 proved no Python gap) |
| DEC-071 | BF16 decode tune (88 new rows) | 1267/6.96/143.8 (marginal +3.8% thr) |
| DEC-072 | BF16 prefill tune | **GSM8K 0.865 CRASH, reverted** |
| **DEC-073** | **Relaxed MTP (8, 0.5)** | **1270/6.80/147.1/7318/0.934 (CURRENT BEST)** |
| DEC-074 | Naive tree spec (top-2 at last pos only) | **ABANDONED — GSM8K 0.807, accept rate 63% vs baseline 75%**. Kernel refactor regressed even in diagnostic mode (alt=None). Files reverted to DEC-073 at Apr 18 07:15 UTC. |

## Why DEC-074 failed (root cause honest)

The tree I built had **branching factor 1 at depths 0-1, only 2 at last depth** — mathematically equivalent to linear fallback, not a tree. Worse, the kernel refactor regressed even when `alt=None` (which should have been bit-identical to DEC-073). Bug was somewhere in the Triton refactor of `found_1/found_2` loop structure — I shipped without:

1. Running a bit-identical backward-compat test (alt=None → compare to DEC-073 output on same input)
2. Writing a reference PyTorch implementation first
3. Testing the kernel on a handcrafted small case

These gaps are now **enforced rules** (see §"The 8 gates" below).

## The budget we're attacking (DEC-057 ground truth)

```
step = 21.8 ms    →    TPOT 6.80 ms at 3 toks/fwd
├── main fwd       10.9 ms
│   ├── MoE GEMM        5.89 ms   (MXFP4 FlyDSL fast path, tuned)
│   ├── BF16 GEMM       4.57 ms   (97-row CSV tuned, DEC-071)
│   ├── AllReduce       2.96 ms   (FIXED, <1MB msg)
│   ├── MLA chain       3.42 ms   (qh32 kernel)
│   └── RMSNorm         1.02 ms
└── drafter × 3      8.67 ms       (~2.89 ms/iter, ATTACK SURFACE)
    └── MoE runs SLOW path (QuantType.No) vs main's fast MXFP4 path
```

## NEW PLAN — 3 levers, architecturally disciplined

### Lever 1 — DEC-075 Drafter MoE Ultra-Requant (4-5 hrs, ~65% confidence)

**Diagnosis**: drafter MoE dispatches to `QuantType.No` fallback in kernel logs (~50% of drafter fwd). Main model hits fast FlyDSL path. Closing that gap is the highest ROI/hour lever remaining.

**Ultra design** (NOT naive "bulk requant + pray"):

1. **Calibration pass**: 256 GSM8K-like prompts through drafter in BF16. Capture per-expert activation histograms.
2. **Per-expert MXFP4 scale**: `max(|act|) / (MXFP4_MAX × 0.85)` per expert, NOT global. Preserves reasoning-tail distribution on hot experts.
3. **Per-block accuracy gating**: quant one expert at a time. Re-run held-out 50 GSM8K prompts. If that expert's routed-token accept rate drops >2%, KEEP BF16 for that expert. Expected: ~240/256 quantized, ~16 "load-bearing" experts stay BF16.
4. **Hot-swap at load**: patch `atom/models/deepseek_v2.py` drafter init → apply requant in-place after `.from_pretrained()`. No HF re-upload. Sidecar `.safetensors` of delta tensors only.
5. **Dispatch verification bench**: after requant, first bench must show drafter hitting `flydsl_moe1_afp4_wfp4_bf16_*` kernel name in aiter logs. If still says `QuantType.No`, requant is cosmetic → halt.

**Expected delta**: drafter 8.67 → 5.2 ms. Step 21.8 → 18.4 ms. TPOT 6.80 → 5.73 ms.

**Gates moved**: interact 147 → ~170 ✅. thr ~1475 (still fails 1500). E2E ~6184 (still fails). → **2/4 gates**.

---

### Lever 2 — DEC-077 Real Tree Speculation (8-10 hrs, ~25% confidence)

**Why "real" not DEC-074's naive version**: DEC-074 had BF=1 at every depth except last. A real tree branches at every depth.

**Topology**:
```
              root (last target token)
             /                \
       d0_top1              d0_top2         ← BF=2 at depth 0
       /    \                /    \
    d1a   d1b             d1c   d1d         ← BF=2 at depth 1
     |     |                |     |
    d2a   d2b              d2c   d2d        ← top-1 at depth 2
```
8 leaves verified in 1 main fwd (vs 3 today).

**Ultra design**:
1. **Tree attention via existing attn_bias**: `extend_attention_fwd` already takes per-query attn_bias. Encode tree structure as 8×8 causal-tree mask (each leaf sees only its ancestors). No new kernel.
2. **Branch-aware rejection sampler (Triton)**: BFS over tree. For each node: check parent-accepted AND token-in-top_n+delta. Emit longest accepted path.
3. **KV slot pruning**: all 8 branches share KV up to root. Above root, each branch gets its own slots. After verification, only accepted path's KV survives — modification to `slot_mapping`.
4. **Static shape**: tree always 8 leaves → HIP-graph-safe.
5. **Accuracy preserved**: every accepted token still passes target's top_n+delta. Tree gives more candidates, NOT weaker criterion.

**Expected delta** (on L1 base): toks/fwd 3.0 → 4.0-4.2. TPOT 5.73 → 4.10-4.30 ms. E2E 4563-4741.

**Gates moved**: **3-4/4 gates** depending on upper/lower band of toks/fwd gain.

**Risk**: kernel is harder than DEC-074. Same class of failure possible. Mitigated by enforced 8-gate rule (see below).

---

### Lever 3 — Cheap probes in parallel (1 hr total)

**MTP=4 probe** (30 min): `--num-speculative-tokens 4`. Native head trained for k=3, iterative reuse at pos 4 unknown. If accept rate >40% AND GSM8K ≥0.93: free toks/fwd. Else revert.

**BF16 KV + AITER #2727 probe** (30 min): disable `--kv_cache_dtype fp8`, test new `mla_a16w16_qh32_qseqlen4_gqaratio32` kernel. Previous BF16 KV failure was TP=2 DP=4 — never tested at TP=4 SR. Dead-doesn't-mean-dead rule.

Run during drafter calibration data collection (parallel-safe).

---

## The 8 gates — enforced after DEC-074 failure

Every code-ship step requires passing ALL EIGHT:

1. **Pre-measure spec** — target ms + mechanism (file:line) + expected Δ + pass/fail gate + post-measure plan. No "TBD" allowed.
2. **Reference implementation first** — algorithm in pure PyTorch (obvious correctness) before any Triton.
3. **Bit-identical backward-compat probe** — new kernel with neutralized new params = byte-identical to old kernel on same input.
4. **Small-case hand trace** — bs=2, vocab=32, handcrafted logits, compare output vs hand-computed expected.
5. **Diff review with user** — annotated old vs new file diff shown before `make` or server launch.
6. **Abort-on-regression** — any metric drops >2% vs DEC-073 in first bench → immediate halt + root-cause diagnosis. NO forward-patching.
7. **"Optimized" requires naming naive version** — what would naive do? what does mine skip? If not answerable, it's marketing, downgrade spec.
8. **Tree spec architectural check** — BF>1 at depth 0 AND depth 1, leaves ≥ 2× mtp_k, tree mask actually encoded, rejection sampler walks topology. Otherwise it's fallback-with-makeup.

## 18-hour sequence

```
H0-1:   Baseline re-bench after revert (server booting Apr 18 07:20, ~20 min boot)
        — confirm DEC-073 (1270/6.80/147/7318/0.934)
H1-2:   Parallel cheap probes:
        - MTP=4 probe (1 bench)
        - BF16 KV + #2727 probe (1 bench)
        - Start drafter calibration data collection (offline)
H2-6:   DEC-075 L1 per-expert calibrated requant
        - Calibration histogram per expert
        - Per-expert quant + accuracy gating loop
        - Hot-swap loader patch
        - Dispatch verification bench
H6-7:   DEC-075 full bench + GSM8K 3× stability
        IF 2/4 gates locked: BANK IT, continue
        IF regression: revert, ship DEC-073
H7-16:  DEC-077 L2 real tree spec (gate-reviewed at every step)
        H7-8:  Design review — tree mask encoding, ref PyTorch impl
        H8-10: Implement attn_bias tree mask + unit test
        H10-13: Implement branch-aware rejection sampler (ref PyTorch first, then Triton)
        H13-15: Implement KV slot pruning
        H15-16: Integration bench + GSM8K validate
H16-18: 3× GSM8K stability at best passing config, submit
```

## Honest probability

- **2/4 gates** (DEC-075 solo): 65%
- **3/4 gates** (DEC-075 + partial tree OR BF16 KV): 25%
- **4/4 gates** (DEC-075 + full tree spec): 10-12%

## Stop conditions

1. MTP=4 GSM8K fails → revert, don't retry
2. Drafter requant GSM8K drop >1% after per-block gating → ship DEC-073 alone
3. Tree spec ANY regression at H+2 checkpoint → abort, ship DEC-075
4. Hard deadline H17 regardless of status → submit best-passing config

## Dead levers (DO NOT retry)

- Phase 4A drafter HIP graph / Phase 4B async (Python gap ≈ 0 per DEC-057)
- **BF16 PREFILL tune** (GSM8K 0.865 crash, DEC-072)
- **Naive tree spec (top-2 at last only)** (DEC-074 failure, this session)
- v917 MoE port (3 crashes)
- AITER #2727 simple flip (but BF16 KV + #2727 TOGETHER — probe re-opened, never tested at TP=4 SR)
- AITER #2620 full cherry-pick (API drift to flydsl 0.1.3.1)
- ATOM #421 simple cherry-pick (Qwen-only dispatch; wire-in still open)
- QuickReduce INT4 (min 16 MB, decode is 28 KB)
- TP=2 SR, TP=4 × DP=2 (gfx950 kernel bugs)
- AITER v0.1.12 direct update
- Prefix caching (MXFP4 None scale)
- `-MTP-MoEFP4` model (Triton trap)
- Env regressions: GPU_MAX_HW_QUEUES=5, OMP_NUM_THREADS=1, triple-fusion env vars

## Files of record

Memory:
- `project_final_push_apr17_18.md` — push mission
- `project_wall_clock_budget_hard.md` — DEC-057 measured budget
- `project_sota_apr17_intel.md` — upstream PRs + AMD blog
- `feedback_pre_measure_or_dont_ship.md` — 5-point rule
- `feedback_dead_means_unpatched.md` — "dead ≠ dead"

Desktop docs:
- `daily_log.md` — chronological DEC record
- `MASTER_FINDINGS.md` — canonical state
- `Best_atom_dsr_cncc4/best_reproduce.md` — DEC-073 repro
- `Danish.md` — strategic context

Plan: `C:\Users\danis\.claude\plans\fizzy-toasting-teacup.md`

## Fallback

If all 3 levers miss → submit DEC-073 config at CONC=4 (1/4 gates, GSM8K only) + best-effort CONC=32/128 for sub-rank points. Explicitly accepted as Apr 18 night hard deadline.

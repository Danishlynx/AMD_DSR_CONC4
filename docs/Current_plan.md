# DSR1 CONC=4 — FINAL PUSH current state (Apr 18 ~07:20 UTC)

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

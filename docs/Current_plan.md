# DSR1 CONC=4 — CURRENT PLAN (Session-10 Apr 20: P0-P8 kernel engineering campaign)

**Last updated**: 2026-04-20 session-10 — **P0 ✅ DONE → 3/4 GATES**
**Status**: Advancing to P1 (TRITON_GEMM fusion enablement)

## 🎯 P0 BREAKTHROUGH (Apr 20 16:50 UTC)

Added `--cudagraph-capture-sizes "[1,2,4,8,16,32]"` to canonical launch. Nothing else changed.

Min-of-3 result: **Thr/GPU 1500 ✅ | Interact 185 ✅ | E2E 5763 ❌ | GSM8K 0.9318 ✅ = 3/4 GATES**

This is +2 gates vs the prior 1/4 floor. TPOT −19% (6.66→5.40 ms). E2E still 763 ms over gate — will close via P1+P2+P7.

**New baseline for subsequent phases**: 1500/5.40/185/5763/0.9318. All phase targets now measured from this.

**Authority**: Danish — "you wont stop until all 4/4 gates are accomplished" + "kernel level engineering like AMD engineer with 15 years experience"
**Master plan file**: `C:\Users\danis\.claude\plans\fizzy-toasting-teacup.md` (full details, 200+ lines)
**Campaign memory**: `memory/project_dsr1_kernel_campaign_apr20.md` (auto-compact survival)
**Archive**: prior session-8 `Current_plan.md` → [archive/Current_plan_session8_HK_qh32.md](archive/Current_plan_session8_HK_qh32.md)

---

## Context

Profiling on Apr 20 (M1 torch.profiler + V1/V4/V5 overlap validation) identified the bottleneck definitively:

- **`hipGraphLaunch` = 77.7% of wall** at CONC=4 (915 calls × 63 µs, measured; V5 confirmed NOT overlapped with GPU work — DSR1 is host-starved, opposite of Kimi)
- **Root cause**: ~1525 nodes per graph (61 layers × ~25 kernels) × HIP runtime ~40 ns/node submission = 61 µs (matches measured 63 µs)
- **Per-node cost already near-optimal** — problem is NODE COUNT
- Profiler skew 3.4× → native native contribution ~30-50% wall (still dominant)

See: [Bottleneck.md](Bottleneck.md) for full profile data.

Prior plans (VSKIP=0, HK qh32 v7/v8, compact-experts, F0a hipEventSync) were either non-applicable on DSR1 or addressed kernels too small to matter (MLA 4.3% of GPU time = 0.1% wall). Campaign now targets the REAL bottleneck: node-count reduction + tokens-per-step boost.

---

## Phase table (current state)

| Phase | Description | Status | Expected TPOT | Expected Gates |
|---|---|---|---:|---:|
| P0 | Environmental hygiene + clean floor + explicit `hipGraphUpload` | **🔨 IN PROGRESS** | 6.60 ms | 1/4 |
| P1 | `ATOM_USE_TRITON_GEMM=1` → unlocks 2 fusions (-122 nodes) | pending | 6.35 ms | 1/4 |
| P2 | **REVISED**: activate ATOM shared-expert fusion scaffold (commented out at `moe.py:435`) + port vLLM #24097 mechanism — 1-8.5% QPS gain confirmed | pending | 6.00 ms | 1/4 → 2/4 threshold |
| P3 | Backport vLLM #27224 (host overhead between decode steps) | pending | 5.85 ms | **2/4** ← first crossing |
| P4 | Drafter graph isolation experiment (time-boxed 2d) | pending | — | — |
| P5 | HK MLA v2 qh32 via metadata-builder template fix (⭐ structural) | pending | 5.65 ms | 2/4 |
| P6 | Full megakernel (stretch, deferred) | deferred | — | — |
| P7 | MTP=4 with HK qseqlen=5 | pending | 4.60 ms | **3/4** |
| P8 | MTP=5 with HK qseqlen=6 | pending | 3.95 ms | **4/4** ✓ |

Wall-clock estimate: 3-4 weeks sustained work; +2-4 if P6 invoked.

---

## Engineering rules (session-10)

1. **No naive paths.** Every patch must reference a kernel trace, GitHub PR, or AMD doc.
2. **Measure before/after.** Each phase writes `dsr_beta/bench_results/p<N>_<short>.json`.
3. **Revert if regresses >2%.** Every patch pre-backs as `.preP<N>`.
4. **Record every experiment** to [EXPERIMENTS.md](EXPERIMENTS.md).
5. **Never declare a lever dead without profile-confirmed reason.**
6. **GitHub push only on new record.**
7. **Boot-command pause**: announce each cold boot before triggering.
8. **Update docs + memory at end of EVERY phase** so auto-compact cannot lose progress.

---

## P0 execution checklist (in flight)

- [ ] P0.0 — Create/refresh doc infrastructure + campaign memory (this commit)
- [ ] P0.1 — Container state check, env audit, add `--cuda-graph-sizes 32`
- [ ] P0.2 — Read `model_runner.py:2013-2018`, apply explicit `hipGraphUpload(graph, stream)` after capture
- [ ] P0.3 — 3× min-of-3 `./dsr1_benchmark perf` + GSM8K, write `P0_clean_floor.json`
- [ ] P0 checkpoint commit: `dsr_beta_final` branch, update this file + EXPERIMENTS.md + memory

---

## Phase-crossing gates (decision points)

- **End of P3**: if 2/4 gates + ≥10% TPOT improvement, GitHub push + leaderboard interim submit. Continue to P5.
- **End of P5**: if HK qh32 qseqlen=5 doesn't land in 7 days, re-scope (possibly invoke P6 megakernel earlier OR research upstream AITER nhead=32 PR status).
- **End of P7**: if 3/4 + E2E gap ≤ 200 ms, retry P3/P1 stacked patches for the last 200 ms.
- **End of P8**: 4/4 achieved → full submit. Otherwise pivot to P6 megakernel (2-4 week stretch).

---

## Math — why 4/4 is reachable

Compounded native gains (3.4× deflated from profiler view):

| Phase | ΔTPOT | TPOT | Thr/GPU | Interact | E2E | Gates |
|---|---:|---:|---:|---:|---:|---:|
| Floor | — | 6.66 | 1351 | 150 | 7221 | 1/4 |
| P1 | −0.31 | 6.35 | 1420 | 157 | 6880 | 1/4 |
| P2 | −0.20 | 6.15 | 1467 | 162 | 6660 | 1/4 |
| P3 | −0.30 | 5.85 | 1541 | 171 | 6340 | **2/4** |
| P5 | −0.20 | 5.65 | 1595 | 177 | 6120 | 2/4 |
| P7 | −1.05 | 4.60 | 1957 | 217 | 5020 | **3/4** |
| P8 | −0.65 | 3.95 | 2280 | 253 | 4350 | **4/4 ✓** |

---

## Pointers

- Full plan with patch-site file:line references: [`../../.claude/plans/fizzy-toasting-teacup.md`](file:///C:/Users/danis/.claude/plans/fizzy-toasting-teacup.md)
- Bottleneck profile data: [Bottleneck.md](Bottleneck.md)
- Floor reproduction: [best_reproduce.md](best_reproduce.md)
- Experiment log: [EXPERIMENTS.md](EXPERIMENTS.md) (append-only)
- Daily log: [HISTORY.md](HISTORY.md)
- Status dashboard: [STATUS.md](STATUS.md)
- Master findings: [FINDINGS.md](FINDINGS.md)

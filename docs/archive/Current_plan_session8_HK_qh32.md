# [ARCHIVED] DSR1 Current_plan.md as of session-8 close (Apr 19 evening)

**Archived**: 2026-04-20 session-10 (P0-P8 campaign lock-in)
**Reason**: Superseded by `Current_plan.md` v2 which tracks the P0-P8 kernel-engineering campaign based on Apr 20 bottleneck profiling findings. The C1 HK qh32 port described below continues as Phase P5 with a different root-cause approach (metadata-builder template fix instead of Python preprocessor).

---

# DSR1 CONC=4 — Current state (Apr 19 session-8 late evening, C1 HK port in flight)

## Floor locked (unchanged)

**TP=4 SR baseline**: `1361 / 6.35 / 157.55 / 6842 / 0.934` → **1/4 gates** (GSM8K only).
Last session-7 pure-floor re-bench: `1341 / 6.47 / 154.63 / 7009 / 0.9356` — row-identical within noise.
Reproduction: [best_reproduce.md](best_reproduce.md). Bench result JSON: [dsr_beta/bench_results/CURRENT_BEST_1361_6p35.json](../dsr_beta/bench_results/CURRENT_BEST_1361_6p35.json).

## Session-8 state: C1 HipKittens qh32 port in flight — ZERO benchmarks produced

Danish authorized 2026-04-19: "timing is not the constraint, build it, I want AMD optimized kernels".

### What's done
1. **HK archaeology** — Discovered HipKittens MLA already integrated at `/app/aiter-test/csrc/kernels/mla/hk/` (2646 LOC). FP8 + DeepSeek MLA shape + runtime `max_seqlen_q` all baked in. Blocker: `static_assert(kBlockM==kQoNumHead, "nhead=128 only")` at `mi3xx_v32_fwd_decode_h128_fp8_fp8.cuh:36`.
2. **Port design** — NEW isolated files strategy (keep proven h128 UNTOUCHED). kNumWarps=2 for h32 (kBlockM=32/kTileM=16=2 warps). All VGPR constants unchanged since kTileM=16 stays.
3. **Patches deployed** (backups `.pre_c1` for each):
   - NEW `/app/aiter-test/csrc/kernels/mla/hk/mi3xx_v32_fwd_decode_h32_fp8_fp8.cuh` — h32 traits + wrapper reusing h128 kernel body via template parameter
   - NEW num_head==32 dispatch branch in `/app/aiter-test/csrc/kernels/mla/hk_decode_fwd.cu`
   - `/app/aiter-test/aiter/jit/optCompilerConfig.json` — h32 header added to `module_hk_mla` srcs
   - `/app/aiter-test/aiter/mla.py:330-437` — use_hk gated on new `AITER_ENABLE_HK_QH32` env + native-supported extended for qh32 qseqlen=5-8
   - `/app/ATOM/atom/config.py:882` — MTP cap lifted 4→8
4. **JIT compile** ✅ **SUCCEEDED** in 34.3s under standalone dummy-tensor test. Template instantiates cleanly at kNumWarps=2. `module_hk_mla.so` built.

### What broke
**First boot attempt** with `AITER_ENABLE_HK_QH32=1 AITER_ENABLE_EXPERIMENTAL=1 --num-speculative-tokens 3` **HUNG**:
- Weights loaded, dynamo compile passed
- Capture phase: ONLY `max_q_len=2` (bs=256→1) — canary warning: MTP silently collapsed to MTP-1
- `max_q_len=4` count = **0** (expected for MTP-3 main verification)
- Uvicorn up at 8890, `/health` OK, but log flooded with `[aiter] No available shared memory broadcast block found in 60.0 seconds` (40+ times)
- pgrep: **2 of 4** workers alive
- Interpretation: HK qh32 crashed silently on rank 2/3 during MTP-3 drafter capture at qseqlen=4. Ranks 0/1 stuck on broadcast acks.
- Killed + `docker restart danish_atom_dsr_beta` cleared 330 zombie pythons + 282GB VRAM leak. Patches intact.

### Session-9 continuation (Apr 19-20)
- v7 kernel qpos loop attempted — HSA fault
- v8/v8b/v8c Python preprocessor attempts — all crashed despite 7 local edge case tests passing
- Conclusion at session-9 close: preprocessor approach is fragile. Root cause = `kPackedQoLenPerWg=128` hardcoded in AITER metadata builder at `v1_1_device.cuh:662` for h128 assumption. See session-10 Current_plan.md P5 for new approach.

## Honest lever status (Apr 19 session-8 close)

| Lever | Status | Notes |
|---|---|---|
| B drafter HIP graph (v1→v6) | ❌ ALL CRASHED session-7 | Fundamentally incompatible with MoE+NCCL on gfx950 |
| C prefix cache (v1→v4) | ❌ ALL CRASHED session-7 | Kernel has byte-level FP8 arithmetic + layout + stride baked |
| A1 hipBLASLt retune | ❌ BLOCKED session-7 | solidx non-round-trip + aiter JIT merge destroyed pristine CSV |
| B2 P-EAGLE position-only gamble | ❌ −31% thr session-7 | Training-free init = near-zero accept at positions t+2/t+3. Reverted |
| B1 drafter FP4 transplant | ✅ ALREADY DEPLOYED DEC-075 | Already baked into floor |
| **C1 HipKittens qh32 port** | 🚧 session-8 JIT ✅, first boot hung | Continues as session-10 Phase P5 |
| C2 tree spec | ⚠️ PROVED DEAD in our stack | All variants no-op or net-neg (needs C1's kernel mask) |
| C3 MTP=4+ | ⏳ BLOCKED on C1 | No qseqlen>4 kernel exists yet |

## Rules in force

1. Autonomous mode, no permission asking
2. CONC=4 only until 4/4 gates
3. GitHub push ONLY on new record
4. "Infeasible never terminal" — 3 ranked paths with file:line blockers, start cheapest
5. Pause before server boot (12-15 min cold boot)
6. Always optimized never naive

## Superseded

- Earlier "Lever A MLA flatten port" plan — ALREADY MERGED Oct-Dec 2025
- Earlier 4-lever patch list — all dead, see MASTER_FINDINGS session-7 + session-8
- TP=8 decision — parked for CONC=32/128 tracks
- C2 tree spec — proved dead in our stack (no per-query attention mask support in AITER MLA)

---

**CONTINUES AT**: `../Current_plan.md` (v2 Apr 20 session-10 — P0-P8 kernel campaign). The HK qh32 work from sessions 8/9 is now Phase P5 with root-cause fix at AITER metadata builder.

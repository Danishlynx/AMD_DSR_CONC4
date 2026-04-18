# 🎯 DSR1 Master Execution Plan — Phase-C Sprint

**Started**: 2026-04-18 post-Phase-A
**Baseline**: DSR_beta + TBO prefill + DUAL_STREAM=1024 = **1344/6.36/157/6896/0.9386** (1/4 gates)
**Target**: E2E ≤ 5000 ms → TPOT ≤ 4.52 ms (need −29% from 6.36)

## Governing rules (Danish mandate)

1. **Nothing left unattempted.** No lever skipped.
2. **Fully optimized versions only.** No naive implementations.
3. **Deep research before each phase.** Understand the mechanism fully.
4. **Patch the system if blocked.** If a constant/flag/guard stops a lever, modify it.
5. **Losing is not an option.**

## Phase 0 — QKN fusion probe (15 min, RUNNING NOW)

**0.1** `ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1` — env var default OFF but wiring exists in `deepseek_v2.py:1487-1583`.
- Server booting with this env set on top of DSR_beta best stack
- Decision: keep if TPOT ≤ 6.36 AND GSM8K ≥ 0.93

## Phase 1 — 🥇 fused_kv_bmm uncomment (3-4 hrs, HIGHEST EV)

**Target**: `atom/model_ops/attention_mla.py:741`
- A single `#` disables a complete 58-line fused Triton kernel wrapper
- Fuses Q-proj + K-up-proj + RoPE + concat + KV-cache-write = 1 kernel launch instead of 2
- 64 MLA calls/step × 1 launch saved × 8.8 μs = **0.3-0.7 ms TPOT savings**

**Steps**:
1.1 Backup `attention_mla.py.pre_kvbmm`
1.2 Uncomment line 741 `q_out = self.fused_kv_bmm(...)`
1.3 Remove replaced separate calls (line 711 `_q_proj_and_k_up_proj`, line 723 `fused_qk_rope_concat_and_cache_mla`) in decode branch only
1.4 Verify guard logic `is_rocm_aiter_fp4bmm_enabled()` — may need env `ATOM_USE_TRITON_MXFP4_BMM=1` set
1.5 Clean restart, verify kernel `fused_fp4_bmm_rope_cat_and_cache_mla` dispatches
1.6 **Bit-identical probe**: same prompt temp=0 greedy, compare tokens to baseline
1.7 If bit-identical: full bench + GSM8K
1.8 If NOT: debug guard logic. If can't fix in 1 hr, revert + move on.

## Phase 2 — TP=8 DSR_beta retest (30 min, orthogonal)

Old Phase 1 TP=8 gave 960 thr/GPU on stale stack. DSR_beta (ROCm 7.2.2 + TBO prefill) may improve.

2.1 Kill TP=4 server
2.2 Relaunch: `-tp 8` + same env (TBO prefill, DUAL_STREAM=1024, patches)
2.3 Bench
2.4 Per-GPU calc: total_thr / 8
- If per-GPU ≥ 1500: **PASSES thr gate**, switch as new baseline
- If 1344 ≤ per-GPU < 1500: reasonable alternative
- If per-GPU < 1344: stay TP=4 SR

## Phase 3 — Sync-reduction patch (2 hrs)

Attacks 25.5% hipEventSynchronize directly.

3.1 Patch `atom/model_engine/model_runner.py:137-150` `send_mtp_status_to_cpu_async`:
- Current: 2 `send_to_cpu_async` (rejected + bonus) → 2 copy_done events → 2 syncs
- New: `torch.stack([num_rejected, num_bonus])` → 1 copy → 1 event → 1 sync
3.2 Patch `recv_mtp_status_async:152-160` symmetrically: 1 recv, split result
3.3 Boot, bit-identical probe, bench, GSM8K

**Expected**: 1 sync saved per MTP step × 8 steps × 1.6 ms = 13 ms / 32 tokens = **0.4 ms TPOT**

## Phase 4 — 🥈 Drafter-rerank tree spec (6-8 hrs, HIGHEST COMPLEXITY)

**Deep research rationale**:
- Classic EAGLE-2 tree verify blocked: aiter `mla_decode_fwd` on gfx950 FP8 qh=32 has NO mask parameter (verified in source)
- Batch-expansion fallback: 2× main compute for 1.2× tokens = NET LOSS at bs=4 MoE
- SGLang's own AMD MLA path is **chain-only** (they hardcode `max_q_len=1`, never feed tree mask)
- DRAFTER-side tree with CHAIN verify circumvents the kernel constraint

**Design**:
- Drafter samples top-2 at each of 3 depths → 8 candidate chains in drafter's internal state
- Score each chain by sum of log-probabilities (drafter's own logits)
- Pick highest-scoring chain
- Submit selected 3-chain to main fwd (unchanged qseqlen=4 verify)

**Drafter cost analysis**:
- Call 0: bs=4 (1 per seq) → sample top-2
- Call 1: bs=8 (2 branches/seq)
- Call 2: bs=16 (4 branches/seq)
- Total drafter tokens: 28 vs current 12 (2.33× compute)
- Drafter: 1.2 → 2.8 ms
- bs=16 IS in default cudagraph_capture_sizes — no capture fail

**Files modified**:
- `atom/spec_decode/eagle.py:193-220` — replace greedy argmax with topk(2), track branches, score chains
- Add `_score_chains(logits_per_step)` helper — cumulative log-prob
- Add `_select_best_chain()` helper — argmax over scored chains
- `rejection_sampler.py`: UNCHANGED (verifies single 3-chain as before)

**Testing**: boot, temp=0 greedy correctness probe (should match baseline), bench, GSM8K

**Expected**: per-step accept 65% → 80% with rerank. Tokens 2.5 → 2.95-3.18. Step 17.6 → 19.2. TPOT 6.36 → 5.9-6.5 ms.

## Phase 5 — Wildcard: TP=8 + BF16 KV + MTP=7 (1-2 hrs)

At TP=8: qh=16. aiter HAS `mla_a16w16_qh16_*_qseqlen=8` kernel (BF16 KV only).
- MTP=7 → 8 positions/seq → fits qseqlen=8 ✓

**Patches required**:
5.1 `atom/config.py:868` — relax `num_speculative_tokens > 4` guard up to 7
5.2 Launch: `-tp 8 --kv_cache_dtype bf16 --num-speculative-tokens 7`
5.3 Full bench + GSM8K

**Math**:
- Depth-7 chain at 70% accept: 3.14 tokens/step (+25%)
- MLA qseqlen=8 ≈ 1.5× vs qseqlen=4
- TP=8 divisor /8 (per-GPU halved if total thr doesn't double)
- Uncertain — but if it works, could unlock E2E gate

## Phase 6 — 🥉 BF16 CSV retune on ROCm 7.2.2 (2-4 hrs autonomous)

Our DEC-071 tune (old ROCm) incompatible (hipBLASLt solidx renumbered).

6.1 Stop server (frees GPUs)
6.2 Run `aiter/gradlib/gradlib/gemm_tuner.py` targeting DSR1 priority shapes (M=1/4/16 LM head, M=16 MLA projections)
6.3 Install tuned CSV at `/app/aiter-test/aiter/configs/model_configs/dsv3_bf16_tuned_gemm.csv`
6.4 Relaunch with winning stack + new CSV, bench

**Expected**: +0.1-0.2 ms TPOT polish.

## Phase 7 — Second-order sweeps (1-2 hrs)

Each: 10-15 min probe. Keep any win >0.1 ms, revert otherwise.

7.1 `ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=0` — may conflict with TBO prefill batch-split
7.2 `ATOM_ENABLE_DS_QKNORM_FUSION=0` — isolate contribution
7.3 `--block-size 8` and `--block-size 32` (vs default 16)
7.4 `--max-num-batched-tokens` sweep: 8192 / 16384 / 32768
7.5 `--scheduler-delay-factor 0.1` / 0.5
7.6 `--level` compilation: 2 / 1 (vs 3)

## Phase 8 — Final stability + submission (1 hr)

8.1 Lock winning stack (full config snapshot)
8.2 3× GSM8K at winning config (min-of-3 ≥ 0.93 for submission stability)
8.3 Multi-CONC: CONC=32, CONC=128 best-effort for those gate tiers
8.4 Commit to `dsr_beta_snapshot` branch
8.5 Push to GitHub
8.6 Submit to HuggingFace leaderboard

## Timeline budget

| Phase | Duration | Cumulative |
|---|---|---|
| 0 QKN probe | 15m | 0:15 |
| 1 fused_kv_bmm | 3-4h | 4:15 |
| 2 TP=8 probe | 30m | 4:45 |
| 3 sync patch | 2h | 6:45 |
| 4 drafter-rerank tree | 6-8h | 14:45 |
| 5 TP=8+BF16+MTP=7 | 1-2h | 16:45 |
| 6 BF16 retune | 2-4h | 20:45 |
| 7 sweeps | 1-2h | 22:45 |
| 8 submit | 1h | **23:45** |

**Fits within 24-hr budget with 15 min buffer.** Realistic parallelism (code while server runs) compresses to ~20 hrs.

## Projected gains (stack all)

| Phase | Conservative | Optimistic |
|---|---|---|
| 0 QKN | 0.05 | 0.3 |
| 1 fused_kv_bmm | 0.3 | 0.7 |
| 3 sync reduction | 0.2 | 0.6 |
| 4 drafter-rerank | 0.1 | 0.5 |
| 6 BF16 retune | 0.1 | 0.2 |
| 7 fusion toggles | 0.0 | 0.2 |
| **Cumulative TPOT drop** | **0.75 ms** | **2.5 ms** |
| Final TPOT | 5.61 ms | 3.86 ms |
| Final E2E (370+1024×TPOT) | 6115 ms | 4324 ms |
| Final interact (1000/TPOT) | 178 | 259 |
| Likely gates | 2/4 | 4/4 |

## Dependencies + parallelism

- Phase 0 → 1 (need QKN result first)
- Phase 1 → 2 (fresh bench after kv_bmm)
- Phases 3, 4, 5 INDEPENDENT code work (can write while other server runs)
- Phase 6 needs exclusive GPU (kill server first)
- Phase 7 after 6 (full stack)
- Phase 8 always last

## Rollback safety

Every phase has a defined rollback path. Winning baseline is `dsr_beta_snapshot@9194e8f` (1344/6.36/157/6896). Ultimate fallback = `main@557285e` (DEC-075 production 1278/6.74).

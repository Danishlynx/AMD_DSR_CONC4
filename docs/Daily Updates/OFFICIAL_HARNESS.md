# OFFICIAL Benchmark Harness — Source of Truth

**Last updated**: 2026-05-04 ~13:00 UTC (May 04 kimbochen-only directive added; May 03 F1/F4 + May 04 F2/G results appended)
**Authority**: this file is the canonical scoring procedure. Any other doc that contradicts it is wrong.

> 🚫 **MAY 04 KIMBOCHEN-ONLY DIRECTIVE (user, supersedes any other harness language in this repo):** the `dsr1_benchmark perf` wrapper IS the harness. Anything else is a fluke for purposes of gate scoring. Dev-bench / direct `benchmark_serving` / `--random-range-ratio 0.8` / no-chat-template / single-warmup ≈ 11-22% better TPOT than reality and cannot be cited as a target/ceiling/regress baseline. Lever WIN promotion = `dsr1_benchmark perf` 4-iter median(2,3,4) on the GOLD snapshot or a successor.

## Why this file exists

Through 2026-04-26 we used an **informal** bench formulation (`--random-range-ratio 0.8`, no `--use-chat-template`, `--num-warmups 1`) and reported "3/4 gates" results. That formulation is **NOT** the bounty's actual scoring harness. Under the actual harness the same stack scores **1/4 gates**. This file documents the official harness so we (and any reviewer) can never confuse the two again.

**The chat-template is what makes the difference.** kimbochen's `benchmark_serving.py` wraps every random-token prompt in DSR1's `<｜begin▁of▁sentence｜><｜User｜>...<｜Assistant｜>` template. The `<｜Assistant｜>` token activates DSR1-R1's **reasoning mode** — the model emits `<think>...</think>` before its actual output. Reasoning mode uses different MoE expert routing and is structurally **~15% slower per output token** than non-reasoning. Every prior "win" measured WITHOUT chat-template was overstating throughput by 11-15%.

## Source

`github.com/danielhua23/amdgpu_bounty_optimization/blob/main/dsr1-fp4-atom-mtp-mi355x/dsr1_benchmark.cpp`

The C++ source is the contract. The text below restates it verbatim where it matters.

## Step 1 — GSM8K accuracy (runs first, gates everything)

```
lm_eval --model local-completions \
  --model_args model=<MODEL>,base_url=http://0.0.0.0:<PORT>/v1/completions,num_concurrent=65,max_retries=1,tokenized_requests=False \
  --tasks gsm8k --num_fewshot 3
```

- Parser pulls `exact_match` value from `flexible-extract` row of the lm_eval table.
- Validation: `gsm8k_metric ≥ BASELINE_GSM8K_METRIC - GSM8K_TOL`. Defaults `BASELINE_GSM8K_METRIC=0.93`, `GSM8K_TOL=0`. So default minimum accepted = **0.93 strict**.
- **If FAIL → entire perf phase is SKIPPED. Result: 0 perf gates.**

**N=3 RULE (locked Apr 27 ~13:25 IST)**: single-shot GSM8K is NOT a valid promotion gate. Spread across same-stack reboots is 0.0113 (larger than lm_eval's reported stderr 0.0072). Use `agents/harness/official_bench_n3.sh` (mirror at `/tmp/official_bench_n3.sh` in container) which runs GSM8K ×3, takes the median, and only proceeds to perf if median ≥0.93.

## Step 2 — Perf bench (only runs if step 1 passes)

```
git clone https://github.com/kimbochen/bench_serving.git /tmp/bmk-<ts>
python3 /tmp/bmk-<ts>/benchmark_serving.py \
  --model <MODEL> --backend vllm --base-url http://0.0.0.0:<PORT> \
  --dataset-name random \
  --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \
  --num-prompts 40 --max-concurrency 4 --request-rate inf --ignore-eos \
  --save-result --num-warmups 8 \
  --percentile-metrics 'ttft,tpot,itl,e2el' \
  --use-chat-template
```

Critical flags that the informal harness got wrong (these are the ones that make all the difference):

| Flag | Informal (wrong) | Official | Effect |
|---|---|---|---|
| `--random-range-ratio` | 0.8 | **1.0** | every prompt is exactly ISL tokens; no short-sequence median pulldown |
| `--use-chat-template` | absent | **PRESENT** | activates DSR1 reasoning mode (`<think>` generation), ~15% slower per token |
| `--num-warmups` | 1 | **8** (= 2×CONC) | ensures all decode cudagraph batch sizes warmed |
| `--num-prompts` | (varied) | **40** for CONC=4 (= CONC×10) | enough samples for median stability |

## Step 3 — Compute & validate gates

```
tput_per_gpu = total_token_throughput / TP_size
            # script HARDCODES /8 for TP=8.
            # OUR DEPLOYMENT IS TP=4 → divide by 4.
interactivity = 1000.0 / median_tpot_ms
median_e2e    = median_e2el_ms
```

The hardcoded `/8` in `dsr1_benchmark.cpp:501-516` is a test-harness default. The **leaderboard scorer** uses `num_GPUs_used = TP_size`. For TP=4 this is `/4`. Confirmed by Daniel (Apr 13-14 Discord) + Ziguan (Apr 15 07:10 Discord: "Q: If we are using TP=4, is tok/s/GPU = total/4 or total/8? A: total_token/s/4"). The bench binary is wrong; the rules text is right.

### Gates by CONC (ISL=8192, OSL=1024)

| CONC | E2E (ms) | Interactivity (tok/s/u) | Tput/GPU (tok/s/GPU) |
|---|---|---|---|
| **4**   | ≤ 5000  | ≥ 165 | ≥ 1500 |
| 32  | ≤ 18000 | ≥ 50  | ≥ 3900 |
| 128 | ≤ 22000 | ≥ 48  | ≥ 6000 |

Plus **GSM8K ≥ 0.93** (gates everything).

---

## Canonical Measurements Under Official Harness

All numbers below are real and measured under the kimbochen-equivalent harness. Anything not in this section is suspect and must be re-measured before citation.

### Apr 27 — A26 baseline RE-MEASURED under official harness (what we thought was 3/4 was actually 1/4)

| Metric | Official (CORRECTED) | Old informal (WRONG) | Gate (CONC=4) | Status |
|---|---|---|---|---|
| GSM8K (num_fewshot=3 flexible) | **0.9378** | 0.9522 (num_fewshot=5) | ≥0.93 | ✅ PASS |
| Median E2E (ms) | **6908.25** | 5240 | ≤5000 | ❌ FAIL (−1908 ms, 38% off) |
| Tput / GPU (TP=4 → /4) | **1356.65** | 1650 | ≥1500 | ❌ FAIL (−10%) |
| Interactivity | **156.97** | 206.61 | ≥165 | ❌ FAIL (−5%) |

**Real status: 1/4 gates** (only GSM8K passes). Every "3/4 gates", "TPOT 4.84 ms", "thr/GPU 1650", "interactivity 207", "E2E gap 240 ms" claim from before Apr 27 ~06:15 IST is wrong and only retained for archival traceability.

### Apr 27 — A26 with RELAXED_TOP_N=9 fails GSM8K under official harness

A26 used `RELAXED_TOP_N=9` for the +0.14 ms TPOT win (under informal harness). Re-measured under official N=3:

| Stack | GSM8K runs | Median | Status |
|---|---|---|---|
| **A26** (TOP_N=9) | 0.9219 / 0.9348 / 0.9287 | **0.9287 FAIL** | gate ≥0.93 |
| **A27** (TOP_N=8) — REAL BASELINE | 0.9356 / 0.9318 / 0.9348 | **0.9348 PASS** | +0.0048 over gate |

`RELAXED_TOP_N=9` had to be reverted to 8. A26's "BREAKTHROUGH" doesn't survive the official harness.

### Apr 27 — A27 baseline (real) under N=3 official

| Metric | A27 | Gate | Status |
|---|---|---|---|
| GSM8K_median | **0.9348** | ≥0.93 | ✅ PASS (+0.0048) |
| Median E2E (ms) | **6578.27** | ≤5000 | ❌ FAIL (−1578, 32% off) |
| Tput/GPU (TP=4) | **1369.25** | ≥1500 | ❌ FAIL (−9%) |
| Interactivity | **162.06** | ≥165 | ❌ FAIL (−1.8%) |
| TPOT_med | 6.171 ms | — | (drives Intvty) |

**A27 = 1/4 gates at CONC=4 and is the real baseline going forward.** Every measurement after this date uses N=3 official harness. No more single-shot, no more random-range=0.8, no more "3/4" shorthand.

### Apr 27 — A27 CONC=32 reference

| Metric | A27 CONC=32 | Gate | Status |
|---|---|---|---|
| GSM8K | 0.9431 | ≥0.93 | ✅ PASS |
| Median E2E | 19044 ms | ≤18000 | ❌ FAIL (−5.8%) |
| Tput/GPU (TP=4) | 3831 | ≥3900 | ❌ FAIL (−1.8%) |
| Interactivity | 56.17 | ≥50 | ✅ PASS (+12%) |
| TPOT_med | 17.80 ms | — | — |

**CONC=32 = 2/4 gates** with sub-6% gaps on both failing metrics. Likely closes 4/4 *before* CONC=4 as levers stack — CONC=32 is closer to gate.

### Apr 28 — L2 v6c WIN under official harness (CONC=4)

| Stack | TPOT_med | E2E_med | Tput/GPU | Intvty | GSM8K_med | Gates |
|---|---:|---:|---:|---:|---:|:---:|
| Apr 28 17:00 clean baseline | 6.619 | 7202 | 1326 | 151.07 | 0.9325 | 1/4 |
| **L2 v6c rerun (3-iter)** | **6.470** | 7158 | 1303 | **154.55** | 0.9332 | 1/4 |
| Combined 6-iter (clean + v6c) | 6.553 | — | — | — | best 0.9416 | 1/4 |

**v6c delivers −0.149 ms TPOT (above ±0.10 noise) and the most comfortable GSM8K margin all session** (best iter 0.9416). **First confirmed shippable source change** since A26. Implementation: env-gate `ATOM_USE_POST_ATTN_RMSNORM_MXFP4_FUSION=1` in `atom/model_ops/layernorm.py` dispatches post-attn RMSNorm to `tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant` with custom-AR window size guard (`inp_size ≤ 8192·8192` bytes; prefill batches >8192·8192 fall back to bf16 path).

### Apr 29 — gold+v6c N=3 official (canonical shipping baseline)

`gold` = Apr 26 canonical env stack (RELAXED_TOP_N=8, INT4 AR, MSCG_K UNSET, MSCCLPP, dual-stream MoE 1024). `v6c` = `ATOM_USE_POST_ATTN_RMSNORM_MXFP4_FUSION=1` env-gate.

| Iter | TPOT (ms) | E2E (ms) | Tput/GPU | Intvty | GSM8K |
|---|---:|---:|---:|---:|---:|
| 1 | 6.260 | 6743 | 1403 | 159.74 | 0.9325 |
| 2 | 6.080 | 6584 | 1390 | **164.48** | 0.9340 |
| 3 | **5.925** | 6846 | 1424 | **168.78** | **0.9386** |
| **Median** | **6.080** | **6743** | **1403** | **164.48** | **0.9340 PASS** |

| Metric | Median | Gate | Status |
|---|---:|---:|:---:|
| GSM8K | 0.9340 | ≥0.93 | ✅ PASS |
| Intvty | 164.48 | ≥165 | ❌ FAIL by 0.52 (within ±0.10 ms TPOT noise) |
| Tput/GPU | 1403 | ≥1500 | ❌ FAIL (−6.5%) |
| E2E | 6743 | ≤5000 | ❌ FAIL (−34%, binding gate) |

**Median 1/4 gates. Best iter (iter3) = 2/4** (Intvty 168.78 PASS, TPOT 5.925 = best official-harness single iter ever for DSR1 this campaign).

**Re-bench (3 more iters Apr 29 03:13 IST):** TPOT 6.111 / 6.357 / 6.376 → median 6.357. Combined 6-iter median ≈ 6.20 ms. Variance ±0.25 ms attributable to host contention drift.

**Snapshot:** `rocm/atom-dev:dsr1_apr29_gold_v6c_2of4` sha `4911df6937b152` + `locked/dsr1:gold_v6c_2of4`.

**v6d (MoE prequant skip via Python dict + setattr) is DEAD on cudagraph incompatibility.** Reverted; gold+v6c is the canonical shipping stack. v6e (tensor-buffer redesign) — pending, never completed.

### Apr 30 — Phase 11 per-phase MTP v3 (canonical 2/4)

Real source-level lever (TRT-LLM port). Per-seq phase tensor tracking `<think>` token transitions. Inside `<think>` block use RELAXED_TOP_N=10 DELTA=0.6; outside (and for end-of-block) use RELAXED_TOP_N=8 DELTA=0.5.

| Metric | Value | Gate | Verdict |
|---|---:|---:|---|
| GSM8K_med (N=3: 0.9356/0.9318/0.9318) | **0.9318** | ≥ 0.93 | ✅ PASS (margin +0.0018) |
| TPOT_med | **5.641 ms** | (drives Intvty) | — |
| Median E2E | 6210 ms | ≤ 5000 | ❌ FAIL (off by 1210) |
| Tput/GPU (/4) | 1449 | ≥ 1500 | ❌ FAIL (off by 51) |
| Interactivity | **177.26** | ≥ 165 | ✅ PASS (margin +12.26) |

**2/4 gates** (GSM8K + Intvty). Tput close, E2E still off by 1.2 sec.

**Snapshot:** `rocm/atom-dev:dsr1_apr30_phase11_v3_2of4` sha `c58cf2ce4512` + `locked/dsr1:phase11_v3_2of4`.

**Caveat about Apr 29 L0-v2 single-iter "2/4"**: that reading was a cold-start outlier per Apr 30 re-bench (1/4 under steady warm). The Apr 30 Phase 11 v3 result is the reproducible 2/4 verdict.

### May 01 — R2 small-M MoE GEMM kernel attempts (closed at 1/4)

R2-A (snapshot tensors) + R2-B (microbench) showed T0=71.88 µs/call, 0.85% FP4 peak compute, 13.82% HBM peak BW → 5-20× theoretical headroom. R2-C M1..M2.13 built bit-exact MFMA `mfma_scale_f32_16x16x128_f8f6f4` MoE GEMM2 kernel with multi-expert atomic accumulator (err_ratio=0.0048 PASS).

**R2-D wired into AITER dispatcher**: bench TPOT 10.203 ms = +4.10 ms regression vs clean baseline. Microbench 27.97 µs/call = 2.57× T0. Root cause: **grid structure issues 7168 CUDA blocks/call (16 sorted × 448 N-tiles) vs FlyDSL atomic_bnt2_persist's ~56 blocks (tile_M=32 tile_N=128 persistent threadblock). 128× more grid launches → host-device dispatch overhead dominates per-call latency.**

R2-D v4 (+ native HW `global_atomic_pk_add_bf16` inline asm): saved 1.65 ms vs CAS-loop but bench still +4.10 ms regression. Fix = persistent-threadblock restructure (multi-day rewrite, not pursued). All sources reverted.

### May 01 — Clean baseline post-cache-wipe

| Snapshot | sha | TPOT | Use |
|---|---|---:|---|
| `rocm/atom-dev:dsr1_may01_p1_aiter_cherry` | `7630c22aee40` | 5.859 | P1 (AITER #2890 + #2823 cherry-picked, gates default OFF) |
| `rocm/atom-dev:dsr1_may01_clean_baseline_5894` | `b6f8c9d03206` | 5.894 | Phase 0 anchor (post-cache-wipe) |

Note: TPOT drift 5.641 (Phase 11 v3 KEEP) → 5.859 (P1) → 5.894 (clean) reflects cache state + host contention; the snapshots are intact.

### May 02 — Drafter cudagraph campaign (W3.2-A/B v1/v2/v3-meta/v4 + MSCG-P6) — closed

| Variant | Result |
|---|---|
| W3.2-B v1 (separate Stream) | SIGABRT first decode (NCCL IPC peer pointer mismatch) |
| W3.2-B v2 (gc.stream + graph_pool) | SIGABRT — `mla_decode_stage1_asm_fwd` non-persistent path at qseqlen=1 gqa=32 has no kernel |
| W3.2-B v3-meta (`prepare_mtp_decode()` populate, gated `if wmd is None`) | bs=1,2,4,8 capture OK; bs=16 SIGABRT (stale wmd from prior bs=8 transition) |
| **W3.2-B v4** (ALWAYS rebuild `work_meta_data`) | **24 captures, 0 SIGABRTs, GSM iter2 0.9310 PASS** but **TPOT +0.317 ms regression** |
| MSCG-P6 (pre-existing `model_runner.capture_cudagraph_multistep`) | Boot OK, smoke PASS, rank 2 SIGABRT mid-GSM8K (same MLA root cause) |

**W3.2-B v4 final 3x bench (May 02 ~03:00 UTC re4c_v10)**:

- Baseline (eagle.py.pre_w32b, 319 lines): TPOT_med 5.810 / E2E_med 6292 / Tput/GPU 1491 / Intvty 172.1 / GSM_med 0.9280 — **1/4 gates**
- v4 (drafter cudagraph end-to-end): TPOT_med 6.127 / E2E_med 6611 / Tput/GPU 1410 / Intvty 163.20 / GSM_med 0.9287 / GSM_iter2 0.9310 PASS — **1/4 gates (worse)**
- Δ: **+0.317 ms TPOT, +319 ms E2E, −81 Tput, −8.9 Intvty**

Why the lever didn't deliver: trace event count ≠ wall-clock dispatch tax. Real dispatch tax was ~0.3 ms TPOT, not the 1.6 ms the breakthrough EV math predicted. v4 introduced replay overhead (3 input copy_() + graph launch + output buffer reads ≈ 0.4 ms) that exceeded the dispatch saving. **Cudagraph lever empirically dead.** Reverted eagle.py to clean baseline.

### May 02 — AITER PR #2927 cherry-pick (qh128 gfx950 gate flip)

3-site patch (`aiter/mla.py:415`, `aiter/ops/attention.py:942`, `csrc/kernels/mla/metadata/v1_2_device.cuh:481`): `gfx942` → `gfx942 OR gfx950`. Enables native qh128 path on gfx950 for persistent MLA dispatch.

3-iter bench result:

| Metric | Pre #2927 | Post #2927 | Δ |
|---|---:|---:|---:|
| TPOT_med | 5.810 | **5.777** | −0.033 ms |
| GSM iter1 | 0.9280 | 0.9303 | +0.0023 |
| Iter1 gates | 1/4 | **2/4** | +Tput passed @ 1500-edge |
| Median gates | 1/4 | 1/4 | unchanged |

**Real improvement but tiny (within noise floor).** Iter1 alone hits 2/4 gates because Tput sits at the 1500 gate edge. Median across iters still 1/4.

### May 03-04 — Phase A/B kernel-author levers (architectural insights, ΔTPOT ≈ 0)

| Lever | Date | Result | Why |
|---|---|---|---|
| F1 force `run_1stage=1` (CSV) | May 03 | DEAD | `aiter.fmoe_g1u1` ASM dispatcher cannot resolve FlyDSL kernel names; ASM registry empty for our shape (cu=256/expert=257/topk=9 FP4×FP4) |
| F1 v2 ASM CSV write (`fmoe_bf16_pertokenMXfp4_g1u1_*_32x512`) | May 03 | NEUTRAL/regress | M=32 only +0.005 ms (within noise); M=4-32 +1.27 ms (tile-32 padding waste 87.5% at M=4) |
| F4 `gemm_moe_tune.py` re-retune | May 03 | NEUTRAL | 5/8 wins fired at decode (M=8/16/32), but median TPOT +0.43 ms; M=4 hot path unchanged + bench variance ±0.5 ms swamps non-M=4 wins |
| F2 tile_m=16 FlyDSL stage1 author | May 04 | NO WIN | t16 always loses to t32 at M=4: same MFMA primitive, t32 has K-loop pipelining advantage at HBM-bound regime |
| F7 `moe_fused_gate` swap | May 04 | DEFERRED | Ceiling −0.02 to −0.05 ms < ±0.5 ms variance floor + GSM regression risk |
| Lever Z `_fp4`-suffix-everything CSV | May 04 | DEAD | GSM 0.9265 FAIL — AMD's selective `_fp4` placement is intentional calibration, not omission |

### May 04 — Lever G `PHASE_RELAXED_TOP_N 10→12` (PROVISIONAL WIN, dev-bench only)

Single-line edit `/app/ATOM/atom/model_ops/rejection_sampler.py:22`. Wider top-N inside `<think>` block lifts MTP acceptance from ~69% → ~72-75%; PHASE_STRICT_TOP_N=8 unchanged → outside-`<think>` GSM accuracy preserved.

| Bench | TPOT_med | GSM | Status |
|---|---:|---:|---|
| dev-bench 4-iter med(2,3,4) | **5.5141 ms** | — | -0.200 ms vs 5.7141 dev-bench baseline |
| GSM canary 3-iter median | — | **0.9401** | PASS, margin +0.0076 over 0.9325 gate |
| **kimbochen 4-iter med(2,3,4)** | **PENDING** | **PENDING** | **NOT YET RUN — required for promotion** |

**Status**: provisional. Source edit lives in `dsr1_fresh` working container only. Until kimbochen 4-iter median confirms, do NOT cite "−0.200 ms WIN" without the "(dev-bench, kimbochen-pending)" qualifier. **G v2 (TOP_N=14) regressed** — saturated at 12.

**Outside-the-box mechanism**: AMD's Phase 1a only tested STRICTER direction (9, 0.55) which regressed +0.51 ms in kimbochen mode; PERMISSIVE direction was untried. Lever direction matters.

### May 03 profiling (supersedes Apr 20 hipGraphLaunch finding)

GPU 78.2% busy (was 2.3% Apr 20), kernel-bound now. **hipGraphLaunch only 3.1% wall** (Apr 20 cudagraph fix landed). hipEventSynchronize 50.9% = CPU waiting on busy GPU. Top kernels: MoE 27%, MLA BF16 GEMM 17.6%, MLA attn 15.7%, AR 4.3%. CONC=4 is fixed-overhead-per-kernel dominated. Lever class for closing E2E = **kernel COUNT reduction (fusion, persistent CTAs)**, not µs-per-kernel tuning.

### May 02 — MTP=4 retest with PR #2927: still crashes

Theory: PR #2927 enables qh128 dispatch where `gqa_ratio==128, sub_Q=128, config_max_seqlen_q=0` (generic qseqlen). MTP=4 (qseqlen=5) should dispatch cleanly through the qh128 kernel.

**Reality**: boot crashed at `asm_mla.cu:308 only support fp8 mla decoding for qo_len <= 4`. Dispatcher hit the gqa=16 branch (NOT gqa=128). ATOM's `attention_mla.py:_forward_decode` pre-folds the q tensor to nhead=16 BEFORE calling aiter. PR #2927 helps inside aiter but never reaches the dispatcher because the fold happens upstream. Patching ATOM to skip fold when arch=gfx950+fp8/fp8+nhead=128 is the next attempt (not yet done; ~2h investigation).

---

## Forbidden shorthand (do not use, ever)

- ❌ "3/4 gates" without harness qualifier — was an informal-harness artifact
- ❌ "TPOT 4.83 ms" / "TPOT 4.84 ms" — informal random-range=0.8 numbers
- ❌ "E2E gap 240 ms" — real gap is 1908 ms under official
- ❌ "thr/GPU 1650" / "1675" / "interactivity 207" — all informal
- ❌ "RELAXED_TOP_N=9 wins" — works under informal, fails GSM8K under official
- ❌ "BREAKTHROUGH" + any number not measured with N=3 + chat-template + ratio=1.0 + 8 warmups
- ✅ Use the real official numbers above. Quote with snapshot SHA + harness version + N=3 if applicable.

## How to run locally

In container `re4c_v10`: `bash /tmp/official_bench_n3.sh` (mirrors the C++ source). Output is in `/tmp/official_bench_n3_<HHMMSS>/`.

For a single run: `bash /tmp/official_bench_3x.sh`.

## Pointers
- Memory entry: `~/.claude/projects/c--Users-danis-OneDrive-Desktop-AMD/memory/reference_official_harness_apr27.md`
- Phase plan: `docs/Daily Updates/Phase.md` (this folder)
- Reproduce: `docs/Daily Updates/REPRODUCE.md`
- Master engineering log: `docs/Daily Updates/MASTER.md`
- Snapshot inventory: `docs/Daily Updates/SNAPSHOT_INVENTORY_apr24.md`

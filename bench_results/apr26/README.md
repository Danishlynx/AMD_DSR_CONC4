# Apr 26 2026 EOD Bench Result JSONs

Raw `benchmark_serving` output from each bench run on `re4c_v10` (DSR1-MXFP4, locked Apr 26 stack). All results ISL=8192 OSL=1024.

## Index

| File | CONC | TP | Warmup | Date | Notes |
|---|---|---|---|---|---|
| `conc4_warm_run1.json` | 4 | 4 | 8-curl | 2026-04-26 11:27 | Canonical 3-run sweep, run 1 |
| `conc4_warm_run2.json` | 4 | 4 | 8-curl | 2026-04-26 11:28 | Canonical 3-run sweep, run 2 |
| `conc4_warm_run3.json` | 4 | 4 | 8-curl | 2026-04-26 11:29 | Canonical 3-run sweep, run 3 |
| `conc4_nowarm_run1.json` | 4 | 4 | none | 2026-04-26 14:11 | Cold-decode-graph Run 1: TPOT_mean=7.148 (vs warm 5.03), thr/GPU=1161 (vs 1660) |
| `conc4_nowarm_run2.json` | 4 | 4 | none | 2026-04-26 14:12 | Equivalent to warm (decode graphs hit during Run 1) |
| `conc4_nowarm_run3.json` | 4 | 4 | none | 2026-04-26 14:13 | Equivalent to warm |
| `conc32_warm_run1.json` | 32 | 4 | 8-curl | 2026-04-26 13:54 | num_prompts=64 |
| `conc32_warm_run2.json` | 32 | 4 | 8-curl | 2026-04-26 13:55 | num_prompts=64 |
| `conc128_nowarm_run1.json` | 128 | 8 | none | 2026-04-26 14:36 | TP=8 cold-boot first bench, TPOT_std=696, p99=4051 ms cold-tail |
| `conc128_nowarm_run2.json` | 128 | 8 | none | 2026-04-26 14:37 | TP=8 cold-boot, second bench (still no explicit warmup) |
| `conc128_warm_run1.json` | 128 | 8 | 8-curl | 2026-04-26 14:40 | After 8-curl warmup, TPOT_std collapsed 696→6.7 (−99%) |
| `conc128_warm_run2.json` | 128 | 8 | 8-curl | 2026-04-26 14:41 | TPOT_std 12.8 |
| `gsm8k_a26_tp8.json` | — | 8 | n/a (eval) | 2026-04-26 16:42 | **GSM8K accuracy eval on TP=8 server, 1319 prompts, 3-shot, temp=0**. Flexible-extract **0.9462** / strict-match **0.9454** — both PASS gate ≥0.93. Ran in 124 sec via `lm_eval --tasks gsm8k --num_fewshot 3 --gen_kwargs temperature=0,max_gen_toks=512 --num_concurrent 16`. |

## Best CONC=4 result (canonical 3/4-gates)

3-run median across `conc4_warm_run{1,2,3}.json`:

| Metric | Median across 3 runs | Gate | Pass? |
|---|---|---|---|
| TPOT_med | **4.840 ms** | ≤ 6.06 | ✅ |
| TTFT_med | 288.93 ms | (no gate) | — |
| Total token throughput / GPU | **1650** | ≥ 1500 | ✅ |
| Interactivity (1000/TPOT_med) | **206.61** tok/s/user | ≥ 165 | ✅ |
| E2E_calc_med (TTFT + 1023×TPOT) | **5240 ms** | ≤ 5000 | ❌ −240 ms |
| GSM8K (separate `lm_eval` run, TP=4 morning) | **0.9522 flexible / 0.9469 strict** | ≥ 0.93 | ✅ |
| GSM8K (separate `lm_eval` run, TP=8 EOD) — see `gsm8k_a26_tp8.json` | **0.9462 flexible / 0.9454 strict** | ≥ 0.93 | ✅ |

GSM8K independence from TP topology is itself confirmation of correctness: 0.9522 (TP=4) vs 0.9462 (TP=8) is well within the ±0.6% statistical noise on 1319 prompts (stderr 0.0062). Both PASS the gate cleanly.

All `conc4_warm_run*.json` were generated with the `bench_apr26.sh` wrapper (in [`scripts/`](../../scripts/)) which does:
1. Set `HF_HUB_OFFLINE=1` environment
2. 8 small `curl /v1/completions` warmup requests with `max_tokens=50`
3. 3 sequential `python -m atom.benchmarks.benchmark_serving --model amd/DeepSeek-R1-0528-MXFP4 --port 8890 --dataset-name random --random-input-len 8192 --random-output-len 1024 --num-prompts 40 --max-concurrency 4 --trust-remote-code`

## Warmup-effect summary

| CONC | Metric | NOWARM | WARM | Δ |
|---|---|---|---|---|
| 4 | TPOT_med | 4.901 | 4.840 | −1.3% (within DVFS noise on warm server) |
| 4 | thr/GPU | 1656 | 1650 | flat |
| 4 | E2E_calc | 5303 | 5240 | −63 ms |
| 4 | TPOT_mean (Run 1 only) | 7.148 | 5.015 | **−42%** (Run 1 cold-tail) |
| 4 | thr/GPU (Run 1 only) | 1161 | 1632 | **+30%** (Run 1 cold-tail) |
| 128 | TPOT_med | 26.50 | 26.26 | −0.9% |
| 128 | thr/GPU | 3051 | 3579 | **+17.3%** |
| 128 | TTFT_med | 18,777 | 12,629 | **−6,148 ms (−33%)** |
| 128 | E2E_calc_med | 45,890 | 39,489 | **−6,401 ms (−14%)** |
| 128 | TPOT std | 696-771 | 6.7-12.8 | **−98% tail** |

Warmup matters most at high concurrency (cold cudagraph compile happens during the first batch, contaminating TTFT and tail). At CONC=4 on a long-running already-warm server, the 8-curl trick is mostly cosmetic for the median — but on cold-boot Run 1, it's the difference between 3/4 gates and missing throughput by 30%.

## CONC=128 vs Apr 10-13 vanilla (`test_mtp3_conc128`, TP=8 MTP=3 fp8kv)

| Metric | Apr 10-13 vanilla | Today WARM | Δ |
|---|---|---|---|
| thr/GPU | 3192 | 3579 | **+12.1%** |
| TPOT_med | 45.95 ms | 26.26 ms | **−42.9%** |
| Interactivity | 21.76 | 38.09 | **+75.0%** |
| E2E_calc_med | 48,394 ms | 39,489 ms | **−18.4%** |

(Vanilla Apr 10-13 numbers are referenced in [`../../docs/REPRODUCE.md`](../../docs/REPRODUCE.md) Section 0 Stack Genealogy from `project_bounty_dir_prior_experiments.md`.)

# Profiling Playbook — replicating today's DSR1 investigation for any MI355X workload

**Context**: On Apr 22 (session-14) we spent a day chasing a "wrapper regression" on DSR1 and learned that:
1. The leaderboard wrapper does NOT match what you measure with your team's in-tree bench
2. Small flag differences (`--use-chat-template`) can completely change the bottleneck regime (host-bound → kernel-bound)
3. The Apr 20 profile captured in clean state is USELESS for reasoning-mode analysis
4. A torch.profiler trace under the actual wrapper workload is the only reliable way to pick optimization targets

This playbook is for **Kimi-Opus teammate** (or anyone running a different model/track). Follow it step by step to:
- Discover what your leaderboard wrapper actually measures
- Determine if your "gold" numbers match what the wrapper measures
- Identify your real kernel bottleneck under submission-flow conditions
- Build a data-driven optimization plan instead of chasing whatever was dominant in a clean trace

---

## Step 0 — Identify your wrapper binary and inspect its source

Every Track has its own wrapper. For DSR1 it's `dsr1_benchmark`; for Kimi it's `kimi_benchmark`.

```bash
find /projects/teamA/danish/repos/amdgpu_bounty_optimization -maxdepth 3 -name "*benchmark*"
```

Expected output for Kimi team:
```
/projects/teamA/danish/repos/amdgpu_bounty_optimization/kimik25-fp4-vllm-mi355x/kimi_benchmark
/projects/teamA/danish/repos/amdgpu_bounty_optimization/kimik25-fp4-vllm-mi355x/kimi_benchmark.cpp
```

**Read the .cpp source** — specifically look at `run_benchmark_serving` and grep for:
- `git clone` → what bench tool does it download?
- `--use-chat-template` → is chat-template activation present? (for Kimi: NO, for DSR1: YES)
- `--ignore-eos` → forced full output
- `--num-warmups` → warmup prompt count (DSR1: 8, Kimi: 2×CONC)
- `--random-range-ratio` → input length variance
- All flags between `benchmark_serving.py` and `--result-filename` → the exact args your wrapper invokes

```bash
grep -nE "use-chat-template|git clone|benchmark_serving\.py|--num-warmups|--percentile" \
  /projects/teamA/danish/repos/amdgpu_bounty_optimization/kimik25-fp4-vllm-mi355x/kimi_benchmark.cpp
```

**What Kimi-Opus will find**:
- Wrapper clones `github.com/kimbochen/bench_serving.git` (same repo DSR1 uses)
- Runs `--backend vllm --ignore-eos --num-warmups {2*CONC}` BUT does NOT pass `--use-chat-template`
- Random tokens are sent raw — NO chat template wrapping → no reasoning-mode activation on Kimi-K2-Thinking

This is different from DSR1. Your bottleneck regime may differ.

---

## Step 1 — Compare wrapper result vs your team's in-tree bench (side-by-side)

Your "gold" numbers probably came from your team's bench tool with DIFFERENT args. You need to know if the wrapper's kimbochen fork + wrapper's flags give you the same numbers.

On your **warm server** (skip GSM8K for fast iteration), run both benches back-to-back, same prompt count, all wrapper flags:

```bash
docker exec <your_container> bash -c '
  # Clone kimbochen fork fresh
  [ ! -d /tmp/kimbochen_bench ] && git clone https://github.com/kimbochen/bench_serving.git /tmp/kimbochen_bench

  # Phase 1: your in-tree bench (whatever tool your team uses)
  cd /app/<your_stack>  # e.g. /app/ATOM or /app/SGLang
  python3 -m atom.benchmarks.benchmark_serving \
    --model <YOUR_MODEL> --port <YOUR_PORT> \
    --dataset-name random --random-input-len 8192 --random-output-len 1024 \
    --num-prompts 40 --max-concurrency 4 --trust-remote-code \
    --ignore-eos \
    --save-result --result-filename /tmp/TOOL_TEAM_run1.json 2>&1 | tail -25

  sleep 5

  # Phase 2: kimbochen bench with the EXACT flags the wrapper uses
  # (for Kimi team: no --use-chat-template)
  python3 /tmp/kimbochen_bench/benchmark_serving.py \
    --backend vllm --base-url http://0.0.0.0:<YOUR_PORT> \
    --model <YOUR_MODEL> --trust-remote-code \
    --dataset-name random --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1 \
    --num-prompts 40 --max-concurrency 4 --request-rate inf --ignore-eos \
    --save-result --num-warmups 8 --percentile-metrics "ttft,tpot,itl,e2el" \
    --result-filename /tmp/TOOL_WRAPPER_run1.json 2>&1 | tail -25
'
```

**What to look for**:
- If `thr_per_gpu` is ≥95% matched → wrapper ≈ your bench, gold numbers are representative
- If `thr_per_gpu` is 90-95% matched → small tool-level difference (pacing, warmup). Worth noting but not critical
- If `thr_per_gpu` is <90% matched → MAJOR discrepancy. Investigate flag differences BEFORE profiling

**For DSR1 we found 14% gap; for Kimi the gap may be smaller (no chat template).**

Compare both JSON results:
```bash
for f in /tmp/TOOL_TEAM_run1.json /tmp/TOOL_WRAPPER_run1.json; do
  jq "{thr_per_gpu: (.total_token_throughput/4), tpot_mean: .mean_tpot_ms, tpot_median: .median_tpot_ms, e2e_median: .median_e2el_ms, duration: .duration}" $f
done
```

---

## Step 2 — Isolate the flag(s) causing any gap

If you see a gap, bisect by running kimbochen bench with one flag removed at a time. E.g., for DSR1 the gap was purely chat-template:

```bash
# Run 3: kimbochen bench WITHOUT chat-template (remove that flag)
python3 /tmp/kimbochen_bench/benchmark_serving.py ... <same-flags-minus-problem-flag> \
  --result-filename /tmp/TOOL_WRAPPER_NOFLAG_run1.json
```

Compare three runs to isolate which specific flag is responsible for the delta. Our Apr 22 result:
- ATOM bench no-chat-template: 1514 thr/GPU
- Kimbochen WITH chat-template: 1308 thr/GPU (-14%)
- Kimbochen WITHOUT chat-template: 1477 thr/GPU (-2.4%)

Conclusion: ~85% of gap = chat-template, ~15% = kimbochen tool quirks.

For Kimi-Opus: your wrapper doesn't pass chat-template, so expect the gap (if any) to be small. Primary suspects are warmup count and percentile-metrics.

---

## Step 3 — Enable torch.profiler on your server

The default launch script may not wire up the torch profiler. Check engine kwargs in the boot log:

```bash
grep "Engine kwargs" /tmp/<your_boot_log>.log | head -1 | grep -oE "torch_profiler_dir[^,]*"
```

If it shows `torch_profiler_dir: None`, you need to explicitly pass the CLI flag in your launch script.

**Env var alone is NOT enough** — ATOM's arg_utils default `None` overrides the env-based default. Add the CLI flag:

```bash
# In your launch script, add before --cudagraph-capture-sizes:
--torch-profiler-dir /tmp/torch_traces_contaminated
```

Also create the dir:
```bash
mkdir -p /tmp/torch_traces_contaminated
```

Rebuild your launch script with this, then:
```bash
docker restart <your_container>  # NECESSARY — pkill leaves zombie VRAM; OOM on re-launch otherwise
# Wait 15s for VRAM to clear
docker exec -d <your_container> bash -c "bash /tmp/<your_launch>.sh > /tmp/<boot_log>.log 2>&1"
# Wait 10-15 min for cold boot with cudagraph captures
```

Verify boot success:
```bash
docker exec <your_container> curl -s --max-time 2 http://localhost:<port>/health
# Should return {"status":"ok"}

docker exec <your_container> grep -oE "torch_profiler_dir[^,]*" /tmp/<boot_log>.log | head -1
# Should show torch_profiler_dir: '/tmp/torch_traces_contaminated'
```

---

## Step 4 — Capture a profiled bench matching the wrapper workload

```bash
docker exec <your_container> bash -c '
  cd /app/<your_stack>
  python3 -m atom.benchmarks.benchmark_serving \
    --model <YOUR_MODEL> --port <YOUR_PORT> \
    --dataset-name random --random-input-len 8192 --random-output-len 1024 \
    --num-prompts 12 --max-concurrency 4 --trust-remote-code \
    --ignore-eos \
    --profile \
    --save-result --result-filename /tmp/PROFILED.json 2>&1 | tail -25
'
```

- `--num-prompts 12` keeps trace size manageable (~1.4 GB per rank for 12 prompts; 40 prompts = ~5GB)
- `--profile` triggers server's `POST /start_profile` and `POST /stop_profile` endpoints
- **For Kimi-Opus: DO NOT add `--use-chat-template` since your wrapper doesn't use it.** Replicate your wrapper's exact flags.

Verify traces were written:
```bash
docker exec <your_container> find /tmp/torch_traces_contaminated -type f -name "*.pt.trace.json.gz"
# Should show 4 files (rank_0 through rank_3), ~1.4 GB each
```

If 0 files: profiler didn't activate. Check:
- Did server boot with `torch_profiler_dir` set (Step 3)?
- Do you see `"Profiling started"` and `"Profiling stopped"` in server log during the bench?
- Does `/tmp/torch_traces_contaminated` exist and is writable?

---

## Step 5 — Parse the trace and identify bottleneck category

Use the parser at `/tmp/diff_traces.py` (on `reproducer_best` container — copy to yours):

```bash
docker cp reproducer_best:/tmp/diff_traces.py - | docker cp - <your_container>:/tmp/diff_traces.py
```

Or write your own minimal version — the key categories to split kernels into:
- MoE (gemm1/gemm2/sort/topk)
- MLA (attention decode)
- GEMM_BF16 (hipBLASLt Cijk + hgemm)
- GEMM_FP8 / GEMM_FP4 (quantized)
- AllReduce (NCCL / quick_reduce)
- RMSNorm, RoPE, Sample

Run:
```bash
docker exec <your_container> python3 /tmp/diff_traces.py \
  /tmp/CLEAN_BASELINE.json.gz \
  /tmp/torch_traces_contaminated/rank_0/*.pt.trace.json.gz
```

(If you don't have a clean baseline, just profile clean-state first with same flags MINUS chat-template or whatever difference you found in Step 2.)

---

## Step 6 — Read the trace's executive summary

Key numbers to extract from the parsed output:
1. **GPU busy %** = `kernel_total_ms / wall_ms`. If >50% you're kernel-bound; if <20% you're host-bound.
2. **Dominant HIP API** = which `cuda_runtime` entry is biggest.
   - `hipGraphLaunch` dominant → host-bound → optimize graph node count, dispatch flow
   - `hipEventSynchronize` dominant → kernel-bound → optimize kernel time directly
3. **Top kernel category by ms** = the #1 optimization target.
4. **Top individual kernel** = specific kernel to profile deeper.

For DSR1 reasoning mode we found:
- GPU 98.2% busy (kernel-bound)
- hipEventSynchronize 60% of wall (waiting for GPU)
- MoE = 37% of kernel time (#1 target)
- moe_gemm1_0 + moe_gemm2_0 = 25% of kernel time (specific kernels)

**For Kimi-Opus**: you'll likely find a different kernel regime since:
- Kimi-K2.5 has different MoE shape (384 experts vs DSR1's 256)
- Kimi wrapper doesn't trigger reasoning
- Different attention backend (Kimi-K2 uses AITER MLA directly, same kernel family but different nhead)

---

## Step 7 — Build an optimization plan based on the numbers

Rank optimizations by:
1. **Percentage of kernel time** (or wall time if host-bound)
2. **Effort to implement** (env flip = 30 min, kernel retune = hours, custom kernel = days)
3. **Risk of regression** (accuracy, crashes, incompatible shapes)

Quick wins (usually stackable):
- `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4` if AllReduce >5% of kernel time
- `HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0=1` if GEMM >15%
- `--enable-prefix-caching` if TTFT dominates (not usually the case under wrapper)

Medium:
- AITER MoE CSV retune for your exact shape (if MoE >20%)
- Verify `VLLM_ROCM_USE_AITER_MOE=1` is actually being used
- Check `ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD` is correct for your batch size

Big:
- Custom HipKittens MLA kernel port (multi-day) — only if MLA >15%
- Swap to ATOM MoE vs aiter MoE path

---

## Step 8 — Validate under wrapper (the ONLY test that counts)

After any optimization, the real gate test is:
```bash
docker exec <your_container> bash -c '
  export MODEL=<YOUR_MODEL> PORT=<YOUR_PORT> TP=4 CONC=4 ISL=8192 OSL=1024 NUM_PROMPTS=40
  export RESULT_FILENAME=/tmp/WRAPPER_run
  /projects/teamA/danish/repos/amdgpu_bounty_optimization/kimik25-fp4-vllm-mi355x/kimi_benchmark perf
'
```

(Or `submit` mode if you want to post to leaderboard.)

**Run 3× for min-of-3**, require all gates pass in EACH run before claiming victory. Direct bench numbers are NOT submission evidence.

---

## Gotchas we hit today (learn from our pain)

1. **`pkill -9 python3` leaves ROCm VRAM allocated**. You MUST `docker restart <container>` to free VRAM before relaunching server. Otherwise: OOM on boot.
2. **Env var set in launch script is insufficient**. Need explicit `--torch-profiler-dir` CLI flag. ATOM's arg_utils default `None` overrides env.
3. **Clean-state profile is NOT representative of wrapper performance**. A 74-second host-bound clean bench and a 23-second kernel-bound reasoning bench have completely different bottlenecks. Always profile under wrapper-equivalent workload.
4. **GSM8K does NOT contaminate server state**. We wasted hours on this hypothesis — don't.
5. **`--use-chat-template` activates reasoning mode on DSR1-R1**. If your wrapper passes it and your model is a reasoning model, EXPECT ~14% TPOT bloat intrinsic to the workload.
6. **Profiler `--profile` flag needs server-side support** (`/start_profile` and `/stop_profile` HTTP endpoints). Check "Profiler trace saved" log line.

---

## Reference files on the shared server

All in `reproducer_best` container, copy to yours via `docker cp reproducer_best:<path> - | docker cp - <your_container>:<path>`:

- `/tmp/diff_traces.py` — trace category parser
- `/tmp/parse_torch_trace.py` — kernel breakdown
- `/tmp/parse_trace_hip_api.py` — HIP API breakdown
- `/tmp/kimbochen_bench/benchmark_serving.py` — the leaderboard's actual bench tool (fresh clone)
- `/tmp/side_by_side.sh` — template for Step 1 comparison
- `/tmp/reasoning_profile.sh` — template for Step 4 profiled capture
- `/tmp/apr20_baseline_rank0.json.gz` — DSR1 clean-state torch.profiler trace (62MB)
- `/tmp/torch_traces_contaminated/rank_0/DeepSeek-R1-0528-MXFP4_ts_20260421_141950_461.pt.trace.json.gz` — DSR1 reasoning-mode trace (1.4GB)

---

## TL;DR for Kimi-Opus (with exact facts from `kimi_benchmark.cpp`)

### Your wrapper's actual bench invocation (confirmed from source)
```
python3 kimbochen_bench/benchmark_serving.py
  --model moonshotai/Kimi-K2.5 --backend vllm --base-url http://0.0.0.0:8888
  --dataset-name random --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1
  --num-prompts {CONC*10} --max-concurrency {CONC}
  --trust-remote-code --request-rate inf --ignore-eos
  --num-warmups {2*CONC}        # 8 at CONC=4, 64 at CONC=32, 256 at CONC=128
  --percentile-metrics 'ttft,tpot,itl,e2el'
  --save-result --result-filename result.json
```

**Explicitly NOT passed**: `--use-chat-template` → random tokens sent raw → **no DSR1-style reasoning activation**. Your perf measurement is what you see in direct random-token benches.

### Your gates (from `BASELINES` map, line ~80 of kimi_benchmark.cpp)
| CONC | E2E ms (≤) | Interact (≥) | Thr/GPU (≥) |
|---|---:|---:|---:|
| 4 | 6000 | 150 | 1350 |
| 32 | 14000 | 65 | 4500 |
| 128 | 24500 | 35 | 5300 |

Gentler at CONC=4 than DSR1 (DSR1: E2E 5000, Interact 165, Thr 1500). Your 1350 gate is easier to hit.

### Your GSM8K threshold
`BASELINE_GSM8K_METRIC=0.9325` (DSR1: 0.93). Environment override: `GSM8K_BASELINE_METRIC` + `GSM8K_TOL` (default 0.0).

### Your TP
Kimi wrapper hardcodes `tput_per_gpu = data['total_token_throughput'] / 8.0` for TP=8 native (no modification needed if you're actually running TP=8). If you're on TP=4 like DSR1 team, change to `/4.0`.

### Your submission leaderboards
- CONC=4: `daniehua-kimik25-fp4-isl8192-osl1024-conc4.hf.space`
- CONC=32: `daniehua-kimik25-fp4-isl8192-osl1024-conc32.hf.space`
- CONC=128: `daniehua-kimik25-fp4-isl8192-osl1024-conc128.hf.space`

### Recommended flow (6 steps)
1. **Skip the DSR1 chat-template debate** — your wrapper doesn't pass `--use-chat-template`. Your raw-random-token perf IS what the leaderboard measures.
2. **Side-by-side**: your team's in-tree bench vs `kimbochen_bench/benchmark_serving.py` with the exact flag set above. If they agree within 3%, you're fine — skip to step 4.
3. **If they disagree**: bisect flags (warmup count is the most likely culprit since Kimi scales it with CONC). Remove one flag at a time from kimbochen bench until numbers converge with your team's bench.
4. **Enable torch profiler** via `--torch-profiler-dir /tmp/kimi_traces` CLI flag in your server launch. Cold restart via `docker restart <container>` (NOT pkill).
5. **Capture profiled bench** matching wrapper flags (12 prompts for manageable trace). No chat template. `--profile` flag triggers server to save trace.
6. **Parse trace** via `/tmp/diff_traces.py`. Identify top kernel category. For Kimi-K2.5 you'll probably see:
   - MoE dominant (384 experts vs DSR1's 256 — bigger MoE weight)
   - MLA attention next (different nhead than DSR1 — may still hit `qh32_qseqlen4` persistent ASM kernel or non-persistent fallback depending on TP)
   - AllReduce may be larger % than DSR1 since Kimi-K2.5 is 1T params with more comms traffic

### Gotchas specific to Kimi
- Kimi-K2.5 is 1T params (vs DSR1 685B) → more weight streaming, potentially memory-bandwidth-bound in some configs
- Kimi's container (`danish_kimi`) image `rocm/atom:rocm7.1.1-ubuntu24.04-pytorch2.9-atom0.1.1-MI350x` is ROCm 7.1.1 (DSR1 team is on 7.2.2). ROCm 7.2.2 has hipBLASLt MXFP4 perf improvements + Origami GEMM selection. Worth considering an upgrade.
- Kimi wrapper uses `num_concurrent=65` for GSM8K (same as DSR1) — GSM8K runs quickly (~90s)
- Baseline `Baseline` struct comment says "max E2E ms, min interactivity, min tput per GPU" but the ordering in the struct init is `{median_e2e, median_intvty, tput_per_gpu}` — careful when parsing

Same universal methodology. Different answer likely. Report back kernel-category breakdown once you profile.

---

## Session-14 (Apr 22) lessons learned for Kimi team

### Don't tune prefill-only MoE shapes
DSR1 team ran `gemm_moe_tune.py` on `token=32768` (the one unmatched shape from boot log) — found a 2.06x kernel speedup per-call. But `token=32768` fires ONLY during PREFILL: ~40 calls in a 70s wrapper bench = ~160ms saved = **0.2% of total wall time = indistinguishable from noise.**

**Wrapper bench is dominated by DECODE** (40 requests × 1024 output tokens = 40960 decode steps, each using small-token shapes (M ≤ 64)). These are ALREADY tuned in `dsv3_fp4_tuned_fmoe.csv`. Re-tuning them via same methodology unlikely to yield more.

**Takeaway for Kimi**: enumerate actual decode-phase MoE shapes from the boot log / profile. If they're all in `kimik2_fp4_tuned_fmoe.csv`, RE.3-style retune is wasted effort. Look for OTHER levers (MLA, GEMM, AR).

### HipKittens qh32 correctness ≠ performance
DSR1 team got HK v7 kernel at qseqlen=4 bit-exact correct vs ASM (`max_abs_diff = 0.0` on 5 shape variants). But under wrapper bench it was -35% slower (880 vs 1360 thr/GPU).

**Hand-tuned ASM is hard to beat.** The persistent ASM kernel (`mla_a8w8_qh32_qseqlen4_gqaratio32_ps`) has optimized MFMA scheduling, occupancy, LDS layout that generic CK/HK kernels can't match without matching the schedule instruction-by-instruction.

**Only swap to HK when ASM can't do the shape.** At qseqlen=8 (MTP=7) ASM crashes → HK is the path. At qseqlen=4 (MTP=3) ASM exists and wins.

### aiter's /tmp/aiter_configs/ is ephemeral
`/tmp/aiter_configs/*.csv` files are REGENERATED on every `import aiter` from the merge of:
- `aiter/configs/<type>.csv` (root)
- `aiter/configs/model_configs/*<type>*.csv` (auto-globbed)

If you write to `/tmp/aiter_configs/`, it'll get wiped on next boot. For persistence, write to `aiter/configs/model_configs/<your_name>.csv`.

### aiter's auto-dedup can wipe your tuned entries
If your new CSV has duplicate keys (same M, N, K, dtype combo appears twice — e.g., once with block_m=128 and once with block_m=64), aiter's merge logic raises "duplicate shape entries" error on server boot, auto-"resolves" by keeping "best us", and may leave your file as header-only. **Submit only one row per shape key.**

### Official aiter tuner has E2E mismatch gate that rejects MXFP4
`gemm_moe_tune.py` with `--compare --update_improved` checks E2E output against torch reference. For MXFP4 with e8m0 scaling, even a correct kernel produces E2E diff > threshold → SKIP. Either widen tolerance or manually insert the winning row (accept the flag). Verified stage-level errors (err1 0.0%, err2 0.3%) are much lower than the E2E gate's internal threshold.

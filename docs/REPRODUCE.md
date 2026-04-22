# DSR1 CONC=4 — REPRODUCE (reproduction recipe)

**Last updated**: 2026-04-22 session-14 end-of-day — **RE.1 INT4 AR is current best submittable config**

## 🏆 CURRENT BEST SUBMITTABLE: RE.1 INT4 AllReduce (session-14)

**Wrapper-measured**: 1353-1365 thr/GPU avg, TPOT 6.15 ms, GSM8K 0.9424, E2E ~7200ms → **1/4 gates** (pass GSM8K only)

Gate gap: -10% thr/GPU (need 1500), -3% interact (162 vs 165), -44% E2E (7200 vs 5000).

### Reproduce RE.1 in 3 steps

**1. Snapshot image**: `rocm/atom-dev:dsr1_RE1_int4_ar_validated_apr22` (ID `e7259e3c94c1`, 474GB). Includes all aiter modules, INT4 AR envs, validated model weights.

**2. Launch script `/tmp/p0_launch_profiled.sh`** (critical env + CLI):
```bash
export HOME=/tmp HF_HOME=/tmp/.cache/huggingface HUGGINGFACE_HUB_CACHE=/tmp/.cache/huggingface/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export AITER_ENABLE_VSKIP=0 ATOM_ENABLE_RELAXED_MTP=1 ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=1024
export HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1 NCCL_MIN_NCHANNELS=16
export HIP_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1
# THE KEY RE.1 CHANGES (from pre-RE.1 FP → INT4):
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4    # aiter actually reads THIS, not VLLM_ROCM_*

exec python3 -m atom.entrypoints.openai_server \
  --model amd/DeepSeek-R1-0528-MXFP4 --server-port 8890 -tp 4 \
  --kv_cache_dtype fp8 --max-model-len 10240 --method mtp --num-speculative-tokens 3 \
  --enable-tbo prefill --max-num-batched-tokens 65536 \
  --cudagraph-capture-sizes "[1,2,4,8,16,32]"
```

**3. Bench via competition wrapper** (the ONLY submittable measurement):
```bash
export MODEL=amd/DeepSeek-R1-0528-MXFP4 PORT=8890 TP=4 CONC=4 ISL=8192 OSL=1024 NUM_PROMPTS=40
/projects/teamA/danish/repos/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x/dsr1_benchmark perf
# Only permitted modification: /8.0 → /4.0 for TP=4 in the wrapper's post-process
```

### Session-14 honest rollup (all experiments tried)

| Lever | Env/files changed | Wrapper bench | Verdict |
|---|---|---:|---|
| **RE.1 INT4 AR** | `VLLM_ROCM_/AITER_QUICK_REDUCE_QUANTIZATION=INT4` | **1360 avg** | ✅ **KEEP** (+7.4%, GSM8K held) |
| RE.2 BF16 naive tuner | custom `hipb_findallsols` tuner, 47 shapes | n/a | ❌ crashes GPU (persistent), 0 effect (/tmp) |
| RE.3 MoE tune prefill | aiter `gemm_moe_tune.py` at token=32768 | 1361 avg | ❌ neutral (prefill-only, 0.2% bench impact) |
| RE.4a HK qh32 sq=4 | `AITER_ENABLE_HK_QH32=1 AITER_ENABLE_EXPERIMENTAL=1` | 879 avg | ❌ correct but -35% (ASM faster) |
| RE.4b HK qh32 sq=8 | metadata crash before kernel even runs | n/a | ⏳ blocked → RE.4c (multi-day) |

**Only RE.1 sticks.** All other work is documented in `RE4_hk_qh32/`, `phase_re_artifacts/`, and git commits 8483c0b, b064caf, 178237a.

### Path to 4/4 gates (next session)

**RE.4c — HK qh32 at qseqlen=8 for MTP=7 unlock** (multi-day, 3-5 days):
1. Fix `get_mla_metadata_v1` to produce valid work_info at `nhead=32 + qseqlen=8`
2. Verify HK v7 kernel handles sq=8 (may have qseqlen=4 bakes-in)
3. Expected: +15-20% TPOT (MTP=7 = 3.5 tokens/step vs MTP=3 = 2.1 tokens/step)
4. Stacked with RE.1 INT4 AR: **1570-1700 thr/GPU → clears 1500 gate with margin**

ASM persistent kernel CRASHES at sq=8 (fold invariant break). HK is the ONLY path. v7 already compiles + passes correctness at sq=4. Metadata unblock is the critical work.

---

## 🚨 HISTORICAL NOTE: direct-bench "gold 3/4 gates" was non-submittable

Yesterday's "3/4 gates at 1500+ thr/GPU" numbers used `python -m atom.benchmarks.benchmark_serving` WITHOUT `--use-chat-template`. The competition leaderboard's `dsr1_benchmark` uses kimbochen's bench WITH chat template, which activates DSR1 reasoning mode (`<think>...</think>`) — ~14% slower per token.

Side-by-side proof (same warm server, 40 prompts, --ignore-eos):
- ATOM bench, no chat template: 1514 thr/GPU, 5.42 ms
- Kimbochen bench + chat-template (wrapper): **1308 thr/GPU, 6.32 ms**
- Kimbochen bench, no chat template: 1477 thr/GPU, 5.47 ms

The gap = ~11% reasoning-mode + ~2.4% tool difference.

**Only wrapper numbers are submittable.** Direct-bench is for component-level debugging only.

---

# DSR1 CONC=4 — Current Best: **3/4 GATES at P0 clean floor (session-10 Apr 20)** ✅ LOCKED

**Last updated**: 2026-04-20 session-10 (P0 crossed 3 gates + container committed to gold image)

## 🔒 GOLD IMAGE LOCKED

Container committed: **`rocm/atom-dev:dsr1_P0_3of4_gates_apr20`** (container ID verified on `mia1-p02-g55`, image size 45 GB)

Any future experimentation MUST happen on a CLONE via `docker run --name <new_name> rocm/atom-dev:dsr1_P0_3of4_gates_apr20 ...` — never modify this gold image.

## P0 REVERIFY min-of-3 (post-recovery, proves gold image reproduces)

| Metric | Run 1 | Run 2 | Run 3 | min-of-3 | Gate | Status |
|---|---:|---:|---:|---:|---:|---|
| Thr/GPU | 1578.71 | 1554.01 | 1573.67 | **1554.01** | ≥1500 | ✅ PASS |
| Interactivity | 192.3 | 190.5 | 188.3 | **188.3** | ≥165 | ✅ PASS |
| Median TPOT | 5.20 | 5.25 | 5.31 | 5.25 | — | — |
| GSM8K flex | 0.9318 | (1 run) | — | — | ≥0.93 | ✅ PASS |

**Gold standard: 1554/5.25/188/0.9318 → 3/4 gates (E2E ~5700 ms remaining gap)**

---



---

# 🎯 P0 CLEAN FLOOR (Apr 20 session-10) — 3/4 GATES

**This REPLACES the prior 1/4 canonical floor. The old floor was measured with suboptimal cudagraph_capture_sizes default (captured 33 unused graph variants at [1,2,4,...,512]).**

**Model**: `amd/DeepSeek-R1-0528-MXFP4` (HuggingFace canonical, NO merged checkpoint)

| Metric | min-of-3 | best run | Gate | Status | vs Prior Floor |
|---|---:|---:|---:|---|---|
| **Thr/GPU (÷4)** | **1500.11** | 1623.68 (run 3) | ≥1500 | ✅ **PASS** | +11% (1351→1500) |
| **Interactivity** | **185.04** | 192.54 (run 3) | ≥165 | ✅ **PASS** | +23% (150→185) |
| **Median TPOT** | 5.40 ms | 5.19 ms (run 3) | — | (derived) | −19% (6.66→5.40) |
| **Median E2E** (run 1) | 5762.86 ms | — | ≤5000 | ❌ **FAIL** | −20% (7221→5763), 763 ms over |
| **GSM8K flex-extract** | **0.9318** | — | ≥0.93 | ✅ **PASS** | equivalent to 0.934 (variance) |
| **GATES** | — | — | 4/4 | **3/4** ✅✅✅ | +2 gates from 1/4 |

**Workload**: ISL=8192, OSL=1024, CONC=4, num_prompts=40 (InferenceX-matching harness)

**Result JSONs**:
- `dsr_beta/bench_results/P0_clean_floor.json` (summary)
- Container: `/tmp/P0_run{1,2,3}.json`

**Gate math for remaining E2E**:
- Current E2E 5763, need ≤5000 = cut 763 ms (−13%)
- P2 shared-expert fusion expected: −15 to −135 ms
- P7 MTP=4 (qseqlen=5, needs HK kernel): expected −1200 ms → cracks E2E

---

## P0 Recipe (REPRODUCIBLE, crosses 3/4 gates)

### Stack
- Container: `danish_atom_dsr_beta` (image: `rocm/atom-dev:latest`)
- ROCm 7.2.2, PyTorch 2.10.0+rocm7.2.2.lw.git40d237bf
- aiter commit `73ad0023e15e9735b3af95b3357b99cf7f801bf1` (main)
- ATOM commit `f8453e3fc0f65191fb2034602dc9a2066a78020b` (main)
- flydsl 0.1.3.1, triton 3.5.1
- TP=4 single-replica, GPUs 0-3 (GPUs 4-7 hold Kimi container)

### Required local patches
1. `rejection_sampler.py`: `RELAXED_TOP_N=8, RELAXED_DELTA=0.5` (ATOM_ENABLE_RELAXED_MTP=1 selects)
2. `attention_mla.py`: `num_kv_splits=None` (was 16)
3. Phase 3 sync-fuse — `model_runner.py`: merge `send_mtp_status_to_cpu_async` rejected+bonus tensors

### Launch command (P0 clean floor — 3/4 gates)

```bash
~/bin/docker exec -d \
  -e HOME=/tmp \
  -e HF_HOME=/tmp/.cache/huggingface \
  -e HUGGINGFACE_HUB_CACHE=/tmp/.cache/huggingface/hub \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e AITER_ENABLE_VSKIP=0 \
  -e ATOM_ENABLE_RELAXED_MTP=1 \
  -e ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=1024 \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e HSA_NO_SCRATCH_RECLAIM=1 \
  -e NCCL_MIN_NCHANNELS=16 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e OMP_NUM_THREADS=1 \
  -e VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=FP \
  -e VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=1 \
  danish_atom_dsr_beta bash -c '
    python3 -m atom.entrypoints.openai_server \
      --model amd/DeepSeek-R1-0528-MXFP4 \
      --server-port 8890 \
      -tp 4 \
      --kv_cache_dtype fp8 \
      --max-model-len 10240 \
      --method mtp \
      --num-speculative-tokens 3 \
      --enable-tbo prefill \
      --max-num-batched-tokens 65536 \
      --cudagraph-capture-sizes "[1,2,4,8,16,32]" \
      > /tmp/p0_boot.log 2>&1
  '
```

**THE KEY CHANGE vs prior floor**: added `--cudagraph-capture-sizes "[1,2,4,8,16,32]"` as the LAST line of the Python command.

Wait ~10-12 min for cold boot. Verify boot success:
```bash
grep "max_q_len=4" /tmp/p0_boot.log                             # Should show 6 captures (bs=1,2,4,8,16,32)
curl http://localhost:8890/health                               # {"status":"ok"}
ps -eo pid,stat,cmd | grep -c multiprocessing-fork              # Should be 4 workers + 1 resource tracker
```

### Bench command (perf only, writes to /tmp/P0_run{N}.json)

```bash
~/bin/docker exec danish_atom_dsr_beta bash -c '
  export HOME=/tmp HF_HOME=/tmp/.cache/huggingface HUGGINGFACE_HUB_CACHE=/tmp/.cache/huggingface/hub
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  cd /app/ATOM
  python3 -m atom.benchmarks.benchmark_serving \
    --model amd/DeepSeek-R1-0528-MXFP4 --port 8890 \
    --dataset-name random --random-input-len 8192 --random-output-len 1024 \
    --num-prompts 40 --max-concurrency 4 --trust-remote-code \
    --save-result --save-detailed --result-filename /tmp/P0_run1.json
'
```

Repeat 3× for min-of-3. Take the minimum Thr/GPU.

### GSM8K (separate run)

```bash
~/bin/docker exec danish_atom_dsr_beta bash -c '
  export HOME=/tmp HF_HOME=/tmp/.cache/huggingface HUGGINGFACE_HUB_CACHE=/tmp/.cache/huggingface/hub
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  lm_eval --model local-completions \
    --model_args model=amd/DeepSeek-R1-0528-MXFP4,base_url=http://0.0.0.0:8890/v1/completions,num_concurrent=65,max_retries=1,tokenized_requests=False \
    --tasks gsm8k --num_fewshot 3
'
```

Look for `gsm8k|...|flexible-extract|...|exact_match|↑|0.9318|±|0.0069|` — gate is ≥0.93 on flexible-extract.

---

## Why `--cudagraph-capture-sizes [1,2,4,8,16,32]` is the big unlock

Default ATOM behavior (when flag not set):
- `cuda_graph_sizes = [512]` (single value)
- Auto-expansion at `model_runner.py:1908-1918`: `[1,2,4,8] + [16,32,...,512 in steps of 16]` = **33 graph variants captured**

At CONC=4 we only hit bs=1,2,4 in steady state. Having 33 graph variants cached:
- Bloats `self.graphs` dispatch dict (lookup latency)
- Consumes device memory for unused graph structures
- Possibly forces graph instantiation/compilation work that hits HIP runtime caches

Cutting to 6 variants `[1,2,4,8,16,32]` — only sizes we actually use — reduces these overheads and gives a measured **−19% TPOT** / **+11% Thr/GPU** / **+23% Interactivity** uplift with zero code changes.

This is pure engine hygiene. The previous 1351/6.66/150/7221/0.934 floor was leaving this on the table.

---

## Historical floor lineage (now superseded)
- DEC-073 floor (merged model): `1270/6.80/147.1/7318/0.934` (Apr 18)
- 1361 floor (merged, session-8): `1361/6.35/157.55/6842/0.934` (Apr 18 evening)
- Apr 20 stock canonical (prior): `1351/6.66/150.23/7221/0.934` (replaced merged per mergability)
- **Apr 20 session-10 P0 (CURRENT)**: `1500/5.40/185/5763/0.9318` ← 3/4 GATES ← **USE THIS**

---

# 🔬 Historical: Older recipe (1/4 gates, superseded by P0)

**Model**: `amd/DeepSeek-R1-0528-MXFP4` (HuggingFace canonical, NO merged checkpoint)

| Metric | Value | Gate | Status |
|---|---|---|---|
| **Thr/GPU (÷4)** | **1351** | ≥1500 | ❌ −9.9% |
| Thr/GPU (÷8 result.json field) | 675.49 | — | reference |
| Total throughput | 5403.96 tok/s | — | — |
| **Median TPOT** | **6.66 ms** | — | (need ≤4.52 for E2E gate) |
| Mean TPOT | 6.21 ms | — | — |
| P99 TPOT | 7.85 ms | — | — |
| **Median TTFT** | **370.15 ms** | — | — |
| P99 TTFT | 1445.91 ms | — | — |
| Median ITL | 16.23 ms | — | — |
| **Median E2E** | **7221.33 ms** | ≤5000 | ❌ +44% |
| P99 E2E | 8956.88 ms | — | — |
| **Interactivity** | **150.23 tok/s/user** | ≥165 | ❌ −9.0% |
| **GSM8K** | **0.934** | ≥0.93 | ✅ PASS |
| **GATES** | **1/4** | 4/4 | GSM8K only |

**Workload**: ISL=8192, OSL=1024, CONC=4, num_prompts=40

**Result file**: `/projects/teamA/danish/experiments/stock_floor_MTP3_TBO_QR_canonical.json`

## Stock floor recipe (REPRODUCIBLE)

### Stack
- Container: `danish_atom_dsr_beta` (rocm/atom-dev sha256:52c5195a712b5d3a)
- ROCm 7.2.2, PyTorch 2.10.0+rocm7.2.2.git40d237bf
- aiter commit `73ad0023e15e9735b3af95b3357b99cf7f801bf1` (main)
- ATOM commit `f8453e3fc0f65191fb2034602dc9a2066a78020b` (main)
- flydsl 0.1.3.1, triton 3.5.1
- TP=4 single-replica, GPUs 0-3 (GPUs 4-7 hold Kimi container)

### Required local patches
1. `rejection_sampler.py`: `RELAXED_TOP_N=8, RELAXED_DELTA=0.5` (ATOM_ENABLE_RELAXED_MTP=1 selects)
2. `attention_mla.py`: `num_kv_splits=None` (was 16)
3. Phase 3 sync-fuse — `model_runner.py`: merge `send_mtp_status_to_cpu_async` rejected+bonus tensors

### Launch command (canonical stock floor)

```bash
~/bin/docker exec -d \
  -e HOME=/tmp \
  -e AITER_ENABLE_VSKIP=0 \
  -e ATOM_ENABLE_RELAXED_MTP=1 \
  -e ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=1024 \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e NCCL_MIN_NCHANNELS=16 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e OMP_NUM_THREADS=1 \
  -e VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=FP \
  -e VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=1 \
  danish_atom_dsr_beta bash -c '
    python3 -m atom.entrypoints.openai_server \
      --model amd/DeepSeek-R1-0528-MXFP4 \
      --server-port 8890 \
      -tp 4 \
      --kv_cache_dtype fp8 \
      --max-model-len 10240 \
      --method mtp \
      --num-speculative-tokens 3 \
      --enable-tbo prefill \
      --max-num-batched-tokens 65536 \
      > /tmp/atom-stock-floor.log 2>&1
  '
```

Wait ~12-15 min for cold boot. Verify boot success:
```bash
grep "max_q_len=4" /tmp/atom-stock-floor.log    # MTP=3 captures present
curl http://localhost:8890/health                # {"status":"ok"}
```

### Bench command
```bash
~/bin/docker exec danish_atom_dsr_beta bash -c '
  cd /projects/teamA/danish/repos/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x
  MODEL=amd/DeepSeek-R1-0528-MXFP4 PORT=8890 ./dsr1_benchmark perf
'
```

### Why merged model was dropped
- Mergability concern: AMD reference benchmark (InferenceX) uses canonical `amd/DeepSeek-R1-0528-MXFP4`, not transplanted variants
- Empirical: merge benefit measured at +0.7% throughput vs stock (within variance)
- Reproducibility: stock model is single canonical artifact; merged required custom transplant recipe

---

# 🔬 C1 HK qh32 kernel port — v6 in flight on STOCK model (E-08-06 series)

**Status as of 2026-04-20 07:15 UTC**: HK kernel port closes [ROCm/aiter Issue #1468](https://github.com/ROCm/aiter/issues/1468) (open since Nov 2025, no AMD progress) — additive opt-in patch via `AITER_ENABLE_HK_QH32=1` env, max mergability.

| Iter | Fix | Result |
|---|---|---|
| v1 | Virtual-warp at Q+K+V | Compiles+boots, MTP=3 active, GARBAGE (Q overflows kNumTilesM=2) |
| v2 | Q+K reverted, V kept | GARBAGE (K 2-warp LDS vs V 8-vwarp mismatch) |
| v3 | Outer K virt-warp re-applied | GARBAGE (inner K still real-warp) |
| v4 | Inner K full-tile virt-warp | GARBAGE (LDS still wrong shape) |
| **v5** | `kNumRowsPerSubBlock = 4` (constant) in KvManagerV2 | **PARTIAL COHERENCE**: qseqlen=1 PERFECT R1 reasoning, qseqlen=4 still garbage |
| **v6** | `s_barrier` between work_idx iterations | **TESTING** (next on stock) |
| v7 | Per-iter barrier inside V virt-warp loop | If v6 garbage |

**Patch files** (active on server):
- `/app/aiter-test/csrc/kernels/mla/hk/mi3xx_v32_fwd_decode_h32_fp8_fp8.cuh` (836 lines, 3 v4 markers + 1 v6 marker)
- `/app/aiter-test/csrc/kernels/mla/hk/hk_mla_buffer_managers.cuh` (1 v5 marker @ line 794)
- `/app/aiter-test/csrc/kernels/mla/hk_decode_fwd.cu` (num_head==32 branch added)
- `/app/aiter-test/aiter/jit/optCompilerConfig.json` (h32 src in module_hk_mla)
- `/app/aiter-test/aiter/mla.py` (use_hk gated on AITER_ENABLE_HK_QH32)
- `/app/ATOM/atom/config.py` line 882 (MTP cap 4→8)

All `.pre_v*` backups preserved. Patches are env-gated/additive — default behavior unchanged when env unset.

## v5+nospec proof (HK kernel correctness at qseqlen=1)

```
Test prompt: "What is 2+2?"

Output (3 runs, all coherent):
Run 1: "Okay, the user asked "What is 2+2?" That's pretty straightforward. 
        Let me think... This is basic arithmetic, so the answer should be 4..."
Run 2: "Okay, the user asked "What is 2+2?" This seems like a very basic 
        math question..."
Run 3: "Okay, the user asked "What is 2+2?" That seems incredibly basic..."
```

TPOT_s=0.0073 (7.3 ms). All real R1 reasoning. **HK kernel is structurally correct** — bug isolated to qseqlen=4 (MTP-3 spec verification) path which v6+ targets.

---

# 🛠️ HK kernel boot recipe (when v6+ produces coherent qseqlen=4 output)

```bash
# Step 1: Container restart to clear VRAM zombies (REQUIRED before every reboot)
~/bin/docker restart danish_atom_dsr_beta

# Step 2: Wipe stale module_hk_mla.so to force JIT rebuild
~/bin/docker exec danish_atom_dsr_beta bash -c '
  find / -name "*module_hk_mla*" 2>/dev/null | xargs rm -rf 2>/dev/null
'

# Step 3: Launch with HK_QH32 env added to canonical stock config above
~/bin/docker exec -d \
  -e HOME=/tmp \
  -e AITER_ENABLE_VSKIP=0 \
  -e AITER_ENABLE_EXPERIMENTAL=1 \
  -e AITER_ENABLE_HK_QH32=1 \
  -e ATOM_ENABLE_RELAXED_MTP=1 \
  -e ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=1024 \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e NCCL_MIN_NCHANNELS=16 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e OMP_NUM_THREADS=1 \
  -e VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=FP \
  -e VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=1 \
  danish_atom_dsr_beta bash -c '
    python3 -m atom.entrypoints.openai_server \
      --model amd/DeepSeek-R1-0528-MXFP4 \
      --server-port 8890 \
      -tp 4 \
      --kv_cache_dtype fp8 \
      --max-model-len 10240 \
      --method mtp \
      --num-speculative-tokens 3 \
      --enable-tbo prefill \
      --max-num-batched-tokens 65536 \
      > /tmp/atom-stock-hk-vN.log 2>&1
  '

# Step 4: Coherence check (file-based JSON to avoid shell quoting issues)
~/bin/docker exec danish_atom_dsr_beta bash -c '
  cat > /tmp/req.json <<EOF
{"model":"amd/DeepSeek-R1-0528-MXFP4","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50,"temperature":0}
EOF
  curl -s http://localhost:8890/v1/chat/completions -H "Content-Type: application/json" --data-binary @/tmp/req.json
'

# Step 5: Bench (only if coherent)
~/bin/docker exec danish_atom_dsr_beta bash -c '
  cd /projects/teamA/danish/repos/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x
  MODEL=amd/DeepSeek-R1-0528-MXFP4 PORT=8890 ./dsr1_benchmark perf
'
```

## MTP=4/MTP=5 extension recipe (after MTP=3 HK proven coherent + bench parity)

Change `--num-speculative-tokens 3` → `4` (or `5`). All other env/flags unchanged. The HK kernel handles qseqlen up to 8 via work_info decomposition (config.py:882 cap already lifted to 8).

---



**The real path to 4/4.** Custom HipKittens MLA kernel for qh32 unblocks MTP=4+ which is the only approach with positive-math gate projection.

| Iteration | Fix | Result |
|---|---|---|
| v1 | Virtual-warp loops at Q+K+V | Compiles, boots, MTP=3 active, garbage output. Q load overflows kNumTilesM=2 buffer dim |
| v2 | Reverted Q+K loops; kept V loop | Boots OK, still garbage. K fills 2-warp LDS, V reads 8-virtual-warp slots = uninitialized |
| **v3** | Virtual-warp loop on K too — both K fill and V use consistent 8-slot layout | **IN JIT REBUILD / BOOT (08:40 UTC wakeup for check)** |
| v4 (planned) | Override kSzLdsKv to 8-warp size if v3 still garbage | pending |
| v5 (last resort) | Native 2-warp buffer manager rewrite (400-600 LOC) | pending |

Kernel active at `/app/aiter-test/csrc/kernels/mla/hk/mi3xx_v32_fwd_decode_h32_fp8_fp8.cuh` = 795 lines. `.pre_c1` backups preserve proven h128 path for instant rollback. Full iteration detail in `STATUS.md` and `HISTORY.md`.

Once v3 (or vN) produces coherent output and bit-matches asm baseline at qseqlen=4, extend kernel to qseqlen=5 (MTP=4) → target 3/4 gates; then qseqlen=6 (MTP=5) → target 4/4.

---

---

# ⚠️ E-08-05 CONFIG — 2/4 gates ON 1-OF-3 RUNS (not submittable)

**Stability test result** (3 back-to-back identical-config runs):

| Run | Interactivity | Pass 165 gate? |
|---|---|---|
| E-08-05 (initial) | 165.35 | ✅ |
| E-08-05b (repeat) | 159.87 | ❌ −3.1% |
| E-08-05c (repeat) | 150.23 | ❌ −9.0% |
| **min-of-3** | **150.23** | ❌ |

**Min-of-3 FAILS gate by 9%.** E-08-05 is a "lucky run" config that bounces around the 165 gate with ~3% run-to-run variance. NOT submittable for 2/4 claim.

**Submission path**: need structural TPOT margin (not just getting close to gate). C1 HK kernel port for MTP=4 is the committed path (multi-day engineering).

**Real current committable floor** = 1/4 gates (GSM8K only), same as historical state.

---

# 🎯 ASPIRATIONAL CONFIG E-08-05 — 2/4 gates (first-ever at TP=4 SR CONC=4)

| Metric | Value | Gate | Status |
|---|---|---|---|
| **Thr/GPU (÷4)** | **1304.35** | ≥1500 | ❌ −13% |
| Thr/GPU (÷8 in result.json `tput_per_gpu` field) | 652.18 | — | reference |
| Total thr (tok/s) | 5217.40 | — | — |
| **Median TPOT** | **6.05 ms** | — | — |
| Mean TPOT | 6.27 ms | — | — |
| P99 TPOT | 8.70 ms | — | — |
| **Median TTFT** | **370.69 ms** | — | — |
| P99 TTFT | 1440.80 ms | — | — |
| Median ITL | 16.24 ms | — | — |
| **Median E2E** | **6591.96 ms** | ≤5000 | ❌ +32% |
| P99 E2E | 9422 ms | — | — |
| **Interactivity** | **165.35** | **≥165** | ✅ **PASS** |
| **GSM8K** | **0.9333** | ≥0.93 | ✅ PASS |
| **Gates** | **🎯 2/4** | 4/4 | GSM8K + Interactivity |

**Artifact JSON**: `/projects/teamA/danish/experiments/E-08-05_NEW_RECORD_2of4_merged_MTP3_TBO_CSV_QR_65536.json`

## Full reproduction recipe

### 1. Stack (DSR_beta container)

| Component | Value |
|---|---|
| Docker image | `rocm/atom-dev@sha256:52c5195a712b5d3a0993d5e63de9b8ffc13a77d0c4b2f31d40afe9e62c12ab5f` (or newer — ROCm 7.2.2 + latest aiter/ATOM/flydsl) |
| ROCm | 7.2.2 |
| PyTorch | 2.10.0+rocm7.2.2.git40d237bf |
| aiter | HEAD (main branch at time of run) |
| ATOM | HEAD (main branch at time of run) |
| flydsl | 0.1.3.1 |
| triton | 3.5.1 |
| Container | `danish_atom_dsr_beta` port 8890 |
| GPUs | 0, 1, 2, 3 (TP=4 single replica) |

### 2. Model (CRITICAL — merged DEC-075 checkpoint, NOT stock HF)

```
MODEL = /projects/teamA/danish/models_merged/DSR1-drafter-FP4
```

This is the DEC-075 merged checkpoint (layer 61 MoE transplanted to FP4 from `amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4` variant, while body layers 0-60 remain from stock `amd/DeepSeek-R1-0528-MXFP4`). Gives 20× drafter fast path without the Triton trap.

Build: `scripts/merge_dec075_v5.py` (~5s, mostly symlinks + 2 cleaned shards).

### 3. Required ATOM code patches (verified in place before bench)

```
/app/ATOM/atom/model_ops/rejection_sampler.py:12-13
    RELAXED_TOP_N = 8
    RELAXED_DELTA = 0.5

/app/ATOM/atom/model_ops/attention_mla.py:596
    num_kv_splits=None  # was 16

/app/ATOM/atom/model_engine/model_runner.py:139-148
    # Phase 3 sync-fuse: torch.stack(num_rejected, num_bonus) into merged 2-row tensor
    # Cuts 2 async D2H copies + 2 syncs to 1. ~1.6ms/step at MTP=3.
```

Verify applied:
```bash
grep -n "RELAXED_TOP_N = 8" /app/ATOM/atom/model_ops/rejection_sampler.py
grep -n "num_kv_splits=None" /app/ATOM/atom/model_ops/attention_mla.py
grep -n "Phase 3 patch" /app/ATOM/atom/model_engine/model_runner.py
```

### 4. BF16 CSV (critical — filtered version, 53 rows)

The full 97-row tuned CSV from DEC-071 has 42 `hipblaslt` rows with solidx values that DON'T round-trip to current hipBLASLt version → HIPBLAS_STATUS_INTERNAL_ERROR at runtime → server hangs at init. **Must filter out `hipblaslt` libtype rows, keep only `flydsl`, `asm`, `triton`.**

Source copy on server: `/projects/teamA/danish/repos/aiter/aiter/configs/model_configs/dsv3_bf16_tuned_gemm.csv` (34473 bytes, 97 rows, older schema without `gfx` column).

Steps to install correctly:

```bash
# Step 1: copy source to active path
~/bin/docker exec danish_atom_dsr_beta cp \
  /projects/teamA/danish/repos/aiter/aiter/configs/model_configs/dsv3_bf16_tuned_gemm.csv \
  /app/aiter-test/aiter/configs/model_configs/dsv3_bf16_tuned_gemm.csv

# Step 2: add gfx column (schema compatibility with current aiter master CSV)
~/bin/docker exec danish_atom_dsr_beta python3 -c '
import sys
p = "/app/aiter-test/aiter/configs/model_configs/dsv3_bf16_tuned_gemm.csv"
lines = open(p).readlines()
out = ["gfx," + lines[0]] + ["gfx950," + ln for ln in lines[1:]]
open(p, "w").writelines(out)
'

# Step 3: filter out hipblaslt rows
~/bin/docker exec danish_atom_dsr_beta python3 -c '
import csv
p = "/app/aiter-test/aiter/configs/model_configs/dsv3_bf16_tuned_gemm.csv"
r = csv.DictReader(open(p))
fn = r.fieldnames
rows = [row for row in r if row.get("libtype") != "hipblaslt"]
with open(p, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fn); w.writeheader(); w.writerows(rows)
print(f"kept {len(rows)} non-hipblaslt rows")
'

# Step 4: on first server boot, aiter will auto-dedup duplicate shape entries
# (keeps best-performing per shape, writes back). Final CSV = 53 rows.
```

### 5. Required environment variables

```bash
# MANDATORY for boot stability + performance
export HOME=/tmp                                      # overlay-FS workaround (/root/.aiter read-only)
export AITER_ENABLE_VSKIP=0                            # prevents MoE aperture faults (AMD Issue #1143)
export ATOM_ENABLE_RELAXED_MTP=1                       # enable relaxed MTP accept
export ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=1024       # dual-stream MoE overlap
export HIP_FORCE_DEV_KERNARG=1                         # device-side kernel args
export NCCL_MIN_NCHANNELS=16                           # all-reduce channels
export HIP_VISIBLE_DEVICES=0,1,2,3                     # TP=4 on GPUs 0-3 only (Kimi on 4-7)
export OMP_NUM_THREADS=1
export AMDGCN_USE_BUFFER_OPS=1

# NEW in E-08-05 (drove the 2/4 breakthrough)
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=FP          # quantized AllReduce
export VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=1      # companion to above
```

### 6. Launch command (direct python3 — CRITICAL, DON'T use `launch_atom_server.sh`)

**⚠️ GOTCHA**: `launch_atom_server.sh` has a FIXED-TEMPLATE ATOM_CMD that SILENTLY IGNORES extra flags like `--num-speculative-tokens 3` and `--enable-tbo prefill`. Must call `python3 -m atom.entrypoints.openai_server` directly.

```bash
~/bin/docker exec -d \
  -e HOME=/tmp \
  -e AITER_ENABLE_VSKIP=0 \
  -e ATOM_ENABLE_RELAXED_MTP=1 \
  -e ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=1024 \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e NCCL_MIN_NCHANNELS=16 \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e OMP_NUM_THREADS=1 \
  -e AMDGCN_USE_BUFFER_OPS=1 \
  -e VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=FP \
  -e VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=1 \
  danish_atom_dsr_beta bash -c '
    python3 -m atom.entrypoints.openai_server \
      --model /projects/teamA/danish/models_merged/DSR1-drafter-FP4 \
      --server-port 8890 \
      -tp 4 \
      --kv_cache_dtype fp8 \
      --max-model-len 10240 \
      --method mtp \
      --num-speculative-tokens 3 \
      --enable-tbo prefill \
      --max-num-batched-tokens 65536 \
      > /tmp/atom-server.log 2>&1
  '
```

Expected cold boot: **~10-12 min** (JIT compile all kernels + safetensors load + capture at mtp_k=3).

### 7. Boot verification markers (MUST check before benching)

```bash
# (a) Health endpoint OK
curl -s http://localhost:8890/health
# Expected: {"status":"ok"}

# (b) Engine config ACTUALLY used num_spec_tokens=3 (not 1)
grep -E "num_spec_tokens=3.*enable_tbo.*True.*max_num_batched_tokens.*65536" /tmp/atom-server.log
# Expected: match on engine kwargs line. Critical — if num_spec_tokens=1 you're running MTP=1 silently.

# (c) Capture phase ran at max_q_len=4 (confirms MTP=3 active)
grep -oE "max_q_len=[0-9]+" /tmp/atom-server.log | sort -u
# Expected: "max_q_len=4" should appear. If only "max_q_len=2" appears, MTP collapsed to MTP-1.

# (d) Drafter FP4 fast path kernel loaded
grep "flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w3_fq" /tmp/atom-server.log | head
# Expected: at least one match during bs=4 inference (confirms merged FP4 drafter active).
```

### 8. Benchmark command

```bash
~/bin/docker exec \
  -e HOME=/tmp \
  -e HF_HOME=/tmp/.cache/huggingface \
  -e MODEL=/projects/teamA/danish/models_merged/DSR1-drafter-FP4 \
  -e PORT=8890 \
  -e HOST=localhost \
  -e ISL=8192 \
  -e OSL=1024 \
  -e CONC=4 \
  -e TP=4 \
  danish_atom_dsr_beta bash -c '
    cd /projects/teamA/danish/repos/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x && \
    ./dsr1_benchmark perf
  '
```

Runs ~3 min total (GSM8K validation + 40-prompt perf run at CONC=4 ISL=8192 OSL=1024).

### 9. Expected result

```
Interactivity: 165+ tokens/s/user (min required: 165) ✅
E2E (median):  6591 ms (max allowed: 5000) ❌
Throughput:    652 tokens/s/GPU (min required: 1500) ❌ [harness ÷8 convention]
GSM8K:         0.9333 (min required: 0.93) ✅

GATES: 2/4 PASSING (Interactivity + GSM8K)
```

JSON output written to `/projects/teamA/danish/repos/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x/result.json`. Save copy to `/projects/teamA/danish/experiments/` for record preservation.

### 10. Known failure modes + fixes

| Failure | Symptom | Fix |
|---|---|---|
| Launch script silent MTP=1 | engine kwargs shows `num_spec_tokens=1` when you passed 3 | Don't use `launch_atom_server.sh` — call python3 directly |
| CSV schema mismatch | `ValueError: Column mismatch...bf16_tuned_gemm.csv` | Add `gfx` column as first field with value `gfx950` on every row |
| CSV duplicate shapes | `RuntimeError: Found 16 duplicate shape entries...Please re-run.` | Just re-run — aiter auto-dedups + saves |
| hipblaslt solidx broken | `hipBLAS error: HIPBLAS_STATUS_INTERNAL_ERROR at hipbsolgemm.cu:1231` + server hang | Filter out `libtype == "hipblaslt"` rows from CSV |
| JIT cache permission denied | `PermissionError at /root/.aiter` or `/root/.cache/huggingface` | Set `HOME=/tmp` (and `HF_HOME=/tmp/.cache/huggingface` for lm_eval) |
| MoE aperture fault | `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` in fused MoE kernel | Set `AITER_ENABLE_VSKIP=0` (AMD Issue #1143) |
| Zombie VRAM after crash | GPUs show ~282 GB used but no procs | `docker restart danish_atom_dsr_beta` clears |
| pgrep shows 0 workers but server healthy | misleading — ATOM worker cmdlines don't match "openai_server" | Don't trust pgrep alone; test with curl /health or actual request |

### 11. Session-8 journey (how we got here)

Starting point: historical best 1361/6.35/157/6842/0.934 → **1/4 gates** (GSM8K only). Interactivity 157 failed 165 gate by 5%.

Session-8 progression (all with MTP=3 + TBO + merged model):
- E-08-03 no CSV, no QUICK_REDUCE, default batched-tokens → 1317/6.64/150/7140/0.9371 → **1/4**
- E-08-04 same but stock model → 1251/6.88/145/7378/0.9333 → 1/4 (merge contribution +5%)
- E-08-05 **+filtered CSV +QUICK_REDUCE +batched-tokens=65536** → 1304/**6.05**/**165.35**/6592/0.9333 → **🎯 2/4**

Key finding: TPOT dropped 9% with these three env/config additions stacked. Interact pushed from 150→165 which just crosses the gate.

### 12. Stability note

E-08-05 interactivity margin is razor-thin (165.35 vs 165 gate = +0.2% margin). Should run 2-3 more benches to confirm the gate stays passed. Run-to-run variance on this platform is typically ±2% on TPOT which could dip below 6.06 threshold and fail gate in some runs.

---

# 🏆 Historical reference: DEC-075 + DSR_beta (1361/6.35) — former best, 1/4 gates

**Measured 2026-04-18 12:45 UTC before session-8 additions.**

## CURRENT BEST FLOOR — locked + reproducible

| Metric | Value | Gate | Status |
|---|---|---|---|
| **Thr/GPU (÷4)** | **1361** | ≥1500 | ❌ −9% |
| **Median TPOT** | **6.35 ms** | — | (TPOT gate 4.52 for E2E) |
| Mean TPOT | 6.10 ms | — | — |
| **Median ITL** | 16.29 ms | — | — |
| **Interactivity** | **157.55** | ≥165 | ❌ −5% (narrowed from −10% at DEC-075) |
| **Median E2E** | **6842 ms** | ≤5000 | ❌ +37% |
| **GSM8K** | **0.934** | ≥0.93 | ✅ |
| **Gates** | **1/4** | 4/4 | GSM8K only |

**Full JSON**: `dsr_beta/bench_results/CURRENT_BEST_1361_6p35.json`

## Current best config (DSR_beta stack)

| Component | Value |
|---|---|
| Docker image | `rocm/atom-dev@sha256:52c5195a712b5d3a0993d5e63de9b8ffc13a77d0c4b2f31d40afe9e62c12ab5f` |
| ROCm | 7.2.2 |
| PyTorch | 2.10.0+rocm7.2.2.git40d237bf |
| aiter | commit `73ad0023e15e9735b3af95b3357b99cf7f801bf1` (main) |
| ATOM | commit `f8453e3fc0f65191fb2034602dc9a2066a78020b` (main) |
| flydsl | 0.1.3.1 |
| triton | 3.5.1 |
| Container | `danish_atom_dsr_beta` port 8890 |
| Model | `/projects/teamA/danish/models_merged/DSR1-drafter-FP4` (DEC-075 merged checkpoint) |

## Required local patches (3)

1. `rejection_sampler.py`: `RELAXED_TOP_N = 8`, `RELAXED_DELTA = 0.5` (was 10, 0.6)
2. `attention_mla.py`: `num_kv_splits=None` (was 16)
3. **Phase 3 sync-fuse** — `model_runner.py`: merge `send_mtp_status_to_cpu_async` rejected+bonus tensors into single stacked tensor. Patch script: `dsr_beta/scripts/phase3_patch.py`

## Required env vars + flags

```bash
export HIP_FORCE_DEV_KERNARG=1
export NCCL_MIN_NCHANNELS=16
export ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=1024
export ATOM_ENABLE_RELAXED_MTP=1
export HIP_VISIBLE_DEVICES=0,1,2,3

python3 -m atom.entrypoints.openai_server \
  --model /projects/teamA/danish/models_merged/DSR1-drafter-FP4 \
  --server-port 8890 -tp 4 \
  --kv_cache_dtype fp8 \
  --method mtp --num-speculative-tokens 3 \
  --max-model-len 10240 \
  --gpu-memory-utilization 0.85 \
  --enable-tbo prefill
```

## Gains vs DEC-075 production floor (1278/6.74/148/7253)

| Metric | DEC-075 prod | Current best | Δ |
|---|---|---|---|
| Thr/GPU | 1278 | 1361 | **+6.5%** |
| Median TPOT | 6.74 | 6.35 | **−5.8%** |
| Interact | 148 | 157 | **+6.4%** |
| Median E2E | 7253 | 6842 | **−5.7%** |

**Binding gate math**: E2E ≤ 5000 → TPOT ≤ 4.52 ms. Need −29% from 6.35 ms. Gate-closing requires either kernel work (not available in 24h) or algorithmic change (tree spec, blocked by MLA kernel qseqlen≤4 on gfx950 FP8).

## Patches in progress (Apr 18)

See `dsr_beta/MASTER_PLAN.md` + `memory/project_PATCH_LIST_breakthrough_apr18.md`:
- **Patch #5** setperfdeterminism 2400: applied, SCLK 1406→2400 verified, but bottleneck NOT compute-bound at CONC=4, no TPOT gain (may help at CONC=128)
- **Patch #1** AITER PR #2622 MoE tiles: IN TEST (5 CSV lines swapped, server booting)
- **Patch #6** TBO all + MORI_SHMEM_MODE=ISOLATION: pending
- **Patch #2** ATOM ds_mtp1 branch (MTP cuda graph fix): pending
- **Patch #3** ATOM ds_prefix_cache: pending
- **Patch #4** MLA flatten fix port: pending

## Historical (pre-DSR_beta) — DEC-075 reference

## DEC-075 = DEC-073 + merged checkpoint with drafter MoE layer 61 swapped to FP4

DEC-075 transplants the FP4-quantized layer 61 MoE weights from `amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4` into our main `amd/DeepSeek-R1-0528-MXFP4` checkpoint (where layer 61 was BF16). Drafter now dispatches to `flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w3_fq` (FP4 fast path) instead of the slow BF16 path. Merged checkpoint at `/projects/teamA/danish/models_merged/DSR1-drafter-FP4` — mostly symlinks + 2 cleaned shards.

Build script: `scripts/merge_dec075_v5.py`. Runs in ~5s.

## Full DEC lineage
| DEC | Change | Result |
|---|---|---|
| DEC-056 | DUAL_STREAM=256 | floor 1209/6.89 |
| DEC-058 | +9-row BF16 CSV tune + NCCL=16 | 1202/7.19 |
| DEC-064 | Relaxed MTP (7, 0.4) | 1253/7.06 (+4.2%) |
| DEC-066 | +new tuned CSV (9 rows total) | 1221/6.73/148.6 |
| DEC-069 | Phase 4A v4 drafter HIP graph | NULL (DEC-057 proved Python gap ≈ 0) |
| DEC-071 | BF16 decode tune (97 rows, added 88) | 1267/6.96/143.8/7495/0.9303 (marginal) |
| DEC-072 | BF16 prefill tune (148 rows) | **FAILED — GSM8K 0.865 crash, reverted** |
| DEC-073 | Relaxed MTP (8, 0.5) | 1270/6.80/147.1/7318/0.934 |
| **DEC-075** | **Drafter FP4 transplant (layer 61 MoE from MoEFP4)** | **1278-1297/6.54-6.74/148-153/7056-7253/0.9454** (+2.4% thr, +3.9% interact over DEC-073) |

## DEC-075 profile reality (Apr 17 evening — measured via torch.profiler)

| Component | % of GPU time | Notes |
|---|---|---|
| hipEventSynchronize | **25.5%** | CPU-side async-copy waits, NOT missing graph. Main fwd IS graph-captured at model_runner.py:1741. |
| MoE GEMM (flydsl stage1+2) | 17.8% | Already on FP4 fast path |
| BF16 GEMM (LM head + Q/K/V proj) | ~10.5% | 97-row CSV tune landed |
| AllReduce (reduce_scatter + 2stage) | ~7.5% | Custom 1-shot XGMI could save 1-2ms (multi-day) |
| hipLaunchKernel overhead | 5.8% | 1710 launches × 8.8μs |
| MLA attention | 5.5% | qh32 kernel (already optimized) |
| Other (MoE sort, misc) | ~23% | Scattered |

**Step breakdown (measured)**:
- Main fwd GPU: ~10 ms (60 MoE layers dominate)
- Drafter GPU: **~0.4 ms** per MTP step (20× cheaper than DEC-057 pre-FP4)
- Non-compute overhead: ~6.7 ms per step (sync + launch + CPU scheduling)
- **Step total**: ~17 ms → at 2.5 tokens/step → TPOT 6.74 ms ✓

**Why tree spec is now viable (wasn't at DEC-057)**: drafter cheap enough (0.4 ms) that 3× widening only adds 0.8 ms. Step becomes ~17.8 ms; if tokens/step grows 2.5 → 3.5 with tree, TPOT = 17.8/3.5 = **5.1 ms** (very close to 4.52 gate).

## Config (exactly what reproduces DEC-073)

- **Model**: `amd/DeepSeek-R1-0528-MXFP4` (NOT `-MTP-MoEFP4` — Triton trap)
- **TP**: 4 single replica (GPUs 0-3)
- **KV cache**: FP8
- **MTP**: 3 speculative tokens, relaxed
- **Relaxed MTP**: **(8, 0.5)** hardcoded in `rejection_sampler.py` lines 11-12
- **DUAL_STREAM**: 256 (`ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=256`)
- **NCCL**: 16 channels (`NCCL_MIN_NCHANNELS=16`)
- **BF16 GEMM**: 97 decode shapes tuned in `dsv3_bf16_tuned_gemm.csv` (backup at `/tmp/dsv3_bf16_tuned_gemm.csv.DEC071_0512`)
- **Container**: `danish_atom_main`
- **ATOM commit**: 108a70e + 3 local mods (rejection_sampler, attention_mla, aiter re-export) + Phase 4A v4 drafter HIP graph patch in eagle.py (harmless, null perf)
- **AITER commit**: f8c1d76bd + re-export patch
- **flydsl**: 0.1.2

## Critical file state

**rejection_sampler.py lines 10-12** (DEC-073):
```python
ATOM_ENABLE_RELAXED_MTP = True  # HARDCODED Danish 2026-04-15 B1a
RELAXED_TOP_N = 8   # DEC-073 tighter top-K
RELAXED_DELTA = 0.5 # DEC-073 wider delta
```

**attention_mla.py** (~line 592):
```python
num_kv_splits=None,  # SESSION6A intervention #1
```

**aiter/__init__.py** (appended):
```python
from aiter.ops.cache import concat_and_cache_mla, fused_qk_rope_concat_and_cache_mla
```

**eagle.py**: contains Phase 4A v4 drafter HIP graph patch. 15632 bytes (vs 11065 clean). Patch is null-perf but harmless — keep as infra.

**CSV `aiter/configs/model_configs/dsv3_bf16_tuned_gemm.csv`**: 98 lines (header + 97 rows). Contains DEC-071 tune covering all priority decode shapes (M=1/4/16 LM head, M=16 MLA projections). Backup at `/tmp/dsv3_bf16_tuned_gemm.csv.DEC071_0512`.

## Reproduction steps (from cold container)

### 1. Enter container + pre-flight
```bash
~/bin/docker start danish_atom_main
~/bin/docker exec danish_atom_main bash -c '
export HOME=/tmp
echo "--- 1. ATOM editable path ---"
python3 -c "import atom; print(atom.__file__)"
# expect: /projects/teamA/danish/repos/ATOM_main/atom/__init__.py
echo "--- 2. Relaxed MTP (expect 8 0.5 True) ---"
python3 -c "from atom.model_ops import rejection_sampler as r; print(r.RELAXED_TOP_N, r.RELAXED_DELTA, r.ATOM_ENABLE_RELAXED_MTP)"
echo "--- 3. aiter re-export ---"
python3 -c "import aiter; print(hasattr(aiter, \"concat_and_cache_mla\"))"
echo "--- 4. flydsl 0.1.2 ---"
python3 -c "from aiter.fused_moe import is_flydsl_available; print(is_flydsl_available())"
echo "--- 5. BF16 CSV row count (expect 98) ---"
wc -l /projects/teamA/danish/repos/aiter/aiter/configs/model_configs/dsv3_bf16_tuned_gemm.csv
echo "--- 6. GPUs clean (expect ~284 MB each on 0-3) ---"
rocm-smi --showmeminfo vram 2>&1 | grep "Used Memory" | head -4
'
```

### 2. Launch server (DEC-075 config — DSR1-drafter-FP4 merged model)

**⚠️ CRITICAL — do NOT just run `bash launch_atom_server.sh`.** That script is missing required flags/env vars and will regress to ~7.14 TPOT (MTP=1 mode). Use the exact command below:

```bash
~/bin/docker exec -d danish_atom_main bash -c '
export HOME=/tmp AITER_BUILD_DIR=/tmp/.aiter_cache TRITON_CACHE_DIR=/tmp/.triton_cache
export HIP_FORCE_DEV_KERNARG=1 NCCL_MIN_NCHANNELS=16
export ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=256 ATOM_ENABLE_RELAXED_MTP=1
export HF_HOME=/projects/teamA/hf_cache HUGGINGFACE_HUB_CACHE=/projects/teamA/hf_cache/hub
unset HF_HUB_OFFLINE
export HIP_VISIBLE_DEVICES=0,1,2,3
cd /workspace/ATOM_main
python3 -m atom.entrypoints.openai_server \
  --model /projects/teamA/danish/models_merged/DSR1-drafter-FP4 \
  --server-port 8888 \
  -tp 4 \
  --kv_cache_dtype fp8 \
  --method mtp \
  --num-speculative-tokens 3 \
  --max-model-len 10240 \
  --gpu-memory-utilization 0.85 > /tmp/atom-server-dec075.stdout 2>&1
'
```

**Verify correct boot** (grep log):
- `grep "Capturing bs=.*max_q_len=4"` → max_q_len=4 means mtp_k=3 ✓ (max_q_len=2 = WRONG, means mtp_k=1)
- `grep "flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w3_fq"` at bs=4 → drafter FP4 fast path ✓

Wait for `Uvicorn running on http://0.0.0.0:8888` (~5-8 min with warm cache, ~10-12 min cold).

### 3. Run official bench (MODEL override required for DEC-075)
```bash
~/bin/docker exec danish_atom_main bash -c '
export HOME=/tmp HF_HOME=/projects/teamA/hf_cache HUGGINGFACE_HUB_CACHE=/projects/teamA/hf_cache/hub
unset HF_HUB_OFFLINE
cd /workspace/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x
source specific_conc_var.sh
export MODEL=/projects/teamA/danish/models_merged/DSR1-drafter-FP4   # CRITICAL override
./dsr1_benchmark perf 2>&1 | tail -50
'
```

Without `MODEL` override, harness asks server for `amd/DeepSeek-R1-0528-MXFP4` but server registers as the local path → 400 error.

### 4. Stale cache wipe (if perf regresses between runs)
If a fresh bench comes back significantly worse (e.g., 7.14 TPOT instead of ~6.7), wipe caches and re-launch:
```bash
~/bin/docker exec danish_atom_main bash -c '
rm -rf /tmp/torchinductor_root /tmp/.cache/atom /tmp/.aiter_cache /tmp/.triton_cache /tmp/.flydsl
'
```

### 4. Expected output
```
Total Token throughput (tok/s):          ~5080
Median TPOT (ms):                        ~6.80
Median E2EL (ms):                        ~7318
GSM8K metric:                            ~0.934
Thr/GPU (÷4):                            ~1270
Interactivity:                           ~147
```

## What's been tried and is DEAD (do NOT retest)

| Test | Result | Reason |
|---|---|---|
| Phase 4A v4 drafter HIP graph | NULL (DEC-069) | Python gap ≈ 0 per DEC-057 profile; patch harmless |
| Phase 4B async scheduling | dropped | same root cause |
| **BF16 PREFILL tune** | **GSM8K 0.865 CRASH (DEC-072)** | **errRatio=0.05 too loose for large-M shapes; accumulated drift across 61 layers** |
| v917 MoE kernel port | 3 crashes | ABI mismatch |
| DEC-059 TODO MLA 32-head fix | −18% thr | aiter qk_batch_ratio bug at 32 heads |
| PR #547 stream parallelism | NEUTRAL | — |
| QKNORM fusion | WORSE | — |
| DEC-068 full CSV merge | CORRUPTED | merge script bug (now fixed indirectly by aiter's auto-resolve) |
| AITER #2727 cherry-pick | DEAD | a16w16 kernel only, we use a8w8 FP8 KV |
| ATOM #421 simple cherry-pick | DEAD | Qwen-only dispatch |
| AITER #2620 full cherry-pick | DEAD | API drift to flydsl 0.1.3.1 |
| QuickReduce INT4 | DEAD | min 16 MB tensor, decode is 28 KB |
| GPU_MAX_HW_QUEUES=5 | −4% regression | MI355X Compass warning |
| OMP_NUM_THREADS=1 | −20% | |
| TP=2 SR | GPU memory fault | |
| TP=4 × DP=2 | gfx950 kernel bugs | |
| AITER v0.1.12 direct update | CRASHED | needs flydsl 0.1.3.1 + destroy_dist_env |
| `--enable-prefix-caching` | accuracy crash | MXFP4 None scale |
| `-MTP-MoEFP4` model | 1.5× slower | Triton MoE trap |
| `--max-num-batched-tokens 4096` | CRASHED | can't fit ISL=8192 |

## Env vars NEVER to set
- `AITER_QUICK_REDUCE_QUANTIZATION=INT4`
- `GPU_MAX_HW_QUEUES=5`
- `OMP_NUM_THREADS=1`
- `AMDGCN_USE_BUFFER_OPS=1`
- `ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=1`
- `ATOM_ENABLE_QK_NORM_ROPE_QUANT_FUSION=1`
- `ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION=1`
- `TORCH_BLAS_PREFER_HIPBLASLT=0`

## Gates path forward

If tree spec (DEC-074) succeeds at expected toks/fwd 3.25:
- TPOT 6.80 → ~6.30 ms
- Interact → ~159 (close to 165 gate, may or may not pass)
- E2E → ~6829 ms (still fails 5000)
- Gates: **2/4 likely** (GSM8K + maybe interact)

To reach 4/4:
- E2E gate requires TPOT ≤ 4.52 ms → impossible without tree spec delivering toks/fwd ≥ 4.0+ OR something else structural

**Realistic Apr 18 night outcome: 2-3/4 gates.** Submit for sub-rank points.

## Rollback to DEC-073 if tree spec breaks

```bash
# 1. Restore Phase-4A-only eagle.py (or full clean)
~/bin/docker exec danish_atom_main bash -c '
cp /projects/teamA/danish/repos/ATOM_main/atom/spec_decode/eagle.py.bak_before_hip_graph \
   /projects/teamA/danish/repos/ATOM_main/atom/spec_decode/eagle.py
# DEC-073 had Phase 4A v4 patch, but clean baseline also works equivalently
'

# 2. Restore rejection_sampler.py to (8, 0.5) state
~/bin/docker exec danish_atom_main bash -c '
ls /projects/teamA/danish/repos/ATOM_main/atom/model_ops/rejection_sampler.py.bak_before_8_0.5_*
# pick the most recent and restore if needed
# OR manually re-edit to TOP_N=8, DELTA=0.5
'

# 3. Restore CSV
cp /tmp/dsv3_bf16_tuned_gemm.csv.DEC071_0512 \
   /projects/teamA/danish/repos/aiter/aiter/configs/model_configs/dsv3_bf16_tuned_gemm.csv

# 4. Relaunch server with DEC-073 config (from §2 above)
```

## Key pointers for future Opus sessions
- Active plan: `C:\Users\danis\.claude\plans\fizzy-toasting-teacup.md`
- Memory: `project_final_push_apr17_18.md`, `project_wall_clock_budget_hard.md`, `project_sota_apr17_intel.md`
- Rule: `feedback_pre_measure_or_dont_ship.md` + `feedback_dead_means_unpatched.md`
- Chronology: `daily_log.md`
- Canonical findings: `MASTER_FINDINGS.md`

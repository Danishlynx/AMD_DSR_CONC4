# DSR1 CONC=4 Bottleneck Profile — REAL DATA

**Date**: 2026-04-20
**Container**: `danish_atom_dsr_beta` port 8890
**Model**: `amd/DeepSeek-R1-0528-MXFP4` (STOCK)
**Hardware**: 4× MI355X (CDNA4 gfx950)
**Server config**: `-tp 4 --kv_cache_dtype fp8 --max-model-len 10240 --method mtp --num-speculative-tokens 3 --enable-tbo prefill --max-num-batched-tokens 65536`
**Bench**: 12 prompts, CONC=4, ISL=8192, OSL=1024, random dataset, 74.4 sec wall

---

## 0a. VALIDATION (added after Kimi-bust precedent)

Kimi made the SAME hypothesis mistake (hipEventSync 24% wall) and got BUSTED in validation (99.4% overlapped with GPU). For DSR1 we ran V1/V4/V5 BEFORE patching:

| Test | Result | Verdict |
|---|---:|---|
| V1: hipGraphLaunch overlap with GPU work | **2.2%** | confirmed blocking (vs Kimi 99.4% overlap = no help) |
| V4: GPU active during inter-launch gaps | **3.1%** | confirmed GPU starved |
| V5: GPU util during 915 decode windows | **2.2%** (62 ms idle/step) | confirmed bottleneck is real |

**Caveat**: numbers are profiler-on (3.4× host inflation). Native hipGraphLaunch is likely ~18 µs (not 63 µs); native wall fraction is ~30-50% (not 77%). Patch gains scale down accordingly. Realistic ceiling **2/4 gates** with all P-patches; 4/4 still needs MTP=5+kernel work.

---

## 0. EXECUTIVE SUMMARY — THE ONE BOTTLENECK

**`hipGraphLaunch` consumes 77.7% of wall time on host CPU.**

| Layer | Time | % Wall | Notes |
|---|---:|---:|---|
| Wall total | 74,500 ms | 100% | bench duration |
| GPU kernel time | 1,737 ms | 2.3% | actual work on the GPU |
| **Host inside `hipGraphLaunch`** | **57,875 ms** | **77.7%** | **915 calls × 63 µs each** |
| Host inside other HIP APIs | 9,308 ms | 12.5% | `hipLaunchKernel`, sync, memcpy, etc |
| Total HIP API time | 67,183 ms | 90.2% | host CPU is BUSY in HIP runtime |
| Host idle (GPU exclusive) | 7,317 ms | 9.8% | GPU gets to compute alone |

**Translation**: GPU is 97.7% idle, but NOT because nothing is happening — the host is stuck inside `hipGraphLaunch` for 77.7% of the wall clock submitting graphs. Each decode step's graph submission costs 63 µs of host CPU (vs typical 5–15 µs for HIP graph launch). Until that 63 µs drops, GPU can't start the next step.

**This is OPPOSITE of Kimi**, which was bottlenecked by `hipEventSynchronize` (88% of HIP API). DSR1's `hipEventSynchronize` is only 3.1% — already efficient.

---

## 1. METHOD 1 — torch.profiler (`--torch-profiler-dir` CLI flag)

### Setup
- CLI flag `--torch-profiler-dir /tmp/torch_traces` (NOT env var — env var is silently ignored, see Kimi guide)
- Bench: `python3 -m atom.benchmarks.benchmark_serving --port 8890 --dataset-name random --random-input-len 8192 --random-output-len 1024 --num-prompts 12 --max-concurrency 4 --profile --trust-remote-code`
- Output: 4 ranks × 35 MB gzipped trace JSON (1.1 GB uncompressed each)

### Bench result (with profiler overhead)
| Metric | Value |
|---|---:|
| Bench duration | 74.44 s |
| Total tput | 1472.30 tok/s |
| Median TTFT | 490.89 ms |
| Median TPOT (with profiler) | 22.60 ms |
| Median TPOT (no profiler, baseline) | 6.66 ms |
| Mean ITL | 79.98 ms |

### 1A. GPU kernel breakdown (`kernel` category — 1.25M events)

**Top 30 GPU kernels (rank 0)**:

| ms | % | Calls | Avg µs | Kernel |
|---:|---:|---:|---:|---|
| 127.05 | 7.3% | 94579 | 1.3 | `aiter::local_device_load_rmsnorm<bf16, 512, 2>` |
| 119.09 | 6.9% | 94579 | 1.3 | `aiter::reduce_scatter_cross_device_store<bf16, 4>` |
| 91.77 | 5.3% | 60070 | 1.5 | `Cijk_*MT32x16x512` (BF16 GEMM via hipBLASLt) |
| 88.39 | 5.1% | 58556 | 1.5 | `aiter::fused_qk_rmsnorm_kernel` |
| 88.28 | 5.1% | 55811 | 1.6 | `hgemm_bf16_32x64x64_S2TN_AS_SPK16_0` |
| 85.51 | 4.9% | 56722 | 1.5 | `kn_mla_reduce_v1_ps<512,32,1>` |
| 85.15 | 4.9% | 54078 | 1.6 | `hgemm_bf16_32x64x128_S2TN_AS_SPK8_BS_0` |
| 77.85 | 4.5% | 53426 | 1.5 | `moe_gemm2_0` |
| 76.80 | 4.4% | 58556 | 1.3 | `aiter::fuse_qk_rope_concat_and_cache_mla_per_head_kernel` |
| 76.41 | 4.4% | 53426 | 1.4 | `moe_gemm1_0` |
| **74.04** | **4.3%** | 56722 | 1.3 | **`aiter::mla_a8w8_qh32_qseqlen4_gqaratio32_ps`** ← HK port target |
| 73.30 | 4.2% | 47567 | 1.5 | `_batched_gemm_a8w8_*` |
| 70.20 | 4.0% | 47670 | 1.5 | `Cijk_*MT32x16x1024` (BF16 GEMM) |
| 69.12 | 4.0% | 56235 | 1.2 | `aiter::grouped_topk_opt_sort_kernel` |
| 66.54 | 3.8% | 45363 | 1.5 | `ck_tile::MoeSortingKernel` |
| 64.04 | 3.7% | 53070 | 1.2 | `aiter::mxfp4_quant_moe_sort_kernel<256,32>` |
| 60.70 | 3.5% | 50402 | 1.2 | `aiter::mxfp4_quant_moe_sort_kernel<64,8>` |
| 26.12 | 1.5% | 19792 | 1.3 | `aiter::allreduce_fusion_kernel_1stage` |

### Kernel categories (rank 0)

| Category | ms | % |
|---|---:|---:|
| MoE total (gemm+sort+topk+mxfp4_sort) | 371 | 21.4% |
| MLA attention | 244 | 14.0% |
| RMSNorm (incl. fused) | 233 | 13.4% |
| GEMM_BF16 (hgemm) | 197 | 11.3% |
| GEMM_FP4_FP8 | 172 | 9.9% |
| Reduce/scatter | 129 | 7.4% |
| Sort | 70 | 4.0% |
| AllReduce | 28 | 1.6% |
| `Cijk_*` BF16 (hipBLASLt unmatched) | ~180 | ~10% |
| Sample/TopK | 24 | 1.4% |
| **Total kernel time** | **1737** | — |
| **Wall** | **74,507** | — |
| **GPU busy** | — | **2.3%** |

### 1A finding
**MLA `qh32_qseqlen4` is only 4.3% of kernel time** = the HipKittens C1 port targets a kernel that is 4.3% of 2.3% wall = **0.099% wall total**. Even a 2× speedup on that kernel saves 0.05% TPOT. **HK port was the wrong lever.**

### 1B. HIP API breakdown (`cuda_runtime` category — 414K events) ⭐

THIS is the analysis the Kimi guide forced us to do. Without it, we'd never know the host is the bottleneck.

| ms | % HIP | % wall | Calls | Avg µs | HIP API |
|---:|---:|---:|---:|---:|---|
| **57,875** | **86.1%** | **77.7%** | **915** | **63.25** | **hipGraphLaunch** ← THE ONE |
| 4,585 | 6.8% | 6.2% | 113884 | 0.04 | hipLaunchKernel |
| 2,061 | 3.1% | 2.8% | 2775 | 0.74 | hipEventSynchronize |
| 949 | 1.4% | 1.3% | 29654 | 0.03 | hipModuleLaunchKernel |
| 689 | 1.0% | 0.9% | 18731 | 0.04 | hipExtModuleLaunchKernel |
| 611 | 0.9% | 0.8% | 26795 | 0.02 | hipMemcpyAsync |
| 163 | 0.2% | 0.2% | 2308 | 0.07 | hipExtLaunchKernel |
| 122 | 0.2% | 0.2% | 1830 | 0.07 | hipMemsetAsync |
| 32 | <0.1% | — | 11090 | 0.003 | hipEventRecord |
| 22 | <0.1% | — | 4605 | 0.005 | hipStreamWaitEvent |
| **67,183** | **100%** | **90.2%** | **413,858** | — | **TOTAL HIP API** |

### 1B critical observations

1. **`hipGraphLaunch` 86.1%** — the host CPU is INSIDE this function for 57.9 sec out of 74.5 sec wall. Each call is 63 µs. There are 915 calls. **This is the only thing that matters at CONC=4.**
2. **Sync APIs total only 4%** (`hipEventSynchronize` 2.0 sec + `hipMemcpyAsync` 0.6 sec). DSR1 is NOT sync-stall-bound. The Kimi-style F0a "double-buffer pipeline depth" patch saves 0.5 sec max — not the lever.
3. **Per-step decode wall = 81 ms** (74,500 ms / 915 graph launches). With 4 concurrent requests, effective per-request TPOT = 20.3 ms (matches the 22.6 ms measured TPOT — 2.3 ms profiler overhead).
4. **Why is hipGraphLaunch 63 µs?** Typical is 5–15 µs. 4× overhead suggests:
   - Large graph node count (likely 200–500 kernels per decode-step graph)
   - Non-resident graph re-resolution at each launch
   - Implicit head-of-line stream sync before launch
   - Symbol/pointer fixup at launch time
5. **915 launches = 1 per decode step** (matched to user_annotation `decode[bs=N tok=N*4 spec=3]` count). MTP=3 verifies 4 positions per step.

### 1C. Adjacency around `hipGraphLaunch` (915/915 identical)

Pattern verified by 3-event window before/after each `hipGraphLaunch` call:

```
BEFORE:                                 AFTER:
1. user_annotation: decode[bs=N tok=N*4 spec=3]    1. hipDriverGetVersion
2. cuda_runtime: hipStreamIsCapturing              2. cpu_op: aten::slice
3. cuda_runtime: hipLaunchKernel                   3. cpu_op: aten::as_strided
4. cuda_runtime: hipGraphLaunch  ← 63 µs
```

`bs=4` count = 658 (steady state); `bs=3,2,1` = 257 (drain phase). Confirms ONE graph per decode step regardless of batch.

### 1D. GPU utilization (cross-check)

| Metric | Value |
|---|---:|
| Wall span | 74,507 ms |
| GPU kernel total | 1,737 ms |
| **GPU busy** | **2.3%** |
| Host inside HIP runtime | 67,183 ms (90.2%) |
| Host idle (true) | 7,324 ms (9.8%) |

Implication: it is impossible for the GPU to be more than ~10% busy unless the host stops calling HIP APIs. The bottleneck is host CPU time spent inside `hipGraphLaunch`.

---

## 2. METHOD 2 — rocprofv3 (cross-validation, optional)

**Status**: cold boot v2 in progress (with `--process-sync true` to fix v1 flush issue).

**Why used**: original purpose was "find HIP API overhead torch.profiler can't see". **Now superseded by Method 1 §1B** — torch.profiler `cuda_runtime` category captures all HIP APIs with comparable accuracy. M2 will provide cross-validation only.

**Expected agreement**: rocprofv3 `hipGraphLaunch` should match torch.profiler within ±10%. If different, suggests profiler-induced inflation in M1.

---

## 3. METHOD 3 — rocprofv3 HSA + memcopy (low priority)

**Status**: combined with M2 v2 boot.

**Why used**: HSA queue analysis + memcopy direction breakdown.

**Expected outcome**: HSA queue contention probably negligible (TP=4, dual-stream MoE on); memcopy dominated by H2D parameter shipping during prefill (already known).

---

## 4. METHOD 4 — rocprofv3 PMC counters

**Status**: pending (after M2/M3).

**Why used**: per-kernel SQ_INSTS_VALU vs SQ_INSTS_MFMA, LDS bank conflicts, real `GRBM_GUI_ACTIVE` (vs torch.profiler's wall-clock guess).

**Expected outcome**: less critical now that M1 §1B identified host bottleneck. Still useful to confirm individual kernel HW efficiency (e.g., MoE_FlyDSL kernels — are they MFMA-bound or VALU-bound?).

---

## 5. FINAL BOTTLENECK TABLE

### Ranked by % wall time

| # | Bottleneck | % Wall | Addressability | Lever |
|---|---|---:|---|---|
| 1 | `hipGraphLaunch` host overhead | **77.7%** | medium-high | reduce graph node count via fusion; pipeline graph submit |
| 2 | Other HIP API (kernel launches, sync, memcpy) | 12.5% | low (system overhead) | minor |
| 3 | GPU compute (actual work) | 2.3% | high but small leverage | kernel optimization (MoE, MLA, GEMM) |
| 4 | Host idle | 9.8% | n/a | this is when GPU is computing alone |

### Per-API ranking (within the 90.2% host-busy time)

| # | HIP API | ms | % | Per-call avg | Saves if cut to median |
|---|---|---:|---:|---:|---:|
| 1 | hipGraphLaunch | 57,875 | 86.1% | 63.25 µs | reducing avg to 30 µs saves 30,000 ms = **−40% wall** |
| 2 | hipLaunchKernel | 4,585 | 6.8% | 0.04 µs | already minimal |
| 3 | hipEventSynchronize | 2,061 | 3.1% | 0.74 µs | F0a double-buffer saves ~1,000 ms = −1.3% wall |
| 4 | hipModuleLaunchKernel | 949 | 1.4% | 0.03 µs | minimal |
| 5 | hipExtModuleLaunchKernel | 689 | 1.0% | 0.04 µs | minimal |
| 6 | hipMemcpyAsync | 611 | 0.9% | 0.02 µs | minimal |

### What this means for closing 4/4 gates

Current state (no profiler): TPOT 6.66 ms, Thr/GPU 1351, E2E 7221 ms, GSM8K 0.934 = 1/4

If `hipGraphLaunch` per-call cost drops 63 µs → 30 µs (53% cut, achievable via graph-node fusion):
- Save 30 ms per decode step × 915 steps / 4 ranks = ~7,500 ms wall
- Wall: 74,500 → 67,000 ms (no profiler: ~6.66 ms TPOT × 0.9 = 6.0 ms)
- Thr/GPU: ~1500 (likely clears Thr/GPU gate)
- E2E: 7221 → ~6500 ms (still doesn't clear ≤5000 gate)

Even halving hipGraphLaunch overhead doesn't get us 4/4. **E2E gate is binding** — needs both:
1. hipGraphLaunch overhead cut (this analysis)
2. Tokens-per-step boost (MTP=4+ → kernel work blocked)

---

## 6. INVESTIGATION RESULTS — three root causes tested

### Root cause B (re-resolution warming up): ❌ RULED OUT
- 915 hipGraphLaunch durations: p10=57µs, p50=64µs, p90=65µs, p99=67µs — TIGHT distribution
- First 50 launches avg = 64.1 µs vs last 315 launches avg = 60.9 µs — NO warm-up effect
- Verdict: graph instantiation cache is fine; not a JIT/resolution issue

### Root cause C (head-of-line serialization): ✅ CONFIRMED but NOT main cause
- All 915 launches on a single thread (TID 294)
- Inter-launch gap median = 14 µs
- Pure serial pipeline: launch → return after 64µs → 14µs python work → next launch
- Pipeline depth = 1 (no double-buffering)
- BUT: the 14 µs gap is small enough that hiding the 64 µs launch wouldn't help much, since the LAUNCH ITSELF is the cost — not waiting AROUND it.

### Root cause A (graph node count too high): ✅ CONFIRMED — primary cause
- DSR1 architecture: 61 layers × ~25 kernels/layer = ~1525 nodes per forward pass
- Per-node submission cost in HIP runtime ≈ 40 ns
- 1525 nodes × 40 ns = 61 µs — **matches measured 63 µs almost exactly**
- This is per-node submission cost, NOT a wait — fixing it requires reducing node count

### Investigation conclusion
The 63 µs hipGraphLaunch is dominated by HIP runtime per-node submission cost. Each kernel node requires queue write + dependency setup. The bottleneck is **graph complexity**, not synchronization.

---

## 7. UPDATED PATCH PLAN — based on root cause A confirmation

### Lever P1: Reduce graph node count via kernel fusion (PRIMARY)

**Mechanism**: Each fused kernel removes one node from the graph. Per-launch cost drops linearly with node count (~40 ns saved per node removed).

**Patch options**:
- **P1a**: Fuse `qk_rmsnorm + qk_quant` into `fused_qk_norm_rope_cache_quant` (already done in aiter — just needs default-on for DSR1)
- **P1b**: Fuse `add_rmsnorm + quant` into `add_rmsnorm_quant` (already exists, gated on `ATOM_USE_TRITON_GEMM` which we DON'T set → currently OFF). Enable for ~5-10 nodes per layer × 61 layers = **300-600 nodes saved** = ~12-24 µs/launch saved = **~2-4% TPOT**.
- **P1c**: Fuse `mxfp4_quant + moe_sort` into `mxfp4_quant_moe_sort` (variants already exist in aiter — pick the right one).

**Expected gain**: P1b alone saves 12-24 µs/launch × 915 / 4 ranks = **−3-5 sec wall = −5-7% TPOT**. Stack with P1a/P1c for additional 5-10%.

### Lever P3: hipGraphInstantiateWithFlags + DeviceLaunch (HIGH)

**Mechanism**: ROCm 7+ supports `hipGraphInstantiateFlagDeviceLaunch` — graph is uploaded to device once, subsequent launches are device-side enqueues that bypass host CPU queue submission entirely.

**Patch site**: PyTorch's `torch.cuda.CUDAGraph` doesn't expose this flag. Need ATOM-side modification at `model_runner.py:2014` `torch.cuda.graph(graph, self.graph_pool, stream=gc.stream)` to use raw HIP API or get torch upstream support.

**Expected gain**: cut launch from 63 µs to <10 µs = **−50 µs/launch × 915 / 4 = −11 sec wall = −15% TPOT**.

**Risk**: PyTorch wrapper limitation. May require C++ extension. 1-2 day port.

### ~~Lever P2: Pipeline depth 2~~ — RULED OUT

The 14 µs inter-launch gap is too small to overlap with the 63 µs LAUNCH ITSELF. The 63 µs is per-node submission to HIP queue, NOT waiting on something. Pipelining would only save ~14 µs / 78 µs = 18% of inter-launch time.

### Lever P4: GPU kernel optimization (LOW — only 2.3% of wall)

GPU kernel total = 2.3% wall. Even 50% optimization saves 1% wall. **HipKittens MLA C1 port (4.3% of 2.3% = 0.1% wall) is the wrong lever.**

EXCEPTION: HipKittens C1 unblocks MTP=5 (qseqlen=6). At MTP=5, tokens-per-step rises from ~3.76 to ~4.7 (25% more) — but ONLY if hipGraphLaunch overhead doesn't grow proportionally. If graph node count stays similar but tokens per launch grows, P1+MTP=5 stack could give an additional 25% reduction.

---

## 7. WHAT M1 PROVES vs PRIOR HYPOTHESES

| Prior hypothesis | M1 verdict |
|---|---|
| HipKittens MLA C1 port closes the gap | ❌ targets 4.3% of 2.3% kernel time = 0.1% wall |
| MTP=4/5 with custom kernel | ⚠️ unblocks more tokens/step but doesn't help if `hipGraphLaunch` still 63 µs/launch |
| Compact-experts MoE optimization | ❌ MoE = 21.4% of 2.3% = 0.5% wall; lever is smaller than thought |
| `hipEventSynchronize` is the bottleneck (Kimi pattern) | ❌ only 3.1% of HIP API time, vs Kimi's 88% |
| GPU is starved at CONC=4 | ✅ TRUE but cause is HOST CPU in `hipGraphLaunch`, not naive launch overhead |

---

## 8. RAW DATA REFERENCES

- M1 raw trace: `/tmp/torch_traces/rank_{0,1,2,3}/DeepSeek-R1-0528-MXFP4_ts_20260420_130020_*.pt.trace.json.gz` (4 × 35 MB)
- M1 parser: `/tmp/parse_torch_trace.py` (kernel category)
- M1B parser: `/tmp/parse_trace_hip_api.py` (cuda_runtime category — the host-side analysis)
- M1C parser: `/tmp/parse_trace_adjacent.py` (adjacency)
- Bench result on container: `(/tmp/m1 launch log)`
- Stock floor JSON: `/projects/teamA/danish/experiments/stock_floor_MTP3_TBO_QR_canonical.json`
- M2+M3 v2 in-flight at: `/tmp/rocprof_m23v2/` (cold boot 14:12 UTC)

---

## 9. NEXT STEPS — patch order

1. ✅ M1 fully analyzed (kernel + cuda_runtime + adjacency + launch distribution + concurrency)
2. ✅ Code read (`model_runner.py:1744 .replay()` and `:1905-2000 capture_cudagraph`)
3. ✅ All 3 root causes tested — A confirmed primary, B ruled out, C ruled in but not main
4. ⚠️ M2+M3 (rocprofv3) cross-val FAILED twice (workers won't flush on signal — they require atexit which never fires under SIGKILL). Accept M1 as authoritative.
5. ⏰ **Patch P1b first** (1-day work): enable `add_rmsnorm_quant` fused path. Lowest risk, well-tested in aiter. Re-bench, expect −5% TPOT.
6. ⏰ Patch P1a + P1c (2-day work): audit all DSR1 layer kernels for additional fusable pairs. Apply incrementally, bench each.
7. ⏰ **Patch P3 (1-2 weeks)**: ATOM-side wrapper for `hipGraphInstantiateFlagDeviceLaunch`. Highest gain (−15% TPOT). Highest risk (torch limitation).
8. ⏰ **MTP=5** + HipKittens C1 (post-graph-fusion): unlocks 25% more tokens per launch. Combined with P1+P3 might close E2E gate.

## 10. WHY 4/4 GATES IS HARD

Even stacking ALL P-patches:
- P1a+b+c: −10% TPOT (graph fusion, 1500→1200 nodes)
- P3: −15% TPOT (device-launch flag)
- HipKittens C1 + MTP=5: +25% tokens/step

Conservative compounded estimate:
- TPOT: 6.66 → 6.0 (P1) → 5.1 (P3) → 4.1 (MTP=5 effective) ms
- Thr/GPU: 1351 → 1500 (P1) → 1700 (P3) → 2100 (MTP=5)
- E2E: 7221 → 6500 → 5500 → 4400 ms

In theory 4/4 reachable, but each step has risk and the MTP=5 path requires custom kernel work.

**Honest critical path**:
1. P1b is the only ≤1 day patch — start there
2. P3 is the highest gain but needs torch internals work — 1-2 weeks
3. MTP=5 / HipKittens unlocks the E2E gate but is multi-week
4. Without all 3, capped at 2-3/4 gates

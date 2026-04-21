# 🚧 SESSION-8 — C1 HipKittens qh32 port in flight (Apr 19 late evening)

**Most current state** — update chain: Apr 17 FINAL PUSH → Apr 18/19 multi-phase plan → Apr 19 session-7 short-patch exhausted → **Apr 19 session-8 C1 full port initiated**.

Danish directive (2026-04-19): **"timing is not the constraint, build it, I want AMD optimized kernels"**.

## Floor (unchanged)

`1361 / 6.35 / 157.55 / 6842 / 0.934` → **1/4 gates**. Last re-bench `1341/6.47/154.63/7009/0.9356` — within noise. **ZERO benchmarks this session**.

## Active plan
- `C:\Users\danis\.claude\plans\fizzy-toasting-teacup.md` (session-8 entry at top)

## Key docs
- `Current_plan.md` — top-of-head summary (updated session-8)
- `MASTER_FINDINGS.md` — canonical results + decision history (updated session-8 with C1 port + HK discovery)
- `daily_log.md` — chronological record (session-8 appended with B2 test + C1 initiation)
- `best_reproduce.md` — floor repro recipe

## Session-8 state (Apr 19 late evening)

- **B2 P-EAGLE position-only tested + reverted**: 30% accept, 1.9 tok/step, −31% thr regression. Training-free init gives near-zero accept at t+2/t+3 as research predicted.
- **C2 short-patch proved DEAD**: all 3 variants (top-K rescore / dual-chain / true tree) are no-op, net-neg, or require C1's kernel mask.
- **C3 MTP=4+ blocked on C1**: no qseqlen>4 kernel exists; `hsa/codegen.py` is a CSV compiler not kernel generator.
- **🎯 HipKittens MLA discovery**: `csrc/kernels/mla/hk/` has 2646 LOC HK MLA already integrated. FP8 + DeepSeek shape + runtime max_seqlen_q all baked in. Blocker: `static_assert(nhead==128)`.
- **C1 patches deployed**: NEW `mi3xx_v32_fwd_decode_h32_fp8_fp8.cuh` + dispatch + mla.py routes + MTP cap lift. Backups `.pre_c1`.
- **JIT compile SUCCEEDED**: 34.3s clean, no template errors. `module_hk_mla.so` built.
- **First boot HUNG**: MTP silently collapsed to MTP-1 (no `max_q_len=4` captures), 2 of 4 TP workers alive, SHM broadcast timeout spam. Likely rank 2/3 silent crash during MTP-3 drafter capture.
- **Container restart cleared VRAM/zombies**. Patches intact.
- **Control boot in progress** (no HK) to isolate cause — running at session close.

## Hard rule reminder
Every intervention needs: measured-target + mechanism + expected-delta + pass/fail gate + post-measurement (see `memory/feedback_pre_measure_or_dont_ship.md`).

## Realistic session-9 outcomes
- If HK debug yields correct MTP-3 at mtp_k=3 + perf match or beat: real shot at MTP-4 at +33% tok/step = 4/4 gates
- If HK proves unviable after debug: revert + submit floor at 1/4 gates
- No other positive-math path identified

---

## Historical (prior session context below)

### 🚨 FINAL PUSH MODE — 2026-04-17 late night (original declaration)

User declaration: **Block 3 / Kimi / May 15 horizon DROPPED. Single mission: 4/4 CONC=4 gates by Apr 18 night or submit at sub-rank.**

Floor DEC-066: 1221 thr/GPU, 6.73 ms TPOT, 148.6 interact, 7663 ms E2E, 0.9378 GSM8K. Binding gate was E2E ≤ 5000 ms → TPOT ≤ 4.52 ms → need **−33%**. (Note: floor improved since to 1361/6.35/157/6842/0.934 via DEC-075 drafter FP4 transplant.)

---

## Original strategic doc below (historical context — note May 15 horizon is SUPERSEDED)

The hardware roofline (the speed limit)
We have 8× MI355X. Each GPU has:

288 GB HBM3e memory
8 TB/s HBM bandwidth peak (~6.5 TB/s realistic)
256 active compute units (CUs)
160 KB LDS per CU (on-chip scratchpad)
10 PFLOPS MXFP4 compute, 5 PFLOPS FP8
8 GPUs all connected pairwise via Infinity Fabric, 153 GB/s bidir per link
The model is DeepSeek-R1-0528 in MXFP4 quantization:

671B total parameters but only 37B active per token (sparse MoE)
61 layers (3 dense MLP + 58 MoE)
256 experts, 9 fire per token (8 routed + 1 shared)
MLA attention with 128 heads, 512-dim KV-LoRA compression
Total weights ~155 GB across 8 GPUs = ~19 GB per GPU
The roofline math: every output token needs to read its weights from HBM. With sparse MoE (9/256 experts active), effective bytes-per-token ≈ 10-12 GB per GPU. At 6.5 TB/s sustained, that's 12/6500 = ~1.5 ms of pure HBM-read time per token. That's the physical floor. With MTP=3 averaging 1.89 accepted tokens per forward, the effective floor drops to ~0.8 ms per output token.

We're at 6 ms TPOT at CONC=4. That's 7.5× above the floor. The other 5+ ms is overhead.

The stack we're running (BEST BASE)

 [Client]
    │ HTTP/JSON
    ▼
 [ATOM api_server.py]                  ← OpenAI-compatible REST endpoint
    │
    ▼
 [ATOM EngineCore busy_loop]            ← Python while-loop, NOT asyncio
    │ pickle over zmq, daemon threads
    ▼
 [ATOM Scheduler]                       ← continuous batching, MTP draft tokens
    │
    ▼
 [ATOM ModelRunner.forward]             ← Python prep + cudagraph replay + postprocess
    │
    ├─ [run_model] → cudagraph replay  ← ALL kernels for one decode step in 1 HIP call
    │       │
    │       ▼
    │  [DeepseekV2DecoderLayer × 61]
    │       │
    │       ├─ RMSNorm + AllReduce (FUSED via aiter::fused_allreduce_rmsnorm)
    │       ├─ MLA Attention
    │       │     ├─ Query/KV projections (BF16 GEMM via hipblasLT)
    │       │     ├─ MLA decode kernel (aiter::mla_a8w8_qh16_qseqlen4_*)
    │       │     └─ MLA reduce kernel (kn_mla_reduce_v1_ps)  ← num_kv_splits=16 problem
    │       ├─ RMSNorm + AllReduce (FUSED)
    │       └─ MoE
    │             ├─ Gate (top-9 routing)
    │             ├─ Token sorting
    │             ├─ Expert compute (FlyDSL stage1+stage2 at decode shapes)
    │             └─ Combine
    │
    └─ [postprocess] OUTSIDE cudagraph  ← Python: index_select, sampler, rejection_sampler
            │
            ▼
       [drafter.propose] OUTSIDE cudagraph  ← MTP/EAGLE drafter forward IN PYTHON
Container: danish_atom_main running rocm/atom:rocm7.1.1-ubuntu24.04-pytorch2.9-atom0.1.1-MI350x, with ATOM main commit 108a70e + AITER main commit a35b45ad9 + flydsl 0.1.2. Two env vars matter: ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=16384 (enables shared-experts-on-alt-stream overlap) and the launch flags --method mtp --num-speculative-tokens 3 --kv_cache_dtype fp8 -tp 8.

Where the 6 ms of TPOT actually goes (the latency budget)
At CONC=4, per output token:

~4 ms (66%) is GPU kernel time — the actual compute work the GPU does
~2 ms (34%) is non-kernel residual — Python scheduler, MTP postprocess (torch.index_select × 2, sampler, rejection sampler), the EAGLE drafter forward pass (which runs in Python OUTSIDE the cudagraph), kernel launch queue latency, cross-GPU sync waits, detokenizer
That ~2 ms of non-kernel overhead is the structural cost of MTP running in Python rather than inside the cudagraph. It's the single biggest CONC=4 lever, but it's also the hardest to fix because the drafter is a separate model with its own forward pass that ATOM hasn't compiled into the same graph as the main model.

Inside the 4 ms of GPU kernel time, the top 10 kernels (post-FlyDSL):

Rank	Kernel	%	What it is
1	moe_gemm1_0	10.4%	FlyDSL MoE stage 1 (decode)
2	mla_a8w8_qh16_qseqlen4_*	9.0%	MLA attention decode (the actual attention computation)
3	kn_mla_reduce_v1_ps	8.3%	MLA partial reduction (num_kv_splits=16 problem)
4	allreduce_fusion_kernel_1stage	7.1%	Fused AllReduce + RMSNorm — already optimized
5	hgemm_bf16_*_SPK4	6.0%	MLA projection (q_a, q_b, kv_a, kv_b — all BF16)
6	ncclDevKernel_Generic_1	5.3%	Cross-GPU NCCL all-reduce
7	_batched_gemm_a8w8_*	4.7%	FP8 batched GEMM (MoE expert compute)
8	hgemm_bf16_*_SPK8	4.4%	More MLA projections
9	MoeSortingKernel	4.1%	Token-to-expert dispatch sort
10	kernel_moe_mxgemm	3.8%	CK MoE prefill (only fires on first step)
Where we are vs the gates (the scoreboard)
Metric	CONC=4	CONC=32	CONC=128
Throughput	738/1500 (49%) ❌	2345/3900 (60%) ❌	3555/6000 (59%) ❌
Interactivity	167/165 ✅	64/50 ✅	24/48 (50%) ❌
E2E	6324/5000 (126%) ❌	16507/18000 ✅	43637/22000 (198%) ❌
GSM8K	0.937 ✅	0.941 ✅	0.942 ✅
3 of 9 gates passing. The hardest are CONC=4 throughput (need ~2× current) and ALL three CONC=128 metrics (need ~2× throughput, ~2× interactivity, ~2× lower E2E).

The 4 levers — the framework for every optimization
Every optimization in LLM inference reduces to ONE of these:

QUANTIZE — fewer bits per weight = fewer HBM bytes per token → higher throughput. We already use MXFP4 weights + FP8 KV. Next would be FP8 MLA projections (PATCH-005, DEAD because crash is in vLLM compiler we can't patch).

FUSE — combine consecutive kernels so intermediate tensors stay in registers/LDS instead of round-tripping through HBM. Already done: AllReduce+RMSNorm. Not done: MTP drafter into main cudagraph (the big CONC=4 win), MLA decode + reduce into one kernel (doesn't exist yet, would need AITER kernel work).

CACHE/REUSE — keep hot data on-chip across tokens or across requests. CUDA graphs are an example (cache the kernel launch sequence). Prefix caching is another (cache shared prompt tokens), but ATOM's prefix caching crashes on MXFP4 — needs an ATOM source patch we deferred.

SKIP — don't do the work at all. MTP (skip rejected draft tokens), sparse MoE (skip experts that don't fire), early exit. We already use MTP=3 and sparse MoE.

Every intervention I propose must be one of these 4. If it isn't, it's a pipe dream.

What we CAN attack right now (today's interventions)
The three patches in Intervention Plan v2 — all source-backed, not gambling:

1. num_kv_splits=16 → None (the patch we're testing right now)

Lever: SKIP — fewer partial reductions to combine
Attacks the 8.3% kn_mla_reduce_v1_ps kernel
Predicted: -0% CONC=4, -3-5% CONC=32, -5-10% CONC=128
One line, fully reversible
2. Re-enable q→FP8 cast in attention_mla.py

Lever: QUANTIZE — query in FP8 instead of BF16
Attacks the 9.0% MLA decode kernel (might dispatch to a faster variant)
Predicted: -0-5% any CONC
3 lines, reversible
3. Layer 0 input_norm fusion

Lever: FUSE — fuse the AllReduce + RMSNorm for layer 0 too
Attacks the unfused AR+RMS calls on layer 0 only
Predicted: -0.5% all CONC
1 line, reversible
These are TONIGHT'S work. Combined optimistic ceiling: -3 to -15% per CONC. Realistic: -2 to -8%.

What we COULD attack but isn't tonight (the bigger fish)
1. MTP drafter into cudagraph (the biggest CONC=4 fish)

Lever: FUSE — currently the drafter forward runs as a separate Python step
Attacks the ~2 ms of Python overhead at CONC=4
Predicted: -10-25% CONC=4 TPOT
Effort: multi-day ATOM source patch + AITER kernel changes
Risk: high (touches the spec decode core path)
2. Compute-communication overlap kernel

Lever: FUSE — overlap AllReduce with the next compute step
Attacks the 13-19% NCCL/all-reduce time at CONC=128
Predicted: -5-10% CONC=128 TPOT
Effort: 1-2 weeks of novel kernel work (this is a real research kernel)
3. MTP=5+ AITER patch

Lever: SKIP — more skipped tokens per forward pass
Attacks the MTP=4 hard limit (AITER asserts qo_len ≤ 4)
Predicted: -10-15% TPOT all CONC
Effort: 3-5 days of AITER ASM kernel modification
4. SGLang + MORI PD disaggregation

Lever: a structural change — split prefill and decode onto different GPU subsets
Attacks the 7-second TTFT at CONC=128 directly
Predicted: potentially -50% TTFT at CONC=128 (Research Report quotes 10× on MI355X)
Effort: 2-day framework switch + setup + verification on single node
Risk: very high (untested on this cluster, unknown if single-node split works)
The honest gate math
Stacking EVERY intervention above optimistically:

CONC	Current	Tonight (#1+#2+#3)	+ Big fish 1-3	+ SGLang PD disagg	Gate
4	738	~770 (+4%)	~950 (+29%)	~950 (no help)	1500 (-37%)
32	2345	~2520 (+7%)	~3000 (+28%)	~3100 (+32%)	3900 (-21%)
128	3555	~3870 (+9%)	~4400 (+24%)	~5500-7000 (+55-97%)	6000
Even with everything, CONC=4 throughput gate is unreachable without a fundamental TP architecture change (which we proved is dead on gfx950 today).
CONC=32 gates are within reach if the big fish land.
CONC=128 gates are only reachable via SGLang PD disagg — and that's a bet, not a plan.

So what's the actual strategy
Two-layer strategy:

Hit gates we CAN hit — close interactivity (need 24→48 at CONC=128, that's the hardest interactivity gap), close E2E (CONC=4 needs 6324→5000, CONC=128 needs 43637→22000). These are achievable with TPOT reduction from interventions #1-#3 plus the big fish.

Maximize sub-rank scoring on the gates we MISS — the hackathon is sub-ranked: 600 throughput points + 400 interactivity points per CONC × 3 CONCs = 3000 total. Points are awarded by RANK among finalists, not by hitting the absolute gate. Moving CONC=4 throughput from 738 → 950 in a 10-finalist field is real sub-rank position movement = real prize money even if 1500 isn't hit.

Track 1 (DSR1) prize structure:

Top 10 finalists each get $10k guaranteed
Top 1 gets $350k grand prize (only if all 9 gates met)
Sub-rank in 2-9 = guaranteed $10k + bigger share of remaining pool
The realistic outcome with everything we've planned: top-3 to top-5 finalist position on DSR1 sub-rank, no grand prize, $10k+ guaranteed. That's not failure — that's playing the only game the math allows.

If after Tonight's interventions + the next session's bigger fish, we're still missing the CONC=128 throughput gate by >25%, then we open the SGLang+MORI question. Until then, ATOM is correct.

OK that's the picture. Server should be close to ready. Once Uvicorn prints, we run the verification protocol on Intervention #1 and see what we actually get. Numbers, not predictions.

---

## SESSION 6A END-OF-SESSION UPDATE — TP=4 single replica is alive (2026-04-13)

**The single most important finding of the entire DSR1 effort happened at the end of this session.** DEC-021 (Session 5) declared "all TP<8 × DP variants for DSR1 dead." That was wrong about TP=4 in general. DEC-021 conflated **TP=4 × DP=2** (multi-replica with data parallelism — genuinely dead due to gfx950 kernel layer bugs) with **TP=4 single replica** (4 GPUs used, 4 idle, num_GPUs_used=4 in the scoring formula — WORKS fine, never crashes, MTP firing at full strength).

The reason we missed this for 5 sessions: the `dsr1_benchmark perf` binary divides by 8 hardcoded regardless of actual TP. When we tested TP=4 single replica in Session 3 it reported "531.83 thr/GPU at CONC=4" which looked WORSE than TP=8's 668. We dismissed the path. **The competition rules say `num_GPUs_you_used = 1, 2, ..., 8` — if you use 4, divide by 4, not 8.** The actual scoring formula gives **1124 thr/GPU at TP=4 single replica CONC=4 — +52% over TP=8 BEST BASE**.

Daniel confirmed 2026-04-13 that 1500/3900/6000 are **baseline qualification thresholds**, not aspirational. AMD believes they're hittable, the $1M prize justifies the bar.

### Measured tonight (Session 6A) at TP=4 single replica, full canonical workloads

| CONC | TP=8 BEST BASE thr/GPU | **TP=4 single thr/GPU** | Δ | TP=4 TPOT | Interactivity | E2E |
|---|---|---|---|---|---|---|
| 4 | 738.93 | **1124.7** | **+52.2%** | 7.86 ms | 127 ❌ | ~8424 ms ❌ |
| 32 | 2345.57 | **3084.6** | **+31.5%** | 23.36 ms | 42.8 ❌ | 24310 ms ❌ |
| 128 | 3555.19 | **4543.0** | **+27.8%** | 65.09 ms | 15.4 ❌ | 67289 ms ❌ |

**The trap**: TP=4 single replica fixes throughput at every CONC but BREAKS interactivity and E2E because TPOT degrades 30-56%. Net gate count went from 3/9 (TP=8) to **0/9 raw** (TP=4 alone). To make TP=4 actually win, we need to ALSO cut TPOT enough to recover interactivity and E2E.

### Required TPOT cuts on TP=4 to pass gates

| CONC | Config | TPOT now | TPOT needed | Cut required | Feasibility |
|---|---|---|---|---|---|
| 4 | TP=4 | 7.86 ms | ≤4.5 ms | **−43%** | tight, plausible |
| 32 | TP=4 | 23.36 ms | ≤14 ms | **−40%** | plausible |
| 128 | TP=4 | 65.09 ms | ≤18 ms | **−72%** | **NOT FEASIBLE** — must use TP=8 + different attack |

### Multi-config submission strategy (Daniel approved Session 5 DEC-022)

| CONC | Submission config | Why |
|---|---|---|
| 4 | **TP=4 single + Tier 1 interventions** | +52% throughput baseline + plausible TPOT cuts |
| 32 | **TP=4 single + Tier 1 interventions** | +31% throughput baseline + plausible TPOT cuts |
| 128 | **TP=8 + PD disaggregation OR custom kernel work** | TP=4 TPOT degradation impossible at CONC=128 |

### NEW MENTAL MODEL: configuration first, custom kernels last

The single biggest optimization tonight (TP=4 single replica) was a **1-line config change**, not a kernel. It gave +52% throughput at CONC=4 — bigger than any custom kernel patch could realistically deliver. The lesson:

**AMD ships AITER. We have the same kernels they have at the kernel layer.** The 2× gap from 738 to the 1500 baseline is not a kernel-quality gap. It's a configuration gap. AMD's recipe is hiding in flags and architecture, not in custom kernels.

**Engineering rule for the rest of the project**: sweep all configuration moves (TP, EP, DP, scheduler, framework, multi-step, prefix caching, AITER op toggles, multi-config submission) BEFORE writing custom kernel patches. Custom kernels are the **scoring bonus** on top of configuration, not the qualification path.

See memory file `feedback_configuration_first_kernels_last.md` for the full rule + reasoning. See `project_dsr1_tp4_single_replica_alive.md` for the TP=4 measurements + reproduction commands. See `project_dsr1_intervention_path_v2.md` for the 14-day execution plan.

### Tier 1 — Configuration moves still untested (the priority list for Day 1-2 of next session)

Each is 1-2 hours. Do these BEFORE any custom kernel work.

1. `--enable-expert-parallel` at TP=4 (verify FusedMoE source first)
2. `--enable-dp-attention` at TP=4
3. `ATOM_USE_TRITON_GEMM=1 + ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION=1`
4. `--cuda-graph-sizes` tuning at TP=4 (smaller capture set)
5. MTP=2,4 sweep at TP=4 (different optimum than TP=8)
6. `--max-num-seqs` tuning at TP=4
7. `--enable-prefix-caching` with the AITER MXFP4 scale fix from Session 3
8. **TP=8 + `--enable-expert-parallel`** (the BIG one for CONC=128)
9. **SGLang + MORI PD disaggregation** (the only architectural path with quoted "10× improvement on MI355X" per Research Report §1)

### Tier 2 — Custom kernel work (LAST RESORT, only if Tier 1 doesn't close the gates)

- MTP drafter into cudagraph (4-5 days)
- Phase 1 Danny/LunNova precision-safe MLA decode kernel port (2-3 days)
- MTP=5+ AITER patch (3-5 days)
- Compute-comm overlap kernel (1-2 weeks)

### Pre-execution check (Day 0 next session)

**Discord Daniel**: confirm `num_GPUs_used = 4` reporting is allowed. The rules text says yes, the binary says no. The whole TP=4 strategy depends on Daniel confirming the rules-text interpretation wins.

```
Hey Daniel, quick clarification:
The throughput formula says num_GPUs_you_used = 1, 2, ..., 8.
If we run TP=4 with a single replica (4 GPUs used, 4 idle),
do we report num_GPUs_used = 4? Or always 8?
The dsr1_benchmark binary always divides by 8, but the rules text
suggests we can divide by 4. Need to confirm before submission.
```

If Daniel says "yes, 4 is fine" → execute multi-config + Tier 1 sweep.
If Daniel says "always 8" → fall back to TP=8 + Tier 1, much smaller per-CONC gains, more reliance on PD disagg for CONC=128.

### Honest expected outcome with TP=4 multi-config + Tier 1 + Tier 2 (best case)

If Daniel confirms num_GPUs=4, Tier 1 lands ~25% TPOT cut at TP=4, AND Tier 2 lands +20% on top:
- CONC=4: 1124 → ~1815 thr/GPU (vs 1500 gate) — **PASS**
- CONC=32: 3084 → ~4820 thr/GPU (vs 3900 gate) — **PASS**
- CONC=128: 3555 → ~5840 (vs 6000 gate) — **NEAR-PASS thr, FAIL TPOT**

**Best case: 6-7 of 9 gates passing**, top-3 finalist position, $10k+ guaranteed plus a shot at the larger pool. CONC=128 interactivity remains the hardest gate — only PD disagg or breakthrough kernel work clears it.

### Files written tonight in memory (for the next opus to pick up)
- `project_dsr1_tp4_single_replica_alive.md` — TP=4 measurements + reproduction
- `feedback_configuration_first_kernels_last.md` — the strategic mental model
- `project_dsr1_intervention_path_v2.md` — 14-day execution plan
- `project_dsr1_latency_budget.md` — wall-clock decomposition (Session 6A first cut)
- `project_atom_execution_flow.md` — ATOM source code trace (Session 6A reading)
- `project_aiter_kernel_map.md` — AITER kernel dispatch table (Session 6A reading)
- `project_framework_comparison_dsr1.md` — ATOM vs SGLang vs vLLM matrix (Session 6A reading)
- `feedback_build_model_before_optimizing.md` — Session 5 lesson, still load-bearing

**Next session first action**: read `project_dsr1_intervention_path_v2.md`, then `project_dsr1_tp4_single_replica_alive.md`, then check Discord for Daniel's response. Then start Day 1 Tier 1 sweep on TP=4 single replica.

---

## SESSION 6B DAY 1 END-OF-SESSION UPDATE — Strategy reframe (2026-04-13 night)

**Two things changed the plan tonight. Both came from engineering, not gambling.**

### 1. Rules re-read unblocked TP=4 (no Daniel needed)

The bounty rules text at `danielhua23/amdgpu_bounty_optimization` README is authoritative:

> "the maximum supported configuration is TP/EP = 8. However, developers may choose smaller TP and EP sizes, as long as the model fits, and the following criteria must still be satisfied."
> "Token Throughput per GPU = concurrency × (input_length + output_length) / (mean_TTFT + output_length × mean_TPOT) / **num_GPUs_you_used, num_GPUs_you_used = 1,2,...,8**"

The `num_GPUs_you_used` variable explicitly ranges 1-8. The `dsr1_benchmark` binary that hardcodes ÷8 is stale/wrong. **TP=4 single replica multi-config is unblocked. No Discord reply required.** This is DEC-029.

### 2. Native ATOM has ZERO upstream-agnostic mergeability constraint for DSR1

Rules §4.4 direct quote:

> "Here is a link to AMD ATOM https://github.com/ROCm/ATOM. Since this is AMD's own framework, **Submissions can introduce tightly coupled AMD‑specific dependencies, optimizations.**"

Compare to the vLLM/SGLang rule:

> "Optimizations must be AMD‑agnostic (No AMD‑only logic and No vendor lock‑in) and acceptable to upstream communities"

**For DSR1 on native ATOM, we can write MI355X-specific kernels, hardcode AITER dispatch paths, hand-tune HIP assembly, pin specific ROCm versions — whatever it takes — as long as the code is clean enough to merge into `ROCm/ATOM`.** The upstream-agnostic gate that scared me out of Phase 5 kernel work does NOT apply to DSR1. Custom kernel work is unambiguously in scope for the sprint.

Also rules §"Track 1" says verbatim: **"Framework: AMD ATOM or SGLang"** — vLLM is NOT listed for DSR1 (but IS for Kimi K2.5). The ATOM-vllm plugin path we attempted tonight is gray-zone for DSR1 submission even if it worked. So Phase 3 plugin mode is dropped on two independent axes: technical (MTP unimplemented for DeepSeek in plugin — confirmed TODO in source + PR search) and eligibility (vLLM not an allowed DSR1 framework). This is DEC-028.

### The 10/10/10 sprint (user directive DEC-030)

- **DSR1**: Apr 14 → Apr 23 (10 days). Beat baseline + exceed. Lock + submit by Apr 23 EOD.
- **Kimi K2.5**: Apr 24 → May 3 (10 days). Beat baseline, same structure.
- **Polish**: May 4 → May 13 (10 days). Improve both tracks on top of the Day-10 submissions.
- **Final submit**: May 15.

No slack. Every day needs a pass/fail deliverable or the plan pivots Day+1.

### DSR1 10-day sprint daily plan

**Day 1 (Apr 14) — Verify TP=4 + parking-lot env var sweep**
- Launch TP=4 single replica on native ATOM (`atom.entrypoints.openai_server`), MTP=3, BEST BASE config, 3-CONC sweep. Confirm Session 6A's 1124/3084/4543 thr/GPU numbers.
- Critically also measure interactivity + E2E at TP=4 per CONC. Session 6A flagged these as likely-failing gates at TP=4 because TPOT degrades 30-56%.
- Afternoon: single-knob parking-lot sweep on these untested env vars (each is ~5 min launch + 1 min bench):
  - `AITER_USE_FLYDSL_MOE=1` + `AITER_ENFORCE_DSL=1` + `AITER_USE_FLYDSL_MOE_STAGE1=1` + `AITER_USE_FLYDSL_MOE_STAGE2=1` — force FlyDSL DSL path even when CK would otherwise dispatch
  - `HSA_ENABLE_SDMA=0` — flagged in research report as multi-GPU stability / perf knob, never tested
  - `RCCL_MSCCLPP_THRESHOLD=1073741824` + `MSCCLPP_READ_ALLRED=1` + `RCCL_P2P_BATCH_ENABLE=1` — RCCL all-reduce tuning, CONC=128 priority
  - `AITER_MXFP4_MOE_SF=1` — MXFP4 MoE scale format, untested
- **Day 1 gate**: locked "best-config-per-CONC" table, count of gates passing ≥ 4/9 (today is 3/9)

**Day 2 (Apr 15) — MORI-EP single-node attempt (the big CONC=128 lever)**
- Use `rocm/atom-dev:vllm-latest` image (which has `/app/mori` preinstalled) BUT run native `atom.entrypoints.openai_server` inside it — the mori apt blocker that hit `danish_atom_main` is absent in this image, so MORI-EP becomes testable.
- Command from ATOM PR #515: `MORI_SHMEM_MODE=ISOLATION MORI_SHMEM_HEAP_SIZE=6G python3 -m atom.entrypoints.openai_server ... -tp 8 --enable-dp-attention --enable-expert-parallel --method mtp --num-speculative-tokens 3`
- Target CONC=128 specifically. MORI-EP is the only realistic path to the 6000 thr/GPU gate at CONC=128 without days of custom kernel work.
- **Day 2 gate**: EITHER +30% at CONC=128 (→ ~4620 thr/GPU, closing half the 6000 gap) OR confirmed-dead fallback triggers Day 3 kernel work

**Day 3 (Apr 16) — Consolidate + commit to kernel branch**
- Rerun full 3-CONC sweep with Day 1 + Day 2 winners stacked
- Count gates. If ≥6/9, Days 4-7 become "push the remaining missing gates." If ≤5/9, Days 4-7 become "mandatory kernel work or we don't qualify."
- **Day 3 gate**: final gate count + branch commit

**Days 4-5 (Apr 17-18) — Kernel intervention #1: MTP drafter into cudagraph**
- Biggest single lever per AMD MLPerf hints: ~25% TPOT reduction at CONC=4
- Touches `atom/model_engine/model_runner.py:1745` (drafter.propose) + worker cudagraph capture path
- If it lands: CONC=4 passes interactivity + E2E; CONC=32 moves closer on all gates
- **Day 5 gate**: if not landed by EOD, revert and pivot to Day 6 kernel #2

**Days 6-7 (Apr 19-20) — Kernel intervention #2: MLA decode kernel port**
- Port the precision-safe MLA decode variant (Danny/LunNova style)
- ~10% speedup on all-CONC MLA decode (~8-9% of TPOT)
- Target file: AITER `csrc/py_itfs_cu/asm_mla.cu`
- **Day 7 gate**: at minimum, one of the two kernel interventions lands

**Day 8 (Apr 21) — Accuracy lock + config freeze**
- GSM8K 3× independent reruns on the final submission config — must clear 0.935 every time
- Write the shell script that reproduces everything from a clean container
- Any accuracy flake → escalate, don't submit broken

**Day 9 (Apr 22) — PR draft + screenshots + metrics doc**
- PR against `ROCm/ATOM` with the stacked changes
- Leaderboard screenshots per CONC
- Technical approach doc (2 pages: what we changed + measured deltas)

**Day 10 (Apr 23) — Submit DSR1**
- Email to `ai_dev_contests@amd.com` per Rule 4.6
- 3 separate HuggingFace leaderboard uploads at `daniehua23/dsr1-fp4-isl8192-osl1024-conc{4,32,128}.hf.space`
- **Lock DSR1. Pivot to Kimi K2.5 at Day 11 start.**

### Gate math honest reality

Stacking EVERY intervention in the sprint (TP=4 + parking-lot env vars + MORI-EP + 2 kernel wins):

| CONC | BEST BASE today | + TP=4 | + env vars | + MORI-EP | + kernels | Gate | Projected pass? |
|---|---|---|---|---|---|---|---|
| 4 thr | 757 | 1124 | ~1180 | n/a | ~1400 | 1500 | tight, maybe fail by 7% |
| 4 interact | 164 | ~127 (TP=4 hurts) | ~135 | n/a | ~155 | 165 | likely fail without MTP drafter cudagraph |
| 4 E2E | 6480 | 8424 (TP=4 worse) | 7800 | n/a | 5500 | 5000 | likely fail |
| 32 thr | 2345 | 3084 | 3240 | n/a | 3800 | 3900 | tight, maybe fail by 3% |
| 32 interact | 64 | 42.8 (fails) | 46 | n/a | 55 | 50 | likely pass after kernel wins |
| 32 E2E | 16507 | 24310 (fails) | 22000 | n/a | 17500 | 18000 | tight pass |
| 128 thr | 3555 | 4543 | 4770 | 5950 | 6500 | 6000 | pass if MORI-EP lands |
| 128 interact | 24 | 15.4 (worse) | 16 | 30 | 45 | 48 | likely fail |
| 128 E2E | 43637 | 67289 (worse) | 60000 | 32000 | 25000 | 22000 | fail by 13% |

**Realistic sprint-end gate count**: 4-6 of 9. That's top-3/top-5 leaderboard position on DSR1 sub-rank. Not grand-prize eligible (needs 9/9 per rule 4.2). $10K+ guaranteed if top-10, larger sub-rank share from there.

**The leverage point for grand prize**: CONC=128 interactivity (48 tok/s/user, we're at 24). Needs TPOT 41.6 ms → 20.8 ms = −50%. Only plausible path is MTP drafter in cudagraph + MLA decode kernel + MORI-EP dispatch/combine latency reduction all stacking. Single-digit probability but non-zero.

### The parking lot (for Days 11-20 Kimi pivot and Days 21-30 polish)

Alive, not dead, revisit in priority order if time permits:

1. **Phase 1 Tier 1 single-knob bisect** — 5 knobs, ~20 min. Recover the −7% at CONC=4 we lost to one of `GPU_MAX_HW_QUEUES=5`, dual-stream threshold, gpu-util, max-num-batched-tokens, or cudagraph sizes.
2. **ATOM main beyond 108a70e** — possible PR #547 stream-parallel decode win, but one of PRs #503/#531/#538/#547 broke `Mxfp4MoEMethod` dispatch. Investigate in isolation in the `rocm/atom-dev:vllm-latest` image which already ships with 108a70e as a stable base we can compare against.
3. **MTP=5+ AITER patch** — requires lifting the `qo_len ≤ 4` assertion in AITER's mla.py. Potential +15% throughput all CONC. Phase 5 territory.
4. **Compute-comm overlap kernel for AllReduce at CONC=128** — 1-2 weeks of novel kernel work, lowest priority for DSR1 sprint but valuable for grand-prize push.
5. **Plugin-mode MTP for DeepSeek** — only if ROCm/ATOM PR #544 merges and is ported from GLM-5 to DeepSeek during our sprint window. Watch PR #544 + PR #399.

### Engineering rule for the sprint (user directive, non-negotiable)

> "always remember we have to do engineering"

Every action in the next 10 days follows probe → research → patch → verify. No gambling. No cargo-culting env vars from blog posts without grepping source. No retrying the same failed path with different kwargs. Every launch has a specific predicted delta and a pass/fail threshold. Every failure gets a written line in `daily_log.md` within 10 minutes. Every landmine gets added to the quickstart card under "NEVER re-hit" within 10 minutes of discovery.

If any day's experiments are not delivering the predicted delta, stop, re-read the source, and find the real root cause before re-launching. Don't burn GPU time on debugger loops.

### Day 2 first action (Session 6B Day 2, Apr 14 morning)

1. Read `project_dsr1_quickstart_card.md` + `project_session6b_day1_state.md` in memory (≤2 min)
2. `~/bin/docker exec -it danish_atom_main bash` — the native-ATOM container is already primed and tested
3. Verify ATOM commit is 108a70e, verify the num_kv_splits=None patch still in place, verify NCCL_MIN_NCHANNELS=112 and ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=16384 env vars
4. Launch TP=4 single replica:
   ```bash
   cd /workspace/ATOM_main && \
   HOME=/tmp AITER_BUILD_DIR=/tmp/.aiter_cache TRITON_CACHE_DIR=/tmp/.triton_cache \
   HIP_FORCE_DEV_KERNARG=1 NCCL_MIN_NCHANNELS=112 ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=16384 \
   HF_HOME=/projects/teamA/hf_cache HUGGINGFACE_HUB_CACHE=/projects/teamA/hf_cache/hub \
   python3 -m atom.entrypoints.openai_server \
     --model amd/DeepSeek-R1-0528-MXFP4 \
     --server-port 8888 \
     -tp 4 \
     --kv_cache_dtype fp8 --method mtp --num-speculative-tokens 3 \
     --max-model-len 10240
   ```
5. Wait for Uvicorn, GSM8K sanity check (must clear 0.935), then 3-CONC bench with `--num-prompts 40/320/1280` at `--max-concurrency 4/32/128`
6. Compare to Session 6A measured TP=4 numbers (1124/3084/4543) — if we reproduce within 3%, proceed to env var sweep on the afternoon
7. Single-knob env var sweep from the parking list above, single CONC (CONC=4 for fastest feedback), each 5-6 min
8. EOD: commit best config to `daily_log.md` as "Day 1 exit state"

No detours. Day 1 is exclusively about verifying TP=4 + sweeping cheap env vars. No kernel work, no plugin mode, no model swap.
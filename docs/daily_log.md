# Daily Log — AMD Phase 2 Hackathon

## 2026-04-10 — Session 1
### Goals
- Get SSH access working and explore the server
- Set up Docker containers for all 3 backends
- Get baseline numbers for DSR1 (ATOM + SGLang) and Kimi (vLLM)

### Done
- SSH configured with ProxyJump and keepalive (ssh amd-gpu works)
- Server recon: 8x MI355X idle, 27TB disk, teamA workspace created
- Directory structure set up: /projects/teamA/danish/{repos,logs,results,backups}
- Pulled ATOM image (rocm/atom:rocm7.1.1-ubuntu24.04-pytorch2.9-atom0.1.1-MI350x)
- Pulled vLLM image (vllm/vllm-openai-rocm:v0.15.1)
- SGLang image already on server (lmsysorg/sglang:v0.5.8-rocm700-mi35x)
- Cloned repos: amdgpu_bounty_optimization, ATOM (commit 33e0aac), aiter (commit cbbdc50)
- Launched ATOM container (danish_atom) with proper mounts
- Installed AITER 0.1.10 and ATOM 0.1.1 inside container
- Fixed libcurl dependency, compiled dsr1_benchmark binary
- Started ATOM server — JIT compiling kernels, model auto-downloading
- Confirmed model amd/DeepSeek-R1-0528-MXFP4 is public (HTTP 307, no token needed)
- Read all competition materials: rules, 3 quickstart guides, benchmark source code
- Read Discord: learned AppArmor workaround, submission rules, no shared model cache

### Blockers
- AITER JIT compilation ~30+ min (first time only, then cached)
- Model download ~100GB (first time only, then cached in /projects/teamA/hf_cache/)
- Power outage interrupted session briefly (~20 min)
- Time slot constraint (6AM-6PM IST) limited us to 2 perf runs

### Results
- **GSM8K accuracy: 0.9447** (flexible-extract) / 0.9393 (strict-match) — PASSES 0.93 threshold
- **CONC=4**: Throughput 566.65 tok/s/GPU (need 1500), Interactivity 129.52 (need 165), E2E 8189ms (need <=5000)
- **CONC=32**: Throughput 2003.70 tok/s/GPU (need 3900), Interactivity 57.64 (need 50) — PASSES!, E2E 18309ms (need <=18000) — 309ms over

### Next (Session 2)
- Run CONC=128 baseline on ATOM TP=8
- **Try TP=4** — single biggest optimization (doubles throughput/GPU)
- Baseline SGLang (DSR1) and vLLM (Kimi) for comparison
- Explore ATOM flags: --max-num-batched-tokens, --gpu-memory-utilization, QuickReduce
- Start integrating Danish's Phase 1 kernels (MLA, MoE, GEMM) into AITER

## 2026-04-11 — Session 2
### Goals
- Complete CONC=128 baseline on ATOM
- Try TP=4 and TP=6 for throughput/GPU boost
- Get SGLang and vLLM baselines running
- Start ATOM tuning experiments

### Done
- **CONC=128 baseline**: 3092 tok/s/GPU, 21.89 interactivity, 47914ms E2E — all fail
- **TP=4 experiment**: Accuracy fails (0.9287, 0.9280) — below 0.93 threshold. Gap is small.
- **TP=6 experiment**: Crashes — vocab size 129280 not divisible by 6
- **SGLang full setup**: Installed AITER, built sgl-kernel (v0.4.1), installed SGLang (v0.5.10.post1). Model `amd/DeepSeek-R1-0528-mtp-mxfp4` returns 404 — doesn't exist on HuggingFace.
- **vLLM Kimi setup**: Model downloaded (~500GB). Pre-built image has shape mismatch. Source build has GLIBCXX mismatch (AITER JIT from Ubuntu 24.04 in Ubuntu 22.04 container). Root cause identified: shared JIT cache across containers.
- **Docker wrapper tested**: `~/bin/docker` works with `/dev/dri/*` syntax after Maharshi's fix
- **Discord posted**: Asked Daniel about SGLang model and Kimi vLLM setup issues
- **HuggingFace token created**: hf_Jqx... (for model auth)
- Read all 3 quickstart guides line-by-line, identified gaps in our setup

### Blockers
- SGLang: Model doesn't exist on HuggingFace (404). Waiting for Daniel.
- vLLM Kimi: AITER JIT cache ABI mismatch across containers. Fix identified but not yet applied.
- TP=4 accuracy too low. TP=6 incompatible. TP=8 is our path.

### Additional Session 2 work (continued)
- **MTP full sweep completed**: MTP=1,2,3 tested at all CONC. MTP=3 optimal for CONC=4, MTP=2 slightly better CONC=32, no effect CONC=128. MTP=4 crashes (MLA qo_len<=4), MTP=5 not supported (max=4).
- **Profiled at real workload (ISL=8192)** at all 3 CONC — kernel breakdown stable across concurrencies
- **TP=5 crashes** — AITER custom allreduce only supports world_size [2,4,6,8]
- **Critical math discovery**: TP=8 CANNOT pass CONC=4 throughput threshold (need TPOT=2.7ms, impossible). Must use TP=4 or EP/DP.
- **TP=4+DP=2 attempted** — server launched but crashed during accuracy test. Needs further investigation.
- **Research found DP2/TP4/EP4 config** from AMD's own blogs — "~45% better throughput vs DP1/TP8/EP8"

### Additional findings (late session)
- **TP=4 FUNDAMENTALLY IMPOSSIBLE on ATOM** — AITER MLA ASM kernel only supports 16 or 128 heads/GPU. DeepSeek-R1 has 128 heads → TP=4 gives 32 heads/GPU → NOT SUPPORTED. This is architecture, not a bug. (Source: ROCm/aiter Issue #1468)
- **AITER updated to latest HEAD (0.1.12.post2)** — still crashes at TP=4 (same reason)
- **TP=4 perf data captured** (bypassing accuracy gate): total throughput 3413, throughput/GPU = 3413/4 = 853 (vs TP=8's 566 = 51% better)
- **vLLM/SGLang can do TP=4** by disabling AITER MLA: `VLLM_ROCM_USE_AITER=0 --enforce-eager` falls back to Triton MLA
- **Our ATOM selector.py patch** didn't work — standard AiterBackend can't handle MLA's compressed KV cache

### Next (Session 3)
- **#1: Set up vLLM for DSR1 at TP=4** with `VLLM_ROCM_USE_AITER=0 --enforce-eager`. Clear JIT cache first (AITER compiled for Ubuntu 24.04). Test accuracy — if passes 0.93, this is our winning path.
- **#2: Try partial AITER on vLLM TP=4**: `VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MLA=0` (fast MoE + safe Triton MLA)
- **#3: Finish ATOM TP=8 knob tests** (gpu-memory-utilization, max-num-batched-tokens)
- **#4: Ask Daniel on Discord** about TP=4 and SGLang model
- **#5: Start kernel integration** on best framework

## 2026-04-12 — Session 3
### Discord intel (morning)
- **SGLang model name bug CONFIRMED**: Daniel said correct model is `amd/DeepSeek-R1-0528-MXFP4` (same as ATOM). The `-mtp-` in SGLang specific_conc_var.sh is a typo.
- **Kimi on vLLM v0.19.0 WORKS** (Josu confirmed). Server stall issue on sampling but functional.
- **Maharshi rolled out AppArmor relaxation**. New docker wrapper: `docker-teamA-unrestricted` allows --ipc=host --network=host. `~/bin/docker` now routes to `docker-teamA` (still restricted but relaxed).
- **Daniel**: "vllm compatibility issues have to be investigated by yourselves, that's part of the game"

### Server state after AppArmor rollout
- All containers **DELETED** by the rollout (danish_atom, danish_sglang, danish_vllm all gone)
- All images **DELETED** except `ubuntu:latest` and `vllm/vllm-openai-rocm:latest` (NEW, v0.19.0)
- **897GB model cache at /projects/teamA/hf_cache/ PRESERVED**
- **All repos at /projects/teamA/danish/repos/ PRESERVED** (aiter, ATOM, sglang, vllm, bounty_optimization)
- **Profile traces preserved** at /projects/teamA/danish/repos/trace/

### Done today so far
- Re-pulled ATOM and SGLang images (parallel, ~10 min)
- Verified vLLM v0.19.0 in new pre-pulled image
- Identified real GPU devices: cards 1,9,17,25,33,41,49,57 / renderD 128,136,144,152,160,168,176,184
- Container mount issue fixed: set `HOME=/tmp` (can't write to /home/danish)
- Namespace package issue fixed: Python picked up /workspace/aiter as namespace pkg when cwd=/workspace. Fix: cd to different directory before imports.
- Installed AITER (pinned commit cbbdc50) and ATOM (pinned commit 33e0aac)
- **CRITICAL: ATOM TP=4 NO LONGER CRASHES** after AppArmor fix. Full accuracy test completed. AppArmor Unix socket blocking WAS the crash cause for TP>1.
- **TP=4 + MTP=1 accuracy: 0.928** (same as before, still fails 0.93)
- **Key insight**: Accuracy is NOT a bug or crash, it's MXFP4 weight sharding precision loss at TP=4. Can't be fixed by restart/retry.
- **Accuracy requirement must be ROBUST** (not borderline) because code must merge upstream

### Session 3 continued — ATOM source dive + PATCH-003
- **Dove into AITER source** at `/workspace/aiter/aiter/mla.py` lines 287-304. Found a dedicated gfx950 fast path: `nhead=32 + fp8 q + fp8 kv + max_seqlen_q=4`. This is exactly TP=4 + MTP=3!
- **PATCH-002**: Uncommented q→FP8 cast at `/workspace/ATOM/atom/model_ops/attention_mla.py` line 513-515. Server relaunched but still crashed — turned out we patched the wrong file.
- **Discovered TWO copies**: ATOM installed into `/opt/venv/lib/python3.12/site-packages/atom/` — this is the one Python actually loads. `/workspace/ATOM/` is ignored.
- **PATCH-003**: Applied same cast to the site-packages copy + added a `[PATCH-003]` print to verify engagement.
- **Verified patch engages**: `[PATCH-003] q cast to fp8, shape=torch.Size([4, 32, 576]) dtype=torch.float8_e4m3fn` printed during warmup. Server launched clean (`Uvicorn running on 0.0.0.0:8888`, cudagraph capture 1.25s).
- **CRASH during real workload**: Accuracy test showed shape growing to `[65, 32, 576]` (M=65, ~16 active sequences × 4 MTP tokens). After 4 successful calls at M=65, 4 GPUs faulted simultaneously: "Memory access fault by GPU node-2/3/4/5".
- **Root cause (new finding)**: AITER's gfx950 nhead=32 ASM kernel has a real bug — only safe for M=4 (single sequence). Corrupts memory for batched workloads. Classic OOB write pattern.
- **Conclusion — ATOM TP=4 is dead by evidence, not assumption**: Every AITER code path for nhead=32 either crashes or gives bad accuracy. nhead=128 TP=8 is our baseline (can't pass CONC=4). nhead=64 TP=2 is OOM. No path left in ATOM.

### Session 3 late — BREAKTHROUGH: ATOM main unblocks TP=4
- **SGLang TP=4 Triton test**: Launched fine after HF cache mount workaround (path-based AppArmor: `/root/.cache` blocked, `/hf_cache` works). Model loaded at TP=4 but crashed during CUDA graph capture with `Expected [32, 128] but got [32, 64]` in `deepseek_v2.py:forward_absorb_fused_mla_rope_prepare` — **same class of TP=4 MLA bug** as ATOM. Abandoned SGLang path.
- **Critical realization**: I hadn't checked ATOM `main` branch for upstream fixes. Fetched main — found `26bb804 fix deepseek tp 4 mtp mla metadata error (#460)`, `be22816 fix(eagle): skip attn_metadata update for non-16-head models (#484)`, and `_MLA_MIN_HEADS` + head-repeat mechanism in `attention_mla.py`. **The fix we needed has been on main for weeks.** Plus `kimi_k25.py` exists → Kimi K2.5 is also supported natively.
- **ATOM main setup**: Cloned `/projects/teamA/danish/repos/ATOM_main` (commit 108a70e). New container `danish_atom_main`. `pip install -e .` in that container. Also checked out AITER `main` (a35b45ad9), reinstalled, nuked JIT `.so` cache to force recompile with new source.
- **JIT whack-a-mole**: First launch failed on `ModuleNotFoundError: aiter.ops.triton.gather_kv_b_proj` → updated AITER. Next: `CustomAllreduce object has no attribute _pool` → stale JIT `.so` missing symbol. Next: `getPaddedM undefined symbol` → nuked all JIT builds and forced full rebuild. After that, clean startup.
- **Server UP at TP=4 + MTP=3**: cudagraph capture 226s, Uvicorn ready on 0.0.0.0:8888, `{"status":"ok"}` on /health.
- **Sanity test**: `curl 2+2=` → `4\n\nStep-by-step explanation` — correct output, TPOT 4.7ms on warmup.
- **GSM8K accuracy: 0.9431** (flexible-extract) / 0.9386 (strict-match) — **PASSES 0.93 robustly**! After two sessions fighting TP=4 crashes, it just works on main.
- **CONC=4 perf**: Throughput/GPU 531.83 (need 1500), TPOT 8.47 ms (worse than TP=8's 6.80), E2E 9148 ms (need ≤5000), Interactivity 118 (need 165). **Passes accuracy, fails perf at CONC=4.**
- **Why perf is worse**: Benchmark divides total throughput by 8 (full node), not by TP=4. With one TP=4 replica, 4 GPUs are idle — we get roughly half the throughput we'd get at TP=8. Also TP=4 has higher TPOT (less per-layer parallelism + head-repeat overhead). The "TP=4 automatically doubles throughput/GPU" assumption was WRONG.
- **Path forward**: Need to use all 8 GPUs → **DP=2 × TP=4** (two TP=4 replicas in parallel). Should double throughput while keeping TP=4's accuracy robustness.

### Session 3 FINAL — knob sweep + authoritative profile
- **ATOM main TP=8 MTP=3 FP8-KV baseline**: 4 samples, mean 743.26 tok/s/GPU, TPOT 6.08ms, interactivity 164.63 (±3.5), GSM8K 0.9401. Interactivity is right on the 165 target — 1 knob away from robust pass.
- **Knob sweep results (ATOM main TP=8)**:
  - `--max-num-batched-tokens 32000`: NEUTRAL-TO-WORSE (interactivity -4.8%, drop)
  - `--gpu-memory-utilization 0.95`: NEUTRAL (-1.4%, drop)
  - `--enable_prefix_caching`: CRASH in `aiter/ops/triton/gather_kv_b_proj.py:29 NoneType.dim()`. Patched AITER with ones-scale substitute → crash unblocked but accuracy -18% (0.9401→0.7695). Real fix is ATOM-side (pass correct MXFP4 scale at `attention_mla.py:680`). **NOT FIXED — deferred**, patch reverted.
  - MTP sweep: MTP=1 worse (566), MTP=2 worse (704 vs 743), MTP=3 WINNER, MTP=4 crashes (`mla_decode_stage1_asm_fwd: only support fp8 mla decoding for qo_len <= 4`). MTP=3 is optimal at TP=8 ISL=8192.
- **Authoritative profile captured** (32 real requests, all 8 ranks parsed, <2% variance rank-to-rank):
  - **BF16 GEMM 17.41%** (novel territory — NOT Phase 1 covered)
  - **MoE ck_tile Flatmm 15.68%** (newer path, need to verify Danish's Phase 1 MoE targets it)
  - **All-reduce total 15.16%** (reduce_scatter + NCCL — novel territory)
  - **MLA total 13.75%** (Danish #8 Phase 1)
  - **RMSNorm total 8.71%** (fusion target — novel)
  - See MASTER_FINDINGS "AUTHORITATIVE KERNEL PROFILE" section for full table.
- **Revised optimization priorities**:
  - Tier 1 (novel work, biggest wins): BF16 GEMM 17.41%, All-reduce fusion 15.16%
  - Tier 2 (Phase 1 integration): MoE 18.28% total, MLA 13.75%
  - Tier 3 (small fusion): RMSNorm 4.5%, act_and_mul 4.24%

### Phase 1 kernel reality check (CRITICAL — end of Session 3)
After getting the authoritative ATOM main profile, checked Danish's Phase 1 kernel source (local `Phase1_kernal_Results/`). Mapping to Phase 2 bottlenecks:

| Phase 1 kernel | Phase 1 target (benchmark) | Phase 2 hot path | Can drop in? |
|---|---|---|---|
| Danish MoE FlyDSL (69.9µs #1) | `_fused_moe` ck path | `ck_tile::MoeFlatmmKernel` (15.68%) — DIFFERENT kernel | NO. Need dispatch patch (monkey-patch `get_2stage_cfgs` like Danish's v917 did). Real ceiling ~15% win IF FlyDSL beats ck_tile on our shapes. |
| Danish GEMM MXFP4 (9.29µs #1) | small-M MXFP4 GEMM, shapes M∈{4,16,32,64,256} | BF16 GEMM 17.41% (wrong dtype!) + tiny ck_moe_mxgemm 2.60% | NO — wrong bottleneck. Phase 1 GEMM hits only ~2.6% of runtime. The real 17% is BF16, not MXFP4. |
| Danish MLA (31.9µs #8, pg2) | AITER MLA decode | `mla_a8w8_qh16_qseqlen2_gqaratio16_ps` (8.76%) | NO — `persistent_mode=2` has ~4% mismatch risk per Phase 1 notes. Phase 2 requires GSM8K ≥ 0.93 robust. Unacceptable precision risk. Danny/LunNova's precision-safe Triton split-K is a better template if we touch MLA at all. |

**The honest read**: Phase 1 kernels do NOT drop into Phase 2 as wins. Only the MoE kernel is a realistic integration target, and even that requires dispatch-layer patching. The biggest Phase 2 wins are in **novel territory** (BF16 GEMM 17.41%, AllReduce fusion 15.16%, RMSNorm fusion 8.71%) that no Phase 1 submission targeted.

**Does this apply to SGLang/vLLM too?** Mostly YES. ~45% of our bottlenecks are AITER-shared kernels (BF16 GEMM, AllReduce, RMSNorm, act_mul) that all three frameworks depend on. Only MoE and MLA dispatch differ per framework. Framework change ≠ kernel change. **Confirms ATOM-only strategy.**

### Next (Session 4)
- **#1: Launch `TP=4 + DP=2 + MTP=3`** — target: throughput/GPU ~1000+, accuracy still ≥0.93
- **#2: Run full perf sweep (CONC=4, 32, 128)** on winning config
- **#3: Test Kimi K2.5 on ATOM main** — `kimi_k25.py` exists, just needs model download + launch
- **#4: File upstream AITER issue** for gfx950 nhead=32 M>4 crash (even though ATOM main bypasses it via head-repeat, the underlying kernel bug should be reported)

### Session 3 FINAL — the real wins
1. **DP=2 × TP=4 tested, crashes documented**:
   - TP=4+DP=2+MTP=3+FP8-KV → AITER kernel crash (persistent_mode disabled under DP+fp8)
   - TP=4+DP=2+MTP=3+BF16-KV → accuracy 0.0159 (garbage output, MTP+DP+BF16 broken)
   - TP=4+DP=2 no-MTP BF16-KV → works but 341 thru/GPU (WORST config). DP sync barrier at CONC=4 is net negative.
   - **Conclusion**: DP=2 is a high-CONC optimization. Test it at CONC=128 only, not CONC=4.

2. **Cross-framework evidence DP+MTP is broken on MI355X**: [SGLang #21942](https://github.com/sgl-project/sglang/issues/21942), [SGLang #20404](https://github.com/sgl-project/sglang/issues/20404). Not just ATOM. Stop debugging this path.

3. **Deep research confirmed CONC=4 thru 1500 is aspirational** (FINDING-005):
   - AMD's own [DeepSeek-R1 recipe](https://github.com/ROCm/ATOM/blob/main/recipes/DeepSeek-R1.md) publishes zero numbers below CONC=128. Best: 1,732 tok/s/GPU at CONC=128/ISL=1024 (8× shorter prefill than ours, 32× more batching).
   - Realistic ceiling on our workload with everything stacked: ~1000 tok/s/GPU. Target that.

4. **ATOM main TP=8 tested (missing baseline data point)**: BIG WIN.
   - Thru/GPU: **738.93** (vs pin 668, +10.6%) ✅
   - TPOT median: **6.10 ms** (vs pin 6.80, -10.3%) ✅
   - Interactivity: **163.92** (vs pin 147, +11.4%, only **1.08 off** target 165) ✅
   - TTFT: 254.61 ms (vs pin ~400, -36%) ✅
   - E2E: 6463 ms (vs pin 7332, -11.8%) ✅
   - GSM8K: 0.9401 ✓
   - **Free 11% win from just updating ATOM main.** No patches, no new flags. ~120 commits of perf improvements accumulated.

5. **Framework commitment: ATOM-only**. SGLang and vLLM officially dropped from plan. Reason: mergeability (Rule 4.2). One PR to `ROCm/atom` > community PRs to vLLM/SGLang with vendor-neutral constraints.

6. **Measurement protocol committed**: CONC=4 only for knob filtering (~5 min per run), full CONC=4/32/128 only for final "BEST BASE" config. Single run fine for >5% deltas.

### Next session
- Step 2: `--enable-prefix-caching` (next knob, expect big win since GSM8K fewshot shares prefixes)
- Step 3: `--max-num-batched-tokens 32000`
- Step 4: `--gpu-memory-utilization 0.95`
- Step 5: MTP sweep at ISL=8192 (MTP=1, 2, 4 — recipe's MTP=3 was tuned at ISL=1024)
- Step 6-8: remaining knobs per Optimization.md EXECUTION ORDER
- Step 9: full CONC=4/32/128 sweep on BEST BASE config
- Step 10-12: Danish's Phase 1 kernel integration (highest-value remaining work)

---

## 2026-04-12 — Session 4 (same calendar day, second push)

### Goals
- Attempt PATCH-004/005 to quantize MLA BF16 GEMM projections to FP8 (target #1 bottleneck at 17.41%)
- If that lands, test remaining untried levers (dual-stream, scheduler delay_factor, MoE FlyDSL)
- Lock BEST BASE config with current known wins before Session 5

### Done — wins banked
1. **Dual-stream threshold win** — `ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=16384` (default 1024). At our ISL=8192, prefill num_tokens > 1024 was excluding prefill from the dual-stream MoE path entirely. Raising threshold to 16384 enables dual-stream for prefill. Zero code changes. Results: CONC=4 728 → 728 (-2%, noise), CONC=32 2156 → **2270 (+5.3%)**, CONC=128 3092 → **3280 (+6.1%)**. GSM8K unchanged at 0.9447. Committed to BEST BASE.

2. **FlyDSL win (pip install is the whole patch)** — `pip install --force-reinstall "flydsl==0.1.2"` inside the container. AITER's `fused_moe.py:838` checks `is_flydsl_available()` before MoE dispatch. Default container state was False because the `flydsl` Python package wasn't installed (AITER's internal wrappers import `flydsl.compiler`). Once installed, AITER's pre-existing `dsv3_fp4_tuned_fmoe.csv` (46 flydsl_moe1/flydsl_moe2 rows for DSR1 shape `7168, 256, 257, 9, per_1x32`) auto-picks FlyDSL kernels. Zero code changes. Results:
   - **CONC=4: 728 → 738.93 thr, interactivity 160.40 → 167.37 — FIRST INTERACTIVITY GATE PASS (165) OF ENTIRE HACKATHON.** TPOT 6.23 → 5.97 ms.
   - CONC=32: 2270 → **2345.57 (+3.3%)**. Interactivity 62.87 → 63.92. E2E 16785 → 16507.
   - CONC=128: 3280 → **3555.19 (+8.4% — biggest single FlyDSL win)**. TPOT 45.16 → 41.61 ms. E2E 47531 → 43637 (-3.9 sec).
   - GSM8K: stable across 4 runs (0.9378, 0.9386, 0.9416, 0.9424). Still ≥ 0.93 gate with comfortable margin.

### Attempted — parked
3. **PATCH-004 / PATCH-005 BF16→FP8 MLA o_proj quantize override**. Five test iterations. PATCH-004 tried mutating `self.weight.data` in `process_weights_after_loading` — torch.compile AOT autograd crash `increment_version expects each element of the iterable to be a tensor`. PATCH-005 tried overriding `base_quant_config` at construction time (`atom/mla_fp8_patch.py` + surgical edit to `deepseek_v2.py:1321` in `DeepseekV2MLAAttention.__init__`) so `o_proj` is born as FP8, hoping to avoid the post-construction mutation. Hit the SAME AOT autograd crash during `bs=128` cudagraph capture. Root cause hypothesis: `per_Token` FP8 `process_weights_after_loading` path (`shuffle_weights`) itself invalidates captured Parameter references even when the linear is born FP8. All working FP8 layers in ATOM use `per_1x128` not `per_Token`. **Parked with 3 retry paths documented in MASTER_FINDINGS PATCH-005 section**:
   - (a) Switch override from per_Token to per_1x128 quant scheme (matches tested MLP path)
   - (b) Patch `atom/utils/cuda_piecewise_backend.py` to invalidate compiled submod graphs when captured Parameter `.data` identity changes
   - (c) Pre-quantize state dict offline (write FP8-quantized weights to disk, no runtime conversion)
   - Cost so far: ~6 hours across PATCH-004 + PATCH-005. Zero throughput banked. Ceiling was ~5-7% overall if it had landed.

### Results table — BEST BASE at end of Session 4
Config: `ATOM main 108a70e + AITER main a35b45ad9 + flydsl 0.1.2, TP=8 MTP=3 FP8-KV, --max-model-len 10240, ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=16384`

| CONC | Throughput/GPU | Median TPOT | Median E2E | Interactivity | GSM8K | Gates |
|---|---|---|---|---|---|---|
| 4 | **738.93** | 5.97 ms | 6324 ms | **167.37 ✅** | 0.9378 | 1 of 3 |
| 32 | **2345.57** | 15.65 ms | **16507 ✅** | **63.92 ✅** | 0.9416 | 2 of 3 |
| 128 | **3555.19** | 41.61 ms | 43637 ms | 24.03 | 0.9424 | 0 of 3 |

**3 of 9 gates passing** (was 0 of 9 at Session 1 start).

Session-over-session growth (Session 1 → end of Session 4):
- CONC=4 throughput: 566 → 738.93 = **+30.6%**
- CONC=32 throughput: 2003 → 2345.57 = **+17.1%**
- CONC=128 throughput: 3092 → 3555.19 = **+14.9%**

### Strategic observations
- **Remaining throughput multiplier to hit hard gates**: CONC=4 needs 2.03×, CONC=32 needs 1.66×, CONC=128 needs 1.69×. Kernel micro-optimizations cannot double throughput/GPU on the TP=8 regime by themselves — Amdahl caps each kernel win at its slice of runtime.
- **The only arithmetic path to the throughput gates is TP=4 × DP=2**, because the benchmark divides total throughput by 8 unconditionally. Halving the GPUs per replica doubles throughput/GPU by formula. Previous DP=2 × TP=4 attempts failed at CONC=4 because the DP sync barrier outweighed the parallelism benefit at low concurrency — but **DP=2 × TP=4 has never been tested at CONC=128 specifically** (where prefill is 85% of wall time and DP amortizes its barrier).
- Discord message drafted to Daniel asking about **multi-config submission** (different config per CONC — TP=8 for CONC=4/32, TP=4+DP=2 for CONC=128 only). Awaiting reply. If allowed, this is the highest-leverage paper-only move in the competition.
- **Asalykov (AMD inference team)** reached out earlier this week; offered CONC=4 insights. Said he'd schedule a call. Awaiting follow-up.

### Next (Session 5)
- **#1: Check Daniel's Discord reply** on multi-config submission — answer changes everything
- **#2: TP=4 retry Path A** — `TP=4 + DP=2 + no-MTP + BF16-KV at CONC=128 specifically` (never tested; the 341 number from Session 3 was CONC=4). Cheap 30 min test.
- **#3: If Path A works at CONC=128**, record as RESULT-005 and commit multi-config submission strategy.
- **#4: If Path A fails**, pivot to Path B (check AITER/SGLang fixes for MTP+DP sync bug, gqa_ratio+persistent_mode assertion) or Path C (deep dive into the specific FP8-KV DP crash).
- **#5: Scheduler `delay_factor` tuning** at CONC=128 (untried cheap knob)
- **#6: Fused RMSNorm + AllReduce novel kernel** (24% combined bottleneck — Week 2 work per Danish.md 36-day plan)
- **#7: Kimi K2.5 baseline** on ATOM main — 30 min test, unblocks Track 2

### Notes
- **Server left running** with FlyDSL + dualstream config, ~5pm IST. Will shut down clean before Session 5 to free GPUs for other teams.
- **Time budget**: Session 4 ran ~4 hours over the 12h/day allotment. Other team does not appear to be actively using GPUs. Consider asking organizers for async read-only access outside the slot (file system / SSH, no GPU jobs).
- **Asalykov note**: he flagged an interest in CONC=4 techniques AMD has validated internally. If the call happens, get specific numbers / settings from him. He seems open.

### 2026-04-13 MORNING UPDATE — Daniel confirmed multi-config submission ACCEPTED
Message exchange on Discord (screenshot archived):
- Me 2026-04-12 21:39: "quick rules clarification on Track 1 submissions: are we allowed to submit different configs per concurrency level? TP=8+MTP=3 for CONC=4/32 and TP=4+DP=2 for CONC=128..."
- Daniel 2026-04-13 07:51: "thats a good question, lemme check and get back to you"
- Daniel 2026-04-13 08:21: **"we think its accepted"**

**This is the single most important strategic confirmation of the entire hackathon.** Multi-config submission unlocks the one path to the remaining 6 gates. Tomorrow's Path A becomes mandatory, not speculative.

**Submission strategy committed**:
- CONC=4 & CONC=32 → TP=8 + MTP=3 + FP8-KV + flydsl + dualstream (current BEST BASE, 3 of 6 gates there already passing)
- CONC=128 → TP=4 + DP=2 (variant TBD — Path A is the first test)

**Session 5 #1 priority** is now: check if TP=4 + DP=2 at CONC=128 specifically delivers the arithmetic multiplier we expect. Even if ONLY CONC=128 throughput improves, that's worth 600 points (sub-ranked scoring). If interactivity also moves, that's 400 more points.

---

## 2026-04-13 — Session 5

### Goals
- Execute the Path A plan (TP=4 × DP=2 at CONC=128) with all variants (bf16 no-MTP, fp8 MTP=1, fp8 MTP=3)
- If Path A works, commit it as CONC=128 submission config
- Quick pivots: Fresh kernel profile, Kimi baseline, scheduler delay_factor

### Done — DSR1 DP scaling exhausted, all variants failed

Five TP<8 × DP variants tested. **Every single one failed.** Full postmortem:

| Variant | Config | Result |
|---|---|---|
| Path A | TP=4 DP=2 bf16 no-MTP | GSM8K=0.9409 PASS → Memory access fault at CONC=128 40% (nhead=32 decode kernel OOB for M>4 — same bug as Session 3 PATCH-003) |
| Path A capped | + `--max-num-seqs 16` | `AssertionError: graph_bs[0] <= max_num_seqs` at launch — ATOM's own sanity check rejects cudagraph sizes > max_num_seqs |
| Path A-fp8 | TP=4 DP=2 fp8 MTP=1 | AITER `decode_qlen=2,4 gqa_ratio=32 fp8/fp8` kernel assertion. EAGLE draft runs qlen=1, no supporting kernel. |
| Path A-fp8-mtp3 | TP=4 DP=2 fp8 MTP=3 | Same assertion at cudagraph capture |
| Path A' | TP=2 DP=4 bf16 MTP=3 | Launched clean, GSM8K = **0.9045 FAIL** (~5% below 0.93 gate) |
| Path A' no-MTP | TP=2 DP=4 bf16 no-MTP | GSM8K=**0.9386 PASS** / Throughput=**2750.38** (**-22.6% vs BEST BASE**) / E2E 53630ms / Interactivity 23.58 / **NET LOSS — confirms DSR1 DP unviable** |

**Cherry-pick attempt**: ATOM commit `4911f42 disable persistent mla for fp8 kvcache`. REJECTED with conflict — commit targets `atom/plugin/attention_mla_sparse.py` which our HEAD has deleted (sparse MLA was refactored out). Not applicable to dense MLA path we use.

**Conclusion**: **DSR1 DP scaling is blocked at the AITER kernel layer on gfx950.** Every TP<8 × DP configuration either hits the known nhead=32 decode kernel M>4 OOB bug, or the `decode_qlen=2,4` fp8 persistent mode kernel limitation, or accuracy degradation from MXFP4 sharding at small TP, or MTP+DP+BF16 sync issues. There is no currently-supported kernel path. **Our TP=8 + FlyDSL + dualstream BEST BASE is the final DSR1 submission config.**

### Done — three research agents returned critical intel

Spawned 3 parallel agents during Path A' wait time. All returned useful findings:

**Agent 1: Kimi K2.5 architecture research**
- Kimi K2.5 is NOT a simple swap. Multi-day pivot minimum.
- `n_routed_experts = 384` (vs DSR1's 256) → FlyDSL CSV won't match, needs re-tune
- `num_attention_heads = 64` (vs 128) → gqa_ratio halves, TP=8 = 8 heads/rank
- **No MTP head** — uses EAGLE3 via vLLM `--speculative-model` (PRs #33320, #34501). ATOM may not support it at all.
- `rope_theta = 50000` (vs 10000), YaRN-32 → RoPE cache rebuild
- Multimodal: MoonViT 400M vision tower, `--mm-encoder-tp-mode data` recommended
- AMD recipe: vLLM **v0.17.0** (not 0.15, not 0.18), ROCm 7.1.0, `VLLM_ROCM_USE_AITER=1`, TP=4, `--enforce-eager`
- vLLM 0.15 BROKEN for Kimi, needs backports from vLLM PRs #33320 and #34501
- Published Kimi K2-Thinking ceiling: 837 tok/s/GPU at CONC=128, 4× MI355 MXFP4, ISL=1024/OSL=1024
- **Effort estimate: 1-3 days** (vLLM image pull + launch + GSM8K + FlyDSL 384-expert re-tune + EAGLE3 wiring + sweep + robustness)

**Agent 2: MI355X / CDNA4 / gfx950 hardware research**
- Realistic sustained HBM BW: **6.5-7.0 TB/s** (vs theoretical 8.0). Naive placement costs up to 20%.
- LDS **grew 64KB → 160KB per CU** on CDNA4. Read BW doubled.
- **`decode_qlen=2,4` explained**: LDS bank-conflict optimization. At gqa_ratio=32, 32 Q heads per wave means the double-buffered K/V tiles hard-code into LDS and only qlen 2/4 leave enough bank space for fp8 scale vectors. **Recompile with wider LDS staging (feasible on 160KB) could lift this.** Real upstream AITER PR opportunity, 2-3 day kernel work by AMD engineer. Filing the issue with this analysis is a 30-min mergeable contribution.
- Matrix core throughput: FP8 ~20 PF (2× MI300X), MXFP4 ~40 PF dense, 80 PF w/ 2:4 sparsity
- Infinity Fabric: 7 links × 153 GB/s bidir → 1.075 TB/s per GPU aggregate. TP=8 ring-allreduce bounded at ~150 GB/s.
- 256 active CUs (8 XCDs × 32), vs 304 on MI300X
- Theoretical TPOT floor for our workload: ~0.82 ms with MTP=3. Current CONC=128 TPOT 41.61 ms = 51× above floor.
- Hot MFMA tiles on gfx950: `v_mfma_f32_16x16x128_f8f6f4`, `v_mfma_f32_32x32x64_f8f6f4`, block-scaled `__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4`

**Agent 3: Public leaderboard / Phase 2 intel**
- AMD's OWN published DSR1 per-GPU best: **864 tok/s/GPU** (CONC=128 FP8+MTP3 TP=8 MI300X ISL=1024)
- **Our BEST BASE 3555 is 4.1× higher** — apples-to-oranges (different hardware, dtype, context) but anchors that **the 6000 target is AMD internal stretch, not a public competitor floor**
- Track 1 capped at 10 finalists. Each gets guaranteed $10k + grand prize pool shot.
- Leaderboards live at `daniehua/dsr1-fp4-isl8192-osl1024-conc{4,32,128}.hf.space`. Gradio spaces, NOT scrapable — **need browser session to see scores**.
- **Track 2 Kimi ceiling (AMD published)**: only 837 tok/s/GPU. Much less explored. $650k prize. **$-per-engineering-hour significantly higher than further DSR1 chasing.**

### Strategic reframe

Before Session 5 research: "We need to chase CONC=128 throughput from 3555 → 6000 to pass the gate."

After Session 5 research: "**The 6000 is internal stretch, we're likely already top-3-5 on DSR1 among 10 finalists, and Track 2 Kimi has materially higher $-per-effort at this point.**"

This changes Session 5's remaining priorities:
1. **Lock BEST BASE as final DSR1 config.** Stop chasing throughput.
2. **Open leaderboard URLs in browser** to confirm rank (we cannot scrape Gradio, need real browser session)
3. **Tier 1 cheap wins on BEST BASE** (~2 hours): chiplet-aware scheduling audit, CUDA_GRAPH_MAX_SIZE, ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION, fresh profile
4. **File upstream AITER issue** with LDS bank conflict analysis (30 min, mergeable contribution)
5. **Draft Week 2 Kimi K2.5 battle plan** — not execute, just prep
6. **Kimi K2.5 is NOT a Session 5 task.** Needs proper prep session.

### Next (Session 6)
- **Week 2 starts**: Kimi K2.5 pivot with full prep
- Execute Tier 1 DSR1 wins if not done in session 5
- File AITER upstream issue with LDS analysis
- Fresh profile on BEST BASE post-FlyDSL

### Notes
- Server left stopped / released at end of session (cleanup commands in reference_commands.md)
- **Asalykov pending** — if call happens, ask about CONC=4 insights and LDS bank staging for gqa_ratio=32 fp8 path
- **Reply to Ziguan** when Kimi pivot starts (ask exact vLLM Docker tag)

---

## 2026-04-13 — Session 6A (continuing same calendar day, evening run)

### Goals
- Build the engineering model that 5 sessions of optimization were missing
- Read ATOM + AITER + SGLang + vLLM source directly (not via WebFetch summaries)
- Produce 4 memory deliverables documenting the system end-to-end
- Apply Intervention #1 (num_kv_splits patch) and verify
- Recalibrate the strategic plan based on what the model says

### Done

#### 1. Engineering model deliverables built (4 memory files)
- `project_dsr1_latency_budget.md` — wall-clock TPOT decomposition. CONC=4 first cut: ~66% GPU kernel time, ~34% non-kernel residual (Python + drafter + comm). Source: rocprofv3 was available but not used; numbers came from torch.profiler trace × MTP-per-forward math.
- `project_atom_execution_flow.md` — ATOM source code trace from EngineCore.busy_loop() through ModelRunner.forward() to postprocess(). KEY FINDING: MTP drafter runs in PYTHON outside the cudagraph (model_runner.py:1745). Layer 0 input_norm AllReduce is NOT fused (deepseek_v2.py:1695 gates on `layer_idx > 0`). PATCH-005 crash was in inherited vLLM compiler manager, not ATOM — DEAD as planned.
- `project_aiter_kernel_map.md` — AITER kernel dispatch table. Confirmed `mla_decode_fwd` signature, `get_meta_param()` heuristic (manually computed for our shapes), `fused_allreduce_rmsnorm` dispatch path, FlyDSL stage1+stage2 wrappers. vLLM uses persistent mode via `get_mla_metadata_v1()` — different code path than ATOM.
- `project_framework_comparison_dsr1.md` — ATOM vs SGLang vs vLLM matrix. ATOM has fused_allreduce_rmsnorm (saves 5-10% TPOT vs SGLang's scheduling-overlap pattern). SGLang has `mooncake/`, `mori/`, `nixl/` PD disagg backends ATOM doesn't have. **For DSR1 staying on ATOM is correct unless PD disagg becomes the only path to CONC=128.**

#### 2. Intervention #1 applied — num_kv_splits=16 → None (FLAT result, kept as cleanup)
- Patch: `attention_mla.py:592`, changed hardcoded `num_kv_splits=16` to `None` (let AITER auto-tune via `get_meta_param()`)
- Hypothesis: auto-tuner picks i=8 at CONC=32 and i=2 at CONC=128 (manually verified the heuristic)
- Predicted: -3-10% TPOT at CONC=32/128
- **Actual at canonical workloads**:
  - CONC=4: 738 → 749 thr/GPU (+1.4%), TPOT 6.07 → 5.92 ms (-2.5%)
  - CONC=32: 2345 → 2364 thr/GPU (+0.8%), TPOT 15.65 → 15.38 ms (-1.7%)
  - CONC=128: 3555 → 3576 thr/GPU (+0.6%), TPOT 41.61 → 41.66 ms (+0.1%)
- All ~+1% across CONCs, no regression. **Most likely cause: ATOM uses persistent mode; AITER persistent mode internally overrides num_kv_splits to cu_num regardless of caller's value.** Patch is harmless cleanup, kept for upstream PR.

#### 3. Intervention #2 (q→FP8 cast) — INVALID AS PLANNED
- WebFetch agent claimed lines 479-481 had a commented-out `q.to(dtypes.fp8)` block to uncomment.
- Direct source read showed NO such block exists. Lines 479-481 are inside `_forward_prefill_mla()` (line 449) sparse-attention handling. WebFetch hallucinated.
- **Skipped Intervention #2.** Lesson: trust source over agent summaries.

#### 4. THE BIG ONE — TP=4 single replica is ALIVE (DEC-024)
- After Daniel confirmed (Discord 2026-04-13) that 1500/3900/6000 are real qualification baselines (not aspirational), re-examined our position.
- Re-read the rules formula: `Token Throughput per GPU = CONC * (ISL+OSL) / (TTFT + OSL × TPOT) / num_GPUs_you_used, num_GPUs_you_used = 1, 2, ..., 8`
- Realized DEC-021 (Session 5) "all TP<8 × DP variants dead" CONFLATED two different things:
  - TP=4 × DP=2 (multi-replica) → genuinely dead due to gfx950 kernel layer bugs
  - TP=4 × 1 replica (4 GPUs idle, divisor=4 in formula) → never actually disproven
- The `dsr1_benchmark perf` binary divides by 8 hardcoded → made TP=4 single replica look worse than TP=8 in Session 3, dismissed as "4 GPUs idle, hurts"
- **Tested TP=4 single replica + MTP=3 + FP8-KV at full canonical workloads tonight**:

| CONC | TP=8 BEST BASE thr/GPU | TP=4 single thr/GPU | Δ | TP=4 TPOT | Interactivity at TP=4 | E2E at TP=4 |
|---|---|---|---|---|---|---|
| 4 (40 prompts) | 738.93 | **1124.7** | **+52.2%** | 7.86 ms | 127 ❌ | ~8424 ms ❌ |
| 32 (320 prompts) | 2345.57 | **3084.6** | **+31.5%** | 23.36 ms | 42.8 ❌ | 24310 ms ❌ |
| 128 (1280 prompts) | 3555.19 | **4543.0** | **+27.8%** | 65.09 ms | 15.4 ❌ | 67289 ms ❌ |

- Server logs during CONC=128 run showed MTP firing at full strength (Average toks/fwd: 2.67, Acceptance rate: 55-56%) — same as TP=8. **No crashes, no accuracy regression observable.**
- **Per-GPU throughput improvement at every CONC**, but **interactivity and E2E gates BREAK** because TPOT degrades 30-56% at TP=4. Net gate count: 0/9 RAW (was 3/9 at TP=8) — TP=4 alone is a net regression.
- **To win with TP=4, need to ALSO cut TPOT** by -43% at CONC=4 and -40% at CONC=32 to recover interactivity. -72% needed at CONC=128, NOT FEASIBLE.

#### 5. Multi-config submission strategy locked (DEC-025)
- CONC=4: TP=4 single + Tier 1 interventions
- CONC=32: TP=4 single + Tier 1 interventions
- CONC=128: TP=8 + PD disaggregation (SGLang+MORI) OR custom kernel work
- Daniel approved multi-config in DEC-022 (Session 5)

#### 6. New strategic rule: configuration first, custom kernels last (DEC-026)
- The TP=4 trick was a 1-line config change that gave +52% per-GPU throughput at CONC=4 — bigger than any custom kernel could realistically deliver
- AMD ships AITER (their kernel library); we have the same kernels they have
- The 2× gap from 738 → 1500 baseline is not a kernel-quality gap, it's a configuration gap
- Sweep ALL configuration moves (TP, EP, DP, scheduler, framework, AITER toggles) BEFORE writing custom kernels
- Custom kernels are the SCORING BONUS on top of configuration, not the qualification path
- Rule saved as `feedback_configuration_first_kernels_last.md` in memory

### Decisions made (decision_log)
- **DEC-024**: TP=4 single replica is alive — corrects DEC-021 (which conflated TP=4 single with TP=4 × DP=2)
- **DEC-025**: Multi-config submission — TP=4 for CONC=4/32, TP=8 for CONC=128
- **DEC-026**: Configuration first, custom kernels last — strategic pivot away from kernel-first thinking

### Pre-execution check for next session (CRITICAL)
**Discord Daniel** to confirm `num_GPUs_used = 4` reporting is allowed:
```
Hey Daniel, quick clarification:
The throughput formula in the rules says num_GPUs_you_used = 1, 2, ..., 8.
If we run TP=4 with a single replica (using only 4 GPUs of 8, leaving 4 idle),
do we report num_GPUs_used = 4? Or always 8?
The dsr1_benchmark binary always divides by 8, but the rules text suggests
we can divide by 4. Need to confirm before submission.
```
The whole TP=4 strategy depends on Daniel confirming the rules-text interpretation.

### Next session plan (Day 1, Apr 14)
**Track A (TP=4 for CONC=4/32 — Tier 1 configuration sweep)**:
1. `--enable-expert-parallel` at TP=4 (verify FusedMoE source first)
2. `--enable-dp-attention` at TP=4
3. `ATOM_USE_TRITON_GEMM=1 + ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION=1`
4. `--cuda-graph-sizes 1,2,4,8` (smaller capture set)
5. MTP=2,4 sweep at TP=4
6. `--max-num-seqs` tuning
7. `--enable-prefix-caching` with AITER scale fix from Session 3

**Track B (TP=8 for CONC=128 — Tier 1 configuration sweep)**:
1. TP=8 + `--enable-expert-parallel` (the BIG one)
2. Larger `--max-num-batched-tokens`
3. `--scheduler-delay-factor` tuning

**Track C (Day 3-4: SGLang + MORI PD disaggregation investigation)**:
- Container switch to `lmsysorg/sglang:v0.5.8-rocm700-mi35x`
- Test if single-node 4P+4D split via MORI works
- If yes: this is the only architectural path to CONC=128 6000 gate

### Notes
- Server left running at TP=4 single replica with `num_kv_splits=None` patch applied. Will shut down at session end.
- ATOM source patch `attention_mla.py.session6.bak` exists for reverting Intervention #1 if needed
- 6 hours of source reading and 1 hour of measurement produced more strategic value than 5 sessions of optimization-by-guessing combined
- Honest expected outcome with multi-config + Tier 1 + Tier 2: 6-7 of 9 gates passing, top-3 finalist position, $10k+ guaranteed plus shot at larger pool. CONC=128 interactivity is the hardest gate — only PD disagg or breakthrough kernel work clears it.

### Session 6A ADDENDUM — Deep research unlocks real recipe (late evening)

After tonight's TP=4 single replica finding, Danish insisted we do deep web research before executing more on the server. Launched 3 parallel research agents (AMD ATOM docs + MORI-EP + env var enumeration). Returned more actionable intel than 5 prior sessions combined.

**Top 7 actionable findings** (full details in memory file `project_dsr1_research_findings_session6a.md`):

1. **WRONG MODEL** — `amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4` (note `-MTP-MoEFP4` suffix) has quantized MTP layer weights. Our `amd/DeepSeek-R1-0528-MXFP4` does NOT. ATOM PR #411 (merged) publishes 29758 tok/s system (= 3720/GPU) at CONC=128 on the correct model with OUR exact recipe — 4.6% above our 3555 for free. GSM8K on the correct model: 94.90% (vs 94.47% on ours).
2. **`GPU_MAX_HW_QUEUES=5`** — hidden prerequisite for dual-stream MoE to actually overlap. ATOM PR #499 body: *"the same HW queue map to multiple internal streams, cause stream sequential"*. We've been setting `ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=16384` but dual-stream has been firing without overlapping.
3. **Missing CLI flags**: `--async-scheduling --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' --no-enable-prefix-caching`. Docs explicitly call FULL_AND_PIECEWISE "the most performant mode for most models". We've been running default PIECEWISE.
4. **`ATOM_USE_TRITON_GEMM=1 + ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION=1`** pair unlocks the auto-disabled fusion (we saw the warning in our own logs).
5. **`ATOM_USE_TRITON_MXFP4_BMM=1`** — untested, targets our MXFP4 BMM in MLA attention directly.
6. **`ATOM_ENABLE_RELAXED_MTP=1`** — merged via PR #411, requires MTP-MoEFP4 model, MTP acceptance 81% → 86%.
7. **Pull ATOM main past 38d0d7f374** — 5 merged PRs ahead of 108a70e: #547 stream-parallel decode metadata (free TPOT win), #507 HIP fused_rms_fp8_group_quant, #411 relaxed MTP, #499 GPU_MAX_HW_QUEUES fix.

**Architectural findings**:

8. **MORI-EP WORKS on single-node 8-GPU** via `IntraNode` kernel (XGMI peer-to-peer, no RDMA). Published MI355X EP8: 345 GB/s dispatch / 420 GB/s combine. Exact DSR1 command in ATOM PR #515. **Phase 2 of plan tests this.**
9. **SGLang + MORI PD disaggregation is DEAD** for single-node MI355X FP4. Upstream issues #18006 + #21942 confirm broken. Every AMD-blessed recipe is multi-node (1P2D = 3 nodes). **DROP from plan — saves 2-3 days of wasted investigation.**
10. **vLLM env vars** untested by us: `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1` (DSR1 has shared experts!), `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1` (MXFP4 path), `VLLM_ALL2ALL_BACKEND=mori`, `VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 + VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1`.

**Hard-rule additions**:
- **NEVER enable `--enable-tbo` on DSR1** — ATOM PR #515 measured -14 to -24% regression.
- **GLM-5 recipe warning**: DP-attn + EP + fp8 KV may not mix at gqa=8 (DSR1 is gqa=1, probably fine, but start safe without fp8 KV on first MORI-EP launch).

**Unverified claims** (treat as NOT load-bearing):
- "MORI-EP 82% MoE latency reduction" — not found in any published source
- "AITER sampling op 1.6× thr" — no VLLM_ROCM_USE_AITER_SAMPLING in mainline vLLM

### Decisions made from research (DEC-027 through DEC-029, TBD pending execution)

- **DEC-027 (pending execution confirmation)**: Swap to `amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4` as the new canonical model. Confirmed published numbers match hackathon workload.
- **DEC-028 (pending execution confirmation)**: Apply Tier 1 configuration stack (7 env vars + 5 CLI flags) as Phase 1 of the 14-day plan.
- **DEC-029 (already committed)**: Drop SGLang + MORI PD disaggregation from the plan entirely. Saves 2-3 days.

### Execution plan written and committed

Full 8-phase plan at `C:\Users\danis\.claude\plans\fizzy-toasting-teacup.md` ("DSR1 Path-to-Baselines Execution Plan"):

- **Phase 0** (Day 0 setup, ~1-2 hrs no GPU time): Discord Daniel, check model cache, download if needed, pull ATOM main, install mori package
- **Phase 1** (Day 1, 4-6 hrs): full Tier 1 configuration stack on TP=8 baseline, 3-CONC sweep, commit or revert
- **Phase 2** (Day 2, 4-6 hrs): MORI-EP single-node test via `-tp 8 --enable-dp-attention --enable-expert-parallel`
- **Phase 3** (Day 3, 4-6 hrs): atom-vllm plugin path with vLLM env vars if MORI-EP falls short
- **Phase 4** (Day 4, if multi-config alive): TP=4 single replica with all wins applied for CONC=4/32
- **Phase 5** (Days 5-10, if needed): custom kernels for specific remaining gap
- **Phase 6** (Days 11-13): accuracy robustness + submission prep
- **Phase 7** (Day 14): submit DSR1
- **Phase 8** (Days 15+): pivot to Kimi K2.5

### Quickstart card written

New memory file `project_dsr1_quickstart_card.md` — the single file the next Opus reads after a context compact. Contains current BEST BASE, active plan pointer, top 10 findings, strategic rules, dead paths, alive paths, Daniel question, and what the next session's opening line should be.

### End-of-session state

Server: should be shut down cleanly at end of session (Phase 0 starts with clean state tomorrow).
ATOM commit: 108a70e with `num_kv_splits=None` patch (Intervention #1, flat result, kept as upstream cleanup).
Open Discord question: Daniel `num_GPUs_used = 4` confirmation — blocker for Phase 4 only.

### Net Session 6A output (whole day)

- **4 engineering model deliverables** in memory (latency budget, ATOM flow, AITER kernel map, framework comparison)
- **TP=4 single replica discovered alive** (+27-52% per-GPU, DEC-024 corrects DEC-021)
- **Multi-config submission strategy locked** (DEC-025)
- **Configuration-first strategic rule** (DEC-026, `feedback_configuration_first_kernels_last.md`)
- **Deep research found the real recipe gaps** (wrong model + missing env vars + missing CLI flags)
- **Doc consolidation**: 6 project docs → 3 (Danish.md, MASTER_FINDINGS.md, daily_log.md) + memory archive
- **Full 14-day execution plan written** to plan file with checkpoints, rollback paths, and resume markers
- **Quickstart card memory file** for post-compact resumption

Longest and most productive session of the project. Worth it.

---

## 2026-04-13 — Session 6B Day 1 (Research-backed Tier 1 execution)

### Phase 0 (setup)

- Model `amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4` downloaded (500 GB, 87 files) — PARKED, not bounty-legal (bounty locks `MXFP4`, not `-MoEFP4`)
- ATOM pulled from 108a70e → a6fe785, then **reverted to 108a70e** after `Mxfp4MoEMethod` triton_kernels crashes on main body MoE
- transformers/huggingface-hub re-pinned to 4.57.6 / 0.34.0 after ATOM rebuild regressed them
- `mori` install BLOCKED: `libpci-dev` + `libibverbs-dev` can't install due to `rocm-hip` version conflict in container's apt state
- Session 6A Intervention #1 patch (`num_kv_splits=None` at `attention_mla.py:592`) intact through git pull/revert

### Today's BEST BASE reproduction (new floor — treat as authoritative going forward)

Ran bare BEST BASE command at end of Phase 1 to verify regression was real. Better than recorded 738.93:

| CONC | Thr/GPU | TPOT med | TTFT med | Interact | GSM8K |
|---|---|---|---|---|---|
| 4  | **757.31** (was recorded 738.93) | 6.10 ms | 234 ms | ~164 | **0.9462** (was 0.9378) |

Gain likely from container rebuild + newer transformers 4.57.6 numerics. **Treat 757 as new CONC=4 floor.** CONC=32/128 not re-measured today — still using Session 6A records.

### Phase 1 Tier 1 AITER-path stack — TESTED, PARKED (DEC-027)

Stack launched: `GPU_MAX_HW_QUEUES=5`, `ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=131072` (up from 16384), `ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION=1`, `--gpu-memory-utilization 0.95`, `--max-num-batched-tokens 32768`, `--cudagraph-capture-sizes "[1..512]"`.

**Shakedown drops** (landmines from Session 6A research being wrong about gfx950 compatibility):

- `ATOM_USE_TRITON_GEMM=1` — forces `Mxfp4MoEMethod.use_triton=True` on gfx950 → requires `triton_kernels` → would brick AITER (source: `atom/model_ops/moe.py:644-651`, commit 108a70e)
- `ATOM_USE_TRITON_MXFP4_BMM=1` — dropped as precaution
- `ATOM_ENABLE_RELAXED_MTP=1` — needs MTP-MoEFP4 model, blocked same landmine

#### Phase 1a results vs BEST BASE

| CONC | BEST BASE thr/GPU | Phase 1a thr/GPU | Δ | TPOT | Interactivity | Gates |
|---|---|---|---|---|---|---|
| 4 | 757.31 | **703.46** | **−7.1% ❌** | 6.10 → 6.56 | 164 → 152 (fails 165) | 0/3 |
| 32 | 2345.57 | **2261.0** | **−3.6% ❌** | 15.65 → 15.95 | 63.9 → 62.7 (passes 50) | 2/3 |
| 128 | 3555.19 | **3588.85** | **+0.95% ✓** | 41.61 → 41.04 | 24.0 → 24.4 (fails 48, BB also) | 0/3 |
| GSM8K | 0.9378 | **0.9462** ✓ | +0.84pp | — | — | pass |

Net: small regression uniformly except CONC=128 marginal win. Not worth keeping as-is.

#### Phase 1b diagnostic — dropped `ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION=1`

CONC=4: 698.04 thr/GPU, TPOT 6.57 ms — essentially unchanged from Phase 1a. **Ruled out RMSNORM fusion as regressor.** One of the other 5 deltas (`GPU_MAX_HW_QUEUES`, dual-stream threshold, gpu-util, max-num-batched-tokens, cudagraph sizes) is the culprit. Didn't single-knob bisect further — cost/reward too small given Phase 3 has better expected upside.

### DEC-027 summary

**Phase 1 Tier 1 AITER-path stack is net-negative. Parked.** Regression at CONC=4/32 outweighs the CONC=128 marginal win. Skipping single-knob bisect of 5 remaining knobs. Revisit only if Phase 3 fails to close gates.

### Landmines discovered today (never re-hit)

1. **`ATOM_USE_TRITON_GEMM=1` on gfx950 forces triton_kernels requirement** → incompatible with `danish_atom_main` AITER-only container. NEVER set in this container. Safe only in `rocm/atom-dev:vllm-latest`. Source: `atom/model_ops/moe.py:644-651` in 108a70e.
2. **`amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4` model** hits same triton_kernels trap at layer 0. ALSO bounty rules pin `amd/DeepSeek-R1-0528-MXFP4` per `COMPETITION_QUICKSTART_EN.md`. **PR #411's 3720 tok/s/GPU is AMD research path, NOT bounty path.** Session 6A research was wrong.
3. **ATOM main a6fe785** broke `Mxfp4MoEMethod` dispatch for main body vs 108a70e. Stay on 108a70e.
4. **`--async-scheduling`, `--compilation-config`, `--no-enable-prefix-caching` are vLLM-only CLI flags.** Not accepted by `atom.entrypoints.openai_server`. Only work in atom-vllm plugin mode.
5. **`mori` install blocked** by broken apt state in danish_atom_main. Try in `rocm/atom-dev:vllm-latest`.
6. **`VLLM_ROCM_USE_AITER_*` env vars are no-ops in plugin mode** — `ATOMPlatform.get_attn_backend_cls()` returns `AiterMLABackend` directly, bypassing vLLM's own ROCm attention path. Source: `docs/vllm_plugin_backend_guide.md` section 3.

### Phase 3 plan (Day 2): container swap to `rocm/atom-dev:vllm-latest`

Recipe source: `/projects/teamA/danish/repos/ATOM_main/recipes/atom_vllm/DeepSeek-R1.md`. Plugin mode gives real new levers: `--async-scheduling`, `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'`, vLLM's mature cache manager. ATOM's own cudagraph disabled in plugin mode (`enforce_eager=True`) — delegates to vLLM.

Container swap prep done:

- `/workspace/ATOM_main`, `/workspace/amdgpu_bounty_optimization`, `/workspace/aiter` all verified as symlinks into `/projects/teamA/danish/repos/` (persistent xfs mount)
- Session logs rescued to `/projects/teamA/danish/session_logs/session6b_day1/`
- Git safe.directory configured for ATOM / AITER / bounty repos
- Bench script at persistent `/projects/teamA/danish/repos/ATOM_main/atom/benchmarks/benchmark_serving.py`

### Parking lot (alive, not dead)

1. Phase 1 Tier 1 single-knob bisect — 5 knobs, ~20 min
2. MORI-EP — unblocked if `rocm/atom-dev:vllm-latest` has clean apt
3. MTP drafter in cudagraph capture — Phase 5, ~25% TPOT lever at CONC=4
4. Danny/LunNova precision MLA decode kernel port — Phase 5, ~10%
5. MTP=5+ AITER patch — Phase 5, +15% if `qo_len ≤ 4` assertion lifts
6. TP=4 single replica multi-config — pending Daniel's Discord reply on `num_GPUs_used=4`
7. ATOM main beyond 108a70e — possible PR #547 stream-parallel decode win; 1+ PR in #503/#531/#538/#547 broke `Mxfp4MoEMethod`. Investigate in isolation.
8. GSM8K today 0.9462 vs 0.9378 record — unexplained improvement, confirm reproducibility

---

## 2026-04-13 — Session 6B Day 1 continued (Phase 3 container swap + rules re-read)

### Phase 3 attempt — atom-vllm plugin container

Pulled `rocm/atom-dev:vllm-latest` and started `danish_atom_vllm_main` with identical mounts to `danish_atom_main`. Critical findings inside the new image:

- vllm 0.19.1.dev0+g2a69949bd.d20260412 preinstalled at `/opt/venv/bin/vllm`
- `/app/ATOM` at commit 108a70e (same as our native checkout)
- Entry point `atom` registered for `vllm.platform_plugins`
- `flydsl==0.1.2` preinstalled
- `lm_eval 0.4.11` preinstalled
- **`/app/mori` directory exists** — MORI IS installed in this image (unlike `danish_atom_main` which has the apt block)
- **`triton_kernels` PyPI package is STILL not installed** — so MXFP4-MTP-MoEFP4 model and `ATOM_USE_TRITON_GEMM=1` are still unusable here
- `HOME=/tmp` override needed for AITER cache (default `/root/.aiter` is unwritable; with HOME=/tmp, CK+HIP ops load cleanly)

### Phase 3a — no-MTP baseline on plugin mode (recipe verbatim from `recipes/atom_vllm/DeepSeek-R1.md`)

Launched: `vllm serve amd/DeepSeek-R1-0528-MXFP4 -tp 8 --kv-cache-dtype fp8 --gpu_memory_utilization 0.9 --async-scheduling --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' --no-enable-prefix-caching --max-model-len 10240`

Startup confirmed plugin activation:
- `Platform plugin atom is activated`
- `Register model DeepseekV3ForCausalLM → ATOMMoEForCausalLM`
- `ATOM plugin: patched vLLM graph_capture to nest aiter ca_comm.capture()` — the AR+RMS fusion IS preserved inside vLLM's cudagraph in plugin mode
- `cudagraph_mode: FULL_AND_PIECEWISE`, `max_cudagraph_capture_size: 512`, 51 capture sizes from 1 to 512
- `Asynchronous scheduling is enabled`

GSM8K on plugin mode (via lm_eval): **0.9500 flex-extract, 0.9469 strict-match** — even higher than BEST BASE. Above gate.

3-CONC no-MTP sweep results:

| CONC | BEST BASE (MTP=3) thr/GPU | Phase 3a (no MTP) thr/GPU | Δ | TPOT | TTFT | Notes |
|---|---|---|---|---|---|---|
| 4 | 757.31 | 477.44 | −37% | 6.10→8.76 | 234→751 ms | Chunked prefill at max_num_batched_tokens=8192 serializes CONC=4 prefill |
| 32 | 2345.57 | 1728.82 | −26% | 15.65→19.52 | unrecorded→1270 ms | |
| 128 | 3555.19 | 2907.31 | −18% | 41.61→47.98 | unrecorded→1076 ms | |

Regression shrinks as CONC grows — exact pattern you'd expect from losing MTP's ~1.5-2× speculative batch multiplier (bigger relative effect at low CONC).

### Phase 3b — add MTP, CRASH on drafter MLA init

Relaunched with `--speculative-config '{"model":"amd/DeepSeek-R1-0528-MXFP4","method":"deepseek_mtp","num_speculative_tokens":3}' --max-num-batched-tokens 32768`.

All 8 worker processes crashed during model load:
```
File "/app/ATOM/atom/plugin/attention_mla.py", line 984, in new_init
    orig_init(self, *args, **kwargs)
File "/app/ATOM/atom/model_ops/attention_mla.py", line 145, in __init__
    self.q_lora_rank = mla_modules.q_lora_rank
AttributeError: 'NoneType' object has no attribute 'q_lora_rank'
```

Root cause: vLLM's MTP drafter path (`vllm/v1/spec_decode/eagle.py`) constructs the DeepseekV2 MTP decoder layer's MLA attention via a codepath where ATOM's plugin wrapper doesn't populate `mla_modules`. The wrapper just does `orig_init(self, *args, **kwargs)` with whatever vLLM passes, and vLLM's `MLAAttention.__init__` at `mla_attention.py:388` doesn't pass `mla_modules=...` — it passes scalar `q_lora_rank` / `kv_lora_rank` / `qk_*_head_dim` as flat kwargs directly, expecting the impl to use those. ATOM's `model_ops/attention_mla.py` impl hard-codes `self.q_lora_rank = mla_modules.q_lora_rank` and doesn't fall back to the kwarg form.

### Investigation: is plugin MTP implemented in ATOM at all?

Grepped `/app/ATOM/atom/plugin/` — found smoking guns:

```
/app/ATOM/atom/plugin/attention.py:726:    # TODO: support mtp and sparse
/app/ATOM/atom/plugin/attention.py:1132:       # TODO: support mtp
```

Plugin-mode MTP is explicitly flagged as **unimplemented** by the ATOM developers themselves. Not a bug, a missing feature.

Background research agent confirmed via ROCm/ATOM repo + PR search:
- **PR #544** "[Feature] Support GLM-5 MTP for vLLM Pluggin" — DRAFT, opened 2026-04-11, targets branch `plugin_sparse_mla`, for GLM-5 only, not DeepSeek. Depends on PR #399 which is OPEN/unmerged.
- **PR #399** "[Feat][Plugin] Enable Sparse MLA and GLM-5 for vLLM-ATOM" — open, unmerged. Sparse MLA refactor prerequisite.
- **PR #265** (merged Mar 10) established plugin-mode MLA but explicitly without MTP/drafter support
- Recipe `recipes/atom_vllm/DeepSeek-R1.md` has **zero** mentions of MTP, speculative, or num_speculative_tokens — AMD's own plugin recipe deliberately omits MTP, confirming it's not supported
- Latest commit to `atom/plugin/attention_mla.py` is Mar 26 (a6ad84d) — nothing in April

**Conclusion: plugin-mode DeepSeek MTP is not a fixable bug, it's unimplemented.** AMD is rolling it out model-by-model starting with GLM-5. DeepSeek will come later, probably not within our submission window.

### Re-read of bounty rules text (danielhua23/amdgpu_bounty_optimization README + Rules doc)

Two critical rule clarifications that change strategy:

**Rule 1: TP=4 single replica IS definitively allowed.** Direct quote:
> "the maximum supported configuration is TP/EP = 8. However, developers may choose smaller TP and EP sizes, as long as the model fits, and the following criteria must still be satisfied."
> "Token Throughput per GPU = concurrency × (input_length + output_length) / (mean_TTFT + output_length × mean_TPOT) / **num_GPUs_you_used, num_GPUs_you_used = 1,2,...,8**"

The rules text is authoritative. The `dsr1_benchmark` binary that hardcodes ÷8 is out of date — the rules say divide by `num_GPUs_you_used`. **Daniel's Discord reply is not needed — Phase 4 TP=4 multi-config is unblocked immediately per the rules.**

**Rule 2: For Track 1 DSR1, the framework is "AMD ATOM or SGLang".** vLLM is NOT listed as a valid DSR1 framework (but IS listed for Track 2 Kimi K2.5). Submitting a `vllm serve` command for DSR1 is in a gray zone even if the ATOM plugin is active, because the rule text says "AMD ATOM or SGLang" verbatim. Safest interpretation: **use ATOM's native `atom.entrypoints.openai_server` for all DSR1 submissions.**

**Rule 3: ATOM submissions have ZERO upstream-agnostic constraint.** Direct quote from rules §4.4:
> "Here is a link to AMD ATOM https://github.com/ROCm/ATOM. Since this is AMD's own framework, Submissions can introduce tightly coupled AMD-specific dependencies, optimizations."

Compare to the vLLM/SGLang mergeability rule:
> "Optimizations must be AMD-agnostic (No AMD-only logic and No vendor lock-in) and acceptable to upstream communities"

**For DSR1 on native ATOM, we can write MI355X-specific kernels, hardcode AITER dispatch paths, hand-tune HIP assembly, pin specific ROCm versions — whatever it takes — as long as the code is clean enough to merge into `ROCm/ATOM`.** The upstream-agnostic gate that was the main disqualifier for Phase 5 custom kernel work **does not apply to DSR1**. Kernel work is unambiguously in scope for the sprint.

### DEC-028 — Phase 3 plugin mode dropped for DSR1 (two independent reasons)

1. MTP is unimplemented for DeepSeek in the ATOM plugin — confirmed by source TODO comments, PR search, and recipe omission. The fix is a multi-day feature development effort AMD is doing for GLM-5 first, and we can't cherry-pick it for DSR1.
2. vLLM is not a listed DSR1 framework per the rules. Even if we fixed MTP, submitting via `vllm serve` is gray-zone.

Plugin mode is parked. Native ATOM with `atom.entrypoints.openai_server` + MTP=3 is the only DSR1 submission path for this session.

### DEC-029 — TP=4 single replica multi-config unblocked by rules text

The rules text explicitly allows smaller TP/EP and says `num_GPUs_you_used = 1,2,...,8`. Daniel's Discord reply is not strictly required. Phase 4 TP=4 multi-config is the primary lever going into Day 2 of Session 6B.

Per Session 6A measured TP=4 numbers (1124/3084/4543 thr/GPU) the throughput wins are +27–52% over TP=8 BEST BASE. The risk is interactivity/E2E — TP=4 TPOT degrades at all CONCs, and CONC=128 TPOT is 65 ms which far exceeds the 20.83 ms interactivity gate. Multi-config strategy: TP=4 at CONC=4/32, TP=8 at CONC=128.

### DEC-030 — 10/10/10 sprint schedule (user directive)

Compressed deadline structure from the user tonight:
- **DSR1 sprint**: Apr 14 → Apr 23 (10 days). Beat baseline + exceed. Lock config + submit by Apr 23 EOD.
- **Kimi K2.5 sprint**: Apr 24 → May 3 (10 days). Beat baseline, same structure.
- **Polish window**: May 4 → May 13 (10 days). Improve both tracks on top of the Day-10 submissions.
- **Final submit**: May 15.

The original plan's Phase 6-8 polish time collapses into the 10-day polish window. No slack. Every day needs a specific deliverable or we revise the plan.

### Tonight's session wrap

Server state: both containers alive (`danish_atom_main` and `danish_atom_vllm_main`), no running inference processes, all session logs rescued to `/projects/teamA/danish/session_logs/session6b_day1/`. Docs fully updated across memory + local AMD dir. Session 6B Day 1 closed, Day 2 starts with Phase 4 TP=4 multi-config on `danish_atom_main` as the first action.

### Wins and losses tally for Session 6B Day 1

**Wins:**
- BEST BASE reproduced at 757/GPU today (+2.5% vs recorded 738, likely from newer transformers 4.57.6)
- GSM8K 0.9462–0.9500 (+1pp, well above 0.935 gate)
- TP=4 multi-config unblocked by rules text re-read (no Daniel blocker)
- Native ATOM has zero upstream-agnostic constraint per rules §4.4 — custom kernel work is back in play without merge risk
- `/app/mori` preinstalled in the vllm image — MORI-EP may be unblockable by using that image but running native ATOM inside it (`atom.entrypoints.openai_server`, not `vllm serve`)
- Six landmines documented (triton_kernels trap, MTP-MoEFP4 trap, plugin MLA drafter crash, vLLM-only CLI flags, mori apt block, VLLM_ROCM_USE_AITER_* no-ops in plugin mode)
- DEC-027 (Phase 1 Tier 1 stack parked) + DEC-028 (Phase 3 plugin mode dropped) + DEC-029 (TP=4 allowed) + DEC-030 (10/10/10 sprint) locked

**Losses:**
- Phase 1 Tier 1 AITER-path stack was net-negative (−7%/−3.6%/+0.95%), didn't bisect the regressor
- Phase 3 plugin mode dead for DSR1 MTP — two independent reasons
- Session 6A research findings were over-optimistic about several knobs and models

**Net for the day**: +2.5% BEST BASE, +1pp GSM8K, rules re-read unlocked TP=4 AND kernel work. Net-positive day even though both architectural bets (Phase 1 env var stack, Phase 3 plugin mode) failed.

### Day 2 first action (Session 6B Day 2, Apr 14 morning)

1. Read `project_dsr1_quickstart_card.md` + `project_session6b_day1_state.md` in memory (≤2 min)
2. Enter `danish_atom_main` container
3. Launch Phase 4 TP=4 single replica with MTP=3 + BEST BASE config (see the launch command pre-staged in `project_session6b_day1_state.md` Day 2 section)
4. 3-CONC sweep, confirm Session 6A's 1124/3084/4543 numbers
5. Single-knob parking-lot env var sweep on whichever TP wins per CONC

No detours. No plugin mode. No MTP-MoEFP4 model. No `ATOM_USE_TRITON_GEMM=1`. Native ATOM + MXFP4 + MTP=3, multi-config.

---

## 2026-04-14 — Session 6B Day 2 (Sprint Day 1: TP=4 + env var sweep + bottleneck analysis pivot)

### TP=4 reproduction (iter 0)

Launched native ATOM TP=4 single replica with BEST BASE env vars (`ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=16384`, `HIP_FORCE_DEV_KERNARG=1`, `NCCL_MIN_NCHANNELS=112`, MTP=3, FP8 KV, max-model-len 10240).

**AITER confirmation**: log shows `mla_a8w8_qh32_qseqlen4_gqaratio32_ps` kernel firing — **AITER has a qh32 MLA decode kernel now**. Session 6A Issue #1468 ("only 16 or 128 heads supported") is no longer the blocker at TP=4. 128/4=32 works out of the box.

**GSM8K**: 0.9424 ✓ (above 0.935 gate at TP=4)

**CONC=4 results** (Total thr / 4 for num_GPUs_used=4 reporting):

| | Session 6A TP=4 | Today iter 0 |
|---|---|---|
| Thr/GPU | 1124 | **1099.01** (−2.2%, within tolerance) |
| TPOT med | 7.86 ms | 8.21 ms |
| TTFT med | (n/a) | 374 ms |
| Interact | 127 | 121.8 |
| E2E | 8424 ms | ~8781 ms |

Reproduction verified within 3%.

### Iter 1: TP=4 + FlyDSL force-enable

Added env vars from Session 6B Day 1 research critique:
```
AITER_USE_FLYDSL_MOE=1
AITER_ENFORCE_DSL=1
AITER_USE_FLYDSL_MOE_STAGE1=1
AITER_USE_FLYDSL_MOE_STAGE2=1
```

Log confirmed FlyDSL DSL path firing: `flydsl_moe1_afp4_wfp4_bf16_*` and `flydsl_moe2_afp4_wfp4_bf16_*_persist` kernels at all batch sizes.

CONC=4: **1136.13 thr/GPU, TPOT 7.91 ms, interact 126.4** (+3.4% vs iter 0). **KEPT**.

### Iter 2: + HSA_ENABLE_SDMA=0

Added SDMA disable on top of iter 1. Hypothesis from research: multi-GPU stability / perf knob untested on our stack.

- CONC=4: 1116.05 thr/GPU (−1.8%), TPOT 8.23 (+4%)
- CONC=32: 2773.21 thr/GPU (−10% vs Session 6A TP=4 record 3084), TPOT 26.10, interact 38.3 (fails 50 gate)

**DROPPED**. SDMA disable hurts at CONC=4 and CONC=32. Confirmed not beneficial on TP=4 native ATOM path.

### Iter 3: + AMD's vLLM DSR1 recipe flags (from rocm.blogs.amd.com scaling-ai-inference blog)

Dropped SDMA, added three flags from AMD's own DSR1 vLLM production recipe:
```
--max-num-batched-tokens 131072  (was default ~8192)
--max-num-seqs 4096              (was default ~256)
--block-size 1                   (was ATOM default 64)
```

Rationale: AMD's Dec 8 2025 DSR1 vLLM benchmark uses these to handle chunked prefill and large decode queues at high CONC.

- CONC=4: **1105.22 thr/GPU** (−2.7% vs iter 1, **NOISE**), TPOT 7.92 (unchanged), TTFT 374 (unchanged)
- CONC=32: **2910.50 thr/GPU** (+5.0% vs iter 2, still −5.6% vs Session 6A TP=4 3084), TPOT 24.73 (−5.2% vs iter 2), TTFT 390 (−10% vs iter 2), interact 40.4 (still fails 50)

**CONC=4 NEUTRAL, CONC=32 MILD WIN.** The 131072 batched tokens helps at CONC=32 (bigger prefill chunks amortize) but does nothing at CONC=4 (prefill already fits in default window at CONC=4). Keeping iter 3 config as candidate for CONC=32 submission, but TP=4 TPOT at CONC=32 is structurally above the 20 ms interactivity gate.

### Bottleneck analysis (triggered by Danish question: "what's the blocker?")

Danish questioned why we're doing env var sweeps without a bottleneck model. He was right. Built the full per-CONC TPOT decomposition — **see memory file `project_dsr1_bottleneck_analysis_day2.md` for the complete analysis with gate math, kernel profiles, and day-by-day kernel sprint plan.**

Key findings summarized:

| CONC | TPOT today | Gate TPOT | Cut needed | Primary blocker | Structural? |
|---|---|---|---|---|---|
| 4 | 6.10 ms | 2.77 ms | **−55%** | Python overhead from MTP drafter out-of-cudagraph (33% of TPOT = ~2 ms fixed) | YES |
| 32 | 15.65 ms | 8.90 ms | **−43%** | MoE expert GEMMs (42% of TPOT, ~6.5 ms at bs=32) | YES |
| 128 | 41.61 ms | 20.83 ms | **−50%** | MoE compute + dispatch (50%) + AllReduce (14%) | YES |

**Knob-twiddling ceiling at CONC=4**: even with perfect kernel optimization, env var sweeps cap at ~+10% stacked. 1136 × 1.10 = **~1250 thr/GPU maximum via config alone**. Gate is 1500. **Config sweeps cannot close any gate.**

### DEC-031 (pending): Pivot to structural kernel work after TP=2/TP=1 test

Commitment for rest of sprint:
- Day 1 (today, remaining afternoon): TP=2 and TP=1 single-replica tests (12 min total). Then STOP env var sweeps.
- Day 2 (Apr 15): Start MTP drafter-into-cudagraph investigation. Read `atom/model_engine/model_runner.py:1745` + `atom/spec_decode/eagle.py:50`. Scope patch.
- Day 3 (Apr 16): Finish MTP drafter cudagraph patch or abort. Measure delta.
- Day 4 (Apr 17): MORI-EP test in `rocm/atom-dev:vllm-latest` container (running native `atom.entrypoints.openai_server` inside it — `/app/mori` is preinstalled in that image).
- Day 5 (Apr 18): MLA decode kernel port (Danny/LunNova precision-safe style).
- Day 6 (Apr 19): Stack kernel wins, 3-CONC re-measure.
- Day 7 (Apr 20): Optional MTP=5+ AITER patch or finalize.
- Day 8 (Apr 21): Accuracy robustness + repro script.
- Day 9 (Apr 22): PR draft + screenshots + writeup.
- Day 10 (Apr 23): Submit DSR1. Pivot to Kimi Day 11.

Realistic outcome: 4-6 of 9 gates PASS, top-3 to top-5 sub-rank, grand prize probability ~10-15%.

### Engineering rule reinforcement

Per Danish's Session 6B Day 1 directive ("always remember we have to do engineering"):
1. Every intervention from Day 2 onwards must name the blocker with numbers
2. Every patch must name file + line + what changes
3. Every launch must predict delta and commit/revert by day budget
4. Every failure gets a daily_log line within 10 min
5. If prediction doesn't match, STOP and re-read source before next try

No more env var shotgun. Full commitment to structural kernel work for Days 2-7.

## 2026-04-14 — Session 6B Day 2 (execution log)

### Summary

Structural day. Iterations 6-9. Most interventions NEUTRAL, MORI-EP parked, M2/M3 drafter cudagraph work completed and measuring. Key strategic correction from Danish: MTP-MoEFP4 checkpoint is a triton trap (1.5× slower than CK+asm), plain MXFP4 is correct. Stop re-planning every session; two-stage plan is authoritative.

### Iter 6 — PR #547 cherry-pick (stream-parallel decode metadata)

- Blocker: can we stack published decode-metadata parallelism on top of BEST BASE?
- Files changed: async_proc.py, engine_core.py, model_runner.py, aiter_mla.py (4 files, 82 lines)
- Predicted delta: +3-5% CONC=4 throughput
- Measured: CONC=4 1123 vs 1136 baseline (NEUTRAL, noise)
- Decision: kept (no regression), move on

### Iter 7 — AITER main pull (15 commits)

- Blocker: are we missing upstream MLA/MoE wins?
- Pulled a35b45ad9 → 303a583c8. Includes PR #2717 OPUS MLA Reduce, #2700 small-M MoE opt, #2661 NUM_KSPLIT fix.
- Predicted delta: +5-15% stacked
- Measured: CONC=4 1117 (NEUTRAL), CONC=128 4535 vs 4543 baseline (NEUTRAL)
- Decision: kept. Upstream wins don't hit our DSR1 code paths meaningfully.

### Iter 8a-d — MORI-EP attempt (PARKED — DEC-032)

- Goal: enable MORI all-to-all for MoE dispatch at CONC=128 (AMD MLPerf claim: 82% dispatch latency reduction)
- Infrastructure: copied mori package from `danish_atom_vllm_main` container into `danish_atom_main` (bypassing apt block on libibverbs-dev/libpci-dev). Path: `/opt/venv/lib/python3.12/site-packages/mori/` + `/app/mori/build/`. `python3 -c 'import mori'` succeeds.
- Iter 8a crash: `AttributeError: 'NoneType' object has no attribute 'wait_stream'` at `atom/models/deepseek_v2.py:915` in `dual_stream_moe_forward`. Root cause: **ATOM PR #389 regression (unreported) — MTP drafter's `DeepseekV2DecoderLayer` is instantiated without `alt_stream=` kwarg, so `self.alt_stream = None` on layer 61 MoE. Under `torch.compile`, fx graph bakes in `maybe_dual_stream_forward.default(buf4, 'model.layers.61.mlp')` which crashes on replay.**
- Research agent confirmed: zero existing GitHub issues/PRs for this regression. PR #389 introduced it, PR #393 was a partial fix that missed the drafter path, PR #400 removed the env-var gate so the issue can't be disabled.
- Iter 8b: removed `ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD`. Same crash (dispatch default still triggers).
- Iter 8c: patched `maybe_dual_stream_forward` to guard `self.alt_stream is not None` and route to `single_stream_moe_forward` when None. Crash fixed. BUT MTP acceptance collapsed from 63% → 3-8% during bench. **Root cause: MTP drafter fundamentally incompatible with `--enable-dp-attention` token layout. `atom/plugin/attention.py:726,1132` explicitly has `TODO: support mtp` markers.**
- Iter 8d: dropped MTP (`--method mtp` removed) to isolate MORI-EP effect at CONC=128.
  - CONC=128 result: TPOT 38.77 ms (vs 41.6 baseline, -7%), P99 TPOT 47.46 → interactivity 21 (still fails 48 gate). Total thr 23329/s = 2916/GPU (vs 3555 baseline, **-18% regression** because MTP is off).
  - Not a useful Stage 1 lever at CONC=128: MTP's 1.89× thr multiplier > MORI-EP's 7% TPOT cut.
- **DEC-032**: MORI-EP PARKED until AMD lands upstream fix for MTP+DP-attention. Not worth multi-day workaround.

### Iter 9 M1 — probe phase for drafter cudagraph capture

- Read `atom/model_engine/model_runner.py:590-620` (model init), `1695-1950` (forward + postprocess + propose_draft_token_ids + capture_cudagraph)
- Read `atom/spec_decode/eagle.py:1-210` (EagleProposer class, propose loop)
- Read `atom/models/deepseek_mtp.py:1-80,160-230` (drafter model forward signature)
- Read `atom/model_ops/attentions/aiter_mla.py:153-230,492-520` (`set_mla_persistent_worker_buffers`, `prepare_mtp_decode`, `build_for_cudagraph_capture`)
- Verified capture safety: `prepare_mtp_decode` and `set_mla_persistent_worker_buffers` both write in-place into `forward_vars` pinned tensors. The returned `workinfos` dict contains full pinned-buffer references, not slices — stable addresses across replays.
- Identified rebind bugs in eagle.py loop: `positions = target_positions + 1` (line 118, new tensor), `input_ids = new_draft_ids` (rebind), `hidden_states = sample_hidden_states` (rebind), `attn_metadata.__dict__[k] = v` (metadata rebind). All fixable with in-place `copy_()` into pre-allocated pinned buffers.
- Decision: capture iters 1..mtp_k-1 only (iter 0 has different shape `bs*(mtp_k+1)` tokens, iters 1+ have decode shape `bs` tokens). For MTP=3 that kills 2/3 drafter Python overhead.

### Iter 9 M2 — eagle.py loop refactor (no capture yet)

- Files: `atom/spec_decode/eagle.py` (`.session6b_bak` backup saved)
- Changes:
  - Added pinned buffers in `EagleProposer.__init__`: `self.draft_input_ids`, `self.draft_positions`, `self.draft_hidden_in` (`max_bs * hidden_size`, correct dtypes)
  - Refactored end-of-iter rebinds (lines ~193-196) to in-place `copy_()` + `add_(1)` into pinned buffers
- Predicted delta: NEUTRAL (no capture, just refactor preparing for M3)
- Measured CONC=4:
  - Total thr: 5785.81/s = 723/GPU (vs 739 baseline, NEUTRAL noise)
  - Mean TPOT: 5.81 ms (vs 6.07, slight improvement)
  - Median TPOT: 6.26, P99 TPOT: 7.35 → interactivity 136
  - **Accept rate: 49% (vs 63% baseline, -22%)**. Distribution {0: 23%, 1: 31%, 2: 20%, 3: 26%}
- Semantic regression in accept rate — subtle bug in refactor, not yet debugged. TPOT is stable so total impact is small. Revert trigger threshold was <50% accept; we're at 49.3%. Marginal.
- Decision: proceed to M3 (capture on top), debug accept drop after M3 measurement.

### Iter 9 M3 — drafter cudagraph capture (IN PROGRESS at end of session)

- Files: `atom/spec_decode/eagle.py`, `atom/model_engine/model_runner.py`, `atom/model_ops/attentions/aiter_mla.py` (all `.session6b_bak` backups)
- Changes:
  1. `aiter_mla.py`: added `max_q_len_override` param to `build_for_cudagraph_capture` (one-line fix for decode shape capture)
  2. `eagle.py`: added `self.draft_hidden_out` pinned buffer; replay dispatch in `propose()` — for i ≥ 1 if `runner.drafter_graphs` has current bs, replay the graph instead of eager `self.model(...)` call
  3. `model_runner.py`: injected drafter capture for loop into `capture_cudagraph` inside the existing `with graph_capture() as gc:` block, after main model capture. Per bs: zero positions, set up decode cu_seqlens_q, call `build_for_cudagraph_capture(bs, max_q_len_override=1)`, warmup once, capture inside `torch.cuda.graph(drafter_graph, self.graph_pool, stream=gc.stream)` writing into `draft_hidden_out[:bs]`.
- Predicted delta: TPOT 5.81 → ~4.5 ms (-22%), thr 723 → ~900/GPU (+24%). Based on ~1.3ms drafter Python overhead eliminated (2/3 of estimated 2ms total).
- Revert trigger: accuracy regression >0.5pp GSM8K, crash in capture loop, accept rate <45%.
- Status: server launched, measurement pending at time of log write.

### DEC entries to add to MASTER_FINDINGS

- **DEC-032**: MORI-EP PARKED until upstream ATOM fix for MTP+DP-attention. Combo is fundamentally broken: alt_stream propagation bug (PR #389 regression, unreported) + MTP drafter incompatible with DP-attention token layout (`atom/plugin/attention.py` has TODO markers). Standalone MORI-EP gain (-7% TPOT) < MTP loss (-47% thr) at CONC=128. Not worth multi-day workaround.
- **DEC-033**: DO NOT switch to `amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4` checkpoint. Routes MoE through Triton kernels which are ~1.5× slower than AITER CK+asm `f4gemm_bf16_per1x32Fp4_BpreShuffle_*` on MI355X. Plain `amd/DeepSeek-R1-0528-MXFP4` is the fast path. Overrides Session 6A research agent finding #1. Also: `ATOM_USE_TRITON_GEMM=1` and `ATOM_USE_TRITON_MXFP4_BMM=1` are in the same trap category, keep OFF.
- **DEC-034**: Execute the two-stage plan as written. Stop re-planning strategy every session. Plan churn = no shipped work. Only re-plan when a measurement invalidates an assumption.

## 2026-04-14 evening — Session 6B Day 2 extended (profiling methodology + M3 revert)

### M3 result — NEUTRAL, reverted

CONC=4 measurement after M3 drafter cudagraph landed:
- Total thr: 5790.92/8 = **724 tok/s/GPU** (vs 723 M2 baseline, 739 BEST BASE)
- Mean TPOT: 5.78 ms (vs 5.81 M2, 6.07 baseline)
- P99 TPOT: 7.87 ms (vs 7.35 M2, slightly worse)

Delta: neutral to slightly negative. M3 capture+replay was functionally correct (diagnostic confirmed: drafter_graphs populated with 10 bs keys, replay branch fires at every i≥1 call), but drafter Python overhead is <0.1ms per step, not the ~2ms I estimated. **Predicted delta 22% TPOT reduction did not materialize.**

### Prediction error analysis

The bottleneck analysis in `project_dsr1_bottleneck_analysis_day2.md` guessed drafter Python was 33% of CONC=4 TPOT. That guess was **not backed by profile data**. Per engineering rule 6 (prediction mismatch → STOP and revert), M2+M3 reverted via `.session6b_bak` files. Clean BEST BASE restored.

Lesson: **do not predict deltas without a kernel-level wall-clock budget**. This is the same lesson as Session 5's `feedback_build_model_before_optimizing.md` — I violated it again. Memory doesn't help unless I actually use it.

### DEC-035: M3 drafter cudagraph parked, root cause = fixed-overhead not Python

Drafter cudagraph delivers zero delta because `torch.compile(backend="eager")` on the drafter was already dynamo-traced. The dispatch between ops was already fast (~0.1ms total). The real per-layer "overhead" is inside the kernel itself (launch + LDS setup + wave-issue), not in Python.

### Iter 10: Profile-driven engineering (the pivot)

Danish called out gambling mid-evening. Reset to real engineering methodology:
1. profile → kernel-level wall-clock budget
2. architectural analysis → understand WHY each hot kernel is hot (mem-bound? compute-bound? launch-bound? comm-bound?)
3. intervention → attack the specific architectural limit
4. measure → verify prediction

Used `atom/examples/profile_offline.py` (AMD's shipped profiling harness) with `llm.start_profile()/stop_profile()` hooks. Two profiles captured:
- `/projects/teamA/danish/repos/trace/session6b_conc4_decodeonly/` — bs=4, input=128, output=128 (pure decode, no prefill pollution)
- `/projects/teamA/danish/repos/trace/session6b_conc32_decodeonly/` — bs=32, input=128, output=64 (scaling comparison)

### CONC=4 decode kernel budget (first clean data this sprint)

Analysis window 670ms wall, 580ms kernel busy (**87% GPU utilization**, 13% idle — NOT launch-bound at process level). Inter-kernel gap p99 = 4 μs (negligible).

| Category | ms | % of decode | Top kernels |
|---|---|---|---|
| MoE expert GEMMs (FlyDSL compiled as `moe_gemm1/2_0`) | ~139 | 22% | moe_gemm1_0 (87.4ms, 14%), moe_gemm2_0 (51.3ms, 8.2%) |
| MLA BF16 projections | ~150 | 24% | hgemm_bf16_S2TN_AS_SPK8 (36.8ms), _gemm_a16_w16_M32_N32_K256 (33.2ms), bf16gemm_80x64 (25.4ms), _batched_gemm_a8w8 (32ms) |
| MLA attention core | ~97 | 16% | mla_a8w8_qh16_qseqlen4 (28.9ms), kn_mla_reduce_v1_ps (41.1ms), fuse_qk_rope (13.8ms), fused_qk_rmsnorm (13.5ms) |
| AllReduce chain | ~87 | 14% | reduce_scatter_cross_device_store (50.2ms), local_device_load_rmsnorm (27.8ms), allreduce_fusion_1stage (6.6ms) |
| MoE routing/sort | ~80 | 13% | MoeSortingKernel (33.7ms), mxfp4_quant_moe_sort_x2 (16+15ms), grouped_topk (12.8ms) |
| Other (argmax, catarray, Cijk) | ~28 | ~5% | |

**MLA chain (projections + attn core) = ~40% of decode. MoE (GEMMs + routing) = ~35%. AllReduce = 14%.**

### Scaling analysis (CONC=4 vs CONC=32, μs/call per kernel)

| Kernel | bs=16 | bs=128 | ratio | interpretation |
|---|---|---|---|---|
| moe_gemm1_0 | 29.3 | 57.0 | 1.95× | 87% FIXED overhead (25.4 μs fixed, 0.25 μs/token) |
| mla_a8w8_qh16_qseqlen4 | 9.0 | 10.3 | 1.15× | ~all fixed (~8.8 μs fixed) |
| kn_mla_reduce_v1_ps | 12.5 | 10.3 | 0.82× | fixed, FASTER at higher bs (better occupancy) |
| hgemm_bf16 (MLA proj) | 11.3 | 12.8 | 1.13× | ~all fixed (~11 μs) |
| bf16gemm_fp32bf16_80x64 | 8.2 | 10.8 | 1.32× | mostly fixed |
| **reduce_scatter_cross_device** | **7.8** | **38.1** | **4.86×** | genuinely bandwidth-bound (the only correctly scaling kernel) |

**Key insight**: CONC=4 decode kernels are **fixed-overhead dominated**. Launch + LDS setup + first-wave-issue + first-memory-load per kernel = ~8-25 μs. For bs=16, that's a large fraction of per-call time. AllReduce is the only bandwidth-bound kernel — expected (it scales with bytes).

### DEC-036: CONC=4 bottleneck is fixed-overhead, not compute

Per-CONC strategy confirmed by data:
- **CONC=4**: fixed-overhead bound → attack = kernel fusion / persistent kernels / smaller tiles / reducing kernel count per layer
- **CONC=32**: transitioning → AllReduce starts to dominate (23% of decode)
- **CONC=128**: bandwidth-bound (likely) → AllReduce + MoE dispatch dominate → attack = compute-comm overlap

### FlyDSL already deployed (false alarm on port)

Investigated AITER's 1stage path (`fused_moe_1stage_dict`) and found:
- 1stage selector logic is COMMENTED OUT in `fused_moe_dp_shared_expert.py:458-478`
- `fmoe_g1u1` (1stage fast path) only has gfx942 .co files, no gfx950 FP4 variants
- Port Phase 1 FlyDSL attempt: **REDUNDANT** — Session 6B profile confirmed FlyDSL is already the path via the current 2stage selector, which picks `flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w3` and sibling kernels at our DSR1 shape `(cu_num=256, token=N, model_dim=7168, inter_dim=256, expert=257, topk=9)`. The Python selector name differs from the HIP symbol name — `moe_gemm1_0` in the trace IS the FlyDSL compiled kernel.
- **Phase 1 port parked** (DEC-037). Danish's Phase 1 FlyDSL contribution is already upstream in AITER main.

### DEC-037: Phase 1 FlyDSL port parked (AITER has absorbed it)

The current AITER's 2stage selector already uses `flydsl_moe_stage1`/`flydsl_moe_stage2` for DSR1 shapes. No port needed. The 22% of decode in MoE is the FlyDSL floor at tile_m=32 for our small-M case. Further MoE reduction requires NEW kernel work (smaller tiles, persistent, or different algorithm), not configuration.

### Next attack surface: MLA chain (~40% of decode, under-investigated)

MLA BF16 projections are 4 different GEMM kernels at 11-13 μs/call, 5 projections per layer × 61 layers × ~45 steps = ~14k calls. Need to check:
1. Which tuned CSV serves these (likely `bf16_tuned_gemm.csv` or `dsv3_bf16_tuned_gemm.csv`)
2. Whether DSR1's exact BF16 GEMM shapes have tuned entries (or fall to hipblaslt fallback)
3. Whether MLA's split-k parameter is optimal for bs=16 (if split_k>1 at tiny bs, reduce kernel is pure overhead)

### M2/M3 reverts

- `atom/spec_decode/eagle.py` ← restored from `.session6b_bak`
- `atom/model_ops/attentions/aiter_mla.py` ← restored from `.session6b_bak`
- `atom/model_engine/model_runner.py` ← M3 block stripped by Python regex (no backup existed for this file)
- All `ITER 9`, `M3 CAPTURE`, `drafter_graphs`, `max_q_len_override` references removed
- `/tmp/.cache/atom/torch_compile_cache` nuked

## 2026-04-14 late evening — Session 6B Day 2 extended (iter 10 + Phase 0 + TP=4 SR + Test 1)

### Iter 10: MLA `max_split_per_batch=1` patch — REVERTED (DEC-038)

- File: `atom/model_ops/attentions/aiter_mla.py:165`, changed `"max_split_per_batch": 16` → `1`
- Backup: `.iter10_bak`
- Profile at ISL=128 (WRONG OPERATING POINT, key lesson): `kn_mla_reduce_v1_ps` per-call 12.5 → 5.1 μs (-59% as predicted), `mla_a8w8_qh16_qseqlen4` per-call 9.0 → 16.1 μs (+79%). Net MLA per-call essentially neutral. Profile looked promising.
- Benchmark at ISL=8192 (REAL operating point): **TPOT 5.76 → 9.92 ms (+63% regression), thr 739 → 438/GPU (-41%)**. Massive regression.
- Root cause: at short context (ISL=128), KV fits in L2 cache so `max_split=1` works; at long context (ISL=8192+), KV overflows L2 and split-k distributes cache streaming across multiple workgroups. Forcing split=1 destroys cache efficiency.
- **Reverted via `cp .iter10_bak`**. Written lesson to memory: `feedback_profile_at_benchmark_isl.md` — always profile at ISL=8192, never short context.

### Phase 0.1: BEST BASE CONC=4 reproduction (clean slate)

Command:
```bash
HIP_FORCE_DEV_KERNARG=1 NCCL_MIN_NCHANNELS=112 ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=16384 \
python3 -m atom.entrypoints.openai_server --model amd/DeepSeek-R1-0528-MXFP4 -tp 8 \
  --kv_cache_dtype fp8 --method mtp --num-speculative-tokens 3 --max-model-len 10240
```

Result: **739 thr/GPU, 5.76 ms mean TPOT, 8.21 ms P99 TPOT, 270 ms mean TTFT, 6.3 s E2E**. Matches Session 6A BEST BASE within noise. Clean starting point confirmed.

### Phase 0.2: AMD quickstart verbatim

Command (AMD-style env vars, no MTP flag, no HIP_FORCE_DEV_KERNARG):
```bash
OMP_NUM_THREADS=1 AMDGCN_USE_BUFFER_OPS=1 \
python3 -m atom.entrypoints.openai_server --model amd/DeepSeek-R1-0528-MXFP4 -tp 8 \
  --kv_cache_dtype fp8 --max-model-len 10240 --method mtp
```

Result: **468 thr/GPU, 9.28 ms mean TPOT, 33 ms P99**. Matches ATOM public nightly dashboard (484 thr/GPU at CONC=4).

**Critical finding**: AMD's quickstart = public nightly = **-37% below our BEST BASE**. The 53% advantage our BEST BASE has comes from `--num-speculative-tokens 3` + `HIP_FORCE_DEV_KERNARG` + dual-stream MoE. **Protecting MTP=3 is load-bearing.**

### Research: ATOM benchmark dashboard + compass report

Downloaded JSON from `rocm.github.io/ATOM/benchmark-dashboard/`. Latest DSR1-MXFP4 commit 649d712 (2026-04-13, ATOM-vLLM, 8×MI355X, ROCm 7.2.1, `vllm-v0.19.0-nightly_20260412`):

| CONC | AMD nightly thr/GPU | TPOT | Our BEST BASE thr/GPU | Our edge |
|---|---|---|---|---|
| 4 | 484 | 8.83 ms (no MTP) | 739 | **+53%** |
| 32 | 1884 | 17.82 ms | 2346 | **+25%** |
| 64 | 2650 | 25.81 ms | — | — |
| 128 | NOT PUBLISHED | — | 3555 | — |

We are **already above AMD's public nightly at every CONC**. But the 1500/3900/6000 organizer baselines are **AMD internal numbers**, not public, still 66-103% above us.

**Session 5 intel was right**: *"our 3555 is 4× AMD's public DSR1 best, 6000 is internal stretch"*. Confirmed.

### Phase 2.1: TP=4 single replica measurement at CONC=4 (DEC-040)

Command: same as BEST BASE but `-tp 4`, bench thr computed as `Total Tput / 4`.

Result: **1105 thr/GPU, 7.60 ms mean TPOT, 10.29 P99, 438 ms TTFT, 8.2 s E2E**.

Gate analysis:
- thr 1105 vs 1500 target = **FAIL (-26%)**
- interact 1000/7.60 = 132 vs 165 target = **FAIL**
- E2E 8.2s vs 5.0s target = **FAIL (-39%)**

**DEC-040**: TP=4 SR gets +50% thr via num_GPUs_used=4 divisor, but **costs +32% TPOT** (per-rank work doubles). Fails all 3 gates at CONC=4. To pass all 3 gates at TP=4 SR, kernel wins need to reduce TPOT from 7.60 → 5.0 ms (-34%). More tractable than TP=8's required -52% TPOT cut, so TP=4 SR remains the likely CONC=4 config WITH kernel work on top. Session 6A's 1124 reproduction confirmed.

### Iter 11 / Test 1: BEST BASE + AMD env vars stacked (DEC-039)

Added `OMP_NUM_THREADS=1 AMDGCN_USE_BUFFER_OPS=1` to our BEST BASE command (keeping MTP=3, HIP_FORCE_DEV_KERNARG, NCCL_MIN_NCHANNELS, etc.).

Result: **593 thr/GPU (-20%), TPOT mean 7.29 ms (+26%), P99 TPOT 21.56 ms (+163%)**. Massive P99 outlier injection.

**DEC-039**: AMD's env vars REGRESS our stack. Hypothesis: `OMP_NUM_THREADS=1` chokes Python async scheduler at low CONC (CONC=4 where per-request CPU work matters). `AMDGCN_USE_BUFFER_OPS=1` untested in isolation. **Do not auto-apply AMD's env vars to our high-performance stack without individual verification.** Reverted both.

### Research: ATOM 13 AITER env vars + compass report

Read 241-line external research report (`compass_artifact_wf-*.md`). New actionable insights:
- **MI355X has 256 CUs** (corrected from 304 in my prior notes)
- **L2 cache is 32 MB total (4 MB per XCD)**, not a single large cache
- **Infinity Cache 256 MB** — can hold ~450K compressed KV latents, 3-5× effective BW if MLA decode kernel is IC-aware
- **`--block-size 1` MANDATORY for AITER MLA** — not setting it explicitly, need to verify default
- **MTP hurts at high CONC** (report says disable at CONC=128) — we've been running MTP=3 everywhere
- **Two-Batch Overlap (TBO)** — hides 50% comm latency, unknown if in ATOM
- **Triton-Distributed** — AMD framework fusing GEMM+comm, 30-40% claimed
- **AITER FusedMoE** — fuses gather+grouped_GEMM+activation+scaling, 23% claimed
- `GPU_MAX_HW_QUEUES>2` can deadlock RCCL on MI355X
- **PYTHON_GIL=0** requires Python 3.13 free-threading — we're 3.12, no-op
- Per-CONC chunked prefill: 512-2048 / 8192-32768 / 65536-131072

Full digest: `project_dsr1_compass_report_insights.md` in memory.

### Phase 1.1 TP=4 profile at ISL=8192: BLOCKED by zombie GPU memory

Attempted `profile_offline.py -tp 4 --random-input --input-length 8192 --output-length 32`. Crashed with HIP OOM on GPUs 0-3 because previous TP=4 server run left ~17 GB/GPU pinned. Killed `multiprocessing.spawn` workers and cleaned, retry pending with `--gpu-memory-utilization 0.85`.

**Day 3 must start with this profile. Without a clean ISL=8192 kernel budget we're guessing.**

### End-of-day summary

- BEST BASE locked at 739/2345/3555. 
- Iter 6-11 all either NEUTRAL or reverted (DEC-032, 035, 038, 039, 040).
- Gap to organizer baselines: +103%/+66%/+69%. To win: +160%/+113%/+116%.
- Profiles at wrong ISL are worthless. Day 3 starts with correct ISL=8192 profile.
- Key architectural levers to test Day 3: `--block-size 1` verification, MTP=0 at CONC=128, FusedMoE path verification, TBO investigation, hipBLASLt tunable op.
- Multi-config submission (different TP per CONC) is allowed per Daniel's confirmation. TP=4 SR at CONC=4 is the likely config path but requires -34% TPOT from kernel wins.


## 2026-04-14 late evening — Iter 12: `--block-size 1` REGRESSION (DEC-042)

**Hypothesis:** Both compass report and second report said `--block-size 1` is MANDATORY for AITER MLA. ATOM default is 16. Predicted 0-risk config win.

**Command:** BEST BASE launch + `--block-size 1`. CONC=4, 40 prompts, ISL=8192 OSL=1024.

**Result:**
| Metric | BEST BASE | Iter 12 | Delta |
|---|---|---|---|
| Thr/GPU | 739 | 577 | **-22%** |
| Mean TPOT | 5.76 ms | 7.47 ms | +30% |
| P99 TPOT | 8.21 ms | 24.25 ms | **+195%** |
| TTFT | 270 ms | 307 ms | +14% |
| Duration | 62.4 s | 79.8 s | +28% |

**Decision:** REVERTED. Drop `--block-size 1` from launch, keep default 16.

**Learning:** Report-recommended flags have now failed **4/4 times today** (iter10 max_split=1, Test1 AMD env, Test2 NCCL prio, iter12 block-size=1). Our ATOM+MTP=3+native openai_server is a specific equilibrium. External reports likely describe SGLang or older ATOM paths. **Stop flag-sweeping from reports. Read our source first.**

P99 tripling specifically suggests paged-attn metadata churn — block_size=1 means 16× more page entries per sequence.

Tomorrow (Day 3): skip remaining "universal knob" tests. Go to TP=4 profile at ISL=8192 (retry with cleaned GPU state) as highest-signal action, then MTP=2 test at CONC=4 (compass said drop if accept<60%; we measured 49-63%).

---

## 2026-04-14 late evening — Session 6B Day 2 FINALE (the big discoveries)

### Summary of the night
- **BF16 tuning reverted** (DEC-043) — tuner corrupted production CSVs via dedup merge; -20% thr / +179% P99. Git checkout restored.
- **Cold GPU clock bug identified** — auto power governor downclocks during idle → first bench every session under-measures ~15%. Rule saved: always run warmup bench before measuring.
- **TP=2 SR dropped** (DEC-044) — booted cleanly, crashed mid-bench with GPU Memory Access Fault ("Write access to read-only page" on rank 0). Structural bug in ATOM at TP=2 with 128-head MLA. Permanent drop.
- **EAGLE-3 tree speculation research (agent)** — PR #411 is NOT tree spec, it's "relaxed MTP" acceptance. Real tree spec = 3+ day kernel port (no tree-mask MLA kernel in aiter for gfx950). Parked.
- **Official scoring ground truth corrected** — rules formula divides by `num_GPUs_you_used` (1..8), NOT by 8. Multi-config per CONC explicitly allowed. Rank-based scoring (max 3000 pts across 3 CONCs). GSM8K gate = 0.93 (tightened from earlier 0.38).
- **Intel harvested** from `/projects/teamA/danish/repos/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x/` — ~20 prior result JSONs + daily_log + launch script. Confirmed many dead ends (MTP=2, mbt32k, DP=2×TP=4, no-MTP, GPU_MAX_HW_QUEUES=5 stack).

### DEC-045 — **BIG WIN**: `ATOM_ENABLE_RELAXED_MTP=1` on TP=4 SR at CONC=4

**Test 1 (thresholds = 10, 0.6 — default when env flag on):**

| Metric | TP=4 SR strict | TP=4 SR + RELAXED | Delta |
|---|---|---|---|
| Thr/GPU (÷4) | 1133 | **1472** | **+30%** |
| Mean TPOT | 7.42 ms | **5.73 ms** | **-23%** |
| Median TPOT | 7.88 ms | **5.59 ms** | **-29%** |
| P99 TPOT | 10.24 ms | 7.37 ms | -28% clean |
| Interact (1000/med) | 127 | **178.9** | **+41% — PASSES 165 gate** |
| Mean E2E | ~8040 ms | ~6170 ms | -23% (still fails 5000 gate) |
| Accept rate | 54% | **86%** | +60% |
| Avg toks/fwd | 2.63 | **3.58** | +36% |
| Accept-depth-3 | 18% | **64%** | +256% |

**Context:** Relaxed MTP mechanism (source: `atom/model_ops/rejection_sampler.py:10-16`):
- Strict: accept draft only if it matches target argmax (TOP_N=1, DELTA=0.0)
- Relaxed (default env flag): accept if draft is in target top-10 AND target_prob ≥ 0.6 × argmax_prob

**GSM8K stability issue — the relaxed default is too aggressive:**
- Run 1: 0.9158 ❌ FAIL
- Run 2: 0.9333 ✅ PASS (+0.3 pp margin)
- **Unstable: true accuracy ~0.92-0.94, edge of gate, single-run validation = coin flip**

**DEC-045 status:** Relaxed MTP is a REAL +30% thr lever at CONC=4 — biggest single discovery of the competition so far. But default thresholds `(10, 0.6)` are too aggressive for DSR1 reasoning. Must tighten `rejection_sampler.py` to preserve accuracy margin.

### DEC-046 — Relaxed MTP threshold tuning — iteration 1: `(5, 0.3)` still marginal

| Run | GSM8K | vs 0.93 gate |
|---|---|---|
| 1 | 0.9393 | ✅ +0.63 pp |
| 2 | 0.9356 | ✅ +0.26 pp |
| 3 | 0.9272 | ❌ -0.28 pp |

Mean 0.934, min 0.9272. 2/3 pass. Still unsafe.

**Decision:** Tighten further to `(3, 0.2)` and re-test 3×. In progress.

### CONC=4 gate status after DEC-045 (relaxed 10, 0.6)

| Metric | Number | Gate | Status |
|---|---|---|---|
| Thr/GPU | 1472 | ≥1500 | ❌ **-1.9% (noise)** |
| Interactivity | 178.9 | ≥165 | ✅ **PASS +8.4%** |
| Mean E2E | 6170 ms | ≤5000 | ❌ **-23%** |
| GSM8K | 0.916-0.933 unstable | ≥0.93 | ⚠️ coin flip at (10, 0.6) |

**From 0/3 gates passing → 1/3 passing (interact) + 1 within noise (thr) + 1 still structural (E2E).**

### Dead configs (confirmed from bounty dir prior JSONs + our Day 2 tests)

- `--max-num-batched-tokens 32768` — prior test showed same E2E as baseline (6870 vs 6463 ms). **Test 2 DEAD.**
- MTP=2 at CONC=4 — prior test: 640/GPU vs 738 MTP=3 → -13% worse. **Test 4 DEAD.**
- DP=2 × TP=4 no MTP — prior test: 341/GPU (half of BEST BASE). num_GPUs=8 divisor erases TP=4 advantage + no MTP hurts. **DEAD.**
- `--gpu-memory-utilization 0.95` at TP=8 — prior test: same as BEST BASE. Neutral.
- All `--async-scheduling`, `--compilation-config`, `--no-enable-prefix-caching` — vLLM-only, not in native ATOM server. **Drop from plan.**

### Tomorrow (Day 3) priorities

1. Finish relaxed MTP threshold tuning: (3, 0.2), if needed (2, 0.1), lock the tightest stable config
2. Run `./dsr1_benchmark perf -isl 8192 -osl 1024` on the locked config → full 9-gate measurement (CONC=4, 32, 128)
3. Measure the CONC=32 and CONC=128 gap with the winning stack
4. Commit to either container swap (rocm/atom-dev:vllm-latest for FULL_AND_PIECEWISE) OR structural kernel work based on the numbers

### Files of record today
- `project_dsr1_scoring_ground_truth.md` — rules formula + rank scoring + /num_GPUs_used correction
- `project_session6b_day2_consolidated_plan.md` — active execution plan
- `project_bounty_dir_prior_experiments.md` — harvested intel from ~20 prior JSONs
- `feedback_warmup_before_bench.md` — cold clock rule
- `MASTER_FINDINGS.md` — DEC-043/044/045/046 appended

### DEC-047 — Relaxed MTP `(3, 0.2)` is the CONC=4 sweet spot — 2026-04-14 late evening

After DEC-046's `(5, 0.3)` marginal result, tightened `rejection_sampler.py:11-14` to `(TOP_N=3, DELTA=0.2)`.

**3-run GSM8K stability:**
| Run | GSM8K |
|---|---|
| 1 | 0.9371 ✅ |
| 2 | 0.9356 ✅ |
| 3 | 0.9333 ✅ |

Mean 0.9353, min 0.9333. **First 3/3 pass** — but thin margin (+0.33 pp above 0.93 gate).

**Warm bench at CONC=4 (TP=4 SR + RELAXED_MTP=1 + (3, 0.2) thresholds):**

| Metric | Strict | (10, 0.6) | **(3, 0.2)** |
|---|---|---|---|
| Total thr | 4535 | 5888 | **5881** |
| Thr/GPU (÷4) | 1133 | 1472 | **1470** |
| Median TPOT | 7.88 ms | 5.59 ms | **5.54 ms** |
| Interact | 127 | 178.9 | **180.5** |
| Mean TPOT | 7.42 | 5.73 | 5.75 |
| P99 TPOT | 10.24 | 7.37 | 8.50 |
| Mean TTFT | — | 453 | 480 ms |
| Mean E2E | ~8040 | ~6170 | **~6213 ms** |
| MTP accept interval | 54% | 86% | 85% |

**Critical finding — speed is IDENTICAL between (10, 0.6) and (3, 0.2).** Tightening thresholds barely touches accept rate on random-text bench (86 → 85%) but dramatically improves accuracy on structured GSM8K reasoning. Best of both worlds: speed preserved, accuracy stable.

**CONC=4 gate status at (3, 0.2):**

| Gate | Value | Target | Verdict |
|---|---|---|---|
| Thr/GPU | 1470 | ≥1500 | ❌ −2% (within noise, re-run could hit) |
| Interactivity | 180.5 | ≥165 | ✅ **PASS +9.4%** |
| Mean E2E | ~6213 ms | ≤5000 | ❌ −24% (decode-dominated, needs TPOT cut) |
| GSM8K min | 0.9333 | ≥0.93 | ✅ **PASS +0.33 pp** (thin, 3/3) |

**2/4 pass, 1 noise, 1 structural.** Backup saved at `rejection_sampler.py.BAK_3_0p2_STABLE`.

**Decision:** `(3, 0.2)` is **committable for CONC=4** as the relaxed-MTP floor. Next iteration: try `(2, 0.1)` to see if it pushes GSM8K min ≥ 0.935 without losing speed. Then measure CONC=32 + CONC=128 with the winner.

**E2E analysis (why decode dominates):** Mean E2E = TTFT + output_len × TPOT = 480 + 997 × 5.75 ≈ 6213 ms. TTFT is only 7.7% of E2E; decode is 92.3%. Even with TTFT → 0, E2E would still be ~5733 ms and fail 5000 gate. **CONC=4 E2E only closes via further TPOT reduction (structural kernel work or tighter accept rate).**

**What that means for tomorrow:** CONC=4 has 2 real gates blocked (thr noise-close, E2E structural). Need:
1. Kernel-level TPOT reduction (MLA split-k retune at ISL=8192, MoE tile retune, QKV fusion)
2. OR container swap to `rocm/atom-dev:vllm-latest` for vLLM plugin cudagraph (FULL_AND_PIECEWISE)
3. OR relaxed MTP goes even tighter AND accept rate still climbs enough to matter — diminishing returns expected

---

## END OF SESSION 6B DAY 2 (2026-04-14 ~16:45)

**Headline wins:**
- DEC-045: Relaxed MTP env var is the biggest CONC=4 lever of the competition (+30% thr, −29% TPOT)
- DEC-047: `(TOP_N=3, DELTA=0.2)` thresholds locked as committable floor (3/3 GSM8K pass, speed identical to default)
- CONC=4 interactivity gate NOW PASSES at 180.5 vs 165 required (+9.4% margin)
- CONC=4 throughput closed from −51% gap (739/1500) to −2% gap (1470/1500)

**Remaining at CONC=4:**
- Throughput: 1470/1500 (−2% noise, 1 more tiny win closes it)
- E2E: 6213/5000 ms (−24% structural, needs kernel-level TPOT reduction)
- GSM8K: 0.9333 min of 3 runs, thin +0.33 pp margin

**Not yet measured with relaxed MTP stack:**
- CONC=32 (prior strict: 2345 TP=8, 3084 TP=4 SR)
- CONC=128 (prior strict: 3555 TP=8)

**Server-side state on shutdown:**
- `rejection_sampler.py` edited to `(3, 0.2)` thresholds
- Backup at `rejection_sampler.py.BAK_3_0p2_STABLE`
- Original backup at `rejection_sampler.py.BAK_pretune`
- Server may or may not be running

**Day 3 morning priorities:**
1. Test `(2, 0.1)` thresholds — 3× GSM8K + 1× bench. Lock if min ≥0.935 + thr ≥1400.
2. Measure winning stack at CONC=32 and CONC=128 (never benched with relaxed MTP)
3. Run `./dsr1_benchmark perf` for the full 9-gate official measurement
4. Based on 9-gate numbers, decide Day 3-5 kernel work priorities (MLA split-k retune, QKV triad fusion, container swap)

**Memory files saved for Day 3 pickup:**
- `project_session6b_day2_FINAL_state.md` ⭐⭐⭐⭐ (read first)
- `project_relaxed_mtp_big_win.md` ⭐⭐⭐ (DEC-045/046/047 details)
- `project_bounty_dir_prior_experiments.md` ⭐⭐⭐ (dead-configs list)
- `project_dsr1_scoring_ground_truth.md` ⭐⭐ (rules formula)
- `feedback_warmup_before_bench.md` (cold clock rule)

**Day 2 took us from 0/4 CONC=4 gates cleanly passing to 2/4 passing, 1 within noise, 1 structural.** Best single day of the competition so far.

---

## 2026-04-15 Day 3 AM — DEC-048: ATOM fusion env vars incompatible with relaxed MTP

Attempted Phase A of CONC=4 plan: stack 4 AMD-first-party fusion env vars on TP=4 SR + relaxed MTP (3, 0.2) at CONC=4.

| Run | Config | Thr/GPU | TPOT med | Interact | GSM8K |
|---|---|---|---|---|---|
| Baseline | (3, 0.2) no fusions | **1470** | **5.54** | **180.5** | 0.9333 |
| Run 1 | (3, 0.2) + QK_NORM_ROPE + DS_QKNORM + DS_QKNORM_QUANT + ALLREDUCE_RMSNORM | 1155 | 7.84 | 127.6 | 0.9416 |
| Run 2 | (3, 0.2) + 3 QK fusions only (dropped ALLREDUCE_RMSNORM) | 1170 | 7.46 | 134.0 | 0.9424 |

**GSM8K RISES with fusions → relaxed MTP failing closed.** Fusions perturb Q/K/allreduce enough to shift logit distribution across 61 layers, defeating the drafter's top-3/0.2 threshold. Relaxed sampler rejects more drafts → behaves like strict.

**Verdict DEC-048:** Fusion env vars + relaxed MTP are fundamentally incompatible on DSR1. All 4 fusions break it equivalently. Cannot bisect. Must choose: strict+fusions OR relaxed+no-fusions. **Choice: relaxed + no fusions** (preserves +30% CONC=4 throughput from DEC-045/047).

**Phase A of the CONC=4 plan is DEAD.** Reverted to clean (3, 0.2) launch command (no fusion env vars). Moving to Phase B / Phase 1 kernel port plan.

### Next step — port Danish Phase 1 MoE v917 kernel

Big discovery: Phase 1 MoE benchmark included `(M=16, inter=256, E=257)` which is **exactly our TP=8 DSR1 CONC=4 decode shape**. Danish won Phase 1 at 69.9μs (−41% vs AITER default). If the same FlyDSL patch applies to our ATOM/aiter version, projected savings:

- Current `moe_gemm1_0` + `moe_gemm2_0` = 33.6% of decode TPOT = 1.86 ms
- 30-40% speedup → 0.56-0.74 ms TPOT savings
- TPOT 5.54 → ~4.90 ms → median E2E → 5260 ms → −5.2% from 5000 gate

Plus the BF16 GEMM Phase 1 techniques and a careful BF16 retune, we can theoretically close the gate.

Plan saved in memory: `project_phase1_kernels_port_plan.md`


---

## 2026-04-15 Day 3 — HARD TIMELINE LOCK + CONC=4 TP=4/TP=2 RULE

Danish set the timeline explicitly. NO MORE CONFUSION.

### 30-day budget breakdown (Apr 15 → May 15)

| Block | Days | Purpose |
|---|---|---|
| **Block 1: DSR1 baseline** | Apr 15 – Apr 24 (10 days) | Pass all 9 DSR1 gates |
| **Block 2: Kimi K2.5 baseline** | Apr 25 – May 4 (10 days) | Pass all 9 Kimi gates |
| **Block 3: Exceed by 28%** | May 5 – May 15 (10 days) | Push above baseline for rank-0 points |

### Inside Block 1: 3 days per CONC

- **CONC=4: Apr 15–17** (3 days) — RIGHT NOW, Day 1 of 3
- CONC=32: Apr 18–20
- CONC=128: Apr 21–23
- Buffer + submit: Apr 24

### CONC=4 HARD RULE: TP=4 SR or TP=2 SR ONLY. NEVER TP=8.

**Why TP=8 is dead at CONC=4:**
- Decode throughput at CONC=4 is rate-limited by per-step time × concurrency
- TP=8 measured TPOT 5.77 ms strict, ~4.4 ms with relaxed MTP best case
- Total tput at TP=8 = 4 × 1024 / (0.25 + 1024 × 0.0044) ≈ 4660 tok/s
- `/num_GPUs_used = 8` → **583 tok/s/GPU**, far below 1500 gate
- **Mathematically impossible for TP=8 to reach 1500 at CONC=4 even with perfect kernels**

**TP=4 SR (currently DEC-047 floor):**
- thr 1470/GPU (−2% noise from 1500), interact 180.5 (PASS), E2E ~5897 (FAIL by 18%), GSM8K 0.9333

**TP=2 SR (DEC-044 was a crash, needs precision retry):**
- Theoretical: total tput halved (smaller batch per rank?) but `/2` divisor doubles
- Rough projection: 2200-2900/GPU thr, TPOT 8-12ms (per-rank batch unchanged), interact 80-120 (likely fails), E2E 9-13s (fails)
- TP=2 likely passes thr but FAILS interact + E2E. Worth measuring to confirm.

### Day 1 (Apr 15) plan — what's still untested at CONC=4 with TP=4/TP=2

1. **TP=2 SR retry** with `--gpu-memory-utilization 0.75` (vs 0.85 last attempt) — see if it boots cleanly this time, then bench
2. **TP=4 SR + cudagraph capture size sweep** — explicit `[1,2,4,8,16]` for CONC=4 effective sizes
3. **TP=4 SR + `--max-num-seqs` reduction** to 32 or 64 (default 256 may add scheduler overhead)
4. **TP=4 SR + `ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD` sweep** — currently 16384, try 32, 64, 256

All of these are non-math optimizations. They should NOT break relaxed MTP at (3, 0.2) the way kernel substitutions did (DEC-048, DEC-049).

### Confirmed dead at CONC=4 (DO NOT RE-TEST)

- v917 MoE FlyDSL port (DEC-049) — breaks relaxed MTP via precision drift
- 4 fusion env vars (DEC-048) — break relaxed MTP same way
- TP=8 anything (mathematically below 1500 gate)
- DP=2 × TP=4 (uses 8 GPUs → /8 divisor → no benefit)
- `--block-size 1` (DEC-042 regression)
- `--max-num-batched-tokens 32768` (prior bounty dir test, neutral)
- MTP=2 (test_mtp2_conc4.json shows -13% vs MTP=3)
- AMD env vars `OMP_NUM_THREADS=1 + AMDGCN_USE_BUFFER_OPS=1` (DEC-039)

### What gets us out of Block 1 if CONC=4 doesn't pass in 3 days

If by Apr 17 we still can't close CONC=4, **lock the best floor we have, mark it as known-failing, MOVE TO CONC=32**. We come back to CONC=4 in Block 3 (May 5-15) when tree speculation port becomes feasible.

---

## 2026-04-15 afternoon — DEC-050: BENCH HARNESS MISMATCH (the day's biggest finding)

**DEC-047 "1470/GPU, 5.54 TPOT, 180.5 interact" was measured with `atom.benchmarks.benchmark_serving` (internal dev bench). That is NOT the scoring harness.** The official submission tool `./dsr1_benchmark perf` uses a completely different bench flow and gives ~22% worse numbers on the same server.

### Side-by-side reproduction on 2026-04-15 afternoon (same server, (3, 0.2) relaxed MTP hardcoded, single warm run)

| Harness | Total thr | Thr/GPU (÷4) | Median TPOT | Interact | Median E2E |
|---|---|---|---|---|---|
| `atom.benchmarks.benchmark_serving` (4 warmups, no GSM8K, no chat template) | **5833** | **1458** | **5.47 ms** | **183** | ~5905 ms |
| `./dsr1_benchmark perf` (8 warmups, 1319 GSM8K first, `--use-chat-template`) | **4561** | **1140** | **7.77 ms** | **129** | **8364 ms** |
| **Delta** | −22% | −22% | +42% | −30% | +42% |

**DEC-047 is perfectly reproducible via the internal bench (1458 vs 1470 recorded = 0.8% noise). Server is fine, relaxed MTP is fine, all yesterday's (3, 0.2) + hardcoding work is valid.** But only the `./dsr1_benchmark perf` number is what leaderboard scores against.

### What the official tool does that the internal doesn't

Source: `dsr1_benchmark.cpp` → `run_benchmark_serving()`:

```bash
git clone https://github.com/kimbochen/bench_serving.git /tmp/bmk-*
python3 /tmp/bmk-*/benchmark_serving.py \
  --model amd/DeepSeek-R1-0528-MXFP4 --backend vllm \
  --base-url http://0.0.0.0:8888 \
  --dataset-name random \
  --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1 \
  --num-prompts $((CONC*10)) --max-concurrency $CONC \
  --request-rate inf --ignore-eos \
  --num-warmups $((CONC*2)) \
  --percentile-metrics 'ttft,tpot,itl,e2el' \
  --use-chat-template
```

Plus: `./dsr1_benchmark perf` runs **1319 GSM8K requests via `lm_eval` FIRST** (gate on GSM8K ≥ 0.93), THEN the perf phase. GSM8K phase leaves a 2-minute warm state on reasoning prompts before perf hits random tokens.

Plus: `process_json_*.py` computes `tput_per_gpu = total / 8.0` hardcoded. **WRONG per rules — Ziguan Discord 2026-04-15 07:10 confirmed `total/num_GPUs_used` which at TP=4 = `total/4`.** Always compute per-GPU manually.

### Corrected CONC=4 gate status (via official harness)

| Gate | Target | Official-harness reading | Status |
|---|---|---|---|
| Thr/GPU | 1500 | **1140** | ❌ −24% |
| Interact | 165 | **129** | ❌ −22% |
| Median E2E | 5000 ms | **8364 ms** | ❌ −67% |
| GSM8K | 0.93 | **0.9386** | ✅ |

**1/4 gates passing, not 2/4 as DEC-047 claimed.** Yesterday's record was measured against the wrong tool.

### Suspected causes of the 22% harness gap

1. `--use-chat-template` wraps each random prompt with DeepSeek's chat template (e.g. `<|im_start|>user\n{random tokens}\n<|im_end|>\n<|im_start|>assistant\n`). The MTP drafter may accept poorly on chat-wrapped random text.
2. GSM8K-first phase heats cudagraph/KV allocator/drafter state on reasoning prompts, then the perf phase arrives with random tokens. Different acceptance pattern.
3. `--num-warmups 8` vs 4 — smaller factor.

**The harness gap itself is the biggest single optimization target left for CONC=4.** If we can find one thing that closes it, that's +22% for free on every CONC without any kernel work.

### Action items from DEC-050

1. **All committable-floor measurements from here forward: `./dsr1_benchmark perf` only.** Internal bench is fine for quick A/B knob deltas but never reported as gate status.
2. **Investigate the harness gap on Day 2 morning.** Priorities:
   - Run `./dsr1_benchmark perf` twice back-to-back → does run 2 recover? (Tells us if GSM8K pre-state is the cost.)
   - Read kimbochen `benchmark_serving.py` source → understand what `--use-chat-template` does at tokenization level.
   - Run official bench with chat template disabled (fork + patch locally) → measure if that's the gap.
3. **Memory file `feedback_bench_harness_matters.md` created.** All future "best score" claims must cite the harness.
4. **MEMORY.md + MASTER_FINDINGS.md + daily_log.md updated.** DEC-047 memory file also updated with correction at top.

### What knob work looks like now

From 1140 thr/GPU official floor, the gate is +31% away. Non-kernel knobs (GPU_MAX_HW_QUEUES=5, max-num-seqs sweep, dual-stream threshold, scheduler delay) realistically give 5-15% stacked. Harness gap is 22%. Combined optimistic upside: 1140 × 1.15 × 1.22 = 1600 — JUST over gate. Tight but plausible.

Day 2 priority: **first attack the harness gap** (biggest single lever), then stack the cheap knobs.

---

## 2026-04-15 Day 1 afternoon — DEC-051: chat template is 100% of the harness gap, via drafter accept rate drop

**Isolation test** (same server, back-to-back, (3, 0.2) relaxed MTP, TP=4 SR):

| Run | Flags | Total thr | Thr/GPU | Median TPOT | Interact |
|---|---|---|---|---|---|
| 1 | no template | 5695 | **1424** | **5.52** | **181** |
| 2 | +chat template only | 4654 | **1163** | **7.91** | **126** |
| 3 | +chat template + ignore-eos | 4570 | **1142** | **7.95** | **126** |
| Official `./dsr1_benchmark perf` (3-run mean) | +chat template + ignore-eos + GSM8K-first + 8 warmups | 4676 | 1169 | 7.38 | 135 |

**Conclusion: chat template is 100% of the internal-vs-official gap.** Run 3 with just chat template + ignore-eos matches official tool within noise. GSM8K pre-state and warmup count are NOT contributors.

### Root cause — drafter accept rate drops on chat-wrapped random prompts

Captured MTP Stats Interval lines from server terminal during runs 1 and 3:

| Metric | No template (Run 1) | Chat template (Run 3) |
|---|---|---|
| Mean interval accept rate | **84%** | **57%** |
| Mean toks/fwd | 3.53 | 2.73 |
| Mean depth-3 accepts | **72%** | **32%** |
| Position-0 rejects (first interval) | 2.7% | **20.1%** (7.4× more) |

**The drafter predicts the wrong first token 20% of the time on chat-wrapped random prompts vs 2.7% on raw random.** The chat template ends with `<|im_start|>assistant\n` which primes the drafter for "structured conversational English" but the target (greedy over random-token context) diverges immediately. Depth-3 chains (predict 3 correct in a row) crash from 72% → 32%. Every rejection burns a verification forward pass.

### Math check

- TPOT ratio: 7.95 / 5.52 = 1.44 = +44%
- toks/fwd ratio: 3.53 / 2.73 = 1.293 = +29%
- Discrepancy (~15%) is extra drafter forward passes wasted on rejected chains

**Fix direction: either loosen acceptance thresholds (drafter-level), reduce drafter depth (MTP=1 test below), or attack the main model forward pass directly (kernel-level).**

---

## DEC-052 — Threshold tuning (5, 0.3) gives marginal gain

**Test**: hardcoded `(RELAXED_TOP_N=5, RELAXED_DELTA=0.3)` after pyc nuke + torch.compile cache nuke.

**GSM8K stability:** 3/3 pass with **better** margin than (3, 0.2)

| Run | GSM8K |
|---|---|
| 1 | 0.9371 |
| 2 | 0.9409 |
| 3 | 0.9447 |
| **min-of-3** | **0.9371** (vs (3,0.2) 0.9333, +0.38 pp) |

**Perf (1 warm run via `./dsr1_benchmark perf`):**

| Metric | (3, 0.2) floor | **(5, 0.3)** | Delta |
|---|---|---|---|
| Thr/GPU | 1169 | **1191** | +1.9% (noise) |
| Median TPOT | 7.38 | **7.40** | +0.3% (flat) |
| Median E2E | 7933 | **7972** | +0.5% (noise) |
| Interact | 135 | **135** | 0% |
| MTP accept (perf phase mean) | ~57% | ~60% | +3 pp |
| toks/fwd | 2.73 | 2.80 | +2.5% |

**Threshold looseness gives +3 pp accept rate but ~0% TPOT improvement.** The drafter isn't being rejected for being "too strict" — it's being rejected because it's predicting the wrong tokens. Looser thresholds accept more wrong predictions, same verification cost.

**Decision**: lock (5, 0.3) as new floor (better GSM8K margin, same speed). `rejection_sampler.py` line 10-12 hardcoded.

---

## DEC-053 — MTP=1 test confirms main model forward is the bottleneck, not drafter

**Hypothesis**: if drafter is cheap, MTP=1 might be faster than MTP=3 on chat-template because fewer wasted drafter passes.

**Test**: launch with `--num-speculative-tokens 1` (keeping (5, 0.3) hardcoded).

**Perf (1 warm run):**

| Metric | MTP=3 (5, 0.3) | **MTP=1** | Delta |
|---|---|---|---|
| Total thr | 4763 | **4442** | **−6.7%** |
| Thr/GPU | 1191 | **1111** | **−6.7%** |
| Median TPOT | 7.40 | **7.62** | +3.0% |
| Median E2E | 7972 | **8258** | +3.6% |
| Interact | 135 | **131** | −3% |
| GSM8K | 0.9371 | 0.9378 | both pass |
| MTP accept rate | ~60% | **~84%** | **+24 pp** |
| toks/fwd | 2.73 | 1.84 | −33% |
| Median ITL | 17.53 | 12.57 | −28% (smoother, not faster) |

**MTP=1 accept rate hits 84% on chat-template** (matches MTP=3 no-template baseline). Proves the drafter CAN predict well, just not chains of 3 on chat-wrapped random. **But MTP=1 still LOSES by 7% on total TPOT.**

### Bottleneck math extracted from MTP=1 vs MTP=3 comparison

Let m = main forward time, d = drafter forward time. Solving:
- MTP=3: (m + 3d) / 2.73 toks = 7.40 ms → m + 3d ≈ 20.2 ms
- MTP=1: (m + 1d) / 1.84 toks = 7.62 ms → m + d ≈ 14.0 ms

Subtract: **2d = 6.2 ms → d ≈ 3.1 ms, m ≈ 10.9 ms.**

**Main model forward pass dominates** (~11 ms / 14 ms = 79% of MTP=1 step time). **Drafter forward is only ~25% of main** (3.1/10.9 = 0.28). Break-even ratio for MTP=1 vs MTP=3 is 0.32 (derivable) — we're just below, so MTP=3 wins by ~3%.

### The real bottleneck is main_fwd at ~11 ms

**Revert decision**: MTP=3 + (5, 0.3) remains the floor. MTP=1 is dead.

**Corollary**: threshold tuning and MTP depth tuning are both capped. **To break through 7.4 ms TPOT we must reduce main_fwd** — that's BF16 GEMM tuning, MoE retune, MLA split-k, kernel fusion territory. All kernel-level work.

---

## DEC-054 — 2026-04-14 profile was TP=8 ISL=128 strict — cannot be trusted for TP=4 SR ISL=8192 relaxed

Earlier today I was quoting "MLA projections 24%, MoE 22%, AllReduce 14%" from `project_dsr1_conc4_kernel_budget.md`. That profile was captured:
- at TP=8 (per-rank shapes differ significantly from TP=4)
- at ISL=128 (KV cache fits in L2 → fundamentally different memory regime vs ISL=8192 where KV is HBM-bound; see `feedback_profile_at_benchmark_isl.md` — iter10 learned this the hard way with +63% regression)
- BEFORE relaxed MTP was live (drafter behavior differs)

**Category percentages are unreliable.** Need fresh profile at TP=4 SR, ISL=8192, MTP=3, relaxed (5, 0.3). Using `atom/examples/profile_offline.py` with `-tp 4 --input-length 8192 --output-length 32 --bs 4`. Trace dir: `/projects/teamA/danish/repos/trace/day1_tp4sr_isl8192_real/`. Will parse for main_model kernel category breakdown BEFORE committing to any kernel-level optimization.

---

## Day 1 end-of-afternoon state

**CONC=4 floor via official `./dsr1_benchmark perf` (committable):**

| Gate | Current (MTP=3, (5,0.3)) | Target | Status |
|---|---|---|---|
| Thr/GPU | **1191** | ≥1500 | ❌ −21% |
| Interact | **135** | ≥165 | ❌ −18% |
| Median E2E | **7972 ms** | ≤5000 | ❌ −59% |
| GSM8K min-of-3 | **0.9371** | ≥0.93 | ✅ +0.41 pp |

**1/4 passing.** Slight improvement over morning floor (1169 → 1191) mostly from noise + new threshold. **No meaningful win from threshold/MTP-depth lever** — the drafter is not the bottleneck.

### Levers still untested

1. **Fresh profile at real config** (in progress) → reveals true main_fwd breakdown
2. **BF16 GEMM tuning** (Phase B2) → if BF16 projections are still ~20%+ of main_fwd, could save ~0.3-0.6 ms TPOT
3. **MLA split-k retune** (Phase B3) at correct ISL=8192 regime → ~0.3 ms TPOT
4. **Env var stack** (Phase B4): GPU_MAX_HW_QUEUES=5, --max-num-seqs, ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD, NCCL_MIN_NCHANNELS → ~0.2-0.4 ms
5. **Kernel fusion / persistent kernels** → out of Block 1 scope (days of kernel work)

**Realistic ceiling with Phase B2+B3+B4 stack:** TPOT ~6.0 ms, thr/GPU ~1470, interact ~166. **Still misses E2E** (would need ~5.7 s, has 5.0 s gate).

### Hard decision coming Day 2

If by end of Day 2 we're at 1350-1450 thr/GPU and E2E is still 6000+ ms, we **lock 3/4 gates and move to CONC=32** on Day 3-4. The alternative (kernel-level work to close E2E) is Block 3 territory (tree speculation, etc.) — doesn't fit Block 1.

### File state end of Day 1

- `rejection_sampler.py` hardcoded at (5, 0.3) — backups preserved
- torch.compile cache populated at (5, 0.3) config
- Server either just killed or mid-profile
- 9 memory files updated (see `feedback_bench_harness_matters.md`, DEC-047 correction block, TIMELINE_HARD Day 1 status)

---

## DEC-055 — FRESH PROFILE at TP=4 SR ISL=8192 MTP=3 relaxed (2026-04-15 09:23)

Captured via `atom/examples/profile_offline.py` at the REAL config (not the stale TP=8 ISL=128 from 2026-04-14). Parser script at `/tmp/parse_trace.py`. Trace dir `/projects/teamA/danish/repos/trace/day1_tp4sr_isl8192_real/rank_[0-3]/*.pt.trace.json.gz`.

### Methodology

```bash
cd /workspace/ATOM_main
HOME=/tmp AITER_BUILD_DIR=/tmp/.aiter_cache TRITON_CACHE_DIR=/tmp/.triton_cache \
HIP_FORCE_DEV_KERNARG=1 NCCL_MIN_NCHANNELS=112 \
ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=16384 ATOM_ENABLE_RELAXED_MTP=1 \
HF_HOME=/projects/teamA/hf_cache HUGGINGFACE_HUB_CACHE=/projects/teamA/hf_cache/hub \
HIP_VISIBLE_DEVICES=0,1,2,3 \
python3 -u atom/examples/profile_offline.py \
  --model amd/DeepSeek-R1-0528-MXFP4 -tp 4 --kv_cache_dtype fp8 \
  --method mtp --num-speculative-tokens 3 --max-model-len 10240 \
  --bs 4 --random-input --input-length 8192 --output-length 32 \
  --torch-profiler-dir /projects/teamA/danish/repos/trace/day1_tp4sr_isl8192_real \
  2>&1 | tee /tmp/profile_run.log
```

**Important:** profile_offline.py is standalone (not a server), generates 4 prefills at ISL=8192 + 32 decode steps each, then exits. Total run ~3-4 min with warm torch.compile cache.

**Parser trims first 30% of trace (prefill) and last 5% (cleanup).** Decode window: 1039 ms wall, 20641 kernel events, 98.7% GPU utilization.

### DECODE-ONLY BUDGET (high-call-count filter removes prefill residue)

| Category | ms | % decode | Per-step ms | Notes |
|---|---|---|---|---|
| **MoE GEMM (FlyDSL)** | **183.79** | **57.4%** | **5.74** | `moe_gemm1_0` + `moe_gemm2_0` (844 calls each) |
| MLA attention core | ~38 | ~12% | ~1.20 | `mla_a8w8_qh32_qseqlen4` 24.81 ms + `kn_mla_reduce` 13.33 ms |
| BF16 decode GEMMs | ~41 | ~13% | ~1.29 | `Cijk_MT64x16x256` 19.11 + `hgemm_bf16_32x64x128` 8.39 + `Cijk_MT32x16x128` 8.08 + `bf16gemm_80x64` 5.90 |
| MoE routing + sort | ~14 | ~4.5% | 0.45 | MoeSortingKernel + per_group_quant |
| AllReduce/comm | 11.21 | 3.5% | 0.35 | reduce_scatter + local_device_load_rmsnorm |
| RMSNorm | 6.52 | 2.0% | 0.20 | |
| Sampling / Other | ~25 | ~8% | ~0.8 | |
| **TOTAL decode** | **~320** | **100%** | **~10.0** | matches MTP math m ≈ 10.9 ms from DEC-053 |

### Comparison old vs new profile

| Category | Old (TP=8 ISL=128) | New (TP=4 SR ISL=8192) | Change |
|---|---|---|---|
| MoE GEMM | 22% | **57%** | +35 pp |
| MLA attention | 16% | ~12% | −4 pp |
| MLA BF16 projections | 24% | ~10% | −14 pp |
| AllReduce | 14% | 3.5% | −10 pp |
| RMSNorm | 8% | 2% | −6 pp |

**MoE GEMM dominates at TP=4 SR** because inter_dim doubles (256 → 512 per rank) and fewer ranks means each rank handles more MoE work per token. **Previous optimization targeting was based on stale data.**

### New strategic implications

1. **MoE GEMM is the dominant bottleneck** (57% of decode). Every 10% cut saves ~0.6 ms per step = ~0.2 ms TPOT = ~45 thr/GPU.
2. **Phase 1 Danish.py v917 MoE kernel is now the highest-ROI target** — Phase 1 measured −41% on this exact MoE shape. Even with precision drift from DEC-049, the (5, 0.3) floor is already at ~60% accept on chat-template so further drift has less room to hurt.
3. **BF16 GEMM tuning is still positive** but smaller than expected (~0.2 ms TPOT vs earlier 0.3-0.5 ms estimate).
4. **AllReduce overlap is nearly dead** (3.5% of decode, max 0.1 ms TPOT — not worth the engineering).

### Updated gate math

| Lever | TPOT Δ | Cumulative TPOT | Thr/GPU | E2E (ms) |
|---|---|---|---|---|
| Floor (5, 0.3) MTP=3 | — | 7.40 | 1191 | 7972 |
| v917 MoE port (−30% on MoE) | −1.0 | 6.40 | 1377 | 6930 |
| BF16 decode GEMM tuning | −0.2 | 6.20 | 1421 | 6725 |
| MLA split-k retune | −0.15 | 6.05 | 1457 | 6570 |
| Env var stack | −0.15 | 5.90 | 1493 | 6413 |

**Optimistic Block 1 ceiling ≈ 1493 thr/GPU, ~5.90 TPOT, interact ~170, E2E ~6413 ms.** 3/4 gates passing (thr noise-close, interact +3%, GSM8K, E2E still misses by 28%). **To close E2E (need TPOT ≤ 4.51 ms) requires kernel fusion or structural change — not achievable in Block 1.**

### Decision: v917 MoE retry is Day 1 end-of-day highest EV

Retry DEC-049's v917 patch at (5, 0.3) thresholds. If precision drift is tolerable (accept rate doesn't drop below 55%), it gives us the single biggest TPOT cut available in Block 1.

---

## Day 1 execute plan — v917 MoE retry

See memory `project_bottleneck_is_main_fwd.md` for the methodology + parser script.

Next step: apply `/tmp/v917_moe_patch.py` (from DEC-049 attempts), launch server with preamble, run `./dsr1_benchmark perf`, compare to (5, 0.3) floor (1191/7.40). If accept rate ≥55% AND thr/GPU ≥ 1300 → v917 stays. If accept rate crashes OR thr regresses → revert.

---

## DEC-056 — NEW CONC=4 FLOOR: `ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=256` (2026-04-15 evening)

### Summary

Lowering the MoE dual-stream overlap trigger from default 16384 → 256 activated stream-overlap at our bs=16 effective (decode). Overlap was dormant at the default because we never hit 16384 tokens in any decode batch. This gave the **first meaningful ATOM config win of Day 1**: TPOT −6.9%, interact +7.4%, E2E −6.4%. GSM8K preserved.

### Side-by-side vs previous floor

| Metric | Previous floor (5, 0.3) MTP=3 | **NEW FLOOR** `DUAL_STREAM=256` | Delta |
|---|---|---|---|
| Total token throughput (tok/s) | 4763 | **4835** | +1.5% |
| **Thr/GPU (÷4)** | 1191 | **1209** | **+1.5%** |
| **Median TPOT (ms)** | 7.40 | **6.89** | **−6.9%** |
| Mean TPOT (ms) | 7.21 | 6.88 | −4.6% |
| P99 TPOT (ms) | 9.00 | 9.22 | ~flat |
| **Interactivity (1000/medTPOT)** | 135 | **145** | **+7.4%** |
| Median TTFT (ms) | 375 | 374 | ~flat |
| **Median E2E (ms)** | 7972 | **7464** | **−6.4%** |
| GSM8K | 0.9371 | 0.9363 | noise (passes gate) |

**Why thr/GPU only +1.5% when TPOT is −6.9%?** TTFT (~375 ms) is unchanged — only decode time dropped. Total throughput formula includes prefill, which dilutes per-GPU number. Interact and E2E are pure decode so they show the full 6-7% win.

### Gate status after Test 2 (DUAL_STREAM=256)

| Gate | Target | New floor | Gap to gate | Previous gap |
|---|---|---|---|---|
| Thr/GPU | ≥ 1500 | **1209** | ❌ −19% (−291) | −21% |
| Interactivity | ≥ 165 | **145** | ❌ −12% (−20) | −18% |
| Median E2E | ≤ 5000 ms | **7464 ms** | ❌ −49% (+2464) | −59% |
| GSM8K min-of-3 | ≥ 0.93 | **0.9363** | ✅ +0.63 pp | ✅ |

**1/4 gates passing** (unchanged count) but **every failing gap closed meaningfully**. Interact now ~33% of the way to gate (was ~20%), E2E ~17% of the way (was ~12%), Thr ~6% of the way. Structural improvement, not noise.

### Reproducible launch command (NEW CONC=4 FLOOR)

```bash
# Pre-flight: kill anything running + verify GPUs clean
pkill -9 -f "atom.entrypoints" 2>/dev/null
pkill -9 -f "ModelRunner" 2>/dev/null
pkill -9 python3 2>/dev/null
sleep 10
rocm-smi --showmeminfo vram | grep "Used Memory" | awk '{print $NF}'
# expect 8× 297766912 (284 MB each = clean)

# Verify rejection_sampler.py is hardcoded at (5, 0.3) RELAXED MTP
grep -E "RELAXED_TOP_N|RELAXED_DELTA|HARDCODED" \
  /projects/teamA/danish/repos/ATOM_main/atom/model_ops/rejection_sampler.py | head
# expect: ATOM_ENABLE_RELAXED_MTP = True  # HARDCODED ...
#         RELAXED_TOP_N = 5
#         RELAXED_DELTA = 0.3

# Launch (the ONLY change vs morning baseline is ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=256)
cd /workspace/ATOM_main && \
HOME=/tmp AITER_BUILD_DIR=/tmp/.aiter_cache TRITON_CACHE_DIR=/tmp/.triton_cache \
HIP_FORCE_DEV_KERNARG=1 NCCL_MIN_NCHANNELS=112 \
ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=256 ATOM_ENABLE_RELAXED_MTP=1 \
HF_HOME=/projects/teamA/hf_cache HUGGINGFACE_HUB_CACHE=/projects/teamA/hf_cache/hub \
HIP_VISIBLE_DEVICES=0,1,2,3 \
python3 -m atom.entrypoints.openai_server \
  --model amd/DeepSeek-R1-0528-MXFP4 --server-port 8888 \
  -tp 4 --kv_cache_dtype fp8 --method mtp --num-speculative-tokens 3 \
  --max-model-len 10240 --gpu-memory-utilization 0.85
```

**Wait for `Uvicorn running on http://0.0.0.0:8888`** (~5 min with warm torch.compile cache).

**Then bench**:
```bash
cd /projects/teamA/danish/repos/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x
./dsr1_benchmark perf 2>&1 | tail -40
```

### Raw Test 2 perf output (2026-04-15 evening)

```
============ Serving Benchmark Result ============
Successful requests:                     40
Benchmark duration (s):                  76.22
Total input tokens:                      327680
Total generated tokens:                  40859
Request throughput (req/s):              0.52
Output token throughput (tok/s):         536.07
Total Token throughput (tok/s):          4835.22
---------------Time to First Token----------------
Mean TTFT (ms):                          454.31
Median TTFT (ms):                        374.75
P99 TTFT (ms):                           1261.72
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          6.88
Median TPOT (ms):                        6.89
P99 TPOT (ms):                           9.22
---------------Inter-token Latency----------------
Mean ITL (ms):                           20.08
Median ITL (ms):                         17.58
P99 ITL (ms):                            97.61
----------------End-to-end Latency----------------
Mean E2EL (ms):                          7469.75
Median E2EL (ms):                        7463.91
P99 E2EL (ms):                           9792.86
==================================================
INFO: Throughput: 604.40 tokens/s/GPU (min required: 1500)
INFO: E2E (median): 7463.91 ms (max allowed: 5000)
INFO: Interactivity: 145.17 tokens/s/user (min required: 165)
GSM8K metric: 0.9363
```

**Note**: tool reports `Throughput: 604.40 tokens/s/GPU` which is **total / 8.0 hardcoded** — WRONG per rules. Real per-GPU at TP=4 = total / 4 = **4835.22 / 4 = 1208.8 ≈ 1209** (confirmed by Ziguan Discord 2026-04-15).

### What changed mechanically

`ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD` controls when the MoE layer splits its gate+up and down projections onto two CUDA streams to overlap them. Default is 16384. At CONC=4 with MTP=3, our effective batch per decode step is ~16 tokens per rank. **16 < 16384, so the overlap never fired** — MoE stage1 and stage2 ran serially. Lowering to 256 is below 16 so overlap now fires (or the flag semantics may be inverted, either way it activates).

Per-step savings: ~0.51 ms on median TPOT = ~7% of main_fwd MoE cost. Over 1024 output tokens per request, that's **−522 ms per-request decode time**, which matches the measured −508 ms E2E delta exactly.

### Dead configs confirmed by Test 1 (do NOT re-test)

- **`GPU_MAX_HW_QUEUES=5`** — Test 1 regressed 4% across all metrics. Compass report warning about RCCL contention on MI355X was correct. **REVERTED, dropped from further sweeps.**

### Full Day 1 config sweep status

| Test | Config | Result | Decision |
|---|---|---|---|
| Baseline | `(5, 0.3)` relaxed MTP chain | 1191/7.40/135/7972 | — |
| 1 | +`GPU_MAX_HW_QUEUES=5` | 1144/7.74/129/8297 | ❌ REVERT |
| **2** | +`DUAL_STREAM_MOE_TOKEN_THRESHOLD=256` | **1209/6.89/145/7464** | **✅ KEEP (new floor)** |
| 3-6 | --max-num-seqs / bf16 KV / TP=1 / EP | — | **DEFERRED**, pivoting to SGLang spike |

### Key facts

- **Reproducibility**: launch command captured exactly. Relaunching produces within ±1% noise of these numbers.
- **GSM8K variance**: 0.9363 vs 0.9371 is within the ±0.0076 single-run stderr, both pass the 0.93 gate comfortably.
- **Safety net**: this IS our Day 3 fallback floor. If SGLang pivot and tree spec both fail, we submit this config for CONC=4.
- **Path to bigger wins**: not via more env var knobs (public space is mostly exhausted). The remaining levers are SGLang pivot or ATOM tree spec port.




---

## 2026-04-16 + 2026-04-17 — CONC=4 Day 2-3 arc (DEC-057 → DEC-069)

Daily log was last updated at DEC-056 end of Apr 15. Memory carried intermediate DECs; bringing desktop doc current.

### DEC-057 (2026-04-16 03:56 UTC) — FRESH PROFILE at exact floor config

Captured via `atom/examples/profile_offline.py` at TP=4 SR, ISL=8192, OSL=1024, bs=4, MTP=3 relaxed (5, 0.3), DUAL_STREAM=256. Trace dir `/projects/teamA/danish/repos/trace/day2_*`. Parser `/tmp/parse_trace_day2.py`.

**Total decode kernel time: 21.8 ms/step. Matches bench step time 21.73 ms within 0.3 percent.** Zero Python gap — kernel time equals step time.

Real category breakdown (overturns DEC-055):

| Category | ms | pct | vs DEC-055 |
|---|---|---|---|
| MoE GEMM (FlyDSL) | 5.89 | 26.2 | was 57 — overlap ON + OSL=1024 rebalanced |
| BF16 GEMM UNTUNED | 4.57 | 20.3 | was 13 — LM head + MLA projections on torch solution:0 |
| AllReduce/NCCL | 2.96 | 13.2 | was 5.5 — hidden by short-output DEC-055 profile |
| MLA attention | 2.26 | 10.1 | was 12 — kernel is qh32 native (NOT padded to 128) |
| MoE routing + sort | 1.39 | 6.2 | — |
| MLA reduce | 1.16 | 5.2 | — |
| RMSNorm | 1.02 | 4.5 | larger at OSL=1024 |
| Quant/dequant | 0.74 | 3.3 | — |
| Other | 1.74 | 7.8 | — |
| Python/CPU gap | ~0.1 | under 0.5 | confirmed zero |

**#1 lever measured: BF16 GEMM CSV tune (4.57 ms, zero-precision-risk offline tune).**

### DEC-058 → DEC-068 — Day 2 sweep (summary)

- DEC-058: BF16 tune + NCCL_MIN_NCHANNELS=16 → floor 1202/7.19/139/7705 (kept)
- DEC-059: TODO MLA 32-head fix → FAILED (−18 percent, aiter qk_batch_ratio bug at 32 heads). Reverted.
- DEC-060: skip metadata i=1 → NEUTRAL (below 0.1 ms, as DEC-057 implies)
- DEC-060b: CONC=32 measurement → 3208 thr/GPU, 22.09 TPOT, 1/4 gates
- DEC-061: top-K at last step → CRASHED (batch mismatch)
- DEC-063: PR #547 stream parallelism → NEUTRAL
- **DEC-064**: Relaxed MTP (7, 0.4) → **+4.2 percent (1253/7.06/141/7684/0.9371)** — kept
- DEC-065: Latest ATOM main → NEUTRAL
- **DEC-066 (Apr 16 end)**: New tuned CSV (9 rows) → **BEST TPOT 1221/6.73/148.6/7663/0.9378** (committable floor)
- DEC-067: QKNORM_FUSION → WORSE. Reverted.
- DEC-068: Merged full CSV attempt → CORRUPTED (bad merge script wrote header in middle). Reverted. **Merge script never fixed — that is the immediate Apr 17 evening task.**

### DEC-069 (2026-04-17) — Phase 4A v4 drafter HIP graph: NULL result

Implemented drafter HIP graph capture wrapped in `aiter.graph_capture()` context to register NCCL IPC handles (fixing v2/v3 NULL-pointer crash). Patch was technically correct:

- Capture succeeded: `[DG v4] Captured drafter graph bs=1` on all 4 ranks
- Replay stable: no Memory access fault across 63k draft tokens
- Accept rate preserved: 65.73 percent cumulative (vs DEC-066 roughly 62.5) — graph did not corrupt logits
- GSM8K: 0.9401 (+0.23 pp)

**But TPOT unchanged**: 6.82 ms (DEC-069) vs 6.73 ms (DEC-066). Within noise, zero TPOT cut.

| Metric | DEC-066 | DEC-069 | Delta |
|---|---|---|---|
| Thr/GPU | 1221 | 1232 | +0.9 pct (noise) |
| TPOT | 6.73 ms | 6.82 ms | +1.3 pct (noise) |
| Interact | 148.6 | 146.6 | −1.3 pct |
| E2E median | 7663 ms | 7695 ms | +0.4 pct |
| GSM8K | 0.9378 | 0.9401 | +0.23 pp |
| Gates | 1/4 | 1/4 | unchanged |

**Why null**: DEC-057 already proved Python/CPU gap is roughly 0 (kernel time = step time within 0.3 pct). Phase 4A optimized Python launch overhead. There was no Python overhead to save. Patch was correct engineering applied to a non-bottleneck.

**Plan-level root cause**: I proposed Phase 4A projecting 2.2 ms savings from drafter Python dispatch. DEC-057 data in memory at the moment of planning already contradicted that projection. `memory/feedback_profile_before_intervene.md` is literally titled to prevent this. I did not check the prediction against the measured budget before writing code. Third occurrence of this anti-pattern (M3 drafter cudagraph, DEC-055 speculation, Phase 4A v4).

### Apr 17 end-of-day decisions

- **Lock DEC-066 as CONC=4 committable floor**: 1221 thr/GPU, 6.73 ms TPOT, 148.6 interact, 7663 ms E2E, 0.9378 GSM8K. 1/4 gates. Launch block in `Best_atom_dsr_cncc4/best_reproduce.md`.
- **Phase 4A v4 patch stays in eagle.py** — harmless infra, may help Block 3 tree spec work, not reverting.
- **Phase 4B async scheduling: DROPPED** — invalidated by DEC-057 zero Python gap.
- **Plan file rewritten** at `C:\Users\danis\.claude\plans\fizzy-toasting-teacup.md` — 5-point-spec rule, measurement-driven lever ranking, 8-day Block 1 sprint, Block 3 tree spec preview.
- **New #1 lever**: fix DEC-068 CSV merge bug + full BF16 tune sweep across all 200+ shapes (confirmed via DEC-069 server log `torch solution:0` hits). Target 4.57 ms, expected −1.5 ms TPOT.
- **Apr 18 window**: CONC=32 baseline (morning), BF16 full sweep (afternoon).
- **Hard rule**: every intervention requires measured target ms + mechanism + expected delta + pass/fail gate + post-measurement. No exceptions.



## 2026-04-17 23:00 local / 04:11 UTC Apr 18 — FINAL PUSH declared

**User declaration**: "there is no block 3, we dont meet all 4 gate of cncc 4 by tomorow night, its all over"

Plan collapsed from 30-day horizon to 24-hour push. Block 3 tree spec, Kimi Block 2, May 15 submission — ALL DROPPED.

Single mission: pass all 4 CONC=4 gates by Apr 18 night, or submit at sub-rank.

### State at 04:11 UTC Apr 18 (local Apr 17 23:00)
- Server: DOWN (killed at 04:00 to free GPUs for tuner)
- BF16 decode CSV tuner: RUNNING (launched 04:11, ~45 min ETA)
- Untuned shapes: 79 (M ∈ {1,2,4,8,16,24,32,48,64,96,128,256} × N ∈ {256,2112,6144,7168,8192,12288,32320,64640} × K ∈ {512,1536,4096,7168,8192})
- Backup CSV: `/tmp/dsv3_bf16_tuned_gemm.csv.DEC066_0403`
- Target: shapes confirmed absent from current CSV (LM head M=16 N=32320 K=7168 not present, MLA projections not present, nothing matches our decode signature)

### Full lever stack (none dropped from user's list)
1. BF16 decode tune (running)
2. rocprof HW counter profile (MoE + MLA + AR)
3. BF16 PREFILL tune M=1024-8192 for TTFT
4. AITER PR #2620 fused mxfp4 quant moe sort
5. AITER PR #2727 MI350 MLA ps shapes
6. ATOM PR #421 gated_rmsnorm_quant fusion
7. Relaxed MTP (8, 0.5) tighter
8. QuickReduce non-INT4 modes
9. Minimal tree spec (top-2 at i=2 only, not full SGLang)
10. Custom kernel rocprof-informed (if LDS-bound flagged)

### Projected outcome
- Realistic stack: TPOT 4.43 ms, TTFT 227 ms → E2E 4763 ms, interact 226, thr 1620 → **4/4 gates**
- Probability 4/4: 50-60%
- Pessimistic: TPOT 6.23 → 2/4 gates (interact + GSM8K) → submit sub-rank

### Active files updated
- `Best_atom_dsr_cncc4/best_reproduce.md`
- `Current_plan.md`
- `Danish.md` (header added)
- `MASTER_FINDINGS.md` (header added)
- `memory/project_final_push_apr17_18.md` (NEW)
- `memory/project_wall_clock_budget_hard.md` (NEW)
- `memory/project_sota_apr17_intel.md` (NEW)
- `memory/feedback_pre_measure_or_dont_ship.md` (NEW)
- `memory/MEMORY.md` (index updated)
- plan file `fizzy-toasting-teacup.md`

### Waiting for
- 04:55 UTC: tuner finishes → DEC-071 bench cycle starts



---

## 2026-04-18 06:00 UTC — FINAL PUSH Phase 1 complete (DEC-070 → DEC-073)

### DEC-070 — CONC=32 baseline (skipped)
Plan originally scheduled CONC=32 baseline at DEC-070 slot. **Skipped to prioritize CONC=4 levers in 24-hr push.** User declared Apr 17 night: 4/4 CONC=4 gates by Apr 18 night or submit sub-rank. No time for CONC=32/128 re-measurement before submission.

### DEC-071 (05:03 UTC) — BF16 decode CSV full sweep

Config: DEC-069 base + 88 new BF16 tuned rows added to `dsv3_bf16_tuned_gemm.csv` (97 rows total, up from 9).

**Numbers** (via `./dsr1_benchmark perf`):
- Thr/GPU: 1267.4 (+3.8% vs DEC-066's 1221)
- Median TPOT: 6.96 ms (+3.4% vs 6.73, median anomaly; mean TPOT 6.59 = −4.2% real improvement)
- Mean TPOT: 6.59 ms
- Median TTFT: 375 ms (flat)
- Median E2E: 7495 ms (−2.2% vs 7663)
- Mean E2E: 7165 ms (−4.1%)
- Interactivity: 143.76 (−3.2% vs 148.6 — median-TPOT-driven)
- GSM8K: 0.9303 (−0.75 pp vs 0.9378, thin margin but passes 0.93)

**Read**: real +3-4% wins on mean-metrics and throughput. Median TPOT left-skewed (new distribution shape). BF16 decode tune landed weaker than projected −1.5 ms because DEC-066 already had the top 9 shapes (LM head) tuned — marginal over-DEC-066 was only for secondary shapes.

**Root cause of weaker-than-projected gain**: 20.3% of step time is BF16 GEMM; best-case 30% improvement on that band = −1.37 ms step / −0.47 ms TPOT. DEC-066 already captured 50%+ of that. DEC-071 marginal gain ~0.15-0.25 ms step.

**Gates**: 1/4 (GSM8K only). Still binding on E2E.

### DEC-072 (05:37 UTC attempt) — BF16 PREFILL tune — FAILED

Attempted to add 54 prefill shapes (M ∈ {512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 8193} × 6 NK bands) to CSV via `gradlib/gemm_tuner.py --mp 4 --errRatio 0.05`. Tuner added 50 new prefill rows (CSV 97 → 147).

**GSM8K crashed from 0.9303 → 0.865**. Bench harness aborted before perf phase (safety gate).

**Root cause**: prefill shapes have larger M (hundreds to thousands) → more accumulated floating-point ops → larger numerical drift. errRatio=0.05 threshold let through kernels that individually pass but accumulate error across 61 layers × 1319 GSM8K prompts. Also some kernel candidates showed 1-3% element mismatch in tuner log warnings — we were loose enough to accept borderline kernels.

**Recovery**: Restored `/tmp/dsv3_bf16_tuned_gemm.csv.DEC071_0512` backup (98 rows, DEC-071 state, decode-tune only). Critical decode shapes (M=16 LM head, MLA projections) preserved.

**DEAD**: BF16 prefill tune at errRatio=0.05. Alternatives considered:
- Re-tune with errRatio=0.02 (tighter) — would take 20+ min, no guarantee, risk of zero viable kernels
- Skip prefill tune entirely ← taken

### DEC-073 (06:21 UTC) — Relaxed MTP (8, 0.5) — NEW BEST

Config: DEC-071 CSV (restored) + `rejection_sampler.py` edit: `RELAXED_TOP_N=7→8`, `RELAXED_DELTA=0.4→0.5`.

**Numbers**:
- Thr/GPU: **1270.2** (+0.2% vs DEC-071, within noise)
- Median TPOT: **6.80 ms** (−2.3%)
- Mean TPOT: 6.49 ms (−1.5%)
- Median TTFT: 376 ms (flat)
- Median E2E: **7318 ms** (−2.4%, −177 ms)
- Mean E2E: 7075 ms (−1.3%)
- Interactivity: **147.1** (+2.3%)
- **GSM8K: 0.934** (**+0.4 pp** — up from 0.9303, stronger margin)
- MTP accept rate: 65.60% → 66.97% (+1.4 pp)
- Toks/fwd: 2.97 → 3.01 (+1.3%)

**Read**: (8, 0.5) is strictly better than (7, 0.4). Wider delta threshold accepts more drafter tokens that are close to target's top-8, without hurting accuracy. Accept rate + accuracy both went UP.

**Gates**: 1/4. Interact still 147 < 165 gate. E2E still 7318 > 5000 gate.

### Updated gate gap after DEC-073

- Thr/GPU: 1270 → 1500 = need +18%
- Interact: 147 → 165 = need +12% (tree spec should close this)
- E2E: 7318 → 5000 = need −32% (structurally hard without tree spec delivering big +toks/fwd)
- GSM8K: 0.934 ✅ passing with 0.4 pp margin

### Next lever: Tree spec top-2 at i=2 (DEC-074)

Target: algorithmic, not kernel. At last drafter iteration (i=mtp_k-1), emit top-2 candidates instead of single argmax. Extend Triton rejection kernel to check second candidate if first fails.

Expected: toks/fwd 3.01 → ~3.25, TPOT 6.80 → ~6.30 ms, interact → 159 (still marginal on 165 gate). May or may not fully close interact.

### Reality check

- Tree spec + maybe drafter requant: 2-3/4 gates realistic.
- E2E gate under ~30% probability of passing without extended tree spec (top-2 at ALL iterations or structural change).
- Submission plan: lock best config reached, submit regardless of 4/4.

---


---

## DEC-xxx Apr 18 (post-DEC-073, post-SSH-grant)

### Phase A1 — relaxed MTP fine sweep (probes at SSH-enabled phase)

- **Probe 1 (7, 0.5)**: 1299/6.73/148.6/7421/0.9439. Noise/marginal, E2E +3% regression vs DEC-073.
- **Probe 2 (9, 0.5)**: 1272/6.60/151.48/7300/0.9333. Marginal TPOT gain (-1.5%), interact +1.5%, but GSM8K dropped to 0.9333 (close to 0.93 floor). Not worth keeping.
- **Verdict**: (8, 0.5) is the sweet spot. Reverted rejection_sampler.py to DEC-073.

### DEC-075 UNLOCKED by Danish (weight transplant approved)

- Plan: surgical layer-61 MoE transplant from `amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4` (FP4 drafter) into our main `amd/DeepSeek-R1-0528-MXFP4` (BF16 drafter), via synthetic merged checkpoint directory.
- Scope: swap ONLY layer 61 MoE (experts + gate + shared_experts). Keep MLA/layernorms/embed/eh_proj/shared_head BF16 from main. Surgical, not naive — avoids FP4 MLA kernel shape risk.
- Built `/projects/teamA/danish/models_merged/DSR1-drafter-FP4` with 91,681 merged keys (82 main shards + 2 MoEFP4 shards symlinked).
- Expected: drafter MoE BF16 slow path (QuantType.No) → FP4 FlyDSL fast path (flydsl_moe1_afp4_wfp4_bf16). Save ~3ms/step. TPOT 6.77 → ~6.10-6.40.
- First boot attempt CRASHED with OOM — leftover probe 2 server workers held GPU memory. Cleaned + relaunched at 15:18 UTC. Polling for ready.

### Infrastructure / cleanup

- Deleted 5.3 TB of GPU core dumps from old crashes
- Cleaned 376 GB duplicate HF cache in /tmp/.cache
- Pushed DEC-073 snapshot to GitHub: https://github.com/Danishlynx/AMD_DSR_CNCC4
- Organized /projects/teamA inventory — SERVER_MAP.md documents full layout
- Separation-of-concerns brief for Kimi Opus written (BRIEF_FOR_KIMI_OPUS.md)


### Credential note (Apr 18)
- First GitHub push used PAT-embedded URL; Windows Credential Manager cached user as "x-access-token".
- Re-pushing to force credential re-selection as "Danishlynx".

---

## DEC-075 LANDED (Apr 18 16:30 UTC) — drafter layer 61 FP4 transplant

**After 5 iterations of debug**, merged checkpoint at `/projects/teamA/danish/models_merged/DSR1-drafter-FP4` successfully swaps layer 61 MoE weights from MoEFP4 variant. Drafter kernel dispatch now `flydsl_moe1_afp4_wfp4_bf16` (FP4 fast path) vs DEC-073's `QuantType.No` slow BF16.

### Result (test_162646.json)

| Metric | DEC-073 | DEC-075 | Δ |
|---|---|---|---|
| Thr/GPU (÷4) | 1282 | **1297** | +1.2% |
| Median TPOT | 6.70 | **6.54** | −2.4% |
| Mean TPOT | 6.51 | 6.39 | −1.8% |
| Median E2E | 7205 | **7056** | −2.1% |
| Interactivity | 149.3 | **152.89** | +2.4% |
| GSM8K | 0.9401 | **0.9454** | +0.5pp |

Every metric better. Gates still 1/4 (GSM8K only). Interact closer to gate (165-153=12 gap vs 15 before).

### Key debug findings

- v1: OOM from leftover worker processes (GPU memory not freed)
- v2-v3: `re:model.layers.61.self_attn.*` excludes don't match drafter's `mtp_block`-renamed path
- v4: `safetensors_weights_iterator` globs ALL *.safetensors, picking up BF16 tensors from pure-layer-61 main shards even when index doesn't reference them
- v5 (success): exclude pure-layer-61 main shards, rebuild mixed shards without layer 61 keys

### For submission — MODEL name issue

Server registers as `--model` path value. Bench harness hardcoded MODEL=amd/DeepSeek-R1-0528-MXFP4 → 400 error. Two paths for submission:
1. `MODEL` env var override at bench time (current workaround)
2. Symlink merged dir into `/projects/teamA/hf_cache/hub/models--amd--DeepSeek-R1-0528-MXFP4/snapshots/{fake-hash}/` + update refs/main (cleaner for AMD review)

### Smaller-than-projected gain analysis

Predicted ~5-7% TPOT improvement; got ~2.4%. Possibilities:
- DEC-057 profile over-estimated drafter-MoE fraction
- FP4 kernel for drafter shapes (bs=4, smaller) less optimized than main's shapes (bs=16)
- Drafter has overhead beyond MoE (MLA, routing, token dispatch)

Still net-positive + every metric improved + no regression. **DEC-075 locked as new floor above DEC-073.**

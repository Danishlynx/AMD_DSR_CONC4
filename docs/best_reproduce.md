# DSR1 CONC=4 — Current Best: DEC-075 (self-contained reproduction)

**Last updated**: 2026-04-17 evening (DEC-075 drafter FP4 transplant + full profile investigation)

> **🚨 FINAL PUSH MODE**: Block 3 / Kimi / May 15 horizon DROPPED. Mission: 4/4 CONC=4 gates by Apr 18 night or submit sub-rank. Next attempt: tree speculation (drafter cheap enough at DEC-075 to justify it).

## Current best floor: DEC-075 (LANDED, reproducible)

| Metric | LANDED (162646) | REPRO (174928) | Gate | Status |
|---|---|---|---|---|
| **Thr/GPU (÷4)** | **1297** | **1278** | ≥1500 | ❌ −14% |
| **Median TPOT** | **6.54 ms** | **6.74 ms** | — | — |
| Mean TPOT | 6.39 | 6.43 | — | — |
| **Median ITL** | 16.5 | 16.46 | — | — (MTP=3 burst ✓) |
| **Interactivity** | **153** | **148** | ≥165 | ❌ −10% |
| **Median E2E** | **7056** | **7253** | ≤5000 | ❌ +45% |
| **GSM8K** | **0.9454** | ≥0.93 | ✅ +1.5 pp |
| **Gates** | **1/4** | 1/4 | 4/4 | GSM8K only |

**Binding gate math**: E2E ≤ 5000 → TPOT ≤ 4.52 ms. Need −33% from 6.74 ms.

**Run-to-run variance**: ±3% on TPOT/thr is normal. Both runs are within noise — DEC-075 is stable.

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

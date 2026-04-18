# DSR_beta Reproduction Recipe

**Captured:** 2026-04-18 ~05:40 UTC
**Result:** 1335 thr/GPU, 6.40 ms TPOT median, 156 interact, 7009 ms E2E median, GSM8K 0.9386 (+4.4% thr / −5.0% TPOT / +5.4% interact over DEC-075 production floor at CONC=4).

## Full state snapshot

### Docker image
- **Tag**: `rocm/atom-dev:latest`
- **Layer digest**: `sha256:7f54c1b431040e3dda265adc76190d3dcf0584d483610e38aba0eb5ecce6d490`
- **Repo digest** (pinned pull ID): `rocm/atom-dev@sha256:52c5195a712b5d3a0993d5e63de9b8ffc13a77d0c4b2f31d40afe9e62c12ab5f`
- **Created**: 2026-04-17T15:23:22Z
- **Size**: ~57.3 GB

To re-pull exact image:
```bash
docker pull rocm/atom-dev@sha256:52c5195a712b5d3a0993d5e63de9b8ffc13a77d0c4b2f31d40afe9e62c12ab5f
```

### Container
- Name: `danish_atom_dsr_beta`
- Container ID (on current server): `7135631e3df30ba42a4c536988c6a59327d3d099e64154ed2abad66ff6dd5ef6`
- Port: **8890** (vs production 8888)
- Based on: `rocm/atom-dev:latest`

### Software versions
| Package | Version | Source |
|---|---|---|
| ROCm | 7.2.2 | /opt/rocm/.info/version |
| Python | 3.12.3 | system |
| PyTorch | 2.10.0+rocm7.2.2.git40d237bf | pip |
| HIP runtime | 7.2.53211 | torch.version.hip |
| aiter (amd-aiter) | 0.0.0 | commit `73ad0023e15e9735b3af95b3357b99cf7f801bf1` on main |
| ATOM | 0.1.3.dev90+gf8453e3fc | commit `f8453e3fc0f65191fb2034602dc9a2066a78020b` on main |
| flydsl | 0.1.3.1 | PyPI |
| triton | 3.5.1 | PyPI |

### Local patches (from stock upstream main)

See [patches/dsr_beta_local_mods.diff](patches/dsr_beta_local_mods.diff) for the diff. Two changes:

**1. `atom/model_ops/rejection_sampler.py`** (relaxed MTP tuning to DEC-073 values):
```diff
-    RELAXED_TOP_N = 10
-    RELAXED_DELTA = 0.6
+    RELAXED_TOP_N = 8
+    RELAXED_DELTA = 0.5
```

**2. `atom/model_ops/attention_mla.py`** (Session 6A intervention):
```diff
-            num_kv_splits=16,
+            num_kv_splits=None,
```

### Checkpoint
- **Path** (server): `/projects/teamA/danish/models_merged/DSR1-drafter-FP4`
- This is the **DEC-075 merged checkpoint** — layer 61 MoE weights swapped from `amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4` (FP4) into main `amd/DeepSeek-R1-0528-MXFP4` (was BF16). Drafter dispatches to FP4 FlyDSL fast path.
- Build script: `scripts/merge_dec075_v5.py`

## Full reproduction steps

### 1. Container creation
```bash
~/bin/docker run -d --name danish_atom_dsr_beta \
  --device /dev/kfd \
  --device /dev/dri/renderD128 --device /dev/dri/renderD136 \
  --device /dev/dri/renderD144 --device /dev/dri/renderD152 \
  --device /dev/dri/renderD160 --device /dev/dri/renderD168 \
  --device /dev/dri/renderD176 --device /dev/dri/renderD184 \
  --group-add video \
  --shm-size=32g \
  -p 8890:8890 \
  -v /projects:/projects \
  -v /home/hackathon:/home/hackathon \
  rocm/atom-dev@sha256:52c5195a712b5d3a0993d5e63de9b8ffc13a77d0c4b2f31d40afe9e62c12ab5f \
  sleep infinity
```

### 2. Apply local patches

```bash
~/bin/docker exec danish_atom_dsr_beta bash -c "
cd /app/ATOM
# Patch 1: relaxed MTP (8, 0.5)
sed -i 's/RELAXED_TOP_N = 10/RELAXED_TOP_N = 8/' atom/model_ops/rejection_sampler.py
sed -i 's/RELAXED_DELTA = 0.6/RELAXED_DELTA = 0.5/' atom/model_ops/rejection_sampler.py
# Patch 2: num_kv_splits=None
sed -i 's/num_kv_splits=16,/num_kv_splits=None,/' atom/model_ops/attention_mla.py

# Verify
grep -nE 'RELAXED_TOP_N|RELAXED_DELTA' atom/model_ops/rejection_sampler.py | head -2
grep -n 'num_kv_splits' atom/model_ops/attention_mla.py | head
"
```

### 3. Ensure checkpoint exists

The merged checkpoint at `/projects/teamA/danish/models_merged/DSR1-drafter-FP4` must exist. If missing:
```bash
~/bin/docker exec danish_atom_dsr_beta bash -c "
cd /projects/teamA/danish/repos/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x
python3 /path/to/merge_dec075_v5.py
"
```

### 4. Launch server (CRITICAL — exact recipe)

```bash
~/bin/docker exec -d danish_atom_dsr_beta bash -c "
export HOME=/tmp
export HF_HOME=/projects/teamA/hf_cache
export HUGGINGFACE_HUB_CACHE=/projects/teamA/hf_cache/hub
unset HF_HUB_OFFLINE
export AITER_BUILD_DIR=/tmp/.aiter_cache TRITON_CACHE_DIR=/tmp/.triton_cache
export HIP_FORCE_DEV_KERNARG=1
export NCCL_MIN_NCHANNELS=16
export ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=256
export ATOM_ENABLE_RELAXED_MTP=1
export HIP_VISIBLE_DEVICES=0,1,2,3
cd /app/ATOM
python3 -m atom.entrypoints.openai_server \
  --model /projects/teamA/danish/models_merged/DSR1-drafter-FP4 \
  --server-port 8890 \
  -tp 4 \
  --kv_cache_dtype fp8 \
  --method mtp \
  --num-speculative-tokens 3 \
  --max-model-len 10240 \
  --gpu-memory-utilization 0.85 \
  --enable-tbo prefill > /tmp/atom-dsr_beta.stdout 2>&1
"
```

Wait for `Uvicorn running on http://0.0.0.0:8890` (~8-10 min cold, ~5 min warm).

**Verify correct boot** via log check:
- `grep "Capturing bs=4.*max_q_len=4"` → max_q_len=4 confirms mtp_k=3 ✓
- `grep "flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w3_fq"` at bs=4 → drafter FP4 fast path ✓
- `grep "enable_tbo: True, enable_tbo_decode: False"` → TBO prefill-only ✓

### 5. Bench

```bash
~/bin/docker exec danish_atom_dsr_beta bash -c "
export HOME=/tmp HF_HOME=/projects/teamA/hf_cache HUGGINGFACE_HUB_CACHE=/projects/teamA/hf_cache/hub
unset HF_HUB_OFFLINE
cd /projects/teamA/danish/repos/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x
source specific_conc_var.sh
export MODEL=/projects/teamA/danish/models_merged/DSR1-drafter-FP4
export PORT=8890
./dsr1_benchmark perf 2>&1 | tail -50
"
```

### 6. Expected output

```
Request throughput (req/s):              0.58
Output token throughput (tok/s):         591
Total Token throughput (tok/s):          5340
Mean TTFT (ms):                          ~437
Median TTFT (ms):                        ~370
Mean TPOT (ms):                          6.18
Median TPOT (ms):                        6.40
P99 TPOT (ms):                           8.92
Mean ITL (ms):                           18.80
Median ITL (ms):                         16.18
P99 ITL (ms):                            57.39
Mean E2EL (ms):                          6746
Median E2EL (ms):                        7009
P99 E2EL (ms):                           9454
INFO: Throughput: 667.43 tokens/s/GPU (but /8 divisor — correct /4 = 1335)
INFO: Interactivity: 156.24 tokens/s/user
GSM8K: 0.9386 ✓
```

## Full bench results (all 4 test runs on DSR_beta)

| Test | Config | Thr/GPU (÷4) | Median TPOT | Median ITL | Interact | Median E2E | GSM8K | Verdict |
|---|---|---|---|---|---|---|---|---|
| **1. Baseline** | Latest stack, no TBO | 1315 | 6.56 | 16.29 | 152.46 | 7150 | 0.93+ | +2.9% vs DEC-075 |
| **2. + TBO prefill** | + `--enable-tbo prefill` | **1335** | **6.40** | **16.18** | **156.24** | **7009** | **0.9386** | **+4.4% vs DEC-075 (BEST)** |
| **3. + TBO all** | `--enable-tbo all` | 939 | 9.53 | 24.86 | 104.96 | 10099 | — | **−30% regression** |
| **4. + MORI on (2)** | TBO prefill + `--all2all-backend low-latency` | 1322 | 6.64 | 16.32 | 150.52 | 7201 | — | −1% (MORI pure overhead at TP-sharded MoE) |

## Gate analysis

| Gate | Target | DEC-075 prod | DSR_beta best | Gap |
|---|---|---|---|---|
| Thr/GPU | ≥ 1500 | 1278 | 1335 | −11% |
| Interactivity | ≥ 165 | 148 | 156 | **−5.5%** ← closest |
| E2E median | ≤ 5000 ms | 7253 | 7009 | +40% |
| GSM8K | ≥ 0.93 | ≥0.93 | **0.9386** | ✓ PASS |

**Still 1/4 gates.** Interactivity gap narrowed from 10.3% → 5.5%.

**Binding gate math**: E2E ≤ 5000 → TPOT ≤ ~4.49 ms. Need −30% TPOT from 6.40. No kernel tuning alone closes this; only tree speculation.

## Known issues flagged for retune/retry

See memory `project_dsr_beta_csv_retune_needed.md` — BF16 GEMM CSV must be re-tuned on ROCm 7.2.2 because hipBLASLt `solidx` values from ROCm 7.1.1 throw `HIPBLAS_STATUS_INTERNAL_ERROR` on new runtime. Expected +1-2% after retune.

## Files in this snapshot

- `REPRODUCTION.md` — this document
- `bench_results/` — all 4 DSR_beta test JSONs
- `patches/dsr_beta_local_mods.diff` — the 2 local patches as unified diff
- `scripts/dsr_beta_launch.sh` — reproducible launch script
- `scripts/dsr_beta_setup.sh` — full container + patch setup script

## Rollback / safety

Production container `danish_atom_main` (port 8888) is **UNTOUCHED** throughout. DSR_beta is fully isolated. Rollback is `~/bin/docker stop danish_atom_dsr_beta` — production unaffected.

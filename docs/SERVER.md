# DSR1 CONC=4 — SERVER COMMANDS & INFRASTRUCTURE (merged: INFRA + SERVER_MAP)

**Last updated**: 2026-04-22 session-14

---

## 🚨 CRITICAL: COMPETITION WRAPPER = KIMBOCHEN BENCH + CHAT TEMPLATE = REASONING MODE

The leaderboard runs `dsr1_benchmark perf` → clones `github.com/kimbochen/bench_serving` → runs it **with `--use-chat-template`** → activates DSR1 reasoning mode. Direct ATOM bench numbers without chat template are NOT what the leaderboard measures.

### Wrapper invocation (what the leaderboard actually does)

```bash
docker exec reproducer_best bash -c '
  export HOME=/tmp HF_HOME=/tmp/.cache/huggingface HUGGINGFACE_HUB_CACHE=/tmp/.cache/huggingface/hub
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export MODEL=amd/DeepSeek-R1-0528-MXFP4 PORT=8890 TP=4 CONC=4 ISL=8192 OSL=1024 NUM_PROMPTS=40
  export RESULT_FILENAME=/tmp/WRAPPER_run1
  /projects/teamA/danish/repos/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x/dsr1_benchmark perf
'
```

Modes: `perf` (GSM8K→validate→bench), `acc` (GSM8K only), `submit` (GSM8K→validate→bench→leaderboard POST). Every mode runs GSM8K first — no bypass.

**Parse wrapper result**:
```bash
jq '{thr_per_gpu: (.total_token_throughput/4), tpot_mean: .mean_tpot_ms, tpot_median: .median_tpot_ms, e2e_median: .median_e2el_ms}' /tmp/WRAPPER_run1.json
```

### Kimbochen bench direct invocation (no server restart needed)

For fast iteration without the 90-second GSM8K prefix, run kimbochen's bench directly with the same flags as the wrapper:

```bash
docker exec reproducer_best bash -c '
  # Clone once if not present
  [ ! -d /tmp/kimbochen_bench ] && git clone https://github.com/kimbochen/bench_serving.git /tmp/kimbochen_bench
  export HOME=/tmp HF_HOME=/tmp/.cache/huggingface HUGGINGFACE_HUB_CACHE=/tmp/.cache/huggingface/hub
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  python3 /tmp/kimbochen_bench/benchmark_serving.py \
    --backend vllm --base-url http://0.0.0.0:8890 \
    --model amd/DeepSeek-R1-0528-MXFP4 \
    --dataset-name random --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1 \
    --num-prompts 40 --max-concurrency 4 --request-rate inf --ignore-eos \
    --save-result --num-warmups 8 --percentile-metrics "ttft,tpot,itl,e2el" \
    --use-chat-template \
    --result-filename /tmp/KIMB_CHAT_run1.json
'
```

Output will numerically match the wrapper's bench phase (already proven: 1308 thr/GPU vs wrapper 1291 thr/GPU within noise).

### Direct ATOM bench (for non-reasoning debugging only — NOT submittable)

```bash
docker exec reproducer_best bash -c '
  export HOME=/tmp HF_HOME=/tmp/.cache/huggingface HUGGINGFACE_HUB_CACHE=/tmp/.cache/huggingface/hub
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  cd /app/ATOM
  python3 -m atom.benchmarks.benchmark_serving \
    --model amd/DeepSeek-R1-0528-MXFP4 --port 8890 \
    --dataset-name random --random-input-len 8192 --random-output-len 1024 \
    --num-prompts 40 --max-concurrency 4 --trust-remote-code \
    --save-result --save-detailed --result-filename /tmp/DIRECT_run1.json
'
```

### Profiled bench (torch.profiler trace)

Server must be launched with `--torch-profiler-dir /tmp/torch_traces_contaminated` CLI flag (env alone is insufficient — CLI default `None` overrides env):

```bash
# In launch script:
python3 -m atom.entrypoints.openai_server ... \
  --torch-profiler-dir /tmp/torch_traces_contaminated \
  --cudagraph-capture-sizes "[1,2,4,8,16,32]"

# Then run bench with --profile to trigger start/stop:
docker exec reproducer_best bash -c '
  cd /app/ATOM
  python3 -m atom.benchmarks.benchmark_serving \
    --model amd/DeepSeek-R1-0528-MXFP4 --port 8890 \
    --dataset-name random --random-input-len 8192 --random-output-len 1024 \
    --num-prompts 12 --max-concurrency 4 --trust-remote-code \
    --use-chat-template --ignore-eos --profile \
    --save-result --result-filename /tmp/PROFILED.json
'
```

Traces land in `/tmp/torch_traces_contaminated/rank_{0..3}/*.pt.trace.json.gz` (~1.4GB per rank for 12 prompts).

**Parse trace** via `/tmp/diff_traces.py` (in `reproducer_best`): see MASTER.md § TOP for expected kernel breakdown.

---

## Quick SSH setup (every session)

```bash
eval "$(ssh-agent -s)"
SSH_PASS=7756932064 SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0 \
  ssh-add ~/.ssh/id_ed25519_new < /dev/null
ssh amd-gpu 'hostname; rocm-smi | head'
```

Passphrase: `7756932064` (via `/tmp/askpass.sh`)

---

## Containers

| Container | Image | Purpose | Port | State |
|---|---|---|---|---|
| `reproducer_best` | `rocm/atom-dev:dsr1_P0_3of4_gates_apr20` | Q3 lab (session-14) | 8890 | Q3.3 applied, perf-tested |
| `danish_atom_dsr_beta` | `rocm/atom-dev:latest` | Session-13 original | 8890 | Idle, PB reverts clean |
| `danish_kimi` | — | Kimi team | 8893+ | Separate team, do not touch |

### Gold image (DO NOT MODIFY)
- Image: `rocm/atom-dev:dsr1_P0_3of4_gates_apr20` (473GB, ID `02e27b1ebcac`)
- Branch: `dsr_best_P0_3of4_apr20` pushed to https://github.com/Danishlynx/AMD_DSR_CNCC4

---

## Key file paths (inside reproducer_best container)

| Purpose | Path |
|---|---|
| ATOM source | `/app/ATOM/` |
| AITER source | `/app/aiter-test/` |
| Model weights | `/tmp/.cache/huggingface/hub/models--amd--DeepSeek-R1-0528-MXFP4` |
| P0 launch script | `/tmp/p0_launch.sh` |
| P0 gold benches | `/tmp/P0_run{1,2,3}.json`, `/tmp/P0_reverify_run{1,2,3}.json` |
| Competition wrapper | `/projects/teamA/danish/repos/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x/dsr1_benchmark` |
| Wrapper source | `/dsr1_benchmark.cpp` |

### Q3.3 patch sites
- `/app/ATOM/atom/model_ops/moe.py:432-438` (FusedMoEModularKernel shared_experts arg)
- `/app/ATOM/atom/models/deepseek_v2.py:~920` (dual_stream_moe_forward gate)
- Backups: `<file>.preQ3.3`

### Q3.2 / Q3.4 / Q3.5 patch sites (pending)
- `/app/ATOM/atom/model_ops/linear.py:40-57` (TRITON_GEMM gate)
- `/app/aiter-test/csrc/kernels/mla/hk/mi3xx_v32_fwd_decode_h32_fp8_fp8.cuh` (Q4 HK kernel base)
- `/app/aiter-test/aiter/jit/optCompilerConfig.json` (kernel registration)

### Cached PR diffs (host, for reference)
- `C:\Users\danis\tmp_research\pr27224.diff` (host overhead)
- `C:\Users\danis\tmp_research\pr2727.diff` (BF16 fold trick)
- `C:\Users\danis\tmp_research\pr39616.diff` (non-persistent fallback)
- `C:\Users\danis\tmp_research\pr27380.diff` (additional fusion)

---

## Boot checklist (cold)

1. SSH + key loaded (see above)
2. Confirm container running: `ssh amd-gpu 'docker ps | grep reproducer_best'`
3. Verify GPUs clean: `ssh amd-gpu 'rocm-smi --showmeminfo vram | head -20'`
4. Kill any stale Python: `ssh amd-gpu 'docker exec reproducer_best bash -c "pkill -9 python3 2>&1; sleep 5"'`
5. Launch P0: `ssh amd-gpu 'docker exec -d reproducer_best bash /tmp/p0_launch.sh'`
6. Wait ~15 min cold boot (cudagraph captures take most of this)
7. Verify server: `ssh amd-gpu 'docker exec reproducer_best curl -s http://localhost:8890/v1/models'`
8. Smoke test: `curl -s http://localhost:8890/v1/completions -d '{"model":"amd/DeepSeek-R1-0528-MXFP4","prompt":"2+2=","max_tokens":5}'`

---

## Cache hygiene (before each bench)

```bash
# Wipe stale torchinductor caches (per feedback_torchinductor_cache_poisons_perf.md)
docker exec reproducer_best bash -c '
  rm -rf /tmp/.cache/atom/torchinductor_cache/* 2>/dev/null || true
  rm -rf /root/.cache/torch/inductor/* 2>/dev/null || true
'
# GPUs should be at default perf (check before bench)
rocm-smi --showperflevel | head -8
```

---

## Original INFRA.md content

# Infrastructure Reference — mia1-p02-g55 (stable reference doc)

**Last updated**: 2026-04-19. Purpose: persistent server/hardware/container map so any session (Opus, Kimi, future) can understand the setup instantly.

---

## 🏠 Server access

```
Physical host:  mia1-p02-g55  (AMD hackathon server)
SSH config:     C:\Users\danis\.ssh\config
Aliases:
  amd-bastion   → 64.139.223.122 (jump host)
  amd-gpu       → mia1-p02-g55   (GPU server, via ProxyJump)

SSH wrapper:    /c/tmp/ssh_helper/ssh_wrap.sh
  (auto-loads agent key with cached passphrase via askpass helper)
```

**From Windows bash (git bash / mingw64)**:
```bash
bash /c/tmp/ssh_helper/ssh_wrap.sh amd-gpu '<command>'
bash /c/tmp/ssh_helper/ssh_wrap.sh amd-gpu '~/bin/docker exec danish_atom_dsr_beta bash -c "cmd"'
```

Key test commands:
```bash
ssh amd-gpu 'hostname'
ssh amd-gpu '~/bin/docker ps'
ssh amd-gpu 'rocm-smi --showmeminfo vram | head -20'
```

---

## 🖥️ Hardware (the roofline)

```
8× AMD Instinct MI355X (CDNA4 gfx950)
  GPUs 0-3 → DSR1 track
  GPUs 4-7 → Kimi track

Per GPU:
  ~288 GB HBM3e memory
  8 TB/s HBM bandwidth peak (~6.5 TB/s realistic)
  256 active compute units (CUs)
  256 VGPRs per wave, 16K VGPRs per CU
  160 KB LDS per CU (on-chip scratchpad)
  10 PFLOPS MXFP4 compute, 5 PFLOPS FP8
  4 SIMD × 64 waves per CU
  FP8 E4M3 MFMA: 16×16×32

Interconnect:
  Infinity Fabric pairwise, 153 GB/s bidir per link

Storage:
  / (28 TB, 20 TB free)
  /projects/teamA/ shared team volume (on /dev/md0)
```

### Roofline math (for DSR1 CONC=4 reference)
- DSR1-0528 MXFP4: 671B total params, 37B active/token (sparse MoE), 61 layers, 256 experts top-8
- Active bytes per token at TP=4: ~10-12 GB per GPU
- HBM-read time: 12/6500 = **~1.5 ms pure read** floor
- With MTP=3 at 1.89 avg accept, effective floor: ~0.8 ms/output token
- Current TPOT 6.35 ms = **7.5× above physical floor** — the gap is overhead, not compute

---

## 🗂️ Host filesystem

```
/projects/teamA/                   ← shared team volume
│
├── danish/                        ← YOUR main work area
│   ├── repos/                     ← DSR1 TRACK code
│   │   ├── ATOM_main/             ← ATOM framework for DSR1 (older, mounted into danish_atom_main)
│   │   ├── aiter/                 ← AITER kernel lib (older)
│   │   ├── sglang/                ← SGLang reference
│   │   ├── vllm/                  ← vLLM reference
│   │   ├── amdgpu_bounty_optimization/   ← the official bench harness
│   │   │   └── dsr1-fp4-atom-mtp-mi355x/ ← launch_atom_server.sh + dsr1_benchmark
│   │   └── ATOM/                  ← older ATOM (abandoned)
│   │
│   ├── kimi/                      ← KIMI TRACK code (isolated)
│   │   ├── ATOM_kimi/
│   │   ├── aiter_kimi/
│   │   ├── vllm_kimi/
│   │   └── amdgpu_bounty_optimization/
│   │
│   ├── dsr_beta/                  ← DSR_beta stack working dir
│   │   ├── caches/
│   │   ├── logs/
│   │   └── repos/
│   │
│   ├── backups/                   ← DEC-073 snapshots
│   ├── logs/                      ← session logs
│   ├── results/                   ← bench results
│   ├── models_merged/             ← merged checkpoints
│   │   └── DSR1-drafter-FP4/      ← DEC-075 transplanted checkpoint
│   ├── c1_hk_port/                ← C1 HipKittens port working dir (session-8)
│   └── _mori_*/                   ← mori library state (all-to-all comm)
│
└── hf_cache/                      ← SHARED model weights (1.6 TB, READ-ONLY for safety)
    └── hub/
        ├── models--amd--DeepSeek-R1-0528-MXFP4               376 GB  ← DSR1 main model
        ├── models--amd--DeepSeek-R1-0528-MXFP4-MTP-MoEFP4    350 GB  ← variant (Triton trap, don't use)
        ├── models--amd--Kimi-K2.5-MXFP4                      521 GB  ← Kimi K2.5
        ├── models--lightseekorg--kimi-k2.5-eagle3              6 GB  ← Kimi Eagle3 drafter
        ├── datasets--gsm8k
        └── datasets--openai--gsm8k

/home/danish@neuralmerge.net/    ← your home, ~28 KB (essentially empty — correct)
└── bin/docker                    ← wrapper script for docker access

/share4/                         ← OTHER TEAM's storage (99% full, ignore)
```

---

## 🐳 Docker containers

| Name | Track | GPUs | Port | Size | Status |
|---|---|---|---|---|---|
| **danish_atom_dsr_beta** | DSR1 (DSR_beta stack, ROCm 7.2.2) | 0-3 | 8890 | — | **Active production DSR1** (up 32+ hrs) |
| danish_atom_main | DSR1 (older stack, rocm 7.1.1) | 0-3 | internal 8888 | 7.76 GB | Legacy, not used currently |
| danish_kimi | Kimi track | 4-7 | 8889→8888 | 1.31 GB | Active (separate Opus session) |
| danish_atom | Experimental | — | 8888 | 205 MB | Idle, don't touch |

### danish_atom_dsr_beta (current DSR1 production)

- Image: `rocm/atom-dev@sha256:52c5195a712b5d3a0993d5e63de9b8ffc13a77d0c4b2f31d40afe9e62c12ab5f`
- Stack: ROCm 7.2.2, PyTorch 2.10.0, aiter `73ad0023`, ATOM `f8453e3f`, flydsl 0.1.3.1, triton 3.5.1
- Mounts: `/projects/teamA/` host → `/projects/teamA/` container (includes HF cache + all repos)
- Server: `python3 -m atom.entrypoints.openai_server --model amd/DeepSeek-R1-0528-MXFP4 -tp 4 --kv_cache_dtype fp8 --method mtp --num-speculative-tokens 3 --enable-tbo prefill`
- JIT cache: `/root/.aiter` is read-only in overlay FS → use `HOME=/tmp` env override for all invocations
- Session-8 C1 patches deployed here (see STATUS.md for list, all with `.pre_c1` backups)

---

## 📁 Key file locations inside danish_atom_dsr_beta

### DSR1 inference stack
```
/app/ATOM/atom/                    ← ATOM (the framework)
├── spec_decode/eagle.py           ← EagleProposer, MTP drafter logic
├── model_ops/
│   ├── attention_mla.py           ← MLA attention wrapper (has `.pre_c1` + other backups)
│   ├── rejection_sampler.py:10-14 ← RELAXED_TOP_N=8, RELAXED_DELTA=0.5 hardcoded
│   └── attentions/aiter_mla.py:348 ← prepare_mtp_decode
├── model_engine/
│   ├── model_runner.py            ← forward dispatch + HIP graph capture
│   └── block_manager.py           ← KV cache block allocation
├── models/deepseek_mtp.py         ← DeepseekV2 MTP model definition
└── config.py:882                  ← MTP cap ValueError (was >4, session-8 lifted to >8)

/app/aiter-test/                   ← AITER kernel library
├── aiter/
│   ├── mla.py:330-362             ← MLA native-supported dispatch gate
│   ├── mla.py:429-437             ← use_hk gate (session-8 extended for qh32)
│   ├── ops/attention.py:1294      ← hk_mla_decode_fwd Python binding
│   └── jit/optCompilerConfig.json ← JIT module build config (has module_hk_mla)
├── csrc/
│   ├── kernels/mla/
│   │   ├── hk_decode_fwd.cu       ← HK dispatch (session-8 added num_head==32 branch)
│   │   └── hk/                    ← HipKittens MLA (2646 LOC total)
│   │       ├── hk_mla_buffer_managers.cuh (1546 LOC)
│   │       ├── hk_mla_softmax.cuh (272 LOC)
│   │       ├── hk_mla_utils.cuh (16 LOC)
│   │       ├── mi3xx_v32_fwd_decode_h128_fp8_fp8.cuh (812 LOC) — ORIGINAL, untouched
│   │       └── mi3xx_v32_fwd_decode_h32_fp8_fp8.cuh (NEW session-8)
│   └── cpp_itfs/mla/              ← assembly kernel wrappers (.cpp.jinja)
├── hsa/gfx950/mla/                ← pre-compiled GPU assembly (.co) kernel blobs
│   ├── mla_a8w8_qh32_qseqlen{2,4}_gqaratio32_ps.co
│   └── mla_asm.csv                ← kernel registry (qType, kvType, Gqa, ps, qSeqLen...)
└── hsa/codegen.py                 ← CSV→C++-header compiler (NOT a kernel generator)

/app/aiter-test/aiter/configs/model_configs/
├── dsv3_bf16_tuned_gemm.csv       ← BF16 GEMM tunings (session-7 destroyed by JIT merge)
├── dsv3_a4w4_blockscale_tuned_gemm.csv
└── dsv3_fp4_tuned_fmoe.csv        ← FP4 MoE tunings
```

### Reproduction harness
```
/projects/teamA/danish/repos/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x/
├── launch_atom_server.sh          ← base launch script
└── dsr1_benchmark                 ← scoring binary (perf + acc modes)
```

### Backups on disk

Session-7 (Lever B/C attempts):
- `eagle.py.pre_lever_b`, `eagle.py.pre_lever_b2`, `eagle.py.pre_lever_b_v6`
- `attention_mla.py.pre_lever_c`, `.pre_lever_c_v3`, `.pre_lever_c_v4`, `.pre_kvbmm_patch`

Session-8 (C1 HK port):
- `hk_decode_fwd.cu.pre_c1`
- `optCompilerConfig.json.pre_c1`
- `mla.py.pre_c1`
- `atom/config.py.pre_c1`

Working dirs (local port copies):
- `/projects/teamA/danish/c1_hk_port/` — all HK source files copied for editing
- `/c/tmp/ssh_helper/` (local) — patch source scripts: `lever_b*.py`, `lever_c*.py`, `lever_b2_peagle_pos_only.py`

---

## 🔐 Track separation — DSR1 vs Kimi

| Aspect | DSR1 | Kimi |
|---|---|---|
| Container | `danish_atom_dsr_beta` | `danish_kimi` |
| GPUs | 0, 1, 2, 3 | 4, 5, 6, 7 |
| ATOM code | `/projects/teamA/danish/repos/ATOM_main`, `/app/ATOM/atom/` (in container) | `/projects/teamA/danish/kimi/ATOM_kimi` |
| aiter code | `/app/aiter-test/aiter/` (in container) | `/projects/teamA/danish/kimi/aiter_kimi` |
| Model | `amd/DeepSeek-R1-0528-MXFP4` | `amd/Kimi-K2.5-MXFP4` |
| Bench harness | `dsr1-fp4-atom-mtp-mi355x` | `kimi-*` |
| Host port | 8890 | 8889 |

### Hard rules to avoid track collision

1. **Do NOT touch `kimi/` subtree** — that's the Kimi track's code
2. **Use GPUs 0-3 only** for DSR1 — never set `HIP_VISIBLE_DEVICES=4,5,6,7` in DSR1 work
3. **Do NOT restart `danish_kimi`** — it's the Kimi Opus session's workspace
4. **Shared HF cache at `/projects/teamA/hf_cache/`** is read-only for our purposes
5. **Never run a DSR1 server on GPUs 4-7** — would conflict with Kimi's active processes

---

## 🛠️ Common commands

### SSH + docker
```bash
# Run command inside DSR1 container
bash /c/tmp/ssh_helper/ssh_wrap.sh amd-gpu "~/bin/docker exec danish_atom_dsr_beta bash -c 'cmd'"

# GPU state
bash /c/tmp/ssh_helper/ssh_wrap.sh amd-gpu "~/bin/docker exec danish_atom_dsr_beta bash -c 'rocm-smi --showmeminfo vram | head -20'"

# Current server processes
bash /c/tmp/ssh_helper/ssh_wrap.sh amd-gpu "~/bin/docker exec danish_atom_dsr_beta bash -c 'pgrep -af openai_server'"

# Container restart (clears zombies + VRAM leak)
bash /c/tmp/ssh_helper/ssh_wrap.sh amd-gpu "~/bin/docker restart danish_atom_dsr_beta"
```

### Launch DSR1 floor server

See STATUS.md for complete launch command. Key env vars: `HOME=/tmp HIP_FORCE_DEV_KERNARG=1 NCCL_MIN_NCHANNELS=16 ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=1024 ATOM_ENABLE_RELAXED_MTP=1`.

### Bench

```bash
# From inside container:
cd /projects/teamA/danish/repos/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x
./dsr1_benchmark perf          # full perf + GSM8K
./dsr1_benchmark acc           # GSM8K only (use min-of-3 for stability gate)
```

### Boot verification markers

Watch for in server log:
- ✅ `[aiter] rank N in world size 4 is assigned as DP rank 0, PP rank 0, TP rank N` × 4 ranks
- ✅ `Capturing bs=4, max_q_len=4` → MTP=3 captured correctly
- ✅ `flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w3_fq` at bs=4 → drafter FP4 fast path
- ✅ `Uvicorn running on http://0.0.0.0:8890`
- ❌ Only `max_q_len=2` captures → MTP silently collapsed to MTP-1 (BAD)
- ❌ `[aiter] No available shared memory broadcast block found` → ranks hung, likely silent crash

---

## 📊 Current state snapshot (2026-04-19 session-8 close)

- DSR1 floor: **1361/6.35/157/6842/0.934** → 1/4 gates at CONC=4
- Session-8 C1 HipKittens qh32 port in flight: JIT compile ✅, first boot HUNG, control boot in progress
- See STATUS.md for full active plan + resume checklist

## Reference links

- **STATUS.md** — current state, active plan, reproduction recipe
- **FINDINGS.md** — canonical decisions (DEC-*) + dead/alive lever inventory
- **HISTORY.md** — chronological session narratives
- **BRIEF_FOR_KIMI_OPUS.md** — cross-agent handoff for Kimi track

---

## Original SERVER_MAP.md content

# Server Infrastructure Map — mia1-p02-g55

**Generated**: 2026-04-18 (DEC-073 floor locked)
**Purpose**: Persistent reference so anyone (you, future Opus, Kimi Opus) can understand the setup instantly.

---

## 🏠 Server & Access

```
Physical host:  mia1-p02-g55  (AMD's hackathon server)
SSH config:     C:\Users\danis\.ssh\config
Aliases:
  amd-bastion   → 64.139.223.122 (jump host)
  amd-gpu       → mia1-p02-g55   (GPU server, via ProxyJump)

Hardware:  8× AMD Instinct MI355X (CDNA4)
           GPUs 0-3 → DSR1 track
           GPUs 4-7 → Kimi track
           
Each GPU: ~288 GB HBM3e
```

**From Windows bash (git bash / mingw64), to run commands on the GPU server**:
```bash
ssh amd-gpu '<command>'
# or with the passphrase-free session key
ssh -F /tmp/ssh_config amd-gpu-s '<command>'
```

---

## 🗂️ Host Filesystem

```
/   (28 TB, 20 TB free — HEALTHY)
└── /projects/teamA/           ← shared team volume (on /dev/md0)
    │
    ├── danish/                ← YOUR main work area
    │   ├── repos/             ← DSR1 TRACK code
    │   │   ├── ATOM_main/     ← ATOM framework for DSR1 (108a70e + our patches)
    │   │   ├── aiter/         ← AITER kernel lib for DSR1 (f8c1d76bd)
    │   │   ├── sglang/        ← SGLang reference
    │   │   ├── vllm/          ← vLLM reference
    │   │   ├── amdgpu_bounty_optimization/  ← the official bench harness
    │   │   └── ATOM/          ← older ATOM (abandoned)
    │   │
    │   ├── kimi/              ← KIMI TRACK code (isolated)
    │   │   ├── ATOM_kimi/     ← ATOM for Kimi
    │   │   ├── aiter_kimi/    ← separate aiter for Kimi
    │   │   ├── vllm_kimi/     ← vLLM for Kimi
    │   │   └── amdgpu_bounty_optimization/
    │   │
    │   ├── backups/           ← DEC-073 snapshots go here
    │   ├── logs/              ← session logs
    │   ├── results/           ← bench results
    │   └── _mori_*/           ← mori library state (all-to-all comm)
    │
    └── hf_cache/              ← SHARED model weights (1.6 TB)
        └── hub/
            ├── models--amd--DeepSeek-R1-0528-MXFP4               376 GB  ← DSR1 model (our main)
            ├── models--amd--DeepSeek-R1-0528-MXFP4-MTP-MoEFP4    350 GB  ← variant (Triton trap, don't use)
            ├── models--amd--Kimi-K2.5-MXFP4                      521 GB  ← Kimi K2.5 model
            ├── models--lightseekorg--kimi-k2.5-eagle3              6 GB  ← Kimi Eagle3 drafter
            ├── datasets--gsm8k
            └── datasets--openai--gsm8k

/home/danish@neuralmerge.net/   ← your home, 28 KB (essentially empty — correct)
└── bin/docker                  ← wrapper script for docker access

/share4/                        ← OTHER TEAM's storage (99% full, ignore)
```

---

## 🐳 Docker Containers

Three containers, all based on `rocm/atom:rocm7.1.1-ubuntu24.04-pytorch2.9-atom0.1.1-MI350x`:

### `danish_atom_main`  ← **YOUR DSR1 MAIN**
- Size: 7.76 GB (after cleanup from 411 GB)
- Uses: GPUs 0-3 (TP=4 single-replica)
- Port: internal 8888 (bench talks to it via docker exec, not host port)
- Mounts:
  - `/projects/teamA/` (host) → `/projects/teamA/` (container)
  - HF cache, all repos visible via this
- Server: `python3 -m atom.entrypoints.openai_server ... --model amd/DeepSeek-R1-0528-MXFP4 -tp 4`
- This is where DEC-073 runs

### `danish_kimi`  ← **KIMI TRACK** (separate Opus session)
- Size: 1.31 GB (virtual 54 GB)
- Uses: GPUs 4-7
- Port: `8889:8888` (exposed on host port 8889)
- Workspace internally: `/ATOM_kimi`, `/aiter_kimi`, etc (mounted from `/projects/teamA/danish/kimi/`)
- **Separate from DSR1 — do NOT touch this container's code or configs**

### `danish_atom`  ← UNUSED/EXPERIMENTAL
- Size: 205 MB
- Port: `8888:8888` (conflicts with main? — probably idle)
- Purpose unclear, likely leftover from earlier session
- Don't touch, don't rely on

---

## 🔐 Separation of Concerns — DSR1 vs Kimi

| Aspect | DSR1 (yours, this Opus) | Kimi (other Opus) |
|---|---|---|
| **Container** | `danish_atom_main` | `danish_kimi` |
| **GPUs** | 0, 1, 2, 3 | 4, 5, 6, 7 |
| **ATOM code** | `/projects/teamA/danish/repos/ATOM_main` | `/projects/teamA/danish/kimi/ATOM_kimi` |
| **aiter code** | `/projects/teamA/danish/repos/aiter` | `/projects/teamA/danish/kimi/aiter_kimi` |
| **Model** | `amd/DeepSeek-R1-0528-MXFP4` | `amd/Kimi-K2.5-MXFP4` |
| **Bench harness** | `/projects/teamA/danish/repos/amdgpu_bounty_optimization/dsr1-*` | `/projects/teamA/danish/kimi/amdgpu_bounty_optimization/kimi-*` |
| **Host port** | internal 8888 (no host mapping) | 8889 (host) → 8888 (container) |
| **Shared** | HF model cache + filesystem root | HF model cache + filesystem root |
| **Floor config** | DEC-073 (1257/6.77/147.8/7390/0.9348) | (ask Kimi Opus) |

### Rules to keep tracks from colliding

1. **Don't touch `kimi/` subtree** — that's their code, their aiter, their vLLM
2. **Use GPUs 0-3 only** — never set HIP_VISIBLE_DEVICES to 4,5,6,7 in DSR1 work
3. **Don't restart `danish_kimi`** — it's their workspace
4. **Shared HF cache is READ-only** for our purposes (both tracks use the same model weights; don't delete anyone's model)
5. **Never run a DSR1 server on GPUs 4-7** — would conflict with Kimi's active processes

### What IS safely shared between tracks

- **HF model cache** at `/projects/teamA/hf_cache/` — immutable model weights, both tracks read from here
- **SGLang/vLLM reference repos** at `/projects/teamA/danish/repos/{sglang,vllm}` — read-only references
- **mori library** at `/projects/teamA/danish/_mori_*` — installed library, not code

---

## 🔑 DEC-073 Best Config (what produces 1/4 gates)

```
Stack:
├── ATOM commit: 108a70e + 3 local patches
│   ├── atom/model_ops/rejection_sampler.py   (relaxed MTP 8, 0.5)
│   ├── atom/model_ops/attention_mla.py        (num_kv_splits=None)
│   └── atom/spec_decode/eagle.py               (Phase 4A v4 drafter graph, null perf)
├── aiter commit: f8c1d76bd
│   └── aiter/configs/model_configs/dsv3_bf16_tuned_gemm.csv  (97 rows, tuned)
└── flydsl: 0.1.2

Launch env:
export HIP_FORCE_DEV_KERNARG=1
export NCCL_MIN_NCHANNELS=16
export ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=256
export ATOM_ENABLE_RELAXED_MTP=1
export HIP_VISIBLE_DEVICES=0,1,2,3

Flags:
-tp 4 --kv_cache_dtype fp8 --method mtp --num-speculative-tokens 3
--max-model-len 10240 --gpu-memory-utilization 0.85
```

---

## 🛠️ Useful Commands

```bash
# From Windows bash
ssh amd-gpu 'hostname'                                      # test SSH
ssh amd-gpu '~/bin/docker ps'                               # list containers
ssh amd-gpu '~/bin/docker exec danish_atom_main bash -c "cmd"'  # run cmd inside DSR1

# Server status
ssh amd-gpu 'rocm-smi --showmeminfo vram'                   # GPU memory
ssh amd-gpu '~/bin/docker ps --format "{{.Names}}\t{{.Status}}"'

# Bench DEC-073 (from inside danish_atom_main)
cd /workspace/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x
./dsr1_benchmark perf
```

---

## 📊 Current State Snapshot (2026-04-18)

- DSR1 submission floor: **DEC-073** (1/4 gates at CONC=4)
- All probes (EP+TP=4, MTP=4, BF16 KV) tested and DEAD
- DSR1 container cleanup: 411 GB → 7.76 GB (duplicate HF cache removed)
- Tree spec path discovered (mla_extend_ref.py in aiter op_tests) — feasibility TBD
- Kimi track: active on GPUs 4-7, 285 GB loaded per GPU (status per Kimi Opus)

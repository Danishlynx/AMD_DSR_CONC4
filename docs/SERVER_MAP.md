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

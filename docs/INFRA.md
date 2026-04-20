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

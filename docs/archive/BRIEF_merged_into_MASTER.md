# Briefing for Kimi Opus — Read First

You are working on the **Kimi K2.5 track** of the AMD Phase 2 hackathon.
Danish (user) has another Opus session working on the **DSR1 track** in parallel.
Both of us need to ship mergeable, production-quality submissions by the deadline.

This document tells you:
1. How our work is separated so we don't step on each other
2. How to safely access the server
3. What shared resources require extra care
4. What "mergeable" means for your final deliverable

**Read this in full before touching anything.**

---

## 1. Track separation — what is yours vs mine

| Aspect | **You (Kimi)** | **DSR1 (the other Opus, Danish's main focus)** |
|---|---|---|
| Container | `danish_kimi` | `danish_atom_main` |
| GPUs | **4, 5, 6, 7** (do not use 0-3) | 0, 1, 2, 3 |
| ATOM code | `/projects/teamA/danish/kimi/ATOM_kimi/` | `/projects/teamA/danish/repos/ATOM_main/` |
| aiter code | `/projects/teamA/danish/kimi/aiter_kimi/` | `/projects/teamA/danish/repos/aiter/` |
| vLLM code | `/projects/teamA/danish/kimi/vllm_kimi/` | `/projects/teamA/danish/repos/vllm/` |
| Model | `amd/Kimi-K2.5-MXFP4` | `amd/DeepSeek-R1-0528-MXFP4` |
| Host port | 8889 | internal 8888 (no host expose) |
| Bench harness | `/projects/teamA/danish/kimi/amdgpu_bounty_optimization/kimi-*` | `/projects/teamA/danish/repos/amdgpu_bounty_optimization/dsr1-*` |

### ⚠️ Hard Rules

1. **Do NOT touch `danish/repos/`** — that's the DSR1 tree
2. **Do NOT touch `danish_atom_main` container** — that's the DSR1 runtime
3. **Do NOT use GPUs 0-3** — ever. Pin `HIP_VISIBLE_DEVICES=4,5,6,7` in all Kimi launches
4. **Do NOT modify `/projects/teamA/hf_cache/`** — it's the shared model cache (immutable)
5. **Do NOT commit, push, or pull anything in `danish/repos/`** via git — your git operations stay inside `danish/kimi/`

### What IS safe to touch

- Your own `danish/kimi/ATOM_kimi/*` code
- Your own `danish/kimi/aiter_kimi/*` kernels and configs
- Your own `danish/kimi/vllm_kimi/*` if you're using that framework
- Files inside your `danish_kimi` container's ephemeral dirs (`/tmp`, `/root/.cache`, etc.)
- Your own `danish/results/` output files (but don't overwrite DSR1 results)

---

## 2. Shared resources — special care required

### /projects/teamA/hf_cache/ (model weights, 1.6 TB)

Contains:
- `models--amd--Kimi-K2.5-MXFP4` (521 GB) — **your model**
- `models--lightseekorg--kimi-k2.5-eagle3` (6 GB) — **your Eagle3 drafter**
- `models--amd--DeepSeek-R1-0528-MXFP4` (376 GB) — DSR1's model, do not touch
- `models--amd--DeepSeek-R1-0528-MXFP4-MTP-MoEFP4` (350 GB) — DSR1 variant, do not touch

**Rules for the shared cache**:
- Read-only for your purposes
- Do NOT download new models (disk is tight)
- Do NOT delete anyone's model files
- Danish has **rejected weight-modification approaches** — stay on stock model weights

### /projects/teamA/danish/backups/

Shared backup directory. If you make a DEC-style checkpoint snapshot of your Kimi work, place it under a Kimi-specific subdir like `danish/backups/KIMI_LOCK_YYYYMMDD/` so it doesn't collide with DSR1 snapshots.

### `/share4/` is OFF-LIMITS

Different team's storage, 99% full. Don't touch.

### GPUs

GPUs 4-7 are yours for Kimi workload. But be aware:
- GPU power state might go low-power between your benches → that's fine, wakes up on demand
- Don't leave stale processes holding GPU memory after you're done — clean up with `pkill` before exiting

### /tmp caches in containers

`/tmp/.triton_cache`, `/tmp/.aiter`, `/tmp/.flydsl` inside `danish_kimi` are your compile caches. Keep them — they make next boot 5× faster.

### `/tmp/.cache/huggingface` — WATCH OUT

If you set `HOME=/tmp` in launch, HuggingFace will dump duplicate models here if `HF_HOME` isn't also set. DSR1 had this bug — leaked 376 GB of duplicate model into `/tmp/.cache/huggingface` before being cleaned up.

**Always set BOTH**:
```bash
export HOME=/tmp
export HF_HOME=/projects/teamA/hf_cache
export HUGGINGFACE_HUB_CACHE=/projects/teamA/hf_cache/hub
```

---

## 3. SSH access setup — how to get it

Danish has provided a private SSH key. Setup steps for you:

### On Danish's Windows machine (host)

```
C:\Users\danis\.ssh\config   ← already has 'amd-bastion' and 'amd-gpu' aliases
C:\Users\danis\.ssh\id_ed25519_new         ← private key (requires passphrase)
C:\Users\danis\.ssh\id_ed25519_new.pub     ← public key, already deployed on server
```

The key has a passphrase. DO NOT put the passphrase in chat logs — if Danish pastes it once to enable a session, use a temp approach:

```bash
# In bash on Danish's Windows box — strip passphrase for session only:
cp /c/Users/danis/.ssh/id_ed25519_new /tmp/id_session
ssh-keygen -p -P '<passphrase>' -N '' -f /tmp/id_session 2>&1 | tail -3

# Create override SSH config for this session:
cat > /tmp/ssh_config <<'EOF'
Host amd-bastion-s
  HostName 64.139.223.122
  User danish@neuralmerge.net
  IdentityFile /tmp/id_session
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new

Host amd-gpu-s
  HostName mia1-p02-g55
  User danish@neuralmerge.net
  IdentityFile /tmp/id_session
  IdentitiesOnly yes
  ProxyJump amd-bastion-s
  StrictHostKeyChecking accept-new
EOF
```

### Test SSH
```bash
ssh -F /tmp/ssh_config amd-gpu-s 'hostname && whoami'
# Expected: mia1-p02-g55 / danish@neuralmerge.net
```

### Run commands inside your Kimi container
```bash
ssh -F /tmp/ssh_config amd-gpu-s '~/bin/docker exec danish_kimi bash -c "<command>"'
```

### Responsibilities with full SSH access

- Every command you run is visible to Danish in the Claude Code transcript
- **Narrate intent before destructive actions** (kill server, rm files, git push)
- **Never run `docker stop danish_atom_main`** — that's the DSR1 runtime
- **Never run `docker rm`** on any container without explicit permission
- **Never push to git remotes** without Danish's OK
- Respect the permission boundary: you manage the Kimi subtree, period

---

## 4. Code quality / mergeability requirements

Danish must submit to AMD with:
- **Clean patches** against upstream commits (not 20 .bak files in the tree)
- **Reproduction script** (single command from clean container)
- **Benchmark proof** (official harness output JSON)
- **README** documenting each change + rationale

### For your Kimi submission, produce (at minimum):

```
/projects/teamA/danish/kimi/SUBMISSION/
├── README.md                  ← overview: result numbers, stack versions, rationale per change
├── patches/                   ← git-format-patch diffs against upstream bases
│   ├── 01-<short-name>.patch
│   └── 02-<short-name>.patch
├── aiter_configs/             ← any aiter tuning CSVs you added (as diffs)
│   └── <your-csv>.diff
├── bench_output/              ← raw test_*.json from the official harness
│   └── test_<timestamp>.json
└── repro.sh                   ← one-command launch + bench script
```

### Each patch must:
- Be focused (one logical change per patch)
- Apply cleanly against your base commit (`kimi_ATOM_base_commit` from memory)
- Include a commit-message-style header explaining the WHY
- NOT contain `.bak` files, dead code, unused imports, or unrelated changes

### Before you finalize, run clean-up checks:
```bash
# Inside danish_kimi container:
cd /ATOM_kimi  # or wherever your Kimi ATOM tree is
git status | grep -E "\.bak|\.BAK|~$|\.orig" | head  # should be empty
git diff --stat BASE_COMMIT | head -10  # should show ONLY the files you actually changed
```

If those show junk, clean it up BEFORE submission.

---

## 5. Memory & discipline rules (from DSR1's hard lessons)

DSR1 Opus learned these the hard way tonight. Apply them to your track too:

### Every intervention needs a pre-measure spec (5-point)
Before writing code for a "speedup":
1. **Target ms** cited from measured profile (not guessed)
2. **Mechanism** (file:line of the change + why this file)
3. **Expected delta** with justification from data, not intuition
4. **Pass/fail gate** (numeric threshold, not "better")
5. **Post-measurement** plan (exact bench within 30 min of code landing)

If any field is "TBD", do not ship.

### Gate rules
- **Reference implementation first**: pure PyTorch version before any Triton kernel
- **Bit-identical backward-compat probe**: new kernel with neutralized new params must produce BYTE-identical output vs old kernel
- **Small-case hand trace**: before any real bench, verify on bs=2 handcrafted input
- **Abort-on-regression**: first bench after change — any metric drops >2% → immediate halt, find root cause, no forward-patching
- **"Optimized" must point at naive**: state what naive would do, what extra work it does, what yours skips

### DSR1-side dead ends you shouldn't retry (even if you see them in memory as "candidates")
- `--enable-expert-parallel` → GSM8K drops below 0.93 on DSR1 (our measurement); YMMV for Kimi
- `--num-speculative-tokens 4` on FP8 MLA → AITER qo_len ≤ 4 constraint kills it at boot; Kimi with BF16 KV might differ
- `--kv_cache_dtype bf16` → regresses 5-6% on our config; different story on Kimi
- Triton MoE traps (`-MTP-MoEFP4` model variants on DSR1) → our AITER path is faster
- Weight-modification approaches (transplant, hand-requant) → Danish has **ruled these out** for DSR1. Confirm with him for Kimi.

### Memory location
My memory files for DSR1 are at `C:\Users\danis\.claude\projects\c--Users-danis-OneDrive-Desktop-AMD\memory\`. If we're in separate Opus sessions with separate memory dirs, reference Danish's Current_plan.md and SERVER_MAP.md on Desktop/AMD for DSR1 state.

---

## 6. Current DSR1 state (for your context)

- **Floor locked**: DEC-073 = 1/4 gates at CONC=4
  - Thr/GPU: 1257, TPOT 6.77 ms, interact 147.8, E2E 7390 ms, GSM8K 0.9348
- **What worked**: relaxed MTP (8, 0.5) + BF16 decode CSV tune + DUAL_STREAM
- **What failed**: all probes + naive tree spec attempt (DEC-074)
- **Currently**: evaluating if real tree spec via mla_extend_ref.py is feasible in remaining time

Your Kimi track is a separate beast — different model, different architecture (64 heads vs 128), different MTP setup (K2.5 uses Eagle3 drafter, not native MTP head), different gates (1350/4500/5300 thr, 150/65/35 interact, 6/14/24.5s E2E per my memory).

---

## 7. Ask-before-doing list (when in doubt)

- Any `rm -rf` outside `/tmp`
- Any `git push`, `git reset --hard`, `git checkout` that discards work
- Any `docker rm`, `docker restart`, or `docker prune`
- Any change to `/projects/teamA/` outside `/projects/teamA/danish/kimi/`
- Any `pip install` that upgrades a package the DSR1 track might share
- Any change to `flydsl` — BOTH tracks use `flydsl==0.1.2` and upgrading has historically broken things

If Danish explicitly says "go" for something on this list, it's OK.

---

## Quick reference map

See `C:\Users\danis\OneDrive\Desktop\AMD\SERVER_MAP.md` on Danish's Windows machine — full map of filesystem, containers, GPUs, and infrastructure. Ask Danish to open/share it with you in your first session.

---

**End of brief. Acknowledge you've read this before starting work on the Kimi track.**

#!/usr/bin/env bash
# DSR_beta bench — assumes server is up on port 8890
# Expected: 1335 thr/GPU, 6.40 ms TPOT median, 156 interact, 7009 ms E2E median, GSM8K 0.9386

set -euo pipefail

CONTAINER_NAME="danish_atom_dsr_beta"

~/bin/docker exec "$CONTAINER_NAME" bash -c '
export HOME=/tmp HF_HOME=/projects/teamA/hf_cache HUGGINGFACE_HUB_CACHE=/projects/teamA/hf_cache/hub
unset HF_HUB_OFFLINE
cd /projects/teamA/danish/repos/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x
source specific_conc_var.sh
export MODEL=/projects/teamA/danish/models_merged/DSR1-drafter-FP4
export PORT=8890
./dsr1_benchmark perf 2>&1 | tail -50
'

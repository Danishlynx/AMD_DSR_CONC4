#!/bin/bash
# Foreman cycle script. Usage: bash agents/harness/cycle.sh {dsr1|kimi}
#
# Runs from laptop. SSH-driven. Foreman (Claude) reads outputs between stages
# and decides next stage / abort. This script is the SHELL of the cycle; the
# brain is the foreman parsing the JSON / log artifacts.
#
# Outputs land under runs/<model>/<ts>/ on laptop. Container artifacts are
# scp'd back at the end of each stage.

set -u
if [ -z "${1:-}" ]; then echo "usage: cycle.sh {dsr1|kimi}" >&2; exit 2; fi
MODEL=$1
TS=$(date +%Y%m%d_%H%M%S)

case "$MODEL" in
  dsr1)
    CONTAINER=re4c_v10
    BOOT_SCRIPT=/tmp/boot_a26.sh
    BENCH_SCRIPT=/tmp/official_bench_n3.sh
    LOCK_FILE=/tmp/locks/gpu.busy.dsr1
    SISTER_LOCK=/tmp/locks/gpu.busy.kimi
    BOOT_PORT=8890
    ;;
  kimi)
    CONTAINER=kimi_latest
    BOOT_SCRIPT=/tmp/boot_kimi.sh        # placeholder — to be created mirror of dsr1
    BENCH_SCRIPT=/tmp/official_bench_n3.sh
    LOCK_FILE=/tmp/locks/gpu.busy.kimi
    SISTER_LOCK=/tmp/locks/gpu.busy.dsr1
    BOOT_PORT=8890
    ;;
  *)
    echo "ERROR: model must be dsr1 or kimi"
    exit 2
    ;;
esac

RUNDIR_LAPTOP="runs/$MODEL/$TS"
RUNDIR_HOST="/tmp/runs/$MODEL/$TS"
mkdir -p "$RUNDIR_LAPTOP"
echo "[cycle] model=$MODEL ts=$TS container=$CONTAINER" | tee "$RUNDIR_LAPTOP/cycle.log"

source /tmp/ssh_agent_env.sh 2>/dev/null

ssh_exec() {
  ssh amd-gpu "docker exec $CONTAINER bash -c '$1'" 2>&1
}

ssh_host() {
  ssh amd-gpu "$1" 2>&1
}

stage() {
  echo "" | tee -a "$RUNDIR_LAPTOP/cycle.log"
  echo "===== STAGE $1: $2 =====" | tee -a "$RUNDIR_LAPTOP/cycle.log"
}

abort() {
  echo "" | tee -a "$RUNDIR_LAPTOP/cycle.log"
  echo "===== ABORT: $1 =====" | tee -a "$RUNDIR_LAPTOP/cycle.log"
  ssh_host "rm -f $LOCK_FILE" >/dev/null
  exit 3
}

# ---- Stage 0: health check ----
stage 0 "health check"
ssh_host "mkdir -p /tmp/locks /tmp/runs/$MODEL/$TS"
HEALTH=$(ssh_host "
  rocm-smi -c | grep -c '2100' || echo 0
  echo ---SISTER---
  cat $SISTER_LOCK 2>/dev/null || echo NO_SISTER_LOCK
  echo ---SERVER---
  docker exec $CONTAINER curl -sf -o /dev/null -w 'http=%{http_code}\n' http://localhost:$BOOT_PORT/v1/models 2>&1 || echo http=000
")
echo "$HEALTH" | tee "$RUNDIR_LAPTOP/00_health.txt"

# Acquire lock
ssh_host "echo $TS > $LOCK_FILE"

# ---- Stage 1: AutoKernel ----
stage 1 "AutoKernel rank"
# Foreman: ship autokernel script to container (rsync), then run.
# For initial bring-up we run a simple rocprof-based profiler. See agents/autokernel/run.py.
# Output: $RUNDIR_HOST/autokernel.json
echo "[stage1] AutoKernel deferred to foreman manual call (agents/autokernel/run.py not yet shipped). Skipping for no-op cycle." | tee -a "$RUNDIR_LAPTOP/cycle.log"

# ---- Stage 2: GEAK-HIP ----
stage 2 "GEAK-HIP generate"
echo "[stage2] GEAK-HIP deferred to foreman manual call (agents/geak_hip/run.py not yet shipped). Skipping for no-op cycle." | tee -a "$RUNDIR_LAPTOP/cycle.log"

# ---- Stage 3: Audit ----
stage 3 "audit"
echo "[stage3] No candidate to audit in no-op cycle. Skip." | tee -a "$RUNDIR_LAPTOP/cycle.log"

# ---- Stage 4: Patch locked container ----
stage 4 "patch locked"
echo "[stage4] No-op cycle: no patch applied. Locked container untouched." | tee -a "$RUNDIR_LAPTOP/cycle.log"

# ---- Stage 5: OFFICIAL bench ----
stage 5 "OFFICIAL harness"
ssh_host "docker exec $CONTAINER curl -sf -o /dev/null http://localhost:$BOOT_PORT/v1/models" || abort "server not responding"

# Make sure host run dir exists
ssh_host "mkdir -p /tmp/runs/$MODEL/$TS"

# Run bench inside container; capture stdout via ssh
echo "[stage5] running official_bench_v1.sh inside $CONTAINER (this takes ~10-12 min)..." | tee -a "$RUNDIR_LAPTOP/cycle.log"
ssh amd-gpu "docker exec $CONTAINER bash $BENCH_SCRIPT" > "$RUNDIR_LAPTOP/05_bench_locked.log" 2>&1

# Pull the summary file (latest one) out of the container
SUMMARY_PATH=$(ssh amd-gpu "docker exec $CONTAINER bash -c 'ls -t /tmp/official_summary_*.txt 2>/dev/null | head -1'" 2>/dev/null | tr -d '\r')
if [ -n "$SUMMARY_PATH" ]; then
  ssh amd-gpu "docker exec $CONTAINER cat $SUMMARY_PATH" > "$RUNDIR_LAPTOP/05_bench_locked_summary.txt" 2>/dev/null
  echo "[stage5] summary captured: $SUMMARY_PATH" | tee -a "$RUNDIR_LAPTOP/cycle.log"
  echo "" | tee -a "$RUNDIR_LAPTOP/cycle.log"
  cat "$RUNDIR_LAPTOP/05_bench_locked_summary.txt" | tee -a "$RUNDIR_LAPTOP/cycle.log"
else
  echo "[stage5] WARN: no summary file produced inside container" | tee -a "$RUNDIR_LAPTOP/cycle.log"
fi

# ---- Stage 6: Fresh-container verify ----
stage 6 "fresh-container verify"
echo "[stage6] No-op cycle: no patch to verify. Skip." | tee -a "$RUNDIR_LAPTOP/cycle.log"

# ---- Stage 7: promote or revert ----
stage 7 "promote-or-revert"
echo "[stage7] No-op cycle: nothing to promote." | tee -a "$RUNDIR_LAPTOP/cycle.log"

# ---- Stage 8: DEC log entry ----
stage 8 "DEC log"
cat > "$RUNDIR_LAPTOP/DEC.md" <<EOF
# DEC entry — $MODEL $TS — NO-OP CYCLE (loop validation)

This cycle ran the harness end-to-end with no patch applied. Purpose:
verify the orchestrator + cycle.sh + official_bench_v1.sh chain works.

- Stage 5 result: see 05_bench_locked_summary.txt
- Stages 1/2/3/4/6/7: skipped (no-op).

If the Stage 5 numbers match A26 baseline (GSM8K 0.9378 / E2E 6908 / Tput 1357 / Intvty 157),
the harness is healthy. Any drift means infra issue, not lever issue.
EOF

ssh_host "rm -f $LOCK_FILE"
echo "[cycle] DONE. Artifacts: $RUNDIR_LAPTOP/"

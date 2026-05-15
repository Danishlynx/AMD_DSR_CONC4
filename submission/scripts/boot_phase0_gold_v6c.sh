#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 0 baseline boot — gold+v6c stack
# Reproduces snapshot rocm/atom-dev:dsr1_apr29_gold_v6c_2of4 sha 4911df6937b152
# Expected kimbochen 3-iter median: TPOT 6.080 ms / E2E 6743 / Tput 1403 / Intvty 164.48 / GSM8K 0.9340
# Source: docs/Daily Updates/REPRODUCE.md "🟢 CANONICAL kimbochen-VALIDATED RECIPE (gold+v6c)"
# ─────────────────────────────────────────────────────────────────────────────

LOGFILE=/tmp/gold_v6c_boot_$(date +%H%M%S).log
echo "$LOGFILE" > /tmp/last_boot_log.txt
echo "=== Phase 0 gold+v6c boot ==="
echo "  log: $LOGFILE"

# ── perf determinism — let GPU boost 2100 → 2396 MHz on MI355X ────────────────
echo "=== reset perf determinism ==="
rocm-smi --resetperfdeterminism -d 0 -d 1 -d 2 -d 3 2>&1 | tail -3

# ── strip stale state from prior session ──────────────────────────────────────
unset AITER_ENABLE_HK_QH32 AITER_ENABLE_HK_QH32_V11 ATOM_CUDAGRAPH_MODE \
      ATOM_USE_CDNA4_MOE_GEMM2 CDNA4_MOE_LIB CDNA4_MOE_DEBUG PYTHONSTARTUP \
      ATOM_MSCG_K   # silent regression Apr 26 — KEEP UNSET (not just =0)

# ── HF cache + offline ────────────────────────────────────────────────────────
export HOME=/tmp HF_HOME=/tmp/.cache/huggingface \
       HUGGINGFACE_HUB_CACHE=/tmp/.cache/huggingface/hub \
       TRANSFORMERS_CACHE=/tmp/.cache/huggingface/hub \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# ── ATOM features ─────────────────────────────────────────────────────────────
export AITER_ENABLE_VSKIP=0 ATOM_ENABLE_RELAXED_MTP=1
export ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=1024   # KEEP at 1024; 0/512 regresses

# ── HIP ───────────────────────────────────────────────────────────────────────
export HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1
export HIP_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1

# ── Collectives — RCCL ROCm 7.1+ knobs (proven +6.8% throughput Apr 22) ───────
export NCCL_MIN_NCHANNELS=16                 # 32 regresses (TPOT +7%)
export RCCL_MSCCLPP_ENABLE=1
export RCCL_MSCCLPP_THRESHOLD=1048576
export RCCL_P2P_BATCH_ENABLE=1

# ── Cold-boot timeout extensions ──────────────────────────────────────────────
export NCCL_TIMEOUT=3600 TORCH_NCCL_BLOCKING_WAIT=0 \
       TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 TORCH_NCCL_TIMEOUT=3600 \
       TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=3600

# ── QuickReduce INT4 AR (RE.1 lock — proven +6.8 % throughput, GSM8K-safe) ────
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4

# ── defaults explicit ─────────────────────────────────────────────────────────
export ATOM_MSCG_P6_REPLAY=0

# ── v6c WIN — gold+v6c = THIS stack with v6c env-gated ON ─────────────────────
export ATOM_USE_POST_ATTN_RMSNORM_MXFP4_FUSION=1

# ── disable any profiler in perf-mode boot ────────────────────────────────────
unset ATOM_DECODE_PROFILE_DIR ATOM_DECODE_PROFILE_SKIP ATOM_DECODE_PROFILE_NUM
unset ATOM_TORCH_PROFILER_DIR ATOM_PROFILER_MORE

# ── server launch ─────────────────────────────────────────────────────────────
cd /app/ATOM
nohup python3 -m atom.entrypoints.openai_server \
  --model amd/DeepSeek-R1-0528-MXFP4 \
  --server-port 8890 -tp 4 --kv_cache_dtype fp8 --max-model-len 10240 \
  --method mtp --num-speculative-tokens 3 --enable-tbo prefill \
  --max-num-batched-tokens 65536 \
  --cudagraph-capture-sizes "[1,2,4,8,16,32]" > "$LOGFILE" 2>&1 &

PID=$!
echo "BOOTED PID=$PID LOG=$LOGFILE"
echo "$PID" > /tmp/last_boot_pid.txt
echo "Tail with: tail -f $LOGFILE"
echo "Wait for: 'Application startup complete' (~12 min cold-boot)"

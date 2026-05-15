#!/bin/bash
# L0-v2 FULL_DECODE_ONLY boot — gold+v6c stack + ATOM_CUDAGRAPH_MODE=FULL_AND_PIECEWISE + ATOM_COMPILE_SIZES=cudagraph_capture_sizes

LOGFILE=/tmp/l0v2_full_decode_only_boot_$(date +%H%M%S).log
echo "$LOGFILE" > /tmp/last_boot_log.txt
echo "=== L0-v2 FULL_DECODE_ONLY boot (gold+v6c + ATOM_USE_TRITON_MXFP4_BMM=1) ==="
echo "  log: $LOGFILE"

# perf determinism — let GPU boost
rocm-smi --resetperfdeterminism -d 0 -d 1 -d 2 -d 3 2>&1 | tail -3

# strip stale state
unset AITER_ENABLE_HK_QH32 AITER_ENABLE_HK_QH32_V11 \
      ATOM_USE_CDNA4_MOE_GEMM2 CDNA4_MOE_LIB CDNA4_MOE_DEBUG PYTHONSTARTUP \
      ATOM_MSCG_K

# HF cache + offline
export HOME=/tmp HF_HOME=/tmp/.cache/huggingface \
       HUGGINGFACE_HUB_CACHE=/tmp/.cache/huggingface/hub \
       TRANSFORMERS_CACHE=/tmp/.cache/huggingface/hub \
       HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# ATOM features (gold+v6c)
export AITER_ENABLE_VSKIP=0 ATOM_ENABLE_RELAXED_MTP=1
export ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=1024

# HIP
export HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1
export HIP_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1

# Collectives
export NCCL_MIN_NCHANNELS=16
export RCCL_MSCCLPP_ENABLE=1
export RCCL_MSCCLPP_THRESHOLD=1048576
export RCCL_P2P_BATCH_ENABLE=1

# Cold-boot timeout extensions
export NCCL_TIMEOUT=3600 TORCH_NCCL_BLOCKING_WAIT=0 \
       TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 TORCH_NCCL_TIMEOUT=3600 \
       TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=3600

# QuickReduce INT4 AR
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4

# defaults
export ATOM_MSCG_P6_REPLAY=0

# v6c WIN
export ATOM_USE_POST_ATTN_RMSNORM_MXFP4_FUSION=1

# === L0-v2 FULL_DECODE_ONLY (sidesteps dual-stream MoE conflict that killed L0 v1) ===
export ATOM_CUDAGRAPH_MODE=FULL_DECODE_ONLY

# DISABLED for L3 baseline: export ATOM_USE_TRITON_MXFP4_BMM=1

# === Phase 1 L0 + L1 ===
# DISABLED for L1-only bisect: export ATOM_CUDAGRAPH_MODE=FULL_AND_PIECEWISE
# DISABLED for L2-only: export ATOM_COMPILE_SIZES=cudagraph_capture_sizes

# clear profilers
unset ATOM_DECODE_PROFILE_DIR ATOM_DECODE_PROFILE_SKIP ATOM_DECODE_PROFILE_NUM
unset ATOM_TORCH_PROFILER_DIR ATOM_PROFILER_MORE

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

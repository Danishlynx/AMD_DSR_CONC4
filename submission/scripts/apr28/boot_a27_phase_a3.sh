#!/bin/bash
# Phase A3 boot — Phase A1 memoization (already applied to source) + ATOM_USE_TRITON_GEMM=1
# to activate the already-shipped fused qkv_a_proj+rmsnorm+quant path
# (deepseek_v2.py:1528, gate flipped from False to True).
LOGFILE=/tmp/a27_phase_a3_boot_$(date +%H%M%S).log
echo "$LOGFILE" > /tmp/last_boot_log.txt
echo "=== reset perf determinism ==="
rocm-smi --resetperfdeterminism -d 0 -d 1 -d 2 -d 3 2>&1 | tail -3
unset AITER_ENABLE_HK_QH32 AITER_ENABLE_HK_QH32_V11
export HOME=/tmp HF_HOME=/tmp/.cache/huggingface HUGGINGFACE_HUB_CACHE=/tmp/.cache/huggingface/hub TRANSFORMERS_CACHE=/tmp/.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export AITER_ENABLE_VSKIP=0 ATOM_ENABLE_RELAXED_MTP=1
export ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=1024
export HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1
export NCCL_MIN_NCHANNELS=16
export RCCL_MSCCLPP_ENABLE=1 RCCL_MSCCLPP_THRESHOLD=1048576 RCCL_P2P_BATCH_ENABLE=1
export HIP_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=1 AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_MSCG_K=2
# Phase A3: activate fused qkv_a_proj+rmsnorm+quant path (gated by use_triton_gemm())
export ATOM_USE_TRITON_GEMM=1
unset ATOM_DECODE_PROFILE_DIR ATOM_DECODE_PROFILE_SKIP ATOM_DECODE_PROFILE_NUM
cd /app/ATOM
nohup python3 -m atom.entrypoints.openai_server \
  --model amd/DeepSeek-R1-0528-MXFP4 \
  --server-port 8890 -tp 4 --kv_cache_dtype fp8 --max-model-len 10240 \
  --method mtp --num-speculative-tokens 3 --enable-tbo prefill \
  --max-num-batched-tokens 65536 \
  --cudagraph-capture-sizes "[1,2,4,8,16,32]" > "$LOGFILE" 2>&1 &
echo "BOOTED PID=$! LOG=$LOGFILE"

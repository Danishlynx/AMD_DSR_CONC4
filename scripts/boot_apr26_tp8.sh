#!/bin/bash
LOGFILE=/tmp/tp8_boot_$(date +%H%M%S).log
echo "$LOGFILE" > /tmp/last_boot_log.txt
echo "=== reset perf determinism ==="
rocm-smi --resetperfdeterminism -d 0 -d 1 -d 2 -d 3 -d 4 -d 5 -d 6 -d 7 2>&1 | tail -3
unset AITER_ENABLE_HK_QH32 AITER_ENABLE_HK_QH32_V11
export HOME=/tmp HF_HOME=/tmp/.cache/huggingface HUGGINGFACE_HUB_CACHE=/tmp/.cache/huggingface/hub TRANSFORMERS_CACHE=/tmp/.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export AITER_ENABLE_VSKIP=0 ATOM_ENABLE_RELAXED_MTP=1
export ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=1024
export HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1
export NCCL_MIN_NCHANNELS=16
export NCCL_TIMEOUT=3600 NCCL_BLOCKING_WAIT=0
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=3600
export RCCL_MSCCLPP_ENABLE=1 RCCL_MSCCLPP_THRESHOLD=1048576 RCCL_P2P_BATCH_ENABLE=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=1 AITER_QUICK_REDUCE_QUANTIZATION=INT4
# export ATOM_MSCG_K=2  # disabled (matches Apr 26 3/4 baseline)
export ATOM_MSCG_P6_REPLAY=0
# CDNA4 MoE GEMM2 atomic shim
export ATOM_USE_CDNA4_MOE_GEMM2=0
# export ATOM_DECODE_PROFILE_DIR=/tmp/decode_profile (off for perf bench)
export CDNA4_MOE_LIB=/tmp/cdna4_gemm2/build/libcdna4_moe_gemm2.so
export CDNA4_MOE_DEBUG=0  # set 1 to log every dispatch
export PYTHONPATH=/tmp/cdna4_gemm2:${PYTHONPATH}
# Preload the shim BEFORE atom imports aiter so the patch installs before
# MOEMetadata is built (functools.partial captures function reference at build time).
export PYTHONSTARTUP=/tmp/cdna4_gemm2/_preload_shim.py
cd /app/ATOM
nohup python3 -m atom.entrypoints.openai_server \
  --model amd/DeepSeek-R1-0528-MXFP4 \
  --server-port 8890 -tp 8 --kv_cache_dtype fp8 --max-model-len 10240 \
  --method mtp --num-speculative-tokens 3 --enable-tbo prefill \
  --max-num-batched-tokens 65536 \
  --cudagraph-capture-sizes "[1,2,4,8,16,32]" > "$LOGFILE" 2>&1 &
echo "BOOTED PID=$! LOG=$LOGFILE"

#!/usr/bin/env bash
# P0 boot with torch profiler enabled via CLI flag (env alone not enough — CLI default None overrides)
set -e

export HOME=/tmp
export HF_HOME=/tmp/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/tmp/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export AITER_ENABLE_VSKIP=0
export ATOM_ENABLE_RELAXED_MTP=1
export ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=1024
export HIP_FORCE_DEV_KERNARG=1
export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_MIN_NCHANNELS=16
export HIP_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=FP
export VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=1
export ATOM_TORCH_PROFILER_DIR=/tmp/torch_traces_contaminated

mkdir -p /tmp/torch_traces_contaminated

cd /app/ATOM
exec python3 -m atom.entrypoints.openai_server \
  --model amd/DeepSeek-R1-0528-MXFP4 \
  --server-port 8890 \
  -tp 4 \
  --kv_cache_dtype fp8 \
  --max-model-len 10240 \
  --method mtp \
  --num-speculative-tokens 3 \
  --enable-tbo prefill \
  --max-num-batched-tokens 65536 \
  --torch-profiler-dir /tmp/torch_traces_contaminated \
  --cudagraph-capture-sizes "[1,2,4,8,16,32]"

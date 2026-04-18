#!/usr/bin/env bash
# DSR_beta launch — reproduces 1335 thr/GPU / 6.40 TPOT / 156 interact / 7009 E2E / 0.9386 GSM8K
# Requires: dsr_beta_setup.sh already run (container danish_atom_dsr_beta exists)

set -euo pipefail

CONTAINER_NAME="danish_atom_dsr_beta"
PORT=8890

echo "=== 1. Clear stale caches (ensure fresh JIT compile) ==="
~/bin/docker exec "$CONTAINER_NAME" bash -c '
rm -rf /tmp/torchinductor_root /tmp/.cache/atom /tmp/.aiter_cache /tmp/.triton_cache /tmp/.flydsl /tmp/aiter_configs
echo CACHES_CLEARED
'

echo "=== 2. Ensure no previous server running ==="
~/bin/docker exec "$CONTAINER_NAME" bash -c 'pkill -TERM -f openai_server 2>&1 || true; sleep 6'

echo "=== 3. Launch server (detached) ==="
~/bin/docker exec -d "$CONTAINER_NAME" bash -c '
export HOME=/tmp
export HF_HOME=/projects/teamA/hf_cache
export HUGGINGFACE_HUB_CACHE=/projects/teamA/hf_cache/hub
unset HF_HUB_OFFLINE
export AITER_BUILD_DIR=/tmp/.aiter_cache TRITON_CACHE_DIR=/tmp/.triton_cache
export HIP_FORCE_DEV_KERNARG=1
export NCCL_MIN_NCHANNELS=16
export ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=256
export ATOM_ENABLE_RELAXED_MTP=1
export HIP_VISIBLE_DEVICES=0,1,2,3
cd /app/ATOM
python3 -m atom.entrypoints.openai_server \
  --model /projects/teamA/danish/models_merged/DSR1-drafter-FP4 \
  --server-port 8890 \
  -tp 4 \
  --kv_cache_dtype fp8 \
  --method mtp \
  --num-speculative-tokens 3 \
  --max-model-len 10240 \
  --gpu-memory-utilization 0.85 \
  --enable-tbo prefill > /tmp/atom-dsr_beta.stdout 2>&1
'

echo "=== 4. Wait for server UP (max 12 min cold boot) ==="
for i in $(seq 1 72); do
  sleep 10
  if ~/bin/docker exec "$CONTAINER_NAME" curl -fs "http://0.0.0.0:$PORT/health" > /dev/null 2>&1; then
    echo "Server UP at port $PORT after $((i*10))s"
    break
  fi
  if [ $i -eq 72 ]; then
    echo "ERROR: Server failed to come up in 12 min" >&2
    ~/bin/docker exec "$CONTAINER_NAME" tail -30 /tmp/atom-dsr_beta.stdout
    exit 1
  fi
done

echo "=== 5. Verify boot markers ==="
~/bin/docker exec "$CONTAINER_NAME" bash -c '
if grep -q "max_q_len=4" /tmp/atom-dsr_beta.stdout; then
  echo "✓ mtp_k=3 (max_q_len=4) confirmed"
else
  echo "⚠ mtp_k NOT 3 — check log" >&2
fi
if grep -q "flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w3_fq" /tmp/atom-dsr_beta.stdout; then
  echo "✓ drafter FP4 fast path confirmed"
else
  echo "⚠ drafter NOT on FP4 fast path" >&2
fi
if grep -q "enable_tbo.*True" /tmp/atom-dsr_beta.stdout; then
  echo "✓ TBO enabled"
else
  echo "⚠ TBO not enabled" >&2
fi
'

echo ""
echo "=== DSR_beta server READY on port $PORT ==="
echo "Run bench: bash scripts/dsr_beta_bench.sh"

#!/bin/bash
# RE.3 v2 — MoE CSV tuning with FIX for stdout buffering (python -u)
# Also: reduce shape set to just token=32768 (the critical missing bucket from boot log)
# That shape takes ~7000 tasks; single shape = maybe 10-20 min, not 2h.
set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HIP_VISIBLE_DEVICES=0
export HOME=/tmp
export PYTHONUNBUFFERED=1  # force stdout flush

cd /app/aiter-test

echo "=== RE.3 v2 MoE tuner start $(date -u) ==="
cat /tmp/re3_moe_untuned_v2.csv
echo

# -u for unbuffered python output
python3 -u csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  -i /tmp/re3_moe_untuned_v2.csv \
  -o /app/aiter-test/aiter/configs/model_configs/dsr1_fp4_tuned_fmoe.csv \
  -o2 /tmp/re3_moe_profile_all_v2.csv \
  --errRatio 0.05 \
  --warmup 3 --iters 20 \
  --batch 20 --mp 1 \
  --sort True --all --compare --update_improved \
  2>&1 | tee /tmp/re3_moe_tune_v2.log

echo "=== DONE $(date -u) ==="
wc -l /app/aiter-test/aiter/configs/model_configs/dsr1_fp4_tuned_fmoe.csv 2>/dev/null || echo "NO CSV CREATED"
head -5 /app/aiter-test/aiter/configs/model_configs/dsr1_fp4_tuned_fmoe.csv 2>/dev/null

#!/bin/bash
# RE.3 — MoE CSV tuning via aiter's official gemm_moe_tune.py
# Adapted from Phase 1 run_tuner_kimi_strict_v2.sh (OOM-resilient config)
set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HIP_VISIBLE_DEVICES=0
export HOME=/tmp
export HF_HOME=/tmp/.cache/huggingface

cd /app/aiter-test

echo "=== RE.3 MoE tuner start $(date -u) ==="
echo "=== Input ==="
cat /tmp/re3_moe_untuned.csv
echo
echo "=== VRAM pre-tune ==="
rocm-smi --showmeminfo vram 2>/dev/null | grep Used | head -4

python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  -i /tmp/re3_moe_untuned.csv \
  -o /app/aiter-test/aiter/configs/model_configs/dsr1_fp4_tuned_fmoe.csv \
  -o2 /tmp/re3_moe_profile_all.csv \
  --errRatio 0.05 \
  --warmup 5 --iters 100 \
  --batch 20 --mp 1 \
  --sort True --all --compare --update_improved \
  2>&1 | tee /tmp/re3_moe_tuner.log

echo "=== DONE $(date -u) ==="
echo "=== Output CSV ==="
wc -l /app/aiter-test/aiter/configs/model_configs/dsr1_fp4_tuned_fmoe.csv 2>/dev/null || echo "OUTPUT NOT CREATED"
head -5 /app/aiter-test/aiter/configs/model_configs/dsr1_fp4_tuned_fmoe.csv 2>/dev/null

#!/usr/bin/env bash
# Build recipe for HK qh32 v8 on server container (reproducer_best).
#
# v8 = v7 + inner q_pos loop over [qo_start, qo_end) + Opt-E s_setprio.
# Coexists with v7 — different symbol, different dispatch gate.
#
# Prerequisites:
#   1. Copy v8_h32.cuh → /app/aiter-test/csrc/kernels/mla/hk/mi3xx_v32_fwd_decode_h32_fp8_fp8_v8.cuh
#   2. Apply aiter_mla_py_patch_v8.diff to /app/aiter-test/aiter/mla.py
#   3. Apply hk_decode_fwd_v8_append.diff to /app/aiter-test/csrc/cpp_itfs/mla/hk_decode_fwd.cu
#
# This script runs INSIDE the container (docker exec reproducer_best bash ...).

set -euo pipefail

# Backup v7 state before v8 intrusion.
cp /app/aiter-test/aiter/mla.py /app/aiter-test/aiter/mla.py.pre_v8
cp /app/aiter-test/csrc/cpp_itfs/mla/hk_decode_fwd.cu /app/aiter-test/csrc/cpp_itfs/mla/hk_decode_fwd.cu.pre_v8

# Wipe JIT cache for module_hk_mla so the new v8 cuh gets picked up.
rm -rf /tmp/.aiter/module_hk_mla

# Trigger JIT build with v8 enabled.
HOME=/tmp \
  AITER_ENABLE_EXPERIMENTAL=1 \
  AITER_ENABLE_HK_QH32_V8=1 \
  python3 -c "import aiter; print('v8 symbol available:', hasattr(aiter, 'hk_mi3xx_mla_v32_fwd_decode_h32_fp8_fp8_v8'))"

echo "Build succeeded. module_hk_mla rebuilt with v8 symbol."
echo "To run correctness test:"
echo "  HOME=/tmp HIP_VISIBLE_DEVICES=1 AITER_ENABLE_EXPERIMENTAL=1 python3 /tmp/test_hk_qh32_v8_correctness.py"
echo ""
echo "To rollback to v7:"
echo "  cp /app/aiter-test/aiter/mla.py.pre_v8 /app/aiter-test/aiter/mla.py"
echo "  cp /app/aiter-test/csrc/cpp_itfs/mla/hk_decode_fwd.cu.pre_v8 /app/aiter-test/csrc/cpp_itfs/mla/hk_decode_fwd.cu"
echo "  rm /app/aiter-test/csrc/kernels/mla/hk/mi3xx_v32_fwd_decode_h32_fp8_fp8_v8.cuh"
echo "  rm -rf /tmp/.aiter/module_hk_mla"

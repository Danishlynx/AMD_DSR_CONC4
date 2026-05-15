#!/bin/bash
# R2-C M1 build script: compile r2_smallm_moe_gemm2.cu for CDNA4 (gfx950).
# Outputs: r2_smallm_moe_gemm2.so (shared library loadable from Python via ctypes).
#
# Per Apr 29 R1 OOB lesson: -O3 may segfault the LLVM machine scheduler when
# sched_group_barrier interleaves with mfma_scale. Start at -O1; bump to -O3 only after parity.

set -e

SRC=/tmp/r2_smallm_moe_gemm2.cu
OUT=/tmp/r2_smallm_moe_gemm2.so
LOG=/tmp/r2_build.log

echo "=== R2-C M1 build: $SRC -> $OUT ===" | tee "$LOG"

# Detect hipcc
HIPCC=$(which hipcc 2>/dev/null)
if [ -z "$HIPCC" ]; then
    HIPCC=/opt/rocm/bin/hipcc
fi
echo "Using: $HIPCC" | tee -a "$LOG"
$HIPCC --version 2>&1 | head -3 | tee -a "$LOG"

# Build flags
# --offload-arch=gfx950: target MI355X CDNA4
# -O1: avoid LLVM scheduler segfault on mfma_scale interleave (Apr 29 R1 lesson)
# -fPIC -shared: build shared lib for Python ctypes load
# -mllvm flags suppressed (not all are accepted on this hipcc version)
"$HIPCC" \
    --offload-arch=gfx950 \
    -O1 \
    -fPIC -shared \
    -std=c++17 \
    -DHIP_ARCH_GFX950=1 \
    -I/opt/rocm/include \
    -o "$OUT" "$SRC" \
    2>&1 | tee -a "$LOG"

if [ ! -f "$OUT" ]; then
    echo "BUILD FAILED" | tee -a "$LOG"
    exit 1
fi

echo "=== Build OK: $OUT ($(stat -c %s "$OUT") bytes) ===" | tee -a "$LOG"

# Verify gfx950 ISA was emitted (look for v_mfma in disassembly if rocm-objdump available)
if command -v rocm-objdump >/dev/null 2>&1; then
    echo "=== Disasm probe ===" | tee -a "$LOG"
    rocm-objdump -d "$OUT" 2>/dev/null | grep -E "v_mfma|ds_read_b64_tr|amdgcn|gfx950" | head -10 | tee -a "$LOG" || true
fi

echo "M1 build complete." | tee -a "$LOG"

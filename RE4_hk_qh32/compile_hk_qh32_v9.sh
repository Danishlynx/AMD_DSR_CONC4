#!/usr/bin/env bash
# Compile v9 kernel into module_hk_mla.so on the re4c_v8 container.
# Idempotent: skips steps that were already applied.
set -e

CSRC=/app/aiter-test/csrc/kernels/mla
HK=$CSRC/hk
BUILD=/app/aiter-test/aiter/jit/build/module_hk_mla/build

# --- Step 0: backups (already done in session-16 opening, safe to re-run) ---
for f in "$CSRC/hk_decode_fwd.cu" "$HK/hk_mla_buffer_managers.cuh"; do
    if [ ! -f "$f.pre_v9" ]; then
        cp "$f" "$f.pre_v9"
        echo "backed up $f -> $f.pre_v9"
    fi
done

# --- Step 1: drop new v9 kernel header into container tree ---
cp /tmp/v9_deliverables/v9_h32.cuh $HK/mi3xx_v32_fwd_decode_h32_fp8_fp8_v9.cuh
echo "installed v9 kernel header"

# --- Step 2: replace hk_decode_fwd.cu with v9-aware dispatcher ---
cp /tmp/v9_deliverables/hk_decode_fwd_v9.cu $CSRC/hk_decode_fwd.cu
echo "installed v9 dispatcher"

# --- Step 3: patch hk_mla_buffer_managers.cuh with load_k_wide_to_gpr + lds_2_gpr_wide ---
# Only apply if not already applied (idempotent check via presence of function name).
if ! grep -q "load_k_wide_to_gpr" $HK/hk_mla_buffer_managers.cuh; then
    # The patch inserts two functions. We need to find insertion points:
    #   (A) in KvManagerV2, right after load_k_to_gpr (before load_v_to_gpr)
    #   (B) in QManagerV4, right after lds_2_gpr (before get_lds_size_per_block_in_byte)
    # Use python for robust multiline insertion.
    python3 /tmp/v9_deliverables/apply_buffer_managers_patch.py \
        $HK/hk_mla_buffer_managers.cuh \
        /tmp/v9_deliverables/buffer_managers_v9.patch.cuh
    echo "patched buffer_managers (load_k_wide_to_gpr + lds_2_gpr_wide)"
else
    echo "buffer_managers already patched, skipping"
fi

# --- Step 4: clean rebuild of module_hk_mla.so ---
echo "cleaning + ninja rebuild..."
rm -f $BUILD/*.o $BUILD/*.so
(cd $BUILD && ninja 2>&1 | tail -30)

# --- Step 5: install into aiter JIT ---
cp $BUILD/module_hk_mla.so /app/aiter-test/aiter/jit/module_hk_mla.so
echo "installed module_hk_mla.so ($(md5sum /app/aiter-test/aiter/jit/module_hk_mla.so | awk '{print $1}'))"

# --- Step 6: verify v9 symbol linked ---
if nm -D /app/aiter-test/aiter/jit/module_hk_mla.so | grep -q "_v9"; then
    echo "PASS: v9 kernel symbol present"
else
    echo "FAIL: v9 kernel symbol missing"
    exit 1
fi

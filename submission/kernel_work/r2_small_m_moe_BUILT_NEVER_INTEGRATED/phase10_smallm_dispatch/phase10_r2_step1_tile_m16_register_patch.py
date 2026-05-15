#!/usr/bin/env python3
"""Phase 10 R2 Step 1 — register tile_m=16 in flydsl stage 1 MoE configs.

Stage 2 already has tile_m=16 (line 142 of moe_kernels.py: `tile_ms = [16, 32, 64, 128] if is_fp4 else [...]`).
Stage 1 is missing tile_m=16 (line 71: `tile_ms = [32, 64, 128]`). At our M=4 decode case the t32
tile-M dimension wastes 28/32 of MFMA slots in `flydsl_moe1_t32x...`. Adding tile_m=16 lets the
codegen pipeline produce a smaller-tile variant that aligns to the 16x16x128 V_MFMA boundary.

Backwards-compatible: existing tile_m=32+ paths unchanged. AOT precompile picks up tile_m=16
at next install / lazy first-boot.

NULL-OP: no kernels are dispatched to the new tile_m=16 variant unless a dispatcher explicitly
selects it (Step 3). This patch ALONE only adds compile-time variants to the registry.
"""
import os
import py_compile
import sys

TARGET = "/app/aiter-test/aiter/ops/flydsl/moe_kernels.py"
BACKUP = "/app/aiter-test/aiter/ops/flydsl/moe_kernels.py.pre_phase10_r2_step1"
PATCH_MARKER = "# >>> phase10_r2_step1_tile_m16 <<<"

# Anchor: the line `tile_ms = [32, 64, 128]` in get_flydsl_stage1_kernels.
ANCHOR = '''    tile_ns = [32, 64, 128] if is_fp4_b else [128]
    tile_ks = [256]
    tile_ms = [32, 64, 128]'''

REPLACEMENT = '''    tile_ns = [32, 64, 128] if is_fp4_b else [128]
    tile_ks = [256]
    # >>> phase10_r2_step1_tile_m16 <<<
    # R2: register tile_m=16 stage1 variant for DSR1 small-M decode (M=4..16).
    # CDNA4 V_MFMA_F32_16x16x128_F8F6F4 aligns naturally; tile_m=32 wastes 28/32
    # MFMA slots at M=4. The MLIR codegen pipeline produces the kernel
    # automatically when this tile size appears in the registered configs.
    # Backwards-compatible: existing tile_m=32+ unchanged.
    tile_ms = [16, 32, 64, 128]
    # <<< phase10_r2_step1_tile_m16 <<<'''


def main():
    with open(TARGET, "r") as f:
        src = f.read()

    if PATCH_MARKER in src:
        print(f"[phase10_r2_step1] already patched, skipping")
        return 0

    if ANCHOR not in src:
        print(f"[phase10_r2_step1] FATAL: anchor not found in {TARGET}", file=sys.stderr)
        sys.exit(1)
    if src.count(ANCHOR) != 1:
        print(
            f"[phase10_r2_step1] FATAL: anchor matched {src.count(ANCHOR)} times, expected 1",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.exists(BACKUP):
        with open(BACKUP, "w") as f:
            f.write(src)
        print(f"[phase10_r2_step1] backup: {BACKUP}")

    new_src = src.replace(ANCHOR, REPLACEMENT, 1)
    with open(TARGET, "w") as f:
        f.write(new_src)
    print(f"[phase10_r2_step1] wrote {TARGET}")

    try:
        py_compile.compile(TARGET, doraise=True)
        print(f"[phase10_r2_step1] py_compile OK")
    except py_compile.PyCompileError as e:
        print(f"[phase10_r2_step1] FATAL py_compile: {e}", file=sys.stderr)
        with open(BACKUP, "r") as f:
            with open(TARGET, "w") as g:
                g.write(f.read())
        print(f"[phase10_r2_step1] ROLLED BACK", file=sys.stderr)
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())

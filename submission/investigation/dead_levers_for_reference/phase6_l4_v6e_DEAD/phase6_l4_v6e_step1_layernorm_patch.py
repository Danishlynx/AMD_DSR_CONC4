#!/usr/bin/env python3
"""Phase 6 L4 v6e Step 1 — layernorm.py patch — RMSNorm stash producer.

When ATOM_USE_V6E_PREQUANT_STASH=1, the v6c MXFP4 fusion kernel's existing
4-tuple output (out_fp4, residual, scale, unquant_bf16) is captured. The first
two outputs (FP4 packed uint8 + E8M0 scale uint8) are stashed on the RMSNorm
instance as `self._v6e_stash` for the consumer (DeepseekV2DecoderLayer.forward
at deepseek_v2.py:1806) to read.

NULL-OP at default. When env unset, the original v6c return path is unchanged.
"""
import os
import py_compile
import sys

TARGET = "/app/ATOM/atom/model_ops/layernorm.py"
BACKUP = "/app/ATOM/atom/model_ops/layernorm.py.pre_phase6_l4_v6e_step1"
PATCH_MARKER = "# >>> phase6_l4_v6e_stash <<<"

ANCHOR = '''            if _use_mxfp4:
                _x_fp4, residual, _x_scale, unquant_bf16 = (
                    tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant(
                        x.contiguous(),
                        residual,
                        self.weight,
                        self.eps,
                    )
                )
                return unquant_bf16, residual'''

REPLACEMENT = '''            if _use_mxfp4:
                _x_fp4, residual, _x_scale, unquant_bf16 = (
                    tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant(
                        x.contiguous(),
                        residual,
                        self.weight,
                        self.eps,
                    )
                )
                # >>> phase6_l4_v6e_stash <<<
                # v6e: when ATOM_USE_V6E_PREQUANT_STASH=1, the kernel's FP4 +
                # E8M0 scale outputs (already produced, currently discarded) are
                # captured on the RMSNorm instance for the MoE consumer to read.
                # NULL-OP at default: env unset → no stash assigned, behavior
                # bit-identical to v6c.
                _v6e_on = (
                    _os_l2v6.environ.get("ATOM_USE_V6E_PREQUANT_STASH", "0") == "1"
                )
                if _v6e_on:
                    self._v6e_stash = (_x_fp4, _x_scale)
                else:
                    self._v6e_stash = None
                # <<< phase6_l4_v6e_stash <<<
                return unquant_bf16, residual'''


def main():
    with open(TARGET, "r") as f:
        src = f.read()

    if PATCH_MARKER in src:
        print(f"[phase6_l4_step1_ln] already patched, skipping")
        return 0

    if ANCHOR not in src:
        print(f"[phase6_l4_step1_ln] FATAL: anchor not found in {TARGET}", file=sys.stderr)
        sys.exit(1)
    if src.count(ANCHOR) != 1:
        print(
            f"[phase6_l4_step1_ln] FATAL: anchor matched {src.count(ANCHOR)} times, expected 1",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.exists(BACKUP):
        with open(BACKUP, "w") as f:
            f.write(src)
        print(f"[phase6_l4_step1_ln] backup: {BACKUP}")

    new_src = src.replace(ANCHOR, REPLACEMENT, 1)
    with open(TARGET, "w") as f:
        f.write(new_src)
    print(f"[phase6_l4_step1_ln] wrote {TARGET}")

    try:
        py_compile.compile(TARGET, doraise=True)
        print(f"[phase6_l4_step1_ln] py_compile OK")
    except py_compile.PyCompileError as e:
        print(f"[phase6_l4_step1_ln] FATAL py_compile: {e}", file=sys.stderr)
        with open(BACKUP, "r") as f:
            with open(TARGET, "w") as g:
                g.write(f.read())
        print(f"[phase6_l4_step1_ln] ROLLED BACK", file=sys.stderr)
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())

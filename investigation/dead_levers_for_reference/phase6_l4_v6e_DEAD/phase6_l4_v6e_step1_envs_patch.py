#!/usr/bin/env python3
"""Phase 6 L4 v6e Step 1 — envs.py patch — register ATOM_USE_V6E_PREQUANT_STASH.

NULL-OP at default ("0"). When "1" enables the v6e tensor-buffer prequant stash
that plumbs the FP4 + E8M0 outputs already produced by v6c
(tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant) into the MoE input
skip branch at aiter/fused_moe.py:1206-1212.
"""
import os
import py_compile
import sys

TARGET = "/app/ATOM/atom/utils/envs.py"
BACKUP = "/app/ATOM/atom/utils/envs.py.pre_phase6_l4_v6e_step1"
PATCH_MARKER = "# >>> phase6_l4_v6e_prequant_stash <<<"

ANCHOR = '''    "ATOM_USE_POST_ATTN_RMSNORM_MXFP4_FUSION": lambda: os.getenv(
        "ATOM_USE_POST_ATTN_RMSNORM_MXFP4_FUSION", "0"
    )
    == "1",'''

INSERT = '''
    # >>> phase6_l4_v6e_prequant_stash <<<
    # v6e: stash FP4 + E8M0 scale outputs already produced by v6c kernel
    # (tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant) and feed them
    # into the dormant skip branch at aiter/fused_moe.py:1206-1212. Eliminates
    # one BF16 hidden_states HBM round trip + one fused_dynamic_mxfp4_quant_moe_sort
    # kernel launch per layer × 60 layers. Mid-est -0.6 ms TPOT.
    "ATOM_USE_V6E_PREQUANT_STASH": lambda: os.getenv(
        "ATOM_USE_V6E_PREQUANT_STASH", "0"
    )
    == "1",
    # <<< phase6_l4_v6e_prequant_stash <<<'''


def main():
    with open(TARGET, "r") as f:
        src = f.read()

    if PATCH_MARKER in src:
        print(f"[phase6_l4_step1_envs] already patched, skipping")
        return 0

    if ANCHOR not in src:
        print(f"[phase6_l4_step1_envs] FATAL: anchor not found in {TARGET}", file=sys.stderr)
        sys.exit(1)
    if src.count(ANCHOR) != 1:
        print(
            f"[phase6_l4_step1_envs] FATAL: anchor matched {src.count(ANCHOR)} times, expected 1",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.exists(BACKUP):
        with open(BACKUP, "w") as f:
            f.write(src)
        print(f"[phase6_l4_step1_envs] backup: {BACKUP}")

    new_src = src.replace(ANCHOR, ANCHOR + INSERT, 1)
    with open(TARGET, "w") as f:
        f.write(new_src)
    print(f"[phase6_l4_step1_envs] wrote {TARGET}")

    try:
        py_compile.compile(TARGET, doraise=True)
        print(f"[phase6_l4_step1_envs] py_compile OK")
    except py_compile.PyCompileError as e:
        print(f"[phase6_l4_step1_envs] FATAL py_compile: {e}", file=sys.stderr)
        with open(BACKUP, "r") as f:
            with open(TARGET, "w") as g:
                g.write(f.read())
        print(f"[phase6_l4_step1_envs] ROLLED BACK", file=sys.stderr)
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Phase 6 L4 v6e Step 2 — DecoderLayer.forward bridge patch.

The Mxfp4MoEMethod.apply consumer at moe.py:1043-1054 (already shipped Apr 27
"L2 v5 Phase 2 prequant branch") reads `layer._prequantized_input` to skip the
fused_dynamic_mxfp4_quant_moe_sort kernel + BF16 hidden_states HBM round trip,
hitting the dormant skip path at aiter/fused_moe.py:1206-1212.

Step 1 (already applied) makes RMSNorm.forward stash `(_x_fp4, _x_scale)` on
self._v6e_stash when ATOM_USE_V6E_PREQUANT_STASH=1.

This Step 2 is the bridge: in DeepseekV2DecoderLayer.forward, after the
post_attention_layernorm call returns, propagate the stash from the RMSNorm
instance into self.mlp.experts._prequantized_input where the existing consumer
will pick it up.

NULL-OP at default. Multi-guarded: only fires on MoE layers (hasattr experts),
only when env=1, only when stash actually populated (skips boot warmup).
"""
import os
import py_compile
import sys

TARGET = "/app/ATOM/atom/models/deepseek_v2.py"
BACKUP = "/app/ATOM/atom/models/deepseek_v2.py.pre_phase6_l4_v6e_step2"
PATCH_MARKER = "# >>> phase6_l4_v6e_bridge <<<"

ANCHOR = '''        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)'''

REPLACEMENT = '''        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # >>> phase6_l4_v6e_bridge <<<
        # v6e: bridge RMSNorm._v6e_stash → FusedMoE._prequantized_input so the
        # Mxfp4MoEMethod.apply skip branch (moe.py:1043-1054) consumes the FP4
        # outputs already produced by the fused AR+RMSNorm+MXFP4 kernel,
        # eliminating fused_dynamic_mxfp4_quant_moe_sort + BF16 hidden_states
        # HBM round trip per layer.
        import os as _os_l4v6e
        if (
            _os_l4v6e.environ.get("ATOM_USE_V6E_PREQUANT_STASH", "0") == "1"
            and hasattr(self.mlp, "experts")
        ):
            _v6e_stash = getattr(self.post_attention_layernorm, "_v6e_stash", None)
            if _v6e_stash is not None:
                self.mlp.experts._prequantized_input = _v6e_stash
            # do NOT reset to None here — consumer at moe.py:1055 sets it back
            # to None after one-shot consume, preserving v6c behavior on
            # subsequent layers without v6e fusion (e.g. dense layers).
        # <<< phase6_l4_v6e_bridge <<<
        hidden_states = self.mlp(hidden_states)'''


def main():
    with open(TARGET, "r") as f:
        src = f.read()

    if PATCH_MARKER in src:
        print(f"[phase6_l4_step2_bridge] already patched, skipping")
        return 0

    if ANCHOR not in src:
        print(f"[phase6_l4_step2_bridge] FATAL: anchor not found in {TARGET}", file=sys.stderr)
        sys.exit(1)
    if src.count(ANCHOR) != 1:
        print(
            f"[phase6_l4_step2_bridge] FATAL: anchor matched {src.count(ANCHOR)} times, expected 1",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.exists(BACKUP):
        with open(BACKUP, "w") as f:
            f.write(src)
        print(f"[phase6_l4_step2_bridge] backup: {BACKUP}")

    new_src = src.replace(ANCHOR, REPLACEMENT, 1)
    with open(TARGET, "w") as f:
        f.write(new_src)
    print(f"[phase6_l4_step2_bridge] wrote {TARGET}")

    try:
        py_compile.compile(TARGET, doraise=True)
        print(f"[phase6_l4_step2_bridge] py_compile OK")
    except py_compile.PyCompileError as e:
        print(f"[phase6_l4_step2_bridge] FATAL py_compile: {e}", file=sys.stderr)
        with open(BACKUP, "r") as f:
            with open(TARGET, "w") as g:
                g.write(f.read())
        print(f"[phase6_l4_step2_bridge] ROLLED BACK", file=sys.stderr)
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())

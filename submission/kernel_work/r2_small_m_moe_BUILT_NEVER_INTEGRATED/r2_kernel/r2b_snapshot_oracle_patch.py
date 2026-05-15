#!/usr/bin/env python3
# Extend the M=4 snapshot to ALSO save the aiter.fused_moe(...) reference output.
# This becomes the M2 numerics-parity oracle.
# One-shot, env-gated via ATOM_R2_SNAPSHOT_TENSORS=1 (already in place from R2-B).
# Idempotent: skips if /tmp/r2_snapshot/out_ref.pt exists.
import py_compile, shutil, sys

MOE_PATH = "/app/ATOM/atom/model_ops/moe.py"
BAK = MOE_PATH + ".pre_r2b_oracle"
shutil.copyfile(MOE_PATH, BAK)
src = open(MOE_PATH).read()

# Anchor: the existing R2B_SNAPSHOT_TENSORS block ended with `_r2b_snapshot_done = True`.
# We want to capture the OUTPUT of fused_moe right AFTER it returns.
# The existing fused_moe call is at moe.py:1107 inside `Mxfp4MoEMethod.apply`.
# We'll wrap it: capture output before return, save when our snapshot fires.

old = """        if self.fused_experts is None:
            return fused_moe("""

new = """        if self.fused_experts is None:
            # >>> R2B_ORACLE_CAPTURE <<<
            # If snapshot fired this layer (M=4 path), call fused_moe and save output.
            import os as _os_r2b_o
            if (_os_r2b_o.environ.get("ATOM_R2_SNAPSHOT_TENSORS", "0") == "1"
                and getattr(layer, "_r2b_snapshot_done", False)
                and not getattr(layer, "_r2b_oracle_done", False)):
                _r2b_out = fused_moe(
                    x, layer.w13_weight, layer.w2_weight,
                    topk_weights, topk_ids,
                    expert_mask=expert_map, activation=activation,
                    quant_type=self.quant_type,
                    w1_scale=layer.w13_weight_scale,
                    w2_scale=layer.w2_weight_scale,
                    a1_scale=a1_scale, a2_scale=a2_scale,
                    doweight_stage1=apply_router_weight_on_input,
                    hidden_pad=self.hidden_pad,
                    intermediate_pad=self.intermediate_pad,
                    bias1=layer.w13_bias, bias2=layer.w2_bias,
                )
                import torch as _r2b_torch_o
                _os_r2b_o.makedirs("/tmp/r2_snapshot", exist_ok=True)
                _r2b_torch_o.save(_r2b_out.clone().contiguous().cpu(),
                                  "/tmp/r2_snapshot/out_ref.pt")
                with open("/tmp/r2_snapshot/oracle_meta.txt", "w") as _f:
                    _f.write(f"out.shape={tuple(_r2b_out.shape)}\\n")
                    _f.write(f"out.dtype={_r2b_out.dtype}\\n")
                    _f.write(f"out.abs().max()={float(_r2b_out.abs().max()):.6e}\\n")
                    _f.write(f"out.abs().mean()={float(_r2b_out.abs().mean()):.6e}\\n")
                layer._r2b_oracle_done = True
                print(f"[R2B-ORACLE] saved aiter.fused_moe output ref to /tmp/r2_snapshot/out_ref.pt", flush=True)
                return _r2b_out
            # <<< R2B_ORACLE_CAPTURE <<<
            return fused_moe("""

if old not in src:
    sys.exit("ERR: anchor not found")

src2 = src.replace(old, new, 1)
open(MOE_PATH, "w").write(src2)
py_compile.compile(MOE_PATH, doraise=True)
print(f"OK R2-B oracle patch applied")
print(f"Backup: {BAK}")

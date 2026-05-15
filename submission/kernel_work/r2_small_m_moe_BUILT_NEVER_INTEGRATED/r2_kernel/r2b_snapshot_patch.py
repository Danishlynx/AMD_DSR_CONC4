#!/usr/bin/env python3
# R2-B diagnostic patch: snapshot real input tensors at the first M=4 MoE call.
# One-shot: writes tensors to /tmp/r2_snapshot/ then never fires again.
# Env-gated by ATOM_R2_SNAPSHOT_TENSORS=1, NULL-OP otherwise.
import py_compile, shutil, sys

MOE_PATH = "/app/ATOM/atom/model_ops/moe.py"
BAK = MOE_PATH + ".pre_r2b_snapshot"
shutil.copyfile(MOE_PATH, BAK)
src = open(MOE_PATH).read()

# Find the entry of Mxfp4MoEMethod.apply where x is the hidden_states. Anchor on
# the existing Apr 27 _prequant comment so we land in the right spot.
anchor = '        # Lever 2 v5 (Apr 27): Phase 2 prequant branch.'
if anchor not in src:
    sys.exit("ERR: anchor not found")

snap_block = """        # >>> R2B_SNAPSHOT_TENSORS <<<
        # Diagnostic one-shot: capture real M=4 inputs at first decode step.
        import os as _os_r2b
        if _os_r2b.environ.get("ATOM_R2_SNAPSHOT_TENSORS", "0") == "1":
            _r2b_done = getattr(layer, "_r2b_snapshot_done", False)
            _r2b_target_M = 4
            if not _r2b_done and x.shape[0] == _r2b_target_M:
                import torch as _r2b_torch
                _r2b_dir = "/tmp/r2_snapshot"
                _os_r2b.makedirs(_r2b_dir, exist_ok=True)
                # Save the tensors. Use .clone().contiguous().cpu() to detach
                # from any cudagraph-captured lifetime.
                def _save(name, t):
                    if t is None:
                        return
                    _r2b_torch.save(t.clone().contiguous().cpu(), f"{_r2b_dir}/{name}.pt")
                _save("hidden_states_M4", x)
                _save("topk_weights", topk_weights)
                _save("topk_ids", topk_ids)
                _save("expert_map", expert_map)
                _save("w13_weight", layer.w13_weight)
                _save("w2_weight", layer.w2_weight)
                _save("w13_weight_scale", layer.w13_weight_scale)
                _save("w2_weight_scale", layer.w2_weight_scale)
                _save("w13_input_scale", a1_scale)
                _save("w2_input_scale", a2_scale)
                _save("w13_bias", layer.w13_bias if hasattr(layer, "w13_bias") else None)
                _save("w2_bias", layer.w2_bias if hasattr(layer, "w2_bias") else None)
                # Save metadata
                with open(f"{_r2b_dir}/meta.txt", "w") as _f:
                    _f.write(f"M={x.shape[0]}\\n")
                    _f.write(f"hidden_size={x.shape[1] if x.ndim > 1 else 'NA'}\\n")
                    _f.write(f"x.dtype={x.dtype}\\n")
                    _f.write(f"x.shape={tuple(x.shape)}\\n")
                    _f.write(f"w13_weight.shape={tuple(layer.w13_weight.shape)}\\n")
                    _f.write(f"w13_weight.dtype={layer.w13_weight.dtype}\\n")
                    _f.write(f"w2_weight.shape={tuple(layer.w2_weight.shape)}\\n")
                    _f.write(f"w13_weight_scale.shape={tuple(layer.w13_weight_scale.shape)}\\n")
                    _f.write(f"topk_ids.shape={tuple(topk_ids.shape)}\\n")
                    _f.write(f"topk_weights.shape={tuple(topk_weights.shape)}\\n")
                    _f.write(f"quant_type={self.quant_type}\\n")
                    _f.write(f"hidden_pad={self.hidden_pad}\\n")
                    _f.write(f"intermediate_pad={self.intermediate_pad}\\n")
                    _f.write(f"num_experts={self.num_experts}\\n")
                    _f.write(f"layer_idx_clue={getattr(layer, 'layer_idx', '?')}\\n")
                layer._r2b_snapshot_done = True
                print(f"[R2B-SNAPSHOT] saved tensors at M={x.shape[0]} to {_r2b_dir}/", flush=True)
        # <<< R2B_SNAPSHOT_TENSORS <<<
        # Lever 2 v5 (Apr 27): Phase 2 prequant branch."""

src2 = src.replace(anchor, snap_block, 1)
if src2 == src:
    sys.exit("ERR: snap_block insertion failed")

open(MOE_PATH, "w").write(src2)
py_compile.compile(MOE_PATH, doraise=True)
print("OK R2-B snapshot patch applied")
print(f"Backup: {BAK}")

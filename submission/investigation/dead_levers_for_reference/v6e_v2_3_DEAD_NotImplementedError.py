#!/usr/bin/env python3
# v6e v2.3 BufferRing + zeros-init + dtype kwarg fix (combined patch)
# Stacks on top of Phase 11 v3 (no conflict — Phase 11 patches different code paths).
# Uses torch.zeros() instead of torch.empty() to avoid HSA exception 0x1016
# from uninitialized E8M0 scale (NaN -> MFMA fault).
import py_compile, shutil, sys

# ============ FILE 1: moe.py ============
MOE_PATH = "/app/ATOM/atom/model_ops/moe.py"
MOE_BAK = MOE_PATH + ".pre_v6e_v2_3"
shutil.copyfile(MOE_PATH, MOE_BAK)
moe_src = open(MOE_PATH).read()

# 1a: Allocate persistent buffers in create_weights (zeros, not empty)
moe_anchor1 = "        self.hidden_pad = self.hidden_size - layer.hidden_size"
if moe_anchor1 not in moe_src:
    sys.exit("ERR: moe anchor1 not found")

moe_alloc_block = """        self.hidden_pad = self.hidden_size - layer.hidden_size
        # >>> v6e_v2_3_persistent_prequant_zeros <<<
        # Pre-allocate persistent (fp4_buf, scale_buf) tuple as layer._prequantized_input.
        # zeros() not empty() to avoid HSA exception from uninitialized E8M0 scale.
        # NULL-OP at default (env unset).
        import os as _os_v6e_a
        if _os_v6e_a.environ.get("ATOM_USE_V6E_PREQUANT_STASH", "0") == "1":
            _v6e_max_tokens = 256
            _fp4_buf = torch.zeros(
                _v6e_max_tokens, hidden_size // 2,
                dtype=torch.float4_e2m1fn_x2, device="cuda",
            )
            _scale_buf = torch.zeros(
                _v6e_max_tokens, hidden_size // 32,
                dtype=torch.uint8, device="cuda",
            )
            layer._v6e_fp4_buf = _fp4_buf
            layer._v6e_scale_buf = _scale_buf
            layer._prequantized_input = (_fp4_buf, _scale_buf)
        # <<< v6e_v2_3_persistent_prequant_zeros <<<
"""
moe_src2 = moe_src.replace(moe_anchor1 + "\n", moe_alloc_block, 1)
if moe_src2 == moe_src: sys.exit("ERR: moe anchor1 replace failed")

# 1b: Consumer + dtype kwarg fix at line ~1043
moe_anchor2 = """        _prequant = getattr(layer, "_prequantized_input", None)
        if (
            _prequant is not None
            and not getattr(self, "use_triton", False)
            and getattr(layer, "dp_size", 1) == 1
        ):
            _q_fp4, _raw_scale = _prequant
            x = _q_fp4
            a1_scale = _raw_scale
            layer._prequantized_input = None  # one-shot consume"""

moe_consumer_v2_3 = """        # >>> v6e_v2_3_consumer <<<
        # Read persistent _prequantized_input tuple. Slice [:bs]. NO setattr to None.
        _v6e_out_dtype = None
        _prequant = getattr(layer, "_prequantized_input", None)
        if (
            _prequant is not None
            and not getattr(self, "use_triton", False)
            and getattr(layer, "dp_size", 1) == 1
        ):
            _v6e_out_dtype = x.dtype  # bf16 - pass to fused_moe as dtype kwarg
            _q_fp4, _raw_scale = _prequant
            _bs_v6e = x.shape[0]
            x = _q_fp4[:_bs_v6e]
            a1_scale = _raw_scale[:_bs_v6e]
        # <<< v6e_v2_3_consumer <<<"""

moe_src3 = moe_src2.replace(moe_anchor2, moe_consumer_v2_3, 1)
if moe_src3 == moe_src2: sys.exit("ERR: moe consumer anchor not matched")

# 1c: Add dtype kwarg to fused_moe call
fused_moe_anchor = """        if self.fused_experts is None:
            return fused_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                expert_mask=expert_map,
                activation=activation,
                quant_type=self.quant_type,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                doweight_stage1=apply_router_weight_on_input,
                hidden_pad=self.hidden_pad,
                intermediate_pad=self.intermediate_pad,
                bias1=layer.w13_bias,
                bias2=layer.w2_bias,
            )"""

fused_moe_v23 = """        if self.fused_experts is None:
            return fused_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                expert_mask=expert_map,
                activation=activation,
                quant_type=self.quant_type,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                doweight_stage1=apply_router_weight_on_input,
                hidden_pad=self.hidden_pad,
                intermediate_pad=self.intermediate_pad,
                bias1=layer.w13_bias,
                bias2=layer.w2_bias,
                dtype=_v6e_out_dtype,
            )"""

moe_src4 = moe_src3.replace(fused_moe_anchor, fused_moe_v23, 1)
if moe_src4 == moe_src3: sys.exit("ERR: fused_moe call anchor not matched")

open(MOE_PATH, "w").write(moe_src4)
py_compile.compile(MOE_PATH, doraise=True)
print("OK moe.py: " + str(len(moe_src)) + " -> " + str(len(moe_src4)))

# ============ FILE 2: layernorm.py - producer copy_ ============
LN_PATH = "/app/ATOM/atom/model_ops/layernorm.py"
LN_BAK = LN_PATH + ".pre_v6e_v2_3"
shutil.copyfile(LN_PATH, LN_BAK)
ln_src = open(LN_PATH).read()

ln_anchor = """                _v6e_on = (
                    _os_l2v6.environ.get("ATOM_USE_V6E_PREQUANT_STASH", "0") == "1"
                )
                if _v6e_on:
                    self._v6e_stash = (_x_fp4, _x_scale)
                else:
                    self._v6e_stash = None"""

ln_v23 = """                _v6e_on = (
                    _os_l2v6.environ.get("ATOM_USE_V6E_PREQUANT_STASH", "0") == "1"
                )
                if _v6e_on and hasattr(self, "_v6e_fp4_buf"):
                    _bs_v6e = _x_fp4.shape[0]
                    self._v6e_fp4_buf[:_bs_v6e].copy_(_x_fp4)
                    self._v6e_scale_buf[:_bs_v6e].copy_(_x_scale)
                self._v6e_stash = None"""

ln_src2 = ln_src.replace(ln_anchor, ln_v23, 1)
if ln_src2 == ln_src: sys.exit("ERR: layernorm anchor not matched")

open(LN_PATH, "w").write(ln_src2)
py_compile.compile(LN_PATH, doraise=True)
print("OK layernorm.py: " + str(len(ln_src)) + " -> " + str(len(ln_src2)))

# ============ FILE 3: deepseek_v2.py - drop bridge + add init alias ============
DS_PATH = "/app/ATOM/atom/models/deepseek_v2.py"
DS_BAK = DS_PATH + ".pre_v6e_v2_3"
shutil.copyfile(DS_PATH, DS_BAK)
ds_src = open(DS_PATH).read()

ds_anchor = """        import os as _os_l4v6e
        if (
            _os_l4v6e.environ.get("ATOM_USE_V6E_PREQUANT_STASH", "0") == "1"
            and hasattr(self.mlp, "experts")
        ):
            _v6e_stash = getattr(self.post_attention_layernorm, "_v6e_stash", None)
            if _v6e_stash is not None:
                self.mlp.experts._prequantized_input = _v6e_stash"""

ds_v23 = """        if False and hasattr(self.mlp, "experts"):
            pass"""

ds_src2 = ds_src.replace(ds_anchor, ds_v23, 1)
if ds_src2 == ds_src: sys.exit("ERR: deepseek_v2 bridge anchor not matched")

ds_init_end_anchor = "        self.routed_scaling_factor = config.routed_scaling_factor"
if ds_init_end_anchor not in ds_src2:
    ds_init_end_anchor = "        self.layer_idx = layer_idx"
    if ds_init_end_anchor not in ds_src2:
        sys.exit("ERR: ds init anchor not found")

ds_alias_block = ds_init_end_anchor + """

        # >>> v6e_v2_3_alias <<<
        import os as _os_v6e_alias
        if (
            _os_v6e_alias.environ.get("ATOM_USE_V6E_PREQUANT_STASH", "0") == "1"
            and hasattr(self, "mlp")
            and hasattr(self.mlp, "experts")
            and hasattr(self.mlp.experts, "_v6e_fp4_buf")
        ):
            self.post_attention_layernorm._v6e_fp4_buf = self.mlp.experts._v6e_fp4_buf
            self.post_attention_layernorm._v6e_scale_buf = self.mlp.experts._v6e_scale_buf
        # <<< v6e_v2_3_alias <<<"""

ds_src3 = ds_src2.replace(ds_init_end_anchor, ds_alias_block, 1)
if ds_src3 == ds_src2: sys.exit("ERR: ds alias anchor not matched")

open(DS_PATH, "w").write(ds_src3)
py_compile.compile(DS_PATH, doraise=True)
print("OK deepseek_v2.py: " + str(len(ds_src)) + " -> " + str(len(ds_src3)))

print("\nALL 3 FILES PATCHED v2.3 (zeros-init, dtype kwarg, py_compile clean)")
print("Backups: " + MOE_BAK + ", " + LN_BAK + ", " + DS_BAK)

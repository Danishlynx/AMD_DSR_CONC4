#!/usr/bin/env python3
"""Phase 2f: add Python plumbing for fused_allreduce_rmsnorm_mxfp4_quant.

Mirrors the existing FP8 chain across 4 files:
1. /app/aiter-test/aiter/dist/device_communicators/custom_all_reduce.py
2. /app/aiter-test/aiter/dist/device_communicators/communicator_cuda.py
3. /app/aiter-test/aiter/dist/parallel_state.py
4. /app/aiter-test/aiter/dist/communication_op.py
"""
import os, sys, shutil

# ============== File 1: custom_all_reduce.py ==============
F1 = "/app/aiter-test/aiter/dist/device_communicators/custom_all_reduce.py"

# Backup
if not os.path.exists(F1 + ".pre_phase2"):
    shutil.copy(F1, F1 + ".pre_phase2")

src1 = open(F1).read()
ORIG1 = src1

# Add post_mxfp4_group_quant flag to fused_ar_rms (extend) and add custom_fused_ar_rms_mxfp4_quant.
# Find the existing else: at the end of fused_ar_rms (the one after the else: that handles FP8 quant)
# and inject new branch at end of fused_ar_rms; then add new method.
ANCHOR_F1 = """    def custom_fused_ar_rms(
        self,
        input: torch.Tensor,
        residual_inp: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        use_1stage: bool,
    ) -> Optional[torch.Tensor]:"""

INSERT_F1 = """    def fused_ar_rms_mxfp4_quant(
        self,
        inp: torch.Tensor,
        res_inp: torch.Tensor,
        w: torch.Tensor,
        eps: float,
        registered: bool,
        use_1stage: bool,
    ):
        \"\"\"MXFP4 (per-32-group e8m0) variant of fused_ar_rms with post quantization.
        out: uint8 packed FP4 (1 byte = 2 fp4 elements), shape = inp.shape but last-dim halved.
        scale_out: uint8 e8m0 scale, shape = inp.shape[:-1] + (hidden//32,).
        \"\"\"
        from ...jit.utils.chip_info import get_gfx
        # Output: pack 2 fp4 per byte. Shape is (..., hidden//2) as uint8.
        last_dim = inp.shape[-1]
        assert last_dim % 32 == 0, f"hidden={last_dim} must be %32 for MXFP4"
        # Allocate FP4 packed output: shape = inp.shape with last-dim halved, dtype=uint8.
        out_shape = inp.shape[:-1] + (last_dim // 2,)
        out = torch.empty(out_shape, dtype=torch.uint8, device=inp.device)
        # Allocate e8m0 scale: shape = inp.shape[:-1] + (hidden//32,), dtype=uint8.
        scale_shape = inp.shape[:-1] + (last_dim // 32,)
        scale_out = torch.empty(scale_shape, dtype=torch.uint8, device=inp.device)
        # Allocate residual_out (BF16, same as input).
        res_out = torch.empty_like(inp)

        reg = 0 if not registered else self._pool["input"].data_ptr
        reg_bytes = 0 if not registered else self._pool["input"].max_size

        ops.fused_allreduce_rmsnorm_mxfp4_quant(
            self._ptr,
            inp,
            res_inp,
            res_out,
            out,
            scale_out,
            w,
            eps,
            reg,
            reg_bytes,
            use_1stage,
        )
        return out, res_out, scale_out

    def custom_fused_ar_rms_mxfp4_quant(
        self,
        input: torch.Tensor,
        residual_inp: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        use_1stage: bool,
    ):
        \"\"\"Public entry from communicator_cuda.py for MXFP4 path.\"\"\"
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.fused_ar_rms_mxfp4_quant(
                    input, residual_inp, w=weight, eps=eps,
                    registered=True, use_1stage=use_1stage,
                )
            else:
                # Eager mode under capturing flag: dummy outputs for cudagraph trace.
                last_dim = input.shape[-1]
                dummy_out = torch.zeros(
                    input.shape[:-1] + (last_dim // 2,), dtype=torch.uint8, device=input.device)
                dummy_scale = torch.zeros(
                    input.shape[:-1] + (last_dim // 32,), dtype=torch.uint8, device=input.device)
                return dummy_out, torch.zeros_like(input), dummy_scale
        else:
            return self.fused_ar_rms_mxfp4_quant(
                input, residual_inp, w=weight, eps=eps,
                registered=False, use_1stage=use_1stage,
            )

    def custom_fused_ar_rms(
        self,
        input: torch.Tensor,
        residual_inp: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        use_1stage: bool,
    ) -> Optional[torch.Tensor]:"""

if "fused_ar_rms_mxfp4_quant" not in src1:
    assert ANCHOR_F1 in src1, "ANCHOR_F1 not found"
    src1 = src1.replace(ANCHOR_F1, INSERT_F1, 1)
    open(F1, 'w').write(src1)
    print(f"PATCHED {F1}")

# ============== File 2: communicator_cuda.py ==============
F2 = "/app/aiter-test/aiter/dist/device_communicators/communicator_cuda.py"
if not os.path.exists(F2 + ".pre_phase2"):
    shutil.copy(F2, F2 + ".pre_phase2")
src2 = open(F2).read()
ORIG2 = src2

ANCHOR_F2 = """    def fused_allreduce_rmsnorm_quant("""
INSERT_F2 = """    def fused_allreduce_rmsnorm_mxfp4_quant(
        self,
        input_,
        res_inp_,
        weight_,
        eps,
        prefill_support: bool = False,
    ):
        \"\"\"MXFP4 path: input bf16 -> AR + RMSNorm + per-32-group FP4 quant.
        Returns (out_uint8, res_out_bf16, scale_out_uint8).
        \"\"\"
        total_bytes = input_.numel() * input_.element_size()
        if (
            int(input_.shape[-1]) in [512, 1024, 2048, 4096]
            and total_bytes <= 4096 * 1024
            and (prefill_support or total_bytes <= 64 * 1024 * 1024)
        ):
            use_1stage = (
                self._ar_1stage_override
                if self._ar_1stage_override is not None
                else (total_bytes <= 128 * 1024)
            )
            res = self.ca_comm.custom_fused_ar_rms_mxfp4_quant(
                input_, res_inp_, weight_, eps, use_1stage
            )
            assert res is not None, "Phase2 MXFP4 fast-path missing (custom AR disabled?)"
            return res
        else:
            # Fallback: AR alone then external triton MXFP4 quant. Not used in DSR1 typical hidden=7168 path.
            raise NotImplementedError(
                f"Phase2 MXFP4 fallback path: hidden_dim={input_.shape[-1]} not in fast-path list"
            )

    def fused_allreduce_rmsnorm_quant("""

if "fused_allreduce_rmsnorm_mxfp4_quant" not in src2:
    assert ANCHOR_F2 in src2, "ANCHOR_F2 not found"
    src2 = src2.replace(ANCHOR_F2, INSERT_F2, 1)
    open(F2, 'w').write(src2)
    print(f"PATCHED {F2}")

# ============== File 3: parallel_state.py ==============
F3 = "/app/aiter-test/aiter/dist/parallel_state.py"
if not os.path.exists(F3 + ".pre_phase2"):
    shutil.copy(F3, F3 + ".pre_phase2")
src3 = open(F3).read()
ORIG3 = src3

# Add 4 things to parallel_state.py:
# - fake/real custom op pair
# - group method
# - _out_place method

ANCHOR_F3a = """def fused_allreduce_rmsnorm_quant_fake("""
INSERT_F3a = """def fused_allreduce_rmsnorm_mxfp4_quant_fake(
    inp: torch.Tensor,
    res_inp: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    last_dim = inp.shape[-1]
    return (
        torch.empty(inp.shape[:-1] + (last_dim // 2,), dtype=torch.uint8, device=inp.device),
        torch.empty_like(inp),
        torch.empty(inp.shape[:-1] + (last_dim // 32,), dtype=torch.uint8, device=inp.device),
    )


@torch_compile_guard(gen_fake=fused_allreduce_rmsnorm_mxfp4_quant_fake)
def fused_allreduce_rmsnorm_mxfp4_quant_(
    inp: torch.Tensor,
    res_inp: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    group_name: str,
    prefill_support: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    return group._fused_allreduce_rmsnorm_mxfp4_quant_out_place(
        inp, res_inp, w, eps, prefill_support
    )


def fused_allreduce_rmsnorm_quant_fake("""

if "fused_allreduce_rmsnorm_mxfp4_quant_fake" not in src3:
    assert ANCHOR_F3a in src3, "ANCHOR_F3a not found"
    src3 = src3.replace(ANCHOR_F3a, INSERT_F3a, 1)
    print(f"  +fake/real op")

ANCHOR_F3b = """    def fused_allreduce_rmsnorm_quant(
        self,
        input_: torch.Tensor,
        residual_inp_: torch.Tensor,
        weight_: torch.Tensor,
        eps: float,
        prefill_support: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return fused_allreduce_rmsnorm_quant_("""

INSERT_F3b = """    def fused_allreduce_rmsnorm_mxfp4_quant(
        self,
        input_: torch.Tensor,
        residual_inp_: torch.Tensor,
        weight_: torch.Tensor,
        eps: float,
        prefill_support: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return fused_allreduce_rmsnorm_mxfp4_quant_(
            input_,
            residual_inp_,
            weight_,
            eps,
            group_name=self.unique_name,
            prefill_support=prefill_support,
        )

    def fused_allreduce_rmsnorm_quant(
        self,
        input_: torch.Tensor,
        residual_inp_: torch.Tensor,
        weight_: torch.Tensor,
        eps: float,
        prefill_support: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return fused_allreduce_rmsnorm_quant_("""

if "def fused_allreduce_rmsnorm_mxfp4_quant(" not in src3:
    assert ANCHOR_F3b in src3, "ANCHOR_F3b not found"
    src3 = src3.replace(ANCHOR_F3b, INSERT_F3b, 1)
    print(f"  +group method")

ANCHOR_F3c = """    def _fused_allreduce_rmsnorm_quant_out_place("""
INSERT_F3c = """    def _fused_allreduce_rmsnorm_mxfp4_quant_out_place(
        self,
        input_: torch.Tensor,
        residual_inp_: torch.Tensor,
        weight_: torch.Tensor,
        eps: float,
        prefill_support: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        return self.device_communicator.fused_allreduce_rmsnorm_mxfp4_quant(
            input_,
            residual_inp_,
            weight_,
            eps,
            prefill_support,
        )

    def _fused_allreduce_rmsnorm_quant_out_place("""

if "_fused_allreduce_rmsnorm_mxfp4_quant_out_place" not in src3:
    assert ANCHOR_F3c in src3, "ANCHOR_F3c not found"
    src3 = src3.replace(ANCHOR_F3c, INSERT_F3c, 1)
    print(f"  +out_place method")

if src3 != ORIG3:
    open(F3, 'w').write(src3)
    print(f"PATCHED {F3}")

# ============== File 4: communication_op.py ==============
F4 = "/app/aiter-test/aiter/dist/communication_op.py"
if not os.path.exists(F4 + ".pre_phase2"):
    shutil.copy(F4, F4 + ".pre_phase2")
src4 = open(F4).read()
ORIG4 = src4

ANCHOR_F4 = """def tensor_model_parallel_fused_allreduce_rmsnorm_quant("""
INSERT_F4 = """def tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant(
    input_: torch.Tensor,
    residual_inp_: torch.Tensor,
    weight_: torch.Tensor,
    eps: float,
    prefill_support: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    \"\"\"Phase 2 fused AR + RMSNorm + MXFP4 (per-32-group e8m0) quantization.
    Returns (out_fp4_packed_uint8, res_out, scale_e8m0_uint8).\"\"\"
    return get_tp_group().fused_allreduce_rmsnorm_mxfp4_quant(
        input_, residual_inp_, weight_, eps, prefill_support
    )


def tensor_model_parallel_fused_allreduce_rmsnorm_quant("""

if "tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant" not in src4:
    assert ANCHOR_F4 in src4, "ANCHOR_F4 not found"
    src4 = src4.replace(ANCHOR_F4, INSERT_F4, 1)
    open(F4, 'w').write(src4)
    print(f"PATCHED {F4}")

print("=== Phase 2f Python plumbing complete ===")

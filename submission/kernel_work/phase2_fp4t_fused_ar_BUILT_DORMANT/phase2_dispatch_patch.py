#!/usr/bin/env python3
"""Phase 2d: add MXFP4 dispatch path to custom_all_reduce host code.

Touches 3 files atomically:
1. /app/aiter-test/csrc/kernels/custom_all_reduce.cu - new entry fn + helper
2. /app/aiter-test/csrc/include/custom_all_reduce.h  - declaration
3. /app/aiter-test/csrc/include/rocm_ops.hpp         - pybind binding
"""
import sys, os

CU_PATH  = "/app/aiter-test/csrc/kernels/custom_all_reduce.cu"
H_PATH   = "/app/aiter-test/csrc/include/custom_all_reduce.h"
PYB_PATH = "/app/aiter-test/csrc/include/rocm_ops.hpp"

# ============== PATCH 1: custom_all_reduce.cu ==============
cu = open(CU_PATH).read()
CU_ORIG = cu

# 1a) Insert _fused_allreduce_rmsnorm_mxfp4 BEFORE existing _fused_allreduce_rmsnorm
ANCHOR_CU_HELPER = """static void _fused_allreduce_rmsnorm(fptr_t _fa,"""

INSERT_CU_HELPER = """// Phase 2: MXFP4 (per-32 e8m0) variant of _fused_allreduce_rmsnorm.
// Uses ar_fusion_epilogue<..., opus::fp4_t> branch added in custom_all_reduce.cuh.
static void _fused_allreduce_rmsnorm_mxfp4(fptr_t _fa,
                                           void* inp, void* residual_inp,
                                           void* residual_out, void* out,
                                           void* scale_out, void* w,
                                           AiterDtype dtype, float eps,
                                           int m, int n,
                                           bool use_1stage)
{
    hipStream_t stream = aiter::getCurrentHIPStream();
    auto fa = reinterpret_cast<aiter::CustomAllreduce*>(_fa);

#define DISPATCH_AR_FUSION_MXFP4(DTYPE)                                  \\
    fa->dispatchFusedAllReduceRMSNormQuant<DTYPE, opus::fp4_t>(          \\
        stream,                                                          \\
        reinterpret_cast<DTYPE*>(inp),                                   \\
        reinterpret_cast<DTYPE*>(residual_inp),                          \\
        reinterpret_cast<DTYPE*>(residual_out),                          \\
        reinterpret_cast<opus::fp4_t*>(out),                             \\
        reinterpret_cast<float*>(scale_out),                             \\
        reinterpret_cast<DTYPE*>(w),                                     \\
        eps, m, n, use_1stage);

    switch(dtype)
    {
#if(__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case AITER_DTYPE_bf16: {
        DISPATCH_AR_FUSION_MXFP4(opus::bf16_t)
        break;
    }
#endif
    case AITER_DTYPE_fp16: {
        DISPATCH_AR_FUSION_MXFP4(opus::fp16_t)
        break;
    }
    default:
        throw std::runtime_error(
            "fused_allreduce_rmsnorm_mxfp4_quant only supports bfloat16/float16 input");
    }
#undef DISPATCH_AR_FUSION_MXFP4
}

"""

if "_fused_allreduce_rmsnorm_mxfp4" not in cu:
    assert ANCHOR_CU_HELPER in cu, "ANCHOR_CU_HELPER not found"
    cu = cu.replace(ANCHOR_CU_HELPER, INSERT_CU_HELPER + ANCHOR_CU_HELPER, 1)

# 1b) Insert fused_allreduce_rmsnorm_mxfp4_quant entry function AFTER existing
# fused_allreduce_rmsnorm_quant (just before "} // namespace aiter")
ANCHOR_CU_NS_END = """} // namespace aiter"""

INSERT_CU_ENTRY = """
void fused_allreduce_rmsnorm_mxfp4_quant(fptr_t _fa,
                                         const aiter_tensor_t& inp,
                                         const aiter_tensor_t& res_inp,
                                         const aiter_tensor_t& res_out,
                                         const aiter_tensor_t& out,
                                         const aiter_tensor_t& scale_out,
                                         const aiter_tensor_t& w,
                                         double eps,
                                         int64_t reg_ptr, int64_t reg_bytes,
                                         bool use_1stage)
{
    HipDeviceGuard device_guard(inp.device_id);
    hipStream_t stream = aiter::getCurrentHIPStream();
    auto dtype     = inp.dtype();
    int64_t numel  = inp.numel();
    int64_t data_bytes = numel * inp.element_size();
    int n = (int)w.numel();
    int m = (int)(numel / w.numel());

    if(reg_ptr != 0)
    {
        if(data_bytes > reg_bytes)
            throw std::runtime_error("registered buffer is too small to contain the input");
        HIP_CALL(hipMemcpyAsync((void*)reg_ptr, inp.data_ptr(), data_bytes,
                                hipMemcpyDeviceToDevice, stream));
        _fused_allreduce_rmsnorm_mxfp4(_fa,
                                       (void*)reg_ptr, res_inp.data_ptr(), res_out.data_ptr(),
                                       out.data_ptr(), scale_out.data_ptr(), w.data_ptr(),
                                       dtype, (float)eps, m, n, use_1stage);
    }
    else
    {
        _fused_allreduce_rmsnorm_mxfp4(_fa,
                                       inp.data_ptr(), res_inp.data_ptr(), res_out.data_ptr(),
                                       out.data_ptr(), scale_out.data_ptr(), w.data_ptr(),
                                       dtype, (float)eps, m, n, use_1stage);
    }
}

"""

if "fused_allreduce_rmsnorm_mxfp4_quant" not in cu:
    # Insert just before the namespace closer
    last_ns_end = cu.rfind(ANCHOR_CU_NS_END)
    assert last_ns_end > 0, "namespace end not found"
    cu = cu[:last_ns_end] + INSERT_CU_ENTRY + cu[last_ns_end:]

if cu != CU_ORIG:
    open(CU_PATH, 'w').write(cu)
    print(f"PATCHED {CU_PATH} (+{cu.count(chr(10)) - CU_ORIG.count(chr(10))} lines)")
else:
    print(f"  {CU_PATH} already patched")

# ============== PATCH 2: custom_all_reduce.h ==============
hh = open(H_PATH).read()
H_ORIG = hh

ANCHOR_H = """void fused_allreduce_rmsnorm_quant(fptr_t _fa,
                                   const aiter_tensor_t& inp,
                                   const aiter_tensor_t& res_inp,
                                   const aiter_tensor_t& res_out,
                                   const aiter_tensor_t& out,
                                   const aiter_tensor_t& scale_out,
                                   const aiter_tensor_t& w,
                                   double eps,
                                   int64_t reg_ptr,
                                   int64_t reg_bytes,
                                   bool use_1stage);"""

INSERT_H = ANCHOR_H + """
void fused_allreduce_rmsnorm_mxfp4_quant(fptr_t _fa,
                                         const aiter_tensor_t& inp,
                                         const aiter_tensor_t& res_inp,
                                         const aiter_tensor_t& res_out,
                                         const aiter_tensor_t& out,
                                         const aiter_tensor_t& scale_out,
                                         const aiter_tensor_t& w,
                                         double eps,
                                         int64_t reg_ptr,
                                         int64_t reg_bytes,
                                         bool use_1stage);"""

if "fused_allreduce_rmsnorm_mxfp4_quant" not in hh:
    assert ANCHOR_H in hh, "ANCHOR_H not found"
    hh = hh.replace(ANCHOR_H, INSERT_H, 1)

if hh != H_ORIG:
    open(H_PATH, 'w').write(hh)
    print(f"PATCHED {H_PATH}")
else:
    print(f"  {H_PATH} already patched")

# ============== PATCH 3: rocm_ops.hpp pybind ==============
pyb = open(PYB_PATH).read()
PYB_ORIG = pyb

ANCHOR_PYB = """    m.def("fused_allreduce_rmsnorm_quant",                                                     \\
          &aiter::fused_allreduce_rmsnorm_quant,                                               \\
          py::arg("_fa"),                                                                      \\
          py::arg("inp"),                                                                      \\
          py::arg("res_inp"),                                                                  \\
          py::arg("res_out"),                                                                  \\
          py::arg("out"),                                                                      \\
          py::arg("scale_out"),                                                                \\
          py::arg("w"),                                                                        \\
          py::arg("eps"),                                                                      \\
          py::arg("reg_ptr"),                                                                  \\
          py::arg("reg_bytes"),                                                                \\
          py::arg("use_1stage"));                                                              \\"""

INSERT_PYB = ANCHOR_PYB + """
    m.def("fused_allreduce_rmsnorm_mxfp4_quant",                                               \\
          &aiter::fused_allreduce_rmsnorm_mxfp4_quant,                                         \\
          py::arg("_fa"),                                                                      \\
          py::arg("inp"),                                                                      \\
          py::arg("res_inp"),                                                                  \\
          py::arg("res_out"),                                                                  \\
          py::arg("out"),                                                                      \\
          py::arg("scale_out"),                                                                \\
          py::arg("w"),                                                                        \\
          py::arg("eps"),                                                                      \\
          py::arg("reg_ptr"),                                                                  \\
          py::arg("reg_bytes"),                                                                \\
          py::arg("use_1stage"));                                                              \\"""

if "fused_allreduce_rmsnorm_mxfp4_quant" not in pyb:
    assert ANCHOR_PYB in pyb, "ANCHOR_PYB not found"
    pyb = pyb.replace(ANCHOR_PYB, INSERT_PYB, 1)

if pyb != PYB_ORIG:
    open(PYB_PATH, 'w').write(pyb)
    print(f"PATCHED {PYB_PATH}")
else:
    print(f"  {PYB_PATH} already patched")

print("=== ALL 3 PATCHES APPLIED ===")

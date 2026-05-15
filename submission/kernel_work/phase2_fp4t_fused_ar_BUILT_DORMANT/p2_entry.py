#!/usr/bin/env python3
"""Insert ONLY the entry function fused_allreduce_rmsnorm_mxfp4_quant."""
PATH = "/app/aiter-test/csrc/kernels/custom_all_reduce.cu"
src = open(PATH).read()

if "void fused_allreduce_rmsnorm_mxfp4_quant(fptr_t _fa," in src:
    print("entry already present")
else:
    INSERT = """
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
    NS_END = "} // namespace aiter"
    pos = src.rfind(NS_END)
    assert pos > 0, "namespace closer not found"
    src = src[:pos] + INSERT + src[pos:]
    open(PATH, "w").write(src)
    print("INSERTED entry function (+", INSERT.count(chr(10)), "lines)")

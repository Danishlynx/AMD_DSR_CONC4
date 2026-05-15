F3 = "/app/aiter-test/aiter/dist/parallel_state.py"
src = open(F3).read()
if "def _fused_allreduce_rmsnorm_mxfp4_quant_out_place(" not in src:
    INSERT = """    def _fused_allreduce_rmsnorm_mxfp4_quant_out_place(
        self,
        input_,
        residual_inp_,
        weight_,
        eps,
        prefill_support: bool = False,
    ):
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
    OLD = "    def _fused_allreduce_rmsnorm_quant_out_place("
    pos = src.find(OLD)
    assert pos > 0, "anchor not found"
    src = src[:pos] + INSERT + src[pos + len(OLD):]
    open(F3, "w").write(src)
    print("INSERTED out_place method")
else:
    print("already present")

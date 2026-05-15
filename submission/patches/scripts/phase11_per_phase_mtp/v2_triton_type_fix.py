#!/usr/bin/env python3
# Fix Triton type mismatch in rejection_phased_sample_kernel
# new_phase needs explicit int8 dtype on constant assignments.
import py_compile, sys

RS_PATH = "/app/ATOM/atom/model_ops/rejection_sampler.py"
src = open(RS_PATH).read()

# Convert int8 load to int32 (uniform with constants), cast back to int8 on store.
old1 = """    phase_val = tl.load(phase_ptr + req_idx)
    is_thinking = phase_val == 1
    new_phase = phase_val"""

new1 = """    phase_val = tl.load(phase_ptr + req_idx).to(tl.int32)
    is_thinking = phase_val == 1
    new_phase = phase_val"""

if old1 not in src:
    sys.exit("ERR: phase load anchor not found")
src = src.replace(old1, new1, 1)

# Final store needs cast back to int8 since phase tensor is int8 dtype.
old2 = "    tl.store(phase_ptr + req_idx, new_phase)"
new2 = "    tl.store(phase_ptr + req_idx, new_phase.to(tl.int8))"
if old2 not in src:
    sys.exit("ERR: phase store anchor not found")
src = src.replace(old2, new2, 1)

open(RS_PATH, "w").write(src)
py_compile.compile(RS_PATH, doraise=True)
print("OK fix applied (int32 in kernel, cast int8 on store)")

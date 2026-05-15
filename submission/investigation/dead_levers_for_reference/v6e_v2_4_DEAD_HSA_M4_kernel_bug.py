#!/usr/bin/env python3
# v6e v2.4 fix: torch.zeros doesn't work for fp4x2 dtype (no fill_cuda).
# Workaround: allocate as uint8 with zeros, then view as float4_e2m1fn_x2.
import py_compile, sys

MOE_PATH = "/app/ATOM/atom/model_ops/moe.py"
src = open(MOE_PATH).read()

old = """            _fp4_buf = torch.zeros(
                _v6e_max_tokens, hidden_size // 2,
                dtype=torch.float4_e2m1fn_x2, device="cuda",
            )"""

new = """            # fp4x2 has no fill_cuda; zero-init via uint8 storage then view-cast.
            _fp4_buf = torch.zeros(
                _v6e_max_tokens, hidden_size // 2,
                dtype=torch.uint8, device="cuda",
            ).view(torch.float4_e2m1fn_x2)"""

if old not in src:
    sys.exit("ERR: fp4 zero-alloc anchor not found")
src = src.replace(old, new, 1)

open(MOE_PATH, "w").write(src)
py_compile.compile(MOE_PATH, doraise=True)
print("OK v6e v2.4 fix applied: fp4x2 zeros via uint8.view")

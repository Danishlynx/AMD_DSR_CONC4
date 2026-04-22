#!/usr/bin/env python3
"""Patch C++ v1_2_device.cuh to add (nhead=32, sq=8, fp8/fp8) as natively_supported."""
import os
import shutil
import sys

PATH = "/app/aiter-test/csrc/kernels/mla/metadata/v1_2_device.cuh"
BACKUP = PATH + ".pre_v8"

old = '''        ((arch_id == "gfx950") && (num_heads == 32) && q_is_fp8 && kv_is_fp8 &&
         (max_seqlen_qo == 4)) ||'''

new = '''        ((arch_id == "gfx950") && (num_heads == 32) && q_is_fp8 && kv_is_fp8 &&
         (max_seqlen_qo == 4)) ||
        ((arch_id == "gfx950") && (num_heads == 32) && q_is_fp8 && kv_is_fp8 &&
         (max_seqlen_qo == 8)) ||'''

if not os.path.exists(BACKUP):
    shutil.copy2(PATH, BACKUP)
    print(f"backup -> {BACKUP}")

with open(PATH, "r") as f:
    s = f.read()

if new in s:
    print("already patched")
    sys.exit(0)

if old not in s:
    print("ERROR: pattern not found", file=sys.stderr)
    sys.exit(1)

s2 = s.replace(old, new, 1)
with open(PATH, "w") as f:
    f.write(s2)

print("C++ v1_2_device.cuh patched")

#!/usr/bin/env python3
"""Patch aiter/ops/attention.py to include max_seqlen_qo=8 in natively_supported.
Run inside container:
  python3 /tmp/patch_metadata_sq8.py
"""
import os
import shutil
import sys

PATH = "/app/aiter-test/aiter/ops/attention.py"
BACKUP = PATH + ".pre_v8"

old = '''            arch_id == "gfx950"
            and num_heads_per_head_k == 32
            and q_is_fp8
            and kv_is_fp8
            and max_seqlen_qo == 4'''

new = '''            arch_id == "gfx950"
            and num_heads_per_head_k == 32
            and q_is_fp8
            and kv_is_fp8
            and max_seqlen_qo in (4, 8)'''

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
assert old not in s2

with open(PATH, "w") as f:
    f.write(s2)

print("patched OK")

#!/usr/bin/env python3
"""Phase 5 L5 envs.py patch — register ATOM_SPEC_V2_OVERLAP env var.

NULL-OP at default ("0"); when "1" enables Spec V2 propose-loop overlap pattern
(prepare_mtp_decode + slot_mapping update on spec_prefetch_stream concurrent
with default-stream positions += 1).
"""
import io
import os
import py_compile
import sys

TARGET = "/app/ATOM/atom/utils/envs.py"
BACKUP = "/app/ATOM/atom/utils/envs.py.pre_phase5_l5"
PATCH_MARKER = "# >>> phase5_l5_spec_v2_overlap <<<"

# Anchor: insert immediately AFTER the ATOM_USE_TRITON_MXFP4_BMM entry
ANCHOR = '''    "ATOM_USE_TRITON_MXFP4_BMM": lambda: os.getenv("ATOM_USE_TRITON_MXFP4_BMM", "0")
    == "1",'''

INSERT = '''
    # >>> phase5_l5_spec_v2_overlap <<<
    # Spec V2 propose-loop overlap: dispatch prepare_mtp_decode + slot_mapping
    # update on spec_prefetch_stream so default stream can run positions += 1
    # concurrently. Pattern mirrors SGLang SGLANG_ENABLE_SPEC_V2.
    "ATOM_SPEC_V2_OVERLAP": lambda: os.getenv("ATOM_SPEC_V2_OVERLAP", "0") == "1",
    # <<< phase5_l5_spec_v2_overlap <<<'''


def main():
    with open(TARGET, "r") as f:
        src = f.read()

    if PATCH_MARKER in src:
        print(f"[phase5_l5_envs] already patched (marker present), skipping")
        return 0

    if ANCHOR not in src:
        print(f"[phase5_l5_envs] FATAL: anchor not found in {TARGET}", file=sys.stderr)
        sys.exit(1)

    if src.count(ANCHOR) != 1:
        print(
            f"[phase5_l5_envs] FATAL: anchor matched {src.count(ANCHOR)} times, "
            f"expected exactly 1",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.exists(BACKUP):
        with open(BACKUP, "w") as f:
            f.write(src)
        print(f"[phase5_l5_envs] backup written: {BACKUP}")

    new_src = src.replace(ANCHOR, ANCHOR + INSERT, 1)

    with open(TARGET, "w") as f:
        f.write(new_src)
    print(f"[phase5_l5_envs] wrote {TARGET}")

    # AST validate
    try:
        py_compile.compile(TARGET, doraise=True)
        print(f"[phase5_l5_envs] py_compile OK")
    except py_compile.PyCompileError as e:
        print(f"[phase5_l5_envs] FATAL: py_compile failed: {e}", file=sys.stderr)
        # rollback
        with open(BACKUP, "r") as f:
            with open(TARGET, "w") as g:
                g.write(f.read())
        print(f"[phase5_l5_envs] ROLLED BACK from backup", file=sys.stderr)
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())

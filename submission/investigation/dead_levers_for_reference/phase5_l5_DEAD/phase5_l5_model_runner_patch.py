#!/usr/bin/env python3
"""Phase 5 L5 model_runner.py patch — allocate spec_prefetch_stream.

Allocates a new torch.cuda.Stream alongside the existing async_execute_stream.
NULL allocation has no runtime cost; eagle.py only uses it when
ATOM_SPEC_V2_OVERLAP=1.
"""
import os
import py_compile
import sys

TARGET = "/app/ATOM/atom/model_engine/model_runner.py"
BACKUP = "/app/ATOM/atom/model_engine/model_runner.py.pre_phase5_l5"
PATCH_MARKER = "# >>> phase5_l5_spec_prefetch_stream <<<"

# Anchor: the existing async_execute_stream allocation in ModelRunner.load_model
ANCHOR = '''        torch.set_default_device(self.device)
        self.async_execute_stream = torch.cuda.Stream(self.device)'''

REPLACEMENT = '''        torch.set_default_device(self.device)
        self.async_execute_stream = torch.cuda.Stream(self.device)
        # >>> phase5_l5_spec_prefetch_stream <<<
        # Side stream for Spec V2 propose-loop overlap (eagle.py prepare_mtp_decode +
        # slot_mapping update) when ATOM_SPEC_V2_OVERLAP=1. Allocation is unconditional
        # but unused if the env flag is off, so this is a NULL-OP at default config.
        self.spec_prefetch_stream = torch.cuda.Stream(self.device)
        # <<< phase5_l5_spec_prefetch_stream <<<'''


def main():
    with open(TARGET, "r") as f:
        src = f.read()

    if PATCH_MARKER in src:
        print(f"[phase5_l5_mr] already patched (marker present), skipping")
        return 0

    if ANCHOR not in src:
        print(f"[phase5_l5_mr] FATAL: anchor not found in {TARGET}", file=sys.stderr)
        sys.exit(1)

    if src.count(ANCHOR) != 1:
        print(
            f"[phase5_l5_mr] FATAL: anchor matched {src.count(ANCHOR)} times, "
            f"expected exactly 1",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.exists(BACKUP):
        with open(BACKUP, "w") as f:
            f.write(src)
        print(f"[phase5_l5_mr] backup written: {BACKUP}")

    new_src = src.replace(ANCHOR, REPLACEMENT, 1)

    with open(TARGET, "w") as f:
        f.write(new_src)
    print(f"[phase5_l5_mr] wrote {TARGET}")

    try:
        py_compile.compile(TARGET, doraise=True)
        print(f"[phase5_l5_mr] py_compile OK")
    except py_compile.PyCompileError as e:
        print(f"[phase5_l5_mr] FATAL: py_compile failed: {e}", file=sys.stderr)
        with open(BACKUP, "r") as f:
            with open(TARGET, "w") as g:
                g.write(f.read())
        print(f"[phase5_l5_mr] ROLLED BACK from backup", file=sys.stderr)
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())

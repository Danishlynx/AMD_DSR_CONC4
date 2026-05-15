#!/usr/bin/env python3
"""
Phase 1 L0 — config.py override fix.

Discovery: atom/config.py line ~872 unconditionally sets
    self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
when level == PIECEWISE. This silently clobbers any user-supplied cudagraph_mode
(including ATOM_CUDAGRAPH_MODE=FULL_AND_PIECEWISE). The fix is to respect a
non-None user value: only force PIECEWISE when cudagraph_mode is still its
sentinel default (None).

This is a 1-line semantic fix:

    BEFORE:
        self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE

    AFTER:
        if self.compilation_config.cudagraph_mode is None:
            self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE

Mergeability: cleanly preserves backward-compat (None default still maps to
PIECEWISE), unblocks the documented FULL/FULL_AND_PIECEWISE/FULL_DECODE_ONLY
modes that the Pipeline already supports but the runtime is silently rejecting.

Apply via:
    python3 phase1_l0_config_override_patch.py /app/ATOM/atom/config.py
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

PATCH_MARKER = "# >>> phase1_l0_cudagraph_mode_user_override <<<"


def patch_file(path: Path) -> int:
    src = path.read_text()
    if PATCH_MARKER in src:
        print(f"[skip] {path} already patched (marker present)")
        return 0

    backup = path.with_suffix(path.suffix + ".pre_phase1_l0_cfgfix")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"[backup] {backup}")

    pattern = re.compile(
        r'(\n)(\s+)(self\.compilation_config\.cudagraph_mode\s*=\s*CUDAGraphMode\.PIECEWISE)\s*\n',
    )
    m = pattern.search(src)
    if not m:
        sys.stderr.write("ERROR: target line not found in config.py\n")
        return 2

    indent = m.group(2)
    replacement = (
        f'\n{indent}{PATCH_MARKER}\n'
        f'{indent}# Phase 1 L0: respect user-supplied cudagraph_mode (e.g. via\n'
        f'{indent}# ATOM_CUDAGRAPH_MODE=FULL_AND_PIECEWISE plumbed in arg_utils.py).\n'
        f'{indent}# Only force PIECEWISE when the field is still its sentinel default.\n'
        f'{indent}if self.compilation_config.cudagraph_mode is None:\n'
        f'{indent}    self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE\n'
    )
    src = src[:m.start()] + replacement + src[m.end():]

    path.write_text(src)
    print(f"[patched] {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="/app/ATOM/atom/config.py")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    p = Path(args.path)
    if not p.exists():
        sys.stderr.write(f"ERROR: {p} not found\n")
        return 1
    if args.dry_run:
        print(f"[dry-run] would patch {p}")
        return 0
    return patch_file(p)


if __name__ == "__main__":
    sys.exit(main())

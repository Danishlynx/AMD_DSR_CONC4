"""Phase A1 memoization v2 — bugfixes from v1:
  Bug 1: helper insertion landed inside a multi-line `from ... import (` block.
         Fix: scan past parens — only consider an import "complete" when paren depth = 0.
  Bug 2: global string replace also replaced the CALL inside helper body itself,
         creating infinite recursion `_cached_get_device_properties` → itself.
         Fix: replace call sites FIRST (in original src), then insert helper text.
         The helper body uses the literal `torch.cuda.get_device_properties` directly.

Idempotent. Backups preserved at .pre_phase_a1_memoize (already exist from v1 run).
"""
import os, shutil, sys, re

PATCHES = [
    "/app/aiter-test/aiter/ops/attention.py",
    "/app/aiter-test/aiter/dist/device_communicators/quick_all_reduce.py",
    "/app/aiter-test/aiter/ops/flydsl/utils.py",
]

MARKER = "# >>> PHASE A1 hipGetDeviceProperties memoization >>>"
HELPER = (
    "\n"
    "# >>> PHASE A1 hipGetDeviceProperties memoization >>>\n"
    "# Eliminates ~1.4 ms/forward host overhead measured Apr 28: 1377 calls/forward of\n"
    "# torch.cuda.get_device_properties (which is constant per-GPU). Module-local cache.\n"
    "_DEV_PROPS_CACHE = {}\n"
    "def _cached_get_device_properties(device):\n"
    "    \"\"\"Memoized torch.cuda.get_device_properties — identical return for same device.\"\"\"\n"
    "    key = device if isinstance(device, int) else getattr(device, \"index\", device)\n"
    "    if key not in _DEV_PROPS_CACHE:\n"
    "        _DEV_PROPS_CACHE[key] = torch.cuda.get_device_properties(device)\n"
    "    return _DEV_PROPS_CACHE[key]\n"
    "# <<< PHASE A1 hipGetDeviceProperties memoization <<<\n"
    "\n"
)


def find_safe_insert_point(src: str) -> int:
    """Return char-offset AFTER the last top-level import statement, accounting for
    multi-line `from X import (a, b, c)` blocks. Returns offset right after the
    closing newline of the last import.
    """
    lines = src.split("\n")
    paren_depth = 0
    last_import_end_line = -1
    in_import = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Track paren depth across lines
        if in_import or stripped.startswith("import ") or stripped.startswith("from "):
            if stripped.startswith("import ") or stripped.startswith("from "):
                in_import = True
                paren_depth = 0
            paren_depth += line.count("(") - line.count(")")
            if paren_depth <= 0:
                # Import statement complete on this line
                last_import_end_line = i
                in_import = False
                paren_depth = 0
        # Don't break early — keep scanning for later imports too
    if last_import_end_line < 0:
        return 0
    # Compute char offset of START of line AFTER last_import_end_line
    offset = 0
    for i, line in enumerate(lines):
        if i == last_import_end_line + 1:
            return offset
        offset += len(line) + 1  # +1 for newline
    return offset  # End of file


def patch_one(path: str) -> bool:
    if not os.path.exists(path):
        print(f"  SKIP (not present): {path}")
        return False
    backup = path + ".pre_phase_a1_memoize"
    with open(path) as f:
        src = f.read()

    if MARKER in src:
        print(f"  ALREADY PATCHED: {path}")
        return True

    # Backup if missing
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"  backup: {backup}")

    # Step 1: Replace call sites in ORIGINAL src (before helper insertion)
    old_call = "torch.cuda.get_device_properties("
    new_call = "_cached_get_device_properties("
    n_replaced = src.count(old_call)
    if n_replaced == 0:
        print(f"  WARN: no call sites in {path}")
        return False
    src_replaced = src.replace(old_call, new_call)

    # Step 2: Find safe insert point (after last complete import block)
    insert_offset = find_safe_insert_point(src_replaced)

    # Step 3: Insert helper at offset
    src_final = src_replaced[:insert_offset] + HELPER + src_replaced[insert_offset:]

    with open(path, "w") as f:
        f.write(src_final)
    print(f"  PATCHED {path} ({n_replaced} sites replaced, helper inserted at offset {insert_offset})")
    return True


if __name__ == "__main__":
    print("=== Phase A1 memoization v2 patch ===")
    for path in PATCHES:
        patch_one(path)
    print("=== done. Run byte-compile to verify ===")

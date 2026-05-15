"""Phase A1 v3 — RETARGETED memoization at the actual hot path.

v1: patched torch.cuda.get_device_properties (4 sites in attention.py + 2 elsewhere).
    Result: 137,678 → 134,177 calls (only −3.5%). The hot path was elsewhere.
v3: patches `triton.runtime.driver.active.utils.get_device_properties` —
    the real hot path called from triton/compiler/compiler.py:131 + 3 sites in
    aiter/ops/triton/_triton_kernels/gated_delta_rule/gated_delta_rule_utils.py.

Strategy: monkey-patch the Triton driver utils method at aiter import time so
EVERY caller gets memoization without modifying triton site-packages.

Inject the monkey-patch into /app/aiter-test/aiter/__init__.py near the top.
The patch replaces `triton.runtime.driver.active.utils.get_device_properties`
with a memoized wrapper. Same return value, same call signature.

Backup: /app/aiter-test/aiter/__init__.py.pre_phase_a1_v3
Idempotent.
"""
import os
import shutil
import sys

PATH = "/app/aiter-test/aiter/__init__.py"
BACKUP = PATH + ".pre_phase_a1_v3"
MARKER = "# >>> PHASE A1 v3 triton.runtime.driver memoization >>>"

PATCH_BLOCK = '''
# >>> PHASE A1 v3 triton.runtime.driver memoization >>>
# Profile (Apr 28) measured 134,177 calls/100-forwards = 1342 calls/forward of
# triton.runtime.driver.active.utils.get_device_properties — pure host overhead
# since GPU props are CONSTANT per-device. Memoize via in-process cache.
#
# Hot path: triton/compiler/compiler.py:131 -> driver.active.utils.get_device_properties
# Called once per kernel autotune lookup; with thousands of triton kernels per
# forward, this dominates host time (~1.4 ms/forward).
try:
    import triton.runtime.driver as _trd
    if not getattr(_trd, "_phase_a1_v3_memoized", False):
        _orig_get_dp = _trd.active.utils.get_device_properties
        _PHASE_A1_V3_CACHE = {}
        def _phase_a1_v3_get_device_properties(device):
            key = device if isinstance(device, int) else getattr(device, "index", device)
            if key not in _PHASE_A1_V3_CACHE:
                _PHASE_A1_V3_CACHE[key] = _orig_get_dp(device)
            return _PHASE_A1_V3_CACHE[key]
        _trd.active.utils.get_device_properties = _phase_a1_v3_get_device_properties
        _trd._phase_a1_v3_memoized = True
        try:
            import logging as _logging
            _logging.getLogger("aiter").info(
                "[PHASE A1 v3] memoized triton.runtime.driver.active.utils.get_device_properties"
            )
        except Exception:
            pass
except Exception as _e:
    # Don't break import if triton runtime isn't ready yet at this point — caller
    # will hit the unmemoized path and we degrade gracefully.
    try:
        import logging as _logging
        _logging.getLogger("aiter").warning(
            f"[PHASE A1 v3] could not memoize triton driver get_device_properties: {_e}"
        )
    except Exception:
        pass
# <<< PHASE A1 v3 triton.runtime.driver memoization <<<

'''


def main():
    if not os.path.exists(PATH):
        print(f"FATAL: {PATH} not found")
        sys.exit(1)
    with open(PATH) as f:
        src = f.read()
    if MARKER in src:
        print(f"ALREADY PATCHED v3: {PATH}")
        return
    if not os.path.exists(BACKUP):
        shutil.copy2(PATH, BACKUP)
        print(f"  backup: {BACKUP}")
    # Insert PATCH_BLOCK after the existing module docstring + first imports.
    # Strategy: walk lines, find first non-{import,from,comment,blank,docstring} line,
    # insert above it. Be conservative: insert AFTER the very first `from`/`import`
    # blocks but BEFORE any function/class definitions.
    lines = src.split("\n")
    insert_idx = None
    in_docstring = False
    docstring_quote = None
    paren_depth = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Track triple-quoted module docstring
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                q = stripped[:3]
                # one-line docstring?
                rest = stripped[3:]
                if rest.endswith(q) and len(rest) >= 3:
                    continue
                in_docstring = True
                docstring_quote = q
                continue
        else:
            if stripped.endswith(docstring_quote):
                in_docstring = False
                docstring_quote = None
            continue
        # Track multi-line imports via parens
        if paren_depth > 0:
            paren_depth += line.count("(") - line.count(")")
            if paren_depth <= 0:
                paren_depth = 0
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            paren_depth = line.count("(") - line.count(")")
            if paren_depth < 0:
                paren_depth = 0
            continue
        if stripped == "" or stripped.startswith("#"):
            continue
        # Found the first non-import/non-docstring/non-comment content line
        insert_idx = i
        break
    if insert_idx is None:
        # File is all imports — append at end
        insert_idx = len(lines)
    new_lines = lines[:insert_idx] + [PATCH_BLOCK] + lines[insert_idx:]
    with open(PATH, "w") as f:
        f.write("\n".join(new_lines))
    print(f"  PATCHED {PATH} (insert at line {insert_idx})")


if __name__ == "__main__":
    print("=== Phase A1 v3 (Triton runtime memoization) ===")
    main()
    print("=== done. Byte-compile + boot + verify call count drop in profile ===")

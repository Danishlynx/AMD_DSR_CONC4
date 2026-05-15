"""Phase A1 — memoize torch.cuda.get_device_properties to eliminate 1377 calls/forward
of pure host overhead. Identified Apr 25, never applied. 5-min fix.

Sites in /app/aiter-test/aiter/ops/attention.py (verified Apr 28 plan-mode investigation):
  186:    device_properties = torch.cuda.get_device_properties(gpu)
  729:    device_properties = torch.cuda.get_device_properties(gpu)
  829:    device_properties = torch.cuda.get_device_properties(device)
  921:    device_properties = torch.cuda.get_device_properties(gpu)

Sites we ALSO memoize (smaller call counts but free wins):
  /app/aiter-test/aiter/dist/device_communicators/quick_all_reduce.py:41
  /app/aiter-test/aiter/ops/flydsl/utils.py:36

Strategy: add module-level _DEV_PROPS_CACHE dict + _cached_get_device_properties helper
in attention.py. Replace 4 calls. For the other 2 files, replicate the cache pattern
(or import from attention.py — but attention.py is sometimes loaded LATER, so each
file gets its own local _DEV_PROPS_CACHE for safety).

Idempotent. Backup at .pre_phase_a1_memoize.
"""
import os, shutil, sys

PATCHES = [
    ("/app/aiter-test/aiter/ops/attention.py", 4),
    ("/app/aiter-test/aiter/dist/device_communicators/quick_all_reduce.py", 1),
    ("/app/aiter-test/aiter/ops/flydsl/utils.py", 1),
]

MARKER = "# >>> PHASE A1 hipGetDeviceProperties memoization >>>"
HELPER = '''
# >>> PHASE A1 hipGetDeviceProperties memoization >>>
# Eliminates ~1.4 ms/forward host overhead measured Apr 28: 1377 calls/forward of
# torch.cuda.get_device_properties (which is constant per-GPU). Module-local cache.
_DEV_PROPS_CACHE = {}
def _cached_get_device_properties(device):
    """Memoized torch.cuda.get_device_properties — identical return for same device."""
    key = device if isinstance(device, int) else getattr(device, "index", device)
    if key not in _DEV_PROPS_CACHE:
        _DEV_PROPS_CACHE[key] = torch.cuda.get_device_properties(device)
    return _DEV_PROPS_CACHE[key]
# <<< PHASE A1 hipGetDeviceProperties memoization <<<
'''


def patch_one(path: str, expected_count: int) -> bool:
    if not os.path.exists(path):
        print(f"  SKIP (not present): {path}")
        return False
    backup = path + ".pre_phase_a1_memoize"
    with open(path) as f:
        src = f.read()

    if MARKER in src:
        print(f"  ALREADY PATCHED: {path}")
        return True

    # Backup
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"  backup: {backup}")

    # Insert helper after the last top-level `import` block
    # Find the first non-import non-comment non-blank line and insert before it
    lines = src.split("\n")
    insert_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            insert_idx = i + 1
        elif stripped == "" or stripped.startswith("#"):
            continue
        elif insert_idx > 0:
            # First non-import, non-comment, non-blank — stop here
            break
    new_lines = lines[:insert_idx] + [HELPER] + lines[insert_idx:]
    src = "\n".join(new_lines)

    # Replace the call sites
    old_call = "torch.cuda.get_device_properties("
    new_call = "_cached_get_device_properties("
    n = src.count(old_call)
    if n == 0:
        print(f"  WARN: no call sites in {path}")
        # restore from backup
        return False
    if n != expected_count:
        print(f"  WARN: found {n} sites in {path}, expected {expected_count} — proceeding anyway")
    src = src.replace(old_call, new_call)

    with open(path, "w") as f:
        f.write(src)
    print(f"  PATCHED {path} ({n} sites)")
    return True


if __name__ == "__main__":
    print("=== Phase A1 memoization patch ===")
    for path, n in PATCHES:
        patch_one(path, n)
    print("=== done. Byte-compile + restart server + 3-iter bench ===")

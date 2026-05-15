#!/usr/bin/env python3
"""
Phase 1 L0 patch — env-gated cudagraph_mode wiring in atom EngineArgs.

Discovery during Phase 1: L1 (compile_sizes) is ALREADY shipped via env var
ATOM_COMPILE_SIZES (a prior session's wire-up at lines 242-244 of
/app/ATOM/atom/model_engine/arg_utils.py reads ATOM_COMPILE_SIZES and injects
compile_sizes=["cudagraph_capture_sizes"] into CompilationConfig).

L0 needs a parallel addition for cudagraph_mode. We mirror the same pattern:
read ATOM_CUDAGRAPH_MODE env var, validate it against the allowed enum values,
and inject into CompilationConfig.cudagraph_mode kwarg.

Lever L0: cudagraph_mode = "FULL_AND_PIECEWISE"
   docs/compilation_cudagraph_guide.md:100-114 calls this "the most performant
   mode for most models" — eliminates per-step kernel-launch overhead at decode
   by capturing the full decode forward in one cudagraph (60 layers × ~10 µs
   = 600 µs/step), while preserving piecewise capture for prefill.

Mergeability: env-gated, default OFF (preserves current behavior), validates
input against the upstream CUDAGraphMode enum. Maps cleanly to a single PR.

Apply via:
    python3 phase1_l0_cudagraph_mode_patch.py /app/ATOM/atom/model_engine/arg_utils.py

Idempotent — re-running detects PATCH_MARKER and skips.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

PATCH_MARKER = "# >>> phase1_l0_cudagraph_mode <<<"
PATCH_MARKER_END = "# <<< phase1_l0_cudagraph_mode >>>"


def patch_file(path: Path) -> int:
    src = path.read_text()
    if PATCH_MARKER in src:
        print(f"[skip] {path} already patched (marker present)")
        return 0

    backup = path.with_suffix(path.suffix + ".pre_phase1_l0")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"[backup] {backup}")

    # ── Locate the existing ATOM_COMPILE_SIZES wiring; insert L0 right after it.
    #
    # Existing lines (per phase 0 inspection of live file):
    #     import os as _os_l3
    #     _l3_compile_sizes = None
    #     if _os_l3.getenv("ATOM_COMPILE_SIZES", "").lower() == "cudagraph_capture_sizes":
    #         _l3_compile_sizes = ["cudagraph_capture_sizes"]
    #     kwargs["compilation_config"] = CompilationConfig(
    #         level=kwargs.pop("level"),
    #         compile_sizes=_l3_compile_sizes,
    #         cudagraph_capture_sizes=(
    #             parse_size_list(kwargs.pop("cudagraph_capture_sizes"))
    #             if self.cudagraph_capture_sizes
    #             else None
    #         ),
    #     )

    anchor_re = re.compile(
        r'(\n)'
        r'(\s+)kwargs\["compilation_config"\]\s*=\s*CompilationConfig\(\s*\n'
        r'(\s+)level=kwargs\.pop\("level"\),\s*\n'
        r'\s+compile_sizes=_l3_compile_sizes,\s*\n'
        r'\s+cudagraph_capture_sizes=\([^)]*\)\s*\n'
        r'\s+if self\.cudagraph_capture_sizes\s*\n'
        r'\s+else None\s*\n'
        r'\s+\),\s*\n'
        r'(\s+)\)\s*\n',
        re.MULTILINE,
    )
    m = anchor_re.search(src)
    if not m:
        # Fallback: weaker anchor (just the CompilationConfig opening)
        weak_re = re.compile(
            r'(\n)(\s+)kwargs\["compilation_config"\]\s*=\s*CompilationConfig\(\s*\n',
            re.MULTILINE,
        )
        m = weak_re.search(src)
        if not m:
            sys.stderr.write("ERROR: CompilationConfig(...) construction not found.\n")
            sys.stderr.write("File layout has drifted; manual inspection required.\n")
            return 2

    indent = m.group(2)

    # ── Inject L0 cudagraph_mode env-read block BEFORE the CompilationConfig call,
    # then add a closing-paren-replacement that includes cudagraph_mode= kwarg.

    # Strategy: find the CompilationConfig block end and inject the cudagraph_mode
    # kwarg into it; also inject the env-read snippet before the call.

    # Find the exact block to replace.
    block_re = re.compile(
        r'(\s+)import os as _os_l3\s*\n'
        r'(\s+)_l3_compile_sizes = None\s*\n'
        r'(\s+)if _os_l3\.getenv\("ATOM_COMPILE_SIZES",[^\n]+\n'
        r'(\s+)_l3_compile_sizes = \["cudagraph_capture_sizes"\]\s*\n'
        r'(\s+)kwargs\["compilation_config"\]\s*=\s*CompilationConfig\(\s*\n'
        r'(\s+)level=kwargs\.pop\("level"\),\s*\n'
        r'(\s+)compile_sizes=_l3_compile_sizes,\s*\n'
        r'(\s+)cudagraph_capture_sizes=\(\s*\n'
        r'(\s+)parse_size_list\(kwargs\.pop\("cudagraph_capture_sizes"\)\)\s*\n'
        r'(\s+)if self\.cudagraph_capture_sizes\s*\n'
        r'(\s+)else None\s*\n'
        r'(\s+)\),\s*\n'
        r'(\s+)\)\s*\n',
        re.MULTILINE,
    )
    m = block_re.search(src)
    if not m:
        sys.stderr.write("ERROR: full L1+CompilationConfig block not found via strict regex.\n")
        sys.stderr.write("Check whether arg_utils.py has been edited since Phase 0 inspection.\n")
        return 3

    indent = m.group(1)
    replacement = (
        f'{indent}import os as _os_l3\n'
        f'{indent}_l3_compile_sizes = None\n'
        f'{indent}if _os_l3.getenv("ATOM_COMPILE_SIZES", "").lower() == "cudagraph_capture_sizes":\n'
        f'{indent}    _l3_compile_sizes = ["cudagraph_capture_sizes"]\n'
        f'{indent}{PATCH_MARKER}\n'
        f'{indent}# L0 — env-gated cudagraph_mode override (FULL_AND_PIECEWISE recommended\n'
        f'{indent}# for DSR1-class MoE per docs/compilation_cudagraph_guide.md L100-114).\n'
        f'{indent}_l0_cudagraph_mode = None\n'
        f'{indent}_l0_mode_env = _os_l3.getenv("ATOM_CUDAGRAPH_MODE", "").upper()\n'
        f'{indent}if _l0_mode_env:\n'
        f'{indent}    _l0_allowed = {{"NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"}}\n'
        f'{indent}    if _l0_mode_env not in _l0_allowed:\n'
        f'{indent}        raise ValueError(\n'
        f'{indent}            f"ATOM_CUDAGRAPH_MODE={{_l0_mode_env!r}} not in {{_l0_allowed}}"\n'
        f'{indent}        )\n'
        f'{indent}    # Lazy import to avoid circulars; CUDAGraphMode is in atom.config\n'
        f'{indent}    try:\n'
        f'{indent}        from atom.config import CUDAGraphMode as _CGM\n'
        f'{indent}        _l0_cudagraph_mode = getattr(_CGM, _l0_mode_env)\n'
        f'{indent}    except (ImportError, AttributeError):\n'
        f'{indent}        # Fallback: pass the raw string; CompilationConfig coerces.\n'
        f'{indent}        _l0_cudagraph_mode = _l0_mode_env\n'
        f'{indent}{PATCH_MARKER_END}\n'
        f'{indent}_cc_kwargs = dict(\n'
        f'{indent}    level=kwargs.pop("level"),\n'
        f'{indent}    compile_sizes=_l3_compile_sizes,\n'
        f'{indent}    cudagraph_capture_sizes=(\n'
        f'{indent}        parse_size_list(kwargs.pop("cudagraph_capture_sizes"))\n'
        f'{indent}        if self.cudagraph_capture_sizes\n'
        f'{indent}        else None\n'
        f'{indent}    ),\n'
        f'{indent})\n'
        f'{indent}if _l0_cudagraph_mode is not None:\n'
        f'{indent}    _cc_kwargs["cudagraph_mode"] = _l0_cudagraph_mode\n'
        f'{indent}kwargs["compilation_config"] = CompilationConfig(**_cc_kwargs)\n'
    )
    src = src[:m.start()] + replacement + src[m.end():]

    path.write_text(src)
    print(f"[patched] {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?",
                        default="/app/ATOM/atom/model_engine/arg_utils.py")
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

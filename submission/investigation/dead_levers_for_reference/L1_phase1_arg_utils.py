#!/usr/bin/env python3
"""
Phase 1 L0+L1 patch — exposes --compilation-config JSON CLI in atom EngineArgs.

Lever L0: cudagraph_mode = "FULL_AND_PIECEWISE"
   docs/compilation_cudagraph_guide.md:100-114 explicitly recommends this as
   "the most performant mode for most models". Currently CompilationConfig.cudagraph_mode
   defaults to PIECEWISE and is not exposed via CLI.

Lever L1: compile_sizes = "cudagraph_capture_sizes"
   atom/config.py:241-256 (CompilationConfig.init_with_cudagraph_sizes) already
   supports the literal string "cudagraph_capture_sizes" — it substitutes the
   actual cudagraph_capture_sizes list at init time. What's missing is a CLI surface.

This patch adds a single new CLI flag --compilation-config that accepts a JSON
object and merges its keys into the CompilationConfig kwargs. Mergeable upstream
because (a) it uses an existing CompilationConfig field path, (b) defaults to
"" so behavior is unchanged when absent, (c) integrates with existing kwargs.

Apply via:
    python3 phase1_arg_utils_l0_l1_patch.py /app/ATOM/atom/model_engine/arg_utils.py

The script is idempotent — re-running detects already-patched state and skips.

Backup created at <file>.pre_phase1_l0l1 per Backup.md §2.4.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

PATCH_MARKER = "# >>> phase1_l0l1_compilation_config <<<"
PATCH_MARKER_END = "# <<< phase1_l0l1_compilation_config >>>"


def patch_file(path: Path) -> int:
    src = path.read_text()
    if PATCH_MARKER in src:
        print(f"[skip] {path} already patched (marker present)")
        return 0

    backup = path.with_suffix(path.suffix + ".pre_phase1_l0l1")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"[backup] {backup}")

    # ── 1. Add the new dataclass field next to cudagraph_capture_sizes ─────────
    field_re = re.compile(
        r'(\s+)cudagraph_capture_sizes:\s*str\s*=\s*"\[[^\]]*\]"\n',
        re.MULTILINE,
    )
    m = field_re.search(src)
    if not m:
        sys.stderr.write("ERROR: cudagraph_capture_sizes field not found — file layout changed?\n")
        return 2
    indent = m.group(1)
    insert_after = m.end()
    new_field = (
        f'{indent}{PATCH_MARKER}\n'
        f'{indent}compilation_config: str = ""\n'
        f'{indent}"""JSON dict merged into CompilationConfig kwargs at engine construction.\n'
        f'{indent}Supported keys: cudagraph_mode (str: "PIECEWISE"|"FULL"|"FULL_DECODE_ONLY"|"FULL_AND_PIECEWISE"),\n'
        f'{indent}                compile_sizes (list[int|str]: the literal "cudagraph_capture_sizes" is\n'
        f'{indent}                substituted with the cudagraph_capture_sizes list at init time).\n'
        f'{indent}Example: --compilation-config \'{{"cudagraph_mode":"FULL_AND_PIECEWISE","compile_sizes":["cudagraph_capture_sizes"]}}\'\n'
        f'{indent}"""\n'
        f'{indent}{PATCH_MARKER_END}\n'
    )
    src = src[:insert_after] + new_field + src[insert_after:]

    # ── 2. Add the argparse argument next to --cudagraph-capture-sizes ─────────
    cli_re = re.compile(
        r'(\s+)parser\.add_argument\(\s*\n'
        r'\s+"--cudagraph-capture-sizes",[^)]*\)\s*\n',
        re.MULTILINE,
    )
    m = cli_re.search(src)
    if not m:
        sys.stderr.write("ERROR: --cudagraph-capture-sizes argparse block not found\n")
        return 3
    cli_indent = m.group(1)
    insert_after = m.end()
    new_cli = (
        f'{cli_indent}{PATCH_MARKER}\n'
        f'{cli_indent}parser.add_argument(\n'
        f'{cli_indent}    "--compilation-config",\n'
        f'{cli_indent}    type=str,\n'
        f'{cli_indent}    default="",\n'
        f'{cli_indent}    help=(\n'
        f'{cli_indent}        \'JSON dict merged into CompilationConfig kwargs. \'\n'
        f'{cli_indent}        \'Example: \\\'{{"cudagraph_mode":"FULL_AND_PIECEWISE",\'\n'
        f'{cli_indent}        \'"compile_sizes":["cudagraph_capture_sizes"]}}\\\'\'\n'
        f'{cli_indent}    ),\n'
        f'{cli_indent})\n'
        f'{cli_indent}{PATCH_MARKER_END}\n'
    )
    src = src[:insert_after] + new_cli + src[insert_after:]

    # ── 3. Plumb into CompilationConfig kwargs at engine construction ──────────
    cc_re = re.compile(
        r'(\s+)kwargs\["compilation_config"\]\s*=\s*CompilationConfig\(\s*\n'
        r'\s+level=kwargs\.pop\("level"\),\s*\n'
        r'\s+cudagraph_capture_sizes=\(\s*\n'
        r'\s+parse_size_list\(kwargs\.pop\("cudagraph_capture_sizes"\)\)\s*\n'
        r'\s+if self\.cudagraph_capture_sizes\s*\n'
        r'\s+else None\s*\n'
        r'\s+\),\s*\n'
        r'\s+\)\s*\n',
        re.MULTILINE,
    )
    m = cc_re.search(src)
    if not m:
        sys.stderr.write("ERROR: CompilationConfig(...) construction block not found\n")
        return 4
    cc_indent = m.group(1)
    cc_replacement = (
        f'{cc_indent}{PATCH_MARKER}\n'
        f'{cc_indent}_cc_user = {{}}\n'
        f'{cc_indent}_cc_user_str = kwargs.pop("compilation_config", "")\n'
        f'{cc_indent}if _cc_user_str:\n'
        f'{cc_indent}    import json as _json\n'
        f'{cc_indent}    _cc_user = _json.loads(_cc_user_str)\n'
        f'{cc_indent}_cc_kwargs = dict(\n'
        f'{cc_indent}    level=kwargs.pop("level"),\n'
        f'{cc_indent}    cudagraph_capture_sizes=(\n'
        f'{cc_indent}        parse_size_list(kwargs.pop("cudagraph_capture_sizes"))\n'
        f'{cc_indent}        if self.cudagraph_capture_sizes else None\n'
        f'{cc_indent}    ),\n'
        f'{cc_indent})\n'
        f'{cc_indent}# CompilationConfig field whitelist (avoid silently dropping bad keys)\n'
        f'{cc_indent}_cc_allowed = {{"cudagraph_mode", "compile_sizes", "use_inductor",\n'
        f'{cc_indent}                "splitting_ops", "use_cudagraph", "full_cuda_graph"}}\n'
        f'{cc_indent}for _k, _v in _cc_user.items():\n'
        f'{cc_indent}    if _k not in _cc_allowed:\n'
        f'{cc_indent}        raise ValueError(\n'
        f'{cc_indent}            f"--compilation-config: key {{_k!r}} not in allowed set {{_cc_allowed}}"\n'
        f'{cc_indent}        )\n'
        f'{cc_indent}    _cc_kwargs[_k] = _v\n'
        f'{cc_indent}kwargs["compilation_config"] = CompilationConfig(**_cc_kwargs)\n'
        f'{cc_indent}{PATCH_MARKER_END}\n'
    )
    src = src[:m.start()] + cc_replacement + src[m.end():]

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

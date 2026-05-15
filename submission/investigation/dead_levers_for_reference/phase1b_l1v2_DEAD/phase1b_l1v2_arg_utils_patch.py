#!/usr/bin/env python3
"""Phase 1b L1-v2 arg_utils.py patch — JSON-list compile_sizes + inductor no-autotune env hooks.

Extends the existing L1 v1 ATOM_COMPILE_SIZES env parsing (which only accepts
"cudagraph_capture_sizes" string) to ALSO accept a JSON list literal like "[4]".
Adds a new ATOM_INDUCTOR_NO_AUTOTUNE=1 env that injects
inductor_compile_config={"max_autotune": False, ...} into CompilationConfig.

This is the dossier-1-derived L1-v2 retry of L1 v1 (which died on Apr 29 13:55 IST
with GSM8K 0.9295 because per-shape Inductor max_autotune picks different fp32
reduction trees per static shape, drifting lm_head numerics by ~0.005). v2 keeps
shape specialization (compile_sizes=[4]) but kills max_autotune to remove the
drift surface.

NULL-OP at default (env unset). Mergeable upstream as additive env config.
"""
import os
import py_compile
import sys

TARGET = "/app/ATOM/atom/model_engine/arg_utils.py"
BACKUP = "/app/ATOM/atom/model_engine/arg_utils.py.pre_phase1b_l1v2"
PATCH_MARKER = "# >>> phase1b_l1v2_inductor_compile_config <<<"

ANCHOR = '''        _l3_compile_sizes = None

        if _os_l3.getenv("ATOM_COMPILE_SIZES", "").lower() == "cudagraph_capture_sizes":

            _l3_compile_sizes = ["cudagraph_capture_sizes"]'''

REPLACEMENT = '''        _l3_compile_sizes = None
        _l1v2_inductor_compile_config = None

        # >>> phase1b_l1v2_inductor_compile_config <<<
        # ATOM_COMPILE_SIZES accepts:
        #   "cudagraph_capture_sizes" (legacy L1 v1 path; per-shape autotune)
        #   "[4]" or "[1,4]" or "4"  (JSON list literal OR single int; L1-v2 path)
        # ATOM_INDUCTOR_NO_AUTOTUNE=1 sets inductor_compile_config to disable
        # max_autotune (kills the lm_head numerics drift surface that killed L1 v1).
        _l1_compile_env = _os_l3.getenv("ATOM_COMPILE_SIZES", "").strip()
        if _l1_compile_env.lower() == "cudagraph_capture_sizes":
            _l3_compile_sizes = ["cudagraph_capture_sizes"]
        elif _l1_compile_env:
            try:
                import json as _json
                _parsed = _json.loads(_l1_compile_env)
                if isinstance(_parsed, int):
                    _l3_compile_sizes = [_parsed]
                elif isinstance(_parsed, list) and all(isinstance(x, int) for x in _parsed):
                    _l3_compile_sizes = _parsed
                else:
                    raise ValueError(
                        f"ATOM_COMPILE_SIZES={_l1_compile_env!r} parsed but not int or list[int]"
                    )
            except (ValueError, _json.JSONDecodeError) as _exc:
                # Try plain int
                try:
                    _l3_compile_sizes = [int(_l1_compile_env)]
                except ValueError:
                    raise ValueError(
                        f"ATOM_COMPILE_SIZES={_l1_compile_env!r} not 'cudagraph_capture_sizes', "
                        f"not JSON int/list[int], not plain int"
                    ) from _exc
        if _os_l3.getenv("ATOM_INDUCTOR_NO_AUTOTUNE", "0") == "1":
            _l1v2_inductor_compile_config = {
                "max_autotune": False,
                "max_autotune_pointwise": False,
                "max_autotune_gemm": False,
                "coordinate_descent_tuning": False,
            }
        # <<< phase1b_l1v2_inductor_compile_config <<<'''


def main():
    with open(TARGET, "r") as f:
        src = f.read()

    if PATCH_MARKER in src:
        print(f"[phase1b_l1v2] already patched, skipping")
        return 0

    if ANCHOR not in src:
        print(f"[phase1b_l1v2] FATAL: anchor not found", file=sys.stderr)
        sys.exit(1)
    if src.count(ANCHOR) != 1:
        print(
            f"[phase1b_l1v2] FATAL: anchor matched {src.count(ANCHOR)} times",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.exists(BACKUP):
        with open(BACKUP, "w") as f:
            f.write(src)
        print(f"[phase1b_l1v2] backup: {BACKUP}")

    new_src = src.replace(ANCHOR, REPLACEMENT, 1)
    with open(TARGET, "w") as f:
        f.write(new_src)
    print(f"[phase1b_l1v2] wrote {TARGET}")

    try:
        py_compile.compile(TARGET, doraise=True)
        print(f"[phase1b_l1v2] py_compile OK")
    except py_compile.PyCompileError as e:
        print(f"[phase1b_l1v2] FATAL py_compile: {e}", file=sys.stderr)
        with open(BACKUP, "r") as f:
            with open(TARGET, "w") as g:
                g.write(f.read())
        print(f"[phase1b_l1v2] ROLLED BACK", file=sys.stderr)
        sys.exit(1)

    # Now patch the _cc_kwargs construction to wire inductor_compile_config
    with open(TARGET, "r") as f:
        src2 = f.read()
    SECOND_ANCHOR = '''        _cc_kwargs = dict(

            level=kwargs.pop("level"),

            compile_sizes=_l3_compile_sizes,'''
    SECOND_REPL = '''        _cc_kwargs = dict(

            level=kwargs.pop("level"),

            compile_sizes=_l3_compile_sizes,

            **(
                {"inductor_compile_config": _l1v2_inductor_compile_config}
                if _l1v2_inductor_compile_config is not None
                else {}
            ),'''
    if SECOND_ANCHOR not in src2:
        print(f"[phase1b_l1v2] FATAL: _cc_kwargs anchor not found", file=sys.stderr)
        # Rollback first patch
        with open(BACKUP, "r") as f:
            with open(TARGET, "w") as g:
                g.write(f.read())
        sys.exit(1)
    new_src2 = src2.replace(SECOND_ANCHOR, SECOND_REPL, 1)
    with open(TARGET, "w") as f:
        f.write(new_src2)
    print(f"[phase1b_l1v2] _cc_kwargs wired")

    try:
        py_compile.compile(TARGET, doraise=True)
        print(f"[phase1b_l1v2] py_compile OK (post _cc_kwargs)")
    except py_compile.PyCompileError as e:
        print(f"[phase1b_l1v2] FATAL py_compile post: {e}", file=sys.stderr)
        with open(BACKUP, "r") as f:
            with open(TARGET, "w") as g:
                g.write(f.read())
        print(f"[phase1b_l1v2] ROLLED BACK", file=sys.stderr)
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# Phase 11 v3.1b: single-constant change. Inside-thinking top-N: 10 -> 12.
# No kernel structure change vs v3. Outside thinking still top-N=8.
# Cheapest possible accept-rate probe.
import py_compile, shutil, sys

RS_PATH = "/app/ATOM/atom/model_ops/rejection_sampler.py"
RS_BAK = RS_PATH + ".pre_v31b_top_n_12"
shutil.copyfile(RS_PATH, RS_BAK)
src = open(RS_PATH).read()

old = """PHASE_RELAXED_TOP_N = 10  # TRT-LLM relaxed_topk (inside thinking)
PHASE_STRICT_TOP_N = 8     # baseline-equivalent top-N (outside thinking) - matches RELAXED_TOP_N=8
PHASE_RELAXED_DELTA = 0.6  # TRT-LLM relaxed_delta"""

new = """PHASE_RELAXED_TOP_N = 12  # v3.1b: single-constant probe (was 10 in v3)
PHASE_STRICT_TOP_N = 8     # baseline-equivalent top-N (outside thinking) - matches RELAXED_TOP_N=8
PHASE_RELAXED_DELTA = 0.6  # TRT-LLM relaxed_delta"""

if old not in src:
    sys.exit("ERR: PHASE_RELAXED_TOP_N anchor not found")
src = src.replace(old, new, 1)

open(RS_PATH, "w").write(src)
py_compile.compile(RS_PATH, doraise=True)
print("OK v3.1b applied: PHASE_RELAXED_TOP_N 10 -> 12 (single-constant change)")
print("Backup: " + RS_BAK)

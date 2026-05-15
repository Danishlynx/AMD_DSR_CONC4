#!/usr/bin/env python3
"""
AutoKernel — Amdahl-rank bottlenecks on the locked baseline.

Approach:
  1. Profile a short representative bench inside the container with rocprofv2 (or rocprof-fallback).
  2. Aggregate kernel time by op-name (longest-prefix string match against a known op list).
  3. Compute Amdahl headroom per op: self_share * (1 - achievable_speedup_estimate).
  4. Emit JSON ranked by headroom; top entries become GEAK-HIP's task queue.

Usage (intended to be called inside the locked container by orchestrator):
  python3 /tmp/agents/autokernel_run.py \
    --baseline-prompt-cmd "python3 -c 'import requests, json; ...'" \
    --num-prompts 10 \
    --out /tmp/runs/dsr1/<ts>/autokernel.json

Output JSON shape:
  {
    "captured_at": "2026-04-27T13:00:00Z",
    "total_ms": 12345.6,
    "ops_ranked": [
      {"op": "fused_a_qkv_proj", "self_ms_total": 832.1, "self_share_pct": 6.7,
       "amdahl_headroom_pct": 4.3, "shape_hint": "M=32 N=2112 K=7168",
       "current_kernel": "aiter::gemm_a16wfp4_preshuffle"},
      ...
    ],
    "top_3_for_geak": ["fused_a_qkv_proj", "moe_gemm2_atomic", "mla_decode"]
  }

DEPENDENCIES INSIDE CONTAINER:
  - rocprofv2 (ships with ROCm 7.x)
  - python3 with json/csv stdlib
  - server already booted on port 8890 (the foreman ensures this in Stage 0)

NOTE: this is the v1 skeleton. v1 captures kernel-level data only (no MFMA-cycle
estimation). v2 will add roofline-based achievable_speedup_estimate from
hardware peak. For now we use a conservative 0.3× speedup ceiling for any op
that's >=80% of peak utilization, 0.5× for 50-80%, 0.7× for <50%.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

# Known op-name patterns we care about (shipped with the project).
# Ordered most-specific first; first match wins.
OP_PATTERNS = [
    ("fused_a_qkv_proj",     ["fused_qkv_a_proj", "_fuse_qkv_a_proj"]),
    ("moe_gemm2_atomic",     ["moe_stage2", "moe2_t32x128x256", "moe_ck2stages"]),
    ("moe_gemm1_dispatch",   ["moe_stage1", "moe1_t32x128x256"]),
    ("moe_sorting",          ["moe_sorting"]),
    ("rmsnorm_quant_fused",  ["fused_rms_mxfp4_quant", "_fuse_rmsnorm_fp4"]),
    ("mla_decode",           ["mla_decode", "fwd_decode_h32_fp8"]),
    ("ar_rmsnorm_fused",     ["fused_allreduce_rmsnorm", "fused_ar_rms"]),
    ("eagle_drafter",        ["eagle.propose", "drafter_loop"]),
    ("rejection_sampler",    ["rejection_sampler", "relaxed_mtp"]),
    ("attention_qk",         ["fused_qk_rmsnorm"]),
    ("ck_gemm_misc",         ["ck_gemm_a", "gemm_a16wfp4_preshuffle"]),
]


def categorize(kernel_name: str) -> str:
    name_lower = kernel_name.lower()
    for op_label, patterns in OP_PATTERNS:
        for p in patterns:
            if p.lower() in name_lower:
                return op_label
    return "uncategorized"


def amdahl_ceiling(self_share_pct: float, util_pct: float = 50.0) -> float:
    """
    Conservative achievable-speedup estimate × Amdahl bound.

    util_pct is hardware utilization for that op. v1 has no telemetry for
    actual util, so we use 50% as a default. v2 wires rocprofv2's
    SQ_ACCUM_PREV_HIRES utilization counter.
    """
    if util_pct >= 80:
        achievable = 0.3   # already near peak; little room
    elif util_pct >= 50:
        achievable = 0.5
    else:
        achievable = 0.7
    # Amdahl: max overall speedup if op_share takes 0 = 1 / (1 - share)
    # Headroom % = self_share * achievable
    return self_share_pct * achievable


def run_rocprof(profile_cmd: list, out_csv: Path) -> int:
    """Run rocprofv2 over the given command, write CSV to out_csv. Return exit code."""
    cmd = ["rocprofv2", "-o", str(out_csv), "--"] + profile_cmd
    print(f"[autokernel] running: {' '.join(shlex.quote(c) for c in cmd)}", file=sys.stderr)
    return subprocess.run(cmd, check=False).returncode


def parse_rocprof_csv(csv_path: Path) -> list:
    """Parse rocprofv2 CSV and return [(kernel_name, duration_ns), ...]."""
    import csv
    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            kn = r.get("KernelName") or r.get("kernel_name") or ""
            dn = r.get("DurationNs") or r.get("duration_ns") or "0"
            try:
                rows.append((kn, int(dn)))
            except ValueError:
                pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-prompt-cmd", required=True,
                    help="Shell command that drives the server (e.g. a Python script that POSTs N prompts).")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    ap.add_argument("--csv-tmp", default="/tmp/autokernel_rocprof.csv",
                    help="Where rocprofv2 writes its raw CSV.")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_tmp = Path(args.csv_tmp)
    if csv_tmp.exists():
        csv_tmp.unlink()

    cmd_list = ["bash", "-c", args.baseline_prompt_cmd]
    rc = run_rocprof(cmd_list, csv_tmp)
    if rc != 0:
        print(f"[autokernel] WARN rocprofv2 exit {rc}", file=sys.stderr)

    if not csv_tmp.exists():
        print(f"[autokernel] ERROR no CSV produced at {csv_tmp}", file=sys.stderr)
        out.write_text(json.dumps({"error": "rocprofv2 produced no CSV", "ops_ranked": []}))
        sys.exit(2)

    rows = parse_rocprof_csv(csv_tmp)
    if not rows:
        out.write_text(json.dumps({"error": "rocprofv2 CSV empty", "ops_ranked": []}))
        sys.exit(2)

    # Aggregate
    total_ns = sum(d for _, d in rows)
    by_op = {}
    for kname, dur in rows:
        op = categorize(kname)
        by_op.setdefault(op, {"self_ns": 0, "kernels": set()})
        by_op[op]["self_ns"] += dur
        by_op[op]["kernels"].add(kname)

    ranked = []
    for op, data in by_op.items():
        share_pct = (data["self_ns"] / total_ns) * 100 if total_ns > 0 else 0
        headroom = amdahl_ceiling(share_pct)
        ranked.append({
            "op": op,
            "self_ms_total": data["self_ns"] / 1e6,
            "self_share_pct": round(share_pct, 2),
            "amdahl_headroom_pct": round(headroom, 2),
            "kernels_observed": sorted(list(data["kernels"]))[:5],  # top 5 representative names
        })

    ranked.sort(key=lambda r: -r["amdahl_headroom_pct"])

    result = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_ms": total_ns / 1e6,
        "ops_ranked": ranked,
        "top_3_for_geak": [r["op"] for r in ranked[:3] if r["op"] != "uncategorized"],
    }
    out.write_text(json.dumps(result, indent=2))
    print(f"[autokernel] wrote {out}", file=sys.stderr)
    print(json.dumps(result["top_3_for_geak"]))


if __name__ == "__main__":
    main()

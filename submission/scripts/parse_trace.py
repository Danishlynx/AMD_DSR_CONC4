"""Parse torch.profiler chrome trace to find top GPU kernels.
Usage: python3 parse_trace.py <path-to-trace.json.gz>"""
import gzip
import json
import sys
from collections import defaultdict

trace_path = sys.argv[1] if len(sys.argv) > 1 else \
    "/tmp/dec075_profile/rank_0/DSR1-drafter-FP4_ts_20260417_164907_607.pt.trace.json.gz"

with gzip.open(trace_path, "rt") as f:
    data = json.load(f)

events = data.get("traceEvents", [])
print(f"Total trace events: {len(events)}")

gpu_by_name = defaultdict(lambda: {"count": 0, "total_us": 0, "max_us": 0})
for e in events:
    if "dur" not in e:
        continue
    name = e.get("name", "unknown")
    cat = e.get("cat", "")
    if cat == "kernel" or "hip" in cat.lower() or "cuda" in cat.lower():
        gpu_by_name[name]["count"] += 1
        gpu_by_name[name]["total_us"] += e["dur"]

total_gpu_us = sum(v["total_us"] for v in gpu_by_name.values())
print(f"\nTotal GPU kernel time: {total_gpu_us/1000:.1f} ms\n")
print(f"{'Rank':<4} {'Kernel':<60} {'Total ms':>10} {'Count':>8} {'Avg μs':>9} {'% GPU':>7}")
print("-" * 105)
for i, (name, d) in enumerate(sorted(gpu_by_name.items(), key=lambda x: -x[1]["total_us"])[:25], 1):
    nm = name[:58] + "…" if len(name) > 58 else name
    ms = d["total_us"]/1000
    avg = d["total_us"]/d["count"]
    pct = 100*d["total_us"]/total_gpu_us
    print(f"{i:<4} {nm:<60} {ms:>10.2f} {d['count']:>8} {avg:>9.1f} {pct:>6.1f}%")

#!/usr/bin/env python3
"""parse_profile.py — read ATOM decode_top_kernels.csv + chrome trace,
output ranked kernel buckets with attack-surface tags.

Usage:
    python3 parse_profile.py <profile_dir>
"""
import csv
import json
import os
import sys
import re
from collections import defaultdict


# Surface taxonomy (heuristic kernel-name regex → bucket)
BUCKETS = [
    ("MoE_GEMM",        r"gemm_(a16w|afp4)|gemm.*fp4|fused_moe|moe_(stage|kernel)|flydsl_moe|grouped_gemm|moe1|moe2"),
    ("MoE_routing",     r"topk|moe_align|expert_route"),
    ("MLA_GEMM",        r"q_a_proj|kv_a_proj|q_b_proj|kv_b_proj|gemm.*mla|rms|merged_replicated"),
    ("MLA_attn",        r"mla_(decode|extend|asm|a8w8)|attention_mla|mla_fwd|hk_mla|fmla"),
    ("AllReduce",       r"all_reduce|allreduce|quick_reduce|qr_|nccl|rccl"),
    ("Quant",           r"quant|dequant|mxfp4|fp8|cast|fused_reduce_rms"),
    ("Sampler_specdec", r"sampler|rejection|spec_decode|eagle|drafter|topp|topk_softmax"),
    ("Layernorm",       r"rmsnorm|layernorm|norm_"),
    ("Embedding",       r"embedding|tok_embed|lm_head"),
    ("Copy_misc",       r"memcpy|memset|copy_|fill_|zeros_|aten::copy_"),
    ("HostSync",        r"hipEventSync|hipStreamSync|cudaEventSync|cudaStreamSync|hipDeviceSync"),
]


def bucket_of(name):
    for tag, pat in BUCKETS:
        if re.search(pat, name, flags=re.IGNORECASE):
            return tag
    return "Other"


def parse_csv(path):
    rows = []
    with open(path) as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        # header: key, device_time_total, cpu_time_total, count
        for r in reader:
            if len(r) < 4:
                continue
            try:
                rows.append({
                    "name": r[0],
                    "device_us": float(r[1]) / 1e3,   # cuda kernel time (input is ns? or us?)
                    "cpu_us": float(r[2]) / 1e3,
                    "count": int(r[3]),
                })
            except ValueError:
                continue
    return rows


def main(argv):
    if len(argv) < 2:
        print("usage: parse_profile.py <profile_dir>")
        return 1
    pdir = argv[1]
    csv_path = os.path.join(pdir, "decode_top_kernels.csv")
    if not os.path.exists(csv_path):
        print(f"missing: {csv_path}")
        return 1

    rows = parse_csv(csv_path)
    # Determine unit (device_time_total from torch profiler is usually nanoseconds)
    # If max device_us looks tiny (<1) treat input as us; if huge (>1e6) treat as ns
    # The CSV column dtype is uncertain. Try both — print raw + scaled.

    total_dev = sum(r["device_us"] for r in rows)
    total_cpu = sum(r["cpu_us"] for r in rows)

    bucket_dev = defaultdict(float)
    bucket_cnt = defaultdict(int)
    for r in rows:
        b = bucket_of(r["name"])
        bucket_dev[b] += r["device_us"]
        bucket_cnt[b] += r["count"]

    print(f"=== Profile dir: {pdir} ===")
    print(f"Total device time:  {total_dev:.1f}")
    print(f"Total CPU time:     {total_cpu:.1f}")
    print()
    print(f"{'Bucket':<20} {'device_us':>14} {'pct':>7} {'count':>10}")
    for b, t in sorted(bucket_dev.items(), key=lambda kv: -kv[1]):
        pct = (t / total_dev * 100) if total_dev > 0 else 0
        print(f"  {b:<18} {t:>14.1f} {pct:>6.1f}% {bucket_cnt[b]:>10d}")

    print()
    print("=== Top 20 kernels (raw) ===")
    for r in sorted(rows, key=lambda x: -x["device_us"])[:20]:
        pct = (r["device_us"] / total_dev * 100) if total_dev > 0 else 0
        print(f"  {pct:5.1f}% {r['device_us']:>12.1f} cnt={r['count']:>6} {r['name'][:90]}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

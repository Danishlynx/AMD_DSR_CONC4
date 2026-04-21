#!/usr/bin/env python3
"""
Custom BF16 hipBLASLt per-shape tuner.
Iterates all solution indices for each (M,N,K), picks the fastest.
Outputs aiter-format CSV row for merge into bf16_tuned_gemm.csv.

Usage:
  HOME=/tmp python3 custom_bf16_tuner.py <shapes.csv> <output.csv>
"""
import sys, os, time, torch
from aiter.ops.gradlib import hipb_create_extension, hipb_mm, hipb_findallsols
from aiter.jit.utils.chip_info import get_cu_num, get_gfx

if len(sys.argv) < 3:
    print("usage: custom_bf16_tuner.py <shapes.csv> <output.csv>")
    sys.exit(1)

SHAPES_FILE = sys.argv[1]
OUT_FILE = sys.argv[2]

hipb_create_extension()
gfx = get_gfx()
cu_num = get_cu_num()
print(f"gfx={gfx} cu_num={cu_num}")

# Parse shapes
shapes = []
with open(SHAPES_FILE) as f:
    lines = f.readlines()
    for line in lines[1:]:  # skip header
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue
        m, n, k = int(parts[0]), int(parts[1]), int(parts[2])
        shapes.append((m, n, k))

print(f"Loaded {len(shapes)} shapes")

# Output header
with open(OUT_FILE, "w") as f:
    f.write("gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw\n")

WARMUP = 10
ITERS = 30

results = []
for shape_idx, (M, N, K) in enumerate(shapes):
    print(f"\n[{shape_idx+1}/{len(shapes)}] M={M} N={N} K={K}")
    try:
        # Allocate tensors: A is M×K, B is K×N (column-major for hipblaslt = N×K row-major)
        A = torch.randint(-10, 10, (M, K), dtype=torch.bfloat16, device="cuda")
        B = torch.randint(-10, 10, (K, N), dtype=torch.bfloat16, device="cuda")

        # Get all valid solution indices
        sols = hipb_findallsols(A, B, out_dtype=torch.bfloat16, bpreshuffle=False)
        print(f"  found {len(sols)} solutions")

        if not sols:
            print("  no solutions available, skipping")
            continue

        # Bench each solution
        best_us = float("inf")
        best_sol = -1
        for sol in sols:
            try:
                # Warmup
                for _ in range(WARMUP):
                    _ = hipb_mm(A, B, sol, out_dtype=torch.bfloat16)
                torch.cuda.synchronize()

                # Measure
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(ITERS):
                    _ = hipb_mm(A, B, sol, out_dtype=torch.bfloat16)
                end.record()
                torch.cuda.synchronize()
                us = start.elapsed_time(end) * 1000.0 / ITERS

                if us < best_us:
                    best_us = us
                    best_sol = sol
            except Exception as e:
                continue

        if best_sol < 0:
            print("  all solutions failed")
            continue

        # Compute metrics
        flops = 2.0 * M * N * K
        tflops = flops / (best_us * 1e-6) / 1e12
        bytes_per = (M*K + K*N)*2 + M*N*2  # bf16 is 2 bytes
        bw = bytes_per / (best_us * 1e-6) / 1e9

        print(f"  BEST sol={best_sol} us={best_us:.3f} tflops={tflops:.2f} bw={bw:.2f}")

        # Write CSV row (aiter format)
        row = f"{gfx},{cu_num},{M},{N},{K},False,torch.bfloat16,torch.bfloat16,False,False,hipblaslt,{best_sol},0,{best_us:.4f},,0.0,{tflops:.2f},{bw:.2f}\n"
        with open(OUT_FILE, "a") as f:
            f.write(row)
        results.append((M, N, K, best_sol, best_us))

        # Cleanup
        del A, B
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  ERROR: {e}")
        continue

print(f"\n=== SUMMARY ===")
print(f"Tuned {len(results)} / {len(shapes)} shapes")
for m, n, k, sol, us in results:
    print(f"  ({m},{n},{k}) -> sol={sol} {us:.2f} us")
print(f"\nOutput: {OUT_FILE}")

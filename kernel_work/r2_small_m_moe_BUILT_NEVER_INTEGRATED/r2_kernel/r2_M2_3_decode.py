#!/usr/bin/env python3
import torch
out1 = torch.load("/tmp/r2_m2_3_probe1_out.pt")
out2 = torch.load("/tmp/r2_m2_3_probe2_out.pt")

print("=== Probe 1 (A varies with m, B all 1) ===")
print("D[m,n] should be 128*fp4_val(m%8)")
print("FP4_VALS = [0, 0.5, 1, 1.5, 2, 3, 4, 6]")
print("Expected D candidates: [0, 64, 128, 192, 256, 384, 512, 768]")
print()
print("All unique values seen in probe1:", sorted(set(out1.flatten().tolist())))
print("All unique values seen in probe2:", sorted(set(out2.flatten().tolist())))
print()
print("Probe1 first 32 lanes (4 outputs per lane):")
for l in range(32):
    print(f"L{l:2d}: " + " ".join(f"{v:>7.1f}" for v in out1[l].tolist()))
print()
print("Probe2 first 32 lanes:")
for l in range(32):
    print(f"L{l:2d}: " + " ".join(f"{v:>7.1f}" for v in out2[l].tolist()))

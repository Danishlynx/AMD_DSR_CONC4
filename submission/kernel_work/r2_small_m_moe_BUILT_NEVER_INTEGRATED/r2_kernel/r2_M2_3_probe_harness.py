#!/usr/bin/env python3
# R2-C M2.3 step 1: probe to empirically derive D matrix lane->(m,n) mapping.
#
# Strategy:
#   - Set A[m, k] = encoded_m_value (for all k) — gives D[m,n] depending only on m
#   - Set B[n, k] = 1.0 for all n, k
#   - D[m, n] = sum_k(A[m,k] * B[n,k] * 1) = K * encoded_m_value
#   - Each lane's 4 output floats reveal which 4 (m,n) entries it holds
#
# FP4 E2M1 encodings: {0, 0.5, 1, 1.5, 2, 3, 4, 6} (positive nibbles 0..7).
# Use index i = 0..7 to pick encoded value; D[m,n] = 128 * fp4_val(i_for_m).
#
# Probe 1: A[m,k] = fp4_val(m & 7) (m from 0..15 → cycles through values)
#          B[n,k] = 1.0
#          D[m,n] = 128 * fp4_val(m & 7)
#          So per-lane output reveals m via the value (modulo m=0..7 vs m=8..15 disambiguation)
#
# Probe 2: A[m,k] = 1.0
#          B[n,k] = fp4_val(n & 7) (n from 0..15)
#          D[m,n] = 128 * fp4_val(n & 7)
#          So per-lane output reveals n.

import ctypes, os, sys, torch
SO_PATH = "/tmp/r2_m2_3_probe.so"

print("=== R2-C M2.3 probe: derive D matrix lane->(m,n) mapping ===")

if not os.path.exists(SO_PATH):
    sys.exit(f"ERR {SO_PATH} not found")
lib = ctypes.CDLL(SO_PATH)
launcher = lib.r2_m2_3_probe_launch
launcher.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
launcher.restype = ctypes.c_int

# fp4 E2M1 encoding: nibble -> value (positive)
# Standard OCP MXFP4: indices 0..7 -> [0, 0.5, 1, 1.5, 2, 3, 4, 6]
FP4_VALS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

def fp4_byte_with_value(idx_lo, idx_hi):
    """Pack two fp4 nibbles (positive only) into a byte."""
    return (idx_lo & 0x0F) | ((idx_hi & 0x0F) << 4)

# ===== Probe 1: A varies with m, B all 1.0 =====
# Goal: D[m,n] = 128 * fp4_val(m % 8)
A_fp4 = torch.zeros(16, 64, dtype=torch.uint8, device="cuda")
B_fp4 = torch.zeros(16, 64, dtype=torch.uint8, device="cuda")

for m in range(16):
    val_idx = m & 7  # 0..7
    byte = fp4_byte_with_value(val_idx, val_idx)  # both nibbles same value
    A_fp4[m] = byte

# B all 1.0 -> nibble = 2 (index of 1.0 in FP4_VALS)
B_fp4[:] = fp4_byte_with_value(2, 2)

out = torch.zeros(64, 4, dtype=torch.float32, device="cuda")
torch.cuda.synchronize()
ret = launcher(A_fp4.data_ptr(), B_fp4.data_ptr(), out.data_ptr(), None)
torch.cuda.synchronize()
if ret != 0:
    sys.exit(f"FAIL: probe1 launcher returned {ret}")

print("\n=== Probe 1: A[m] = fp4_val(m%8), B all 1.0 ===")
print("Expected D[m,n] = 128 * fp4_val(m%8)")
print("Expected values for m%8 in 0..7: ", [128 * v for v in FP4_VALS])

# Decode: for each lane, what value does each of its 4 output floats have?
out_cpu = out.cpu()
print("\nLane | C_acc[0] | C_acc[1] | C_acc[2] | C_acc[3]")
print("-" * 60)
for lane in range(64):
    vals = out_cpu[lane].tolist()
    print(f" {lane:3d} | {vals[0]:>8.1f} | {vals[1]:>8.1f} | {vals[2]:>8.1f} | {vals[3]:>8.1f}")

# Save raw data for further analysis
torch.save(out_cpu, "/tmp/r2_m2_3_probe1_out.pt")

# ===== Probe 2: B varies with n, A all 1.0 =====
A_fp4_2 = torch.full((16, 64), fp4_byte_with_value(2, 2), dtype=torch.uint8, device="cuda")
B_fp4_2 = torch.zeros(16, 64, dtype=torch.uint8, device="cuda")
for n in range(16):
    val_idx = n & 7
    B_fp4_2[n] = fp4_byte_with_value(val_idx, val_idx)

out2 = torch.zeros(64, 4, dtype=torch.float32, device="cuda")
torch.cuda.synchronize()
ret = launcher(A_fp4_2.data_ptr(), B_fp4_2.data_ptr(), out2.data_ptr(), None)
torch.cuda.synchronize()
if ret != 0:
    sys.exit(f"FAIL: probe2 launcher returned {ret}")

print("\n\n=== Probe 2: A all 1.0, B[n] = fp4_val(n%8) ===")
print("Expected D[m,n] = 128 * fp4_val(n%8)")
out2_cpu = out2.cpu()
print("\nLane | C_acc[0] | C_acc[1] | C_acc[2] | C_acc[3]")
print("-" * 60)
for lane in range(64):
    vals = out2_cpu[lane].tolist()
    print(f" {lane:3d} | {vals[0]:>8.1f} | {vals[1]:>8.1f} | {vals[2]:>8.1f} | {vals[3]:>8.1f}")

torch.save(out2_cpu, "/tmp/r2_m2_3_probe2_out.pt")

# ===== Decode lane->(m,n) mapping =====
print("\n\n=== DECODING ===")
print("For each lane, infer (m, n) for each of its 4 output positions.")
print("Probe 1 tells us m (via D = 128*fp4_val(m%8))")
print("Probe 2 tells us n (via D = 128*fp4_val(n%8))")

def find_m_from_value(v):
    """Given D = 128 * fp4_val(m%8), return m%8 if value matches."""
    for i, fv in enumerate(FP4_VALS):
        if abs(v - 128.0 * fv) < 0.5:
            return i
    return -1

print("\nLane | Slot | from probe1 (m%8) | from probe2 (n%8)")
print("-" * 60)
for lane in range(64):
    for slot in range(4):
        v1 = out_cpu[lane][slot].item()
        v2 = out2_cpu[lane][slot].item()
        m_mod = find_m_from_value(v1)
        n_mod = find_m_from_value(v2)
        print(f" {lane:3d} |  {slot}  |   m%8 = {m_mod:>2}   |   n%8 = {n_mod:>2}")
    print("-" * 60)

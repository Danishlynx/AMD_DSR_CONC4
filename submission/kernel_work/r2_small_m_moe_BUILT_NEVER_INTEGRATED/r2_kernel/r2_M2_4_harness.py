#!/usr/bin/env python3
# R2-C M2.4 harness: verified D-mapping + LDS load. Compare against torch FP4 dequant ref.
import ctypes, os, sys, torch
SO_PATH = "/tmp/r2_m2_4.so"

print("=== R2-C M2.4 — verified D-mapping + LDS load ===")

if not os.path.exists(SO_PATH):
    sys.exit(f"ERR {SO_PATH} not found")
lib = ctypes.CDLL(SO_PATH)
launcher = lib.r2_m2_4_launch
launcher.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
launcher.restype = ctypes.c_int

FP4_VALS_POS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
DEQUANT = torch.tensor(FP4_VALS_POS + [-v for v in FP4_VALS_POS], dtype=torch.float32, device="cuda")

def dequant_fp4(packed, M, K):
    nlo = (packed & 0x0F).long()
    nhi = ((packed >> 4) & 0x0F).long()
    out = torch.empty(M, K, dtype=torch.float32, device=packed.device)
    out[:, 0::2] = DEQUANT[nlo]
    out[:, 1::2] = DEQUANT[nhi]
    return out

# === Test 1: deterministic input (all 1.0) ===
# A[m,k] = 1.0 for all → byte = (2 | 2<<4) = 0x22
# B[n,k] = 1.0 → same
# D[m,n] should = 128 (sum of 128 K positions of 1*1)
A_fp4 = torch.full((16, 64), 0x22, dtype=torch.uint8, device="cuda")
B_fp4 = torch.full((16, 64), 0x22, dtype=torch.uint8, device="cuda")
C_bf16 = torch.zeros((4, 16), dtype=torch.bfloat16, device="cuda")

torch.cuda.synchronize()
ret = launcher(A_fp4.data_ptr(), B_fp4.data_ptr(), C_bf16.data_ptr(), None)
torch.cuda.synchronize()
print(f"Test 1 (all 1.0): hipError={ret}")
print(f"  C output:\n{C_bf16}")
expected = 128.0
actual_max = C_bf16.float().abs().max().item()
print(f"  Expected D[m,n] = {expected}, observed max = {actual_max}")
print(f"  Ratio = {actual_max / expected:.4f}")

# === Test 2: random ===
torch.manual_seed(42)
A_fp4_r = torch.randint(0, 256, (16, 64), dtype=torch.uint8, device="cuda")
B_fp4_r = torch.randint(0, 256, (16, 64), dtype=torch.uint8, device="cuda")
C_bf16_r = torch.zeros((4, 16), dtype=torch.bfloat16, device="cuda")

A_dq = dequant_fp4(A_fp4_r, 16, 128)  # [16, 128]
B_dq = dequant_fp4(B_fp4_r, 16, 128)  # [16, 128]
C_ref = torch.matmul(A_dq[:4], B_dq.T)  # [4, 16]
C_ref_bf = C_ref.to(torch.bfloat16)

torch.cuda.synchronize()
ret = launcher(A_fp4_r.data_ptr(), B_fp4_r.data_ptr(), C_bf16_r.data_ptr(), None)
torch.cuda.synchronize()
print(f"\nTest 2 (random): hipError={ret}")
print(f"  Kernel C[0]: {C_bf16_r[0]}")
print(f"  Ref    C[0]: {C_ref_bf[0]}")
print(f"  Kernel abs max: {C_bf16_r.abs().max().item():.2f}")
print(f"  Ref    abs max: {C_ref_bf.abs().max().item():.2f}")

err = (C_bf16_r.float() - C_ref_bf.float()).abs()
err_ratio = err.max().item() / max(C_ref_bf.abs().max().item(), 1e-6)
print(f"  err max: {err.max().item():.2f}")
print(f"  err mean: {err.mean().item():.2f}")
print(f"  err ratio: {err_ratio:.4f}")

if err_ratio < 1e-2:
    print("\n=== M2.4 PASS: numerics within 1% of torch ref ===")
elif err_ratio < 0.1:
    print(f"\n=== M2.4 CLOSE: numerics within 10% (err_ratio={err_ratio:.4f}) ===")
elif err_ratio < 1.0:
    print(f"\n=== M2.4 PARTIAL: numerics within 100% (err_ratio={err_ratio:.4f}) — D layout fix improved over M2.2's 24x ===")
else:
    print(f"\n=== M2.4 STILL OFF: err_ratio={err_ratio:.4f} — input layout still wrong ===")

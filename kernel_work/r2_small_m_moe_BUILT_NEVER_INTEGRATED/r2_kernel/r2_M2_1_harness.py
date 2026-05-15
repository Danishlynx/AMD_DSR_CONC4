#!/usr/bin/env python3
# R2-C M2.1 harness: launch single-tile FP4 GEMM on synthesized inputs.
# Goal: smoke test only — no parity yet (M2.1 has known WRONG output due to
# simplified LDS load layout). M2.2 fixes layout.
import ctypes, os, sys, torch

SO_PATH = "/tmp/r2_m2_1.so"

print("=== R2-C M2.1 single-tile FP4 GEMM smoke test ===")

if not os.path.exists(SO_PATH):
    sys.exit(f"ERR: {SO_PATH} not found")
lib = ctypes.CDLL(SO_PATH)

launcher = lib.r2_m2_1_launch
launcher.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,  # A_fp4, B_fp4
    ctypes.c_void_p, ctypes.c_void_p,  # Sa, Sb
    ctypes.c_void_p,                   # C_bf16
    ctypes.c_void_p,                   # stream
]
launcher.restype = ctypes.c_int

# Synthesized inputs at the M2.1 shape (M=4, N=16, K=128)
# A: [4, 64] uint8 packed FP4 (each byte = 2 fp4 elements)
# B: [16, 64] uint8 packed FP4
# Sa: [4, 4] uint8 E8M0 scales (K=128 / 32 = 4 scale groups per row)
# Sb: [16, 4] uint8
# C: [4, 16] BF16 output

torch.manual_seed(0)
A_fp4  = torch.randint(0, 256, (4, 64),  dtype=torch.uint8, device="cuda")
B_fp4  = torch.randint(0, 256, (16, 64), dtype=torch.uint8, device="cuda")
Sa     = torch.full((4, 4),  0x7f, dtype=torch.uint8, device="cuda")  # all 1.0
Sb     = torch.full((16, 4), 0x7f, dtype=torch.uint8, device="cuda")
C_bf16 = torch.zeros((4, 16), dtype=torch.bfloat16, device="cuda")

print(f"A_fp4.shape={tuple(A_fp4.shape)} B_fp4.shape={tuple(B_fp4.shape)}")
print(f"Sa.shape={tuple(Sa.shape)} Sb.shape={tuple(Sb.shape)}")
print(f"C_bf16.shape={tuple(C_bf16.shape)}")

print("Launching kernel...")
torch.cuda.synchronize()
ret = launcher(
    A_fp4.data_ptr(), B_fp4.data_ptr(),
    Sa.data_ptr(),    Sb.data_ptr(),
    C_bf16.data_ptr(), None
)
print(f"  hipError_t = {ret}")
torch.cuda.synchronize()

if ret != 0:
    sys.exit(f"\n=== M2.1 FAIL: launcher returned {ret} ===")

# Smoke checks
print(f"\nC output (first row): {C_bf16[0]}")
print(f"C abs max: {C_bf16.abs().max().item():.4f}")
print(f"C abs mean: {C_bf16.abs().mean().item():.4f}")
print(f"C nonzero count: {(C_bf16 != 0).sum().item()} / {C_bf16.numel()}")

if C_bf16.abs().max().item() == 0:
    print("\n=== M2.1 WARNING: output all zero (kernel might not be writing) ===")
elif torch.isnan(C_bf16).any() or torch.isinf(C_bf16).any():
    print("\n=== M2.1 WARNING: NaN/Inf in output ===")
else:
    print("\n=== M2.1 PASS: kernel launched, MFMA executed, output non-zero non-NaN ===")
    print("(Numerics are NOT correct yet — M2.2 fixes LDS layout for proper k-interleaved MFMA.)")

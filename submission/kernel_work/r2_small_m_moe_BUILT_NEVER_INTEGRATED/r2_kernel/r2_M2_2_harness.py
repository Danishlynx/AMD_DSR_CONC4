#!/usr/bin/env python3
# R2-C M2.2 harness: launch kernel with corrected lane mapping + compare against
# torch FP4-dequant BF16 reference GEMM.
import ctypes, os, sys, torch

SO_PATH = "/tmp/r2_m2_2.so"

print("=== R2-C M2.2 corrected-layout FP4 GEMM ===")

if not os.path.exists(SO_PATH):
    sys.exit(f"ERR: {SO_PATH} not found")
lib = ctypes.CDLL(SO_PATH)
launcher = lib.r2_m2_2_launch
launcher.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
launcher.restype = ctypes.c_int

# FP4 dequant table per OCP MXFP4 spec (E2M1 = 1 bit sign, 2 bit exp, 1 bit mantissa)
# 16 possible values for 4-bit FP4
def make_fp4_dequant_table():
    table = torch.zeros(16, dtype=torch.float32)
    # E2M1: bits = SEEM, S=sign, E=exp(2 bits), M=mantissa(1 bit)
    # Special: 000=+0, 100=-0, 001=0.5, 010=1, 011=1.5, 100=-0, 101=-0.5, etc.
    # Standard E2M1 values (subnormals + normals):
    vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    for i in range(8):
        table[i]     = vals[i]
        table[i + 8] = -vals[i]
    return table.cuda()

DEQUANT = make_fp4_dequant_table()

def dequant_fp4(packed_bytes, M, K):
    """packed_bytes: uint8 [M, K/2]. Each byte holds 2 fp4 elements (low nibble first)."""
    nibbles_lo = (packed_bytes & 0x0F).long()
    nibbles_hi = ((packed_bytes >> 4) & 0x0F).long()
    out = torch.empty(M, K, dtype=torch.float32, device=packed_bytes.device)
    out[:, 0::2] = DEQUANT[nibbles_lo]
    out[:, 1::2] = DEQUANT[nibbles_hi]
    return out

# ===== Test inputs =====
torch.manual_seed(42)
M, N, K = 16, 16, 128
A_fp4 = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device="cuda")
B_fp4 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
C_bf16 = torch.zeros((4, N), dtype=torch.bfloat16, device="cuda")

# Reference: dequant + BF16 GEMM (no scales for this test)
A_dq = dequant_fp4(A_fp4, M, K)  # [16, 128]
B_dq = dequant_fp4(B_fp4, N, K)  # [16, 128]
# C = A @ B^T (since B[N,K] and we want [M,N])
C_ref = torch.matmul(A_dq, B_dq.T)  # [16, 16] FP32
# Only first 4 M-rows are valid in our M2.2 kernel
C_ref_bf16 = C_ref[:4].to(torch.bfloat16)  # [4, 16]

# ===== Launch kernel =====
print(f"A_fp4.shape={tuple(A_fp4.shape)} B_fp4.shape={tuple(B_fp4.shape)}")
torch.cuda.synchronize()
ret = launcher(A_fp4.data_ptr(), B_fp4.data_ptr(), C_bf16.data_ptr(), None)
torch.cuda.synchronize()
print(f"hipError_t = {ret}")
if ret != 0:
    sys.exit(f"FAIL launcher returned {ret}")

# ===== Compare =====
print(f"\nC_kernel[0]: {C_bf16[0]}")
print(f"C_ref[0]:    {C_ref_bf16[0]}")
print(f"\nC_kernel abs max: {C_bf16.abs().max().item():.4f}")
print(f"C_ref abs max:    {C_ref_bf16.abs().max().item():.4f}")

if torch.isnan(C_bf16).any():
    print("\n=== FAIL: NaN in kernel output ===")
    sys.exit(1)

err = (C_bf16.float() - C_ref_bf16.float()).abs()
err_ratio = err.max().item() / max(C_ref_bf16.abs().max().item(), 1e-6)
print(f"\nerr max: {err.max().item():.4f}")
print(f"err mean: {err.mean().item():.4f}")
print(f"err ratio: {err_ratio:.4f}")

if err_ratio < 1e-2:
    print("\n=== M2.2 PASS: numerics match torch reference (err < 1%) ===")
elif err_ratio < 0.5:
    print(f"\n=== M2.2 PARTIAL: numerics close ({err_ratio*100:.1f}% err) — layout mostly right ===")
else:
    print(f"\n=== M2.2 SIGNIFICANT NUMERICS GAP: err_ratio={err_ratio:.4f} ===")
    print("Likely needs: real ds_read_b64_tr_b4 transpose + lane-shuffled D output mapping.")

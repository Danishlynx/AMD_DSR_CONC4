#!/usr/bin/env python3
# R2-C M1 harness: load the R2 skeleton .so via ctypes and launch it.
# Goal: prove kernel runs on snapshot tensors without HSA exception.
# NO parity check (M2). NO timing comparison (M3).

import ctypes, os, sys, torch

SO_PATH      = "/tmp/r2_smallm_moe_gemm2.so"
SNAPSHOT_DIR = "/tmp/r2_snapshot"

print("=== R2-C M1 harness: launch + no crash ===")

# 1. Load the shared library
if not os.path.exists(SO_PATH):
    sys.exit(f"ERR: {SO_PATH} not found. Run r2_build.sh first.")
lib = ctypes.CDLL(SO_PATH)
print(f"Loaded {SO_PATH}")

# 2. Bind the launcher signature
# extern "C" hipError_t r2_launch_smallm_moe_gemm2_skeleton(
#   const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*,
#   const int*, const int*, const int*, const float*, __hip_bfloat16*,
#   int, int, int, int, hipStream_t)
launcher = lib.r2_launch_smallm_moe_gemm2_skeleton
launcher.argtypes = [
    ctypes.c_void_p,  # inter_states
    ctypes.c_void_p,  # w2
    ctypes.c_void_p,  # inter_scales
    ctypes.c_void_p,  # w2_scales
    ctypes.c_void_p,  # sorted_token_ids
    ctypes.c_void_p,  # sorted_expert_ids
    ctypes.c_void_p,  # num_valid_ids
    ctypes.c_void_p,  # topk_weights
    ctypes.c_void_p,  # output
    ctypes.c_int,     # num_tokens
    ctypes.c_int,     # hidden
    ctypes.c_int,     # intermediate
    ctypes.c_int,     # topk
    ctypes.c_void_p,  # stream
]
launcher.restype = ctypes.c_int  # hipError_t

# 3. Build minimal valid input tensors from snapshot
# (M1: kernel is a no-op, so we don't strictly need real values, but pointers must be valid GPU addresses.)
print("Loading snapshot tensors...")
hidden_states = torch.load(f"{SNAPSHOT_DIR}/hidden_states_M4.pt").cuda()  # (4, 7168) bf16
w2            = torch.load(f"{SNAPSHOT_DIR}/w2_weight.pt").cuda()         # (257, 7168, 256) fp4x2
w2_scale      = torch.load(f"{SNAPSHOT_DIR}/w2_weight_scale.pt").cuda()   # (257, 7168, 8) uint8
topk_w        = torch.load(f"{SNAPSHOT_DIR}/topk_weights.pt").cuda()      # (4, 9) f32
topk_ids      = torch.load(f"{SNAPSHOT_DIR}/topk_ids.pt").cuda()          # (4, 9) i32

# Inter_states is the M2-stage input (output of stage1). M1 placeholder: use raw bytes from hidden_states.
# In real flow, stage1 produces inter_states of shape (num_tokens*topk, intermediate/2) FP4.
inter_states = torch.zeros(4 * 9, 256, dtype=torch.uint8, device="cuda")  # placeholder
inter_scales = torch.zeros(4 * 9, 16, dtype=torch.uint8, device="cuda")   # placeholder

# Sorted IDs / num_valid (M1 placeholder)
sorted_token_ids  = torch.zeros(4 * 9, dtype=torch.int32, device="cuda")
sorted_expert_ids = torch.zeros(4 * 9, dtype=torch.int32, device="cuda")
num_valid_ids     = torch.zeros(1, dtype=torch.int32, device="cuda")

# Output: (num_tokens, hidden) BF16
output = torch.zeros(4, 7168, dtype=torch.bfloat16, device="cuda")

print(f"  hidden_states: {tuple(hidden_states.shape)} {hidden_states.dtype}")
print(f"  w2:            {tuple(w2.shape)} {w2.dtype}")
print(f"  w2_scale:      {tuple(w2_scale.shape)} {w2_scale.dtype}")
print(f"  output:        {tuple(output.shape)} {output.dtype}")

# 4. Launch the kernel
print("\nLaunching kernel...")
torch.cuda.synchronize()
ret = launcher(
    inter_states.data_ptr(),
    w2.data_ptr(),
    inter_scales.data_ptr(),
    w2_scale.data_ptr(),
    sorted_token_ids.data_ptr(),
    sorted_expert_ids.data_ptr(),
    num_valid_ids.data_ptr(),
    topk_w.data_ptr(),
    output.data_ptr(),
    4,    # num_tokens
    7168, # hidden
    512,  # intermediate
    9,    # topk
    None, # stream = default
)
print(f"  hipError_t = {ret}")
torch.cuda.synchronize()
print(f"  cudaDeviceSynchronize OK (no HSA exception)")

# 5. M1 success criterion: launcher returned 0 (hipSuccess)
if ret == 0:
    print("\n=== M1 PASS: kernel compiled, launched, returned hipSuccess ===")
    sys.exit(0)
else:
    print(f"\n=== M1 FAIL: launcher returned {ret} ===")
    sys.exit(1)

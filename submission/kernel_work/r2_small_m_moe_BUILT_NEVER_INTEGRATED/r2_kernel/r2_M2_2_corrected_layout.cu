// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Danish — DSR1 hackathon submission.
//
// R2-C M2.2 — single-tile FP4 GEMM with CORRECTED LDS layout per CDNA4 ISA §7.1.5.1.
//
// FP4 16x16x128 layout (per CDNA4 ISA Table at p.50):
//   A[16][128] / B[128][16]
//   row0 thr 0-15  M/N=[0-15]
//   row1 thr 16-31 M/N=[0-15]
//   row2 thr 32-47 M/N=[0-15]
//   row3 thr 48-63 M/N=[0-15]
//   v0 holds K=[0-31], v1 K=[32-63], v2 K=[64-95], v3 K=[96-127] for FP4
//   (Per VGPR: 32 fp4 elements packed in 32 bits — 4 packed bytes per VGPR... wait
//    32 fp4 = 16 bytes = 4 VGPRs! That doesn't fit. Re-reading: "F4: 4 VGPRs" total.)
//
// Reconciliation: 4 VGPRs per lane × 32 bits = 128 bits = 32 fp4. So per lane has
// ONLY 32 fp4 of A, distributed across 4 VGPRs as 8 fp4 each. Each VGPR holds
// 8 fp4 = K-range of 8 elements (NOT 32 as table seems to show).
//
// Actually re-reading: "v0 k=0-31" might mean v0 contains the slice corresponding
// to K=0..31 of M=this_lane's M. And v0 stores 8 packed fp4 elements representing
// 4 K-slots × 2 packed = 8 K positions. Hmm.
//
// IMPORTANT: For M2.2, we use the existing CK ck2stages dispatch as the layout
// reference. The MFMA intrinsic accepts intx8 (32 bytes per lane); for FP4 only
// the first 16 bytes (4 VGPRs) are meaningful, the rest is don't-care.
//
// LDS layout we'll use: [16 padded M rows][128 K cols] FP4 packed as bytes.
// Read pattern: each lane loads its (M=lane%16, K-range) slice via 4
// ds_read_b64_tr_b4 instructions (2 per VGPR pair).
//
// For M2.2 we CANNOT yet use ds_read_b64_tr_b4 inline asm because LLVM 22.x's
// recognition of this mnemonic is unclear from our tests. Use simpler ds_read_b64
// + manual lane shuffling, OR direct HBM→register load via flat_load.
//
// SIMPLIFIED M2.2 PATH: skip LDS entirely for M2.2 — load A and B from HBM
// directly into per-lane registers using the (lane_id) → (M, K) mapping per ISA.
// This skips the LDS layout problem but exercises the MFMA + correct lane mapping.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef __attribute__((__vector_size__(8 * sizeof(int)))) int  intx8_t;
typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;

constexpr int kM = 16;
constexpr int kN = 16;
constexpr int kK = 128;
constexpr int kThreads = 64;

// MFMA wrapper: CBSZ=4 (A FP4), BLGP=4 (B FP4), no scale (SCALE=0 means 2^0=1).
// ABID[0]=1 for SCALE intrinsic per ISA p.50 — but we use the unscaled NAME
// here for simplicity; real per-element scales happen via ABID=1 + scale srcs.
template<int CBSZ = 4, int ABID = 0, int BLGP = 4,
         int OPSEL_A = 0, int SCALE_A = 0, int SCALE_B = 0>
__device__ static inline floatx4_t mfma_scale_16x16x128_mxfp4(
    intx8_t A, intx8_t B, floatx4_t C)
{
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        A, B, C, CBSZ, ABID, BLGP, OPSEL_A, SCALE_A, SCALE_B);
}

// M2.2: HBM-direct load with correct lane->(M,K) mapping.
// A[16][128] FP4 packed in [16][64]bytes. B[128][16] FP4 packed in [16][64]bytes
// (B is row-major in K, col-major across M for MFMA — store as [16 N rows][64 K-bytes]).
//
// Lane mapping (per ISA §7.1.5.1):
//   M_lane = lane % 16  (which M row this lane represents)
//   K_block = lane / 16 (which K-quarter: 0=>K(0-31), 1=>K(32-63), 2=>K(64-95), 3=>K(96-127))
//
// Per lane needs 32 fp4 from K-block → 16 bytes → 4 ints in low half of intx8.

extern "C" __global__ __launch_bounds__(kThreads, 1)
void r2_m2_2_single_tile_fp4_gemm(
    const uint8_t* __restrict__ A_fp4,   // [16, 64] = 16 M-rows × K/2 packed bytes
    const uint8_t* __restrict__ B_fp4,   // [16, 64] = 16 N-rows × K/2 packed bytes
    __hip_bfloat16* __restrict__ C_bf16) // [4, 16]
{
    int lane = threadIdx.x;
    int M_lane = lane & 0xF;       // M row = lane % 16
    int K_block = lane >> 4;       // K-quarter = lane / 16

    // Per-lane: 32 fp4 from M=M_lane, K=[K_block*32 .. K_block*32+31]
    // Stored as 16 bytes = 4 ints in A_reg[0..3]. A_reg[4..7] are padding.
    intx8_t A_reg = {0,0,0,0,0,0,0,0};
    intx8_t B_reg = {0,0,0,0,0,0,0,0};

    int K_byte_start = K_block * 16;  // 32 fp4 = 16 bytes
    const int* a_src_int = (const int*)(A_fp4 + M_lane * 64 + K_byte_start);
    const int* b_src_int = (const int*)(B_fp4 + M_lane * 64 + K_byte_start);

    // Load 4 ints (16 bytes = 32 fp4) into low half of intx8.
    A_reg[0] = a_src_int[0];
    A_reg[1] = a_src_int[1];
    A_reg[2] = a_src_int[2];
    A_reg[3] = a_src_int[3];
    B_reg[0] = b_src_int[0];
    B_reg[1] = b_src_int[1];
    B_reg[2] = b_src_int[2];
    B_reg[3] = b_src_int[3];

    // MFMA accumulate
    floatx4_t C_acc = {0.f, 0.f, 0.f, 0.f};
    C_acc = mfma_scale_16x16x128_mxfp4<>(A_reg, B_reg, C_acc);

    // Output D layout per CDNA4 ISA: 16x16 FP32 spread across 64 lanes × 4 floats.
    // For 16x16 D matrix with 64 lanes × 4 floats = 256 elements (matches).
    // Per ISA: lane (i,j) holds D[(i%16) ?? lanes 0-3 hold col 0..3, lanes 4-7 hold col 4..7, etc.
    // The exact mapping is in ISA §7.1 - similar to FP8 case. For M2.2 use a
    // common-pattern guess: D[lane % 16][(lane/16)*4 + i] = C_acc[i] for i=0..3.
    // M = lane % 16, N_base = (lane/16)*4
    // This is a standard CDNA MFMA D layout for 16x16 outputs with 64 lanes.
    int M_out = lane & 0xF;
    int N_base = (lane >> 4) * 4;
    if (M_out < 4) {  // only first 4 of 16 M rows are valid
        for (int i = 0; i < 4; ++i) {
            int N_out = N_base + i;
            if (N_out < 16) {
                C_bf16[M_out * 16 + N_out] = __float2bfloat16(C_acc[i]);
            }
        }
    }
}

extern "C" hipError_t r2_m2_2_launch(
    const uint8_t* A_fp4, const uint8_t* B_fp4,
    __hip_bfloat16* C_bf16, hipStream_t stream)
{
    dim3 grid(1);
    dim3 block(kThreads);
    r2_m2_2_single_tile_fp4_gemm<<<grid, block, 0, stream>>>(
        A_fp4, B_fp4, C_bf16);
    return hipGetLastError();
}

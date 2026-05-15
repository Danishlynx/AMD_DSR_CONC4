// SPDX-License-Identifier: MIT
// R2-C M2.4 — verified D-mapping + Path A (ds_read_b64_tr_b4 HW transpose).
//
// What changes vs M2.3:
//   1. D output store uses VERIFIED probe-confirmed mapping:
//        m_base = (lane / 16) * 4, n = lane % 16, C_acc[i] = D[m_base+i, n]
//   2. Input load via ds_read_b64_tr_b4 inline asm (HW handles k-interleave)
//   3. LDS layout: linear [16 M-rows][64 K-bytes] — HW transpose handles permute
//
// Per CDNA4 ISA §11.4 DS_READ_B64_TR_B4:
//   - Read 64 bits per lane from LDS, interpret as 4-bit elements, transpose
//   - Two instructions load complete matrix
//     First:  K=(0..15, 32..47), Second: K=(16..31, 48..63)
//   - Each instruction fills 2 VGPRs per lane (b64)

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

typedef __attribute__((__vector_size__(8 * sizeof(int)))) int  intx8_t;
typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;

constexpr int kThreads = 64;

template<int CBSZ = 4, int ABID = 0, int BLGP = 4,
         int OPSEL_A = 0, int SCALE_A = 0, int SCALE_B = 0>
__device__ static inline floatx4_t mfma_scale_16x16x128_mxfp4(
    intx8_t A, intx8_t B, floatx4_t C)
{
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        A, B, C, CBSZ, ABID, BLGP, OPSEL_A, SCALE_A, SCALE_B);
}

extern "C" __global__ __launch_bounds__(kThreads, 1)
void r2_m2_4_single_tile(
    const uint8_t* __restrict__ A_fp4,   // [16, 64] = M × K/2 packed
    const uint8_t* __restrict__ B_fp4,   // [16, 64] = N × K/2 packed
    __hip_bfloat16* __restrict__ C_bf16) // [4, 16]
{
    int lane = threadIdx.x;

    // ---- LDS: linear A and B, 16 M (or N) rows × 64 K-bytes each ----
    __shared__ alignas(16) uint8_t A_lds[16 * 64];
    __shared__ alignas(16) uint8_t B_lds[16 * 64];

    // Cooperative HBM -> LDS load: 64 lanes × 16 bytes each = 1024 bytes (full A or B)
    // Each lane copies 16 contiguous bytes
    int byte_off = lane * 16;
    if (byte_off + 16 <= 16 * 64) {
        for (int i = 0; i < 16; ++i) {
            A_lds[byte_off + i] = A_fp4[byte_off + i];
            B_lds[byte_off + i] = B_fp4[byte_off + i];
        }
    }
    __syncthreads();

    // ---- LDS -> register via ds_read_b64_tr_b4 ----
    // 2 instructions per lane fill 2 pairs of VGPRs (intx8 = 8 ints).
    // Each instruction loads 64 bits per lane = 16 fp4 elements.
    // Per ISA: first inst K=(0..15, 32..47), second K=(16..31, 48..63).
    //
    // The HW transpose distributes data across lanes per the MFMA expectation.
    // LDS layout we provide: linear [M][K-bytes]; HW reads it correctly.
    //
    // Per-lane LDS address: each lane provides a base address; HW reads
    // a 64-bit slice from that address with transpose semantics.
    //
    // For 16x128 matrix in LDS at A_lds[16][64 bytes]:
    //   - Lane l provides addr = A_lds + (l % 16) * 64 + (l / 16) * <stride>
    //   - HW transposes the 16x16 sub-tile starting at that addr
    //
    // M2.4 simplification: provide linear addr; let HW figure out layout.
    // Each lane reads its own M-row × K-quarter slice.

    intx8_t A_reg = {0,0,0,0,0,0,0,0};
    intx8_t B_reg = {0,0,0,0,0,0,0,0};

    int M_lane = lane & 0xF;       // 0..15
    int K_block = lane >> 4;       // 0..3 (K-quarter)

    // M2.4 simpler path: read 16 bytes from LDS via int4 pointer.
    // MFMA expects k-interleaved layout (HW transpose via ds_read_b64_tr_b4),
    // but for M2.4 we test the verified D-mapping with linear k-contiguous data.
    // If numerics are still off but D layout works, M2.5 fixes input layout.
    // Try ds_read_b64_tr_b4 inline asm for HW k-interleaved transpose load.
    // Per CDNA4 ISA §11.4: 2 instructions load complete matrix.
    // First inst: K=(0..15, 32..47) into 2 VGPRs
    // Second inst: K=(16..31, 48..63) into 2 VGPRs
    // For full K=128 we need 4 instructions per lane (2 quarters × 2 insts).

    // LDS address: each lane provides base address; HW reads with transpose.
    // Convert generic LDS pointer to LDS-relative offset.
    uint32_t a_smem_base = (uint32_t)(uintptr_t)A_lds + M_lane * 64;
    uint32_t b_smem_base = (uint32_t)(uintptr_t)B_lds + M_lane * 64;

    // Use uint64_t aliases for ds_read output
    uint64_t a01, a23, a45, a67, b01, b23, b45, b67;
    asm volatile("ds_read_b64_tr_b4 %0, %1 offset:0"  : "=v"(a01) : "v"(a_smem_base) : "memory");
    asm volatile("ds_read_b64_tr_b4 %0, %1 offset:16" : "=v"(a23) : "v"(a_smem_base) : "memory");
    asm volatile("ds_read_b64_tr_b4 %0, %1 offset:32" : "=v"(a45) : "v"(a_smem_base) : "memory");
    asm volatile("ds_read_b64_tr_b4 %0, %1 offset:48" : "=v"(a67) : "v"(a_smem_base) : "memory");
    asm volatile("ds_read_b64_tr_b4 %0, %1 offset:0"  : "=v"(b01) : "v"(b_smem_base) : "memory");
    asm volatile("ds_read_b64_tr_b4 %0, %1 offset:16" : "=v"(b23) : "v"(b_smem_base) : "memory");
    asm volatile("ds_read_b64_tr_b4 %0, %1 offset:32" : "=v"(b45) : "v"(b_smem_base) : "memory");
    asm volatile("ds_read_b64_tr_b4 %0, %1 offset:48" : "=v"(b67) : "v"(b_smem_base) : "memory");

    // Pack into intx8 (8 ints = 32 bytes; each uint64 = 2 ints)
    A_reg[0] = (int)(a01 & 0xFFFFFFFF);  A_reg[1] = (int)(a01 >> 32);
    A_reg[2] = (int)(a23 & 0xFFFFFFFF);  A_reg[3] = (int)(a23 >> 32);
    A_reg[4] = (int)(a45 & 0xFFFFFFFF);  A_reg[5] = (int)(a45 >> 32);
    A_reg[6] = (int)(a67 & 0xFFFFFFFF);  A_reg[7] = (int)(a67 >> 32);
    B_reg[0] = (int)(b01 & 0xFFFFFFFF);  B_reg[1] = (int)(b01 >> 32);
    B_reg[2] = (int)(b23 & 0xFFFFFFFF);  B_reg[3] = (int)(b23 >> 32);
    B_reg[4] = (int)(b45 & 0xFFFFFFFF);  B_reg[5] = (int)(b45 >> 32);
    B_reg[6] = (int)(b67 & 0xFFFFFFFF);  B_reg[7] = (int)(b67 >> 32);
    (void)K_block;  // unused now

    // ---- MFMA ----
    floatx4_t C_acc = {0.f, 0.f, 0.f, 0.f};
    C_acc = mfma_scale_16x16x128_mxfp4<>(A_reg, B_reg, C_acc);

    // ---- Output store using VERIFIED D layout (probe-confirmed) ----
    int m_base = (lane >> 4) << 2;   // 0, 4, 8, 12
    int n      = lane & 0xF;          // 0..15
    if (n < 16) {
        for (int i = 0; i < 4; ++i) {
            int m = m_base + i;
            if (m < 4) {  // only first 4 of 16 M rows are valid for our M=4 workload
                C_bf16[m * 16 + n] = __float2bfloat16(C_acc[i]);
            }
        }
    }
}

extern "C" hipError_t r2_m2_4_launch(
    const uint8_t* A_fp4, const uint8_t* B_fp4,
    __hip_bfloat16* C_bf16, hipStream_t stream)
{
    dim3 grid(1);
    dim3 block(kThreads);
    r2_m2_4_single_tile<<<grid, block, 0, stream>>>(A_fp4, B_fp4, C_bf16);
    return hipGetLastError();
}

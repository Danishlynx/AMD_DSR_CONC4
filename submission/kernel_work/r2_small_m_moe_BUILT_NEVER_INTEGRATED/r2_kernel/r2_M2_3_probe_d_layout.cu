// SPDX-License-Identifier: MIT
// R2-C M2.3 step 1: probe kernel to empirically determine the
// D matrix lane->(m, n) mapping for V_MFMA_F32_16x16x128_F8F6F4 (FP4 mode).
//
// Strategy: feed identity-like A and B, run MFMA, write each lane's
// 4 floats to a per-lane slot in a global buffer. Python decodes which
// (m, n) each lane's i-th float corresponds to.
//
// A trick: if we set A[m, k] = some marker(m) and B[n, k] = some marker(n),
// then D[m, n] = sum over k of A[m,k]*B[n,k]*scale = K * marker(m)*marker(n).
// By choosing markers as powers of 2, we can read off m and n from D[m,n]'s value.
//
// For FP4 we have only 16 quantization levels, can't encode m and n
// in arbitrary precision. But we can use a SIMPLER probe: set A[m,k]=1
// for all m, k. Set B[n,k]=1 for all n, k. Then D[m,n] = K = 128 for all entries.
// That doesn't help distinguish lanes.
//
// Better: set A[m,k] = 1.0 for ALL m, k. Set B[n,k] = (n is special) ? 0 : ... no.
//
// Best probe: Use a constant input + per-lane writes. The MFMA's D output
// is then D[i,j] = K * 1 * 1 = 128 (or whatever scale). Each lane holds 4
// of the 256 output values. By writing each lane's (lane_id, c_acc[0..3])
// to a known slot, we can later set A,B such that D[m,n] = encode(m,n)
// and read back to decode.

#include <hip/hip_runtime.h>

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

// Probe: A is all 1.0 in fp4 (= 0x22 byte = two 1.0 fp4 = bits 0010_0010).
// B is all 1.0 in fp4 (same byte pattern).
// D[m, n] = sum_k(1*1) = 128. Every D entry = 128.0.
//
// To distinguish lanes, we OUTPUT the lane_id and the 4 floats per lane.
// Python decodes: which 4 (m,n) entries does each lane hold?
//
// To get a unique signature per (m,n), use a TWO-PASS probe:
// Pass 1: A[m, k] = 2^m (encoded in fp4 as best we can). Ah, fp4 max is 6.
//         So we can't do per-m encoding directly.
// Pass 2: Use B[n, k] = scaled by E8M0 per-lane to inject n information.
//         Too complex.
//
// SIMPLEST: write TWO probe kernels.
//   Probe A: D_target[m,n] = m+1     -> A[m,k]=fp4_value_for(m+1), B all 1
//   Probe B: D_target[m,n] = n+1     -> A all 1, B[n,k]=fp4_value_for(n+1)
// Then per-lane output value gives m or n directly.
//
// But fp4 can only encode {0, 0.5, 1, 1.5, 2, 3, 4, 6} so D = K * fp4_a * fp4_b
// = 128 * 1 * fp4_b = 128 * fp4_b can hit values 64, 128, 192, ...
//
// Even simpler: we don't need to FULLY decode m,n. We just need a fingerprint
// that tells us the lane->(m,n) mapping. Just dump per-lane output and
// REVERSE-ENGINEER.

extern "C" __global__ __launch_bounds__(kThreads, 1)
void r2_m2_3_probe(
    const uint8_t* __restrict__ A_fp4,  // [16, 64]
    const uint8_t* __restrict__ B_fp4,  // [16, 64]
    float* __restrict__ out_per_lane)   // [64, 4] — per-lane 4 floats output
{
    int lane = threadIdx.x;
    int M_lane = lane & 0xF;
    int K_block = lane >> 4;

    // Load 16 bytes per lane (32 fp4) into A_reg/B_reg low 4 ints.
    intx8_t A_reg = {0,0,0,0,0,0,0,0};
    intx8_t B_reg = {0,0,0,0,0,0,0,0};

    int K_byte = K_block * 16;
    const int* a_int = (const int*)(A_fp4 + M_lane * 64 + K_byte);
    const int* b_int = (const int*)(B_fp4 + M_lane * 64 + K_byte);
    A_reg[0] = a_int[0]; A_reg[1] = a_int[1]; A_reg[2] = a_int[2]; A_reg[3] = a_int[3];
    B_reg[0] = b_int[0]; B_reg[1] = b_int[1]; B_reg[2] = b_int[2]; B_reg[3] = b_int[3];

    floatx4_t C_acc = {0.f, 0.f, 0.f, 0.f};
    C_acc = mfma_scale_16x16x128_mxfp4<>(A_reg, B_reg, C_acc);

    // Dump this lane's 4 output floats to out_per_lane[lane, 0..3]
    out_per_lane[lane * 4 + 0] = C_acc[0];
    out_per_lane[lane * 4 + 1] = C_acc[1];
    out_per_lane[lane * 4 + 2] = C_acc[2];
    out_per_lane[lane * 4 + 3] = C_acc[3];
}

extern "C" hipError_t r2_m2_3_probe_launch(
    const uint8_t* A_fp4, const uint8_t* B_fp4,
    float* out_per_lane, hipStream_t stream)
{
    dim3 grid(1);
    dim3 block(kThreads);
    r2_m2_3_probe<<<grid, block, 0, stream>>>(A_fp4, B_fp4, out_per_lane);
    return hipGetLastError();
}

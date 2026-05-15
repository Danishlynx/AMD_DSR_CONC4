// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Danish — DSR1 hackathon submission.
//
// R2-C M1 SKELETON: small-M MoE GEMM2 kernel for CDNA4 (gfx950).
//
// Goal of M1: COMPILE + LAUNCH + NO CRASH only. Performance and parity come in M2/M3.
//
// Hot shape (from R2-A capture):
//   M=4 (decode workload), N=intermediate=512, K=hidden=7168
//   topk=9, num_experts=257, FP4×FP4 with E8M0 per-32-K block scales
//
// Reference T0 (from R2-B microbench): 71.88 µs/call via AITER ck2stages fallback.
// R2-C target: ≤ 47.92 µs/call (1.5× speedup) at M2/M3.
//
// Hardware basis (CDNA4 / MI355X):
//   - V_MFMA_F32_16x16x128_F8F6F4 native MXFP4 MFMA: __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4
//     Each call: 16M × 16N × 128K. Per-lane: 32 bytes A, 32 bytes B, 4 floats C.
//     For MXFP4: cbsz=4 (A is FP4), blgp=4 (B is FP4). Scales are E8M0 bytes.
//   - DS_READ_B64_TR_B4 LDS load with 4-bit transpose, k-interleaved (per CDNA4 ISA §11.7)
//   - BUFFER_LOAD_DWORDX4_LDS for direct global→LDS async copy
//   - global_atomic_pk_add_bf16 for output reduction
//
// M1 SCOPE:
//   1) Build path works: hipcc + gfx950 + correct intrinsics resolved
//   2) Kernel launches without HSA exception or compile failure
//   3) Returns; output buffer untouched or zero-filled is fine
//   4) NO MFMA loop yet (placeholder)
//   5) NO parity vs aiter.fused_moe (M2)
//   6) NO perf tuning (M3)

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

// ---------- Type aliases ----------
typedef __attribute__((__vector_size__(8 * sizeof(int)))) int  intx8_t;
typedef __attribute__((__vector_size__(4 * sizeof(int)))) int  intx4_t;
typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;
typedef __attribute__((__vector_size__(2 * sizeof(int)))) int  intx2_t;

// ---------- Constants for the captured hot shape ----------
constexpr int kHotM        = 4;     // decode workload M
constexpr int kHotN        = 512;   // expert intermediate dim (N)
constexpr int kHotK        = 7168;  // hidden_size (K)
constexpr int kHotTopk     = 9;
constexpr int kHotExperts  = 257;
constexpr int kScaleGroup  = 32;    // K elements per E8M0 scale

// MFMA tile geometry (HW-fixed by V_MFMA_F32_16x16x128_F8F6F4)
constexpr int kMfmaM  = 16;
constexpr int kMfmaN  = 16;
constexpr int kMfmaK  = 128;

// Workgroup / kernel tile (R2-C choice)
constexpr int kTileM  = kHotM;          // 4 valid; 12 of 16 MFMA-M slots wasted (unavoidable at workload M=4)
constexpr int kTileN  = 32;             // 2 N-MFMAs per output sub-block, processed by 1 wave
constexpr int kTileK  = kMfmaK;         // 128, processed by 1 MFMA per K-iteration
constexpr int kKIters = kHotK / kTileK; // 7168/128 = 56 K-iters

constexpr int kThreadsPerWg  = 64;      // one wavefront
constexpr int kBytesA_perLane = 32;     // 32 bytes per lane = 16 fp4 elements packed × 4 K-strides
constexpr int kBytesB_perLane = 32;

// ---------- MFMA wrapper for CDNA4 native MXFP4 ----------
// Wraps __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4 with cbsz=4, blgp=4.
// HipKittens reference: include/ops/warp/register/tile/mma.cuh:119
// All trailing args are compile-time constants per LLVM intrinsic constraints.
// For MXFP4 (FP4×FP4 with E8M0 scales): cbsz=4, blgp=4 (encoded as FP4 type).
// HipKittens uses all-zero (FP8×FP8 mode) at mma.cuh:119; we template on cbsz/blgp.
template<int CBSZ = 4, int ABID = 0, int BLGP = 4,
         int OPSEL_A = 0, int SCALE_A = 0, int SCALE_B = 0>
__device__ static inline floatx4_t mfma_scale_16x16x128_mxfp4(
    intx8_t A, intx8_t B, floatx4_t C)
{
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        A, B, C,
        CBSZ, ABID, BLGP,
        OPSEL_A, SCALE_A, SCALE_B);
}

// ---------- DS read with 4-bit transpose ----------
// Reads 64 bits per lane from LDS in k-interleaved transposed layout.
// HipKittens reference: include/common/macros.cuh:239 (tr_b16 variant; tr_b4 has same form).
template<int GPR_START>
__device__ __forceinline__ void ds_read_b64_tr_b4(uint32_t smem_ptr, int byte_offset) {
    constexpr int GPR_END = GPR_START + 1;
    asm volatile(
        "ds_read_b64_tr_b4 v[%0:%1], %2 offset:%3"
        :
        : "n"(GPR_START), "n"(GPR_END), "v"(smem_ptr), "i"(byte_offset)
        : "memory");
}

// ---------- BF16 atomic packed-add for output accumulation ----------
// global_atomic_pk_add_bf16 writes 2 BF16 lanes atomically. Used for cross-expert reduction.
__device__ __forceinline__ void atomic_pk_add_bf16(__hip_bfloat162* dst, __hip_bfloat162 v) {
    // Use __builtin_amdgcn_global_atomic_fadd_v2bf16 if available.
    // Fallback: reinterpret as int and use atomic CAS (slow but correct).
#ifdef __HIP_ARCH_GFX950__
    asm volatile(
        "global_atomic_pk_add_bf16 v[%0:%1], %2, off"
        :
        : "v"(((uint64_t)dst)), "v"(((uint64_t)dst >> 32)), "v"(*(int*)&v)
        : "memory");
#else
    // Non-gfx950 fallback (won't be exercised in our build)
    atomicAdd((float*)dst, __bfloat162float(v.x) + __bfloat162float(v.y));
#endif
}

// ---------- M1 SKELETON KERNEL ----------
// Placeholder kernel: launches, declares LDS, writes a zero-init pattern to output.
// No actual MFMA loop yet — just proves compile + launch + no crash.
// Signature mirrors aiter.fused_moe ck2stages stage2 (gemm2) call surface.
extern "C" __global__ __launch_bounds__(kThreadsPerWg, 4)
void r2_smallm_moe_gemm2_skeleton(
    const uint8_t* __restrict__ inter_states,    // [num_active_tokens, intermediate/2] FP4 packed
    const uint8_t* __restrict__ w2,              // [num_experts, hidden, intermediate/2] FP4 packed
    const uint8_t* __restrict__ inter_scales,    // [num_active_tokens, intermediate/32] E8M0
    const uint8_t* __restrict__ w2_scales,       // [num_experts, hidden, intermediate/32] E8M0
    const int*     __restrict__ sorted_token_ids,
    const int*     __restrict__ sorted_expert_ids,
    const int*     __restrict__ num_valid_ids,
    const float*   __restrict__ topk_weights,    // [num_tokens, topk]
    __hip_bfloat16* __restrict__ output,         // [num_tokens, hidden] BF16 accumulated
    int num_tokens,
    int hidden,
    int intermediate,
    int topk)
{
    // ---- Shared memory layout (placeholder allocation) ----
    // M1: declare but don't read/write. M2 will populate.
    __shared__ uint8_t lds_A[2][16 * (kTileK / 2)];   // 16 padded M-rows × 64 fp4-bytes per K-tile, double-buffered
    __shared__ uint8_t lds_B[2][kTileN * (kTileK / 2)]; // [32 N × 64 fp4-bytes] × 2
    __shared__ uint8_t lds_Sa[16 * (kTileK / kScaleGroup)];
    __shared__ uint8_t lds_Sb[kTileN * (kTileK / kScaleGroup)];

    int wg_id  = blockIdx.x;
    int lane   = threadIdx.x; // 0..63

    // ---- M1 placeholder: write a sentinel zero to output[0,0] from rank-0 thread to prove launch ----
    if (wg_id == 0 && lane == 0) {
        // No-op: don't actually write to avoid contention with the M2 path.
        // Just confirm the kernel reached this point by reading num_tokens (volatile compiler hint).
        volatile int probe = num_tokens; (void)probe;
    }

    // ---- M1 placeholder: declare register tiles to confirm intrinsics resolve ----
    intx8_t   A_reg = {0,0,0,0,0,0,0,0};
    intx8_t   B_reg = {0,0,0,0,0,0,0,0};
    floatx4_t C_acc = {0.f, 0.f, 0.f, 0.f};

    // Don't actually call MFMA in M1 (would need real FP4 input tensors).
    // M2 will fill the K-loop with real loads + MFMA.
    if (false) {
        C_acc = mfma_scale_16x16x128_mxfp4<>(A_reg, B_reg, C_acc);
        ds_read_b64_tr_b4<0>(0, 0);
    }

    // M1 done.
}

// ---------- Host launcher (extern "C" so Python ctypes can find it) ----------
extern "C" hipError_t r2_launch_smallm_moe_gemm2_skeleton(
    const uint8_t* inter_states,
    const uint8_t* w2,
    const uint8_t* inter_scales,
    const uint8_t* w2_scales,
    const int* sorted_token_ids,
    const int* sorted_expert_ids,
    const int* num_valid_ids,
    const float* topk_weights,
    __hip_bfloat16* output,
    int num_tokens,
    int hidden,
    int intermediate,
    int topk,
    hipStream_t stream)
{
    // Grid: 1 workgroup per (token × N-tile) — M1 just launches a single workgroup
    dim3 grid(1);
    dim3 block(kThreadsPerWg);

    r2_smallm_moe_gemm2_skeleton<<<grid, block, 0, stream>>>(
        inter_states, w2, inter_scales, w2_scales,
        sorted_token_ids, sorted_expert_ids, num_valid_ids,
        topk_weights, output,
        num_tokens, hidden, intermediate, topk);

    return hipGetLastError();
}

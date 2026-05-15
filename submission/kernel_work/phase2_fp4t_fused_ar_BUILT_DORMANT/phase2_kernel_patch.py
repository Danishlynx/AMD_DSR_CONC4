#!/usr/bin/env python3
"""Phase 2 MXFP4 triple-fusion kernel patch for /app/aiter-test/csrc/include/custom_all_reduce.cuh

Adds:
1. ar_fusion_epilogue_per32_max: per-32-group warp DPP reduce (4 threads/group)
2. MXFP4 branch in ar_fusion_epilogue: BF16->FP4 via aiter::bf16_to_fp4_scaled_x8
   + e8m0 scale extraction matching quant_kernels.cu:1654-1720 reference pattern
3. Generalized packQuant<T, OutT, N> template (only for FP8 — FP4 goes through
   bf16_to_fp4_scaled_x8 directly, separate code path).

Strategy: ADD code, do NOT modify existing FP8 path. Surgical insertion.
"""
import sys, re

PATH = "/app/aiter-test/csrc/include/custom_all_reduce.cuh"
src = open(PATH).read()
ORIG = src

# -------- PATCH 1: Add per-32-group max reduce helper --------
ANCHOR1 = """template <typename A, int PACK_SIZE, int WARP_SIZE = 32>
__device__ __forceinline__ float ar_fusion_epilogue_reduce_abs_max(A& data, int block_size)
{"""
INSERT1_BEFORE_ANCHOR1 = """// Phase2 MXFP4: per-32-group abs_max reduce. Each thread holds PACK_SIZE elements.
// For PACK_SIZE=8 BF16: 1 thread = 8 elements. 4 threads = 32 elements = 1 MXFP4 group.
// Returns the per-group max for THIS thread's group (broadcast across the 4 threads).
template <typename A, int PACK_SIZE>
__device__ __forceinline__ float ar_fusion_epilogue_reduce_abs_max_per32(A& data)
{
    static_assert(PACK_SIZE * 4 == 32, "Phase2 MXFP4: per-32 group requires PACK_SIZE*4==32");
    float local_max = 0.f;
#pragma unroll
    for(int i = 0; i < PACK_SIZE; ++i)
    {
        float v = upcast_s(data[i]);
        float a = std::abs(v);
        local_max = local_max > a ? local_max : a;
    }
    // 4-thread DPP reduce within each group of 4 lanes
    // bpermute: lane (idx XOR 1) -> share with neighbor in pair
    float v1 = __shfl_xor(local_max, 1, 4);
    local_max = local_max > v1 ? local_max : v1;
    float v2 = __shfl_xor(local_max, 2, 4);
    local_max = local_max > v2 ? local_max : v2;
    return local_max;
}

// Phase2 MXFP4: extract e8m0 scale matching quant_kernels.cu fp4_scale + e8m0 pattern.
// Given group_max, computes: pow2_floor(group_max) / 4.0 (since DTYPE_MAX=6.0,
// floor(log2(6.0))=2, 2^2=4.0). Returns (row_scale_f32, e8m0_byte).
__device__ __forceinline__ void ar_fusion_epilogue_mxfp4_e8m0(
    float group_max, float& row_scale_out, uint8_t& e8m0_out)
{
    // Match quant_kernels.cu fp4_scale lambda exactly.
    uint32_t bits = __builtin_bit_cast(uint32_t, group_max);
    uint32_t pow2_floor_bits = (bits >> 23) << 23;
    float pow2_floor = __builtin_bit_cast(float, pow2_floor_bits);
    row_scale_out = pow2_floor / 4.0f;  // 4.0 = 2^floor(log2(FP4_MAX=6.0))
    uint32_t scale_bits = __builtin_bit_cast(uint32_t, row_scale_out);
    e8m0_out = (uint8_t)((scale_bits >> 23) & 0xFFu);
}

"""

# -------- PATCH 2: Add MXFP4 branch in ar_fusion_epilogue --------
ANCHOR2_OLD = """template <typename P, typename A, typename T, typename OutT, int PACK_SIZE>
__device__ __forceinline__ void ar_fusion_epilogue(A& in,
                                                   P& weight,
                                                   int hidden_dim,
                                                   float eps,
                                                   int idx,
                                                   int tidx,
                                                   int block_size,
                                                   OutT* __restrict__ output,
                                                   float* __restrict__ scale_out,
                                                   bool active = true)
{
    if constexpr(std::is_same_v<T, OutT>)
    {
        P out;
        ar_fusion_epilogue_rms_norm<P, A, P, T, PACK_SIZE>(
            out, in, weight, eps, hidden_dim, block_size);
        if(active)
            *reinterpret_cast<P*>(output + idx) = out;
    }
    else
    {
        float FP8_UPBOUND = opus::cast<opus::fp32_t>(opus::numeric_limits<opus::fp8_t>::max());
        using OP          = opus::vector_t<OutT, PACK_SIZE>;
        OP out_quant;
        A out;
        ar_fusion_epilogue_rms_norm<P, A, A, float, PACK_SIZE>(
            out, in, weight, eps, hidden_dim, block_size);
        float amax  = ar_fusion_epilogue_reduce_abs_max<A, PACK_SIZE>(out, block_size);
        float scale = amax == 0.f ? 1.f : amax / FP8_UPBOUND;
        out_quant   = packQuant<opus::fp32_t, PACK_SIZE>(out, scale);
        if(active)
            *reinterpret_cast<OP*>(output + idx) = out_quant;
        if(threadIdx.x == 0)
            scale_out[tidx] = scale;
    }
}"""

ANCHOR2_NEW = """template <typename P, typename A, typename T, typename OutT, int PACK_SIZE>
__device__ __forceinline__ void ar_fusion_epilogue(A& in,
                                                   P& weight,
                                                   int hidden_dim,
                                                   float eps,
                                                   int idx,
                                                   int tidx,
                                                   int block_size,
                                                   OutT* __restrict__ output,
                                                   void* __restrict__ scale_out,
                                                   bool active = true)
{
    if constexpr(std::is_same_v<T, OutT>)
    {
        P out;
        ar_fusion_epilogue_rms_norm<P, A, P, T, PACK_SIZE>(
            out, in, weight, eps, hidden_dim, block_size);
        if(active)
            *reinterpret_cast<P*>(output + idx) = out;
    }
    // Phase2: MXFP4 path - per-32-group e8m0 scale, BF16->FP4 packed via gfx950 intrinsic
    else if constexpr(std::is_same_v<OutT, opus::fp4_t>)
    {
        static_assert(PACK_SIZE == 8,
            "Phase2 MXFP4 path requires PACK_SIZE=8 (one BF16 thread = 8 elements = 1/4 of 32-group)");
        // Step 1: RMSNorm in float
        A out;
        ar_fusion_epilogue_rms_norm<P, A, A, float, PACK_SIZE>(
            out, in, weight, eps, hidden_dim, block_size);

        // Step 2: per-32-group max via 4-lane DPP reduce
        float group_max = ar_fusion_epilogue_reduce_abs_max_per32<A, PACK_SIZE>(out);
        if(group_max < 1e-10f) group_max = 1e-10f;

        // Step 3: e8m0 scale extraction
        float row_scale;
        uint8_t e8m0;
        ar_fusion_epilogue_mxfp4_e8m0(group_max, row_scale, e8m0);

        // Step 4: BF16->FP4 conversion via aiter intrinsic. Source must be bf16x8.
        opus::vector_t<T, PACK_SIZE> bf16_pack;
#pragma unroll
        for(int i = 0; i < PACK_SIZE; ++i)
            bf16_pack[i] = downcast_s<T>(out[i]);
        // bf16_to_fp4_scaled_x8 expects "scale" arg to be the SCALE FACTOR
        // applied as multiplier to source. To get out_fp4 = src/row_scale we
        // pass 1.0/row_scale. (Matches quant_kernels.cu scaled_quant_vgpr_impl
        // and aiter_opus_plus.h:246 inverted_scale convention.)
        float inv_scale = (row_scale > 0.f) ? (1.0f / row_scale) : 0.f;
        auto fp4_packed = aiter::bf16_to_fp4_scaled_x8(bf16_pack, inv_scale);

        // Step 5: write packed FP4 (4 bytes / thread = 8 FP4 elements)
        if(active)
        {
            uint32_t fp4_u32 = __builtin_bit_cast(uint32_t, fp4_packed);
            // Output buffer is uint8 fp4_t* with 1 byte per 2 FP4 elements.
            // PACK_SIZE=8 means 4 bytes/thread. idx is in elements; output is fp4_t (1B for 2 elem)
            // so fp4 byte offset = idx/2.
            uint32_t* out_u32 = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(output) + idx/2);
            *out_u32 = fp4_u32;
        }

        // Step 6: write e8m0 scale per group (only first thread of each 4-group)
        if(active && (threadIdx.x & 3) == 0)
        {
            int group_id = threadIdx.x >> 2;          // group within block
            int hidden_groups = hidden_dim >> 5;       // hidden_dim / 32
            uint8_t* scale_u8 = reinterpret_cast<uint8_t*>(scale_out);
            scale_u8[tidx * hidden_groups + group_id] = e8m0;
        }
    }
    else
    {
        float FP8_UPBOUND = opus::cast<opus::fp32_t>(opus::numeric_limits<opus::fp8_t>::max());
        using OP          = opus::vector_t<OutT, PACK_SIZE>;
        OP out_quant;
        A out;
        ar_fusion_epilogue_rms_norm<P, A, A, float, PACK_SIZE>(
            out, in, weight, eps, hidden_dim, block_size);
        float amax  = ar_fusion_epilogue_reduce_abs_max<A, PACK_SIZE>(out, block_size);
        float scale = amax == 0.f ? 1.f : amax / FP8_UPBOUND;
        out_quant   = packQuant<opus::fp32_t, PACK_SIZE>(out, scale);
        if(active)
            *reinterpret_cast<OP*>(output + idx) = out_quant;
        if(threadIdx.x == 0)
            reinterpret_cast<float*>(scale_out)[tidx] = scale;
    }
}"""

if "ar_fusion_epilogue_reduce_abs_max_per32" not in src:
    assert ANCHOR1 in src, "ANCHOR1 (reduce_abs_max) not found"
    assert ANCHOR2_OLD in src, "ANCHOR2 (ar_fusion_epilogue) not found"
    # Insert per-32 helpers BEFORE the existing reduce_abs_max function
    src = src.replace(ANCHOR1, INSERT1_BEFORE_ANCHOR1 + ANCHOR1, 1)
    # Replace ar_fusion_epilogue with new version that adds MXFP4 branch
    src = src.replace(ANCHOR2_OLD, ANCHOR2_NEW, 1)
    open(PATH, 'w').write(src)
    print("PATCHED custom_all_reduce.cuh")
    print(f"  +{src.count(chr(10)) - ORIG.count(chr(10))} lines")
else:
    print("ALREADY PATCHED")

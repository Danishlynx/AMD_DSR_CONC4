#!/usr/bin/env python3
"""Phase 2 kernel v2: Replace bf16_to_fp4_scaled_x8 helper call with INLINE AMD builtin
intrinsic + drop aiter_opus_plus.h include (avoids `max` ambiguity in unrelated FP8 path).
"""
PATH = "/app/aiter-test/csrc/include/custom_all_reduce.cuh"
src = open(PATH).read()
ORIG = src

# Step 1: drop aiter_opus_plus.h include
OLD_INC = '#include "opus/opus.hpp"\n#include "aiter_opus_plus.h"'
NEW_INC = '#include "opus/opus.hpp"'
if '#include "aiter_opus_plus.h"' in src:
    src = src.replace(OLD_INC, NEW_INC, 1)
    print("removed aiter_opus_plus.h include")

# Step 2: replace bf16_to_fp4_scaled_x8 call with inline AMD builtin
OLD_CONV = """        // Step 4: BF16->FP4 conversion via aiter intrinsic. Source must be bf16x8.
        opus::vector_t<T, PACK_SIZE> bf16_pack;
#pragma unroll
        for(int i = 0; i < PACK_SIZE; ++i)
            bf16_pack[i] = downcast_s<T>(out[i]);
        // bf16_to_fp4_scaled_x8 expects "scale" arg to be the SCALE FACTOR
        // applied as multiplier to source. To get out_fp4 = src/row_scale we
        // pass 1.0/row_scale. (Matches quant_kernels.cu scaled_quant_vgpr_impl
        // and aiter_opus_plus.h:246 inverted_scale convention.)
        float inv_scale = (row_scale > 0.f) ? (1.0f / row_scale) : 0.f;
        auto fp4_packed = bf16_to_fp4_scaled_x8(bf16_pack, inv_scale);

        // Step 5: write packed FP4 (4 bytes / thread = 8 FP4 elements)
        if(active)
        {
            uint32_t fp4_u32 = __builtin_bit_cast(uint32_t, fp4_packed);
            // Output buffer is uint8 fp4_t* with 1 byte per 2 FP4 elements.
            // PACK_SIZE=8 means 4 bytes/thread. idx is in elements; output is fp4_t (1B for 2 elem)
            // so fp4 byte offset = idx/2.
            uint32_t* out_u32 = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(output) + idx/2);
            *out_u32 = fp4_u32;
        }"""

NEW_CONV = """        // Step 4: BF16->FP4 conversion using AMD builtin intrinsic directly.
        // 8 BF16 elements -> 4 packed FP4 bytes (4 bytes total in one u32).
        // inv_scale: out_fp4 = src * inv_scale = src / row_scale.
        float inv_scale = (row_scale > 0.f) ? (1.0f / row_scale) : 0.f;
        // bf16x2_t = bf16_t __attribute__((ext_vector_type(2)))
        using bf16x2_v = opus::bf16_t __attribute__((ext_vector_type(2)));
        bf16x2_v p0 = {downcast_s<opus::bf16_t>(out[0]), downcast_s<opus::bf16_t>(out[1])};
        bf16x2_v p1 = {downcast_s<opus::bf16_t>(out[2]), downcast_s<opus::bf16_t>(out[3])};
        bf16x2_v p2 = {downcast_s<opus::bf16_t>(out[4]), downcast_s<opus::bf16_t>(out[5])};
        bf16x2_v p3 = {downcast_s<opus::bf16_t>(out[6]), downcast_s<opus::bf16_t>(out[7])};
        unsigned int fp4_u32 = 0;
        fp4_u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(fp4_u32, p0, inv_scale, 0);
        fp4_u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(fp4_u32, p1, inv_scale, 1);
        fp4_u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(fp4_u32, p2, inv_scale, 2);
        fp4_u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(fp4_u32, p3, inv_scale, 3);

        // Step 5: write packed FP4 (4 bytes / thread = 8 FP4 elements).
        // Output buffer is uint8 fp4_t* with 1 byte per 2 FP4 elements.
        // PACK_SIZE=8 means 4 bytes/thread. idx is in elements; output is fp4_t (1B for 2 elem)
        // so fp4 byte offset = idx/2.
        if(active)
        {
            unsigned int* out_u32 = reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(output) + idx/2);
            *out_u32 = fp4_u32;
        }"""

if "Step 4: BF16->FP4 conversion using AMD builtin" not in src:
    assert OLD_CONV in src, "OLD_CONV anchor not found"
    src = src.replace(OLD_CONV, NEW_CONV, 1)
    print("inlined AMD intrinsic for bf16->fp4 conversion")
else:
    print("AMD inline already present")

if src != ORIG:
    open(PATH, 'w').write(src)
    print(f"WROTE {PATH}")

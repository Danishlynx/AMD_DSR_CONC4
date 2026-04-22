// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
// session-16 V9: v9 dispatch added for num_head==32 + AITER_ENABLE_HK_QH32_V9=1.
// v9 uses mfma_scale_f32_16x16x128_f8f6f4 for NoPE QK^T (4× K-depth/call).

#include "mla.h"
#include "hk/mi3xx_v32_fwd_decode_h128_fp8_fp8.cuh"
#include "hk/mi3xx_v32_fwd_decode_h32_fp8_fp8.cuh"
#include "hk/mi3xx_v32_fwd_decode_h32_fp8_fp8_v8.cuh"
#include "hk/mi3xx_v32_fwd_decode_h32_fp8_fp8_v9.cuh"

#include <cstdlib>
#include <cstring>

static inline bool hk_env_is_true(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr) return false;
    if (v[0] == '\0') return false;
    if (std::strcmp(v, "0") == 0) return false;
    return true;
}

void hk_mla_decode_fwd(
    torch::Tensor& query,
    torch::Tensor& kv_buffer,
    const torch::Tensor& qo_indptr,
    const torch::Tensor& kv_indptr,
    const torch::Tensor& kv_page_indices,
    const torch::Tensor& kv_last_page_lens,
    const torch::Tensor& work_indptr,
    const torch::Tensor& work_info_set,
    const int max_seqlen_q,
    const float softmax_scale,
    torch::Tensor& split_output,
    torch::Tensor& split_lse,
    torch::Tensor& final_output)
{
    const int32_t num_head = query.size(1);

    if (num_head == 128) {
        hk_mi3xx_mla_v32_fwd_decode_h128_fp8_fp8(
            query, kv_buffer, qo_indptr, kv_indptr, kv_page_indices, kv_last_page_lens,
            work_indptr, work_info_set, max_seqlen_q, softmax_scale,
            split_output, split_lse, final_output);
    } else if (num_head == 32) {
        // Dispatch priority: V9 > V8 > V7.
        // V9: mfma_1616128 NoPE upgrade (4× K-depth/call on NoPE QK^T).
        // V8: v7 + Opt-E s_setprio.
        // V7: session-14 baseline.
        if (hk_env_is_true("AITER_ENABLE_HK_QH32_V9")) {
            hk_mi3xx_mla_v32_fwd_decode_h32_fp8_fp8_v9(
                query, kv_buffer, qo_indptr, kv_indptr, kv_page_indices, kv_last_page_lens,
                work_indptr, work_info_set, max_seqlen_q, softmax_scale,
                split_output, split_lse, final_output);
        } else if (hk_env_is_true("AITER_ENABLE_HK_QH32_V8")) {
            hk_mi3xx_mla_v32_fwd_decode_h32_fp8_fp8_v8(
                query, kv_buffer, qo_indptr, kv_indptr, kv_page_indices, kv_last_page_lens,
                work_indptr, work_info_set, max_seqlen_q, softmax_scale,
                split_output, split_lse, final_output);
        } else {
            hk_mi3xx_mla_v32_fwd_decode_h32_fp8_fp8(
                query, kv_buffer, qo_indptr, kv_indptr, kv_page_indices, kv_last_page_lens,
                work_indptr, work_info_set, max_seqlen_q, softmax_scale,
                split_output, split_lse, final_output);
        }
    } else {
        TORCH_CHECK(false,
            "hk_mla_decode_fwd currently supports num_head in {128, 32}, got num_head = ",
            num_head);
    }
}

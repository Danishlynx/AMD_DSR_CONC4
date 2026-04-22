// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
// session-15 RE.4c: v8 dispatch added for num_head==32 + AITER_ENABLE_HK_QH32_V8=1.

#include "mla.h"
#include "hk/mi3xx_v32_fwd_decode_h128_fp8_fp8.cuh"
#include "hk/mi3xx_v32_fwd_decode_h32_fp8_fp8.cuh"
#include "hk/mi3xx_v32_fwd_decode_h32_fp8_fp8_v8.cuh"

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
        // v8 adds inner q_pos loop over [qo_start, qo_end) so metadata emitting
        // qo_end-qo_start > 1 (e.g. sq=8 MTP=7) is handled correctly. Also has
        // s_setprio coverage around QK/PV MFMAs (Opt-E).
        // Gate: AITER_ENABLE_HK_QH32_V8=1. Default off → fall through to v7 (session-14).
        if (hk_env_is_true("AITER_ENABLE_HK_QH32_V8")) {
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

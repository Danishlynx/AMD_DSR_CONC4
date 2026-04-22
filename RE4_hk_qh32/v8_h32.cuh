// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// v8 (RE.4c, session-15, Apr 22-23 2026) — SESSION-14 HK v7 + two surgical changes:
//   (1) INNER q_pos LOOP over [qo_start, qo_end) so the kernel processes all
//       query positions within one work_info entry. At qseqlen=8 (MTP=7),
//       get_mla_metadata_v1 (aiter main) emits ONE work_info per batch with
//       qo_end-qo_start=8. v7 only processed qo_start -> wrong output at
//       positions 1-7 (reproduces bug class of aiter issue #2720).
//       v8 iterates q_pos through [qo_start, qo_end) so each position runs
//       the full sq=1 pipeline (Q-load → QK^T → softmax → PV → output).
//       K/V are re-streamed per position (same HBM traffic as a Python split).
//       This is the CORRECTNESS fix for MTP=7 — it does NOT claim a perf gain
//       over ASM at sq=4 (that requires the MFMA-width rewrite tracked as v9).
//
//   (2) OPT-E s_setprio COVERAGE around QK^T and PV MFMAs to enable VALU
//       dual-issue during MFMA execution. v7 only s_setprio'd the oaccu
//       rescale block (lines 482-522). Hand-tuned ASM uses finer priority
//       changes across all MFMA blocks. Expected: +1-3% TPOT at sq=4.
//
// NOT changed vs v7 (scoped out of this session):
//   - MFMA opcode (v7 uses 16x16x32 fp8; upgrade to 16x16x128 requires
//     full LDS/tile rewrite — tracked as v9).
//   - LDS XOR swizzle (requires rocprof bank-conflict counter data to
//     verify target exists — tracked as v9.2).
//   - Parallel virtual-warp unroll (needs SASS inspection to confirm
//     v7's #pragma unroll already emits back-to-back buffer_load — tracked
//     as v9.1 pending server-side SASS dump).
//   - kNumWarps=4 (infeasible at nhead=32 given static_assert kBlockM==kQoNumHead).
//   - Direct-to-LDS buffer_load_lds (v7 already uses this at line 900 of
//     hk_mla_buffer_managers.cuh — no-op).
//
// Compile/dispatch:
//   Gate via AITER_ENABLE_HK_QH32_V8=1 in launch script. Co-exists with v7.
//   See aiter_mla_py_patch_v8.diff for dispatch changes.

#pragma once

#include "hk_mla_buffer_managers.cuh"
#include "hk_mla_softmax.cuh"
#include "mla.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/python.h>

template <typename q_t_, typename kv_t_, typename out_t_, int32_t kQoNumHead_>
struct HkMlaDecodeFwdTraitsH32V8
{
    static constexpr int32_t kQoNumHead     = kQoNumHead_;
    static constexpr int32_t kKvNumHead     = 1;
    static constexpr int32_t kKvLoraRank    = 512;
    static constexpr int32_t kQkNopeHeadDim = kKvLoraRank;
    static constexpr int32_t kQkRopeHeadDim = 64;
    static constexpr int32_t kQkHeadDim     = kQkNopeHeadDim + kQkRopeHeadDim;
    static constexpr int32_t kVoHeadDim     = kKvLoraRank;
    static constexpr int32_t kPageSize      = 1;
    static constexpr int32_t kNumWarps      = 2;
    static constexpr int32_t kNumThreads    = kNumWarps * ckt::get_warp_size();
    static constexpr int32_t kVirtualWarps   = 8;
    static constexpr int32_t kVirtualPerReal = kVirtualWarps / kNumWarps;
    static constexpr int32_t kOccupancy     = 1;
    static constexpr int32_t kBlockM        = 32;
    static constexpr int32_t kBlockN        = 32;
    static constexpr int32_t kBlockK        = 32;
    static constexpr int32_t kTileM         = kBlockM / kNumWarps;
    static constexpr int32_t kNumTilesM     = kBlockM / kTileM;
    static constexpr int32_t kRoundMode     = 1;

    static_assert(kBlockM == kQoNumHead, "h32/v8: kBlockM must match kQoNumHead (32)");
    static_assert(kNumWarps == 2, "h32/v8: kNumWarps=2 for 2-warp-native port");
    static_assert(kTileM == 16, "h32/v8: kTileM=16 for MFMA fp8 16x16x32");
    static_assert(kVirtualPerReal == 4, "h32/v8: virtual/real ratio must be 4");

    using q_t   = q_t_;
    using kv_t  = kv_t_;
    using out_t = out_t_;
    using gl_q = hk::gl<q_t, -1, kNumTilesM, kTileM, kQkHeadDim>;
    using gl_kv =
        hk::gl<kv_t, -1, kPageSize, kKvNumHead, kQkHeadDim>;
    using gl_o    = hk::gl<out_t, 1, -1, kQoNumHead, kVoHeadDim>;
    using gl_so   = hk::gl<float, 1, -1, kQoNumHead, kVoHeadDim>;
    using gl_slse = hk::gl<float, 1, -1, kQoNumHead, 1>;
    static_assert(std::is_same_v<kv_t, hk::bf16> || std::is_same_v<kv_t, hk::fp8e4m3>);
    using st_kv_nope = std::conditional_t<std::is_same_v<kv_t, hk::fp8e4m3>,
                                          hk::st_fp8e4m3<kBlockN, kKvLoraRank, hk::st_16x16_s>,
                                          hk::st_bf<kBlockN, kKvLoraRank, hk::st_16x16_s>>;
    using st_kv_rope = std::conditional_t<std::is_same_v<kv_t, hk::fp8e4m3>,
                                          hk::st_fp8e4m3<kBlockN, kQkRopeHeadDim, hk::st_16x16_s>,
                                          hk::st_bf<kBlockN, kQkRopeHeadDim, hk::st_16x16_s>>;
};

template <typename T>
__global__ __launch_bounds__(T::kNumThreads, T::kOccupancy) __attribute__((
    amdgpu_num_vgpr(72))) void kn_mla_v32_fwd_decode_h32_fp8_fp8_v8(HkMlaDecodeFwdParams<T> params)
{
    using q_t     = T::q_t;
    using kv_t    = T::kv_t;
    using out_t   = T::out_t;
    using comp_t  = float;
    using split_t = float;

    using G = hk::group<T::kNumWarps>;

    constexpr comp_t log2e = 1.4426950408889634;

    const int32_t worker_idx     = blockIdx.x;
    const int32_t work_start_idx = __builtin_amdgcn_readfirstlane(params.p_work_indptr[worker_idx]);
    const int32_t work_end_idx =
        __builtin_amdgcn_readfirstlane(params.p_work_indptr[worker_idx + 1]);
    if(work_start_idx >= work_end_idx)
    {
        return;
    }

    // Reg tiles — identical layout to v7.
    constexpr uint32_t k_o_sz      = 128;
    constexpr uint32_t k_p_mfma_sz = 2;
    constexpr uint32_t k_p_comp_sz = 8;
    constexpr uint32_t k_kv_size   = 4;
    constexpr uint32_t k_q_rope_sz = 4;
    constexpr uint32_t k_q_nope_sz = 32;

    constexpr uint32_t k_o_end        = 255;
    constexpr uint32_t k_o_begin      = k_o_end - k_o_sz + 1;
    constexpr uint32_t k_p_comp_end   = k_o_begin - 1;
    constexpr uint32_t k_p_comp_begin = k_p_comp_end - k_p_comp_sz + 1;
    constexpr uint32_t k_p_mfma_end   = k_p_comp_begin + k_p_mfma_sz - 1;
    constexpr uint32_t k_p_mfma_begin = k_p_mfma_end - k_p_mfma_sz + 1;
    constexpr uint32_t k_kv_1_end     = k_p_comp_begin - 1;
    constexpr uint32_t k_kv_1_begin   = k_kv_1_end - k_kv_size + 1;
    constexpr uint32_t k_kv_0_end     = k_kv_1_begin - 1;
    constexpr uint32_t k_kv_0_begin   = k_kv_0_end - k_kv_size + 1;
    constexpr uint32_t k_q_rope_end   = k_kv_0_begin - 1;
    constexpr uint32_t k_q_rope_begin = k_q_rope_end - k_q_rope_sz + 1;
    constexpr uint32_t k_q_nope_end   = k_q_rope_begin - 1;
    constexpr uint32_t k_q_nope_begin = k_q_nope_end - k_q_nope_sz + 1;

    using q_nope_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_q_nope_begin, k_q_nope_end>>, 2>;
    using q_rope_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_q_rope_begin, k_q_rope_end>>, 2>;
    using kv_0_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_kv_0_begin, k_kv_0_end>>, 2>;
    using kv_1_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_kv_1_begin, k_kv_1_end>>, 2>;
    using p_comp_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_p_comp_begin, k_p_comp_end>>, 4>;
    using p_mfma_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_p_mfma_begin, k_p_mfma_end>>, 2>;
    using o_ranges =
        hkdart::split_many_t<hkdart::type_list<hkdart::range<k_o_begin, k_o_end>>, 4>;

    hkdart::clobber<q_nope_ranges>();
    hkdart::clobber<q_rope_ranges>();
    hkdart::clobber<kv_0_ranges>();
    hkdart::clobber<kv_1_ranges>();
    hkdart::clobber<p_comp_ranges>();
    hkdart::clobber<p_mfma_ranges>();
    hkdart::clobber<o_ranges>();

    QManagerV3<T> q_manager;
    KvManagerV2<T> kv_manager;
    VtManagerV1<T> vt_manager;
    OManager16bitsV2<T, out_t> o_manager;
    OManager32bitsV2<T, split_t> split_o_manager;

    hk::art<kv_t, T::kBlockK, T::kBlockN, hk::row_l, hk::rt_16x32_s, kv_0_ranges> kv_0;
    hk::art<kv_t, T::kBlockK, T::kBlockN, hk::row_l, hk::rt_16x32_s, kv_1_ranges> kv_1;
    hk::art<comp_t, T::kBlockN, T::kTileM, hk::col_l, hk::rt_16x16_s, p_comp_ranges> p_comp;
    hk::art<kv_t, T::kTileM, T::kBlockN, hk::row_l, hk::rt_16x32_s, p_mfma_ranges> p_mfma;
    hk::art<comp_t, T::kTileM, T::kVoHeadDim, hk::row_l, hk::rt_16x16_s, o_ranges> oaccu;

    const uint32_t warp_idx           = ckt::get_warp_id();
    const uint32_t lane_idx           = ckt::get_lane_id();
    const uint32_t kv_ld_row_base_idx = kv_manager.get_kv_ld_row_base_idx(warp_idx);
    const uint32_t kv_ld_col_base     = kv_manager.get_kv_ld_col_base(warp_idx);

    const uintptr_t out_as_int       = reinterpret_cast<uintptr_t>(params.final_output.raw_ptr);
    const uint64_t out_as_u64        = static_cast<uint64_t>(out_as_int);
    const hk::buffer_resource out_br = hk::make_buffer_resource(out_as_u64, 0xFFFFFFFF, 0x00020000);
    const uintptr_t split_out_as_int = reinterpret_cast<uintptr_t>(params.split_output.raw_ptr);
    const uint64_t split_out_as_u64  = static_cast<uint64_t>(split_out_as_int);
    const hk::buffer_resource split_out_br =
        hk::make_buffer_resource(split_out_as_u64, 0xFFFFFFFF, 0x00020000);

    extern __shared__ int32_t p_lds[];

    constexpr uint32_t kSzLdsQ  = q_manager.get_lds_size_in_byte();
    constexpr uint32_t kSzLdsKv = kv_manager.get_lds_size_in_byte();
    constexpr uint32_t kSzLdsTv = vt_manager.get_lds_size_in_byte();
    constexpr uint32_t kSzLdsO =
        ckt::max(o_manager.get_lds_size_in_byte(), split_o_manager.get_lds_size_in_byte());

    static_assert(kSzLdsO <= kSzLdsKv,
                  "kSzLdsO must be less than or equal to kSzLdsKv because we want to reuse p_lds_o "
                  "and p_lds_kv_next.");

    const uintptr_t p_lds_vt = reinterpret_cast<uintptr_t>(p_lds);
    const uintptr_t p_lds_q  = p_lds_vt + kSzLdsTv;

    for(int32_t work_idx = work_start_idx; work_idx < work_end_idx; ++work_idx)
    {
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        const int32_t partial_qo_loc = __builtin_amdgcn_readfirstlane(
            params.p_work_info_set[work_idx * kSizeMlaWorkInfoInDw + 1]);
        const int32_t qo_start = __builtin_amdgcn_readfirstlane(
            params.p_work_info_set[work_idx * kSizeMlaWorkInfoInDw + 2]);
        const int32_t qo_end = __builtin_amdgcn_readfirstlane(
            params.p_work_info_set[work_idx * kSizeMlaWorkInfoInDw + 3]);
        const int32_t kv_start = __builtin_amdgcn_readfirstlane(
            params.p_work_info_set[work_idx * kSizeMlaWorkInfoInDw + 4]);
        const int32_t kv_end = __builtin_amdgcn_readfirstlane(
            params.p_work_info_set[work_idx * kSizeMlaWorkInfoInDw + 5]);
        const int32_t kv_len = kv_end - kv_start;

        // v8 CHANGE #1: inner loop over all qo positions in this work_info.
        // v7 hardcoded qo_start (single position). At qseqlen=8, metadata emits
        // qo_end-qo_start=8 and v7 silently dropped positions 1-7.
        // Each q_pos runs the full sq=1 pipeline with its own K/V sweep.
        // Correctness-first; perf-parity with "Python split" approach.
        const int32_t qo_count = qo_end - qo_start;

        for(int32_t qp_rel = 0; qp_rel < qo_count; ++qp_rel)
        {
            const int32_t qo_pos = qo_start + qp_rel;

            // Between q_pos iterations: drain all outstanding LDS writes + ensure
            // no VMEM stragglers from previous position's K/V fetch. Needed
            // because p_lds_kv_curr is reused across positions.
            if(qp_rel > 0)
            {
                __builtin_amdgcn_s_waitcnt(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);
            }

            uintptr_t p_lds_kv_curr = p_lds_q + kSzLdsQ;
            uintptr_t p_lds_kv_next = p_lds_kv_curr + kSzLdsKv;

            comp_t row_max;
            comp_t row_sum_e;

            int32_t row_kv_ld;
            if(kv_len < T::kBlockN)
            {
                row_kv_ld =
                    get_kv_ld_row<true>(params.p_kv_indices, kv_ld_row_base_idx, kv_start, kv_end);
            }
            else
            {
                row_kv_ld = get_kv_ld_row<false>(
                    params.p_kv_indices, kv_ld_row_base_idx, kv_start, kv_start + T::kBlockN);
            }

            // Load Q for THIS position.
            q_manager.template load_q_to_gpr<k_q_nope_begin, k_q_rope_begin>(
                params.query, warp_idx, qo_pos, p_lds_q);
            __builtin_amdgcn_sched_barrier(0);

            // First K tile — virtual-warp loop (v7 logic, unchanged).
            #pragma unroll
            for (uint32_t vi = 0; vi < T::kVirtualPerReal; vi++) {
                const uint32_t vwarp_k = warp_idx + vi * T::kNumWarps;
                const uint32_t vkv_ld_row_base = kv_manager.get_kv_ld_row_base_idx(vwarp_k);
                const uint32_t vkv_ld_col = kv_manager.get_kv_ld_col_base(vwarp_k);
                int32_t vrow_kv_ld;
                if (kv_len < T::kBlockN)
                    vrow_kv_ld = get_kv_ld_row<true>(params.p_kv_indices, vkv_ld_row_base, kv_start, kv_end);
                else
                    vrow_kv_ld = get_kv_ld_row<false>(params.p_kv_indices, vkv_ld_row_base, kv_start, kv_start + T::kBlockN);
                if (kv_len < T::kBlockN) {
                    kv_manager.template async_load_k<false, true>(
                        p_lds_kv_curr, vwarp_k, params.kv_buffer, vrow_kv_ld, vkv_ld_col);
                } else {
                    kv_manager.template async_load_k<false, false>(
                        p_lds_kv_curr, vwarp_k, params.kv_buffer, vrow_kv_ld, vkv_ld_col);
                }
            }

            int32_t row_kv_ld_next_next = -1;
            if(kv_len >= 2 * T::kBlockN)
            {
                row_kv_ld_next_next = get_kv_ld_row<false>(params.p_kv_indices,
                                                           kv_ld_row_base_idx,
                                                           kv_start + T::kBlockN,
                                                           kv_start + 2 * T::kBlockN);
            }
            else if(kv_len > T::kBlockN)
            {
                row_kv_ld_next_next = get_kv_ld_row<true>(
                    params.p_kv_indices, kv_ld_row_base_idx, kv_start + T::kBlockN, kv_end);
            }

            auto mla_main = [&]<bool kIsFirstIter,
                                PvGemmEpilogueType kEpilogueType,
                                bool kCheckBoundary,
                                bool kCheckBoundaryNext>(const int32_t kv_tile_start,
                                                         const int32_t kv_tile_end) {
                constexpr bool kIsLastIter = (kEpilogueType != PvGemmEpilogueType::None);

                static_assert((kCheckBoundary == false) || (kIsLastIter == true));
                static_assert((kIsLastIter == false) || (kCheckBoundaryNext == false));

                __builtin_amdgcn_s_waitcnt(0);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                uintptr_t p_lds_kv_next_warp;
                int32_t row_kv_ld_next;
                if constexpr(kIsLastIter == false)
                {
                    p_lds_kv_next_warp = kv_manager.get_p_lds_kv_warp_base(warp_idx, p_lds_kv_next);
                    row_kv_ld_next     = row_kv_ld_next_next;
                }

                if constexpr(kIsLastIter == false) {
                    #pragma unroll
                    for (uint32_t vi = 0; vi < T::kVirtualPerReal; vi++) {
                        const uint32_t vwarp_k = warp_idx + vi * T::kNumWarps;
                        const uint32_t vkv_ld_row_base = kv_manager.get_kv_ld_row_base_idx(vwarp_k);
                        const uint32_t vkv_ld_col = kv_manager.get_kv_ld_col_base(vwarp_k);
                        int32_t vrow_kv_ld_next;
                        if constexpr(kCheckBoundaryNext) {
                            vrow_kv_ld_next = get_kv_ld_row<true>(params.p_kv_indices,
                                vkv_ld_row_base, kv_tile_start + T::kBlockN, kv_end);
                        } else {
                            vrow_kv_ld_next = get_kv_ld_row<false>(params.p_kv_indices,
                                vkv_ld_row_base, kv_tile_start + T::kBlockN, kv_tile_start + 2 * T::kBlockN);
                        }
                        if constexpr(kCheckBoundaryNext) {
                            kv_manager.template async_load_k<false, true>(
                                p_lds_kv_next, vwarp_k, params.kv_buffer, vrow_kv_ld_next, vkv_ld_col);
                        } else {
                            kv_manager.template async_load_k<false, false>(
                                p_lds_kv_next, vwarp_k, params.kv_buffer, vrow_kv_ld_next, vkv_ld_col);
                        }
                    }
                }

                // GEMM on NoPE — v8 CHANGE #2: s_setprio coverage around MFMAs.
                // v7 only priority-tuned oaccu rescale. ASM pattern: raise prio
                // to 14 during MFMA, lower to 0 during VALU softmax to let
                // the two categories dual-issue on CDNA4.
                constexpr uint32_t num_nope_iter = (k_q_nope_end + 1 - k_q_nope_begin) / 4;
                ckt::static_for<0, num_nope_iter, 1>{}([&](auto idx) {
                    constexpr uint32_t reg_start = idx.value * 4 + k_q_nope_begin;
                    using q_range_0 =
                        hkdart::split_many_t<hkdart::type_list<hkdart::range<reg_start, reg_start + 1>>,
                                             2>;
                    using q_range_1 = hkdart::
                        split_many_t<hkdart::type_list<hkdart::range<reg_start + 2, reg_start + 3>>, 2>;
                    hk::art<q_t, T::kTileM, T::kBlockK, hk::row_l, hk::rt_16x32_s, q_range_0> q_0;
                    hk::art<q_t, T::kTileM, T::kBlockK, hk::row_l, hk::rt_16x32_s, q_range_1> q_1;

                    constexpr int32_t tile_idx = (reg_start - k_q_nope_begin) / 2;
                    kv_manager.template load_k_to_gpr<0, (tile_idx + 0) * T::kBlockK>(kv_0, p_lds_kv_curr);
                    kv_manager.template load_k_to_gpr<16, (tile_idx + 0) * T::kBlockK>(kv_0, p_lds_kv_curr);
                    kv_manager.template load_k_to_gpr<0, (tile_idx + 1) * T::kBlockK>(kv_1, p_lds_kv_curr);
                    kv_manager.template load_k_to_gpr<16, (tile_idx + 1) * T::kBlockK>(kv_1, p_lds_kv_curr);

                    asm volatile("s_waitcnt lgkmcnt(2)");
                    __builtin_amdgcn_s_setprio(14);  // v8 OPT-E: prio up for MFMA
                    if constexpr(idx.value == 0)
                    {
                        hk::mma_ABt(p_comp, kv_0, q_0);
                    }
                    else
                    {
                        hk::mma_ABt(p_comp, kv_0, q_0, p_comp);
                    }
                    asm volatile("s_waitcnt lgkmcnt(0)");
                    hk::mma_ABt(p_comp, kv_1, q_1, p_comp);
                    __builtin_amdgcn_s_setprio(0);   // v8 OPT-E: prio down for non-MFMA
                });

                constexpr uint32_t num_rope_iter = (k_q_rope_end + 1 - k_q_rope_begin) / 4;
                ckt::static_for<0, num_rope_iter, 1>{}([&](auto idx) {
                    constexpr uint32_t reg_start = idx.value * 4 + k_q_rope_begin;
                    using q_range_0 =
                        hkdart::split_many_t<hkdart::type_list<hkdart::range<reg_start, reg_start + 1>>,
                                             2>;
                    using q_range_1 = hkdart::
                        split_many_t<hkdart::type_list<hkdart::range<reg_start + 2, reg_start + 3>>, 2>;
                    hk::art<q_t, T::kTileM, T::kBlockK, hk::row_l, hk::rt_16x32_s, q_range_0> q_0;
                    hk::art<q_t, T::kTileM, T::kBlockK, hk::row_l, hk::rt_16x32_s, q_range_1> q_1;

                    constexpr int32_t tile_idx = (reg_start - k_q_rope_begin) / 2;
                    kv_manager.template load_k_to_gpr<0, (tile_idx + 0 + 16) * T::kBlockK>(kv_0, p_lds_kv_curr);
                    kv_manager.template load_k_to_gpr<16, (tile_idx + 0 + 16) * T::kBlockK>(kv_0, p_lds_kv_curr);
                    kv_manager.template load_k_to_gpr<0, (tile_idx + 1 + 16) * T::kBlockK>(kv_1, p_lds_kv_curr);
                    kv_manager.template load_k_to_gpr<16, (tile_idx + 1 + 16) * T::kBlockK>(kv_1, p_lds_kv_curr);

                    asm volatile("s_waitcnt lgkmcnt(2)");
                    __builtin_amdgcn_s_setprio(14);  // v8 OPT-E
                    hk::mma_ABt(p_comp, kv_0, q_0, p_comp);
                    asm volatile("s_waitcnt lgkmcnt(0)");
                    hk::mma_ABt(p_comp, kv_1, q_1, p_comp);
                    __builtin_amdgcn_s_setprio(0);   // v8 OPT-E
                });

                v8ui v;
                kv_manager.load_v_to_gpr(&v, warp_idx, p_lds_kv_curr);

                if constexpr((kIsLastIter == false) && (kCheckBoundaryNext == false))
                {
                    if((kv_tile_start + 2 * T::kBlockN) < kv_end)
                    {
                        if((kv_tile_start + 3 * T::kBlockN) <= kv_end)
                        {
                            row_kv_ld_next_next = get_kv_ld_row<false>(params.p_kv_indices,
                                                                       kv_ld_row_base_idx,
                                                                       kv_tile_start + 2 * T::kBlockN,
                                                                       kv_tile_end + 2 * T::kBlockN);
                        }
                        else
                        {
                            row_kv_ld_next_next = get_kv_ld_row<true>(params.p_kv_indices,
                                                                      kv_ld_row_base_idx,
                                                                      kv_tile_start + 2 * T::kBlockN,
                                                                      kv_end);
                        }
                    }
                }

                const uint32_t col_0_idx = lane_idx >> 4;
                softmax_scale_p<kCheckBoundary, k_p_comp_begin>(
                    col_0_idx * 4 + kv_tile_start, kv_end, params.softmax_scale);

                comp_t local_max = max_8<k_p_comp_begin, comp_t>();

                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_sched_barrier(0);

                constexpr int32_t reduce_range = ckt::get_warp_size();
                constexpr int32_t stop_stride  = ckt::get_warp_size() / 4 - 1;
                local_max                      = aiter::
                    warpReduce<aiter::MaxFunctor, decltype(local_max), reduce_range, stop_stride>(
                        local_max);
                vt_manager.transpose_v(&v);

                const comp_t new_row_max = kIsFirstIter ? local_max : ckt::max(local_max, row_max);
                const comp_t rescale =
                    kIsFirstIter ? 1.0f : __builtin_amdgcn_exp2f((row_max - new_row_max) * log2e);
                row_max = new_row_max;

                softmax_p1<kIsFirstIter, k_p_comp_begin>(&row_sum_e, row_max, rescale);

                const uintptr_t p_lds_o    = kIsLastIter ? p_lds_kv_curr : 0;
                const float reci_row_sum_e = kIsLastIter ? (1.0f / row_sum_e) : .0f;

                vt_manager.store_transposed_v_to_lds(p_lds_vt, warp_idx, v);
                #pragma unroll
                for (uint32_t vi = 1; vi < T::kVirtualPerReal; vi++) {
                    const uint32_t vwarp_v = warp_idx + vi * T::kNumWarps;
                    v8ui v_slice;
                    kv_manager.load_v_to_gpr(&v_slice, vwarp_v, p_lds_kv_curr);
                    asm volatile("s_waitcnt lgkmcnt(0)");
                    vt_manager.transpose_v(&v_slice);
                    vt_manager.store_transposed_v_to_lds(p_lds_vt, vwarp_v, v_slice);
                }

                if constexpr(kIsFirstIter == false)
                {
                    __builtin_amdgcn_s_setprio(8);
                    hk::mul_vgpr<0, 0>(oaccu, oaccu, rescale);
                    hk::mul_vgpr<0, 1>(oaccu, oaccu, rescale);
                    hk::mul_vgpr<0, 2>(oaccu, oaccu, rescale);
                    hk::mul_vgpr<0, 3>(oaccu, oaccu, rescale);
                    __builtin_amdgcn_s_setprio(7);
                    hk::mul_vgpr<0, 4>(oaccu, oaccu, rescale);
                    hk::mul_vgpr<0, 5>(oaccu, oaccu, rescale);
                    hk::mul_vgpr<0, 6>(oaccu, oaccu, rescale);
                    hk::mul_vgpr<0, 7>(oaccu, oaccu, rescale);
                    __builtin_amdgcn_s_setprio(6);
                    hk::mul_vgpr<0, 8>(oaccu, oaccu, rescale);
                    hk::mul_vgpr<0, 9>(oaccu, oaccu, rescale);
                    hk::mul_vgpr<0, 10>(oaccu, oaccu, rescale);
                    hk::mul_vgpr<0, 11>(oaccu, oaccu, rescale);
                    __builtin_amdgcn_s_setprio(5);
                    hk::mul_vgpr<0, 12>(oaccu, oaccu, rescale);
                    hk::mul_vgpr<0, 13>(oaccu, oaccu, rescale);
                    hk::mul_vgpr<0, 14>(oaccu, oaccu, rescale);
                    hk::mul_vgpr<0, 15>(oaccu, oaccu, rescale);
                    __builtin_amdgcn_s_setprio(4);
                    hk::mul_vgpr<0, 16>(oaccu, oaccu, rescale);
                    hk::mul_vgpr<0, 17>(oaccu, oaccu, rescale);
                    hk::mul_vgpr<0, 18>(oaccu, oaccu, rescale);
                    hk::mul_vgpr<0, 19>(oaccu, oaccu, rescale);
                    __builtin_amdgcn_s_setprio(3);
                    hk::mul_vgpr<0, 20>(oaccu, oaccu, rescale);
                    hk::mul_vgpr<0, 21>(oaccu, oaccu, rescale);
                    hk::mul_vgpr<0, 22>(oaccu, oaccu, rescale);
                    hk::mul_vgpr<0, 23>(oaccu, oaccu, rescale);
                    __builtin_amdgcn_s_setprio(2);
                    hk::mul_vgpr<0, 24>(oaccu, oaccu, rescale);
                    hk::mul_vgpr<0, 25>(oaccu, oaccu, rescale);
                    hk::mul_vgpr<0, 26>(oaccu, oaccu, rescale);
                    hk::mul_vgpr<0, 27>(oaccu, oaccu, rescale);
                    __builtin_amdgcn_s_setprio(1);
                    hk::mul_vgpr<0, 28>(oaccu, oaccu, rescale);
                    hk::mul_vgpr<0, 29>(oaccu, oaccu, rescale);
                    hk::mul_vgpr<0, 30>(oaccu, oaccu, rescale);
                    hk::mul_vgpr<0, 31>(oaccu, oaccu, rescale);
                    __builtin_amdgcn_s_setprio(0);
                }

                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                pack_4f32_to_fp8<k_p_mfma_begin, k_p_comp_begin, true>();
                pack_4f32_to_fp8<k_p_mfma_begin, k_p_comp_begin + 2, false>();
                pack_4f32_to_fp8<k_p_mfma_begin + 1, k_p_comp_begin + 4, true>();
                pack_4f32_to_fp8<k_p_mfma_begin + 1, k_p_comp_begin + 6, false>();

                constexpr uint32_t num_pv_iter = T::kVoHeadDim / (T::kBlockK * 2);
                ckt::static_for<0, num_pv_iter, 1>{}([&](auto idx) {
                    constexpr uint32_t oaccu_base = k_o_begin + idx.value * 8 * 2;
                    using oaccu_range_0           = hkdart::split_many_t<
                        hkdart::type_list<hkdart::range<oaccu_base + 0, oaccu_base + 8 - 1>>, 4>;
                    using oaccu_range_1 = hkdart::split_many_t<
                        hkdart::type_list<hkdart::range<oaccu_base + 8, oaccu_base + 16 - 1>>, 4>;
                    hk::art<comp_t, T::kBlockK, T::kTileM, hk::col_l, hk::rt_16x16_s, oaccu_range_0> oaccu_0;
                    hk::art<comp_t, T::kBlockK, T::kTileM, hk::col_l, hk::rt_16x16_s, oaccu_range_1> oaccu_1;

                    constexpr uint32_t kColOffsetDelta = T::kBlockK / 2;
                    constexpr uint32_t kColOffset0     = idx.value * T::kBlockK * 2;
                    constexpr uint32_t kColOffset1     = kColOffset0 + kColOffsetDelta * 1;
                    constexpr uint32_t kColOffset2     = kColOffset0 + kColOffsetDelta * 2;
                    constexpr uint32_t kColOffset3     = kColOffset0 + kColOffsetDelta * 3;

                    vt_manager.template load_transposed_v_to_gpr<kColOffset0, k_kv_0_begin>(p_lds_vt);
                    vt_manager.template load_transposed_v_to_gpr<kColOffset1, k_kv_0_begin + 2>(p_lds_vt);
                    vt_manager.template load_transposed_v_to_gpr<kColOffset2, k_kv_1_begin>(p_lds_vt);
                    vt_manager.template load_transposed_v_to_gpr<kColOffset3, k_kv_1_begin + 2>(p_lds_vt);

                    asm volatile("s_waitcnt lgkmcnt(4)");
                    __builtin_amdgcn_s_setprio(14);  // v8 OPT-E: PV MFMA prio
                    if constexpr(kIsFirstIter)
                    {
                        hk::mma_ABt(oaccu_0, kv_0, p_mfma);
                    }
                    else
                    {
                        hk::mma_ABt(oaccu_0, kv_0, p_mfma, oaccu_0);
                    }

                    asm volatile("s_waitcnt lgkmcnt(0)");
                    if constexpr(kIsFirstIter)
                    {
                        hk::mma_ABt(oaccu_1, kv_1, p_mfma);
                    }
                    else
                    {
                        hk::mma_ABt(oaccu_1, kv_1, p_mfma, oaccu_1);
                    }
                    __builtin_amdgcn_s_setprio(0);   // v8 OPT-E

                    if constexpr(kIsLastIter)
                    {
                        constexpr uint32_t col_offset = idx.value * (T::kBlockK * 2);

                        hk::mul_vgpr(oaccu_0, oaccu_0, reci_row_sum_e);
                        hk::mul_vgpr(oaccu_1, oaccu_1, reci_row_sum_e);

                        if constexpr(kEpilogueType == PvGemmEpilogueType::OutputFinal)
                        {
                            // v8 CHANGE #1 (cont'd): output at qo_pos (this position's slot).
                            o_manager.template output_to_vram<oaccu_base, col_offset>(
                                params.final_output.raw_ptr, warp_idx, qo_pos, p_lds_o);
                            o_manager.template output_to_vram<oaccu_base + 8, col_offset + T::kBlockK>(
                                params.final_output.raw_ptr, warp_idx, qo_pos, p_lds_o);
                        }
                        else
                        {
                            split_o_manager.template output_to_vram<oaccu_base, col_offset>(
                                params.split_output.raw_ptr, warp_idx, partial_qo_loc, p_lds_o);
                            split_o_manager
                                .template output_to_vram<oaccu_base + 8, col_offset + T::kBlockK>(
                                    params.split_output.raw_ptr, warp_idx, partial_qo_loc, p_lds_o);
                        }
                    }
                });

                if constexpr(kIsLastIter == false)
                {
                    std::swap(p_lds_kv_curr, p_lds_kv_next);
                }
                else if constexpr(kEpilogueType == PvGemmEpilogueType::OutputSplit)
                {
                    constexpr uint32_t kMfmaResultRows = 16;
                    if(lane_idx < kMfmaResultRows)
                    {
                        constexpr comp_t inv_log2e = 1.0 / log2e;
                        const uint32_t row_idx =
                            lane_idx % 16 + warp_idx * 16 + partial_qo_loc * T::kQoNumHead;
                        const comp_t lse = row_max + __builtin_amdgcn_logf(row_sum_e) * inv_log2e;
                        params.split_lse.raw_ptr[row_idx] = lse;
                    }
                }
            };

            if(kv_len < T::kBlockN)
            {
                if(partial_qo_loc < 0)
                {
                    mla_main.template operator()<true, PvGemmEpilogueType::OutputFinal, true, false>(
                        kv_start, kv_end);
                }
                else
                {
                    mla_main.template operator()<true, PvGemmEpilogueType::OutputSplit, true, false>(
                        kv_start, kv_end);
                }
            }
            else if(kv_len == T::kBlockN)
            {
                if(partial_qo_loc < 0)
                {
                    mla_main.template operator()<true, PvGemmEpilogueType::OutputFinal, false, false>(
                        kv_start, kv_end);
                }
                else
                {
                    mla_main.template operator()<true, PvGemmEpilogueType::OutputSplit, false, false>(
                        kv_start, kv_end);
                }
            }
            else
            {
                const int32_t kv_1st_end = kv_start + T::kBlockN;
                if((kv_1st_end + T::kBlockN - 1) < kv_end)
                {
                    mla_main.template operator()<true, PvGemmEpilogueType::None, false, false>(
                        kv_start, kv_1st_end);
                }
                else
                {
                    mla_main.template operator()<true, PvGemmEpilogueType::None, false, true>(
                        kv_start, kv_1st_end);
                }

                int32_t kv_idx = kv_1st_end;
                while((kv_idx + T::kBlockN) < kv_end)
                {
                    if((kv_idx + 2 * T::kBlockN - 1) < kv_end)
                    {
                        mla_main.template operator()<false, PvGemmEpilogueType::None, false, false>(
                            kv_idx, kv_idx + T::kBlockN);
                    }
                    else
                    {
                        mla_main.template operator()<false, PvGemmEpilogueType::None, false, true>(
                            kv_idx, kv_idx + T::kBlockN);
                    }
                    kv_idx += T::kBlockN;
                }

                if((kv_idx + T::kBlockN) == kv_end)
                {
                    if(partial_qo_loc < 0)
                    {
                        mla_main
                            .template operator()<false, PvGemmEpilogueType::OutputFinal, false, false>(
                                kv_idx, kv_end);
                    }
                    else
                    {
                        mla_main
                            .template operator()<false, PvGemmEpilogueType::OutputSplit, false, false>(
                                kv_idx, kv_end);
                    }
                }
                else
                {
                    if(partial_qo_loc < 0)
                    {
                        mla_main.template
                        operator()<false, PvGemmEpilogueType::OutputFinal, true, false>(kv_idx, kv_end);
                    }
                    else
                    {
                        mla_main.template
                        operator()<false, PvGemmEpilogueType::OutputSplit, true, false>(kv_idx, kv_end);
                    }
                }
            }
        }   // end of v8 inner q_pos loop
    }
}

template <typename Traits>
void mla_v32_fwd_decode_h32_fp8_fp8_v8(torch::Tensor& query,
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
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));

    const hipStream_t stream = at::hip::getCurrentHIPStream();

    HkMlaDecodeFwdParams<Traits> params = {
        hk::make_gl<typename Traits::gl_q>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(query.data_ptr())),
            query.size(0),
            Traits::kNumTilesM,
            Traits::kTileM,
            Traits::kQkHeadDim),
        hk::make_gl<typename Traits::gl_kv>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(kv_buffer.data_ptr())),
            kv_buffer.size(0),
            Traits::kPageSize,
            Traits::kKvNumHead,
            Traits::kQkHeadDim),
        kv_page_indices.data_ptr<int32_t>(),
        work_indptr.data_ptr<int32_t>(),
        work_info_set.data_ptr<int32_t>(),
        hk::make_gl<typename Traits::gl_o>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(final_output.data_ptr())),
            1,
            final_output.size(0),
            Traits::kQoNumHead,
            Traits::kVoHeadDim),
        hk::make_gl<typename Traits::gl_so>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(split_output.data_ptr())),
            1,
            split_output.size(0),
            Traits::kQoNumHead,
            Traits::kVoHeadDim),
        hk::make_gl<typename Traits::gl_slse>(
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(split_lse.data_ptr())),
            1,
            split_lse.size(0),
            Traits::kQoNumHead,
            1),
        softmax_scale};

    const dim3 grid        = dim3(dev_prop.multiProcessorCount);
    const int32_t lds_size = dev_prop.maxSharedMemoryPerMultiProcessor / Traits::kOccupancy;

    kn_mla_v32_fwd_decode_h32_fp8_fp8_v8<Traits>
        <<<grid, Traits::kNumThreads, lds_size, stream>>>(params);
}

void hk_mi3xx_mla_v32_fwd_decode_h32_fp8_fp8_v8(torch::Tensor& query,
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
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(final_output));

    const bool q_is_fp8 = (query.scalar_type() == at::ScalarType::Float8_e4m3fn) ||
                          (query.scalar_type() == at::ScalarType::Float8_e4m3fnuz);
    const bool kv_is_fp8 = (kv_buffer.scalar_type() == at::ScalarType::Float8_e4m3fn) ||
                           (kv_buffer.scalar_type() == at::ScalarType::Float8_e4m3fnuz);

    if(q_is_fp8 && kv_is_fp8)
    {
        using Traits = HkMlaDecodeFwdTraitsH32V8<hk::fp8e4m3, hk::fp8e4m3, hk::bf16, 32>;
        mla_v32_fwd_decode_h32_fp8_fp8_v8<Traits>(query,
                                                   kv_buffer,
                                                   qo_indptr,
                                                   kv_indptr,
                                                   kv_page_indices,
                                                   kv_last_page_lens,
                                                   work_indptr,
                                                   work_info_set,
                                                   max_seqlen_q,
                                                   softmax_scale,
                                                   split_output,
                                                   split_lse,
                                                   final_output);
    }
    else
    {
        TORCH_CHECK(false,
                    "hk_mi3xx_mla_v32_fwd_decode_h32_fp8_fp8_v8 doesn't support q type ",
                    toString(query.scalar_type()),
                    " and kv type",
                    toString(kv_buffer.scalar_type()),
                    ".");
    }
}

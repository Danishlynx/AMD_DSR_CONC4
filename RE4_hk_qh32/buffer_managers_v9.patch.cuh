// ============================================================================
// V9 WIDE LOAD FUNCTIONS for hk_mla_buffer_managers.cuh
//
// Two additions, both strictly additive (existing narrow loads unchanged):
//
// 1. load_k_wide_to_gpr in KvManagerV2 — reads 16×128 fp8 per call, 8 VGPRs/lane
// 2. lds_2_gpr_wide in QManagerV4       — reads 16×128 fp8 per call, 8 VGPRs/lane
//
// Both follow the MFMA 16x16x128 quadrant lane layout:
//   row = lane_idx % 16
//   quadrant = lane_idx / 16 ∈ {0,1,2,3}
//   quadrant q owns K-cols [q*32, (q+1)*32) of its row (32 cols, contiguous)
//
// 128-col span crosses 2 LDS blocks (each block is 64 cols):
//   quadrants 0,1 read from block (kColOffset/64)
//   quadrants 2,3 read from block (kColOffset/64)+1
//   half within block = quadrant % 2 (0 or 32-col offset)
// ============================================================================


// === PATCH_SECTION_A_KVMANAGERV2_LOAD_K_WIDE ===
// Insert INSIDE KvManagerV2 class body (after existing load_k_to_gpr), circa line 990:

    // V9: Load 16x128 fp8 from LDS to GPR using NARROW-INTERLEAVED lane layout.
    // Same lane mapping as 4 adjacent load_k_to_gpr calls at kColOffsets {0,32,64,96},
    // but writes 8 VGPRs per lane (not just 2). Per-lane data: cols {0-7, 32-39, 64-71, 96-103}
    // of row (4 non-contiguous 8-col stripes at 32-col stride).
    //
    // Covers 1 row-half (16 rows); call twice at kRowOffset={0,16} for 32 rows of outer art.
    // kColOffset is 32-aligned (not 128); the 128-col span is kColOffset..kColOffset+127.
    template <uint32_t kRowOffset, uint32_t kColOffset, hkdart::all RT>
    __device__ __forceinline__ static void load_k_wide_to_gpr(RT& dst, const uintptr_t p_lds_kv)
    {
        static_assert(((kRowOffset % 16) == 0) && (kRowOffset < 32),
                      "load_k_wide_to_gpr(): Unsupported row offset!");
        static_assert(((kColOffset % 32) == 0) && (kColOffset + 128 <= T::kQkHeadDim),
                      "load_k_wide_to_gpr(): kColOffset must be 32-aligned and span 128 cols in head_dim!");

        constexpr uint32_t kMfmaRows = 16;
        constexpr uint32_t kMfmaElemPerThr = 8;

        const uint32_t lane_idx = ckt::get_lane_id();
        const uint32_t row      = lane_idx % kMfmaRows;
        const uint32_t row_phy  = (row / 2) * 4 + (row % 2);
        const uint32_t col      = lane_idx / kMfmaRows * kMfmaElemPerThr;  // 0, 8, 16, 24

        const uintptr_t p_lds_kv_lane = p_lds_kv
            + (row_phy / 4) * kNumBytesPerSubBlock
            + (row_phy % 4) * kNumBytesPerRow
            + (col % kNumCols) * sizeof(kv_t);

        // 4 narrow slabs at kColOffsets {0, 32, 64, 96} relative to kColOffset.
        constexpr uint32_t kCO0 = kColOffset + 0;
        constexpr uint32_t kCO1 = kColOffset + 32;
        constexpr uint32_t kCO2 = kColOffset + 64;
        constexpr uint32_t kCO3 = kColOffset + 96;
        constexpr uint32_t kRowBase = (kRowOffset / 16) * 2 * kNumBytesPerRow;
        constexpr uint32_t kFixed0 = kRowBase + (kCO0 % kNumCols) * sizeof(kv_t) + (kCO0 / kNumCols) * kNumBytesPerBlock;
        constexpr uint32_t kFixed1 = kRowBase + (kCO1 % kNumCols) * sizeof(kv_t) + (kCO1 / kNumCols) * kNumBytesPerBlock;
        constexpr uint32_t kFixed2 = kRowBase + (kCO2 % kNumCols) * sizeof(kv_t) + (kCO2 / kNumCols) * kNumBytesPerBlock;
        constexpr uint32_t kFixed3 = kRowBase + (kCO3 % kNumCols) * sizeof(kv_t) + (kCO3 / kNumCols) * kNumBytesPerBlock;

        using range_type = hkdart::get_nth_range_t<typename RT::register_ranges, kRowOffset / 16>;
        static_assert(range_type::hi - range_type::lo + 1 == 8,
                      "load_k_wide_to_gpr requires 8 consecutive VGPRs per row-half");

        hkm::ds_read_b64<range_type::lo + 0>(p_lds_kv_lane, kFixed0);
        hkm::ds_read_b64<range_type::lo + 2>(p_lds_kv_lane, kFixed1);
        hkm::ds_read_b64<range_type::lo + 4>(p_lds_kv_lane, kFixed2);
        hkm::ds_read_b64<range_type::lo + 6>(p_lds_kv_lane, kFixed3);
    }


// === PATCH_SECTION_B_QMANAGERV4_LDS_2_GPR_WIDE ===
// Insert INSIDE QManagerV4 class body (after existing lds_2_gpr), circa line 435:

    // V9: Read 16×128 fp8 from Q LDS to GPR. Lane layout matches load_k_wide_to_gpr.
    // kColBase must be 128-aligned; spans 2 adjacent 64-col Q-blocks in LDS.
    // Takes base p_lds + warp_idx; computes per-warp offset internally.
    template <uint32_t GPR_START, uint32_t kColBase>
    __device__ __forceinline__ void lds_2_gpr_wide(const uintptr_t p_lds, const uint32_t warp_idx)
    {
        const uintptr_t p_lds_warp = p_lds + warp_idx * get_lds_size_per_block_in_byte();
        static_assert(kColBase % 128 == 0, "lds_2_gpr_wide: kColBase must be 128-aligned");
        static_assert(kColBase + 128 <= T::kQkHeadDim, "lds_2_gpr_wide: kColBase+128 exceeds head_dim");

        const uint32_t lane_idx = ckt::get_lane_id();
        const uint32_t row      = lane_idx % 16;
        const uint32_t row_phy  = (row / 8) * 2 + (row % 8) / 2 * 4 + (row % 2) * 1;
        const uint32_t quadrant = lane_idx / 16;
        const uint32_t q_block  = quadrant / 2;
        const uint32_t q_half   = quadrant % 2;

        // Q LDS layout per block (via vram_2_lds): kNumElemPerCol=16 rows, 64 cols, padded 4DW per 4 rows.
        // Block i of 9 blocks starts at p_lds_warp + i * get_lds_size_per_block_in_byte() (=1088 bytes).
        constexpr uint32_t kBytesPerBlock = kNumElemPerCol / 4 * kNumBytesPer4Rows; // 1088
        constexpr uint32_t kColBaseBlockIdx = kColBase / 64;

        const uintptr_t p_lds_lane = p_lds_warp
            + (kColBaseBlockIdx + q_block) * kBytesPerBlock
            + (row_phy / 4) * kNumBytesPer4Rows
            + (row_phy % 4) * kNumElemPerRow * sizeof(q_t)
            + q_half * 32 * sizeof(q_t);

        // 2 × ds_read_b128 = 32 bytes per lane (8 VGPRs, cols 0-31 of quadrant)
        hkm::ds_read_b128<GPR_START + 0>(p_lds_lane, 0);
        hkm::ds_read_b128<GPR_START + 4>(p_lds_lane, 16);
    }

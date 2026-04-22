---
name: CDNA4 (gfx950 / MI355X) ISA digest — FP8 MLA kernel reference
description: Distilled AMD CDNA4 ISA Reference Guide. MFMA instruction set for FP8 + block scaling, LDS bank rules + new ds_read_b64_tr_b8 transpose-on-read for gfx950, buffer_load/global_load_lds, dependency NOP tables, register budget (512 VGPR/wave = 256 regular + 256 AGPR), FP_ROUND/FP_DENORM mode controls. From 600-page ISA PDF at c:/Users/danis/OneDrive/Desktop/AMD/amd-instinct-cdna4-instruction-set-architecture.txt.
type: reference
originSessionId: 03086f88-0708-42af-ae59-566e34823620
---
# CDNA4 ISA digest — MI355X FP8 MLA kernel

Source: AMD CDNA4 Instruction Set Architecture Reference Guide (Aug 2025, 600 pages). Full text at `c:/Users/danis/OneDrive/Desktop/AMD/amd-instinct-cdna4-instruction-set-architecture.txt` (22867 lines, 949 KB).

## 1. MFMA instructions for FP8 on gfx950

### Dense FP8 MFMA list
| Instruction | MxNxK | Cycles | A,B | C,D | Notes |
|---|---|---:|---|---|---|
| `v_mfma_f32_16x16x32_f8_f8` | 16×16×32 | 16 | FP8 | F32 | **Currently used by v7/v8 HK (narrow)** |
| `v_mfma_f32_32x32x16_f8_f8` | 32×32×16 | 32 | FP8 | F32 | Large M-tile variant |
| `v_mfma_f32_16x16x128_f8f6f4` | 16×16×128 | 32 (if F8) | FP8/6/4 | F32 | **v9 target, 4× K-depth/call** |
| `v_mfma_f32_32x32x64_f8f6f4` | 32×32×64 | 64 (if F8) | FP8/6/4 | F32 | Larger M-tile wide variant |
| `v_mfma_scale_f32_16x16x128_f8f6f4` | 16×16×128 | 33 (F8+load) | FP8/6/4 + E8M0 | F32 | **MXFP block-scale variant** |
| `v_mfma_scale_f32_32x32x64_f8f6f4` | 32×32×64 | 65 | FP8/6/4 + E8M0 | F32 | |

### Block Exponent Scaling (Section 7.2.1) — MXFP
- Scale format: **E8M0** (8-bit exponent only, bias=127, 0xFF=NaN)
- Block size: **32 K-elements per scale** (at K=128: 4 scales/row; at K=64: 2 scales/row)
- 4-DWORD instruction: DWORD 0-1 = load-scale (VOP3P format 0xCC35), DWORD 2-3 = MFMA opcode
- Per-lane: 1 scale value, broadcast to all 64 lanes for their block
- Scale exponents ADD to dot-product exponent: `d_exp = Σ(a_exp+b_exp) + c_exp + scale_a + scale_b`
- NaN scale treated as exponent=0 (no scaling)
- DSR1 MXFP4 uses 32-element blocks → directly compatible

### 8-bit matrix register layout (Section 7.1.5)
For `v_mfma_f32_16x16x32_f8_f8`:
- **A [16×32 fp8]**: 4 fp8 per VGPR (items 0-3 in bits [7:0]-[31:24]). Lane i%16 = row; lane_group (i/16) = K-chunk. Lane 0 owns `A[0, K=0..7]` (2 VGPRs, 8 fp8 total). K_L = 32/4 = 8 per lane across 4 lane-groups.
- **B [32×16 fp8]**: Similar. Lane 0 owns `B[K=0..7, 0]`.
- **C/D [16×16 f32]**: 4 f32 per lane × 4 VGPRs = 16 rows × 16 cols. Lane 0 owns `D[0..3, 0]` + `D[8..11, 0]`.

### Encoding bits (VOP3P)
```
[31:24] Opcode
[23:21] CBSZ[2:0] — A format or broadcast size (3'b000=E4M3, 3'b001=E5M2)
[20:18] BLGP[2:0] — B format or B lane permutation
[17:14] ABID[3:0] — A broadcast block ID
[13]    ACC_CD   — C/D in AGPR (1) vs VGPR (0)
[12:11] ACC[1:0] — A/B in AGPR (bit 0→A, bit 1→B)
```

## 2. MFMA dependency resolution (NOP table, Section 7.6)

| First | Second | Required NOPs |
|---|---|---:|
| Non-MFMA VALU write VGPR | MFMA read VGPR | **2** |
| MFMA write VGPR | MFMA read same (SrcC accumulator) | **0** (chain allowed) |
| MFMA write VGPR | MFMA read same (SrcA or B) | **3** |
| MFMA write VGPR | MFMA read different opcode | **3** |
| MFMA write VGPR | VALU/VMEM/LDS read overlapping vDst | **5-20** (size-dependent) |
| XDL (16x16x32_F8) write | XDL read SrcC (exact match) | **2** |
| XDL write | XDL read SrcC (overlap) | **4-18** |

**Practical**: For back-to-back same-opcode MFMA with accumulator chaining (FA-style K-loop), 0 NOPs. Switch MFMA shape = 3 NOPs. VALU after MFMA = 5+ NOPs.

## 3. Register budget (gfx950)

| Resource | Per-wave limit | Per-CU total |
|---|---:|---:|
| VGPR (regular) | 256 | 512 VGPR/wave max (256 reg + 256 AGPR) |
| AGPR (accumulator) | 256 | — |
| SGPR | 102 | — |
| LDS | 160 KB per workgroup | 160 KB per CU |
| Workgroup size | 1024 threads | 16 waves of 64 lanes |
| Max waves/CU | 16 | — |

**Occupancy formula**: `waves_per_CU = min(16, floor(256 / regs_per_wave), floor(160KB / LDS_per_wave))`

**AGPR**: separate 256-entry pool, interchangeable with VGPR via `V_ACCVGPR_READ_B32 / V_ACCVGPR_WRITE_B32`. MFMA writes go to AGPR pool when `ACC_CD=1` in opcode. For chains of MFMA that immediately feed next MFMA, keep C in AGPR (no cross-move needed).

**Compiler directives**:
- `__launch_bounds__(blocksize, waves_per_cu)` — set occupancy target
- `__attribute__((amdgpu_num_vgpr(N)))` — hard cap VGPR usage (v7 uses 72)
- `__attribute__((amdgpu_num_sgpr(N)))` — hard cap SGPR

## 4. LDS (gfx950, 160 KB, 64 banks × 4B)

### Bank formula
```
bank = (byte_address / 4) mod 64
```

### ds_read instructions
| Instruction | Bits/lane | Phases | Bank pattern |
|---|---:|---:|---|
| `ds_read_b32` | 32 | 2 | Lane i → bank (addr+i*4)/4 mod 64 |
| `ds_read_b64` | 64 | 4 | Sequential 4-phase |
| `ds_read_b128` | 128 | 8 | Sequential 8-phase, **highest conflict risk** |
| `ds_read_b256` | 256 | 16 | — |
| `ds_read2_b{32,64}` | 2× @ different offsets | — | Useful for strided |
| `ds_read2st64_b{32,64}` | same, offsets ×64 | — | Stride=64 LDS chunks |

### **NEW gfx950: MFMA Transpose Load from LDS (Section 11.4)** 🎯
Game-changer for FP8 kernels — avoids manual transpose via VALU:

| Instruction | Element | VGPRs written | K indices loaded per call |
|---|---|---:|---|
| `ds_read_b64_tr_b16` | 16-bit | 2 | {0-3, 8-11} then {4-7, 12-15} |
| **`ds_read_b64_tr_b8`** | **8-bit (FP8!)** | **2** | **{0-7, 16-23, 32-39, 48-55} then rest** |
| `ds_read_b64_tr_b4` | 4-bit (sub-byte) | 2 | {0-15, 32-47} then {16-31, 48-63} |
| `ds_read_b96_tr_b6` | 6-bit | 3 | (no even VGPR alignment requirement) |

**Constraints**:
- EXEC mask must be all-1s (no divergent lanes)
- LDS address aligned to element size (8B for B8, 16B for B16)
- Even VGPR alignment (except B6)

**Savings**: 2-4 fewer VALU instructions per matrix tile vs manual bpermute transpose. **For FP8 MLA, replace k_load + transpose with single `ds_read_b64_tr_b8` pair** — huge reduction in MLA kernel's scalar overhead.

### Bank conflict mitigation
1. **Stride-4 (default)**: lane i → bank i, conflict-free
2. **Stride-32 HAZARD**: lane i stride 32 bytes → bank (VGPR/4 + 8*i) mod 64 = all lanes hit first 16 banks → **16x serialization**
3. **XOR swizzle** (zero LDS overhead): `addr' = addr ^ ((row >> 1 & 7) ^ ((pair>>1^pair>>2)&1)) << 4`
4. **Padding** (12.5-25% LDS overhead): pad rows to non-64-byte strides
5. **ds_permute_b32** (runtime permute): extra VALU cost but enables arbitrary reordering

## 5. Vector memory (Chapter 9)

### Instructions
- `buffer_load_dwordx{1,2,3,4}` — 32/64/96/128 bit per lane (**x3 NEW gfx950**)
- `global_load_b{32,64,96,128}` — flat address, no resource descriptor
- **`global_load_lds_{b32,b96,b128}`** (NEW gfx950) — HBM → LDS bypassing VGPRs! M0[17:0] = LDS offset
- `buffer_load_*_lds` — MUBUF variant with LDS=1 bit

### Cache control (SC[1:0], NT)
| Use case | SC | NT | L1/L2/IC behavior |
|---|---:|---:|---|
| KV cache read (reuse expected) | 0,0 | 0 | Hit LRU everywhere |
| Prefetch (no L1 pollution) | 0,0 | 1 | L1 Miss Evict, L2 Stream, IC Evict |
| Shared w/ workgroup | 1,0 | 0 | Hit LRU all |
| Device-scope write | 0,1 | 0 | Miss Evict L1, coherent bypass multi-L2 |

### Wait counters
- **vmcnt** = vector memory (buffer/flat/global/scratch) in-flight count
- **lgkmcnt** = LDS + GDS + scalar cache + messages
- **expcnt** = exports (unused in compute)
- **s_waitcnt vmcnt(N)**: continue when ≤N vmcnt operations outstanding
- Typical: `s_waitcnt vmcnt(0)` before MFMA reads VGPR from VMEM; `s_waitcnt lgkmcnt(0)` before MFMA reads VGPR from LDS
- **Tuning**: allow vmcnt=[4..16] during MFMA to overlap HBM prefetch

## 6. Softmax-relevant VALU (Chapter 6)

| Instruction | Latency | Throughput | Notes |
|---|---:|---|---|
| `v_fma_f32` | 4 cyc | 1/cyc | Core softmax scaling |
| `v_exp_f32` | **16 cyc** | 1/4 cyc | Requires 1 NOP before consumer (trans→non-trans) |
| `v_log_f32` | 16 cyc | 1/4 cyc | — |
| `v_rcp_f32` | 16 cyc | 1/4 cyc | Softmax denominator |
| `v_rsq_f32` | 16 cyc | 1/4 cyc | — |
| `v_max_f32` | 4 cyc | 1/cyc | Softmax row-max |
| `v_pk_fma_f16` | 2 cyc | 2× F32 | Packed (use for BF16/FP16 paths) |
| `v_cvt_f32_fp8` (SDWA) | 4 cyc | 1/cyc | Unpack fp8 byte to f32 |
| `v_cvt_pk_fp8_f32` | 4 cyc | 1/cyc | Pack 2 f32 → 2 fp8 in one VGPR |

Cross-lane:
- `ds_bpermute_b32` (backward permute via LDS index): 2-4 cyc
- `v_readlane_b32 / v_writelane_b32`: requires 4 NOPs after VALU writes lane-select SGPR
- `ds_permute_b32` / `ds_bpermute_b32`: no LDS RW; uses LDS routing hardware

## 7. Mode register (rounding/denorm)

`S_SETREG_B32 MODE, <val>`:
- **FP_ROUND[3:0]**: [1:0]=single, [3:2]=double/half (0=RNE, 1=+∞, 2=-∞, 3=trunc)
- **FP_DENORM[7:4]**: [5:4]=single, [7:6]=double/half (0=flush in+out, 1=allow in/flush out, 2=flush in/allow out, 3=allow in+out)
- **IEEE[9]**: NaN passthrough (0=DX10_CLAMP forces NaN→0)
- **DX10_CLAMP[8]**: NaN→0 clamp
- **FP16_OVFL[23]**: FP16 overflow clamp to ±MAX

**MFMA FP8 handling (important!)**:
- MFMA FP8/BF8 ops **ignore MODE.fp_denorm** (always allow denorms)
- Force RNE rounding (ignore MODE.fp_round)
- Respect FP16_OVFL for FP16 overflow only
- **SH_MEM_CONFIG bit[8] must be 1** for correct BF8/FP8 arithmetic on gfx950

## 8. Critical operational patterns

### FA3-style ping-pong (HBM→LDS overlap with MFMA)
```
// Iter 0: MFMA on bank_B, load next to bank_A
buffer_load_lds_dwordx4 ..., m0=lds_offset_A   // HBM→LDS direct
s_waitcnt vmcnt(8)                              // Allow 8 outstanding
v_mfma_f32_16x16x32_f8_f8 ...                   // Compute on bank_B
ds_read_b64_tr_b8 v[X:X+1], bank_B_addr, 0      // Read transposed
// Iter 1: roles swap
```

### MFMA accumulator chain (0-NOP back-to-back)
```
v_mfma_f32_16x16x128_f8f6f4 v[a:a+3], v[Q0:Q7], v[K0:K7], v[a:a+3]   // chain start
v_mfma_f32_16x16x128_f8f6f4 v[a:a+3], v[Q8:Q15], v[K8:K15], v[a:a+3] // 0 NOPs needed
```

### Softmax row-max + exp (pipeline-friendly)
```
v_max_f32 v_rowmax, v_score_0, v_score_1
v_max_f32 v_rowmax, v_rowmax, v_score_2
ds_bpermute_b32 v_cross_max, v_lane_xor, v_rowmax   // cross-lane reduce
v_sub_f32 v_shifted, v_score_0, v_rowmax
v_exp_f32 v_exp_val, v_shifted                      // 16-cycle trans
s_nop 0                                             // 1 NOP required trans→VALU
v_add_f32 v_sum, v_sum, v_exp_val
v_rcp_f32 v_inv_sum, v_sum                          // 16-cycle
s_nop 0
v_mul_f32 v_softmax, v_exp_val, v_inv_sum
```

### XOR swizzle for LDS K tile (zero overhead)
```cpp
uint32_t swizzle_addr(uint32_t row, uint32_t col, uint32_t col_stride) {
    uint32_t pair = (row >> 1) & 7;
    uint32_t perm = pair ^ (((pair >> 1) ^ (pair >> 2)) & 1);
    return (row * col_stride + col) ^ (perm << 4);
}
```

## 9. For DSR1 sq=8 MLA kernel

- Use **`v_mfma_f32_16x16x128_f8f6f4`** (not scaled variant — DSR1 uses per-tensor scale, applied outside MFMA)
- Each MFMA = 32 cycles (F8 input) + 3 NOP if SrcA/B changes, 0 NOP if chaining SrcC
- Per (16-row × 128-K) MFMA: 8 VGPRs A + 8 VGPRs B + 4 VGPRs D per lane
- Prefer **`ds_read_b64_tr_b8`** for K/V LDS → register transpose (saves 4+ VALU per load)
- Use **`global_load_lds_b128`** for HBM→LDS KV prefetch (zero VGPR staging)
- Keep accumulators in AGPR via ACC_CD=1 (chain without cross-pool moves)
- 8-wave ping-pong: waves 0-3 compute, waves 4-7 prefetch (mirror FA3 pattern)
- Budget check: 8 VGPR Q + 8 VGPR K (wide) + 4 VGPR accumulator per 16×16 output tile. For sq=8 × nhead=32 split across waves = manageable under 256/wave.

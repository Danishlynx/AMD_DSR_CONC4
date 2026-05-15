# L3 — Triton FP4 KV-Cache MLA Decode Kernel

**Status**: ✅ Triton kernel, **3.04× faster standalone than aiter `mla_decode_fwd`** at production shape (23 µs vs 70 µs at bs=4, head=16, seq=8192). Numerics within MXFP4 envelope (max_abs_err < 0.05). Integration into ATOM regressed +1.10 ms due to dispatch gate bug.

## Microbench results (May 10 standalone)

`aiter_compare.py` ran L3 Triton kernel vs aiter `mla_decode_fwd` at production shape (bs=4, head=16, seq=8192, prefix BF16 KV):

| Kernel | `num_kv_splits` | Time (µs) |
|---|---:|---:|
| **L3 Triton FP4 KV** | 8 | 141.10 |
| **L3 Triton FP4 KV** | 16 | 72.76 |
| **L3 Triton FP4 KV** | 32 | 38.92 |
| **L3 Triton FP4 KV** | **64** | **23.04** ⭐ |
| Aiter `mla_decode_fwd` (BF16 KV) | auto | 72.02 |
| Aiter `mla_decode_fwd` (BF16 KV) | 8 | 69.76 |
| Aiter `mla_decode_fwd` (BF16 KV) | 32 | 69.22 |
| Aiter `mla_decode_fwd` (BF16 KV) | 64 | 70.17 |
| Triton load-only floor (FP4 dequant + LDS) | 64 | 15.50 |

### Key findings
1. **Aiter ignores `num_kv_splits`** in its compute time — kernel internally fixed-tiled. Likely ASM path doesn't honor the Python knob.
2. **Triton scales linearly with KS until KS=64** (saturates at ~256 program count = CDNA4 CU sweet spot).
3. **L3 at KS=64 (23 µs) beats aiter at any KS (70 µs) by 3.04×.**
4. Distance to load-only floor (15.5 µs) = compute overhead ≈ 7.5 µs (~33% over BW-bound floor).

## Numerics validation

`numerics_test.py` (Triton kernel vs NumPy fp4-dequant + matmul reference):

| `NUM_KV_SPLITS` | max_abs_err | mean_abs_err | Status |
|---|---|---|---|
| 8 | 0.001180 | 0.000224 | PASS |
| 32 | 0.001115 | 0.000222 | PASS |
| 64 | 0.001080 | 0.000213 | PASS |

All within MXFP4 envelope (< 0.05).

## Why integration regressed

Production wire-up at `l3_decode_wrapper.py:34`:

```python
def should_dispatch_l3(forward_context):
    if forward_context.max_seqlen_q != 1:
        return False
    ...
```

The verifier in MTP=3 spec-decode folds 4 query tokens together (`MQ=4`), so `max_seqlen_q != 1` and **L3 never fires**. The dispatch check ran on every decode call but always returned False → added pure overhead (~25 µs/call × 60 layers = 1.5 ms) without any kernel benefit. Net: **+1.10 ms TPOT regress** under production workload.

## Path to unblock

Author an `MQ=4` specialized kernel variant:
- Reuse existing kernel structure
- Specialize for `head=128` fold case (4 logical queries × 32 effective head dim)
- Update gate at `l3_decode_wrapper.py:34` to accept `max_seqlen_q in (1, 2, 4)`

Estimated 3-5 days kernel + 1 week tune. Expected production TPOT delta: −0.3 to −0.5 ms (smaller than standalone because the BW savings have less headroom at MQ=4 — the 4 queries share the KV read).

## Files

### Core kernels
| File | Purpose |
|---|---|
| `mla_decode_fp4_kv.py` | L3 stage-1: FP4 KV → BF16 logits per split. FP8-cast `tl.dot` for matmul; per-32-group E8M0 scale. |
| `mla_decode_fp4_kv_stage2.py` | L3 stage-2 combine: merge per-split logits → final attention output. |
| `mla_fp4_kv_write.py` | KV-cache write kernel for FP4 KV format (the "other half" — encoder of the FP4 KV cache). |

### Reference + integration
| File | Purpose |
|---|---|
| `mla_decode_rope_REFERENCE.py` | Reference implementation with RoPE applied (numerics oracle). |
| `l3_decode_wrapper.py` | The dispatch wrapper into ATOM's `aiter_mla` decode path. **Line 34 is where the MQ=1-only gate bug lives.** |

### Bench harness
| File | Purpose |
|---|---|
| `aiter_compare.py` | The microbench that produced the 3.04× standalone speedup table above. |
| `parse_profile.py` | Chrome-trace bucketer used during integration diagnostic. |

## Reproduce standalone bench

```bash
# After installing the L3 kernel into PYTHONPATH:
python3 aiter_compare.py --kv_splits 64 --shape "bs=4,head=16,seq=8192"
# Expected: L3 Triton 23 µs vs aiter mla_decode_fwd 70 µs (3.04× faster)
```

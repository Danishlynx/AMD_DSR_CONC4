#!/usr/bin/env python3
# Phase 11 v3.1 - per-position top-N inside thinking
# Position 0 (closest future): top-N=10 (= v3)
# Position 1 (mid future):     top-N=10 (= v3)
# Position 2 (furthest future): top-N=12 (relaxed +2)
# Outside thinking: top-N=8 (= v3, baseline parity)
# Conservative: only bump position 2 because that's where accept rate is lowest.
import py_compile, shutil, sys

RS_PATH = "/app/ATOM/atom/model_ops/rejection_sampler.py"
RS_BAK = RS_PATH + ".pre_phase11_v31"
shutil.copyfile(RS_PATH, RS_BAK)
src = open(RS_PATH).read()

# 1. Update globals - add per-position top-N values + bump MAX_TOP_N to fit pos2's higher value
old_globals = """PHASE_RELAXED_TOP_N = 10  # TRT-LLM relaxed_topk (inside thinking)
PHASE_STRICT_TOP_N = 8     # baseline-equivalent top-N (outside thinking) - matches RELAXED_TOP_N=8
PHASE_RELAXED_DELTA = 0.6  # TRT-LLM relaxed_delta"""

new_globals = """PHASE_RELAXED_TOP_N = 10  # TRT-LLM relaxed_topk (inside thinking) - default for pos 0/1
PHASE_STRICT_TOP_N = 8     # baseline-equivalent top-N (outside thinking) - matches RELAXED_TOP_N=8
PHASE_RELAXED_DELTA = 0.6  # TRT-LLM relaxed_delta
# v3.1: per-position top-N inside thinking. Position 2 (furthest future MTP draft) has
# lowest accept rate; relax it more. Others stay at PHASE_RELAXED_TOP_N=10.
PHASE_THINK_TOP_N_POS0 = 10
PHASE_THINK_TOP_N_POS1 = 10
PHASE_THINK_TOP_N_POS2 = 12
PHASE_MAX_TOP_N = 12  # max across all positions; Python topk uses this; kernel iterates this many"""

if old_globals not in src:
    sys.exit("ERR: globals anchor not found")
src = src.replace(old_globals, new_globals, 1)

# 2. Update kernel signature - replace TOP_N_RELAXED_C with per-pos triple
old_sig = """    MAX_TOP_N: tl.constexpr,
    TOP_N_STRICT_C: tl.constexpr,
    TOP_N_RELAXED_C: tl.constexpr,
    THINK_TOKEN_ID_C: tl.constexpr,
    ENDTHINK_TOKEN_ID_C: tl.constexpr,
):"""

new_sig = """    MAX_TOP_N: tl.constexpr,
    TOP_N_STRICT_C: tl.constexpr,
    TOP_N_THINK_P0_C: tl.constexpr,
    TOP_N_THINK_P1_C: tl.constexpr,
    TOP_N_THINK_P2_C: tl.constexpr,
    THINK_TOKEN_ID_C: tl.constexpr,
    ENDTHINK_TOKEN_ID_C: tl.constexpr,
):"""

if old_sig not in src:
    sys.exit("ERR: kernel signature anchor not found")
src = src.replace(old_sig, new_sig, 1)

# 3. Update kernel branch - effective_top_n now depends on pos when thinking
old_branch = """            found = False
            # Phase-aware effective top-N: TOP_N_STRICT outside thinking, TOP_N_RELAXED inside.
            # Always iterate MAX_TOP_N; mask with k < effective_top_n.
            effective_top_n = TOP_N_RELAXED_C if is_thinking else TOP_N_STRICT_C
            for k in range(MAX_TOP_N):
                if k < effective_top_n:
                    candidate_id = tl.load(topn_ids_ptr + base_offset + k)
                    if candidate_id == draft_token_id:
                        found = True"""

new_branch = """            found = False
            # Phase-aware + per-position effective top-N.
            # Outside thinking: TOP_N_STRICT (=8, baseline parity).
            # Inside thinking: per-position (pos0/pos1=10, pos2=12) since pos2 (furthest
            # MTP future) has lowest accept rate.
            if is_thinking:
                if pos == 0:
                    effective_top_n = TOP_N_THINK_P0_C
                elif pos == 1:
                    effective_top_n = TOP_N_THINK_P1_C
                else:
                    effective_top_n = TOP_N_THINK_P2_C
            else:
                effective_top_n = TOP_N_STRICT_C
            for k in range(MAX_TOP_N):
                if k < effective_top_n:
                    candidate_id = tl.load(topn_ids_ptr + base_offset + k)
                    if candidate_id == draft_token_id:
                        found = True"""

if old_branch not in src:
    sys.exit("ERR: kernel branch anchor not found")
src = src.replace(old_branch, new_branch, 1)

# 4. Update Python dispatch call - use PHASE_MAX_TOP_N for topk + pass per-pos constexprs
old_call = """    if ATOM_ENABLE_PER_PHASE_RELAXED_MTP and _spec_phase_tensor is not None:
        probs = target_probs.softmax(dim=-1, dtype=torch.float32)
        topn_probs, topn_ids = torch.topk(probs, PHASE_RELAXED_TOP_N, dim=-1)
        top1_probs = topn_probs[:, 0:1]
        valid_mask = topn_probs >= (top1_probs - PHASE_RELAXED_DELTA)
        topn_ids[~valid_mask] = -1
        topn_ids = topn_ids.to(torch.int32).contiguous()
        _phase_slice = _spec_phase_tensor[:batch_size].contiguous()
        rejection_phased_sample_kernel[(batch_size,)](
            output_token_ids,
            num_bonus_tokens,
            cu_num_draft_tokens,
            draft_token_ids,
            topn_ids,
            bonus_token_ids,
            _phase_slice,
            num_spec_steps,
            PHASE_RELAXED_TOP_N,
            PHASE_STRICT_TOP_N,
            PHASE_RELAXED_TOP_N,
            THINK_TOKEN_ID,
            ENDTHINK_TOKEN_ID,
            num_warps=1,
        )"""

new_call = """    if ATOM_ENABLE_PER_PHASE_RELAXED_MTP and _spec_phase_tensor is not None:
        probs = target_probs.softmax(dim=-1, dtype=torch.float32)
        topn_probs, topn_ids = torch.topk(probs, PHASE_MAX_TOP_N, dim=-1)
        top1_probs = topn_probs[:, 0:1]
        valid_mask = topn_probs >= (top1_probs - PHASE_RELAXED_DELTA)
        topn_ids[~valid_mask] = -1
        topn_ids = topn_ids.to(torch.int32).contiguous()
        _phase_slice = _spec_phase_tensor[:batch_size].contiguous()
        rejection_phased_sample_kernel[(batch_size,)](
            output_token_ids,
            num_bonus_tokens,
            cu_num_draft_tokens,
            draft_token_ids,
            topn_ids,
            bonus_token_ids,
            _phase_slice,
            num_spec_steps,
            PHASE_MAX_TOP_N,
            PHASE_STRICT_TOP_N,
            PHASE_THINK_TOP_N_POS0,
            PHASE_THINK_TOP_N_POS1,
            PHASE_THINK_TOP_N_POS2,
            THINK_TOKEN_ID,
            ENDTHINK_TOKEN_ID,
            num_warps=1,
        )"""

if old_call not in src:
    sys.exit("ERR: dispatch call anchor not found")
src = src.replace(old_call, new_call, 1)

open(RS_PATH, "w").write(src)
py_compile.compile(RS_PATH, doraise=True)
print("OK Phase 11 v3.1 per-position top-N patch applied (pos 0=10, pos 1=10, pos 2=12)")
print("Backup: " + RS_BAK)

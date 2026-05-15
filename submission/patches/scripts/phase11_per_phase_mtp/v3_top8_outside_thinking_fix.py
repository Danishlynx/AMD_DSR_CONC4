#!/usr/bin/env python3
# v3 fix: outside-thinking uses top-N=8 (matches baseline RELAXED_TOP_N=8 behavior)
# inside-thinking uses top-N=10 (per TRT-LLM use_relaxed_acceptance_for_thinking)
# Net: per-phase only RELAXES MORE inside thinking, never stricter than baseline.
import py_compile, sys

RS_PATH = "/app/ATOM/atom/model_ops/rejection_sampler.py"
src = open(RS_PATH).read()

# Replace the kernel branch logic to take top_n_strict + top_n_relaxed as kernel args
old_kernel_branch = """            found = False
            if is_thinking:
                # Relaxed: any top-N candidate (delta filter applied in Python)
                for k in range(MAX_TOP_N):
                    candidate_id = tl.load(topn_ids_ptr + base_offset + k)
                    if candidate_id == draft_token_id:
                        found = True
            else:
                # Strict: only top-1
                if draft_token_id == top1_id:
                    found = True"""

new_kernel_branch = """            found = False
            # Phase-aware effective top-N: TOP_N_STRICT outside thinking, TOP_N_RELAXED inside.
            # Always iterate MAX_TOP_N; mask with k < effective_top_n.
            effective_top_n = TOP_N_RELAXED_C if is_thinking else TOP_N_STRICT_C
            for k in range(MAX_TOP_N):
                if k < effective_top_n:
                    candidate_id = tl.load(topn_ids_ptr + base_offset + k)
                    if candidate_id == draft_token_id:
                        found = True"""

if old_kernel_branch not in src:
    sys.exit("ERR: kernel branch anchor not found")
src = src.replace(old_kernel_branch, new_kernel_branch, 1)

# Update kernel signature: add TOP_N_STRICT_C and TOP_N_RELAXED_C constexprs
old_sig = """def rejection_phased_sample_kernel(
    output_token_ids_ptr,
    num_bonus_tokens_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    topn_ids_ptr,           # [num_tokens, MAX_TOP_N]
    bonus_token_ids_ptr,
    phase_ptr,              # [batch_size] int8: 0=NOT_THINKING 1=THINKING 2=DONE_THINKING
    num_spec_steps,
    MAX_TOP_N: tl.constexpr,
    THINK_TOKEN_ID_C: tl.constexpr,
    ENDTHINK_TOKEN_ID_C: tl.constexpr,
):"""

new_sig = """def rejection_phased_sample_kernel(
    output_token_ids_ptr,
    num_bonus_tokens_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    topn_ids_ptr,           # [num_tokens, MAX_TOP_N]
    bonus_token_ids_ptr,
    phase_ptr,              # [batch_size] int8: 0=NOT_THINKING 1=THINKING 2=DONE_THINKING
    num_spec_steps,
    MAX_TOP_N: tl.constexpr,
    TOP_N_STRICT_C: tl.constexpr,
    TOP_N_RELAXED_C: tl.constexpr,
    THINK_TOKEN_ID_C: tl.constexpr,
    ENDTHINK_TOKEN_ID_C: tl.constexpr,
):"""

if old_sig not in src:
    sys.exit("ERR: kernel signature anchor not found")
src = src.replace(old_sig, new_sig, 1)

# Update the call site to pass new args (positional after MAX_TOP_N)
old_call = """        rejection_phased_sample_kernel[(batch_size,)](
            output_token_ids,
            num_bonus_tokens,
            cu_num_draft_tokens,
            draft_token_ids,
            topn_ids,
            bonus_token_ids,
            _phase_slice,
            num_spec_steps,
            PHASE_RELAXED_TOP_N,
            THINK_TOKEN_ID,
            ENDTHINK_TOKEN_ID,
            num_warps=1,
        )"""

new_call = """        rejection_phased_sample_kernel[(batch_size,)](
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

if old_call not in src:
    sys.exit("ERR: kernel call anchor not found")
src = src.replace(old_call, new_call, 1)

# Add PHASE_STRICT_TOP_N global
old_globals = """PHASE_RELAXED_TOP_N = 10  # TRT-LLM relaxed_topk
PHASE_RELAXED_DELTA = 0.6  # TRT-LLM relaxed_delta"""

new_globals = """PHASE_RELAXED_TOP_N = 10  # TRT-LLM relaxed_topk (inside thinking)
PHASE_STRICT_TOP_N = 8     # baseline-equivalent top-N (outside thinking) - matches RELAXED_TOP_N=8
PHASE_RELAXED_DELTA = 0.6  # TRT-LLM relaxed_delta"""

if old_globals not in src:
    sys.exit("ERR: globals anchor not found")
src = src.replace(old_globals, new_globals, 1)

open(RS_PATH, "w").write(src)
py_compile.compile(RS_PATH, doraise=True)
print("OK v3 fix applied: top_n_strict=8, top_n_relaxed=10")

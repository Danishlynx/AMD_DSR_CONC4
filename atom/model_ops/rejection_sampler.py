from typing import Optional

import torch
import triton
import triton.language as tl
from atom.utils import envs
from atom.utils.forward_context import SpecDecodeMetadata
from torch import nn

ATOM_ENABLE_RELAXED_MTP = True  # HARDCODED Danish 2026-04-15 B1a (5,0.3)
RELAXED_TOP_N = 8  # DEC-073 tighter top-K
RELAXED_DELTA = 0.5  # DEC-073 wider delta

# >>> phase11_per_phase_mtp_globals <<<
# Phase 11 v3 (TRT-LLM thinking port): per-sequence phase-tracking relaxed MTP.
# Inside <think>...</think> reasoning blocks: top-N=PHASE_RELAXED_TOP_N=10 (TRT-LLM
# published `relaxed_topk`). Outside thinking: top-N=PHASE_STRICT_TOP_N=8 (= the
# baseline RELAXED_TOP_N above -- never stricter than baseline anywhere).
# The Triton kernel reads/writes the per-sequence phase from a module-level
# tensor registered by model_runner via set_spec_phase_tensor() -- avoids any
# Python setattr in the forward path so the kernel stays cudagraph-safe.
ATOM_ENABLE_PER_PHASE_RELAXED_MTP = getattr(envs, "ATOM_ENABLE_PER_PHASE_RELAXED_MTP", False)
PHASE_RELAXED_TOP_N = 10   # TRT-LLM relaxed_topk (inside thinking)
PHASE_STRICT_TOP_N = 8     # = RELAXED_TOP_N above (outside thinking = baseline)
PHASE_RELAXED_DELTA = 0.6  # TRT-LLM relaxed_delta
THINK_TOKEN_ID = 128798    # DeepSeek-R1 <think> token id
ENDTHINK_TOKEN_ID = 128799 # DeepSeek-R1 </think> token id

_spec_phase_tensor = None  # set by model_runner at init via set_spec_phase_tensor()

def set_spec_phase_tensor(t):
    """Register the per-sequence phase tensor allocated by model_runner.

    Called once at ModelRunner.__init__ time. The tensor is int8[max_num_seqs]
    on the GPU. Phase encoding: 0=NOT_THINKING, 1=THINKING (inside <think>...),
    2=DONE_THINKING (already saw </think>, back in answer phase).
    """
    global _spec_phase_tensor
    _spec_phase_tensor = t
# <<< phase11_per_phase_mtp_globals <<<


class RejectionSampler(nn.Module):
    def forward(
        self,
        metadata: SpecDecodeMetadata,
        # [num_tokens, vocab_size]
        target_logits: torch.Tensor,
        # [batch_size, 1]
        bonus_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Ensure target_logits is contiguous. For greedy sampling, we can use
        # logits directly (argmax is the same for logits and probs), but we
        # need to ensure it's contiguous to satisfy the assertion in rejection_sample.
        target_logits = target_logits.contiguous()

        # Validate shapes match expectations
        expected_num_tokens = len(metadata.draft_token_ids)
        if target_logits.shape[0] != expected_num_tokens:
            raise ValueError(
                f"target_logits shape mismatch: expected first dimension to be "
                f"{expected_num_tokens} (len(draft_token_ids)), but got {target_logits.shape[0]}"
            )

        output_token_ids = rejection_sample(
            metadata.draft_token_ids,
            # metadata.num_draft_tokens_np,
            metadata.num_spec_steps,
            metadata.cu_num_draft_tokens,
            None,
            target_logits,
            bonus_token_ids,
        )
        return output_token_ids


def rejection_sample(
    # [num_tokens]
    draft_token_ids: torch.Tensor,
    # # [batch_size]
    # num_draft_tokens: list[int],
    num_spec_steps: int,
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor,
    # [num_tokens, vocab_size]
    draft_probs: Optional[torch.Tensor],
    # [num_tokens, vocab_size]
    target_probs: torch.Tensor,
    # [batch_size, 1]
    bonus_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert draft_token_ids.ndim == 1
    assert draft_probs is None or draft_probs.ndim == 2
    assert cu_num_draft_tokens.ndim == 1
    assert target_probs.ndim == 2

    batch_size = len(cu_num_draft_tokens)
    num_tokens = draft_token_ids.shape[0]
    vocab_size = target_probs.shape[-1]
    device = target_probs.device
    assert draft_token_ids.is_contiguous()
    assert draft_probs is None or draft_probs.is_contiguous()
    assert target_probs.is_contiguous()
    assert bonus_token_ids.is_contiguous()
    assert target_probs.shape == (num_tokens, vocab_size)

    # Create output buffer.
    output_token_ids = torch.empty(
        (batch_size, num_spec_steps + 1),
        dtype=torch.int32,  # Consistent with SamplerOutput.sampled_token_ids.
        device=device,
    )
    num_bonus_tokens = torch.empty(batch_size, dtype=torch.int32, device=device)

    # >>> phase11_per_phase_mtp_dispatch <<<
    # Phase 11 v3 (TRT-LLM thinking port). Env-gated: when ATOM_ENABLE_PER_PHASE_RELAXED_MTP=1
    # AND model_runner has registered a per-sequence phase tensor, dispatch the
    # phased kernel which branches per-request on the current reasoning phase:
    #   inside <think>...</think>: accept if draft is in top-PHASE_RELAXED_TOP_N (=10) with delta filter
    #   outside thinking:          accept if draft is in top-PHASE_STRICT_TOP_N  (=8, baseline)
    # The kernel also updates the phase tensor by scanning committed tokens for
    # <think>/</think> token IDs.
    if ATOM_ENABLE_PER_PHASE_RELAXED_MTP and _spec_phase_tensor is not None:
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
            THINK_TOKEN_ID,
            ENDTHINK_TOKEN_ID,
            num_warps=1,
        )
        return output_token_ids, num_bonus_tokens
    # <<< phase11_per_phase_mtp_dispatch <<<

    if RELAXED_TOP_N <= 1:
        # Strict greedy path: draft must exactly match target argmax
        target_argmax = target_probs.argmax(dim=-1)
        rejection_greedy_sample_kernel[(batch_size,)](
            output_token_ids,
            num_bonus_tokens,
            cu_num_draft_tokens,
            draft_token_ids,
            target_argmax,
            bonus_token_ids,
            num_spec_steps,
            num_warps=1,
        )
    else:
        # Relaxed acceptance path: accept if draft is among top-N
        # candidates with prob >= (top1_prob - delta)
        probs = target_probs.softmax(dim=-1, dtype=torch.float32)
        topn_probs, topn_ids = torch.topk(probs, RELAXED_TOP_N, dim=-1)

        top1_probs = topn_probs[:, 0:1]
        valid_mask = topn_probs >= (top1_probs - RELAXED_DELTA)
        topn_ids[~valid_mask] = -1
        topn_ids = topn_ids.to(torch.int32).contiguous()

        rejection_relaxed_sample_kernel[(batch_size,)](
            output_token_ids,
            num_bonus_tokens,
            cu_num_draft_tokens,
            draft_token_ids,
            topn_ids,
            bonus_token_ids,
            num_spec_steps,
            RELAXED_TOP_N,
            num_warps=1,
        )

    return output_token_ids, num_bonus_tokens


@triton.jit(do_not_specialize=["num_spec_steps"])
# TODO use the same sampler as main model
def rejection_greedy_sample_kernel(
    output_token_ids_ptr,  # [batch_size, num_spec_steps + 1]
    num_bonus_tokens_ptr,
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    num_spec_steps,
):
    req_idx = tl.program_id(0)

    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    num_bonus_token = -1
    INVALID_TOKEN: tl.constexpr = -1
    for pos in range(num_draft_tokens):
        if rejected:
            target_argmax_id = INVALID_TOKEN
        else:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos)
            target_argmax_id = tl.cast(target_argmax_id, tl.int32)
            if draft_token_id != target_argmax_id:
                # rejected = False
                rejected = True
            num_bonus_token += 1
        tl.store(
            output_token_ids_ptr + req_idx * (num_spec_steps + 1) + pos,
            target_argmax_id,
        )

    if rejected:
        bonus_token_id = INVALID_TOKEN
    else:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        num_bonus_token += 1
    tl.store(
        output_token_ids_ptr + req_idx * (num_spec_steps + 1) + num_draft_tokens,
        bonus_token_id,
    )
    tl.store(num_bonus_tokens_ptr + req_idx, num_bonus_token)


@triton.jit(do_not_specialize=["num_spec_steps", "top_n"])
def rejection_relaxed_sample_kernel(
    output_token_ids_ptr,  # [batch_size, num_spec_steps + 1]
    num_bonus_tokens_ptr,
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    topn_ids_ptr,  # [num_tokens, top_n] — candidate token ids, -1 = invalid
    bonus_token_ids_ptr,  # [batch_size]
    num_spec_steps,
    top_n,
):
    req_idx = tl.program_id(0)

    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    num_bonus_token = -1
    INVALID_TOKEN: tl.constexpr = -1

    for pos in range(num_draft_tokens):
        if rejected:
            output_id = INVALID_TOKEN
        else:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)

            base_offset = (start_idx + pos) * top_n
            top1_id = tl.load(topn_ids_ptr + base_offset)

            found = False
            for k in range(top_n):
                candidate_id = tl.load(topn_ids_ptr + base_offset + k)
                if candidate_id == draft_token_id:
                    found = True

            if found:
                output_id = draft_token_id
            else:
                output_id = top1_id
                rejected = True

            num_bonus_token += 1

        tl.store(
            output_token_ids_ptr + req_idx * (num_spec_steps + 1) + pos,
            output_id,
        )

    if rejected:
        bonus_token_id = INVALID_TOKEN
    else:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        num_bonus_token += 1
    tl.store(
        output_token_ids_ptr + req_idx * (num_spec_steps + 1) + num_draft_tokens,
        bonus_token_id,
    )
    tl.store(num_bonus_tokens_ptr + req_idx, num_bonus_token)


# >>> phase11_per_phase_mtp_kernel <<<
# Phase 11 v3 (TRT-LLM thinking port). Per-sequence-phase relaxed-acceptance kernel.
#
# Each program processes one request. For each draft position the kernel checks
# the request's current reasoning phase (loaded from phase_ptr):
#   - is_thinking=True  -> accept if the draft token is in top-TOP_N_RELAXED_C (=10)
#                          with the delta filter already pre-applied in Python (sentinel -1
#                          for masked-out candidates).
#   - is_thinking=False -> accept if the draft token is in top-TOP_N_STRICT_C  (=8,
#                          matches the baseline RELAXED_TOP_N -- never stricter).
# After each committed token (including bonus), the kernel scans for the
# DeepSeek-R1 <think>/</think> token IDs and advances the phase in-place
# (tl.store -- no Python setattr in forward path, cudagraph-safe).
@triton.jit(do_not_specialize=["num_spec_steps"])
def rejection_phased_sample_kernel(
    output_token_ids_ptr,    # [batch_size, num_spec_steps + 1]
    num_bonus_tokens_ptr,    # [batch_size]
    cu_num_draft_tokens_ptr, # [batch_size]
    draft_token_ids_ptr,     # [num_tokens]
    topn_ids_ptr,            # [num_tokens, MAX_TOP_N] -- top-N=PHASE_RELAXED_TOP_N with delta filter applied (masked = -1)
    bonus_token_ids_ptr,     # [batch_size]
    phase_ptr,               # [batch_size] int8: 0=NOT_THINKING, 1=THINKING, 2=DONE_THINKING
    num_spec_steps,
    MAX_TOP_N: tl.constexpr,        # = PHASE_RELAXED_TOP_N (10)
    TOP_N_STRICT_C: tl.constexpr,   # = PHASE_STRICT_TOP_N  (8)
    THINK_TOKEN_ID_C: tl.constexpr,
    ENDTHINK_TOKEN_ID_C: tl.constexpr,
):
    req_idx = tl.program_id(0)

    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    phase_val = tl.load(phase_ptr + req_idx)
    is_thinking = phase_val == 1
    new_phase = phase_val

    rejected = False
    num_bonus_token = -1
    INVALID_TOKEN: tl.constexpr = -1

    for pos in range(num_draft_tokens):
        if rejected:
            output_id = INVALID_TOKEN
        else:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            base_offset = (start_idx + pos) * MAX_TOP_N
            top1_id = tl.load(topn_ids_ptr + base_offset)

            # Effective top-N per phase (v3 fix: NEVER stricter than baseline).
            effective_top_n = TOP_N_STRICT_C
            if is_thinking:
                effective_top_n = MAX_TOP_N

            found = False
            for k in range(MAX_TOP_N):
                if k < effective_top_n:
                    candidate_id = tl.load(topn_ids_ptr + base_offset + k)
                    if candidate_id == draft_token_id:
                        found = True

            if found:
                output_id = draft_token_id
            else:
                output_id = top1_id
                rejected = True
            num_bonus_token += 1

        # Phase update on each committed token (including INVALID = no-op).
        if output_id == THINK_TOKEN_ID_C:
            new_phase = 1
        elif output_id == ENDTHINK_TOKEN_ID_C:
            new_phase = 2

        tl.store(
            output_token_ids_ptr + req_idx * (num_spec_steps + 1) + pos,
            output_id,
        )

    if rejected:
        bonus_token_id = INVALID_TOKEN
    else:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        num_bonus_token += 1
        if bonus_token_id == THINK_TOKEN_ID_C:
            new_phase = 1
        elif bonus_token_id == ENDTHINK_TOKEN_ID_C:
            new_phase = 2

    tl.store(
        output_token_ids_ptr + req_idx * (num_spec_steps + 1) + num_draft_tokens,
        bonus_token_id,
    )
    tl.store(num_bonus_tokens_ptr + req_idx, num_bonus_token)
    tl.store(phase_ptr + req_idx, new_phase)
# <<< phase11_per_phase_mtp_kernel <<<

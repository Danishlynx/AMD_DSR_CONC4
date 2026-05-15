#!/usr/bin/env python3
# Phase 11 Per-Phase Relaxed MTP - TRT-LLM port to ATOM
# Detects <think>/</think> token IDs (128798/128799) per-sequence.
# Inside <think>: top-N=10 + delta=0.6 (relaxed). Outside: top-1 strict.
# Cudagraph-safe: phase tensor on GPU, kernel reads/writes via tl.load/tl.store, no Python setattr.
import py_compile, shutil, sys

# ============ FILE 1: envs.py ============
ENVS_PATH = "/app/ATOM/atom/utils/envs.py"
ENVS_BAK = ENVS_PATH + ".pre_per_phase_mtp"
shutil.copyfile(ENVS_PATH, ENVS_BAK)
envs_src = open(ENVS_PATH).read()

envs_anchor = '''    "ATOM_ENABLE_RELAXED_MTP": lambda: os.getenv("ATOM_ENABLE_RELAXED_MTP", "0").lower()
    == "1",'''
if envs_anchor not in envs_src:
    sys.exit("ERR: envs anchor not found (multi-line variant)")

envs_new = envs_anchor + '''
    "ATOM_ENABLE_PER_PHASE_RELAXED_MTP": lambda: os.getenv("ATOM_ENABLE_PER_PHASE_RELAXED_MTP", "0").lower()
    == "1",'''
envs_src2 = envs_src.replace(envs_anchor, envs_new, 1)
if envs_src2 == envs_src:
    sys.exit("ERR: envs replace failed")
open(ENVS_PATH, "w").write(envs_src2)
py_compile.compile(ENVS_PATH, doraise=True)
print("OK envs.py: " + str(len(envs_src)) + " -> " + str(len(envs_src2)))

# ============ FILE 2: model_runner.py - allocate phase + reset on prefill + register with sampler ============
MR_PATH = "/app/ATOM/atom/model_engine/model_runner.py"
MR_BAK = MR_PATH + ".pre_per_phase_mtp"
shutil.copyfile(MR_PATH, MR_BAK)
mr_src = open(MR_PATH).read()

# 2a: Allocate spec_phase tensor + register with rejection sampler. Anchor: self.max_bs = self.config.max_num_seqs (line ~1011)
mr_anchor1 = "        self.max_bs = self.config.max_num_seqs"
if mr_anchor1 not in mr_src:
    sys.exit("ERR: model_runner anchor1 not found")

mr_alloc_block = mr_anchor1 + """

        # >>> phase11_per_phase_mtp_alloc <<<
        # Per-Phase Relaxed MTP: per-sequence phase tensor (0=NOT_THINKING, 1=THINKING, 2=DONE_THINKING).
        # Allocated unconditionally (sized by max_num_seqs); Triton kernel reads/writes during sampling.
        # Connection to rejection_sampler module via set_spec_phase_tensor (module global, no setattr in forward).
        self.spec_phase = torch.zeros(self.max_bs, dtype=torch.int8, device=self.device)
        try:
            from atom.model_ops import rejection_sampler as _rs_mod
            if hasattr(_rs_mod, "set_spec_phase_tensor"):
                _rs_mod.set_spec_phase_tensor(self.spec_phase)
        except Exception as _e:
            pass  # rejection_sampler may not have setter; per-phase lever NULL-OP
        # <<< phase11_per_phase_mtp_alloc <<<"""

mr_src2 = mr_src.replace(mr_anchor1, mr_alloc_block, 1)
if mr_src2 == mr_src:
    sys.exit("ERR: model_runner alloc inject failed")

# 2b: Reset prefill slot phases in prepare_inputs. Anchor: is_prefill = batch.total_tokens_num_prefill > 0 (line ~1681)
mr_anchor2 = "        is_prefill = batch.total_tokens_num_prefill > 0\n        bs = batch.total_seqs_num"
if mr_anchor2 not in mr_src2:
    sys.exit("ERR: model_runner anchor2 (prepare_inputs prefill block) not found")

mr_reset_block = mr_anchor2 + """
        # >>> phase11_per_phase_mtp_reset <<<
        # Reset spec_phase for newly-prefilling sequences (slot indices 0..num_prefill_seqs-1).
        # Decode-only batches leave phase untouched; the rejection sampler kernel updates it.
        import os as _os_p11
        if _os_p11.environ.get("ATOM_ENABLE_PER_PHASE_RELAXED_MTP", "0") == "1" and is_prefill:
            _num_prefill = batch.total_seqs_num_prefill
            if _num_prefill > 0 and hasattr(self, "spec_phase"):
                self.spec_phase[:_num_prefill].zero_()
        # <<< phase11_per_phase_mtp_reset <<<"""

mr_src3 = mr_src2.replace(mr_anchor2, mr_reset_block, 1)
if mr_src3 == mr_src2:
    sys.exit("ERR: model_runner reset inject failed")

open(MR_PATH, "w").write(mr_src3)
py_compile.compile(MR_PATH, doraise=True)
print("OK model_runner.py: " + str(len(mr_src)) + " -> " + str(len(mr_src3)))

# ============ FILE 3: rejection_sampler.py - add phased kernel + dispatch ============
RS_PATH = "/app/ATOM/atom/model_ops/rejection_sampler.py"
RS_BAK = RS_PATH + ".pre_per_phase_mtp"
shutil.copyfile(RS_PATH, RS_BAK)
rs_src = open(RS_PATH).read()

# 3a: Add module-level globals + setter at top, after the existing constants
rs_anchor1 = """ATOM_ENABLE_RELAXED_MTP = envs.ATOM_ENABLE_RELAXED_MTP
if ATOM_ENABLE_RELAXED_MTP:
    RELAXED_TOP_N = 8
    RELAXED_DELTA = 0.5
else:
    RELAXED_TOP_N = 1
    RELAXED_DELTA = 0.0"""

if rs_anchor1 not in rs_src:
    sys.exit("ERR: rejection_sampler anchor1 not found")

rs_globals_block = rs_anchor1 + """

# >>> phase11_per_phase_mtp_globals <<<
# Per-Phase Relaxed MTP globals. Module-level so model_runner can register
# the phase tensor without forward-path setattr (cudagraph-safe).
ATOM_ENABLE_PER_PHASE_RELAXED_MTP = getattr(envs, "ATOM_ENABLE_PER_PHASE_RELAXED_MTP", False)
PHASE_RELAXED_TOP_N = 10  # TRT-LLM relaxed_topk
PHASE_RELAXED_DELTA = 0.6  # TRT-LLM relaxed_delta
THINK_TOKEN_ID = 128798   # DSR1 R1 <think>
ENDTHINK_TOKEN_ID = 128799  # DSR1 R1 </think>

_spec_phase_tensor = None  # set by model_runner at init via set_spec_phase_tensor

def set_spec_phase_tensor(t):
    global _spec_phase_tensor
    _spec_phase_tensor = t
# <<< phase11_per_phase_mtp_globals <<<"""

rs_src2 = rs_src.replace(rs_anchor1, rs_globals_block, 1)
if rs_src2 == rs_src:
    sys.exit("ERR: rejection_sampler globals inject failed")

# 3b: Add phased kernel after the relaxed kernel.
# Anchor: end of rejection_relaxed_sample_kernel - the final `tl.store(num_bonus_tokens_ptr + req_idx, num_bonus_token)` of that kernel.
# We'll find the end by locating the trailing definition right before the next function.
rs_anchor2 = """    tl.store(num_bonus_tokens_ptr + req_idx, num_bonus_token)
"""

# Need to find LAST occurrence of this since both kernels end this way
last_pos = rs_src2.rfind(rs_anchor2)
if last_pos == -1:
    sys.exit("ERR: rejection_sampler last tl.store anchor not found")

rs_phased_kernel = """    tl.store(num_bonus_tokens_ptr + req_idx, num_bonus_token)


@triton.jit(do_not_specialize=["num_spec_steps"])
def rejection_phased_sample_kernel(
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
):
    # Per-Phase Relaxed MTP kernel.
    # Inside <think>: accept if draft is in any of top-MAX_TOP_N (already delta-filtered by Python).
    # Outside <think>: accept only if draft == top1 (strict).
    # Phase update: scan committed tokens for THINK/ENDTHINK IDs, write back.
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

            found = False
            if is_thinking:
                # Relaxed: any top-N candidate (delta filter applied in Python)
                for k in range(MAX_TOP_N):
                    candidate_id = tl.load(topn_ids_ptr + base_offset + k)
                    if candidate_id == draft_token_id:
                        found = True
            else:
                # Strict: only top-1
                if draft_token_id == top1_id:
                    found = True

            if found:
                output_id = draft_token_id
            else:
                output_id = top1_id
                rejected = True
            num_bonus_token += 1

        # Phase update on every committed token (incl. INVALID = no-op)
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
"""

rs_src3 = rs_src2[:last_pos] + rs_phased_kernel + rs_src2[last_pos + len(rs_anchor2):]

# 3c: Modify rejection_sample() dispatch logic. Insert phased path BEFORE the existing if-else.
# Anchor: the existing dispatch starting with `if RELAXED_TOP_N <= 1:`
rs_anchor3 = """    if RELAXED_TOP_N <= 1:
        # Strict greedy path: draft must exactly match target argmax"""

if rs_anchor3 not in rs_src3:
    sys.exit("ERR: rejection_sampler dispatch anchor not found")

rs_dispatch_block = """    # >>> phase11_per_phase_mtp_dispatch <<<
    # Per-Phase Relaxed MTP: when env=1 + phase tensor wired, always compute top-N=10 + delta=0.6
    # filter, dispatch phased kernel which branches per-req on spec_phase (set by Python at prefill).
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
            THINK_TOKEN_ID,
            ENDTHINK_TOKEN_ID,
            num_warps=1,
        )
    elif RELAXED_TOP_N <= 1:
        # Strict greedy path: draft must exactly match target argmax"""

rs_src4 = rs_src3.replace(rs_anchor3, rs_dispatch_block, 1)
if rs_src4 == rs_src3:
    sys.exit("ERR: rejection_sampler dispatch inject failed")

open(RS_PATH, "w").write(rs_src4)
py_compile.compile(RS_PATH, doraise=True)
print("OK rejection_sampler.py: " + str(len(rs_src)) + " -> " + str(len(rs_src4)))

print("\nALL 3 FILES PATCHED (py_compile clean).")
print("Backups: " + ENVS_BAK + ", " + MR_BAK + ", " + RS_BAK)
print("Lever name: ATOM_ENABLE_PER_PHASE_RELAXED_MTP=1 (NULL-OP at default).")

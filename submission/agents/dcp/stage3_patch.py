"""DCP=4 Stage 3 — capture softmax_lse from mla_decode_fwd.

KEY DISCOVERY: aiter/mla.py:mla_decode_fwd already returns (logits, final_lse)
at line 631. ATOM's call sites just ignore the return value. So Stage 3 is
not multi-day kernel surgery; it's a few-line patch to ATOM that:
  1. Passes return_lse=True when dcp_world_size > 1
  2. Captures the returned lse
  3. Stashes it on attn_metadata for Stage 4 to read

Default behavior at dcp=1: completely unchanged (don't pass return_lse).

File: /app/ATOM/atom/model_ops/attention_mla.py
Two call sites:
  ~line 495: _forward_extend_or_prefill (sparse FP8 path, less common)
  ~line 587: _forward_decode (main decode path, what DSR1 hits at conc=4)

We patch the main _forward_decode (line 587 area). The other call site
stays untouched until we have evidence DCP needs to support sparse too.

Backups: *.pre_dcp_stage3
"""
import os, sys

PATH = '/app/ATOM/atom/model_engine/model_runner.py'  # not really used; main is below
ATTN = '/app/ATOM/atom/model_ops/attention_mla.py'


def patch():
    src = open(ATTN).read()
    if "DCP Stage 3 (Apr 27)" in src:
        print(f"already patched: {ATTN}")
        return

    # Backup
    bk = f"{ATTN}.pre_dcp_stage3"
    if not os.path.exists(bk):
        open(bk, 'w').write(src)
        print(f"  backup: {bk}")

    # Anchor: the _forward_decode mla_decode_fwd call (line ~587).
    # We need to (a) detect dcp_size, (b) pass return_lse=True conditionally,
    # (c) capture returned lse, (d) stash on attn_metadata.

    OLD = '''        mla_decode_fwd(
            q,
            kv_buffer.view(-1, 1, 1, q.shape[-1]),
            o,
            attn_metadata.cu_seqlens_q,
            paged_kv_indptr,
            paged_kv_indices,
            attn_metadata.kv_last_page_lens,
            attn_metadata.max_seqlen_q,
            num_kv_splits=None,
            sm_scale=self.scale,
            work_meta_data=work_meta_data,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
            q_scale=self._q_scale,
            kv_scale=self._k_scale,
        )'''

    NEW = '''        # DCP Stage 3 (Apr 27): capture softmax_lse from mla_decode_fwd when dcp_world_size > 1.
        # aiter/mla.py:mla_decode_fwd already returns (logits, final_lse) at its
        # final return statement. Default ATOM behavior ignores it.
        # When DCP is active, we pass return_lse=True (allocates a small
        # (total_s, nhead) fp32 tensor) and stash it on attn_metadata so the
        # Stage 4 MLA-forward wrapper in deepseek_v2.py can pass it to
        # cp_lse_ag_out_rs for cross-rank attention combine.
        try:
            from aiter.dist.parallel_state import get_decode_context_parallel_world_size
            _dcp_size = get_decode_context_parallel_world_size()
        except (AssertionError, AttributeError, ImportError):
            _dcp_size = 1
        _dcp_need_lse = _dcp_size > 1

        if _dcp_need_lse:
            _, _dcp_lse = mla_decode_fwd(
                q,
                kv_buffer.view(-1, 1, 1, q.shape[-1]),
                o,
                attn_metadata.cu_seqlens_q,
                paged_kv_indptr,
                paged_kv_indices,
                attn_metadata.kv_last_page_lens,
                attn_metadata.max_seqlen_q,
                num_kv_splits=None,
                sm_scale=self.scale,
                work_meta_data=work_meta_data,
                work_indptr=work_indptr,
                work_info_set=work_info_set,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                q_scale=self._q_scale,
                kv_scale=self._k_scale,
                return_lse=True,
            )
            # Stash on attn_metadata for Stage 4 wrapper.
            attn_metadata.dcp_local_lse = _dcp_lse
        else:
            mla_decode_fwd(
                q,
                kv_buffer.view(-1, 1, 1, q.shape[-1]),
                o,
                attn_metadata.cu_seqlens_q,
                paged_kv_indptr,
                paged_kv_indices,
                attn_metadata.kv_last_page_lens,
                attn_metadata.max_seqlen_q,
                num_kv_splits=None,
                sm_scale=self.scale,
                work_meta_data=work_meta_data,
                work_indptr=work_indptr,
                work_info_set=work_info_set,
                reduce_indptr=reduce_indptr,
                reduce_final_map=reduce_final_map,
                reduce_partial_map=reduce_partial_map,
                q_scale=self._q_scale,
                kv_scale=self._k_scale,
            )'''

    if OLD in src:
        src = src.replace(OLD, NEW, 1)
        open(ATTN, 'w').write(src)
        print(f"PATCHED {ATTN}")
    else:
        print(f"  WARN: anchor not found in {ATTN}")


if __name__ == "__main__":
    print("=== DCP=4 Stage 3 patch (capture LSE from mla_decode_fwd) ===")
    patch()
    print("=== Stage 3 patch complete ===")
    print("byte-compile + boot --dcp 1 (no-op) to validate.")

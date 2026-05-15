"""DCP=4 Stage 4 — wire MLA forward with cp_lse_ag_out_ar combine.

Two parts:
  1. Drop-in: /app/aiter-test/aiter/ops/triton/cp_lse.py (Triton kernel + AR variant).
  2. Patch ATOM /app/ATOM/atom/model_ops/attention_mla.py: after mla_decode_fwd
     in the dcp>1 branch, call cp_lse_ag_out_ar(o, attn_metadata.dcp_local_lse,
     get_dcp_group()) and overwrite o.

Gated: when dcp_world_size == 1, no traffic flows. Stage 4 is no-op.

Backups: *.pre_dcp_stage4
"""
import os, sys, shutil

ATTN = '/app/ATOM/atom/model_ops/attention_mla.py'
CP_LSE_DEST = '/app/aiter-test/aiter/ops/triton/cp_lse.py'


def install_kernel():
    """Copy our cp_lse.py into aiter triton ops dir."""
    src = '/tmp/cp_lse.py'
    if not os.path.exists(src):
        print(f"  ERROR: source kernel file not found at {src}")
        return False
    if os.path.exists(CP_LSE_DEST):
        # Compare; if same content, skip
        if open(src).read() == open(CP_LSE_DEST).read():
            print(f"  cp_lse.py already installed at {CP_LSE_DEST}, identical")
            return True
        # Backup existing
        bk = CP_LSE_DEST + ".pre_dcp_stage4"
        if not os.path.exists(bk):
            shutil.copy(CP_LSE_DEST, bk)
            print(f"  backup: {bk}")
    shutil.copy(src, CP_LSE_DEST)
    print(f"  installed: {CP_LSE_DEST}")
    return True


def patch_attention_mla():
    """Wire the cp_lse_ag_out_ar call after mla_decode_fwd in the dcp>1 branch."""
    src = open(ATTN).read()
    if "DCP Stage 4 (Apr 27)" in src:
        print(f"already patched: {ATTN}")
        return
    if "DCP Stage 3 (Apr 27)" not in src:
        print(f"  ERROR: Stage 3 not present; apply Stage 3 first")
        return

    # Backup
    bk = ATTN + ".pre_dcp_stage4"
    if not os.path.exists(bk):
        open(bk, 'w').write(src)
        print(f"  backup: {bk}")

    # Anchor: the Stage 3 stash line `attn_metadata.dcp_local_lse = _dcp_lse`
    OLD = '''            # Stash on attn_metadata for Stage 4 wrapper.
            attn_metadata.dcp_local_lse = _dcp_lse'''

    NEW = '''            # Stash on attn_metadata for Stage 4 wrapper.
            attn_metadata.dcp_local_lse = _dcp_lse

            # DCP Stage 4 (Apr 27): combine partial attention across DCP ranks.
            # Each rank computed attention on its KV shard (Stage 2 ensured
            # only owned tokens were written to local cache). Now LSE-
            # renormalize per-position then all-reduce across DCP ranks.
            # Output shape [total_s, nhead, head_dim] unchanged (heads not
            # sharded by DCP — only KV).
            from aiter.ops.triton.cp_lse import cp_lse_ag_out_ar
            from aiter.dist.parallel_state import get_dcp_group
            # o is [total_s, nhead, head_dim]; cp_attn_out wants [B, H, D].
            # _dcp_lse is [total_s, nhead]; cp_attn_lse wants [B, H]. Already match.
            o = cp_lse_ag_out_ar(
                o,
                _dcp_lse,
                get_dcp_group(),
                return_lse=False,
                is_lse_base_on_e=True,
            )'''

    if OLD in src:
        src = src.replace(OLD, NEW, 1)
        open(ATTN, 'w').write(src)
        print(f"PATCHED {ATTN}")
    else:
        print(f"  WARN: Stage 3 stash anchor not found; cannot wire Stage 4")


if __name__ == "__main__":
    print("=== DCP=4 Stage 4 patch (wire cp_lse_ag_out_ar combine) ===")
    install_kernel()
    patch_attention_mla()
    print("=== Stage 4 patch complete ===")
    print("byte-compile + boot --dcp 1 (no-op) then --dcp 4 (real test)")

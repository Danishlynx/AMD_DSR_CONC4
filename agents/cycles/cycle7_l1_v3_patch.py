"""Cycle 7 — L1 v3: pre-allocate kv_indptr scratch buffer at boot.

L1 v1 attempt (Apr 27 ~14:55) failed because it allocated `torch.zeros_like()`
INSIDE eagle.py:182 — which is the captured cudagraph region under
ATOM_MSCG_P6_REPLAY=1. Per-rank divergent allocator timing → NCCL deadlock.

L1 v3 fix: pre-allocate the scratch buffer at BOOT inside
`model_runner.allocate_forward_vars()`, alongside the existing MSCG P6 PROPER
buffers. Address is stable across all cudagraph captures + replays. The
in-graph operations (copy_, arithmetic, copy_back) all happen on the same
fixed-address tensor → no allocator divergence, no deadlock.

Three patches:
  1. /app/ATOM/atom/model_engine/model_runner.py::allocate_forward_vars()
     Add `self.mscg_kv_indptr_scratch = torch.zeros(max_bs+1, dtype=int32)`.
  2. /app/ATOM/atom/spec_decode/eagle.py:182
     Replace `kv_indptr[1:bs+1] -= cumsum(num_reject_tokens)` with
     scratch-buffered version: scratch.copy_(kv_indptr); scratch[1:bs+1] -=
     cumsum; kv_indptr.copy_(scratch).
  3. /app/ATOM/atom/model_ops/attentions/aiter_mla.py:357-359
     Replace `kv_indptr += var["cu_seqlens_q"].gpu[:bs+1]` with same
     scratch-buffered pattern.

Backups: *.pre_l1_v3
"""
import os, sys, re

PATHS = {
    'model_runner': '/app/ATOM/atom/model_engine/model_runner.py',
    'eagle': '/app/ATOM/atom/spec_decode/eagle.py',
    'aiter_mla': '/app/ATOM/atom/model_ops/attentions/aiter_mla.py',
}


def backup(path, tag='l1_v3'):
    bk = f"{path}.pre_{tag}"
    if not os.path.exists(bk):
        with open(path) as f:
            src = f.read()
        with open(bk, 'w') as f:
            f.write(src)
        print(f"  backup: {bk}")


def patch_model_runner():
    """Add mscg_kv_indptr_scratch buffer in allocate_forward_vars."""
    path = PATHS['model_runner']
    src = open(path).read()
    if "L1 v3 (Apr 27)" in src:
        print(f"already patched: {path}")
        return
    backup(path)

    # Anchor: end of MSCG P6 PROPER buffers section, just before next def.
    OLD = '''        if _mtp_k > 0:
            self.mscg_num_reject_A = torch.zeros(_mbs, dtype=torch.int32, device=self.device)
            # mtp_k as a device-side scalar tensor for in-graph subtraction.
            # Storing as 1-D so broadcasting against mscg_bonus_A[bs] works without view ops.
            self.mscg_mtp_k_dev = torch.tensor(_mtp_k, dtype=torch.int32, device=self.device)
        # <<< MSCG P6 PROPER buffers <<<'''

    NEW = '''        if _mtp_k > 0:
            self.mscg_num_reject_A = torch.zeros(_mbs, dtype=torch.int32, device=self.device)
            # mtp_k as a device-side scalar tensor for in-graph subtraction.
            # Storing as 1-D so broadcasting against mscg_bonus_A[bs] works without view ops.
            self.mscg_mtp_k_dev = torch.tensor(_mtp_k, dtype=torch.int32, device=self.device)
        # <<< MSCG P6 PROPER buffers <<<

        # >>> L1 v3 (Apr 27) kv_indptr scratch buffer >>>
        # Pre-allocated at boot (OUTSIDE any cudagraph capture region) so its
        # address is stable across all replays. The in-graph operations
        # (copy_, arithmetic, copy_back) target this fixed-address tensor —
        # no per-rank allocator divergence, no NCCL deadlock.
        # Used by:
        #   - /app/ATOM/atom/spec_decode/eagle.py:182 (kv_indptr -= cumsum(num_reject_tokens))
        #   - /app/ATOM/atom/model_ops/attentions/aiter_mla.py:357-359 (kv_indptr += cu_seqlens_q)
        # Sized to max_bs+1 (kv_indptr is always [bs+1] indexed).
        self.mscg_kv_indptr_scratch = torch.zeros(
            _mbs + 1, dtype=torch.int32, device=self.device
        )
        # <<< L1 v3 (Apr 27) <<<'''

    if OLD in src:
        src = src.replace(OLD, NEW, 1)
        open(path, 'w').write(src)
        print(f"PATCHED {path}")
    else:
        print(f"  WARN: anchor not found in {path}")


def patch_eagle():
    """Wire eagle.py:182 to use the boot-allocated scratch."""
    path = PATHS['eagle']
    src = open(path).read()
    if "L1 v3 (Apr 27)" in src:
        print(f"already patched: {path}")
        return
    backup(path)

    OLD = "kv_indptr[1 : bs + 1] -= torch.cumsum(num_reject_tokens, dim=0)"

    NEW_LINES = [
        "# L1 v3 (Apr 27): cudagraph-safe via boot-allocated scratch",
        "# (model_runner.allocate_forward_vars sets self.runner.mscg_kv_indptr_scratch).",
        "_l1_scratch = self.runner.mscg_kv_indptr_scratch[: bs + 1]",
        "_l1_scratch.copy_(kv_indptr[: bs + 1])",
        "_l1_scratch[1 : bs + 1] -= torch.cumsum(num_reject_tokens, dim=0)",
        "kv_indptr[: bs + 1].copy_(_l1_scratch[: bs + 1])",
    ]

    # Auto-detect indent of the OLD line
    m = re.search(r'(?m)^([ \t]*)' + re.escape(OLD), src)
    if m is None:
        print(f"  WARN: anchor not found in {path}")
        return
    indent = m.group(1)
    NEW = '\n'.join(indent + ln for ln in NEW_LINES)

    src = src.replace(indent + OLD, NEW, 1)
    open(path, 'w').write(src)
    print(f"PATCHED {path} (indent={len(indent)} chars)")


def patch_aiter_mla():
    """Wire aiter_mla.py:357-359 (prepare_mtp_decode) to use scratch."""
    path = PATHS['aiter_mla']
    src = open(path).read()
    if "L1 v3 (Apr 27)" in src:
        print(f"already patched: {path}")
        return
    backup(path)

    OLD = 'kv_indptr += var["cu_seqlens_q"].gpu[: bs + 1]'

    NEW_LINES = [
        "# L1 v3 (Apr 27): cudagraph-safe via boot-allocated scratch.",
        "_l1_scratch = self.model_runner.mscg_kv_indptr_scratch[: bs + 1]",
        "_l1_scratch.copy_(kv_indptr[: bs + 1])",
        '_l1_scratch += var["cu_seqlens_q"].gpu[: bs + 1]',
        "kv_indptr[: bs + 1].copy_(_l1_scratch[: bs + 1])",
    ]

    m = re.search(r'(?m)^([ \t]*)' + re.escape(OLD), src)
    if m is None:
        print(f"  WARN: anchor not found in {path}")
        return
    indent = m.group(1)
    NEW = '\n'.join(indent + ln for ln in NEW_LINES)

    src = src.replace(indent + OLD, NEW, 1)
    open(path, 'w').write(src)
    print(f"PATCHED {path} (indent={len(indent)} chars)")


if __name__ == "__main__":
    print("=== Cycle 7 L1 v3 patch (kv_indptr scratch at boot) ===")
    patch_model_runner()
    patch_eagle()
    patch_aiter_mla()
    print("=== L1 v3 patch complete ===")
    print("byte-compile then boot with ATOM_MSCG_P6_REPLAY=1")

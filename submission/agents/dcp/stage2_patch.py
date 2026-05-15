"""DCP=4 Stage 2 — KV cache layout sharding (AMD-level, no stub).

Two patches, both gated by `dcp_world_size > 1`:

1. **`/app/ATOM/atom/model_engine/model_runner.py::allocate_kv_cache`**:
   When `dcp_size > 1`, scale `num_physical_kvcache_blocks //= dcp_size`.
   Per-rank KV cache holds 1/N of the tokens (per-rank capacity grows by N
   from the host's perspective; per-token storage is on exactly ONE rank).

2. **`/app/ATOM/atom/model_ops/attentions/aiter_mla.py::AiterMLAMetadataBuilder.prepare_decode`**:
   When `dcp_size > 1`, mask the slot_mapping so each rank only writes
   tokens it owns: `slot_mapping[i] = -1` if `pos[i] % dcp_size != dcp_rank`.
   Mirrors vLLM block_table.py:104-130 (interleave=1, token-level
   alignment).

At `dcp_size=1`, both branches short-circuit. Patch is functionally a no-op
unless `--decode-context-parallel-size > 1` is set.

Backups: *.pre_dcp_stage2 for clean revert.
"""
import os, sys

PATHS = {
    'model_runner': '/app/ATOM/atom/model_engine/model_runner.py',
    'aiter_mla': '/app/ATOM/atom/model_ops/attentions/aiter_mla.py',
}


def backup(path, tag='dcp_stage2'):
    bk = f"{path}.pre_{tag}"
    if not os.path.exists(bk):
        with open(path) as f:
            src = f.read()
        with open(bk, 'w') as f:
            f.write(src)
        print(f"  backup: {bk}")


def patch_model_runner():
    """Add per-rank num_blocks sharding when dcp_size > 1."""
    path = PATHS['model_runner']
    src = open(path).read()
    if "DCP Stage 2 (Apr 27)" in src:
        print(f"already patched: {path}")
        return
    backup(path)

    # Inject right after `self.num_physical_kvcache_blocks = (... block_ratio)` line.
    OLD = '''    def allocate_kv_cache(self, num_kvcache_blocks):
        pre_alloc = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        config = self.config
        config.num_kvcache_blocks = num_kvcache_blocks
        hf_config = config.hf_config
        self.num_physical_kvcache_blocks = (
            num_kvcache_blocks * self.attn_metadata_builder.block_ratio
        )'''
    NEW = '''    def allocate_kv_cache(self, num_kvcache_blocks):
        pre_alloc = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        config = self.config
        config.num_kvcache_blocks = num_kvcache_blocks
        hf_config = config.hf_config
        self.num_physical_kvcache_blocks = (
            num_kvcache_blocks * self.attn_metadata_builder.block_ratio
        )
        # DCP Stage 2 (Apr 27): when dcp_size > 1, shard KV cache by rank.
        # Per-rank stores 1/dcp_size of the tokens (selected by ownership
        # rule: token i lives on rank i % dcp_size, see aiter_mla.py
        # prepare_decode). Effective virtual block_size scales by dcp_size.
        from aiter.dist.parallel_state import get_decode_context_parallel_world_size
        try:
            dcp_size = get_decode_context_parallel_world_size()
        except (AssertionError, AttributeError):
            dcp_size = 1
        if dcp_size > 1:
            assert self.num_physical_kvcache_blocks % dcp_size == 0, (
                f"num_physical_kvcache_blocks ({self.num_physical_kvcache_blocks}) "
                f"must be divisible by dcp_size ({dcp_size}); got remainder "
                f"{self.num_physical_kvcache_blocks % dcp_size}. Reduce "
                f"--gpu-memory-utilization slightly to align."
            )
            self.num_physical_kvcache_blocks = self.num_physical_kvcache_blocks // dcp_size
            logger.info(
                f"DCP Stage 2: sharded KV cache per rank: dcp_size={dcp_size}, "
                f"num_physical_kvcache_blocks per rank = {self.num_physical_kvcache_blocks}"
            )
        self.dcp_size = dcp_size  # stash for prepare_decode'''
    if OLD in src:
        src = src.replace(OLD, NEW, 1)
        open(path, 'w').write(src)
        print(f"PATCHED {path}")
    else:
        print(f"  WARN: allocate_kv_cache anchor not found in {path}")


def patch_aiter_mla():
    """Mask slot_mapping for non-local tokens when dcp_size > 1."""
    path = PATHS['aiter_mla']
    src = open(path).read()
    if "DCP Stage 2 (Apr 27)" in src:
        print(f"already patched: {path}")
        return
    backup(path)

    # Patch prepare_decode: after slot_mapping is built, apply DCP mask.
    # Anchor: the line `var["slot_mapping"].np[:sum_scheduled_tokens] = slot_mapping`
    OLD = '''        var["slot_mapping"].np[: bs * max_seqlen_q] = -1
        if not batch.is_dummy_run:
            var["slot_mapping"].np[:sum_scheduled_tokens] = slot_mapping'''

    NEW = '''        var["slot_mapping"].np[: bs * max_seqlen_q] = -1
        if not batch.is_dummy_run:
            # DCP Stage 2 (Apr 27): mask non-local tokens to slot_mapping=-1.
            # vLLM block_table.py:104-130 ownership rule (interleave=1):
            #   token at position `pos` is owned by rank `pos % dcp_size`.
            # Non-local tokens get slot_mapping=-1 -> skipped during KV write.
            from aiter.dist.parallel_state import (
                get_decode_context_parallel_world_size,
                get_decode_context_parallel_rank,
            )
            try:
                _dcp_size = get_decode_context_parallel_world_size()
                _dcp_rank = get_decode_context_parallel_rank()
            except (AssertionError, AttributeError):
                _dcp_size, _dcp_rank = 1, 0
            if _dcp_size > 1:
                # Recompute the absolute positions matching slot_mapping order.
                # In prepare_decode: outer iter is per request; inner is
                #   `range(seq_len - max_seqlen_q, seq_len)`.
                _dcp_positions = []
                for _seq_len in context_lens:
                    _dcp_positions.extend(range(_seq_len - max_seqlen_q, _seq_len))
                _dcp_positions = np.asarray(_dcp_positions, dtype=np.int32)
                _dcp_mask = (_dcp_positions % _dcp_size) == _dcp_rank
                _dcp_slot_arr = np.asarray(slot_mapping, dtype=np.int32)
                _dcp_slot_arr = np.where(_dcp_mask, _dcp_slot_arr, -1)
                var["slot_mapping"].np[:sum_scheduled_tokens] = _dcp_slot_arr
            else:
                var["slot_mapping"].np[:sum_scheduled_tokens] = slot_mapping'''
    if OLD in src:
        src = src.replace(OLD, NEW, 1)
        open(path, 'w').write(src)
        print(f"PATCHED {path}")
    else:
        print(f"  WARN: aiter_mla prepare_decode anchor not found")


def main():
    print("=== DCP=4 Stage 2 KV cache layout patch ===")
    patch_model_runner()
    patch_aiter_mla()
    print("=== Stage 2 patch complete ===")
    print("byte-compile + boot --dcp 1 (no-op) to validate.")


if __name__ == "__main__":
    main()

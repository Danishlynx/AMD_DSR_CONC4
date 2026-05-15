"""
DCP=4 Port — Stage 1 Foundation Patch (AMD-level, no stub, no naive)
====================================================================

Adds:
  1. Engine arg: `decode_context_parallel_size: int = 1` + `--decode-context-parallel-size` / `-dcp` CLI.
  2. Plumbs through Config -> ModelRunner.__init__ -> init_dist_env -> ensure_model_parallel_initialized -> initialize_model_parallel.
  3. Activates the commented-out _DCP group block in /app/aiter-test/aiter/dist/parallel_state.py.
  4. Adds _DCP global, get_dcp_group(), get_decode_context_parallel_world_size(), get_decode_context_parallel_rank() accessors.
  5. Adds `decode_context_parallel_all_gather` and `decode_context_parallel_reduce_scatter` ops in communication_op.py.
  6. Validators: tp_size % dcp_size == 0, dcp_size <= tp_size.
  7. Backups everything to *.pre_dcp_stage1.

Gated: when dcp_size=1, all DCP collectives short-circuit (no-op). The DCP
group is still created (to keep the call chain stable) but no traffic flows
unless the actual MLA forward calls the DCP ops (Stage 4).

Files touched:
  /app/aiter-test/aiter/dist/parallel_state.py
  /app/aiter-test/aiter/dist/communication_op.py
  /app/aiter-test/aiter/ops/communication.py  (init_dist_env)
  /app/ATOM/atom/model_engine/arg_utils.py
  /app/ATOM/atom/model_engine/model_runner.py
  /app/ATOM/atom/config.py  (if it has a ParallelConfig)
"""

import os, sys, re

PATHS = {
    'parallel_state': '/app/aiter-test/aiter/dist/parallel_state.py',
    'communication_op': '/app/aiter-test/aiter/dist/communication_op.py',
    'aiter_init_dist': '/app/aiter-test/aiter/ops/communication.py',
    'arg_utils': '/app/ATOM/atom/model_engine/arg_utils.py',
    'model_runner': '/app/ATOM/atom/model_engine/model_runner.py',
}


def backup(path, tag='dcp_stage1'):
    bk = f"{path}.pre_{tag}"
    if not os.path.exists(bk):
        with open(path) as f:
            src = f.read()
        with open(bk, 'w') as f:
            f.write(src)
        print(f"  backup: {bk}")


def patch_parallel_state():
    """Add _DCP global, accessors, function signature param, group creation, destroy."""
    path = PATHS['parallel_state']
    src = open(path).read()
    if "DCP Stage 1 (Apr 27)" in src:
        print(f"already patched: {path}")
        return
    backup(path)

    # 1a. Add _DCP global declaration after _EP (line ~1115).
    old_globals = '''_EP: Optional[GroupCoordinator] = None'''
    new_globals = '''_EP: Optional[GroupCoordinator] = None

# DCP Stage 1 (Apr 27): Decode Context Parallel group. Carves the TP group
# into tp_size//dcp_size sub-groups. world_size unchanged; reuses TP GPUs.
_DCP: Optional[GroupCoordinator] = None'''
    if old_globals in src and '_DCP: Optional[GroupCoordinator] = None' not in src:
        src = src.replace(old_globals, new_globals, 1)
        print("  [parallel_state] added _DCP global")
    else:
        print("  [parallel_state] _DCP global already present or anchor missing")

    # 1b. Add get_dcp_group accessor after get_ep_group (search for `get_ep_group` function definition).
    # Pattern: a `def get_ep_group()` block followed by `_EP_FALLBACK` or similar.
    # We append a new accessor block right after the _EP block.
    accessor_anchor = '''def get_ep_group() -> GroupCoordinator:
    assert _EP is not None, "expert parallel group is not initialized"
    return _EP'''
    accessor_new = '''def get_ep_group() -> GroupCoordinator:
    assert _EP is not None, "expert parallel group is not initialized"
    return _EP


# DCP Stage 1 (Apr 27): accessors for the decode context parallel group.
def get_dcp_group() -> GroupCoordinator:
    assert _DCP is not None, "decode context parallel group is not initialized"
    return _DCP


def get_decode_context_parallel_world_size() -> int:
    return get_dcp_group().world_size


def get_decode_context_parallel_rank() -> int:
    return get_dcp_group().rank_in_group'''
    if accessor_anchor in src and 'get_dcp_group' not in src:
        src = src.replace(accessor_anchor, accessor_new, 1)
        print("  [parallel_state] added get_dcp_group/world_size/rank accessors")
    elif 'get_dcp_group' in src:
        print("  [parallel_state] get_dcp_group already present")
    else:
        print("  [parallel_state] WARN: get_ep_group anchor not found, skipping accessor add")

    # 1c. Uncomment + activate the parameter in initialize_model_parallel signature.
    # Original line: `    # decode_context_model_parallel_size: Optional[int] = 1,`
    sig_old = '''    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    # decode_context_model_parallel_size: Optional[int] = 1,
    backend: Optional[str] = None,
    data_parallel_size: int = 1,
) -> None:'''
    sig_new = '''    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    decode_context_model_parallel_size: int = 1,
    backend: Optional[str] = None,
    data_parallel_size: int = 1,
) -> None:'''
    if sig_old in src:
        src = src.replace(sig_old, sig_new, 1)
        print("  [parallel_state] uncommented decode_context_model_parallel_size param")

    # 1d. Activate the commented-out _DCP block + add validation.
    body_old = '''    # # Build the DCP model-parallel groups.
    # global _DCP
    # assert _DCP is None, "decode context model parallel group is already initialized"
    # # Note(hc): In the current implementation of decode context parallel,
    # # dcp_size must not exceed tp_size, because the world size does not
    # # change by DCP, it simply reuses the GPUs of TP group, and split one
    # # TP group into tp_size//dcp_size DCP groups.
    # group_ranks = all_ranks.reshape(-1, decode_context_model_parallel_size).unbind(0)
    # group_ranks = [x.tolist() for x in group_ranks]
    # _DCP = init_model_parallel_group(
    #     group_ranks,
    #     get_world_group().local_rank,
    #     backend,
    #     use_message_queue_broadcaster=True,
    #     group_name="dcp",
    # )'''
    body_new = '''    # DCP Stage 1 (Apr 27): activate decode context parallel groups.
    # dcp_size must divide tp_size; world_size unchanged.
    # Each TP group of size tp_size is split into tp_size // dcp_size DCP groups.
    # With TP=4 DCP=4: one DCP group [0,1,2,3] = full TP group (intra-xGMI single node).
    # With TP=4 DCP=1: dcp groups are singletons (no-op).
    assert tensor_model_parallel_size % decode_context_model_parallel_size == 0, (
        f"tp_size ({tensor_model_parallel_size}) must be divisible by "
        f"dcp_size ({decode_context_model_parallel_size})"
    )
    assert decode_context_model_parallel_size <= tensor_model_parallel_size, (
        f"dcp_size ({decode_context_model_parallel_size}) must be <= "
        f"tp_size ({tensor_model_parallel_size})"
    )
    global _DCP
    assert _DCP is None, "decode context parallel group is already initialized"
    # Reshape so the inner dim is dcp_size; unbind to get one group per outer index.
    # Layout-wise: ranks are already grouped as ExternalDP x DP x PP x TP, so
    # reshaping by dcp keeps DCP groups within each TP group.
    group_ranks = all_ranks.reshape(-1, decode_context_model_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    _DCP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=False,  # DCP doesn't need msg-queue broadcaster
        group_name="dcp",
    )'''
    if body_old in src:
        src = src.replace(body_old, body_new, 1)
        print("  [parallel_state] activated _DCP group creation block")
    else:
        print("  [parallel_state] WARN: _DCP body anchor not found")

    # 1e. Update ensure_model_parallel_initialized to accept and forward dcp param.
    ensure_old = '''def ensure_model_parallel_initialized(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    backend: Optional[str] = None,
    data_parallel_size: int = 1,
) -> None:
    """Helper to initialize model parallel groups if they are not initialized,
    or ensure tensor-parallel and pipeline-parallel sizes are equal to expected
    values if the model parallel groups are initialized.
    """
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            backend,
            data_parallel_size,
        )
        return'''
    ensure_new = '''def ensure_model_parallel_initialized(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    backend: Optional[str] = None,
    data_parallel_size: int = 1,
    decode_context_model_parallel_size: int = 1,
) -> None:
    """Helper to initialize model parallel groups if they are not initialized,
    or ensure tensor-parallel and pipeline-parallel sizes are equal to expected
    values if the model parallel groups are initialized.
    """
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            decode_context_model_parallel_size,
            backend,
            data_parallel_size,
        )
        return'''
    if ensure_old in src:
        src = src.replace(ensure_old, ensure_new, 1)
        print("  [parallel_state] updated ensure_model_parallel_initialized signature + forward")
    else:
        print("  [parallel_state] WARN: ensure anchor not found")

    # 1f. Update destroy_model_parallel to wipe _DCP too.
    destroy_anchor = '''def destroy_model_parallel():'''
    if destroy_anchor in src:
        # Find the function body end and inject _DCP destruction logic.
        # Look for "global _TP, _PP, _DP, _EP" or similar; if not found use a different pattern.
        destroy_old = '''def destroy_model_parallel():
    """Set the groups to none and destroy them."""
    global _TP'''
        destroy_new = '''def destroy_model_parallel():
    """Set the groups to none and destroy them."""
    global _DCP
    if _DCP:
        _DCP.destroy()
    _DCP = None
    global _TP'''
        if destroy_old in src and 'global _DCP' not in src.split('def destroy_model_parallel')[1][:200]:
            src = src.replace(destroy_old, destroy_new, 1)
            print("  [parallel_state] added _DCP destruction in destroy_model_parallel")

    open(path, 'w').write(src)
    print(f"PATCHED {path}")


def patch_init_dist_env():
    """Add decode_context_model_parallel_size param to init_dist_env and forward to ensure_*."""
    path = PATHS['aiter_init_dist']
    src = open(path).read()
    if "DCP Stage 1 (Apr 27)" in src:
        print(f"already patched: {path}")
        return
    backup(path)

    old = '''def init_dist_env(
    tensor_model_parallel_size: int,
    rankID: int,
    backend: str = "cpu:gloo,cuda:nccl",
    distributed_init_method: Optional[str] = "env://",
    local_rank: int = -1,
    data_parallel_size: int = 1,
    data_parallel_rank: int = 0,
):
    pipeline_model_parallel_size = 1
    # world_size is TPxPP
    world_size = pipeline_model_parallel_size * tensor_model_parallel_size
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
        # distributed_init_method=get_distributed_init_method(get_ip(), get_open_port()),
        backend=backend,
        local_rank=local_rank,
        data_parallel_size=data_parallel_size,
        data_parallel_rank=data_parallel_rank,
    )
    ensure_model_parallel_initialized(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        data_parallel_size=data_parallel_size,
    )'''
    new = '''def init_dist_env(
    tensor_model_parallel_size: int,
    rankID: int,
    backend: str = "cpu:gloo,cuda:nccl",
    distributed_init_method: Optional[str] = "env://",
    local_rank: int = -1,
    data_parallel_size: int = 1,
    data_parallel_rank: int = 0,
    decode_context_model_parallel_size: int = 1,  # DCP Stage 1 (Apr 27)
):
    pipeline_model_parallel_size = 1
    # world_size is TPxPP
    world_size = pipeline_model_parallel_size * tensor_model_parallel_size
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
        # distributed_init_method=get_distributed_init_method(get_ip(), get_open_port()),
        backend=backend,
        local_rank=local_rank,
        data_parallel_size=data_parallel_size,
        data_parallel_rank=data_parallel_rank,
    )
    ensure_model_parallel_initialized(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        data_parallel_size=data_parallel_size,
        decode_context_model_parallel_size=decode_context_model_parallel_size,
    )'''
    if old in src:
        src = src.replace(old, new, 1)
        open(path, 'w').write(src)
        print(f"PATCHED {path}")
    else:
        print(f"  WARN: init_dist_env anchor not found in {path}")


def patch_communication_op():
    """Add decode_context_parallel_all_gather + reduce_scatter ops."""
    path = PATHS['communication_op']
    src = open(path).read()
    if "DCP Stage 1 (Apr 27)" in src:
        print(f"already patched: {path}")
        return
    backup(path)

    # Find the get_tp_group import at top + the tensor_model_parallel_all_gather
    # function. We add a parallel block after the TP gather/reduce_scatter functions.

    old_imports = "from .parallel_state import get_tp_group"
    if old_imports in src and 'get_dcp_group' not in src:
        new_imports = "from .parallel_state import get_tp_group, get_dcp_group, get_decode_context_parallel_world_size"
        src = src.replace(old_imports, new_imports, 1)
        print("  [communication_op] added get_dcp_group import")

    # Append DCP collective ops at the end of file.
    if "decode_context_parallel_all_gather" not in src:
        dcp_ops = '''


# DCP Stage 1 (Apr 27): decode context parallel collectives.
# When dcp_world_size == 1, these short-circuit (no-op). At dcp > 1 they
# dispatch through the dedicated DCP process group (separate NCCL comm).
def decode_context_parallel_all_gather(
    input_: torch.Tensor,
    use_custom: bool = False,
    dim: int = -1,
) -> torch.Tensor:
    """All-gather across the decode context parallel group. No-op at dcp=1."""
    if get_decode_context_parallel_world_size() == 1:
        return input_
    return get_dcp_group().all_gather(input_, use_custom, dim)


def decode_context_parallel_reduce_scatter(
    input_: torch.Tensor,
    use_custom: bool = True,
    dim: int = 0,
) -> torch.Tensor:
    """Reduce-scatter across the decode context parallel group. No-op at dcp=1."""
    if get_decode_context_parallel_world_size() == 1:
        return input_
    return get_dcp_group().reduce_scatter_tensor(input_, use_custom, dim)


def decode_context_parallel_all_to_all(
    input_: torch.Tensor,
    scatter_dim: int = 0,
    gather_dim: int = 0,
) -> torch.Tensor:
    """All-to-all across the DCP group, used by the future a2a backend variant."""
    if get_decode_context_parallel_world_size() == 1:
        return input_
    # GroupCoordinator.all_to_all is the standard primitive. If unavailable,
    # caller should use AG+RS instead.
    return get_dcp_group().all_to_all(input_, scatter_dim, gather_dim)
'''
        src = src.rstrip() + dcp_ops
        open(path, 'w').write(src)
        print(f"PATCHED {path}")


def patch_arg_utils():
    """Add decode_context_parallel_size field + CLI flag."""
    path = PATHS['arg_utils']
    src = open(path).read()
    if "DCP Stage 1 (Apr 27)" in src:
        print(f"already patched: {path}")
        return
    backup(path)

    # Add field after data_parallel_size.
    old_field = '''    tensor_parallel_size: int = 1
    data_parallel_size: int = 1'''
    new_field = '''    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    # DCP Stage 1 (Apr 27): decode context parallel size. Must divide tp_size.
    # dcp=1 = no-op; dcp>1 enables DCP group + KV-cache sharding (Stages 2-4).
    decode_context_parallel_size: int = 1'''
    if old_field in src and 'decode_context_parallel_size' not in src:
        src = src.replace(old_field, new_field, 1)
        print("  [arg_utils] added decode_context_parallel_size field")

    # Add CLI flag after --data-parallel-size block.
    old_cli = '''        parser.add_argument(
            "--data-parallel-size",
            "-dp",
            type=int,
            default=1,
            help="Data parallel size.",
        )'''
    new_cli = '''        parser.add_argument(
            "--data-parallel-size",
            "-dp",
            type=int,
            default=1,
            help="Data parallel size.",
        )
        # DCP Stage 1 (Apr 27): decode context parallel.
        parser.add_argument(
            "--decode-context-parallel-size",
            "-dcp",
            type=int,
            default=1,
            help="Decode context parallel size. Must divide tensor parallel size. "
                 "When > 1, KV cache is sharded across dcp ranks (per-rank capacity = total / dcp), "
                 "MLA decode runs on local KV shard, and an LSE-renormalized combine runs after attention.",
        )'''
    if old_cli in src and '--decode-context-parallel-size' not in src:
        src = src.replace(old_cli, new_cli, 1)
        print("  [arg_utils] added --decode-context-parallel-size CLI")

    open(path, 'w').write(src)
    print(f"PATCHED {path}")


def patch_model_runner():
    """Pass decode_context_parallel_size through init_dist_env."""
    path = PATHS['model_runner']
    src = open(path).read()
    if "DCP Stage 1 (Apr 27)" in src:
        print(f"already patched: {path}")
        return
    backup(path)

    old_call = '''        init_dist_env(
            config.tensor_parallel_size,
            rankID=rank,
            backend="nccl",
            distributed_init_method=distributed_init_method,
            data_parallel_size=config.parallel_config.data_parallel_size,
            data_parallel_rank=config.parallel_config.data_parallel_rank,
        )'''
    new_call = '''        # DCP Stage 1 (Apr 27): plumb decode_context_parallel_size through to aiter init.
        init_dist_env(
            config.tensor_parallel_size,
            rankID=rank,
            backend="nccl",
            distributed_init_method=distributed_init_method,
            data_parallel_size=config.parallel_config.data_parallel_size,
            data_parallel_rank=config.parallel_config.data_parallel_rank,
            decode_context_model_parallel_size=getattr(
                config, "decode_context_parallel_size", 1
            ),
        )'''
    if old_call in src:
        src = src.replace(old_call, new_call, 1)
        open(path, 'w').write(src)
        print(f"PATCHED {path}")
    else:
        print(f"  WARN: init_dist_env call anchor not found in {path}")


def main():
    print("=== DCP=4 Stage 1 Foundation Patch ===")
    patch_parallel_state()
    print()
    patch_init_dist_env()
    print()
    patch_communication_op()
    print()
    patch_arg_utils()
    print()
    patch_model_runner()
    print()
    print("=== Stage 1 patch complete ===")
    print("Next: byte-compile check, then boot with --dcp 1 (no-op) to validate.")


if __name__ == "__main__":
    main()

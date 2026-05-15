#!/usr/bin/env python3
"""Phase 5 L5 eagle.py patch — Spec V2 propose-loop overlap.

When ATOM_SPEC_V2_OVERLAP=1, dispatches `prepare_mtp_decode` + the dependent
`slot_mapping` update on `self.runner.spec_prefetch_stream` so that the default
stream can run `positions += 1` and the dict-merge concurrently. A
`wait_stream` at the end ensures the next forward sees the updated metadata.

NULL-OP at default (env unset). Numerics-safe because both streams write to the
same memory; only dispatch order shifts. GSM8K canary still mandatory before perf.
"""
import os
import py_compile
import sys

TARGET = "/app/ATOM/atom/spec_decode/eagle.py"
BACKUP = "/app/ATOM/atom/spec_decode/eagle.py.pre_phase5_l5"
PATCH_MARKER = "# >>> phase5_l5_spec_v2_overlap <<<"

# Anchor 1: import block — append `from atom.utils import envs` after the existing
# `from atom.utils import CpuGpuBuffer, resolve_obj_by_qualname` line.
IMPORT_ANCHOR = (
    "from atom.utils import CpuGpuBuffer, resolve_obj_by_qualname"
)
IMPORT_REPLACEMENT = (
    "from atom.utils import CpuGpuBuffer, resolve_obj_by_qualname\n"
    "from atom.utils import envs as atom_envs  # phase5_l5 spec v2 overlap"
)

# Anchor 2: the existing serial metadata-update block in EagleProposer.propose.
# Match the literal contiguous region from "# update metadata" through
# "hidden_states = sample_hidden_states" (last line before loop continues).
SERIAL_BLOCK = '''                    # update metadata
                    attn_metadata.max_seqlen_k += 1
                    workinfos = self.runner.attn_metadata_builder.prepare_mtp_decode(
                        bs,
                        (
                            attn_metadata.max_seqlen_q
                            if not do_attn_metadata_update
                            else i0_max_seqlen_q
                        ),
                        attn_metadata.max_seqlen_k,
                        only_update=do_attn_metadata_update,
                        num_reject_tokens=num_reject_tokens if i == 0 else None,
                    )
                    for k, v in workinfos.items():
                        attn_metadata.__dict__[k] = v
                    slot_mapping[:] = kv_indices[kv_indptr[1 : bs + 1] - 1]
                    input_ids = new_draft_ids
                    positions += 1
                    hidden_states = sample_hidden_states'''

# Replacement: env-gated branch. Default path is bit-identical to original;
# overlap path runs prepare_mtp_decode + slot_mapping write on side stream.
OVERLAP_BLOCK = '''                    # >>> phase5_l5_spec_v2_overlap <<<
                    if atom_envs.ATOM_SPEC_V2_OVERLAP:
                        # Dispatch metadata regen + slot_mapping write on side
                        # stream; default stream runs positions += 1 concurrently.
                        spec_stream = self.runner.spec_prefetch_stream
                        spec_stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(spec_stream):
                            attn_metadata.max_seqlen_k += 1
                            workinfos = self.runner.attn_metadata_builder.prepare_mtp_decode(
                                bs,
                                (
                                    attn_metadata.max_seqlen_q
                                    if not do_attn_metadata_update
                                    else i0_max_seqlen_q
                                ),
                                attn_metadata.max_seqlen_k,
                                only_update=do_attn_metadata_update,
                                num_reject_tokens=num_reject_tokens if i == 0 else None,
                            )
                            for k, v in workinfos.items():
                                attn_metadata.__dict__[k] = v
                            slot_mapping[:] = kv_indices[kv_indptr[1 : bs + 1] - 1]
                        input_ids = new_draft_ids
                        positions += 1
                        hidden_states = sample_hidden_states
                        # Sync side stream completion before next forward dispatch.
                        torch.cuda.current_stream().wait_stream(spec_stream)
                    else:
                        # update metadata
                        attn_metadata.max_seqlen_k += 1
                        workinfos = self.runner.attn_metadata_builder.prepare_mtp_decode(
                            bs,
                            (
                                attn_metadata.max_seqlen_q
                                if not do_attn_metadata_update
                                else i0_max_seqlen_q
                            ),
                            attn_metadata.max_seqlen_k,
                            only_update=do_attn_metadata_update,
                            num_reject_tokens=num_reject_tokens if i == 0 else None,
                        )
                        for k, v in workinfos.items():
                            attn_metadata.__dict__[k] = v
                        slot_mapping[:] = kv_indices[kv_indptr[1 : bs + 1] - 1]
                        input_ids = new_draft_ids
                        positions += 1
                        hidden_states = sample_hidden_states
                    # <<< phase5_l5_spec_v2_overlap <<<'''


def main():
    with open(TARGET, "r") as f:
        src = f.read()

    if PATCH_MARKER in src:
        print(f"[phase5_l5_eagle] already patched (marker present), skipping")
        return 0

    if IMPORT_ANCHOR not in src:
        print(
            f"[phase5_l5_eagle] FATAL: import anchor not found in {TARGET}",
            file=sys.stderr,
        )
        sys.exit(1)
    if src.count(IMPORT_ANCHOR) != 1:
        print(
            f"[phase5_l5_eagle] FATAL: import anchor matched "
            f"{src.count(IMPORT_ANCHOR)} times, expected 1",
            file=sys.stderr,
        )
        sys.exit(1)

    if SERIAL_BLOCK not in src:
        print(
            f"[phase5_l5_eagle] FATAL: serial block anchor not found",
            file=sys.stderr,
        )
        sys.exit(1)
    if src.count(SERIAL_BLOCK) != 1:
        print(
            f"[phase5_l5_eagle] FATAL: serial block anchor matched "
            f"{src.count(SERIAL_BLOCK)} times, expected 1",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.exists(BACKUP):
        with open(BACKUP, "w") as f:
            f.write(src)
        print(f"[phase5_l5_eagle] backup written: {BACKUP}")

    new_src = src.replace(IMPORT_ANCHOR, IMPORT_REPLACEMENT, 1)
    new_src = new_src.replace(SERIAL_BLOCK, OVERLAP_BLOCK, 1)

    with open(TARGET, "w") as f:
        f.write(new_src)
    print(f"[phase5_l5_eagle] wrote {TARGET}")

    try:
        py_compile.compile(TARGET, doraise=True)
        print(f"[phase5_l5_eagle] py_compile OK")
    except py_compile.PyCompileError as e:
        print(f"[phase5_l5_eagle] FATAL: py_compile failed: {e}", file=sys.stderr)
        with open(BACKUP, "r") as f:
            with open(TARGET, "w") as g:
                g.write(f.read())
        print(f"[phase5_l5_eagle] ROLLED BACK from backup", file=sys.stderr)
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())

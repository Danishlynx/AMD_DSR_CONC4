#!/usr/bin/env python3
"""DEC-075 OPTIMIZED merge — surgical drafter fast-path transplant.

Swaps ONLY layer 61 MoE (experts + gate + shared_experts + mlp.norm) to FP4 from
amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4.

Keeps BF16 from amd/DeepSeek-R1-0528-MXFP4 for layer 61:
  - self_attn.* (MLA projections)
  - input_layernorm, post_attention_layernorm, enorm, hnorm
  - embed_tokens, eh_proj, shared_head.*

This captures ~95% of drafter speedup with minimum kernel-shape risk (avoids FP4 MLA).
"""
import json
import os
import shutil
from pathlib import Path

MAIN = Path("/projects/teamA/hf_cache/hub/models--amd--DeepSeek-R1-0528-MXFP4/snapshots/913fc83b2d3962dbc2682d6b97e9ef31acb4bf5a")
MOE  = Path("/projects/teamA/hf_cache/hub/models--amd--DeepSeek-R1-0528-MXFP4-MTP-MoEFP4/snapshots/0d5e9928feff2f4f735823c7062177773acf63ce")
OUT  = Path("/projects/teamA/danish/models_merged/DSR1-drafter-FP4")


def keep_main_layer61(key: str) -> bool:
    """Return True if layer 61 key should stay BF16 from main checkpoint."""
    return (
        key.startswith("model.layers.61.self_attn.")
        or key.startswith("model.layers.61.input_layernorm")
        or key.startswith("model.layers.61.post_attention_layernorm")
        or key.startswith("model.layers.61.enorm")
        or key.startswith("model.layers.61.hnorm")
        or key.startswith("model.layers.61.embed_tokens")
        or key.startswith("model.layers.61.eh_proj")
        or key.startswith("model.layers.61.shared_head")
    )


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    main_idx = json.loads((MAIN / "model.safetensors.index.json").read_text())
    moe_idx  = json.loads((MOE  / "model.safetensors.index.json").read_text())
    main_wmap = main_idx["weight_map"]
    moe_wmap  = moe_idx["weight_map"]

    print(f"Main:   {len(main_wmap)} keys, {len(set(main_wmap.values()))} shards")
    print(f"MoEFP4: {len(moe_wmap)} keys, {len(set(moe_wmap.values()))} shards")

    merged_wmap = {}

    # Start with all main keys except the layer-61 ones we'll replace
    main_from_main = 0
    main_dropped_l61 = 0
    for k, fname in main_wmap.items():
        if "layers.61" in k:
            if keep_main_layer61(k):
                merged_wmap[k] = fname
                main_from_main += 1
            else:
                main_dropped_l61 += 1
        else:
            merged_wmap[k] = fname
            main_from_main += 1

    # Add FP4 layer-61 MoE keys from MoEFP4 (plus their weight_scales)
    added_moe = 0
    for k, fname in moe_wmap.items():
        if "layers.61" in k and not keep_main_layer61(k):
            merged_wmap[k] = "moefp4-" + fname
            added_moe += 1

    print(f"\nMerged: {len(merged_wmap)} keys")
    print(f"  From main: {main_from_main} (layers 0-60 + lm_head + misc + layer 61 MLA/LN/embed/eh_proj/shared_head)")
    print(f"  From MoEFP4 FP4 for layer 61: {added_moe}")
    print(f"  Dropped from main layer 61 (replaced): {main_dropped_l61}")

    # Symlink main shards
    main_shards = set(main_wmap.values())
    for shard in main_shards:
        os.symlink(MAIN / shard, OUT / shard)

    # Symlink MoEFP4 shards that we reference
    moe_shards = set()
    for k, fname in moe_wmap.items():
        if "layers.61" in k and not keep_main_layer61(k):
            moe_shards.add(fname)
    for shard in moe_shards:
        os.symlink(MOE / shard, OUT / ("moefp4-" + shard))

    print(f"\nSymlinks: {len(main_shards)} main + {len(moe_shards)} moefp4 shards")

    # Write merged index.json
    merged_idx = {
        "metadata": main_idx.get("metadata", {}),
        "weight_map": merged_wmap,
    }
    (OUT / "model.safetensors.index.json").write_text(json.dumps(merged_idx, indent=2))

    # Write modified config.json — precise layer 61 excludes
    config = json.loads((MAIN / "config.json").read_text())
    qc = config["quantization_config"]

    # Remove the catch-all layer 61 exclusion
    new_exclude = [e for e in qc["exclude"] if e != "re:model.layers.61.*"]
    # Add specific layer-61 excludes matching keep_main_layer61 policy
    layer61_bf16_excludes = [
        "re:model.layers.61.self_attn.*",
        "re:model.layers.61.input_layernorm.*",
        "re:model.layers.61.post_attention_layernorm.*",
        "re:model.layers.61.enorm",
        "re:model.layers.61.hnorm",
        "model.layers.61.embed_tokens",
        "model.layers.61.eh_proj",
        "re:model.layers.61.shared_head.*",
    ]
    new_exclude.extend(layer61_bf16_excludes)
    qc["exclude"] = new_exclude
    (OUT / "config.json").write_text(json.dumps(config, indent=2))

    # Symlink auxiliary files
    for f in [
        "tokenizer.json", "tokenizer_config.json",
        "configuration_deepseek.py", "modeling_deepseek.py",
        "generation_config.json", "chat_template.jinja",
        "special_tokens_map.json",
    ]:
        src = MAIN / f
        if src.exists():
            dst = OUT / f
            if not dst.exists():
                os.symlink(src, dst)

    print(f"\n=== DONE: merged checkpoint at {OUT} ===")
    files = sorted(p.name for p in OUT.iterdir())
    print(f"Files ({len(files)}): {files[:5]} ... {files[-3:]}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""DEC-075 v5 merge: fully isolate layer 61 sources.

Root cause of v2-v4 crashes: safetensors_weights_iterator globs ALL *.safetensors
in the dir. Even if weight_map doesn't reference a shard, its keys still get
yielded — so main's layer-61 BF16 tensors leaked in alongside moefp4's FP4.

Fix:
  1. Main shards 00077-80 (pure layer 61): DO NOT include in merged dir
  2. Main shards 00076, 00081, 00082 (mixed layer 61 + other): rebuild with layer 61 stripped
  3. Main shards 00001-00075: symlink as-is
  4. MoEFP4 shards 00074-76 (layer 61 only): symlink as 'moefp4-*'
  5. Index maps correctly
"""
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MAIN = Path("/projects/teamA/hf_cache/hub/models--amd--DeepSeek-R1-0528-MXFP4/snapshots/913fc83b2d3962dbc2682d6b97e9ef31acb4bf5a")
MOE  = Path("/projects/teamA/hf_cache/hub/models--amd--DeepSeek-R1-0528-MXFP4-MTP-MoEFP4/snapshots/0d5e9928feff2f4f735823c7062177773acf63ce")
OUT  = Path("/projects/teamA/danish/models_merged/DSR1-drafter-FP4")


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    main_idx = json.loads((MAIN / "model.safetensors.index.json").read_text())
    moe_idx  = json.loads((MOE  / "model.safetensors.index.json").read_text())
    main_wmap = main_idx["weight_map"]
    moe_wmap  = moe_idx["weight_map"]

    # Classify main shards by content: pure-layer-61, mixed, non-layer-61
    from collections import defaultdict
    shard_keys = defaultdict(list)
    for k, s in main_wmap.items():
        shard_keys[s].append(k)

    pure_l61 = set()
    mixed = {}
    pure_other = set()
    for s, keys in shard_keys.items():
        l61_keys = [k for k in keys if "layers.61" in k]
        non_l61 = [k for k in keys if "layers.61" not in k]
        if l61_keys and not non_l61:
            pure_l61.add(s)
        elif l61_keys and non_l61:
            mixed[s] = non_l61   # keys we want to keep
        else:
            pure_other.add(s)

    print(f"Main shards: pure_layer61={len(pure_l61)}, mixed={len(mixed)}, pure_other={len(pure_other)}")

    # Build merged weight_map
    merged_wmap = {}

    # 1. For non-layer-61 keys from main: route to appropriate shard
    for k, fname in main_wmap.items():
        if "layers.61" in k:
            continue   # drop, will come from MoEFP4
        # If fname is a pure_l61 shard (shouldn't be — layer 61 not here) skip
        if fname in pure_l61:
            continue
        # If mixed shard, we'll write a CLEANED version
        if fname in mixed:
            merged_wmap[k] = f"cleaned-{fname}"
        else:
            # pure_other shard: use as-is via symlink
            merged_wmap[k] = fname

    # 2. For layer-61 keys: route to MoEFP4 (renamed with moefp4- prefix)
    for k, fname in moe_wmap.items():
        if "layers.61" in k:
            merged_wmap[k] = f"moefp4-{fname}"

    print(f"Merged weight_map: {len(merged_wmap)} keys")

    # 3. Symlink pure_other main shards (layers 0-60, lm_head, norm)
    for s in pure_other:
        if s in set(main_wmap.values()):
            os.symlink(MAIN / s, OUT / s)
    print(f"Symlinked {len(pure_other)} pure_other main shards")

    # 4. For mixed main shards: rebuild without layer 61 keys
    for s, keep_keys in mixed.items():
        src_path = MAIN / s
        out_name = f"cleaned-{s}"
        out_path = OUT / out_name
        tensors = {}
        with safe_open(src_path, framework="pt") as f:
            for k in keep_keys:
                tensors[k] = f.get_tensor(k)
        save_file(tensors, out_path)
        print(f"   cleaned {s}: {len(keep_keys)} keys kept (was {len(shard_keys[s])})")

    # 5. Symlink MoEFP4 shards that hold layer 61 keys
    moe_shards = set()
    for k, fname in moe_wmap.items():
        if "layers.61" in k:
            moe_shards.add(fname)
    for s in moe_shards:
        os.symlink(MOE / s, OUT / f"moefp4-{s}")
    print(f"Symlinked {len(moe_shards)} MoEFP4 layer-61 shards")

    # 6. Write merged index.json
    (OUT / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": main_idx.get("metadata", {}),
        "weight_map": merged_wmap,
    }, indent=2))

    # 7. Merged config: base from MAIN, quant_config from MoEFP4 for layer 61 handling
    config = json.loads((MAIN / "config.json").read_text())
    qc_main = config["quantization_config"]
    qc_moe  = json.loads((MOE / "config.json").read_text())["quantization_config"]

    # Remove layer-61 catch-all from main's excludes
    merged_excl = [e for e in qc_main["exclude"] if e != "re:model.layers.61.*"]
    # Add MoEFP4's layer-61-specific excludes (embed_tokens, eh_proj, shared_head.head)
    for e in qc_moe["exclude"]:
        if "layers.61" in e and e not in merged_excl:
            merged_excl.append(e)
    config["quantization_config"]["exclude"] = merged_excl
    config["quantization_config"]["layer_quant_config"] = qc_moe.get("layer_quant_config", {})
    (OUT / "config.json").write_text(json.dumps(config, indent=2))

    # 8. Aux files
    for f in ["tokenizer.json", "tokenizer_config.json", "configuration_deepseek.py",
              "modeling_deepseek.py", "generation_config.json", "chat_template.jinja",
              "special_tokens_map.json"]:
        src = MAIN / f
        if src.exists() and not (OUT / f).exists():
            os.symlink(src, OUT / f)

    # Verify: no stray main layer-61 shards
    for s in pure_l61:
        assert not (OUT / s).exists(), f"LEAK: {s} still in merged dir"

    print(f"\n=== DONE: {OUT} ===")
    print(f"Total files in merged dir: {len(list(OUT.iterdir()))}")
    print(f"No main layer-61 shards present (verified)")


if __name__ == "__main__":
    main()

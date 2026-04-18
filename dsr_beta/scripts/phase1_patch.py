#!/usr/bin/env python3
"""Phase 1: replace 2-kernel Q-proj+RoPE+cache path with single fused_kv_bmm.

Guarded by ATOM_USE_TRITON_MXFP4_BMM=1 env var (via is_rocm_aiter_fp4bmm_enabled()).
Run inside the DSR_beta container:
    python3 /tmp/phase1_patch.py
"""
import sys

src = "/app/ATOM/atom/model_ops/attention_mla.py"
with open(src) as f:
    content = f.read()

old_block = '''        else:
            q_nope, q_rope = self._q_proj_and_k_up_proj(q, x_scale=q_scale)

            q_out = torch.empty(
                (
                    q_nope.shape[0],
                    self.num_heads,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                ),
                dtype=attn_metadata.dtype_q,
                device=q_nope.device,
            )
            if kv_cache.numel() > 0:
                fused_qk_rope_concat_and_cache_mla(
                    q_nope,
                    q_rope,
                    k_nope,
                    k_rope,
                    kv_cache.view(
                        kv_cache.shape[0], -1, self.kv_lora_rank + self.qk_rope_head_dim
                    ),
                    q_out,
                    attn_metadata.slot_mapping,
                    self._k_scale,
                    self._q_scale,
                    positions,
                    self.rotary_emb.cos_cache,
                    self.rotary_emb.sin_cache,
                    is_neox=self.rotary_emb.is_neox_style,
                    is_nope_first=True,
                )
                # q_out = self.fused_kv_bmm(q, q_scale, k_nope, k_rope, positions, kv_cache, attn_metadata)'''

new_block = '''        else:
            # Phase 1 patch: if ATOM_USE_TRITON_MXFP4_BMM=1, use single fused kernel
            if is_rocm_aiter_fp4bmm_enabled() and kv_cache.numel() > 0:
                q_out = self.fused_kv_bmm(q, q_scale, k_nope, k_rope, positions, kv_cache, attn_metadata)
            else:
                q_nope, q_rope = self._q_proj_and_k_up_proj(q, x_scale=q_scale)

                q_out = torch.empty(
                    (
                        q_nope.shape[0],
                        self.num_heads,
                        self.kv_lora_rank + self.qk_rope_head_dim,
                    ),
                    dtype=attn_metadata.dtype_q,
                    device=q_nope.device,
                )
                if kv_cache.numel() > 0:
                    fused_qk_rope_concat_and_cache_mla(
                        q_nope,
                        q_rope,
                        k_nope,
                        k_rope,
                        kv_cache.view(
                            kv_cache.shape[0], -1, self.kv_lora_rank + self.qk_rope_head_dim
                        ),
                        q_out,
                        attn_metadata.slot_mapping,
                        self._k_scale,
                        self._q_scale,
                        positions,
                        self.rotary_emb.cos_cache,
                        self.rotary_emb.sin_cache,
                        is_neox=self.rotary_emb.is_neox_style,
                        is_nope_first=True,
                    )'''

if old_block not in content:
    print("OLD_BLOCK_NOT_FOUND")
    sys.exit(1)

new_content = content.replace(old_block, new_block)
with open(src, "w") as f:
    f.write(new_content)
print("PATCH_APPLIED")

#!/usr/bin/env python3
# R2-B microbench harness: time aiter.fused_moe at exact M=4 dispatched shape.
# Loads snapshotted tensors, runs N=1000 iters with cuda events, reports T0.
# Computes roofline: actual TFLOPS vs MI355X peak FP4 = 1.3 PetaFLOPS.
import os, time, json
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("ATOM_ENABLE_PER_PHASE_RELAXED_MTP", "1")  # match v3 stack
os.environ.setdefault("ATOM_USE_POST_ATTN_RMSNORM_MXFP4_FUSION", "1")
os.environ["ATOM_R2_SNAPSHOT_TENSORS"] = "0"  # don't fire snapshot during bench

import torch
import aiter
from aiter import QuantType, ActivationType
from aiter.fused_moe import fused_moe

SNAP = "/tmp/r2_snapshot"

print("=== R2-B microbench: M=4 fused_moe roofline ===")
print(f"loading tensors from {SNAP}/")

x          = torch.load(f"{SNAP}/hidden_states_M4.pt").cuda()
w13        = torch.load(f"{SNAP}/w13_weight.pt").cuda()
w2         = torch.load(f"{SNAP}/w2_weight.pt").cuda()
w13_scale  = torch.load(f"{SNAP}/w13_weight_scale.pt").cuda()
w2_scale   = torch.load(f"{SNAP}/w2_weight_scale.pt").cuda()
topk_w     = torch.load(f"{SNAP}/topk_weights.pt").cuda()
topk_ids   = torch.load(f"{SNAP}/topk_ids.pt").cuda()

print(f"x.shape={tuple(x.shape)} dtype={x.dtype}")
print(f"w13.shape={tuple(w13.shape)} dtype={w13.dtype}")
print(f"w2.shape={tuple(w2.shape)} dtype={w2.dtype}")
print(f"topk_ids.shape={tuple(topk_ids.shape)}")

M = x.shape[0]
H = x.shape[1]  # hidden_size = 7168
intermediate = w13.shape[1] // 2  # 1024 / 2 = 512 (gate + up combined as 2*intermediate)
num_experts = w13.shape[0]
topk = topk_ids.shape[1]

print(f"M={M} H={H} intermediate={intermediate} num_experts={num_experts} topk={topk}")

# fused_moe call - mirror moe.py:1087 exactly
def _fused_moe_call():
    return fused_moe(
        x,
        w13, w2,
        topk_w, topk_ids,
        expert_mask=None,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        a1_scale=None,
        a2_scale=None,
        doweight_stage1=True,
        hidden_pad=0,
        intermediate_pad=0,
        bias1=None,
        bias2=None,
    )

# Warmup
print("warmup 5 iters...")
for _ in range(5):
    out = _fused_moe_call()
torch.cuda.synchronize()

# Bench: cuda events, 1000 iters
N_ITERS = 1000
print(f"timing {N_ITERS} iters via cuda.Event...")
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(N_ITERS):
    out = _fused_moe_call()
end.record()
torch.cuda.synchronize()
total_ms = start.elapsed_time(end)
mean_ms_per_iter = total_ms / N_ITERS
mean_us_per_iter = mean_ms_per_iter * 1000

print(f"\n=== T0 = {mean_us_per_iter:.2f} us per call (mean of {N_ITERS}) ===")

# Roofline analysis
# FP4 GEMM at M=4: each token -> topk (8 routed + 1 shared) experts.
# Per expert per token: gate+up GEMM = M * 2*intermediate * H FP4 FMAs (FMA = 2 ops)
#                       down GEMM    = M * H * intermediate FMAs
# Total per call:
flops_per_token_per_expert = 2 * (2 * intermediate * H + H * intermediate)  # (gate+up + down) * 2 ops/FMA
# Actually each token routes to topk experts (8 routed) + 1 shared expert
flops_per_token = topk * flops_per_token_per_expert
flops_per_call  = M * flops_per_token

# MI355X peak FP4 TFLOPS (per GPU) = 1.3 PFLOPS = 1.3e15 FLOPS/s
# But we run TP=4 split across 4 GPUs. Each expert's weights are sharded.
# The kernel runs on 1 GPU per worker; the per-call FLOPS we computed is for ONE rank.
# MI355X per-GPU peak FP4 = 1.3e15 / num_gpus_per_card... actually each MI355X die
# has its own peak. Let me use 1.3e15 as the per-GPU peak for now.
PEAK_FLOPS_PER_GPU = 1.3e15  # MI355X FP4
TIME_S = mean_ms_per_iter / 1000.0
ACHIEVED_FLOPS = flops_per_call / TIME_S
PCT_PEAK = 100.0 * ACHIEVED_FLOPS / PEAK_FLOPS_PER_GPU

print(f"\n=== Roofline (compute-side) ===")
print(f"FLOPS per call (1 rank): {flops_per_call:.3e}")
print(f"Achieved TFLOPS at M=4: {ACHIEVED_FLOPS / 1e12:.2f}")
print(f"Peak FP4 TFLOPS (MI355X est): {PEAK_FLOPS_PER_GPU / 1e12:.2f}")
print(f"Percent of peak: {PCT_PEAK:.2f}%")

# HBM-bound check: how much HBM read per call?
# Per call we read: w13 (per active expert per token) + w2 (per active expert per token) + scales
# At M=4, topk=9: 4*9 = 36 expert activations. Bytes:
# w13 per expert: 1024 * 3584 = 3.7 MB FP4 + 1024*224 = 230 KB scale ~ 3.9 MB
# w2 per expert: 7168 * 256 = 1.8 MB FP4 + 7168*8 = 57 KB scale ~ 1.9 MB
# Total per expert activation: ~5.8 MB
# Total HBM read (no caching, all 36 expert activations): 36 * 5.8 = 209 MB
# Conservative (assumes cache hit on common experts, real usage ~ unique experts * 5.8 MB)
hbm_per_expert_mb = (w13.shape[1] * w13.shape[2] + w13_scale.shape[1] * w13_scale.shape[2] + w2.shape[1] * w2.shape[2] + w2_scale.shape[1] * w2_scale.shape[2]) / 1e6
unique_experts = len(torch.unique(topk_ids).tolist())
hbm_total_mb = unique_experts * hbm_per_expert_mb  # weights bytes
hbm_per_call_mb = hbm_total_mb  # plus a tiny amount for activations and scales
HBM_PEAK_GBS = 5300  # MI355X HBM3e ~5.3 TB/s
hbm_bandwidth_used_gbs = (hbm_per_call_mb / 1000.0) / TIME_S
hbm_pct = 100.0 * hbm_bandwidth_used_gbs / HBM_PEAK_GBS

print(f"\n=== Roofline (HBM-side) ===")
print(f"Unique experts touched at M=4: {unique_experts}")
print(f"Estimated HBM read per call: {hbm_per_call_mb:.1f} MB")
print(f"Achieved HBM bandwidth: {hbm_bandwidth_used_gbs:.1f} GB/s")
print(f"Peak HBM bandwidth (MI355X est): {HBM_PEAK_GBS} GB/s")
print(f"Percent of peak HBM: {hbm_pct:.2f}%")

# Verdict
print(f"\n=== R2-B verdict ===")
if PCT_PEAK > 80:
    print("COMPUTE-BOUND: kernel near peak. R2 has limited headroom. STOP candidate.")
elif hbm_pct > 80:
    print("HBM-BOUND: bandwidth saturated. R2 has limited headroom. STOP candidate.")
else:
    print(f"NEITHER COMPUTE NOR HBM BOUND: compute={PCT_PEAK:.1f}% peak, HBM={hbm_pct:.1f}% peak.")
    print("R2 has structural headroom (likely tile-M waste at M=4). PROCEED to R2-C.")

# Save JSON for record
result = {
    "M": M,
    "H": H,
    "intermediate": intermediate,
    "num_experts": num_experts,
    "topk": topk,
    "T0_us_per_call": mean_us_per_iter,
    "n_iters": N_ITERS,
    "achieved_tflops": ACHIEVED_FLOPS / 1e12,
    "peak_fp4_tflops_estimate": PEAK_FLOPS_PER_GPU / 1e12,
    "pct_compute_peak": PCT_PEAK,
    "hbm_bandwidth_gbs": hbm_bandwidth_used_gbs,
    "hbm_peak_gbs_estimate": HBM_PEAK_GBS,
    "pct_hbm_peak": hbm_pct,
    "unique_experts": unique_experts,
}
with open(f"{SNAP}/r2b_microbench_result.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nResults saved to {SNAP}/r2b_microbench_result.json")

# Apr 26 2026 Boot + Bench Scripts

Verbatim copies of the production boot + bench scripts that produced the Apr 26 3/4-gates result. All shell-runnable on a docker container based on `rocm/atom-dev:dsr1_apr26_3of4_validated_warm_bench` (sha `8e844757ad6c`) or `dsr1_session17_re1_best_3of4_apr23_0627` (sha `2286b9de5107`) with the Apr 26 patches from [`patches/`](../patches/).

## Files

| Script | Purpose | Used for |
|---|---|---|
| [`boot_apr26_tp4.sh`](boot_apr26_tp4.sh) | TP=4 server launch with locked Apr 26 env stack (RELAXED 9/0.5, INT4 QR, RCCL_MSCCLPP, TBO prefill, MTP=3, etc.) | All CONC=4 and CONC=32 benches |
| [`boot_apr26_tp8.sh`](boot_apr26_tp8.sh) | TP=8 server launch (same env stack, `HIP_VISIBLE_DEVICES=0..7`, `-tp 8`) | CONC=128 sweep |
| [`bench_apr26.sh`](bench_apr26.sh) | 8-curl warmup + 3× sequential `benchmark_serving` runs at CONC=4 ISL=8192 OSL=1024 num_prompts=40 | The CONC=4 canonical 3-run sweep that produced TPOT_med 4.84 ms |

## How to use

After applying patches from [`../patches/`](../patches/) to ATOM + aiter source trees, building aiter as documented, and starting a container based on the canonical image:

```bash
# 1) Inside the container, copy boot script to /tmp
docker cp scripts/boot_apr26_tp4.sh dsr1_repro:/tmp/boot_cdna4_moe.sh
docker cp scripts/bench_apr26.sh    dsr1_repro:/tmp/proper_warmup_bench.sh

# 2) Reset perf-determinism (let GPU SCLK boost)
docker exec dsr1_repro rocm-smi --resetperfdeterminism -d 0 -d 1 -d 2 -d 3

# 3) Boot the server in background
docker exec -d dsr1_repro bash /tmp/boot_cdna4_moe.sh
# Wait ~10-13 minutes; tail /tmp/cdna4_boot_*.log until "Application startup complete"

# 4) Run warmup + 3x bench
docker exec dsr1_repro bash /tmp/proper_warmup_bench.sh
```

Expected output: `TPOT_med ≈ 4.84 ms`, `thr/GPU ≈ 1650`, `Interact ≈ 207`, `E2E_calc ≈ 5240 ms`, `GSM8K ≈ 0.94+` (separately).

## Boot env stack reference

The full env stack with inline justifications is in [`../docs/REPRODUCE.md`](../docs/REPRODUCE.md) Section 3. Key envs `boot_apr26_tp4.sh` exports:

```bash
# Cache + offline
export HOME=/tmp HF_HOME=/tmp/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# ATOM features
export AITER_ENABLE_VSKIP=0 ATOM_ENABLE_RELAXED_MTP=1
export ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=1024  # KEEP at 1024
# HIP
export HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1
export HIP_VISIBLE_DEVICES=0,1,2,3
# Collectives — RCCL ROCm 7.1+ knobs
export NCCL_MIN_NCHANNELS=16
export RCCL_MSCCLPP_ENABLE=1 RCCL_MSCCLPP_THRESHOLD=1048576 RCCL_P2P_BATCH_ENABLE=1
# Cold-boot timeout extension (NEW Apr 26)
export NCCL_TIMEOUT=3600 NCCL_BLOCKING_WAIT=0
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=3600
# QuickReduce INT4 AR
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
# Removed Apr 26 (silent regression)
# export ATOM_MSCG_K=2  ← DISABLED
export ATOM_MSCG_P6_REPLAY=0
# Future-stage scaffolds (default OFF for canonical bench)
export ATOM_USE_CDNA4_MOE_GEMM2=0
```

Server CLI flags:
```
--model amd/DeepSeek-R1-0528-MXFP4
--server-port 8890 -tp 4
--kv_cache_dtype fp8
--max-model-len 10240
--method mtp --num-speculative-tokens 3
--enable-tbo prefill
--max-num-batched-tokens 65536
--cudagraph-capture-sizes "[1,2,4,8,16,32]"
```

## What's the 8-curl warmup doing?

Hits all decode cudagraph capture sizes (1, 2, 4, 8) in succession with small `max_tokens=50` requests. Each curl pushes a new request through the scheduler at progressively-rising batch size, exercising every cudagraph variant before the bench tool starts. Without this, the **first bench's first 1-2 batches eat the cudagraph compile + JIT cost**, contaminating TPOT_mean and the throughput count.

Empirical impact (CONC=4, cold-boot Run 1, 1-replica):
- TPOT_mean: **5.02 ms with warmup → 7.15 ms without (+42%)**
- thr/GPU:   **1660 with warmup → 1161 without (−30%)**
- p99 TPOT:  **6.86 ms with warmup → 23.33 ms without (+240%)**

At CONC=128 cold-boot the warmup effect is more dramatic still — see [`../bench_results/apr26/README.md`](../bench_results/apr26/README.md).

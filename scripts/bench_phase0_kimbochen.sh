#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 0 baseline bench — kimbochen-official 3-iter
# Source: kimbochen/dsr1_benchmark.cpp (the only valid scoring harness)
# Gates (CONC=4): GSM8K ≥ 0.93, Intvty ≥ 165, Tput/GPU ≥ 1500, E2E ≤ 5000
# Expected gold+v6c medians: TPOT 6.080 / E2E 6743 / Tput 1403 / Intvty 164.48 / GSM8K 0.9340 = 1/4
# ─────────────────────────────────────────────────────────────────────────────

set -e

BENCH=/projects/teamA/danish/repos/amdgpu_bounty_optimization/dsr1-fp4-atom-mtp-mi355x/dsr1_benchmark
if [[ ! -x "$BENCH" ]]; then
  echo "ERROR: kimbochen binary not found at $BENCH"
  echo "Locate via: find /projects/teamA -name dsr1_benchmark -type f 2>/dev/null"
  exit 1
fi

export MODEL=amd/DeepSeek-R1-0528-MXFP4
export PORT=8890
export TP=4
export CONC=4
export ISL=8192
export OSL=1024
export RANDOM_RANGE_RATIO=1.0
export NUM_PROMPTS=40

RUNDIR=/tmp/phase0_bench_$(date +%H%M%S)
mkdir -p "$RUNDIR"
echo "=== Phase 0 bench RUNDIR=$RUNDIR ==="

# 8-curl warmup — eliminates iter-1 cold-decode TPOT spike (15 ms → 6 ms)
echo "=== 8-curl warmup ==="
for i in 1 2 3 4 5 6 7 8; do
  curl -sf -X POST http://0.0.0.0:8890/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"amd/DeepSeek-R1-0528-MXFP4\",\"prompt\":\"hi $i\",\"max_tokens\":8,\"temperature\":0}" \
    > "$RUNDIR/warm_$i.txt" 2>&1 || echo "warmup $i failed"
done

# 3-iter perf bench
echo "=== 3-iter kimbochen perf ==="
for ITER in 1 2 3; do
  export RESULT_FILENAME=$RUNDIR/result_iter${ITER}
  echo "  iter $ITER → $RESULT_FILENAME"
  $BENCH perf 2>&1 | tee "$RUNDIR/iter${ITER}.log"
done

# Parse + emit summary
echo "=== Summary ==="
python3 - <<EOF
import json, statistics, sys
results = []
for it in (1,2,3):
    p = f"$RUNDIR/result_iter{it}.json"
    try:
        d = json.load(open(p))
    except FileNotFoundError:
        print(f"iter {it}: MISSING {p}")
        continue
    tpot = d.get("median_tpot_ms")
    e2e = d.get("median_e2el_ms")
    tput = d.get("total_token_throughput", 0) / 4.0   # TP=4 override
    intvty = 1000.0 / tpot if tpot else 0
    gsm = d.get("gsm8k_metric", "n/a")
    print(f"iter {it}: TPOT {tpot:.3f} | E2E {e2e:.0f} | Tput/GPU {tput:.0f} | Intvty {intvty:.2f} | GSM8K {gsm}")
    results.append((tpot, e2e, tput, intvty, gsm))
if len(results) >= 2:
    tpots = [r[0] for r in results]
    e2es = [r[1] for r in results]
    tputs = [r[2] for r in results]
    intvtys = [r[3] for r in results]
    gsms = [r[4] for r in results if isinstance(r[4], (int,float))]
    print(f"\nMEDIAN: TPOT {statistics.median(tpots):.3f} | E2E {statistics.median(e2es):.0f} | Tput/GPU {statistics.median(tputs):.0f} | Intvty {statistics.median(intvtys):.2f} | GSM8K {statistics.median(gsms) if gsms else 'n/a'}")
    print("\nGate check (CONC=4):")
    print(f"  GSM8K  {statistics.median(gsms):.4f} {'PASS' if statistics.median(gsms) >= 0.93 else 'FAIL'}  (gate ≥ 0.93)")
    print(f"  Intvty {statistics.median(intvtys):.2f} {'PASS' if statistics.median(intvtys) >= 165 else 'FAIL'}  (gate ≥ 165)")
    print(f"  Tput   {statistics.median(tputs):.0f}    {'PASS' if statistics.median(tputs) >= 1500 else 'FAIL'}  (gate ≥ 1500)")
    print(f"  E2E    {statistics.median(e2es):.0f}    {'PASS' if statistics.median(e2es) <= 5000 else 'FAIL'}  (gate ≤ 5000)")
EOF

echo ""
echo "RUNDIR=$RUNDIR"
echo "Copy to host bind: cp -r $RUNDIR /projects/teamA/danish/apr29_evidence/phase0_baseline/"

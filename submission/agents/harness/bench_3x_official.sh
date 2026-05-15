#!/bin/bash
# Run the OFFICIAL bench (single GSM8K + single perf, mirrors kimbochen/dsr1_benchmark.cpp)
# THREE TIMES back-to-back. 3 GSM8K runs + 3 perf runs total. No N=3 wrapper.
export HOME=/tmp HF_HOME=/tmp/.cache/huggingface HUGGINGFACE_HUB_CACHE=/tmp/.cache/huggingface/hub TRANSFORMERS_CACHE=/tmp/.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Warmup once before all three iterations.
for i in 1 2 3 4 5 6 7 8; do
  curl -sf -X POST http://localhost:8890/v1/completions -H "Content-Type: application/json" \
    -d '{"model":"amd/DeepSeek-R1-0528-MXFP4","prompt":"hi","max_tokens":8,"temperature":0}' > /dev/null
done

for ITER in 1 2 3; do
  TS=$(date +%H%M%S)
  RUNDIR=/tmp/official_3x_iter${ITER}_${TS}
  mkdir -p "$RUNDIR"
  SUMMARY="$RUNDIR/summary.txt"

  echo "==== ITER ${ITER}/3 begin $(date +%T) ====" | tee "$SUMMARY"

  # ---- Step 1: GSM8K (official: num_fewshot=3 num_concurrent=65 max_retries=1) ----
  echo "==== STEP 1: GSM8K (single run) ====" | tee -a "$SUMMARY"
  GSM_LOG="$RUNDIR/gsm8k.log"
  lm_eval --model local-completions \
    --model_args "model=amd/DeepSeek-R1-0528-MXFP4,base_url=http://localhost:8890/v1/completions,num_concurrent=65,max_retries=1,tokenized_requests=False" \
    --tasks gsm8k --num_fewshot 3 2>&1 | tee "$GSM_LOG" >/dev/null

  GSM_VAL=$(grep -E "^\|gsm8k\|.*flexible-extract.*exact_match" "$GSM_LOG" | head -1 | python3 -c "
import sys, re
line = sys.stdin.read().strip()
m = re.search(r'\|\s*([0-9]+\.[0-9]+)\s*\|\s*±', line)
print(m.group(1) if m else 'NA')
")
  echo "GSM8K = $GSM_VAL" | tee -a "$SUMMARY"

  PASS=$(python3 -c "v='$GSM_VAL'; import sys; sys.exit(0 if v != 'NA' and float(v) >= 0.93 else 1)" && echo YES || echo NO)
  echo "GSM8K_PASS=$PASS (gate >=0.93)" | tee -a "$SUMMARY"

  if [ "$PASS" != "YES" ]; then
    echo "FAIL: GSM8K below gate. Performance bench SKIPPED for iter $ITER (per official rule)." | tee -a "$SUMMARY"
    echo "==== ITER ${ITER}/3 end $(date +%T) (perf SKIPPED) ====" | tee -a "$SUMMARY"
    continue
  fi

  # ---- Step 2: Perf bench (official params: random-range=1.0, num-warmups=8, use-chat-template, num_prompts=40, max-conc=4) ----
  echo "" | tee -a "$SUMMARY"
  echo "==== STEP 2: Perf bench ====" | tee -a "$SUMMARY"
  PERF_RESULT="$RUNDIR/perf_result.json"
  cd /app/ATOM
  python3 -m atom.benchmarks.benchmark_serving \
    --model amd/DeepSeek-R1-0528-MXFP4 \
    --backend vllm --base-url http://localhost:8890 \
    --dataset-name random --random-input-len 8192 --random-output-len 1024 \
    --random-range-ratio 1.0 \
    --num-prompts 40 --max-concurrency 4 \
    --request-rate inf --ignore-eos \
    --save-result --result-filename "$(basename $PERF_RESULT)" --result-dir "$(dirname $PERF_RESULT)/" \
    --num-warmups 8 \
    --use-chat-template \
    --percentile-metrics ttft,tpot,itl,e2el 2>&1 | tee "$RUNDIR/perf.log" >/dev/null

  python3 - <<PYEOF | tee -a "$SUMMARY"
import json, glob
results = sorted(glob.glob("$RUNDIR/perf_result*.json"))
if not results:
    print("ERROR: no perf result JSON")
    raise SystemExit(2)
d = json.load(open(results[-1]))
e2e_med = d.get('median_e2el_ms', 0.0)
tpot_med = d.get('median_tpot_ms', 0.0)
tput_total = d.get('total_token_throughput', 0.0)
tput_per_gpu = tput_total / 4.0
intvty = (1000.0 / tpot_med) if tpot_med > 0 else 0.0
gsm = float("$GSM_VAL")

print('')
print('==== OFFICIAL GATES (CONC=4, TP=4) ====')
print(f'GSM8K        : {gsm:.4f}        gate >=0.93   ' + ('PASS' if gsm >= 0.93 else 'FAIL'))
print(f'Median E2E   : {e2e_med:.2f} ms  gate <=5000   ' + ('PASS' if e2e_med <= 5000 else 'FAIL'))
print(f'Tput/GPU(/4) : {tput_per_gpu:.2f}   gate >=1500   ' + ('PASS' if tput_per_gpu >= 1500 else 'FAIL'))
print(f'Interactivity: {intvty:.2f} tok/s/u gate >=165    ' + ('PASS' if intvty >= 165 else 'FAIL'))
print(f'TPOT_med     : {tpot_med:.3f} ms')
print('')
gates = [gsm>=0.93, e2e_med<=5000, tput_per_gpu>=1500, intvty>=165]
print(f'OVERALL: {sum(gates)}/4 gates passed')
PYEOF

  echo "==== ITER ${ITER}/3 end $(date +%T) ====" | tee -a "$SUMMARY"
  echo "Run dir: $RUNDIR" | tee -a "$SUMMARY"
done

echo ""
echo "==== ALL 3 OFFICIAL BENCHES COMPLETE ===="
ls -td /tmp/official_3x_iter*/ 2>/dev/null | head -3

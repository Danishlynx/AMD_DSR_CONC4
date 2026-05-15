#!/bin/bash
# Official harness with N=3 GSM8K runs + median gate.
# Replaces single-shot /tmp/official_bench_v1.sh in the harness loop.
#
# Sequence:
#   - Run GSM8K 3 times (num_fewshot=3, num_concurrent=65, max_retries=1)
#   - Take median.
#   - If median >= 0.93, run official perf bench (random-range=1.0, use-chat-template, num-warmups=8, num-prompts=40, max-conc=4).
#   - Compute tput_per_gpu = total_token_throughput / 4 (TP=4).
#   - Print "OFFICIAL GATES" summary with all metrics.
#
# Logs: /tmp/official_n3_<TS>/{gsm8k_run{1,2,3}.log, perf.log, summary.txt}
# Result JSON: /tmp/official_n3_<TS>/perf_result.json

export HOME=/tmp HF_HOME=/tmp/.cache/huggingface HUGGINGFACE_HUB_CACHE=/tmp/.cache/huggingface/hub TRANSFORMERS_CACHE=/tmp/.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

TS=$(date +%H%M%S)
RUNDIR=/tmp/official_n3_$TS
mkdir -p "$RUNDIR"
SUMMARY="$RUNDIR/summary.txt"

echo "==== STEP 1: GSM8K accuracy ×3 (num_fewshot=3, concurrent=65, max_retries=1) ====" | tee "$SUMMARY"

GSM_VALS=()
for RUN in 1 2 3; do
  GSM_LOG="$RUNDIR/gsm8k_run${RUN}.log"
  echo "[gsm8k run $RUN/3]" | tee -a "$SUMMARY"
  lm_eval --model local-completions \
    --model_args "model=amd/DeepSeek-R1-0528-MXFP4,base_url=http://localhost:8890/v1/completions,num_concurrent=65,max_retries=1,tokenized_requests=False" \
    --tasks gsm8k --num_fewshot 3 2>&1 | tee "$GSM_LOG" >/dev/null

  VAL=$(grep -E "^\|gsm8k\|.*flexible-extract.*exact_match" "$GSM_LOG" | head -1 | python3 -c "
import sys, re
line = sys.stdin.read().strip()
m = re.search(r'\|\s*([0-9]+\.[0-9]+)\s*\|\s*±', line)
print(m.group(1) if m else 'NA')
")
  echo "  run $RUN: GSM8K = $VAL" | tee -a "$SUMMARY"
  GSM_VALS+=("$VAL")
done

# Compute median (write vals to temp file to avoid shell-quoting hell)
echo "${GSM_VALS[@]}" > "$RUNDIR/gsm8k_vals.txt"
GSM_MEDIAN=$(python3 -c "
vals = open('$RUNDIR/gsm8k_vals.txt').read().split()
vals = [float(v) for v in vals if v != 'NA']
vals.sort()
if vals:
    print(f'{vals[len(vals)//2]:.4f}')
else:
    print('NA')
")

echo "" | tee -a "$SUMMARY"
echo "GSM8K runs: ${GSM_VALS[@]}" | tee -a "$SUMMARY"
echo "GSM8K_median = $GSM_MEDIAN" | tee -a "$SUMMARY"

PASS=$(python3 -c "v='$GSM_MEDIAN'; import sys; sys.exit(0 if v != 'NA' and float(v) >= 0.93 else 1)" && echo YES || echo NO)
echo "GSM8K_PASS_MEDIAN=$PASS (gate >=0.93)" | tee -a "$SUMMARY"

if [ "$PASS" != "YES" ]; then
  echo "" | tee -a "$SUMMARY"
  echo "FAIL: median GSM8K below gate. Performance benchmark SKIPPED." | tee -a "$SUMMARY"
  exit 1
fi

echo "" | tee -a "$SUMMARY"
echo "==== STEP 2: Perf bench (random-range=1.0, num-warmups=8, use-chat-template, num_prompts=40 conc=4) ====" | tee -a "$SUMMARY"

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
gsm = float("$GSM_MEDIAN")

print('')
print('==== OFFICIAL GATES (CONC=4, TP=4, GSM8K=median(N=3)) ====')
print(f'GSM8K_med    : {gsm:.4f}        gate >=0.93   ' + ('PASS' if gsm >= 0.93 else 'FAIL'))
print(f'Median E2E   : {e2e_med:.2f} ms  gate <=5000   ' + ('PASS' if e2e_med <= 5000 else 'FAIL'))
print(f'Tput/GPU(/4) : {tput_per_gpu:.2f}   gate >=1500   ' + ('PASS' if tput_per_gpu >= 1500 else 'FAIL'))
print(f'Interactivity: {intvty:.2f} tok/s/u gate >=165    ' + ('PASS' if intvty >= 165 else 'FAIL'))
print(f'TPOT_med     : {tpot_med:.3f} ms')
print('')
gates = [gsm>=0.93, e2e_med<=5000, tput_per_gpu>=1500, intvty>=165]
print(f'OVERALL: {sum(gates)}/4 gates passed')
PYEOF

echo "" | tee -a "$SUMMARY"
echo "Run dir: $RUNDIR"
echo "Summary: $SUMMARY"

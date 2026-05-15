#!/bin/bash
# Same as official_bench_n3.sh but for CONC=32 (gates: e2e<=18000, intvty>=50, tput/GPU>=3900).
# num-prompts = CONC*10 = 320, num-warmups = 2*CONC = 64.
# Tput/GPU = total_token_throughput / 4 (TP=4).

export HOME=/tmp HF_HOME=/tmp/.cache/huggingface HUGGINGFACE_HUB_CACHE=/tmp/.cache/huggingface/hub TRANSFORMERS_CACHE=/tmp/.cache/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

TS=$(date +%H%M%S)
RUNDIR=/tmp/official_n3_conc32_$TS
mkdir -p "$RUNDIR"
SUMMARY="$RUNDIR/summary.txt"

echo "==== STEP 1: GSM8K accuracy (single run, num_fewshot=3, concurrent=65) ====" | tee "$SUMMARY"
echo "(GSM8K already validated at CONC=4 baseline — running once for record only.)" | tee -a "$SUMMARY"

GSM_LOG="$RUNDIR/gsm8k_run1.log"
lm_eval --model local-completions \
  --model_args "model=amd/DeepSeek-R1-0528-MXFP4,base_url=http://localhost:8890/v1/completions,num_concurrent=65,max_retries=1,tokenized_requests=False" \
  --tasks gsm8k --num_fewshot 3 2>&1 | tee "$GSM_LOG" >/dev/null

VAL=$(grep -E "^\|gsm8k\|.*flexible-extract.*exact_match" "$GSM_LOG" | head -1 | python3 -c "
import sys, re
line = sys.stdin.read().strip()
m = re.search(r'\|\s*([0-9]+\.[0-9]+)\s*\|\s*±', line)
print(m.group(1) if m else 'NA')
")
echo "GSM8K_run1 = $VAL" | tee -a "$SUMMARY"

echo "" | tee -a "$SUMMARY"
echo "==== STEP 2: Perf bench CONC=32 (random-range=1.0, num-warmups=64, use-chat-template, num_prompts=320) ====" | tee -a "$SUMMARY"

PERF_RESULT="$RUNDIR/perf_result.json"
cd /app/ATOM
python3 -m atom.benchmarks.benchmark_serving \
  --model amd/DeepSeek-R1-0528-MXFP4 \
  --backend vllm --base-url http://localhost:8890 \
  --dataset-name random --random-input-len 8192 --random-output-len 1024 \
  --random-range-ratio 1.0 \
  --num-prompts 320 --max-concurrency 32 \
  --request-rate inf --ignore-eos \
  --save-result --result-filename "$(basename $PERF_RESULT)" --result-dir "$(dirname $PERF_RESULT)/" \
  --num-warmups 64 \
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
gsm = float("$VAL") if "$VAL" != "NA" else 0.0

print('')
print('==== OFFICIAL GATES (CONC=32, TP=4) ====')
print(f'GSM8K        : {gsm:.4f}        gate >=0.93   ' + ('PASS' if gsm >= 0.93 else 'FAIL'))
print(f'Median E2E   : {e2e_med:.2f} ms  gate <=18000  ' + ('PASS' if e2e_med <= 18000 else 'FAIL'))
print(f'Tput/GPU(/4) : {tput_per_gpu:.2f}   gate >=3900  ' + ('PASS' if tput_per_gpu >= 3900 else 'FAIL'))
print(f'Interactivity: {intvty:.2f} tok/s/u gate >=50   ' + ('PASS' if intvty >= 50 else 'FAIL'))
print(f'TPOT_med     : {tpot_med:.3f} ms')
print('')
gates = [gsm>=0.93, e2e_med<=18000, tput_per_gpu>=3900, intvty>=50]
print(f'OVERALL: {sum(gates)}/4 gates passed')
PYEOF

echo ""
echo "Run dir: $RUNDIR"
echo "Summary: $SUMMARY"

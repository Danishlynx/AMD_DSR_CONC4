#!/bin/bash
# Proper warmup + bench per REPRODUCE.md step 4 protocol.
# Use 8 small "Hello world N" curls THEN 3 perf benches.
set -e
export HOME=/tmp HF_HOME=/tmp/.cache/huggingface HUGGINGFACE_HUB_CACHE=/tmp/.cache/huggingface/hub
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MODEL=amd/DeepSeek-R1-0528-MXFP4
export PORT=8890 TP=4 CONC=4 ISL=8192 OSL=1024 NUM_PROMPTS=40

LOG=/tmp/proper_warmup_$(date +%H%M%S).log
echo "LOG=$LOG"

echo "===== WARMUP (8 small curls, REPRODUCE.md style) =====" | tee -a "$LOG"
for i in 1 2 3 4 5 6 7 8; do
  curl -s http://0.0.0.0:8890/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"amd/DeepSeek-R1-0528-MXFP4\",\"prompt\":\"Hello world $i\",\"max_tokens\":50,\"temperature\":0}" > /dev/null
done
echo "warmup_done" | tee -a "$LOG"

for i in 1 2 3; do
  echo "===== BENCH RUN $i =====" | tee -a "$LOG"
  cd /app/ATOM
  python3 -m atom.benchmarks.benchmark_serving \
    --model $MODEL --port $PORT \
    --dataset-name random --random-input-len $ISL --random-output-len $OSL \
    --num-prompts $NUM_PROMPTS --max-concurrency $CONC --trust-remote-code \
    --save-result --result-filename /tmp/proper_run${i}.json 2>&1 | tail -25 | tee -a "$LOG"
  sleep 5
done

echo "===== EXTRACT =====" | tee -a "$LOG"
python3 /tmp/extract_runs_proper.py | tee -a "$LOG"

echo "DONE LOG=$LOG"

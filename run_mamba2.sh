#!/bin/bash
# run_mamba2.sh — Run Entropic Deviation experiment on Mamba2-2.7B
#
# Prerequisites:
#   pip install transformers torch mamba_ssm causal_conv1d
#   (mamba_ssm and causal_conv1d optional but recommended for CUDA kernels)
#
# Hardware: single P40 (24 GB VRAM) is sufficient for Mamba2-2.7B in float16.
#
# Usage:
#   ./run_mamba2.sh              # full run (semantic + neutral, 3 temps)
#   ./run_mamba2.sh --resume     # resume interrupted run

set -euo pipefail

MODEL="state-spaces/mamba2-2.7b"
MODEL_NAME="mamba2-2.7b"
PROMPTS_SEM="prompts/prompts.jsonl"
PROMPTS_NEU="prompts/prompts_neutral.jsonl"
TEMPS="0.7 1.0 1.3"
MAX_TOKENS=128
SAVE_INTERVAL=20
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results"
LOGS_DIR="logs"

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

RESUME_FLAG=""
if [[ "${1:-}" == "--resume" ]]; then
    RESUME_FLAG="--resume"
    echo "Resuming from previous run..."
fi

echo "=== Mamba2-2.7B ED Experiment ==="
echo "Model: $MODEL"
echo "Timestamp: $TIMESTAMP"
echo ""

# Step 1: Semantic prompts
echo "--- Semantic prompts (800 × 3 temps = 2400 generations) ---"
python generate_logits_hf.py \
    --model "$MODEL" \
    --model-name "$MODEL_NAME" \
    --prompts "$PROMPTS_SEM" \
    --temps $TEMPS \
    --max_tokens $MAX_TOKENS \
    --ed-out "${RESULTS_DIR}/ed_${MODEL_NAME}_prompts_${TIMESTAMP}.csv" \
    --save_interval $SAVE_INTERVAL \
    --shuffle \
    --dtype float16 \
    --log "${LOGS_DIR}/${MODEL_NAME}_prompts_${TIMESTAMP}.log" \
    --progress-file "${RESULTS_DIR}/progress_${MODEL_NAME}_prompts.txt" \
    $RESUME_FLAG

echo ""

# Step 2: Neutral prompts
echo "--- Neutral prompts (1000 × 3 temps = 3000 generations) ---"
python generate_logits_hf.py \
    --model "$MODEL" \
    --model-name "$MODEL_NAME" \
    --prompts "$PROMPTS_NEU" \
    --temps $TEMPS \
    --max_tokens $MAX_TOKENS \
    --ed-out "${RESULTS_DIR}/ed_${MODEL_NAME}_prompts_neutral_${TIMESTAMP}.csv" \
    --save_interval $SAVE_INTERVAL \
    --shuffle \
    --dtype float16 \
    --log "${LOGS_DIR}/${MODEL_NAME}_prompts_neutral_${TIMESTAMP}.log" \
    --progress-file "${RESULTS_DIR}/progress_${MODEL_NAME}_prompts_neutral.txt" \
    $RESUME_FLAG

echo ""

# Step 3: Statistical tests
echo "--- Running F1-F8 tests ---"
for PTYPE in prompts prompts_neutral; do
    CSV="${RESULTS_DIR}/ed_${MODEL_NAME}_${PTYPE}_${TIMESTAMP}.csv"
    if [[ -f "$CSV" ]]; then
        python calculate_metrics.py "$CSV" \
            --extra-csv results/ed_results_combined.csv \
            --out "${RESULTS_DIR}/FT_${MODEL_NAME}_${PTYPE}_${TIMESTAMP}.csv"
    fi
done

# Step 4: Combined file
echo "--- Creating combined CSV ---"
SEM_CSV="${RESULTS_DIR}/ed_${MODEL_NAME}_prompts_${TIMESTAMP}.csv"
NEU_CSV="${RESULTS_DIR}/ed_${MODEL_NAME}_prompts_neutral_${TIMESTAMP}.csv"
COMBINED="${RESULTS_DIR}/ed_${MODEL_NAME}_combined_${TIMESTAMP}.csv"

if [[ -f "$SEM_CSV" && -f "$NEU_CSV" ]]; then
    head -1 "$SEM_CSV" > "$COMBINED"
    tail -n +2 -q "$SEM_CSV" "$NEU_CSV" >> "$COMBINED"
    python calculate_metrics.py "$COMBINED" \
        --extra-csv results/ed_results_combined.csv \
        --out "${RESULTS_DIR}/FT_${MODEL_NAME}_combined_${TIMESTAMP}.csv"
fi

echo ""
echo "=== Done ==="
echo "Results in ${RESULTS_DIR}/ed_${MODEL_NAME}_*_${TIMESTAMP}.csv"

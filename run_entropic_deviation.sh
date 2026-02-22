#!/bin/bash

# =========================================================
# run_entropic_deviation.sh
# Full pipeline: generate logits → calculate ED → run statistical tests
# =========================================================

set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# --- Configuration (edit as needed) -----------------------
MODEL_PATH="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
PROMPTS_PATH="prompts/prompts.jsonl"
MODEL_NAME="Mistral-7B"
TEMPS="0.7 1.0 1.3"
MAX_TOKENS=128
N_CTX=512
N_GPU_LAYERS=-1
SAVE_INTERVAL=5
RESULTS_DIR="results"
LOG_DIR="logs"
# ----------------------------------------------------------

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# Validate inputs
for f in "$SCRIPT_DIR/$MODEL_PATH" "$SCRIPT_DIR/$PROMPTS_PATH"; do
    if [ ! -f "$f" ]; then
        echo "Error: $f not found."
        exit 1
    fi
done

LOGITS_PREFIX="$RESULTS_DIR/logits_${TIMESTAMP}"
ED_CSV="$RESULTS_DIR/ed_results_${TIMESTAMP}.csv"
FT_CSV="$RESULTS_DIR/FTresults_${TIMESTAMP}.csv"

echo "====================================================="
echo "Entropic Deviation Pipeline"
echo "Model:   $MODEL_PATH"
echo "Prompts: $PROMPTS_PATH"
echo "Temps:   $TEMPS"
echo "Results: $RESULTS_DIR"
echo "====================================================="

# Step 1: Generate logits
echo "[1/3] Generating logits..."
python "$SCRIPT_DIR/generate_logits.py" \
    --model "$MODEL_PATH" \
    --prompts "$PROMPTS_PATH" \
    --temps $TEMPS \
    --max_tokens "$MAX_TOKENS" \
    --n_ctx "$N_CTX" \
    --n_gpu_layers "$N_GPU_LAYERS" \
    --save_interval "$SAVE_INTERVAL" \
    --out "$LOGITS_PREFIX" \
    --log "$LOG_DIR/generate_${TIMESTAMP}.log"

# Step 2: Calculate ED
echo "[2/3] Computing Entropic Deviation..."
python "$SCRIPT_DIR/calculate_ed.py" \
    --pattern "${LOGITS_PREFIX}_gpu*_chkpt_*.pt" \
    --out "$ED_CSV" \
    --model-name "$MODEL_NAME"

# Step 3: Statistical tests
echo "[3/3] Running statistical tests (F1-F8)..."
python "$SCRIPT_DIR/calculate_metrics.py" "$ED_CSV" \
    --out "$FT_CSV"

echo "====================================================="
echo "Pipeline complete!"
echo "  ED results: $ED_CSV"
echo "  FT results: $FT_CSV"
echo "====================================================="

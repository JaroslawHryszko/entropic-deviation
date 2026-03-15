#!/bin/bash
# run_multilingual.sh — Cross-lingual ED experiment on Qwen-2.5-32B
#
# Prerequisites:
#   pip install datasets nltk
#   python prompts/build_multilingual_prompts.py
#
# Design:
#   Compare ED across 4 non-English languages (pl, zh, ar, ja) using
#   Qwen-32B which has strong multilingual support. English baseline
#   from existing prompts_20260304 data. Neutral prompts are already
#   language-independent — reuse existing neutral results.
#
# Expected: if ED is stable across languages, non-randomness is
# language-independent. If it varies, language structure contributes.

set -euo pipefail

MODEL="models/Qwen2.5-32B-Instruct-Q4_K_M.gguf"
MODEL_NAME="Qwen2.5-32B-Instruct"
PROMPTS="prompts/prompts_multilingual.jsonl"
TEMPS="0.7 1.0 1.3"
MAX_TOKENS=128
SAVE_INTERVAL=20
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results"
LOGS_DIR="logs"

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# Step 0: Build multilingual prompts if not present
if [[ ! -f "$PROMPTS" ]]; then
    echo "Building multilingual prompts..."
    python prompts/build_multilingual_prompts.py
fi

PROMPT_COUNT=$(wc -l < "$PROMPTS")
echo "=== Multilingual ED Experiment ==="
echo "Model: $MODEL_NAME"
echo "Prompts: $PROMPT_COUNT × ${#TEMPS} temps"
echo "Timestamp: $TIMESTAMP"
echo ""

RESUME_FLAG=""
if [[ "${1:-}" == "--resume" ]]; then
    RESUME_FLAG="--resume"
    echo "Resuming from previous run..."
fi

# Step 1: Generate ED
echo "--- Running multilingual prompts ---"
python generate_logits.py \
    --model "$MODEL" \
    --model-name "$MODEL_NAME" \
    --prompts "$PROMPTS" \
    --temps $TEMPS \
    --max_tokens $MAX_TOKENS \
    --ed-out "${RESULTS_DIR}/ed_${MODEL_NAME}_multilingual_${TIMESTAMP}.csv" \
    --save_interval $SAVE_INTERVAL \
    --shuffle \
    --log "${LOGS_DIR}/${MODEL_NAME}_multilingual_${TIMESTAMP}.log" \
    --progress-file "${RESULTS_DIR}/progress_${MODEL_NAME}_multilingual.txt" \
    $RESUME_FLAG

echo ""

# Step 2: F1-F8
echo "--- Running statistical tests ---"
CSV="${RESULTS_DIR}/ed_${MODEL_NAME}_multilingual_${TIMESTAMP}.csv"
python calculate_metrics.py "$CSV" \
    --extra-csv results/ed_results_combined.csv \
    --out "${RESULTS_DIR}/FT_${MODEL_NAME}_multilingual_${TIMESTAMP}.csv"

echo ""
echo "=== Done ==="
echo "Results: ${CSV}"
echo ""
echo "Key analysis to run:"
echo "  - Compare ED across wiki_pl, wiki_zh, wiki_ar, wiki_ja domains"
echo "  - Compare with existing English wiki ED from ed_${MODEL_NAME}_prompts_*.csv"
echo "  - Kruskal-Wallis (F7) will test cross-lingual uniformity"

#!/bin/bash

# =========================================================
# run_entropic_deviation.sh
# Full pipeline: for each model × prompt set, generate logits,
# compute ED, run statistical tests, then produce combined analysis.
# =========================================================

set -euo pipefail

# GPU lockfile — prevents cron jobs from competing for GPU
LOCKFILE="/tmp/ed_experiment.lock"
touch "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# --- Configuration ----------------------------------------
TEMPS="0.7 1.0 1.3"
MAX_TOKENS=128
N_CTX=512
N_GPU_LAYERS=-1
SAVE_INTERVAL=5
MODELS_DIR="models"
RESULTS_DIR="results"
LOG_DIR="logs"

PROMPT_SETS=(
    "prompts/prompts.jsonl"
    "prompts/prompts_neutral.jsonl"
)
# ----------------------------------------------------------

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# --- Discover models --------------------------------------
MODELS=("$SCRIPT_DIR/$MODELS_DIR"/*.gguf)

if [ ${#MODELS[@]} -eq 0 ] || [ ! -f "${MODELS[0]}" ]; then
    echo "Error: No .gguf models found in $MODELS_DIR/"
    exit 1
fi

echo "====================================================="
echo "Entropic Deviation — Full Experiment"
echo "Timestamp:    $TIMESTAMP"
echo "Models found: ${#MODELS[@]}"
for m in "${MODELS[@]}"; do echo "  - $(basename "$m")"; done
echo "Prompt sets:  ${#PROMPT_SETS[@]}"
for p in "${PROMPT_SETS[@]}"; do echo "  - $p"; done
echo "Temps:        $TEMPS"
echo "Max tokens:   $MAX_TOKENS"
echo "====================================================="

ED_CSVS=()

for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_FILE=$(basename "$MODEL_PATH")
    # Derive a short label: strip .gguf and quantization suffix
    MODEL_NAME=$(echo "$MODEL_FILE" | sed 's/\.gguf$//; s/-Q[0-9].*//; s/_Q[0-9].*//')

    for PROMPTS_PATH in "${PROMPT_SETS[@]}"; do
        PROMPT_TAG=$(basename "$PROMPTS_PATH" .jsonl)

        RUN_ID="${MODEL_NAME}_${PROMPT_TAG}_${TIMESTAMP}"
        LOGITS_PREFIX="$RESULTS_DIR/logits_${RUN_ID}"
        ED_CSV="$RESULTS_DIR/ed_${RUN_ID}.csv"
        FT_CSV="$RESULTS_DIR/FT_${RUN_ID}.csv"

        echo ""
        echo "-----------------------------------------------------"
        echo "Model:   $MODEL_NAME ($MODEL_FILE)"
        echo "Prompts: $PROMPTS_PATH"
        echo "Run ID:  $RUN_ID"
        echo "-----------------------------------------------------"

        # Validate
        if [ ! -f "$SCRIPT_DIR/$PROMPTS_PATH" ]; then
            echo "Warning: $PROMPTS_PATH not found, skipping."
            continue
        fi

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
            --log "$LOG_DIR/${RUN_ID}.log"

        # Step 2: Calculate ED
        echo "[2/3] Computing Entropic Deviation..."
        python "$SCRIPT_DIR/calculate_ed.py" \
            --pattern "${LOGITS_PREFIX}_chkpt_*.pt" \
            --out "$ED_CSV" \
            --model-name "$MODEL_NAME"

        # Step 3: Statistical tests
        echo "[3/3] Running statistical tests (F1-F8)..."
        python "$SCRIPT_DIR/calculate_metrics.py" "$ED_CSV" \
            --out "$FT_CSV"

        ED_CSVS+=("$ED_CSV")

        echo "Done: $RUN_ID"
    done
done

# --- Combined analysis ------------------------------------
if [ ${#ED_CSVS[@]} -gt 1 ]; then
    echo ""
    echo "====================================================="
    echo "Combined analysis across all runs"
    echo "====================================================="

    COMBINED_CSV="$RESULTS_DIR/ed_combined_${TIMESTAMP}.csv"
    COMBINED_FT="$RESULTS_DIR/FT_combined_${TIMESTAMP}.csv"

    head -1 "${ED_CSVS[0]}" > "$COMBINED_CSV"
    for csv in "${ED_CSVS[@]}"; do
        tail -n +2 "$csv" >> "$COMBINED_CSV"
    done

    python "$SCRIPT_DIR/calculate_metrics.py" "$COMBINED_CSV" \
        --out "$COMBINED_FT"

    echo "Combined ED:  $COMBINED_CSV"
    echo "Combined FT:  $COMBINED_FT"
fi

echo ""
echo "====================================================="
echo "Experiment complete!"
echo "All results in: $RESULTS_DIR/"
echo "====================================================="

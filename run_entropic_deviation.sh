#!/bin/bash

# =========================================================
# run_entropic_deviation.sh
# Full pipeline: for each model × prompt set, generate logits
# with inline ED computation, then run statistical tests.
#
# Supports graceful stop: send SIGINT (Ctrl+C) or SIGTERM to
# cleanly finish the current step and exit. The signal is forwarded
# to the running Python subprocess so it can save its own state.
# Re-run the script with --resume to continue from where it stopped.
# =========================================================

set -uo pipefail

# --- Graceful shutdown machinery --------------------------
CHILD_PID=""
STOP_REQUESTED=0

on_signal() {
    STOP_REQUESTED=1
    echo ""
    echo "[STOP] Graceful shutdown requested — finishing current step..."
    if [ -n "$CHILD_PID" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
        kill -TERM "$CHILD_PID"
        wait "$CHILD_PID" 2>/dev/null
    fi
}

trap on_signal SIGINT SIGTERM

# Run a command in foreground, capture its PID for signal forwarding.
# Returns the command's exit code. If stop was requested, caller checks STOP_REQUESTED.
run_step() {
    "$@" &
    CHILD_PID=$!
    wait "$CHILD_PID"
    local rc=$?
    CHILD_PID=""
    return $rc
}
# ----------------------------------------------------------

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
SAVE_INTERVAL=20
MODELS_DIR="models"
RESULTS_DIR="results"
LOG_DIR="logs"
PROGRESS_FILE="results/.progress"
RESUME_FLAG=""

PROMPT_SETS=(
    "prompts/prompts.jsonl"
    "prompts/prompts_neutral.jsonl"
)

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --resume) RESUME_FLAG="--resume" ;;
    esac
done
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
if [ -n "$RESUME_FLAG" ]; then echo "Mode:         RESUME"; fi
echo "====================================================="

ED_CSVS=()

for MODEL_PATH in "${MODELS[@]}"; do
    MODEL_FILE=$(basename "$MODEL_PATH")
    # Derive a short label: strip .gguf and quantization suffix
    MODEL_NAME=$(echo "$MODEL_FILE" | sed 's/\.gguf$//; s/-Q[0-9].*//; s/_Q[0-9].*//')

    for PROMPTS_PATH in "${PROMPT_SETS[@]}"; do
        PROMPT_TAG=$(basename "$PROMPTS_PATH" .jsonl)

        RUN_ID="${MODEL_NAME}_${PROMPT_TAG}_${TIMESTAMP}"
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

        # Step 1: Generate logits + compute ED
        echo "[1/2] Generating logits + computing ED..."
        run_step python "$SCRIPT_DIR/generate_logits.py" \
            --model "$MODEL_PATH" \
            --prompts "$PROMPTS_PATH" \
            --temps $TEMPS \
            --max_tokens "$MAX_TOKENS" \
            --n_ctx "$N_CTX" \
            --n_gpu_layers "$N_GPU_LAYERS" \
            --save_interval "$SAVE_INTERVAL" \
            --ed-out "$ED_CSV" \
            --model-name "$MODEL_NAME" \
            --log "$LOG_DIR/${RUN_ID}.log" \
            --progress-file "$PROGRESS_FILE" \
            $RESUME_FLAG

        if [ "$STOP_REQUESTED" -eq 1 ]; then
            echo "[STOP] Stopped after logits generation. Re-run with --resume to continue."
            exit 0
        fi

        # Step 2: Statistical tests
        echo "[2/2] Running statistical tests (F1-F8)..."
        run_step python "$SCRIPT_DIR/calculate_metrics.py" "$ED_CSV" \
            --out "$FT_CSV" \
            --log "$LOG_DIR/${RUN_ID}.log"

        if [ "$STOP_REQUESTED" -eq 1 ]; then
            echo "[STOP] Stopped after statistical tests. Re-run with --resume to continue."
            exit 0
        fi

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

    run_step python "$SCRIPT_DIR/calculate_metrics.py" "$COMBINED_CSV" \
        --out "$COMBINED_FT"

    if [ "$STOP_REQUESTED" -eq 1 ]; then
        echo "[STOP] Stopped during combined analysis."
        exit 0
    fi

    echo "Combined ED:  $COMBINED_CSV"
    echo "Combined FT:  $COMBINED_FT"
fi

echo ""
echo "====================================================="
echo "Experiment complete!"
echo "All results in: $RESULTS_DIR/"
echo "====================================================="

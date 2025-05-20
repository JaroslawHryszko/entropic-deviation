#!/bin/bash

# =========================================================
# run_entropic_deviation.sh
# Script to run the memory-optimized Entropic Deviation experiment
# =========================================================

# Directory setup
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
MODEL_PATH="models/meta-llama-3-8b-instruct.Q4_K_M.gguf"
PROMPTS_PATH="prompts/prompts.jsonl"
RESULTS_DIR="results"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create necessary directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"

# Check if the Python script exists
if [ ! -f "$SCRIPT_DIR/ed_experiment/generate_multi_gpu_lowmem.py" ]; then
    echo "Error: generate_multi_gpu_lowmem.py not found!"
    echo "Please save the memory-optimized script first."
    exit 1
fi

# Check if the model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    echo "Please download the model first or update the path in this script."
    exit 1
fi

# Check if the prompts file exists
if [ ! -f "$PROMPTS_PATH" ]; then
    echo "Error: Prompts file not found at $PROMPTS_PATH"
    exit 1
fi

echo "====================================================="
echo "Starting Entropic Deviation Experiment (Low Memory Mode)"
echo "Model: $MODEL_PATH"
echo "Prompts: $PROMPTS_PATH"
echo "Timestamp: $TIMESTAMP"
echo "====================================================="

# Run the memory-optimized experiment
python "$SCRIPT_DIR/ed_experiment/generate_multi_gpu_lowmem.py" \
    --model "$MODEL_PATH" \
    --prompts "$PROMPTS_PATH" \
    --max_tokens 64 \
    --n_ctx 512 \
    --n_batch 32 \
    --micro_batch 2 \
    --sleep_time 2.0 \
    --n_gpu_layers 32 \
    --save_interval 5 \
    --out "$RESULTS_DIR/logits_$TIMESTAMP" \
    --log "$LOG_DIR/experiment_$TIMESTAMP.log"

# Check if the experiment completed successfully
if [ $? -eq 0 ]; then
    echo "Experiment completed successfully!"
    echo "Results saved to: $RESULTS_DIR/logits_$TIMESTAMP"
    echo "Log saved to: $LOG_DIR/experiment_$TIMESTAMP.log"
    
    # Run the ED calculation on the results
    echo "Computing Entropic Deviation metrics..."
    python "$SCRIPT_DIR/ed_experiment/ed.py" \
        --logits "$RESULTS_DIR/logits_${TIMESTAMP}_combined.pt" \
        --out "$RESULTS_DIR/ed_results_$TIMESTAMP.csv"
    
    # Run the statistical tests
    echo "Running statistical tests..."
    python "$SCRIPT_DIR/ed_experiment/stats.py" "$RESULTS_DIR/ed_results_$TIMESTAMP.csv" \
        --out "$RESULTS_DIR/results_Ftests_$TIMESTAMP.csv"
    
    echo "====================================================="
    echo "Entropic Deviation Experiment Pipeline Complete!"
    echo "Check the results in the $RESULTS_DIR directory."
    echo "====================================================="
else
    echo "ERROR: Experiment failed or was interrupted."
    echo "Check the log file for details: $LOG_DIR/experiment_$TIMESTAMP.log"
    echo "You may want to resume the experiment with --start_idx parameter."
fi
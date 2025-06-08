# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Entropic Deviation** research framework that investigates "proto-agency" in Large Language Models by measuring how much a model's token probability distribution deviates from uniform distribution. The project implements the methodology from "Emergence of Proto-Agency via Entropic Deviation in High-Scale LLMs".

## Key Commands

### Data Generation
```bash
# Generate logits with multi-GPU support
python generate_logits.py --model models/MODEL.gguf --prompts prompts/prompts.jsonl --temps 0.7 1.0 1.3 --max_tokens 128 --out OUTPUT_DIR

# Resume interrupted generation
python generate_logits.py --resume --checkpoint_dir CHECKPOINT_DIR
```

### Analysis Pipeline
```bash
# Calculate entropic deviation from checkpoint files
python calculate_ed.py --pattern "logits_results_gpu*_chkpt_*.pt" --out ed_results.csv --model-name "ModelName"

# Run statistical falsification tests (F1-F8)
python calculate_metrics.py INPUT.csv --out OUTPUT.csv

# Merge checkpoint files for processing
python merge_checkpoints.py INPUT_PATTERN OUTPUT_FILE
```

### Orchestration Scripts
```bash
# Run complete ED experiment pipeline
./run_entropic_deviation.sh

# Run Mistral-specific experiments
./mistral.sh
```

## Architecture

### Core Pipeline
1. **`generate_logits.py`**: Multi-GPU inference engine that processes prompts and generates logit checkpoints
2. **`calculate_ed.py`**: Streaming processor that computes entropic deviation from checkpoint files using KL divergence
3. **`calculate_metrics.py`**: Statistical analysis engine implementing 8 falsification tests (F1-F8)

### Key Features
- **Streaming processing**: Handles large datasets via checkpointing without loading everything into memory
- **Multi-GPU support**: Dynamic load balancing across available GPUs with memory monitoring
- **Resume capability**: Can restart from interruptions using checkpoint files
- **Model support**: Llama-3-8B, Phi-3-mini-4K, Mistral-7B (all GGUF Q4_K_M quantized)

### Data Flow
```
prompts.jsonl → generate_logits.py → checkpoint files (.pt) → calculate_ed.py → ed_results.csv → calculate_metrics.py → FTresults.csv
```

### Experimental Parameters
- **Temperature testing**: 0.7, 1.0, 1.3 to analyze model behavior variation
- **Domain diversity**: 800 prompts across Wikipedia (400), News (200), Fiction (120), Code (80)
- **Checkpointing**: Every 5 generations with garbage collection every 20
- **Token generation**: 128 max tokens per prompt

### File Patterns
- **Checkpoints**: `logits_gpu{id}_chkpt_{n}.pt`
- **Results**: `ed_results_{model}.csv`, `FTresults_{model}.csv`
- **Models**: GGUF format in `models/` directory
- **Prompts**: JSONL format with domain tags

### Dependencies
Critical packages: `torch`, `llama-cpp-python`, `pandas`, `scipy`, `statsmodels`, `psutil`

## Development Notes

- The project uses streaming and checkpointing extensively for memory efficiency
- All statistical tests achieve extremely high significance (p < 10⁻¹⁰⁰)
- Model size mapping in `calculate_ed.py` enables cross-architecture analysis
- GPU memory management is critical - monitor with `nvidia-smi` during long runs
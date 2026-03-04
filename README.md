# Entropic Deviation

![License](https://img.shields.io/badge/license-MIT-blue.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18732011.svg)](https://zenodo.org/records/18732011)

A research framework for studying **randomness and non-randomness** in Large Language Models by measuring Entropic Deviation (ED) — the degree to which a model's token probability distribution deviates from uniform.

## What is Entropic Deviation?

ED is defined as:

```
ED_t = KL(p || uniform) / log(|V|)
```

where `p = softmax(logits)` and `|V|` is the vocabulary size. ED = 0 means uniform distribution (maximum randomness), ED = 1 means concentration on a single token (no randomness).

## Pipeline Overview

```
prompts.jsonl → generate_logits.py → ed_results.csv → calculate_metrics.py → FTresults.csv
```

## Step-by-Step Guide

### 1. Clone and set up

```bash
git clone https://github.com/JaroslawHryszko/entropic-deviation.git
cd entropic-deviation

conda create -n text python=3.12 && conda activate text
pip install -r requirements.txt
pip install llama-cpp-python  # requires CUDA toolkit
```

### 2. Download a model

Place GGUF-quantized models in the `models/` directory. Example using `huggingface-cli`:

```bash
mkdir -p models

# Qwen-2.5-32B (~20 GB)
huggingface-cli download bartowski/Qwen2.5-32B-Instruct-GGUF \
    --include "Qwen2.5-32B-Instruct-Q4_K_M.gguf" --local-dir models/

# Gemma-2-27B (~17 GB)
huggingface-cli download bartowski/gemma-2-27b-it-GGUF \
    --include "gemma-2-27b-it-Q4_K_M.gguf" --local-dir models/

# Llama-3.3-70B (~43 GB, requires 2 GPUs)
huggingface-cli download bartowski/Llama-3.3-70B-Instruct-GGUF \
    --include "Llama-3.3-70B-Instruct-Q4_K_M.gguf" --local-dir models/
```

### 3. Generate logits and compute ED

The inference engine auto-detects GPUs, spreads the model across all available devices, and computes ED inline — no intermediate checkpoint files needed.

```bash
python generate_logits.py \
    --model models/Qwen2.5-32B-Instruct-Q4_K_M.gguf \
    --prompts prompts/prompts.jsonl \
    --temps 0.7 1.0 1.3 \
    --max_tokens 128 \
    --ed-out results/ed_results_qwen32b.csv \
    --model-name "Qwen-2.5-32B" \
    --save_interval 20 \
    --log logs/qwen32b.log
```

Key options:
- `--temps` — temperature values to test (default: 0.7 1.0 1.3)
- `--max_tokens` — tokens to generate per prompt (default: 128)
- `--ed-out` — output CSV for ED results (default: `{out}_ed.csv`)
- `--model-name` — model label for CSV (auto-derived from filename if not given)
- `--save_interval` — flush CSV every N generations (default: 20)
- `--save-logits` — also save `.pt` checkpoint files with full logit tensors
- `--resume` — resume from the last saved position in ED CSV
- `--n_gpu_layers` — layers to offload to GPU (-1 = all, default)

The script handles SIGINT/SIGTERM gracefully, flushing buffered ED records to CSV before exit.

Output CSV columns: `prompt, temp, seq_len, gen_time, timestamp, ED_mean, ED_std, model, model_size, domain`.

### 4. Run statistical tests (F1-F8)

```bash
python calculate_metrics.py results/ed_results_qwen32b.csv \
    --out results/FTresults_qwen32b.csv
```

### 5. Repeat for each model

Run steps 3-4 for each model in your experiment. To analyze all models together, concatenate ED results:

```bash
# Merge ED CSVs (skip headers from subsequent files)
head -1 results/ed_results_qwen32b.csv > results/ed_results_combined.csv
tail -n +2 -q results/ed_results_*.csv >> results/ed_results_combined.csv

# Run combined analysis
python calculate_metrics.py results/ed_results_combined.csv \
    --out results/FTresults_combined.csv
```

### Alternative: run the full pipeline

The shell wrapper runs all steps for every model found in `models/`:

```bash
./run_entropic_deviation.sh
```

Edit the configuration variables at the top of the script to customize prompts, temperatures, and output directories.

## Prompt Sets

The framework includes two prompt sets designed to test different hypotheses:

| File | Prompts | Domains | Purpose |
|------|---------|---------|---------|
| `prompts/prompts.jsonl` | 800 | wiki, news, fiction, code | Domain prompts with semantic context |
| `prompts/prompts_neutral.jsonl` | 1000 | empty, random, explicit, neutral, nonsense | Neutral prompts with minimal semantic constraint |

The neutral prompts are designed to disentangle two hypotheses:
- **H1**: Non-randomness is intrinsic to the model's learned representations
- **H2**: Non-randomness is induced by the semantic constraints of the input prompts

To rebuild prompts from source:

```bash
python prompts/build_prompts_en.py       # domain prompts (requires datasets library)
python prompts/build_neutral_prompts.py   # neutral prompts (no external dependencies)
```

## Statistical Tests (F1-F8)

| Test | Null Hypothesis | Method |
|------|----------------|--------|
| F1 | Mean ED = 0 | One-sample t-test |
| F2 | No temperature effect | One-way ANOVA + Tukey HSD |
| F3 | No model size effect | OLS regression |
| F4 | ED independent of temperature | Pearson correlation |
| F5 | No autoregressive persistence | AR(1) coefficient |
| F6 | ED independent of sequence length | Pearson correlation |
| F7 | Uniform ED across domains | Kruskal-Wallis |
| F8 | ED independent of generation rank | OLS slope |

## Utility Scripts

- **`calculate_ed.py`** — standalone checkpoint-to-CSV processor (for reprocessing `.pt` files saved with `--save-logits`):
  ```bash
  python calculate_ed.py --pattern "results/logits_*_chkpt_*.pt" --out ed.csv --model-name "ModelName"
  ```
- **`merge_checkpoints.py`** — merge multiple `.pt` checkpoint files into one:
  ```bash
  python merge_checkpoints.py --pattern "results/logits_*_chkpt_*.pt" --output merged.pt
  ```

## Testing

```bash
conda activate text
python -m pytest tests/
```

## Project Structure

```
entropic-deviation/
├── generate_logits.py          # Step 1: multi-GPU inference + inline ED
├── calculate_ed.py             # Standalone: compute ED from .pt checkpoints
├── calculate_metrics.py        # Step 2: statistical tests F1-F8
├── merge_checkpoints.py        # Utility: merge checkpoint files
├── run_entropic_deviation.sh   # Full pipeline wrapper
├── requirements.txt
├── prompts/
│   ├── prompts.jsonl           # 800 domain prompts
│   ├── prompts_neutral.jsonl   # 1000 neutral prompts
│   ├── build_prompts_en.py     # Domain prompt generator
│   └── build_neutral_prompts.py # Neutral prompt generator
├── tests/                      # Test suite (pytest)
├── results/                    # Experiment output (CSV)
└── models/                     # GGUF models (gitignored)
```

## Requirements

- Python 3.10+ (conda recommended)
- PyTorch 2.x (with CUDA support)
- NVIDIA GPU (Pascal architecture or newer)
- `llama-cpp-python` (not in `requirements.txt` — install separately with CUDA support)

## Citation

```bibtex
@article{hryszko2025ed,
  title={Entropic Deviation as a Measure of Non-Randomness in Large Language Models},
  author={Jaros{\l}aw Hryszko},
  journal={Zenodo},
  doi={10.5281/zenodo.18732011},
  url={https://zenodo.org/records/18732011},
  year={2025}
}
```

## License

MIT License — see [LICENSE](LICENSE).

## Contact

- **Author**: Jaroslaw Hryszko
- **Institution**: Institute of Computer Science, Jagiellonian University, Krakow, Poland
- **Email**: jaroslaw.hryszko@uj.edu.pl
- **ORCID**: [0000-0002-4207-1080](https://orcid.org/0000-0002-4207-1080)

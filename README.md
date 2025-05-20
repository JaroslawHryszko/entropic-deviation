# Entropic Deviation

![License](https://img.shields.io/badge/license-MIT-blue.svg)

A research framework for investigating proto-agency in Large Language Models through entropic deviation (ED) analysis. This project implements the methodology described in *"Entropic Deviation Reveals Proto-Agency in Large Language Models"*.

## Overview

Entropic Deviation (ED) measures how much a model's token probability distribution deviates from a uniform distribution. This metric may indicate "proto-agency" or non-random behavior in language models. This framework enables:

1. Generating responses from LLMs with various temperatures
2. Collecting logits for each token
3. Computing entropy deviation metrics
4. Running statistical tests to analyze the results

## Key Features

- Multi-GPU support for parallel processing
- Temperature variation (0.7, 1.0, 1.3) to test model behavior
- Domain-diverse prompts (Wikipedia, News, Fiction, Code)
- Statistical significance tests (F1-F8)
- Simple CLI interface for running experiments

## Project Structure

| Directory/File | Description |
|----------------|-------------|
| `ed_experiment/ed.py` | Core implementation for computing entropic deviation |
| `ed_experiment/generate3.py` | Single-GPU generation with logit collection |
| `ed_experiment/generate_multi_gpu.py` | Multi-GPU parallel processing |
| `ed_experiment/build_prompts.py` | Creates 800 diverse prompts for testing |
| `ed_experiment/stats.py` | Eight falsification tests (F1-F8) |
| `ed_experiment/cli_run/` | CLI tools for experiment automation |
| `prompts/prompts.jsonl` | 800 pre-built prompts across domains |
| `models/` | GGUF-quantized model checkpoints (not committed) |
| `results/` | Output directory for experiment results |

## Requirements

- Python 3.x
- PyTorch 2.6.0 (with CUDA support)
- NVIDIA GPU with 24GB+ VRAM (RTX 3090 or better recommended)
- Dependencies listed in `requirements.txt`:
  - llama_cpp_python (for GGUF model loading)
  - numpy, pandas, scipy
  - statsmodels (for AR(1) & OLS analysis)
  - datasets, nltk, tqdm

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourhandle/entropic-deviation.git
cd entropic-deviation

# Set up environment
python -m venv edenv && source edenv/bin/activate
pip install -r ed_experiment/requirements.txt

# Download model (if needed)
bash scripts/get_model.sh

# Single-GPU workflow
# 1. Generate completions and collect logits
python ed_experiment/generate3.py \
      --model models/llama-3-8b-instruct.Q4_K_M.gguf \
      --prompts prompts/prompts.jsonl \
      --out results/logits.pt

# 2. Compute entropic deviation
python ed_experiment/ed.py \
      --logits results/logits.pt \
      --out results/ed_results.csv

# 3. Run statistical tests
python ed_experiment/stats.py results/ed_results.csv \
      --out results/results_Ftests.csv

# Multi-GPU workflow
python ed_experiment/generate_multi_gpu.py \
      --model models/llama-3-8b-instruct.Q4_K_M.gguf \
      --prompts prompts/prompts.jsonl \
      --temps 0.7 1.0 1.3 \
      --max_tokens 100 \
      --out_file results/logits_multi_gpu.pt \
      --gpu_split_ratio 0.5
```

## Statistical Tests (F1-F8)

The project runs eight falsification tests on the collected data:

1. **F1**: Mean ED != 0 (one-sample t-test)
2. **F2**: Temperature effect (one-way ANOVA)
3. **F3**: Model size slope effect
4. **F4**: Correlation between ED and temperature
5. **F5**: AR(1) coefficient test
6. **F6**: Correlation between ED and sequence length
7. **F7**: Domain uniformity test
8. **F8**: Rank-independence test

## CLI Tools

The `cli_run` directory contains tools for running experiments through command-line interfaces:

- `generate_cli.py`: Invokes llama-cli.exe with the --logits-all option
- `parse_logits.py`: Processes log files into PyTorch format

## Building Custom Prompts

Use `build_prompts.py` to create domain-diverse prompt sets:
- Wikipedia articles (400 prompts)
- News articles (200 prompts)
- Fiction (120 prompts)
- Code snippets (80 prompts)

```bash
python ed_experiment/build_prompts.py \
      --output prompts/custom_prompts.jsonl \
      --count 800
```

## Extending the Framework

- Add new model checkpoints in the `models/` directory
- Create custom prompt sets with `build_prompts.py`
- Implement additional statistical tests in `stats.py`

## Recent Updates

- Multi-GPU support for parallel processing
- Fixed index bug in prompt handling
- Updated build_prompts.py for improved prompt quality

## Citation

If you use this code in your research, please cite:

```bibtex
@article{hryszko2025ed,
  title={Entropic Deviation Reveals Proto-Agency in Large Language Models},
  author={JaroslawHryszko},
  journal={ArXiv (let's hope)},
  year={2025}
}
```

## License

© 2025 Jarosław Hryszko — MIT License
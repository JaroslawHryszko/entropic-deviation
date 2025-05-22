# Entropic Deviation

![License](https://img.shields.io/badge/license-MIT-blue.svg)

A research framework for investigating proto-agency in Large Language Models through entropic deviation (ED) analysis. This project implements the methodology described in *"Emergence of Proto-Agency via Entropic Deviation in High-Scale LLMs"*.

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
| `calculate_ed.py` | Core implementation for computing entropic deviation |
| `generate_logits.py` | Multi-GPU generation with logit collection |
| `calculate_metrics.py` | Eight falsification tests (F1-F8) |
| `prompts/prompts.jsonl` | 800 pre-built prompts across domains |
| `requirements.txt` | Python dependencies |
| `results/` | Experiment results |
| `prompts/` | Prompts - see below |

## Requirements

- Python 3.x
- PyTorch 2.6.0 (with CUDA support)
- NVIDIA GPU (min. Pascal architecture)
- Dependencies listed in `requirements.txt`:
  - llama_cpp_python (for GGUF model loading)
  - numpy, pandas, scipy
  - statsmodels (for AR(1) & OLS analysis)
  - datasets, nltk, tqdm

## Quick Start

```bash
# Clone repository
git clone https://github.com/JaroslawHryszko/entropic-deviation.git
cd entropic-deviation

# Set up environment
python -m venv edenv && source edenv/bin/activate
pip install -r requirements.txt

# Download model (if needed)
# Models should be placed in models/ directory

# Multi-GPU workflow
python generate_logits.py \
      --model models/llama-3-8b-instruct.Q4_K_M.gguf \
      --prompts prompts/prompts.jsonl \
      --temps 0.7 1.0 1.3 \
      --max_tokens 128 \
      --out logits_results

# Compute entropic deviation
python calculate_ed.py \
      --pattern "logits_results_gpu*_chkpt_*.pt" \
      --out ed_results.csv \
      --model-name "Llama-3-8B"

# Run statistical tests
python calculate_metrics.py ed_results.csv \
      --out FTresults_llama3.csv
```

## Statistical Tests (F1-F8)

The project runs eight falsification tests on the collected data:

1. **F1**: Mean ED ≠ 0 (one-sample t-test)
2. **F2**: Temperature effect (one-way ANOVA)
3. **F3**: Model size slope effect
4. **F4**: Correlation between ED and temperature
5. **F5**: AR(1) coefficient test
6. **F6**: Correlation between ED and sequence length
7. **F7**: Domain uniformity test
8. **F8**: Rank-independence test

## Experimental Results

My multi-architecture experiment across three models yielded compelling evidence for structured behavioral patterns:

- **Models tested**: Llama-3-8B, Phi-3-mini-4K, Mistral-7B
- **Total samples**: 7,200 (800 prompts × 3 temperatures × 3 models)
- **Key findings**: 6/8 falsification tests achieved astronomical significance (p < 10⁻¹⁰⁰)

### Key Results Summary

| Test | Llama-3-8B | Phi-3-mini | Mistral-7B | Combined |
|------|------------|------------|------------|----------|
| F1 (Mean ≠ 0) | < 10⁻¹⁵ | < 10⁻¹⁵ | < 10⁻¹⁵ | < 10⁻¹⁵ |
| F2 (Temp effect) | 5.26×10⁻²⁷ | 2.11×10⁻¹⁴ | 1.73×10⁻⁷² | 5.55×10⁻⁶⁶ |
| F5 (AR(1)) | 1.27×10⁻¹⁴³ | 1.97×10⁻¹⁴⁵ | 1.51×10⁻³⁸ | 3.50×10⁻²⁰⁰ |

## Building Custom Prompts

Use the domain-balanced prompt construction:
- Wikipedia articles (400 prompts)
- News articles (200 prompts)
- Fiction (120 prompts)
- Code snippets (80 prompts)

The prompts are pre-built in `prompts/prompts.jsonl` with domain tags and length normalization.

## Model Support

Tested architectures:
- **Meta-Llama-3-8B-Instruct** (GGUF Q4_K_M)
- **Phi-3-mini-4K-Instruct** (GGUF Q4_K_M)
- **Mistral-7B-Instruct-v0.1** (GGUF Q4_K_M)

All models use 4-bit quantization for computational efficiency while preserving representational capacity.

## Hardware Requirements

- **Recommended**: 2× NVIDIA RTX 3090 (24GB VRAM each)
- **Minimum**: 1× GPU with 16GB+ VRAM
- **RAM**: 32GB+ system memory
- **Storage**: NVMe SSD for model caching

## Extending the Framework

- Add new model checkpoints in the `models/` directory
- Modify domain distributions in prompt construction
- Implement additional statistical tests in `calculate_metrics.py`
- Extend behavioral-drift probes (see paper Appendix A)

## Recent Updates

- Multi-GPU support for parallel processing
- Cross-architecture validation framework
- Comprehensive falsification test battery
- Enhanced statistical power analysis

## Citation

If you use this code in your research, please cite:

```bibtex
@article{hryszko2025ed,
  title={Emergence of Proto-Agency via Entropic Deviation in High-Scale LLMs},
  author={Jarosław Hryszko},
  journal={arXiv preprint},
  year={2025}
}
```

## Contributing

Contributions are welcome! Please see our guidelines for:
- Adding new model architectures
- Implementing additional behavioral probes
- Extending statistical test battery
- Improving computational efficiency

## License

© 2025 Jarosław Hryszko — MIT License

## Contact

- **Author**: Jarosław Hryszko
- **Institution**: Institute of Computer Science, Jagiellonian University, Kraków, Poland
- **Email**: jaroslaw.hryszko@uj.edu.pl
- **ORCID**: [0000-0002-4207-1080](https://orcid.org/0000-0002-4207-1080)

---

**Note**: This research investigates emergent behavioral patterns in AI systems. The findings have implications for AI safety and should be considered in the context of responsible AI development.
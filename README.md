# Entropic-Deviation Pilot

This repository contains the **exact code, prompts, and instructions** used
to reproduce the pilot experiment reported in  
*“Entropic Deviation Reveals Proto-Agency in Large Language Models”*.

| Folder | Contents |
|--------|----------|
| `models/` | GGUF-quantised checkpoints (4-bit) — **not* committed; download script provided |
| `prompts/` | `prompts.jsonl` (800 lines × diverse domains) |
| `ed_experiment/` | `generate.py`, `ed.py`, `stats.py`, `requirements.txt` |
| `results/` | Sample CSV outputs (`ed_results.csv`, `results_Ftests.csv`) |
| `notebooks/` | Optional Jupyter notebook for quick ED visualisation |

---

## Quick start (RTX 3090 / 24 GB)

```bash
git clone https://github.com/yourhandle/entropic-deviation.git
cd entropic-deviation

python -m venv edenv && source edenv/bin/activate
pip install -r ed_experiment/requirements.txt

# download 4-bit Llama-3 checkpoint (~4 GB)
bash scripts/get_model.sh        # or manual wget (see script)

# 1) sample logits
python ed_experiment/generate.py \
      --model models/llama-3-8b-instruct.Q4_K_M.gguf \
      --prompts prompts/prompts.jsonl \
      --out results/logits.pt

# 2) compute per-token ED and sequence means
python ed_experiment/ed.py --logits results/logits.pt \
                           --out    results/ed_results.csv

# 3) run the eight falsification tests (Table F1–F8)
python ed_experiment/stats.py results/ed_results.csv \
                              --out results/results_Ftests.csv
````

The CSV files match the values reported in **Table 1** of the paper.
If you add extra checkpoints or temperatures, the code automatically
extends the analysis.

---

## File glossary

| File                           | Purpose                                                       |
| ------------------------------ | ------------------------------------------------------------- |
| `generate.py`                  | streams completions with *llama-cpp-python*, dumps raw logits |
| `ed.py`                        | converts logits → per-token ED → per-sequence mean            |
| `stats.py`                     | implements the eight pre-registered tests (F1–F8)             |
| `scripts/get_model.sh`         | convenience downloader for 8-B Q4 K M checkpoint              |
| `notebooks/ed_visualise.ipynb` | optional histogram & correlation plots                        |

---

## Re-running the Seed-Sweep probe (BP1)

A single command re-uses the same logits to compute
the Kolmogorov–Smirnov statistic described in Section 5.2:

```bash
python ed_experiment/seed_sweep.py --logits results/logits.pt
```

---

## Citation

If you use this code, please cite:

```
@article{hryszko2025ed,
  title={Entropic Deviation Reveals Proto-Agency in Large Language Models},
  author={JaroslawHryszko},
  journal={ArXiv (let's hope)},
  year={2025}
}
```

---

© 2025 Jarosław Hryszko — MIT License


# Entropic-Deviation Pilot 🎲📊

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
git clone https://github.com/JaroslawHryszko/entropic-deviation.git
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

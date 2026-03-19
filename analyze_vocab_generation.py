#!/usr/bin/env python3
"""
Generation-side vocabulary analysis for the multilingual ED experiment.

Runs short inference pass (50 prompts x 5 languages x temp=1.0) to measure:
  - vocab_used: unique token IDs in model output per language
  - vocab_concentration: mean probability mass in top-K tokens per position
  - fertility: tokens per character in prompts

Complements analyze_vocab_multilingual.py (prompt-side / tokenizer partition).
"""

import gc
import logging
import os
import random
import signal
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = "models/Qwen2.5-32B-Instruct-Q4_K_M.gguf"
MULTILINGUAL_CSV = "results/ed_Qwen2.5-32B-Instruct_multilingual_20260315_212747.csv"
ENGLISH_CSV = "results/ed_Qwen2.5-32B-Instruct_prompts_20260304_235106.csv"
OUTPUT_CSV = "results/vocab_generation_multilingual.csv"
PROGRESS_FILE = "results/.progress"
TOKENIZER_NAME = "Qwen/Qwen2.5-32B-Instruct"

LANGUAGES = ["en", "pl", "ja", "zh", "ar"]
ED_MEANS = {"en": 0.329, "ja": 0.388, "zh": 0.395, "pl": 0.401, "ar": 0.408}

N_PROMPTS = 50
TEMPERATURE = 1.0
MAX_TOKENS = 128
N_CTX = 512
SEED = 42
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "vocab_generation.log")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger():
    """Configure console + file logging."""
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger("vocab_gen")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


log = setup_logger()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_prefix(prompt, lang):
    """Remove domain prefix(es) from prompt text."""
    for pfx in [f"wiki_{lang}: wiki_{lang}: ", f"wiki_{lang}: ",
                "wiki: wiki: ", "wiki: "]:
        if prompt.startswith(pfx):
            return prompt[len(pfx):]
    return prompt


def detect_gpus():
    try:
        import subprocess
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, check=True)
        return len(r.stdout.strip().split('\n'))
    except Exception:
        return 0


def write_progress(processed, total, t_start, lang=None):
    """Write progress file for gpu_monitor (atomic overwrite)."""
    pct = processed / total * 100 if total else 0
    elapsed = time.time() - t_start
    if processed > 0:
        eta_s = elapsed / processed * (total - processed)
        eta_h, eta_rem = divmod(int(eta_s), 3600)
        eta_m, eta_sec = divmod(eta_rem, 60)
        eta_str = f"{eta_h}h {eta_m:02d}m {eta_sec:02d}s"
    else:
        eta_str = "-"
    elapsed_h, elapsed_rem = divmod(int(elapsed), 3600)
    elapsed_m, elapsed_sec = divmod(elapsed_rem, 60)
    elapsed_str = f"{elapsed_h}h {elapsed_m:02d}m {elapsed_sec:02d}s"

    lines = [
        f"model:    Qwen2.5-32B-Instruct (vocab analysis)",
        f"progress: {processed}/{total}  ({pct:.1f}%)",
    ]
    if lang:
        lines.append(f"language: {lang}")
    lines += [
        f"elapsed:  {elapsed_str}",
        f"ETA:      {eta_str}",
        f"updated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, 'w') as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, PROGRESS_FILE)


def sample_prompts(n_per_lang):
    """Sample unique prompts per language from existing ED results."""
    rng = random.Random(SEED)
    df_m = pd.read_csv(MULTILINGUAL_CSV)
    df_m["language"] = df_m["domain"].str.replace("wiki_", "")
    df_e = pd.read_csv(ENGLISH_CSV)
    df_e_wiki = df_e[df_e["domain"] == "wiki"]

    out = {}
    for lang in LANGUAGES:
        df_lang = df_e_wiki if lang == "en" else df_m[df_m["language"] == lang]
        pool = df_lang["prompt"].unique().tolist()
        out[lang] = rng.sample(pool, min(n_per_lang, len(pool)))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    log.info("=== Vocab Generation Analysis ===")
    log.info(f"Config: {N_PROMPTS} prompts/lang x {len(LANGUAGES)} langs x temp={TEMPERATURE}")
    log.info(f"Max tokens: {MAX_TOKENS}, n_ctx: {N_CTX}, seed: {SEED}")
    log.info(f"Model: {MODEL_PATH}")
    log.info(f"Output: {OUTPUT_CSV}")
    log.info(f"Log: {LOG_FILE}")

    log.info("Loading HuggingFace tokenizer...")
    hf_tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    log.info(f"Tokenizer loaded (vocab size: {len(hf_tok):,})")

    prompts_by_lang = sample_prompts(N_PROMPTS)
    total_gens = sum(len(v) for v in prompts_by_lang.values())
    for lang, ps in prompts_by_lang.items():
        log.info(f"  {lang}: {len(ps)} prompts sampled")

    # Load model
    from llama_cpp import Llama
    n_gpus = detect_gpus()
    log.info(f"Loading model on {n_gpus} GPU(s)...")
    llm_kwargs = dict(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,
        n_ctx=N_CTX,
        logits_all=True,
        verbose=False,
    )
    if n_gpus > 1:
        llm_kwargs["tensor_split"] = [1.0 / n_gpus] * n_gpus
    llm = Llama(**llm_kwargs)
    log.info("Model loaded.")

    # --- State for SIGINT handler ---
    results = []
    gen_done = 0
    t_all = time.time()

    def save_and_exit(signum, frame):
        log.warning(f"Signal {signum} received, saving partial results...")
        if results:
            df_partial = pd.DataFrame(results)
            df_partial.to_csv(OUTPUT_CSV, index=False)
            log.info(f"Partial results ({len(results)} languages) saved to {OUTPUT_CSV}")
        write_progress(gen_done, total_gens, t_all, lang="INTERRUPTED")
        sys.exit(0)

    signal.signal(signal.SIGINT, save_and_exit)
    signal.signal(signal.SIGTERM, save_and_exit)

    # --- Generation loop ---
    write_progress(0, total_gens, t_all)

    for lang in LANGUAGES:
        prompts = prompts_by_lang[lang]
        sampled_ids_all = set()
        argmax_ids_all = set()
        topk_accum = {100: [], 1000: [], 5000: []}
        fertilities = []
        prompt_tok_counts = []
        gen_tok_counts = []
        skipped = 0
        t_lang = time.time()

        log.info(f"--- {lang.upper()} ({len(prompts)} prompts) ---")

        for i, prompt in enumerate(prompts):
            t_gen = time.perf_counter()
            try:
                llm.reset()
                prompt_n = len(llm.tokenize(prompt.encode()))

                resp = llm(prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
                gen_text = resp["choices"][0]["text"]
                t_infer = time.perf_counter()

                # Sampled token IDs (HF tokenizer on generated text)
                sampled_ids = hf_tok.encode(gen_text, add_special_tokens=False)
                sampled_ids_all.update(sampled_ids)
                gen_tok_counts.append(len(sampled_ids))

                # Logits -> argmax + top-K concentration
                if hasattr(llm, 'scores') and llm.scores is not None:
                    arr = np.array(llm.scores, dtype=np.float32)
                    if arr.shape[0] > prompt_n:
                        arr = arr[prompt_n:]
                    gen_n = arr.shape[0]

                    argmax_ids_all.update(arr.argmax(axis=-1).tolist())

                    logits_t = torch.from_numpy(arr).float()
                    probs = torch.softmax(logits_t, dim=-1)
                    topk_vals, _ = torch.topk(probs, 5000, dim=-1, sorted=True)

                    for k in (100, 1000, 5000):
                        mass = topk_vals[:, :k].sum(dim=-1).mean().item()
                        topk_accum[k].append(mass)

                    del logits_t, probs, topk_vals, arr
                else:
                    gen_n = 0
                    log.warning(f"  [{lang.upper()} {i+1}] No scores available")

                # Prompt fertility
                clean = strip_prefix(prompt, lang)
                if len(clean) > 0:
                    ptids = hf_tok.encode(clean, add_special_tokens=False)
                    fertilities.append(len(ptids) / len(clean))
                    prompt_tok_counts.append(len(ptids))

                t_done_gen = time.perf_counter()

                log.info(
                    f"  [{lang.upper()} {i+1}/{len(prompts)}] "
                    f"gen: {gen_n} tok in {t_infer - t_gen:.1f}s | "
                    f"total: {t_done_gen - t_gen:.1f}s | "
                    f"vocab: {len(sampled_ids_all):,}"
                )

            except Exception as e:
                log.error(f"  [{lang.upper()} {i+1}/{len(prompts)}] FAILED: {e}")
                skipped += 1
                continue

            gen_done += 1
            gc.collect()

            if (i + 1) % 10 == 0:
                write_progress(gen_done, total_gens, t_all, lang=lang.upper())

        if skipped:
            log.warning(f"  {lang.upper()}: {skipped}/{len(prompts)} generations skipped")

        if not topk_accum[100]:
            log.error(f"  {lang.upper()}: no successful generations, skipping")
            continue

        row = {
            "language": lang,
            "ED_mean": ED_MEANS[lang],
            "vocab_used": len(sampled_ids_all),
            "vocab_used_argmax": len(argmax_ids_all),
            "vocab_concentration_top100": round(np.mean(topk_accum[100]), 4),
            "vocab_concentration_top1000": round(np.mean(topk_accum[1000]), 4),
            "vocab_concentration_top5000": round(np.mean(topk_accum[5000]), 4),
            "mean_fertility": round(np.mean(fertilities), 4),
            "mean_prompt_tokens": round(np.mean(prompt_tok_counts), 1),
            "mean_gen_tokens": round(np.mean(gen_tok_counts), 1),
            "total_gen_tokens": sum(gen_tok_counts),
            "n_prompts": len(prompts) - skipped,
        }
        results.append(row)

        t_lang_done = time.time() - t_lang
        log.info(
            f"  {lang.upper()} complete: {t_lang_done:.0f}s | "
            f"vocab_used={row['vocab_used']:,} | "
            f"top100={row['vocab_concentration_top100']:.3f} | "
            f"top1000={row['vocab_concentration_top1000']:.3f}"
        )
        write_progress(gen_done, total_gens, t_all, lang=lang.upper())

        gc.collect()
        torch.cuda.empty_cache()

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # --- Save ---
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    t_total = time.time() - t_all
    log.info(f"\n{'='*60}")
    log.info(f"RESULTS ({t_total/60:.1f} min)")
    log.info("=" * 60)
    log.info("\n" + df.to_string(index=False))

    # Correlations
    log.info("--- Spearman correlations with ED_mean (n=5) ---")
    for col in ["vocab_used", "vocab_used_argmax",
                "vocab_concentration_top100", "vocab_concentration_top1000",
                "vocab_concentration_top5000", "mean_fertility"]:
        rho, p = spearmanr(df["ED_mean"], df[col])
        log.info(f"  ED vs {col:35s}  rho={rho:+.3f}  p={p:.3f}")

    # EN vs PL
    en = df.loc[df["language"] == "en"].iloc[0]
    pl = df.loc[df["language"] == "pl"].iloc[0]
    log.info(f"  EN vs PL:")
    log.info(f"    vocab_used: {int(en['vocab_used']):,} vs {int(pl['vocab_used']):,}")
    log.info(f"    top100:     {en['vocab_concentration_top100']:.3f} vs "
             f"{pl['vocab_concentration_top100']:.3f}")
    log.info(f"    ED:         {en['ED_mean']:.3f} vs {pl['ED_mean']:.3f}")

    log.info(f"Saved to {OUTPUT_CSV}")
    write_progress(gen_done, total_gens, t_all, lang="DONE")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
generate_logits_hf.py — HuggingFace inference engine for Entropic Deviation.

Drop-in alternative to generate_logits.py for models that don't have GGUF
quantizations or aren't supported by llama-cpp-python (e.g. Mamba2, RWKV).

Uses HuggingFace transformers with model.generate(output_scores=True) to
extract per-token logits, computes ED inline, and writes the same CSV
format as generate_logits.py.

Tested with:
  - state-spaces/mamba2-2.7b  (Mamba2 SSM, requires mamba_ssm + causal_conv1d)
  - state-spaces/mamba-2.8b-hf (Mamba v1, pure HF)
  - tiiuae/falcon-mamba-7b     (Falcon Mamba)

Usage:
    python generate_logits_hf.py \
        --model state-spaces/mamba2-2.7b \
        --prompts prompts/prompts.jsonl \
        --temps 0.7 1.0 1.3 \
        --max_tokens 128 \
        --ed-out results/ed_mamba2_2.7b.csv
"""
import argparse
import json
import math
import os
import re
import sys
import gc
import signal
import time
import logging
from datetime import datetime

import torch
import numpy as np
import pandas as pd


# --- ED computation (identical to generate_logits.py) ---------------------

def entropic_deviation(logits):
    """Compute per-token ED: KL(softmax(logits) || uniform) / log(vocab_size)."""
    logits = logits.float()
    p = torch.softmax(logits, dim=-1)
    n = p.size(-1)
    log_p = torch.log(p.clamp(min=1e-45))
    kl = torch.sum(p * (log_p - math.log(1.0 / n)), dim=-1)
    return kl / math.log(n)


# --- Helpers (shared with generate_logits.py) -----------------------------

def setup_logger(log_file=None):
    logger = logging.getLogger("ed_hf")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def parse_domain(prompt):
    """Extract domain from prompt prefix like 'wiki: ...'."""
    if ":" in prompt:
        return prompt.split(":", 1)[0].strip()
    return ""


def _parse_model_size(model_name):
    """Extract parameter count from model name (e.g. 'mamba2-2.7b' → 2.7e9)."""
    m = re.search(r'(\d+(?:\.\d+)?)\s*[Bb]', model_name)
    if m:
        return int(float(m.group(1)) * 1_000_000_000)
    return None


def find_resume_point(ed_csv):
    """Count existing rows in CSV to support --resume."""
    if not os.path.exists(ed_csv):
        return 0
    try:
        with open(ed_csv, 'r', encoding='utf-8') as f:
            return max(0, sum(1 for _ in f) - 1)  # subtract header
    except Exception:
        return 0


def flush_records(records, ed_csv, write_header):
    """Append buffered records to CSV."""
    if not records:
        return
    df = pd.DataFrame(records)
    df.to_csv(ed_csv, mode='a', header=write_header, index=False)


def write_progress(path, model_name, processed, total, t_start_wall, ed_mean_last=None):
    """Write a human-readable progress file, atomically."""
    if not path:
        return
    elapsed = time.time() - t_start_wall
    rate = processed / elapsed if elapsed > 0 else 0
    eta = (total - processed) / rate if rate > 0 else float('inf')
    tmp = path + ".tmp"
    try:
        with open(tmp, 'w') as f:
            f.write(f"model: {model_name}\n")
            f.write(f"progress: {processed}/{total} ({100*processed/total:.1f}%)\n")
            f.write(f"elapsed: {elapsed/3600:.1f}h\n")
            f.write(f"rate: {rate:.2f} gen/s\n")
            f.write(f"eta: {eta/3600:.1f}h\n")
            if ed_mean_last is not None:
                f.write(f"last_ED: {ed_mean_last:.4f}\n")
            f.write(f"updated: {datetime.now().isoformat()}\n")
        os.replace(tmp, path)
    except Exception:
        pass


# --- Model loading --------------------------------------------------------

def load_model_and_tokenizer(model_id, dtype, device, logger):
    """Load HF model + tokenizer with appropriate backend."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    logger.info(f"Loading tokenizer from {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Ensure pad token exists (Mamba models often don't have one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading model from {model_id} (dtype={dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # Log model info
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {n_params/1e9:.2f}B parameters, "
                f"vocab_size={model.config.vocab_size}")

    return model, tokenizer


# --- Generation with logit extraction ------------------------------------

def generate_with_logits(model, tokenizer, prompt, max_tokens, temperature, device):
    """Generate tokens and return stacked logits for generated tokens only.

    Returns:
        logits: torch.Tensor of shape (gen_tokens, vocab_size) in float32
        gen_n_tokens: int, number of generated tokens
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_k=0,           # disable top-k to match llama-cpp behavior
            top_p=1.0,         # disable nucleus sampling
            output_scores=True,
            return_dict_in_generate=True,
        )

    # outputs.scores is a tuple of (gen_tokens,) tensors, each (batch, vocab)
    if not outputs.scores:
        return None, 0

    # Stack into (gen_tokens, vocab_size) and move to float32
    logits = torch.stack(outputs.scores, dim=0).squeeze(1).float()
    gen_n_tokens = logits.shape[0]

    return logits, gen_n_tokens


# --- Main -----------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="HuggingFace inference for Entropic Deviation"
    )
    p.add_argument("--model", required=True,
                   help="HuggingFace model ID (e.g. state-spaces/mamba2-2.7b)")
    p.add_argument("--model-name", default=None,
                   help="Display name for CSV (default: derived from --model)")
    p.add_argument("--prompts", required=True,
                   help="Path to prompts JSONL file")
    p.add_argument("--temps", nargs="+", type=float, default=[0.7, 1.0, 1.3],
                   help="Temperature values (default: 0.7 1.0 1.3)")
    p.add_argument("--max_tokens", type=int, default=128,
                   help="Max tokens to generate per prompt")
    p.add_argument("--ed-out", required=True,
                   help="Output CSV path for ED results")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"],
                   help="Model precision (default: float16)")
    p.add_argument("--device", default="auto",
                   help="Device: 'auto', 'cuda', 'cuda:0', 'cpu'")
    p.add_argument("--save_interval", type=int, default=20,
                   help="Flush to CSV every N generations")
    p.add_argument("--resume", action="store_true",
                   help="Resume from existing CSV")
    p.add_argument("--log", default=None,
                   help="Log file path")
    p.add_argument("--progress-file", default=None,
                   help="Progress status file")
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle prompt×temperature combinations (recommended)")
    args = p.parse_args()

    logger = setup_logger(args.log)

    # Model name for CSV
    model_name = args.model_name or args.model.split("/")[-1]
    model_size = _parse_model_size(model_name) or _parse_model_size(args.model)

    # Dtype mapping
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}, dtype: {args.dtype}")

    # Load prompts
    prompts = []
    with open(args.prompts) as f:
        for line in f:
            prompts.append(json.loads(line)["prompt"])
    logger.info(f"Loaded {len(prompts)} prompts from {args.prompts}")

    # Build all (temp, prompt) combos
    combos = [(t, p) for t in args.temps for p in prompts]
    total = len(combos)
    logger.info(f"Total combinations: {total} ({len(args.temps)} temps × {len(prompts)} prompts)")

    if args.shuffle:
        import random
        random.seed(42)  # reproducible shuffle
        random.shuffle(combos)
        logger.info("Shuffled prompt×temperature order (seed=42)")

    # Resume
    resume_idx = 0
    if args.resume:
        resume_idx = find_resume_point(args.ed_out)
        logger.info(f"Resuming from row {resume_idx}")

    write_header = (resume_idx == 0) or not os.path.exists(args.ed_out)

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, dtype, device, logger)

    # If device_map="auto" was used, find the actual device
    if hasattr(model, 'device'):
        actual_device = model.device
    else:
        actual_device = next(model.parameters()).device
    logger.info(f"Model on device: {actual_device}")

    # --- Buffers and state ---
    ed_records = []
    t_wall_start = time.time()
    processed = 0
    ed_mean_last = None

    # --- Signal handler ---
    def save_and_exit(signum, frame):
        if ed_records:
            flush_records(ed_records, args.ed_out,
                          write_header and processed == len(ed_records))
            logger.info(f"Signal {signum}: flushed {len(ed_records)} records")
        write_progress(args.progress_file, model_name,
                       resume_idx + processed, total, t_wall_start, ed_mean_last)
        sys.exit(0)

    signal.signal(signal.SIGINT, save_and_exit)
    signal.signal(signal.SIGTERM, save_and_exit)

    # --- Generation loop ---
    for i, (temp, prompt) in enumerate(combos):
        if i < resume_idx:
            continue

        t_start = time.perf_counter()

        try:
            logits, gen_n_tokens = generate_with_logits(
                model, tokenizer, prompt, args.max_tokens, temp, actual_device
            )
        except Exception as e:
            logger.warning(f"[{i+1}/{total}] Generation failed: {e}")
            continue

        if logits is None or gen_n_tokens == 0:
            logger.warning(f"[{i+1}/{total}] No tokens generated, skipping")
            continue

        # Compute ED
        with torch.no_grad():
            ed_t = entropic_deviation(logits)
        ed_mean = ed_t.mean().item()
        ed_std = ed_t.std().item()
        ed_mean_last = ed_mean

        t_end = time.perf_counter()

        global_rank = resume_idx + processed
        ed_records.append({
            'prompt': prompt,
            'temp': temp,
            'seq_len': gen_n_tokens,
            'gen_time': time.time(),
            'timestamp': datetime.now().isoformat(),
            'ED_mean': ed_mean,
            'ED_std': ed_std,
            'model': model_name,
            'model_size': model_size,
            'domain': parse_domain(prompt),
            'rank': global_rank,
            'chkpt_id': global_rank // args.save_interval,
        })

        processed += 1

        # Flush
        if processed % args.save_interval == 0:
            flush_records(ed_records, args.ed_out, write_header)
            write_header = False
            logger.info(f"[{i+1}/{total}] Flushed {len(ed_records)} records")
            ed_records.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            write_progress(args.progress_file, model_name,
                           resume_idx + processed, total, t_wall_start, ed_mean_last)

        if processed % 10 == 0:
            elapsed = t_end - t_start
            logger.info(
                f"[{i+1}/{total}] gen #{processed} | "
                f"{elapsed:.1f}s | tokens: {gen_n_tokens} | "
                f"ED: {ed_mean:.4f} ± {ed_std:.4f}"
            )

        # Clean up tensors
        del logits, ed_t

    # Final flush
    if ed_records:
        flush_records(ed_records, args.ed_out, write_header)
        logger.info(f"Final flush: {len(ed_records)} records")

    write_progress(args.progress_file, model_name,
                   resume_idx + processed, total, t_wall_start, ed_mean_last)
    logger.info(f"Done. {processed} generations saved to {args.ed_out}")


if __name__ == "__main__":
    main()

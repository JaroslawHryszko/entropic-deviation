#!/usr/bin/env python3
"""
generate_logits.py — Multi-GPU inference engine for Entropic Deviation.

Loads a single GGUF model spread across all available GPUs,
generates responses for all (prompt × temperature) combinations,
and saves logit tensors to .pt checkpoint files.
"""
import argparse
import json
import os
import sys
import gc
import signal
import time
import logging
from datetime import datetime

import psutil
import torch
import numpy as np
from llama_cpp import Llama


# --- Helpers -----------------------------------------------------------

def setup_logger(log_file=None):
    """Configure console (and optional file) logging."""
    logger = logging.getLogger("ed_inference")
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


def detect_gpus():
    """Return number of available CUDA GPUs."""
    try:
        import subprocess
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, check=True
        )
        count = len(r.stdout.strip().split('\n'))
        return count
    except Exception:
        return 0


def load_prompts(path, logger):
    """Read prompts from JSONL or plain TXT."""
    logger.info(f"Loading prompts from {path}")
    prompts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('{'):
                try:
                    record = json.loads(line)
                    prompt = f"{record['domain']}: {record['prompt']}"
                    prompts.append(prompt)
                except Exception:
                    continue
            else:
                prompts.append(line)
    logger.info(f"Loaded {len(prompts)} prompts")
    return prompts


def find_resume_point(prefix):
    """
    Scan for existing checkpoints matching prefix_chkpt_*.pt
    and return the highest checkpoint index found (= number of
    generations already saved). Returns 0 if no checkpoints exist.
    """
    import glob, re
    pattern = f"{prefix}_chkpt_*.pt"
    files = glob.glob(pattern)
    if not files:
        return 0
    indices = []
    for fn in files:
        m = re.search(r'_chkpt_(\d+)\.pt$', fn)
        if m:
            indices.append(int(m.group(1)))
    return max(indices) if indices else 0


# --- Main processing --------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Generate logits for Entropic Deviation analysis"
    )
    p.add_argument("--model", required=True, help="Path to GGUF model")
    p.add_argument("--prompts", required=True, help="JSONL or TXT file with prompts")
    p.add_argument("--temps", nargs="+", type=float,
                   default=[0.7, 1.0, 1.3], help="Temperatures")
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--n_ctx", type=int, default=512)
    p.add_argument("--out", default="logits", help="Output prefix for checkpoint files")
    p.add_argument("--log", default=None, help="Log file path")
    p.add_argument("--save_interval", type=int, default=5,
                   help="Checkpoint every N generations")
    p.add_argument("--reset_interval", type=int, default=20,
                   help="Full GC every N generations")
    p.add_argument("--sleep_time", type=float, default=1.0,
                   help="Sleep between generations (seconds)")
    p.add_argument("--n_gpu_layers", type=int, default=-1,
                   help="Number of layers to offload to GPU (-1 = all)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from last checkpoint")
    args = p.parse_args()

    logger = setup_logger(args.log)

    # --- Detect GPUs and build tensor split ---
    n_gpus = detect_gpus()
    if n_gpus == 0:
        logger.warning("No GPUs detected, running on CPU")
        tensor_split = None
    else:
        tensor_split = [1.0 / n_gpus] * n_gpus
        logger.info(f"Detected {n_gpus} GPU(s), tensor_split={tensor_split}")

    # --- Load prompts and build combinations ---
    all_prompts = load_prompts(args.prompts, logger)
    combos = [(t, prompt) for t in args.temps for prompt in all_prompts]
    total = len(combos)
    logger.info(f"Total combinations: {total} ({len(all_prompts)} prompts × {len(args.temps)} temps)")

    # --- Resume logic ---
    resume_idx = 0
    if args.resume:
        resume_idx = find_resume_point(args.out)
        if resume_idx > 0:
            logger.info(f"Resuming from checkpoint {resume_idx} ({resume_idx}/{total} done)")
        else:
            logger.info("No checkpoints found, starting from scratch")

    # --- Load model ---
    logger.info(f"Loading model: {args.model}")
    llm_kwargs = dict(
        model_path=args.model,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        logits_all=True,
        verbose=False,
    )
    if tensor_split is not None:
        llm_kwargs["tensor_split"] = tensor_split
    llm = Llama(**llm_kwargs)
    logger.info("Model loaded")

    # --- Signal handler for graceful shutdown ---
    out_logits = []
    out_meta = []
    checkpoint_counter = resume_idx

    def save_and_exit(signum, frame):
        if out_logits:
            checkpoint_counter_now = checkpoint_counter + len(out_logits)
            cp = f"{args.out}_chkpt_{checkpoint_counter_now}.pt"
            torch.save({'logits': out_logits, 'meta': out_meta}, cp)
            logger.info(f"Signal {signum}: saved emergency checkpoint {cp}")
        sys.exit(0)

    signal.signal(signal.SIGINT, save_and_exit)
    signal.signal(signal.SIGTERM, save_and_exit)

    # --- Generation loop ---
    processed = 0
    for i, (temp, prompt) in enumerate(combos):
        if i < resume_idx:
            continue

        llm.reset()
        with torch.no_grad():
            resp = llm(prompt, max_tokens=args.max_tokens, temperature=temp, logprobs=5)

        if hasattr(llm, 'scores') and llm.scores is not None:
            try:
                arr = np.array(llm.scores, dtype=np.float32)
                if arr.size == 0:
                    logger.warning(f"[{i+1}/{total}] Empty scores, skipping")
                    continue
                out_logits.append(torch.from_numpy(arr))
                out_meta.append({
                    'prompt': prompt,
                    'temp': temp,
                    'seq_len': arr.shape[0],
                    'gen_time': time.time(),
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.warning(f"[{i+1}/{total}] Failed to extract scores: {e}")
                continue
        else:
            logger.warning(f"[{i+1}/{total}] No scores available, skipping")
            continue

        processed += 1

        # Checkpoint
        if processed % args.save_interval == 0:
            checkpoint_counter += len(out_logits)
            cp = f"{args.out}_chkpt_{checkpoint_counter}.pt"
            torch.save({'logits': out_logits, 'meta': out_meta}, cp)
            logger.info(f"[{i+1}/{total}] Checkpoint saved: {cp}")
            out_logits.clear()
            out_meta.clear()
            gc.collect()
            torch.cuda.empty_cache()

        # Periodic GC
        if processed % args.reset_interval == 0:
            gc.collect()
            torch.cuda.empty_cache()

        if processed % 50 == 0:
            logger.info(f"Progress: {i+1}/{total} ({processed} generated)")

        time.sleep(args.sleep_time)

    # --- Final save ---
    if out_logits:
        checkpoint_counter += len(out_logits)
        cp = f"{args.out}_chkpt_{checkpoint_counter}.pt"
        torch.save({'logits': out_logits, 'meta': out_meta}, cp)
        logger.info(f"Final checkpoint saved: {cp}")

    logger.info(f"Done. {processed} generations saved to {args.out}_chkpt_*.pt")


if __name__ == "__main__":
    main()

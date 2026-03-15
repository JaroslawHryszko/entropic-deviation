#!/usr/bin/env python3
"""
generate_logits.py — Multi-GPU inference engine for Entropic Deviation.

Loads a single GGUF model spread across all available GPUs,
generates responses for all (prompt × temperature) combinations,
computes Entropic Deviation inline, and writes results to CSV.
Optionally saves logit tensors to .pt checkpoint files (--save-logits).
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

import psutil
import torch
import numpy as np
import pandas as pd


# --- ED computation ---------------------------------------------------

def entropic_deviation(logits):
    """Compute per-token ED: KL(softmax(logits) || uniform) / log(vocab_size)."""
    logits = logits.float()
    p = torch.softmax(logits, dim=-1)
    n = p.size(-1)
    log_p = torch.log(p.clamp(min=1e-45))
    kl = torch.sum(p * (log_p - math.log(1.0 / n)), dim=-1)
    return kl / math.log(n)


def _parse_model_size(model_name):
    """Extract parameter count from model name (e.g. 'Llama-3.3-70B' → 70e9)."""
    m = re.search(r'(\d+(?:\.\d+)?)\s*[Bb]', model_name)
    if m:
        return int(float(m.group(1)) * 1_000_000_000)
    return None


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


def find_resume_point(ed_csv):
    """Count data rows in existing ED CSV to determine resume index."""
    if not os.path.exists(ed_csv):
        return 0
    count = 0
    with open(ed_csv, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # skip header
            if line.strip():
                count += 1
    return count


def derive_model_name(model_path):
    """Derive a short model label from GGUF filename."""
    name = os.path.basename(model_path)
    name = re.sub(r'\.gguf$', '', name)
    name = re.sub(r'[-_]Q[0-9].*', '', name)
    return name


def parse_domain(prompt):
    """Extract domain from prompt prefix like 'wiki: ...'."""
    if ":" in prompt:
        return prompt.split(":", 1)[0].strip()
    return ""


# --- CSV I/O -----------------------------------------------------------

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
    pct = processed / total * 100 if total else 0
    elapsed = time.time() - t_start_wall
    if processed > 0:
        eta_s = elapsed / processed * (total - processed)
        eta_h, eta_rem = divmod(int(eta_s), 3600)
        eta_m, eta_sec = divmod(eta_rem, 60)
        eta_str = f"{eta_h}h {eta_m:02d}m {eta_sec:02d}s"
    else:
        eta_str = "—"
    elapsed_h, elapsed_rem = divmod(int(elapsed), 3600)
    elapsed_m, elapsed_sec = divmod(elapsed_rem, 60)
    elapsed_str = f"{elapsed_h}h {elapsed_m:02d}m {elapsed_sec:02d}s"

    lines = [
        f"model:    {model_name}",
        f"progress: {processed}/{total}  ({pct:.1f}%)",
        f"elapsed:  {elapsed_str}",
        f"ETA:      {eta_str}",
        f"updated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if ed_mean_last is not None:
        lines.insert(2, f"last ED:  {ed_mean_last:.4f}")

    tmp = path + ".tmp"
    with open(tmp, 'w') as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)


# --- Main processing --------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Generate logits and compute Entropic Deviation"
    )
    p.add_argument("--model", required=True, help="Path to GGUF model")
    p.add_argument("--prompts", required=True, help="JSONL or TXT file with prompts")
    p.add_argument("--temps", nargs="+", type=float,
                   default=[0.7, 1.0, 1.3], help="Temperatures")
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--n_ctx", type=int, default=512)
    p.add_argument("--out", default="logits", help="Output prefix for checkpoint files (used with --save-logits)")
    p.add_argument("--ed-out", default=None, help="CSV path for ED results (default: {out}_ed.csv)")
    p.add_argument("--model-name", default=None, help="Model label for CSV (derived from model path if not given)")
    p.add_argument("--log", default=None, help="Log file path")
    p.add_argument("--save_interval", type=int, default=20,
                   help="Flush CSV every N generations")
    p.add_argument("--save-logits", action="store_true",
                   help="Also save .pt checkpoint files with full logit tensors")
    p.add_argument("--n_gpu_layers", type=int, default=-1,
                   help="Number of layers to offload to GPU (-1 = all)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from last saved position in ED CSV")
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle prompt×temperature order (recommended to avoid F8 ordering confounds)")
    p.add_argument("--progress-file", default=None,
                   help="Path to progress status file (updated every save_interval)")
    args = p.parse_args()

    logger = setup_logger(args.log)

    # Derive defaults
    if args.ed_out is None:
        args.ed_out = f"{args.out}_ed.csv"
    if args.model_name is None:
        args.model_name = derive_model_name(args.model)

    model_size = _parse_model_size(args.model_name)

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

    if args.shuffle:
        import random
        random.seed(42)  # reproducible shuffle
        random.shuffle(combos)
        logger.info("Shuffled prompt×temperature order (seed=42)")

    # --- Resume logic (from ED CSV) ---
    resume_idx = 0
    if args.resume:
        resume_idx = find_resume_point(args.ed_out)
        if resume_idx > 0:
            logger.info(f"Resuming from row {resume_idx} ({resume_idx}/{total} done)")
        else:
            logger.info("No existing results found, starting from scratch")

    # --- Load model ---
    from llama_cpp import Llama
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

    # --- CSV header ---
    write_header = (resume_idx == 0) or not os.path.exists(args.ed_out)

    # --- Buffers and state ---
    ed_records = []
    out_logits = []
    out_meta = []
    t_wall_start = time.time()
    processed = 0
    ed_mean_last = None

    # --- Signal handler for graceful shutdown ---
    def save_and_exit(signum, frame):
        if ed_records:
            flush_records(ed_records, args.ed_out, write_header and processed == len(ed_records))
            logger.info(f"Signal {signum}: flushed {len(ed_records)} ED records to {args.ed_out}")
        if args.save_logits and out_logits:
            cp = f"{args.out}_chkpt_{resume_idx + processed}.pt"
            torch.save({'logits': out_logits, 'meta': out_meta}, cp)
            logger.info(f"Signal {signum}: saved emergency checkpoint {cp}")
        write_progress(args.progress_file, args.model_name, resume_idx + processed, total, t_wall_start, ed_mean_last)
        sys.exit(0)

    signal.signal(signal.SIGINT, save_and_exit)
    signal.signal(signal.SIGTERM, save_and_exit)

    # --- Generation loop ---
    for i, (temp, prompt) in enumerate(combos):
        if i < resume_idx:
            continue

        t_start = time.perf_counter()

        llm.reset()

        # Tokenize prompt to determine prompt length for trimming
        prompt_tokens = llm.tokenize(prompt.encode())
        prompt_n_tokens = len(prompt_tokens)

        t_pre = time.perf_counter()
        resp = llm(prompt, max_tokens=args.max_tokens, temperature=temp, logprobs=5)
        t_infer = time.perf_counter()

        if hasattr(llm, 'scores') and llm.scores is not None:
            try:
                arr = np.array(llm.scores, dtype=np.float32)
                if arr.size == 0:
                    logger.warning(f"[{i+1}/{total}] Empty scores, skipping")
                    continue

                # Trim to generated tokens only (skip prompt tokens)
                if arr.shape[0] > prompt_n_tokens:
                    arr = arr[prompt_n_tokens:]
                gen_n_tokens = arr.shape[0]

                # Compute ED on float32 logits (before any precision loss)
                logit_tensor_f32 = torch.from_numpy(arr)
                with torch.no_grad():
                    ed_t = entropic_deviation(logit_tensor_f32)
                ed_mean = ed_t.mean().item()
                ed_std = ed_t.std().item()
                ed_mean_last = ed_mean

                t_extract = time.perf_counter()

                # Build ED record
                global_rank = resume_idx + processed
                ed_records.append({
                    'prompt': prompt,
                    'temp': temp,
                    'seq_len': gen_n_tokens,
                    'gen_time': time.time(),
                    'timestamp': datetime.now().isoformat(),
                    'ED_mean': ed_mean,
                    'ED_std': ed_std,
                    'model': args.model_name,
                    'model_size': model_size,
                    'domain': parse_domain(prompt),
                    'rank': global_rank,
                    'chkpt_id': global_rank // args.save_interval,
                })

                # Optionally save logit tensors
                if args.save_logits:
                    logit_tensor = logit_tensor_f32.half()
                    out_logits.append(logit_tensor)
                    out_meta.append({
                        'prompt': prompt,
                        'temp': temp,
                        'seq_len': gen_n_tokens,
                        'prompt_n_tokens': prompt_n_tokens,
                        'gen_n_tokens': gen_n_tokens,
                        'gen_time': time.time(),
                        'timestamp': datetime.now().isoformat()
                    })

                del logit_tensor_f32, ed_t
            except Exception as e:
                logger.warning(f"[{i+1}/{total}] Failed to extract scores: {e}")
                continue
        else:
            logger.warning(f"[{i+1}/{total}] No scores available, skipping")
            continue

        processed += 1
        t_end = time.perf_counter()

        # Flush at save_interval
        if processed % args.save_interval == 0:
            flush_records(ed_records, args.ed_out, write_header)
            write_header = False
            logger.info(
                f"[{i+1}/{total}] Flushed {len(ed_records)} records to {args.ed_out}"
            )
            ed_records.clear()

            if args.save_logits and out_logits:
                cp_idx = resume_idx + processed
                cp = f"{args.out}_chkpt_{cp_idx}.pt"
                t_save_start = time.perf_counter()
                torch.save({'logits': out_logits, 'meta': out_meta}, cp)
                t_save_end = time.perf_counter()
                logger.info(f"  Checkpoint saved: {cp} (save: {t_save_end - t_save_start:.1f}s)")
                out_logits.clear()
                out_meta.clear()

            gc.collect()
            torch.cuda.empty_cache()
            write_progress(args.progress_file, args.model_name,
                           resume_idx + processed, total, t_wall_start, ed_mean_last)

        if processed % 10 == 0:
            logger.info(
                f"[{i+1}/{total}] gen #{processed} | "
                f"infer: {t_infer - t_pre:.1f}s | "
                f"extract: {t_extract - t_infer:.1f}s | "
                f"total: {t_end - t_start:.1f}s | "
                f"tokens: {gen_n_tokens} (prompt: {prompt_n_tokens}) | "
                f"ED: {ed_mean:.4f} ± {ed_std:.4f}"
            )

    # --- Final flush ---
    if ed_records:
        flush_records(ed_records, args.ed_out, write_header)
        logger.info(f"Final flush: {len(ed_records)} records to {args.ed_out}")

    if args.save_logits and out_logits:
        cp_idx = resume_idx + processed
        cp = f"{args.out}_chkpt_{cp_idx}.pt"
        torch.save({'logits': out_logits, 'meta': out_meta}, cp)
        logger.info(f"Final checkpoint saved: {cp}")

    write_progress(args.progress_file, args.model_name,
                   resume_idx + processed, total, t_wall_start, ed_mean_last)
    logger.info(f"Done. {processed} generations → {args.ed_out}")


if __name__ == "__main__":
    main()

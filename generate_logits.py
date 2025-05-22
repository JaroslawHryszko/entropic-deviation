#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
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
import threading

# --- Helpers -----------------------------------------------------------

def setup_logger(log_file=None):
    """Configure console (and optional file) logging."""
    logger = logging.getLogger("multi_gpu_inference")
    logger.handlers.clear()  # avoid duplicate handlers
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

def check_gpu_usage(gpu_id):
    """Return (gpu_util_pct, used_memory_mb)."""
    try:
        r = subprocess.run(
            ['nvidia-smi', '-i', str(gpu_id),
             '--query-gpu=utilization.gpu,memory.used',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        util, mem = map(int, r.stdout.strip().split(','))
        return util, mem
    except Exception:
        return 0, 0

def find_resume_point(prefix, gpu_id, combinations):
    """
    If final file exists, return (len(combinations), logits, meta).
    Else if temp exists, load and compute how many combos done.
    Otherwise return (0, [], []).
    """
    final_f = f"{prefix}_gpu{gpu_id}.pt"
    tmp_f   = f"{prefix}_gpu{gpu_id}_temp.pt"

    if os.path.exists(final_f):
        logger.info(f"GPU{gpu_id}: loading final {final_f}")
        data = torch.load(final_f, map_location='cpu')
        return len(combinations), data['logits'], data['meta']

    if os.path.exists(tmp_f):
        logger.info(f"GPU{gpu_id}: loading temp {tmp_f}")
        data = torch.load(tmp_f, map_location='cpu')
        # assume meta entries == number of processed combos
        return len(data.get('meta', [])), data.get('logits', []), data.get('meta', [])

    return 0, [], []

def load_prompts(path):
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
                    #prompts.append(json.loads(line)['prompt'])
                    record = json.loads(line)
                    prompt = f"{record['domain']}: {record['prompt']}"
                    prompts.append(prompt)
                except Exception:
                    continue
            else:
                prompts.append(line)
    logger.info(f"Loaded {len(prompts)} prompts")
    return prompts

def combination_gen(temps, prompts):
    """Yield (temp, prompt) without building full list in memory."""
    for t in temps:
        for p in prompts:
            yield t, p

def save_and_exit(signum, frame):
    """On SIGINT/SIGTERM, persist each GPU state and exit."""
    for gpu_id, s in state.items():
        prefix = s['prefix']
        tf = f"{prefix}_gpu{gpu_id}_temp.pt"
        torch.save({'logits': s['out_logits'], 'meta': s['meta']}, tf)
        logger.info(f"GPU{gpu_id}: saved temp state to {tf}")
    sys.exit(0)

# --- Main processing --------------------------------------------------

state = {}  # for signal handler to persist

def process_gpu(model_path, prompts, temps, max_tokens, prefix,
                gpu_id, save_interval, reset_interval,
                sleep_time, n_ctx, chat_format, n_gpu_layers):
    torch.cuda.set_device(gpu_id)
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        chat_format=chat_format,
        logits_all=True,
        main_gpu=gpu_id,
        tensor_split=[1.0, 0.0] if gpu_id == 0 else [0.0, 1.0],
        verbose=False
    )

    # build combos list only once for resume logic
    combos = [(t, p) for t in temps for p in prompts]
    resume_idx, out_logits, meta = find_resume_point(prefix, gpu_id, combos)
    state[gpu_id] = {'prefix': prefix, 'out_logits': out_logits, 'meta': meta}

    gen = combination_gen(temps, prompts)
    processed = 0

    for i, (t, prompt) in enumerate(gen):
        if i < resume_idx:
            continue

        util, mem = check_gpu_usage(gpu_id)
        if util > 90:
            # back off if GPU is too busy
            time.sleep((util - 80) * 0.5)

        llm.reset()
        with torch.no_grad():
            resp = llm(prompt, max_tokens=max_tokens, temperature=t, logprobs=5)

        if hasattr(llm, 'scores'):
            arr = np.array(llm.scores, dtype=np.float32)
            out_logits.append(torch.from_numpy(arr))
            meta.append({
                'prompt': prompt,
                'temp': t,
                'seq_len': arr.shape[0],
                'gpu_id': gpu_id,
                'gen_time': time.time(),
                'timestamp': datetime.now().isoformat()
            })

        processed += 1

        # checkpoint + clear buffers
        if processed % save_interval == 0:
            cp = f"{prefix}_gpu{gpu_id}_chkpt_{processed}.pt"
            torch.save({'logits': out_logits, 'meta': meta}, cp)
            logger.info(f"GPU{gpu_id}: saved checkpoint {cp}")
            out_logits.clear()
            meta.clear()
            gc.collect()
            torch.cuda.empty_cache()

        # periodic full cleanup
        if processed % reset_interval == 0:
            gc.collect()
            torch.cuda.empty_cache()

        time.sleep(sleep_time)

    # final save + cleanup
    final_f = f"{prefix}_gpu{gpu_id}.pt"
    torch.save({'logits': out_logits, 'meta': meta}, final_f)
    logger.info(f"GPU{gpu_id}: final results saved to {final_f}")
    out_logits.clear()
    meta.clear()
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, save_and_exit)
    signal.signal(signal.SIGTERM, save_and_exit)

    p = argparse.ArgumentParser(
        description="Multi-GPU inference, memory optimized"
    )
    p.add_argument("--model",       required=True, help="Path to GGUF model")
    p.add_argument("--prompts",     required=True, help="File with prompts")
    p.add_argument("--temps",       nargs="+", type=float,
                        default=[0.7, 1.0, 1.3], help="Temperatures")
    p.add_argument("--max_tokens",  type=int, default=64)
    p.add_argument("--n_ctx",       type=int, default=512)
    p.add_argument("--out",         default="logits")
    p.add_argument("--log",         default=None, help="Log file")
    p.add_argument("--gpu_split",   type=float, default=0.5,
                        help="Fraction of prompts for GPU0")
    p.add_argument("--save_interval",  type=int, default=5,
                        help="Checkpoint every N generations")
    p.add_argument("--reset_interval", type=int, default=20,
                        help="Full GC every N generations")
    p.add_argument("--sleep_time",     type=float, default=1.0)
    p.add_argument("--n_gpu_layers",   type=int, default=-1)

    args = p.parse_args()
    logger = setup_logger(args.log)
    all_prompts = load_prompts(args.prompts)

    split = int(len(all_prompts) * args.gpu_split)
    p0, p1 = all_prompts[:split], all_prompts[split:]
    logger.info(f"Total prompts: {len(all_prompts)} – GPU0: {len(p0)}, GPU1: {len(p1)}")

    threads = []
    for gid, prom in enumerate([p0, p1]):
        t = threading.Thread(
            target=process_gpu,
            args=(args.model, prom, args.temps, args.max_tokens,
                  args.out, gid, args.save_interval,
                  args.reset_interval, args.sleep_time,
                  args.n_ctx, args.log, args.n_gpu_layers)
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

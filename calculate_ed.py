#!/usr/bin/env python3
"""
calculate_ed.py — Streaming checkpoint processor for Entropic Deviation.

Computes ED per token as normalized KL divergence from uniform,
aggregates to ED_mean and ED_std per generation.
"""
import torch, math, argparse, pandas as pd, glob, os, gc, re, logging
from datetime import datetime


def setup_logger(log_file=None):
    logger = logging.getLogger("ed_calculate")
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


def _parse_model_size(model_name):
    """Extract parameter count from model name (e.g. 'Llama-3.3-70B' → 70e9).
    Returns None if no size pattern is found."""
    m = re.search(r'(\d+(?:\.\d+)?)\s*[Bb]', model_name)
    if m:
        return int(float(m.group(1)) * 1_000_000_000)
    return None


def entropic_deviation(logits):
    """Compute per-token ED: KL(softmax(logits) || uniform) / log(vocab_size)."""
    # Upcast float16 → float32 for numerical precision in softmax
    logits = logits.float()
    p = torch.softmax(logits, dim=-1)
    n = p.size(-1)
    # Clamp to avoid log(0) = -inf which causes NaN in p * log(p)
    log_p = torch.log(p.clamp(min=1e-45))
    kl = torch.sum(p * (log_p - math.log(1.0 / n)), dim=-1)
    return kl / math.log(n)


def parse_index(fn):
    """Extract checkpoint number from filename *chkpt_{N}.pt*."""
    m = re.search(r'_chkpt_(\d+)\.pt$', fn)
    return int(m.group(1)) if m else float('inf')


def process_one_bundle(fp, out_csv, write_header, model_name, logger=None):
    try:
        bundle = torch.load(fp, map_location='cpu', weights_only=False)
    except Exception as e:
        if logger:
            logger.error(f"Failed to load {fp}: {e}")
        raise
    seqs, meta = bundle["logits"], bundle["meta"]
    records = []
    for i, logits in enumerate(seqs):
        try:
            with torch.no_grad():
                ed_t = entropic_deviation(logits)
            rec = meta[i].copy()
            rec["ED_mean"] = ed_t.mean().item()
            rec["ED_std"]  = ed_t.std().item()
            rec["model"]   = model_name
            rec["timestamp_processed"] = datetime.now().isoformat()
            rec["model_size"] = _parse_model_size(model_name)
            rec["rank"] = i
            rec["chkpt_id"] = parse_index(fp)
            if "prompt" in rec and ":" in rec["prompt"]:
                rec["domain"] = rec["prompt"].split(":", 1)[0].strip()
            records.append(rec)
        except Exception as e:
            if logger:
                logger.error(f"Failed to compute ED for entry {i} in {fp}: {e}")
            continue
    df = pd.DataFrame(records)
    df.to_csv(out_csv, mode='a', header=write_header, index=False)
    del bundle, seqs, meta, records, df
    gc.collect()


def main():
    ap = argparse.ArgumentParser(
        description="Compute Entropic Deviation from logits checkpoints"
    )
    ap.add_argument(
        "--pattern", default="logits_*_chkpt_*.pt",
        help="Glob pattern for checkpoint files"
    )
    ap.add_argument(
        "--out", default="ed_results.csv",
        help="Output CSV path"
    )
    ap.add_argument(
        "--model-name", required=True,
        help="Model label for the 'model' column in CSV"
    )
    ap.add_argument("--log", default=None, help="Log file path")
    args = ap.parse_args()

    logger = setup_logger(args.log)

    if os.path.exists(args.out):
        os.remove(args.out)

    files = sorted(glob.glob(args.pattern), key=parse_index)
    if not files:
        logger.warning(f"No files matching pattern: {args.pattern}")
        return

    logger.info(f"Found {len(files)} checkpoint(s) matching {args.pattern}")

    write_header = True
    for fp in files:
        logger.info(f"Processing {fp} ...")
        try:
            process_one_bundle(fp, args.out, write_header, args.model_name, logger)
            write_header = False
        except Exception as e:
            logger.error(f"Skipping {fp} due to error: {e}")
            continue

    logger.info(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()

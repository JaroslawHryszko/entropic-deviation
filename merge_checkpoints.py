#!/usr/bin/env python3
"""
merge_checkpoints.py — Merge multiple .pt checkpoint files into one bundle.
"""
import argparse
import glob
import logging
import os
import torch


def setup_logger(log_file=None):
    logger = logging.getLogger("ed_merge")
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


def parse_index(fn):
    # expects filenames like: prefix_chkpt_{n}.pt
    base = os.path.basename(fn)
    parts = base.split('_chkpt_')
    if len(parts) != 2:
        return float('inf')
    num_part = parts[1].split('.pt')[0]
    try:
        return int(num_part)
    except Exception:
        return float('inf')


def main():
    p = argparse.ArgumentParser(
        description="Merge .pt checkpoint files into a single logits+meta bundle"
    )
    p.add_argument(
        "--pattern", required=True,
        help="Glob pattern for checkpoint files, e.g. 'logits_*_chkpt_*.pt'"
    )
    p.add_argument(
        "--output", default="logits_merged.pt",
        help="Output .pt file with merged logits and meta"
    )
    p.add_argument("--log", default=None, help="Log file path")
    args = p.parse_args()

    logger = setup_logger(args.log)

    files = sorted(glob.glob(args.pattern), key=parse_index)
    if not files:
        logger.warning(f"No files matching pattern: {args.pattern}")
        return

    logger.info(f"Found {len(files)} checkpoint(s) matching {args.pattern}")

    all_logits = []
    all_meta   = []

    for f in files:
        try:
            data = torch.load(f, map_location="cpu", weights_only=False)
            logits = data.get("logits", [])
            meta   = data.get("meta", [])
            all_logits.extend(logits)
            all_meta.extend(meta)
            logger.info(f"  Loaded {len(logits)} entries from {f}")
        except Exception as e:
            logger.error(f"  Failed to load {f}: {e}")
            continue

    torch.save({"logits": all_logits, "meta": all_meta}, args.output)
    logger.info(f"Merged {len(files)} files -> {args.output} ({len(all_logits)} entries)")


if __name__ == "__main__":
    main()

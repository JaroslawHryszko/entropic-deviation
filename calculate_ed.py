#!/usr/bin/env python3
"""
calculate_ed.py — Streaming checkpoint processor for Entropic Deviation.

Computes ED per token as normalized KL divergence from uniform,
aggregates to ED_mean and ED_std per generation.
"""
import torch, math, argparse, pandas as pd, glob, os, gc, re
from datetime import datetime

SIZE_MAP = {
    # Current models
    "Qwen-2.5-32B":    32_000_000_000,
    "Llama-3.3-70B":   70_000_000_000,
    "Gemma-2-27B":     27_000_000_000,
    # Legacy models
    "Llama-3-8B":       8_000_000_000,
    "Llama-3-8B-Q4":    8_000_000_000,
    "Mistral-7B":       7_300_000_000,
    "Phi-3-mini-4k":    3_800_000_000,
}


def entropic_deviation(logits):
    """Compute per-token ED: KL(softmax(logits) || uniform) / log(vocab_size)."""
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


def process_one_bundle(fp, out_csv, write_header, model_name):
    bundle = torch.load(fp, map_location='cpu', weights_only=False)
    seqs, meta = bundle["logits"], bundle["meta"]
    records = []
    for i, logits in enumerate(seqs):
        ed_t = entropic_deviation(logits)
        rec = meta[i].copy()
        rec["ED_mean"] = ed_t.mean().item()
        rec["ED_std"]  = ed_t.std().item()
        rec["model"]   = model_name
        rec["timestamp_processed"] = datetime.now().isoformat()
        rec["model_size"] = SIZE_MAP.get(model_name, None)
        rec["rank"] = i
        rec["chkpt_id"] = parse_index(fp)
        if "prompt" in rec and ":" in rec["prompt"]:
            rec["domain"] = rec["prompt"].split(":", 1)[0].strip()
        records.append(rec)
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
        "--model-name", default="Mistral-7B",
        help="Model label for the 'model' column in CSV"
    )
    args = ap.parse_args()

    if os.path.exists(args.out):
        os.remove(args.out)

    files = sorted(glob.glob(args.pattern), key=parse_index)
    if not files:
        print(f"No files matching pattern: {args.pattern}")
        return

    write_header = True
    for fp in files:
        print(f"Processing {fp} ...")
        process_one_bundle(fp, args.out, write_header, args.model_name)
        write_header = False

    print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()

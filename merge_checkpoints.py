#!/usr/bin/env python3
import argparse
import glob
import os
import torch

def parse_index(fn):
    # expects filenames like: prefix_chkpt_{n}.pt
    base = os.path.basename(fn)
    parts = base.split('_chkpt_')
    if len(parts) != 2:
        return float('inf')
    num_part = parts[1].split('.pt')[0]
    try:
        return int(num_part)
    except:
        return float('inf')

def main():
    p = argparse.ArgumentParser(
        description="Scal checkpointy .pt w jeden bundle logits+meta"
    )
    p.add_argument(
        "--pattern", required=True,
        help="Glob pattern for checkpoint files, e.g. 'logits_*_chkpt_*.pt'"
    )
    p.add_argument(
        "--output", default="logits_merged.pt",
        help="Docelowy plik .pt z połączonymi logits i meta"
    )
    args = p.parse_args()

    files = sorted(glob.glob(args.pattern), key=parse_index)
    if not files:
        print(f"Brak plików pasujących do wzorca {args.pattern}")
        return

    all_logits = []
    all_meta   = []

    for f in files:
        data = torch.load(f, map_location="cpu", weights_only=False)
        logits = data.get("logits", [])
        meta   = data.get("meta", [])
        all_logits.extend(logits)
        all_meta.extend(meta)
        print(f"  + załadowano {len(logits)} wpisów z {f}")

    torch.save({"logits": all_logits, "meta": all_meta}, args.output)
    print(f"Scalono {len(files)} plików → {args.output}")
    print(f"Łącznie wpisów: {len(all_logits)}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# ed.py
import torch, math, argparse, pandas as pd

def entropic_deviation(logits):
    p = torch.softmax(logits, dim=-1)
    n = p.size(-1)
    kl = torch.sum(p * (p.log() - math.log(1.0 / n)), dim=-1)
    return kl / math.log(n)              # [seq_len]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", default="logits.pt")
    ap.add_argument("--out", default="ed_results.csv")
    args = ap.parse_args()

    bundle = torch.load(args.logits)
    seqs, meta = bundle["logits"], bundle["meta"]
    records = []

    for i, logits in enumerate(seqs):
        ed_t = entropic_deviation(logits)            # [seq_len]
        rec = meta[i].copy()
        rec["ED_mean"] = ed_t.mean().item()
        rec["ED_std"]  = ed_t.std().item()
        rec["model"]   = "Llama-3-8B-Q4"
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(args.out, index=False)
    print("saved →", args.out)

if __name__ == "__main__":
    main()

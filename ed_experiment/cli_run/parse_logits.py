#!/usr/bin/env python
"""
Parses every *.log file produced by generate_cli.py,
collects full logit tensors, and stores a dict {logits, meta} to .pt.
"""
import argparse, re, torch, numpy as np, glob, os, json

LOGIT_RE = re.compile(r"^logits\s+\t(.+)$")

def parse_single(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LOGIT_RE.match(line)
            if m:
                rows.append(np.fromstring(m.group(1), sep=" "))
    return torch.tensor(rows, dtype=torch.float32)  # [seq, vocab]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="results/logits_raw")
    ap.add_argument("--out_pt",  default="results/logits.pt")
    args = ap.parse_args()

    logits_tensors, meta = [], []
    for fp in glob.glob(os.path.join(args.raw_dir, "*.log")):
        tensor = parse_single(fp)
        if tensor.numel() == 0:
            continue
        logits_tensors.append(tensor)
        # metadata from filename pXXXX_Ty.y.log
        base = os.path.basename(fp)
        p_idx = int(base[1:5])
        temp  = float(base.split("_T")[1][:-4])
        meta.append({"prompt_idx": p_idx, "temp": temp, "seq_len": tensor.shape[0]})

    torch.save({"logits": logits_tensors, "meta": meta}, args.out_pt)
    print("✅ logits.pt saved →", args.out_pt)

if __name__ == "__main__":
    main()

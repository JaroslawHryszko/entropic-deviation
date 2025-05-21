#!/usr/bin/env python3
# ed_stream.py – strumieniowe przetwarzanie checkpointów
import torch, math, argparse, pandas as pd, glob, os, gc, re
from datetime import datetime

def entropic_deviation(logits):
    p = torch.softmax(logits, dim=-1)
    n = p.size(-1)
    kl = torch.sum(p * (p.log() - math.log(1.0 / n)), dim=-1)
    return kl / math.log(n)

def parse_index(fn):
    """Wyciąga numer checkpointu z nazwy *chkpt_{N}.pt*"""
    m = re.search(r'_chkpt_(\d+)\.pt$', fn)
    return int(m.group(1)) if m else float('inf')

def process_one_bundle(fp, out_csv, write_header, model_name="Llama-3-8B-Q4"):
    bundle = torch.load(fp, map_location='cpu')
    seqs, meta = bundle["logits"], bundle["meta"]
    records = []
    for i, logits in enumerate(seqs):
        ed_t = entropic_deviation(logits)
        rec = meta[i].copy()
        rec["ED_mean"] = ed_t.mean().item()
        rec["ED_std"]  = ed_t.std().item()
        rec["model"]   = model_name
        rec["timestamp_processed"] = datetime.now().isoformat()
        records.append(rec)
    df = pd.DataFrame(records)
    df.to_csv(out_csv, mode='a', header=write_header, index=False)
    # zwolnij pamięć
    del bundle, seqs, meta, records, df
    gc.collect()

def main():
    ap = argparse.ArgumentParser(
        description="Strumieniowe obliczanie ED na checkpointach"
    )
    ap.add_argument(
        "--pattern", required=True,
        help="Glob pattern do checkpointów: e.g. 'logits_gpu*_chkpt_*.pt'"
    )
    ap.add_argument(
        "--out", default="ed_results.csv",
        help="Ścieżka do wyniku CSV"
    )
    ap.add_argument(
        "--model-name", default="Llama-3-8B-Q4",
        help="Etykieta pola 'model' w CSV"
    )
    args = ap.parse_args()

    # usuń stary plik wynikowy, jeśli jest
    if os.path.exists(args.out):
        os.remove(args.out)

    # znajdź i posortuj pliki checkpoint
    files = sorted(glob.glob(args.pattern), key=parse_index)
    if not files:
        print(f"Brak plików pasujących do wzorca {args.pattern}")
        return

    write_header = True
    for fp in files:
        print(f"⏳ Przetwarzam {fp} …")
        process_one_bundle(fp, args.out, write_header, args.model_name)
        write_header = False

    print(f"✅ Zapisano wyniki do {args.out}")

if __name__ == "__main__":
    main()

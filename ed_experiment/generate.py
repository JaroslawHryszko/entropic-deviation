#!/usr/bin/env python
# generate.py
import argparse, json, os, itertools, torch, numpy as np, tqdm
from llama_cpp import Llama

def load_prompts(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line)["prompt"] if line.strip().startswith("{")
                else line.strip() for line in f]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)          # GGUF path
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--temps", nargs="+", type=float, default=[0.7,1.0,1.3])
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--out", default="logits.pt")
    args = ap.parse_args()

    llama = Llama(model_path=args.model, n_gpu_layers=-1, n_ctx=args.max_tokens+64)

    prompts = load_prompts(args.prompts)
    out_logits, meta = [], []

    for t, prompt in tqdm.tqdm(list(itertools.product(args.temps, prompts))):
        res = llama.create_completion(prompt, temperature=t,
                                      max_tokens=args.max_tokens,
                                      logits_all=True, stream=False)
        # logits shape: [seq_len, vocab]
        logits = np.array(res["logits"], dtype=np.float32)
        out_logits.append(torch.from_numpy(logits))
        meta.append({"prompt": prompt, "temp": t, "seq_len": logits.shape[0]})

    torch.save({"logits": torch.stack(out_logits), "meta": meta}, args.out)
    print(f"wrote {args.out} with {len(out_logits)} sequences")

if __name__ == "__main__":
    main()

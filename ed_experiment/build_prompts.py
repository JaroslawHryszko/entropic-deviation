#!/usr/bin/env python
"""
build_prompts.py  —  create prompts.jsonl (800 lines)
Domains and quotas:
  wiki       400
  news       200
  fiction    120
  code        80
Total        800
All sources under CC-BY, CC-BY-SA, MIT, or Public Domain.
"""

import random, json, os, re, argparse, tqdm
from datasets import load_dataset
import nltk


nltk.download("punkt", quiet=True)

DOMAINS = {
    "wiki":   ("wikipedia",        "20220301.en", 400),
    "news":   ("cc_news",          None,          200),
    "fiction":("ai2_arc",          "ARC-Challenge", 120),
    "code":   ("code_search_net",  "python",       80),
}


OUT_PATH = "prompts/prompts.jsonl"
os.makedirs("prompts", exist_ok=True)

def to_sentences(text, target_len=1, max_len=40):
    """Return candidate sentences limited by length."""
    out = []
    for s in nltk.sent_tokenize(text):
        w = s.strip().replace("\n", " ")
        if target_len <= len(w.split()) <= max_len:
            out.append(w)
    return out

def sample_wiki(n):
    ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True, trust_remote_code=True)
    sents = []
    for row in ds.take(50_000):
        sents.extend(to_sentences(row["text"]))
        if len(sents) > n*3:
            break
    return random.sample(sents, n)

def sample_cc_news(n):
    ds = load_dataset("cc_news", split="train", streaming=True, trust_remote_code=True)
    sents = []
    for row in ds.take(20_000):
        sents.extend(to_sentences(row["text"]))
        if len(sents) > n*3:
            break
    return random.sample(sents, n)

def sample_arc(n):
    import json
    ds = load_dataset("ai2_arc", "ARC-Challenge", split="train", streaming=True)
    sents = []
    for row in ds.take(10_000):
        if isinstance(row, str):  # JSON string in streaming mode
            row = json.loads(row)

        q = row.get("question", {}).get("stem", "")
        a = row.get("answerKey", "")
        if q and a and len(q.split()) > 4:
            sents.append(f"{q.strip()} (Answer: {a.strip()})")
        if len(sents) >= n:
            break
    return random.sample(sents, n)



def sample_code(n):
    ds = load_dataset("code_search_net", "python", split="train", streaming=True, trust_remote_code=True)
    prompts = []
    for row in ds.take(n*3):
        code = row["code"]
        # take the first docstring line or function signature
        m = re.search(r'"""(.*?)"""', code, re.S)
        snippet = m.group(1).strip().split("\n")[0] if m else code.split("\n")[0]
        snippet = snippet.strip()
        if 3 <= len(snippet.split()) <= 30:
            prompts.append(f"Explain what this code does:\n{snippet}")
        if len(prompts) >= n:
            break
    return random.sample(prompts, n)

SAMPLERS = {
    "wiki": sample_wiki,
    "news": sample_cc_news,
    "fiction": sample_arc,
    "code": sample_code,
}

def main():
    random.seed(42)
    all_prompts = []
    for dom, (ds_name, cfg, quota) in DOMAINS.items():
        print(f"sampling {quota} from {ds_name} …")
        samples = SAMPLERS[dom](quota)
        for p in samples:
            all_prompts.append({
                "prompt": p,
                "domain": dom,
                "len": len(p.split())
            })

    random.shuffle(all_prompts)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for rec in all_prompts:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"✅  wrote {len(all_prompts)} prompts → {OUT_PATH}")

if __name__ == "__main__":
    main()

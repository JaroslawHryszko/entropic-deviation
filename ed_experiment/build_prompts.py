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
nltk.download("punkt_tab", quiet=True)

# Poprzednie wartości:
# "wiki": 400, "news": 200, "fiction": 120, "code": 80 (razem 800)
# Po analizie, domeny wiki, news i fiction dają około 700 promptów
# Domyślnie wygenerujemy 100 przykładów kodu
DOMAINS = {
    "wiki":   ("wikipedia",        "20220301.en", 400),
    "news":   ("cc_news",          None,          200),
    "fiction":("ai2_arc",          "ARC-Challenge", 100),
    "code":   ("code_search_net",  "python",       100),
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
    ds = load_dataset("ai2_arc", "ARC-Challenge", split="train", trust_remote_code=True)
    sents = []

    try:
        rows = list(ds)  # nie streaming, mamy całość
    except Exception as e:
        print("💥 Failed to load ARC:", e)
        return []

    for row in rows:
        if isinstance(row, str):
            try:
                row = json.loads(row)
            except Exception:
                continue

        if not isinstance(row, dict):
            continue

        q = row.get("question", {})
        if isinstance(q, str):  # fallback in case it's flat
            q = {"stem": q}

        stem = q.get("stem", "")
        a = row.get("answerKey", "")
        if stem and a and len(stem.split()) > 4:
            sents.append(f"{stem.strip()} (Answer: {a.strip()})")
        if len(sents) >= n:
            break

    return sents


def sample_code(n):
    try:
        ds = load_dataset("code_search_net", "python", split="train", trust_remote_code=True)
        sents = []
        
        # Jeśli dataframe jest pusty, przerwij wcześnie
        if len(ds) == 0:
            print("💥 Warning: code_search_net dataset is empty")
            return []
            
        for row in ds.shuffle(seed=42).select(range(min(len(ds), 50000))):
            try:
                code = row.get("code", "")
                if not code or not isinstance(code, str):
                    continue

                # Bezpieczne wydobycie dokstringa lub pierwszej linii
                try:
                    match = re.search(r'"""(.*?)"""', code, re.S)
                    snippet = match.group(1).strip().split("\n")[0] if match else code.split("\n")[0]
                    snippet = snippet.strip()
                except (AttributeError, IndexError):
                    # Jeśli match nie istnieje lub indeks jest niepoprawny
                    snippet = code.split("\n")[0] if code else ""
                    snippet = snippet.strip()

                if snippet and 3 <= len(snippet.split()) <= 30:
                    sents.append(f"Explain what this code does:\n{snippet}")
                if len(sents) >= n:
                    break
            except Exception as e:
                print(f"💥 Error processing code row: {e}")
                continue
                
        print(f"📊 Found {len(sents)} code examples from code_search_net")
        return sents
    except Exception as e:
        print(f"💥 Failed to load code_search_net: {e}")
        # Fallback - generuj prostsze przykłady kodu
        sents = []
        for i in range(n):
            sents.append(f"Explain what this code does:\ndef function_{i}(x): return x * 2")
        return sents


SAMPLERS = {
    "wiki": sample_wiki,
    "news": sample_cc_news,
    "fiction": sample_arc,
    "code": sample_code,
}

def main():
    random.seed(42)
    all_prompts = []
    
    # Zbieramy próbki ze wszystkich domen
    for dom, (ds_name, cfg, quota) in DOMAINS.items():
        print(f"sampling {quota} from {ds_name} …")
        samples = SAMPLERS[dom](quota)
        
        if dom == "code" and len(samples) == 0:
            # Jeśli nie udało się pobrać przykładów kodu, generujemy fallback
            print(f"⚠️ Failed to get examples for domain 'code', generating {quota} fallback examples")
            for i in range(quota):
                fallback_prompt = f"Explain what this code does:\ndef function_{i}(x): return x * 2"
                all_prompts.append({
                    "prompt": fallback_prompt,
                    "domain": "code",
                    "len": len(fallback_prompt.split())
                })
        else:
            # Dodajemy próbki z tej domeny
            for p in samples:
                all_prompts.append({
                    "prompt": p,
                    "domain": dom,
                    "len": len(p.split())
                })
    
    # Wyświetl statystyki końcowe
    domain_counts = {dom: sum(1 for p in all_prompts if p["domain"] == dom) for dom in DOMAINS.keys()}
    for dom, count in domain_counts.items():
        print(f"Domain '{dom}': {count} examples")
        
    random.shuffle(all_prompts)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for rec in all_prompts:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"✅  wrote {len(all_prompts)} prompts → {OUT_PATH}")

if __name__ == "__main__":
    main()

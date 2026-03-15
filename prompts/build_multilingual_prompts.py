#!/usr/bin/env python3
"""
build_multilingual_prompts.py — Generate semantic prompts in multiple languages
for cross-lingual Entropic Deviation analysis.

Sources:
  - Wikipedia in each target language (CC-BY-SA)
  - Code prompts are language-independent (reused from English)

Output: prompts/prompts_multilingual.jsonl
Format: {"prompt": "wiki_pl: ...", "domain": "wiki_pl", "lang": "pl", "len": N}

Languages:
  en  — English (baseline, from existing prompts.jsonl)
  pl  — Polish (Slavic, Latin script)
  zh  — Chinese (Sino-Tibetan, CJK script)
  ar  — Arabic (Semitic, RTL Arabic script)
  ja  — Japanese (Japonic, mixed script: kanji + kana)

Design rationale:
  Four non-English languages chosen for maximum diversity:
  different language families, different scripts, different tokenization.
  If ED is stable across these, it's strong evidence for architecture-level
  non-randomness independent of linguistic structure.

Usage:
    pip install datasets nltk
    python prompts/build_multilingual_prompts.py

    # Then run with Qwen-32B (which supports all these languages):
    python generate_logits.py \
        --model models/Qwen2.5-32B-Instruct-Q4_K_M.gguf \
        --prompts prompts/prompts_multilingual.jsonl \
        --temps 0.7 1.0 1.3 \
        --ed-out results/ed_qwen32b_multilingual.csv \
        --model-name "Qwen-2.5-32B" \
        --shuffle
"""
import random
import json
import os
import sys

# Optional: nltk for sentence tokenization (English, some European)
try:
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

from datasets import load_dataset

# --- Configuration --------------------------------------------------------

QUOTA_PER_LANG = 200   # prompts per language
SEED = 42

# Wikipedia dataset configs per language.
# Use wikimedia/wikipedia (official, newer dumps) with 20231101 snapshots.
# Fallback: HuggingFaceFW/finewiki if wikimedia/wikipedia fails.
WIKI_CONFIGS = {
    "pl": ("wikimedia/wikipedia", "20231101.pl"),
    "zh": ("wikimedia/wikipedia", "20231101.zh"),
    "ar": ("wikimedia/wikipedia", "20231101.ar"),
    "ja": ("wikimedia/wikipedia", "20231101.ja"),
}

OUT_PATH = "prompts/prompts_multilingual.jsonl"

# --- Sentence extraction --------------------------------------------------

def split_sentences_generic(text, lang):
    """Split text into sentences. Uses nltk for Latin-script languages,
    simple heuristics for CJK and Arabic."""
    text = text.strip().replace("\n", " ")

    if lang in ("en", "pl") and HAS_NLTK:
        return nltk.sent_tokenize(text)

    if lang == "zh":
        # Chinese sentence boundaries
        import re
        sents = re.split(r'([。！？；])', text)
        result = []
        for i in range(0, len(sents) - 1, 2):
            s = (sents[i] + sents[i + 1]).strip()
            if s:
                result.append(s)
        # Handle trailing text
        if len(sents) % 2 == 1 and sents[-1].strip():
            result.append(sents[-1].strip())
        return result

    if lang == "ja":
        # Japanese sentence boundaries
        import re
        sents = re.split(r'([。！？])', text)
        result = []
        for i in range(0, len(sents) - 1, 2):
            s = (sents[i] + sents[i + 1]).strip()
            if s:
                result.append(s)
        if len(sents) % 2 == 1 and sents[-1].strip():
            result.append(sents[-1].strip())
        return result

    if lang == "ar":
        # Arabic sentence boundaries
        import re
        sents = re.split(r'([.؟!،؛])', text)
        result = []
        for i in range(0, len(sents) - 1, 2):
            s = (sents[i] + sents[i + 1]).strip()
            if s:
                result.append(s)
        if len(sents) % 2 == 1 and sents[-1].strip():
            result.append(sents[-1].strip())
        return result

    # Fallback: split on period
    return [s.strip() for s in text.split(".") if s.strip()]


def is_valid_sentence(s, lang, min_chars=20, max_chars=300):
    """Filter sentences by length. Uses character count for CJK,
    word count for others."""
    s = s.strip()
    if not s:
        return False

    if lang in ("zh", "ja"):
        # CJK: character count (no spaces between words)
        return min_chars // 3 <= len(s) <= max_chars // 2
    else:
        # Space-separated: word count
        words = len(s.split())
        return 5 <= words <= 40


def sample_wiki(lang, config, n, max_articles=50000):
    """Sample n sentences from Wikipedia in given language."""
    ds_name, ds_config = config
    print(f"  Loading {ds_name}/{ds_config}...")

    try:
        ds = load_dataset(ds_name, ds_config, split="train",
                          streaming=True, trust_remote_code=True)
    except Exception as e1:
        # Fallback to HuggingFaceFW/finewiki
        print(f"  Primary source failed ({e1}), trying HuggingFaceFW/finewiki...")
        try:
            ds = load_dataset("HuggingFaceFW/finewiki", lang, split="train",
                              streaming=True, trust_remote_code=True)
        except Exception as e2:
            print(f"  Fallback also failed: {e2}")
            return []

    sents = []
    for row in ds.take(max_articles):
        text = row.get("text", "")
        if not text:
            continue
        for s in split_sentences_generic(text, lang):
            if is_valid_sentence(s, lang):
                sents.append(s)
        if len(sents) > n * 3:
            break

    if len(sents) < n:
        print(f"  WARNING: only got {len(sents)} sentences for {lang} "
              f"(needed {n})")
        return sents

    return random.sample(sents, n)


# --- Main -----------------------------------------------------------------

def main():
    random.seed(SEED)
    os.makedirs("prompts", exist_ok=True)

    all_prompts = []

    for lang, config in WIKI_CONFIGS.items():
        print(f"Sampling {QUOTA_PER_LANG} from Wikipedia/{lang}...")
        try:
            sentences = sample_wiki(lang, config, QUOTA_PER_LANG)
        except Exception as e:
            print(f"  ERROR: Failed to load Wikipedia/{lang}: {e}")
            continue

        domain = f"wiki_{lang}"
        for s in sentences:
            all_prompts.append({
                "prompt": f"{domain}: {s}",
                "domain": domain,
                "lang": lang,
                "len": len(s.split()) if lang not in ("zh", "ja") else len(s),
            })
        print(f"  Got {len(sentences)} prompts for {lang}")

    # Shuffle
    random.shuffle(all_prompts)

    # Write
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for rec in all_prompts:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Summary
    from collections import Counter
    lang_counts = Counter(p["lang"] for p in all_prompts)
    print(f"\nWrote {len(all_prompts)} prompts → {OUT_PATH}")
    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang}: {count}")


if __name__ == "__main__":
    main()

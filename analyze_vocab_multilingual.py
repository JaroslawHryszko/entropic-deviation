#!/usr/bin/env python3
"""
Analyze effective vocabulary per language for Qwen2.5-32B-Instruct.

Measures tokenizer vocabulary allocation and prompt-side token usage
per language to test H_vocab vs H_deeper for the ED gradient:
    EN (0.329) < JA (0.388) < ZH (0.395) < PL (0.401) < AR (0.408)

H_vocab: ED gradient is explained by tokenizer vocab allocation per language.
H_deeper: ED gradient persists after vocab correction -> structural effect.
"""

import pandas as pd
import numpy as np
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from transformers import AutoTokenizer
from scipy.stats import spearmanr

# --- Config ---
MULTILINGUAL_CSV = "results/ed_Qwen2.5-32B-Instruct_multilingual_20260315_212747.csv"
ENGLISH_CSV = "results/ed_Qwen2.5-32B-Instruct_prompts_20260304_235106.csv"
OUTPUT_CSV = "results/vocab_analysis_multilingual.csv"
TOKENIZER_NAME = "Qwen/Qwen2.5-32B-Instruct"

LANGUAGES = ["en", "pl", "ja", "zh", "ar"]
ED_MEANS = {"en": 0.329, "ja": 0.388, "zh": 0.395, "pl": 0.401, "ar": 0.408}


# ---------------------------------------------------------------------------
# Unicode script classification
# ---------------------------------------------------------------------------

# Arabic script ranges (main block + supplements + presentation forms)
ARABIC_RANGES = [
    (0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF), (0xFE70, 0xFEFF),
]

# CJK Unified Ideographs (shared by Chinese and Japanese)
CJK_RANGES = [
    (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F), (0x2B740, 0x2B81F), (0xF900, 0xFAFF),
    (0x2F800, 0x2FA1F),
]

# Latin script (Basic Latin letters through Latin Extended + supplement)
LATIN_RANGES = [
    (0x0041, 0x005A), (0x0061, 0x007A),   # Basic A-Z, a-z
    (0x00C0, 0x024F),                       # Latin Extended-A/B + supplement
    (0x1E00, 0x1EFF),                       # Latin Extended Additional
    (0x2C60, 0x2C7F),                       # Latin Extended-C
    (0xA720, 0xA7FF),                       # Latin Extended-D
]

POLISH_DIACRITICS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")


def in_ranges(cp, ranges):
    return any(lo <= cp <= hi for lo, hi in ranges)


def classify_char(cp):
    """Classify a single Unicode codepoint to a script category."""
    cat = unicodedata.category(chr(cp))

    # Whitespace / control
    if cat.startswith('Z') or cat.startswith('C'):
        return "control"
    # Punctuation, symbols, digits -> shared
    if cat.startswith('P') or cat.startswith('S') or cat.startswith('N'):
        return "shared"

    if in_ranges(cp, ARABIC_RANGES):
        return "arabic"
    if in_ranges(cp, CJK_RANGES):
        return "cjk"
    if 0x3040 <= cp <= 0x309F:
        return "hiragana"
    if 0x30A0 <= cp <= 0x30FF:
        return "katakana"
    if in_ranges(cp, LATIN_RANGES):
        return "latin"
    if 0x0400 <= cp <= 0x052F:
        return "cyrillic"
    if 0x0370 <= cp <= 0x03FF:
        return "greek"
    if 0x1100 <= cp <= 0x11FF or 0xAC00 <= cp <= 0xD7AF:
        return "hangul"
    if 0x0900 <= cp <= 0x097F:
        return "devanagari"
    if 0x0E00 <= cp <= 0x0E7F:
        return "thai"
    return "other"


def classify_token_script(text):
    """Return the dominant Unicode script of a decoded token."""
    if not text:
        return "empty"
    scripts = Counter()
    for ch in text:
        scripts[classify_char(ord(ch))] += 1
    if not scripts:
        return "empty"
    return scripts.most_common(1)[0][0]


# Script -> which languages can use tokens of this script
SCRIPT_TO_LANGS = {
    "latin":      ["en", "pl"],
    "arabic":     ["ar"],
    "cjk":        ["zh", "ja"],
    "hiragana":   ["ja"],
    "katakana":   ["ja"],
    "shared":     LANGUAGES,
    "control":    LANGUAGES,
    "cyrillic":   [],
    "greek":      [],
    "hangul":     [],
    "devanagari": [],
    "thai":       [],
    "other":      [],
    "empty":      [],
}


# ---------------------------------------------------------------------------
# Part 1: Tokenizer vocabulary analysis
# ---------------------------------------------------------------------------

def analyze_tokenizer_vocab(tokenizer):
    """Partition the full vocabulary by Unicode script and map to languages."""
    vocab_size = len(tokenizer)
    print(f"Total vocabulary size: {vocab_size:,}")

    script_counts = Counter()
    lang_tokens = defaultdict(set)
    polish_specific_tokens = set()

    for tid in range(vocab_size):
        try:
            text = tokenizer.decode([tid])
        except Exception:
            continue

        script = classify_token_script(text)
        script_counts[script] += 1

        for lang in SCRIPT_TO_LANGS.get(script, []):
            lang_tokens[lang].add(tid)

        # Track Polish-specific tokens (Latin tokens with Polish diacritics)
        if script == "latin" and any(c in POLISH_DIACRITICS for c in text):
            polish_specific_tokens.add(tid)

    print("\n--- Vocabulary by script ---")
    for script, count in sorted(script_counts.items(), key=lambda x: -x[1]):
        print(f"  {script:12s}: {count:>7,} ({count / vocab_size * 100:5.1f}%)")

    print(f"\n  Polish-specific (with diacritics): {len(polish_specific_tokens):,}")

    print("\n--- Effective vocab allocated per language ---")
    for lang in LANGUAGES:
        n = len(lang_tokens[lang])
        print(f"  {lang}: {n:>7,} ({n / vocab_size * 100:5.1f}%)")

    return lang_tokens


# ---------------------------------------------------------------------------
# Part 2: Prompt tokenization analysis
# ---------------------------------------------------------------------------

def strip_domain_prefix(prompt, lang):
    """Remove domain prefix(es) from prompt text."""
    prefixes = [
        f"wiki_{lang}: wiki_{lang}: ",   # double prefix (known bug)
        f"wiki_{lang}: ",
        "wiki: wiki: ",
        "wiki: ",
    ]
    for pfx in prefixes:
        if prompt.startswith(pfx):
            return prompt[len(pfx):]
    return prompt


def analyze_prompts(tokenizer, lang_tokens):
    """Tokenize prompts per language and compute usage statistics."""
    df_multi = pd.read_csv(MULTILINGUAL_CSV)
    df_multi["language"] = df_multi["domain"].str.replace("wiki_", "")

    df_en = pd.read_csv(ENGLISH_CSV)
    df_en_wiki = df_en[df_en["domain"] == "wiki"].copy()
    df_en_wiki["language"] = "en"

    rows = []

    for lang in LANGUAGES:
        df_lang = df_en_wiki if lang == "en" else df_multi[df_multi["language"] == lang]
        prompts = df_lang["prompt"].unique()

        all_token_ids = []        # every token occurrence (for frequency concentration)
        unique_tokens = set()
        token_counts = []
        fertilities = []

        for prompt in prompts:
            clean = strip_domain_prefix(prompt, lang)
            tids = tokenizer.encode(clean, add_special_tokens=False)

            all_token_ids.extend(tids)
            unique_tokens.update(tids)
            token_counts.append(len(tids))

            if len(clean) > 0:
                fertilities.append(len(tids) / len(clean))

        # Token-frequency-based concentration (proxy for model-side concentration)
        freq = Counter(all_token_ids)
        total = sum(freq.values())
        sorted_freqs = sorted(freq.values(), reverse=True)

        def topk_conc(k):
            return sum(sorted_freqs[:min(k, len(sorted_freqs))]) / total

        conc_100 = topk_conc(100)
        conc_1000 = topk_conc(1000)
        conc_5000 = topk_conc(5000)

        row = {
            "language":                    lang,
            "ED_mean":                     ED_MEANS[lang],
            "vocab_allocated":             len(lang_tokens.get(lang, set())),
            "vocab_used":                  len(unique_tokens),
            "vocab_concentration_top100":  round(conc_100, 4),
            "vocab_concentration_top1000": round(conc_1000, 4),
            "vocab_concentration_top5000": round(conc_5000, 4),
            "mean_fertility":              round(np.mean(fertilities), 4),
            "mean_prompt_tokens":          round(np.mean(token_counts), 1),
        }
        rows.append(row)

        print(f"\n--- {lang.upper()} (ED={ED_MEANS[lang]}) ---")
        print(f"  Prompts:              {len(prompts)}")
        print(f"  Vocab allocated:      {row['vocab_allocated']:,}")
        print(f"  Vocab used (prompts): {row['vocab_used']:,}")
        print(f"  Mean prompt tokens:   {row['mean_prompt_tokens']}")
        print(f"  Mean fertility:       {row['mean_fertility']} tok/char")
        print(f"  Concentration (freq): top100={conc_100:.3f}  "
              f"top1000={conc_1000:.3f}  top5000={conc_5000:.3f}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 3: Correlations and interpretation
# ---------------------------------------------------------------------------

def compute_correlations(df):
    """Spearman correlations between vocab metrics and ED (n=5)."""
    print("\n--- Spearman correlations with ED_mean (n=5) ---")
    print("  (n=5 limits statistical power; interpret direction, not p-value)\n")

    cols = [
        "vocab_allocated", "vocab_used",
        "vocab_concentration_top100", "vocab_concentration_top1000",
        "mean_fertility", "mean_prompt_tokens",
    ]
    for col in cols:
        rho, p = spearmanr(df["ED_mean"], df[col])
        print(f"  ED vs {col:30s}  rho={rho:+.3f}  p={p:.3f}")


def interpret(df):
    """Print brief interpretation."""
    en_alloc = df.loc[df["language"] == "en", "vocab_allocated"].iloc[0]
    pl_alloc = df.loc[df["language"] == "pl", "vocab_allocated"].iloc[0]

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)

    if en_alloc == pl_alloc:
        print(
            f"\n  EN and PL share identical vocab allocation ({en_alloc:,} tokens)"
            f"\n  yet ED differs: EN={ED_MEANS['en']:.3f} vs PL={ED_MEANS['pl']:.3f}."
            f"\n  -> Vocab allocation ALONE cannot explain the full ED gradient."
            f"\n  -> H_deeper has partial support: language structure matters"
            f"\n     beyond raw tokenizer coverage."
        )

    rho_alloc, _ = spearmanr(df["ED_mean"], df["vocab_allocated"])
    if rho_alloc < -0.5:
        print(
            f"\n  Vocab allocated anti-correlates with ED (rho={rho_alloc:+.3f})."
            f"\n  Languages with fewer allocated tokens tend to have higher ED."
            f"\n  -> H_vocab has partial support for cross-script comparisons"
            f"\n     (AR, JA, ZH vs EN/PL)."
        )
    else:
        print(
            f"\n  Vocab allocated does NOT strongly anti-correlate with ED"
            f"\n  (rho={rho_alloc:+.3f})."
            f"\n  -> H_vocab is NOT supported; ED gradient driven by other factors."
        )

    print(
        "\n  NOTE: vocab_concentration here is frequency-based (from prompt tokens),"
        "\n  not probability-mass-based. For true model-side concentration,"
        "\n  generation logits would be needed (re-run with checkpoint saving)."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading Qwen2.5-32B-Instruct tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

    print("\n" + "=" * 60)
    print("PART 1: Tokenizer Vocabulary Partition by Script")
    print("=" * 60)
    lang_tokens = analyze_tokenizer_vocab(tokenizer)

    print("\n" + "=" * 60)
    print("PART 2: Prompt Tokenization Analysis")
    print("=" * 60)
    df_results = analyze_prompts(tokenizer, lang_tokens)

    print("\n" + "=" * 60)
    print("PART 3: Correlations")
    print("=" * 60)
    compute_correlations(df_results)

    interpret(df_results)

    # Save
    output_cols = [
        "language", "vocab_used", "vocab_concentration_top100",
        "vocab_concentration_top1000", "mean_fertility", "mean_prompt_tokens",
    ]
    # Also save full version with extra columns
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFull results saved to {OUTPUT_CSV}")
    print("\n" + df_results.to_string(index=False))


if __name__ == "__main__":
    main()

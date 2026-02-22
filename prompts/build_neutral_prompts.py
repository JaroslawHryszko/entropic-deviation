#!/usr/bin/env python3
"""
build_neutral_prompts.py — generate prompts that minimise semantic constraint
on the output distribution, for baseline ED measurement.

Categories (200 prompts each, 1000 total):
  empty      — empty string / whitespace / BOS-only contexts
  random     — random token sequences with no semantic coherence
  explicit   — explicit requests for random/uniform output
  neutral    — minimal semantic stubs ("Continue:", "Output:", etc.)
  nonsense   — phonotactically plausible but meaningless strings

The key idea: if ED remains elevated under these conditions,
non-randomness is model-intrinsic (H1). If ED collapses, the patterns
observed with domain prompts are prompt-driven (H2).
"""

import json
import os
import random
import string

random.seed(42)

OUT_PATH = "prompts/prompts_neutral.jsonl"
os.makedirs("prompts", exist_ok=True)

N_PER_CATEGORY = 200


# ---------------------------------------------------------------------------
# Category 1: empty — minimal or empty contexts
# ---------------------------------------------------------------------------
def build_empty(n):
    prompts = []
    variants = [
        "",
        " ",
        "  ",
        "\n",
        "\t",
        ".",
        "...",
        "-",
        "—",
        "_",
    ]
    for i in range(n):
        prompts.append(variants[i % len(variants)])
    return prompts


# ---------------------------------------------------------------------------
# Category 2: random — random ASCII / token-like sequences
# ---------------------------------------------------------------------------
def build_random(n):
    prompts = []
    charset = string.ascii_letters + string.digits + " " * 5
    for _ in range(n):
        length = random.randint(10, 60)
        seq = "".join(random.choice(charset) for _ in range(length))
        prompts.append(seq)
    return prompts


# ---------------------------------------------------------------------------
# Category 3: explicit — direct requests for random generation
# ---------------------------------------------------------------------------
def build_explicit(n):
    templates = [
        "Generate a sequence of random characters:",
        "Output random letters and numbers:",
        "Produce a uniformly random string:",
        "Simulate rolling a six-sided die 20 times:",
        "Generate random words with no meaning:",
        "Output a random sequence of tokens:",
        "Write a string of random ASCII characters:",
        "Produce random text with no coherent meaning:",
        "Generate 50 random characters:",
        "Simulate a random number generator, output digits:",
        "List random letters from the alphabet:",
        "Create a nonsensical random sequence:",
        "Output random bytes as text:",
        "Generate uniformly distributed random text:",
        "Write random characters without forming words:",
        "Produce a stream of random symbols:",
        "Simulate white noise as text:",
        "Output random Unicode characters:",
        "Generate text with maximum entropy:",
        "Write a sequence with no pattern or meaning:",
    ]
    prompts = []
    for i in range(n):
        prompts.append(templates[i % len(templates)])
    return prompts


# ---------------------------------------------------------------------------
# Category 4: neutral — minimal semantic stubs
# ---------------------------------------------------------------------------
def build_neutral(n):
    stubs = [
        "Continue:",
        "Output:",
        "Text:",
        "Next:",
        "Begin:",
        "Start:",
        "Write:",
        "Generate:",
        "Produce:",
        "Result:",
        "Response:",
        "Answer:",
        "Content:",
        "Data:",
        "Sequence:",
        "The",
        "A",
        "It",
        "This",
        "One",
    ]
    prompts = []
    for i in range(n):
        prompts.append(stubs[i % len(stubs)])
    return prompts


# ---------------------------------------------------------------------------
# Category 5: nonsense — phonotactically plausible gibberish
# ---------------------------------------------------------------------------
def build_nonsense(n):
    onsets = ["bl", "cr", "dr", "fl", "gr", "kl", "pr", "sl", "tr", "sp",
              "st", "br", "fr", "gl", "pl", "sk", "sm", "sn", "sw", "tw",
              "b", "c", "d", "f", "g", "h", "j", "k", "l", "m",
              "n", "p", "r", "s", "t", "v", "w", "z"]
    vowels = ["a", "e", "i", "o", "u", "ai", "ou", "ea", "ie", "oo"]
    codas = ["n", "m", "l", "r", "s", "t", "k", "ng", "nt", "nd", ""]

    def random_word():
        syl_count = random.randint(1, 3)
        word = ""
        for _ in range(syl_count):
            word += random.choice(onsets) + random.choice(vowels) + random.choice(codas)
        return word

    prompts = []
    for _ in range(n):
        word_count = random.randint(3, 10)
        sentence = " ".join(random_word() for _ in range(word_count))
        prompts.append(sentence)
    return prompts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
BUILDERS = {
    "empty":    build_empty,
    "random":   build_random,
    "explicit": build_explicit,
    "neutral":  build_neutral,
    "nonsense": build_nonsense,
}


def main():
    all_prompts = []
    for domain, builder in BUILDERS.items():
        samples = builder(N_PER_CATEGORY)
        for p in samples:
            all_prompts.append({
                "prompt": p,
                "domain": domain,
                "len": len(p.split()) if p.strip() else 0,
            })
        print(f"  {domain}: {len(samples)} prompts")

    random.shuffle(all_prompts)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for rec in all_prompts:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_prompts)} neutral prompts -> {OUT_PATH}")


if __name__ == "__main__":
    main()

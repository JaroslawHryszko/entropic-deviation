"""Tests for prompt generation scripts."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestNeutralPrompts:
    """Tests for the neutral prompts builder and output."""

    def test_neutral_prompts_file_exists(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "prompts", "prompts_neutral.jsonl"
        )
        assert os.path.exists(path)

    def test_neutral_prompts_count(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "prompts", "prompts_neutral.jsonl"
        )
        with open(path) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 1000

    def test_neutral_prompts_five_domains(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "prompts", "prompts_neutral.jsonl"
        )
        domains = set()
        with open(path) as f:
            for line in f:
                if line.strip():
                    domains.add(json.loads(line)["domain"])
        assert domains == {"empty", "random", "explicit", "neutral", "nonsense"}

    def test_neutral_prompts_200_per_domain(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "prompts", "prompts_neutral.jsonl"
        )
        from collections import Counter
        counts = Counter()
        with open(path) as f:
            for line in f:
                if line.strip():
                    counts[json.loads(line)["domain"]] += 1
        for domain, count in counts.items():
            assert count == 200, f"{domain}: expected 200, got {count}"

    def test_neutral_prompts_valid_json(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "prompts", "prompts_neutral.jsonl"
        )
        with open(path) as f:
            for i, line in enumerate(f):
                if line.strip():
                    record = json.loads(line)
                    assert "prompt" in record, f"Line {i}: missing 'prompt'"
                    assert "domain" in record, f"Line {i}: missing 'domain'"
                    assert "len" in record, f"Line {i}: missing 'len'"


class TestDomainPrompts:
    """Tests for the domain prompts file."""

    def test_domain_prompts_file_exists(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "prompts", "prompts.jsonl"
        )
        assert os.path.exists(path)

    def test_domain_prompts_count(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "prompts", "prompts.jsonl"
        )
        with open(path) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 800

    def test_domain_prompts_four_domains(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "prompts", "prompts.jsonl"
        )
        domains = set()
        with open(path) as f:
            for line in f:
                if line.strip():
                    domains.add(json.loads(line)["domain"])
        assert domains == {"wiki", "news", "fiction", "code"}

    def test_domain_prompts_valid_json(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "prompts", "prompts.jsonl"
        )
        with open(path) as f:
            for i, line in enumerate(f):
                if line.strip():
                    record = json.loads(line)
                    assert "prompt" in record
                    assert "domain" in record
                    assert "len" in record

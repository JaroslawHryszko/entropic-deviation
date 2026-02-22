"""Tests for calculate_ed.py — ED computation and checkpoint processing."""
import math
import os

import pandas as pd
import pytest
import torch

# Import from project root
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calculate_ed import entropic_deviation, parse_index, process_one_bundle, SIZE_MAP


class TestEntropicDeviation:
    """Tests for the core ED calculation."""

    def test_uniform_logits_give_zero_ed(self, uniform_logits):
        """Uniform distribution should yield ED ≈ 0."""
        ed = entropic_deviation(uniform_logits)
        assert ed.shape == (10,)
        assert torch.allclose(ed, torch.zeros(10), atol=1e-5)

    def test_peaked_logits_give_high_ed(self, peaked_logits):
        """Peaked distribution should yield ED close to 1."""
        ed = entropic_deviation(peaked_logits)
        assert ed.shape == (10,)
        assert (ed > 0.9).all()

    def test_ed_bounded_zero_one(self, sample_logits):
        """ED values must be in [0, 1]."""
        ed = entropic_deviation(sample_logits)
        assert (ed >= 0).all()
        assert (ed <= 1.0 + 1e-6).all()

    def test_ed_shape_matches_seq_len(self):
        """Output shape should match number of tokens."""
        logits = torch.randn(42, 200)
        ed = entropic_deviation(logits)
        assert ed.shape == (42,)

    def test_ed_different_vocab_sizes(self):
        """ED should work with different vocabulary sizes."""
        for vocab_size in [100, 1000, 32000, 128256]:
            logits = torch.randn(5, vocab_size)
            ed = entropic_deviation(logits)
            assert ed.shape == (5,)
            assert (ed >= 0).all()
            assert (ed <= 1.0 + 1e-6).all()

    def test_ed_deterministic(self, sample_logits):
        """Same input should produce same output."""
        ed1 = entropic_deviation(sample_logits)
        ed2 = entropic_deviation(sample_logits)
        assert torch.allclose(ed1, ed2)

    def test_ed_single_token(self):
        """Should work with a single token."""
        logits = torch.randn(1, 100)
        ed = entropic_deviation(logits)
        assert ed.shape == (1,)


class TestParseIndex:
    """Tests for checkpoint filename parsing."""

    def test_standard_filename(self):
        assert parse_index("logits_chkpt_5.pt") == 5

    def test_with_prefix(self):
        assert parse_index("results/logits_qwen32b_chkpt_100.pt") == 100

    def test_no_match_returns_inf(self):
        assert parse_index("random_file.pt") == float('inf')

    def test_zero_index(self):
        assert parse_index("logits_chkpt_0.pt") == 0

    def test_large_index(self):
        assert parse_index("logits_chkpt_99999.pt") == 99999


class TestProcessOneBundle:
    """Tests for checkpoint-to-CSV processing."""

    def test_produces_csv(self, sample_checkpoint, tmp_dir):
        out_csv = str(tmp_dir / "test_ed.csv")
        process_one_bundle(str(sample_checkpoint), out_csv, True, "TestModel")
        assert os.path.exists(out_csv)
        df = pd.read_csv(out_csv)
        assert len(df) == 1
        assert "ED_mean" in df.columns
        assert "ED_std" in df.columns

    def test_csv_has_correct_columns(self, sample_checkpoint, tmp_dir):
        out_csv = str(tmp_dir / "test_ed.csv")
        process_one_bundle(str(sample_checkpoint), out_csv, True, "TestModel")
        df = pd.read_csv(out_csv)
        expected = {"ED_mean", "ED_std", "model", "model_size", "rank",
                    "chkpt_id", "domain", "prompt", "temp", "seq_len",
                    "gen_time", "timestamp", "timestamp_processed"}
        assert expected.issubset(set(df.columns))

    def test_domain_parsed_from_prompt(self, sample_checkpoint, tmp_dir):
        out_csv = str(tmp_dir / "test_ed.csv")
        process_one_bundle(str(sample_checkpoint), out_csv, True, "TestModel")
        df = pd.read_csv(out_csv)
        assert df.iloc[0]["domain"] == "wiki"

    def test_model_name_propagated(self, sample_checkpoint, tmp_dir):
        out_csv = str(tmp_dir / "test_ed.csv")
        process_one_bundle(str(sample_checkpoint), out_csv, True, "Qwen-2.5-32B")
        df = pd.read_csv(out_csv)
        assert df.iloc[0]["model"] == "Qwen-2.5-32B"
        assert df.iloc[0]["model_size"] == 32_000_000_000

    def test_unknown_model_gets_none_size(self, sample_checkpoint, tmp_dir):
        out_csv = str(tmp_dir / "test_ed.csv")
        process_one_bundle(str(sample_checkpoint), out_csv, True, "UnknownModel")
        df = pd.read_csv(out_csv)
        assert pd.isna(df.iloc[0]["model_size"])

    def test_append_mode(self, sample_checkpoint, tmp_dir):
        out_csv = str(tmp_dir / "test_ed.csv")
        process_one_bundle(str(sample_checkpoint), out_csv, True, "Model1")
        process_one_bundle(str(sample_checkpoint), out_csv, False, "Model2")
        df = pd.read_csv(out_csv)
        assert len(df) == 2

    def test_ed_values_reasonable(self, sample_checkpoint, tmp_dir):
        out_csv = str(tmp_dir / "test_ed.csv")
        process_one_bundle(str(sample_checkpoint), out_csv, True, "TestModel")
        df = pd.read_csv(out_csv)
        assert 0 <= df.iloc[0]["ED_mean"] <= 1
        assert df.iloc[0]["ED_std"] >= 0


class TestSizeMap:
    """Tests for the model size mapping."""

    def test_current_models_present(self):
        assert "Qwen-2.5-32B" in SIZE_MAP
        assert "Llama-3.3-70B" in SIZE_MAP
        assert "Gemma-2-27B" in SIZE_MAP

    def test_legacy_models_present(self):
        assert "Llama-3-8B" in SIZE_MAP
        assert "Mistral-7B" in SIZE_MAP
        assert "Phi-3-mini-4k" in SIZE_MAP

    def test_sizes_are_positive(self):
        for name, size in SIZE_MAP.items():
            assert size > 0, f"{name} has invalid size {size}"

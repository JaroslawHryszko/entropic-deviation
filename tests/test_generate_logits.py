"""Tests for generate_logits.py — helpers, ED inline computation, and CSV output."""
import json
import logging
import os
import sys

import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_logits import (
    setup_logger,
    load_prompts,
    find_resume_point,
    derive_model_name,
    parse_domain,
    flush_records,
    entropic_deviation,
    _parse_model_size,
)


# --- setup_logger() ---------------------------------------------------

class TestSetupLogger:
    def test_returns_logger(self):
        logger = setup_logger()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "ed_inference"

    def test_console_handler_present(self):
        logger = setup_logger()
        handler_types = [type(h) for h in logger.handlers]
        assert logging.StreamHandler in handler_types

    def test_file_handler_when_path_given(self, tmp_dir):
        log_path = str(tmp_dir / "test.log")
        logger = setup_logger(log_path)
        handler_types = [type(h) for h in logger.handlers]
        assert logging.FileHandler in handler_types

    def test_creates_log_directory(self, tmp_dir):
        log_path = str(tmp_dir / "subdir" / "test.log")
        setup_logger(log_path)
        assert os.path.isdir(str(tmp_dir / "subdir"))

    def test_no_duplicate_handlers(self):
        setup_logger()
        logger = setup_logger()
        stream_handlers = [h for h in logger.handlers
                          if isinstance(h, logging.StreamHandler)
                          and not isinstance(h, logging.FileHandler)]
        assert len(stream_handlers) == 1


# --- load_prompts() ---------------------------------------------------

class TestLoadPrompts:
    def test_load_jsonl(self, prompts_jsonl):
        logger = setup_logger()
        prompts = load_prompts(str(prompts_jsonl), logger)
        assert len(prompts) == 3
        assert prompts[0] == "wiki: The history of computing"
        assert prompts[1] == "news: Breaking news today"

    def test_load_plain_text(self, prompts_txt):
        logger = setup_logger()
        prompts = load_prompts(str(prompts_txt), logger)
        assert len(prompts) == 3
        assert "Hello world" in prompts

    def test_skips_blank_lines(self, prompts_txt):
        logger = setup_logger()
        prompts = load_prompts(str(prompts_txt), logger)
        assert "" not in prompts

    def test_jsonl_domain_prefix(self, prompts_jsonl):
        logger = setup_logger()
        prompts = load_prompts(str(prompts_jsonl), logger)
        for p in prompts:
            assert ": " in p

    def test_empty_file(self, tmp_dir):
        path = tmp_dir / "empty.jsonl"
        path.write_text("")
        logger = setup_logger()
        prompts = load_prompts(str(path), logger)
        assert prompts == []

    def test_malformed_json_skipped(self, tmp_dir):
        path = tmp_dir / "bad.jsonl"
        path.write_text('{"prompt": "good", "domain": "wiki", "len": 1}\n{bad json\n')
        logger = setup_logger()
        prompts = load_prompts(str(path), logger)
        assert len(prompts) == 1


# --- find_resume_point() (now CSV-based) ------------------------------

class TestFindResumePoint:
    def test_nonexistent_file_returns_zero(self, tmp_dir):
        assert find_resume_point(str(tmp_dir / "nonexistent.csv")) == 0

    def test_empty_csv_with_header_only(self, tmp_dir):
        path = tmp_dir / "ed.csv"
        path.write_text("prompt,temp,ED_mean\n")
        assert find_resume_point(str(path)) == 0

    def test_counts_data_rows(self, tmp_dir):
        path = tmp_dir / "ed.csv"
        path.write_text("prompt,temp,ED_mean\na,1.0,0.5\nb,0.7,0.6\nc,1.3,0.4\n")
        assert find_resume_point(str(path)) == 3

    def test_ignores_trailing_blank_lines(self, tmp_dir):
        path = tmp_dir / "ed.csv"
        path.write_text("prompt,temp,ED_mean\na,1.0,0.5\nb,0.7,0.6\n\n\n")
        assert find_resume_point(str(path)) == 2

    def test_single_row(self, tmp_dir):
        path = tmp_dir / "ed.csv"
        path.write_text("col1,col2\nval1,val2\n")
        assert find_resume_point(str(path)) == 1


# --- derive_model_name() ---------------------------------------------

class TestDeriveModelName:
    @pytest.mark.parametrize("path, expected", [
        ("models/gemma-2-27b-it-Q4_K_M.gguf", "gemma-2-27b-it"),
        ("models/Llama-3.3-70B-Instruct-Q4_K_M.gguf", "Llama-3.3-70B-Instruct"),
        ("/absolute/path/Qwen2.5-32B-Instruct-Q4_K_M.gguf", "Qwen2.5-32B-Instruct"),
        ("model.gguf", "model"),
    ])
    def test_derive_model_name(self, path, expected):
        assert derive_model_name(path) == expected


# --- parse_domain() ---------------------------------------------------

class TestParseDomain:
    @pytest.mark.parametrize("prompt, expected", [
        ("wiki: Some article text", "wiki"),
        ("news: Breaking news", "news"),
        ("code: def foo(): pass", "code"),
        ("fiction: Once upon a time", "fiction"),
        ("no domain prefix here", ""),
        ("empty: ", "empty"),
    ])
    def test_parse_domain(self, prompt, expected):
        assert parse_domain(prompt) == expected


# --- flush_records() --------------------------------------------------

class TestFlushRecords:
    def test_creates_file_with_header(self, tmp_dir):
        csv_path = str(tmp_dir / "out.csv")
        records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        flush_records(records, csv_path, write_header=True)
        lines = (tmp_dir / "out.csv").read_text().strip().split("\n")
        assert lines[0] == "a,b"
        assert len(lines) == 3

    def test_appends_without_header(self, tmp_dir):
        csv_path = str(tmp_dir / "out.csv")
        (tmp_dir / "out.csv").write_text("a,b\n1,2\n")
        records = [{"a": 5, "b": 6}]
        flush_records(records, csv_path, write_header=False)
        lines = (tmp_dir / "out.csv").read_text().strip().split("\n")
        assert lines[0] == "a,b"
        assert len(lines) == 3

    def test_empty_records_noop(self, tmp_dir):
        csv_path = str(tmp_dir / "out.csv")
        flush_records([], csv_path, write_header=True)
        assert not (tmp_dir / "out.csv").exists()

    def test_roundtrip_with_find_resume(self, tmp_dir):
        """flush_records → find_resume_point should count correctly."""
        csv_path = str(tmp_dir / "ed.csv")
        records = [{"ED_mean": 0.5, "temp": 1.0} for _ in range(7)]
        flush_records(records, csv_path, write_header=True)
        assert find_resume_point(csv_path) == 7


# --- entropic_deviation() (inline copy) ------------------------------

class TestEntropicDeviationInline:
    def test_uniform_logits_give_zero_ed(self, uniform_logits):
        ed = entropic_deviation(uniform_logits)
        assert ed.shape == (10,)
        assert torch.allclose(ed, torch.zeros(10), atol=1e-5)

    def test_peaked_logits_give_high_ed(self, peaked_logits):
        ed = entropic_deviation(peaked_logits)
        assert ed.shape == (10,)
        assert (ed > 0.9).all()

    def test_ed_bounded_zero_one(self, sample_logits):
        ed = entropic_deviation(sample_logits)
        assert (ed >= 0).all()
        assert (ed <= 1.0 + 1e-6).all()

    def test_no_nan_on_extreme_logits(self):
        logits = torch.randn(10, 300) * 100
        ed = entropic_deviation(logits)
        assert not torch.isnan(ed).any()

    def test_float16_input_upcasted(self):
        logits = torch.randn(10, 100).half()
        ed = entropic_deviation(logits)
        assert ed.dtype == torch.float32
        assert not torch.isnan(ed).any()

    def test_deterministic(self, sample_logits):
        ed1 = entropic_deviation(sample_logits)
        ed2 = entropic_deviation(sample_logits)
        assert torch.allclose(ed1, ed2)

    def test_higher_concentration_gives_higher_ed(self):
        flat = torch.randn(20, 100) * 0.1
        peaked = torch.randn(20, 100) * 10.0
        ed_flat = entropic_deviation(flat).mean()
        ed_peaked = entropic_deviation(peaked).mean()
        assert ed_peaked > ed_flat


# --- _parse_model_size() (inline copy) --------------------------------

class TestParseModelSizeInline:
    @pytest.mark.parametrize("name, expected", [
        ("Llama-3.3-70B-Instruct", 70_000_000_000),
        ("Qwen-2.5-32B-Instruct", 32_000_000_000),
        ("gemma-2-27b-it", 27_000_000_000),
        ("Phi-3-mini-4K", None),
        ("Llama-3-8B", 8_000_000_000),
        ("model-0.5B-small", 500_000_000),
    ])
    def test_parse_model_size(self, name, expected):
        assert _parse_model_size(name) == expected


# --- Consistency: inline vs calculate_ed copies -----------------------

class TestConsistencyWithCalculateEd:
    """Verify that generate_logits.py and calculate_ed.py ED functions agree."""

    def test_same_results(self, sample_logits):
        from calculate_ed import entropic_deviation as ed_standalone
        ed_inline = entropic_deviation(sample_logits)
        ed_standalone_result = ed_standalone(sample_logits)
        assert torch.allclose(ed_inline, ed_standalone_result)

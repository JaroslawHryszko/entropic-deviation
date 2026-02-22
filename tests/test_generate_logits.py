"""Tests for generate_logits.py — helpers and prompt loading."""
import json
import logging
import os
import sys
from unittest.mock import MagicMock

import pytest

# Mock llama_cpp before importing generate_logits
sys.modules["llama_cpp"] = MagicMock()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_logits import setup_logger, load_prompts, find_resume_point


class TestSetupLogger:
    """Tests for logger configuration."""

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
        """Calling setup_logger twice should not duplicate handlers."""
        setup_logger()
        logger = setup_logger()
        stream_handlers = [h for h in logger.handlers
                          if isinstance(h, logging.StreamHandler)
                          and not isinstance(h, logging.FileHandler)]
        assert len(stream_handlers) == 1


class TestLoadPrompts:
    """Tests for prompt file loading."""

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


class TestFindResumePoint:
    """Tests for checkpoint resume detection."""

    def test_no_checkpoints_returns_zero(self, tmp_dir):
        assert find_resume_point(str(tmp_dir / "nonexistent")) == 0

    def test_finds_highest_checkpoint(self, sample_checkpoint_pair):
        prefix = str(sample_checkpoint_pair[0]).replace("_chkpt_5.pt", "")
        idx = find_resume_point(prefix)
        assert idx == 10

    def test_single_checkpoint(self, sample_checkpoint):
        prefix = str(sample_checkpoint).replace("_chkpt_5.pt", "")
        idx = find_resume_point(prefix)
        assert idx == 5

"""Tests for merge_checkpoints.py — checkpoint merging."""
import os
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from merge_checkpoints import parse_index


class TestMergeParseIndex:
    """Tests for the merge_checkpoints parse_index function."""

    def test_standard_filename(self):
        assert parse_index("logits_chkpt_5.pt") == 5

    def test_with_directory(self):
        assert parse_index("/some/path/logits_chkpt_42.pt") == 42

    def test_no_match(self):
        assert parse_index("not_a_checkpoint.pt") == float('inf')


class TestMergeCLI:
    """Test merge_checkpoints via CLI."""

    def _run_merge(self, pattern, output):
        result = subprocess.run(
            [sys.executable, "merge_checkpoints.py",
             "--pattern", pattern, "--output", str(output)],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        return result

    def test_merges_two_checkpoints(self, sample_checkpoint_pair, tmp_dir):
        pattern = str(sample_checkpoint_pair[0]).replace("_chkpt_5.pt", "_chkpt_*.pt")
        output = tmp_dir / "merged.pt"
        result = self._run_merge(pattern, output)
        assert result.returncode == 0, result.stderr
        assert output.exists()

        data = torch.load(output, map_location="cpu", weights_only=False)
        assert len(data["logits"]) == 2
        assert len(data["meta"]) == 2

    def test_preserves_logit_shapes(self, sample_checkpoint_pair, tmp_dir):
        pattern = str(sample_checkpoint_pair[0]).replace("_chkpt_5.pt", "_chkpt_*.pt")
        output = tmp_dir / "merged.pt"
        self._run_merge(pattern, output)

        data = torch.load(output, map_location="cpu", weights_only=False)
        for logits in data["logits"]:
            assert logits.shape == (3, 50)

    def test_no_matching_files(self, tmp_dir):
        output = tmp_dir / "merged.pt"
        result = self._run_merge(str(tmp_dir / "nonexistent_*.pt"), output)
        assert result.returncode == 0  # graceful exit
        assert not output.exists()

    def test_order_by_checkpoint_index(self, sample_checkpoint_pair, tmp_dir):
        pattern = str(sample_checkpoint_pair[0]).replace("_chkpt_5.pt", "_chkpt_*.pt")
        output = tmp_dir / "merged.pt"
        self._run_merge(pattern, output)

        data = torch.load(output, map_location="cpu", weights_only=False)
        # First entry should be from chkpt_5, second from chkpt_10
        assert "Prompt 0" in data["meta"][0]["prompt"]
        assert "Prompt 1" in data["meta"][1]["prompt"]

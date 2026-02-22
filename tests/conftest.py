"""Shared fixtures for entropic-deviation tests."""
import json
import os
import tempfile

import pandas as pd
import pytest
import torch


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def sample_logits():
    """Create sample logits tensor (5 tokens, vocab_size=100)."""
    torch.manual_seed(42)
    return torch.randn(5, 100)


@pytest.fixture
def uniform_logits():
    """Logits that produce near-uniform distribution (ED ≈ 0)."""
    return torch.zeros(10, 100)


@pytest.fixture
def peaked_logits():
    """Logits that produce a very peaked distribution (ED → 1)."""
    logits = torch.full((10, 100), -100.0)
    logits[:, 0] = 100.0  # all probability on token 0
    return logits


@pytest.fixture
def sample_checkpoint(tmp_dir, sample_logits):
    """Create a minimal .pt checkpoint file."""
    meta = [
        {
            "prompt": "wiki: Test prompt about science",
            "temp": 1.0,
            "seq_len": 5,
            "gen_time": 1700000000.0,
            "timestamp": "2025-01-01T00:00:00",
        }
    ]
    path = tmp_dir / "logits_chkpt_5.pt"
    torch.save({"logits": [sample_logits], "meta": meta}, path)
    return path


@pytest.fixture
def sample_checkpoint_pair(tmp_dir):
    """Create two checkpoint files for merge testing."""
    torch.manual_seed(42)
    paths = []
    for i, idx in enumerate([5, 10]):
        logits = torch.randn(3, 50)
        meta = [
            {
                "prompt": f"wiki: Prompt {i}",
                "temp": 0.7,
                "seq_len": 3,
                "gen_time": 1700000000.0 + i,
                "timestamp": "2025-01-01T00:00:00",
            }
        ]
        path = tmp_dir / f"logits_chkpt_{idx}.pt"
        torch.save({"logits": [logits], "meta": meta}, path)
        paths.append(path)
    return paths


@pytest.fixture
def sample_ed_csv(tmp_dir):
    """Create a minimal ED results CSV for metrics testing."""
    data = []
    for temp in [0.7, 1.0, 1.3]:
        for domain in ["wiki", "news", "code"]:
            for rank in range(5):
                data.append({
                    "ED_mean": 0.85 + temp * 0.02 + (0.01 if domain == "wiki" else 0.0)
                              + torch.randn(1).item() * 0.01,
                    "ED_std": 0.05,
                    "temp": temp,
                    "domain": domain,
                    "seq_len": 128,
                    "model": "TestModel",
                    "model_size": 8_000_000_000,
                    "rank": rank,
                    "chkpt_id": rank + 1,
                })
    df = pd.DataFrame(data)
    path = tmp_dir / "ed_results.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def multi_model_ed_csv(tmp_dir):
    """Create ED results CSV with multiple model sizes for F3 testing."""
    data = []
    for model, size in [("Small", 3_800_000_000), ("Large", 70_000_000_000)]:
        for temp in [0.7, 1.0, 1.3]:
            for rank in range(5):
                data.append({
                    "ED_mean": 0.85 + (size / 1e11) * 0.01
                              + torch.randn(1).item() * 0.005,
                    "ED_std": 0.05,
                    "temp": temp,
                    "domain": "wiki",
                    "seq_len": 128,
                    "model": model,
                    "model_size": size,
                    "rank": rank,
                    "chkpt_id": rank + 1,
                })
    df = pd.DataFrame(data)
    path = tmp_dir / "ed_multi_model.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def prompts_jsonl(tmp_dir):
    """Create a minimal prompts JSONL file."""
    prompts = [
        {"prompt": "The history of computing", "domain": "wiki", "len": 4},
        {"prompt": "Breaking news today", "domain": "news", "len": 3},
        {"prompt": "def hello():", "domain": "code", "len": 2},
    ]
    path = tmp_dir / "prompts.jsonl"
    with open(path, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")
    return path


@pytest.fixture
def prompts_txt(tmp_dir):
    """Create a minimal plain-text prompts file."""
    path = tmp_dir / "prompts.txt"
    with open(path, "w") as f:
        f.write("Hello world\n")
        f.write("Tell me a story\n")
        f.write("\n")  # blank line — should be skipped
        f.write("What is 2+2?\n")
    return path

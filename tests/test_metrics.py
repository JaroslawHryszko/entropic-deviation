"""Tests for calculate_metrics.py — statistical falsification tests F1-F8."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from calculate_metrics import main as metrics_main
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="statsmodels not installed")


class TestMetrics:
    """Test the metrics computation."""

    def _run_metrics(self, csv_path, out_path):
        """Run calculate_metrics.main() with mocked sys.argv."""
        old_argv = sys.argv
        sys.argv = ["calculate_metrics.py", str(csv_path), "--out", str(out_path)]
        try:
            metrics_main()
        finally:
            sys.argv = old_argv

    def test_produces_output_csv(self, sample_ed_csv, tmp_dir):
        out = tmp_dir / "ft.csv"
        self._run_metrics(sample_ed_csv, out)
        assert out.exists()

    def test_all_eight_tests_present(self, sample_ed_csv, tmp_dir):
        out = tmp_dir / "ft.csv"
        self._run_metrics(sample_ed_csv, out)
        df = pd.read_csv(out)
        assert list(df["TestID"]) == ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]

    def test_f1_rejects_nonzero_mean(self, sample_ed_csv, tmp_dir):
        """ED_mean ~ 0.87, so F1 should strongly reject H0: mean=0."""
        out = tmp_dir / "ft.csv"
        self._run_metrics(sample_ed_csv, out)
        df = pd.read_csv(out)
        p_f1 = df.loc[df["TestID"] == "F1", "p_value"].iloc[0]
        assert p_f1 < 0.01

    def test_f3_nan_for_single_model(self, sample_ed_csv, tmp_dir):
        """Single model size -> F3 should be NaN."""
        out = tmp_dir / "ft.csv"
        self._run_metrics(sample_ed_csv, out)
        df = pd.read_csv(out)
        p_f3 = df.loc[df["TestID"] == "F3", "p_value"].iloc[0]
        assert pd.isna(p_f3)

    def test_f3_computes_for_multi_model(self, multi_model_ed_csv, tmp_dir):
        """Multiple model sizes -> F3 should produce a p-value."""
        out = tmp_dir / "ft.csv"
        self._run_metrics(multi_model_ed_csv, out)
        df = pd.read_csv(out)
        p_f3 = df.loc[df["TestID"] == "F3", "p_value"].iloc[0]
        assert pd.notna(p_f3)
        assert 0 <= p_f3 <= 1

    def test_f6_nan_for_uniform_length(self, sample_ed_csv, tmp_dir):
        """All seq_len=128 -> F6 should be NaN."""
        out = tmp_dir / "ft.csv"
        self._run_metrics(sample_ed_csv, out)
        df = pd.read_csv(out)
        p_f6 = df.loc[df["TestID"] == "F6", "p_value"].iloc[0]
        assert pd.isna(p_f6)

    def test_f7_domain_test_runs(self, sample_ed_csv, tmp_dir):
        """Multiple domains -> F7 should produce a p-value."""
        out = tmp_dir / "ft.csv"
        self._run_metrics(sample_ed_csv, out)
        df = pd.read_csv(out)
        p_f7 = df.loc[df["TestID"] == "F7", "p_value"].iloc[0]
        assert pd.notna(p_f7)
        assert 0 <= p_f7 <= 1

    def test_pvalues_in_valid_range(self, sample_ed_csv, tmp_dir):
        """All p-values should be in [0, 1] or NaN."""
        out = tmp_dir / "ft.csv"
        self._run_metrics(sample_ed_csv, out)
        df = pd.read_csv(out)
        for _, row in df.iterrows():
            if pd.notna(row["p_value"]):
                assert 0 <= row["p_value"] <= 1, f"{row['TestID']}: p={row['p_value']}"

    def test_tukey_posthoc_file_created(self, sample_ed_csv, tmp_dir):
        """Tukey post-hoc CSV should be written next to --out."""
        out = tmp_dir / "ft.csv"
        self._run_metrics(sample_ed_csv, out)
        expected = tmp_dir / f"f2_posthoc_{os.path.basename(sample_ed_csv)}"
        assert expected.exists()
        df = pd.read_csv(expected)
        assert len(df) > 0

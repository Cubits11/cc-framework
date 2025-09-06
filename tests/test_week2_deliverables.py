"""Test that Week 2 deliverables are complete"""

import subprocess
from pathlib import Path

import pandas as pd


def test_smoke_runs():
    """200 session smoke test completes"""
    result = subprocess.run(["make", "reproduce-smoke"], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    assert Path("results/smoke/aggregates/summary.csv").exists()


def test_unit_tests_pass():
    """Unit tests are green"""
    result = subprocess.run(["pytest", "tests/unit", "-q"], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


def test_three_figures_exist():
    """Exactly 3 figures generated"""
    figures = [
        "paper/figures/cc_convergence.pdf",
        "paper/figures/phase_diagram.pdf",
        "paper/figures/roc_comparison.pdf",
    ]
    for fig in figures:
        assert Path(fig).exists(), f"Missing figure: {fig}"


def test_documentation_exists():
    """Core documentation present"""
    assert Path("docs/index.md").exists()
    assert Path("docs/experiments-guide.md").exists()
    assert Path("docs/reproducibility.md").exists()
    assert Path("docs/architecture/protocol_sequence.md").exists()
    assert Path("paper/draft.md").exists()


def test_results_schema():
    """Results have correct schema"""
    summary = pd.read_csv("results/aggregates/summary.csv")
    required_cols = ["experiment_id", "n_sessions", "cc_max", "ci_lower", "ci_upper"]
    assert all(col in summary.columns for col in required_cols)
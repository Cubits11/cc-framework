"""Test that Week 2 deliverables are complete"""

import json
from pathlib import Path

import pandas as pd
import pytest

from cc.analysis.generate_figures import main as gen_main


@pytest.fixture
def smoke_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    """Build smoke-style figures and summary in a test-owned directory."""
    history = tmp_path / "audit.jsonl"
    with history.open("w", encoding="utf-8") as f:
        for i in range(15):
            record = {
                "cfg": {"epsilon": 0.1 + 0.01 * i, "T": 5 + i, "samples": 200 + i},
                "metrics": {"CC_max": 1.0 + 0.01 * i},
            }
            f.write(json.dumps(record) + "\n")

    fig_dir = tmp_path / "figures"
    out_dir = tmp_path / "aggregates"
    gen_main(["--history", str(history), "--fig-dir", str(fig_dir), "--out-dir", str(out_dir)])
    return fig_dir, out_dir


def test_smoke_runs(smoke_artifacts: tuple[Path, Path]):
    """Smoke-style artifact generation completes in a temp directory."""
    _, out_dir = smoke_artifacts
    assert (out_dir / "summary.csv").exists()


def test_three_figures_exist(smoke_artifacts: tuple[Path, Path]):
    """Exactly 3 figures generated"""
    fig_dir, _ = smoke_artifacts
    figures = ["cc_convergence.pdf", "phase_diagram.pdf", "roc_comparison.pdf"]
    for fig in figures:
        assert (fig_dir / fig).exists(), f"Missing figure: {fig}"


def test_documentation_exists():
    """Core documentation present"""
    assert Path("docs/index.md").exists()
    assert Path("docs/experiments-guide.md").exists()
    assert Path("docs/reproducibility.md").exists()
    assert Path("docs/architecture/protocol_sequence.md").exists()
    assert Path("paper/draft.md").exists()


def test_results_schema(smoke_artifacts: tuple[Path, Path]):
    """Results have correct schema"""
    _, out_dir = smoke_artifacts
    summary = pd.read_csv(out_dir / "summary.csv")
    required_cols = ["experiment_id", "n_sessions", "cc_max", "ci_lower", "ci_upper"]
    assert all(col in summary.columns for col in required_cols)

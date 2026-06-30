from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from cc.evals.dependence_benchmark import (
    KeywordBlockerAdapter,
    load_dataset,
    run_benchmark,
    verify_benchmark_summary,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "dependence_benchmark_harmful.csv"


class MockLlamaGuard:
    name = "llama_guard"
    version = "mock-llama-guard@integration"

    def blocks(self, prompt: str, response: str | None, metadata: Mapping[str, object]) -> bool:
        return "blocked-by-llama" in prompt


def test_mocked_llamaguard_keyword_fixture_produces_verified_summary() -> None:
    rows, dataset_hash = load_dataset(FIXTURE)

    summary = run_benchmark(
        rows,
        [
            MockLlamaGuard(),
            KeywordBlockerAdapter(keywords=("jailbreak", "exploit"), threshold=0.0),
        ],
        dataset_id=str(FIXTURE),
        dataset_sha256=dataset_hash,
        run_id="integration-fixture",
    )

    assert verify_benchmark_summary(summary) == []
    assert summary["dataset"]["sha256"] == dataset_hash
    assert summary["n_harmful"] == 4
    assert summary["events"]["stack_unsafe_pass"]["pairwise_tightened"]["fh_width"] == 0.0


def test_dependence_benchmark_cli_keyword_only(tmp_path: Path) -> None:
    out = tmp_path / "summary.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cc.evals.dependence_benchmark",
            "--dataset",
            str(FIXTURE),
            "--adapters",
            "keyword_blocker",
            "--keyword-terms",
            "jailbreak,exploit",
            "--out",
            str(out),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert verify_benchmark_summary(payload) == []
    assert payload["labels"] == ["keyword_blocker"]

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_FILES = {
    "table_1_classical_frechet_bounds.csv",
    "table_2_metric_examples.csv",
    "table_3_witness_verification.csv",
    "figure_1_fh_interval.png",
    "figure_2_independence_regret.png",
    "minimal_bounds.json",
    "minimal_witnesses.json",
    "minimal_bundle.json",
    "environment.json",
    "manifest.json",
}


def test_reproduce_paper_creates_required_artifacts_and_matching_manifest(tmp_path: Path) -> None:
    out_dir = tmp_path / "paper"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_paper.py",
            "--out",
            str(out_dir),
        ],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    produced = {path.name for path in out_dir.iterdir()}
    assert produced >= REQUIRED_FILES

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    entries = {entry["filename"]: entry for entry in manifest["files"]}
    for filename in REQUIRED_FILES - {"manifest.json"}:
        path = out_dir / filename
        assert entries[filename]["sha256"] == _sha256(path)
        assert entries[filename]["bytes"] == path.stat().st_size


def test_minimal_example_runs_and_returns_expected_keys() -> None:
    result = subprocess.run(
        [sys.executable, "examples/minimal/run_bounds.py"],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    expected_keys = {
        "failure_event_convention",
        "guardrails",
        "declared_marginals",
        "query",
        "lower_bound",
        "upper_bound",
        "fh_width",
        "fh_position",
        "independent_baseline",
        "independence_regret",
        "witnesses_verified",
    }
    assert expected_keys <= set(payload)
    assert payload["witnesses_verified"] == {"lower": True, "upper": True}


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    return env


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

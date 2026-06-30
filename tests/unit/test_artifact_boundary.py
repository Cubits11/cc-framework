from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[2]


def load_checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_artifact_boundary",
        ROOT / "scripts" / "check_artifact_boundary.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load artifact boundary checker")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_static_checker_rejects_forbidden_tracked_runtime_file() -> None:
    checker = load_checker()

    findings = checker.validate_tracked_paths(
        ["checkpoints/exp_123/checkpoint_000100.json"]
    )

    assert findings
    assert "runtime-only generated directory" in findings[0].message


def test_static_checker_allows_whitelisted_paper_artifact() -> None:
    checker = load_checker()

    findings = checker.validate_tracked_paths(["artifacts/paper/manifest.json"])

    assert findings == []


def test_static_checker_allows_declared_fixture() -> None:
    checker = load_checker()

    findings = checker.validate_tracked_paths(["tests/fixtures/week5_scan/scan.csv"])

    assert findings == []


def test_static_checker_reports_missing_paper_manifest(tmp_path: Path) -> None:
    checker = load_checker()

    findings = checker.validate_paper_manifest(tmp_path)

    assert findings
    assert findings[0].path == "artifacts/paper/manifest.json"
    assert "missing required paper artifact manifest" in findings[0].message


def test_after_run_status_rejects_new_untracked_file_outside_runtime_roots() -> None:
    checker = load_checker()

    findings = checker.validate_after_run_status(
        ["?? scratch.json", "?? results/local/out.json"],
        baseline_status_lines=[],
    )

    assert len(findings) == 1
    assert findings[0].path == "scratch.json"
    assert "new untracked file" in findings[0].message


def test_after_run_status_ignores_baseline_entries() -> None:
    checker = load_checker()

    findings = checker.validate_after_run_status(
        [" M docs/release/ARTIFACT_BOUNDARY.md", "?? scratch.json"],
        baseline_status_lines=[" M docs/release/ARTIFACT_BOUNDARY.md"],
    )

    assert len(findings) == 1
    assert findings[0].path == "scratch.json"

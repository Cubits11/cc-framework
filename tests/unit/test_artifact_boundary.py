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

    findings = checker.validate_tracked_paths(["checkpoints/exp_123/checkpoint_000100.json"])

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


# --- E1 manifest boundary -------------------------------------------------
#
# validate_e1_manifest reads required_files and the hashed file records out of
# the manifest. Manifest schema v2 moved both under identity_payload so the
# digest stays invariant under relocation, and the checker silently stopped
# finding them. These tests pin the shape it depends on.


def _write_e1_manifest(root: Path, manifest: dict) -> Path:
    import json

    target = root / "artifacts" / "empirical" / "e1"
    target.mkdir(parents=True, exist_ok=True)
    path = target / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _valid_identity() -> dict:
    return {
        "required_files": ["study.json", "coverage.csv", "manifest.json"],
        "files": [
            {"filename": "study.json", "sha256": "0" * 64, "bytes": 1},
            {"filename": "coverage.csv", "sha256": "1" * 64, "bytes": 1},
        ],
    }


def test_committed_e1_manifest_satisfies_the_boundary() -> None:
    """The real artifact must pass, not just a synthetic fixture."""

    checker = load_checker()

    assert checker.validate_e1_manifest(ROOT) == []


def test_e1_manifest_checker_accepts_schema_v2_shape(tmp_path: Path) -> None:
    checker = load_checker()
    _write_e1_manifest(tmp_path, {"identity_payload": _valid_identity()})

    assert checker.validate_e1_manifest(tmp_path) == []


def test_e1_manifest_checker_rejects_a_manifest_without_identity_payload(tmp_path: Path) -> None:
    """A pre-v2 manifest must fail loudly rather than pass vacuously."""

    checker = load_checker()
    _write_e1_manifest(tmp_path, _valid_identity())  # fields at top level, v1 style

    findings = checker.validate_e1_manifest(tmp_path)

    assert findings
    assert "identity_payload" in findings[0].message


def test_e1_manifest_checker_rejects_allowlist_drift(tmp_path: Path) -> None:
    checker = load_checker()
    identity = _valid_identity()
    identity["required_files"] = ["study.json"]
    _write_e1_manifest(tmp_path, {"identity_payload": identity})

    findings = checker.validate_e1_manifest(tmp_path)

    assert any("required_files does not exactly match" in f.message for f in findings)


def test_e1_manifest_checker_rejects_hashed_file_drift(tmp_path: Path) -> None:
    checker = load_checker()
    identity = _valid_identity()
    identity["files"] = [{"filename": "study.json", "sha256": "0" * 64, "bytes": 1}]
    _write_e1_manifest(tmp_path, {"identity_payload": identity})

    findings = checker.validate_e1_manifest(tmp_path)

    assert any("file entries do not exactly match" in f.message for f in findings)

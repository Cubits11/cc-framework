# tests/integration/test_claim_governance_capsule.py

from __future__ import annotations

import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CAPSULE = ROOT / "examples" / "claim_governance_capsule"
OUTPUTS = CAPSULE / "outputs"

MANIFEST_TRACKED_ARTIFACTS = [
    "audit_log.jsonl",
    "bounds.json",
    "calibration.json",
    "cc_report.json",
    "claim_envelope.json",
    "claim_governance_audit.json",
    "confirmatory_failure_matrix.json",
    "confirmatory_protocol.json",
    "decay_policy.json",
    "extremal_lower.json",
    "extremal_upper.json",
]

GENERATED_ARTIFACTS = [
    *MANIFEST_TRACKED_ARTIFACTS,
    "capsule_manifest.json",
]


def test_claim_governance_capsule_reproduces_expected_manifest() -> None:
    result = run_capsule()

    generated_path = OUTPUTS / "capsule_manifest.json"
    expected_path = CAPSULE / "manifest.expected.json"
    assert generated_path.exists(), result.stderr + result.stdout

    generated = load_json(generated_path)
    expected = load_json(expected_path)
    audit = load_json(OUTPUTS / "claim_governance_audit.json")
    envelope = load_json(OUTPUTS / "claim_envelope.json")

    assert_json_equal(generated, expected, label="capsule manifest")
    assert result.returncode == 0, result.stderr + result.stdout

    assert generated["governance_verdict"] == "pass"
    assert generated["pass_caveat"] == (
        "PASS means internal consistency under verifier rules; it does not mean the AI "
        "system is safe in deployment."
    )

    manifest_files = {item["filename"]: item for item in generated["files"]}
    assert sorted(manifest_files) == sorted(MANIFEST_TRACKED_ARTIFACTS)

    assert audit["schema"] == "cc/claim-governance-audit.v1"
    assert "schema_" not in audit
    assert audit["verdict"] == "pass"
    assert audit["required_human_review"] is False
    assert audit["reasons"] == []
    assert audit["boundary"]["mandatory_non_claims_missing"] == []
    assert len(audit["non_claims"]) >= 1
    assert any("deployment safety" in item.lower() for item in audit["non_claims"])
    assert any("not a release claim" in item.lower() for item in audit["non_claims"])

    evidence_by_path = {item["path"]: item for item in audit["evidence_artifacts"]}
    for required_path in (
        "bounds.json",
        "calibration.json",
        "confirmatory_failure_matrix.json",
        "confirmatory_protocol.json",
        "decay_policy.json",
        "extremal_lower.json",
        "extremal_upper.json",
        "audit_log.jsonl",
    ):
        assert required_path in evidence_by_path
        assert evidence_by_path[required_path]["status"] == "present"
        assert (
            evidence_by_path[required_path]["sha256_actual"]
            == evidence_by_path[required_path]["sha256_expected"]
        )

    assert audit["decay"]["status"] == "fresh"
    assert audit["scenarios"]["scenario_count"] == 2
    assert audit["scenarios"]["infeasible_count"] == 0
    assert sorted(audit["scenarios"]["kinds"]) == ["frechet_endpoint", "frechet_endpoint"]

    assert audit["confirmatory_protocols"]["present"] is True
    assert audit["confirmatory_protocols"]["artifact_count"] == 1
    assert audit["confirmatory_protocols"]["failed_count"] == 0
    assert audit["confirmatory_protocols"]["review_count"] == 0
    assert audit["confirmatory_protocols"]["audits"][0]["status"] == "pass"
    assert all(
        check["status"] == "pass"
        for check in audit["confirmatory_protocols"]["audits"][0]["checks"]
    )

    assert audit["envelope_support"]["strongest_non_integrity_strength"] == "confirmatory"
    assert audit["envelope_support"]["relation_counts"]["confirmatory_tests"] >= 1
    assert audit["envelope_support"]["unsupported_role_refs"] == []
    assert audit["envelope_support"]["unknown_role_refs"] == 0

    assert envelope["schema"] == "cc.claim_envelope.v1"
    assert envelope["governance_state"]["verdict"] == "pass"
    assert envelope["governance_state"]["required_human_review"] is False
    assert envelope["governance_state"]["support_summary"]["strongest_non_integrity_strength"] == (
        "confirmatory"
    )


def test_claim_governance_capsule_second_run_artifacts_are_identical() -> None:
    first = run_capsule()
    assert first.returncode == 0, first.stderr + first.stdout

    first_hashes = artifact_hashes()

    second = run_capsule()
    assert second.returncode == 0, second.stderr + second.stdout

    assert artifact_hashes() == first_hashes


def test_claim_governance_capsule_temp_copy_regeneration_is_byte_identical(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "capsule-a"
    second_root = tmp_path / "capsule-b"
    shutil.copytree(CAPSULE, first_root)
    shutil.copytree(CAPSULE, second_root)

    first = run_capsule_at(first_root)
    second = run_capsule_at(second_root)

    assert first.returncode == 0, first.stderr + first.stdout
    assert second.returncode == 0, second.stderr + second.stdout

    first_hashes = artifact_hashes_at(first_root / "outputs")
    second_hashes = artifact_hashes_at(second_root / "outputs")

    assert first_hashes == second_hashes
    for name in GENERATED_ARTIFACTS:
        assert (first_root / "outputs" / name).read_bytes() == (
            second_root / "outputs" / name
        ).read_bytes()


def test_claim_governance_capsule_tamper_changes_manifest_and_governance_verdict(
    tmp_path: Path,
) -> None:
    result = run_capsule()
    assert result.returncode == 0, result.stderr + result.stdout

    tampered_root = tmp_path / "capsule"
    shutil.copytree(CAPSULE, tampered_root)

    tampered_outputs = tampered_root / "outputs"
    bounds_path = tampered_outputs / "bounds.json"

    payload = load_json(bounds_path)
    payload["interval"]["upper"] = 0.999
    bounds_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest_check = subprocess.run(
        [
            sys.executable,
            "build_capsule.py",
            "--verify-only",
        ],
        cwd=tampered_root,
        env=env(extra_pythonpath=str(ROOT / "src")),
        capture_output=True,
        text=True,
        check=False,
    )

    assert manifest_check.returncode == 1
    assert "generated artifact differs" in manifest_check.stdout

    governance_check = subprocess.run(
        [
            sys.executable,
            "-m",
            "cc.reporting.cli",
            "verify-claim-governance",
            str(tampered_outputs / "cc_report.json"),
            "--base-dir",
            str(tampered_outputs),
            "--now",
            "2026-01-02T00:00:00Z",
            "--out",
            str(tampered_outputs / "tampered_audit.json"),
        ],
        cwd=ROOT,
        env=env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert governance_check.returncode == 2
    assert "Claim governance verdict: FAIL" in governance_check.stdout

    tampered_audit = load_json(tampered_outputs / "tampered_audit.json")
    assert tampered_audit["verdict"] == "fail"
    assert any(
        "sha-256" in reason.lower()
        or "hash" in reason.lower()
        or "mismatch" in reason.lower()
        or "does not match" in reason.lower()
        for reason in tampered_audit["reasons"]
    )
    assert any(
        artifact["path"] == "bounds.json"
        and artifact["status"] == "invalid"
        and "sha-256" in artifact["reason"].lower()
        and "does not match" in artifact["reason"].lower()
        for artifact in tampered_audit["evidence_artifacts"]
    )
    assert tampered_audit["receipt"]["artifact_hashes_verified"] is False


def test_claim_governance_capsule_readme_includes_pass_caveat_and_non_claims() -> None:
    readme = (CAPSULE / "README.md").read_text(encoding="utf-8").lower()

    required_phrases = [
        "pass governance audit means internal consistency under verifier rules",
        "does not mean the ai system is safe in deployment",
        "non-claims",
        "not a deployment-safety proof",
    ]

    for phrase in required_phrases:
        assert phrase in readme


def run_capsule() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["examples/claim_governance_capsule/reproduce.sh"],
        cwd=ROOT,
        env=env(),
        capture_output=True,
        text=True,
        check=False,
    )


def run_capsule_at(capsule_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(CAPSULE / "build_capsule.py"),
            "--capsule-dir",
            str(capsule_dir),
        ],
        cwd=ROOT,
        env=env(),
        capture_output=True,
        text=True,
        check=False,
    )


def env(*, extra_pythonpath: str | None = None) -> dict[str, str]:
    values = os.environ.copy()
    values["PYTHONHASHSEED"] = "0"

    pythonpath = "src"
    if extra_pythonpath is not None:
        pythonpath = f"{extra_pythonpath}:{pythonpath}"

    values["PYTHONPATH"] = pythonpath
    return values


def artifact_hashes() -> dict[str, str]:
    return artifact_hashes_at(OUTPUTS)


def artifact_hashes_at(outputs_dir: Path) -> dict[str, str]:
    return {
        name: sha256(outputs_dir / name)
        for name in GENERATED_ARTIFACTS
        if (outputs_dir / name).exists()
    }


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def assert_json_equal(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    label: str,
) -> None:
    if actual == expected:
        return

    actual_text = json.dumps(actual, indent=2, sort_keys=True).splitlines()
    expected_text = json.dumps(expected, indent=2, sort_keys=True).splitlines()
    diff = "\n".join(
        difflib.unified_diff(
            expected_text,
            actual_text,
            fromfile=f"expected {label}",
            tofile=f"actual {label}",
            lineterm="",
        )
    )
    raise AssertionError(f"{label} differs from golden artifact:\n{diff}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

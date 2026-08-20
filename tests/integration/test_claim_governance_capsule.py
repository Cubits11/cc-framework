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


def test_capsule_marks_current_realized_fpr_as_asserted_not_matrix_derived() -> None:
    result = run_capsule()
    assert result.returncode == 0, result.stderr + result.stdout

    calibration = load_json(OUTPUTS / "calibration.json")
    provenance = calibration["realized_fpr_provenance"]

    assert provenance["schema"] == "cc.calibration.realized-fpr-provenance.v1"
    assert provenance["field"] == "realized_fpr"
    assert provenance["semantics"]["status"] == "unresolved"
    assert provenance["origin"] == {"kind": "asserted"}
    assert provenance["external_anchor"] is None
    assert provenance["verification"]["origin"]["status"] == "asserted_not_derivable"
    assert any("does not infer a numerator" in item.lower() for item in provenance["non_claims"])


def test_forged_asserted_calibration_value_remains_regenerable_and_visibly_asserted(
    tmp_path: Path,
) -> None:
    """CH-001 still holds: an asserted upstream value can be regenerated coherently."""

    capsule_root = tmp_path / "asserted-forgery"
    shutil.copytree(CAPSULE, capsule_root)
    config_path = capsule_root / "inputs" / "capsule_config.json"
    config = load_json(config_path)
    config["calibration"]["realized_fpr"] = 0.011111111111
    write_json(config_path, config)

    result = run_capsule_at(capsule_root, "--update-expected")
    assert result.returncode == 0, result.stderr + result.stdout

    calibration = load_json(capsule_root / "outputs" / "calibration.json")
    provenance = calibration["realized_fpr_provenance"]
    assert calibration["realized_fpr"] == 0.011111111111
    assert provenance["origin"] == {"kind": "asserted"}
    assert provenance["verification"]["origin"]["status"] == "asserted_not_derivable"


def test_capsule_checks_opt_in_deterministically_derived_calibration_rate(tmp_path: Path) -> None:
    capsule_root = tmp_path / "derived-capsule"
    shutil.copytree(CAPSULE, capsule_root)
    config_path = capsule_root / "inputs" / "capsule_config.json"
    config = load_json(config_path)
    config["calibration"]["realized_fpr"] = 0.375
    config["calibration"]["realized_fpr_provenance"] = {
        "schema": "cc.calibration.realized-fpr-provenance.v1",
        "field": "realized_fpr",
        "semantics": {
            "status": "defined",
            "numerator": "Rows whose guardrail_keyword value equals one.",
            "denominator": "All checked-in failure-matrix rows.",
            "population": "The checked-in capsule failure-matrix rows.",
        },
        "origin": {
            "kind": "deterministic_derivation",
            "algorithm": "binary_column_rate/v1",
            "source": {
                "path": "inputs/failure_matrix.csv",
                "sha256": sha256(capsule_root / "inputs" / "failure_matrix.csv"),
            },
            "column": "guardrail_keyword",
            "inclusion_rule": "all_rows",
            "exact_result": {"numerator": 9, "denominator": 24},
            "rendering": {"decimal_places": 12, "rounding": "half_even"},
        },
        "external_anchor": None,
    }
    write_json(config_path, config)

    result = run_capsule_at(capsule_root, "--update-expected")
    assert result.returncode == 0, result.stderr + result.stdout

    calibration = load_json(capsule_root / "outputs" / "calibration.json")
    verification = calibration["realized_fpr_provenance"]["verification"]["origin"]
    assert verification == {
        "computed_value": "0.375000000000",
        "denominator_count": 24,
        "numerator_count": 9,
        "source_sha256": sha256(capsule_root / "inputs" / "failure_matrix.csv"),
        "status": "derivation_verified_from_declared_bytes",
    }

    config["calibration"]["realized_fpr_provenance"]["origin"]["source"]["sha256"] = "0" * 64
    write_json(config_path, config)
    stale_source = run_capsule_at(capsule_root)
    assert stale_source.returncode == 1
    assert "source sha256 does not match matrix" in stale_source.stdout


def test_capsule_rejects_a_wrong_opt_in_derived_calibration_rate(tmp_path: Path) -> None:
    capsule_root = tmp_path / "mismatched-derived-capsule"
    shutil.copytree(CAPSULE, capsule_root)
    config_path = capsule_root / "inputs" / "capsule_config.json"
    config = load_json(config_path)
    config["calibration"]["realized_fpr"] = 0.011111111111
    config["calibration"]["realized_fpr_provenance"] = {
        "schema": "cc.calibration.realized-fpr-provenance.v1",
        "field": "realized_fpr",
        "semantics": {
            "status": "defined",
            "numerator": "Rows whose guardrail_keyword value equals one.",
            "denominator": "All checked-in failure-matrix rows.",
            "population": "The checked-in capsule failure-matrix rows.",
        },
        "origin": {
            "kind": "deterministic_derivation",
            "algorithm": "binary_column_rate/v1",
            "source": {
                "path": "inputs/failure_matrix.csv",
                "sha256": sha256(capsule_root / "inputs" / "failure_matrix.csv"),
            },
            "column": "guardrail_keyword",
            "inclusion_rule": "all_rows",
            "exact_result": {"numerator": 9, "denominator": 24},
            "rendering": {"decimal_places": 12, "rounding": "half_even"},
        },
        "external_anchor": None,
    }
    write_json(config_path, config)

    result = run_capsule_at(capsule_root)
    assert result.returncode == 1
    assert "does not match the deterministic guardrail_keyword rate" in result.stdout


def test_capsule_records_but_does_not_fetch_an_external_anchor(tmp_path: Path) -> None:
    capsule_root = tmp_path / "anchored-capsule"
    shutil.copytree(CAPSULE, capsule_root)
    config_path = capsule_root / "inputs" / "capsule_config.json"
    config = load_json(config_path)
    config["calibration"]["realized_fpr_provenance"]["external_anchor"] = {
        "mechanism": "reference_only/v1",
        "uri": "https://example.test/calibration/receipt",
        "subject_sha256": calibration_anchor_subject_hash(config["calibration"]),
        "issued_at": "2026-01-01T00:00:00Z",
        "issuer": "fixture instrument",
    }
    write_json(config_path, config)

    result = run_capsule_at(capsule_root, "--update-expected")
    assert result.returncode == 0, result.stderr + result.stdout

    calibration = load_json(capsule_root / "outputs" / "calibration.json")
    assert calibration["realized_fpr_provenance"]["verification"]["external_anchor"]["status"] == (
        "reference_recorded_not_externally_verified"
    )

    config["calibration"]["realized_fpr"] = 0.125
    write_json(config_path, config)
    wrong_subject = run_capsule_at(capsule_root)
    assert wrong_subject.returncode == 1
    assert "does not bind this exact subject" in wrong_subject.stdout


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


def run_capsule_at(capsule_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(CAPSULE / "build_capsule.py"),
            "--capsule-dir",
            str(capsule_dir),
            *args,
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def calibration_anchor_subject_hash(calibration: dict[str, Any]) -> str:
    provenance = calibration["realized_fpr_provenance"]
    subject = {
        "schema": provenance["schema"],
        "field": provenance["field"],
        "value": str(calibration["realized_fpr"]),
        "semantics": provenance["semantics"],
        "origin": provenance["origin"],
    }
    encoded = json.dumps(
        subject,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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

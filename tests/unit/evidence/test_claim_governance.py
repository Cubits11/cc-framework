from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from cc.evidence import ClaimGovernanceAudit, GovernanceVerdict, verify_claim_governance
from cc.evidence.claim_governance import ClaimFreshnessStatus
from cc.evidence.decay import ClaimDecayPolicy, ClaimDecayRecord, VersionWatchSet
from cc.evidence.extremal_scenario import ExtremalScenario
from cc.kernel.frechet_classes import frechet_bounds
from cc.reporting.report import (
    CalibrationSummary,
    ClaimSummary,
    EnvironmentMetadata,
    EvidenceArtifact,
    GitMetadata,
    MeasurementSummary,
    RunSummary,
    build_cc_report,
    sha256_file,
    write_cc_report,
)

ROOT = Path(__file__).resolve().parents[3]


def test_public_claim_governance_imports_are_intentional() -> None:
    assert ClaimGovernanceAudit.__name__ == "ClaimGovernanceAudit"
    assert GovernanceVerdict.PASS.value == "pass"
    assert callable(verify_claim_governance)


def test_passing_claim_package_verifies_governance(tmp_path: Path) -> None:
    report_path = _write_package(tmp_path)

    audit = verify_claim_governance(report_path, now=_issued_at() + timedelta(days=1))

    assert audit.verdict is GovernanceVerdict.PASS
    assert audit.required_human_review is False
    assert audit.receipt.report_hash_verified is True
    assert audit.receipt.artifact_hashes_verified is True
    assert audit.decay.status is ClaimFreshnessStatus.FRESH
    assert audit.scenarios.scenario_count == 2
    assert audit.scenarios.infeasible_count == 0
    assert audit.boundary.mandatory_non_claims_missing == []
    assert "safe in deployment" in verify_claim_governance.__doc__


def test_degraded_decay_requires_review(tmp_path: Path) -> None:
    report_path = _write_package(tmp_path)

    audit = verify_claim_governance(report_path, now=_issued_at() + timedelta(days=8))

    assert audit.verdict is GovernanceVerdict.NEEDS_REVIEW
    assert audit.decay.status is ClaimFreshnessStatus.DEGRADED
    assert any("degraded" in reason for reason in audit.reasons)


def test_expired_decay_fails(tmp_path: Path) -> None:
    report_path = _write_package(tmp_path)

    audit = verify_claim_governance(report_path, now=_issued_at() + timedelta(days=15))

    assert audit.verdict is GovernanceVerdict.FAIL
    assert audit.decay.status is ClaimFreshnessStatus.EXPIRED
    assert any("expired" in reason for reason in audit.reasons)


def test_version_watch_change_expires_decay_when_report_has_observed_dependency(
    tmp_path: Path,
) -> None:
    decay = _decay_payload()
    decay["version_watch_set"] = {
        "model_versions": {},
        "guardrail_versions": {},
        "data_versions": {},
        "dependency_versions": {"numpy": "1.0"},
    }
    report_path = _write_package(
        tmp_path,
        decay_payload=decay,
        package_snapshot={"numpy": "2.0"},
    )

    audit = verify_claim_governance(report_path, now=_issued_at() + timedelta(days=1))

    assert audit.verdict is GovernanceVerdict.FAIL
    assert audit.decay.status is ClaimFreshnessStatus.EXPIRED
    assert "version_changed:dependency_versions.numpy" in audit.decay.trigger_summary


def test_artifact_hash_mismatch_fails_even_when_receipt_hash_is_valid(tmp_path: Path) -> None:
    report_path = _write_package(tmp_path)
    (tmp_path / "decay.json").write_text('{"tampered": true}\n', encoding="utf-8")

    audit = verify_claim_governance(report_path, now=_issued_at() + timedelta(days=1))

    assert audit.verdict is GovernanceVerdict.FAIL
    assert audit.receipt.report_hash_verified is True
    assert audit.receipt.artifact_hashes_verified is False
    assert any("SHA-256" in reason for reason in audit.reasons)


def test_missing_mandatory_scenario_non_claim_requires_review(tmp_path: Path) -> None:
    upper = _scenario_payload("upper")
    lower = _scenario_payload("lower")
    upper["non_claims"] = ["This endpoint scenario has reviewer-facing diagnostics."]
    lower["non_claims"] = ["This endpoint scenario has reviewer-facing diagnostics."]
    report_path = _write_package(tmp_path, scenario_payloads=[upper, lower])

    audit = verify_claim_governance(report_path, now=_issued_at() + timedelta(days=1))

    assert audit.verdict is GovernanceVerdict.NEEDS_REVIEW
    assert "extremal_scenario_not_likely_world_proof" in (
        audit.boundary.mandatory_non_claims_missing
    )


def test_exploratory_interval_leakage_fails(tmp_path: Path) -> None:
    upper = _scenario_payload("upper")
    upper["metadata"]["exploratory_ci"] = [0.8, 1.0]
    report_path = _write_package(tmp_path, scenario_payloads=[upper, _scenario_payload("lower")])

    audit = verify_claim_governance(report_path, now=_issued_at() + timedelta(days=1))

    assert audit.verdict is GovernanceVerdict.FAIL
    assert any("Exploratory evidence leaked" in reason for reason in audit.reasons)


def test_unknown_role_requires_review_or_fails_in_strict_mode(tmp_path: Path) -> None:
    mystery = tmp_path / "mystery.txt"
    mystery.write_text("opaque evidence\n", encoding="utf-8")
    extra = EvidenceArtifact(
        path="mystery.txt",
        sha256=sha256_file(mystery),
        bytes=mystery.stat().st_size,
        role="mystery_role",
    )
    report_path = _write_package(tmp_path, extra_artifacts=[extra])

    loose = verify_claim_governance(report_path, now=_issued_at() + timedelta(days=1))
    strict = verify_claim_governance(
        report_path,
        now=_issued_at() + timedelta(days=1),
        strict_unknown_roles=True,
    )

    assert loose.verdict is GovernanceVerdict.NEEDS_REVIEW
    assert strict.verdict is GovernanceVerdict.FAIL
    assert any("Unknown evidence role" in reason for reason in loose.reasons)


def test_exploratory_redteam_cannot_support_bounded_claim_level(tmp_path: Path) -> None:
    redteam = tmp_path / "redteam.json"
    _write_json(
        redteam,
        {
            "redteam_id": "adaptive-redteam-1",
            "discovery_protocol": "adaptive dependence-cliff search",
            "findings": [{"case_id": "candidate-cliff"}],
            "non_claims": [
                "Exploratory red-team evidence is not confirmatory evidence.",
            ],
        },
    )
    report_path = _write_package(
        tmp_path,
        extra_artifacts=[_artifact(redteam, "exploratory_redteam")],
    )

    audit = verify_claim_governance(report_path, now=_issued_at() + timedelta(days=1))

    assert audit.verdict is GovernanceVerdict.NEEDS_REVIEW
    assert any(
        "exploratory_redteam" in reason and "cannot support claim level" in reason
        for reason in audit.reasons
    )


def test_human_review_note_cannot_reduce_review_when_hashes_do_not_match(
    tmp_path: Path,
) -> None:
    report_path = _write_package(
        tmp_path,
        claim_level="release_claim",
        review_note_payload={
            "review_id": "review-partial",
            "reviewer": "external-reviewer",
            "reviewed_artifact_hashes": [],
            "reviewed_claim_level": "release_claim",
            "decision": "approved_with_conditions",
            "non_claims": [
                "Human review does not upgrade underlying statistical evidence.",
            ],
        },
    )

    audit = verify_claim_governance(report_path, now=_issued_at() + timedelta(days=1))

    assert audit.verdict is GovernanceVerdict.NEEDS_REVIEW
    assert audit.required_human_review is True
    assert any("does not cover current artifact hash set" in reason for reason in audit.reasons)


def test_hash_matched_human_review_note_can_satisfy_scoped_release_review(
    tmp_path: Path,
) -> None:
    report_path = _write_package(
        tmp_path,
        claim_level="release_claim",
        review_note_payload={
            "review_id": "review-complete",
            "reviewer": "external-reviewer",
            "reviewed_artifact_hashes": "__ALL_BOUND_ARTIFACT_HASHES__",
            "reviewed_claim_level": "release_claim",
            "decision": "approved_with_conditions",
            "non_claims": [
                "Human review does not upgrade underlying statistical evidence.",
            ],
        },
    )

    audit = verify_claim_governance(report_path, now=_issued_at() + timedelta(days=1))

    assert audit.verdict is GovernanceVerdict.PASS
    assert audit.required_human_review is False
    assert any("Hash-matched human_review_note" in reason for reason in audit.reasons)


def test_naive_verification_time_fails_closed(tmp_path: Path) -> None:
    report_path = _write_package(tmp_path)

    audit = verify_claim_governance(report_path, now=datetime(2026, 1, 2))

    assert audit.verdict is GovernanceVerdict.FAIL
    assert any("timezone-aware" in reason for reason in audit.reasons)


def test_cli_writes_audit_json_and_uses_verdict_exit_codes(tmp_path: Path) -> None:
    report_path = _write_package(tmp_path)
    audit_path = tmp_path / "audit.json"

    passed = _run_cli(
        [
            "verify-claim-governance",
            str(report_path),
            "--now",
            "2026-01-02T00:00:00Z",
            "--out",
            str(audit_path),
        ]
    )
    assert passed.returncode == 0, passed.stderr
    assert "Claim governance verdict: PASS" in passed.stdout
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "cc/claim-governance-audit.v1"
    assert payload["verdict"] == "pass"

    expired = _run_cli(
        [
            "verify-claim-governance",
            str(report_path),
            "--now",
            "2026-01-16T00:00:00Z",
        ]
    )
    assert expired.returncode == 2
    assert "Claim governance verdict: FAIL" in expired.stdout


def _write_package(
    tmp_path: Path,
    *,
    claim_level: str = "bounded_empirical",
    decay_payload: dict[str, Any] | None = None,
    scenario_payloads: list[dict[str, Any]] | None = None,
    extra_artifacts: list[EvidenceArtifact] | None = None,
    review_note_payload: dict[str, Any] | None = None,
    package_snapshot: dict[str, str] | None = None,
) -> Path:
    decay_path = tmp_path / "decay.json"
    upper_path = tmp_path / "extremal_upper.json"
    lower_path = tmp_path / "extremal_lower.json"
    _write_json(decay_path, decay_payload or _decay_payload())
    scenarios = scenario_payloads or [_scenario_payload("upper"), _scenario_payload("lower")]
    _write_json(upper_path, scenarios[0])
    _write_json(lower_path, scenarios[1])

    artifacts = [
        _artifact(decay_path, "claim_decay"),
        _artifact(upper_path, "extremal_scenario"),
        _artifact(lower_path, "extremal_scenario"),
        *(extra_artifacts or []),
    ]
    if review_note_payload is not None:
        review_payload = dict(review_note_payload)
        if review_payload.get("reviewed_artifact_hashes") == "__ALL_BOUND_ARTIFACT_HASHES__":
            review_payload["reviewed_artifact_hashes"] = [artifact.sha256 for artifact in artifacts]
        review_note_path = tmp_path / "human_review_note.json"
        _write_json(review_note_path, review_payload)
        artifacts.append(_artifact(review_note_path, "human_review_note"))

    report = build_cc_report(
        run=RunSummary(
            run_id="governance-smoke",
            config_path=None,
            config_hash=None,
            seed=7,
            command="fixture governance verifier",
        ),
        calibration=CalibrationSummary(
            target_fpr=0.05,
            alpha_cap=0.05,
            realized_fpr=0.04,
            calibration_window={"lower": 0.04, "upper": 0.06},
            threshold=0.1,
            status="pass",
        ),
        measurement=MeasurementSummary(
            metric_family="CC",
            point_estimate=1.1,
            interval_lower=1.0,
            interval_upper=1.2,
            confidence_level=0.95,
            interval_method="FH-Bernstein",
            sample_sizes={"n": 64},
        ),
        claim=ClaimSummary(
            statement="Bounded empirical CC claim for the governance verifier fixture.",
            allowed_claim_level=claim_level,
            non_claims=[
                "A receipt verifies artifact integrity, not statistical validity or deployment safety.",
                "This report does not certify production safety.",
            ],
        ),
        evidence_artifacts=artifacts,
        report_id="cc-governance-smoke",
        created_at="2026-01-01T00:00:00Z",
        framework_version="0.3.1-fixture",
        git=GitMetadata(commit="0" * 40, dirty=False, branch="main"),
        environment=EnvironmentMetadata(
            python_version="3.12.0",
            platform="fixture-platform",
            package_snapshot=package_snapshot,
        ),
    )
    report_path = tmp_path / "report.json"
    write_cc_report(report_path, report)
    return report_path


def _issued_at() -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc)


def _decay_payload() -> dict[str, Any]:
    return ClaimDecayRecord(
        claim_id="governance-smoke-claim",
        issued_at=_issued_at(),
        policy=ClaimDecayPolicy(
            policy_id="ttl",
            degraded_after_days=7.0,
            expires_after_days=14.0,
        ),
        version_watch_set=VersionWatchSet(),
    ).to_dict()


def _scenario_payload(endpoint: str) -> dict[str, Any]:
    result = frechet_bounds([0.4, 0.6], event="and", return_distributions=True)
    scenario = ExtremalScenario.from_frechet_result(result, endpoint=endpoint)  # type: ignore[arg-type]
    return scenario.model_dump(mode="json")


def _artifact(path: Path, role: str) -> EvidenceArtifact:
    return EvidenceArtifact(
        path=path.name,
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        role=role,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    return subprocess.run(
        [sys.executable, "-m", "cc.reporting.cli", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

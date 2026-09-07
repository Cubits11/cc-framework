# tests/unit/evidence/test_claim_governance.py
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from cc.evidence import ClaimGovernanceAudit, GovernanceVerdict, verify_claim_governance
from cc.evidence.claim_envelope import compile_claim_envelope
from cc.evidence.claim_governance import (
    CLAIM_LEVEL_NON_CLAIM,
    GOVERNANCE_PASS_CAVEAT,
    RECEIPT_NON_CLAIM,
    ClaimFreshnessStatus,
    _empty_envelope_support_summary,
)
from cc.evidence.decay import ClaimDecayPolicy, ClaimDecayRecord, VersionWatchSet
from cc.evidence.extremal_scenario import (
    FEASIBILITY_NOT_LIKELIHOOD_NON_CLAIM,
    ExtremalScenario,
)
from cc.kernel.frechet_classes import frechet_bounds
from cc.reporting.canonical import LEGACY_SORT_KEYS, sha256_canonical
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


def test_claim_governance_audit_schema_alias_preserves_public_json(tmp_path: Path) -> None:
    report_path = _write_package(tmp_path)

    audit = verify_claim_governance(report_path, now=_issued_at() + timedelta(days=1))
    payload = audit.model_dump(mode="json", by_alias=True)

    assert payload["schema"] == "cc/claim-governance-audit.v1"
    assert "schema_" not in payload


def test_claim_governance_audit_accepts_public_schema_key() -> None:
    audit = ClaimGovernanceAudit(
        schema="cc/claim-governance-audit.v1",
        report_id="schema-alias-smoke",
        evaluated_at="2026-01-02T00:00:00Z",
        verdict=GovernanceVerdict.PASS,
        allowed_claim_level="diagnostic",
        claim_statement="Schema alias smoke test.",
        receipt={
            "report_hash_verified": None,
            "artifact_hashes_verified": False,
            "canonical_hash": None,
            "reason": "No receipt checked in alias smoke test.",
        },
        evidence_artifacts=[],
        decay={
            "present": False,
            "status": ClaimFreshnessStatus.NOT_EVALUATED,
            "reason": "No decay checked in alias smoke test.",
            "evaluated_at": None,
        },
        scenarios={
            "present": False,
            "scenario_count": 0,
            "scenario_ids": [],
            "kinds": [],
            "infeasible_count": 0,
            "excluded_evidence_fields": [],
        },
        confirmatory_protocols={
            "present": False,
            "artifact_count": 0,
            "protocol_ids": [],
            "run_ids": [],
            "failed_count": 0,
            "review_count": 0,
        },
        boundary={
            "claim_non_claim_count": 0,
            "artifact_non_claim_count": 0,
            "mandatory_non_claims_missing": [],
        },
        required_human_review=False,
        reasons=[],
        non_claims=[
            GOVERNANCE_PASS_CAVEAT,
            RECEIPT_NON_CLAIM,
            CLAIM_LEVEL_NON_CLAIM,
        ],
        envelope_support=_empty_envelope_support_summary(),
    )

    assert audit.schema_ == "cc/claim-governance-audit.v1"


def test_claim_governance_pass_requires_pass_caveat_non_claim() -> None:
    with pytest.raises(ValidationError, match="PASS caveat"):
        ClaimGovernanceAudit(
            schema="cc/claim-governance-audit.v1",
            report_id="missing-pass-caveat",
            evaluated_at="2026-01-02T00:00:00Z",
            verdict=GovernanceVerdict.PASS,
            allowed_claim_level="diagnostic",
            claim_statement="This malformed audit lacks mandatory governance caveats.",
            receipt={
                "report_hash_verified": None,
                "artifact_hashes_verified": False,
                "canonical_hash": None,
                "reason": "No receipt checked in malformed test.",
            },
            evidence_artifacts=[],
            decay={
                "present": False,
                "status": ClaimFreshnessStatus.NOT_EVALUATED,
                "reason": "No decay checked in malformed test.",
                "evaluated_at": None,
            },
            scenarios={
                "present": False,
                "scenario_count": 0,
                "scenario_ids": [],
                "kinds": [],
                "infeasible_count": 0,
                "excluded_evidence_fields": [],
            },
            confirmatory_protocols={
                "present": False,
                "artifact_count": 0,
                "protocol_ids": [],
                "run_ids": [],
                "failed_count": 0,
                "review_count": 0,
            },
            boundary={
                "claim_non_claim_count": 0,
                "artifact_non_claim_count": 0,
                "mandatory_non_claims_missing": [],
            },
            required_human_review=False,
            reasons=[],
            non_claims=[],
            envelope_support=_empty_envelope_support_summary(),
        )


def test_unsupported_report_schema_version_fails_closed(tmp_path: Path) -> None:
    report_path = _write_package(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["schema_version"] = "cc.report.v999"
    _write_json(report_path, report)

    audit = verify_claim_governance(report_path, now=_issued_at() + timedelta(days=1))

    assert audit.verdict is GovernanceVerdict.FAIL
    assert any("schema_version must be cc.report.v0.3.1" in reason for reason in audit.reasons)


def test_structural_failure_audit_preserves_governance_non_claims(tmp_path: Path) -> None:
    report_path = tmp_path / "broken.json"
    report_path.write_text("[]\n", encoding="utf-8")

    audit = verify_claim_governance(report_path, now=_issued_at() + timedelta(days=1))

    assert audit.verdict is GovernanceVerdict.FAIL
    assert audit.required_human_review is True
    assert GOVERNANCE_PASS_CAVEAT in audit.non_claims
    assert RECEIPT_NON_CLAIM in audit.non_claims
    assert CLAIM_LEVEL_NON_CLAIM in audit.non_claims


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
    assert GOVERNANCE_PASS_CAVEAT in audit.non_claims
    assert RECEIPT_NON_CLAIM in audit.non_claims
    assert CLAIM_LEVEL_NON_CLAIM in audit.non_claims
    assert "does not prove safety" in GOVERNANCE_PASS_CAVEAT
    assert "deployment safety" in audit.receipt.reason


def test_governance_verifies_a_valid_legacy_receipt_under_its_declared_profile(
    tmp_path: Path,
) -> None:
    report_path = _write_package(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["receipt"]["canonicalization_method"] = LEGACY_SORT_KEYS
    report["receipt"]["canonical_hash"] = sha256_canonical(report, profile=LEGACY_SORT_KEYS)
    _write_json(report_path, report)

    audit = verify_claim_governance(report_path, now=_issued_at() + timedelta(days=1))

    assert audit.verdict is GovernanceVerdict.PASS
    assert audit.receipt.report_hash_verified is True
    assert audit.receipt.canonical_hash == report["receipt"]["canonical_hash"]

    # Changing the declared profile without reissuing the receipt must fail.
    report["receipt"]["canonicalization_method"] = "unsupported-profile"
    _write_json(report_path, report)
    tampered = verify_claim_governance(report_path, now=_issued_at() + timedelta(days=1))
    assert tampered.verdict is GovernanceVerdict.FAIL
    assert tampered.receipt.report_hash_verified is False


def test_builder_supplies_extremal_scenario_non_claim_for_governance_pass(
    tmp_path: Path,
) -> None:
    upper = _scenario_payload("upper")
    lower = _scenario_payload("lower")

    assert FEASIBILITY_NOT_LIKELIHOOD_NON_CLAIM in upper["non_claims"]
    assert FEASIBILITY_NOT_LIKELIHOOD_NON_CLAIM in lower["non_claims"]
    assert "extremal scenario" in FEASIBILITY_NOT_LIKELIHOOD_NON_CLAIM.lower()

    report_path = _write_package(tmp_path, scenario_payloads=[upper, lower])
    audit = verify_claim_governance(report_path, now=_issued_at() + timedelta(days=1))

    assert audit.verdict is GovernanceVerdict.PASS
    assert audit.boundary.mandatory_non_claims_missing == []
    assert FEASIBILITY_NOT_LIKELIHOOD_NON_CLAIM in audit.non_claims


def test_governance_pass_is_explicitly_not_safety_certification(tmp_path: Path) -> None:
    report_path = _write_package(tmp_path)

    audit = verify_claim_governance(report_path, now=_issued_at() + timedelta(days=1))
    serialized = json.dumps(audit.model_dump(mode="json", by_alias=True), sort_keys=True).lower()

    assert audit.verdict is GovernanceVerdict.PASS
    assert "internal consistency under verifier rules only" in serialized
    assert "does not prove safety" in serialized
    assert "deployment validity" in serialized
    assert "production readiness" in serialized
    assert "compliance" in serialized


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


def test_evidence_artifact_paths_cannot_escape_report_base_dir(tmp_path: Path) -> None:
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text(
        "outside evidence must not be loadable through report paths\n", encoding="utf-8"
    )
    escaping_artifacts = [
        EvidenceArtifact(
            path="../outside.txt",
            sha256=sha256_file(outside),
            bytes=outside.stat().st_size,
            role="artifact",
        ),
        EvidenceArtifact(
            path=str(outside),
            sha256=sha256_file(outside),
            bytes=outside.stat().st_size,
            role="artifact",
        ),
    ]
    report_path = _write_package(package_dir, extra_artifacts=escaping_artifacts)

    audit = verify_claim_governance(report_path, now=_issued_at() + timedelta(days=1))

    escaping_reasons = [
        artifact.reason
        for artifact in audit.evidence_artifacts
        if artifact.path in {"../outside.txt", str(outside)}
    ]
    assert audit.verdict is GovernanceVerdict.FAIL
    assert len(escaping_reasons) == 2
    assert all("escapes base_dir" in reason for reason in escaping_reasons)


def test_missing_extremal_scenario_non_claim_requires_review_or_fails(
    tmp_path: Path,
) -> None:
    upper = _scenario_payload("upper")
    lower = _scenario_payload("lower")
    upper["non_claims"] = ["This endpoint scenario has reviewer-facing diagnostics."]
    lower["non_claims"] = ["This endpoint scenario has reviewer-facing diagnostics."]
    report_path = _write_package(tmp_path, scenario_payloads=[upper, lower])

    audit = verify_claim_governance(report_path, now=_issued_at() + timedelta(days=1))

    assert audit.verdict in {GovernanceVerdict.NEEDS_REVIEW, GovernanceVerdict.FAIL}
    assert audit.required_human_review is True
    assert (
        "extremal_scenario_not_likely_world_proof" in audit.boundary.mandatory_non_claims_missing
        or any(
            "scenario is missing mandatory non-claims" in artifact.reason
            for artifact in audit.evidence_artifacts
        )
    )


def test_exploratory_interval_leakage_fails(tmp_path: Path) -> None:
    upper = _scenario_payload("upper")
    upper["metadata"]["exploratory_ci"] = [0.8, 1.0]
    report_path = _write_package(tmp_path, scenario_payloads=[upper, _scenario_payload("lower")])

    audit = verify_claim_governance(report_path, now=_issued_at() + timedelta(days=1))

    assert audit.verdict in {GovernanceVerdict.NEEDS_REVIEW, GovernanceVerdict.FAIL}
    assert any(
        "Exploratory evidence leaked into confirmatory surface" in reason
        for reason in audit.reasons
    )


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


def test_confirmatory_protocol_artifact_is_audited_by_governance(tmp_path: Path) -> None:
    protocol_path = tmp_path / "confirmatory_protocol.json"
    _write_json(protocol_path, _confirmatory_protocol_payload())
    report_path = _write_package(
        tmp_path,
        extra_artifacts=[_artifact(protocol_path, "confirmatory_protocol")],
    )

    audit = verify_claim_governance(report_path, now=_issued_at() + timedelta(days=1))

    assert audit.verdict is GovernanceVerdict.PASS
    assert audit.confirmatory_protocols.present is True
    assert audit.confirmatory_protocols.protocol_ids == ["confirmatory-protocol-1"]
    assert audit.confirmatory_protocols.failed_count == 0
    assert audit.envelope_support.relation_counts["confirmatory_tests"] >= 1
    assert audit.envelope_support.strongest_non_integrity_strength == "confirmatory"


def test_confirmatory_protocol_failure_fails_governance(tmp_path: Path) -> None:
    payload = _confirmatory_protocol_payload()
    payload["run"]["source_role"] = "exploratory_redteam"
    protocol_path = tmp_path / "confirmatory_protocol.json"
    _write_json(protocol_path, payload)
    report_path = _write_package(
        tmp_path,
        extra_artifacts=[_artifact(protocol_path, "confirmatory_protocol")],
    )

    audit = verify_claim_governance(report_path, now=_issued_at() + timedelta(days=1))

    assert audit.verdict is GovernanceVerdict.FAIL
    assert audit.confirmatory_protocols.failed_count == 1
    assert audit.envelope_support.relation_counts["invalidates"] >= 1
    assert audit.envelope_support.strongest_non_integrity_strength == "diagnostic"
    assert any("role rename" in reason for reason in audit.reasons)


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


def test_hash_matched_human_review_note_does_not_upgrade_evidence_strength(
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
    assert any("without upgrading evidence strength" in reason for reason in audit.reasons)
    assert audit.envelope_support.strongest_non_integrity_strength in {
        "theoretical_bound",
        "confirmatory",
        "diagnostic",
    }


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
    assert "schema_" not in payload
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


def _confirmatory_protocol_payload() -> dict[str, Any]:
    non_claims = [
        "Confirmatory validity depends on the pre-registered protocol and run separation, "
        "not on report polish.",
        "A confirmatory_protocol artifact does not certify deployment safety or external validity.",
        "Adaptive discovery evidence may motivate a hypothesis but cannot become confirmatory "
        "evidence by renaming its role.",
    ]
    return {
        "schema_version": "cc.confirmatory_protocol.v1",
        "artifact_id": "confirmatory-artifact-1",
        "plan": {
            "protocol_id": "confirmatory-protocol-1",
            "hypothesis": "The held-out matrix bounds the AND failure endpoint under the plan.",
            "discovery_ref": {
                "artifact_id": "adaptive-redteam-1",
                "artifact_role": "exploratory_redteam",
                "artifact_sha256": "a" * 64,
                "adaptive": True,
                "description": "Adaptive red-team discovery generated the hypothesis only.",
                "discovered_at": "2026-01-01T00:00:00Z",
            },
            "protocol_mode": "held_out_matrix",
            "created_at": "2026-01-02T00:00:00Z",
            "primary_endpoint": "and_failure_rate",
            "fixed_analysis_plan": {
                "analysis_id": "analysis-v1",
                "estimand": "AND composed guardrail failure rate",
                "interval_method": "fixed-binomial-upper-bound",
                "alpha": 0.05,
                "multiplicity_adjustment": "none_predeclared_single_endpoint",
                "frozen": True,
            },
            "sample_plan": {
                "sampling_frame": "held-out prompt matrix v1",
                "unit": "prompt",
                "target_n": 128,
                "held_out_selection": "pre_failure_pattern",
                "clustered_data": False,
            },
            "stopping_rule": {
                "rule_id": "fixed-n",
                "description": "Evaluate exactly the predeclared held-out matrix.",
                "max_samples": 128,
                "max_looks": 1,
                "early_stopping_allowed": False,
            },
            "cluster_blocking": None,
            "exclusion_rules": [
                {
                    "rule_id": "deduplicate-prompts",
                    "field": "prompt_id",
                    "reason": "Duplicate held-out prompts are excluded before the run starts.",
                }
            ],
            "decision_rule": {
                "rule_id": "upper-bound-gate",
                "description": "Pass only if the predeclared upper bound is below threshold.",
                "threshold": 0.05,
                "pass_condition": "upper_bound <= 0.05",
                "fail_condition": "upper_bound > 0.05",
            },
            "non_claims": non_claims,
        },
        "run": {
            "run_id": "confirmatory-run-1",
            "started_at": "2026-01-03T00:00:00Z",
            "completed_at": "2026-01-03T01:00:00Z",
            "artifact_id": "confirmatory-matrix-1",
            "artifact_role": "confirmatory_failure_matrix",
            "source_role": None,
            "artifact_sha256": "b" * 64,
            "primary_endpoint": "and_failure_rate",
            "analysis_plan_id": "analysis-v1",
            "used_adaptive_discovery_data": False,
            "held_out_set_chosen_after_failure_pattern": False,
            "clustered_data_observed": False,
            "non_claims": [],
        },
        "non_claims": non_claims,
    }


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


def test_claim_envelope_preserves_governance_audit_schema(tmp_path: Path) -> None:
    report_path = _write_package(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    audit = verify_claim_governance(report_path, now=_issued_at() + timedelta(days=1))

    envelope = compile_claim_envelope(report, governance_audit=audit)

    assert envelope.governance_state.verifier_schema == "cc/claim-governance-audit.v1"
    assert envelope.governance_state.verdict == "pass"
    assert envelope.governance_state.required_human_review is False


def test_claim_envelope_preserves_governance_non_claim_boundaries(tmp_path: Path) -> None:
    report_path = _write_package(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    audit = verify_claim_governance(report_path, now=_issued_at() + timedelta(days=1))

    envelope = compile_claim_envelope(report, governance_audit=audit)
    serialized = json.dumps(envelope.model_dump(mode="json"), sort_keys=True).lower()

    assert "internal consistency under verifier rules only" in serialized
    assert "does not prove safety" in serialized
    assert "not claim lifecycle states" in serialized


def test_cli_strict_unknown_roles_fails_unknown_evidence_role(tmp_path: Path) -> None:
    mystery = tmp_path / "mystery.txt"
    mystery.write_text("opaque evidence\n", encoding="utf-8")
    report_path = _write_package(
        tmp_path,
        extra_artifacts=[
            EvidenceArtifact(
                path="mystery.txt",
                sha256=sha256_file(mystery),
                bytes=mystery.stat().st_size,
                role="mystery_role",
            )
        ],
    )

    result = _run_cli(
        [
            "verify-claim-governance",
            str(report_path),
            "--now",
            "2026-01-02T00:00:00Z",
            "--strict-unknown-roles",
        ]
    )

    assert result.returncode == 2
    assert "Claim governance verdict: FAIL" in result.stdout

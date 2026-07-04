from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from cc.evidence import (
    ArtifactRef,
    ClaimEnvelope,
    GovernanceVerdict,
    SupportEdge,
    SupportGraph,
    claim_envelope_sha256,
    compile_claim_envelope,
    verify_claim_governance,
)
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


def test_receipt_evidence_cannot_support_statistical_validity() -> None:
    with pytest.raises(ValidationError, match="receipt_integrity evidence cannot support"):
        SupportGraph(
            receipt_refs=(
                ArtifactRef(
                    artifact_id="receipt:report",
                    subject_ref="report:1",
                    role="receipt_integrity",
                ),
            ),
            support_edges=(
                SupportEdge(
                    source_artifact_id="receipt:report",
                    target_claim_fragment="claim.statistical_validity",
                    relation="integrity_binds",
                    strength="integrity_only",
                ),
            ),
        )


def test_claim_decay_evidence_cannot_support_deployment_safety() -> None:
    with pytest.raises(ValidationError, match="claim_decay evidence cannot support"):
        SupportGraph(
            decay_refs=(
                ArtifactRef(
                    artifact_id="evidence:decay",
                    subject_ref="report:1",
                    role="claim_decay",
                ),
            ),
            support_edges=(
                SupportEdge(
                    source_artifact_id="evidence:decay",
                    target_claim_fragment="claim.deployment_safety",
                    relation="qualifies",
                    strength="diagnostic",
                ),
            ),
        )


def test_extremal_scenario_evidence_cannot_support_likelihood() -> None:
    with pytest.raises(ValidationError, match="extremal_scenario evidence cannot support"):
        SupportGraph(
            scenario_refs=(
                ArtifactRef(
                    artifact_id="evidence:scenario",
                    subject_ref="report:1",
                    role="extremal_scenario",
                ),
            ),
            support_edges=(
                SupportEdge(
                    source_artifact_id="evidence:scenario",
                    target_claim_fragment="claim.likelihood",
                    relation="bounds",
                    strength="diagnostic",
                ),
            ),
        )


def test_human_review_cannot_support_evidence_it_did_not_review() -> None:
    with pytest.raises(ValidationError, match="human review cannot support evidence"):
        SupportGraph(
            review_refs=(
                ArtifactRef(
                    artifact_id="review:1",
                    subject_ref="report:1",
                    role="human_review",
                    metadata={"reviewed_artifact_ids": ["evidence.reviewed"]},
                ),
            ),
            support_edges=(
                SupportEdge(
                    source_artifact_id="review:1",
                    target_claim_fragment="evidence.unreviewed",
                    relation="qualifies",
                    strength="weak",
                ),
            ),
        )


def test_unknown_evidence_role_is_preserved_without_strengthening_verdict(
    tmp_path: Path,
) -> None:
    report_path = _write_report_with_unknown_role(tmp_path)

    audit = verify_claim_governance(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    envelope = compile_claim_envelope(report, governance_audit=audit)

    unknown_refs = [
        ref for ref in envelope.support_graph.evidence_refs if ref.role == "mystery_role"
    ]
    unknown_edges = [
        edge
        for edge in envelope.support_graph.support_edges
        if edge.source_artifact_id == unknown_refs[0].artifact_id
    ]

    assert audit.verdict is GovernanceVerdict.NEEDS_REVIEW
    assert envelope.governance_state.verdict == "needs_review"
    assert audit.envelope_support.unknown_role_refs == 1
    assert len(unknown_refs) == 1
    assert unknown_edges
    assert all(edge.strength == "weak" for edge in unknown_edges)
    assert all(edge.relation == "requires_review" for edge in unknown_edges)


def test_support_edges_survive_round_trip_serialization(tmp_path: Path) -> None:
    report_path = _write_basic_report(tmp_path)
    audit = verify_claim_governance(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    envelope = compile_claim_envelope(report, governance_audit=audit)

    restored = ClaimEnvelope.model_validate_json(envelope.model_dump_json())

    assert restored.support_graph.support_edges == envelope.support_graph.support_edges
    assert restored.support_graph.receipt_refs[0].role == "receipt_integrity"
    assert claim_envelope_sha256(restored) == claim_envelope_sha256(envelope)
    assert audit.envelope_support.support_edge_count == len(envelope.support_graph.support_edges)


def test_claim_envelope_models_reject_extra_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        SupportEdge(
            source_artifact_id="receipt:report",
            target_claim_fragment="claim.integrity.receipt",
            relation="integrity_binds",
            strength="integrity_only",
            surprise="not allowed",
        )


def _write_report_with_unknown_role(tmp_path: Path) -> Path:
    mystery = tmp_path / "mystery.txt"
    mystery.write_text("opaque evidence\n", encoding="utf-8")
    return _write_basic_report(
        tmp_path,
        evidence_artifacts=[
            EvidenceArtifact(
                path=mystery.name,
                sha256=sha256_file(mystery),
                bytes=mystery.stat().st_size,
                role="mystery_role",
            )
        ],
    )


def _write_basic_report(
    tmp_path: Path,
    *,
    evidence_artifacts: list[EvidenceArtifact] | None = None,
) -> Path:
    report = build_cc_report(
        run=RunSummary(
            run_id="claim-envelope-smoke",
            config_path=None,
            config_hash=None,
            seed=11,
            command="fixture claim envelope",
        ),
        calibration=CalibrationSummary(
            target_fpr=0.05,
            alpha_cap=0.05,
            realized_fpr=0.04,
            calibration_window={"lower": 0.04, "upper": 0.06},
            threshold=0.2,
            status="pass",
        ),
        measurement=MeasurementSummary(
            metric_family="CC",
            point_estimate=0.5,
            interval_lower=0.4,
            interval_upper=0.6,
            confidence_level=0.95,
            interval_method="fixture-interval",
            sample_sizes={"n": 20},
        ),
        claim=ClaimSummary(
            statement="Diagnostic claim-envelope fixture claim.",
            allowed_claim_level="diagnostic",
            non_claims=[
                "A receipt verifies artifact integrity, not statistical validity or deployment safety.",
            ],
        ),
        evidence_artifacts=evidence_artifacts or [],
        report_id="cc-claim-envelope-smoke",
        created_at="2026-01-01T00:00:00Z",
        framework_version="0.3.1-fixture",
        git=GitMetadata(commit="0" * 40, dirty=False, branch="main"),
        environment=EnvironmentMetadata(
            python_version="3.12.0",
            platform="fixture-platform",
        ),
    )
    report_path = tmp_path / "report.json"
    write_cc_report(report_path, report)
    return report_path

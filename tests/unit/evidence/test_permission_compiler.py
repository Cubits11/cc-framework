from __future__ import annotations

import pytest
from pydantic import ValidationError

from cc.evidence.permission_compiler import (
    CANONICAL_FORBIDDEN_CLAIM_MARKERS,
    EvidenceArtifactRef,
    ForbiddenClaimExplanation,
    PermissionCompilationResult,
    PermissionEdge,
    PermissionReviewRequirement,
    compile_epistemic_permissions,
    explain_forbidden_claim,
    strongest_permission_strength,
)


def test_measurement_role_emits_diagnostic_bounded_permission() -> None:
    result = compile_epistemic_permissions(
        [
            EvidenceArtifactRef(
                artifact_id="measurement-1",
                role="measurement_evidence",
                payload={
                    "metric_family": "failure_rate",
                    "interval": [0.0, 0.1],
                },
            )
        ]
    )

    assert result.schema_version == "cc.epistemic_permission_compilation.v1"
    assert len(result.permissions) == 1

    edge = result.permissions[0]
    assert edge.source_artifact_id == "measurement-1"
    assert edge.source_role == "measurement_evidence"
    assert edge.relation == "bounds"
    assert edge.strength == "diagnostic"
    assert edge.target_claim_fragments == ("claim.statistical_interval",)
    assert edge.support_scope == "named_measurement_interval_under_report_scope"
    assert "measurement_evidence_not_deployment_safety" in edge.mandatory_non_claim_ids
    assert "deployment_safety" in edge.forbidden_claim_types

    assert "measurement_evidence_not_deployment_safety" in result.non_claims
    assert result.review_requirements == ()
    assert result.unknown_roles == ()
    assert result.rejected_roles == ()


def test_receipt_integrity_emits_integrity_only_permission_only() -> None:
    result = compile_epistemic_permissions(
        [
            {
                "artifact_id": "receipt-1",
                "role": "receipt_integrity",
                "payload": {"sha256": "abc"},
            }
        ]
    )

    assert len(result.permissions) == 1

    edge = result.permissions[0]
    assert edge.source_artifact_id == "receipt-1"
    assert edge.source_role == "receipt_integrity"
    assert edge.relation == "integrity_binds"
    assert edge.strength == "integrity_only"
    assert strongest_permission_strength(result.permissions) == "integrity_only"
    assert "statistical_validity" in edge.forbidden_claim_types
    assert "deployment_safety" in edge.forbidden_claim_types
    assert (
        "receipt_integrity_not_statistical_validity_or_deployment_safety"
        in edge.mandatory_non_claim_ids
    )


def test_integrity_bind_edges_cannot_have_non_integrity_strength() -> None:
    with pytest.raises(ValidationError, match="integrity_binds permission edges"):
        PermissionEdge(
            source_artifact_id="receipt-1",
            source_role="receipt_integrity",
            relation="integrity_binds",
            strength="diagnostic",
            target_claim_fragments=("claim.integrity.receipt",),
            support_scope="report_and_artifact_byte_integrity",
        )


def test_unknown_role_emits_no_support_and_requires_review() -> None:
    result = compile_epistemic_permissions(
        [
            EvidenceArtifactRef(
                artifact_id="mystery-1",
                role="mystery_role",
                payload={"anything": True},
            )
        ]
    )

    assert result.permissions == ()
    assert result.unknown_roles == ("mystery_role",)
    assert result.rejected_roles == ()
    assert result.review_requirements

    requirement = result.review_requirements[0]
    assert isinstance(requirement, PermissionReviewRequirement)
    assert requirement.requirement_id == "REQUIRES_ROLE_RESOLUTION_REVIEW"
    assert requirement.source_artifact_id == "mystery-1"
    assert requirement.source_role == "mystery_role"

    assert result.forbidden_claims
    assert result.forbidden_claims[0].claim_type == "any_claim_strengthening"


@pytest.mark.parametrize(
    "role",
    [
        "draft",
        "supported",
        "bounded",
        "challenged",
        "weakened",
        "expired",
        "revoked",
        "superseded",
        "non_claim",
    ],
)
def test_lifecycle_state_name_as_role_is_rejected(role: str) -> None:
    result = compile_epistemic_permissions(
        [
            EvidenceArtifactRef(
                artifact_id=f"{role}-artifact",
                role=role,
                payload={},
            )
        ]
    )

    assert result.permissions == ()
    assert result.unknown_roles == ()
    assert role in result.rejected_roles
    assert result.review_requirements
    assert result.review_requirements[0].requirement_id == ("REJECTED_LIFECYCLE_STATE_USED_AS_ROLE")


def test_deployment_safety_claim_is_forbidden_with_measurement_and_receipt() -> None:
    result = compile_epistemic_permissions(
        [
            EvidenceArtifactRef(
                artifact_id="measurement-1",
                role="measurement_evidence",
                payload={"metric_family": "failure_rate", "interval": [0.0, 0.1]},
            ),
            EvidenceArtifactRef(
                artifact_id="receipt-1",
                role="receipt_integrity",
                payload={"sha256": "abc"},
            ),
        ]
    )

    explanation = explain_forbidden_claim(
        "The system is validated for production and safe for deployment.",
        result,
    )

    assert isinstance(explanation, ForbiddenClaimExplanation)
    assert explanation.schema_version == "cc.forbidden_claim_explanation.v1"
    assert explanation.forbidden is True
    assert "DEPLOYMENT_SAFETY_CERTIFICATION" in explanation.matched_claim_types
    assert "deployment_safety" in explanation.matched_claim_types
    assert "measurement_evidence" in explanation.implicated_roles
    assert "receipt_integrity" in explanation.implicated_roles
    assert "measurement_evidence_not_deployment_safety" in explanation.required_non_claims
    assert (
        "receipt_integrity_not_statistical_validity_or_deployment_safety"
        in explanation.required_non_claims
    )


def test_canonical_deployment_marker_expansion_catches_paraphrase() -> None:
    assert "safe for deployment" in CANONICAL_FORBIDDEN_CLAIM_MARKERS["deployment_safety"]

    result = compile_epistemic_permissions(
        [
            EvidenceArtifactRef(
                artifact_id="measurement-1",
                role="measurement_evidence",
                payload={"metric_family": "failure_rate", "interval": [0.0, 0.1]},
            )
        ]
    )

    explanation = explain_forbidden_claim(
        "This model is safe for deployment.",
        result,
    )

    assert explanation.forbidden is True
    assert "deployment_safety" in explanation.matched_claim_types
    assert "safe for deployment" in explanation.matched_markers


def test_exploratory_redteam_cannot_emit_confirmatory_permission() -> None:
    result = compile_epistemic_permissions(
        [
            EvidenceArtifactRef(
                artifact_id="redteam-1",
                role="exploratory_redteam",
                payload={
                    "redteam_id": "adaptive-search-1",
                    "discovery_protocol": "adaptive cliff search",
                    "findings": [],
                },
            )
        ]
    )

    assert result.permissions
    assert {edge.strength for edge in result.permissions} == {"weak"}
    assert {edge.relation for edge in result.permissions} == {
        "exploratory_suggests",
        "requires_review",
    }

    explanation = explain_forbidden_claim(
        "This exploratory red-team run is confirmatory evidence for release.",
        result,
    )

    assert explanation.forbidden is True
    assert "confirmatory_evidence" in explanation.matched_claim_types
    assert "release_claim" in explanation.matched_claim_types
    assert "exploratory_redteam_not_confirmatory_certificate" in explanation.required_non_claims


def test_confirmatory_protocol_emits_confirmatory_but_not_deployment_safety() -> None:
    result = compile_epistemic_permissions(
        [
            EvidenceArtifactRef(
                artifact_id="protocol-1",
                role="confirmatory_protocol",
                payload={
                    "schema_version": "cc.confirmatory_protocol.v1",
                    "artifact_id": "protocol-1",
                    "plan": {"id": "plan-1"},
                    "run": {"id": "run-1"},
                    "non_claims": ["Confirmatory evidence is scoped to the declared protocol."],
                },
            )
        ]
    )

    assert any(edge.strength == "confirmatory" for edge in result.permissions)
    assert any(edge.relation == "confirmatory_tests" for edge in result.permissions)
    assert strongest_permission_strength(result.permissions) == "confirmatory"

    explanation = explain_forbidden_claim(
        "The confirmatory protocol certifies deployment safety.",
        result,
    )

    assert explanation.forbidden is True
    assert "deployment_safety" in explanation.matched_claim_types
    assert "confirmatory_protocol_not_deployment_safety" in explanation.required_non_claims


def test_mandatory_non_claims_are_preserved_across_roles() -> None:
    result = compile_epistemic_permissions(
        [
            EvidenceArtifactRef(
                artifact_id="scenario-1",
                role="extremal_scenario",
                payload={
                    "schema_version": "cc.extremal_scenario.v1",
                    "scenario_id": "upper-world",
                    "kind": "frechet_endpoint",
                    "feasibility": {"status": "feasible"},
                    "non_claims": ["Endpoint feasibility is not likelihood."],
                },
            ),
            EvidenceArtifactRef(
                artifact_id="review-1",
                role="human_review",
                payload={},
            ),
        ]
    )

    assert "extremal_scenario_not_likely_world_proof" in result.non_claims
    assert "human_review_not_evidence_upgrade" in result.non_claims

    edge_non_claims = {
        non_claim_id for edge in result.permissions for non_claim_id in edge.mandatory_non_claim_ids
    }
    assert edge_non_claims <= set(result.non_claims)


def test_human_review_does_not_upgrade_statistical_evidence() -> None:
    result = compile_epistemic_permissions(
        [
            EvidenceArtifactRef(
                artifact_id="review-1",
                role="human_review",
                payload={},
            )
        ]
    )

    assert result.permissions
    assert {edge.strength for edge in result.permissions} == {"weak"}

    explanation = explain_forbidden_claim(
        "Human review proves statistical validity and makes the system safe.",
        result,
    )

    assert explanation.forbidden is True
    assert "statistical_validity" in explanation.matched_claim_types
    assert "deployment_safety" in explanation.matched_claim_types
    assert "human_review_not_evidence_upgrade" in explanation.required_non_claims


def test_result_rejects_permission_edge_referencing_missing_non_claim() -> None:
    edge = PermissionEdge(
        source_artifact_id="measurement-1",
        source_role="measurement_evidence",
        relation="bounds",
        strength="diagnostic",
        target_claim_fragments=("claim.statistical_interval",),
        support_scope="named_measurement_interval_under_report_scope",
        mandatory_non_claim_ids=("missing_non_claim",),
        forbidden_claim_types=("deployment_safety",),
    )

    with pytest.raises(ValidationError, match="absent from mandatory_non_claims"):
        PermissionCompilationResult(
            permissions=(edge,),
            forbidden_claims=(),
            mandatory_non_claims=(),
            review_requirements=(),
            unknown_roles=(),
            rejected_roles=(),
            non_claims=(),
        )


def test_result_rejects_non_claim_projection_drift() -> None:
    result = compile_epistemic_permissions(
        [
            EvidenceArtifactRef(
                artifact_id="measurement-1",
                role="measurement_evidence",
                payload={"metric_family": "failure_rate", "interval": [0.0, 0.1]},
            )
        ]
    )

    with pytest.raises(ValidationError, match="non_claims must exactly project"):
        PermissionCompilationResult(
            permissions=result.permissions,
            forbidden_claims=result.forbidden_claims,
            mandatory_non_claims=result.mandatory_non_claims,
            review_requirements=result.review_requirements,
            unknown_roles=result.unknown_roles,
            rejected_roles=result.rejected_roles,
            non_claims=(),
        )


def test_result_rejects_unknown_role_support_leak() -> None:
    edge = PermissionEdge(
        source_artifact_id="mystery-1",
        source_role="mystery_role",
        relation="supports",
        strength="weak",
        target_claim_fragments=("claim.anything",),
        support_scope="invalid_unknown_role_support",
    )

    with pytest.raises(ValidationError, match="unknown or rejected roles"):
        PermissionCompilationResult(
            permissions=(edge,),
            forbidden_claims=(),
            mandatory_non_claims=(),
            review_requirements=(),
            unknown_roles=("mystery_role",),
            rejected_roles=(),
            non_claims=(),
        )


def test_no_permission_result_claims_safety_certification_or_truth() -> None:
    result = compile_epistemic_permissions(
        [
            EvidenceArtifactRef(
                artifact_id="measurement-1",
                role="measurement_evidence",
                payload={"metric_family": "failure_rate", "interval": [0.0, 0.1]},
            ),
            EvidenceArtifactRef(
                artifact_id="receipt-1",
                role="receipt_integrity",
                payload={"sha256": "abc"},
            ),
            EvidenceArtifactRef(
                artifact_id="protocol-1",
                role="confirmatory_protocol",
                payload={
                    "schema_version": "cc.confirmatory_protocol.v1",
                    "artifact_id": "protocol-1",
                    "plan": {},
                    "run": {},
                    "non_claims": [],
                },
            ),
        ]
    )

    serialized = result.model_dump_json().lower()

    forbidden_phrases = (
        "certified safe",
        "proves safety",
        "deployment approved",
        "truth proof",
        "globally true",
    )

    for phrase in forbidden_phrases:
        assert phrase not in serialized

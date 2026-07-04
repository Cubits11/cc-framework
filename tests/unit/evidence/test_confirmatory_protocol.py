from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from cc.evidence.confirmatory_protocol import (
    ConfirmatoryProtocolPlan,
    ProtocolAuditStatus,
    verify_confirmatory_protocol_artifact,
)


def test_valid_confirmatory_protocol_passes_v0_checks() -> None:
    audit = verify_confirmatory_protocol_artifact(_payload())

    assert audit.status is ProtocolAuditStatus.PASS
    assert audit.temporal_validity_check is True
    assert audit.protocol_id == "confirmatory-protocol-1"
    assert audit.run_id == "confirmatory-run-1"


def test_plan_timestamp_after_run_timestamp_fails() -> None:
    payload = _payload()
    payload["plan"]["created_at"] = "2026-01-04T00:00:00Z"

    audit = verify_confirmatory_protocol_artifact(payload)

    assert audit.status is ProtocolAuditStatus.FAIL
    assert audit.temporal_validity_check is False
    assert any("before the confirmatory run starts" in reason for reason in audit.reasons)


def test_plan_timestamp_equal_to_run_timestamp_fails() -> None:
    payload = _payload()
    payload["plan"]["created_at"] = payload["run"]["started_at"]

    audit = verify_confirmatory_protocol_artifact(payload)

    assert audit.status is ProtocolAuditStatus.FAIL
    assert audit.temporal_validity_check is False


def test_adaptive_discovery_artifact_cannot_be_confirmatory_by_role_rename() -> None:
    payload = _payload()
    payload["run"]["source_role"] = "exploratory_redteam"

    audit = verify_confirmatory_protocol_artifact(payload)

    assert audit.status is ProtocolAuditStatus.FAIL
    assert any("role rename" in reason for reason in audit.reasons)


def test_adaptive_discovery_hash_reuse_fails_firewall() -> None:
    payload = _payload()
    payload["run"]["artifact_sha256"] = payload["plan"]["discovery_ref"]["artifact_sha256"]

    audit = verify_confirmatory_protocol_artifact(payload)

    assert audit.status is ProtocolAuditStatus.FAIL
    assert any("adaptive discovery artifact hash is reused" in reason for reason in audit.reasons)


def test_held_out_set_chosen_after_failure_pattern_fails() -> None:
    payload = _payload()
    payload["plan"]["sample_plan"]["held_out_selection"] = "post_failure_pattern"

    audit = verify_confirmatory_protocol_artifact(payload)

    assert audit.status is ProtocolAuditStatus.FAIL
    assert any("after seeing the failure pattern" in reason for reason in audit.reasons)


def test_missing_stopping_rule_reviews_diagnostic_and_fails_confirmatory_surface() -> None:
    payload = _payload()
    payload["plan"]["stopping_rule"] = None

    diagnostic = verify_confirmatory_protocol_artifact(payload, claim_level="diagnostic")
    bounded = verify_confirmatory_protocol_artifact(payload, claim_level="bounded_empirical")

    assert diagnostic.status is ProtocolAuditStatus.NEEDS_REVIEW
    assert bounded.status is ProtocolAuditStatus.FAIL
    assert any("Missing stopping rule" in reason for reason in diagnostic.reasons)
    assert any("Missing stopping rule" in reason for reason in bounded.reasons)


def test_clustered_data_without_cluster_blocking_requires_review() -> None:
    payload = _payload()
    payload["plan"]["sample_plan"]["clustered_data"] = True
    payload["run"]["clustered_data_observed"] = True

    audit = verify_confirmatory_protocol_artifact(payload)

    assert audit.status is ProtocolAuditStatus.NEEDS_REVIEW
    assert any("Clustered data without" in reason for reason in audit.reasons)


def test_confirmatory_artifact_must_reference_both_plan_and_run() -> None:
    payload = _payload()
    del payload["run"]

    audit = verify_confirmatory_protocol_artifact(payload)

    assert audit.status is ProtocolAuditStatus.FAIL
    assert audit.protocol_id == "<invalid-protocol>"
    assert any("Confirmatory artifact is invalid" in reason for reason in audit.reasons)


def test_protocol_models_reject_extra_fields() -> None:
    plan = deepcopy(_payload()["plan"])
    plan["surprise_adaptation"] = True

    with pytest.raises(ValidationError, match="Extra inputs"):
        ConfirmatoryProtocolPlan.model_validate(plan)


def _payload() -> dict[str, object]:
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

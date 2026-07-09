# tests/unit/evidence/test_role_ontology.py
from __future__ import annotations

import pytest
from pydantic import ValidationError

from cc.evidence import (
    EvidenceRoleDefinition,
    classify_role,
    get_role_definition,
    mandatory_non_claims_for,
    support_permissions_for,
    validate_role_payload,
)


def test_role_ontology_definitions_reject_extra_fields() -> None:
    payload = get_role_definition("claim_decay").model_dump(mode="json")
    payload["surprise_policy_engine"] = True

    with pytest.raises(ValidationError, match="Extra inputs"):
        EvidenceRoleDefinition.model_validate(payload)


def test_required_and_forbidden_fields_are_enforced_on_payloads() -> None:
    missing = validate_role_payload(
        "claim_decay",
        {"schema_version": "cc.claim_decay.v1", "policy": {"policy_id": "ttl"}},
    )
    forbidden = validate_role_payload(
        "exploratory_redteam",
        {
            "redteam_id": "adaptive-search-1",
            "discovery_protocol": "adaptive cliff search",
            "findings": [],
            "confirmatory_ci": [0.1, 0.2],
        },
    )

    assert missing.valid is False
    assert "claim_id" in missing.missing_required_fields
    assert "issued_at" in missing.missing_required_fields
    assert forbidden.valid is False
    assert forbidden.matched_forbidden_fields == ("$.confirmatory_ci",)


def test_receipt_integrity_has_integrity_only_support() -> None:
    permissions = support_permissions_for("receipt_integrity")

    assert len(permissions) == 1
    assert permissions[0].relation == "integrity_binds"
    assert permissions[0].strength == "integrity_only"
    assert "statistical_validity" in {
        item.claim_type for item in get_role_definition("receipt_integrity").does_not_support
    }


def test_unknown_roles_have_no_support_power() -> None:
    validation = validate_role_payload("mystery_role", {"anything": True})

    assert classify_role("mystery_role") == "unknown"
    assert support_permissions_for("mystery_role") == ()
    assert validation.valid is True
    assert validation.known_role is False
    assert validation.review_required is True


def test_mandatory_non_claims_are_machine_readable() -> None:
    non_claims = mandatory_non_claims_for("extremal_scenario")

    assert non_claims
    assert non_claims[0].non_claim_id == "extremal_scenario_not_likely_world_proof"
    assert any("likely" in phrase for group in non_claims[0].phrase_groups for phrase in group)


def test_confirmatory_protocol_role_requires_plan_and_run() -> None:
    validation = validate_role_payload(
        "confirmatory_protocol",
        {
            "schema_version": "cc.confirmatory_protocol.v1",
            "artifact_id": "confirmatory-artifact-1",
            "non_claims": ["Confirmatory validity depends on the protocol, not on report polish."],
        },
    )

    assert validation.valid is False
    assert "plan" in validation.missing_required_fields
    assert "run" in validation.missing_required_fields


def test_confirmatory_protocol_has_scoped_confirmatory_support() -> None:
    definition = get_role_definition("confirmatory_protocol")
    permissions = support_permissions_for("confirmatory_protocol")

    assert definition.confirmatory_status == "confirmatory"
    assert "deployment_safety" in {item.claim_type for item in definition.does_not_support}
    assert any(permission.strength == "confirmatory" for permission in permissions)

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from cc.evidence.decay import (
    ClaimDecayPolicy,
    ClaimDecayRecord,
    ConfiguredHazardCovariates,
    ConfiguredHazardPolicy,
    DecayState,
    VersionWatchSet,
    evaluate_claim_decay,
)


def _issued_at() -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc)


def _ttl_record() -> ClaimDecayRecord:
    return ClaimDecayRecord(
        claim_id="ttl-claim",
        issued_at=_issued_at(),
        policy=ClaimDecayPolicy(
            policy_id="ttl",
            degraded_after_days=7.0,
            expires_after_days=14.0,
        ),
        version_watch_set=VersionWatchSet(
            model_versions={"model": "v1"},
            guardrail_versions={"rail": "a1"},
            dependency_versions={"numpy": "2.0"},
        ),
    )


def _configured_hazard_policy() -> ConfiguredHazardPolicy:
    return ConfiguredHazardPolicy(
        rationale="Policy prior from reviewer-approved freshness playbook; not calibration.",
        baseline_hazard_per_day=math.log(2.0) / 10.0,
        degraded_after_half_lives=1.0,
        expired_after_half_lives=2.0,
    )


def test_claim_decay_record_uses_strict_pydantic_validation() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        ClaimDecayRecord(
            claim_id="claim-1",
            issued_at=_issued_at(),
            policy=ClaimDecayPolicy(policy_id="ttl", expires_after_days=14.0),
            freshness="fresh",
        )

    with pytest.raises(ValidationError, match="timezone-aware"):
        ClaimDecayRecord(
            claim_id="claim-1",
            issued_at=datetime(2026, 1, 1),
            policy=ClaimDecayPolicy(policy_id="ttl", expires_after_days=14.0),
        )


def test_configured_hazard_policy_requires_explicit_rationale() -> None:
    with pytest.raises(ValidationError, match="rationale"):
        ConfiguredHazardPolicy(
            baseline_hazard_per_day=math.log(2.0) / 10.0,
        )

    with pytest.raises(ValidationError, match="rationale"):
        ConfiguredHazardPolicy(
            rationale="   ",
            baseline_hazard_per_day=math.log(2.0) / 10.0,
        )


def test_unknown_configured_hazard_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        ConfiguredHazardPolicy(
            rationale="Configured heuristic only.",
            baseline_hazard_per_day=math.log(2.0) / 10.0,
            survival_probability=0.9,
        )


def test_configured_hazard_policy_computes_heuristic_thresholds() -> None:
    configured_hazard = _configured_hazard_policy()
    record = ClaimDecayRecord(
        claim_id="hazard-claim",
        issued_at=_issued_at(),
        policy=ClaimDecayPolicy(policy_id="hazard", configured_hazard=configured_hazard),
    )

    assert configured_hazard.configured_half_life_days(record.hazard_covariates) == pytest.approx(
        10.0
    )
    assert evaluate_claim_decay(record, now=_issued_at() + timedelta(days=9)) is DecayState.FRESH
    assert (
        evaluate_claim_decay(record, now=_issued_at() + timedelta(days=10)) is DecayState.DEGRADED
    )
    assert evaluate_claim_decay(record, now=_issued_at() + timedelta(days=20)) is DecayState.EXPIRED


def test_configured_hazard_covariates_apply_heuristic_multiplier() -> None:
    configured_hazard = ConfiguredHazardPolicy(
        rationale="Configured heuristic only.",
        baseline_hazard_per_day=math.log(2.0) / 10.0,
        coefficients={"model_update_count": math.log(2.0)},
    )
    covariates = ConfiguredHazardCovariates(model_update_count=1)

    assert configured_hazard.configured_risk_rate_per_day(covariates) == pytest.approx(
        math.log(2.0) / 5.0
    )
    assert configured_hazard.configured_half_life_days(covariates) == pytest.approx(5.0)


def test_hazard_non_claims_appear_in_serialized_record() -> None:
    record = ClaimDecayRecord(
        claim_id="hazard-non-claims",
        issued_at=_issued_at(),
        policy=ClaimDecayPolicy(
            policy_id="hazard",
            configured_hazard=_configured_hazard_policy(),
        ),
    )

    payload = record.to_dict()
    non_claims = "\n".join(payload["non_claims"])

    assert "does not estimate a statistically calibrated claim-failure probability" in non_claims
    assert "does not prove the claim remains valid in deployment" in non_claims
    assert "verification-time staleness/risk heuristic" in non_claims


def test_serialization_keeps_live_freshness_out_of_signed_record() -> None:
    record = ClaimDecayRecord(
        claim_id="serialized-claim",
        claim_hash="a" * 64,
        issued_at=_issued_at(),
        policy=ClaimDecayPolicy(
            policy_id="hazard",
            expires_after_days=30.0,
            configured_hazard=_configured_hazard_policy(),
        ),
        hazard_covariates=ConfiguredHazardCovariates(data_drift_score=0.2),
        evidence_refs=("report.json",),
    )

    payload = record.to_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert "freshness" not in payload
    assert "state" not in payload
    assert "decay_state" not in payload
    assert "configured_risk_rate_per_day" not in encoded
    assert "configured_half_life_days" not in encoded
    assert '"fresh"' not in encoded
    assert '"degraded"' not in encoded
    assert '"expired"' not in encoded
    assert evaluate_claim_decay(record, now=_issued_at() + timedelta(days=31)) is DecayState.EXPIRED
    assert ClaimDecayRecord.model_validate_json(record.model_dump_json()) == record


def test_ttl_record_serializes_policy_at_issued_at_not_live_state() -> None:
    record = _ttl_record()
    payload = record.to_dict()

    assert payload["issued_at"].startswith("2026-01-01T00:00:00")
    assert payload["policy"]["degraded_after_days"] == 7.0
    assert payload["policy"]["expires_after_days"] == 14.0
    assert "fresh" not in json.dumps(payload)


def test_ttl_evaluation_before_degraded_threshold() -> None:
    assert (
        evaluate_claim_decay(_ttl_record(), now=_issued_at() + timedelta(days=6))
        is DecayState.FRESH
    )


def test_ttl_evaluation_after_degraded_before_expiry() -> None:
    assert (
        evaluate_claim_decay(_ttl_record(), now=_issued_at() + timedelta(days=7))
        is DecayState.DEGRADED
    )
    assert (
        evaluate_claim_decay(_ttl_record(), now=_issued_at() + timedelta(days=13, hours=23))
        is DecayState.DEGRADED
    )


def test_ttl_evaluation_after_expiry() -> None:
    assert (
        evaluate_claim_decay(_ttl_record(), now=_issued_at() + timedelta(days=14))
        is DecayState.EXPIRED
    )


def test_version_trigger_change_expires_claim() -> None:
    assert (
        evaluate_claim_decay(
            _ttl_record(),
            now=_issued_at() + timedelta(days=1),
            current_versions=VersionWatchSet(
                model_versions={"model": "v2"},
                guardrail_versions={"rail": "a1"},
                dependency_versions={"numpy": "2.0"},
            ),
        )
        is DecayState.EXPIRED
    )


def test_multiple_version_trigger_changes_expire_claim() -> None:
    assert (
        evaluate_claim_decay(
            _ttl_record(),
            now=_issued_at() + timedelta(days=1),
            current_versions=VersionWatchSet(
                model_versions={"model": "v2"},
                guardrail_versions={"rail": "b2"},
                dependency_versions={"numpy": "2.1"},
            ),
        )
        is DecayState.EXPIRED
    )


def test_evaluate_claim_decay_is_deterministic_for_fixed_now() -> None:
    record = _ttl_record()
    now = _issued_at() + timedelta(days=8)

    assert evaluate_claim_decay(record, now=now) is evaluate_claim_decay(record, now=now)


def test_timezone_aware_datetimes_are_normalized_by_datetime_arithmetic() -> None:
    eastern = timezone(timedelta(hours=-5))
    issued = datetime(2026, 1, 1, 0, 0, tzinfo=eastern)
    record = ClaimDecayRecord(
        claim_id="tz-claim",
        issued_at=issued,
        policy=ClaimDecayPolicy(policy_id="ttl", degraded_after_days=1.0),
    )

    assert (
        evaluate_claim_decay(record, now=datetime(2026, 1, 2, 5, 0, tzinfo=timezone.utc))
        is DecayState.DEGRADED
    )


def test_naive_verification_time_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        evaluate_claim_decay(_ttl_record(), now=datetime(2026, 1, 2))


def test_now_and_as_of_cannot_both_be_supplied() -> None:
    with pytest.raises(ValueError, match="only one"):
        evaluate_claim_decay(
            _ttl_record(),
            now=_issued_at() + timedelta(days=1),
            as_of=_issued_at() + timedelta(days=1),
        )

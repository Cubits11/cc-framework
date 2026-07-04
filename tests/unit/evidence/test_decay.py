from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from cc.evidence.decay import (
    ClaimDecayPolicy,
    ClaimDecayRecord,
    DecayState,
    HazardCovariates,
    HazardDecayPolicy,
    VersionWatchSet,
    evaluate_claim_decay,
)


def _issued_at() -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc)


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


def test_hazard_policy_computes_half_life_and_status_thresholds() -> None:
    hazard = HazardDecayPolicy(
        baseline_hazard_per_day=math.log(2.0) / 10.0,
        degraded_after_half_lives=1.0,
        expired_after_half_lives=2.0,
    )
    record = ClaimDecayRecord(
        claim_id="hazard-claim",
        issued_at=_issued_at(),
        policy=ClaimDecayPolicy(policy_id="hazard", hazard=hazard),
    )

    assert hazard.half_life_days(record.covariates) == pytest.approx(10.0)
    assert evaluate_claim_decay(record, as_of=_issued_at() + timedelta(days=9)) is DecayState.FRESH
    assert (
        evaluate_claim_decay(record, as_of=_issued_at() + timedelta(days=10)) is DecayState.DEGRADED
    )
    assert (
        evaluate_claim_decay(record, as_of=_issued_at() + timedelta(days=20)) is DecayState.EXPIRED
    )


def test_hazard_covariates_apply_proportional_hazard_multiplier() -> None:
    hazard = HazardDecayPolicy(
        baseline_hazard_per_day=math.log(2.0) / 10.0,
        coefficients={"model_update_count": math.log(2.0)},
    )
    covariates = HazardCovariates(model_update_count=1)

    assert hazard.hazard_rate_per_day(covariates) == pytest.approx(math.log(2.0) / 5.0)
    assert hazard.half_life_days(covariates) == pytest.approx(5.0)


def test_ttl_and_version_triggers_expire_claims_at_verification_time() -> None:
    record = ClaimDecayRecord(
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
        ),
    )

    assert evaluate_claim_decay(record, as_of=_issued_at() + timedelta(days=6)) is DecayState.FRESH
    assert (
        evaluate_claim_decay(record, as_of=_issued_at() + timedelta(days=7)) is DecayState.DEGRADED
    )
    assert (
        evaluate_claim_decay(record, as_of=_issued_at() + timedelta(days=14)) is DecayState.EXPIRED
    )
    assert (
        evaluate_claim_decay(
            record,
            as_of=_issued_at() + timedelta(days=1),
            current_versions=VersionWatchSet(
                model_versions={"model": "v2"},
                guardrail_versions={"rail": "a1"},
            ),
        )
        is DecayState.EXPIRED
    )


def test_serialization_keeps_live_freshness_out_of_signed_record() -> None:
    record = ClaimDecayRecord(
        claim_id="serialized-claim",
        claim_hash="a" * 64,
        issued_at=_issued_at(),
        policy=ClaimDecayPolicy(policy_id="ttl", expires_after_days=30.0),
        covariates=HazardCovariates(data_drift_score=0.2),
        evidence_refs=("report.json",),
    )

    payload = record.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True)

    assert "freshness" not in payload
    assert "state" not in payload
    assert "decay_state" not in payload
    assert '"fresh"' not in encoded
    assert '"expired"' not in encoded
    assert (
        evaluate_claim_decay(record, as_of=_issued_at() + timedelta(days=31)) is DecayState.EXPIRED
    )
    assert ClaimDecayRecord.model_validate_json(record.model_dump_json()) == record

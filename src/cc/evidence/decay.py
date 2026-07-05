"""Deterministic claim-decay policies for evidence freshness checks.

This module defines signed claim-decay artifacts and verification-time freshness
evaluation.

Important semantic boundary
---------------------------
`DecayState` is a freshness/support-state projection. It is not a claim
lifecycle state.

`DecayState.EXPIRED` means the claim-decay policy no longer permits the attached
evidence to be treated as fresh under this verifier's clock and observed version
context. It does not mean the historical claim was false, revoked, unsafe, or
fraudulent.

Future `cc.claims.ClaimState` may consume decay evaluations as inputs to a
lifecycle transition, but this module does not own lifecycle transitions.

This module owns:

- signed freshness policy records;
- TTL-based freshness evaluation;
- version-watch invalidation;
- configured heuristic hazard scoring;
- mandatory decay non-claims.

This module does not own:

- claim lifecycle state;
- claim revocation;
- claim challenge semantics;
- deployment safety;
- calibrated survival modeling;
- production/compliance certification.

Configured hazard support in this module is a configured heuristic risk score,
not a fitted or calibrated survival model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CLAIM_DECAY_SCHEMA_VERSION: Literal["cc.claim_decay.v1"] = "cc.claim_decay.v1"

CONFIGURED_HAZARD_NOTICE = (
    "This is a configured heuristic review-pressure score, not a fitted or calibrated "
    "survival model and not a probability that the claim is false."
)

DECAY_STATE_NON_LIFECYCLE_NOTICE = (
    "DecayState is a verification-time support projection, not a claim lifecycle state."
)

_DEFAULT_DECAY_NON_CLAIMS = (
    "A claim_decay artifact does not prove the system is currently safe; it defines when "
    "the claim's supporting evidence should be rechecked, degraded, or treated as expired.",
    "A claim_decay artifact records signed policy, not live deployment validity.",
    "A claim_decay artifact does not revoke, falsify, or certify a claim by itself.",
    DECAY_STATE_NON_LIFECYCLE_NOTICE,
)

_CONFIGURED_HAZARD_NON_CLAIMS = (
    "This decay policy does not estimate a statistically calibrated claim-failure probability.",
    "This decay policy does not prove the claim remains valid in deployment.",
    "This decay policy is a verification-time staleness/risk heuristic and "
    "review-pressure heuristic unless "
    "externally calibrated.",
    "Configured hazard coefficients are policy parameters, not fitted survival-model parameters.",
)

RESERVED_LIFECYCLE_STATE_NAMES = frozenset(
    {
        "draft",
        "supported",
        "bounded",
        "challenged",
        "weakened",
        "revoked",
        "superseded",
        "non_claim",
    }
)


class DecayModel(BaseModel):
    """Strict base model for claim-decay artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DecayState(str, Enum):
    """Verification-time freshness state for a claim or evidence item.

    This is not a claim lifecycle state.

    - FRESH: the policy does not currently impose degradation or expiry.
    - DEGRADED: the policy requires review or reduced reliance.
    - EXPIRED: the policy no longer permits fresh reliance on the support.
    """

    FRESH = "fresh"
    DEGRADED = "degraded"
    EXPIRED = "expired"


DecayStatus = DecayState


class DecayEvaluation(DecayModel):
    """Audit-friendly result of evaluating a claim-decay record.

    This object is intentionally separate from the signed `ClaimDecayRecord`.
    The signed record stores policy. The evaluation stores live verifier output.
    """

    claim_id: str = Field(min_length=1)
    state: DecayState
    evaluated_at: datetime
    age_days: float = Field(ge=0.0)
    triggered_rules: tuple[str, ...] = Field(default_factory=tuple)
    version_changes: tuple[str, ...] = Field(default_factory=tuple)
    non_claims: tuple[str, ...] = Field(default_factory=lambda: _DEFAULT_DECAY_NON_CLAIMS)

    @field_validator("evaluated_at", mode="before")
    @classmethod
    def _require_timezone(cls, value: datetime | str) -> datetime:
        return _coerce_timezone_aware_datetime(value, "evaluated_at")

    @field_validator("triggered_rules", "version_changes", "non_claims", mode="before")
    @classmethod
    def _coerce_string_tuple(cls, value: Any) -> tuple[str, ...]:
        return _coerce_clean_string_tuple(value)

    @model_validator(mode="after")
    def _expired_has_trigger(self) -> DecayEvaluation:
        if self.state is DecayState.EXPIRED and not (self.triggered_rules or self.version_changes):
            raise ValueError("expired decay evaluations must include a trigger")
        return self


class ConfiguredHazardCovariates(DecayModel):
    """Covariates for a configured heuristic risk score, not a calibrated model."""

    model_update_count: int = Field(default=0, ge=0)
    guardrail_update_count: int = Field(default=0, ge=0)
    dependency_update_count: int = Field(default=0, ge=0)
    data_drift_score: float = Field(default=0.0, ge=0.0)
    threat_activity_level: float = Field(default=0.0, ge=0.0)
    custom: dict[str, float] = Field(default_factory=dict)

    @field_validator("data_drift_score", "threat_activity_level")
    @classmethod
    def _finite_nonnegative_float(cls, value: float) -> float:
        item = float(value)
        if not math.isfinite(item) or item < 0.0:
            raise ValueError("hazard covariates must be finite non-negative values")
        return item

    @field_validator("custom")
    @classmethod
    def _validate_custom_covariates(cls, value: dict[str, float]) -> dict[str, float]:
        cleaned: dict[str, float] = {}
        for raw_key, raw_item in value.items():
            key = str(raw_key).strip()
            item = float(raw_item)
            if not key:
                raise ValueError("custom covariate names must be non-empty")
            if not math.isfinite(item):
                raise ValueError(f"custom covariate {key!r} must be finite")
            cleaned[key] = item
        return dict(sorted(cleaned.items()))

    def feature_map(self) -> dict[str, float]:
        """Return the configured heuristic covariate vector used for beta dot x scoring."""

        features = {
            "model_update_count": float(self.model_update_count),
            "guardrail_update_count": float(self.guardrail_update_count),
            "dependency_update_count": float(self.dependency_update_count),
            "data_drift_score": float(self.data_drift_score),
            "threat_activity_level": float(self.threat_activity_level),
        }
        features.update(self.custom)
        return dict(sorted(features.items()))


class ConfiguredHazardPolicy(DecayModel):
    """Configured proportional-hazard heuristic review-pressure policy.

    This is a configured heuristic risk score, not a fitted or calibrated
    survival model. Callers supply the baseline hazard and coefficients, and
    evaluation uses:

        lambda = baseline_hazard_per_day * exp(beta dot x)

    The resulting half-life is a policy-derived review-pressure clock, not an
    empirically established half-life of truth, safety, or deployment validity.
    """

    rationale: str = Field(
        min_length=1,
        description=(
            "Required explanation for using a configured heuristic risk score; "
            "this is not calibration evidence."
        ),
    )
    baseline_hazard_per_day: float = Field(
        gt=0.0,
        description=("Configured heuristic baseline hazard per day. " + CONFIGURED_HAZARD_NOTICE),
    )
    coefficients: dict[str, float] = Field(
        default_factory=dict,
        description="Configured heuristic coefficients; not fitted survival-model parameters.",
    )
    degraded_after_half_lives: float = Field(
        default=1.0,
        gt=0.0,
        description="Configured heuristic threshold in half-life units, not an empirical half-life.",
    )
    expired_after_half_lives: float = Field(
        default=2.0,
        gt=0.0,
        description="Configured heuristic threshold in half-life units, not an empirical half-life.",
    )
    calibration_notice: Literal["configured_heuristic_not_calibrated"] = (
        "configured_heuristic_not_calibrated"
    )

    @field_validator("rationale")
    @classmethod
    def _validate_rationale(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("configured hazard rationale must be non-empty")
        return stripped

    @field_validator("baseline_hazard_per_day")
    @classmethod
    def _validate_baseline_hazard(cls, value: float) -> float:
        item = float(value)
        if not math.isfinite(item) or item <= 0.0:
            raise ValueError("baseline_hazard_per_day must be positive and finite")
        return item

    @field_validator("coefficients")
    @classmethod
    def _validate_coefficients(cls, value: dict[str, float]) -> dict[str, float]:
        cleaned: dict[str, float] = {}
        for raw_key, raw_item in value.items():
            key = str(raw_key).strip()
            item = float(raw_item)
            if not key:
                raise ValueError("hazard coefficient names must be non-empty")
            if not math.isfinite(item):
                raise ValueError(f"hazard coefficient {key!r} must be finite")
            cleaned[key] = item
        return dict(sorted(cleaned.items()))

    @model_validator(mode="after")
    def _validate_threshold_order(self) -> ConfiguredHazardPolicy:
        if self.expired_after_half_lives < self.degraded_after_half_lives:
            raise ValueError("expired_after_half_lives must be >= degraded_after_half_lives")
        return self

    def configured_risk_rate_per_day(self, covariates: ConfiguredHazardCovariates) -> float:
        """Return configured heuristic ``baseline_hazard_per_day * exp(beta dot x)``."""

        features = covariates.feature_map()
        linear = 0.0
        for key, beta in self.coefficients.items():
            linear += float(beta) * float(features.get(key, 0.0))
        try:
            multiplier = math.exp(linear)
        except OverflowError as exc:
            raise ValueError("hazard linear predictor overflowed") from exc
        hazard = float(self.baseline_hazard_per_day) * multiplier
        if not math.isfinite(hazard) or hazard <= 0.0:
            raise ValueError("configured hazard must evaluate to a positive finite value")
        return hazard

    def configured_half_life_days(self, covariates: ConfiguredHazardCovariates) -> float:
        """Return heuristic half-life ``ln(2) / lambda``; this is not calibrated."""

        return math.log(2.0) / self.configured_risk_rate_per_day(covariates)


class VersionWatchSet(DecayModel):
    """Pinned versions that invalidate freshness when observed versions change."""

    model_versions: dict[str, str] = Field(default_factory=dict)
    guardrail_versions: dict[str, str] = Field(default_factory=dict)
    data_versions: dict[str, str] = Field(default_factory=dict)
    dependency_versions: dict[str, str] = Field(default_factory=dict)

    @field_validator(
        "model_versions",
        "guardrail_versions",
        "data_versions",
        "dependency_versions",
    )
    @classmethod
    def _validate_versions(cls, value: dict[str, str]) -> dict[str, str]:
        cleaned: dict[str, str] = {}
        for raw_key, raw_item in value.items():
            key = str(raw_key).strip()
            item = str(raw_item).strip()
            if not key:
                raise ValueError("version watch keys must be non-empty")
            if not item:
                raise ValueError(f"version for {key!r} must be non-empty")
            cleaned[key] = item
        return dict(sorted(cleaned.items()))

    def changes_from(self, current: VersionWatchSet | dict[str, Any]) -> tuple[str, ...]:
        """Return watched version paths whose observed values differ.

        Missing observed keys do not trigger expiry because absence is ambiguous.
        They should instead be handled as a governance review gap by callers.
        """

        observed = (
            current
            if isinstance(current, VersionWatchSet)
            else VersionWatchSet.model_validate(current)
        )
        changed: list[str] = []
        for field_name in (
            "model_versions",
            "guardrail_versions",
            "data_versions",
            "dependency_versions",
        ):
            expected_versions = getattr(self, field_name)
            observed_versions = getattr(observed, field_name)
            for key, expected in sorted(expected_versions.items()):
                if key in observed_versions and observed_versions[key] != expected:
                    changed.append(f"{field_name}.{key}")
        return tuple(changed)

    def missing_from(self, current: VersionWatchSet | dict[str, Any]) -> tuple[str, ...]:
        """Return watched version paths that are absent from observed versions."""

        observed = (
            current
            if isinstance(current, VersionWatchSet)
            else VersionWatchSet.model_validate(current)
        )
        missing: list[str] = []
        for field_name in (
            "model_versions",
            "guardrail_versions",
            "data_versions",
            "dependency_versions",
        ):
            expected_versions = getattr(self, field_name)
            observed_versions = getattr(observed, field_name)
            for key in sorted(expected_versions):
                if key not in observed_versions:
                    missing.append(f"{field_name}.{key}")
        return tuple(missing)

    @property
    def is_empty(self) -> bool:
        """Return true when no versions are pinned."""

        return not any(
            (
                self.model_versions,
                self.guardrail_versions,
                self.data_versions,
                self.dependency_versions,
            )
        )


class ClaimDecayPolicy(DecayModel):
    """Declarative freshness policy attached to a claim-decay record."""

    policy_id: str = Field(min_length=1)
    degraded_after_days: float | None = Field(default=None, gt=0.0)
    expires_after_days: float | None = Field(default=None, gt=0.0)
    configured_hazard: ConfiguredHazardPolicy | None = Field(
        default=None,
        description=CONFIGURED_HAZARD_NOTICE,
    )

    @field_validator("policy_id")
    @classmethod
    def _policy_id_is_not_lifecycle_state(cls, value: str) -> str:
        stripped = value.strip()
        if stripped in RESERVED_LIFECYCLE_STATE_NAMES:
            raise ValueError("policy_id must not reuse a claim lifecycle state name")
        return stripped

    @model_validator(mode="after")
    def _validate_policy(self) -> ClaimDecayPolicy:
        if (
            self.degraded_after_days is None
            and self.expires_after_days is None
            and self.configured_hazard is None
        ):
            raise ValueError("policy must include TTL thresholds or a hazard policy")
        if (
            self.degraded_after_days is not None
            and self.expires_after_days is not None
            and self.expires_after_days < self.degraded_after_days
        ):
            raise ValueError("expires_after_days must be >= degraded_after_days")
        return self


class ClaimDecayRecord(DecayModel):
    """Signed claim-decay artifact without a live freshness verdict.

    This object intentionally stores policy, not live evaluation result.

    Do not add a `state`, `status`, `fresh`, `expired`, or `verdict` field here.
    Use `evaluate_claim_decay` or `evaluate_claim_decay_with_details` at
    verification time.
    """

    schema_version: Literal["cc.claim_decay.v1"] = CLAIM_DECAY_SCHEMA_VERSION
    claim_id: str = Field(min_length=1)
    claim_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    issued_at: datetime
    policy: ClaimDecayPolicy
    hazard_covariates: ConfiguredHazardCovariates = Field(
        default_factory=ConfiguredHazardCovariates,
        description=(
            "Inputs for configured heuristic risk scoring; not observed survival-model data."
        ),
    )
    version_watch_set: VersionWatchSet = Field(default_factory=VersionWatchSet)
    evidence_refs: tuple[str, ...] = Field(default_factory=tuple)
    notes: tuple[str, ...] = Field(default_factory=tuple)
    non_claims: tuple[str, ...] = Field(default_factory=lambda: _DEFAULT_DECAY_NON_CLAIMS)

    @model_validator(mode="before")
    @classmethod
    def _reject_live_state_and_ensure_non_claims(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        payload = dict(data)
        forbidden_live_fields = {
            "state",
            "status",
            "freshness_status",
            "verdict",
            "is_fresh",
            "is_expired",
            "evaluated_at",
        }
        present = sorted(forbidden_live_fields & set(payload))
        if present:
            raise ValueError(
                "ClaimDecayRecord stores signed policy only and must not contain live "
                f"freshness fields: {present}"
            )
        non_claims = _coerce_clean_string_tuple(
            payload.get("non_claims") or _DEFAULT_DECAY_NON_CLAIMS
        )
        non_claims = _append_missing(non_claims, _DEFAULT_DECAY_NON_CLAIMS)
        if _policy_has_configured_hazard(payload.get("policy")):
            non_claims = _append_missing(non_claims, _CONFIGURED_HAZARD_NON_CLAIMS)
        payload["non_claims"] = non_claims
        return payload

    @field_validator("claim_id")
    @classmethod
    def _claim_id_is_not_lifecycle_state(cls, value: str) -> str:
        stripped = value.strip()
        if stripped in RESERVED_LIFECYCLE_STATE_NAMES:
            raise ValueError("claim_id must not reuse a claim lifecycle state name")
        return stripped

    @field_validator("issued_at", mode="before")
    @classmethod
    def _require_timezone(cls, value: datetime | str) -> datetime:
        return _coerce_timezone_aware_datetime(value, "issued_at")

    @field_validator("evidence_refs", "notes", "non_claims", mode="before")
    @classmethod
    def _coerce_string_collections(cls, value: Any) -> tuple[str, ...]:
        return _coerce_clean_string_tuple(value)

    @field_validator("evidence_refs", "notes", "non_claims")
    @classmethod
    def _non_empty_strings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _coerce_clean_string_tuple(value)

    def to_dict(self) -> dict[str, Any]:
        """Serialize signed policy data only, never live fresh/degraded/expired state."""

        return self.model_dump(mode="json")


def evaluate_claim_decay(
    record: ClaimDecayRecord | Mapping[str, Any],
    *,
    now: datetime | None = None,
    as_of: datetime | None = None,
    current_versions: VersionWatchSet | dict[str, Any] | None = None,
) -> DecayState:
    """Evaluate a claim's freshness state at verification time.

    Returns only `DecayState` for backward compatibility. Use
    `evaluate_claim_decay_with_details` when audit details are needed.
    """

    return evaluate_claim_decay_with_details(
        record,
        now=now,
        as_of=as_of,
        current_versions=current_versions,
    ).state


def evaluate_claim_decay_with_details(
    record: ClaimDecayRecord | Mapping[str, Any],
    *,
    now: datetime | None = None,
    as_of: datetime | None = None,
    current_versions: VersionWatchSet | dict[str, Any] | None = None,
) -> DecayEvaluation:
    """Evaluate freshness and return an audit-friendly deterministic result."""

    if not isinstance(record, ClaimDecayRecord):
        record = ClaimDecayRecord.model_validate(record)

    if now is not None and as_of is not None:
        raise ValueError("provide only one of now or as_of")

    verification_time = now or as_of or datetime.now(timezone.utc)
    verification_time = _coerce_timezone_aware_datetime(verification_time, "now/as_of")

    if verification_time < record.issued_at:
        raise ValueError("now/as_of cannot be earlier than record.issued_at")

    age_days = (verification_time - record.issued_at).total_seconds() / 86400.0
    triggered_rules: list[str] = []
    version_changes: tuple[str, ...] = ()

    if current_versions is not None:
        version_changes = record.version_watch_set.changes_from(current_versions)
        if version_changes:
            triggered_rules.append("version_watch_set_changed")

    state = DecayState.FRESH

    if version_changes:
        state = DecayState.EXPIRED
    else:
        if (
            record.policy.expires_after_days is not None
            and age_days >= record.policy.expires_after_days
        ):
            state = DecayState.EXPIRED
            triggered_rules.append("expires_after_days")
        elif (
            record.policy.degraded_after_days is not None
            and age_days >= record.policy.degraded_after_days
        ):
            state = DecayState.DEGRADED
            triggered_rules.append("degraded_after_days")

        if record.policy.configured_hazard is not None:
            half_life = record.policy.configured_hazard.configured_half_life_days(
                record.hazard_covariates
            )
            expired_threshold = record.policy.configured_hazard.expired_after_half_lives * half_life
            degraded_threshold = (
                record.policy.configured_hazard.degraded_after_half_lives * half_life
            )

            if age_days >= expired_threshold:
                state = DecayState.EXPIRED
                triggered_rules.append("configured_hazard_expired_threshold")
            elif state is DecayState.FRESH and age_days >= degraded_threshold:
                state = DecayState.DEGRADED
                triggered_rules.append("configured_hazard_degraded_threshold")

    return DecayEvaluation(
        claim_id=record.claim_id,
        state=state,
        evaluated_at=verification_time,
        age_days=age_days,
        triggered_rules=_dedupe(triggered_rules),
        version_changes=version_changes,
        non_claims=record.non_claims,
    )


def _policy_has_configured_hazard(policy: Any) -> bool:
    if isinstance(policy, ClaimDecayPolicy):
        return policy.configured_hazard is not None
    if isinstance(policy, Mapping):
        return policy.get("configured_hazard") is not None
    return False


def _append_missing(existing: tuple[str, ...], required: tuple[str, ...]) -> tuple[str, ...]:
    values = list(existing)
    for item in required:
        if item not in values:
            values.append(item)
    return tuple(values)


def _coerce_timezone_aware_datetime(value: datetime | str, field_name: str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _coerce_clean_string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    values = (value,) if isinstance(value, str) else tuple(value)
    cleaned = tuple(str(item).strip() for item in values)
    if any(not item for item in cleaned):
        raise ValueError("string collections must contain non-empty strings")
    return _dedupe(cleaned)


def _dedupe(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)


__all__ = [
    "CLAIM_DECAY_SCHEMA_VERSION",
    "CONFIGURED_HAZARD_NOTICE",
    "DECAY_STATE_NON_LIFECYCLE_NOTICE",
    "ClaimDecayPolicy",
    "ClaimDecayRecord",
    "ConfiguredHazardCovariates",
    "ConfiguredHazardPolicy",
    "DecayEvaluation",
    "DecayState",
    "DecayStatus",
    "VersionWatchSet",
    "evaluate_claim_decay",
    "evaluate_claim_decay_with_details",
]

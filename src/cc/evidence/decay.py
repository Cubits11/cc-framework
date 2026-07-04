"""Deterministic claim-decay policies for evidence freshness checks.

The models in this module are intended for signed artifacts: they record the
policy, covariates, and version watch set, but they do not store a live
freshness verdict.  Call :func:`evaluate_claim_decay` at verification time to
derive the current state against the verifier's clock and observed versions.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CLAIM_DECAY_SCHEMA_VERSION = "cc.claim_decay.v1"


class DecayModel(BaseModel):
    """Strict base model for claim-decay artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DecayState(str, Enum):
    """Verification-time freshness state for a claim or evidence item."""

    FRESH = "fresh"
    DEGRADED = "degraded"
    EXPIRED = "expired"


class HazardCovariates(DecayModel):
    """Named covariates consumed by configured proportional-hazard policies."""

    model_update_count: int = Field(default=0, ge=0)
    guardrail_update_count: int = Field(default=0, ge=0)
    dependency_update_count: int = Field(default=0, ge=0)
    data_drift_score: float = Field(default=0.0, ge=0.0)
    threat_activity_level: float = Field(default=0.0, ge=0.0)
    custom: dict[str, float] = Field(default_factory=dict)

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
        return cleaned

    def feature_map(self) -> dict[str, float]:
        """Return the covariate vector used for beta dot x scoring."""

        features = {
            "model_update_count": float(self.model_update_count),
            "guardrail_update_count": float(self.guardrail_update_count),
            "dependency_update_count": float(self.dependency_update_count),
            "data_drift_score": float(self.data_drift_score),
            "threat_activity_level": float(self.threat_activity_level),
        }
        features.update(self.custom)
        return features


class HazardDecayPolicy(DecayModel):
    """Configured proportional-hazard scorer.

    This is deliberately not a fitted Cox estimator.  Callers supply the
    baseline hazard and coefficients, and evaluation uses
    ``lambda = baseline_hazard_per_day * exp(beta dot x)``.
    """

    baseline_hazard_per_day: float = Field(gt=0.0)
    coefficients: dict[str, float] = Field(default_factory=dict)
    degraded_after_half_lives: float = Field(default=1.0, gt=0.0)
    expired_after_half_lives: float = Field(default=2.0, gt=0.0)

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
        return cleaned

    @model_validator(mode="after")
    def _validate_threshold_order(self) -> HazardDecayPolicy:
        if self.expired_after_half_lives < self.degraded_after_half_lives:
            raise ValueError("expired_after_half_lives must be >= degraded_after_half_lives")
        return self

    def hazard_rate_per_day(self, covariates: HazardCovariates) -> float:
        """Return ``baseline_hazard_per_day * exp(beta dot x)``."""

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

    def half_life_days(self, covariates: HazardCovariates) -> float:
        """Return the hazard-implied half-life, ``ln(2) / lambda``."""

        return math.log(2.0) / self.hazard_rate_per_day(covariates)


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
        return cleaned

    def changes_from(self, current: VersionWatchSet | dict[str, Any]) -> tuple[str, ...]:
        """Return watched version paths whose observed values differ."""

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
    hazard: HazardDecayPolicy | None = None

    @model_validator(mode="after")
    def _validate_policy(self) -> ClaimDecayPolicy:
        if (
            self.degraded_after_days is None
            and self.expires_after_days is None
            and self.hazard is None
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
    """Signed claim-decay artifact without a live freshness verdict."""

    schema_version: Literal["cc.claim_decay.v1"] = CLAIM_DECAY_SCHEMA_VERSION
    claim_id: str = Field(min_length=1)
    claim_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    issued_at: datetime
    policy: ClaimDecayPolicy
    covariates: HazardCovariates = Field(default_factory=HazardCovariates)
    version_watch_set: VersionWatchSet = Field(default_factory=VersionWatchSet)
    evidence_refs: tuple[str, ...] = Field(default_factory=tuple)
    notes: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("issued_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("issued_at must be timezone-aware")
        return value

    @field_validator("evidence_refs", "notes")
    @classmethod
    def _non_empty_strings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("string collections must contain non-empty strings")
        return value


def evaluate_claim_decay(
    record: ClaimDecayRecord,
    *,
    as_of: datetime | None = None,
    current_versions: VersionWatchSet | dict[str, Any] | None = None,
) -> DecayState:
    """Evaluate a claim's freshness state at verification time."""

    if not isinstance(record, ClaimDecayRecord):
        record = ClaimDecayRecord.model_validate(record)
    verification_time = as_of or datetime.now(timezone.utc)
    if verification_time.tzinfo is None or verification_time.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    if verification_time < record.issued_at:
        raise ValueError("as_of cannot be earlier than record.issued_at")

    if current_versions is not None and record.version_watch_set.changes_from(current_versions):
        return DecayState.EXPIRED

    age_days = (verification_time - record.issued_at).total_seconds() / 86400.0
    state = DecayState.FRESH

    if (
        record.policy.expires_after_days is not None
        and age_days >= record.policy.expires_after_days
    ):
        state = DecayState.EXPIRED
    elif (
        record.policy.degraded_after_days is not None
        and age_days >= record.policy.degraded_after_days
    ):
        state = DecayState.DEGRADED

    if record.policy.hazard is not None:
        half_life = record.policy.hazard.half_life_days(record.covariates)
        if age_days >= record.policy.hazard.expired_after_half_lives * half_life:
            return DecayState.EXPIRED
        if (
            state is DecayState.FRESH
            and age_days >= record.policy.hazard.degraded_after_half_lives * half_life
        ):
            state = DecayState.DEGRADED

    return state


__all__ = [
    "CLAIM_DECAY_SCHEMA_VERSION",
    "ClaimDecayPolicy",
    "ClaimDecayRecord",
    "DecayState",
    "HazardCovariates",
    "HazardDecayPolicy",
    "VersionWatchSet",
    "evaluate_claim_decay",
]

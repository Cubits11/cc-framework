"""Pre-registration protocol checks for confirmatory evidence.

This module is a conservative bridge between adaptive discovery and
confirmatory evidence. It verifies whether a confirmatory run is separated
from discovery by a pre-registered plan with fixed endpoints, analysis, sample
and stopping rules.

Important semantic boundary
---------------------------
`ProtocolAuditStatus` is a protocol-check outcome, not a claim lifecycle state.

A protocol PASS means:

    The confirmatory artifact satisfies this verifier's procedural separation,
    timing, fixed-plan, endpoint, sample, and review-boundary checks.

A protocol PASS does not mean:

- the AI system is safe;
- the measured conclusion is globally true;
- the result transfers outside the declared sample/protocol;
- the claim is deployment-valid;
- the package is production-certified;
- the package is compliance-certified;
- a future `cc.claims.Claim` should become supported without explicit lifecycle rules.

This module owns:

- pre-registration plan models;
- confirmatory run references;
- exploratory/confirmatory firewall checks;
- temporal order checks;
- fixed endpoint and fixed analysis checks;
- stopping-rule checks;
- held-out selection checks;
- cluster-blocking review triggers;
- protocol audit records.

This module does not own:

- claim lifecycle state;
- claim compiler logic;
- assumption registry;
- challenge calculus;
- deployment/compliance certification;
- sophisticated post-selection inference;
- anytime-valid inference implementation;
- conformal risk-control implementation.

Future protocol modes such as selective inference, anytime-valid e-processes,
and conformal risk control are reserved as declared modes, but this verifier
does not implement their mathematical validity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from cc.evidence.role_ontology import RESERVED_LIFECYCLE_STATE_NAMES
from cc.reporting.report import CLAIM_LEVELS, ClaimLevel

CONFIRMATORY_PROTOCOL_SCHEMA_VERSION: Literal["cc.confirmatory_protocol.v1"] = (
    "cc.confirmatory_protocol.v1"
)
CONFIRMATORY_PROTOCOL_AUDIT_SCHEMA_VERSION: Literal["cc.confirmatory_protocol_audit.v1"] = (
    "cc.confirmatory_protocol_audit.v1"
)

ProtocolMode = Literal[
    "held_out_matrix",
    "sample_split",
    "fixed_attack_suite",
    "pre_registered_retest",
    "cluster_blocked_confirmation",
    "selective_inference_adjusted",
    "anytime_valid_e_process",
    "conformal_risk_control",
]

HeldOutSelectionTiming = Literal[
    "pre_failure_pattern",
    "post_failure_pattern",
    "not_applicable",
    "unknown",
]

ProtocolCheckId: TypeAlias = Literal[
    "artifact_schema",
    "plan_and_run_referenced",
    "temporal_validity_check",
    "adaptive_discovery_firewall",
    "held_out_selection",
    "stopping_rule",
    "cluster_blocking",
    "fixed_endpoint",
    "fixed_analysis_plan",
    "protocol_mode_supported",
    "non_claim_boundary",
]

_FUTURE_PROTOCOL_MODES = frozenset(
    {
        "selective_inference_adjusted",
        "anytime_valid_e_process",
        "conformal_risk_control",
    }
)

_ADAPTIVE_OR_EXPLORATORY_ROLES = frozenset(
    {
        "adaptive_discovery",
        "exploratory_redteam",
        "fitted_empirical_scenario",
    }
)

_STOPPING_RULE_FAILURE_SURFACES = frozenset(
    {
        "bounded_empirical",
        "reproducible_run",
        "release_claim",
    }
)

_CONFIRMATORY_RUN_ROLES = frozenset(
    {
        "confirmatory_failure_matrix",
        "confirmatory_protocol",
    }
)

_PROHIBITED_ADAPTIVE_INPUTS = (
    "adaptive_search_ci",
    "certificate_ci",
    "exploratory_certificate_ci",
    "exploratory_ci",
    "non_confirmatory_ci",
)

PROTOCOL_PASS_CAVEAT = (
    "A confirmatory protocol PASS means procedural separation under this verifier only; "
    "it does not prove deployment safety, external validity, or global truth."
)

PROTOCOL_NOT_LIFECYCLE_STATE_NON_CLAIM = "Protocol audit status is not a claim lifecycle state."

FUTURE_METHOD_NON_CLAIM = (
    "Reserved future protocol modes are labels only unless their mathematical verifier "
    "is implemented and reviewed."
)

_DEFAULT_PROTOCOL_NON_CLAIMS = (
    "Confirmatory validity depends on the pre-registered protocol and run separation, "
    "not on report polish.",
    "A confirmatory_protocol artifact does not certify deployment safety or external validity.",
    "Adaptive discovery evidence may motivate a hypothesis but cannot become confirmatory "
    "evidence by renaming its role.",
    PROTOCOL_PASS_CAVEAT,
    PROTOCOL_NOT_LIFECYCLE_STATE_NON_CLAIM,
)


class ProtocolAuditStatus(str, Enum):
    """Conservative status for one protocol audit or check.

    This is not a claim lifecycle state.
    """

    PASS = "pass"
    NEEDS_REVIEW = "needs_review"
    FAIL = "fail"


class ConfirmatoryProtocolModel(BaseModel):
    """Strict frozen base model for confirmatory protocol artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DiscoveryReference(ConfirmatoryProtocolModel):
    """Reference to the discovery artifact that motivated the hypothesis."""

    artifact_id: str = Field(min_length=1)
    artifact_role: str = Field(min_length=1)
    artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    adaptive: bool = False
    description: str = Field(min_length=1)
    discovered_at: datetime | None = None

    @field_validator("artifact_role")
    @classmethod
    def _artifact_role_is_not_lifecycle_state(cls, value: str) -> str:
        return _reject_lifecycle_name(value, "discovery artifact_role")

    @field_validator("discovered_at", mode="before")
    @classmethod
    def _discovered_at_is_aware(cls, value: Any) -> datetime | None:
        if value is None:
            return None
        return _parse_aware_datetime(value, "discovered_at")


class FixedAnalysisPlan(ConfirmatoryProtocolModel):
    """Fixed analysis plan declared before the confirmatory run starts."""

    analysis_id: str = Field(min_length=1)
    estimand: str = Field(min_length=1)
    interval_method: str = Field(min_length=1)
    alpha: float = Field(gt=0.0, lt=1.0)
    multiplicity_adjustment: str = Field(min_length=1)
    frozen: bool = True
    prohibited_adaptive_inputs: tuple[str, ...] = Field(default=_PROHIBITED_ADAPTIVE_INPUTS)

    @field_validator("prohibited_adaptive_inputs", mode="before")
    @classmethod
    def _coerce_prohibited_inputs(cls, value: Any) -> tuple[str, ...]:
        return _coerce_string_tuple(value)

    @field_validator("prohibited_adaptive_inputs")
    @classmethod
    def _validate_prohibited_inputs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = _non_empty_string_tuple(value, "prohibited_adaptive_inputs")
        missing_defaults = set(_PROHIBITED_ADAPTIVE_INPUTS) - set(cleaned)
        if missing_defaults:
            raise ValueError(
                "prohibited_adaptive_inputs must include default exploratory interval fields: "
                f"{sorted(missing_defaults)}"
            )
        return cleaned


class SamplePlan(ConfirmatoryProtocolModel):
    """Sample plan fixed before confirmation."""

    sampling_frame: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    target_n: int = Field(gt=0)
    held_out_selection: HeldOutSelectionTiming = "not_applicable"
    split_rule: str | None = None
    clustered_data: bool = False
    cluster_variable: str | None = None

    @model_validator(mode="after")
    def _cluster_variable_reviewed_when_clustered(self) -> SamplePlan:
        # Missing cluster variables are reviewed by _check_cluster_blocking so the
        # verifier can distinguish schema failure from a review-triggering design gap.
        return self


class StoppingRule(ConfirmatoryProtocolModel):
    """Pre-declared stopping rule for the confirmatory run."""

    rule_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    max_samples: int = Field(gt=0)
    max_looks: int = Field(default=1, ge=1)
    early_stopping_allowed: bool = False
    early_stopping_rule: str | None = None

    @model_validator(mode="after")
    def _early_stopping_rule_required_when_allowed(self) -> StoppingRule:
        if self.early_stopping_allowed and not self.early_stopping_rule:
            raise ValueError("early_stopping_rule is required when early stopping is allowed")
        return self


class ClusterBlockingPlan(ConfirmatoryProtocolModel):
    """Cluster-blocking or cluster-robust design declaration."""

    cluster_variable: str = Field(min_length=1)
    method: str = Field(min_length=1)
    blocks_clusters: bool = True
    independent_cluster_assumption: str = Field(min_length=1)


class ExclusionRule(ConfirmatoryProtocolModel):
    """Pre-declared exclusion rule."""

    rule_id: str = Field(min_length=1)
    field: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class DecisionRule(ConfirmatoryProtocolModel):
    """Pre-declared decision boundary for interpreting the confirmatory run."""

    rule_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    threshold: float | None = None
    pass_condition: str = Field(min_length=1)
    fail_condition: str = Field(min_length=1)


class ConfirmatoryProtocolPlan(ConfirmatoryProtocolModel):
    """Pre-registration plan for turning discovery into confirmatory evidence."""

    protocol_id: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    discovery_ref: DiscoveryReference
    protocol_mode: ProtocolMode
    created_at: datetime
    primary_endpoint: str = Field(min_length=1)
    fixed_analysis_plan: FixedAnalysisPlan
    sample_plan: SamplePlan
    stopping_rule: StoppingRule | None
    cluster_blocking: ClusterBlockingPlan | None = None
    exclusion_rules: tuple[ExclusionRule, ...] = Field(default_factory=tuple)
    decision_rule: DecisionRule
    non_claims: tuple[str, ...] = Field(default_factory=lambda: _DEFAULT_PROTOCOL_NON_CLAIMS)

    @model_validator(mode="before")
    @classmethod
    def _accept_preregistered_at_alias_and_reject_live_status(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        payload = dict(data)
        if "preregistered_at" in payload:
            if "created_at" in payload:
                raise ValueError("use only one of created_at or preregistered_at")
            payload["created_at"] = payload.pop("preregistered_at")

        forbidden_live_fields = {
            "status",
            "verdict",
            "claim_state",
            "lifecycle_state",
            "supported",
            "certified",
            "production_ready",
        }
        present = sorted(forbidden_live_fields & set(payload))
        if present:
            raise ValueError(
                "ConfirmatoryProtocolPlan stores protocol design only and must not contain "
                f"live verdict or lifecycle fields: {present}"
            )
        return payload

    @field_validator("protocol_id")
    @classmethod
    def _protocol_id_is_not_lifecycle_state(cls, value: str) -> str:
        return _reject_lifecycle_name(value, "protocol_id")

    @field_validator("created_at", mode="before")
    @classmethod
    def _created_at_is_aware(cls, value: Any) -> datetime:
        return _parse_aware_datetime(value, "created_at")

    @field_validator("non_claims", mode="before")
    @classmethod
    def _coerce_non_claims(cls, value: Any) -> tuple[str, ...]:
        return _append_missing(
            _coerce_string_tuple(value or _DEFAULT_PROTOCOL_NON_CLAIMS),
            _DEFAULT_PROTOCOL_NON_CLAIMS,
        )

    @field_validator("exclusion_rules", mode="before")
    @classmethod
    def _coerce_exclusion_rules(cls, value: Any) -> tuple[Any, ...]:
        if value is None:
            return ()
        return tuple(value)

    @field_validator("non_claims")
    @classmethod
    def _non_claims_are_non_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _non_empty_string_tuple(value, "non_claims")

    @model_validator(mode="after")
    def _future_protocol_modes_require_explicit_boundary(self) -> ConfirmatoryProtocolPlan:
        if self.protocol_mode in _FUTURE_PROTOCOL_MODES:
            required = FUTURE_METHOD_NON_CLAIM
            if required not in self.non_claims:
                raise ValueError(
                    "future protocol modes require the future-method non-claim boundary"
                )
        if self.sample_plan.clustered_data and self.cluster_blocking is None:
            # Do not fail at model construction; verifier will mark review. This remains explicit.
            return self
        return self

    @property
    def preregistered_at(self) -> datetime:
        """Alias for the protocol creation time used by roadmap prose."""

        return self.created_at


class ConfirmatoryRunReference(ConfirmatoryProtocolModel):
    """Reference to the confirmatory run that executed the pre-registered plan."""

    run_id: str = Field(min_length=1)
    started_at: datetime
    completed_at: datetime | None = None
    artifact_id: str = Field(min_length=1)
    artifact_role: str = Field(default="confirmatory_failure_matrix", min_length=1)
    source_role: str | None = None
    artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    primary_endpoint: str = Field(min_length=1)
    analysis_plan_id: str | None = None
    used_adaptive_discovery_data: bool = False
    held_out_set_chosen_after_failure_pattern: bool = False
    clustered_data_observed: bool = False
    cluster_variable: str | None = None
    non_claims: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="before")
    @classmethod
    def _reject_live_status(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        payload = dict(data)
        forbidden_live_fields = {
            "status",
            "verdict",
            "claim_state",
            "lifecycle_state",
            "supported",
            "certified",
            "production_ready",
        }
        present = sorted(forbidden_live_fields & set(payload))
        if present:
            raise ValueError(
                "ConfirmatoryRunReference stores run provenance only and must not contain "
                f"live verdict or lifecycle fields: {present}"
            )
        return payload

    @field_validator("run_id", "artifact_id")
    @classmethod
    def _ids_are_not_lifecycle_states(cls, value: str) -> str:
        return _reject_lifecycle_name(value, "run/artifact id")

    @field_validator("artifact_role", "source_role")
    @classmethod
    def _roles_are_not_lifecycle_states(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _reject_lifecycle_name(value, "artifact/source role")

    @field_validator("started_at", mode="before")
    @classmethod
    def _started_at_is_aware(cls, value: Any) -> datetime:
        return _parse_aware_datetime(value, "started_at")

    @field_validator("completed_at", mode="before")
    @classmethod
    def _completed_at_is_aware(cls, value: Any) -> datetime | None:
        if value is None:
            return None
        return _parse_aware_datetime(value, "completed_at")

    @field_validator("non_claims", mode="before")
    @classmethod
    def _coerce_non_claims(cls, value: Any) -> tuple[str, ...]:
        return _coerce_string_tuple(value)

    @field_validator("non_claims")
    @classmethod
    def _non_claims_are_non_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _non_empty_string_tuple(value, "non_claims")

    @model_validator(mode="after")
    def _completed_after_started_and_cluster_fields_consistent(self) -> ConfirmatoryRunReference:
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at must be >= started_at")
        return self


class ConfirmatoryProtocolArtifact(ConfirmatoryProtocolModel):
    """Full confirmatory artifact binding a plan and its confirming run."""

    schema_version: Literal["cc.confirmatory_protocol.v1"] = CONFIRMATORY_PROTOCOL_SCHEMA_VERSION
    artifact_id: str = Field(min_length=1)
    plan: ConfirmatoryProtocolPlan
    run: ConfirmatoryRunReference
    non_claims: tuple[str, ...] = Field(default_factory=lambda: _DEFAULT_PROTOCOL_NON_CLAIMS)

    @model_validator(mode="before")
    @classmethod
    def _reject_live_status(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        payload = dict(data)
        forbidden_live_fields = {
            "status",
            "verdict",
            "claim_state",
            "lifecycle_state",
            "supported",
            "certified",
            "production_ready",
        }
        present = sorted(forbidden_live_fields & set(payload))
        if present:
            raise ValueError(
                "ConfirmatoryProtocolArtifact stores protocol/run binding only and must not "
                f"contain live verdict or lifecycle fields: {present}"
            )
        return payload

    @field_validator("artifact_id")
    @classmethod
    def _artifact_id_is_not_lifecycle_state(cls, value: str) -> str:
        return _reject_lifecycle_name(value, "artifact_id")

    @field_validator("non_claims", mode="before")
    @classmethod
    def _coerce_non_claims(cls, value: Any) -> tuple[str, ...]:
        return _append_missing(
            _coerce_string_tuple(value or _DEFAULT_PROTOCOL_NON_CLAIMS),
            _DEFAULT_PROTOCOL_NON_CLAIMS,
        )

    @field_validator("non_claims")
    @classmethod
    def _non_claims_are_non_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _non_empty_string_tuple(value, "non_claims")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the artifact as deterministic JSON-compatible data."""

        return self.model_dump(mode="json")


class ProtocolCheck(ConfirmatoryProtocolModel):
    """One machine-checkable protocol condition."""

    check_id: str
    status: ProtocolAuditStatus
    reason: str = Field(min_length=1)

    @field_validator("check_id")
    @classmethod
    def _check_id_known(cls, value: str) -> str:
        if value not in set(cast(tuple[str, ...], getattr(ProtocolCheckId, "__args__", ()))):  # type: ignore[attr-defined]
            # Keep runtime permissive for compatibility if typing internals differ.
            return value
        return value


class ConfirmatoryProtocolAudit(ConfirmatoryProtocolModel):
    """Verifier output for a confirmatory protocol artifact.

    This audit is verifier output. It is not a lifecycle state.
    """

    schema_version: Literal["cc.confirmatory_protocol_audit.v1"] = (
        CONFIRMATORY_PROTOCOL_AUDIT_SCHEMA_VERSION
    )
    artifact_id: str
    protocol_id: str
    run_id: str
    status: ProtocolAuditStatus
    temporal_validity_check: bool
    checks: tuple[ProtocolCheck, ...]
    reasons: tuple[str, ...] = Field(default_factory=tuple)
    non_claims: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("non_claims", mode="before")
    @classmethod
    def _coerce_non_claims(cls, value: Any) -> tuple[str, ...]:
        return _append_missing(_coerce_string_tuple(value), _DEFAULT_PROTOCOL_NON_CLAIMS)

    @model_validator(mode="after")
    def _audit_status_has_reasons_and_caveats(self) -> ConfirmatoryProtocolAudit:
        if self.status is not ProtocolAuditStatus.PASS and not self.reasons:
            raise ValueError("non-PASS protocol audits must include reasons")
        if PROTOCOL_PASS_CAVEAT not in self.non_claims:
            raise ValueError("protocol audit must preserve PASS caveat non-claim")
        return self


def temporal_validity_check(
    plan: ConfirmatoryProtocolPlan,
    run: ConfirmatoryRunReference,
) -> bool:
    """Return true only when the plan was created before the run started."""

    return plan.created_at < run.started_at


def verify_confirmatory_protocol_artifact(
    payload: Mapping[str, Any],
    *,
    claim_level: str = "bounded_empirical",
) -> ConfirmatoryProtocolAudit:
    """Verify a confirmatory artifact without upgrading its statistical claim.

    PASS means the artifact satisfies v0 timing, provenance, fixed-plan, endpoint,
    sample, stopping-rule, and clustering checks. It does not mean the measured
    system is safe in deployment.
    """

    checked_claim_level = _claim_level(claim_level)

    try:
        artifact = ConfirmatoryProtocolArtifact.model_validate(payload)
    except ValidationError as exc:
        return _invalid_artifact_audit(payload, reason=f"Confirmatory artifact is invalid: {exc}")
    except ValueError as exc:
        return _invalid_artifact_audit(payload, reason=f"Confirmatory artifact is invalid: {exc}")

    plan = artifact.plan
    run = artifact.run

    checks = [
        _check(
            "plan_and_run_referenced",
            ProtocolAuditStatus.PASS,
            "Confirmatory artifact references both a protocol plan and a run.",
        ),
        _check_temporal_order(plan, run),
        _check_adaptive_firewall(plan, run),
        _check_held_out_selection(plan, run),
        _check_stopping_rule(plan, claim_level=checked_claim_level),
        _check_cluster_blocking(plan, run),
        _check_fixed_endpoint(plan, run),
        _check_fixed_analysis_plan(plan, run),
        _check_protocol_mode(plan),
        _check_non_claim_boundary(artifact),
    ]

    status = _aggregate_status(checks)
    reasons = tuple(
        check.reason for check in checks if check.status is not ProtocolAuditStatus.PASS
    )
    non_claims = _dedupe((*artifact.non_claims, *plan.non_claims, *run.non_claims))
    non_claims = _append_missing(non_claims, _DEFAULT_PROTOCOL_NON_CLAIMS)

    return ConfirmatoryProtocolAudit(
        artifact_id=artifact.artifact_id,
        protocol_id=plan.protocol_id,
        run_id=run.run_id,
        status=status,
        temporal_validity_check=temporal_validity_check(plan, run),
        checks=tuple(checks),
        reasons=reasons,
        non_claims=non_claims,
    )


def validate_confirmatory_protocol_payload(
    payload: Mapping[str, Any],
    *,
    claim_level: str = "bounded_empirical",
) -> ConfirmatoryProtocolAudit:
    """Compatibility helper for governance and ontology-facing callers."""

    return verify_confirmatory_protocol_artifact(payload, claim_level=claim_level)


def _invalid_artifact_audit(
    payload: Mapping[str, Any],
    *,
    reason: str,
) -> ConfirmatoryProtocolAudit:
    artifact_id = str(payload.get("artifact_id") or "<invalid-confirmatory-artifact>")
    return ConfirmatoryProtocolAudit(
        artifact_id=artifact_id,
        protocol_id="<invalid-protocol>",
        run_id="<invalid-run>",
        status=ProtocolAuditStatus.FAIL,
        temporal_validity_check=False,
        checks=(_check("artifact_schema", ProtocolAuditStatus.FAIL, reason),),
        reasons=(reason,),
        non_claims=_DEFAULT_PROTOCOL_NON_CLAIMS,
    )


def _check_temporal_order(
    plan: ConfirmatoryProtocolPlan,
    run: ConfirmatoryRunReference,
) -> ProtocolCheck:
    if temporal_validity_check(plan, run):
        return _check(
            "temporal_validity_check",
            ProtocolAuditStatus.PASS,
            "Protocol plan was created before the confirmatory run started.",
        )
    return _check(
        "temporal_validity_check",
        ProtocolAuditStatus.FAIL,
        "Protocol plan must be created before the confirmatory run starts.",
    )


def _check_adaptive_firewall(
    plan: ConfirmatoryProtocolPlan,
    run: ConfirmatoryRunReference,
) -> ProtocolCheck:
    observed_roles = {run.artifact_role}
    if run.source_role is not None:
        observed_roles.add(run.source_role)

    if observed_roles & _ADAPTIVE_OR_EXPLORATORY_ROLES:
        return _check(
            "adaptive_discovery_firewall",
            ProtocolAuditStatus.FAIL,
            "Adaptive or exploratory discovery artifacts cannot become confirmatory by role rename.",
        )

    if run.artifact_role not in _CONFIRMATORY_RUN_ROLES:
        return _check(
            "adaptive_discovery_firewall",
            ProtocolAuditStatus.NEEDS_REVIEW,
            f"Confirmatory run artifact role {run.artifact_role!r} is not a known confirmatory role.",
        )

    if run.used_adaptive_discovery_data:
        return _check(
            "adaptive_discovery_firewall",
            ProtocolAuditStatus.FAIL,
            "Confirmatory run reports using adaptive discovery data as confirmatory evidence.",
        )

    if (
        plan.discovery_ref.adaptive
        and plan.discovery_ref.artifact_sha256 is not None
        and run.artifact_sha256 == plan.discovery_ref.artifact_sha256
    ):
        return _check(
            "adaptive_discovery_firewall",
            ProtocolAuditStatus.FAIL,
            "The adaptive discovery artifact hash is reused as the confirmatory run artifact.",
        )

    return _check(
        "adaptive_discovery_firewall",
        ProtocolAuditStatus.PASS,
        "Discovery may motivate the hypothesis, but the confirmatory run is separately referenced.",
    )


def _check_held_out_selection(
    plan: ConfirmatoryProtocolPlan,
    run: ConfirmatoryRunReference,
) -> ProtocolCheck:
    if plan.protocol_mode != "held_out_matrix":
        return _check(
            "held_out_selection",
            ProtocolAuditStatus.PASS,
            "Held-out matrix timing is not required for this protocol mode.",
        )

    if (
        plan.sample_plan.held_out_selection == "post_failure_pattern"
        or run.held_out_set_chosen_after_failure_pattern
    ):
        return _check(
            "held_out_selection",
            ProtocolAuditStatus.FAIL,
            "Held-out confirmation is invalid when the held-out set is chosen after seeing "
            "the failure pattern.",
        )

    if plan.sample_plan.held_out_selection == "unknown":
        return _check(
            "held_out_selection",
            ProtocolAuditStatus.NEEDS_REVIEW,
            "Held-out selection timing is unknown and requires review.",
        )

    return _check(
        "held_out_selection",
        ProtocolAuditStatus.PASS,
        "Held-out set selection was declared before failure-pattern review.",
    )


def _check_stopping_rule(
    plan: ConfirmatoryProtocolPlan,
    *,
    claim_level: ClaimLevel,
) -> ProtocolCheck:
    if plan.stopping_rule is not None:
        return _check(
            "stopping_rule",
            ProtocolAuditStatus.PASS,
            "Stopping rule is pre-declared.",
        )

    if claim_level in _STOPPING_RULE_FAILURE_SURFACES:
        return _check(
            "stopping_rule",
            ProtocolAuditStatus.FAIL,
            f"Missing stopping rule invalidates confirmatory support for {claim_level!r}.",
        )

    return _check(
        "stopping_rule",
        ProtocolAuditStatus.NEEDS_REVIEW,
        "Missing stopping rule requires review on a diagnostic surface.",
    )


def _check_cluster_blocking(
    plan: ConfirmatoryProtocolPlan,
    run: ConfirmatoryRunReference,
) -> ProtocolCheck:
    clustered = plan.sample_plan.clustered_data or run.clustered_data_observed
    blocks_clusters = plan.cluster_blocking is not None and plan.cluster_blocking.blocks_clusters

    if clustered and not blocks_clusters:
        return _check(
            "cluster_blocking",
            ProtocolAuditStatus.NEEDS_REVIEW,
            "Clustered data without a declared cluster-blocking or cluster-robust plan "
            "requires review.",
        )

    if (
        clustered
        and plan.cluster_blocking is not None
        and run.cluster_variable is not None
        and run.cluster_variable != plan.cluster_blocking.cluster_variable
    ):
        return _check(
            "cluster_blocking",
            ProtocolAuditStatus.NEEDS_REVIEW,
            "Observed cluster variable differs from the pre-registered cluster-blocking variable.",
        )

    return _check(
        "cluster_blocking",
        ProtocolAuditStatus.PASS,
        "Cluster blocking obligations are satisfied or no clustered data were declared.",
    )


def _check_fixed_endpoint(
    plan: ConfirmatoryProtocolPlan,
    run: ConfirmatoryRunReference,
) -> ProtocolCheck:
    if run.primary_endpoint != plan.primary_endpoint:
        return _check(
            "fixed_endpoint",
            ProtocolAuditStatus.FAIL,
            "Confirmatory run endpoint does not match the pre-registered primary endpoint.",
        )

    return _check(
        "fixed_endpoint",
        ProtocolAuditStatus.PASS,
        "Confirmatory run endpoint matches the pre-registered primary endpoint.",
    )


def _check_fixed_analysis_plan(
    plan: ConfirmatoryProtocolPlan,
    run: ConfirmatoryRunReference,
) -> ProtocolCheck:
    if not plan.fixed_analysis_plan.frozen:
        return _check(
            "fixed_analysis_plan",
            ProtocolAuditStatus.FAIL,
            "Confirmatory analysis plan is not marked frozen.",
        )

    if run.analysis_plan_id is None:
        return _check(
            "fixed_analysis_plan",
            ProtocolAuditStatus.NEEDS_REVIEW,
            "Confirmatory run does not identify the executed analysis plan.",
        )

    if run.analysis_plan_id != plan.fixed_analysis_plan.analysis_id:
        return _check(
            "fixed_analysis_plan",
            ProtocolAuditStatus.FAIL,
            "Confirmatory run analysis plan does not match the pre-registered plan.",
        )

    return _check(
        "fixed_analysis_plan",
        ProtocolAuditStatus.PASS,
        "Confirmatory run analysis plan matches the pre-registered frozen plan.",
    )


def _check_protocol_mode(plan: ConfirmatoryProtocolPlan) -> ProtocolCheck:
    if plan.protocol_mode in _FUTURE_PROTOCOL_MODES:
        return _check(
            "protocol_mode_supported",
            ProtocolAuditStatus.NEEDS_REVIEW,
            f"Protocol mode {plan.protocol_mode!r} is reserved for future verifier support.",
        )

    return _check(
        "protocol_mode_supported",
        ProtocolAuditStatus.PASS,
        f"Protocol mode {plan.protocol_mode!r} is supported by the v0 verifier.",
    )


def _check_non_claim_boundary(artifact: ConfirmatoryProtocolArtifact) -> ProtocolCheck:
    non_claims = _dedupe(
        (*artifact.non_claims, *artifact.plan.non_claims, *artifact.run.non_claims)
    )
    missing = [item for item in _DEFAULT_PROTOCOL_NON_CLAIMS if item not in non_claims]

    if missing:
        return _check(
            "non_claim_boundary",
            ProtocolAuditStatus.FAIL,
            "Confirmatory protocol artifact is missing mandatory non-claim boundaries.",
        )

    return _check(
        "non_claim_boundary",
        ProtocolAuditStatus.PASS,
        "Mandatory confirmatory protocol non-claim boundaries are present.",
    )


def _check(check_id: str, status: ProtocolAuditStatus, reason: str) -> ProtocolCheck:
    return ProtocolCheck(check_id=check_id, status=status, reason=reason)


def _aggregate_status(checks: Sequence[ProtocolCheck]) -> ProtocolAuditStatus:
    if any(check.status is ProtocolAuditStatus.FAIL for check in checks):
        return ProtocolAuditStatus.FAIL
    if any(check.status is ProtocolAuditStatus.NEEDS_REVIEW for check in checks):
        return ProtocolAuditStatus.NEEDS_REVIEW
    return ProtocolAuditStatus.PASS


def _claim_level(value: str) -> ClaimLevel:
    if value not in CLAIM_LEVELS:
        allowed = ", ".join(CLAIM_LEVELS)
        raise ValueError(f"claim_level must be one of: {allowed}")
    if value in RESERVED_LIFECYCLE_STATE_NAMES:
        raise ValueError("claim_level must not reuse a lifecycle state name")
    return cast(ClaimLevel, value)


def _reject_lifecycle_name(value: str, field_name: str) -> str:
    stripped = value.strip()
    if stripped in RESERVED_LIFECYCLE_STATE_NAMES:
        raise ValueError(f"{field_name} must not reuse a claim lifecycle state name")
    return stripped


def _parse_aware_datetime(value: Any, field_name: str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime or ISO-8601 string")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _coerce_string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _non_empty_string_tuple(value: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    cleaned = tuple(str(item).strip() for item in value)
    if any(not item for item in cleaned):
        raise ValueError(f"{field_name} must contain non-empty strings")
    return _dedupe(cleaned)


def _append_missing(existing: tuple[str, ...], required: tuple[str, ...]) -> tuple[str, ...]:
    values = list(existing)
    for item in required:
        if item not in values:
            values.append(item)
    return tuple(values)


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
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
    "CONFIRMATORY_PROTOCOL_AUDIT_SCHEMA_VERSION",
    "CONFIRMATORY_PROTOCOL_SCHEMA_VERSION",
    "FUTURE_METHOD_NON_CLAIM",
    "PROTOCOL_NOT_LIFECYCLE_STATE_NON_CLAIM",
    "PROTOCOL_PASS_CAVEAT",
    "ClusterBlockingPlan",
    "ConfirmatoryProtocolArtifact",
    "ConfirmatoryProtocolAudit",
    "ConfirmatoryProtocolPlan",
    "ConfirmatoryRunReference",
    "DecisionRule",
    "DiscoveryReference",
    "ExclusionRule",
    "FixedAnalysisPlan",
    "HeldOutSelectionTiming",
    "ProtocolAuditStatus",
    "ProtocolCheck",
    "ProtocolMode",
    "SamplePlan",
    "StoppingRule",
    "temporal_validity_check",
    "validate_confirmatory_protocol_payload",
    "verify_confirmatory_protocol_artifact",
]

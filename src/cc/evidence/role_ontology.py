"""Conservative evidence-role ontology for evidence-bound claims.

An evidence role is a typed permission set, not a decorative label.

This module answers one narrow question:

    Given an evidence artifact role, what may that role support, what must it
    never support, which non-claims must remain visible, and which semantic
    payload fields are required or forbidden?

Important semantic boundary
---------------------------
This module owns evidence-role support semantics.

It does not own claim lifecycle state.

Lifecycle states such as draft/supported/bounded/challenged/weakened/expired/
revoked/superseded/non_claim belong in future `cc.claims.ClaimState`, not here.

Report maturity/support levels such as diagnostic/bounded_empirical/
reproducible_run/release_claim are imported from `cc.reporting.report` so this
module does not silently grow a second claim-level taxonomy.

The intended ontology split is:

- `cc.reporting.report`: report-facing claim maturity/support labels.
- `cc.evidence.role_ontology`: evidence-role support permissions.
- `cc.evidence.claim_governance`: verification of evidence/report consistency.
- future `cc.claims`: lifecycle state, transitions, assumptions, challenges,
  expiry, and first-class claim objects.

Do not add new claim lifecycle states to this file.
Do not use an evidence role as proof of safety.
Do not allow integrity/provenance/review evidence to upgrade semantic truth.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cc.reporting.report import CLAIM_LEVELS, ClaimLevel

ROLE_ONTOLOGY_SCHEMA_VERSION: Literal["cc.evidence_role_ontology.v1"] = (
    "cc.evidence_role_ontology.v1"
)

# ---------------------------------------------------------------------------
# Ontology boundary constants
# ---------------------------------------------------------------------------

# These names are reserved for future `cc.claims.ClaimState`.
# They must never become report claim levels or evidence roles.
RESERVED_LIFECYCLE_STATE_NAMES = frozenset(
    {
        "draft",
        "supported",
        "bounded",
        "challenged",
        "weakened",
        "expired",
        "revoked",
        "superseded",
        "non_claim",
    }
)

# Report-facing maturity/support levels imported from cc.reporting.report.
# Keep this alias local so the role ontology can describe its own intent without
# redefining the taxonomy.
ClaimMaturityLevel: TypeAlias = ClaimLevel

_ALL_CLAIM_LEVELS: tuple[ClaimLevel, ...] = CLAIM_LEVELS
_CONFIRMATORY_CLAIM_LEVELS: tuple[ClaimLevel, ...] = (
    "bounded_empirical",
    "reproducible_run",
    "release_claim",
)

if set(_ALL_CLAIM_LEVELS) & RESERVED_LIFECYCLE_STATE_NAMES:  # pragma: no cover
    raise RuntimeError("report claim levels must remain disjoint from lifecycle state names")

# ---------------------------------------------------------------------------
# Type vocabularies
# ---------------------------------------------------------------------------

SupportRelation = Literal[
    "supports",
    "bounds",
    "qualifies",
    "invalidates",
    "requires_review",
    "integrity_binds",
    "exploratory_suggests",
    "confirmatory_tests",
]

SupportStrength = Literal[
    "weak",
    "diagnostic",
    "confirmatory",
    "integrity_only",
]

SemanticClass = Literal[
    "measurement",
    "calibration",
    "integrity",
    "temporal",
    "scenario",
    "protocol",
    "redteam",
    "review",
    "generic",
    "unknown",
]

StalenessBehavior = Literal[
    "none",
    "verification_time",
    "artifact_timestamp",
    "review_expiry",
    "unknown",
]

ExploratoryStatus = Literal[
    "not_exploratory",
    "exploratory",
    "mixed",
    "unknown",
]

ConfirmatoryStatus = Literal[
    "not_confirmatory",
    "confirmatory",
    "requires_confirmation",
    "unknown",
]

FieldMatchMode = Literal[
    "path",
    "leaf",
    "path_or_leaf",
]

InvalidationSeverity = Literal[
    "review",
    "invalidates",
    "expires",
]


class RoleOntologyModel(BaseModel):
    """Strict frozen base model for role ontology records."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SupportPermission(RoleOntologyModel):
    """A narrow support edge an evidence role is allowed to provide.

    This is not a claim lifecycle state.

    A support permission describes a permitted evidence-to-claim-fragment edge:

    - relation: how support is allowed to behave;
    - strength: conservative support strength;
    - target_claim_fragments: machine-readable target fragments;
    - support_scope: human-readable scope boundary.

    Example:
        receipt_integrity may integrity-bind report bytes.
        It may not support deployment safety or statistical validity.
    """

    relation: SupportRelation
    strength: SupportStrength
    target_claim_fragments: tuple[str, ...] = Field(default_factory=tuple)
    support_scope: str = Field(min_length=1)

    @field_validator("target_claim_fragments")
    @classmethod
    def _target_fragments_are_non_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _clean_str_tuple(value, "target_claim_fragments")


class UnsupportedClaim(RoleOntologyModel):
    """A claim family this evidence role must not be used to support.

    This is a negative support rule. It is not yet the same thing as a future
    `cc.claims.NonClaim`, but it is one source from which mandatory non-claims
    can be generated.
    """

    claim_type: str = Field(min_length=1)
    target_markers: tuple[str, ...] = Field(min_length=1)
    reason: str = Field(min_length=1)

    @field_validator("target_markers")
    @classmethod
    def _target_markers_are_non_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(item.lower() for item in _clean_str_tuple(value, "target_markers"))


class MandatoryNonClaim(RoleOntologyModel):
    """Machine-readable non-claim boundary required by an evidence role.

    This is the evidence layer's seed for first-class non-claims. Future
    `cc.claims.NonClaim` may consume or project this object, but this role
    ontology remains responsible for saying which non-claims are mandatory for
    each evidence role.
    """

    non_claim_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    phrase_groups: tuple[tuple[str, ...], ...] = Field(default_factory=tuple)

    @field_validator("phrase_groups")
    @classmethod
    def _phrase_groups_are_non_empty(
        cls,
        value: tuple[tuple[str, ...], ...],
    ) -> tuple[tuple[str, ...], ...]:
        cleaned: list[tuple[str, ...]] = []
        for group in value:
            cleaned.append(_clean_str_tuple(group, "phrase_groups"))
        return tuple(cleaned)


class PayloadFieldRule(RoleOntologyModel):
    """Required or forbidden semantic-payload field rule."""

    field: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    match: FieldMatchMode = "path_or_leaf"


class ReviewRule(RoleOntologyModel):
    """A conservative review rule implied by an evidence role."""

    rule_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    requires_human_review: bool = True


class InvalidationTrigger(RoleOntologyModel):
    """A machine-readable event that weakens, expires, or invalidates support.

    This is not a full challenge calculus. Future `cc.claims.challenge` may
    translate these triggers into claim lifecycle transitions.
    """

    trigger_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    severity: InvalidationSeverity


class EvidenceRoleDefinition(RoleOntologyModel):
    """Typed permission set for one evidence role."""

    role: str = Field(min_length=1)
    semantic_class: SemanticClass
    supports: tuple[SupportPermission, ...] = Field(default_factory=tuple)
    does_not_support: tuple[UnsupportedClaim, ...] = Field(default_factory=tuple)
    mandatory_non_claims: tuple[MandatoryNonClaim, ...] = Field(default_factory=tuple)
    allowed_claim_levels: tuple[ClaimLevel, ...] = Field(default_factory=tuple)
    staleness_behavior: StalenessBehavior
    exploratory_status: ExploratoryStatus
    confirmatory_status: ConfirmatoryStatus
    required_fields: tuple[PayloadFieldRule, ...] = Field(default_factory=tuple)
    forbidden_fields: tuple[PayloadFieldRule, ...] = Field(default_factory=tuple)
    review_rules: tuple[ReviewRule, ...] = Field(default_factory=tuple)
    invalidation_triggers: tuple[InvalidationTrigger, ...] = Field(default_factory=tuple)

    @field_validator("role")
    @classmethod
    def _role_is_normalized(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("role must be non-empty")
        if stripped in RESERVED_LIFECYCLE_STATE_NAMES:
            raise ValueError("evidence role must not reuse a lifecycle state name")
        return stripped

    @model_validator(mode="after")
    def _claim_levels_are_unique_and_not_lifecycle_states(self) -> EvidenceRoleDefinition:
        if len(set(self.allowed_claim_levels)) != len(self.allowed_claim_levels):
            raise ValueError("allowed_claim_levels must not contain duplicates")
        overlap = set(self.allowed_claim_levels) & RESERVED_LIFECYCLE_STATE_NAMES
        if overlap:
            raise ValueError(
                "allowed_claim_levels must be report maturity levels, not lifecycle states: "
                f"{sorted(overlap)}"
            )
        return self


class RolePayloadValidation(RoleOntologyModel):
    """Result of validating a role-specific semantic payload."""

    role: str
    known_role: bool
    valid: bool
    review_required: bool
    missing_required_fields: tuple[str, ...] = Field(default_factory=tuple)
    matched_forbidden_fields: tuple[str, ...] = Field(default_factory=tuple)
    errors: tuple[str, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)


def get_role_definition(role: str) -> EvidenceRoleDefinition:
    """Return a role definition, using a support-free definition for unknown roles."""

    normalized = _normalize_role(role)
    definition = _ROLE_REGISTRY.get(normalized)
    if definition is not None:
        return definition
    return _unknown_role_definition(normalized or "<empty>")


def classify_role(role: str) -> SemanticClass:
    """Return the semantic class for a role, or ``unknown``."""

    return get_role_definition(role).semantic_class


def is_known_role(role: str) -> bool:
    """Return true when the role has a registered ontology definition."""

    return _normalize_role(role) in _ROLE_REGISTRY


def known_evidence_roles() -> frozenset[str]:
    """Return the registered evidence-role names."""

    return frozenset(_ROLE_REGISTRY)


def validate_role_payload(role: str, payload: Mapping[str, Any]) -> RolePayloadValidation:
    """Validate required and forbidden fields for a role-specific payload.

    Unknown roles are read-only accepted for preservation, but they never gain
    support permissions and always require review.
    """

    definition = get_role_definition(role)
    if not is_known_role(role):
        return RolePayloadValidation(
            role=definition.role,
            known_role=False,
            valid=True,
            review_required=True,
            warnings=("Unknown evidence role is preserved for review but has no support power.",),
        )

    missing = tuple(
        rule.field
        for rule in definition.required_fields
        if not _payload_field_matches(payload, rule)
    )
    matched_forbidden = tuple(
        path
        for rule in definition.forbidden_fields
        for path in _payload_field_matches(payload, rule)
    )
    errors = [f"missing required field {field!r}" for field in missing] + [
        f"forbidden field present at {path}" for path in matched_forbidden
    ]
    return RolePayloadValidation(
        role=definition.role,
        known_role=True,
        valid=not errors,
        review_required=bool(definition.review_rules),
        missing_required_fields=missing,
        matched_forbidden_fields=matched_forbidden,
        errors=tuple(errors),
    )


def support_permissions_for(role: str) -> tuple[SupportPermission, ...]:
    """Return the support permissions for a role.

    Unknown roles deliberately return an empty tuple.
    """

    if not is_known_role(role):
        return ()
    return get_role_definition(role).supports


def mandatory_non_claims_for(role: str) -> tuple[MandatoryNonClaim, ...]:
    """Return mandatory non-claims implied by a role."""

    if not is_known_role(role):
        return ()
    return get_role_definition(role).mandatory_non_claims


def roles_requiring_semantic_payload_validation() -> frozenset[str]:
    """Return roles whose attached JSON payloads should be ontology-checked."""

    roles = {
        role
        for role, definition in _ROLE_REGISTRY.items()
        if definition.required_fields or definition.forbidden_fields
    }
    return frozenset(roles - {"receipt_integrity"})


def role_support_matrix() -> dict[str, dict[str, list[str]]]:
    """Return a backward-compatible support matrix derived from the ontology."""

    matrix: dict[str, dict[str, list[str]]] = {}
    for role, definition in sorted(_ROLE_REGISTRY.items()):
        if not definition.supports and not definition.does_not_support:
            continue
        matrix[role] = {
            "supports": sorted({permission.support_scope for permission in definition.supports}),
            "does_not_support": sorted(
                {unsupported.claim_type for unsupported in definition.does_not_support}
            ),
            "requires_review_if": sorted(
                {
                    trigger.trigger_id
                    for trigger in definition.invalidation_triggers
                    if trigger.severity == "review"
                }
            ),
        }
    return matrix


def role_claim_level_matrix() -> dict[str, list[str]]:
    """Return role-to-report-claim-level permissions.

    This is intentionally report maturity support, not lifecycle state.
    """

    return {
        role: list(definition.allowed_claim_levels)
        for role, definition in sorted(_ROLE_REGISTRY.items())
    }


def role_non_claim_matrix() -> dict[str, list[str]]:
    """Return role-to-mandatory-non-claim identifiers."""

    return {
        role: [non_claim.non_claim_id for non_claim in definition.mandatory_non_claims]
        for role, definition in sorted(_ROLE_REGISTRY.items())
        if definition.mandatory_non_claims
    }


def validate_role_ontology_invariants() -> None:
    """Validate import-time ontology invariants.

    This is intentionally callable from tests so future taxonomy drift is caught
    without relying only on import-time exceptions.
    """

    if len(_ROLE_REGISTRY) != len(_ROLE_DEFINITIONS):
        raise RuntimeError("duplicate evidence role definitions")

    if set(_ALL_CLAIM_LEVELS) & RESERVED_LIFECYCLE_STATE_NAMES:
        raise RuntimeError("claim maturity levels overlap lifecycle state names")

    for role, definition in _ROLE_REGISTRY.items():
        if role in RESERVED_LIFECYCLE_STATE_NAMES:
            raise RuntimeError(f"role {role!r} reuses a lifecycle state name")

        unknown_levels = set(definition.allowed_claim_levels) - set(_ALL_CLAIM_LEVELS)
        if unknown_levels:
            raise RuntimeError(
                f"role {role!r} uses unknown report claim levels: {sorted(unknown_levels)}"
            )

        if not definition.supports and definition.allowed_claim_levels:
            # This is allowed for generic/context roles such as artifact/audit_log,
            # but it is intentionally visible to reviewers through the matrix.
            continue

        if definition.semantic_class == "integrity":
            for permission in definition.supports:
                if permission.strength != "integrity_only":
                    raise RuntimeError(
                        f"integrity role {role!r} must not emit non-integrity support strength"
                    )


def _normalize_role(role: str) -> str:
    return str(role or "").strip()


def _clean_str_tuple(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    cleaned = tuple(str(item).strip() for item in values)
    if any(not item for item in cleaned):
        raise ValueError(f"{field_name} must contain non-empty strings")
    return cleaned


def _payload_field_matches(payload: Mapping[str, Any], rule: PayloadFieldRule) -> tuple[str, ...]:
    field = rule.field.strip()
    matches: list[str] = []
    for path, key in _walk_payload_keys(payload):
        if (rule.match in {"path", "path_or_leaf"} and _path_matches(path, field)) or (
            rule.match in {"leaf", "path_or_leaf"} and key == field
        ):
            matches.append(path)
    return tuple(matches)


def _path_matches(path: str, field: str) -> bool:
    normalized_field = field[2:] if field.startswith("$.") else field
    normalized_path = path[2:] if path.startswith("$.") else path
    return normalized_path == normalized_field


def _walk_payload_keys(value: Any, path: str = "$") -> tuple[tuple[str, str], ...]:
    out: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}"
            out.append((child_path, key))
            out.extend(_walk_payload_keys(child, child_path))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            out.extend(_walk_payload_keys(child, f"{path}[{idx}]"))
    return tuple(out)


def _unknown_role_definition(role: str) -> EvidenceRoleDefinition:
    return EvidenceRoleDefinition(
        role=role,
        semantic_class="unknown",
        supports=(),
        does_not_support=(
            UnsupportedClaim(
                claim_type="any_claim_strengthening",
                target_markers=("claim.", "statistical", "deployment", "release"),
                reason="Unknown roles are preserved but cannot strengthen a claim.",
            ),
        ),
        mandatory_non_claims=(),
        allowed_claim_levels=(),
        staleness_behavior="unknown",
        exploratory_status="unknown",
        confirmatory_status="unknown",
        required_fields=(),
        forbidden_fields=(),
        review_rules=(
            ReviewRule(
                rule_id="unknown_role_resolution",
                description="Unknown evidence roles require human role-resolution review.",
            ),
        ),
        invalidation_triggers=(
            InvalidationTrigger(
                trigger_id="unknown_role",
                description="Unknown role cannot be interpreted by the ontology.",
                severity="review",
            ),
        ),
    )


def _sp(
    relation: SupportRelation,
    strength: SupportStrength,
    target: str | tuple[str, ...],
    scope: str,
) -> SupportPermission:
    targets = (target,) if isinstance(target, str) else target
    return SupportPermission(
        relation=relation,
        strength=strength,
        target_claim_fragments=targets,
        support_scope=scope,
    )


def _no_support(claim_type: str, markers: tuple[str, ...], reason: str) -> UnsupportedClaim:
    return UnsupportedClaim(claim_type=claim_type, target_markers=markers, reason=reason)


def _nc(
    non_claim_id: str,
    description: str,
    phrase_groups: tuple[tuple[str, ...], ...],
) -> MandatoryNonClaim:
    return MandatoryNonClaim(
        non_claim_id=non_claim_id,
        description=description,
        phrase_groups=phrase_groups,
    )


def _required(field: str, reason: str) -> PayloadFieldRule:
    return PayloadFieldRule(field=field, reason=reason)


def _forbidden(field: str, reason: str) -> PayloadFieldRule:
    return PayloadFieldRule(field=field, reason=reason)


def _review(rule_id: str, description: str) -> ReviewRule:
    return ReviewRule(rule_id=rule_id, description=description)


def _trigger(
    trigger_id: str,
    description: str,
    severity: InvalidationSeverity,
) -> InvalidationTrigger:
    return InvalidationTrigger(
        trigger_id=trigger_id,
        description=description,
        severity=severity,
    )


_ROLE_DEFINITIONS: tuple[EvidenceRoleDefinition, ...] = (
    EvidenceRoleDefinition(
        role="receipt_integrity",
        semantic_class="integrity",
        supports=(
            _sp(
                "integrity_binds",
                "integrity_only",
                "claim.integrity.receipt",
                "report_and_artifact_byte_integrity",
            ),
        ),
        does_not_support=(
            _no_support(
                "statistical_validity",
                ("statistical_validity", "statistical validity"),
                "Hashes do not prove experimental design, labels, or inference validity.",
            ),
            _no_support(
                "deployment_safety",
                ("deployment_safety", "deployment safety", "safety"),
                "Byte integrity is not deployment evidence.",
            ),
        ),
        mandatory_non_claims=(
            _nc(
                "receipt_integrity_not_statistical_validity_or_deployment_safety",
                "Receipt integrity checks do not prove statistical validity or deployment safety.",
                (("receipt", "hash"), ("statistical validity", "deployment safety", "safe")),
            ),
        ),
        allowed_claim_levels=_ALL_CLAIM_LEVELS,
        staleness_behavior="none",
        exploratory_status="not_exploratory",
        confirmatory_status="not_confirmatory",
        required_fields=(),
        forbidden_fields=(
            _forbidden(
                "deployment_safety_support",
                "Receipt integrity must not claim deployment-safety support.",
            ),
        ),
        review_rules=(),
        invalidation_triggers=(
            _trigger(
                "hash_mismatch",
                "Report or artifact hash mismatch invalidates integrity support.",
                "invalidates",
            ),
        ),
    ),
    EvidenceRoleDefinition(
        role="measurement_evidence",
        semantic_class="measurement",
        supports=(
            _sp(
                "bounds",
                "diagnostic",
                "claim.statistical_interval",
                "named_measurement_interval_under_report_scope",
            ),
        ),
        does_not_support=(
            _no_support(
                "deployment_safety",
                ("deployment_safety", "deployment safety", "safety"),
                "A measurement interval is scoped to its report population and assumptions.",
            ),
            _no_support(
                "external_validity",
                ("generalization", "external_validity", "representativeness"),
                "Measurement evidence does not establish representativeness by itself.",
            ),
        ),
        mandatory_non_claims=(
            _nc(
                "measurement_evidence_not_deployment_safety",
                "Measurement evidence is scoped and does not certify deployment safety.",
                (("measurement", "interval"), ("does not", "not"), ("safe", "deployment")),
            ),
        ),
        allowed_claim_levels=("diagnostic", "bounded_empirical", "reproducible_run"),
        staleness_behavior="artifact_timestamp",
        exploratory_status="not_exploratory",
        confirmatory_status="requires_confirmation",
        required_fields=(
            _required("metric_family", "Measurement evidence must identify the metric family."),
            _required("interval", "Measurement evidence must include the reported interval."),
        ),
        forbidden_fields=(
            _forbidden(
                "deployment_safety_support",
                "Measurement evidence cannot claim deployment safety.",
            ),
        ),
        review_rules=(),
        invalidation_triggers=(
            _trigger(
                "scope_shift",
                "Population, label, or measurement-scope drift requires review.",
                "review",
            ),
        ),
    ),
    EvidenceRoleDefinition(
        role="calibration_evidence",
        semantic_class="calibration",
        supports=(
            _sp(
                "qualifies",
                "diagnostic",
                "claim.operating_point",
                "calibration_boundary_for_the_named_operating_point",
            ),
        ),
        does_not_support=(
            _no_support(
                "deployment_safety",
                ("deployment_safety", "deployment safety", "safety"),
                "Calibration qualifies an operating point, not deployment safety.",
            ),
            _no_support(
                "external_validity",
                ("external_validity", "generalization", "representativeness"),
                "Calibration does not prove transfer outside the calibration window.",
            ),
        ),
        mandatory_non_claims=(),
        allowed_claim_levels=("diagnostic", "bounded_empirical", "reproducible_run"),
        staleness_behavior="artifact_timestamp",
        exploratory_status="not_exploratory",
        confirmatory_status="requires_confirmation",
        required_fields=(
            _required("status", "Calibration evidence must include a calibration status."),
        ),
        forbidden_fields=(
            _forbidden(
                "deployment_safety_support",
                "Calibration evidence cannot claim deployment safety.",
            ),
        ),
        review_rules=(),
        invalidation_triggers=(
            _trigger(
                "calibration_window_shift",
                "A shifted calibration window requires review.",
                "review",
            ),
        ),
    ),
    EvidenceRoleDefinition(
        role="claim_decay",
        semantic_class="temporal",
        supports=(
            _sp("qualifies", "diagnostic", "claim.staleness", "time_bounding"),
            _sp("requires_review", "diagnostic", "claim.review_pressure", "staleness_review"),
            _sp("invalidates", "diagnostic", "claim.freshness", "freshness_invalidation"),
        ),
        does_not_support=(
            _no_support(
                "deployment_safety",
                ("deployment_safety", "deployment safety", "safety"),
                "Claim-decay evidence controls freshness and review pressure only.",
            ),
            _no_support(
                "statistical_validity",
                ("statistical_validity", "statistical validity"),
                "Claim decay does not validate the underlying statistical evidence.",
            ),
            _no_support(
                "confirmatory_evidence",
                ("confirmatory", "certificate"),
                "A staleness policy is not confirmatory evidence.",
            ),
        ),
        mandatory_non_claims=(
            _nc(
                "claim_decay_not_current_safety_proof",
                "Claim decay does not prove the system is currently safe; it defines recheck, "
                "degrade, or expiry conditions.",
                (("claim decay",), ("does not prove", "not prove"), ("safe",)),
            ),
        ),
        allowed_claim_levels=_ALL_CLAIM_LEVELS,
        staleness_behavior="verification_time",
        exploratory_status="not_exploratory",
        confirmatory_status="not_confirmatory",
        required_fields=(
            _required("schema_version", "Claim-decay payloads must declare their schema."),
            _required("claim_id", "Claim-decay payloads must identify the claim."),
            _required("issued_at", "Claim-decay payloads must have an issue time."),
            _required("policy", "Claim-decay payloads must include a decay policy."),
        ),
        forbidden_fields=(
            _forbidden(
                "live_status_as_signed_truth",
                "Signed decay artifacts must not store live freshness as signed truth.",
            ),
        ),
        review_rules=(_review("degraded_claim_decay", "Degraded decay state requires review."),),
        invalidation_triggers=(
            _trigger("expired", "Expired decay state invalidates freshness.", "expires"),
            _trigger("degraded", "Degraded decay state requires review.", "review"),
            _trigger(
                "triggered_versions",
                "Watched version changes expire the decay state.",
                "expires",
            ),
        ),
    ),
    EvidenceRoleDefinition(
        role="extremal_scenario",
        semantic_class="scenario",
        supports=(
            _sp(
                "bounds",
                "diagnostic",
                "claim.endpoint_feasibility",
                "dependence_endpoint_explanation",
            ),
            _sp(
                "qualifies",
                "diagnostic",
                "claim.endpoint_feasibility",
                "counterfactual_feasibility",
            ),
        ),
        does_not_support=(
            _no_support(
                "likelihood",
                ("likelihood", "likely"),
                "Endpoint scenarios witness feasibility, not probability in deployment.",
            ),
            _no_support(
                "deployment_realization",
                ("deployment_realization", "deployment realization"),
                "Endpoint worlds are not deployment-realization claims.",
            ),
            _no_support(
                "model_truth_claim",
                ("model_truth_claim", "model truth"),
                "Fitted or extremal scenario models are not truth claims.",
            ),
        ),
        mandatory_non_claims=(
            _nc(
                "extremal_scenario_not_likely_world_proof",
                "Extremal scenarios do not prove the endpoint or fitted world is likely.",
                (("extremal scenario",), ("does not prove", "not prove"), ("likely",)),
            ),
        ),
        allowed_claim_levels=_ALL_CLAIM_LEVELS,
        staleness_behavior="artifact_timestamp",
        exploratory_status="mixed",
        confirmatory_status="requires_confirmation",
        required_fields=(
            _required("schema_version", "Scenario payloads must declare their schema."),
            _required("scenario_id", "Scenario payloads must identify the scenario."),
            _required("kind", "Scenario payloads must state the scenario kind."),
            _required("feasibility", "Scenario payloads must include feasibility diagnostics."),
            _required("non_claims", "Scenario payloads must preserve non-claims."),
        ),
        forbidden_fields=(
            _forbidden(
                "deployment_realization_claim",
                "Scenario artifacts cannot claim deployment realization.",
            ),
            _forbidden("likelihood_claim", "Scenario artifacts cannot claim likelihood."),
        ),
        review_rules=(
            _review(
                "excluded_evidence_fields",
                "Excluded or adaptive evidence fields require review.",
            ),
            _review(
                "fitted_without_confirmation",
                "Fitted empirical scenarios require separate confirmation.",
            ),
        ),
        invalidation_triggers=(
            _trigger(
                "infeasible",
                "Infeasible scenarios invalidate scenario support.",
                "invalidates",
            ),
            _trigger(
                "excluded_evidence_fields",
                "Excluded evidence fields require review.",
                "review",
            ),
            _trigger(
                "fitted_without_confirmation",
                "Fitted empirical scenario without confirmation requires review.",
                "review",
            ),
        ),
    ),
    EvidenceRoleDefinition(
        role="exploratory_redteam",
        semantic_class="redteam",
        supports=(
            _sp(
                "exploratory_suggests",
                "weak",
                "claim.redteam_hypothesis",
                "hypothesis_generation_and_triage",
            ),
            _sp(
                "requires_review",
                "weak",
                "claim.confirmatory_boundary",
                "confirmatory_firewall_pressure",
            ),
        ),
        does_not_support=(
            _no_support(
                "confirmatory_evidence",
                ("confirmatory", "certificate", "confirmatory_ci"),
                "Exploratory red-team evidence is not confirmatory evidence.",
            ),
            _no_support(
                "release_claim",
                ("release", "deployment_safety", "deployment safety"),
                "Exploratory evidence cannot support release claims.",
            ),
        ),
        mandatory_non_claims=(
            _nc(
                "exploratory_redteam_not_confirmatory_certificate",
                "Exploratory red-team evidence is not a confirmatory certificate.",
                (("exploratory",), ("confirmatory",)),
            ),
        ),
        allowed_claim_levels=("diagnostic",),
        staleness_behavior="artifact_timestamp",
        exploratory_status="exploratory",
        confirmatory_status="not_confirmatory",
        required_fields=(
            _required("redteam_id", "Exploratory red-team payloads must identify the run."),
            _required(
                "discovery_protocol",
                "Exploratory red-team payloads must identify the discovery protocol.",
            ),
            _required("findings", "Exploratory red-team payloads must contain findings."),
        ),
        forbidden_fields=(
            _forbidden(
                "confirmatory_ci",
                "Exploratory red-team payloads must not surface confirmatory intervals.",
            ),
        ),
        review_rules=(
            _review(
                "exploratory_claim_level",
                "Exploratory evidence above diagnostic use requires review.",
            ),
        ),
        invalidation_triggers=(
            _trigger(
                "confirmatory_field_leak",
                "Confirmatory fields in exploratory payloads fail the firewall.",
                "invalidates",
            ),
        ),
    ),
    EvidenceRoleDefinition(
        role="confirmatory_protocol",
        semantic_class="protocol",
        supports=(
            _sp(
                "confirmatory_tests",
                "confirmatory",
                "claim.confirmatory_boundary",
                "pre_registered_plan_and_confirmatory_run_separation",
            ),
            _sp(
                "qualifies",
                "confirmatory",
                "claim.statistical_interval",
                "fixed_endpoint_analysis_sample_and_stopping_protocol",
            ),
            _sp(
                "requires_review",
                "weak",
                "claim.confirmatory_boundary",
                "confirmatory_protocol_review_trigger",
            ),
            _sp(
                "invalidates",
                "diagnostic",
                "claim.confirmatory_boundary",
                "confirmatory_protocol_firewall_failure",
            ),
        ),
        does_not_support=(
            _no_support(
                "deployment_safety",
                ("deployment_safety", "deployment safety", "safety"),
                "Confirmatory protocols still do not certify deployment safety.",
            ),
            _no_support(
                "external_validity",
                ("external_validity", "generalization", "representativeness"),
                "Confirmation is scoped to the declared sample, endpoint, and run.",
            ),
            _no_support(
                "adaptive_discovery_reuse",
                ("adaptive", "exploratory", "post-selection"),
                "Adaptive discovery can motivate a protocol but cannot become "
                "confirmatory evidence by relabeling.",
            ),
        ),
        mandatory_non_claims=(
            _nc(
                "confirmatory_protocol_validity_depends_on_protocol_not_polish",
                "Confirmatory validity depends on the protocol and run separation, not "
                "report polish.",
                (("confirmatory", "protocol"), ("not", "depends"), ("polish",)),
            ),
            _nc(
                "confirmatory_protocol_not_deployment_safety",
                "Confirmatory protocol evidence is scoped and does not certify deployment safety.",
                (("confirmatory", "protocol"), ("does not", "not"), ("deployment", "safe")),
            ),
        ),
        allowed_claim_levels=_CONFIRMATORY_CLAIM_LEVELS,
        staleness_behavior="artifact_timestamp",
        exploratory_status="not_exploratory",
        confirmatory_status="confirmatory",
        required_fields=(
            _required(
                "schema_version",
                "Confirmatory protocol artifacts must declare their schema.",
            ),
            _required("artifact_id", "Confirmatory protocol artifacts must identify themselves."),
            _required("plan", "Confirmatory protocol artifacts must reference the plan."),
            _required("run", "Confirmatory protocol artifacts must reference the run."),
            _required("non_claims", "Confirmatory protocol artifacts must preserve non-claims."),
        ),
        forbidden_fields=(
            _forbidden(
                "adaptive_search_ci",
                "Adaptive search intervals cannot be presented as confirmatory protocol output.",
            ),
            _forbidden(
                "exploratory_ci",
                "Exploratory intervals cannot be presented as confirmatory protocol output.",
            ),
            _forbidden(
                "non_confirmatory_ci",
                "Non-confirmatory intervals cannot be relabeled by a confirmatory protocol.",
            ),
        ),
        review_rules=(
            _review(
                "missing_or_weak_stopping_rule",
                "Missing stopping rules require review or invalidate stronger claim surfaces.",
            ),
            _review(
                "clustered_without_blocking",
                "Clustered data without cluster blocking requires review.",
            ),
        ),
        invalidation_triggers=(
            _trigger(
                "temporal_order_violation",
                "The protocol plan must be created before the confirmatory run starts.",
                "invalidates",
            ),
            _trigger(
                "adaptive_discovery_reuse",
                "Adaptive discovery artifacts cannot be confirmatory evidence by role rename.",
                "invalidates",
            ),
            _trigger(
                "clustered_without_blocking",
                "Clustered data without cluster blocking requires review.",
                "review",
            ),
        ),
    ),
    EvidenceRoleDefinition(
        role="confirmatory_failure_matrix",
        semantic_class="scenario",
        supports=(
            _sp(
                "confirmatory_tests",
                "confirmatory",
                "claim.failure_matrix",
                "predeclared_or_heldout_failure_matrix",
            ),
            _sp(
                "bounds",
                "confirmatory",
                "claim.statistical_interval",
                "confirmatory_failure_interval",
            ),
        ),
        does_not_support=(
            _no_support(
                "deployment_safety",
                ("deployment_safety", "deployment safety", "safety"),
                "Confirmatory failure matrices still do not certify deployment safety.",
            ),
            _no_support(
                "external_validity",
                ("external_validity", "generalization", "representativeness"),
                "Confirmation is scoped to the declared sample and protocol.",
            ),
        ),
        mandatory_non_claims=(
            _nc(
                "confirmatory_failure_matrix_not_deployment_safety",
                "Confirmatory failure-matrix evidence is scoped and does not certify "
                "deployment safety.",
                (("confirmatory",), ("does not", "not"), ("deployment", "safe")),
            ),
        ),
        allowed_claim_levels=_CONFIRMATORY_CLAIM_LEVELS,
        staleness_behavior="artifact_timestamp",
        exploratory_status="not_exploratory",
        confirmatory_status="confirmatory",
        required_fields=(
            _required(
                "failure_matrix",
                "Confirmatory failure-matrix payloads must include the matrix.",
            ),
            _required(
                "confirmatory_ci",
                "Confirmatory failure-matrix payloads must include the confirmatory interval.",
            ),
            _required("protocol_id", "Confirmatory payloads must identify the protocol."),
        ),
        forbidden_fields=(
            _forbidden(
                "adaptive_search_ci",
                "Adaptive intervals cannot be used as confirmatory intervals.",
            ),
        ),
        review_rules=(),
        invalidation_triggers=(
            _trigger(
                "missing_confirmatory_ci",
                "Missing confirmatory interval invalidates confirmatory support.",
                "invalidates",
            ),
        ),
    ),
    EvidenceRoleDefinition(
        role="fitted_empirical_scenario",
        semantic_class="scenario",
        supports=(
            _sp(
                "bounds",
                "diagnostic",
                "claim.fitted_scenario",
                "fitted_empirical_scenario_description",
            ),
        ),
        does_not_support=(
            _no_support(
                "model_truth_claim",
                ("model_truth_claim", "model truth"),
                "A fitted empirical scenario is not a claim that the fitted model is true.",
            ),
            _no_support(
                "deployment_safety",
                ("deployment_safety", "deployment safety", "safety"),
                "Fitted scenarios do not certify deployment safety.",
            ),
        ),
        mandatory_non_claims=(
            _nc(
                "fitted_empirical_scenario_not_model_truth",
                "A fitted empirical scenario does not prove the fitted dependence model is true.",
                (("fitted",), ("does not prove", "not prove"), ("model", "true")),
            ),
        ),
        allowed_claim_levels=("diagnostic", "bounded_empirical"),
        staleness_behavior="artifact_timestamp",
        exploratory_status="mixed",
        confirmatory_status="requires_confirmation",
        required_fields=(
            _required("scenario_id", "Fitted scenarios must identify the scenario."),
            _required("fit_protocol", "Fitted scenarios must identify the fit protocol."),
            _required("non_claims", "Fitted scenarios must preserve non-claims."),
        ),
        forbidden_fields=(
            _forbidden(
                "model_truth_claim",
                "Fitted empirical scenarios cannot claim the fitted model is true.",
            ),
        ),
        review_rules=(
            _review(
                "fitted_without_confirmation",
                "Fitted empirical scenarios require review unless separately confirmed.",
            ),
        ),
        invalidation_triggers=(
            _trigger(
                "model_truth_claim",
                "Model-truth claims in fitted artifacts invalidate the role boundary.",
                "invalidates",
            ),
        ),
    ),
    EvidenceRoleDefinition(
        role="human_review_note",
        semantic_class="review",
        supports=(
            _sp(
                "qualifies",
                "weak",
                ("claim.review_authorization", "evidence.*"),
                "scoped_authorization_of_reviewed_artifact_set",
            ),
        ),
        does_not_support=(
            _no_support(
                "statistical_validity",
                ("statistical_validity", "statistical validity"),
                "Human review does not upgrade the underlying statistical evidence.",
            ),
            _no_support(
                "deployment_safety",
                ("deployment_safety", "deployment safety", "safety"),
                "Human review notes are scoped authorization, not deployment proof.",
            ),
            _no_support(
                "unreviewed_artifacts",
                ("unreviewed", "unreviewed_artifact"),
                "Review notes cannot authorize artifacts they did not review.",
            ),
        ),
        mandatory_non_claims=(
            _nc(
                "human_review_note_not_evidence_upgrade",
                "Human review does not upgrade underlying statistical evidence.",
                (("human review",), ("does not upgrade", "not upgrade", "cannot upgrade")),
            ),
        ),
        allowed_claim_levels=_ALL_CLAIM_LEVELS,
        staleness_behavior="review_expiry",
        exploratory_status="not_exploratory",
        confirmatory_status="not_confirmatory",
        required_fields=(
            _required("review_id", "Human review notes must identify the review."),
            _required("reviewer", "Human review notes must identify the reviewer."),
            _required(
                "reviewed_artifact_hashes",
                "Human review notes must bind reviewed artifact hashes.",
            ),
            _required(
                "reviewed_claim_level",
                "Human review notes must identify the reviewed claim level.",
            ),
            _required("decision", "Human review notes must record a scoped decision."),
        ),
        forbidden_fields=(
            _forbidden(
                "replaces_artifact_hashes",
                "Human review notes cannot replace artifact hashes they did not review.",
            ),
            _forbidden(
                "hash_overrides",
                "Human review notes cannot override report-bound artifact hashes.",
            ),
        ),
        review_rules=(
            _review(
                "hash_bound_review",
                "Review can reduce a review requirement only for the reviewed hash set.",
            ),
        ),
        invalidation_triggers=(
            _trigger(
                "partial_artifact_set",
                "A review over a partial artifact set cannot satisfy review requirements.",
                "review",
            ),
            _trigger(
                "hash_replacement_attempt",
                "Attempted artifact-hash replacement invalidates review semantics.",
                "invalidates",
            ),
        ),
    ),
    EvidenceRoleDefinition(
        role="human_review",
        semantic_class="review",
        supports=(
            _sp(
                "qualifies",
                "weak",
                ("claim.review_authorization", "evidence.*"),
                "scoped_authorization_of_reviewed_artifact_set",
            ),
        ),
        does_not_support=(
            _no_support(
                "statistical_validity",
                ("statistical_validity", "statistical validity"),
                "Human review does not upgrade statistical evidence.",
            ),
            _no_support(
                "deployment_safety",
                ("deployment_safety", "deployment safety", "safety"),
                "Human review is not a deployment-safety proof.",
            ),
        ),
        mandatory_non_claims=(
            _nc(
                "human_review_not_evidence_upgrade",
                "Human review does not upgrade underlying statistical evidence.",
                (("human review",), ("does not upgrade", "not upgrade", "cannot upgrade")),
            ),
        ),
        allowed_claim_levels=_ALL_CLAIM_LEVELS,
        staleness_behavior="review_expiry",
        exploratory_status="not_exploratory",
        confirmatory_status="not_confirmatory",
        required_fields=(),
        forbidden_fields=(
            _forbidden(
                "hash_overrides",
                "Human review cannot override report-bound artifact hashes.",
            ),
        ),
        review_rules=(
            _review(
                "hash_bound_review",
                "Review can affect review state only for the reviewed hash set.",
            ),
        ),
        invalidation_triggers=(
            _trigger(
                "partial_artifact_set",
                "A review over a partial artifact set cannot satisfy review requirements.",
                "review",
            ),
        ),
    ),
    EvidenceRoleDefinition(
        role="artifact",
        semantic_class="generic",
        supports=(),
        does_not_support=(
            _no_support(
                "unstated_support",
                ("claim.",),
                "Generic artifacts are preserved but do not create support edges.",
            ),
        ),
        mandatory_non_claims=(),
        allowed_claim_levels=_ALL_CLAIM_LEVELS,
        staleness_behavior="none",
        exploratory_status="unknown",
        confirmatory_status="unknown",
        required_fields=(),
        forbidden_fields=(),
        review_rules=(),
        invalidation_triggers=(),
    ),
    EvidenceRoleDefinition(
        role="audit_log",
        semantic_class="generic",
        supports=(),
        does_not_support=(
            _no_support(
                "unstated_support",
                ("claim.",),
                "Audit logs are integrity-bound context, not support edges by themselves.",
            ),
        ),
        mandatory_non_claims=(),
        allowed_claim_levels=_ALL_CLAIM_LEVELS,
        staleness_behavior="none",
        exploratory_status="unknown",
        confirmatory_status="unknown",
        required_fields=(),
        forbidden_fields=(),
        review_rules=(),
        invalidation_triggers=(),
    ),
    EvidenceRoleDefinition(
        role="figure_manifest",
        semantic_class="generic",
        supports=(),
        does_not_support=(
            _no_support(
                "unstated_support",
                ("claim.",),
                "Figure manifests bind presentation artifacts; they do not prove claims.",
            ),
        ),
        mandatory_non_claims=(),
        allowed_claim_levels=_ALL_CLAIM_LEVELS,
        staleness_behavior="none",
        exploratory_status="unknown",
        confirmatory_status="unknown",
        required_fields=(),
        forbidden_fields=(),
        review_rules=(),
        invalidation_triggers=(),
    ),
)

_ROLE_REGISTRY: dict[str, EvidenceRoleDefinition] = {
    definition.role: definition for definition in _ROLE_DEFINITIONS
}

validate_role_ontology_invariants()


__all__ = [
    "RESERVED_LIFECYCLE_STATE_NAMES",
    "ROLE_ONTOLOGY_SCHEMA_VERSION",
    "ClaimMaturityLevel",
    "EvidenceRoleDefinition",
    "InvalidationTrigger",
    "MandatoryNonClaim",
    "PayloadFieldRule",
    "ReviewRule",
    "RolePayloadValidation",
    "SupportPermission",
    "UnsupportedClaim",
    "classify_role",
    "get_role_definition",
    "is_known_role",
    "known_evidence_roles",
    "mandatory_non_claims_for",
    "role_claim_level_matrix",
    "role_non_claim_matrix",
    "role_support_matrix",
    "roles_requiring_semantic_payload_validation",
    "support_permissions_for",
    "validate_role_ontology_invariants",
    "validate_role_payload",
]

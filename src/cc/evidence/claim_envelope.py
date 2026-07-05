"""Typed claim and boundary envelopes for evidence-bound CC reports.

The envelope is an intermediate representation around ``cc.report.v0.3.1``.

Important semantic boundary
---------------------------
This module compiles a report plus optional governance audit into a typed,
hashable envelope. It does not change the stable report schema, it does not
promote evidence roles into broader claims than they are allowed to support,
and it does not own claim lifecycle semantics.

This file owns:

- report-to-envelope projection;
- claim fragments targeted by evidence support edges;
- boundary scope, assumptions, non-claims, defeaters, invalidation conditions;
- support graph construction from conservative evidence-role ontology;
- governance audit projection into envelope fields.

This file does not own:

- claim lifecycle state machine;
- claim transitions;
- claim compiler;
- assumption registry;
- challenge calculus;
- dashboard display state;
- deployment/compliance certification.

Future lifecycle states such as draft/supported/bounded/challenged/weakened/
expired/revoked/superseded/non_claim belong in ``cc.claims.ClaimState``.

The ``GovernanceState`` in this module is only an audit projection attached to
an envelope. It is not the same thing as a future claim lifecycle state.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cc.evidence.role_ontology import (
    RESERVED_LIFECYCLE_STATE_NAMES,
    SupportPermission,
    classify_role,
    get_role_definition,
    support_permissions_for,
)
from cc.reporting.canonical import canonical_json_bytes
from cc.reporting.report import ALLOWED_CLAIM_LEVELS, CLAIM_LEVELS, SCHEMA_VERSION, ClaimLevel

CLAIM_ENVELOPE_SCHEMA_VERSION: Literal["cc.claim_envelope.v1"] = "cc.claim_envelope.v1"
BOUNDARY_ENVELOPE_SCHEMA_VERSION: Literal["cc.boundary_envelope.v1"] = "cc.boundary_envelope.v1"
SUPPORT_GRAPH_SCHEMA_VERSION: Literal["cc.support_graph.v1"] = "cc.support_graph.v1"
SUPPORT_SUMMARY_SCHEMA_VERSION: Literal["cc.claim_envelope.support_summary.v1"] = (
    "cc.claim_envelope.support_summary.v1"
)
GOVERNANCE_STATE_SCHEMA_VERSION: Literal["cc.claim_envelope.governance_state.v1"] = (
    "cc.claim_envelope.governance_state.v1"
)

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
SupportStrength = Literal["weak", "diagnostic", "confirmatory", "integrity_only"]

# Audit projection verdicts, not claim lifecycle states.
GovernanceVerdict: TypeAlias = Literal["not_evaluated", "pass", "needs_review", "fail"]

# Audit projection freshness, not lifecycle state. Future cc.claims may translate
# this into lifecycle transitions, but this envelope only records the verifier view.
ClaimFreshnessStatus: TypeAlias = Literal["not_evaluated", "fresh", "degraded", "expired"]

DefeaterStatus: TypeAlias = Literal["active", "unresolved", "resolved", "unknown"]
InvalidationSeverity: TypeAlias = Literal["review", "invalidates", "expires", "unknown"]
ReviewStatus: TypeAlias = Literal["required", "not_required", "satisfied", "unknown"]

_STRENGTH_RANK: dict[SupportStrength, int] = {
    "integrity_only": 0,
    "weak": 1,
    "diagnostic": 2,
    "confirmatory": 3,
}

_ALLOWED_GOVERNANCE_VERDICTS = frozenset(
    {"not_evaluated", "pass", "needs_review", "fail"}
)
_ALLOWED_FRESHNESS_STATUSES = frozenset(
    {"not_evaluated", "fresh", "degraded", "expired"}
)

_NON_PROOF_RECEIPT = (
    "Receipt evidence binds artifact integrity only; it does not prove statistical validity "
    "or deployment safety.",
)
_NON_PROOF_MEASUREMENT = (
    "Measurement evidence is scoped to the report population, assumptions, calibration "
    "window, and interval method; it does not certify deployment safety.",
)
_NON_PROOF_DECAY = (
    "Claim-decay evidence supports staleness, time bounding, or review pressure only; it "
    "does not support deployment safety or statistical validity.",
)
_NON_PROOF_SCENARIO = (
    "Extremal scenarios support endpoint feasibility or counterfactual bounds; they do "
    "not support likelihood or deployment realization.",
)
_NON_PROOF_REVIEW = (
    "Human review can authorize scoped use of an evidence package; it does not upgrade "
    "underlying statistical evidence.",
)
_NON_PROOF_CONFIRMATORY = (
    "Confirmatory evidence remains scoped to its protocol, sample, endpoint, and declared "
    "run separation; it does not certify deployment safety.",
)
_UNKNOWN_ROLE_NON_CLAIM = (
    "Unknown evidence roles are preserved for review but cannot strengthen the claim.",
)


class EnvelopeModel(BaseModel):
    """Strict frozen base model for claim-envelope artifacts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
        strict=True,
    )


class ClaimFragment(EnvelopeModel):
    """Named claim fragment that support edges may target."""

    fragment_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    fragment_type: str = Field(min_length=1)


class ClaimIdentity(EnvelopeModel):
    """Stable identity for a compiled claim envelope."""

    artifact_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    subject_ref: str = Field(min_length=1)
    source_report_id: str = Field(min_length=1)
    source_report_schema: str = Field(min_length=1)
    source_report_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    created_at: str | None = None
    evaluated_at: str | None = None


class ClaimProposition(EnvelopeModel):
    """Positive claim text and report maturity level.

    ``allowed_claim_level`` is a report maturity/support label from
    ``cc.reporting.report``. It is not a lifecycle state.
    """

    statement: str = Field(min_length=1)
    allowed_claim_level: ClaimLevel
    fragments: tuple[ClaimFragment, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _validate_claim_level(self) -> ClaimProposition:
        if self.allowed_claim_level not in ALLOWED_CLAIM_LEVELS:
            raise ValueError("allowed_claim_level is not supported by cc.report.v0.3.1")
        if self.allowed_claim_level in RESERVED_LIFECYCLE_STATE_NAMES:
            raise ValueError("allowed_claim_level must not reuse a lifecycle state name")
        return self


class MeasurementInterval(EnvelopeModel):
    """Report measurement interval scoped into the boundary envelope."""

    lower: float
    upper: float
    point_estimate: float | None = None
    confidence_level: float | None = None
    delta: float | None = None

    @model_validator(mode="after")
    def _interval_is_ordered(self) -> MeasurementInterval:
        if self.lower > self.upper:
            raise ValueError("measurement interval lower cannot exceed upper")
        if self.point_estimate is not None and not self.lower <= self.point_estimate <= self.upper:
            raise ValueError("point_estimate must lie inside measurement interval")
        return self


class BoundaryScope(EnvelopeModel):
    """Machine-readable scope inherited from a report."""

    subject_ref: str = Field(min_length=1)
    report_id: str = Field(min_length=1)
    report_schema_version: str = Field(min_length=1)
    run_id: str | None = None
    metric_family: str | None = None
    interval_method: str | None = None
    measurement_interval: MeasurementInterval | None = None
    sample_sizes: dict[str, int] = Field(default_factory=dict)
    calibration_status: str | None = None
    config_path: str | None = None
    evidence_artifact_ids: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("sample_sizes")
    @classmethod
    def _sample_sizes_are_non_negative(cls, value: dict[str, int]) -> dict[str, int]:
        out: dict[str, int] = {}
        for label, sample_size in value.items():
            if not label:
                raise ValueError("sample size labels must be non-empty")
            if isinstance(sample_size, bool) or sample_size < 0:
                raise ValueError("sample sizes must be non-negative integers")
            out[str(label)] = int(sample_size)
        return dict(sorted(out.items()))


class BoundaryDefeater(EnvelopeModel):
    """Condition or gap that can weaken, invalidate, or block reliance.

    This is a proto-challenge boundary artifact. It is not a full future
    ``cc.claims.Challenge`` object and it does not own lifecycle transitions.
    """

    defeater_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    source_ref: str | None = None
    status: DefeaterStatus = "unresolved"


class InvalidationCondition(EnvelopeModel):
    """Machine-readable condition under which support no longer holds.

    This is an envelope support-boundary condition. It is not the same thing as
    revoking a future ``cc.claims.Claim``.
    """

    condition_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    trigger: str = Field(min_length=1)
    source_ref: str | None = None
    severity: InvalidationSeverity = "review"


class ReviewRequirement(EnvelopeModel):
    """Human or verifier review requirement attached to the claim boundary."""

    requirement_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    source_ref: str | None = None
    required: bool = True
    status: ReviewStatus = "required"

    @model_validator(mode="after")
    def _status_matches_required_flag(self) -> ReviewRequirement:
        if self.required and self.status == "not_required":
            raise ValueError("required review requirement cannot have status not_required")
        return self


class BoundaryEnvelope(EnvelopeModel):
    """Boundary, non-claims, defeaters, invalidators, and review requirements."""

    schema_: Literal["cc.boundary_envelope.v1"] = Field(
        default=BOUNDARY_ENVELOPE_SCHEMA_VERSION,
        alias="schema",
    )
    scope: BoundaryScope
    assumptions: tuple[str, ...] = Field(default_factory=tuple)
    non_claims: tuple[str, ...] = Field(default_factory=tuple)
    defeaters: tuple[BoundaryDefeater, ...] = Field(default_factory=tuple)
    invalidation_conditions: tuple[InvalidationCondition, ...] = Field(default_factory=tuple)
    review_requirements: tuple[ReviewRequirement, ...] = Field(default_factory=tuple)

    @field_validator("assumptions", "non_claims")
    @classmethod
    def _clean_boundary_strings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _dedupe(value)


class ArtifactRef(EnvelopeModel):
    """Evidence, receipt, scenario, decay, or review artifact reference."""

    artifact_id: str = Field(min_length=1)
    subject_ref: str = Field(min_length=1)
    role: str = Field(min_length=1)
    path: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    bytes: int | None = Field(default=None, ge=0)
    schema_: str | None = Field(default=None, alias="schema")
    status: str | None = None
    reason: str | None = None
    created_at: str | None = None
    evaluated_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("role")
    @classmethod
    def _role_is_not_lifecycle_state(cls, value: str) -> str:
        role = value.strip()
        if role in RESERVED_LIFECYCLE_STATE_NAMES:
            raise ValueError("artifact role must not reuse a lifecycle state name")
        return role


class SupportEdge(EnvelopeModel):
    """Typed edge from one artifact to one claim fragment."""

    source_artifact_id: str = Field(min_length=1)
    target_claim_fragment: str = Field(min_length=1)
    relation: SupportRelation
    strength: SupportStrength
    non_claims: tuple[str, ...] = Field(default_factory=tuple)
    rationale: str | None = None

    @field_validator("non_claims")
    @classmethod
    def _non_claims_are_clean(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _dedupe(value)


class EnvelopeSupportSummary(EnvelopeModel):
    """Compact verifier-facing summary of support graph semantics."""

    schema_: Literal["cc.claim_envelope.support_summary.v1"] = Field(
        default=SUPPORT_SUMMARY_SCHEMA_VERSION,
        alias="schema",
    )
    support_edge_count: int = Field(ge=0)
    relation_counts: dict[str, int] = Field(default_factory=dict)
    strength_counts: dict[str, int] = Field(default_factory=dict)
    strongest_non_integrity_strength: SupportStrength | None = None
    integrity_only_edges: int = Field(default=0, ge=0)
    unknown_role_refs: int = Field(default=0, ge=0)
    unsupported_role_refs: tuple[str, ...] = Field(default_factory=tuple)
    review_edges: int = Field(default=0, ge=0)

    @classmethod
    def from_graph(cls, graph: SupportGraph) -> EnvelopeSupportSummary:
        relation_counts: dict[str, int] = {}
        strength_counts: dict[str, int] = {}
        strongest: SupportStrength | None = None

        for edge in graph.support_edges:
            relation_counts[edge.relation] = relation_counts.get(edge.relation, 0) + 1
            strength_counts[edge.strength] = strength_counts.get(edge.strength, 0) + 1
            if edge.strength != "integrity_only" and (
                strongest is None or _STRENGTH_RANK[edge.strength] > _STRENGTH_RANK[strongest]
            ):
                strongest = edge.strength

        refs = graph.all_refs()
        unsupported = tuple(
            ref.artifact_id for ref in refs if classify_role(ref.role) == "unknown"
        )

        return cls(
            support_edge_count=len(graph.support_edges),
            relation_counts=dict(sorted(relation_counts.items())),
            strength_counts=dict(sorted(strength_counts.items())),
            strongest_non_integrity_strength=strongest,
            integrity_only_edges=strength_counts.get("integrity_only", 0),
            unknown_role_refs=len(unsupported),
            unsupported_role_refs=unsupported,
            review_edges=relation_counts.get("requires_review", 0),
        )


class SupportGraph(EnvelopeModel):
    """Typed support graph over evidence roles and claim fragments."""

    schema_: Literal["cc.support_graph.v1"] = Field(
        default=SUPPORT_GRAPH_SCHEMA_VERSION,
        alias="schema",
    )
    evidence_refs: tuple[ArtifactRef, ...] = Field(default_factory=tuple)
    scenario_refs: tuple[ArtifactRef, ...] = Field(default_factory=tuple)
    decay_refs: tuple[ArtifactRef, ...] = Field(default_factory=tuple)
    receipt_refs: tuple[ArtifactRef, ...] = Field(default_factory=tuple)
    review_refs: tuple[ArtifactRef, ...] = Field(default_factory=tuple)
    support_edges: tuple[SupportEdge, ...] = Field(default_factory=tuple)

    def all_refs(self) -> tuple[ArtifactRef, ...]:
        return (
            *self.evidence_refs,
            *self.scenario_refs,
            *self.decay_refs,
            *self.receipt_refs,
            *self.review_refs,
        )

    @model_validator(mode="after")
    def _validate_edges(self) -> SupportGraph:
        refs = self.all_refs()
        ref_by_id = {ref.artifact_id: ref for ref in refs}
        if len(ref_by_id) != len(refs):
            raise ValueError("artifact_id values must be unique across a support graph")

        for edge in self.support_edges:
            source = ref_by_id.get(edge.source_artifact_id)
            if source is None:
                raise ValueError(f"support edge source {edge.source_artifact_id!r} is unknown")
            _validate_role_support_edge(source, edge)

        return self


class GovernanceState(EnvelopeModel):
    """Computed or attached governance audit projection for the envelope.

    This is not a claim lifecycle state.

    ``verdict`` records the verifier's envelope/report consistency outcome.
    ``freshness_status`` records the verifier's freshness projection.

    Future ``cc.claims`` may translate these values into lifecycle transitions,
    but this module only stores the audit projection.
    """

    schema_: Literal["cc.claim_envelope.governance_state.v1"] = Field(
        default=GOVERNANCE_STATE_SCHEMA_VERSION,
        alias="schema",
    )
    verdict: GovernanceVerdict = "not_evaluated"
    verifier_schema: str | None = None
    evaluated_at: str | None = None
    freshness_status: ClaimFreshnessStatus = "not_evaluated"
    required_human_review: bool = True
    reasons: tuple[str, ...] = Field(default_factory=tuple)
    support_summary: EnvelopeSupportSummary

    @field_validator("reasons")
    @classmethod
    def _reasons_are_clean(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _dedupe(value)

    @model_validator(mode="after")
    def _governance_consistency(self) -> GovernanceState:
        if self.verdict == "fail" and not self.reasons:
            raise ValueError("failed governance projection must include reasons")
        if self.required_human_review and self.verdict == "pass" and self.support_summary.review_edges:
            # This is conservative but not fatal. A pass can still require human review
            # in external release processes, but review edges must remain visible.
            return self
        return self


class ClaimEnvelope(EnvelopeModel):
    """Typed intermediate representation for one evidence-bound claim."""

    schema_: Literal["cc.claim_envelope.v1"] = Field(
        default=CLAIM_ENVELOPE_SCHEMA_VERSION,
        alias="schema",
    )
    identity: ClaimIdentity
    proposition: ClaimProposition
    boundary: BoundaryEnvelope
    support_graph: SupportGraph
    governance_state: GovernanceState

    @model_validator(mode="after")
    def _envelope_is_boundary_honest(self) -> ClaimEnvelope:
        if self.proposition.allowed_claim_level != "diagnostic" and not self.boundary.non_claims:
            raise ValueError("non-diagnostic envelopes require explicit non-claims")

        fragment_ids = {fragment.fragment_id for fragment in self.proposition.fragments}
        for edge in self.support_graph.support_edges:
            if edge.target_claim_fragment not in fragment_ids:
                raise ValueError(
                    f"support edge targets missing claim fragment {edge.target_claim_fragment!r}"
                )

        return self

    def to_canonical_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-native payload for serialization or hashing."""

        return self.model_dump(mode="json", by_alias=True)

    def to_canonical_json(self) -> str:
        """Serialize the envelope with stable sorted-key compact JSON."""

        return canonical_json_bytes(self.to_canonical_dict()).decode("utf-8")

    def canonical_hash(self) -> str:
        """Return SHA-256 over the envelope's canonical JSON bytes."""

        return hashlib.sha256(canonical_json_bytes(self.to_canonical_dict())).hexdigest()


def compile_claim_envelope(
    report: Mapping[str, Any],
    *,
    governance_audit: Mapping[str, Any] | BaseModel | None = None,
    subject_ref: str | None = None,
    evaluated_at: str | None = None,
) -> ClaimEnvelope:
    """Compile a ``cc.report.v0.3.1`` report into a typed claim envelope.

    The compiler is intentionally conservative. Report-bound artifacts are
    preserved as references, but only evidence roles with explicit semantics
    receive support edges, and those edges target narrow claim fragments.

    This function does not create a future ``cc.claims.Claim`` and does not
    assign lifecycle state.
    """

    audit = _audit_mapping(governance_audit)
    report_id = _required_str(report, "report_id")
    source_schema = _required_str(report, "schema_version")
    if source_schema != SCHEMA_VERSION:
        raise ValueError(f"report schema_version must be {SCHEMA_VERSION}")

    claim = _required_mapping(report, "claim")
    claim_statement = _required_str(claim, "statement")
    allowed_claim_level = _claim_level(_required_str(claim, "allowed_claim_level"))

    subject = subject_ref or f"{source_schema}:{report_id}"
    report_hash = _receipt_hash(report)

    report_evidence_refs = _compile_evidence_refs(report, audit=audit, subject_ref=subject)
    measurement_ref = _measurement_ref(report, subject_ref=subject)
    calibration_ref = _calibration_ref(report, subject_ref=subject)

    all_evidence_refs = (measurement_ref, calibration_ref, *report_evidence_refs)

    scenario_refs = tuple(ref for ref in all_evidence_refs if ref.role == "extremal_scenario")
    decay_refs = tuple(ref for ref in all_evidence_refs if ref.role == "claim_decay")
    review_refs = tuple(
        ref for ref in all_evidence_refs if ref.role in {"human_review", "human_review_note"}
    )
    evidence_refs = tuple(
        ref
        for ref in all_evidence_refs
        if ref.role
        not in {"claim_decay", "extremal_scenario", "human_review", "human_review_note"}
    )

    receipt_refs = (
        ArtifactRef(
            artifact_id="receipt:report",
            subject_ref=subject,
            role="receipt_integrity",
            sha256=report_hash,
            schema="cc.report.receipt.v0.3.1",
            created_at=_optional_str(report.get("created_at")),
            evaluated_at=_audit_str(audit, ("evaluated_at",)) or evaluated_at,
            reason="Canonical report receipt binds the report bytes and named evidence hashes.",
        ),
    )

    graph = SupportGraph(
        evidence_refs=evidence_refs,
        scenario_refs=scenario_refs,
        decay_refs=decay_refs,
        receipt_refs=receipt_refs,
        review_refs=review_refs,
        support_edges=_compile_support_edges(
            evidence_refs=all_evidence_refs,
            receipt_refs=receipt_refs,
            audit=audit,
        ),
    )
    summary = EnvelopeSupportSummary.from_graph(graph)

    boundary = BoundaryEnvelope(
        scope=_boundary_scope(
            report,
            subject_ref=subject,
            evidence_artifact_ids=tuple(ref.artifact_id for ref in all_evidence_refs),
        ),
        assumptions=_str_tuple(report.get("assumptions")),
        non_claims=_dedupe(
            [
                *_str_tuple(claim.get("non_claims")),
                *_str_tuple(_audit_get(audit, ("non_claims",))),
                *_mandatory_edge_non_claims(graph),
            ]
        ),
        defeaters=_compile_defeaters(audit),
        invalidation_conditions=_compile_invalidation_conditions(audit),
        review_requirements=_compile_review_requirements(audit, allowed_claim_level),
    )

    return ClaimEnvelope(
        identity=ClaimIdentity(
            artifact_id=f"claim-envelope:{_slug(report_id)}",
            claim_id=_slug(report_id),
            subject_ref=subject,
            source_report_id=report_id,
            source_report_schema=source_schema,
            source_report_hash=report_hash,
            created_at=_optional_str(report.get("created_at")),
            evaluated_at=_audit_str(audit, ("evaluated_at",)) or evaluated_at,
        ),
        proposition=ClaimProposition(
            statement=claim_statement,
            allowed_claim_level=allowed_claim_level,
            fragments=_claim_fragments(report, graph),
        ),
        boundary=boundary,
        support_graph=graph,
        governance_state=GovernanceState(
            verdict=_governance_verdict(_audit_str(audit, ("verdict",))),
            verifier_schema=_audit_str(audit, ("schema",)) or _audit_str(audit, ("schema_",)),
            evaluated_at=_audit_str(audit, ("evaluated_at",)) or evaluated_at,
            freshness_status=_freshness_status(_audit_str(audit, ("decay", "status"))),
            required_human_review=_audit_bool(audit, ("required_human_review",), True),
            reasons=_str_tuple(_audit_get(audit, ("reasons",))),
            support_summary=summary,
        ),
    )


def claim_envelope_to_canonical_json(envelope: ClaimEnvelope) -> str:
    """Return canonical JSON for a claim envelope."""

    return envelope.to_canonical_json()


def claim_envelope_sha256(envelope: ClaimEnvelope) -> str:
    """Return SHA-256 over a claim envelope's canonical JSON representation."""

    return envelope.canonical_hash()


def _validate_role_support_edge(source: ArtifactRef, edge: SupportEdge) -> None:
    role = source.role
    target = edge.target_claim_fragment.lower()

    permissions = support_permissions_for(role)
    if classify_role(role) != "unknown":
        role_definition = get_role_definition(role)
        for unsupported in role_definition.does_not_support:
            if any(marker in target for marker in unsupported.target_markers):
                raise ValueError(f"{role} evidence cannot support {edge.target_claim_fragment!r}")

    if role in {"human_review", "human_review_note"}:
        _validate_human_review_edge(source, edge)

    if classify_role(role) == "unknown" and (
        edge.relation not in {"requires_review", "qualifies"} or edge.strength != "weak"
    ):
        raise ValueError("unknown evidence roles cannot strengthen a claim")

    if classify_role(role) == "unknown":
        return

    if not _edge_matches_support_permission(edge, permissions):
        raise ValueError(f"{role} evidence may only provide ontology-permitted support")


def _edge_matches_support_permission(
    edge: SupportEdge,
    permissions: tuple[SupportPermission, ...],
) -> bool:
    for permission in permissions:
        if edge.relation != permission.relation or edge.strength != permission.strength:
            continue
        if not permission.target_claim_fragments:
            return True
        if any(
            _target_fragment_matches(edge.target_claim_fragment, item)
            for item in permission.target_claim_fragments
        ):
            return True
    return False


def _target_fragment_matches(target: str, allowed: str) -> bool:
    if allowed.endswith("*"):
        return target.startswith(allowed[:-1])
    return target == allowed


def _validate_human_review_edge(source: ArtifactRef, edge: SupportEdge) -> None:
    if edge.strength == "confirmatory":
        raise ValueError("human review cannot provide confirmatory statistical support")

    target = edge.target_claim_fragment
    if target.startswith("evidence."):
        reviewed = {str(item) for item in source.metadata.get("reviewed_artifact_ids", [])}
        reviewed.update(str(item) for item in source.metadata.get("reviewed_artifact_hashes", []))
        evidence_id = target.removeprefix("evidence.")
        if evidence_id not in reviewed:
            raise ValueError("human review cannot support evidence it did not review")


def _compile_support_edges(
    *,
    evidence_refs: Sequence[ArtifactRef],
    receipt_refs: Sequence[ArtifactRef],
    audit: Mapping[str, Any] | None,
) -> tuple[SupportEdge, ...]:
    edges: list[SupportEdge] = [
        SupportEdge(
            source_artifact_id=receipt_refs[0].artifact_id,
            target_claim_fragment="claim.integrity.receipt",
            relation="integrity_binds",
            strength="integrity_only",
            non_claims=_NON_PROOF_RECEIPT,
            rationale="Receipt hashing binds report and artifact bytes, but only as integrity.",
        )
    ]

    for ref in evidence_refs:
        if ref.role == "measurement_evidence":
            edges.append(
                SupportEdge(
                    source_artifact_id=ref.artifact_id,
                    target_claim_fragment="claim.statistical_interval",
                    relation="bounds",
                    strength="diagnostic",
                    non_claims=_NON_PROOF_MEASUREMENT,
                    rationale="Report measurement bounds the named metric under stated scope.",
                )
            )
        elif ref.role == "calibration_evidence":
            edges.append(
                SupportEdge(
                    source_artifact_id=ref.artifact_id,
                    target_claim_fragment="claim.operating_point",
                    relation="qualifies",
                    strength="diagnostic",
                    non_claims=_NON_PROOF_MEASUREMENT,
                    rationale="Calibration qualifies the operating point for the report claim.",
                )
            )
        elif ref.role == "claim_decay":
            edges.extend(_decay_edges(ref, audit))
        elif ref.role == "extremal_scenario":
            edges.append(
                SupportEdge(
                    source_artifact_id=ref.artifact_id,
                    target_claim_fragment="claim.endpoint_feasibility",
                    relation="bounds",
                    strength="diagnostic",
                    non_claims=_NON_PROOF_SCENARIO,
                    rationale="Scenario artifacts witness feasible endpoint or counterfactual worlds.",
                )
            )
        elif ref.role in {"human_review", "human_review_note"}:
            edges.append(
                SupportEdge(
                    source_artifact_id=ref.artifact_id,
                    target_claim_fragment="claim.review_authorization",
                    relation="qualifies",
                    strength="weak",
                    non_claims=_NON_PROOF_REVIEW,
                    rationale="Review artifacts can authorize scoped use only.",
                )
            )
        elif ref.role == "exploratory_redteam":
            edges.append(
                SupportEdge(
                    source_artifact_id=ref.artifact_id,
                    target_claim_fragment="claim.redteam_hypothesis",
                    relation="exploratory_suggests",
                    strength="weak",
                    non_claims=(
                        "Exploratory red-team evidence is not a confirmatory certificate.",
                    ),
                    rationale="Exploratory red-team artifacts generate hypotheses and review pressure.",
                )
            )
        elif ref.role == "confirmatory_protocol":
            edges.append(_confirmatory_protocol_edge(ref))
        elif ref.role == "confirmatory_failure_matrix":
            edges.append(
                SupportEdge(
                    source_artifact_id=ref.artifact_id,
                    target_claim_fragment="claim.failure_matrix",
                    relation="confirmatory_tests",
                    strength="confirmatory",
                    non_claims=_NON_PROOF_CONFIRMATORY,
                    rationale="Confirmatory failure matrices test a predeclared or held-out matrix.",
                )
            )
        elif ref.role == "fitted_empirical_scenario":
            edges.append(
                SupportEdge(
                    source_artifact_id=ref.artifact_id,
                    target_claim_fragment="claim.fitted_scenario",
                    relation="bounds",
                    strength="diagnostic",
                    non_claims=(
                        "A fitted empirical scenario does not prove the fitted dependence model is true.",
                    ),
                    rationale="Fitted scenario artifacts describe scoped empirical dependence structure.",
                )
            )
        elif classify_role(ref.role) == "unknown":
            edges.append(
                SupportEdge(
                    source_artifact_id=ref.artifact_id,
                    target_claim_fragment="claim.evidence_role_resolution",
                    relation="requires_review",
                    strength="weak",
                    non_claims=_UNKNOWN_ROLE_NON_CLAIM,
                    rationale="Unknown role is preserved without strengthening the claim.",
                )
            )

    return tuple(edges)


def _confirmatory_protocol_edge(ref: ArtifactRef) -> SupportEdge:
    if ref.status == "invalid":
        return SupportEdge(
            source_artifact_id=ref.artifact_id,
            target_claim_fragment="claim.confirmatory_boundary",
            relation="invalidates",
            strength="diagnostic",
            non_claims=(
                "Failed confirmatory protocol evidence invalidates confirmatory support; "
                "it does not certify deployment safety.",
            ),
            rationale="Confirmatory protocol validation failed under governance checks.",
        )

    if ref.reason is not None and "requires review" in ref.reason:
        return SupportEdge(
            source_artifact_id=ref.artifact_id,
            target_claim_fragment="claim.confirmatory_boundary",
            relation="requires_review",
            strength="weak",
            non_claims=(
                "Confirmatory protocol evidence requiring review is not a confirmatory "
                "certificate until resolved.",
            ),
            rationale="Confirmatory protocol validation raised a review trigger.",
        )

    return SupportEdge(
        source_artifact_id=ref.artifact_id,
        target_claim_fragment="claim.confirmatory_boundary",
        relation="confirmatory_tests",
        strength="confirmatory",
        non_claims=(
            "Confirmatory protocol evidence remains scoped to its pre-registered "
            "plan and run separation; it does not certify deployment safety.",
        ),
        rationale="Confirmatory protocol artifacts bind a pre-registered plan to a separate run.",
    )


def _decay_edges(ref: ArtifactRef, audit: Mapping[str, Any] | None) -> tuple[SupportEdge, ...]:
    status = _freshness_status(_audit_str(audit, ("decay", "status")))
    edges = [
        SupportEdge(
            source_artifact_id=ref.artifact_id,
            target_claim_fragment="claim.staleness",
            relation="qualifies",
            strength="diagnostic",
            non_claims=_NON_PROOF_DECAY,
            rationale="Decay policy qualifies freshness and review timing.",
        )
    ]

    if status == "expired":
        edges.append(
            SupportEdge(
                source_artifact_id=ref.artifact_id,
                target_claim_fragment="claim.freshness",
                relation="invalidates",
                strength="diagnostic",
                non_claims=_NON_PROOF_DECAY,
                rationale="Expired decay state invalidates freshness, not historical integrity.",
            )
        )
    elif status == "degraded":
        edges.append(
            SupportEdge(
                source_artifact_id=ref.artifact_id,
                target_claim_fragment="claim.review_pressure",
                relation="requires_review",
                strength="diagnostic",
                non_claims=_NON_PROOF_DECAY,
                rationale="Degraded decay state requires conservative review.",
            )
        )

    return tuple(edges)


def _compile_evidence_refs(
    report: Mapping[str, Any],
    *,
    audit: Mapping[str, Any] | None,
    subject_ref: str,
) -> tuple[ArtifactRef, ...]:
    entries = _report_evidence_entries(report)
    audits = _artifact_audits_by_path_role(audit)
    refs: list[ArtifactRef] = []

    for idx, entry in enumerate(entries):
        role = _optional_str(entry.get("role")) or "artifact"
        path = _optional_str(entry.get("path"))
        artifact_id = _entry_artifact_id(role, idx, path)
        audit_item = audits.get((path or "", role), {})

        refs.append(
            ArtifactRef(
                artifact_id=artifact_id,
                subject_ref=subject_ref,
                role=role,
                path=path,
                sha256=_optional_str(entry.get("sha256")),
                bytes=_optional_int(entry.get("bytes")),
                status=_audit_str(audit_item, ("status",)) or None,
                reason=_audit_str(audit_item, ("reason",)) or None,
            )
        )

    return tuple(refs)


def _measurement_ref(report: Mapping[str, Any], *, subject_ref: str) -> ArtifactRef:
    measurement = _required_mapping(report, "measurement")
    digest = hashlib.sha256(canonical_json_bytes(measurement)).hexdigest()
    return ArtifactRef(
        artifact_id="report:measurement",
        subject_ref=subject_ref,
        role="measurement_evidence",
        sha256=digest,
        schema="cc.report.measurement.v0.3.1",
        created_at=_optional_str(report.get("created_at")),
        metadata={"source_path": "report.measurement"},
    )


def _calibration_ref(report: Mapping[str, Any], *, subject_ref: str) -> ArtifactRef:
    calibration = _required_mapping(report, "calibration")
    digest = hashlib.sha256(canonical_json_bytes(calibration)).hexdigest()
    return ArtifactRef(
        artifact_id="report:calibration",
        subject_ref=subject_ref,
        role="calibration_evidence",
        sha256=digest,
        schema="cc.report.calibration.v0.3.1",
        created_at=_optional_str(report.get("created_at")),
        metadata={"source_path": "report.calibration"},
    )


def _boundary_scope(
    report: Mapping[str, Any],
    *,
    subject_ref: str,
    evidence_artifact_ids: tuple[str, ...],
) -> BoundaryScope:
    measurement = _required_mapping(report, "measurement")
    interval = measurement.get("interval")
    measurement_interval = None

    if isinstance(interval, Mapping):
        lower = interval.get("lower")
        upper = interval.get("upper")
        if lower is None or upper is None:
            raise ValueError("measurement.interval must include lower and upper")
        measurement_interval = MeasurementInterval(
            lower=float(lower),
            upper=float(upper),
            point_estimate=_optional_float(measurement.get("point_estimate")),
            confidence_level=_optional_float(measurement.get("confidence_level")),
            delta=_optional_float(measurement.get("delta")),
        )

    run = _mapping_or_empty(report.get("run"))
    calibration = _mapping_or_empty(report.get("calibration"))
    sample_sizes_raw = measurement.get("sample_sizes")
    sample_sizes = {
        str(key): int(value)
        for key, value in (
            sample_sizes_raw.items() if isinstance(sample_sizes_raw, Mapping) else []
        )
    }

    return BoundaryScope(
        subject_ref=subject_ref,
        report_id=_required_str(report, "report_id"),
        report_schema_version=_required_str(report, "schema_version"),
        run_id=_optional_str(run.get("run_id")),
        metric_family=_optional_str(measurement.get("metric_family")),
        interval_method=_optional_str(measurement.get("interval_method")),
        measurement_interval=measurement_interval,
        sample_sizes=sample_sizes,
        calibration_status=_optional_str(calibration.get("status")),
        config_path=_optional_str(run.get("config_path")),
        evidence_artifact_ids=evidence_artifact_ids,
    )


def _claim_fragments(
    report: Mapping[str, Any],
    graph: SupportGraph,
) -> tuple[ClaimFragment, ...]:
    claim = _required_mapping(report, "claim")
    fragment_ids = {
        "claim.statement": "Main report claim.",
        "claim.statistical_interval": "Measured interval and point estimate under report scope.",
        "claim.operating_point": "Calibration and operating-point scope.",
        "claim.integrity.receipt": "Report and evidence-byte integrity.",
        "claim.evidence_role_resolution": "Evidence role resolution and unknown-role review.",
    }

    if graph.decay_refs:
        fragment_ids["claim.staleness"] = "Claim freshness, staleness, and review pressure."
        fragment_ids["claim.freshness"] = "Verification-time freshness state."
        fragment_ids["claim.review_pressure"] = "Review pressure caused by decay or staleness."

    if graph.scenario_refs:
        fragment_ids["claim.endpoint_feasibility"] = (
            "Endpoint feasibility and counterfactual dependence bounds."
        )

    if graph.review_refs:
        fragment_ids["claim.review_authorization"] = "Scoped human review authorization."

    for edge in graph.support_edges:
        if edge.target_claim_fragment == "claim.redteam_hypothesis":
            fragment_ids["claim.redteam_hypothesis"] = (
                "Exploratory red-team hypothesis or triage signal."
            )
        elif edge.target_claim_fragment == "claim.failure_matrix":
            fragment_ids["claim.failure_matrix"] = "Confirmatory failure-matrix evidence."
        elif edge.target_claim_fragment == "claim.fitted_scenario":
            fragment_ids["claim.fitted_scenario"] = "Fitted empirical scenario evidence."
        elif edge.target_claim_fragment == "claim.confirmatory_boundary":
            fragment_ids["claim.confirmatory_boundary"] = "Exploratory/confirmatory firewall."

    return (
        *(
            ClaimFragment(
                fragment_id=fragment_id,
                text=text,
                fragment_type=fragment_id.split(".")[1],
            )
            for fragment_id, text in sorted(fragment_ids.items())
        ),
        ClaimFragment(
            fragment_id="claim.original_text",
            text=_required_str(claim, "statement"),
            fragment_type="statement",
        ),
    )


def _compile_defeaters(audit: Mapping[str, Any] | None) -> tuple[BoundaryDefeater, ...]:
    defeaters: list[BoundaryDefeater] = []

    for idx, item in enumerate(
        _str_tuple(_audit_get(audit, ("boundary", "mandatory_non_claims_missing")))
    ):
        defeaters.append(
            BoundaryDefeater(
                defeater_id=f"defeater:missing-non-claim:{idx}",
                description=f"Mandatory non-claim boundary is missing: {item}.",
                source_ref="governance.boundary",
                status="active",
            )
        )

    for idx, reason in enumerate(_str_tuple(_audit_get(audit, ("reasons",)))):
        if "Unknown evidence role" in reason:
            defeaters.append(
                BoundaryDefeater(
                    defeater_id=f"defeater:unknown-role:{idx}",
                    description=reason,
                    source_ref="governance.evidence_artifacts",
                    status="unresolved",
                )
            )

    return tuple(defeaters)


def _compile_invalidation_conditions(
    audit: Mapping[str, Any] | None,
) -> tuple[InvalidationCondition, ...]:
    conditions: list[InvalidationCondition] = []
    if not audit:
        return tuple(conditions)

    receipt_verified = _audit_get(audit, ("receipt", "report_hash_verified"))
    if receipt_verified is False:
        conditions.append(
            InvalidationCondition(
                condition_id="invalidates:receipt-hash",
                description="Canonical report hash did not verify.",
                trigger="receipt_hash_mismatch",
                source_ref="receipt",
                severity="invalidates",
            )
        )

    if _freshness_status(_audit_str(audit, ("decay", "status"))) == "expired":
        conditions.append(
            InvalidationCondition(
                condition_id="expires:claim-decay",
                description="Claim decay evaluation is expired.",
                trigger="claim_decay_expired",
                source_ref="decay",
                severity="expires",
            )
        )

    if int(_audit_get(audit, ("scenarios", "infeasible_count")) or 0) > 0:
        conditions.append(
            InvalidationCondition(
                condition_id="invalidates:scenario-infeasible",
                description="One or more extremal scenario artifacts are infeasible.",
                trigger="scenario_infeasible",
                source_ref="scenarios",
                severity="invalidates",
            )
        )

    for idx, reason in enumerate(_str_tuple(_audit_get(audit, ("reasons",)))):
        lowered = reason.lower()
        if (
            "sha-256" in reason
            or "unreadable" in lowered
            or "exploratory evidence leaked" in reason
        ):
            conditions.append(
                InvalidationCondition(
                    condition_id=f"invalidates:governance-reason:{idx}",
                    description=reason,
                    trigger="governance_failure_reason",
                    source_ref="governance.reasons",
                    severity="invalidates",
                )
            )

    return tuple(conditions)


def _compile_review_requirements(
    audit: Mapping[str, Any] | None,
    allowed_claim_level: ClaimLevel,
) -> tuple[ReviewRequirement, ...]:
    requirements: list[ReviewRequirement] = []
    required = _audit_bool(
        audit, ("required_human_review",), allowed_claim_level == "release_claim"
    )
    status: Literal["required", "not_required"] = "required" if required else "not_required"
    reason = (
        "Governance verifier requires human review."
        if required
        else "Governance verifier did not require human review under its narrow rules."
    )

    requirements.append(
        ReviewRequirement(
            requirement_id="review:governance-verdict",
            reason=reason,
            source_ref="governance.required_human_review",
            required=required,
            status=status,
        )
    )

    if allowed_claim_level == "release_claim":
        requirements.append(
            ReviewRequirement(
                requirement_id="review:release-claim",
                reason="release_claim requires accountable external human review.",
                source_ref="claim.allowed_claim_level",
                required=True,
                status="required",
            )
        )

    for idx, reason_item in enumerate(_str_tuple(_audit_get(audit, ("reasons",)))):
        if "review" in reason_item.lower() or "Unknown evidence role" in reason_item:
            requirements.append(
                ReviewRequirement(
                    requirement_id=f"review:reason:{idx}",
                    reason=reason_item,
                    source_ref="governance.reasons",
                    required=True,
                    status="required",
                )
            )

    return tuple(requirements)


def _report_evidence_entries(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    evidence = _required_mapping(report, "evidence")
    entries: list[Mapping[str, Any]] = []
    raw_artifacts = evidence.get("artifacts", [])

    if isinstance(raw_artifacts, Sequence) and not isinstance(raw_artifacts, (str, bytes)):
        entries.extend(item for item in raw_artifacts if isinstance(item, Mapping))

    for key, role in (("audit_log", "audit_log"), ("figure_manifest", "figure_manifest")):
        item = evidence.get(key)
        if isinstance(item, Mapping):
            entry = dict(item)
            entry["role"] = str(entry.get("role") or role)
            entries.append(entry)

    return entries


def _artifact_audits_by_path_role(
    audit: Mapping[str, Any] | None,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    items = _audit_get(audit, ("evidence_artifacts",))

    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return out

    for item in items:
        if not isinstance(item, Mapping):
            continue
        path = _optional_str(item.get("path")) or ""
        role = _optional_str(item.get("role")) or ""
        out[(path, role)] = item

    return out


def _entry_artifact_id(role: str, idx: int, path: str | None) -> str:
    stem = _slug(path or role)
    return f"evidence:{_slug(role)}:{idx:04d}:{stem}"


def _mandatory_edge_non_claims(graph: SupportGraph) -> tuple[str, ...]:
    return _dedupe(
        non_claim
        for edge in graph.support_edges
        for non_claim in edge.non_claims
    )


def _audit_mapping(value: Mapping[str, Any] | BaseModel | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    return value


def _receipt_hash(report: Mapping[str, Any]) -> str | None:
    receipt = report.get("receipt")
    if not isinstance(receipt, Mapping):
        return None
    value = receipt.get("canonical_hash")
    return value if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) else None


def _required_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _required_str(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _audit_get(audit: Mapping[str, Any] | None, path: tuple[str, ...]) -> Any:
    value: Any = audit
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _audit_str(audit: Mapping[str, Any] | None, path: tuple[str, ...]) -> str | None:
    return _optional_str(_audit_get(audit, path))


def _audit_bool(audit: Mapping[str, Any] | None, path: tuple[str, ...], default: bool) -> bool:
    value = _audit_get(audit, path)
    return value if isinstance(value, bool) else default


def _str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


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


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.:-]+", "-", value.strip()).strip("-")
    return slug or "unknown"


def _claim_level(value: str) -> ClaimLevel:
    if value not in CLAIM_LEVELS:
        allowed = ", ".join(CLAIM_LEVELS)
        raise ValueError(f"allowed_claim_level must be one of: {allowed}")
    if value in RESERVED_LIFECYCLE_STATE_NAMES:
        raise ValueError("allowed_claim_level must not reuse a lifecycle state name")
    return cast(ClaimLevel, value)


def _governance_verdict(value: str | None) -> GovernanceVerdict:
    if value is None:
        return "not_evaluated"
    if value not in _ALLOWED_GOVERNANCE_VERDICTS:
        raise ValueError(f"governance verdict {value!r} is not recognized")
    return cast(GovernanceVerdict, value)


def _freshness_status(value: str | None) -> ClaimFreshnessStatus:
    if value is None:
        return "not_evaluated"
    if value not in _ALLOWED_FRESHNESS_STATUSES:
        raise ValueError(f"claim freshness status {value!r} is not recognized")
    return cast(ClaimFreshnessStatus, value)


__all__ = [
    "BOUNDARY_ENVELOPE_SCHEMA_VERSION",
    "CLAIM_ENVELOPE_SCHEMA_VERSION",
    "GOVERNANCE_STATE_SCHEMA_VERSION",
    "SUPPORT_GRAPH_SCHEMA_VERSION",
    "SUPPORT_SUMMARY_SCHEMA_VERSION",
    "ArtifactRef",
    "BoundaryDefeater",
    "BoundaryEnvelope",
    "BoundaryScope",
    "ClaimEnvelope",
    "ClaimFragment",
    "ClaimFreshnessStatus",
    "ClaimIdentity",
    "ClaimProposition",
    "DefeaterStatus",
    "EnvelopeSupportSummary",
    "GovernanceState",
    "GovernanceVerdict",
    "InvalidationCondition",
    "InvalidationSeverity",
    "MeasurementInterval",
    "ReviewRequirement",
    "ReviewStatus",
    "SupportEdge",
    "SupportGraph",
    "SupportRelation",
    "SupportStrength",
    "claim_envelope_sha256",
    "claim_envelope_to_canonical_json",
    "compile_claim_envelope",
]
"""Epistemic permission compiler for evidence-bound claims.

This module compiles evidence-role ontology records into explicit permission
edges, forbidden claim families, mandatory non-claims, and review requirements.

Semantic boundary
-----------------
This is not a claim lifecycle engine.

It does not decide whether a claim is true, safe, certified, deployed, approved,
or globally valid. It only answers:

    Given the declared evidence role, what is this artifact permitted to support,
    what must it never support, what non-claims must remain attached, and when
    review is required?

Evidence in. Permissions out.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cc.evidence.role_ontology import (
    RESERVED_LIFECYCLE_STATE_NAMES,
    MandatoryNonClaim,
    ReviewRule,
    SupportRelation,
    SupportStrength,
    UnsupportedClaim,
    get_role_definition,
    is_known_role,
    mandatory_non_claims_for,
    support_permissions_for,
)

PERMISSION_COMPILATION_SCHEMA_VERSION: Literal["cc.epistemic_permission_compilation.v1"] = (
    "cc.epistemic_permission_compilation.v1"
)

FORBIDDEN_CLAIM_EXPLANATION_SCHEMA_VERSION: Literal["cc.forbidden_claim_explanation.v1"] = (
    "cc.forbidden_claim_explanation.v1"
)

STRENGTH_ORDER: Mapping[SupportStrength, int] = {
    "integrity_only": 0,
    "weak": 1,
    "diagnostic": 2,
    "confirmatory": 3,
}

# Deterministic overclaim lexicon. This must remain explicit and reviewable.
# Do not replace this with fuzzy matching or LLM calls.
OVERCLAIM_MARKERS: Mapping[str, tuple[str, ...]] = {
    "DEPLOYMENT_SAFETY_CERTIFICATION": (
        "deployment safe",
        "deployment safety",
        "safe for deployment",
        "certified safe",
        "certifies safety",
        "safety certification",
        "production safe",
        "validated for production",
        "production ready",
        "ready for deployment",
        "safe enough for customers",
        "operationally trustworthy",
        "real world validated",
        "real-world validated",
        "system safe",
        "system is safe",
        "makes the system safe",
        "model safe",
        "model is safe",
        "makes the model safe",
    ),
    "ALIGNMENT_PROOF": (
        "alignment proof",
        "proves alignment",
        "aligned model",
        "model is aligned",
        "guaranteed aligned",
    ),
    "EMPIRICAL_GLOBAL_TRUTH": (
        "proves truth",
        "globally true",
        "empirical truth",
        "proven true",
        "ground truth proof",
        "conclusive proof",
    ),
    "FUTURE_PERFORMANCE_GUARANTEE": (
        "will remain safe",
        "future performance",
        "guaranteed performance",
        "guarantees robustness",
        "will not fail",
        "cannot fail",
    ),
    "INDEPENDENCE_AS_DEFAULT": (
        "assume independence",
        "independent by default",
        "failures are independent",
        "multiply the failure rates",
        "product proves",
        "product baseline proves",
    ),
}

# Canonical expansion lets ontology claim families catch equivalent surface
# phrasing without forcing every role definition to duplicate every phrase.
CANONICAL_FORBIDDEN_CLAIM_MARKERS: Mapping[str, tuple[str, ...]] = {
    "deployment_safety": OVERCLAIM_MARKERS["DEPLOYMENT_SAFETY_CERTIFICATION"],
    "release_claim": (
        "release claim",
        "release ready",
        "ready for release",
        "production ready",
        "validated for production",
        "safe for deployment",
    ),
    "statistical_validity": (
        "statistical validity",
        "statistically valid",
        "valid inference",
        "validates the statistics",
        "proves statistical validity",
    ),
    "confirmatory_evidence": (
        "confirmatory evidence",
        "confirmatory certificate",
        "confirmation certificate",
        "confirmatory proof",
        "confirmatory ci",
    ),
    "external_validity": (
        "external validity",
        "generalizes",
        "generalization",
        "representative",
        "representativeness",
        "world representative",
    ),
    "likelihood": (
        "likely",
        "likelihood",
        "probable",
        "probability in deployment",
    ),
    "model_truth_claim": (
        "model truth",
        "true model",
        "fitted model is true",
        "proves the model",
    ),
}


class PermissionCompilerModel(BaseModel):
    """Strict frozen base model for permission compiler records."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class EvidenceArtifactRef(PermissionCompilerModel):
    """Minimal compiler input for one evidence artifact.

    The compiler intentionally requires only identity, role, and optional payload.
    Payload validation belongs mostly to role_ontology.validate_role_payload; this
    compiler consumes the role semantics and emits permission surfaces.
    """

    artifact_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    payload: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("artifact_id", "role")
    @classmethod
    def _clean_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must be non-empty")
        return stripped


class PermissionEdge(PermissionCompilerModel):
    """Explicit evidence-to-claim-fragment permission edge.

    A permission edge is not a truth claim. It is a typed support allowance
    emitted from the role ontology.
    """

    source_artifact_id: str = Field(min_length=1)
    source_role: str = Field(min_length=1)
    relation: SupportRelation
    strength: SupportStrength
    target_claim_fragments: tuple[str, ...] = Field(default_factory=tuple)
    support_scope: str = Field(min_length=1)
    mandatory_non_claim_ids: tuple[str, ...] = Field(default_factory=tuple)
    forbidden_claim_types: tuple[str, ...] = Field(default_factory=tuple)
    review_required: bool = False
    invalidation_triggers: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator(
        "target_claim_fragments",
        "mandatory_non_claim_ids",
        "forbidden_claim_types",
        "invalidation_triggers",
    )
    @classmethod
    def _clean_tuple(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in value)
        if any(not item for item in cleaned):
            raise ValueError("tuples must contain non-empty strings")
        return cleaned

    @model_validator(mode="after")
    def _edge_strength_respects_integrity_boundary(self) -> PermissionEdge:
        if self.relation == "integrity_binds" and self.strength != "integrity_only":
            raise ValueError("integrity_binds permission edges must be integrity_only")
        return self


class ForbiddenClaimFamily(PermissionCompilerModel):
    """A claim family forbidden by one or more compiled evidence roles."""

    claim_type: str = Field(min_length=1)
    target_markers: tuple[str, ...] = Field(default_factory=tuple)
    reason: str = Field(min_length=1)
    implicated_roles: tuple[str, ...] = Field(default_factory=tuple)
    source_artifact_ids: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("target_markers", "implicated_roles", "source_artifact_ids")
    @classmethod
    def _clean_tuple(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in value)
        if any(not item for item in cleaned):
            raise ValueError("tuples must contain non-empty strings")
        return cleaned


class PermissionReviewRequirement(PermissionCompilerModel):
    """Review obligation emitted by role semantics or compiler degradation."""

    requirement_id: str = Field(min_length=1)
    source_artifact_id: str = Field(min_length=1)
    source_role: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class PermissionCompilationResult(PermissionCompilerModel):
    """Complete compiler receipt for evidence-to-permission compilation.

    This model is the stable output boundary of the epistemic permission compiler.

    It is not a claim lifecycle object, not a governance PASS, not a safety
    certificate, and not a truth judgment. It is a structured record of what
    support edges, forbidden claim families, non-claim boundaries, and review
    obligations were emitted from the evidence-role ontology.

    Field naming rules:
    - schema_version avoids Pydantic's BaseModel.schema shadowing warning.
    - review_requirements is plural and stable because the compiler may emit
      zero, one, or many review obligations.
    - PermissionReviewRequirement is the compiler-specific review type; it must
      not collide with claim_envelope.ReviewRequirement.
    """

    schema_version: Literal["cc.epistemic_permission_compilation.v1"] = (
        PERMISSION_COMPILATION_SCHEMA_VERSION
    )
    permissions: tuple[PermissionEdge, ...] = Field(default_factory=tuple)
    forbidden_claims: tuple[ForbiddenClaimFamily, ...] = Field(default_factory=tuple)
    mandatory_non_claims: tuple[MandatoryNonClaim, ...] = Field(default_factory=tuple)
    review_requirements: tuple[PermissionReviewRequirement, ...] = Field(default_factory=tuple)
    unknown_roles: tuple[str, ...] = Field(default_factory=tuple)
    rejected_roles: tuple[str, ...] = Field(default_factory=tuple)
    non_claims: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _result_is_internally_consistent(self) -> PermissionCompilationResult:
        permission_non_claims = {
            non_claim_id
            for permission in self.permissions
            for non_claim_id in permission.mandatory_non_claim_ids
        }
        declared_non_claims = {non_claim.non_claim_id for non_claim in self.mandatory_non_claims}
        projected_non_claims = set(self.non_claims)

        if not permission_non_claims <= declared_non_claims:
            missing = sorted(permission_non_claims - declared_non_claims)
            raise ValueError(
                "permission edges reference mandatory non-claims absent from "
                f"mandatory_non_claims: {missing}"
            )

        if projected_non_claims != declared_non_claims:
            raise ValueError("non_claims must exactly project mandatory_non_claims.non_claim_id")

        unknown_or_rejected = set(self.unknown_roles) | set(self.rejected_roles)
        support_roles = {permission.source_role for permission in self.permissions}
        leaked_roles = sorted(unknown_or_rejected & support_roles)
        if leaked_roles:
            raise ValueError(
                f"unknown or rejected roles must not emit permission edges: {leaked_roles}"
            )

        return self


class ForbiddenClaimExplanation(PermissionCompilerModel):
    """Deterministic explanation for why a proposed claim is forbidden."""

    schema_version: Literal["cc.forbidden_claim_explanation.v1"] = (
        FORBIDDEN_CLAIM_EXPLANATION_SCHEMA_VERSION
    )
    claim_text: str = Field(min_length=1)
    forbidden: bool
    matched_claim_types: tuple[str, ...] = Field(default_factory=tuple)
    matched_markers: tuple[str, ...] = Field(default_factory=tuple)
    reasons: tuple[str, ...] = Field(default_factory=tuple)
    implicated_roles: tuple[str, ...] = Field(default_factory=tuple)
    required_non_claims: tuple[str, ...] = Field(default_factory=tuple)


def compile_epistemic_permissions(
    artifacts: Sequence[EvidenceArtifactRef | Mapping[str, Any]],
) -> PermissionCompilationResult:
    """Compile evidence artifacts into explicit epistemic permission surfaces.

    Conservative rules:
    - unknown roles are preserved but emit no support edges;
    - reserved lifecycle state names are rejected as roles;
    - integrity roles can emit only integrity_only strength;
    - role mandatory non-claims are always preserved;
    - review rules and invalidation triggers are surfaced but not interpreted as
      lifecycle transitions.
    """

    normalized = tuple(_coerce_artifact(item) for item in artifacts)

    permissions: list[PermissionEdge] = []
    forbidden_claims: list[ForbiddenClaimFamily] = []
    mandatory_non_claims: list[MandatoryNonClaim] = []
    review_requirements: list[PermissionReviewRequirement] = []
    unknown_roles: list[str] = []
    rejected_roles: list[str] = []

    for artifact in normalized:
        role = artifact.role

        if role in RESERVED_LIFECYCLE_STATE_NAMES:
            rejected_roles.append(role)
            review_requirements.append(
                PermissionReviewRequirement(
                    requirement_id="REJECTED_LIFECYCLE_STATE_USED_AS_ROLE",
                    source_artifact_id=artifact.artifact_id,
                    source_role=role,
                    reason=(
                        "Lifecycle state names must not be used as evidence roles. "
                        "This belongs to future cc.claims lifecycle semantics."
                    ),
                )
            )
            continue

        definition = get_role_definition(role)

        if not is_known_role(role):
            unknown_roles.append(role)
            review_requirements.append(
                PermissionReviewRequirement(
                    requirement_id="REQUIRES_ROLE_RESOLUTION_REVIEW",
                    source_artifact_id=artifact.artifact_id,
                    source_role=role,
                    reason="Unknown evidence role is preserved but cannot support claims.",
                )
            )
            forbidden_claims.append(
                ForbiddenClaimFamily(
                    claim_type="any_claim_strengthening",
                    target_markers=("claim.", "statistical", "deployment", "release"),
                    reason="Unknown roles cannot strengthen claims.",
                    implicated_roles=(role,),
                    source_artifact_ids=(artifact.artifact_id,),
                )
            )
            continue

        role_non_claims = mandatory_non_claims_for(role)
        role_forbidden_types = tuple(item.claim_type for item in definition.does_not_support)
        role_review_required = bool(definition.review_rules)
        role_invalidation_triggers = tuple(
            trigger.trigger_id for trigger in definition.invalidation_triggers
        )

        mandatory_non_claims.extend(role_non_claims)

        for unsupported in definition.does_not_support:
            forbidden_claims.append(
                _forbidden_family_from_unsupported(
                    unsupported=unsupported,
                    role=role,
                    artifact_id=artifact.artifact_id,
                )
            )

        for review_rule in definition.review_rules:
            review_requirements.append(
                _review_requirement_from_rule(
                    rule=review_rule,
                    artifact_id=artifact.artifact_id,
                    role=role,
                )
            )

        for permission in support_permissions_for(role):
            strength = permission.strength
            if definition.semantic_class == "integrity":
                strength = "integrity_only"

            permissions.append(
                PermissionEdge(
                    source_artifact_id=artifact.artifact_id,
                    source_role=role,
                    relation=permission.relation,
                    strength=strength,
                    target_claim_fragments=permission.target_claim_fragments,
                    support_scope=permission.support_scope,
                    mandatory_non_claim_ids=tuple(
                        non_claim.non_claim_id for non_claim in role_non_claims
                    ),
                    forbidden_claim_types=role_forbidden_types,
                    review_required=role_review_required,
                    invalidation_triggers=role_invalidation_triggers,
                )
            )

    deduped_non_claims = _dedupe_non_claims(mandatory_non_claims)
    return PermissionCompilationResult(
        permissions=tuple(permissions),
        forbidden_claims=_dedupe_forbidden_claims(forbidden_claims),
        mandatory_non_claims=deduped_non_claims,
        review_requirements=tuple(review_requirements),
        unknown_roles=tuple(dict.fromkeys(unknown_roles)),
        rejected_roles=tuple(dict.fromkeys(rejected_roles)),
        non_claims=tuple(non_claim.non_claim_id for non_claim in deduped_non_claims),
    )


def explain_forbidden_claim(
    claim_text: str,
    compilation: PermissionCompilationResult,
) -> ForbiddenClaimExplanation:
    """Explain whether a proposed claim text breaches compiled boundaries.

    This is deterministic marker matching only. No LLMs. No fuzzy semantics.

    Matching uses:
    - hardcoded overclaim markers;
    - ontology-projected forbidden claim markers;
    - canonical deterministic expansions for known forbidden claim families.
    """

    stripped = claim_text.strip()
    if not stripped:
        raise ValueError("claim_text must be non-empty")

    normalized = _normalize_text(stripped)

    matched_types: list[str] = []
    matched_markers: list[str] = []
    reasons: list[str] = []
    implicated_roles: list[str] = []

    for claim_type, markers in OVERCLAIM_MARKERS.items():
        hits = _matching_markers(normalized, markers)
        if hits:
            matched_types.append(claim_type)
            matched_markers.extend(hits)
            reasons.append(f"Claim matches hardcoded overclaim family {claim_type}.")

    for forbidden in compilation.forbidden_claims:
        markers = _expanded_markers_for_forbidden_claim(forbidden)
        hits = _matching_markers(normalized, markers)
        if hits:
            matched_types.append(forbidden.claim_type)
            matched_markers.extend(hits)
            reasons.append(forbidden.reason)
            implicated_roles.extend(forbidden.implicated_roles)

    required_non_claims = tuple(
        dict.fromkeys(non_claim.non_claim_id for non_claim in compilation.mandatory_non_claims)
    )

    return ForbiddenClaimExplanation(
        claim_text=stripped,
        forbidden=bool(matched_types),
        matched_claim_types=tuple(dict.fromkeys(matched_types)),
        matched_markers=tuple(dict.fromkeys(matched_markers)),
        reasons=tuple(dict.fromkeys(reasons)),
        implicated_roles=tuple(dict.fromkeys(implicated_roles)),
        required_non_claims=required_non_claims,
    )


def strongest_permission_strength(
    permissions: Sequence[PermissionEdge],
) -> SupportStrength | None:
    """Return the strongest emitted permission strength, if any."""

    if not permissions:
        return None
    return max(permissions, key=lambda item: STRENGTH_ORDER[item.strength]).strength


def _coerce_artifact(item: EvidenceArtifactRef | Mapping[str, Any]) -> EvidenceArtifactRef:
    if isinstance(item, EvidenceArtifactRef):
        return item

    artifact_id = (
        item.get("artifact_id")
        or item.get("id")
        or item.get("source_artifact_id")
        or item.get("evidence_id")
    )
    role = item.get("role") or item.get("evidence_role") or item.get("source_role")
    payload = item.get("payload") or {}

    return EvidenceArtifactRef(
        artifact_id=str(artifact_id or ""),
        role=str(role or ""),
        payload=payload,
    )


def _forbidden_family_from_unsupported(
    unsupported: UnsupportedClaim,
    role: str,
    artifact_id: str,
) -> ForbiddenClaimFamily:
    return ForbiddenClaimFamily(
        claim_type=unsupported.claim_type,
        target_markers=unsupported.target_markers,
        reason=unsupported.reason,
        implicated_roles=(role,),
        source_artifact_ids=(artifact_id,),
    )


def _review_requirement_from_rule(
    rule: ReviewRule,
    artifact_id: str,
    role: str,
) -> PermissionReviewRequirement:
    return PermissionReviewRequirement(
        requirement_id=rule.rule_id,
        source_artifact_id=artifact_id,
        source_role=role,
        reason=rule.description,
    )


def _dedupe_non_claims(
    non_claims: Sequence[MandatoryNonClaim],
) -> tuple[MandatoryNonClaim, ...]:
    out: dict[str, MandatoryNonClaim] = {}
    for non_claim in non_claims:
        out.setdefault(non_claim.non_claim_id, non_claim)
    return tuple(out.values())


def _dedupe_forbidden_claims(
    forbidden_claims: Sequence[ForbiddenClaimFamily],
) -> tuple[ForbiddenClaimFamily, ...]:
    grouped: dict[tuple[str, str], ForbiddenClaimFamily] = {}

    for forbidden in forbidden_claims:
        key = (forbidden.claim_type, forbidden.reason)
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = forbidden
            continue

        grouped[key] = ForbiddenClaimFamily(
            claim_type=existing.claim_type,
            target_markers=tuple(dict.fromkeys(existing.target_markers + forbidden.target_markers)),
            reason=existing.reason,
            implicated_roles=tuple(
                dict.fromkeys(existing.implicated_roles + forbidden.implicated_roles)
            ),
            source_artifact_ids=tuple(
                dict.fromkeys(existing.source_artifact_ids + forbidden.source_artifact_ids)
            ),
        )

    return tuple(grouped.values())


def _expanded_markers_for_forbidden_claim(
    forbidden: ForbiddenClaimFamily,
) -> tuple[str, ...]:
    canonical = CANONICAL_FORBIDDEN_CLAIM_MARKERS.get(forbidden.claim_type, ())
    return tuple(dict.fromkeys(forbidden.target_markers + canonical))


def _matching_markers(
    normalized_claim_text: str,
    markers: Sequence[str],
) -> tuple[str, ...]:
    hits: list[str] = []
    for marker in markers:
        normalized_marker = _normalize_text(marker)
        if normalized_marker and normalized_marker in normalized_claim_text:
            hits.append(marker)
    return tuple(dict.fromkeys(hits))


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


__all__ = [
    "CANONICAL_FORBIDDEN_CLAIM_MARKERS",
    "FORBIDDEN_CLAIM_EXPLANATION_SCHEMA_VERSION",
    "OVERCLAIM_MARKERS",
    "PERMISSION_COMPILATION_SCHEMA_VERSION",
    "STRENGTH_ORDER",
    "EvidenceArtifactRef",
    "ForbiddenClaimExplanation",
    "ForbiddenClaimFamily",
    "PermissionCompilationResult",
    "PermissionEdge",
    "PermissionReviewRequirement",
    "compile_epistemic_permissions",
    "explain_forbidden_claim",
    "strongest_permission_strength",
]

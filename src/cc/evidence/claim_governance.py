"""Read-only governance verifier for evidence-bound CC claims.

This module verifies the internal consistency of a CC report/evidence package.

Important semantic boundary
---------------------------
`GovernanceVerdict` is a verifier outcome, not a claim lifecycle state.

A PASS verdict means:

    The evidence-bound claim package is internally consistent under this
    verifier's declared rules at verification time.

A PASS verdict does not mean:

- the AI system is safe;
- the claim is globally true;
- the claim is deployment-valid;
- the package is production-certified;
- the package is compliance-certified;
- the claim should transition to a future `cc.claims.ClaimState.SUPPORTED`
  without explicit lifecycle rules.

This module owns:

- report/evidence package verification;
- artifact hash and byte-count checks;
- evidence-role ontology checks;
- semantic payload validation;
- claim-decay freshness projection;
- extremal-scenario validation;
- confirmatory-protocol validation;
- mandatory non-claim checks;
- conservative pass/needs_review/fail verdicts.

This module does not own:

- claim lifecycle transitions;
- claim compiler logic;
- assumption registry;
- challenge calculus;
- dashboard display state;
- deployment/compliance certification.

Future `cc.claims` may consume this audit as evidence, but must not treat this
audit as a lifecycle state machine.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cc.evidence.claim_envelope import EnvelopeSupportSummary, SupportGraph, compile_claim_envelope
from cc.evidence.confirmatory_protocol import (
    ProtocolAuditStatus,
    verify_confirmatory_protocol_artifact,
)
from cc.evidence.decay import ClaimDecayRecord, DecayState, VersionWatchSet, evaluate_claim_decay
from cc.evidence.extremal_scenario import ExtremalScenario, ScenarioKind
from cc.evidence.role_ontology import (
    RESERVED_LIFECYCLE_STATE_NAMES,
    get_role_definition,
    is_known_role,
    mandatory_non_claims_for,
    role_support_matrix,
    roles_requiring_semantic_payload_validation,
    validate_role_payload,
)
from cc.reporting.canonical import sha256_canonical, strict_json_loads
from cc.reporting.report import (
    ALLOWED_CLAIM_LEVELS,
    CLAIM_LEVELS,
    SCHEMA_VERSION,
    ClaimLevel,
    sha256_file,
)

CLAIM_GOVERNANCE_AUDIT_SCHEMA: Literal["cc/claim-governance-audit.v1"] = (
    "cc/claim-governance-audit.v1"
)

ROLE_SUPPORT_MATRIX: dict[str, dict[str, list[str]]] = role_support_matrix()

GOVERNANCE_PASS_CAVEAT = (
    "A governance PASS means internal consistency under verifier rules only; "
    "it does not prove safety, deployment validity, production readiness, or compliance."
)

RECEIPT_NON_CLAIM = (
    "Receipt verification checks artifact integrity only; it does not prove statistical validity, "
    "deployment safety, production readiness, or compliance."
)

CLAIM_LEVEL_NON_CLAIM = (
    "Claim levels are report maturity/support labels, not claim lifecycle states."
)

_DEFAULT_GOVERNANCE_NON_CLAIMS = (
    GOVERNANCE_PASS_CAVEAT,
    RECEIPT_NON_CLAIM,
    CLAIM_LEVEL_NON_CLAIM,
)

_SEMANTIC_PAYLOAD_ROLES = roles_requiring_semantic_payload_validation()

_EXPLORATORY_INTERVAL_FIELDS = frozenset(
    {
        "adaptive_search_ci",
        "certificate_ci",
        "exploratory_certificate_ci",
        "exploratory_ci",
        "non_confirmatory_ci",
    }
)

_REQUIRED_ROLES_BY_CLAIM_LEVEL: dict[ClaimLevel, tuple[str, ...]] = {
    "diagnostic": (),
    "bounded_empirical": ("claim_decay", "extremal_scenario"),
    "reproducible_run": ("claim_decay",),
    "release_claim": ("claim_decay", "extremal_scenario"),
}

if set(CLAIM_LEVELS) & RESERVED_LIFECYCLE_STATE_NAMES:  # pragma: no cover
    raise RuntimeError("claim maturity labels must remain disjoint from lifecycle state names")


class GovernanceVerdict(str, Enum):
    """Conservative verifier verdict for an evidence-bound claim package.

    This is not a lifecycle state.
    """

    PASS = "pass"
    NEEDS_REVIEW = "needs_review"
    FAIL = "fail"


class EvidenceRoleStatus(str, Enum):
    """Verification status for one report-bound evidence artifact."""

    PRESENT = "present"
    MISSING = "missing"
    INVALID = "invalid"
    UNREADABLE = "unreadable"
    UNKNOWN_ROLE = "unknown_role"


class ClaimFreshnessStatus(str, Enum):
    """Verification-time claim freshness projection.

    This is not the same thing as a future `cc.claims.ClaimState`.
    """

    NOT_EVALUATED = "not_evaluated"
    FRESH = "fresh"
    DEGRADED = "degraded"
    EXPIRED = "expired"


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        populate_by_name=True,
        validate_assignment=True,
    )


class EvidenceArtifactAudit(_StrictModel):
    path: str
    role: str
    sha256_expected: str
    sha256_actual: str | None
    bytes_expected: int | None
    bytes_actual: int | None
    status: EvidenceRoleStatus
    reason: str

    @field_validator("role")
    @classmethod
    def _role_is_not_lifecycle_state(cls, value: str) -> str:
        role = value.strip()
        if role in RESERVED_LIFECYCLE_STATE_NAMES:
            raise ValueError("evidence artifact role must not reuse a lifecycle state name")
        return role


class DecayAudit(_StrictModel):
    present: bool
    status: ClaimFreshnessStatus
    reason: str
    evaluated_at: str | None
    trigger_summary: list[str] = Field(default_factory=list)
    non_claims: list[str] = Field(default_factory=list)


class ScenarioAudit(_StrictModel):
    present: bool
    scenario_count: int
    scenario_ids: list[str]
    kinds: list[str]
    infeasible_count: int
    excluded_evidence_fields: list[dict[str, Any]]
    non_claims: list[str] = Field(default_factory=list)


class ConfirmatoryProtocolGovernanceAudit(_StrictModel):
    present: bool
    artifact_count: int
    protocol_ids: list[str]
    run_ids: list[str]
    failed_count: int
    review_count: int
    failed_reasons: list[str] = Field(default_factory=list)
    review_reasons: list[str] = Field(default_factory=list)
    audits: list[dict[str, Any]] = Field(default_factory=list)
    non_claims: list[str] = Field(default_factory=list)


class BoundaryAudit(_StrictModel):
    claim_non_claim_count: int
    artifact_non_claim_count: int
    mandatory_non_claims_missing: list[str]
    unresolved_defeaters_or_gaps: list[str] = Field(default_factory=list)


class ReceiptAudit(_StrictModel):
    report_hash_verified: bool | None
    artifact_hashes_verified: bool
    canonical_hash: str | None
    reason: str


class ClaimGovernanceAudit(_StrictModel):
    schema_: Literal["cc/claim-governance-audit.v1"] = Field(
        default=CLAIM_GOVERNANCE_AUDIT_SCHEMA,
        alias="schema",
    )
    report_id: str
    evaluated_at: str
    verdict: GovernanceVerdict
    allowed_claim_level: ClaimLevel | Literal["unknown"]
    claim_statement: str
    receipt: ReceiptAudit
    evidence_artifacts: list[EvidenceArtifactAudit]
    decay: DecayAudit
    scenarios: ScenarioAudit
    confirmatory_protocols: ConfirmatoryProtocolGovernanceAudit
    boundary: BoundaryAudit
    required_human_review: bool
    reasons: list[str]
    non_claims: list[str]
    envelope_support: EnvelopeSupportSummary

    @field_validator("non_claims", mode="before")
    @classmethod
    def _coerce_non_claims(cls, value: Any) -> list[str]:
        """Coerce non-claims without silently repairing provided boundaries.

        If non_claims is omitted, use the governance defaults. If the caller
        provides non_claims, preserve them and let the after-validator reject
        missing mandatory governance caveats. This prevents boundary tampering
        from being silently repaired during validation.
        """

        if value is None:
            return list(_DEFAULT_GOVERNANCE_NON_CLAIMS)
        return list(_coerce_string_tuple(value))

    @model_validator(mode="after")
    def _verdict_is_boundary_honest(self) -> ClaimGovernanceAudit:
        if self.verdict is not GovernanceVerdict.PASS and not self.reasons:
            raise ValueError("non-PASS governance audits must include at least one reason")

        if self.verdict is GovernanceVerdict.PASS and self.required_human_review:
            raise ValueError("PASS verdict cannot require unresolved human review")

        if (
            self.allowed_claim_level != "unknown"
            and self.allowed_claim_level in RESERVED_LIFECYCLE_STATE_NAMES
        ):
            raise ValueError("allowed_claim_level must not be a lifecycle state")

        if GOVERNANCE_PASS_CAVEAT not in self.non_claims:
            raise ValueError("governance audit must preserve the PASS caveat non-claim")
        if RECEIPT_NON_CLAIM not in self.non_claims:
            raise ValueError("governance audit must preserve the receipt non-claim")
        if CLAIM_LEVEL_NON_CLAIM not in self.non_claims:
            raise ValueError("governance audit must preserve the claim-level non-claim")

        return self


def verify_claim_governance(
    report_path: Path,
    *,
    now: datetime | None = None,
    base_dir: Path | None = None,
    strict_unknown_roles: bool = False,
) -> ClaimGovernanceAudit:
    """Verify a CC report's evidence-bound claim package without mutating artifacts.

    A PASS verdict means the evidence-bound claim package is internally
    consistent under this verifier's rules at verification time.

    It does not mean the AI system is safe in deployment.
    """

    report_file = Path(report_path)
    evaluation_time = _verification_time(now)
    if evaluation_time is None:
        return _failure_audit(
            report_id=report_file.stem or "<unknown>",
            evaluated_at=_utc_iso(datetime.now(timezone.utc)),
            reason="Verification time must be timezone-aware.",
        )

    evaluated_at = _utc_iso(evaluation_time)

    try:
        # A repeated key here would change which report is verified (F-07).
        report = strict_json_loads(report_file.read_text(encoding="utf-8"))
    except Exception as exc:
        return _failure_audit(
            report_id=report_file.stem or "<unreadable>",
            evaluated_at=evaluated_at,
            reason=f"Report cannot be read as JSON: {exc}",
        )

    if not isinstance(report, dict):
        return _failure_audit(
            report_id=report_file.stem or "<unusable>",
            evaluated_at=evaluated_at,
            reason="Report is structurally unusable: top-level JSON value is not an object.",
        )

    shape_errors = _basic_report_shape_errors(report)
    if shape_errors:
        return _failure_audit(
            report_id=str(report.get("report_id") or report_file.stem or "<unusable>"),
            evaluated_at=evaluated_at,
            reason="Report is structurally unusable: " + "; ".join(shape_errors),
            allowed_claim_level=_unknown_or_claim_level(
                _nested_str(report, ("claim", "allowed_claim_level"), "unknown")
            ),
            claim_statement=_nested_str(report, ("claim", "statement"), ""),
        )

    report_id = str(report["report_id"])
    claim = cast(Mapping[str, Any], report["claim"])
    claim_statement = str(claim["statement"])
    allowed_claim_level = _claim_level(str(claim["allowed_claim_level"]))
    root_dir = Path(base_dir) if base_dir is not None else report_file.parent

    reasons: list[str] = []
    fail_reasons: list[str] = []
    review_reasons: list[str] = []
    unresolved: list[str] = []

    raw_entries = _report_evidence_entries(report)
    evidence_audits: list[EvidenceArtifactAudit] = []
    semantic_payloads: list[tuple[Mapping[str, Any], EvidenceArtifactAudit]] = []

    for raw_entry in raw_entries:
        artifact_audit = _audit_artifact(raw_entry, base_dir=root_dir)
        evidence_audits.append(artifact_audit)

        role = artifact_audit.role

        if artifact_audit.status in {EvidenceRoleStatus.UNREADABLE, EvidenceRoleStatus.INVALID}:
            fail_reasons.append(artifact_audit.reason)
            continue

        if not is_known_role(role):
            artifact_audit.status = EvidenceRoleStatus.UNKNOWN_ROLE
            artifact_audit.reason = (
                "Hash verified, but evidence role is not in the governance verifier ontology."
            )
            message = f"Unknown evidence role {role!r} on {artifact_audit.path}."
            if strict_unknown_roles:
                fail_reasons.append(message)
            else:
                review_reasons.append(message)
                unresolved.append(message)
            continue

        role_definition = get_role_definition(role)
        if allowed_claim_level not in role_definition.allowed_claim_levels:
            message = (
                f"Evidence role {role!r} cannot support claim level "
                f"{allowed_claim_level!r} under the role ontology."
            )
            review_reasons.append(message)
            unresolved.append(message)

        if role in _SEMANTIC_PAYLOAD_ROLES:
            payload = _read_semantic_payload(artifact_audit, root_dir)
            if payload is None:
                if _invalid_role_payload_is_failure(role) or strict_unknown_roles:
                    fail_reasons.append(artifact_audit.reason)
                else:
                    review_reasons.append(artifact_audit.reason)
                    unresolved.append(artifact_audit.reason)
                continue

            payload_validation = validate_role_payload(role, payload)
            if not payload_validation.valid:
                artifact_audit.status = EvidenceRoleStatus.INVALID
                artifact_audit.reason = (
                    f"Evidence role ontology rejected {role!r} payload: "
                    + "; ".join(payload_validation.errors)
                )
                if _invalid_role_payload_is_failure(role) or strict_unknown_roles:
                    fail_reasons.append(artifact_audit.reason)
                else:
                    review_reasons.append(artifact_audit.reason)
                    unresolved.append(artifact_audit.reason)
                continue

            if payload_validation.review_required:
                for warning in payload_validation.warnings:
                    review_reasons.append(warning)
                    unresolved.append(warning)

            semantic_payloads.append((payload, artifact_audit))

    receipt = _audit_receipt(report, evidence_audits)
    if receipt.report_hash_verified is False:
        fail_reasons.append(receipt.reason)

    roles_present = {
        audit.role for audit in evidence_audits if audit.status is not EvidenceRoleStatus.MISSING
    }

    missing_roles = [
        role
        for role in _REQUIRED_ROLES_BY_CLAIM_LEVEL.get(allowed_claim_level, ())
        if role not in roles_present
    ]
    for role in missing_roles:
        message = (
            f"Claim level {allowed_claim_level!r} conservatively requires evidence role {role!r}."
        )
        review_reasons.append(message)
        unresolved.append(message)

    decay_audit = _audit_decay_artifacts(
        semantic_payloads,
        report=report,
        evaluated_at=evaluated_at,
        now=evaluation_time,
    )
    scenario_audit = _audit_scenario_artifacts(semantic_payloads)
    confirmatory_protocol_audit = _audit_confirmatory_protocol_artifacts(
        semantic_payloads,
        allowed_claim_level=allowed_claim_level,
    )

    if decay_audit.present:
        if decay_audit.status is ClaimFreshnessStatus.EXPIRED:
            fail_reasons.append(f"Claim decay status is expired: {decay_audit.reason}")
        elif decay_audit.status is ClaimFreshnessStatus.DEGRADED:
            message = f"Claim decay status is degraded: {decay_audit.reason}"
            review_reasons.append(message)
            unresolved.append(message)

    if scenario_audit.infeasible_count:
        fail_reasons.append("One or more extremal_scenario artifacts are infeasible.")

    if scenario_audit.excluded_evidence_fields:
        message = "Endpoint or fitted scenarios contain excluded evidence fields requiring review."
        review_reasons.append(message)
        unresolved.append(message)

    if _has_fitted_without_confirmation(semantic_payloads):
        message = "Fitted empirical scenario exists without confirmatory evidence."
        review_reasons.append(message)
        unresolved.append(message)

    for reason in confirmatory_protocol_audit.failed_reasons:
        fail_reasons.append(reason)

    for reason in confirmatory_protocol_audit.review_reasons:
        review_reasons.append(reason)
        unresolved.append(reason)

    if "confirmatory_failure_matrix" in roles_present and not confirmatory_protocol_audit.present:
        message = (
            "confirmatory_failure_matrix evidence requires a confirmatory_protocol artifact "
            "that binds both the plan and the run."
        )
        review_reasons.append(message)
        unresolved.append(message)

    leakage_paths = _exploratory_leakage_paths(report, semantic_payloads)
    if leakage_paths:
        fail_reasons.append(
            "Exploratory evidence leaked into confirmatory surface: "
            + ", ".join(sorted(leakage_paths))
        )

    non_claims = _collect_non_claims(
        report,
        decay_audit,
        scenario_audit,
        confirmatory_protocol_audit,
        semantic_payloads,
    )

    mandatory_missing = _mandatory_non_claims_missing(
        roles_present=roles_present,
        non_claims=non_claims,
        exploratory_boundary_present=bool(scenario_audit.excluded_evidence_fields or leakage_paths),
    )
    if mandatory_missing:
        message = "Mandatory non-claims are missing: " + ", ".join(mandatory_missing)
        review_reasons.append(message)
        unresolved.append(message)

    review_note_satisfied, review_note_reasons = _human_review_note_satisfies_review(
        report=report,
        evidence_audits=evidence_audits,
        semantic_payloads=semantic_payloads,
        allowed_claim_level=allowed_claim_level,
    )
    for reason in review_note_reasons:
        review_reasons.append(reason)
        unresolved.append(reason)

    if allowed_claim_level == "release_claim":
        if review_note_satisfied:
            reasons.append(
                "Hash-matched human_review_note satisfies the scoped release-claim review "
                "requirement without upgrading evidence strength."
            )
        else:
            message = "release_claim packages require accountable external human review."
            review_reasons.append(message)
            unresolved.append(message)

    artifact_non_claims = [
        *decay_audit.non_claims,
        *scenario_audit.non_claims,
        *confirmatory_protocol_audit.non_claims,
    ]
    boundary = BoundaryAudit(
        claim_non_claim_count=len(_claim_non_claims(report)),
        artifact_non_claim_count=len(artifact_non_claims),
        mandatory_non_claims_missing=mandatory_missing,
        unresolved_defeaters_or_gaps=_dedupe(unresolved),
    )

    reasons = _dedupe([*fail_reasons, *review_reasons, *reasons])
    if fail_reasons:
        verdict = GovernanceVerdict.FAIL
    elif review_reasons:
        verdict = GovernanceVerdict.NEEDS_REVIEW
    else:
        verdict = GovernanceVerdict.PASS

    governance_audit = ClaimGovernanceAudit(
        report_id=report_id,
        evaluated_at=evaluated_at,
        verdict=verdict,
        allowed_claim_level=allowed_claim_level,
        claim_statement=claim_statement,
        receipt=receipt,
        evidence_artifacts=evidence_audits,
        decay=decay_audit,
        scenarios=scenario_audit,
        confirmatory_protocols=confirmatory_protocol_audit,
        boundary=boundary,
        required_human_review=verdict is not GovernanceVerdict.PASS,
        reasons=reasons,
        non_claims=non_claims,
        envelope_support=_empty_envelope_support_summary(),
    )
    return _attach_envelope_support(governance_audit, report)


def _failure_audit(
    *,
    report_id: str,
    evaluated_at: str,
    reason: str,
    allowed_claim_level: ClaimLevel | Literal["unknown"] = "unknown",
    claim_statement: str = "",
) -> ClaimGovernanceAudit:
    return ClaimGovernanceAudit(
        report_id=report_id,
        evaluated_at=evaluated_at,
        verdict=GovernanceVerdict.FAIL,
        allowed_claim_level=allowed_claim_level,
        claim_statement=claim_statement,
        receipt=ReceiptAudit(
            report_hash_verified=None,
            artifact_hashes_verified=False,
            canonical_hash=None,
            reason="Report-level verification could not be completed. " + RECEIPT_NON_CLAIM,
        ),
        evidence_artifacts=[],
        decay=DecayAudit(
            present=False,
            status=ClaimFreshnessStatus.NOT_EVALUATED,
            reason="Report-level verification failed before decay artifacts could be evaluated.",
            evaluated_at=None,
        ),
        scenarios=ScenarioAudit(
            present=False,
            scenario_count=0,
            scenario_ids=[],
            kinds=[],
            infeasible_count=0,
            excluded_evidence_fields=[],
        ),
        confirmatory_protocols=ConfirmatoryProtocolGovernanceAudit(
            present=False,
            artifact_count=0,
            protocol_ids=[],
            run_ids=[],
            failed_count=0,
            review_count=0,
        ),
        boundary=BoundaryAudit(
            claim_non_claim_count=0,
            artifact_non_claim_count=0,
            mandatory_non_claims_missing=[],
        ),
        required_human_review=True,
        reasons=[reason],
        non_claims=_base_governance_non_claims(),
        envelope_support=_empty_envelope_support_summary(),
    )


def _empty_envelope_support_summary() -> EnvelopeSupportSummary:
    return EnvelopeSupportSummary.from_graph(SupportGraph())


def _attach_envelope_support(
    audit: ClaimGovernanceAudit,
    report: Mapping[str, Any],
) -> ClaimGovernanceAudit:
    try:
        envelope = compile_claim_envelope(report, governance_audit=audit)
    except Exception as exc:
        reasons = _dedupe([*audit.reasons, f"ClaimEnvelope compilation failed: {exc}"])
        return audit.model_copy(
            update={
                "verdict": GovernanceVerdict.FAIL,
                "required_human_review": True,
                "reasons": reasons,
            }
        )
    return audit.model_copy(update={"envelope_support": envelope.governance_state.support_summary})


def _verification_time(now: datetime | None) -> datetime | None:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        return None
    return value.astimezone(timezone.utc)


def _basic_report_shape_errors(report: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []

    for field in ("schema_version", "report_id", "claim", "evidence", "receipt"):
        if field not in report:
            errors.append(f"missing {field}")

    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")

    claim = report.get("claim")
    if not isinstance(claim, Mapping):
        errors.append("claim must be an object")
    else:
        for field in ("statement", "allowed_claim_level", "non_claims"):
            if field not in claim:
                errors.append(f"missing claim.{field}")

        level = claim.get("allowed_claim_level")
        if level not in ALLOWED_CLAIM_LEVELS:
            errors.append("claim.allowed_claim_level is not supported")
        if level in RESERVED_LIFECYCLE_STATE_NAMES:
            errors.append("claim.allowed_claim_level must not be a lifecycle state")
        if not isinstance(claim.get("non_claims"), list):
            errors.append("claim.non_claims must be a list")
        elif level != "diagnostic" and not claim.get("non_claims"):
            errors.append("non-diagnostic claim levels require explicit non_claims")

    evidence = report.get("evidence")
    if not isinstance(evidence, Mapping):
        errors.append("evidence must be an object")
    elif not isinstance(evidence.get("artifacts"), list):
        errors.append("evidence.artifacts must be a list")
    else:
        for idx, item in enumerate(evidence.get("artifacts", [])):
            if not isinstance(item, Mapping):
                errors.append(f"evidence.artifacts[{idx}] must be an object")
        for field in ("audit_log", "figure_manifest"):
            item = evidence.get(field)
            if item is not None and not isinstance(item, Mapping):
                errors.append(f"evidence.{field} must be an object or null")

    receipt = report.get("receipt")
    if not isinstance(receipt, Mapping):
        errors.append("receipt must be an object")

    return errors


def _report_evidence_entries(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    evidence = report["evidence"]
    assert isinstance(evidence, Mapping)

    entries: list[Mapping[str, Any]] = []
    for item in evidence.get("artifacts", []):
        if isinstance(item, Mapping):
            entries.append(item)

    for key, role in (("audit_log", "audit_log"), ("figure_manifest", "figure_manifest")):
        item = evidence.get(key)
        if isinstance(item, Mapping):
            entry = dict(item)
            entry["role"] = str(entry.get("role") or role)
            entries.append(entry)

    return entries


def _audit_artifact(raw_entry: Mapping[str, Any], *, base_dir: Path) -> EvidenceArtifactAudit:
    path = str(raw_entry.get("path") or "")
    role = str(raw_entry.get("role") or "")
    expected_hash = str(raw_entry.get("sha256") or "")
    expected_bytes = raw_entry.get("bytes")
    bytes_expected = int(expected_bytes) if isinstance(expected_bytes, int) else None

    if role in RESERVED_LIFECYCLE_STATE_NAMES:
        return EvidenceArtifactAudit(
            path=path,
            role=role,
            sha256_expected=expected_hash,
            sha256_actual=None,
            bytes_expected=bytes_expected,
            bytes_actual=None,
            status=EvidenceRoleStatus.INVALID,
            reason="Evidence artifact role reuses a reserved claim lifecycle state name.",
        )

    if not path or not role or not expected_hash or bytes_expected is None:
        return EvidenceArtifactAudit(
            path=path,
            role=role,
            sha256_expected=expected_hash,
            sha256_actual=None,
            bytes_expected=bytes_expected,
            bytes_actual=None,
            status=EvidenceRoleStatus.INVALID,
            reason="Evidence artifact entry is missing path, role, sha256, or bytes.",
        )

    try:
        resolved = _resolve_path(path, base_dir)
        actual_hash = sha256_file(resolved)
        actual_bytes = resolved.stat().st_size
    except Exception as exc:
        return EvidenceArtifactAudit(
            path=path,
            role=role,
            sha256_expected=expected_hash,
            sha256_actual=None,
            bytes_expected=bytes_expected,
            bytes_actual=None,
            status=EvidenceRoleStatus.UNREADABLE,
            reason=f"Evidence artifact is unreadable: {exc}",
        )

    if actual_hash != expected_hash:
        return EvidenceArtifactAudit(
            path=path,
            role=role,
            sha256_expected=expected_hash,
            sha256_actual=actual_hash,
            bytes_expected=bytes_expected,
            bytes_actual=actual_bytes,
            status=EvidenceRoleStatus.INVALID,
            reason="Evidence artifact SHA-256 does not match the report.",
        )

    if actual_bytes != bytes_expected:
        return EvidenceArtifactAudit(
            path=path,
            role=role,
            sha256_expected=expected_hash,
            sha256_actual=actual_hash,
            bytes_expected=bytes_expected,
            bytes_actual=actual_bytes,
            status=EvidenceRoleStatus.INVALID,
            reason="Evidence artifact byte count does not match the report.",
        )

    return EvidenceArtifactAudit(
        path=path,
        role=role,
        sha256_expected=expected_hash,
        sha256_actual=actual_hash,
        bytes_expected=bytes_expected,
        bytes_actual=actual_bytes,
        status=EvidenceRoleStatus.PRESENT,
        reason="Evidence artifact hash and byte count verified.",
    )


def _read_semantic_payload(
    artifact_audit: EvidenceArtifactAudit,
    root_dir: Path,
) -> Mapping[str, Any] | None:
    try:
        payload = strict_json_loads(_resolve_path(artifact_audit.path, root_dir).read_text())
    except Exception as exc:
        artifact_audit.status = EvidenceRoleStatus.UNREADABLE
        artifact_audit.reason = f"Artifact could not be parsed as JSON: {exc}"
        return None

    if not isinstance(payload, Mapping):
        artifact_audit.status = EvidenceRoleStatus.INVALID
        artifact_audit.reason = "Semantic evidence artifact JSON is not an object."
        return None

    return payload


def _audit_receipt(
    report: Mapping[str, Any],
    evidence_audits: Sequence[EvidenceArtifactAudit],
) -> ReceiptAudit:
    receipt = report.get("receipt")
    canonical_hash = receipt.get("canonical_hash") if isinstance(receipt, Mapping) else None
    canonicalization_method = (
        receipt.get("canonicalization_method") if isinstance(receipt, Mapping) else None
    )

    hash_verified: bool | None
    reason_parts: list[str] = []

    if isinstance(canonical_hash, str):
        try:
            if not isinstance(canonicalization_method, str):
                raise ValueError("receipt.canonicalization_method is absent or not a string")
            computed_hash = sha256_canonical(report, profile=canonicalization_method)
            hash_verified = computed_hash == canonical_hash
            if hash_verified:
                # Preserve the historical audit bytes for already-bound capsules.
                # The declared profile controls verification above, not prose drift.
                reason_parts.append("Canonical report hash verified.")
            else:
                reason_parts.append(
                    f"Canonical report hash mismatch: expected {canonical_hash}, "
                    f"computed {computed_hash} under profile {canonicalization_method!r}."
                )
        except Exception as exc:
            hash_verified = False
            reason_parts.append(f"Canonical report hash verification failed: {exc}")
    else:
        hash_verified = None
        reason_parts.append("Canonical report hash is absent or not a string.")

    artifact_hashes_verified = all(
        audit.sha256_actual == audit.sha256_expected and audit.bytes_actual == audit.bytes_expected
        for audit in evidence_audits
    )

    if artifact_hashes_verified:
        reason_parts.append("Evidence artifact hashes verified.")
    else:
        reason_parts.append("One or more evidence artifact hashes or byte counts did not verify.")

    reason_parts.append(RECEIPT_NON_CLAIM)

    return ReceiptAudit(
        report_hash_verified=hash_verified,
        artifact_hashes_verified=artifact_hashes_verified,
        canonical_hash=canonical_hash if isinstance(canonical_hash, str) else None,
        reason=" ".join(reason_parts),
    )


def _audit_decay_artifacts(
    semantic_payloads: Sequence[tuple[Mapping[str, Any], EvidenceArtifactAudit]],
    *,
    report: Mapping[str, Any],
    evaluated_at: str,
    now: datetime,
) -> DecayAudit:
    states: list[DecayState] = []
    reasons: list[str] = []
    trigger_summary: list[str] = []
    non_claims: list[str] = []
    observed_versions = _observed_versions_from_report(report)
    count = 0

    for payload, artifact in semantic_payloads:
        if artifact.role != "claim_decay":
            continue

        count += 1
        try:
            record = ClaimDecayRecord.model_validate(payload)
            version_changes = (
                record.version_watch_set.changes_from(observed_versions)
                if observed_versions is not None
                else ()
            )
            state = evaluate_claim_decay(record, now=now, current_versions=observed_versions)
        except Exception as exc:
            artifact.status = EvidenceRoleStatus.INVALID
            artifact.reason = f"claim_decay artifact is semantically invalid: {exc}"
            reasons.append(artifact.reason)
            continue

        states.append(state)
        non_claims.extend(record.non_claims)
        trigger_summary.extend(_decay_trigger_summary(record, now=now, state=state))
        trigger_summary.extend(f"version_changed:{item}" for item in version_changes)
        reasons.append(f"{artifact.path} evaluated as {state.value}.")

    if count == 0:
        return DecayAudit(
            present=False,
            status=ClaimFreshnessStatus.NOT_EVALUATED,
            reason="No claim_decay artifact was present.",
            evaluated_at=None,
            non_claims=[],
        )

    if not states:
        return DecayAudit(
            present=True,
            status=ClaimFreshnessStatus.NOT_EVALUATED,
            reason="; ".join(reasons) or "No valid claim_decay artifact could be evaluated.",
            evaluated_at=evaluated_at,
            non_claims=_dedupe(non_claims),
        )

    if DecayState.EXPIRED in states:
        status = ClaimFreshnessStatus.EXPIRED
    elif DecayState.DEGRADED in states:
        status = ClaimFreshnessStatus.DEGRADED
    else:
        status = ClaimFreshnessStatus.FRESH

    return DecayAudit(
        present=True,
        status=status,
        reason="; ".join(reasons),
        evaluated_at=evaluated_at,
        trigger_summary=_dedupe(trigger_summary),
        non_claims=_dedupe(non_claims),
    )


def _observed_versions_from_report(report: Mapping[str, Any]) -> VersionWatchSet | None:
    environment = report.get("environment")
    if not isinstance(environment, Mapping):
        return None

    package_snapshot = environment.get("package_snapshot")
    if not isinstance(package_snapshot, Mapping):
        return None

    dependency_versions = {
        str(key): str(value)
        for key, value in package_snapshot.items()
        if str(key).strip() and str(value).strip()
    }
    if not dependency_versions:
        return None

    return VersionWatchSet(dependency_versions=dependency_versions)


def _decay_trigger_summary(
    record: ClaimDecayRecord,
    *,
    now: datetime,
    state: DecayState,
) -> list[str]:
    age_days = (now - record.issued_at).total_seconds() / 86400.0
    summary = [f"age_days={age_days:.6g}", f"policy_id={record.policy.policy_id}"]

    if state is DecayState.DEGRADED:
        summary.append("age crossed a degraded threshold")
    if state is DecayState.EXPIRED:
        summary.append("age crossed an expiry threshold or a watched version changed")
    if not record.version_watch_set.is_empty:
        summary.append("version_watch_set present; live observed versions were not supplied")

    return summary


def _audit_scenario_artifacts(
    semantic_payloads: Sequence[tuple[Mapping[str, Any], EvidenceArtifactAudit]],
) -> ScenarioAudit:
    scenario_ids: list[str] = []
    kinds: list[str] = []
    infeasible_count = 0
    excluded_fields: list[dict[str, Any]] = []
    non_claims: list[str] = []
    count = 0

    for payload, artifact in semantic_payloads:
        if artifact.role != "extremal_scenario":
            continue

        count += 1
        try:
            scenario = _validate_extremal_scenario_json(payload)
        except Exception as exc:
            artifact.status = EvidenceRoleStatus.INVALID
            artifact.reason = f"extremal_scenario artifact is semantically invalid: {exc}"
            infeasible_count += 1
            continue

        scenario_ids.append(scenario.scenario_id)
        kinds.append(scenario.kind.value)
        non_claims.extend(scenario.non_claims)

        if not scenario.feasibility.is_feasible:
            infeasible_count += 1

        for item in scenario.excluded_evidence_fields:
            excluded_fields.append(item.model_dump(mode="json"))
            non_claims.append(item.reason)

    return ScenarioAudit(
        present=count > 0,
        scenario_count=count,
        scenario_ids=_dedupe(scenario_ids),
        kinds=kinds,
        infeasible_count=infeasible_count,
        excluded_evidence_fields=excluded_fields,
        non_claims=_dedupe(non_claims),
    )


def _audit_confirmatory_protocol_artifacts(
    semantic_payloads: Sequence[tuple[Mapping[str, Any], EvidenceArtifactAudit]],
    *,
    allowed_claim_level: ClaimLevel,
) -> ConfirmatoryProtocolGovernanceAudit:
    protocol_ids: list[str] = []
    run_ids: list[str] = []
    audits: list[dict[str, Any]] = []
    failed_reasons: list[str] = []
    review_reasons: list[str] = []
    non_claims: list[str] = []
    count = 0

    for payload, artifact in semantic_payloads:
        if artifact.role != "confirmatory_protocol":
            continue

        count += 1
        audit = verify_confirmatory_protocol_artifact(
            payload,
            claim_level=allowed_claim_level,
        )
        audits.append(audit.model_dump(mode="json"))
        protocol_ids.append(audit.protocol_id)
        run_ids.append(audit.run_id)
        non_claims.extend(audit.non_claims)

        if audit.status is ProtocolAuditStatus.FAIL:
            reason = f"confirmatory_protocol {artifact.path} failed validation: " + "; ".join(
                audit.reasons
            )
            artifact.status = EvidenceRoleStatus.INVALID
            artifact.reason = reason
            failed_reasons.append(reason)
        elif audit.status is ProtocolAuditStatus.NEEDS_REVIEW:
            reason = f"confirmatory_protocol {artifact.path} requires review: " + "; ".join(
                audit.reasons
            )
            artifact.reason = reason
            review_reasons.append(reason)

    return ConfirmatoryProtocolGovernanceAudit(
        present=count > 0,
        artifact_count=count,
        protocol_ids=_dedupe(protocol_ids),
        run_ids=_dedupe(run_ids),
        failed_count=len(failed_reasons),
        review_count=len(review_reasons),
        failed_reasons=_dedupe(failed_reasons),
        review_reasons=_dedupe(review_reasons),
        audits=audits,
        non_claims=_dedupe(non_claims),
    )


def _has_fitted_without_confirmation(
    semantic_payloads: Sequence[tuple[Mapping[str, Any], EvidenceArtifactAudit]],
) -> bool:
    for payload, artifact in semantic_payloads:
        if artifact.role != "extremal_scenario":
            continue

        try:
            scenario = _validate_extremal_scenario_json(payload)
        except Exception:
            continue

        searchable = " ".join(
            str(item)
            for item in (
                scenario.kind.value,
                scenario.source,
                scenario.source_kernel,
                scenario.endpoint,
            )
        ).lower()

        if "empirical" in searchable or "fitted" in searchable:
            if scenario.kind is ScenarioKind.CONFIRMATORY_FAILURE_MATRIX:
                if "confirmatory_ci" not in scenario.metadata:
                    return True
            elif "confirmatory_ci" not in scenario.metadata:
                return True

    return False


def _validate_extremal_scenario_json(payload: Mapping[str, Any]) -> ExtremalScenario:
    return ExtremalScenario.model_validate_json(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
    )


def _invalid_role_payload_is_failure(role: str) -> bool:
    return role in {
        "confirmatory_protocol",
        "extremal_scenario",
        "exploratory_redteam",
        "confirmatory_failure_matrix",
        "fitted_empirical_scenario",
    }


def _exploratory_leakage_paths(
    report: Mapping[str, Any],
    semantic_payloads: Sequence[tuple[Mapping[str, Any], EvidenceArtifactAudit]],
) -> set[str]:
    paths = set(_find_exploratory_keys(report))

    for payload, artifact in semantic_payloads:
        if artifact.role == "extremal_scenario":
            paths.update(f"{artifact.path}:{path}" for path in _find_exploratory_keys(payload))

    for entry in _report_evidence_entries(report):
        role = str(entry.get("role") or "")
        if role in _EXPLORATORY_INTERVAL_FIELDS:
            paths.add(f"evidence role {role!r}")

    return paths


def _find_exploratory_keys(value: Any, path: str = "$") -> set[str]:
    if _path_is_excluded_evidence_field(path):
        return set()

    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_str = str(key)
            child_path = f"{path}.{key_str}"
            if key_str in _EXPLORATORY_INTERVAL_FIELDS:
                found.add(child_path)
            found.update(_find_exploratory_keys(item, child_path))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            found.update(_find_exploratory_keys(item, f"{path}[{idx}]"))

    return found


def _path_is_excluded_evidence_field(path: str) -> bool:
    return ".excluded_evidence_fields[" in path or path.startswith("$.excluded_evidence_fields[")


def _collect_non_claims(
    report: Mapping[str, Any],
    decay: DecayAudit,
    scenarios: ScenarioAudit,
    confirmatory_protocols: ConfirmatoryProtocolGovernanceAudit,
    semantic_payloads: Sequence[tuple[Mapping[str, Any], EvidenceArtifactAudit]],
) -> list[str]:
    return _dedupe(
        [
            *_base_governance_non_claims(),
            *_claim_non_claims(report),
            *decay.non_claims,
            *scenarios.non_claims,
            *confirmatory_protocols.non_claims,
            *_semantic_payload_non_claims(semantic_payloads),
        ]
    )


def _base_governance_non_claims() -> list[str]:
    return list(_DEFAULT_GOVERNANCE_NON_CLAIMS)


def _semantic_payload_non_claims(
    semantic_payloads: Sequence[tuple[Mapping[str, Any], EvidenceArtifactAudit]],
) -> list[str]:
    values: list[str] = []
    for payload, _artifact in semantic_payloads:
        raw = payload.get("non_claims")
        if isinstance(raw, (list, tuple)):
            values.extend(str(item) for item in raw if str(item).strip())
    return values


_AUTHORIZING_REVIEW_DECISIONS = frozenset(
    {
        "approved",
        "approved_for_diagnostic_use",
        "approved_with_conditions",
    }
)


def _human_review_note_satisfies_review(
    *,
    report: Mapping[str, Any],
    evidence_audits: Sequence[EvidenceArtifactAudit],
    semantic_payloads: Sequence[tuple[Mapping[str, Any], EvidenceArtifactAudit]],
    allowed_claim_level: ClaimLevel,
) -> tuple[bool, list[str]]:
    """Return whether a review note covers the current bound evidence hash set."""

    notes = [
        (payload, artifact)
        for payload, artifact in semantic_payloads
        if artifact.role == "human_review_note" and artifact.status is EvidenceRoleStatus.PRESENT
    ]
    if not notes:
        return False, []

    bound_hashes = {
        audit.sha256_expected
        for audit in evidence_audits
        if audit.role != "human_review_note" and audit.sha256_expected
    }
    if not bound_hashes:
        return False, ["human_review_note cannot reduce review: no reviewed artifact set found."]

    report_hash = _report_receipt_hash(report)
    mismatch_reasons: list[str] = []

    for payload, artifact in notes:
        decision = str(payload.get("decision") or "").strip()
        if decision not in _AUTHORIZING_REVIEW_DECISIONS:
            mismatch_reasons.append(
                f"human_review_note {artifact.path} decision {decision!r} does not "
                "authorize review reduction."
            )
            continue

        reviewed_claim_level = str(payload.get("reviewed_claim_level") or "").strip()
        if reviewed_claim_level != allowed_claim_level:
            mismatch_reasons.append(
                f"human_review_note {artifact.path} reviewed claim level "
                f"{reviewed_claim_level!r}, not {allowed_claim_level!r}."
            )
            continue

        reviewed_report_hash = payload.get("reviewed_report_hash")
        if (
            isinstance(reviewed_report_hash, str)
            and report_hash is not None
            and reviewed_report_hash != report_hash
        ):
            mismatch_reasons.append(
                f"human_review_note {artifact.path} does not match the report receipt hash."
            )
            continue

        reviewed_hashes = _reviewed_artifact_hashes(payload)
        missing_hashes = sorted(bound_hashes - reviewed_hashes)
        if missing_hashes:
            mismatch_reasons.append(
                f"human_review_note {artifact.path} does not cover current artifact hash set."
            )
            continue

        return True, []

    return False, _dedupe(mismatch_reasons)


def _reviewed_artifact_hashes(payload: Mapping[str, Any]) -> set[str]:
    raw = payload.get("reviewed_artifact_hashes")
    if not isinstance(raw, list):
        return set()
    return {str(item).strip() for item in raw if str(item).strip()}


def _report_receipt_hash(report: Mapping[str, Any]) -> str | None:
    receipt = report.get("receipt")
    if not isinstance(receipt, Mapping):
        return None
    value = receipt.get("canonical_hash")
    return value if isinstance(value, str) and value.strip() else None


def _claim_non_claims(report: Mapping[str, Any]) -> list[str]:
    claim = report.get("claim")
    if not isinstance(claim, Mapping):
        return []
    raw = claim.get("non_claims")
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if str(item).strip()]


def _mandatory_non_claims_missing(
    *,
    roles_present: set[str],
    non_claims: Sequence[str],
    exploratory_boundary_present: bool,
) -> list[str]:
    role_set = set(roles_present)
    role_set.add("receipt_integrity")

    if exploratory_boundary_present:
        role_set.add("exploratory_redteam")

    checks: list[tuple[str, tuple[tuple[str, ...], ...]]] = []
    for role in sorted(role_set):
        for non_claim in mandatory_non_claims_for(role):
            checks.append((non_claim.non_claim_id, non_claim.phrase_groups))

    return [
        check_id
        for check_id, phrase_groups in checks
        if not _non_claim_present(non_claims, phrase_groups)
    ]


def _non_claim_present(
    non_claims: Sequence[str],
    phrase_groups: tuple[tuple[str, ...], ...],
) -> bool:
    normalized = [_normalize_text(item) for item in non_claims]
    return any(
        all(any(_normalize_text(option) in item for option in group) for group in phrase_groups)
        for item in normalized
    )


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def _resolve_path(path: str, base_dir: Path) -> Path:
    candidate = Path(path)
    root = base_dir.resolve(strict=False)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Evidence artifact path escapes base_dir: {path}") from exc
    return resolved


def _nested_str(report: Mapping[str, Any], path: tuple[str, ...], default: str) -> str:
    value: Any = report
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return str(value)


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _coerce_string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    values = (value,) if isinstance(value, str) else tuple(value)
    return tuple(str(item).strip() for item in values if str(item).strip())


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _claim_level(value: str) -> ClaimLevel:
    if value not in CLAIM_LEVELS:
        allowed = ", ".join(CLAIM_LEVELS)
        raise ValueError(f"allowed_claim_level must be one of: {allowed}")
    if value in RESERVED_LIFECYCLE_STATE_NAMES:
        raise ValueError("allowed_claim_level must not be a lifecycle state")
    return cast(ClaimLevel, value)


def _unknown_or_claim_level(value: str) -> ClaimLevel | Literal["unknown"]:
    if value == "unknown":
        return "unknown"
    if value in CLAIM_LEVELS:
        return cast(ClaimLevel, value)
    return "unknown"


__all__ = [
    "CLAIM_GOVERNANCE_AUDIT_SCHEMA",
    "CLAIM_LEVEL_NON_CLAIM",
    "GOVERNANCE_PASS_CAVEAT",
    "RECEIPT_NON_CLAIM",
    "ROLE_SUPPORT_MATRIX",
    "BoundaryAudit",
    "ClaimFreshnessStatus",
    "ClaimGovernanceAudit",
    "ConfirmatoryProtocolGovernanceAudit",
    "DecayAudit",
    "EnvelopeSupportSummary",
    "EvidenceArtifactAudit",
    "EvidenceRoleStatus",
    "GovernanceVerdict",
    "ReceiptAudit",
    "ScenarioAudit",
    "_empty_envelope_support_summary",
    "verify_claim_governance",
]

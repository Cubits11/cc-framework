"""Compile portable, evidence-bound claim packages.

The compiler deliberately does less than its name can invite people to infer.
It does not decide whether a claim is true, safe, deployable, certified, or
ready for release.  It takes an already-verifiable ``cc.report.v0.3.1`` report,
copies its *bound* evidence into a portable layout, and preserves the report's
governance audit, claim envelope, support edges, non-claims, and review
requirements.

The copied ``report.json`` is byte-for-byte identical to the source report.
Evidence remains at the report's original relative paths under ``evidence/``;
that is what lets ``verify_claim_governance`` replay from inside the package
without rewriting the report or invalidating its receipt.

A package-level verifier is provided as a second check.  Its default ``now``
is the compiler's recorded verification time so a recipient can reproduce the
packaged audit exactly.  Supplying a different ``now`` deliberately performs a
freshness check at that time instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from cc.evidence.claim_envelope import ClaimEnvelope, SupportEdge, compile_claim_envelope
from cc.evidence.claim_governance import (
    ClaimGovernanceAudit,
    GovernanceVerdict,
    verify_claim_governance,
)
from cc.reporting.canonical import strict_json_loads
from cc.reporting.report import (
    CCReport,
    MeasurementIntervalModel,
    QuantitativePropositionModel,
    sha256_file,
)

CLAIM_PACKAGE_MANIFEST_SCHEMA_VERSION: Literal["cc.claim_package_manifest.v1"] = (
    "cc.claim_package_manifest.v1"
)
CLAIM_PACKAGE_AUDIT_SCHEMA_VERSION: Literal["cc.claim_package_audit.v1"] = (
    "cc.claim_package_audit.v1"
)
CLAIM_PACKAGE_LIFECYCLE_PROJECTION_SCHEMA_VERSION: Literal[
    "cc.claim_package.lifecycle_projection.v1"
] = "cc.claim_package.lifecycle_projection.v1"
CLAIM_PACKAGE_REVIEW_STATUS_SCHEMA_VERSION: Literal["cc.claim_package.review_status.v1"] = (
    "cc.claim_package.review_status.v1"
)

PACKAGE_PASS_CAVEAT = (
    "A package-level PASS means only that the package's declared checks passed at the recorded "
    "time; it does not establish free-text claim truth, source validity, or deployment safety."
)
PACKAGE_INTEGRITY_NON_CLAIM = (
    "Package hashes bind copied bytes and report references; they do not prove statistical "
    "validity, data representativeness, label correctness, or sufficiency for a release decision."
)
PACKAGE_LIFECYCLE_NON_CLAIM = (
    "The claim compiler does not assign, transition, revoke, or approve a claim lifecycle state."
)


class ClaimPackageError(ValueError):
    """Raised when a portable claim package cannot be compiled safely."""


class ClaimPackageModel(BaseModel):
    """Strict, frozen base model for public claim-package artifacts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
        strict=True,
    )


def _require_relative_package_path(value: str) -> str:
    """Return a normalized portable relative path or reject it.

    Report evidence paths are copied underneath ``evidence/``.  Accepting
    absolute paths, parent traversal, or Windows drive forms would either make
    the package non-portable or let a report escape the package root.
    """

    text = str(value).strip().replace("\\", "/")
    if not text:
        raise ValueError("path must be non-empty")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError("path must be a portable relative path without traversal")
    if any(part == "" for part in path.parts):
        raise ValueError("path contains an empty component")
    return path.as_posix()


class PackageSubjectReport(ClaimPackageModel):
    """Identity and byte binding for the untouched subject report."""

    report_id: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    package_path: str = "report.json"
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)
    canonical_receipt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("package_path")
    @classmethod
    def _report_path_is_fixed(cls, value: str) -> str:
        if value != "report.json":
            raise ValueError("subject report package_path must be report.json")
        return value


class PackageArtifact(ClaimPackageModel):
    """One report-bound source artifact copied into ``evidence/``."""

    artifact_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    package_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)

    @field_validator("source_path")
    @classmethod
    def _source_path_is_portable(cls, value: str) -> str:
        return _require_relative_package_path(value)

    @field_validator("package_path")
    @classmethod
    def _package_path_is_portable(cls, value: str) -> str:
        normalized = _require_relative_package_path(value)
        if not normalized.startswith("evidence/"):
            raise ValueError("package evidence artifact paths must be under evidence/")
        return normalized


class PackageVerifierResult(ClaimPackageModel):
    """The governance verifier result pinned when the package was compiled."""

    verdict: Literal["pass", "needs_review"]
    evaluated_at: str = Field(min_length=1)
    audit_path: str = "audits/claim_governance_audit.json"
    required_human_review: bool

    @field_validator("audit_path")
    @classmethod
    def _audit_path_is_fixed(cls, value: str) -> str:
        if value != "audits/claim_governance_audit.json":
            raise ValueError("governance audit path must be audits/claim_governance_audit.json")
        return value


class PackageLifecycleProjection(ClaimPackageModel):
    """An explicit non-lifecycle projection for the required package directory.

    The project deliberately does not yet own a claim lifecycle state machine.
    This record makes that absence visible instead of rebranding a governance
    verdict as a lifecycle state.
    """

    schema_: Literal["cc.claim_package.lifecycle_projection.v1"] = Field(
        default=CLAIM_PACKAGE_LIFECYCLE_PROJECTION_SCHEMA_VERSION,
        alias="schema",
    )
    lifecycle_owner: Literal["not_owned_by_claim_compiler"] = "not_owned_by_claim_compiler"
    claim_lifecycle_state: None = None
    verifier_verdict: Literal["pass", "needs_review"]
    freshness_status: str = Field(min_length=1)
    non_claims: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _states_absence_is_explicit(self) -> PackageLifecycleProjection:
        if PACKAGE_LIFECYCLE_NON_CLAIM not in self.non_claims:
            raise ValueError("lifecycle projection must preserve its non-lifecycle boundary")
        return self


class PackageHumanReviewStatus(ClaimPackageModel):
    """Review requirement projection without treating review as evidence upgrade."""

    schema_: Literal["cc.claim_package.review_status.v1"] = Field(
        default=CLAIM_PACKAGE_REVIEW_STATUS_SCHEMA_VERSION,
        alias="schema",
    )
    required_human_review: bool
    reviewed_artifact_paths: tuple[str, ...] = Field(default_factory=tuple)
    non_claims: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("reviewed_artifact_paths")
    @classmethod
    def _reviewed_paths_are_portable(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_require_relative_package_path(item) for item in value)


class PackageReproducibility(ClaimPackageModel):
    """How to reproduce the recorded package-level verification, and how to
    adversarially falsify the package's own tamper-evidence claim."""

    fixed_now: str = Field(min_length=1)
    verification_command: str = Field(min_length=1)
    package_verification_command: str = Field(min_length=1)
    challenge_command: str = Field(min_length=1)


class PackageIntegrityChecks(ClaimPackageModel):
    """Hashes for generated package surfaces not named by the source report."""

    source_governance_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_envelope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lifecycle_projection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_status_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    readme_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    challenge_doc_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PackageEntailmentResult(ClaimPackageModel):
    """Whether a declared structured proposition follows from report interval data.

    The compiler never infers a proposition from prose.  ``not_checked`` is the
    expected result when the report has only a free-text claim statement.
    """

    status: Literal["pass", "fail", "not_checked"]
    check: Literal["structured_measurement_interval", "not_checked", "unavailable"]
    proposition: QuantitativePropositionModel | None = None
    measurement_interval: MeasurementIntervalModel | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _result_has_the_right_evidence(self) -> PackageEntailmentResult:
        if self.status == "not_checked":
            if (
                self.check != "not_checked"
                or self.proposition is not None
                or self.measurement_interval is not None
            ):
                raise ValueError("not_checked entailment must not carry a proposition")
            return self
        if self.check == "unavailable":
            if self.status != "fail" or self.proposition is not None:
                raise ValueError("unavailable entailment must be a failure without a proposition")
            return self
        if (
            self.check != "structured_measurement_interval"
            or self.proposition is None
            or self.measurement_interval is None
        ):
            raise ValueError("checked entailment requires a structured proposition and interval")
        return self


class PackageIndependenceResult(ClaimPackageModel):
    """Independence evidence visible to the package verifier.

    Claim-package v1 has no independently verified witness declaration, so it
    must report ``none`` rather than promote a same-author build or challenge
    into an independence result.
    """

    status: Literal["none"]
    reason: str = Field(min_length=1)


def _unstructured_entailment_result() -> PackageEntailmentResult:
    return PackageEntailmentResult(
        status="not_checked",
        check="not_checked",
        reason=(
            "No structured quantitative proposition was declared; free-text claim prose is "
            "not parsed or semantic-entailment checked."
        ),
    )


def _no_independence_result() -> PackageIndependenceResult:
    return PackageIndependenceResult(
        status="none",
        reason=(
            "No independent implementation, recomputation, or external review is established "
            "by claim-package v1."
        ),
    )


def _evaluate_entailment(report: Mapping[str, Any]) -> PackageEntailmentResult:
    """Evaluate the one deliberately narrow machine-checkable proposition.

    This function never interprets ``claim.statement``.  A report author must
    explicitly declare an upper- or lower-bound proposition over the exact
    measurement metric before the compiler compares it with the report's
    already receipt-bound interval.
    """

    try:
        parsed = CCReport.model_validate(report)
    except ValidationError as exc:
        return PackageEntailmentResult(
            status="fail",
            check="unavailable",
            reason=(
                "The packaged report cannot be validated, so structured quantitative "
                f"entailment is unavailable: {exc}"
            ),
        )

    proposition = parsed.claim.quantitative_proposition
    if proposition is None:
        return _unstructured_entailment_result()

    interval = parsed.measurement.interval
    metric_family = parsed.measurement.metric_family
    if proposition.metric_family != metric_family:
        return PackageEntailmentResult(
            status="fail",
            check="structured_measurement_interval",
            proposition=proposition,
            measurement_interval=interval,
            reason=(
                "Structured proposition metric_family does not match the report measurement "
                f"metric (proposition={proposition.metric_family!r}, measurement={metric_family!r})."
            ),
        )

    if proposition.relation == "upper_bound":
        passed = interval.upper <= proposition.threshold
        comparison = (
            f"interval upper {interval.upper:g} <= declared threshold {proposition.threshold:g}"
        )
    else:
        passed = interval.lower >= proposition.threshold
        comparison = (
            f"interval lower {interval.lower:g} >= declared threshold {proposition.threshold:g}"
        )

    return PackageEntailmentResult(
        status="pass" if passed else "fail",
        check="structured_measurement_interval",
        proposition=proposition,
        measurement_interval=interval,
        reason=(
            f"Structured {proposition.relation.replace('_', ' ')} check {'passed' if passed else 'failed'}: "
            f"{comparison}."
        ),
    )


class ClaimPackageManifest(ClaimPackageModel):
    """The portable manifest for one evidence-bound claim package."""

    schema_: Literal["cc.claim_package_manifest.v1"] = Field(
        default=CLAIM_PACKAGE_MANIFEST_SCHEMA_VERSION,
        alias="schema",
    )
    package_id: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    evidence_base_dir: str = "evidence"
    subject_report: PackageSubjectReport
    artifacts: tuple[PackageArtifact, ...] = Field(default_factory=tuple)
    support_edges: tuple[SupportEdge, ...] = Field(default_factory=tuple)
    verifier_result: PackageVerifierResult
    lifecycle_state: PackageLifecycleProjection
    human_review_status: PackageHumanReviewStatus
    non_claims: tuple[str, ...] = Field(default_factory=tuple)
    reproducibility: PackageReproducibility
    integrity_checks: PackageIntegrityChecks
    entailment: PackageEntailmentResult = Field(default_factory=_unstructured_entailment_result)
    independence: PackageIndependenceResult = Field(default_factory=_no_independence_result)

    @model_validator(mode="after")
    def _manifest_preserves_boundaries(self) -> ClaimPackageManifest:
        if self.evidence_base_dir != "evidence":
            raise ValueError("evidence_base_dir must be evidence in claim-package v1")
        artifact_ids = [artifact.artifact_id for artifact in self.artifacts]
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("package artifact IDs must be unique")
        for non_claim in (
            PACKAGE_PASS_CAVEAT,
            PACKAGE_INTEGRITY_NON_CLAIM,
            PACKAGE_LIFECYCLE_NON_CLAIM,
        ):
            if non_claim not in self.non_claims:
                raise ValueError("manifest is missing a required package non-claim")
        return self

    def to_canonical_dict(self) -> dict[str, Any]:
        """Return the stable JSON-native manifest representation."""

        return self.model_dump(mode="json", by_alias=True)


class PackageArtifactAudit(ClaimPackageModel):
    """Outcome of checking one copied source artifact."""

    artifact_id: str = Field(min_length=1)
    package_path: str = Field(min_length=1)
    sha256_expected: str = Field(pattern=r"^[0-9a-f]{64}$")
    sha256_actual: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    bytes_expected: int = Field(ge=0)
    bytes_actual: int | None = Field(default=None, ge=0)
    valid: bool
    reason: str = Field(min_length=1)


class ClaimPackageAudit(ClaimPackageModel):
    """Read-only verification result for a compiled claim package."""

    schema_: Literal["cc.claim_package_audit.v1"] = Field(
        default=CLAIM_PACKAGE_AUDIT_SCHEMA_VERSION,
        alias="schema",
    )
    package_id: str = Field(min_length=1)
    evaluated_at: str = Field(min_length=1)
    verdict: Literal["pass", "needs_review", "fail"]
    integrity_verdict: Literal["pass", "fail"]
    entailment: PackageEntailmentResult
    independence: PackageIndependenceResult
    manifest_valid: bool
    report_integrity_valid: bool
    artifacts: tuple[PackageArtifactAudit, ...] = Field(default_factory=tuple)
    governance_verdict: Literal["pass", "needs_review", "fail"]
    support_edges_preserved: bool
    reasons: tuple[str, ...] = Field(default_factory=tuple)
    non_claims: tuple[str, ...] = Field(default_factory=tuple)


def compile_claim_package(
    report_path: Path,
    out_dir: Path,
    *,
    package_id: str | None = None,
    now: datetime | None = None,
    base_dir: Path | None = None,
    strict_unknown_roles: bool = False,
    require_pass: bool = False,
) -> ClaimPackageManifest:
    """Compile a portable package from one verified report.

    ``FAIL`` governance input always aborts.  ``NEEDS_REVIEW`` can be packaged
    (unless ``require_pass`` is set) because a package that honestly carries an
    unresolved review requirement can still be useful for scrutiny.  In both
    cases, hash mismatches, missing files, and non-portable evidence paths
    abort before the requested output directory is created.
    """

    source_report = Path(report_path).resolve()
    source_root = Path(base_dir).resolve() if base_dir is not None else source_report.parent
    destination = Path(out_dir).resolve()
    verification_time = _normalized_time(now)

    if not source_report.is_file():
        raise ClaimPackageError(f"Report file not found: {source_report}")
    if not source_root.is_dir():
        raise ClaimPackageError(f"Evidence base directory not found: {source_root}")
    if destination.exists():
        raise ClaimPackageError(f"Package destination already exists: {destination}")
    if destination == source_root or source_root in destination.parents:
        raise ClaimPackageError(
            "Package destination must not be inside the source evidence directory"
        )

    report = _read_json_object(source_report, "report")
    try:
        # Governance has a deliberately lightweight shape check for useful
        # failure audits.  Compilation needs the stronger report contract too:
        # a package must never preserve a malformed or receipt-invalid report
        # just because it was structurally readable.
        CCReport.model_validate_json(source_report.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise ClaimPackageError(f"Source report failed strict receipt validation: {exc}") from exc
    entailment = _evaluate_entailment(report)
    independence = _no_independence_result()
    source_audit = verify_claim_governance(
        source_report,
        now=verification_time,
        base_dir=source_root,
        strict_unknown_roles=strict_unknown_roles,
    )
    if source_audit.verdict is GovernanceVerdict.FAIL:
        raise ClaimPackageError(
            "Claim governance verification failed; refusing to package inconsistent evidence: "
            + "; ".join(source_audit.reasons)
        )
    if require_pass and source_audit.verdict is not GovernanceVerdict.PASS:
        raise ClaimPackageError(
            "Claim governance requires review; --require-pass refuses to package it: "
            + "; ".join(source_audit.reasons)
        )

    source_artifacts = _source_artifacts_from_report(report, source_root)
    for artifact in source_artifacts:
        _assert_source_artifact_matches(artifact, source_root)

    try:
        compile_claim_envelope(report, governance_audit=source_audit)
    except Exception as exc:  # pragma: no cover - defensive boundary around existing compiler
        raise ClaimPackageError(f"ClaimEnvelope compilation failed: {exc}") from exc

    package_identifier = package_id or _default_package_id(report)
    _require_nonempty(package_identifier, "package_id")
    _validate_destination_collisions(source_artifacts)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        _create_package_layout(temp_dir)
        _copy_file(source_report, temp_dir / "report.json")
        for artifact in source_artifacts:
            _copy_file(source_root / artifact.source_path, temp_dir / artifact.package_path)

        package_audit = verify_claim_governance(
            temp_dir / "report.json",
            now=verification_time,
            base_dir=temp_dir / "evidence",
            strict_unknown_roles=strict_unknown_roles,
        )
        if package_audit.verdict is GovernanceVerdict.FAIL:
            raise ClaimPackageError(
                "Copied package failed in-package governance verification: "
                + "; ".join(package_audit.reasons)
            )
        if package_audit.verdict is not source_audit.verdict:
            raise ClaimPackageError(
                "Copied package governance verdict changed during compilation: "
                f"source={source_audit.verdict.value}, package={package_audit.verdict.value}"
            )

        package_envelope = compile_claim_envelope(report, governance_audit=package_audit)
        lifecycle = _lifecycle_projection(package_audit)
        review_status = _review_status(package_audit, source_artifacts)

        audit_path = temp_dir / "audits" / "claim_governance_audit.json"
        envelope_path = temp_dir / "envelope" / "claim_envelope.json"
        lifecycle_path = temp_dir / "lifecycle" / "projection.json"
        review_path = temp_dir / "reviews" / "review_status.json"
        _write_json(audit_path, package_audit.model_dump(mode="json", by_alias=True))
        _write_json(envelope_path, package_envelope.to_canonical_dict())
        _write_json(lifecycle_path, lifecycle.model_dump(mode="json", by_alias=True))
        _write_json(review_path, review_status.model_dump(mode="json", by_alias=True))
        readme_path = temp_dir / "README.md"
        challenge_path = temp_dir / "CHALLENGE.md"
        _write_readme(temp_dir, package_audit, source_artifacts, entailment, independence)
        _write_challenge_doc(temp_dir, source_artifacts)

        manifest = _build_manifest(
            package_id=package_identifier,
            source_report=source_report,
            report=report,
            artifacts=source_artifacts,
            audit=package_audit,
            envelope=package_envelope,
            lifecycle=lifecycle,
            review_status=review_status,
            entailment=entailment,
            independence=independence,
            now=verification_time,
            audit_path=audit_path,
            envelope_path=envelope_path,
            lifecycle_path=lifecycle_path,
            review_path=review_path,
            readme_path=readme_path,
            challenge_path=challenge_path,
        )
        _write_json(temp_dir / "manifest.json", manifest.to_canonical_dict())

        package_check = verify_claim_package(
            temp_dir,
            now=verification_time,
            strict_unknown_roles=strict_unknown_roles,
        )
        if package_check.integrity_verdict == "fail":
            raise ClaimPackageError(
                "Newly compiled package did not pass its integrity check: "
                + "; ".join(package_check.reasons)
            )
        if package_check.entailment != entailment or package_check.independence != independence:
            raise ClaimPackageError(
                "Newly compiled package did not reproduce its declared verdict axes: "
                + "; ".join(package_check.reasons)
            )
        if require_pass and package_check.verdict != "pass":
            raise ClaimPackageError("Newly compiled package still requires review")

        os.replace(temp_dir, destination)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return manifest


def verify_claim_package(
    package_dir: Path,
    *,
    now: datetime | None = None,
    strict_unknown_roles: bool = False,
) -> ClaimPackageAudit:
    """Verify a portable claim package without modifying it.

    Without ``now``, verification reuses the package's recorded fixed time and
    therefore checks reproducibility of the packaged verdict.  Supplying
    ``now`` reruns the governance verifier at that time, allowing an explicit
    freshness check whose verdict may legitimately differ from the original.
    """

    root = Path(package_dir).resolve()
    manifest_path = root / "manifest.json"
    fallback_time = _normalized_time(now)
    if not root.is_dir():
        return _package_failure_audit(
            package_id="<unknown>",
            evaluated_at=_utc_iso(fallback_time),
            reason=f"Package directory not found: {root}",
        )

    try:
        manifest = _load_model_json(manifest_path, ClaimPackageManifest, "manifest")
    except (OSError, ValueError, ValidationError) as exc:
        return _package_failure_audit(
            package_id="<unknown>",
            evaluated_at=_utc_iso(fallback_time),
            reason=f"Package manifest is unreadable or invalid: {exc}",
        )

    use_recorded_time = now is None
    if use_recorded_time:
        try:
            effective_time = _parse_aware_time(manifest.reproducibility.fixed_now)
        except ValueError as exc:
            return _package_failure_audit(
                package_id=manifest.package_id,
                evaluated_at=_utc_iso(fallback_time),
                reason=f"Package manifest fixed_now is invalid: {exc}",
            )
    else:
        effective_time = fallback_time

    fatal_reasons: list[str] = []
    integrity_reasons: list[str] = []
    review_reasons: list[str] = []
    artifact_audits: list[PackageArtifactAudit] = []
    report_path = root / manifest.subject_report.package_path
    report_valid = _file_matches(
        report_path,
        manifest.subject_report.sha256,
        manifest.subject_report.bytes,
    )
    if not report_valid[0]:
        reason = f"Subject report integrity failed: {report_valid[1]}"
        fatal_reasons.append(reason)
        integrity_reasons.append(reason)

    for artifact in manifest.artifacts:
        file_path = root / artifact.package_path
        valid, reason, actual_hash, actual_bytes = _file_matches(
            file_path, artifact.sha256, artifact.bytes
        )
        artifact_audits.append(
            PackageArtifactAudit(
                artifact_id=artifact.artifact_id,
                package_path=artifact.package_path,
                sha256_expected=artifact.sha256,
                sha256_actual=actual_hash,
                bytes_expected=artifact.bytes,
                bytes_actual=actual_bytes,
                valid=valid,
                reason=reason,
            )
        )
        if not valid:
            integrity_reason = f"Artifact {artifact.package_path} integrity failed: {reason}"
            fatal_reasons.append(integrity_reason)
            integrity_reasons.append(integrity_reason)

    _generated_valid, generated_reasons = _verify_generated_surfaces(root, manifest)
    fatal_reasons.extend(generated_reasons)
    integrity_reasons.extend(generated_reasons)

    try:
        packaged_report = _read_json_object(report_path, "packaged report")
        entailment = _evaluate_entailment(packaged_report)
    except (OSError, ValueError) as exc:
        entailment = PackageEntailmentResult(
            status="fail",
            check="unavailable",
            reason=f"The packaged report cannot be read for entailment checking: {exc}",
        )
    independence = _no_independence_result()

    governance = verify_claim_governance(
        report_path,
        now=effective_time,
        base_dir=root / "evidence",
        strict_unknown_roles=strict_unknown_roles,
    )
    if governance.verdict is GovernanceVerdict.FAIL:
        fatal_reasons.append(
            "In-package claim governance verification failed: " + "; ".join(governance.reasons)
        )
    elif governance.verdict is GovernanceVerdict.NEEDS_REVIEW:
        review_reasons.extend(governance.reasons)

    support_edges_preserved = _support_edges_match(root, manifest, governance)
    if not support_edges_preserved:
        reason = "Package manifest support edges do not match the compiled claim envelope."
        fatal_reasons.append(reason)
        integrity_reasons.append(reason)

    # The manifest is the package's root of trust, so it must carry no authority
    # a verifier cannot re-derive. Recompute every verdict-bearing manifest field
    # from the receipt-bound report and the package files; a manifest that
    # disagrees with what the package actually contains is a tampered manifest.
    manifest_reasons = _manifest_matches_package(root, manifest, governance)
    fatal_reasons.extend(manifest_reasons)
    integrity_reasons.extend(manifest_reasons)

    if use_recorded_time:
        stored_audit_path = root / manifest.verifier_result.audit_path
        try:
            stored_audit = _load_model_json(
                stored_audit_path, ClaimGovernanceAudit, "stored governance audit"
            )
            if stored_audit.model_dump(mode="json", by_alias=True) != governance.model_dump(
                mode="json", by_alias=True
            ):
                fatal_reasons.append(
                    "Stored governance audit does not reproduce at the package's recorded verification time."
                )
        except (OSError, ValueError, ValidationError) as exc:
            fatal_reasons.append(f"Stored governance audit is unreadable or invalid: {exc}")
    elif governance.verdict.value != manifest.verifier_result.verdict:
        review_reasons.append(
            "Fresh governance verdict differs from the packaged verdict at the supplied verification time."
        )

    if entailment.status == "fail":
        fatal_reasons.append("Structured quantitative entailment failed: " + entailment.reason)

    if fatal_reasons:
        verdict: Literal["pass", "needs_review", "fail"] = "fail"
    elif governance.verdict is GovernanceVerdict.NEEDS_REVIEW or review_reasons:
        verdict = "needs_review"
    else:
        verdict = "pass"

    return ClaimPackageAudit(
        package_id=manifest.package_id,
        evaluated_at=_utc_iso(effective_time),
        verdict=verdict,
        integrity_verdict="fail" if integrity_reasons else "pass",
        entailment=entailment,
        independence=independence,
        manifest_valid=True,
        report_integrity_valid=report_valid[0],
        artifacts=tuple(artifact_audits),
        governance_verdict=governance.verdict.value,
        support_edges_preserved=support_edges_preserved,
        reasons=tuple(dict.fromkeys([*fatal_reasons, *review_reasons])),
        non_claims=tuple(
            dict.fromkeys(
                [
                    PACKAGE_PASS_CAVEAT,
                    PACKAGE_INTEGRITY_NON_CLAIM,
                    PACKAGE_LIFECYCLE_NON_CLAIM,
                    *manifest.non_claims,
                ]
            )
        ),
    )


def _source_artifacts_from_report(
    report: Mapping[str, Any],
    source_root: Path,
) -> tuple[PackageArtifact, ...]:
    evidence = report.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ClaimPackageError("Report evidence must be an object")

    entries: list[Mapping[str, Any]] = []
    artifacts = evidence.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ClaimPackageError("Report evidence.artifacts must be an array")
    entries.extend(item for item in artifacts if isinstance(item, Mapping))
    if len(entries) != len(artifacts):
        raise ClaimPackageError("Report evidence.artifacts must contain objects")
    for key, fallback_role in (("audit_log", "audit_log"), ("figure_manifest", "figure_manifest")):
        item = evidence.get(key)
        if item is not None:
            if not isinstance(item, Mapping):
                raise ClaimPackageError(f"Report evidence.{key} must be an object or null")
            item_with_role = dict(item)
            item_with_role.setdefault("role", fallback_role)
            entries.append(item_with_role)

    package_artifacts: list[PackageArtifact] = []
    for index, entry in enumerate(entries):
        raw_path = entry.get("path")
        raw_hash = entry.get("sha256")
        raw_bytes = entry.get("bytes")
        raw_role = entry.get("role", "artifact")
        if not isinstance(raw_path, str) or not isinstance(raw_hash, str):
            raise ClaimPackageError(f"Evidence entry {index} is missing path or sha256")
        if isinstance(raw_bytes, bool) or not isinstance(raw_bytes, int) or raw_bytes < 0:
            raise ClaimPackageError(f"Evidence entry {index} has an invalid byte count")
        if not isinstance(raw_role, str) or not raw_role.strip():
            raise ClaimPackageError(f"Evidence entry {index} has an invalid role")
        try:
            source_path = _require_relative_package_path(raw_path)
        except ValueError as exc:
            raise ClaimPackageError(
                f"Evidence path {raw_path!r} cannot be made portable: {exc}"
            ) from exc
        source_file = (source_root / source_path).resolve()
        if not _is_within(source_file, source_root):
            raise ClaimPackageError(f"Evidence path escapes source base directory: {raw_path}")
        artifact_id = f"artifact-{index:03d}-{_slug(raw_role)}"
        package_artifacts.append(
            PackageArtifact(
                artifact_id=artifact_id,
                role=raw_role.strip(),
                source_path=source_path,
                package_path=f"evidence/{source_path}",
                sha256=raw_hash,
                bytes=raw_bytes,
            )
        )
    return tuple(package_artifacts)


def _assert_source_artifact_matches(artifact: PackageArtifact, source_root: Path) -> None:
    valid, reason, _actual_hash, _actual_bytes = _file_matches(
        source_root / artifact.source_path, artifact.sha256, artifact.bytes
    )
    if not valid:
        raise ClaimPackageError(
            f"Source evidence {artifact.source_path} does not match its report binding: {reason}"
        )


def _create_package_layout(root: Path) -> None:
    for path in (
        root / "evidence",
        root / "audits",
        root / "lifecycle",
        root / "envelope",
        root / "reviews",
    ):
        path.mkdir(parents=True, exist_ok=False)


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _lifecycle_projection(audit: ClaimGovernanceAudit) -> PackageLifecycleProjection:
    return PackageLifecycleProjection(
        verifier_verdict=audit.verdict.value,
        freshness_status=audit.decay.status.value,
        non_claims=(PACKAGE_LIFECYCLE_NON_CLAIM,),
    )


def _review_status(
    audit: ClaimGovernanceAudit,
    artifacts: Sequence[PackageArtifact],
) -> PackageHumanReviewStatus:
    review_paths = tuple(
        artifact.package_path
        for artifact in artifacts
        if artifact.role in {"human_review", "human_review_note"}
    )
    return PackageHumanReviewStatus(
        required_human_review=audit.required_human_review,
        reviewed_artifact_paths=review_paths,
        non_claims=(
            "Human review can authorize scoped use of a bound package; it does not upgrade underlying statistical evidence.",
        ),
    )


def _reproducibility_projection(fixed_now: str) -> PackageReproducibility:
    """Return the fully re-derived command projection for a recorded time."""

    return PackageReproducibility(
        fixed_now=fixed_now,
        verification_command=(
            "python -m cc.reporting.cli verify-claim-governance report.json "
            f"--base-dir evidence --now {fixed_now}"
        ),
        package_verification_command=(
            f"python -m cc.reporting.cli verify-claim-package . --now {fixed_now}"
        ),
        challenge_command="python -m cc.reporting.cli challenge-claim-package .",
    )


def _build_manifest(
    *,
    package_id: str,
    source_report: Path,
    report: Mapping[str, Any],
    artifacts: tuple[PackageArtifact, ...],
    audit: ClaimGovernanceAudit,
    envelope: ClaimEnvelope,
    lifecycle: PackageLifecycleProjection,
    review_status: PackageHumanReviewStatus,
    entailment: PackageEntailmentResult,
    independence: PackageIndependenceResult,
    now: datetime,
    audit_path: Path,
    envelope_path: Path,
    lifecycle_path: Path,
    review_path: Path,
    readme_path: Path,
    challenge_path: Path,
) -> ClaimPackageManifest:
    receipt = report.get("receipt")
    canonical_hash = receipt.get("canonical_hash") if isinstance(receipt, Mapping) else None
    subject = PackageSubjectReport(
        report_id=str(report.get("report_id") or source_report.stem),
        schema_version=str(report.get("schema_version") or "unknown"),
        source_path=str(source_report),
        sha256=sha256_file(source_report),
        bytes=source_report.stat().st_size,
        canonical_receipt_sha256=canonical_hash if isinstance(canonical_hash, str) else None,
    )
    non_claims = tuple(
        dict.fromkeys(
            [
                PACKAGE_PASS_CAVEAT,
                PACKAGE_INTEGRITY_NON_CLAIM,
                PACKAGE_LIFECYCLE_NON_CLAIM,
                *audit.non_claims,
                *envelope.boundary.non_claims,
            ]
        )
    )
    now_iso = _utc_iso(now)
    return ClaimPackageManifest(
        package_id=package_id,
        created_at=now_iso,
        evidence_base_dir="evidence",
        subject_report=subject,
        artifacts=artifacts,
        support_edges=envelope.support_graph.support_edges,
        verifier_result=PackageVerifierResult(
            verdict=audit.verdict.value,
            evaluated_at=now_iso,
            required_human_review=audit.required_human_review,
        ),
        lifecycle_state=lifecycle,
        human_review_status=review_status,
        entailment=entailment,
        independence=independence,
        non_claims=non_claims,
        reproducibility=_reproducibility_projection(now_iso),
        integrity_checks=PackageIntegrityChecks(
            source_governance_audit_sha256=_sha256_file_at(audit_path),
            claim_envelope_sha256=_sha256_file_at(envelope_path),
            lifecycle_projection_sha256=_sha256_file_at(lifecycle_path),
            review_status_sha256=_sha256_file_at(review_path),
            readme_sha256=_sha256_file_at(readme_path),
            challenge_doc_sha256=_sha256_file_at(challenge_path),
        ),
    )


def _render_readme(
    audit: ClaimGovernanceAudit,
    artifacts: Sequence[PackageArtifact],
    entailment: PackageEntailmentResult,
    independence: PackageIndependenceResult,
) -> str:
    artifact_lines = (
        "\n".join(
            f"- `{artifact.package_path}` — `{artifact.role}` — `{artifact.sha256}`"
            for artifact in artifacts
        )
        or "- No report-bound evidence artifacts."
    )
    review_line = (
        "Human review is required by the recorded governance audit."
        if audit.required_human_review
        else "The recorded governance audit did not require unresolved human review."
    )
    text = f"""# Portable Evidence-Bound Claim Package

This directory is a portable copy of one already-bound CC report and its evidence.
The report is copied byte-for-byte as `report.json`; its evidence paths resolve under
`evidence/`, so the original receipt stays meaningful without being rewritten.

## Separate recorded verdicts

At `{audit.evaluated_at}`:

- **Integrity:** checked by `verify-claim-package` against the copied bytes,
  manifest bindings, and generated surfaces. It does not establish claim truth.
- **Governance:** `{audit.verdict.value.upper()}` under the report's evidence and
  freshness rules.
- **Entailment:** `{entailment.status.upper()}` — {entailment.reason}
- **Independence:** `{independence.status.upper()}` — {independence.reason}

The compiler does not infer entailment from `claim.statement`. Free-text prose
therefore remains `NOT_CHECKED`; only an explicitly declared structured upper-
or lower-bound proposition is compared with the report measurement interval.

This file is itself a bound surface: its bytes are hashed in `manifest.json` and
re-derived by the verifier, so the boundary stated below cannot be rewritten
without verification falling to `FAIL`.

{PACKAGE_PASS_CAVEAT}

{PACKAGE_INTEGRITY_NON_CLAIM}

{PACKAGE_LIFECYCLE_NON_CLAIM}

The package can make an asserted input reproducible and internally consistent.
It cannot turn an asserted input into a measurement or establish that its source
data were valid; that boundary remains outside repository-local verification.

{review_line}

## Reproduce the recorded check

From this directory, with CC-Framework installed or available on `PYTHONPATH`:

```bash
python -m cc.reporting.cli verify-claim-governance report.json --base-dir evidence --now {audit.evaluated_at}
python -m cc.reporting.cli verify-claim-package . --now {audit.evaluated_at}
```

Use another `--now` value only when intentionally checking freshness at a new
verification time; the verdict may then change without changing the historical
package.

## Report-bound evidence

{artifact_lines}
"""
    return text


def _write_readme(
    root: Path,
    audit: ClaimGovernanceAudit,
    artifacts: Sequence[PackageArtifact],
    entailment: PackageEntailmentResult,
    independence: PackageIndependenceResult,
) -> None:
    (root / "README.md").write_text(
        _render_readme(audit, artifacts, entailment, independence), encoding="utf-8"
    )


def _render_challenge_doc(artifacts: Sequence[PackageArtifact]) -> str:
    """Render the falsification protocol so a recipient need not trust the compiler.

    The package asserts a narrow byte-integrity property. This document tells a
    skeptic how to disprove that assertion offline: apply the named one-byte
    mutations and watch the integrity verdict fall to FAIL.
    """

    surfaces = (
        "\n".join(
            f"- `{artifact.package_path}` (bound evidence, role `{artifact.role}`)"
            for artifact in artifacts
        )
        or "- (no report-bound evidence artifacts)"
    )
    text = f"""# Falsify this package

This package claims one narrow, checkable thing: it detects the **specific
single-byte mutations** this challenge applies to its report, bound evidence,
generated audit surfaces, and boundary text. A detected mutation makes the
package's **integrity verdict** `FAIL`. Do not trust that claim. Break it.

## The one-command challenge

From this directory, with CC-Framework installed or on `PYTHONPATH`:

```bash
python -m cc.reporting.cli challenge-claim-package .
```

The challenge copies this package to a scratch directory, then for each bound
surface applies the minimal mutation (flips or appends a single byte), re-runs
`verify_claim_package` at the recorded time, and records whether the integrity
verdict fell to `FAIL`. It restores nothing in place — your package is never modified —
and it prints one line per surface. Every mutated surface must be **detected**.
A control run over the untouched copy must reproduce the package's recorded
integrity, governance, entailment, and independence results, so a harness that
simply always fails cannot pass the challenge.

## The surfaces it mutates

- `report.json` (the byte-bound subject report)
- `manifest.json` (the package manifest itself)
- the generated audit, envelope, lifecycle, and review surfaces
- `README.md` (the PASS caveat and the non-claims) and `CHALLENGE.md` (this
  protocol) — the boundary text a reader relies on to know what the PASS means
{surfaces}

## What a PASS on the challenge does and does not mean

It means: the specific single-byte mutations the challenge applies to each named
surface are all detected, and the untouched control reproduces. It does **not**
mean the underlying claim is true, the evidence is valid, or that no undetectable
modification of any kind exists — only that this package's integrity binding
catches the mutations it is challenged with. It does not test semantic
entailment or independence. Integrity is not validity.
"""
    return text


def _write_challenge_doc(root: Path, artifacts: Sequence[PackageArtifact]) -> None:
    (root / "CHALLENGE.md").write_text(_render_challenge_doc(artifacts), encoding="utf-8")


def _verify_generated_surfaces(
    root: Path,
    manifest: ClaimPackageManifest,
) -> tuple[bool, list[str]]:
    checks = (
        (
            root / manifest.verifier_result.audit_path,
            manifest.integrity_checks.source_governance_audit_sha256,
            "governance audit",
        ),
        (
            root / "envelope" / "claim_envelope.json",
            manifest.integrity_checks.claim_envelope_sha256,
            "claim envelope",
        ),
        (
            root / "lifecycle" / "projection.json",
            manifest.integrity_checks.lifecycle_projection_sha256,
            "lifecycle projection",
        ),
        (
            root / "reviews" / "review_status.json",
            manifest.integrity_checks.review_status_sha256,
            "review status",
        ),
        (
            root / "README.md",
            manifest.integrity_checks.readme_sha256,
            "package README",
        ),
        (
            root / "CHALLENGE.md",
            manifest.integrity_checks.challenge_doc_sha256,
            "challenge document",
        ),
    )
    reasons: list[str] = []
    for path, expected_hash, label in checks:
        valid, reason, _actual_hash, _actual_bytes = _file_matches(path, expected_hash, None)
        if not valid:
            reasons.append(f"{label} integrity failed: {reason}")
    return not reasons, reasons


def _manifest_matches_package(
    root: Path,
    manifest: ClaimPackageManifest,
    governance: ClaimGovernanceAudit,
) -> list[str]:
    """Re-derive the manifest's verdict-bearing fields and reject disagreement.

    Everything the manifest asserts about what the package contains is recomputed
    from the packaged report and files. This includes the two human-facing
    surfaces — ``README.md`` (which carries the PASS caveat and the non-claims)
    and ``CHALLENGE.md`` (which carries the falsification protocol) — because a
    package whose boundary text can be rewritten without detection is one whose
    PASS can be misrepresented. Only pure input labels the verifier cannot
    re-derive — ``package_id`` and ``created_at`` — remain outside this check, and
    those carry no verdict. Any tampered artifact list, subject-report binding, or
    generated-surface integrity hash is caught here even if the attacker edited
    the manifest to match a tampered file.
    """

    reasons: list[str] = []
    try:
        report = _read_json_object(root / "report.json", "packaged report")
    except (OSError, ValueError) as exc:
        return [f"Packaged report is unreadable while re-deriving the manifest: {exc}"]

    try:
        expected_artifacts = _source_artifacts_from_report(report, root / "evidence")
    except ClaimPackageError as exc:
        return [f"Manifest artifact list cannot be re-derived from the packaged report: {exc}"]
    if _artifacts_projection(expected_artifacts) != _artifacts_projection(manifest.artifacts):
        reasons.append("Manifest artifact list does not match the packaged report's evidence.")

    subject = manifest.subject_report
    expected_report_id = str(report.get("report_id") or "report")
    if subject.report_id != expected_report_id:
        reasons.append("Manifest subject report_id does not match the packaged report.")
    expected_schema = str(report.get("schema_version") or "unknown")
    if subject.schema_version != expected_schema:
        reasons.append("Manifest subject schema_version does not match the packaged report.")
    receipt = report.get("receipt")
    expected_receipt_hash = receipt.get("canonical_hash") if isinstance(receipt, Mapping) else None
    if subject.canonical_receipt_sha256 != expected_receipt_hash:
        reasons.append(
            "Manifest subject canonical_receipt_sha256 does not match the packaged report receipt."
        )

    expected_entailment = _evaluate_entailment(report)
    if manifest.entailment != expected_entailment:
        reasons.append(
            "Manifest entailment result does not match the re-derived report interval check."
        )
    expected_independence = _no_independence_result()
    if manifest.independence != expected_independence:
        reasons.append(
            "Manifest independence result does not match claim-package v1's available evidence."
        )
    expected_reproducibility = _reproducibility_projection(manifest.reproducibility.fixed_now)
    if manifest.reproducibility != expected_reproducibility:
        reasons.append(
            "Manifest reproducibility commands do not match the re-derived package commands."
        )

    try:
        envelope = compile_claim_envelope(report, governance_audit=governance)
        expected = {
            "source_governance_audit_sha256": _surface_sha256(
                governance.model_dump(mode="json", by_alias=True)
            ),
            "claim_envelope_sha256": _surface_sha256(envelope.to_canonical_dict()),
            "lifecycle_projection_sha256": _surface_sha256(
                _lifecycle_projection(governance).model_dump(mode="json", by_alias=True)
            ),
            "review_status_sha256": _surface_sha256(
                _review_status(governance, expected_artifacts).model_dump(
                    mode="json", by_alias=True
                )
            ),
            "readme_sha256": _text_sha256(
                _render_readme(
                    governance,
                    expected_artifacts,
                    expected_entailment,
                    expected_independence,
                )
            ),
            "challenge_doc_sha256": _text_sha256(_render_challenge_doc(expected_artifacts)),
        }
    except Exception as exc:  # pragma: no cover - defensive re-derivation boundary
        return [*reasons, f"Manifest integrity checks cannot be re-derived: {exc}"]

    recorded = manifest.integrity_checks.model_dump(mode="json", by_alias=True)
    for key, expected_hash in expected.items():
        if recorded.get(key) != expected_hash:
            reasons.append(
                f"Manifest integrity check {key} does not match the re-derived surface; "
                "the manifest's recorded hash is not authoritative."
            )

    # The claim boundary and the lifecycle/review projections are re-derived from
    # the receipt-bound report and governance audit, so a tampered non-claim
    # string or projection value cannot ride along inside the manifest.
    expected_non_claims = tuple(
        dict.fromkeys(
            [
                PACKAGE_PASS_CAVEAT,
                PACKAGE_INTEGRITY_NON_CLAIM,
                PACKAGE_LIFECYCLE_NON_CLAIM,
                *governance.non_claims,
                *envelope.boundary.non_claims,
            ]
        )
    )
    if manifest.non_claims != expected_non_claims:
        reasons.append("Manifest non-claims do not match the re-derived claim boundary.")
    if manifest.lifecycle_state.model_dump(mode="json", by_alias=True) != _lifecycle_projection(
        governance
    ).model_dump(mode="json", by_alias=True):
        reasons.append("Manifest lifecycle projection does not match the re-derived verdict.")
    if manifest.human_review_status.model_dump(mode="json", by_alias=True) != _review_status(
        governance, expected_artifacts
    ).model_dump(mode="json", by_alias=True):
        reasons.append("Manifest human-review projection does not match the re-derived audit.")

    # The recorded verifier result is bound to the stored audit, not asserted by
    # the manifest. Without this, flipping ``required_human_review`` to false in
    # the manifest would let a package understate its own review obligation while
    # still verifying. The stored audit is itself hash-bound and re-derived above,
    # so checking against it holds at the recorded time and under a freshness run.
    stored_audit_path = root / manifest.verifier_result.audit_path
    try:
        stored = _load_model_json(
            stored_audit_path, ClaimGovernanceAudit, "stored governance audit"
        )
    except (OSError, ValueError, ValidationError) as exc:
        reasons.append(
            f"Manifest verifier result cannot be re-derived from the stored audit: {exc}"
        )
        return reasons
    if manifest.verifier_result.verdict != stored.verdict.value:
        reasons.append("Manifest verifier verdict does not match the stored governance audit.")
    if manifest.verifier_result.required_human_review != stored.required_human_review:
        reasons.append("Manifest required_human_review does not match the stored governance audit.")
    if manifest.verifier_result.evaluated_at != stored.evaluated_at:
        reasons.append("Manifest verifier evaluated_at does not match the stored governance audit.")
    return reasons


def _artifacts_projection(artifacts: Sequence[PackageArtifact]) -> list[dict[str, Any]]:
    return [
        {
            "artifact_id": artifact.artifact_id,
            "role": artifact.role,
            "package_path": artifact.package_path,
            "sha256": artifact.sha256,
            "bytes": artifact.bytes,
        }
        for artifact in artifacts
    ]


def _surface_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a generated surface exactly as ``_write_json`` serializes it to disk."""

    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _support_edges_match(
    root: Path,
    manifest: ClaimPackageManifest,
    governance: ClaimGovernanceAudit,
) -> bool:
    try:
        report = _read_json_object(root / "report.json", "packaged report")
        envelope = _load_model_json(
            root / "envelope" / "claim_envelope.json", ClaimEnvelope, "claim envelope"
        )
        recomputed = compile_claim_envelope(report, governance_audit=governance)
    except (OSError, ValueError, ValidationError):
        return False
    return (
        envelope.support_graph.support_edges == manifest.support_edges
        and recomputed.support_graph.support_edges == manifest.support_edges
    )


def _file_matches(
    path: Path,
    expected_hash: str,
    expected_bytes: int | None,
) -> tuple[bool, str, str | None, int | None]:
    if not path.is_file():
        return False, "file is missing", None, None
    actual_bytes = path.stat().st_size
    actual_hash = sha256_file(path)
    if expected_bytes is not None and actual_bytes != expected_bytes:
        return (
            False,
            f"byte count mismatch (expected {expected_bytes}, got {actual_bytes})",
            actual_hash,
            actual_bytes,
        )
    if actual_hash != expected_hash:
        return (
            False,
            f"SHA-256 mismatch (expected {expected_hash}, got {actual_hash})",
            actual_hash,
            actual_bytes,
        )
    return True, "hash and byte count match", actual_hash, actual_bytes


def _package_failure_audit(*, package_id: str, evaluated_at: str, reason: str) -> ClaimPackageAudit:
    return ClaimPackageAudit(
        package_id=package_id,
        evaluated_at=evaluated_at,
        verdict="fail",
        integrity_verdict="fail",
        entailment=PackageEntailmentResult(
            status="fail",
            check="unavailable",
            reason="Entailment could not be evaluated because package verification failed: "
            + reason,
        ),
        independence=_no_independence_result(),
        manifest_valid=False,
        report_integrity_valid=False,
        artifacts=(),
        governance_verdict="fail",
        support_edges_preserved=False,
        reasons=(reason,),
        non_claims=(
            PACKAGE_PASS_CAVEAT,
            PACKAGE_INTEGRITY_NON_CLAIM,
            PACKAGE_LIFECYCLE_NON_CLAIM,
        ),
    )


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    value = strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _load_model_json(path: Path, model_type: Any, label: str) -> Any:
    """Strict-check duplicate keys before asking Pydantic to parse JSON types.

    Pydantic's JSON mode correctly maps JSON arrays back to strict tuple fields.
    The preliminary ``strict_json_loads`` preserves the repository's no-duplicate
    key policy before the model parser sees the document.
    """

    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    strict_json_loads(raw)
    return model_type.model_validate_json(raw)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _text_sha256(text: str) -> str:
    """Hash a generated text surface exactly as it is written to disk."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file_at(path: Path) -> str:
    if not path.is_file():
        raise ClaimPackageError(f"Generated package surface is missing: {path}")
    return sha256_file(path)


def _normalized_time(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ClaimPackageError("Verification time must be timezone-aware")
    return value.astimezone(timezone.utc)


def _parse_aware_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_package_id(report: Mapping[str, Any]) -> str:
    report_id = str(report.get("report_id") or "claim")
    return f"claim-package-{_slug(report_id)}"


def _slug(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value.strip())
    return "-".join(part for part in cleaned.split("-") if part) or "artifact"


def _is_within(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True


def _validate_destination_collisions(artifacts: Sequence[PackageArtifact]) -> None:
    """Reject file-vs-directory collisions before creating a package tree."""

    paths = {artifact.package_path for artifact in artifacts}
    for path in sorted(paths):
        prefix = f"{path}/"
        if any(other.startswith(prefix) for other in paths if other != path):
            raise ClaimPackageError(
                "Report evidence paths cannot be copied portably because a file path "
                f"collides with a descendant path: {path}"
            )


def _require_nonempty(value: str, field: str) -> None:
    if not value.strip():
        raise ClaimPackageError(f"{field} must be non-empty")


__all__ = [
    "CLAIM_PACKAGE_AUDIT_SCHEMA_VERSION",
    "CLAIM_PACKAGE_LIFECYCLE_PROJECTION_SCHEMA_VERSION",
    "CLAIM_PACKAGE_MANIFEST_SCHEMA_VERSION",
    "CLAIM_PACKAGE_REVIEW_STATUS_SCHEMA_VERSION",
    "PACKAGE_INTEGRITY_NON_CLAIM",
    "PACKAGE_LIFECYCLE_NON_CLAIM",
    "PACKAGE_PASS_CAVEAT",
    "ClaimPackageAudit",
    "ClaimPackageError",
    "ClaimPackageManifest",
    "PackageArtifact",
    "PackageArtifactAudit",
    "PackageEntailmentResult",
    "PackageHumanReviewStatus",
    "PackageIndependenceResult",
    "PackageIntegrityChecks",
    "PackageLifecycleProjection",
    "PackageReproducibility",
    "PackageSubjectReport",
    "PackageVerifierResult",
    "compile_claim_package",
    "verify_claim_package",
]

"""Build machine-checkable CC reports with canonical SHA-256 receipts."""

from __future__ import annotations

import hashlib
import json
import platform as platform_module
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cc import __version__ as framework_version_default
from cc.reporting.canonical import sha256_canonical

SCHEMA_VERSION = "cc.report.v0.3.1"
HASH_ALGORITHM = "sha256"
CANONICALIZATION_METHOD = (
    "json.dumps(sort_keys=True,separators=(',', ':'),ensure_ascii=False,allow_nan=False); "
    "receipt.canonical_hash excluded"
)
CLAIM_LEVELS = ("diagnostic", "bounded_empirical", "reproducible_run", "release_claim")
ALLOWED_CLAIM_LEVELS = frozenset(CLAIM_LEVELS)
CLAIM_LEVEL_DESCRIPTIONS = {
    "diagnostic": "Exploratory or debugging evidence only; not a release or safety claim.",
    "bounded_empirical": (
        "A measured bound or interval scoped to the named run, evaluation distribution, "
        "calibration window, and assumptions."
    ),
    "reproducible_run": (
        "A receipt for rerunning or auditing the named run and artifacts; not a claim "
        "that conclusions transfer outside that setup."
    ),
    "release_claim": (
        "A release-gate claim only within an external review process; not standalone "
        "certification of production safety or compliance."
    ),
}
CALIBRATION_STATUSES = frozenset({"pass", "fail"})


class ReportValidationError(ValueError):
    """Raised when report inputs would produce an invalid CC report."""


@dataclass(frozen=True)
class GitMetadata:
    commit: str | None
    dirty: bool
    branch: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"commit": self.commit, "dirty": self.dirty, "branch": self.branch}


@dataclass(frozen=True)
class EnvironmentMetadata:
    python_version: str
    platform: str
    dependency_hash: str | None = None
    package_snapshot: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "python_version": self.python_version,
            "platform": self.platform,
        }
        if self.dependency_hash is not None:
            payload["dependency_hash"] = self.dependency_hash
        if self.package_snapshot is not None:
            payload["package_snapshot"] = dict(sorted(self.package_snapshot.items()))
        return payload


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    config_path: str | None = None
    config_hash: str | None = None
    seed: int | None = None
    command: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_path": self.config_path,
            "config_hash": self.config_hash,
            "seed": self.seed,
            "command": self.command,
        }


@dataclass(frozen=True)
class CalibrationSummary:
    status: str
    target_fpr: float | None = None
    alpha_cap: float | None = None
    realized_fpr: float | None = None
    calibration_window: Mapping[str, Any] | None = None
    threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_fpr": self.target_fpr,
            "alpha_cap": self.alpha_cap,
            "realized_fpr": self.realized_fpr,
            "calibration_window": dict(self.calibration_window or {}),
            "threshold": self.threshold,
            "status": self.status,
        }


@dataclass(frozen=True)
class MeasurementSummary:
    metric_family: str
    point_estimate: float
    interval_lower: float
    interval_upper: float
    interval_method: str
    sample_sizes: Mapping[str, int]
    confidence_level: float | None = None
    delta: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_family": self.metric_family,
            "point_estimate": self.point_estimate,
            "interval": {
                "lower": self.interval_lower,
                "upper": self.interval_upper,
            },
            "confidence_level": self.confidence_level,
            "delta": self.delta,
            "interval_method": self.interval_method,
            "sample_sizes": dict(sorted(self.sample_sizes.items())),
        }


@dataclass(frozen=True)
class EvidenceArtifact:
    path: str
    sha256: str
    bytes: int
    role: str = "artifact"

    @classmethod
    def from_path(cls, path: str | Path, *, role: str = "artifact") -> EvidenceArtifact:
        artifact_path = Path(path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Evidence file not found: {artifact_path}")
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Evidence path is not a file: {artifact_path}")
        return cls(
            path=str(path),
            sha256=sha256_file(artifact_path),
            bytes=artifact_path.stat().st_size,
            role=role,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "role": self.role,
        }


@dataclass(frozen=True)
class ClaimSummary:
    statement: str
    allowed_claim_level: str
    non_claims: Sequence[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "allowed_claim_level": self.allowed_claim_level,
            "non_claims": list(self.non_claims),
        }


def build_cc_report(
    *,
    run: RunSummary,
    calibration: CalibrationSummary,
    measurement: MeasurementSummary,
    claim: ClaimSummary,
    evidence_artifacts: Sequence[EvidenceArtifact] = (),
    audit_log: EvidenceArtifact | None = None,
    figure_manifest: EvidenceArtifact | None = None,
    assumptions: Sequence[str] = (),
    report_id: str | None = None,
    created_at: str | None = None,
    framework_version: str | None = None,
    git: GitMetadata | None = None,
    environment: EnvironmentMetadata | None = None,
    previous_hash: str | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """Build and validate a CC report dictionary with a canonical receipt."""

    _validate_inputs(calibration=calibration, measurement=measurement, claim=claim)

    created = created_at or _utc_now()
    report = {
        "schema_version": SCHEMA_VERSION,
        "report_id": report_id or f"cc-report-{run.run_id}",
        "created_at": created,
        "framework_version": framework_version or framework_version_default,
        "git": (git or detect_git_metadata(cwd=cwd)).to_dict(),
        "environment": (environment or detect_environment_metadata()).to_dict(),
        "run": run.to_dict(),
        "assumptions": list(assumptions),
        "calibration": calibration.to_dict(),
        "measurement": measurement.to_dict(),
        "evidence": {
            "artifacts": [artifact.to_dict() for artifact in evidence_artifacts],
            "audit_log": audit_log.to_dict() if audit_log is not None else None,
            "figure_manifest": figure_manifest.to_dict() if figure_manifest is not None else None,
        },
        "claim": claim.to_dict(),
        "receipt": {
            "canonical_hash": None,
            "hash_algorithm": HASH_ALGORITHM,
            "canonicalization_method": CANONICALIZATION_METHOD,
            "previous_hash": previous_hash,
        },
    }
    report["receipt"]["canonical_hash"] = sha256_canonical(report)
    return report


def write_cc_report(path: str | Path, report: Mapping[str, Any]) -> None:
    """Write a report JSON file in a deterministic pretty-printed form."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_git_metadata(*, cwd: Path | None = None) -> GitMetadata:
    commit = _git(["rev-parse", "HEAD"], cwd=cwd)
    dirty = bool(_git(["status", "--porcelain"], cwd=cwd))
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    if branch == "HEAD":
        branch = None
    return GitMetadata(commit=commit, dirty=dirty, branch=branch)


def detect_environment_metadata() -> EnvironmentMetadata:
    return EnvironmentMetadata(
        python_version=sys.version.split()[0],
        platform=platform_module.platform(),
    )


def _validate_inputs(
    *,
    calibration: CalibrationSummary,
    measurement: MeasurementSummary,
    claim: ClaimSummary,
) -> None:
    if calibration.status not in CALIBRATION_STATUSES:
        raise ReportValidationError("calibration.status must be one of: pass, fail")
    if calibration.target_fpr is None and calibration.alpha_cap is None:
        raise ReportValidationError("calibration must include target_fpr or alpha_cap")

    if not measurement.metric_family.strip():
        raise ReportValidationError("measurement.metric_family cannot be empty")
    if not measurement.interval_method.strip():
        raise ReportValidationError("measurement.interval_method cannot be empty")
    if measurement.confidence_level is None and measurement.delta is None:
        raise ReportValidationError("measurement must include confidence_level or delta")
    if measurement.interval_lower > measurement.interval_upper:
        raise ReportValidationError("measurement interval lower cannot exceed upper")

    if not claim.statement.strip():
        raise ReportValidationError("claim.statement cannot be empty")
    non_claims = list(claim.non_claims)
    if any(not isinstance(item, str) or not item.strip() for item in non_claims):
        raise ReportValidationError("claim.non_claims must contain non-empty strings")

    if claim.allowed_claim_level not in ALLOWED_CLAIM_LEVELS:
        allowed = ", ".join(sorted(ALLOWED_CLAIM_LEVELS))
        raise ReportValidationError(f"claim.allowed_claim_level must be one of: {allowed}")
    if claim.allowed_claim_level != "diagnostic":
        if not (
            measurement.interval_lower <= measurement.point_estimate <= measurement.interval_upper
        ):
            raise ReportValidationError(
                "measurement point_estimate must lie within interval for non-diagnostic claims"
            )
        if not non_claims:
            raise ReportValidationError("non-diagnostic claims require explicit non_claims")

    for label, sample_size in measurement.sample_sizes.items():
        if not isinstance(label, str) or not label:
            raise ReportValidationError("sample size labels must be non-empty strings")
        if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 0:
            raise ReportValidationError("sample sizes must be non-negative integers")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git(args: Sequence[str], *, cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    return value or None

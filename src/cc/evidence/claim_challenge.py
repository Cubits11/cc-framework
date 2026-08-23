"""Falsify a claim package's own tamper-evidence claim.

A compiled claim package asserts one narrow, checkable property: it is
*tamper-evident*. If any byte of the byte-bound report, the bound evidence, the
manifest, or the generated audit surfaces is altered, package verification must
fall to ``FAIL``. This module is the adversary that tries to disprove that
assertion, so a recipient does not have to trust the compiler that made the
package.

It is deliberately a *separate module* from the compiler. The builder and the
attacker share no private assumptions: the challenge re-derives everything from
the shipped ``manifest.json`` and the public ``verify_claim_package`` entry
point, mutating one byte of each surface on a scratch copy and recording whether
the verdict fell. The original package is never modified.

What a passing challenge establishes is exactly as narrow as the property it
tests: that the specific single-byte mutations applied to each named surface are
all detected, and that the untouched control reproduces the recorded verdict (so
a harness that merely always fails cannot pass). It is not a proof that no
undetectable modification exists, and — like every surface in this subsystem —
integrity is not validity.
"""

from __future__ import annotations

import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from cc.evidence.claim_compiler import (
    PACKAGE_INTEGRITY_NON_CLAIM,
    PACKAGE_LIFECYCLE_NON_CLAIM,
    PACKAGE_PASS_CAVEAT,
    ClaimPackageManifest,
    ClaimPackageModel,
    _load_model_json,
    _normalized_time,
    _parse_aware_time,
    _utc_iso,
    verify_claim_package,
)

CLAIM_PACKAGE_CHALLENGE_SCHEMA_VERSION: Literal["cc.claim_package_challenge.v1"] = (
    "cc.claim_package_challenge.v1"
)

CHALLENGE_COMPLETENESS_NON_CLAIM = (
    "The challenge demonstrates detection of the specific single-byte mutations it applies to "
    "each named surface; it is not a proof that no undetectable modification of the package exists."
)

# Surfaces every package carries that must be verdict-bearing. Bound evidence
# artifacts are discovered from the manifest and appended to this fixed set.
_FIXED_SURFACES: tuple[str, ...] = (
    "report.json",
    "manifest.json",
    "audits/claim_governance_audit.json",
    "envelope/claim_envelope.json",
    "lifecycle/projection.json",
    "reviews/review_status.json",
)


class ClaimPackageChallengeSurface(ClaimPackageModel):
    """The outcome of mutating one bound surface and re-verifying."""

    package_path: str = Field(min_length=1)
    mutation: Literal["flip_byte", "append_byte"]
    verdict_after_mutation: Literal["pass", "needs_review", "fail"]
    detected: bool
    reason: str = Field(min_length=1)


class ClaimPackageChallengeReport(ClaimPackageModel):
    """Whether a package survives an adversarial byte-mutation challenge."""

    schema_: Literal["cc.claim_package_challenge.v1"] = Field(
        default=CLAIM_PACKAGE_CHALLENGE_SCHEMA_VERSION,
        alias="schema",
    )
    package_id: str = Field(min_length=1)
    evaluated_at: str = Field(min_length=1)
    control_verdict: Literal["pass", "needs_review", "fail"]
    control_reproduced: bool
    surfaces: tuple[ClaimPackageChallengeSurface, ...] = Field(default_factory=tuple)
    tamper_evident: bool
    reasons: tuple[str, ...] = Field(default_factory=tuple)
    non_claims: tuple[str, ...] = Field(default_factory=tuple)


def challenge_claim_package(
    package_dir: Path,
    *,
    now: datetime | None = None,
    strict_unknown_roles: bool = False,
) -> ClaimPackageChallengeReport:
    """Attempt to defeat a package's tamper-evidence, without modifying it.

    The challenge copies the package to a scratch directory and, for each bound
    surface, applies the minimal single-byte mutation, re-runs
    ``verify_claim_package`` at the package's recorded time, and records whether
    the verdict fell to ``fail``. A control run over the untouched copy must
    reproduce the recorded verdict. ``tamper_evident`` is true only if the
    control reproduced *and* every mutated surface was detected.

    Passing ``now`` runs the whole challenge at that time instead of the
    recorded time; the control then checks reproduction against a fresh verdict.
    """

    root = Path(package_dir).resolve()
    fallback_time = _normalized_time(now)
    manifest_path = root / "manifest.json"

    if not root.is_dir():
        return _challenge_failure(
            package_id="<unknown>",
            evaluated_at=_utc_iso(fallback_time),
            reason=f"Package directory not found: {root}",
        )
    try:
        manifest = _load_model_json(manifest_path, ClaimPackageManifest, "manifest")
    except (OSError, ValueError, ValidationError) as exc:
        return _challenge_failure(
            package_id="<unknown>",
            evaluated_at=_utc_iso(fallback_time),
            reason=f"Package manifest is unreadable or invalid: {exc}",
        )

    if now is None:
        try:
            effective_time = _parse_aware_time(manifest.reproducibility.fixed_now)
        except ValueError as exc:
            return _challenge_failure(
                package_id=manifest.package_id,
                evaluated_at=_utc_iso(fallback_time),
                reason=f"Package manifest fixed_now is invalid: {exc}",
            )
    else:
        effective_time = fallback_time

    surfaces_to_test = _surfaces_for(manifest)
    reasons: list[str] = []

    temp_root = Path(tempfile.mkdtemp(prefix="cc-claim-challenge-"))
    try:
        work = temp_root / root.name
        shutil.copytree(root, work)

        control = verify_claim_package(
            work, now=effective_time, strict_unknown_roles=strict_unknown_roles
        )
        control_verdict = control.verdict
        control_reproduced = control_verdict == manifest.verifier_result.verdict
        if not control_reproduced:
            reasons.append(
                "Control run did not reproduce the recorded verdict "
                f"(recorded={manifest.verifier_result.verdict}, control={control_verdict}); "
                "the package must self-verify before its tamper-evidence can be challenged."
            )

        surface_results: list[ClaimPackageChallengeSurface] = []
        for relative in surfaces_to_test:
            target = work / relative
            if not target.is_file():
                surface_results.append(
                    ClaimPackageChallengeSurface(
                        package_path=relative,
                        mutation="flip_byte",
                        verdict_after_mutation="fail",
                        detected=False,
                        reason="surface missing from package; a bound surface cannot be absent",
                    )
                )
                reasons.append(f"Expected bound surface is missing: {relative}")
                continue
            original = target.read_bytes()
            mutation, mutated = _minimal_mutation(original)
            try:
                target.write_bytes(mutated)
                audit = verify_claim_package(
                    work, now=effective_time, strict_unknown_roles=strict_unknown_roles
                )
                verdict = audit.verdict
            finally:
                target.write_bytes(original)
            detected = verdict == "fail"
            surface_results.append(
                ClaimPackageChallengeSurface(
                    package_path=relative,
                    mutation=mutation,
                    verdict_after_mutation=verdict,
                    detected=detected,
                    reason=(
                        "mutation detected: verdict fell to FAIL"
                        if detected
                        else f"mutation NOT detected: verdict stayed {verdict.upper()}"
                    ),
                )
            )
            if not detected:
                reasons.append(
                    f"Undetected mutation of {relative}: verdict stayed {verdict.upper()}."
                )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    tamper_evident = control_reproduced and all(s.detected for s in surface_results)
    return ClaimPackageChallengeReport(
        package_id=manifest.package_id,
        evaluated_at=_utc_iso(effective_time),
        control_verdict=control_verdict,
        control_reproduced=control_reproduced,
        surfaces=tuple(surface_results),
        tamper_evident=tamper_evident,
        reasons=tuple(dict.fromkeys(reasons)),
        non_claims=(
            CHALLENGE_COMPLETENESS_NON_CLAIM,
            PACKAGE_INTEGRITY_NON_CLAIM,
            PACKAGE_PASS_CAVEAT,
            PACKAGE_LIFECYCLE_NON_CLAIM,
        ),
    )


def _surfaces_for(manifest: ClaimPackageManifest) -> tuple[str, ...]:
    ordered = list(_FIXED_SURFACES)
    for artifact in manifest.artifacts:
        if artifact.package_path not in ordered:
            ordered.append(artifact.package_path)
    return tuple(ordered)


def _minimal_mutation(data: bytes) -> tuple[Literal["flip_byte", "append_byte"], bytes]:
    """Return the smallest mutation that changes the bytes.

    A non-empty surface has its middle byte flipped by one bit — enough to break
    a SHA-256 binding or a JSON parse without depending on structural position.
    An empty surface is grown by one byte, since there is no byte to flip.
    """

    if not data:
        return "append_byte", b"\x00"
    index = len(data) // 2
    mutated = bytearray(data)
    mutated[index] ^= 0x01
    return "flip_byte", bytes(mutated)


def _challenge_failure(
    *, package_id: str, evaluated_at: str, reason: str
) -> ClaimPackageChallengeReport:
    return ClaimPackageChallengeReport(
        package_id=package_id,
        evaluated_at=evaluated_at,
        control_verdict="fail",
        control_reproduced=False,
        surfaces=(),
        tamper_evident=False,
        reasons=(reason,),
        non_claims=(
            CHALLENGE_COMPLETENESS_NON_CLAIM,
            PACKAGE_INTEGRITY_NON_CLAIM,
            PACKAGE_PASS_CAVEAT,
            PACKAGE_LIFECYCLE_NON_CLAIM,
        ),
    )


def render_challenge_report(report: ClaimPackageChallengeReport) -> str:
    """One human-readable line per surface, for CLI output."""

    lines = [
        f"Challenge: {report.package_id}",
        f"Control verdict: {report.control_verdict.upper()} "
        f"(reproduced recorded verdict: {report.control_reproduced})",
    ]
    for surface in report.surfaces:
        mark = "detected" if surface.detected else "UNDETECTED"
        lines.append(f"  [{mark}] {surface.package_path} — {surface.reason}")
    lines.append(
        f"Tamper-evident: {report.tamper_evident}"
        + ("" if report.tamper_evident else " — " + "; ".join(report.reasons))
    )
    return "\n".join(lines)


__all__ = [
    "CHALLENGE_COMPLETENESS_NON_CLAIM",
    "CLAIM_PACKAGE_CHALLENGE_SCHEMA_VERSION",
    "ClaimPackageChallengeReport",
    "ClaimPackageChallengeSurface",
    "challenge_claim_package",
    "render_challenge_report",
]

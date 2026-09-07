"""The claim-package compiler must fail closed against every tamper vector.

The compiler's one promise is tamper-evidence: a compiled package copies an
already-verified report and its bound evidence into a portable layout, records
what it should contain, and re-verifies. If any byte of the report, the bound
evidence, or the generated audit surfaces is altered — or if the manifest is
edited to lie about any of them — package verification must fall to FAIL.

These tests assert exactly that, plus the compile-time refusals (expired
governance, non-portable evidence paths, existing destination) and the two
verification modes (reproduce at the recorded time, freshness at a supplied
time). A green run here is the evidence behind the tamper-evidence claim; the
adversarial challenge in test_claim_challenge.py is the second, independent one.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cc.evidence import (
    ClaimPackageError,
    ClaimPackageManifest,
    compile_claim_package,
    verify_claim_package,
)
from cc.evidence.claim_compiler import _source_artifacts_from_report
from cc.reporting.canonical import sha256_canonical

ROOT = Path(__file__).resolve().parents[3]
CAPSULE = ROOT / "examples" / "claim_governance_capsule" / "expected"
REPORT = CAPSULE / "cc_report.json"
RECORDED = datetime(2026, 1, 2, tzinfo=timezone.utc)
EXPIRED = datetime(2026, 8, 23, tzinfo=timezone.utc)


@pytest.fixture
def package(tmp_path: Path) -> Path:
    out = tmp_path / "pkg"
    compile_claim_package(REPORT, out, base_dir=CAPSULE, now=RECORDED)
    return out


# --------------------------------------------------------------------------- #
# Happy path                                                                  #
# --------------------------------------------------------------------------- #


def test_public_api_exports() -> None:
    assert callable(compile_claim_package)
    assert callable(verify_claim_package)
    assert ClaimPackageManifest.__name__ == "ClaimPackageManifest"
    assert issubclass(ClaimPackageError, ValueError)


def test_compile_and_self_verify_pass(package: Path) -> None:
    audit = verify_claim_package(package, now=RECORDED)
    assert audit.verdict == "pass"
    assert audit.integrity_verdict == "pass"
    assert audit.entailment.status == "not_checked"
    assert audit.independence.status == "none"
    assert audit.report_integrity_valid is True
    assert audit.support_edges_preserved is True
    assert audit.governance_verdict == "pass"
    assert all(a.valid for a in audit.artifacts)


def test_package_ships_layout_and_docs(package: Path) -> None:
    for relative in (
        "report.json",
        "manifest.json",
        "README.md",
        "CHALLENGE.md",
        "audits/claim_governance_audit.json",
        "envelope/claim_envelope.json",
        "lifecycle/projection.json",
        "reviews/review_status.json",
    ):
        assert (package / relative).is_file(), f"missing {relative}"
    assert list((package / "evidence").glob("*")), "no bound evidence copied"


def test_report_copied_byte_for_byte(package: Path) -> None:
    assert (package / "report.json").read_bytes() == REPORT.read_bytes()


def test_manifest_records_working_reproduce_and_challenge_commands(package: Path) -> None:
    manifest = json.loads((package / "manifest.json").read_text())
    repro = manifest["reproducibility"]
    assert repro["package_verification_command"].startswith(
        "python -m cc.reporting.cli verify-claim-package ."
    )
    assert repro["challenge_command"] == "python -m cc.reporting.cli challenge-claim-package ."


def test_unstructured_claim_prose_is_not_semantically_checked(package: Path) -> None:
    """Free text must remain NOT_CHECKED rather than inherit integrity PASS."""

    audit = verify_claim_package(package, now=RECORDED)

    assert audit.integrity_verdict == "pass"
    assert audit.entailment.status == "not_checked"
    assert audit.entailment.check == "not_checked"
    assert audit.independence.status == "none"
    assert "not parsed" in audit.entailment.reason


def test_structured_false_upper_bound_fails_entailment_but_not_integrity(tmp_path: Path) -> None:
    """A reissued report cannot turn an interval-inconsistent bound into semantic PASS."""

    source = tmp_path / "source"
    shutil.copytree(CAPSULE, source)
    source_report = source / "cc_report.json"
    report = json.loads(source_report.read_text(encoding="utf-8"))
    report["claim"]["quantitative_proposition"] = {
        "metric_family": "CC",
        "relation": "upper_bound",
        "threshold": 0.01,
    }
    _reissue_receipt(report)
    source_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    package = tmp_path / "pkg"
    manifest = compile_claim_package(source_report, package, base_dir=source, now=RECORDED)
    audit = verify_claim_package(package, now=RECORDED)

    assert manifest.entailment.status == "fail"
    assert audit.integrity_verdict == "pass"
    assert audit.entailment.status == "fail"
    assert audit.entailment.check == "structured_measurement_interval"
    assert audit.independence.status == "none"
    assert audit.verdict == "fail"
    assert "interval upper" in audit.entailment.reason


def test_structured_true_upper_bound_passes_entailment(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(CAPSULE, source)
    source_report = source / "cc_report.json"
    report = json.loads(source_report.read_text(encoding="utf-8"))
    report["claim"]["quantitative_proposition"] = {
        "metric_family": "CC",
        "relation": "upper_bound",
        "threshold": 0.2,
    }
    _reissue_receipt(report)
    source_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    package = tmp_path / "pkg"
    compile_claim_package(source_report, package, base_dir=source, now=RECORDED)
    audit = verify_claim_package(package, now=RECORDED)

    assert audit.verdict == "pass"
    assert audit.integrity_verdict == "pass"
    assert audit.entailment.status == "pass"
    assert audit.independence.status == "none"


# --------------------------------------------------------------------------- #
# Compile-time refusals                                                       #
# --------------------------------------------------------------------------- #


def test_existing_destination_refused(tmp_path: Path) -> None:
    out = tmp_path / "pkg"
    out.mkdir()
    with pytest.raises(ClaimPackageError, match="already exists"):
        compile_claim_package(REPORT, out, base_dir=CAPSULE, now=RECORDED)


def test_expired_governance_refused(tmp_path: Path) -> None:
    with pytest.raises(ClaimPackageError, match=r"governance verification failed|expired"):
        compile_claim_package(REPORT, tmp_path / "pkg", base_dir=CAPSULE, now=EXPIRED)


def test_missing_report_refused(tmp_path: Path) -> None:
    with pytest.raises(ClaimPackageError, match="not found"):
        compile_claim_package(tmp_path / "nope.json", tmp_path / "pkg", now=RECORDED)


def test_require_pass_accepts_a_passing_capsule(tmp_path: Path) -> None:
    manifest = compile_claim_package(
        REPORT, tmp_path / "pkg", base_dir=CAPSULE, now=RECORDED, require_pass=True
    )
    assert manifest.verifier_result.verdict == "pass"


@pytest.mark.parametrize("evil", ["../escape.json", "/abs/path.json", "a/../../b.json"])
def test_non_portable_evidence_paths_refused(evil: str) -> None:
    report = {
        "evidence": {"artifacts": [{"path": evil, "sha256": "0" * 64, "bytes": 1, "role": "x"}]}
    }
    with pytest.raises(ClaimPackageError):
        _source_artifacts_from_report(report, CAPSULE)


# --------------------------------------------------------------------------- #
# Tamper vectors — each must drive verification to FAIL                       #
# --------------------------------------------------------------------------- #


def _flip_middle_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))


def _reissue_receipt(report: dict[str, object]) -> None:
    receipt = report["receipt"]
    assert isinstance(receipt, dict)
    profile = receipt["canonicalization_method"]
    assert isinstance(profile, str)
    receipt["canonical_hash"] = sha256_canonical(report, profile=profile)


def _first_evidence_file(package: Path) -> Path:
    return sorted((package / "evidence").glob("*"))[0]


@pytest.mark.parametrize(
    "surface",
    [
        "report.json",
        "manifest.json",
        "audits/claim_governance_audit.json",
        "envelope/claim_envelope.json",
        "lifecycle/projection.json",
        "reviews/review_status.json",
        "README.md",
        "CHALLENGE.md",
    ],
)
def test_byte_tamper_of_each_surface_fails_closed(package: Path, surface: str) -> None:
    _flip_middle_byte(package / surface)
    audit = verify_claim_package(package, now=RECORDED)
    assert audit.verdict == "fail", f"{surface} tamper was not detected"


def test_byte_tamper_of_bound_evidence_fails_closed(package: Path) -> None:
    target = _first_evidence_file(package)
    _flip_middle_byte(target)
    audit = verify_claim_package(package, now=RECORDED)
    assert audit.verdict == "fail"
    assert any(not a.valid for a in audit.artifacts)


def test_deleted_bound_evidence_fails_closed(package: Path) -> None:
    _first_evidence_file(package).unlink()
    audit = verify_claim_package(package, now=RECORDED)
    assert audit.verdict == "fail"


def test_manifest_lying_about_an_artifact_hash_fails_closed(package: Path) -> None:
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # Repoint the first artifact's expected hash to a plausible but wrong value:
    # the real file no longer matches, and the re-derivation catches the lie too.
    manifest["artifacts"][0]["sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    audit = verify_claim_package(package, now=RECORDED)
    assert audit.verdict == "fail"


@pytest.mark.parametrize(
    ("path", "value", "reason_fragment"),
    [
        (
            ("subject_report", "canonical_receipt_sha256"),
            "f" * 64,
            "canonical_receipt_sha256",
        ),
        (
            ("reproducibility", "challenge_command"),
            "echo forged-challenge",
            "reproducibility commands",
        ),
    ],
)
def test_manifest_fields_are_rederived_not_trusted(
    package: Path,
    path: tuple[str, str],
    value: str,
    reason_fragment: str,
) -> None:
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[path[0]][path[1]] = value
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    audit = verify_claim_package(package, now=RECORDED)

    assert audit.integrity_verdict == "fail"
    assert audit.verdict == "fail"
    assert any(reason_fragment in reason for reason in audit.reasons)


def test_appending_a_safety_claim_to_the_readme_fails_closed(package: Path) -> None:
    """The boundary text is a bound surface, not decoration.

    README.md carries the PASS caveat and every non-claim. Before these surfaces
    were bound, appending "this package CERTIFIES the system is SAFE" to it left
    verification at PASS -- so a recipient could be handed a package whose stated
    boundary had been silently inverted. That is the one misrepresentation this
    package exists to make impossible.
    """

    readme = package / "README.md"
    readme.write_text(
        readme.read_text() + "\n\nThis package CERTIFIES the system is SAFE for deployment.\n"
    )
    audit = verify_claim_package(package, now=RECORDED)
    assert audit.verdict == "fail"
    assert any("README" in reason for reason in audit.reasons)


def test_manifest_relabelling_a_tampered_readme_fails_closed(package: Path) -> None:
    """The manifest cannot bless boundary text the verifier can re-derive."""

    import hashlib

    readme = package / "README.md"
    readme.write_text(readme.read_text() + "\n\nThis package CERTIFIES safety.\n")
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["integrity_checks"]["readme_sha256"] = hashlib.sha256(readme.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    audit = verify_claim_package(package, now=RECORDED)
    assert audit.verdict == "fail", "a manifest that re-labels tampered boundary text must fail"
    assert any("readme_sha256" in reason for reason in audit.reasons)


def test_challenge_doc_is_a_bound_surface(package: Path) -> None:
    """CHALLENGE.md tells a recipient how to falsify the package; it must bind."""

    doc = package / "CHALLENGE.md"
    doc.write_text(doc.read_text().replace("Do not trust that claim", "Trust this claim"))
    assert verify_claim_package(package, now=RECORDED).verdict == "fail"


def test_manifest_smuggling_an_extra_non_claim_fails_closed(package: Path) -> None:
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["non_claims"] = [*manifest["non_claims"], "This system is certified safe."]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    audit = verify_claim_package(package, now=RECORDED)
    assert audit.verdict == "fail", "a manifest that smuggles a false non-claim must be rejected"


# --------------------------------------------------------------------------- #
# Verification modes                                                          #
# --------------------------------------------------------------------------- #


def test_reproduce_at_recorded_time_is_pass(package: Path) -> None:
    assert verify_claim_package(package).verdict == "pass"  # now=None reuses fixed_now


def test_freshness_check_at_expired_time_is_not_pass(package: Path) -> None:
    audit = verify_claim_package(package, now=EXPIRED)
    assert audit.verdict != "pass"
    assert audit.governance_verdict == "fail"

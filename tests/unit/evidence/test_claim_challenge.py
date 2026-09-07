"""The built-in falsifier must prove tamper-evidence, and refuse to fake it.

challenge_claim_package copies a package, mutates one byte of each bound surface
on the copy, and records whether verification fell to FAIL. The original is never
modified. tamper_evident is true only if an untouched control reproduces the
recorded verdict AND every mutated surface is detected — so a harness that merely
always fails cannot pass, and a package with an unbound surface cannot pass
either.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cc.evidence import challenge_claim_package, compile_claim_package, verify_claim_package
from cc.evidence.claim_challenge import (
    CHALLENGE_COMPLETENESS_NON_CLAIM,
    render_challenge_report,
)
from cc.reporting.canonical import sha256_canonical

ROOT = Path(__file__).resolve().parents[3]
CAPSULE = ROOT / "examples" / "claim_governance_capsule" / "expected"
REPORT = CAPSULE / "cc_report.json"
RECORDED = datetime(2026, 1, 2, tzinfo=timezone.utc)


@pytest.fixture
def package(tmp_path: Path) -> Path:
    out = tmp_path / "pkg"
    compile_claim_package(REPORT, out, base_dir=CAPSULE, now=RECORDED)
    return out


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_capsule_package_is_tamper_evident(package: Path) -> None:
    report = challenge_claim_package(package)
    assert report.tamper_evident is True
    assert report.control_reproduced is True
    assert report.control_verdict == "pass"
    assert len(report.surfaces) >= 8
    assert all(s.detected for s in report.surfaces)
    # every bound evidence artifact and the fixed surfaces were exercised
    tested = {s.package_path for s in report.surfaces}
    assert "report.json" in tested
    assert "manifest.json" in tested
    assert any(p.startswith("evidence/") for p in tested)


def test_challenge_never_modifies_the_original(package: Path) -> None:
    before = _tree_digest(package)
    challenge_claim_package(package)
    assert _tree_digest(package) == before
    assert verify_claim_package(package, now=RECORDED).verdict == "pass"


def test_challenge_reports_the_completeness_non_claim(package: Path) -> None:
    report = challenge_claim_package(package)
    assert CHALLENGE_COMPLETENESS_NON_CLAIM in report.non_claims
    # integrity is not validity, and the falsifier carries that boundary
    assert any("validity" in nc.lower() for nc in report.non_claims)
    assert len(report.non_claims) >= 4


def test_missing_package_is_not_tamper_evident(tmp_path: Path) -> None:
    report = challenge_claim_package(tmp_path / "does-not-exist")
    assert report.tamper_evident is False
    assert report.reasons


def test_a_package_missing_a_bound_surface_cannot_pass_the_challenge(
    package: Path, tmp_path: Path
) -> None:
    broken = tmp_path / "broken"
    shutil.copytree(package, broken)
    next(iter((broken / "evidence").glob("*"))).unlink()
    report = challenge_claim_package(broken)
    # the control no longer reproduces (verification already fails), so no honest
    # tamper-evidence can be asserted
    assert report.tamper_evident is False
    assert report.control_reproduced is False


def test_a_corrupt_manifest_cannot_pass_the_challenge(package: Path, tmp_path: Path) -> None:
    broken = tmp_path / "broken"
    shutil.copytree(package, broken)
    (broken / "manifest.json").write_text("{ not json", encoding="utf-8")
    report = challenge_claim_package(broken)
    assert report.tamper_evident is False


def test_render_is_one_line_per_surface(package: Path) -> None:
    report = challenge_claim_package(package)
    rendered = render_challenge_report(report)
    assert "Tamper-evident: True" in rendered
    assert rendered.count("[detected]") == len(report.surfaces)


def test_challenge_checks_integrity_even_when_structured_entailment_fails(tmp_path: Path) -> None:
    """The byte challenge must not mistake a semantic FAIL for broken integrity."""

    source = tmp_path / "source"
    shutil.copytree(CAPSULE, source)
    report_path = source / "cc_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["claim"]["quantitative_proposition"] = {
        "metric_family": "CC",
        "relation": "upper_bound",
        "threshold": 0.01,
    }
    receipt = payload["receipt"]
    receipt["canonical_hash"] = sha256_canonical(
        payload, profile=receipt["canonicalization_method"]
    )
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    package = tmp_path / "pkg"
    compile_claim_package(report_path, package, base_dir=source, now=RECORDED)
    audit = verify_claim_package(package, now=RECORDED)
    challenge = challenge_claim_package(package)

    assert audit.verdict == "fail"
    assert audit.integrity_verdict == "pass"
    assert audit.entailment.status == "fail"
    assert challenge.control_verdict == "fail"
    assert challenge.control_reproduced is True
    assert challenge.tamper_evident is True
    assert all(surface.integrity_verdict_after_mutation == "fail" for surface in challenge.surfaces)


# --------------------------------------------------------------------------- #
# CLI integration — the commands packages actually record                     #
# --------------------------------------------------------------------------- #


def test_cli_compile_verify_challenge_roundtrip(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from cc.reporting.cli import main

    out = tmp_path / "pkg"
    assert (
        main(
            [
                "compile-claim-package",
                str(REPORT),
                str(out),
                "--base-dir",
                str(CAPSULE),
                "--now",
                "2026-01-02T00:00:00Z",
            ]
        )
        == 0
    )
    compile_output = capsys.readouterr().out
    assert "Entailment: NOT_CHECKED" in compile_output
    assert "Independence: NONE" in compile_output
    assert main(["verify-claim-package", str(out), "--now", "2026-01-02T00:00:00Z"]) == 0
    verify_output = capsys.readouterr().out
    assert "Integrity verdict: PASS" in verify_output
    assert "Entailment: NOT_CHECKED" in verify_output
    assert "Independence: NONE" in verify_output
    assert main(["challenge-claim-package", str(out)]) == 0
    capsys.readouterr()

    # a tampered package makes the shipped verify command exit non-zero
    data = bytearray((out / "report.json").read_bytes())
    data[len(data) // 2] ^= 0x01
    (out / "report.json").write_bytes(bytes(data))
    assert main(["verify-claim-package", str(out), "--now", "2026-01-02T00:00:00Z"]) == 2


def test_cli_challenge_json_out(tmp_path: Path) -> None:
    from cc.reporting.cli import main

    out = tmp_path / "pkg"
    compile_claim_package(REPORT, out, base_dir=CAPSULE, now=RECORDED)
    report_path = tmp_path / "challenge.json"
    assert main(["challenge-claim-package", str(out), "--out", str(report_path)]) == 0
    payload = json.loads(report_path.read_text())
    assert payload["schema"] == "cc.claim_package_challenge.v1"
    assert payload["tamper_evident"] is True

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CAPSULE = ROOT / "examples" / "claim_governance_capsule"
OUTPUTS = CAPSULE / "outputs"


def test_claim_governance_capsule_reproduces_expected_manifest() -> None:
    result = run_capsule()

    assert result.returncode == 0, result.stderr + result.stdout
    generated = json.loads((OUTPUTS / "capsule_manifest.json").read_text(encoding="utf-8"))
    expected = json.loads((CAPSULE / "manifest.expected.json").read_text(encoding="utf-8"))
    audit = json.loads((OUTPUTS / "claim_governance_audit.json").read_text(encoding="utf-8"))
    envelope = json.loads((OUTPUTS / "claim_envelope.json").read_text(encoding="utf-8"))

    assert generated == expected
    assert generated["governance_verdict"] == "pass"
    assert audit["verdict"] == "pass"
    assert audit["required_human_review"] is False
    assert envelope["governance_state"]["verdict"] == "pass"


def test_claim_governance_capsule_second_run_manifest_is_identical() -> None:
    first = run_capsule()
    assert first.returncode == 0, first.stderr + first.stdout
    first_hash = sha256(OUTPUTS / "capsule_manifest.json")

    second = run_capsule()
    assert second.returncode == 0, second.stderr + second.stdout
    assert sha256(OUTPUTS / "capsule_manifest.json") == first_hash


def test_claim_governance_capsule_tamper_changes_verification() -> None:
    result = run_capsule()
    assert result.returncode == 0, result.stderr + result.stdout

    bounds_path = OUTPUTS / "bounds.json"
    payload = json.loads(bounds_path.read_text(encoding="utf-8"))
    payload["interval"]["upper"] = 0.999
    bounds_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest_check = subprocess.run(
        [
            sys.executable,
            "examples/claim_governance_capsule/build_capsule.py",
            "--verify-only",
        ],
        cwd=ROOT,
        env=env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert manifest_check.returncode == 1
    assert "generated artifact differs" in manifest_check.stdout

    governance_check = subprocess.run(
        [
            sys.executable,
            "-m",
            "cc.reporting.cli",
            "verify-claim-governance",
            "examples/claim_governance_capsule/outputs/cc_report.json",
            "--base-dir",
            "examples/claim_governance_capsule/outputs",
            "--now",
            "2026-01-02T00:00:00Z",
        ],
        cwd=ROOT,
        env=env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert governance_check.returncode == 2
    assert "Claim governance verdict: FAIL" in governance_check.stdout

    restore = run_capsule()
    assert restore.returncode == 0, restore.stderr + restore.stdout


def test_claim_governance_capsule_readme_includes_pass_caveat_and_non_claims() -> None:
    readme = (CAPSULE / "README.md").read_text(encoding="utf-8").lower()

    assert "pass governance audit means internal consistency under verifier rules" in readme
    assert "does not mean the ai system is safe in deployment" in readme
    assert "non-claims" in readme
    assert "not a deployment-safety proof" in readme


def run_capsule() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["examples/claim_governance_capsule/reproduce.sh"],
        cwd=ROOT,
        env=env(),
        capture_output=True,
        text=True,
        check=False,
    )


def env() -> dict[str, str]:
    values = os.environ.copy()
    values["PYTHONHASHSEED"] = "0"
    values["PYTHONPATH"] = "src"
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

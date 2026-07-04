from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CAPSULE = ROOT / "examples" / "claim_governance_capsule"
OUTPUTS = CAPSULE / "outputs"

GENERATED_ARTIFACTS = [
    "audit_log.jsonl",
    "bounds.json",
    "calibration.json",
    "capsule_manifest.json",
    "cc_report.json",
    "claim_envelope.json",
    "claim_governance_audit.json",
    "confirmatory_failure_matrix.json",
    "confirmatory_protocol.json",
    "decay_policy.json",
    "extremal_lower.json",
    "extremal_upper.json",
]


def test_claim_governance_capsule_reproduces_expected_manifest() -> None:
    result = run_capsule()

    assert result.returncode == 0, result.stderr + result.stdout

    generated = load_json(OUTPUTS / "capsule_manifest.json")
    expected = load_json(CAPSULE / "manifest.expected.json")
    audit = load_json(OUTPUTS / "claim_governance_audit.json")
    envelope = load_json(OUTPUTS / "claim_envelope.json")

    assert generated == expected
    assert generated["governance_verdict"] == "pass"

    assert audit["schema"] == "cc/claim-governance-audit.v1"
    assert audit["verdict"] == "pass"
    assert audit["required_human_review"] is False
    assert audit["reasons"] == []
    assert len(audit["non_claims"]) >= 1
    assert len(audit["evidence_artifacts"]) >= 1

    assert envelope["schema"] == "cc.claim_envelope.v1"
    assert envelope["governance_state"]["verdict"] == "pass"


def test_claim_governance_capsule_second_run_artifacts_are_identical() -> None:
    first = run_capsule()
    assert first.returncode == 0, first.stderr + first.stdout

    first_hashes = artifact_hashes()

    second = run_capsule()
    assert second.returncode == 0, second.stderr + second.stdout

    assert artifact_hashes() == first_hashes


def test_claim_governance_capsule_tamper_changes_manifest_and_governance_verdict(
    tmp_path: Path,
) -> None:
    result = run_capsule()
    assert result.returncode == 0, result.stderr + result.stdout

    tampered_root = tmp_path / "capsule"
    shutil.copytree(CAPSULE, tampered_root)

    tampered_outputs = tampered_root / "outputs"
    bounds_path = tampered_outputs / "bounds.json"

    payload = load_json(bounds_path)
    payload["interval"]["upper"] = 0.999
    bounds_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest_check = subprocess.run(
        [
            sys.executable,
            "build_capsule.py",
            "--verify-only",
        ],
        cwd=tampered_root,
        env=env(extra_pythonpath=str(ROOT / "src")),
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
            str(tampered_outputs / "cc_report.json"),
            "--base-dir",
            str(tampered_outputs),
            "--now",
            "2026-01-02T00:00:00Z",
            "--out",
            str(tampered_outputs / "tampered_audit.json"),
        ],
        cwd=ROOT,
        env=env(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert governance_check.returncode == 2
    assert "Claim governance verdict: FAIL" in governance_check.stdout

    tampered_audit = load_json(tampered_outputs / "tampered_audit.json")
    assert tampered_audit["verdict"] == "fail"
    assert any(
        "hash" in reason.lower() or "mismatch" in reason.lower()
        for reason in tampered_audit["reasons"]
    )


def test_claim_governance_capsule_readme_includes_pass_caveat_and_non_claims() -> None:
    readme = (CAPSULE / "README.md").read_text(encoding="utf-8").lower()

    required_phrases = [
        "pass governance audit means internal consistency under verifier rules",
        "does not mean the ai system is safe in deployment",
        "non-claims",
        "not a deployment-safety proof",
    ]

    for phrase in required_phrases:
        assert phrase in readme


def run_capsule() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["examples/claim_governance_capsule/reproduce.sh"],
        cwd=ROOT,
        env=env(),
        capture_output=True,
        text=True,
        check=False,
    )


def env(*, extra_pythonpath: str | None = None) -> dict[str, str]:
    values = os.environ.copy()
    values["PYTHONHASHSEED"] = "0"

    pythonpath = "src"
    if extra_pythonpath is not None:
        pythonpath = f"{extra_pythonpath}:{pythonpath}"

    values["PYTHONPATH"] = pythonpath
    return values


def artifact_hashes() -> dict[str, str]:
    return {
        name: sha256(OUTPUTS / name)
        for name in GENERATED_ARTIFACTS
        if (OUTPUTS / name).exists()
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
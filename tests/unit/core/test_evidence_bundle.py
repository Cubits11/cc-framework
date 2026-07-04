import json
from pathlib import Path

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("blake3")

from cc.core.evidence_bundle import EvidenceBundleConfig, run_evidence_bundle


def _bundle_config(
    tmp_path: Path, prompt_text: str = "user email: test@example.com\n"
) -> EvidenceBundleConfig:
    prompt_source = tmp_path / "prompts.txt"
    prompt_source.write_text(prompt_text, encoding="utf-8")

    return EvidenceBundleConfig(
        prompt_source=prompt_source,
        guardrails=[{"name": "keyword_blocker", "params": {"keywords": ["secret"]}}],
        output_dir=tmp_path,
        run_id="bundle_test",
        seed=123,
        enable_plots=False,
        env_gates={"allow_real": False},
        unsigned=True,
    )


def test_evidence_bundle_emits_leak_safe_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = run_evidence_bundle(_bundle_config(tmp_path))
    output_dir = Path(result["output_dir"])

    results_text = (output_dir / "results.jsonl").read_text(encoding="utf-8")
    metrics_text = (output_dir / "metrics.json").read_text(encoding="utf-8")
    manifest_text = (output_dir / "manifest.json").read_text(encoding="utf-8")

    assert "test@example.com" not in results_text
    assert "test@example.com" not in metrics_text
    assert "test@example.com" not in manifest_text

    assert (output_dir / "ledger.jsonl").exists()
    assert (output_dir / "transparency_log.jsonl").exists()
    assert (output_dir / "attestation.json").exists()
    assert (output_dir / "bundle_hashes.json").exists()
    assert not list(output_dir.glob("*private*key*"))


def test_evidence_bundle_default_attestation_is_unsigned(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = run_evidence_bundle(_bundle_config(tmp_path))
    output_dir = Path(result["output_dir"])
    attestation = (output_dir / "attestation.json").read_text(encoding="utf-8")

    assert '"signature_status": "unsigned"' in attestation
    assert '"signature": null' in attestation
    assert "PRIVATE KEY" not in attestation


def test_evidence_bundle_requires_key_or_explicit_unsigned_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    config = _bundle_config(tmp_path)
    config.unsigned = False

    with pytest.raises(ValueError, match="private_key_path"):
        run_evidence_bundle(config)


def test_evidence_bundle_signs_with_external_key(tmp_path: Path, monkeypatch) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    monkeypatch.chdir(tmp_path)

    key_path = tmp_path / "external_ed25519.pem"
    private_key = ed25519.Ed25519PrivateKey.generate()
    key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )

    config = _bundle_config(tmp_path)
    config.private_key_path = key_path
    config.unsigned = False

    result = run_evidence_bundle(config)
    output_dir = Path(result["output_dir"])
    attestation = json.loads((output_dir / "attestation.json").read_text(encoding="utf-8"))

    assert attestation["signature_status"] == "signed"
    assert not list(output_dir.glob("*private*key*"))

    signature = bytes.fromhex(attestation["signature"])
    attestation["signature"] = None
    public_key = ed25519.Ed25519PublicKey.from_public_bytes(
        bytes.fromhex(attestation["public_key"])
    )
    message = json.dumps(attestation, sort_keys=True, separators=(",", ":")).encode("utf-8")
    public_key.verify(signature, message)

"""Adversarial tests for the private transparency log design."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("blake3")

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from cc.core.evidence_bundle import (
    EvidenceBundleConfig,
    _attestation_signing_bytes,
    run_evidence_bundle,
    verify_evidence_bundle,
)
from cc.evidence.merkle_log import MerkleLog, verify_consistency, verify_inclusion


def _write_ed25519_key(path: Path) -> ed25519.Ed25519PrivateKey:
    private_key = ed25519.Ed25519PrivateKey.generate()
    path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return private_key


def _bundle_config(
    tmp_path: Path,
    *,
    signing_key_path: Path,
    witness_key_path: Path | None = None,
) -> EvidenceBundleConfig:
    prompt_source = tmp_path / "prompts.txt"
    prompt_source.write_text("ordinary prompt\nsecret prompt\n", encoding="utf-8")
    return EvidenceBundleConfig(
        prompt_source=prompt_source,
        guardrails=[{"name": "keyword_blocker", "params": {"keywords": ["secret"]}}],
        output_dir=tmp_path / "out",
        run_id="transparency_attack_test",
        seed=123,
        enable_plots=False,
        env_gates={"allow_real": False},
        private_key_path=signing_key_path,
        witness_private_key_path=witness_key_path,
        witness_id="independent-witness",
    )


def _trusted_witness_map(witness_private_key: ed25519.Ed25519PrivateKey) -> dict[str, str]:
    public_key = witness_private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return {"independent-witness": public_key.hex()}


def test_full_log_rewrite_and_rehash_must_fail_verification(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    signing_key_path = tmp_path / "signing.pem"
    witness_key_path = tmp_path / "witness.pem"
    _write_ed25519_key(signing_key_path)
    witness_private_key = _write_ed25519_key(witness_key_path)
    trusted_witnesses = _trusted_witness_map(witness_private_key)

    result = run_evidence_bundle(
        _bundle_config(
            tmp_path,
            signing_key_path=signing_key_path,
            witness_key_path=witness_key_path,
        )
    )
    output_dir = Path(result["output_dir"])
    ok, reason = verify_evidence_bundle(
        output_dir,
        require_anchor=True,
        trusted_witness_public_keys=trusted_witnesses,
    )
    assert ok, reason

    forged_records = [
        {"test_case_id": "prompt_0001", "composition_decision": "allow", "tampered": True},
        {"test_case_id": "prompt_0002", "composition_decision": "allow", "tampered": True},
    ]
    (output_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in forged_records) + "\n",
        encoding="utf-8",
    )
    transparency_log_path = output_dir / "transparency_log.jsonl"
    transparency_log_path.unlink()
    forged_log = MerkleLog(transparency_log_path, records=forged_records)

    attestation_path = output_dir / "attestation.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["results_merkle_root"] = forged_log.root_hash()
    attestation["transparency_log"]["root_hash"] = forged_log.root_hash()
    attestation["transparency_log"]["tree_size"] = forged_log.tree_size
    attestation_path.write_text(json.dumps(attestation, indent=2, sort_keys=True), encoding="utf-8")

    ok, reason = verify_evidence_bundle(
        output_dir,
        require_anchor=True,
        trusted_witness_public_keys=trusted_witnesses,
    )
    assert not ok
    assert reason in {"signature verification failed", "root anchor verification failed"}


def test_merkle_proof_forgery_attempt_must_fail_verification() -> None:
    records = [{"record": "alpha"}, {"record": "beta"}, {"record": "gamma"}]
    log = MerkleLog(records=records)
    proof = log.inclusion_proof(1)

    assert verify_inclusion(records[1], proof, root_hash=log.root_hash())
    assert not verify_inclusion({"record": "forged-beta"}, proof, root_hash=log.root_hash())

    forged_proof = proof.to_dict()
    forged_proof["proof"][0]["hash"] = "00" * 32
    assert not verify_inclusion(records[1], forged_proof, root_hash=log.root_hash())

    old_root = MerkleLog(records=records[:2]).root_hash()
    new_root = log.root_hash()
    consistency = log.consistency_proof(old_root, new_root)
    assert verify_consistency(consistency, old_root=old_root, new_root=new_root)

    forged_consistency = consistency.to_dict()
    forged_consistency["proof"][0] = "ff" * 32
    assert not verify_consistency(forged_consistency, old_root=old_root, new_root=new_root)


def test_replay_old_valid_attestation_under_new_run_context_must_fail(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    signing_key_path = tmp_path / "signing.pem"
    _write_ed25519_key(signing_key_path)

    result = run_evidence_bundle(_bundle_config(tmp_path, signing_key_path=signing_key_path))
    output_dir = Path(result["output_dir"])
    ok, reason = verify_evidence_bundle(output_dir)
    assert ok, reason

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_nonce"] = "new-context-nonce"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    ok, reason = verify_evidence_bundle(output_dir)
    assert not ok
    assert reason == "manifest hash mismatch"


def test_root_anchoring_bypass_attempt_must_fail_when_witness_configured(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    signing_key_path = tmp_path / "signing.pem"
    witness_key_path = tmp_path / "witness.pem"
    signing_private_key = _write_ed25519_key(signing_key_path)
    witness_private_key = _write_ed25519_key(witness_key_path)
    trusted_witnesses = _trusted_witness_map(witness_private_key)

    result = run_evidence_bundle(
        _bundle_config(
            tmp_path,
            signing_key_path=signing_key_path,
            witness_key_path=witness_key_path,
        )
    )
    output_dir = Path(result["output_dir"])
    ok, reason = verify_evidence_bundle(
        output_dir,
        require_anchor=True,
        trusted_witness_public_keys=trusted_witnesses,
    )
    assert ok, reason

    attestation_path = output_dir / "attestation.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["anchor"] = None
    attestation["signature"] = None
    attestation["signature"] = signing_private_key.sign(
        _attestation_signing_bytes(attestation)
    ).hex()
    attestation_path.write_text(json.dumps(attestation, indent=2, sort_keys=True), encoding="utf-8")

    ok, reason = verify_evidence_bundle(
        output_dir,
        require_anchor=True,
        trusted_witness_public_keys=trusted_witnesses,
    )
    assert not ok
    assert reason == "required root anchor is missing"

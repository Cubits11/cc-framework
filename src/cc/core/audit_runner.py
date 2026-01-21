# src/cc/core/audit_runner.py
"""Minimal audit runner: prompts -> guardrails -> evidence artifacts.

This module provides a small, deterministic audit flow suitable for demos:

1. Load prompts from .txt/.csv/.jsonl.
2. Evaluate guardrail stack (per prompt).
3. Write results.jsonl + execution_manifest.json.
4. Build a Merkle root over results.
5. Sign a run attestation (Ed25519).

It intentionally avoids external infrastructure (queues, DBs) while producing
artifacts that map to the broader Assurance Docker vision.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from cc.core.guardrail_api import GuardrailAdapter
from cc.core.manifest import (
    RunManifest,
    build_config_hashes,
    emit_run_manifest,
    guardrail_versions_from_instances,
)
from cc.core.registry import build_guardrails


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(data: dict[str, Any]) -> str:
    return _sha256_bytes(json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return None


def _read_text_lines(path: Path) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
    return lines


def _load_prompts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt source not found: {path}")

    suffix = path.suffix.lower()
    prompts: list[dict[str, Any]] = []

    if suffix in {".txt"}:
        for idx, line in enumerate(_read_text_lines(path), start=1):
            prompts.append({"id": f"prompt_{idx:04d}", "prompt": line})
        return prompts

    if suffix in {".jsonl"}:
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                prompt = payload.get("prompt") or payload.get("text")
                if not prompt:
                    raise ValueError(f"JSONL line {idx} missing prompt/text field.")
                prompts.append({"id": payload.get("id") or f"prompt_{idx:04d}", "prompt": prompt})
        return prompts

    if suffix in {".csv"}:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("CSV prompt source has no header row.")
            for idx, row in enumerate(reader, start=1):
                prompt = row.get("prompt") or row.get("text")
                if not prompt:
                    raise ValueError(f"CSV row {idx} missing prompt/text column.")
                prompts.append({"id": row.get("id") or f"prompt_{idx:04d}", "prompt": prompt})
        return prompts

    raise ValueError(f"Unsupported prompt source format: {path.suffix}")


def _merkle_root(lines: Iterable[str]) -> str:
    hashes = [hashlib.sha256(line.encode("utf-8")).digest() for line in lines]
    if not hashes:
        return _sha256_bytes(b"")
    while len(hashes) > 1:
        if len(hashes) % 2 == 1:
            hashes.append(hashes[-1])
        next_level = []
        for left, right in zip(hashes[0::2], hashes[1::2], strict=False):
            next_level.append(hashlib.sha256(left + right).digest())
        hashes = next_level
    return hashes[0].hex()


def _load_private_key(path: Path) -> ed25519.Ed25519PrivateKey:
    if path.exists():
        return serialization.load_pem_private_key(path.read_bytes(), password=None)
    key = ed25519.Ed25519PrivateKey.generate()
    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return key


@dataclass
class AuditRunConfig:
    prompt_source: Path
    guardrails: list[dict[str, Any]]
    output_dir: Path
    composition: str = "any_block"
    benign_calibration_source: Path | None = None
    private_key_path: Path | None = None
    run_id: str | None = None


def _compose_decision(decisions: Sequence[bool], mode: str) -> str:
    mode = mode.lower().strip()
    if mode in {"any_block", "or"}:
        return "block" if any(decisions) else "allow"
    if mode in {"all_block", "and"}:
        return "block" if decisions and all(decisions) else "allow"
    if mode in {"majority"}:
        if not decisions:
            return "allow"
        return "block" if sum(1 for d in decisions if d) >= (len(decisions) / 2) else "allow"
    raise ValueError(f"Unknown composition mode: {mode}")


def run_audit(config: AuditRunConfig) -> dict[str, Any]:
    run_id = config.run_id or f"run_{uuid4().hex[:12]}"
    output_dir = config.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_rows = _load_prompts(config.prompt_source)
    guardrail_instances = build_guardrails(config.guardrails)
    adapters = [GuardrailAdapter(g) for g in guardrail_instances]

    if config.benign_calibration_source:
        benign_prompts = [p["prompt"] for p in _load_prompts(config.benign_calibration_source)]
        for guardrail in guardrail_instances:
            if hasattr(guardrail, "calibrate"):
                guardrail.calibrate(benign_prompts, target_fpr=0.05)

    results_path = output_dir / "results.jsonl"
    results_lines: list[str] = []

    for row in prompt_rows:
        prompt = row["prompt"]
        per_guardrail: list[dict[str, Any]] = []
        decisions: list[bool] = []
        for adapter in adapters:
            blocked, score = adapter.evaluate(prompt)
            per_guardrail.append(
                {
                    "guardrail": adapter.guardrail.__class__.__name__,
                    "blocked": blocked,
                    "score": round(float(score), 6),
                    "threshold": round(float(adapter.threshold), 6),
                }
            )
            decisions.append(bool(blocked))

        composition_decision = _compose_decision(decisions, config.composition)
        record = {
            "test_case_id": row["id"],
            "prompt": prompt,
            "guardrails": per_guardrail,
            "composition_decision": composition_decision,
            "timestamp": _utc_now(),
        }
        line = json.dumps(record, sort_keys=True)
        results_lines.append(line)

    results_path.write_text(
        "\n".join(results_lines) + ("\n" if results_lines else ""), encoding="utf-8"
    )

    execution_manifest = {
        "run_id": run_id,
        "created_at": _utc_now(),
        "git_commit": _git_commit(),
        "prompt_source": str(config.prompt_source),
        "guardrails": config.guardrails,
        "composition": config.composition,
        "environment": {
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "platform": os.sys.platform,
        },
    }
    manifest_path = output_dir / "execution_manifest.json"
    manifest_path.write_text(
        json.dumps(execution_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    config_payload = {
        "prompt_source": str(config.prompt_source),
        "guardrails": config.guardrails,
        "composition": config.composition,
        "benign_calibration_source": (
            str(config.benign_calibration_source) if config.benign_calibration_source else None
        ),
    }
    dataset_ids = [str(config.prompt_source)]
    if config.benign_calibration_source:
        dataset_ids.append(str(config.benign_calibration_source))

    run_manifest = RunManifest(
        run_id=run_id,
        config_hashes=build_config_hashes(config_payload, label="audit_config_blake3"),
        dataset_ids=dataset_ids,
        guardrail_versions=guardrail_versions_from_instances(guardrail_instances),
        git_sha=_git_commit(),
    )
    manifest_artifacts = emit_run_manifest(run_manifest)

    results_merkle_root = _merkle_root(results_lines)
    manifest_hash = _sha256_json(execution_manifest)

    key_path = config.private_key_path or output_dir / "attestation_private_key.pem"
    private_key = _load_private_key(key_path)
    public_key = private_key.public_key()
    public_key_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    attestation = {
        "run_id": run_id,
        "timestamp": _utc_now(),
        "execution_manifest_hash": manifest_hash,
        "results_merkle_root": results_merkle_root,
        "previous_attestation_hash": None,
        "public_key": public_key_bytes.hex(),
    }
    attestation_message = json.dumps(attestation, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    signature = private_key.sign(attestation_message).hex()
    attestation["signature"] = signature

    attestation_path = output_dir / "attestation.json"
    attestation_path.write_text(json.dumps(attestation, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "run_manifest_path": manifest_artifacts["manifest_path"],
        "run_manifest_chain": manifest_artifacts["chain_path"],
        "run_manifest_chain_head": manifest_artifacts["chain_head"],
        "results_path": str(results_path),
        "attestation_path": str(attestation_path),
    }


def verify_attestation(
    attestation_path: Path, manifest_path: Path, results_path: Path
) -> tuple[bool, str]:
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results_lines = results_path.read_text(encoding="utf-8").splitlines()

    expected_manifest_hash = _sha256_json(manifest)
    if attestation.get("execution_manifest_hash") != expected_manifest_hash:
        return False, "manifest hash mismatch"

    expected_merkle_root = _merkle_root(results_lines)
    if attestation.get("results_merkle_root") != expected_merkle_root:
        return False, "results merkle root mismatch"

    public_key = ed25519.Ed25519PublicKey.from_public_bytes(
        bytes.fromhex(attestation["public_key"])
    )
    message = json.dumps(
        {
            "execution_manifest_hash": attestation["execution_manifest_hash"],
            "previous_attestation_hash": attestation["previous_attestation_hash"],
            "public_key": attestation["public_key"],
            "results_merkle_root": attestation["results_merkle_root"],
            "run_id": attestation["run_id"],
            "timestamp": attestation["timestamp"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    try:
        public_key.verify(bytes.fromhex(attestation["signature"]), message)
    except Exception:
        return False, "signature verification failed"

    return True, "attestation verified"


__all__ = [
    "AuditRunConfig",
    "run_audit",
    "verify_attestation",
]

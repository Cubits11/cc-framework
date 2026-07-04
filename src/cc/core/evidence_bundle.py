# src/cc/core/evidence_bundle.py
"""Assurance Evidence Bundle runner for guardrail audits.

Outputs deterministic, leak-safe artifacts:
- results.jsonl (prompt summaries + guardrail decisions)
- metrics.json (summary metrics)
- manifest.json (rerun command, seed, env gates)
- ledger.jsonl (tamper-evident chain)
- attestation.json (explicitly unsigned, or Ed25519-signed with an external key)
- plots/ (quantitative plots)
"""

from __future__ import annotations

import hashlib
import json
import random
import secrets
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from cc.adapters.base import (
    AUDIT_SCHEMA_VERSION,
    build_audit_payload,
    error_summary_from_exception,
    fingerprint_payload,
    summarize_value,
)
from cc.cartographer import audit as audit_chain
from cc.core.guardrail_api import GuardrailAdapter
from cc.core.manifest import (
    RunManifest,
    build_config_hashes,
    emit_run_manifest,
    guardrail_versions_from_instances,
)
from cc.core.registry import build_guardrails
from cc.evidence.anchoring import (
    Ed25519Witness,
    RootCheckpoint,
    Witness,
    anchor_root,
    verify_anchor,
)
from cc.evidence.merkle_log import HASH_ALGORITHM, MerkleLog
from cc.utils.artifacts import detect_git_commit, write_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(data: Any) -> bytes:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(data: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_json_bytes(data))


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


_SENSITIVE_ARG_NAMES = {
    "--api-key",
    "--key",
    "--password",
    "--private-key",
    "--private-key-path",
    "--secret",
    "--token",
}


def _redact_argv(argv: Sequence[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for arg in argv:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue

        key, sep, value = arg.partition("=")
        key_l = key.lower()
        if key_l in _SENSITIVE_ARG_NAMES:
            if sep:
                redacted.append(f"{key}=<redacted>")
            else:
                redacted.append(arg)
                redact_next = True
            continue
        if any(marker in key_l for marker in ("password", "secret", "token")):
            redacted.append(f"{key}=<redacted>" if sep else "<redacted>")
            continue
        redacted.append(arg if not value else f"{key}{sep}{value}")
    return redacted


def _load_required_private_key(path: Path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    if not path.exists():
        raise FileNotFoundError(
            f"Signing key not found: {path}. Generate and manage signing keys outside "
            "the evidence output directory, or pass explicit unsigned=True/--unsigned "
            "mode for a local unsigned attestation."
        )
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, ed25519.Ed25519PrivateKey):
        raise TypeError("Evidence bundle signing requires an Ed25519 private key.")
    return key


def _assert_key_outside_output(private_key_path: Path, output_dir: Path) -> None:
    key_path = private_key_path.expanduser().resolve()
    run_output = output_dir.resolve()
    if key_path == run_output or key_path.is_relative_to(run_output):
        raise ValueError(
            "private_key_path must live outside the evidence output directory; "
            "otherwise the bundle can leak the signing key."
        )


def _attestation_signing_payload(attestation: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(attestation)
    payload["signature"] = None
    return payload


def _attestation_signing_bytes(attestation: Mapping[str, Any]) -> bytes:
    return _canonical_json_bytes(_attestation_signing_payload(attestation))


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
        import csv

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


@dataclass
class EvidenceBundleConfig:
    prompt_source: Path
    guardrails: list[dict[str, Any]]
    output_dir: Path
    composition: str = "any_block"
    benign_calibration_source: Path | None = None
    private_key_path: Path | None = None
    unsigned: bool = False
    run_id: str | None = None
    run_nonce: str | None = None
    seed: int = 1337
    enable_plots: bool = True
    env_gates: dict[str, Any] | None = None
    witness: Witness | None = None
    witness_private_key_path: Path | None = None
    witness_id: str = "cc-evidence-witness"


def _event_id(run_id: str, prompt_hash: str, guardrail_name: str, index: int) -> str:
    return fingerprint_payload(
        {
            "run_id": run_id,
            "prompt_hash": prompt_hash,
            "guardrail": guardrail_name,
            "index": index,
        },
        strict=False,
    )


def _render_block_rate_plot(path: Path, labels: Sequence[str], values: Sequence[float]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color="#4C78A8")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Block rate")
    ax.set_title("Guardrail block rate")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_evidence_bundle(config: EvidenceBundleConfig) -> dict[str, Any]:
    run_id = config.run_id or f"bundle_{uuid4().hex[:12]}"
    output_dir = config.output_dir / run_id

    if config.private_key_path is None and not config.unsigned:
        raise ValueError(
            "Evidence bundle attestation requires private_key_path, or explicit "
            "unsigned=True/--unsigned mode."
        )
    if config.private_key_path is not None and config.unsigned:
        raise ValueError("Use either private_key_path or unsigned=True, not both.")

    if config.witness is not None and config.witness_private_key_path is not None:
        raise ValueError("Use either witness or witness_private_key_path, not both.")
    if config.private_key_path is not None:
        _assert_key_outside_output(config.private_key_path, output_dir)
    if config.witness_private_key_path is not None:
        _assert_key_outside_output(config.witness_private_key_path, output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    run_nonce = config.run_nonce or secrets.token_hex(16)
    random.seed(config.seed)

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
    result_records: list[dict[str, Any]] = []
    guardrail_block_counts: dict[str, int] = {}
    guardrail_review_counts: dict[str, int] = {}
    guardrail_totals: dict[str, int] = {}
    composition_blocks = 0

    for _prompt_idx, row in enumerate(prompt_rows):
        prompt = row["prompt"]
        prompt_summary = summarize_value(prompt)
        per_guardrail: list[dict[str, Any]] = []
        decisions: list[bool] = []
        verdicts: list[str] = []

        for gr_idx, adapter in enumerate(adapters):
            guardrail_name = adapter.guardrail.__class__.__name__
            started_at = datetime.now(timezone.utc).timestamp()
            error_summary = None
            verdict = "review"
            category = None
            score: float | None = None
            blocked = False

            try:
                blocked, score = adapter.evaluate(prompt)
                verdict = "block" if blocked else "allow"
            except Exception as exc:
                error_summary = error_summary_from_exception(exc, where="guardrail.evaluate")
                verdict = "review"
                category = "adapter_error"
                blocked = False

            completed_at = datetime.now(timezone.utc).timestamp()
            prompt_hash = prompt_summary["sha256"]
            event_id = _event_id(run_id, prompt_hash, guardrail_name, gr_idx)
            audit_payload = build_audit_payload(
                prompt=prompt,
                response=None,
                adapter_name=guardrail_name,
                adapter_version=getattr(adapter.guardrail, "version", "local"),
                parameters={"config": config.guardrails[gr_idx]},
                decision=verdict,
                category=category,
                rationale=None,
                started_at=started_at,
                completed_at=completed_at,
                metadata={"run_id": run_id, "prompt_id": row["id"]},
                error_summary=error_summary,
                event_id=event_id,
            )
            audit_payload["schema"] = AUDIT_SCHEMA_VERSION

            per_guardrail.append(
                {
                    "guardrail": guardrail_name,
                    "verdict": verdict,
                    "blocked": bool(blocked),
                    "score": None if score is None else round(float(score), 6),
                    "threshold": round(float(adapter.threshold), 6),
                    "category": category,
                    "event_id": event_id,
                    "event_hash": audit_payload["event_hash"],
                    "audit": audit_payload,
                }
            )

            guardrail_totals[guardrail_name] = guardrail_totals.get(guardrail_name, 0) + 1
            if verdict == "block":
                guardrail_block_counts[guardrail_name] = (
                    guardrail_block_counts.get(guardrail_name, 0) + 1
                )
            if verdict == "review":
                guardrail_review_counts[guardrail_name] = (
                    guardrail_review_counts.get(guardrail_name, 0) + 1
                )

            decisions.append(bool(blocked))
            verdicts.append(verdict)

        composition_decision = _compose_decision(decisions, config.composition)
        if composition_decision == "block":
            composition_blocks += 1

        record = {
            "test_case_id": row["id"],
            "prompt_summary": prompt_summary,
            "guardrails": per_guardrail,
            "composition_decision": composition_decision,
            "verdicts": verdicts,
            "timestamp": _utc_now(),
        }
        result_records.append(record)
        line = json.dumps(record, sort_keys=True)
        results_lines.append(line)

    results_path.write_text(
        "\n".join(results_lines) + ("\n" if results_lines else ""), encoding="utf-8"
    )

    metrics = {
        "run_id": run_id,
        "prompt_count": len(prompt_rows),
        "composition_block_rate": (composition_blocks / len(prompt_rows)) if prompt_rows else 0.0,
        "guardrail_block_rates": {
            name: (guardrail_block_counts.get(name, 0) / total)
            for name, total in guardrail_totals.items()
        },
        "guardrail_review_rates": {
            name: (guardrail_review_counts.get(name, 0) / total)
            for name, total in guardrail_totals.items()
        },
    }

    manifest = {
        "run_id": run_id,
        "run_nonce": run_nonce,
        "created_at": _utc_now(),
        "prompt_source": str(config.prompt_source),
        "guardrails": config.guardrails,
        "composition": config.composition,
        "seed": config.seed,
        "rerun_argv": _redact_argv(sys.argv)
        if sys.argv
        else ["python", "-m", "cc.core.evidence_bundle"],
        "env_gates": config.env_gates or {},
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": sys.platform,
    }

    metrics_path = output_dir / "metrics.json"
    manifest_path = output_dir / "manifest.json"
    write_json(metrics_path, metrics)
    write_json(manifest_path, manifest)

    config_payload = {
        "prompt_source": str(config.prompt_source),
        "guardrails": config.guardrails,
        "composition": config.composition,
        "seed": config.seed,
        "benign_calibration_source": (
            str(config.benign_calibration_source) if config.benign_calibration_source else None
        ),
    }
    dataset_ids = [str(config.prompt_source)]
    if config.benign_calibration_source:
        dataset_ids.append(str(config.benign_calibration_source))
    run_manifest = RunManifest(
        run_id=run_id,
        config_hashes=build_config_hashes(config_payload, label="evidence_config_blake3"),
        dataset_ids=dataset_ids,
        guardrail_versions=guardrail_versions_from_instances(guardrail_instances),
        git_sha=detect_git_commit(),
    )
    manifest_artifacts = emit_run_manifest(run_manifest)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_paths: list[str] = []
    if config.enable_plots:
        labels = list(metrics["guardrail_block_rates"].keys())
        values = [metrics["guardrail_block_rates"][label] for label in labels]
        if labels:
            plot_path = plots_dir / "guardrail_block_rates.png"
            _render_block_rate_plot(plot_path, labels, values)
            plot_paths.append(str(plot_path))

    transparency_log_path = output_dir / "transparency_log.jsonl"
    if transparency_log_path.exists():
        transparency_log_path.unlink()
    transparency_log_path.touch()
    log_id = f"cc-evidence-bundle:{run_id}"
    transparency_log = MerkleLog(transparency_log_path, log_id=log_id)
    for result_record in result_records:
        transparency_log.append(result_record)
    transparency_root_hash = transparency_log.root_hash()
    transparency_tree_size = transparency_log.tree_size

    witness = config.witness
    if witness is None and config.witness_private_key_path is not None:
        witness = Ed25519Witness.from_private_key_path(
            config.witness_private_key_path,
            witness_id=config.witness_id,
        )

    anchor_payload: dict[str, Any] | None = None
    if witness is not None:
        checkpoint = RootCheckpoint.create(
            log_id=log_id,
            tree_size=transparency_tree_size,
            root_hash=transparency_root_hash,
            run_nonce=run_nonce,
        )
        anchor_payload = anchor_root(checkpoint, witness).to_dict()

    ledger_path = output_dir / "ledger.jsonl"
    ledger_record = {
        "record_type": "evidence_bundle",
        "run_id": run_id,
        "run_nonce": run_nonce,
        "created_at": _utc_now(),
        "metrics_hash": _sha256_json(metrics),
        "manifest_hash": _sha256_json(manifest),
        "results_merkle_root": transparency_root_hash,
        "transparency_log": {
            "hash_algorithm": HASH_ALGORITHM,
            "log_id": log_id,
            "root_hash": transparency_root_hash,
            "tree_size": transparency_tree_size,
        },
        "plot_paths": plot_paths,
    }
    audit_chain.append_jsonl(str(ledger_path), ledger_record)

    attestation = {
        "schema": "cc/evidence-attestation.v2",
        "run_id": run_id,
        "run_nonce": run_nonce,
        "timestamp": _utc_now(),
        "manifest_hash": _sha256_json(manifest),
        "metrics_hash": _sha256_json(metrics),
        "results_merkle_root": transparency_root_hash,
        "transparency_log": {
            "hash_algorithm": HASH_ALGORITHM,
            "log_id": log_id,
            "root_hash": transparency_root_hash,
            "tree_size": transparency_tree_size,
        },
        "ledger_tail_hash": audit_chain.tail_sha(str(ledger_path)),
        "anchor": anchor_payload,
        "public_key": None,
        "signature": None,
        "signature_status": "unsigned" if config.unsigned else "pending",
        "unsigned_reason": "explicit_unsigned_mode" if config.unsigned else None,
    }
    if config.private_key_path is not None:
        from cryptography.hazmat.primitives import serialization

        private_key = _load_required_private_key(config.private_key_path)
        public_key = private_key.public_key()
        public_key_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        attestation["public_key"] = public_key_bytes.hex()
        attestation["signature_status"] = "signed"
        attestation["unsigned_reason"] = None
        attestation["signature"] = private_key.sign(_attestation_signing_bytes(attestation)).hex()
    attestation_path = output_dir / "attestation.json"
    write_json(attestation_path, attestation)

    bundle_hashes = {
        "results.jsonl": _sha256_file(results_path),
        "metrics.json": _sha256_file(metrics_path),
        "manifest.json": _sha256_file(manifest_path),
        "ledger.jsonl": _sha256_file(ledger_path),
        "transparency_log.jsonl": _sha256_file(transparency_log_path),
        "attestation.json": _sha256_file(attestation_path),
    }
    hashes_path = output_dir / "bundle_hashes.json"
    write_json(hashes_path, bundle_hashes)

    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "results_path": str(results_path),
        "metrics_path": str(metrics_path),
        "manifest_path": str(manifest_path),
        "ledger_path": str(ledger_path),
        "transparency_log_path": str(transparency_log_path),
        "transparency_root_hash": transparency_root_hash,
        "transparency_tree_size": transparency_tree_size,
        "attestation_path": str(attestation_path),
        "hashes_path": str(hashes_path),
        "plot_paths": plot_paths,
        "run_manifest_path": manifest_artifacts["manifest_path"],
        "run_manifest_chain": manifest_artifacts["chain_path"],
        "run_manifest_chain_head": manifest_artifacts["chain_head"],
    }


def _read_results_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(f"results line {line_number} is not a JSON object")
            records.append(payload)
    return records


def _verify_attestation_signature(
    attestation: Mapping[str, Any],
    *,
    require_signature: bool,
) -> tuple[bool, str]:
    signature_status = attestation.get("signature_status")
    if signature_status == "unsigned":
        if require_signature:
            return False, "attestation is unsigned"
        if attestation.get("signature") is not None or attestation.get("public_key") is not None:
            return False, "unsigned attestation contains signature material"
        return True, "unsigned attestation accepted"

    if signature_status != "signed":
        return False, "attestation signature_status is invalid"

    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519

        public_key_hex = attestation.get("public_key")
        signature_hex = attestation.get("signature")
        if not isinstance(public_key_hex, str) or not isinstance(signature_hex, str):
            return False, "signed attestation missing public key or signature"
        public_key = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
        public_key.verify(bytes.fromhex(signature_hex), _attestation_signing_bytes(attestation))
        return True, "signature verified"
    except Exception:
        return False, "signature verification failed"


def verify_evidence_bundle(
    output_dir: Path,
    *,
    require_signature: bool = True,
    require_anchor: bool = False,
    trusted_witness_public_keys: Mapping[str, str] | None = None,
) -> tuple[bool, str]:
    """Verify an evidence bundle without trusting its host storage."""

    try:
        output_dir = Path(output_dir)
        attestation_path = output_dir / "attestation.json"
        manifest_path = output_dir / "manifest.json"
        metrics_path = output_dir / "metrics.json"
        results_path = output_dir / "results.jsonl"
        transparency_log_path = output_dir / "transparency_log.jsonl"

        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        result_records = _read_results_records(results_path)

        if attestation.get("manifest_hash") != _sha256_json(manifest):
            return False, "manifest hash mismatch"
        if attestation.get("metrics_hash") != _sha256_json(metrics):
            return False, "metrics hash mismatch"
        if attestation.get("run_id") != manifest.get("run_id"):
            return False, "run_id mismatch"
        if not manifest.get("run_nonce") or attestation.get("run_nonce") != manifest.get(
            "run_nonce"
        ):
            return False, "run_nonce mismatch"

        transparency = attestation.get("transparency_log")
        if not isinstance(transparency, Mapping):
            return False, "missing transparency log checkpoint"

        expected_log = MerkleLog(
            records=result_records,
            log_id=str(transparency.get("log_id", "cc-transparency-log")),
        )
        expected_root = expected_log.root_hash()
        expected_tree_size = expected_log.tree_size

        if transparency.get("hash_algorithm") != HASH_ALGORITHM:
            return False, "unsupported transparency hash algorithm"
        if transparency.get("root_hash") != expected_root:
            return False, "transparency root mismatch"
        if transparency.get("tree_size") != expected_tree_size:
            return False, "transparency tree size mismatch"
        if attestation.get("results_merkle_root") != expected_root:
            return False, "results merkle root mismatch"

        if transparency_log_path.exists():
            disk_log = MerkleLog(transparency_log_path)
            if disk_log.root_hash() != expected_root or disk_log.tree_size != expected_tree_size:
                return False, "transparency log file mismatch"
            if disk_log.records != result_records:
                return False, "transparency log records mismatch"

        ok, reason = _verify_attestation_signature(
            attestation,
            require_signature=require_signature,
        )
        if not ok:
            return False, reason

        anchor = attestation.get("anchor")
        if require_anchor and anchor is None:
            return False, "required root anchor is missing"
        if anchor is not None:
            if not trusted_witness_public_keys:
                return False, "trusted witness public keys are required for anchor verification"
            if not verify_anchor(
                anchor,
                trusted_witness_public_keys=trusted_witness_public_keys,
                expected_root_hash=expected_root,
                expected_tree_size=expected_tree_size,
                expected_log_id=str(transparency["log_id"]),
                expected_run_nonce=str(attestation["run_nonce"]),
            ):
                return False, "root anchor verification failed"

        return True, "evidence bundle verified"
    except Exception as exc:
        return False, f"evidence bundle verification error: {exc}"


def _parse_guardrail_config(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("guardrail config must be a JSON list")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    argv = list(argv if argv is not None else sys.argv[1:])
    if argv and argv[0] == "run":
        argv = argv[1:]

    parser = argparse.ArgumentParser(description="Generate an Assurance Evidence Bundle")
    parser.add_argument("--prompt-source", type=Path, required=True)
    parser.add_argument("--guardrails-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/evidence"))
    parser.add_argument("--composition", type=str, default="any_block")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--benign-calibration-source", type=Path, default=None)
    parser.add_argument("--private-key-path", type=Path, default=None)
    parser.add_argument("--unsigned", action="store_true")
    parser.add_argument("--witness-private-key-path", type=Path, default=None)
    parser.add_argument("--witness-id", type=str, default="cc-evidence-witness")
    parser.add_argument("--disable-plots", action="store_true")
    args = parser.parse_args(argv)

    config = EvidenceBundleConfig(
        prompt_source=args.prompt_source,
        guardrails=_parse_guardrail_config(args.guardrails_config),
        output_dir=args.output_dir,
        composition=args.composition,
        benign_calibration_source=args.benign_calibration_source,
        private_key_path=args.private_key_path,
        unsigned=args.unsigned,
        run_id=args.run_id,
        seed=args.seed,
        enable_plots=not args.disable_plots,
        env_gates={"allow_real": False},
        witness_private_key_path=args.witness_private_key_path,
        witness_id=args.witness_id,
    )

    result = run_evidence_bundle(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# src/cc/core/evidence_bundle.py
"""Assurance Evidence Bundle runner for guardrail audits.

Outputs deterministic, leak-safe artifacts:
- results.jsonl (prompt summaries + guardrail decisions)
- metrics.json (summary metrics)
- manifest.json (rerun command, seed, env gates)
- ledger.jsonl (tamper-evident chain)
- attestation.json (Ed25519 signature)
- plots/ (quantitative plots)
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
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
from cc.utils.artifacts import detect_git_commit, write_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(data: Dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(payload)


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _merkle_root(lines: Iterable[str]) -> str:
    hashes = [hashlib.sha256(line.encode("utf-8")).digest() for line in lines]
    if not hashes:
        return _sha256_bytes(b"")
    while len(hashes) > 1:
        if len(hashes) % 2 == 1:
            hashes.append(hashes[-1])
        next_level = []
        for left, right in zip(hashes[0::2], hashes[1::2]):
            next_level.append(hashlib.sha256(left + right).digest())
        hashes = next_level
    return hashes[0].hex()


def _load_private_key(path: Path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

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


def _read_text_lines(path: Path) -> List[str]:
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
    return lines


def _load_prompts(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt source not found: {path}")

    suffix = path.suffix.lower()
    prompts: List[Dict[str, Any]] = []

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
    guardrails: List[Dict[str, Any]]
    output_dir: Path
    composition: str = "any_block"
    benign_calibration_source: Optional[Path] = None
    private_key_path: Optional[Path] = None
    run_id: Optional[str] = None
    seed: int = 1337
    enable_plots: bool = True
    env_gates: Optional[Dict[str, Any]] = None


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


def run_evidence_bundle(config: EvidenceBundleConfig) -> Dict[str, Any]:
    run_id = config.run_id or f"bundle_{uuid4().hex[:12]}"
    output_dir = config.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

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
    results_lines: List[str] = []
    guardrail_block_counts: Dict[str, int] = {}
    guardrail_review_counts: Dict[str, int] = {}
    guardrail_totals: Dict[str, int] = {}
    composition_blocks = 0

    for prompt_idx, row in enumerate(prompt_rows):
        prompt = row["prompt"]
        prompt_summary = summarize_value(prompt)
        per_guardrail: List[Dict[str, Any]] = []
        decisions: List[bool] = []
        verdicts: List[str] = []

        for gr_idx, adapter in enumerate(adapters):
            guardrail_name = adapter.guardrail.__class__.__name__
            started_at = datetime.now(timezone.utc).timestamp()
            error_summary = None
            verdict = "review"
            category = None
            score: Optional[float] = None
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

    rerun_command = " ".join(sys.argv) if sys.argv else "python -m cc.core.evidence_bundle"
    manifest = {
        "run_id": run_id,
        "created_at": _utc_now(),
        "prompt_source": str(config.prompt_source),
        "guardrails": config.guardrails,
        "composition": config.composition,
        "seed": config.seed,
        "rerun_command": rerun_command,
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
    plot_paths: List[str] = []
    if config.enable_plots:
        labels = list(metrics["guardrail_block_rates"].keys())
        values = [metrics["guardrail_block_rates"][label] for label in labels]
        if labels:
            plot_path = plots_dir / "guardrail_block_rates.png"
            _render_block_rate_plot(plot_path, labels, values)
            plot_paths.append(str(plot_path))

    ledger_path = output_dir / "ledger.jsonl"
    results_merkle_root = _merkle_root(results_lines)
    ledger_record = {
        "record_type": "evidence_bundle",
        "run_id": run_id,
        "created_at": _utc_now(),
        "metrics_hash": _sha256_json(metrics),
        "manifest_hash": _sha256_json(manifest),
        "results_merkle_root": results_merkle_root,
        "plot_paths": plot_paths,
    }
    audit_chain.append_jsonl(str(ledger_path), ledger_record)

    from cryptography.hazmat.primitives import serialization

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
        "manifest_hash": _sha256_json(manifest),
        "metrics_hash": _sha256_json(metrics),
        "results_merkle_root": results_merkle_root,
        "ledger_tail_hash": audit_chain.tail_sha(str(ledger_path)),
        "public_key": public_key_bytes.hex(),
    }
    attestation_message = json.dumps(attestation, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    signature = private_key.sign(attestation_message).hex()
    attestation["signature"] = signature
    attestation_path = output_dir / "attestation.json"
    write_json(attestation_path, attestation)

    bundle_hashes = {
        "results.jsonl": _sha256_file(results_path),
        "metrics.json": _sha256_file(metrics_path),
        "manifest.json": _sha256_file(manifest_path),
        "ledger.jsonl": _sha256_file(ledger_path),
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
        "attestation_path": str(attestation_path),
        "hashes_path": str(hashes_path),
        "plot_paths": plot_paths,
        "run_manifest_path": manifest_artifacts["manifest_path"],
        "run_manifest_chain": manifest_artifacts["chain_path"],
        "run_manifest_chain_head": manifest_artifacts["chain_head"],
    }


def _parse_guardrail_config(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("guardrail config must be a JSON list")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate an Assurance Evidence Bundle")
    parser.add_argument("--prompt-source", type=Path, required=True)
    parser.add_argument("--guardrails-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/evidence"))
    parser.add_argument("--composition", type=str, default="any_block")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--benign-calibration-source", type=Path, default=None)
    parser.add_argument("--private-key-path", type=Path, default=None)
    parser.add_argument("--disable-plots", action="store_true")
    args = parser.parse_args(argv)

    config = EvidenceBundleConfig(
        prompt_source=args.prompt_source,
        guardrails=_parse_guardrail_config(args.guardrails_config),
        output_dir=args.output_dir,
        composition=args.composition,
        benign_calibration_source=args.benign_calibration_source,
        private_key_path=args.private_key_path,
        run_id=args.run_id,
        seed=args.seed,
        enable_plots=not args.disable_plots,
        env_gates={"allow_real": False},
    )

    result = run_evidence_bundle(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

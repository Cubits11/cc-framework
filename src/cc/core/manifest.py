# src/cc/core/manifest.py
"""Run manifest lineage helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from pydantic import Field

from cc.cartographer import audit as audit_chain
from cc.core.models import ModelBase, _hash_json, _now_unix


class RunManifest(ModelBase):
    """Reproducibility manifest for a single run."""

    run_id: str
    created_at: float = Field(default_factory=_now_unix)
    config_hashes: Dict[str, Any] = Field(default_factory=dict)
    dataset_ids: List[str] = Field(default_factory=list)
    guardrail_versions: Dict[str, str] = Field(default_factory=dict)
    git_sha: Optional[str] = None


def build_config_hashes(
    payload: Mapping[str, Any],
    *,
    label: str = "config_blake3",
) -> Dict[str, str]:
    """Hash a config payload for manifest lineage."""
    return {label: _hash_json(payload)}


def guardrail_versions_from_instances(guardrails: Sequence[Any]) -> Dict[str, str]:
    """Resolve guardrail version metadata from instantiated guardrails."""
    versions: Dict[str, str] = {}
    for guardrail in guardrails:
        name = guardrail.__class__.__name__
        versions[name] = _resolve_version(guardrail)
    return versions


def _resolve_version(obj: Any) -> str:
    for attr in ("version", "__version__", "VERSION"):
        val = getattr(obj, attr, None)
        if val:
            return str(val)
    module = sys.modules.get(obj.__class__.__module__)
    if module is not None:
        module_version = getattr(module, "__version__", None)
        if module_version:
            return str(module_version)
    return "unknown"


def emit_run_manifest(
    manifest: RunManifest,
    *,
    output_dir: Path = Path("runs/manifest"),
) -> Dict[str, str]:
    """Write the manifest JSON and append it to a per-run hash chain."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{manifest.run_id}.json"
    chain_path = output_dir / f"{manifest.run_id}.jsonl"

    payload = manifest.model_dump(mode="json", by_alias=True)
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    chain_head = audit_chain.append_jsonl(str(chain_path), payload)
    return {
        "manifest_path": str(manifest_path),
        "chain_path": str(chain_path),
        "chain_head": chain_head,
    }


__all__ = [
    "RunManifest",
    "build_config_hashes",
    "guardrail_versions_from_instances",
    "emit_run_manifest",
]

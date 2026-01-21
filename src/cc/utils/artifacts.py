# src/cc/utils/artifacts.py
"""Artifact helpers for deterministic, audit-grade experiments."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactManifest:
    """Manifest describing an experiment run for reproducibility."""

    run_id: str
    rerun_command: str
    python_version: str
    platform: str
    git_commit: str | None
    adapter_versions: dict[str, str]


def ensure_artifact_dir(base_dir: Path, run_id: str) -> Path:
    """Create a deterministic artifact directory."""
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON deterministically with stable ordering."""
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def detect_git_commit() -> str | None:
    """Best-effort git commit detection (no external calls)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def build_manifest(
    run_id: str, rerun_command: str, adapter_versions: dict[str, str]
) -> ArtifactManifest:
    """Build a manifest for artifact directories."""
    return ArtifactManifest(
        run_id=run_id,
        rerun_command=rerun_command,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        git_commit=detect_git_commit(),
        adapter_versions=adapter_versions,
    )


def manifest_to_payload(manifest: ArtifactManifest) -> dict[str, Any]:
    return {
        "run_id": manifest.run_id,
        "rerun_command": manifest.rerun_command,
        "python_version": manifest.python_version,
        "platform": manifest.platform,
        "git_commit": manifest.git_commit,
        "adapter_versions": manifest.adapter_versions,
    }


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1 + (z**2) / total
    center = (phat + (z**2) / (2 * total)) / denom
    margin = z * ((phat * (1 - phat) + (z**2) / (4 * total)) / total) ** 0.5 / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def summarize_latency(samples_ms: Iterable[float]) -> dict[str, float]:
    """Compute deterministic latency summary stats."""
    values = sorted(float(v) for v in samples_ms)
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    mean = sum(values) / len(values)

    def percentile(p: float) -> float:
        idx = round((p / 100) * (len(values) - 1))
        return values[idx]

    return {
        "mean": mean,
        "p50": percentile(50),
        "p95": percentile(95),
        "p99": percentile(99),
    }

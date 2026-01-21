# tests/performance/test_adapter_perf.py
"""Performance benchmarks for adapters (gated by env)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from cc.adapters.guardrails_ai import GuardrailsAIAdapter
from cc.utils.artifacts import (
    build_manifest,
    ensure_artifact_dir,
    manifest_to_payload,
    summarize_latency,
    write_json,
)

pytestmark = pytest.mark.perf


class _FastGuard:
    version = "fast"

    def validate(self, text: str):
        class Result:
            validation_passed = True
            is_valid = True

        return Result()


def test_adapter_perf(tmp_path: Path) -> None:
    if os.getenv("CC_RUN_PERF") != "1":
        pytest.skip("Set CC_RUN_PERF=1 to run perf tests.")

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    adapter = GuardrailsAIAdapter(guard=_FastGuard())
    run_id = os.getenv("CC_PERF_RUN_ID", "perf-run")
    artifact_dir = ensure_artifact_dir(Path("artifacts/perf"), run_id)

    latencies = []
    for i in range(50):
        start = time.perf_counter()
        adapter.check(f"prompt-{i}", None, {})
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)

    metrics = summarize_latency(latencies)
    write_json(artifact_dir / "metrics.json", {"latency_ms": metrics, "samples": len(latencies)})

    plt.figure()
    plt.hist(latencies, bins=10)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(artifact_dir / "latency_hist.png")
    plt.close()

    manifest = build_manifest(
        run_id=run_id,
        rerun_command="CC_RUN_PERF=1 pytest -m perf tests/performance/test_adapter_perf.py",
        adapter_versions={adapter.name: adapter.version},
    )
    write_json(artifact_dir / "manifest.json", manifest_to_payload(manifest))

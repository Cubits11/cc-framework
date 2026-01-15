# tests/experiments/test_experiment_leak_metrics.py
"""Leak canary experiment (gated by marker/env)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import sys
from typing import List

import pytest

from cc.adapters.llama_guard import LlamaGuardAdapter
from cc.utils.artifacts import (
    build_manifest,
    ensure_artifact_dir,
    manifest_to_payload,
    summarize_latency,
    wilson_interval,
    write_json,
)


pytestmark = pytest.mark.experiment


def _should_run() -> bool:
    return os.getenv("CC_RUN_EXPERIMENTS") == "1"


def _make_canaries(n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    return [f"canary-{i}-{rng.randint(1000, 9999)}" for i in range(n)]


def _generator(prompt_text: str):
    return "safe", 0.0, {"vendor_field": "ok"}


def test_experiment_leak_metrics(tmp_path: Path) -> None:
    if not _should_run():
        pytest.skip("Set CC_RUN_EXPERIMENTS=1 to run leak experiments.")

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    adapter = LlamaGuardAdapter(generator=_generator, model_name="mock-llama-guard")
    run_id = os.getenv("CC_EXPERIMENT_RUN_ID", "canary-seed-0")
    artifact_dir = ensure_artifact_dir(Path("artifacts/adapters"), run_id)

    canaries = _make_canaries(20, seed=0)
    leak_flags = []
    latencies_ms = []
    for canary in canaries:
        start = 0.0
        decision = adapter.check(f"prompt {canary}", None, {})
        end = start
        latencies_ms.append((end - start) * 1000.0)
        blob = json.dumps(decision.audit, sort_keys=True) + json.dumps(decision.raw, sort_keys=True)
        leak_flags.append(1 if canary in blob else 0)

    total = len(leak_flags)
    leaks = sum(leak_flags)
    rate = leaks / total if total else 0.0
    ci_low, ci_high = wilson_interval(leaks, total)

    metrics = {
        "leak_rate": rate,
        "leak_rate_ci_low": ci_low,
        "leak_rate_ci_high": ci_high,
        "latency_ms": summarize_latency(latencies_ms),
        "total_samples": total,
        "leaks": leaks,
    }
    write_json(artifact_dir / "metrics.json", metrics)

    plt.figure()
    plt.hist(leak_flags, bins=2)
    plt.xlabel("Leak Flag")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(artifact_dir / "leak_score_hist.png")
    plt.close()

    manifest = build_manifest(
        run_id=run_id,
        rerun_command="CC_RUN_EXPERIMENTS=1 pytest -m experiment tests/experiments/test_experiment_leak_metrics.py",
        adapter_versions={adapter.name: adapter.version},
    )
    write_json(artifact_dir / "manifest.json", manifest_to_payload(manifest))

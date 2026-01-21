"""Unit tests for analysis utilities and reporting helpers."""

import hashlib

import pytest

from cc.analysis.cc_estimation import estimate_cc_metrics
from cc.analysis.reporting import metrics_to_csv, metrics_to_markdown, summarize_metrics
from tests._factories import mk_attack_result


def _sha256_hex(s: str) -> str:
    """Deterministic valid transcript_hash for synthetic tests."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _synthetic_results():
    """Create a small deterministic dataset for testing."""
    results = []

    # World 0: 3 successes out of 5 -> p0 = 0.6
    successes_w0 = [True, True, False, True, False]
    for i, success in enumerate(successes_w0):
        results.append(
            mk_attack_result(
                world_bit=0,
                success=success,
                attack_id=f"a{i}",
                guardrails_applied="",
                rng_seed=i,
            )
        )

    # World 1: 1 success out of 5 -> p1 = 0.2
    successes_w1 = [False, False, True, False, False]
    offset = len(successes_w0)
    for i, success in enumerate(successes_w1, start=offset):
        results.append(
            mk_attack_result(
                world_bit=1,
                success=success,
                attack_id=f"a{i}",
                guardrails_applied="g1",
                rng_seed=i,
            )
        )

    return results


def test_empirical_cc_metrics_and_reporting():
    results = _synthetic_results()
    metrics = estimate_cc_metrics(results, {"g1": 0.3})

    # Basic metric checks
    assert metrics["j_statistic"] == pytest.approx(0.4)
    assert metrics["p0"] == pytest.approx(0.6)
    assert metrics["p1"] == pytest.approx(0.2)
    assert metrics["cc_max"] == pytest.approx(4 / 3)
    assert metrics["delta_add"] == pytest.approx(0.1)

    summary = dict(summarize_metrics(metrics))
    md = metrics_to_markdown(summary.items())
    csv = metrics_to_csv(summary.items())

    assert "| j_statistic | 0.4 |" in md
    assert "j_statistic,0.4" in csv

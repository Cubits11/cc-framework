"""cc/analysis/cc_estimation.py
===============================

Utilities for estimating empirical composability metrics from collections of
:class:`~cc.core.models.AttackResult` objects.  These functions provide a thin
wrapper over the lower level routines in :mod:`cc.core.stats` and are intended
for use in analysis notebooks or reporting pipelines.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from cc.core.models import AttackResult
from cc.core.stats import compute_composability_coefficients, compute_j_statistic


def estimate_j_statistics(results: Iterable[AttackResult]) -> Dict[str, float]:
    """Compute empirical J statistic and world success rates.

    Args:
        results: Iterable of attack results from the two‑world protocol.

    Returns:
        Dictionary containing the J statistic (``j_statistic``) and the success
        rates in each world (``p0`` and ``p1``).
    """

    result_list = list(results)
    j_stat, p0, p1 = compute_j_statistic(result_list)
    return {"j_statistic": j_stat, "p0": p0, "p1": p1}


def estimate_cc_metrics(
    results: Iterable[AttackResult],
    individual_j: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute empirical composability metrics from attack results.

    This routine first estimates the J statistic and, if individual guardrail
    J‑statistics are provided, computes the composability coefficients using
    :func:`cc.core.stats.compute_composability_coefficients`.

    Args:
        results: Iterable of :class:`AttackResult` objects.
        individual_j: Optional mapping from guardrail identifier to its
            individual J‑statistic.  When provided, composability metrics are
            added to the returned dictionary.

    Returns:
        Dictionary with empirical metrics.  Always includes ``j_statistic``,
        ``p0`` and ``p1``.  If ``individual_j`` is supplied, the dictionary also
        contains ``cc_max``, ``delta_add`` and ``cc_multiplicative`` among other
        fields returned by :func:`compute_composability_coefficients`.
    """

    metrics = estimate_j_statistics(results)
    if individual_j:
        cc_metrics = compute_composability_coefficients(
            metrics["j_statistic"], individual_j, metrics["p0"]
        )
        metrics.update(cc_metrics)
    return metrics
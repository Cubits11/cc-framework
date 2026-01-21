"""Confidence interval helpers for Week 5 analyses."""

from __future__ import annotations

from collections.abc import Iterable
from statistics import NormalDist

import numpy as np


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    p_hat = successes / n
    dist = NormalDist()
    z = dist.inv_cdf(1 - alpha / 2)
    denom = 1 + z**2 / n
    centre = p_hat + z**2 / (2 * n)
    adj = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n)
    lower = max(0.0, (centre - adj) / denom)
    upper = min(1.0, (centre + adj) / denom)
    return float(lower), float(upper)


def bootstrap_ci(
    samples: Iterable[float], n_resamples: int = 200, alpha: float = 0.05, random_state: int = 42
) -> tuple[float, float]:
    """Percentile bootstrap interval for a scalar statistic.

    If ``n_resamples`` <= 0 the function simply returns the alpha/2 and (1-alpha/2)
    quantiles of the provided samples (useful when caller already generated
    bootstrap replicates).
    """
    arr = np.asarray(list(samples), dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if n_resamples <= 0:
        lo = float(np.quantile(arr, alpha / 2))
        hi = float(np.quantile(arr, 1 - alpha / 2))
        return lo, hi
    rng = np.random.default_rng(random_state)
    boot_means = []
    for _ in range(max(1, n_resamples)):
        idx = rng.integers(0, arr.size, arr.size)
        boot_means.append(float(np.mean(arr[idx])))
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return lo, hi

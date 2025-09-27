"""Confidence interval helpers for Week-5 analyses."""

from __future__ import annotations

from math import sqrt
from statistics import NormalDist
from typing import Sequence, Tuple, Union

import numpy as np


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("Sample size n must be positive for Wilson CI")
    if successes < 0 or successes > n:
        raise ValueError("successes must lie in [0, n]")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0,1)")

    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = (z / denom) * sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return float(lo), float(hi)


def bootstrap_ci(
    samples: Union[Sequence[float], Tuple[Sequence[float], Sequence[float]]],
    n_resamples: int = 200,
    alpha: float = 0.05,
    *,
    random_state: Union[int, np.random.Generator, None] = None,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for a mean or mean difference.

    If ``samples`` is a tuple of two sequences, the CI is computed for the
    difference of means (first minus second) with independent resampling.
    Otherwise, the CI is computed for the mean of the provided sample.
    """

    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0,1)")

    rng = (
        np.random.default_rng(random_state)
        if not isinstance(random_state, np.random.Generator)
        else random_state
    )

    if isinstance(samples, tuple) and len(samples) == 2:
        arr0 = np.asarray(samples[0], dtype=float).ravel()
        arr1 = np.asarray(samples[1], dtype=float).ravel()
        if arr0.size == 0 or arr1.size == 0:
            raise ValueError("Both sample arrays must be non-empty for difference CI")
        diffs = np.empty(n_resamples, dtype=float)
        for i in range(n_resamples):
            s0 = arr0[rng.integers(0, arr0.size, size=arr0.size)]
            s1 = arr1[rng.integers(0, arr1.size, size=arr1.size)]
            diffs[i] = float(s0.mean() - s1.mean())
        lo = float(np.quantile(diffs, alpha / 2.0))
        hi = float(np.quantile(diffs, 1.0 - alpha / 2.0))
        return lo, hi

    arr = np.asarray(samples, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("Sample must be non-empty for bootstrap CI")
    stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        resample = arr[rng.integers(0, arr.size, size=arr.size)]
        stats[i] = float(np.mean(resample))
    lo = float(np.quantile(stats, alpha / 2.0))
    hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
    return lo, hi

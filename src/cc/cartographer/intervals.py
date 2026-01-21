# src/cc/cartographer/intervals.py
"""
Interval estimators for Bernoulli proportions and CC propagation.

Provides:
  - wilson_ci_from_counts(k, n, delta)
  - wilson_ci(phat, n, delta)
  - bootstrap_proportion_ci(samples, delta, B=2000, seed=None)
  - cc_ci_wilson(p1_hat, n1, p0_hat, n0, D, delta=0.05)
  - cc_ci_bootstrap(y1_samples, y0_samples, D, delta=0.05, B=2000, seed=None)

Notes:
  * Wilson CI is preferred over Wald for small n.
  * CC CI propagation: if D = [L_D, U_D] for Δ := p1 - p0,
      then CC = (1 - Δ)/D (linear, decreasing), so
        Δ ∈ [L_D, U_D]  =>  CC ∈ [ (1-U_D)/D , (1-L_D)/D ].
"""

from __future__ import annotations

from math import log, sqrt

import numpy as np
from numpy.typing import NDArray

# ---------- Utilities ----------


def _validate_prob(name: str, x: float) -> None:
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"{name} must be in [0,1]. Got {x}.")


def _validate_n(name: str, n: int) -> None:
    if not (isinstance(n, (int, np.integer)) and n > 0):
        raise ValueError(f"{name} must be a positive integer. Got {n}.")


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


# Acklam's rational approximation to the standard normal inverse CDF (double precision).
# Source: https://web.archive.org/web/20150910044759/http://home.online.no/~pjacklam/notes/invnorm/
def _norm_ppf(p: float) -> float:
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0,1), got {p}")
    # Coefficients
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = sqrt(-2 * log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = sqrt(-2 * log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


# ---------- Wilson CI for a Bernoulli proportion ----------


def wilson_ci_from_counts(k: int, n: int, delta: float = 0.05) -> tuple[float, float]:
    """Two-sided Wilson score interval for a proportion with success count k.

    Args:
        k: number of successes (0..n)
        n: sample size (>0)
        delta: two-sided tail probability (e.g., 0.05 for 95% CI)

    Returns:
        (lo, hi) within [0,1].
    """
    _validate_n("n", n)
    if not (0 <= k <= n):
        raise ValueError(f"k must be in [0,n]. Got k={k}, n={n}")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1).")
    phat = k / n
    z = _norm_ppf(1.0 - 0.5 * delta)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    half = (z * sqrt((phat * (1.0 - phat) + z2 / (4 * n)) / n)) / denom
    lo = _clip01(center - half)
    hi = _clip01(center + half)
    return lo, hi


def wilson_ci(phat: float, n: int, delta: float = 0.05) -> tuple[float, float]:
    """Wilson score interval from phat and n."""
    _validate_prob("phat", phat)
    k = round(phat * n)
    return wilson_ci_from_counts(k, n, delta)


# ---------- Bootstrap percentile CI for a proportion ----------


def bootstrap_proportion_ci(
    samples: NDArray[np.float64],
    delta: float = 0.05,
    *,
    B: int = 2000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a Bernoulli proportion from {0,1} samples."""
    if samples.ndim != 1:
        samples = samples.ravel()
    n = samples.size
    _validate_n("n", int(n))
    if not ((samples == 0).all() or (samples == 1).any() or np.isin(samples, [0, 1]).all()):
        # Allow float 0/1 inputs; reject out-of-support values
        bad = np.setdiff1d(np.unique(samples), np.array([0.0, 1.0]))
        if bad.size:
            raise ValueError(f"samples must be 0/1; found {bad.tolist()}")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1).")
    rng = np.random.default_rng(seed)
    means = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n, endpoint=False)
        means[b] = float(np.mean(samples[idx]))
    lo = float(np.quantile(means, delta / 2.0, method="linear"))
    hi = float(np.quantile(means, 1.0 - delta / 2.0, method="linear"))
    return _clip01(lo), _clip01(hi)


# ---------- CC CIs via Wilson and Bootstrap ----------


def _diff_interval(p1_lo: float, p1_hi: float, p0_lo: float, p0_hi: float) -> tuple[float, float]:
    """Interval arithmetic for Δ = p1 - p0 given [p1_lo, p1_hi], [p0_lo, p0_hi]."""
    return p1_lo - p0_hi, p1_hi - p0_lo


def cc_ci_from_diff_interval(D: float, delta_lo: float, delta_hi: float) -> tuple[float, float]:
    """
    Map Δ-interval to CC-interval via CC = (1 - Δ) / D (monotone decreasing in Δ).
    """
    if D <= 0:
        raise ValueError("D must be > 0.")
    cc_lo = (1.0 - delta_hi) / D
    cc_hi = (1.0 - delta_lo) / D
    return cc_lo, cc_hi


def cc_ci_wilson(
    p1_hat: float,
    n1: int,
    p0_hat: float,
    n0: int,
    D: float,
    delta: float = 0.05,
) -> tuple[float, float]:
    """Two-sided CC CI by Wilson intervals on p1 and p0 and interval propagation."""
    p1_lo, p1_hi = wilson_ci(p1_hat, n1, delta)
    p0_lo, p0_hi = wilson_ci(p0_hat, n0, delta)
    d_lo, d_hi = _diff_interval(p1_lo, p1_hi, p0_lo, p0_hi)
    return cc_ci_from_diff_interval(D, d_lo, d_hi)


def cc_ci_bootstrap(
    y1_samples: NDArray[np.float64],
    y0_samples: NDArray[np.float64],
    D: float,
    delta: float = 0.05,
    *,
    B: int = 2000,
    seed: int | None = None,
) -> tuple[float, float]:
    """
    Two-sided CC CI via independent percentile bootstrap of p1 and p0.

    Args:
        y1_samples: 0/1 array for Y=1 composite (A∧B)
        y0_samples: 0/1 array for Y=0 composite (A∨B)
        D: denominator (>0)
        delta: two-sided risk
        B: bootstrap replicates
        seed: RNG seed
    """
    p1_lo, p1_hi = bootstrap_proportion_ci(y1_samples, delta, B=B, seed=seed)
    p0_lo, p0_hi = bootstrap_proportion_ci(
        y0_samples, delta, B=B, seed=None if seed is None else seed + 1
    )
    d_lo, d_hi = _diff_interval(p1_lo, p1_hi, p0_lo, p0_hi)
    return cc_ci_from_diff_interval(D, d_lo, d_hi)

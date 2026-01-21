"""cc/analysis/cc_estimation.py
===============================

Utilities for estimating empirical composability metrics from collections of
:class:`~cc.core.models.AttackResult` objects. These functions provide a thin
wrapper over lower-level routines in :mod:`cc.core.stats` and add the Week-3
FH-Bernstein finite-sample methods for CC at a fixed operating point θ.

New in Week-3 (FH-Bernstein):
  • `estimate_cc_methods_from_rates(...)`:
      Given (p1_hat, p0_hat, D, TPR/FPR pairs, n1, n0, α-cap),
      returns CC_hat, FH intervals I1/I0, variance envelopes, a finite-sample
      two-sided CI for CC, and optional sample-size targets for a tolerance t.
  • `estimate_cc_methods(...)`:
      Same as above, but derives (p1_hat, p0_hat) from the provided results via
      :func:`cc.core.stats.compute_j_statistic`.

This file keeps backward compatibility with:
  • `estimate_j_statistics(results)`
  • `estimate_cc_metrics(results, individual_j=None)`

Author: Pranav Bhave
Refined: 2025-09-12
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

# Week-3 methods (FH-Bernstein)
from cc.cartographer.bounds import (
    cc_confint,
    cc_two_sided_bound,
    fh_intervals,
    fh_var_envelope,
    needed_n_bernstein,
)
from cc.cartographer.intervals import cc_ci_bootstrap, cc_ci_wilson
from cc.core.models import AttackResult
from cc.core.stats import (
    compute_composability_coefficients,
    compute_j_statistic,
)

__all__ = [
    "cc_ci_bootstrap_from_samples",
    # Convenience CI wrappers
    "cc_ci_wilson_from_rates",
    "cc_confint_newcombe",
    "estimate_cc_methods",
    # Week-3 FH-Bernstein API
    "estimate_cc_methods_from_rates",
    "estimate_cc_metrics",
    # Back-compat
    "estimate_j_statistics",
    "newcombe_diff_ci",
    # Wilson/Newcombe helpers (exported for direct import)
    "wilson_interval",
]

# ---------------------------------------------------------------------------
# Backward-compatible helpers (existing API)
# ---------------------------------------------------------------------------


def estimate_j_statistics(results: Iterable[AttackResult]) -> dict[str, float]:
    """Compute empirical J statistic and world success rates."""
    result_list = list(results)
    j_stat, p0, p1 = compute_j_statistic(result_list)
    return {"j_statistic": j_stat, "p0": p0, "p1": p1}


def estimate_cc_metrics(
    results: Iterable[AttackResult],
    individual_j: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute empirical composability metrics from attack results."""
    metrics = estimate_j_statistics(results)
    if individual_j:
        cc_metrics = compute_composability_coefficients(
            metrics["j_statistic"], individual_j, metrics["p0"]
        )
        metrics.update(cc_metrics)
    return metrics


# ---------------------------------------------------------------------------
# Week-3 FH-Bernstein: CC at a fixed θ with policy binding and finite-sample CI
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CCPoint:
    """Point-estimate and context for CC at θ."""

    cc_hat: float
    D: float
    p1_hat: float
    p0_hat: float
    tpr_a: float
    tpr_b: float
    fpr_a: float
    fpr_b: float
    n1: int
    n0: int
    alpha_cap: float | None


@dataclass(frozen=True)
class CCBounds:
    """FH intervals and variance envelopes used in the Bernstein bound."""

    I1: tuple[float, float]  # for p1 (AND on Y=1)
    I0: tuple[float, float]  # for p0 (OR on Y=0)
    vbar1: float
    vbar0: float


@dataclass(frozen=True)
class CCCI:
    """Finite-sample CC confidence interval and planning outputs."""

    delta: float
    lo: float
    hi: float
    # Optional planning fields (only populated if target_t provided)
    target_t: float | None = None
    n1_star: float | None = None
    n0_star: float | None = None


def _validate_prob(name: str, x: float) -> None:
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"{name} must be in [0,1], got {x}.")


def _cc_hat(p1_hat: float, p0_hat: float, D: float) -> float:
    if D <= 0:
        raise ValueError("D must be > 0 for CC computation.")
    return (1.0 - (p1_hat - p0_hat)) / D


def estimate_cc_methods_from_rates(
    *,
    # Empirical rates at θ
    p1_hat: float,
    p0_hat: float,
    # Denominator and class-conditional operating rates
    D: float,
    tpr_a: float,
    tpr_b: float,
    fpr_a: float,
    fpr_b: float,
    # Sample sizes per class
    n1: int,
    n0: int,
    # Policy + risk
    alpha_cap: float | None = None,
    delta: float = 0.05,
    # Optional planner target for |CC_hat - CC|
    target_t: float | None = None,
    # Optional risk split across classes (δ1, δ0); if None, δ/2 each
    split: tuple[float, float] | None = None,
) -> dict[str, float | tuple[float, float] | dict]:
    """End-to-end FH-Bernstein CC workflow given scalar rates at θ."""
    # Validate inputs
    for nm, val in [
        ("p1_hat", p1_hat),
        ("p0_hat", p0_hat),
        ("tpr_a", tpr_a),
        ("tpr_b", tpr_b),
        ("fpr_a", fpr_a),
        ("fpr_b", fpr_b),
    ]:
        _validate_prob(nm, val)
    if n1 <= 0 or n0 <= 0:
        raise ValueError("n1 and n0 must be positive integers.")
    if D <= 0:
        raise ValueError("D must be > 0.")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1).")
    if split is not None:
        d1, d0 = split
        if d1 <= 0.0 or d0 <= 0.0 or abs((d1 + d0) - delta) > 1e-12:
            raise ValueError("split must be positive and sum to delta.")

    # FH intervals + variance envelopes (policy bind on I0)
    I1, I0 = fh_intervals(tpr_a, tpr_b, fpr_a, fpr_b, alpha_cap=alpha_cap)
    vbar1 = fh_var_envelope(I1)
    vbar0 = fh_var_envelope(I0)

    # Point estimate
    cc_hat = _cc_hat(p1_hat, p0_hat, D)

    # Finite-sample CI via Bernstein inversion
    lo, hi = cc_confint(
        n1=n1,
        n0=n0,
        p1_hat=p1_hat,
        p0_hat=p0_hat,
        D=D,
        I1=I1,
        I0=I0,
        delta=delta,
        split=split,
    )

    # Optional planning for a desired half-width (t)
    n1_star = n0_star = None
    if target_t is not None:
        n1_star, n0_star = needed_n_bernstein(
            t=target_t, D=D, I1=I1, I0=I0, delta=delta, split=split
        )

    # Convenience: probability bound at the realized half-width t=(hi-lo)/2
    t_obs = max(0.0, (hi - lo) / 2.0)
    bern_at_t_obs = cc_two_sided_bound(n1, n0, t_obs, D, I1, I0)

    # Package results
    point = CCPoint(
        cc_hat=cc_hat,
        D=D,
        p1_hat=p1_hat,
        p0_hat=p0_hat,
        tpr_a=tpr_a,
        tpr_b=tpr_b,
        fpr_a=fpr_a,
        fpr_b=fpr_b,
        n1=n1,
        n0=n0,
        alpha_cap=alpha_cap,
    )
    bounds = CCBounds(I1=I1, I0=I0, vbar1=vbar1, vbar0=vbar0)
    ci = CCCI(delta=delta, lo=lo, hi=hi, target_t=target_t, n1_star=n1_star, n0_star=n0_star)

    return {
        "point": asdict(point),
        "bounds": asdict(bounds),
        "ci": asdict(ci),
        "audit": {
            "bernstein_bound_at_halfwidth": bern_at_t_obs,
            "observed_halfwidth_t": t_obs,
            "delta": delta,
        },
    }


def estimate_cc_methods(
    results: Iterable[AttackResult],
    *,
    # Denominator and operating rates at θ (caller supplies from single-rail analysis)
    D: float,
    tpr_a: float,
    tpr_b: float,
    fpr_a: float,
    fpr_b: float,
    # Sample sizes per class
    n1: int,
    n0: int,
    # Policy + risk
    alpha_cap: float | None = None,
    delta: float = 0.05,
    # Optional planner target and risk split
    target_t: float | None = None,
    split: tuple[float, float] | None = None,
) -> dict[str, float | tuple[float, float] | dict]:
    """Derive (p1_hat, p0_hat) from results, then run FH-Bernstein CC workflow."""
    res = list(results)
    _j_stat, p0_hat, p1_hat = compute_j_statistic(res)
    return estimate_cc_methods_from_rates(
        p1_hat=p1_hat,
        p0_hat=p0_hat,
        D=D,
        tpr_a=tpr_a,
        tpr_b=tpr_b,
        fpr_a=fpr_a,
        fpr_b=fpr_b,
        n1=n1,
        n0=n0,
        alpha_cap=alpha_cap,
        delta=delta,
        target_t=target_t,
        split=split,
    )


# ---------------------------------------------------------------------------
# Convenience: Wilson and bootstrap CC CIs (delegating to cartographer.intervals)
# ---------------------------------------------------------------------------


def cc_ci_wilson_from_rates(
    p1_hat: float, n1: int, p0_hat: float, n0: int, D: float, delta: float = 0.05
) -> tuple[float, float]:
    """Two-sided CC CI using Wilson/Newcombe (difference of proportions) under denominator D."""
    return cc_ci_wilson(p1_hat, n1, p0_hat, n0, D, delta)


def cc_ci_bootstrap_from_samples(
    y1_samples,
    y0_samples,
    D: float,
    delta: float = 0.05,
    B: int = 2000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Two-sided CC CI via bootstrap from class-labeled samples."""
    return cc_ci_bootstrap(y1_samples, y0_samples, D, delta, B=B, seed=seed)


# ---------------------------------------------------------------------------
# Wilson / Newcombe helpers (exported for direct import)
# ---------------------------------------------------------------------------

from statistics import NormalDist


def _z_for(alpha: float = 0.05) -> float:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def wilson_interval(x: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a single proportion (two-sided)."""
    if n <= 0:
        return (0.0, 1.0)
    if x < 0 or x > n:
        raise ValueError("x must lie in [0, n].")
    z = _z_for(alpha)
    p = x / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z / denom) * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)) ** 0.5)
    lo, hi = center - margin, center + margin
    return (0.0 if lo < 0.0 else lo, 1.0 if hi > 1.0 else hi)


def newcombe_diff_ci(
    x1: int, n1: int, x0: int, n0: int, alpha: float = 0.05
) -> tuple[float, float]:
    """
    Newcombe (1998) method 10: Wilson intervals on each proportion, then ΔCI = (L1 - U0, U1 - L0),
    where Δ = p1 - p0. Better than Wald for small n or extreme p.
    """
    L1, U1 = wilson_interval(x1, n1, alpha)
    L0, U0 = wilson_interval(x0, n0, alpha)
    return (L1 - U0, U1 - L0)


def cc_confint_newcombe(
    x1: int,
    n1: int,
    x0: int,
    n0: int,
    D: float,
    alpha: float = 0.05,
    clamp01: bool = False,
) -> tuple[float, float]:
    """
    Convert Newcombe ΔCI to a CC CI via CC = (1 - Δ) / D.
    Mapping is monotone decreasing in Δ:
      Δ_lo, Δ_hi -> CC_lo = (1 - Δ_hi) / D, CC_hi = (1 - Δ_lo) / D
    """
    if D <= 0.0:
        raise ValueError("D must be > 0.")
    d_lo, d_hi = newcombe_diff_ci(x1, n1, x0, n0, alpha)
    cc_lo = (1.0 - d_hi) / D
    cc_hi = (1.0 - d_lo) / D
    if clamp01:
        cc_lo, cc_hi = max(0.0, cc_lo), min(1.0, cc_hi)
    return cc_lo, cc_hi

"""cc/analysis/cc_estimation.py
===============================

Utilities for estimating empirical composability metrics from collections of
:class:`~cc.core.models.AttackResult` objects. These functions provide a thin
wrapper over lower-level routines in :mod:`cc.core.stats` and add the Week-3
FH–Bernstein finite-sample methods for CC at a fixed operating point θ.

New in Week-3 (FH–Bernstein):
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
Refined: 2025-09-11
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Optional, Tuple, Union

from cc.core.models import AttackResult
from cc.core.stats import (
    compute_composability_coefficients,
    compute_j_statistic,
)

# Week-3 methods (FH–Bernstein) — single source of truth lives here:
from cc.cartographer.bounds import (
    fh_intervals,
    fh_var_envelope,
    cc_confint,
    needed_n_bernstein,
    cc_two_sided_bound,
)
from cc.cartographer.intervals import cc_ci_wilson, cc_ci_bootstrap
# ---------------------------------------------------------------------------
# Backward-compatible helpers (existing API)
# ---------------------------------------------------------------------------

def estimate_j_statistics(results: Iterable[AttackResult]) -> Dict[str, float]:
    """Compute empirical J statistic and world success rates.

    Args:
        results: Iterable of attack results from the two-world protocol.

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
    J-statistics are provided, computes the composability coefficients using
    :func:`cc.core.stats.compute_composability_coefficients`.

    Args:
        results: Iterable of :class:`AttackResult` objects.
        individual_j: Optional mapping {guardrail_id: J_r}. When provided,
            composability metrics are added via :func:`compute_composability_coefficients`.

    Returns:
        Dictionary with empirical metrics. Always includes ``j_statistic``,
        ``p0`` and ``p1``. If ``individual_j`` is supplied, the dictionary also
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


# ---------------------------------------------------------------------------
# Week-3 FH–Bernstein: CC at a fixed θ with policy binding and finite-sample CI
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
    alpha_cap: Optional[float]


@dataclass(frozen=True)
class CCBounds:
    """FH intervals and variance envelopes used in the Bernstein bound."""
    I1: Tuple[float, float]  # for p1 (AND on Y=1)
    I0: Tuple[float, float]  # for p0 (OR on Y=0)
    vbar1: float
    vbar0: float


@dataclass(frozen=True)
class CCCI:
    """Finite-sample CC confidence interval and planning outputs."""
    delta: float
    lo: float
    hi: float
    # Optional planning fields (only populated if target_t provided)
    target_t: Optional[float] = None
    n1_star: Optional[float] = None
    n0_star: Optional[float] = None


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
    alpha_cap: Optional[float] = None,
    delta: float = 0.05,
    # Optional planner target for |CC_hat - CC|
    target_t: Optional[float] = None,
    # Optional risk split across classes (δ1, δ0); if None, δ/2 each
    split: Optional[Tuple[float, float]] = None,
) -> Dict[str, Union[float, Tuple[float, float], dict]]:
    """End-to-end FH–Bernstein CC workflow given scalar rates at θ.

    Returns a dictionary containing:
      • point:  CCPoint
      • bounds: CCBounds (I1/I0 + vbar1/vbar0)
      • ci:     CCCI     (two-sided CI for CC at risk δ; plus planner outputs if target_t)
      • audit:  {'bernstein_bound_at_halfwidth': ..., 'delta': ...} convenience fields

    Notes:
      - p1_hat = P(A∧B=1 | Y=1), p0_hat = P(A∨B=1 | Y=0) empirically at θ
      - D      = min_r(1 - J_r(θ_r^*)), supplied by caller (single-rail reference)
      - FPR policy α-cap is bound into I0 (upper) automatically if provided
    """
    # Validate inputs
    for nm, val in [
        ("p1_hat", p1_hat), ("p0_hat", p0_hat),
        ("tpr_a", tpr_a), ("tpr_b", tpr_b),
        ("fpr_a", fpr_a), ("fpr_b", fpr_b),
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
        n1=n1, n0=n0,
        p1_hat=p1_hat, p0_hat=p0_hat,
        D=D, I1=I1, I0=I0,
        delta=delta, split=split,
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
        cc_hat=cc_hat, D=D, p1_hat=p1_hat, p0_hat=p0_hat,
        tpr_a=tpr_a, tpr_b=tpr_b, fpr_a=fpr_a, fpr_b=fpr_b,
        n1=n1, n0=n0, alpha_cap=alpha_cap,
    )
    bounds = CCBounds(I1=I1, I0=I0, vbar1=vbar1, vbar0=vbar0)
    ci = CCCI(delta=delta, lo=lo, hi=hi, target_t=target_t,
              n1_star=n1_star, n0_star=n0_star)

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
    alpha_cap: Optional[float] = None,
    delta: float = 0.05,
    # Optional planner target and risk split
    target_t: Optional[float] = None,
    split: Optional[Tuple[float, float]] = None,
) -> Dict[str, Union[float, Tuple[float, float], dict]]:
    """Derive (p1_hat, p0_hat) from results, then run FH–Bernstein CC workflow.

    Args:
        results: Iterable of AttackResult from the two-world protocol at θ.
        D: Denominator (>0) from single-rail reference, D = min_r (1 - J_r(θ_r*)).
        tpr_a, tpr_b, fpr_a, fpr_b: Operating TPR/FPR for rails A,B at θ.
        n1, n0: Class-conditional sample sizes.
        alpha_cap: Optional FPR policy cap applied to I0.
        delta: Total two-sided risk for the CI (default 0.05).
        target_t: Optional desired half-width for planning.
        split: Optional class-wise risk split (δ1, δ0) summing to δ.

    Returns:
        Dictionary with 'point', 'bounds', 'ci', and 'audit' entries
        (see `estimate_cc_methods_from_rates` for fields).
    """
    res = list(results)
    j_stat, p0_hat, p1_hat = compute_j_statistic(res)
    return estimate_cc_methods_from_rates(
        p1_hat=p1_hat, p0_hat=p0_hat,
        D=D, tpr_a=tpr_a, tpr_b=tpr_b, fpr_a=fpr_a, fpr_b=fpr_b,
        n1=n1, n0=n0, alpha_cap=alpha_cap, delta=delta,
        target_t=target_t, split=split,
    )

def cc_ci_wilson_from_rates(p1_hat: float, n1: int, p0_hat: float, n0: int, D: float, delta: float=0.05):
    return cc_ci_wilson(p1_hat, n1, p0_hat, n0, D, delta)

def cc_ci_bootstrap_from_samples(y1_samples, y0_samples, D: float, delta: float=0.05, B: int=2000, seed=None):
    return cc_ci_bootstrap(y1_samples, y0_samples, D, delta, B=B, seed=seed)
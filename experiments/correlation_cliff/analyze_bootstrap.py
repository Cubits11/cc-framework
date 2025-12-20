# experiments/correlation_cliff/analyze_bootstrap.py
from __future__ import annotations

"""
Correlation Cliff — Bootstrap + BCa Uncertainty Module
=====================================================

This module provides *rigorous uncertainty quantification* for the Correlation Cliff
experiment, targeted at the exact data shape produced by simulate.py:

- Each (lambda, replicate) produces *multinomial cell counts* for each world:
    world 0: n00_0, n01_0, n10_0, n11_0
    world 1: n00_1, n01_1, n10_1, n11_1
  with per-world sample size n (usually equal across worlds).

- From counts, we compute:
    pA_hat^w, pB_hat^w, p11_hat^w, pC_hat^w,
    J_A_hat, J_B_hat, Jbest_hat, J_C_hat, CC_hat,
    plus dependence summaries (phi, Kendall tau-a).

What this module does
---------------------
A) Per-lambda BCa CIs (parametric multinomial bootstrap + jackknife acceleration)
   for metrics such as CC_hat and JC_hat.

   Why parametric multinomial bootstrap?
   - In this experiment, the observable data at each lambda is a 2x2 contingency table
     (A,B) in each world. The multinomial is the natural sampling model.
   - The parametric bootstrap draws tables from Multinomial(n, p_hat_cells) where
     p_hat_cells are empirical frequencies from the observed table.
   - This gives fast, stable CIs that preserve the multinomial structure.

   Why BCa (vs percentile)?
   - BCa corrects for bias (z0) and skewness (via acceleration a).
   - In ratio metrics like CC = JC/Jbest, skewness is common; BCa is the right default.

B) Optional curve-level threshold bootstrap:
   - For an "observed curve" (one replicate across all lambdas), resample each lambda's
     tables via multinomial bootstrap, recompute CC(lambda), and solve for lambda* where
     CC crosses 1 by interpolation. This gives a sampling distribution over lambda*.

Important caveats (and why they’re OK here)
-------------------------------------------
1) Kinks from absolute values:
   JC = |pC^1 - pC^0| can have a non-smooth point when (pC^1 - pC^0) crosses 0.
   BCa and jackknife rely on smoothness assumptions; near a kink, any asymptotics can wobble.

   What we do:
   - We still compute BCa (it often remains useful),
   - But we emit diagnostics flags if we detect sign flips or tiny gaps.

2) Denominator noise in CC:
   CC_hat uses Jbest_hat in the denominator, which can be close to 0 in edge configs.
   We guard: if Jbest_hat <= eps, CC_hat becomes NaN and we refuse to compute BCa.

3) “Observed dataset” selection:
   For simulation experiments, you usually have multiple independent replicates per lambda.
   BCa is defined for *one* observed dataset; we implement BCa for a chosen replicate ID.
   (You can run BCa on rep=0, or for multiple reps and summarize across reps.)

CLI usage
---------
Compute BCa table from sim_long.csv (choose replicate 0):
  python analyze_bootstrap.py --sim_long artifacts/.../sim_long.csv --out artifacts/.../bca --rep 0 --B 2000

Then merge BCa columns into sim_summary.csv for plotting:
  (run_all.py intentionally keeps this optional; figures.py will auto-prefer BCa cols if present.)

Dependencies
------------
- numpy, pandas
- NO SciPy required: we implement normal CDF/PPF internally.

"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Literal

import math
import json

import numpy as np
import pandas as pd

Rule = Literal["OR", "AND"]


# =============================================================================
# Normal CDF / PPF (no SciPy)
# =============================================================================
def _norm_cdf(x: float) -> float:
    # Phi(x) via erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """
    Inverse standard normal CDF (Acklam approximation).
    Accurate to ~1e-9 in central region, sufficient for BCa quantiles.
    """
    if not (0.0 < p < 1.0):
        if p == 0.0:
            return -float("inf")
        if p == 1.0:
            return float("inf")
        raise ValueError(f"p must be in (0,1), got {p}")

    # Coefficients from Peter J. Acklam's approximation
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
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den

    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return -(num / den)

    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return num / den


def _quantile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


# =============================================================================
# Stat computation from multinomial counts
# =============================================================================
def _phi_from_counts(pA: float, pB: float, p11: float) -> float:
    denom = pA * (1 - pA) * pB * (1 - pB)
    if denom <= 0.0:
        return float("nan")
    return (p11 - pA * pB) / math.sqrt(denom)


def _tau_a_from_cells(p00: float, p01: float, p10: float, p11: float) -> float:
    # tau-a = 2(p00 p11 - p01 p10)
    return 2.0 * (p00 * p11 - p01 * p10)


def _pC_from_pA_pB_p11(rule: Rule, pA: float, pB: float, p11: float) -> float:
    if rule == "OR":
        return pA + pB - p11
    if rule == "AND":
        return p11
    raise ValueError(f"Unknown rule: {rule}")


def compute_metrics_from_counts(
    *,
    rule: Rule,
    n00_0: int, n01_0: int, n10_0: int, n11_0: int,
    n00_1: int, n01_1: int, n10_1: int, n11_1: int,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Compute (JC_hat, CC_hat, dependence summaries, etc.) from cell counts.

    The counts are assumed to correspond to the joint distribution of (A,B) with cells:
      00, 01, 10, 11
    in each world.

    Returns a dict with keys:
      pA_hat_0, pB_hat_0, p11_hat_0, pC_hat_0, phi_hat_0, tau_hat_0,
      pA_hat_1, ..., phi_hat_1, tau_hat_1,
      JA_hat, JB_hat, Jbest_hat, dC_hat, JC_hat, CC_hat,
      phi_hat_avg, tau_hat_avg
    """
    n0 = n00_0 + n01_0 + n10_0 + n11_0
    n1 = n00_1 + n01_1 + n10_1 + n11_1
    if n0 <= 0 or n1 <= 0:
        raise ValueError(f"Non-positive sample size in counts: n0={n0}, n1={n1}")

    # World 0 proportions
    p00_0 = n00_0 / n0
    p01_0 = n01_0 / n0
    p10_0 = n10_0 / n0
    p11_0 = n11_0 / n0
    pA0 = p10_0 + p11_0
    pB0 = p01_0 + p11_0
    pC0 = _pC_from_pA_pB_p11(rule, pA0, pB0, p11_0)
    phi0 = _phi_from_counts(pA0, pB0, p11_0)
    tau0 = _tau_a_from_cells(p00_0, p01_0, p10_0, p11_0)

    # World 1 proportions
    p00_1 = n00_1 / n1
    p01_1 = n01_1 / n1
    p10_1 = n10_1 / n1
    p11_1 = n11_1 / n1
    pA1 = p10_1 + p11_1
    pB1 = p01_1 + p11_1
    pC1 = _pC_from_pA_pB_p11(rule, pA1, pB1, p11_1)
    phi1 = _phi_from_counts(pA1, pB1, p11_1)
    tau1 = _tau_a_from_cells(p00_1, p01_1, p10_1, p11_1)

    JA = abs(pA1 - pA0)
    JB = abs(pB1 - pB0)
    Jbest = max(JA, JB)

    dC = pC1 - pC0
    JC = abs(dC)

    if Jbest <= eps:
        CC = float("nan")
    else:
        CC = JC / Jbest

    return {
        "n0": float(n0),
        "n1": float(n1),
        # World 0
        "pA_hat_0": float(pA0),
        "pB_hat_0": float(pB0),
        "p11_hat_0": float(p11_0),
        "pC_hat_0": float(pC0),
        "phi_hat_0": float(phi0),
        "tau_hat_0": float(tau0),
        # World 1
        "pA_hat_1": float(pA1),
        "pB_hat_1": float(pB1),
        "p11_hat_1": float(p11_1),
        "pC_hat_1": float(pC1),
        "phi_hat_1": float(phi1),
        "tau_hat_1": float(tau1),
        # Metrics
        "JA_hat": float(JA),
        "JB_hat": float(JB),
        "Jbest_hat": float(Jbest),
        "dC_hat": float(dC),
        "JC_hat": float(JC),
        "CC_hat": float(CC),
        "phi_hat_avg": float(0.5 * (phi0 + phi1)),
        "tau_hat_avg": float(0.5 * (tau0 + tau1)),
        # Kink diagnostic
        "dC_sign": float(1.0 if dC > 0 else (-1.0 if dC < 0 else 0.0)),
        "kink_risk": float(1.0 if abs(dC) < 3e-3 else 0.0),
    }


# =============================================================================
# Multinomial parametric bootstrap
# =============================================================================
def _multinomial_draw(rng: np.random.Generator, n: int, p: np.ndarray) -> np.ndarray:
    # p length 4 for [00,01,10,11]
    return rng.multinomial(n, p, size=None)


def _counts_to_probs_4(n00: int, n01: int, n10: int, n11: int) -> np.ndarray:
    n = n00 + n01 + n10 + n11
    if n <= 0:
        raise ValueError("non-positive total count for probs")
    p = np.array([n00 / n, n01 / n, n10 / n, n11 / n], dtype=float)
    # guard numeric drift
    p = np.clip(p, 0.0, 1.0)
    s = float(p.sum())
    if s <= 0:
        raise ValueError("invalid probability sum")
    return p / s


# =============================================================================
# Jackknife acceleration (multinomial leave-one-out)
# =============================================================================
def _jackknife_acceleration_multinomial(
    *,
    rule: Rule,
    counts0: Tuple[int, int, int, int],
    counts1: Tuple[int, int, int, int],
    stat_key: Literal["CC_hat", "JC_hat"] = "CC_hat",
    eps: float = 1e-12,
) -> float:
    """
    Acceleration a for BCa via a multinomial leave-one-out jackknife.

    Key idea:
    - The observed data in each world is n draws from 4 categories (00,01,10,11).
    - Leave-one-out removes a single observation from one of these categories.
    - There are only 4 unique patterns per world, but each repeats count times.
      We compute a weighted jackknife over these patterns.

    We treat the *combined dataset* across both worlds with total N = n0 + n1
    leave-one-out replications, where removing one observation affects only
    the world it belongs to.

    If the jackknife variance collapses (denom==0), return a=0.
    """
    (n00_0, n01_0, n10_0, n11_0) = counts0
    (n00_1, n01_1, n10_1, n11_1) = counts1

    n0 = n00_0 + n01_0 + n10_0 + n11_0
    n1 = n00_1 + n01_1 + n10_1 + n11_1
    N = n0 + n1
    if N <= 1:
        return 0.0

    # Weighted jackknife estimates theta_(i) for each leave-one-out observation
    thetas: List[float] = []
    weights: List[int] = []

    def add_pattern(weight: int, c0: Tuple[int, int, int, int], c1: Tuple[int, int, int, int]) -> None:
        if weight <= 0:
            return
        m = compute_metrics_from_counts(
            rule=rule,
            n00_0=c0[0], n01_0=c0[1], n10_0=c0[2], n11_0=c0[3],
            n00_1=c1[0], n01_1=c1[1], n10_1=c1[2], n11_1=c1[3],
            eps=eps,
        )
        theta = float(m[stat_key])
        thetas.append(theta)
        weights.append(int(weight))

    # World 0 removals
    if n00_0 > 0: add_pattern(n00_0, (n00_0 - 1, n01_0, n10_0, n11_0), (n00_1, n01_1, n10_1, n11_1))
    if n01_0 > 0: add_pattern(n01_0, (n00_0, n01_0 - 1, n10_0, n11_0), (n00_1, n01_1, n10_1, n11_1))
    if n10_0 > 0: add_pattern(n10_0, (n00_0, n01_0, n10_0 - 1, n11_0), (n00_1, n01_1, n10_1, n11_1))
    if n11_0 > 0: add_pattern(n11_0, (n00_0, n01_0, n10_0, n11_0 - 1), (n00_1, n01_1, n10_1, n11_1))

    # World 1 removals
    if n00_1 > 0: add_pattern(n00_1, (n00_0, n01_0, n10_0, n11_0), (n00_1 - 1, n01_1, n10_1, n11_1))
    if n01_1 > 0: add_pattern(n01_1, (n00_0, n01_0,_

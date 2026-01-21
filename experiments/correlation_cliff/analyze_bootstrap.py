# experiments/correlation_cliff/analyze_bootstrap.py
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

Important caveats (and why they're OK here)
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

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        return num / den

    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        return -(num / den)

    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
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
    n00_0: int,
    n01_0: int,
    n10_0: int,
    n11_0: int,
    n00_1: int,
    n01_1: int,
    n10_1: int,
    n11_1: int,
    eps: float = 1e-12,
) -> dict[str, float]:
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

    CC = float("nan") if Jbest <= eps else JC / Jbest

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
    counts0: tuple[int, int, int, int],
    counts1: tuple[int, int, int, int],
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
    thetas: list[float] = []
    weights: list[int] = []

    def add_pattern(
        weight: int, c0: tuple[int, int, int, int], c1: tuple[int, int, int, int]
    ) -> None:
        if weight <= 0:
            return
        m = compute_metrics_from_counts(
            rule=rule,
            n00_0=c0[0],
            n01_0=c0[1],
            n10_0=c0[2],
            n11_0=c0[3],
            n00_1=c1[0],
            n01_1=c1[1],
            n10_1=c1[2],
            n11_1=c1[3],
            eps=eps,
        )
        theta = float(m[stat_key])
        thetas.append(theta)
        weights.append(int(weight))

    # World 0 removals
    if n00_0 > 0:
        add_pattern(n00_0, (n00_0 - 1, n01_0, n10_0, n11_0), (n00_1, n01_1, n10_1, n11_1))
    if n01_0 > 0:
        add_pattern(n01_0, (n00_0, n01_0 - 1, n10_0, n11_0), (n00_1, n01_1, n10_1, n11_1))
    if n10_0 > 0:
        add_pattern(n10_0, (n00_0, n01_0, n10_0 - 1, n11_0), (n00_1, n01_1, n10_1, n11_1))
    if n11_0 > 0:
        add_pattern(n11_0, (n00_0, n01_0, n10_0, n11_0 - 1), (n00_1, n01_1, n10_1, n11_1))

    # World 1 removals
    if n00_1 > 0:
        add_pattern(n00_1, (n00_0, n01_0, n10_0, n11_0), (n00_1 - 1, n01_1, n10_1, n11_1))
    if n01_1 > 0:
        add_pattern(n01_1, (n00_0, n01_0, n10_0, n11_0), (n00_1, n01_1 - 1, n10_1, n11_1))
    if n10_1 > 0:
        add_pattern(n10_1, (n00_0, n01_0, n10_0, n11_0), (n00_1, n01_1, n10_1 - 1, n11_1))
    if n11_1 > 0:
        add_pattern(n11_1, (n00_0, n01_0, n10_0, n11_0), (n00_1, n01_1, n10_1, n11_1 - 1))

    if not thetas:
        return 0.0

    th = np.asarray(thetas, dtype=float)
    w = np.asarray(weights, dtype=float)
    W = float(w.sum())
    if W <= 0:
        return 0.0

    theta_dot = float(np.sum(w * th) / W)

    dif = theta_dot - th
    s2 = float(np.sum(w * dif * dif))
    s3 = float(np.sum(w * dif * dif * dif))

    den = 6.0 * (s2**1.5)
    if den <= 0.0 or not math.isfinite(den):
        return 0.0

    a = s3 / den
    if not math.isfinite(a):
        return 0.0
    return float(a)


# =============================================================================
# BCa core
# =============================================================================
def bca_interval(
    *,
    theta_hat: float,
    theta_boot: np.ndarray,
    acceleration: float,
    alpha: float = 0.05,
) -> dict[str, float]:
    """
    Compute BCa CI endpoints for a scalar statistic.

    Returns:
      {
        "bca_lo": ...,
        "bca_hi": ...,
        "z0": ...,
        "a": ...,
        "alpha1": ...,
        "alpha2": ...,
      }
    """
    x = np.asarray(theta_boot, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 20:
        return {
            "bca_lo": float("nan"),
            "bca_hi": float("nan"),
            "z0": float("nan"),
            "a": float(acceleration),
            "alpha1": float("nan"),
            "alpha2": float("nan"),
        }

    # Bias correction z0
    prop_less = float(np.mean(x < theta_hat))
    # Guard endpoints
    prop_less = min(1.0 - 1e-9, max(1e-9, prop_less))
    z0 = _norm_ppf(prop_less)

    z_low = _norm_ppf(alpha / 2.0)
    z_high = _norm_ppf(1.0 - alpha / 2.0)

    a = float(acceleration)

    def adj_alpha(z: float) -> float:
        num = z0 + z
        den = 1.0 - a * num
        if den == 0.0:
            return float("nan")
        return _norm_cdf(z0 + (num / den))

    alpha1 = adj_alpha(z_low)
    alpha2 = adj_alpha(z_high)
    alpha1 = min(1.0, max(0.0, alpha1))
    alpha2 = min(1.0, max(0.0, alpha2))

    lo = _quantile(np.sort(x), alpha1)
    hi = _quantile(np.sort(x), alpha2)

    return {
        "bca_lo": float(lo),
        "bca_hi": float(hi),
        "z0": float(z0),
        "a": float(a),
        "alpha1": float(alpha1),
        "alpha2": float(alpha2),
    }


# =============================================================================
# Per-lambda BCa computation
# =============================================================================
@dataclass(frozen=True)
class BcaConfig:
    rule: Rule
    B: int = 2000
    alpha: float = 0.05
    seed: int = 123
    eps: float = 1e-12
    stat_keys: tuple[Literal["CC_hat", "JC_hat"], ...] = ("CC_hat", "JC_hat")


def bca_for_one_lambda(
    *,
    lam: float,
    rule: Rule,
    counts0: tuple[int, int, int, int],
    counts1: tuple[int, int, int, int],
    cfg: BcaConfig,
) -> dict[str, float]:
    """
    Compute BCa CIs for CC_hat and JC_hat for a single lambda row.
    """
    rng = np.random.default_rng(cfg.seed + round(lam * 1e6) % 10_000_000)

    (n00_0, n01_0, n10_0, n11_0) = counts0
    (n00_1, n01_1, n10_1, n11_1) = counts1

    # Original stat
    m0 = compute_metrics_from_counts(
        rule=rule,
        n00_0=n00_0,
        n01_0=n01_0,
        n10_0=n10_0,
        n11_0=n11_0,
        n00_1=n00_1,
        n01_1=n01_1,
        n10_1=n10_1,
        n11_1=n11_1,
        eps=cfg.eps,
    )

    n0 = int(n00_0 + n01_0 + n10_0 + n11_0)
    n1 = int(n00_1 + n01_1 + n10_1 + n11_1)
    p0 = _counts_to_probs_4(n00_0, n01_0, n10_0, n11_0)
    p1 = _counts_to_probs_4(n00_1, n01_1, n10_1, n11_1)

    out: dict[str, float] = {"lambda": float(lam)}

    # Some diagnostics (useful in paper + debugging)
    out["kink_risk"] = float(m0["kink_risk"])
    out["dC_sign"] = float(m0["dC_sign"])
    out["Jbest_hat"] = float(m0["Jbest_hat"])

    for key in cfg.stat_keys:
        theta_hat = float(m0[key])
        if not math.isfinite(theta_hat):
            out[f"{key}_bca_lo"] = float("nan")
            out[f"{key}_bca_hi"] = float("nan")
            out[f"{key}_z0"] = float("nan")
            out[f"{key}_a"] = float("nan")
            continue

        # Acceleration via multinomial jackknife
        a = _jackknife_acceleration_multinomial(
            rule=rule,
            counts0=counts0,
            counts1=counts1,
            stat_key=key,
            eps=cfg.eps,
        )

        # Bootstrap distribution
        boots = np.empty(cfg.B, dtype=float)
        for b in range(cfg.B):
            c0 = _multinomial_draw(rng, n0, p0)  # [00,01,10,11]
            c1 = _multinomial_draw(rng, n1, p1)
            mb = compute_metrics_from_counts(
                rule=rule,
                n00_0=int(c0[0]),
                n01_0=int(c0[1]),
                n10_0=int(c0[2]),
                n11_0=int(c0[3]),
                n00_1=int(c1[0]),
                n01_1=int(c1[1]),
                n10_1=int(c1[2]),
                n11_1=int(c1[3]),
                eps=cfg.eps,
            )
            boots[b] = float(mb[key])

        ci = bca_interval(theta_hat=theta_hat, theta_boot=boots, acceleration=a, alpha=cfg.alpha)

        out[f"{key}_hat"] = float(theta_hat)
        out[f"{key}_bca_lo"] = float(ci["bca_lo"])
        out[f"{key}_bca_hi"] = float(ci["bca_hi"])
        out[f"{key}_z0"] = float(ci["z0"])
        out[f"{key}_a"] = float(ci["a"])
        out[f"{key}_alpha1"] = float(ci["alpha1"])
        out[f"{key}_alpha2"] = float(ci["alpha2"])

        # For convenience, also store simple percentile CI (sanity)
        xs = np.sort(boots[np.isfinite(boots)])
        out[f"{key}_pct_lo"] = _quantile(xs, cfg.alpha / 2.0)
        out[f"{key}_pct_hi"] = _quantile(xs, 1.0 - cfg.alpha / 2.0)

    return out


def bca_table_for_curve(
    *,
    df_counts: pd.DataFrame,
    rule: Rule,
    B: int = 2000,
    alpha: float = 0.05,
    seed: int = 123,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Compute BCa table for each lambda row.

    df_counts must contain columns:
      lambda,
      n00_0,n01_0,n10_0,n11_0,
      n00_1,n01_1,n10_1,n11_1
    """
    required = [
        "lambda",
        "n00_0",
        "n01_0",
        "n10_0",
        "n11_0",
        "n00_1",
        "n01_1",
        "n10_1",
        "n11_1",
    ]
    missing = [c for c in required if c not in df_counts.columns]
    if missing:
        raise ValueError(f"df_counts missing required columns: {missing}")

    cfg = BcaConfig(rule=rule, B=int(B), alpha=float(alpha), seed=int(seed), eps=float(eps))
    rows: list[dict[str, float]] = []

    d = df_counts.sort_values("lambda").reset_index(drop=True)
    for _, r in d.iterrows():
        lam = float(r["lambda"])
        counts0 = (int(r["n00_0"]), int(r["n01_0"]), int(r["n10_0"]), int(r["n11_0"]))
        counts1 = (int(r["n00_1"]), int(r["n01_1"]), int(r["n10_1"]), int(r["n11_1"]))
        rows.append(
            bca_for_one_lambda(lam=lam, rule=rule, counts0=counts0, counts1=counts1, cfg=cfg)
        )

    return pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)


# =============================================================================
# Selecting an "observed curve" from sim_long
# =============================================================================
def select_observed_counts_from_sim_long(
    df_long: pd.DataFrame,
    *,
    rep: int = 0,
) -> pd.DataFrame:
    """
    Select one replicate per lambda from sim_long, returning only the count columns.

    Accepts either column name 'rep' or 'replicate' for replicate id.
    """
    rep_col = None
    for c in ["rep", "replicate", "rep_id", "replicate_id"]:
        if c in df_long.columns:
            rep_col = c
            break
    if rep_col is None:
        raise ValueError(
            "sim_long DataFrame must include a replicate id column (rep/replicate/rep_id/replicate_id)."
        )

    d = df_long[df_long[rep_col].astype(int) == int(rep)].copy()
    if d.empty:
        reps = sorted(df_long[rep_col].astype(int).unique().tolist())
        raise ValueError(f"No rows found for rep={rep}. Available reps include: {reps[:10]}...")

    needed = [
        "lambda",
        "n00_0",
        "n01_0",
        "n10_0",
        "n11_0",
        "n00_1",
        "n01_1",
        "n10_1",
        "n11_1",
    ]
    missing = [c for c in needed if c not in d.columns]
    if missing:
        raise ValueError(
            "sim_long does not have required count columns. "
            "Make sure simulate.py outputs multinomial counts per world.\n"
            f"Missing: {missing}"
        )

    return d[needed].sort_values("lambda").reset_index(drop=True)


# =============================================================================
# Curve-level threshold bootstrap (percentile)
# =============================================================================
def _interp_root(x: np.ndarray, y: np.ndarray, target: float) -> float | None:
    if len(x) < 2:
        return None
    for i in range(len(x) - 1):
        y0, y1 = float(y[i]), float(y[i + 1])
        if y0 == target:
            return float(x[i])
        if (y0 - target) * (y1 - target) <= 0:
            if y1 == y0:
                return float(x[i])
            t = (target - y0) / (y1 - y0)
            return float(x[i] + t * (x[i + 1] - x[i]))
    return None


def threshold_bootstrap_percentile(
    *,
    df_counts: pd.DataFrame,
    rule: Rule,
    B: int = 2000,
    seed: int = 123,
    eps: float = 1e-12,
) -> dict[str, float]:
    """
    Bootstrap the entire CC(lambda) curve and estimate a sampling distribution for lambda*.

    Approach:
      For b in 1..B:
        For each lambda row:
          draw multinomial tables in each world from empirical p_hat_cells
          compute CC_hat(lambda)
        Solve CC_hat(lambda)=1 by interpolation.
      Return percentile CI of lambda* (ignoring replicates with no crossing).

    This is *curve-level UQ* and is what you want for a paper when you report lambda*.
    """
    required = [
        "lambda",
        "n00_0",
        "n01_0",
        "n10_0",
        "n11_0",
        "n00_1",
        "n01_1",
        "n10_1",
        "n11_1",
    ]
    missing = [c for c in required if c not in df_counts.columns]
    if missing:
        raise ValueError(f"df_counts missing required columns: {missing}")

    d = df_counts.sort_values("lambda").reset_index(drop=True)

    lambdas = d["lambda"].to_numpy(dtype=float)
    n0 = (d["n00_0"] + d["n01_0"] + d["n10_0"] + d["n11_0"]).to_numpy(dtype=int)
    n1 = (d["n00_1"] + d["n01_1"] + d["n10_1"] + d["n11_1"]).to_numpy(dtype=int)

    p0s = np.vstack(
        [
            _counts_to_probs_4(int(r.n00_0), int(r.n01_0), int(r.n10_0), int(r.n11_0))
            for r in d.itertuples(index=False)
        ]
    )
    p1s = np.vstack(
        [
            _counts_to_probs_4(int(r.n00_1), int(r.n01_1), int(r.n10_1), int(r.n11_1))
            for r in d.itertuples(index=False)
        ]
    )

    rng = np.random.default_rng(int(seed))

    lam_stars: list[float] = []
    n_lam = len(lambdas)

    for _b in range(int(B)):
        CCs = np.empty(n_lam, dtype=float)
        for i in range(n_lam):
            c0 = rng.multinomial(int(n0[i]), p0s[i])
            c1 = rng.multinomial(int(n1[i]), p1s[i])
            mb = compute_metrics_from_counts(
                rule=rule,
                n00_0=int(c0[0]),
                n01_0=int(c0[1]),
                n10_0=int(c0[2]),
                n11_0=int(c0[3]),
                n00_1=int(c1[0]),
                n01_1=int(c1[1]),
                n10_1=int(c1[2]),
                n11_1=int(c1[3]),
                eps=eps,
            )
            CCs[i] = float(mb["CC_hat"])

        lam_star = _interp_root(lambdas, CCs, 1.0)
        if lam_star is not None and math.isfinite(lam_star):
            lam_stars.append(float(lam_star))

    arr = np.asarray(lam_stars, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size < 50:
        return {
            "lambda_star_boot_mean": float("nan"),
            "lambda_star_pct_lo": float("nan"),
            "lambda_star_pct_hi": float("nan"),
            "n_valid_boot": float(arr.size),
        }

    arr.sort()
    return {
        "lambda_star_boot_mean": float(arr.mean()),
        "lambda_star_pct_lo": float(np.quantile(arr, 0.025)),
        "lambda_star_pct_hi": float(np.quantile(arr, 0.975)),
        "n_valid_boot": float(arr.size),
    }


# =============================================================================
# CLI
# =============================================================================
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Compute BCa bootstrap CIs for correlation_cliff outputs."
    )
    ap.add_argument(
        "--sim_long",
        type=str,
        required=True,
        help="Path to sim_long.csv (from run_all.py / simulate_grid).",
    )
    ap.add_argument("--out", type=str, required=True, help="Output directory for BCa artifacts.")
    ap.add_argument(
        "--rep", type=int, default=0, help="Which replicate to treat as the observed curve for BCa."
    )
    ap.add_argument(
        "--rule",
        type=str,
        default=None,
        help="Override rule (OR/AND). If None, tries to read from CSV.",
    )
    ap.add_argument("--B", type=int, default=2000, help="Bootstrap replications per lambda.")
    ap.add_argument("--alpha", type=float, default=0.05, help="Two-sided alpha (0.05 => 95%% CI).")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed.")
    ap.add_argument(
        "--do_threshold_boot",
        action="store_true",
        help="Also run curve-level threshold bootstrap (percentile).",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    sim_long_path = Path(args.sim_long)
    out_dir = Path(args.out)
    _ensure_dir(out_dir)

    df_long = pd.read_csv(sim_long_path)

    # Determine rule
    rule: Rule
    if args.rule is not None:
        rule = args.rule.strip().upper()
        if rule not in ("OR", "AND"):
            raise ValueError("--rule must be OR or AND")
        rule = rule  # type: ignore[assignment]
    else:
        if "rule" not in df_long.columns:
            raise ValueError(
                "Could not infer rule from sim_long.csv (missing 'rule'). Provide --rule OR/AND."
            )
        rule_val = str(df_long["rule"].iloc[0]).strip().upper()
        if rule_val not in ("OR", "AND"):
            raise ValueError(f"Invalid rule in sim_long.csv: {rule_val}")
        rule = rule_val  # type: ignore[assignment]

    df_counts = select_observed_counts_from_sim_long(df_long, rep=int(args.rep))
    df_counts.to_csv(out_dir / "observed_counts.csv", index=False)

    df_bca = bca_table_for_curve(
        df_counts=df_counts,
        rule=rule,
        B=int(args.B),
        alpha=float(args.alpha),
        seed=int(args.seed),
    )
    df_bca.to_csv(out_dir / "bca_by_lambda.csv", index=False)

    meta = {
        "sim_long": str(sim_long_path),
        "rep": int(args.rep),
        "rule": rule,
        "B": int(args.B),
        "alpha": float(args.alpha),
        "seed": int(args.seed),
        "n_lambdas": int(df_bca.shape[0]),
    }
    _write_json(out_dir / "bca_meta.json", meta)

    if bool(args.do_threshold_boot):
        tb = threshold_bootstrap_percentile(
            df_counts=df_counts,
            rule=rule,
            B=int(args.B),
            seed=int(args.seed),
        )
        _write_json(out_dir / "threshold_bootstrap.json", tb)

    print(f"[analyze_bootstrap] wrote: {out_dir / 'bca_by_lambda.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

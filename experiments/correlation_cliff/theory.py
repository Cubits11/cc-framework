# experiments/correlation_cliff/theory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Tuple

import math
import numpy as np
import pandas as pd

Rule = Literal["OR", "AND"]


# =============================================================================
# PhD-level theory core
# =============================================================================
"""
Correlation Cliff Theory Module
===============================

This module defines the *exact* probability geometry and derived metrics for the
two-world, two-rail binary composition experiment.

The experiment:
- Two rails A,B produce binary indicators in each world w ∈ {0,1}.
- Marginals pA^w, pB^w are held fixed.
- Dependence (unknown copula risk) is swept by choosing the feasible joint mass
  p11^w within Fréchet–Hoeffding (FH) bounds:
    L^w = max(0, pA^w + pB^w - 1),  U^w = min(pA^w, pB^w)
- Along the FH-linear path:
    p11^w(λ) = L^w + λ (U^w - L^w),   λ ∈ [0,1]

Composition rules:
- OR:  C = A ∨ B  => pC^w = pA^w + pB^w - p11^w
- AND: C = A ∧ B  => pC^w = p11^w

Leakage metric:
- J_X = |pX^1 - pX^0| for X ∈ {A,B,C}
- J_best = max(J_A, J_B)
- CC_max(λ) = J_C(λ) / J_best

Key facts used throughout:
- Under fixed marginals, pC^w is affine in p11^w, hence Δ(λ)=pC^1-pC^0 is affine
  in (p11^1 - p11^0). This is why the OR/AND case admits closed form on FH-linear.

Uncertainty tie-in:
- Later modules compute delta-method approximations for var(CC) and BCa bootstrap
  CIs, as recommended in ResearchOS. :contentReference[oaicite:5]{index=5}
"""


@dataclass(frozen=True)
class WorldMarginals:
    pA: float
    pB: float


@dataclass(frozen=True)
class TwoWorldMarginals:
    w0: WorldMarginals
    w1: WorldMarginals


@dataclass(frozen=True)
class FHBounds:
    L: float
    U: float

    @property
    def width(self) -> float:
        return self.U - self.L


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def fh_bounds(pA: float, pB: float) -> FHBounds:
    """
    Fréchet–Hoeffding bounds for p11 given marginals pA, pB.
    """
    if not (0.0 <= pA <= 1.0 and 0.0 <= pB <= 1.0):
        raise ValueError(f"marginals must be in [0,1], got pA={pA}, pB={pB}")
    L = max(0.0, pA + pB - 1.0)
    U = min(pA, pB)
    if U < L - 1e-15:
        raise ValueError(f"Infeasible marginals: L={L} > U={U} for pA={pA}, pB={pB}")
    return FHBounds(L=L, U=U)


def p11_fh_linear(pA: float, pB: float, lam: float) -> float:
    """
    FH-linear joint mass p11(λ)=L+λ(U-L).
    """
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lambda must be in [0,1], got {lam}")
    b = fh_bounds(pA, pB)
    return b.L + lam * (b.U - b.L)


def joint_cells_from_marginals(pA: float, pB: float, p11: float) -> Dict[str, float]:
    """
    Construct the full 2x2 joint table:
        p00, p01, p10, p11
    where:
        p10 = P(A=1,B=0) = pA - p11
        p01 = P(A=0,B=1) = pB - p11
        p00 = 1 - pA - pB + p11
    """
    p10 = pA - p11
    p01 = pB - p11
    p00 = 1.0 - pA - pB + p11

    # Tiny negatives can happen due to floating error; clip very lightly.
    eps = 1e-12
    if (p00 < -eps) or (p01 < -eps) or (p10 < -eps) or (p11 < -eps):
        raise ValueError(
            f"Invalid joint probs from marginals: p00={p00}, p01={p01}, p10={p10}, p11={p11}"
        )

    p00 = _clip01(p00)
    p01 = _clip01(p01)
    p10 = _clip01(p10)
    p11 = _clip01(p11)

    s = p00 + p01 + p10 + p11
    if not (abs(s - 1.0) < 1e-9):
        # Renormalize if needed (should be extremely rare with valid inputs).
        p00, p01, p10, p11 = (p00 / s, p01 / s, p10 / s, p11 / s)

    return {"p00": p00, "p01": p01, "p10": p10, "p11": p11}


def pC_from_joint(rule: Rule, cells: Dict[str, float], pA: float, pB: float) -> float:
    """
    Compute pC given composition rule.
    """
    if rule == "OR":
        return pA + pB - cells["p11"]  # = 1 - p00
    if rule == "AND":
        return cells["p11"]
    raise ValueError(f"Unknown rule: {rule}")


def phi_from_joint(pA: float, pB: float, p11: float) -> float:
    """
    Phi coefficient (Pearson correlation for binary indicators) for a single world.
        φ = (p11 - pA pB) / sqrt(pA(1-pA)pB(1-pB))

    If denominator is 0 (degenerate marginals), returns NaN.
    """
    denom = pA * (1 - pA) * pB * (1 - pB)
    if denom <= 0.0:
        return float("nan")
    return (p11 - pA * pB) / math.sqrt(denom)


def kendall_tau_a_from_joint(cells: Dict[str, float]) -> float:
    """
    Kendall's tau-a for a 2x2 distribution.

    For binary (A,B), concordant pairs are (00,11), discordant are (01,10).
    If we pick two i.i.d. draws, unordered:
        P(concordant) = 2 p00 p11
        P(discordant) = 2 p01 p10
    tau-a = P(conc) - P(disc) = 2(p00 p11 - p01 p10)

    Note: tau-b would adjust for ties; tau-a is the clean population functional here.
    """
    p00, p01, p10, p11 = cells["p00"], cells["p01"], cells["p10"], cells["p11"]
    return 2.0 * (p00 * p11 - p01 * p10)


def compute_metrics_for_lambda(
    marg: TwoWorldMarginals,
    rule: Rule,
    lam: float,
) -> Dict[str, float]:
    """
    Compute analytic probabilities and derived metrics for a single lambda.
    """
    # World 0
    p11_0 = p11_fh_linear(marg.w0.pA, marg.w0.pB, lam)
    c0 = joint_cells_from_marginals(marg.w0.pA, marg.w0.pB, p11_0)
    pC_0 = pC_from_joint(rule, c0, marg.w0.pA, marg.w0.pB)

    # World 1
    p11_1 = p11_fh_linear(marg.w1.pA, marg.w1.pB, lam)
    c1 = joint_cells_from_marginals(marg.w1.pA, marg.w1.pB, p11_1)
    pC_1 = pC_from_joint(rule, c1, marg.w1.pA, marg.w1.pB)

    # Leakage gaps
    JA = abs(marg.w1.pA - marg.w0.pA)
    JB = abs(marg.w1.pB - marg.w0.pB)
    Jbest = max(JA, JB)

    dC = pC_1 - pC_0
    JC = abs(dC)
    CC = JC / Jbest if Jbest > 0 else float("nan")

    # Dependence summaries
    phi0 = phi_from_joint(marg.w0.pA, marg.w0.pB, c0["p11"])
    phi1 = phi_from_joint(marg.w1.pA, marg.w1.pB, c1["p11"])
    phi_avg = 0.5 * (phi0 + phi1)

    tau0 = kendall_tau_a_from_joint(c0)
    tau1 = kendall_tau_a_from_joint(c1)
    tau_avg = 0.5 * (tau0 + tau1)

    return {
        "lambda": lam,
        # World 0
        "pA_0": marg.w0.pA,
        "pB_0": marg.w0.pB,
        "p11_0": c0["p11"],
        "p00_0": c0["p00"],
        "p01_0": c0["p01"],
        "p10_0": c0["p10"],
        "pC_0": pC_0,
        "phi_0": phi0,
        "tau_0": tau0,
        # World 1
        "pA_1": marg.w1.pA,
        "pB_1": marg.w1.pB,
        "p11_1": c1["p11"],
        "p00_1": c1["p00"],
        "p01_1": c1["p01"],
        "p10_1": c1["p10"],
        "pC_1": pC_1,
        "phi_1": phi1,
        "tau_1": tau1,
        # Metrics
        "JA": JA,
        "JB": JB,
        "Jbest": Jbest,
        "dC": dC,
        "JC": JC,
        "CC": CC,
        "phi_avg": phi_avg,
        "tau_avg": tau_avg,
    }


def theory_curve_fh_linear(
    marg: TwoWorldMarginals,
    rule: Rule,
    lambdas: Iterable[float],
) -> pd.DataFrame:
    rows = [compute_metrics_for_lambda(marg, rule, float(lam)) for lam in lambdas]
    df = pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)
    return df


def _interp_root(x: np.ndarray, y: np.ndarray, target: float) -> Optional[float]:
    """
    Find root for y(x)=target by linear interpolation on a monotone-ish grid.
    Returns None if bracket doesn't exist.
    """
    if len(x) != len(y) or len(x) < 2:
        return None

    # Find first segment that brackets target.
    for i in range(len(x) - 1):
        y0, y1 = y[i], y[i + 1]
        if (y0 - target) == 0:
            return float(x[i])
        if (y0 - target) * (y1 - target) <= 0:
            # linear interpolation; guard division by zero
            if y1 == y0:
                return float(x[i])
            t = (target - y0) / (y1 - y0)
            return float(x[i] + t * (x[i + 1] - x[i]))
    return None


def find_lambda_star_for_cc_eq_1(df: pd.DataFrame, target_cc: float = 1.0) -> Optional[float]:
    """
    Solve for lambda* such that CC(lambda*) = target_cc by interpolation over an existing curve df.
    """
    x = df["lambda"].to_numpy(dtype=float)
    y = df["CC"].to_numpy(dtype=float)
    return _interp_root(x, y, target_cc)


def fh_pC_interval(pA: float, pB: float, rule: Rule) -> Tuple[float, float]:
    """
    Induce feasible interval for pC^w from FH bounds, per world. :contentReference[oaicite:6]{index=6}
      - OR: pC ∈ [pA+pB-U, pA+pB-L]
      - AND: pC ∈ [L, U]
    """
    b = fh_bounds(pA, pB)
    if rule == "OR":
        lo = pA + pB - b.U
        hi = pA + pB - b.L
    elif rule == "AND":
        lo, hi = b.L, b.U
    else:
        raise ValueError(f"Unknown rule: {rule}")
    return (lo, hi)


def jc_envelope_from_intervals(int0: Tuple[float, float], int1: Tuple[float, floa]()_

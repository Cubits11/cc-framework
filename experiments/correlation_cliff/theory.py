# experiments/correlation_cliff/theory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Tuple, List, Sequence, TypedDict, Union, cast

import math
import numpy as np
import pandas as pd

Rule = Literal["OR", "AND"]

# -----------------------------
# Optional SciPy (Gaussian path)
# -----------------------------
try:
    # SciPy is optional. If missing, Gaussian-copula p11 falls back to Monte Carlo.
    from scipy.stats import norm  # type: ignore
    from scipy.stats import multivariate_normal  # type: ignore

    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


"""
Correlation Cliff Theory Module
===============================

This module is the analytic “bedrock” for the Correlation Cliff experiment.

It is intentionally:
- **mathematically explicit** (definitions, lemmas, proof sketches in docstrings),
- **numerically defensive** (validation & envelopes to catch silent bugs),
- **experiment-oriented** (returns plot-ready DataFrames),
- **extensible** (dependence paths are pluggable; FH-linear remains the canonical baseline).

Core experiment
--------------
We consider two “worlds” w ∈ {0,1} (e.g., protected vs unprotected) and two rails
A,B ∈ {0,1} producing binary triggers. We fix world-wise marginals:

    pA^w = P(A=1 | w)
    pB^w = P(B=1 | w)

The unknown risk is the *joint* dependence between A and B inside each world.
Let p11^w = P(A=1, B=1 | w). By probability axioms (Fréchet–Hoeffding bounds):

    L^w = max(0, pA^w + pB^w - 1)
    U^w = min(pA^w, pB^w)
    L^w ≤ p11^w ≤ U^w

Composition rules
----------------
Define composed output C = f(A,B) where:

- OR:  C = A ∨ B  ⇒  pC^w = P(C=1|w) = pA^w + pB^w − p11^w = 1 − p00^w
- AND: C = A ∧ B  ⇒  pC^w = p11^w

Leakage & composability metric
------------------------------
We use a two-world distinguishability gap (Youden-like difference in rates):

    J_X := |pX^1 − pX^0|   for X ∈ {A,B,C}

Normalize by the best singleton rail:

    J_best := max(J_A, J_B)
    CC_max := J_C / J_best

Interpretation:
- CC_max > 1 ⇒ composition is *destructive* relative to the best single rail
- CC_max < 1 ⇒ composition is *constructive* (reduces leakage vs best singleton)

Dependence parameterization
---------------------------
The canonical baseline path is FH-linear, per world:

    p11^w(λ) = L^w + λ (U^w − L^w),   λ ∈ [0,1]

This is the convex interpolation between FH extremes; it is “singular” in the
copula sense, but extremely useful as a *model-free scanner* of feasible joints.

Key lemma (Affine composition)
------------------------------
Fix marginals pA^w, pB^w. Then pC^w is affine in p11^w for OR/AND.
Therefore for any per-world paths p11^w(t), the two-world gap:

    Δ_C(t) := pC^1(t) − pC^0(t)

is affine in δ11(t) := p11^1(t) − p11^0(t).

Under FH-linear, Δ_C(λ) is affine in λ; hence J_C(λ)=|Δ_C(λ)| is piecewise affine.
This is why closed-form λ* exists under a “no sign-flip” condition.

Dependence summaries (interpretability)
---------------------------------------
We provide two dependence summaries per world, and their simple average:

- Phi coefficient (binary Pearson correlation):
      φ^w = (p11^w − pA^w pB^w) / sqrt(pA^w(1−pA^w)pB^w(1−pB^w))

- Kendall tau-a (population concordance functional for 2×2):
      τ_a^w = 2(p00^w p11^w − p01^w p10^w)

Proof sketch for tau-a (binary):
Let (A1,B1),(A2,B2) be i.i.d. draws. Kendall tau-a is:
    τ_a = E[ sgn((A1−A2)(B1−B2)) ].
For binary, only pairs with A1≠A2 and B1≠B2 contribute ±1; those correspond to
concordant (00 vs 11) and discordant (01 vs 10) cases. Computing the probability
mass of those cases yields τ_a = 2(p00 p11 − p01 p10).

Sanity envelopes (bug catchers)
-------------------------------
Given fixed marginals in each world, FH bounds induce a feasible interval for pC^w,
and therefore a feasible envelope for J_C = |pC^1 − pC^0| even when dependence is unknown.
If empirical estimates violate these envelopes, either:
- the code is wrong,
- the inputs are inconsistent,
- or the experiment violated the “fixed marginals + FH feasible joint” assumptions.

What belongs here vs elsewhere
------------------------------
- This module: exact analytic quantities, envelopes, closed forms, mappings.
- simulate.py: sampling, RNG, empirical estimates.
- analyze_bootstrap.py: bootstrap / BCa, delta-method SEs for empirical curves, hypothesis testing.
- figures.py: plotting.

"""


# =========================
# Typed outputs (optional)
# =========================

class MetricRow(TypedDict, total=False):
    lambda_: float

    # world 0
    pA_0: float
    pB_0: float
    p00_0: float
    p01_0: float
    p10_0: float
    p11_0: float
    pC_0: float
    phi_0: float
    tau_0: float

    # world 1
    pA_1: float
    pB_1: float
    p00_1: float
    p01_1: float
    p10_1: float
    p11_1: float
    pC_1: float
    phi_1: float
    tau_1: float

    # metrics
    JA: float
    JB: float
    Jbest: float
    dC: float
    JC: float
    CC: float
    phi_avg: float
    tau_avg: float

    # diagnostics
    fh_L_0: float
    fh_U_0: float
    fh_L_1: float
    fh_U_1: float
    path_name: str


# =========================
# Data classes & validation
# =========================

@dataclass(frozen=True)
class WorldMarginals:
    pA: float
    pB: float

    def validate(self) -> None:
        if not (0.0 <= self.pA <= 1.0) or not (0.0 <= self.pB <= 1.0):
            raise ValueError(f"marginals must be in [0,1], got pA={self.pA}, pB={self.pB}")


@dataclass(frozen=True)
class TwoWorldMarginals:
    w0: WorldMarginals
    w1: WorldMarginals

    def validate(self) -> None:
        self.w0.validate()
        self.w1.validate()


@dataclass(frozen=True)
class FHBounds:
    L: float
    U: float

    @property
    def width(self) -> float:
        return self.U - self.L

    def validate(self) -> None:
        if not (0.0 <= self.L <= 1.0) or not (0.0 <= self.U <= 1.0):
            raise ValueError(f"FH bounds must be in [0,1], got L={self.L}, U={self.U}")
        if self.U < self.L - 1e-15:
            raise ValueError(f"Infeasible FH: L={self.L} > U={self.U}")


# =========================
# Numerical helpers
# =========================

def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _almost_equal(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(a - b) <= tol


def _safe_div(num: float, den: float, eps: float = 0.0) -> float:
    """
    Safe division with optional denom floor. If den <= eps, return NaN.
    """
    if den <= eps:
        return float("nan")
    return num / den


# =========================
# FH bounds
# =========================

def fh_bounds(pA: float, pB: float) -> FHBounds:
    """
    Fréchet–Hoeffding bounds for p11 given Bernoulli marginals pA, pB.

    Derivation sketch:
      - Upper bound: p11 <= P(A=1)=pA and p11 <= P(B=1)=pB ⇒ p11 <= min(pA,pB)
      - Lower bound: by inclusion-exclusion,
            P(A∪B) = pA + pB − p11 <= 1  ⇒  p11 >= pA + pB − 1
        and of course p11 >= 0 ⇒ p11 >= max(0, pA+pB−1)
    """
    if not (0.0 <= pA <= 1.0 and 0.0 <= pB <= 1.0):
        raise ValueError(f"marginals must be in [0,1], got pA={pA}, pB={pB}")
    L = max(0.0, pA + pB - 1.0)
    U = min(pA, pB)
    b = FHBounds(L=L, U=U)
    b.validate()
    return b


# =========================
# Dependence paths (p11^w(t))
# =========================

DependencePath = Literal["fh_linear", "fh_power", "fh_scurve", "gaussian_copula"]


def p11_fh_linear(pA: float, pB: float, lam: float) -> float:
    """
    FH-linear p11(λ) = L + λ(U−L), λ∈[0,1].
    """
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lambda must be in [0,1], got {lam}")
    b = fh_bounds(pA, pB)
    return b.L + lam * b.width


def p11_fh_power(pA: float, pB: float, lam: float, k: float = 2.0) -> float:
    """
    Curved but still FH-feasible: p11(λ)=L + λ^k (U−L), λ∈[0,1], k>0.

    Interpretation:
      - k>1: spends more time near the lower bound (slow start, fast finish)
      - k<1: spends more time near the upper bound (fast start, slow finish)

    This is NOT a different copula family; it's a different path through the feasible p11 interval.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    return p11_fh_linear(pA, pB, lam ** k)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def p11_fh_scurve(pA: float, pB: float, lam: float, alpha: float = 6.0) -> float:
    """
    S-curve path through [L,U]:
      s(λ) = sigmoid(alpha*(λ-0.5))
      rescaled to [0,1]: s01 = (s - s(0)) / (s(1) - s(0))
      p11 = L + s01*(U-L)

    This yields a “sharper” transition around λ=0.5 for larger alpha.
    """
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lambda must be in [0,1], got {lam}")
    b = fh_bounds(pA, pB)
    s0 = _sigmoid(alpha * (0.0 - 0.5))
    s1 = _sigmoid(alpha * (1.0 - 0.5))
    s = _sigmoid(alpha * (lam - 0.5))
    s01 = (s - s0) / (s1 - s0) if (s1 != s0) else lam
    return b.L + float(s01) * b.width


def p11_gaussian_copula(
    pA: float,
    pB: float,
    rho: float,
    *,
    method: Literal["auto", "scipy", "mc"] = "auto",
    n_mc: int = 200_000,
    seed: Optional[int] = 12345,
) -> float:
    """
    Gaussian copula-based p11 (Bernoulli margins via latent thresholding).

    Construction:
      Let (Z1,Z2) ~ N(0, Σ) with corr(Z1,Z2)=rho.
      Define A=1{Z1 <= tA}, B=1{Z2 <= tB} where:
          tA = Φ^{-1}(pA), tB = Φ^{-1}(pB)
      Then:
          p11 = P(A=1,B=1) = P(Z1<=tA, Z2<=tB) = Φ2(tA,tB; rho)

    Important nuance:
      With discrete margins, the Gaussian-copula family cannot generally achieve
      the FH extremes exactly. So this path is “realistic smooth dependence,” not
      a full scanner of all feasible joints. It is still valuable as a sensitivity check.

    If SciPy is available:
      - use multivariate_normal.cdf for Φ2
    Otherwise:
      - fall back to Monte Carlo (deterministic seed by default)
    """
    if not (-1.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [-1,1], got {rho}")
    if not (0.0 <= pA <= 1.0 and 0.0 <= pB <= 1.0):
        raise ValueError(f"marginals must be in [0,1], got pA={pA}, pB={pB}")

    # Degenerate margins: if pA or pB is 0/1, p11 collapses.
    if pA in (0.0, 1.0) or pB in (0.0, 1.0):
        return float(pA * pB)

    if method == "auto":
        method = "scipy" if _HAVE_SCIPY else "mc"

    if method == "scipy":
        if not _HAVE_SCIPY:
            raise RuntimeError("SciPy not available; cannot use method='scipy' for gaussian_copula.")
        tA = float(norm.ppf(pA))
        tB = float(norm.ppf(pB))
        cov = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
        # multivariate_normal.cdf expects x as a vector
        val = float(multivariate_normal.cdf([tA, tB], mean=[0.0, 0.0], cov=cov))
        return _clip01(val)

    if method == "mc":
        if n_mc <= 10_000:
            # keep MC reasonably stable; user can override
            n_mc = 10_000
        rng = np.random.default_rng(seed)
        cov = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
        Z = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n_mc)
        # A=1{Z1<=tA}, B=1{Z2<=tB}
        if _HAVE_SCIPY:
            tA = float(norm.ppf(pA))
            tB = float(norm.ppf(pB))
        else:
            # If SciPy missing, approximate inverse CDF via erfinv
            # Φ^{-1}(p)=sqrt(2)*erfinv(2p-1)
            tA = math.sqrt(2.0) * math.erf(1.0)  # placeholder to satisfy mypy
            # Proper approximation:
            # Python doesn't expose erfinv in math, but numpy does.
            tA = float(np.sqrt(2.0) * np.erfinv(2.0 * pA - 1.0))  # type: ignore[attr-defined]
            tB = float(np.sqrt(2.0) * np.erfinv(2.0 * pB - 1.0))  # type: ignore[attr-defined]
        val = float(np.mean((Z[:, 0] <= tA) & (Z[:, 1] <= tB)))
        return _clip01(val)

    raise ValueError(f"Unknown method: {method}")


def p11_path(
    pA: float,
    pB: float,
    lam: float,
    *,
    path: DependencePath = "fh_linear",
    path_params: Optional[Dict[str, float]] = None,
) -> float:
    """
    Unified p11 path interface.

    Parameters
    ----------
    pA, pB : marginals in [0,1]
    lam    : sweep parameter in [0,1]
    path   : one of:
      - 'fh_linear'      : L + λ(U-L)
      - 'fh_power'       : L + λ^k (U-L)
      - 'fh_scurve'      : L + s(λ)(U-L) with logistic s
      - 'gaussian_copula': Bernoulli-from-Gaussian copula with rho = 2λ-1 (by default)

    path_params:
      - for fh_power:  {'k': ...}
      - for fh_scurve: {'alpha': ...}
      - for gaussian_copula:
           {'rho_map': 0/1?} not used here; you can implement custom mapping externally
           {'n_mc': ...} for MC fallback, {'seed': ...}
    """
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lambda must be in [0,1], got {lam}")
    path_params = path_params or {}

    if path == "fh_linear":
        return p11_fh_linear(pA, pB, lam)

    if path == "fh_power":
        k = float(path_params.get("k", 2.0))
        return p11_fh_power(pA, pB, lam, k=k)

    if path == "fh_scurve":
        alpha = float(path_params.get("alpha", 6.0))
        return p11_fh_scurve(pA, pB, lam, alpha=alpha)

    if path == "gaussian_copula":
        # Map lam ∈ [0,1] to rho ∈ [-1,1]. Default: rho = 2λ − 1
        rho = float(path_params.get("rho", 2.0 * lam - 1.0))
        method = cast(Literal["auto", "scipy", "mc"], path_params.get("method", "auto"))
        n_mc = int(path_params.get("n_mc", 200_000))
        seed = int(path_params.get("seed", 12345))
        val = p11_gaussian_copula(pA, pB, rho, method=method, n_mc=n_mc, seed=seed)

        # Ensure feasibility (Gaussian can drift numerically; clip to FH as a guardrail)
        b = fh_bounds(pA, pB)
        if val < b.L - 1e-6 or val > b.U + 1e-6:
            # If this happens materially, something is off (numeric, method, or mapping).
            # We clip to keep downstream stable, but you should surface this in analysis.
            val = float(min(b.U, max(b.L, val)))
        return val

    raise ValueError(f"Unknown dependence path: {path}")


# =========================
# Joint table construction & checks
# =========================

def joint_cells_from_marginals(pA: float, pB: float, p11: float) -> Dict[str, float]:
    """
    Construct full 2×2 joint table given marginals and overlap p11.

        p10 = pA - p11
        p01 = pB - p11
        p00 = 1 - pA - pB + p11

    We lightly clip tiny negatives due to float error and renormalize.

    Raises if probs are materially invalid (negative beyond eps).
    """
    p10 = pA - p11
    p01 = pB - p11
    p00 = 1.0 - pA - pB + p11

    eps = 1e-12
    if (p00 < -eps) or (p01 < -eps) or (p10 < -eps) or (p11 < -eps):
        raise ValueError(
            f"Invalid joint probs: p00={p00}, p01={p01}, p10={p10}, p11={p11} "
            f"(pA={pA}, pB={pB})"
        )

    p00 = _clip01(p00)
    p01 = _clip01(p01)
    p10 = _clip01(p10)
    p11 = _clip01(p11)

    s = p00 + p01 + p10 + p11
    if not _almost_equal(s, 1.0, tol=1e-9):
        p00, p01, p10, p11 = (p00 / s, p01 / s, p10 / s, p11 / s)

    return {"p00": p00, "p01": p01, "p10": p10, "p11": p11}


def validate_joint_cells(cells: Dict[str, float], tol: float = 1e-10) -> None:
    """
    Strict validation for a joint table.
    """
    for k in ("p00", "p01", "p10", "p11"):
        if k not in cells:
            raise ValueError(f"Missing cell {k} in {cells}")
        v = float(cells[k])
        if v < -tol or v > 1.0 + tol:
            raise ValueError(f"Cell {k} out of [0,1]: {v}")

    s = float(cells["p00"] + cells["p01"] + cells["p10"] + cells["p11"])
    if abs(s - 1.0) > 1e-8:
        raise ValueError(f"Cells do not sum to 1: sum={s}, cells={cells}")


# =========================
# Composition probabilities
# =========================

def pC_from_joint(rule: Rule, cells: Dict[str, float], pA: float, pB: float) -> float:
    """
    Compute pC given composition rule.
    """
    if rule == "OR":
        return _clip01(pA + pB - cells["p11"])
    if rule == "AND":
        return _clip01(cells["p11"])
    raise ValueError(f"Unknown rule: {rule}")


# =========================
# Dependence summaries
# =========================

def phi_from_joint(pA: float, pB: float, p11: float) -> float:
    """
    Phi coefficient (binary Pearson correlation), single world:

        φ = (p11 - pA pB) / sqrt(pA(1-pA)pB(1-pB))

    Proof sketch:
      For Bernoulli A,B:
        E[A]=pA, E[B]=pB, E[AB]=p11
        cov(A,B)=E[AB]-E[A]E[B]=p11 - pA pB
        var(A)=pA(1-pA), var(B)=pB(1-pB)
        corr(A,B)=cov/sqrt(varA varB) = φ

    Returns NaN if denom is 0 (degenerate marginals).
    """
    denom = pA * (1.0 - pA) * pB * (1.0 - pB)
    if denom <= 0.0:
        return float("nan")
    return (p11 - pA * pB) / math.sqrt(denom)


def kendall_tau_a_from_joint(cells: Dict[str, float]) -> float:
    """
    Kendall tau-a for a 2×2 distribution (population functional):

        τ_a = 2(p00 p11 - p01 p10)

    Proof sketch:
      Define τ_a = E[sgn((A1-A2)(B1-B2))] for i.i.d. draws (A1,B1),(A2,B2).
      Only comparisons with A1≠A2 and B1≠B2 contribute ±1.
      Concordant: (00,11) or (11,00) → probability 2 p00 p11
      Discordant: (01,10) or (10,01) → probability 2 p01 p10
      Thus τ_a = 2 p00 p11 − 2 p01 p10.
    """
    p00, p01, p10, p11 = cells["p00"], cells["p01"], cells["p10"], cells["p11"]
    return 2.0 * (p00 * p11 - p01 * p10)


def avg_ignore_nan(x: float, y: float) -> float:
    if math.isnan(x) and math.isnan(y):
        return float("nan")
    if math.isnan(x):
        return y
    if math.isnan(y):
        return x
    return 0.5 * (x + y)


# =========================
# Leakage metrics & CC
# =========================

def leakage_J(p1: float, p0: float) -> float:
    return abs(p1 - p0)


def compute_singleton_Js(marg: TwoWorldMarginals) -> Tuple[float, float, float]:
    """
    Returns (J_A, J_B, J_best).
    """
    marg.validate()
    JA = leakage_J(marg.w1.pA, marg.w0.pA)
    JB = leakage_J(marg.w1.pB, marg.w0.pB)
    return JA, JB, max(JA, JB)


# =========================
# Core metrics (single lambda)
# =========================

def compute_metrics_for_lambda(
    marg: TwoWorldMarginals,
    rule: Rule,
    lam: float,
    *,
    path: DependencePath = "fh_linear",
    path_params: Optional[Dict[str, float]] = None,
) -> MetricRow:
    """
    Compute analytic probabilities and derived metrics at a single lambda.

    Returns a flat dict (MetricRow) suitable for DataFrame construction.
    """
    marg.validate()
    path_params = path_params or {}

    # FH bounds for diagnostics
    b0 = fh_bounds(marg.w0.pA, marg.w0.pB)
    b1 = fh_bounds(marg.w1.pA, marg.w1.pB)

    # World 0
    p11_0 = p11_path(marg.w0.pA, marg.w0.pB, lam, path=path, path_params=path_params)
    c0 = joint_cells_from_marginals(marg.w0.pA, marg.w0.pB, p11_0)
    validate_joint_cells(c0)
    pC_0 = pC_from_joint(rule, c0, marg.w0.pA, marg.w0.pB)

    # World 1
    p11_1 = p11_path(marg.w1.pA, marg.w1.pB, lam, path=path, path_params=path_params)
    c1 = joint_cells_from_marginals(marg.w1.pA, marg.w1.pB, p11_1)
    validate_joint_cells(c1)
    pC_1 = pC_from_joint(rule, c1, marg.w1.pA, marg.w1.pB)

    # Leakage gaps
    JA = leakage_J(marg.w1.pA, marg.w0.pA)
    JB = leakage_J(marg.w1.pB, marg.w0.pB)
    Jbest = max(JA, JB)

    dC = pC_1 - pC_0
    JC = abs(dC)
    CC = (JC / Jbest) if (Jbest > 0.0) else float("nan")

    # Dependence summaries
    phi0 = phi_from_joint(marg.w0.pA, marg.w0.pB, c0["p11"])
    phi1 = phi_from_joint(marg.w1.pA, marg.w1.pB, c1["p11"])
    phi_avg = avg_ignore_nan(phi0, phi1)

    tau0 = kendall_tau_a_from_joint(c0)
    tau1 = kendall_tau_a_from_joint(c1)
    tau_avg = 0.5 * (tau0 + tau1)

    row: MetricRow = {
        "lambda_": float(lam),
        "path_name": str(path),

        # bounds (diagnostics)
        "fh_L_0": float(b0.L),
        "fh_U_0": float(b0.U),
        "fh_L_1": float(b1.L),
        "fh_U_1": float(b1.U),

        # world 0
        "pA_0": float(marg.w0.pA),
        "pB_0": float(marg.w0.pB),
        "p00_0": float(c0["p00"]),
        "p01_0": float(c0["p01"]),
        "p10_0": float(c0["p10"]),
        "p11_0": float(c0["p11"]),
        "pC_0": float(pC_0),
        "phi_0": float(phi0),
        "tau_0": float(tau0),

        # world 1
        "pA_1": float(marg.w1.pA),
        "pB_1": float(marg.w1.pB),
        "p00_1": float(c1["p00"]),
        "p01_1": float(c1["p01"]),
        "p10_1": float(c1["p10"]),
        "p11_1": float(c1["p11"]),
        "pC_1": float(pC_1),
        "phi_1": float(phi1),
        "tau_1": float(tau1),

        # metrics
        "JA": float(JA),
        "JB": float(JB),
        "Jbest": float(Jbest),
        "dC": float(dC),
        "JC": float(JC),
        "CC": float(CC),
        "phi_avg": float(phi_avg),
        "tau_avg": float(tau_avg),
    }
    return row


# =========================
# Curve generation (vectorized where possible)
# =========================

def theory_curve(
    marg: TwoWorldMarginals,
    rule: Rule,
    lambdas: Iterable[float],
    *,
    path: DependencePath = "fh_linear",
    path_params: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Compute theory curve across lambdas.

    Implementation:
      - For general paths: simple loop (still fast for ~21-200 points)
      - For fh_linear: loop is already trivial, but you can vectorize later if you want.

    Returns DataFrame sorted by lambda_.
    """
    path_params = path_params or {}
    rows = [compute_metrics_for_lambda(marg, rule, float(lam), path=path, path_params=path_params) for lam in lambdas]
    df = pd.DataFrame(rows).sort_values("lambda_").reset_index(drop=True)
    return df


# =========================
# Affine structure (FH-linear OR/AND) + closed form λ*
# =========================

def _affine_form_or_and_one_world(pA: float, pB: float, rule: Rule) -> Tuple[float, float]:
    """
    For FH-linear, p11(λ)=L+λW with W=(U-L).

    OR:
      pC(λ) = pA+pB - p11(λ) = (pA+pB-L) - W λ  => a - bλ
    AND:
      pC(λ) = p11(λ)         = L + W λ          => a + bλ

    Return (a, signed_b) such that pC(λ)=a + signed_b * λ.
    """
    bnd = fh_bounds(pA, pB)
    L, W = bnd.L, bnd.width
    if rule == "OR":
        a = pA + pB - L
        signed_b = -W
        return float(a), float(signed_b)
    if rule == "AND":
        a = L
        signed_b = +W
        return float(a), float(signed_b)
    raise ValueError(f"Unknown rule: {rule}")


def deltaC_affine_params_fh_linear(marg: TwoWorldMarginals, rule: Rule) -> Dict[str, float]:
    """
    Under FH-linear:
        pC^w(λ) = a_w + b_w λ
        Δ(λ) = (a_1-a_0) + (b_1-b_0) λ = delta0 + slope λ

    Returns:
      a0,a1,b0,b1, delta0, slope
    """
    marg.validate()
    a0, b0 = _affine_form_or_and_one_world(marg.w0.pA, marg.w0.pB, rule)
    a1, b1 = _affine_form_or_and_one_world(marg.w1.pA, marg.w1.pB, rule)
    return {
        "a0": float(a0),
        "a1": float(a1),
        "b0": float(b0),
        "b1": float(b1),
        "delta0": float(a1 - a0),
        "slope": float((b1 - b0)),
    }


def detect_abs_kink_fh_linear(marg: TwoWorldMarginals, rule: Rule) -> Optional[float]:
    """
    Detect whether Δ(λ)=pC^1-pC^0 crosses 0 on λ∈[0,1] under FH-linear.

    If Δ changes sign, J_C(λ)=|Δ(λ)| has a kink at λ_kink where Δ(λ_kink)=0.

    Returns λ_kink if it exists in [0,1], else None.
    """
    pars = deltaC_affine_params_fh_linear(marg, rule)
    delta0 = pars["delta0"]
    slope = pars["slope"]

    # Δ(λ) = delta0 + slope*λ
    if abs(slope) < 1e-15:
        return 0.0 if abs(delta0) < 1e-12 else None

    lam0 = 0.0
    lam1 = 1.0
    d0 = delta0 + slope * lam0
    d1 = delta0 + slope * lam1

    if d0 == 0.0:
        return 0.0
    if d0 * d1 < 0.0:
        # Root exists
        lam = -delta0 / slope
        if 0.0 <= lam <= 1.0:
            return float(lam)
    return None


def lambda_star_closed_form_fh_linear(
    marg: TwoWorldMarginals,
    rule: Rule,
    target_cc: float = 1.0,
    *,
    require_no_kink: bool = True,
) -> Optional[float]:
    """
    Closed-form λ* for FH-linear when solving CC(λ*)=target_cc.

    We solve:
        |Δ(λ)| = target_cc * J_best
    where
        Δ(λ) = delta0 + slope λ  (affine)
        J_best is constant for fixed marginals.

    There are up to two solutions:
        delta0 + slope λ = +rhs
        delta0 + slope λ = -rhs
    We return the feasible one in [0,1] consistent with “no kink” if required.

    Notes:
    - If Δ crosses 0 (kink), there can be multiple crossings of CC=target.
      In that case, prefer grid root-finding in analysis for clarity.
    """
    marg.validate()
    JA, JB, Jbest = compute_singleton_Js(marg)
    if Jbest <= 0.0:
        return None

    if require_no_kink and (detect_abs_kink_fh_linear(marg, rule) is not None):
        return None

    pars = deltaC_affine_params_fh_linear(marg, rule)
    delta0 = pars["delta0"]
    slope = pars["slope"]
    rhs = float(target_cc) * float(Jbest)

    if abs(slope) < 1e-15:
        # Δ is constant
        if abs(abs(delta0) - rhs) < 1e-12:
            return 0.0
        return None

    candidates = [
        (rhs - delta0) / slope,
        (-rhs - delta0) / slope,
    ]
    feas = [float(l) for l in candidates if -1e-12 <= l <= 1.0 + 1e-12]
    if not feas:
        return None

    feas = [min(1.0, max(0.0, l)) for l in feas]

    # If multiple, pick the one whose Δ sign matches Δ(0) (typical “no flip” behavior).
    d_at_0 = delta0
    sgn0 = 1.0 if d_at_0 >= 0.0 else -1.0

    def sgn(x: float) -> float:
        return 1.0 if x >= 0.0 else -1.0

    filtered = []
    for l in feas:
        d = delta0 + slope * l
        if d == 0.0:
            continue
        if sgn(d) == sgn0:
            filtered.append(l)

    if filtered:
        return float(min(filtered, key=lambda x: abs(x - 0.5)))
    return float(min(feas, key=lambda x: abs(x - 0.5)))


# =========================
# Root finding on a grid
# =========================

def find_all_roots_linear_interp(
    x: np.ndarray,
    y: np.ndarray,
    target: float,
) -> List[float]:
    """
    Find all roots of y(x)=target using bracketing + linear interpolation on a grid.
    """
    roots: List[float] = []
    if len(x) != len(y) or len(x) < 2:
        return roots

    for i in range(len(x) - 1):
        y0 = float(y[i] - target)
        y1 = float(y[i + 1] - target)

        if y0 == 0.0:
            roots.append(float(x[i]))
            continue

        if y0 * y1 < 0.0:
            # interpolate crossing
            denom = (y[i + 1] - y[i])
            if denom == 0.0:
                roots.append(float(x[i]))
            else:
                t = float((target - y[i]) / denom)
                roots.append(float(x[i] + t * (x[i + 1] - x[i])))

    roots.sort()
    # dedupe
    out: List[float] = []
    for r in roots:
        if not out or abs(r - out[-1]) > 1e-10:
            out.append(r)
    return out


def find_lambda_star_from_curve(
    df: pd.DataFrame,
    *,
    target_cc: float = 1.0,
    prefer_first: bool = True,
) -> Optional[float]:
    """
    Find λ* such that CC(λ*)=target_cc from a curve DataFrame.

    Expected columns:
      - "lambda_" and "CC" OR "CC_hat"

    If prefer_first=True, return first root in increasing λ.
    Otherwise return root closest to 0.5 (often visually stable).
    """
    if "lambda_" not in df.columns:
        raise ValueError("df must contain 'lambda_'")

    if "CC" in df.columns:
        ycol = "CC"
    elif "CC_hat" in df.columns:
        ycol = "CC_hat"
    else:
        raise ValueError("df must contain 'CC' or 'CC_hat'")

    x = df["lambda_"].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    roots = find_all_roots_linear_interp(x, y, float(target_cc))
    if not roots:
        return None
    if prefer_first:
        return float(roots[0])
    return float(min(roots, key=lambda r: abs(r - 0.5)))


# =========================
# FH envelopes (sanity constraints)
# =========================

def fh_pC_interval(pA: float, pB: float, rule: Rule) -> Tuple[float, float]:
    """
    FH-induced feasible interval for pC in a single world.

    - OR: pC = pA+pB-p11, p11∈[L,U] ⇒ pC∈[pA+pB-U, pA+pB-L]
    - AND: pC = p11, p11∈[L,U] ⇒ pC∈[L,U]
    """
    b = fh_bounds(pA, pB)
    if rule == "OR":
        lo = pA + pB - b.U
        hi = pA + pB - b.L
    elif rule == "AND":
        lo, hi = b.L, b.U
    else:
        raise ValueError(f"Unknown rule: {rule}")
    return (_clip01(lo), _clip01(hi))


def jc_envelope_from_intervals(int0: Tuple[float, float], int1: Tuple[float, float]) -> Tuple[float, float]:
    """
    From I0=[a0,b0], I1=[a1,b1], derive feasible envelope for JC=|x-y| with x∈I1,y∈I0.

    - J_min is 0 if intervals overlap, else the distance between them.
    - J_max is the maximum endpoint distance.
    """
    a0, b0 = float(int0[0]), float(int0[1])
    a1, b1 = float(int1[0]), float(int1[1])
    if b0 < a0:
        a0, b0 = b0, a0
    if b1 < a1:
        a1, b1 = b1, a1

    overlap = (b0 >= a1) and (b1 >= a0)
    if overlap:
        jmin = 0.0
    else:
        jmin = min(abs(a1 - b0), abs(a0 - b1))

    jmax = max(abs(a1 - b0), abs(b1 - a0), abs(a1 - a0), abs(b1 - b0))
    return (float(_clip01(jmin)), float(_clip01(jmax)))


def compute_fh_jc_envelope(marg: TwoWorldMarginals, rule: Rule) -> Tuple[float, float]:
    """
    Absolute sanity envelope for JC implied only by marginals + FH feasibility.
    """
    marg.validate()
    int0 = fh_pC_interval(marg.w0.pA, marg.w0.pB, rule)
    int1 = fh_pC_interval(marg.w1.pA, marg.w1.pB, rule)
    return jc_envelope_from_intervals(int0, int1)


# =========================
# Dependence mapping at λ*
# =========================

def dependence_at_lambda(
    marg: TwoWorldMarginals,
    lam: float,
    *,
    path: DependencePath = "fh_linear",
    path_params: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute (phi_0,phi_1,phi_avg,tau_0,tau_1,tau_avg) at a given lambda and path.
    """
    marg.validate()
    path_params = path_params or {}

    p11_0 = p11_path(marg.w0.pA, marg.w0.pB, lam, path=path, path_params=path_params)
    c0 = joint_cells_from_marginals(marg.w0.pA, marg.w0.pB, p11_0)
    validate_joint_cells(c0)

    p11_1 = p11_path(marg.w1.pA, marg.w1.pB, lam, path=path, path_params=path_params)
    c1 = joint_cells_from_marginals(marg.w1.pA, marg.w1.pB, p11_1)
    validate_joint_cells(c1)

    phi0 = phi_from_joint(marg.w0.pA, marg.w0.pB, c0["p11"])
    phi1 = phi_from_joint(marg.w1.pA, marg.w1.pB, c1["p11"])
    phi_avg = avg_ignore_nan(phi0, phi1)

    tau0 = kendall_tau_a_from_joint(c0)
    tau1 = kendall_tau_a_from_joint(c1)
    tau_avg = 0.5 * (tau0 + tau1)

    return {
        "phi_0": float(phi0),
        "phi_1": float(phi1),
        "phi_avg": float(phi_avg),
        "tau_0": float(tau0),
        "tau_1": float(tau1),
        "tau_avg": float(tau_avg),
    }


def solve_lambda_star_and_dependence(
    marg: TwoWorldMarginals,
    rule: Rule,
    lambdas: Iterable[float],
    *,
    target_cc: float = 1.0,
    prefer_closed_form_fh_linear: bool = True,
    path: DependencePath = "fh_linear",
    path_params: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    End-to-end:
      1) compute theory curve
      2) solve λ* such that CC(λ*) = target_cc
      3) map λ* to dependence summaries

    Returns:
      lambda_star, phi_avg_star, tau_avg_star, found (0/1), plus some diagnostics.
    """
    path_params = path_params or {}
    df = theory_curve(marg, rule, lambdas, path=path, path_params=path_params)

    lam_star: Optional[float] = None
    if prefer_closed_form_fh_linear and path == "fh_linear":
        lam_star = lambda_star_closed_form_fh_linear(marg, rule, target_cc=target_cc, require_no_kink=True)

    if lam_star is None:
        lam_star = find_lambda_star_from_curve(df, target_cc=target_cc, prefer_first=True)

    if lam_star is None:
        return {
            "found": 0.0,
            "lambda_star": float("nan"),
            "phi_avg_star": float("nan"),
            "tau_avg_star": float("nan"),
        }

    dep = dependence_at_lambda(marg, lam_star, path=path, path_params=path_params)
    return {
        "found": 1.0,
        "lambda_star": float(lam_star),
        "phi_avg_star": float(dep["phi_avg"]),
        "tau_avg_star": float(dep["tau_avg"]),
    }


# =========================
# Delta-method variance hooks (fast analytic SEs)
# =========================

def var_diff_of_props(p1: float, p0: float, n1: int, n0: int) -> float:
    """
    Asymptotic var( (p1_hat - p0_hat) ) under independent binomial sampling.

    var(p_hat) ≈ p(1-p)/n  (for large n)

    If your sampling uses multinomial draws for the 2x2 table, pC_hat is still a
    sum of multinomial cells and has the same variance pC(1-pC)/n.
    """
    if n1 <= 0 or n0 <= 0:
        return float("nan")
    return (p1 * (1.0 - p1) / n1) + (p0 * (1.0 - p0) / n0)


def var_JC_delta(
    marg: TwoWorldMarginals,
    rule: Rule,
    lam: float,
    *,
    n_per_world: int,
    path: DependencePath = "fh_linear",
    path_params: Optional[Dict[str, float]] = None,
    assume_no_kink: bool = True,
) -> float:
    """
    Delta-method approximation for var( J_C_hat ) at a given lambda, under
    independent sampling in each world.

    If Δ = pC^1 - pC^0 does not change sign (no kink), then:
        J_C = |Δ| and locally J_C_hat behaves like Δ_hat in variance.

    If kink exists or Δ≈0, the absolute value makes local asymptotics trickier;
    we return NaN unless assume_no_kink=False (you can still use it as a rough approx).
    """
    path_params = path_params or {}
    m = compute_metrics_for_lambda(marg, rule, lam, path=path, path_params=path_params)
    dC = float(m["dC"])
    if assume_no_kink and abs(dC) < 1e-10:
        return float("nan")
    pC1 = float(m["pC_1"])
    pC0 = float(m["pC_0"])
    return float(var_diff_of_props(pC1, pC0, n_per_world, n_per_world))


def var_CC_delta(
    marg: TwoWorldMarginals,
    rule: Rule,
    lam: float,
    *,
    n_per_world: int,
    path: DependencePath = "fh_linear",
    path_params: Optional[Dict[str, float]] = None,
    assume_Jbest_fixed: bool = True,
) -> float:
    """
    Delta-method approximation for var( CC_hat ) at a given lambda.

    Simplest (and typical in this experiment) assumption:
      - J_best is fixed because marginals are configured by design.
      Then:
        CC = JC / Jbest  ⇒  var(CC_hat) ≈ var(JC_hat) / Jbest^2

    If you estimate marginals from data and therefore estimate J_best, you can
    extend this with covariance terms (left to analyze_bootstrap.py).
    """
    path_params = path_params or {}
    _, _, Jbest = compute_singleton_Js(marg)
    if Jbest <= 0.0:
        return float("nan")
    varJC = var_JC_delta(
        marg,
        rule,
        lam,
        n_per_world=n_per_world,
        path=path,
        path_params=path_params,
        assume_no_kink=True,
    )
    if math.isnan(varJC):
        return float("nan")
    return float(varJC / (Jbest * Jbest))


def se_from_var(v: float) -> float:
    return float(math.sqrt(v)) if (v >= 0.0 and not math.isnan(v)) else float("nan")


def approx_cc_ci_normal(
    cc_hat: float,
    var_cc: float,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Quick normal-approx CI:
        cc_hat ± z_{1-alpha/2} * sqrt(var_cc)

    This is an *analytic* approximation. Use BCa bootstrap for high rigor,
    especially near kinks or boundaries.
    """
    if math.isnan(var_cc):
        return (float("nan"), float("nan"))
    se = math.sqrt(max(0.0, var_cc))
    # hardcode z for alpha=0.05 to avoid scipy dependency
    z = 1.959963984540054  # Φ^{-1}(0.975)
    lo = cc_hat - z * se
    hi = cc_hat + z * se
    return (float(lo), float(hi))


# =========================
# Paper-writing diagnostics
# =========================

def summarize_cliff_conditions(
    marg: TwoWorldMarginals,
    rule: Rule,
    *,
    path: DependencePath = "fh_linear",
    path_params: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compact “defense-ready” diagnostics for your Methods/Theory writeup.

    Includes:
      - J_A, J_B, J_best
      - Δ(0), Δ(1) and CC(0), CC(1)
      - FH widths
      - kink location (FH-linear only)
    """
    path_params = path_params or {}
    marg.validate()
    JA, JB, Jbest = compute_singleton_Js(marg)

    b0 = fh_bounds(marg.w0.pA, marg.w0.pB)
    b1 = fh_bounds(marg.w1.pA, marg.w1.pB)

    m0 = compute_metrics_for_lambda(marg, rule, 0.0, path=path, path_params=path_params)
    m1 = compute_metrics_for_lambda(marg, rule, 1.0, path=path, path_params=path_params)

    out = {
        "JA": float(JA),
        "JB": float(JB),
        "Jbest": float(Jbest),
        "FH_width_w0": float(b0.width),
        "FH_width_w1": float(b1.width),
        "dC_at_0": float(m0["dC"]),
        "dC_at_1": float(m1["dC"]),
        "CC_at_0": float(m0["CC"]),
        "CC_at_1": float(m1["CC"]),
        "path": float(0.0),  # placeholder numeric to keep JSON-friendly in some pipelines
    }

    if path == "fh_linear":
        kink = detect_abs_kink_fh_linear(marg, rule)
        out["kink_lambda"] = float(kink) if kink is not None else float("nan")
        pars = deltaC_affine_params_fh_linear(marg, rule)
        out["delta0"] = float(pars["delta0"])
        out["slope"] = float(pars["slope"])
    else:
        out["kink_lambda"] = float("nan")
        out["delta0"] = float("nan")
        out["slope"] = float("nan")

    return out


# =========================
# Minimal self-check harness
# =========================

if __name__ == "__main__":
    # This block is intentionally tiny and deterministic.
    # Your real validation lives in unit tests + run_all.py.

    # Example S1-like marginals (replace with your config values)
    marg = TwoWorldMarginals(
        w0=WorldMarginals(pA=0.20, pB=0.15),
        w1=WorldMarginals(pA=0.70, pB=0.55),
    )

    for rule in ("OR", "AND"):
        rule = cast(Rule, rule)
        diag = summarize_cliff_conditions(marg, rule)
        print(f"\n== {rule} diagnostics ==")
        for k, v in diag.items():
            print(f"{k:>14s} : {v}")

        lambdas = np.linspace(0.0, 1.0, 21)
        df = theory_curve(marg, rule, lambdas, path="fh_linear")
        lam_star = lambda_star_closed_form_fh_linear(marg, rule, target_cc=1.0, require_no_kink=True)
        if lam_star is None:
            lam_star = find_lambda_star_from_curve(df, target_cc=1.0)

        print(f"lambda_star ≈ {lam_star}")
        if lam_star is not None:
            dep = dependence_at_lambda(marg, lam_star)
            print(f"phi_avg* ≈ {dep['phi_avg']}, tau_avg* ≈ {dep['tau_avg']}")

"""
theory_core.py
==============

A research-grade, rigor-first *core* module for two-rail, two-world compositional
uncertainty analysis under *unknown dependence*.

This file is intentionally "boring and defensible":

- No plotting.
- No pandas requirement.
- Minimal dependencies (NumPy only; SciPy optional for Gaussian copula CDF).
- Brutally strict validation and explicit degeneracy handling.
- Frozen dataclasses and a semantic exception hierarchy.
- Invariants are enforced via a configurable policy: raise / warn / ignore.

If you later add "curve building", "lambda* solving", "CI/Bootstrap", "report
diagnostics", those belong in `theory_analysis.py` and should import from here.

-------------------------------------------------------------------------------
Core Problem (identifiability)
-------------------------------------------------------------------------------
Given two binary guardrails A and B with marginals:

    pA = P(A=1),  pB = P(B=1),

the joint overlap:

    p11 = P(A=1, B=1)

is not identifiable from (pA,pB) alone. Probability axioms imply only the
Fréchet-Hoeffding feasibility envelope:

    L = max(0, pA + pB - 1)
    U = min(pA, pB)
    L <= p11 <= U

We provide:
- FH bounds and joint feasibility validation
- Composition maps C = f(A,B): AND / OR / COND_OR
- Dependence parameterizations (FH paths, Clayton copula, Gaussian copula)
- Joint table construction (p00,p01,p10,p11)
- Dependence summaries (phi, Kendall tau-a) for interpretability
- Two-world primitives and *dependence-agnostic* JC/CC bounds

-------------------------------------------------------------------------------
Mathematical conventions
-------------------------------------------------------------------------------
Binary variables A,B ∈ {0,1}. 2×2 joint cells:
    p11 = P(A=1,B=1)
    p10 = P(A=1,B=0) = pA - p11
    p01 = P(A=0,B=1) = pB - p11
    p00 = P(A=0,B=0) = 1 - pA - pB + p11

Composition rules (guardrail "triggers"):
- AND:      C = A ∧ B    =>  pC = p11
- OR:       C = A ∨ B    =>  pC = pA + pB - p11
- COND_OR:  "residual-independence OR" (degenerate w.r.t p11)
            pC = pA + (1-pA)*pB

Two worlds (world0, world1) have marginals (pA0,pB0) and (pA1,pB1).
We define:
    J_A = |pA1 - pA0|
    J_B = |pB1 - pB0|
    J_best = max(J_A, J_B)

For a chosen composition rule, each world induces an interval of feasible pC
over all feasible dependence. Therefore the *dependence-agnostic* JC bounds:
    JC = |pC1 - pC0|
    JC_min/max are computed from interval geometry (no dependence assumption)

And CC bounds:
    CC = JC / J_best
    If J_best ≈ 0, CC is defined as 0 (degenerate normalization).

-------------------------------------------------------------------------------
Design principles (what makes this "PhD-level")
-------------------------------------------------------------------------------
1) Strict validation and semantic errors: domain checks, NaN/Inf rejection,
   feasibility checks, consistent tolerances.

2) Degeneracy is first-class: pA=0/1, pB=0/1, J_best=0, COND_OR ignoring p11,
   Gaussian copula at extreme margins, etc.

3) Optional dependencies done honestly: SciPy is optional; no silent import
   failures for "scipy" method.

4) Invariant policy: users can require hard errors (production), warnings
   (research notebooks), or ignore (exploratory scans).

-------------------------------------------------------------------------------
Public API
-------------------------------------------------------------------------------
- Errors: TheoryError, InputValidationError, FeasibilityError, NumericalStabilityError
- Config: TheoryConfig, CONFIG, set_config(), temporary_config()
- FH feasibility: fh_bounds(), validate_joint()
- Joint table: joint_cells_from_marginals(), validate_joint_cells()
- Composition: composed_rate(), composed_rate_bounds()
- Dependence: p11_from_lambda(), p11_clayton_copula(), p11_gaussian_copula()
- Summaries: phi_from_joint(), kendall_tau_a_from_cells()
- Two-world: TwoWorldMarginals, singleton_gaps(), jc_bounds(), cc_bounds()

"""

from __future__ import annotations

import contextlib
import math
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Literal,
    TypedDict,
)

import numpy as np

# =============================================================================
# Semantic errors (do not use ValueError directly in public functions)
# =============================================================================


class TheoryError(Exception):
    """Base error for this module."""


class InputValidationError(TheoryError, ValueError):
    """Inputs violate the contract: domain/type/shape/unknown parameters."""


class FeasibilityError(TheoryError):
    """Requested joint probability is infeasible under FH constraints."""


class NumericalStabilityError(TheoryError, FloatingPointError):
    """Invariant violation or numerical instability (policy-controlled)."""


# =============================================================================
# Configuration and invariant policy
# =============================================================================

InvariantPolicy = Literal["raise", "warn", "ignore"]


@dataclass(frozen=True)
class TheoryConfig:
    """
    Global core configuration.

    eps_prob:
        Tolerance used for:
        - treating tiny excursions outside [0,1] as clip-able,
        - feasibility/invariant comparisons.

    strict_params:
        If True (default), unknown keys in path/call parameters raise.

    invariant_policy:
        - "raise": violations raise NumericalStabilityError (default, production)
        - "warn":  violations emit RuntimeWarning (research notebooks)
        - "ignore": violations are ignored (exploratory use ONLY)

    mc_warn_threshold:
        For Gaussian copula Monte Carlo: if n_mc < threshold, warn (still run).

    ppf_clip:
        Gaussian copula needs Φ^{-1}(u). u must be in (0,1). We clip to:
            [ppf_clip, 1-ppf_clip]

    renorm_tol:
        Joint-cell sums within this tolerance of 1.0 are accepted without renorm.
        Beyond this, we renormalize if all cells are nonnegative within eps.
    """

    eps_prob: float = 1e-12
    strict_params: bool = True
    invariant_policy: InvariantPolicy = "raise"
    mc_warn_threshold: int = 50_000
    ppf_clip: float = 1e-16
    renorm_tol: float = 1e-9


CONFIG = TheoryConfig()


def set_config(**kwargs: Any) -> None:
    """
    Update global CONFIG immutably.

    Example:
        set_config(eps_prob=1e-10, invariant_policy="warn")

    Notes:
      - This mutates module-global state; prefer temporary_config() in tests/notebooks.
    """
    global CONFIG
    allowed = set(TheoryConfig.__dataclass_fields__.keys())
    for k in kwargs:
        if k not in allowed:
            raise InputValidationError(f"Unknown config field: {k!r}")
    new = {**CONFIG.__dict__, **kwargs}
    CONFIG = TheoryConfig(**new)


@contextlib.contextmanager
def temporary_config(**kwargs: Any) -> Iterator[None]:
    """
    Context manager to apply config changes and restore them afterward.

    This prevents tests/notebooks from leaking CONFIG changes into other tests.

    Example:
        with temporary_config(invariant_policy="warn"):
            ...
    """
    old = CONFIG
    set_config(**kwargs)
    try:
        yield
    finally:
        globals()["CONFIG"] = old


def _invariant(msg: str) -> None:
    """
    Handle invariant violations based on CONFIG.invariant_policy.
    """
    pol = str(CONFIG.invariant_policy).strip().lower()
    if pol == "ignore":
        return
    if pol == "warn":
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return
    if pol == "raise":
        raise NumericalStabilityError(msg)
    raise InputValidationError(f"Unknown invariant_policy={CONFIG.invariant_policy!r}")


# =============================================================================
# Core validation helpers
# =============================================================================


def _is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.floating)) and math.isfinite(float(x))


def _require_prob(p: Any, name: str) -> float:
    """
    Require p to be a finite scalar probability in [0,1] (within eps clip).

    - Rejects NaN/Inf and non-numbers.
    - Allows tiny excursions outside [0,1] within CONFIG.eps_prob, then clips.
    """
    if not _is_finite_number(p):
        raise InputValidationError(f"{name} must be a finite number, got {p!r}")
    x = float(p)
    eps = float(CONFIG.eps_prob)

    if x < -eps or x > 1.0 + eps:
        raise InputValidationError(f"{name} must be in [0,1], got {x}")
    if x < 0.0 and x >= -eps:
        x = 0.0
    if x > 1.0 and x <= 1.0 + eps:
        x = 1.0
    return x


def _require_int(n: Any, name: str, *, min_value: int = 1) -> int:
    if not isinstance(n, int):
        raise InputValidationError(f"{name} must be an int, got {type(n).__name__}")
    if n < min_value:
        raise InputValidationError(f"{name} must be >= {min_value}, got {n}")
    return n


def _require_lambda(lam: Any) -> float:
    return _require_prob(lam, "lambda")


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


# =============================================================================
# Canonical rule set (core)
# =============================================================================

Rule = Literal["AND", "OR", "COND_OR"]
_RULES: tuple[str, ...] = ("AND", "OR", "COND_OR")


def _require_rule(rule: Any) -> Rule:
    if not isinstance(rule, str):
        raise InputValidationError(f"rule must be a string, got {type(rule).__name__}")
    r = rule.strip().upper()
    if r not in _RULES:
        raise InputValidationError(f"Unknown rule {rule!r}. Allowed: {list(_RULES)}")
    return r  # type: ignore[return-value]


# =============================================================================
# Data structures
# =============================================================================


@dataclass(frozen=True, slots=True)
class FHBounds:
    """
    Fréchet-Hoeffding bounds container for p11 given (pA,pB):

        lower = max(0, pA+pB-1)
        upper = min(pA,pB)

    Invariants (within CONFIG.eps_prob):
      - 0 <= lower <= upper <= 1
    """

    lower: float
    upper: float
    pA: float
    pB: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "pA", _require_prob(self.pA, "pA"))
        object.__setattr__(self, "pB", _require_prob(self.pB, "pB"))
        lo = float(self.lower)
        hi = float(self.upper)

        if not math.isfinite(lo) or not math.isfinite(hi):
            raise FeasibilityError(f"FH bounds must be finite, got lower={lo}, upper={hi}")

        eps = float(CONFIG.eps_prob)
        if lo < -eps or hi > 1.0 + eps:
            raise FeasibilityError(f"FH bounds out of [0,1]: [{lo},{hi}]")
        if lo > hi + eps:
            raise FeasibilityError(f"Invalid FH ordering: lower={lo} > upper={hi}")

        # Clip tiny excursions to [0,1]
        if lo < 0.0 and lo >= -eps:
            lo = 0.0
        if hi > 1.0 and hi <= 1.0 + eps:
            hi = 1.0
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)

    @property
    def width(self) -> float:
        return float(self.upper - self.lower)

    @property
    def is_degenerate(self) -> bool:
        return abs(self.width) <= float(CONFIG.eps_prob)

    def contains(self, p11: float) -> bool:
        x = float(p11)
        eps = float(CONFIG.eps_prob)
        return (self.lower - eps) <= x <= (self.upper + eps)


@dataclass(frozen=True, slots=True)
class TwoWorldMarginals:
    """
    Two worlds with marginals:
        world0: (pA0, pB0)
        world1: (pA1, pB1)

    Stored as floats in [0,1] with strict validation.
    """

    pA0: float
    pB0: float
    pA1: float
    pB1: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "pA0", _require_prob(self.pA0, "pA0"))
        object.__setattr__(self, "pB0", _require_prob(self.pB0, "pB0"))
        object.__setattr__(self, "pA1", _require_prob(self.pA1, "pA1"))
        object.__setattr__(self, "pB1", _require_prob(self.pB1, "pB1"))


class JointCells(TypedDict):
    """Typed 2×2 joint table."""

    p00: float
    p01: float
    p10: float
    p11: float


# =============================================================================
# FH feasibility primitives
# =============================================================================


def fh_bounds(pA: Any, pB: Any) -> tuple[float, float]:
    """
    Return Fréchet-Hoeffding bounds (lower, upper) for p11 given marginals pA,pB.

    Mathematical truth (for Bernoulli A,B):
        lower = max(0, pA + pB - 1)
        upper = min(pA, pB)

    Output is clipped to [0,1] and order is invariant-checked.
    """
    a = _require_prob(pA, "pA")
    b = _require_prob(pB, "pB")
    lo = max(0.0, a + b - 1.0)
    hi = min(a, b)
    lo = _clip01(lo)
    hi = _clip01(hi)
    if lo > hi + float(CONFIG.eps_prob):
        _invariant(f"FH bounds violated: lo={lo} > hi={hi} for pA={a}, pB={b}")
    return (lo, hi)


def fh_bounds_obj(pA: Any, pB: Any) -> FHBounds:
    """
    Convenience: return FHBounds dataclass (validated) for pA,pB.
    """
    lo, hi = fh_bounds(pA, pB)
    return FHBounds(lower=lo, upper=hi, pA=float(pA), pB=float(pB))


def validate_joint(pA: Any, pB: Any, p11: Any) -> float:
    """
    Validate that p11 is feasible given marginals (pA,pB) under FH bounds.

    - Raises FeasibilityError if p11 violates FH beyond CONFIG.eps_prob.
    - If p11 is within eps of the FH boundary, it is clipped to the boundary.

    Returns:
        A float p11 that is guaranteed to satisfy FH within tolerance.
    """
    a = _require_prob(pA, "pA")
    b = _require_prob(pB, "pB")
    x = _require_prob(p11, "p11")
    lo, hi = fh_bounds(a, b)
    eps = float(CONFIG.eps_prob)

    if x < lo - eps or x > hi + eps:
        raise FeasibilityError(
            f"Infeasible p11={x} for marginals pA={a}, pB={b} with FH=[{lo},{hi}]"
        )
    if x < lo:
        x = lo
    if x > hi:
        x = hi
    return float(x)


# =============================================================================
# Joint table construction (2×2)
# =============================================================================


def joint_cells_from_marginals(pA: Any, pB: Any, p11: Any) -> JointCells:
    """
    Construct a full 2×2 joint distribution from marginals and overlap p11.

    Cells:
        p10 = pA - p11
        p01 = pB - p11
        p00 = 1 - pA - pB + p11

    Guarantees:
      - Validates p11 feasibility (FH).
      - Clips tiny negative cells within eps.
      - If cells sum deviates slightly from 1, optionally renormalizes (only if
        all cells are nonnegative within eps).

    Raises:
      - FeasibilityError if p11 infeasible.
      - NumericalStabilityError/InputValidationError if cells are materially invalid.
    """
    a = _require_prob(pA, "pA")
    b = _require_prob(pB, "pB")
    x = validate_joint(a, b, p11)

    p10 = a - x
    p01 = b - x
    p00 = 1.0 - a - b + x

    eps = float(CONFIG.eps_prob)

    # Reject material negativity (beyond eps).
    if (p00 < -eps) or (p01 < -eps) or (p10 < -eps) or (x < -eps):
        raise NumericalStabilityError(
            f"Invalid joint cells beyond tolerance: p00={p00}, p01={p01}, p10={p10}, p11={x} "
            f"(pA={a}, pB={b})"
        )

    # Clip tiny negatives
    p00 = 0.0 if (p00 < 0.0 and p00 >= -eps) else p00
    p01 = 0.0 if (p01 < 0.0 and p01 >= -eps) else p01
    p10 = 0.0 if (p10 < 0.0 and p10 >= -eps) else p10

    # Clip to [0,1] defensively (should already hold)
    p00 = _clip01(p00)
    p01 = _clip01(p01)
    p10 = _clip01(p10)
    x = _clip01(x)

    s = p00 + p01 + p10 + x
    if abs(s - 1.0) > float(CONFIG.renorm_tol):
        # Only renormalize if sum is positive and cells are nonnegative (already checked)
        if s <= 0.0:
            _invariant(f"Joint cells sum is nonpositive: sum={s}")
        p00, p01, p10, x = (p00 / s, p01 / s, p10 / s, x / s)

    cells: JointCells = {"p00": float(p00), "p01": float(p01), "p10": float(p10), "p11": float(x)}
    validate_joint_cells(cells)
    return cells


def validate_joint_cells(cells: Mapping[str, Any], *, tol: float = 1e-8) -> None:
    """
    Strict validation for a joint table.

    Requirements:
      - keys p00,p01,p10,p11 exist
      - each is finite and within [0,1] up to tol
      - sum equals 1 within tol

    Raises InputValidationError if contract is violated.
    """
    required = ("p00", "p01", "p10", "p11")
    for k in required:
        if k not in cells:
            raise InputValidationError(f"Missing cell {k} in cells={dict(cells)}")
        v = cells[k]
        if not _is_finite_number(v):
            raise InputValidationError(f"Cell {k} must be finite, got {v!r}")
        vf = float(v)
        if vf < -tol or vf > 1.0 + tol:
            raise InputValidationError(f"Cell {k} out of [0,1]: {vf}")

    s = float(cells["p00"] + cells["p01"] + cells["p10"] + cells["p11"])
    if abs(s - 1.0) > tol:
        raise InputValidationError(f"Joint cells do not sum to 1: sum={s}, cells={dict(cells)}")


# =============================================================================
# Composition primitives
# =============================================================================


def composed_rate(rule: Any, pA: Any, pB: Any, p11: Any) -> float:
    """
    Compute composed rate pC = P(C=1) given rule and joint overlap p11.

    AND:
        pC = p11
    OR:
        pC = pA + pB - p11
    COND_OR:
        pC = pA + (1-pA)*pB   (independent residual channel; ignores p11)

    Returns pC clipped to [0,1].
    """
    r = _require_rule(rule)
    a = _require_prob(pA, "pA")
    b = _require_prob(pB, "pB")

    if r == "COND_OR":
        return _clip01(a + (1.0 - a) * b)

    x = validate_joint(a, b, p11)
    if r == "AND":
        return _clip01(x)

    # OR
    return _clip01(a + b - x)


def composed_rate_bounds(rule: Any, pA: Any, pB: Any) -> tuple[float, float]:
    """
    Tight bounds for pC over all feasible p11 consistent with (pA,pB).

    For a fixed world:
      - AND: pC in [L,U]
      - OR:  pC = pA+pB - p11 is decreasing in p11
             => pC in [pA+pB-U, pA+pB-L]
      - COND_OR: degenerate interval [v,v] where v = pA + (1-pA)*pB
    """
    r = _require_rule(rule)
    a = _require_prob(pA, "pA")
    b = _require_prob(pB, "pB")
    lo, hi = fh_bounds(a, b)

    if r == "COND_OR":
        v = _clip01(a + (1.0 - a) * b)
        return (v, v)

    if r == "AND":
        return (lo, hi)

    # OR
    loC = _clip01(a + b - hi)
    hiC = _clip01(a + b - lo)
    if loC > hiC + float(CONFIG.eps_prob):
        _invariant(f"OR bounds reversed: [{loC},{hiC}] for pA={a}, pB={b}")
    return (loC, hiC)


# =============================================================================
# Dependence paths: FH family + copulas
# =============================================================================


def _validate_params(params: Mapping[str, Any] | None, allowed: Sequence[str]) -> dict[str, Any]:
    """
    Validate/whitelist path parameters.

    If CONFIG.strict_params is True:
      unknown keys raise InputValidationError.

    Returns dict containing only allowed keys.
    """
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise InputValidationError("path_params must be a mapping/dict or None")
    d = dict(params)
    unknown = set(d.keys()) - set(allowed)
    if unknown and CONFIG.strict_params:
        raise InputValidationError(
            f"Unknown path_params keys {sorted(unknown)}; allowed={list(allowed)}"
        )
    return {k: d[k] for k in allowed if k in d}


def p11_from_lambda(
    path: Any,
    lam: Any,
    pA: Any,
    pB: Any,
    path_params: Mapping[str, Any] | None = None,
) -> float:
    """
    Map lambda ∈ [0,1] to a feasible p11 along a specified *FH-envelope path*.

    Supported paths (scanner paths, not "truth claims"):
      - "fh_linear": p11 = L + lam*(U-L)
      - "fh_power":  p11 = L + lam^power*(U-L)   (power>0)
      - "fh_scurve": p11 = L + s(lam)*(U-L) where s is logistic-symmetric

    Returns:
      A feasible p11 clipped/validated within FH.

    Notes:
      - These paths explore the feasible dependence envelope; they are not copulas.
      - For copula-based dependence, use p11_clayton_copula / p11_gaussian_copula.
    """
    if not isinstance(path, str):
        raise InputValidationError("path must be a string")
    name = path.strip().lower()
    l = _require_lambda(lam)
    a = _require_prob(pA, "pA")
    b = _require_prob(pB, "pB")
    L, U = fh_bounds(a, b)
    W = U - L

    if name == "fh_linear":
        return float(L + l * W)

    if name == "fh_power":
        pp = _validate_params(path_params, allowed=("power",))
        power = float(pp.get("power", 1.0))
        if not math.isfinite(power) or power <= 0.0:
            raise InputValidationError("fh_power requires power>0 and finite")
        return float(L + (l**power) * W)

    if name == "fh_scurve":
        pp = _validate_params(path_params, allowed=("alpha",))
        alpha = float(pp.get("alpha", 8.0))
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise InputValidationError("fh_scurve requires alpha>0 and finite")

        def sigmoid(x: float) -> float:
            # stable logistic
            if x >= 0:
                z = math.exp(-x)
                return 1.0 / (1.0 + z)
            z = math.exp(x)
            return z / (1.0 + z)

        # symmetric logistic mapped to [0,1] exactly at endpoints
        s0 = sigmoid(alpha * (0.0 - 0.5))
        s1 = sigmoid(alpha * (1.0 - 0.5))
        s = sigmoid(alpha * (l - 0.5))
        s01 = (s - s0) / (s1 - s0) if (s1 != s0) else l
        return float(L + s01 * W)

    raise InputValidationError(f"Unknown path={path!r}. Allowed: fh_linear, fh_power, fh_scurve")


def p11_clayton_copula(pA: Any, pB: Any, theta: Any) -> float:
    """
    Clayton copula overlap:
        C(u,v) = (u^{-θ} + v^{-θ} - 1)^(-1/θ),  θ >= 0
    with limit θ→0 being independence: uv.

    Returns:
      p11 = C(pA,pB; theta), validated for FH feasibility.

    Note:
      For Bernoulli margins, C(u,v) is still a valid joint CDF at (u,v),
      but discrete margins imply not all dependence patterns are reachable by
      a given copula family. This is still useful as a smooth dependence model.
    """
    u = _require_prob(pA, "pA")
    v = _require_prob(pB, "pB")
    if not _is_finite_number(theta):
        raise InputValidationError("theta must be a finite number")
    th = float(theta)
    if th < 0.0:
        raise InputValidationError("Clayton theta must be >= 0")

    if abs(th) <= 1e-15:
        x = u * v
    else:
        term = (u ** (-th)) + (v ** (-th)) - 1.0
        # Numerical guard: due to float error, term could be tiny <=0
        if term <= 0.0:
            term = max(term, float(CONFIG.eps_prob))
        x = term ** (-1.0 / th)

    return validate_joint(u, v, x)


# --- Normal quantile: Acklam approximation (no SciPy dependency) ---
def _normal_ppf(p: float) -> float:
    """
    Approximate inverse CDF (quantile) for N(0,1), p in (0,1).

    Uses Peter J. Acklam's rational approximation (widely used, high accuracy).

    Raises InputValidationError if p is not in (0,1).
    """
    p = float(p)
    if not (0.0 < p < 1.0):
        raise InputValidationError("ppf requires p in (0,1)")

    # Coefficients for Acklam approximation
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
    phigh = 1.0 - plow

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )

    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )

    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def p11_gaussian_copula(
    pA: Any,
    pB: Any,
    rho: Any,
    *,
    method: str = "mc",
    n_mc: int = 100_000,
    seed: int | None = None,
    antithetic: bool = True,
) -> float:
    """
    Gaussian copula overlap:
        p11 = Φ_2( Φ^{-1}(pA), Φ^{-1}(pB) ; rho )

    method:
      - "scipy": use scipy.stats.multivariate_normal.cdf (exact-ish)
      - "mc":    Monte Carlo fallback (deterministic by seed)

    Guarantees:
      - Output is validated/clipped to FH feasibility.
      - rho validated in [-1,1] with eps clipping.

    Notes (honest research nuance):
      - With discrete margins, Gaussian copula is not a "scanner" of FH extremes.
      - Still valuable as a smooth dependence sensitivity family.
    """
    u = _require_prob(pA, "pA")
    v = _require_prob(pB, "pB")

    if not _is_finite_number(rho):
        raise InputValidationError("rho must be finite")
    r = float(rho)
    eps = float(CONFIG.eps_prob)
    if r < -1.0 - eps or r > 1.0 + eps:
        raise InputValidationError("rho must be in [-1,1]")
    r = max(-1.0, min(1.0, r))

    m = str(method).strip().lower()
    if m not in ("mc", "scipy"):
        raise InputValidationError("method must be 'mc' or 'scipy'")

    # Degenerate margins: if u or v is 0/1, overlap is forced.
    if u in (0.0, 1.0) or v in (0.0, 1.0):
        return validate_joint(u, v, u * v)

    # Clip for quantiles
    clip = float(CONFIG.ppf_clip)
    uu = min(max(u, clip), 1.0 - clip)
    vv = min(max(v, clip), 1.0 - clip)
    a = _normal_ppf(uu)
    b = _normal_ppf(vv)

    if m == "scipy":
        try:
            from scipy.stats import multivariate_normal  # type: ignore
        except Exception as e:
            raise InputValidationError("SciPy not available for method='scipy'") from e

        mean = np.array([0.0, 0.0], dtype=float)
        cov = np.array([[1.0, r], [r, 1.0]], dtype=float)
        x = float(multivariate_normal(mean=mean, cov=cov).cdf([a, b]))
        return validate_joint(u, v, x)

    # Monte Carlo
    _require_int(n_mc, "n_mc", min_value=1)
    if n_mc < int(CONFIG.mc_warn_threshold):
        warnings.warn(
            f"Gaussian copula MC with n_mc={n_mc} (<{CONFIG.mc_warn_threshold}) may be noisy.",
            RuntimeWarning,
            stacklevel=2,
        )

    rng = np.random.default_rng(seed)
    # Correlated normals via linear construction:
    # y = rho*x + sqrt(1-rho^2)*z
    s = math.sqrt(max(0.0, 1.0 - r * r))
    x = rng.standard_normal(size=n_mc)
    z = rng.standard_normal(size=n_mc)
    y = r * x + s * z

    if antithetic:
        x = np.concatenate([x, -x])
        y = np.concatenate([y, -y])

    est = float(np.mean((x <= a) & (y <= b)))
    return validate_joint(u, v, est)


# =============================================================================
# Dependence summaries (interpretability)
# =============================================================================


def phi_from_joint(pA: Any, pB: Any, p11: Any) -> float:
    """
    Phi coefficient (binary Pearson correlation):
        φ = (p11 - pA*pB) / sqrt(pA(1-pA)pB(1-pB))

    Returns NaN if denom is 0 (degenerate marginals).
    """
    a = _require_prob(pA, "pA")
    b = _require_prob(pB, "pB")
    x = validate_joint(a, b, p11)
    denom = a * (1.0 - a) * b * (1.0 - b)
    if denom <= 0.0:
        return float("nan")
    return float((x - a * b) / math.sqrt(denom))


def kendall_tau_a_from_cells(cells: Mapping[str, Any]) -> float:
    """
    Kendall tau-a for a 2×2 distribution:
        τ_a = 2(p00*p11 - p01*p10)

    Requires valid joint cells summing to 1.
    """
    validate_joint_cells(cells)
    p00 = float(cells["p00"])
    p01 = float(cells["p01"])
    p10 = float(cells["p10"])
    p11 = float(cells["p11"])
    return float(2.0 * (p00 * p11 - p01 * p10))


def avg_ignore_nan(x: float, y: float) -> float:
    """
    Average that ignores NaN (used in analysis; harmless utility in core).
    """
    x = float(x)
    y = float(y)
    if math.isnan(x) and math.isnan(y):
        return float("nan")
    if math.isnan(x):
        return y
    if math.isnan(y):
        return x
    return 0.5 * (x + y)


# =============================================================================
# Two-world primitives and dependence-agnostic bounds
# =============================================================================


def singleton_gaps(w: TwoWorldMarginals) -> tuple[float, float, float]:
    """
    Singleton gaps:
        J_A = |pA1 - pA0|
        J_B = |pB1 - pB0|
        J_best = max(J_A, J_B)
    """
    jA = abs(w.pA1 - w.pA0)
    jB = abs(w.pB1 - w.pB0)
    return (float(jA), float(jB), float(max(jA, jB)))


def _interval_gap_minmax(i0: tuple[float, float], i1: tuple[float, float]) -> tuple[float, float]:
    """
    Given intervals I0=[a,b], I1=[c,d], compute:
      min gap = min_{x in I0, y in I1} |x-y|
      max gap = max_{x in I0, y in I1} |x-y|

    Assumes a<=b and c<=d (we enforce defensively).
    """
    a, b = float(i0[0]), float(i0[1])
    c, d = float(i1[0]), float(i1[1])
    if b < a:
        a, b = b, a
    if d < c:
        c, d = d, c

    # min distance between intervals
    if b < c:
        mn = c - b
    elif d < a:
        mn = a - d
    else:
        mn = 0.0

    # max distance between endpoints
    mx = max(abs(a - d), abs(b - c))
    return (float(mn), float(mx))


def jc_bounds(w: TwoWorldMarginals, rule: Any) -> tuple[float, float]:
    """
    JC bounds over *all feasible dependence*.

    Steps:
      1) For each world, compute pC interval under rule via composed_rate_bounds.
      2) With pC0 ∈ I0 and pC1 ∈ I1, JC=|pC1-pC0| has min/max given by interval geometry.

    Returns:
      (JC_min, JC_max)
    """
    r = _require_rule(rule)
    I0 = composed_rate_bounds(r, w.pA0, w.pB0)
    I1 = composed_rate_bounds(r, w.pA1, w.pB1)
    mn, mx = _interval_gap_minmax(I0, I1)

    # Sanity: JC_max cannot exceed 1
    if mx > 1.0 + float(CONFIG.eps_prob):
        _invariant(f"JC_max > 1: {mx} from intervals {I0}, {I1}")
    return (float(mn), float(min(1.0, mx)))


def cc_bounds(w: TwoWorldMarginals, rule: Any) -> tuple[float, float]:
    """
    CC bounds = JC bounds / J_best, with degeneracy handling.

    Definition:
      If J_best <= eps, CC is defined as 0 and bounds are [0,0].

    Returns:
      (CC_min, CC_max)
    """
    _, _, jbest = singleton_gaps(w)
    if jbest <= float(CONFIG.eps_prob):
        return (0.0, 0.0)

    lo, hi = jc_bounds(w, rule)
    return (float(lo / jbest), float(hi / jbest))


# =============================================================================
# Export surface
# =============================================================================

__all__ = [
    "CONFIG",
    # data structures
    "FHBounds",
    "FeasibilityError",
    "InputValidationError",
    "InvariantPolicy",
    "JointCells",
    "NumericalStabilityError",
    # composition
    "Rule",
    # config
    "TheoryConfig",
    # errors
    "TheoryError",
    "TwoWorldMarginals",
    "avg_ignore_nan",
    "cc_bounds",
    "composed_rate",
    "composed_rate_bounds",
    # FH feasibility
    "fh_bounds",
    "fh_bounds_obj",
    "jc_bounds",
    # joint table
    "joint_cells_from_marginals",
    "kendall_tau_a_from_cells",
    "p11_clayton_copula",
    # dependence
    "p11_from_lambda",
    "p11_gaussian_copula",
    # summaries
    "phi_from_joint",
    "set_config",
    # two-world
    "singleton_gaps",
    "temporary_config",
    "validate_joint",
    "validate_joint_cells",
]

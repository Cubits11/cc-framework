"""
theory_analysis.py
==================

"Brutally PhD-level" analysis utilities that sit *on top of* `theory_core.py`.

This file is where you do **experiments**, **scans**, and **comparative claims**—
while keeping the pure mathematical contracts in `theory_core.py`.

Core design goals:
- Reproducible, deterministic scans (explicit grids, explicit seeds).
- Separation of concerns: core math in `theory_core`, analytics here.
- Honest about what is and isn't identifiable: FH envelope vs chosen dependence family.
- Provides both 1D *tied* scans (λ0=λ1) and 2D surfaces (λ0, λ1 independent).
- Supports FH-path lambda parameterization *and* copula families.

-------------------------------------------------------------------------------
What this module gives you
-------------------------------------------------------------------------------
1) Dependence scans (FH paths):
   - `scan_fh_path_tied_lambda`: λ shared across both worlds
   - `scan_fh_path_surface`:    λ0 and λ1 free -> JC/CC surface

2) Dependence scans (copula families):
   - `scan_clayton_theta_tied`: θ shared across both worlds
   - `scan_gaussian_rho_tied`:  ρ shared across both worlds (SciPy or MC)

3) Bounds & comparisons:
   - `bounds_summary`: compare FH worst-case vs chosen path/coplan family
   - inflation / contraction metrics: does your chosen family under-cover FH?

4) Inversion / calibration helpers:
   - `find_lambda_for_target_p11` (monotone along FH paths)
   - `find_lambda_for_target_phi`  (typically monotone-ish but not guaranteed)
   - `find_lambda_for_extreme_JC`  (grid argmin/argmax and refine)

-------------------------------------------------------------------------------
Important interpretability note (non-negotiable)
-------------------------------------------------------------------------------
- FH bounds are *axiomatic* given only marginals.
- Any path/coplan family is an *assumption* that chooses a subset of feasible joints.
- Your analysis must always report whether results depend on:
    (a) FH worst-case, or
    (b) assumed dependence family/path.

This module makes that distinction explicit in structures and reporting.

-------------------------------------------------------------------------------
Dependencies
-------------------------------------------------------------------------------
- NumPy required.
- SciPy optional only for Gaussian copula exact CDF (handled in `theory_core`).

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Literal, Mapping, Optional, Tuple

import numpy as np

from .theory_core import (
    CONFIG,
    InputValidationError,
    Rule,
    TwoWorldMarginals,
    _require_prob,  # internal; used only for strict validation here
    _require_rule,  # internal; used only for strict validation here
    cc_bounds,
    composed_rate,
    composed_rate_bounds,
    fh_bounds,
    jc_bounds,
    p11_clayton_copula,
    p11_from_lambda,
    p11_gaussian_copula,
    phi_from_joint,
    singleton_gaps,
    validate_joint,
)

# =============================================================================
# Public dataclasses
# =============================================================================

DependenceFamily = Literal["fh_path", "clayton", "gaussian"]


@dataclass(frozen=True, slots=True)
class Grid1D:
    """
    A strictly increasing grid over a dependence parameter.

    For FH-path scans: values are lambda ∈ [0,1]
    For Clayton scans: values are theta ∈ [0, +∞) (you choose a finite window)
    For Gaussian scans: values are rho ∈ [-1,1]

    Invariants:
      - 1D numpy array, finite
      - strictly increasing
    """

    values: np.ndarray
    name: str

    def __post_init__(self) -> None:
        v = np.asarray(self.values, dtype=float)
        if v.ndim != 1 or v.size < 2:
            raise InputValidationError("Grid1D.values must be a 1D array with size >= 2")
        if not np.all(np.isfinite(v)):
            raise InputValidationError("Grid1D.values must be finite")
        if not np.all(np.diff(v) > 0):
            raise InputValidationError("Grid1D.values must be strictly increasing")
        object.__setattr__(self, "values", v)
        object.__setattr__(self, "name", str(self.name))


@dataclass(frozen=True, slots=True)
class CurveResult:
    """
    1D scan result with tied dependence parameter across worlds.

    Arrays all have length N=len(grid).
    """

    family: DependenceFamily
    rule: Rule
    grid: Grid1D

    # p11 and pC in each world
    p11_0: np.ndarray
    p11_1: np.ndarray
    pC0: np.ndarray
    pC1: np.ndarray

    # gaps
    JC: np.ndarray
    CC: np.ndarray

    # optional interpretability stats
    phi0: Optional[np.ndarray] = None
    phi1: Optional[np.ndarray] = None
    tau0: Optional[np.ndarray] = None
    tau1: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        n = self.grid.values.size
        for name in ("p11_0", "p11_1", "pC0", "pC1", "JC", "CC"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.shape != (n,):
                raise InputValidationError(f"{name} must have shape ({n},), got {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise InputValidationError(f"{name} must be finite")
            object.__setattr__(self, name, arr)

        for opt in ("phi0", "phi1", "tau0", "tau1"):
            arr = getattr(self, opt)
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=float)
            if arr.shape != (n,):
                raise InputValidationError(f"{opt} must have shape ({n},), got {arr.shape}")
            object.__setattr__(self, opt, arr)

    @property
    def JC_min(self) -> float:
        return float(np.min(self.JC))

    @property
    def JC_max(self) -> float:
        return float(np.max(self.JC))

    @property
    def CC_min(self) -> float:
        return float(np.min(self.CC))

    @property
    def CC_max(self) -> float:
        return float(np.max(self.CC))

    def argmin_JC(self) -> int:
        return int(np.argmin(self.JC))

    def argmax_JC(self) -> int:
        return int(np.argmax(self.JC))

    def argmin_CC(self) -> int:
        return int(np.argmin(self.CC))

    def argmax_CC(self) -> int:
        return int(np.argmax(self.CC))


@dataclass(frozen=True, slots=True)
class SurfaceResult:
    """
    2D scan result where dependence parameters can differ across worlds.

    Useful when you refuse the (often implicit) assumption that dependence is
    "stable" across worlds.

    Output matrices are shape (N0, N1), where:
      - axis 0 indexes parameter for world0
      - axis 1 indexes parameter for world1
    """

    family: DependenceFamily
    rule: Rule
    grid0: Grid1D
    grid1: Grid1D

    JC: np.ndarray
    CC: np.ndarray

    def __post_init__(self) -> None:
        n0 = self.grid0.values.size
        n1 = self.grid1.values.size
        for name in ("JC", "CC"):
            mat = np.asarray(getattr(self, name), dtype=float)
            if mat.shape != (n0, n1):
                raise InputValidationError(f"{name} must have shape ({n0},{n1}), got {mat.shape}")
            if not np.all(np.isfinite(mat)):
                raise InputValidationError(f"{name} must be finite")
            object.__setattr__(self, name, mat)

    def minmax(self) -> Dict[str, Tuple[float, float]]:
        return {
            "JC": (float(np.min(self.JC)), float(np.max(self.JC))),
            "CC": (float(np.min(self.CC)), float(np.max(self.CC))),
        }


@dataclass(frozen=True, slots=True)
class BoundsSummary:
    """
    A structured comparison between:

      A) FH worst-case (dependence-agnostic) bounds from theory_core
      B) Bounds actually realized under a chosen path/coplan family scan

    You can use this to prevent yourself from accidentally claiming robustness
    when you only scanned an assumption family that under-covers FH extremes.
    """

    rule: Rule

    # FH worst-case
    JC_FH_min: float
    JC_FH_max: float
    CC_FH_min: float
    CC_FH_max: float

    # realized under scan
    JC_scan_min: float
    JC_scan_max: float
    CC_scan_min: float
    CC_scan_max: float

    # coverage diagnostics
    JC_undercoverage_low: float  # max(0, FH_min - scan_min)
    JC_undercoverage_high: float  # max(0, scan_max - FH_max) but typically 0; kept for symmetry
    JC_missing_extremes: (
        float  # (FH_range - scan_range)+, i.e., how much of FH range you didn't cover
    )
    JC_range_ratio: float  # scan_range / FH_range (NaN if FH_range==0)

    CC_missing_extremes: float
    CC_range_ratio: float

    family: DependenceFamily
    grid_name: str


# =============================================================================
# Small utilities
# =============================================================================


def default_lambda_grid(n: int = 201) -> Grid1D:
    """
    Default λ grid for FH scans.
    """
    if not isinstance(n, int) or n < 2:
        raise InputValidationError("n must be an int >= 2")
    return Grid1D(values=np.linspace(0.0, 1.0, n, dtype=float), name="lambda")


def default_rho_grid(n: int = 201) -> Grid1D:
    """
    Default ρ grid for Gaussian copula scans.
    """
    if not isinstance(n, int) or n < 2:
        raise InputValidationError("n must be an int >= 2")
    return Grid1D(values=np.linspace(-1.0, 1.0, n, dtype=float), name="rho")


def default_theta_grid(
    theta_max: float = 20.0,
    n: int = 201,
    *,
    include_zero: bool = True,
) -> Grid1D:
    """
    Default θ grid for Clayton copula scans.

    θ=0 is independence.
    Large θ approaches comonotone-like dependence (for continuous margins).
    For Bernoulli margins, this is still a smooth dependence family, not FH-extreme scanner.

    theta_max should be chosen based on how "strong" you want to probe.
    """
    theta_max = float(theta_max)
    if not math.isfinite(theta_max) or theta_max <= 0:
        raise InputValidationError("theta_max must be finite and > 0")
    if not isinstance(n, int) or n < 2:
        raise InputValidationError("n must be an int >= 2")

    if include_zero:
        vals = np.linspace(0.0, theta_max, n, dtype=float)
    else:
        vals = np.linspace(theta_max / (n + 1), theta_max, n, dtype=float)
    # Ensure strictly increasing
    vals = np.unique(vals)
    if vals.size < 2:
        raise InputValidationError("theta grid collapsed; choose larger n or different theta_max")
    return Grid1D(values=vals, name="theta")


def _safe_divide(num: np.ndarray, den: float) -> np.ndarray:
    den = float(den)
    if abs(den) <= float(CONFIG.eps_prob):
        return np.zeros_like(num, dtype=float)
    return num / den


def _compute_CC(JC: np.ndarray, w: TwoWorldMarginals) -> np.ndarray:
    _, _, jbest = singleton_gaps(w)
    return _safe_divide(np.asarray(JC, dtype=float), float(jbest))


def _compute_optional_stats(
    pA: float,
    pB: float,
    p11_arr: np.ndarray,
    *,
    want_phi: bool,
    want_tau: bool,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    phi_arr = None
    tau_arr = None
    if want_phi:
        # vectorized phi
        denom = pA * (1.0 - pA) * pB * (1.0 - pB)
        if denom <= 0.0:
            phi_arr = np.full_like(p11_arr, np.nan, dtype=float)
        else:
            phi_arr = (p11_arr - pA * pB) / math.sqrt(denom)

    if want_tau:
        # tau_a from cells: 2(p00*p11 - p01*p10)
        # build cells analytically for speed (still feasible due to FH validation in generators)
        p11 = p11_arr
        p10 = pA - p11
        p01 = pB - p11
        p00 = 1.0 - pA - pB + p11
        tau_arr = 2.0 * (p00 * p11 - p01 * p10)

    return phi_arr, tau_arr


# =============================================================================
# FH-path scans (lambda parameterization)
# =============================================================================


def scan_fh_path_tied_lambda(
    w: TwoWorldMarginals,
    rule: Any,
    *,
    path: str = "fh_linear",
    grid: Optional[Grid1D] = None,
    path_params: Optional[Mapping[str, Any]] = None,
    want_phi: bool = True,
    want_tau: bool = True,
) -> CurveResult:
    """
    Scan a Fréchet–Hoeffding envelope path with a shared λ across worlds (λ0=λ1=λ).

    This is an *assumption*: that the dependence rank-position in the FH envelope
    is stable across worlds.

    Parameters
    ----------
    w:
        Two-world marginals.
    rule:
        Composition rule ("AND", "OR", "COND_OR").
    path:
        One of: "fh_linear", "fh_power", "fh_scurve" (see theory_core.p11_from_lambda)
    grid:
        Grid1D over λ; defaults to default_lambda_grid(201).
    path_params:
        Optional parameters for the FH path:
          - fh_power:  {"power": >0}
          - fh_scurve: {"alpha": >0}
    want_phi, want_tau:
        Compute interpretability stats for each world.

    Returns
    -------
    CurveResult
    """
    r = _require_rule(rule)
    if grid is None:
        grid = default_lambda_grid()

    lam = grid.values
    # Generate p11 arrays via scalar path mapping (robust > micro-optimized)
    p11_0 = np.array(
        [p11_from_lambda(path, float(l), w.pA0, w.pB0, path_params) for l in lam], dtype=float
    )
    p11_1 = np.array(
        [p11_from_lambda(path, float(l), w.pA1, w.pB1, path_params) for l in lam], dtype=float
    )

    # Compose
    if r == "COND_OR":
        # COND_OR ignores p11; still fill p11 arrays for interpretability, but pC is constant.
        pC0 = np.full_like(lam, w.pA0 + (1.0 - w.pA0) * w.pB0, dtype=float)
        pC1 = np.full_like(lam, w.pA1 + (1.0 - w.pA1) * w.pB1, dtype=float)
    else:
        pC0 = np.array([composed_rate(r, w.pA0, w.pB0, float(x)) for x in p11_0], dtype=float)
        pC1 = np.array([composed_rate(r, w.pA1, w.pB1, float(x)) for x in p11_1], dtype=float)

    JC = np.abs(pC1 - pC0)
    CC = _compute_CC(JC, w)

    phi0, tau0 = _compute_optional_stats(w.pA0, w.pB0, p11_0, want_phi=want_phi, want_tau=want_tau)
    phi1, tau1 = _compute_optional_stats(w.pA1, w.pB1, p11_1, want_phi=want_phi, want_tau=want_tau)

    return CurveResult(
        family="fh_path",
        rule=r,
        grid=grid,
        p11_0=p11_0,
        p11_1=p11_1,
        pC0=pC0,
        pC1=pC1,
        JC=JC,
        CC=CC,
        phi0=phi0,
        phi1=phi1,
        tau0=tau0,
        tau1=tau1,
    )


def scan_fh_path_surface(
    w: TwoWorldMarginals,
    rule: Any,
    *,
    path: str = "fh_linear",
    grid0: Optional[Grid1D] = None,
    grid1: Optional[Grid1D] = None,
    path_params: Optional[Mapping[str, Any]] = None,
) -> SurfaceResult:
    """
    2D surface scan of FH paths allowing λ0 and λ1 to vary independently.

    This is the "no hidden stability assumption" version: dependence can change
    across worlds arbitrarily within the chosen FH path family.

    Output:
      JC[i,j] = |pC(world1; λ1_j) - pC(world0; λ0_i)|
    """
    r = _require_rule(rule)
    if grid0 is None:
        grid0 = default_lambda_grid()
    if grid1 is None:
        grid1 = default_lambda_grid()

    lam0 = grid0.values
    lam1 = grid1.values

    p11_0 = np.array(
        [p11_from_lambda(path, float(l), w.pA0, w.pB0, path_params) for l in lam0], dtype=float
    )
    p11_1 = np.array(
        [p11_from_lambda(path, float(l), w.pA1, w.pB1, path_params) for l in lam1], dtype=float
    )

    if r == "COND_OR":
        pC0 = np.full_like(lam0, w.pA0 + (1.0 - w.pA0) * w.pB0, dtype=float)
        pC1 = np.full_like(lam1, w.pA1 + (1.0 - w.pA1) * w.pB1, dtype=float)
    else:
        pC0 = np.array([composed_rate(r, w.pA0, w.pB0, float(x)) for x in p11_0], dtype=float)
        pC1 = np.array([composed_rate(r, w.pA1, w.pB1, float(x)) for x in p11_1], dtype=float)

    # Broadcast to surface
    JC = np.abs(pC1.reshape(1, -1) - pC0.reshape(-1, 1))
    CC = _compute_CC(JC, w)

    return SurfaceResult(
        family="fh_path",
        rule=r,
        grid0=grid0,
        grid1=grid1,
        JC=JC,
        CC=CC,
    )


# =============================================================================
# Copula scans (tied dependence parameters)
# =============================================================================


def scan_clayton_theta_tied(
    w: TwoWorldMarginals,
    rule: Any,
    *,
    grid: Optional[Grid1D] = None,
    want_phi: bool = True,
    want_tau: bool = True,
) -> CurveResult:
    """
    Clayton copula scan with shared θ across worlds.

    θ=0 corresponds to independence in this family.

    This is *not* guaranteed to span FH extremes, but provides a structured,
    smooth dependence sensitivity curve.
    """
    r = _require_rule(rule)
    if grid is None:
        grid = default_theta_grid()

    th = grid.values
    p11_0 = np.array([p11_clayton_copula(w.pA0, w.pB0, float(t)) for t in th], dtype=float)
    p11_1 = np.array([p11_clayton_copula(w.pA1, w.pB1, float(t)) for t in th], dtype=float)

    if r == "COND_OR":
        pC0 = np.full_like(th, w.pA0 + (1.0 - w.pA0) * w.pB0, dtype=float)
        pC1 = np.full_like(th, w.pA1 + (1.0 - w.pA1) * w.pB1, dtype=float)
    else:
        pC0 = np.array([composed_rate(r, w.pA0, w.pB0, float(x)) for x in p11_0], dtype=float)
        pC1 = np.array([composed_rate(r, w.pA1, w.pB1, float(x)) for x in p11_1], dtype=float)

    JC = np.abs(pC1 - pC0)
    CC = _compute_CC(JC, w)

    phi0, tau0 = _compute_optional_stats(w.pA0, w.pB0, p11_0, want_phi=want_phi, want_tau=want_tau)
    phi1, tau1 = _compute_optional_stats(w.pA1, w.pB1, p11_1, want_phi=want_phi, want_tau=want_tau)

    return CurveResult(
        family="clayton",
        rule=r,
        grid=grid,
        p11_0=p11_0,
        p11_1=p11_1,
        pC0=pC0,
        pC1=pC1,
        JC=JC,
        CC=CC,
        phi0=phi0,
        phi1=phi1,
        tau0=tau0,
        tau1=tau1,
    )


def scan_gaussian_rho_tied(
    w: TwoWorldMarginals,
    rule: Any,
    *,
    grid: Optional[Grid1D] = None,
    method: str = "mc",
    n_mc: int = 100_000,
    seed: Optional[int] = 0,
    antithetic: bool = True,
    want_phi: bool = True,
    want_tau: bool = True,
) -> CurveResult:
    """
    Gaussian copula scan with shared ρ across worlds.

    method:
      - "scipy" (exact-ish) if SciPy installed
      - "mc" Monte Carlo, deterministic with seed

    NOTE: MC scanning can be expensive. Use fewer grid points or SciPy.
    """
    r = _require_rule(rule)
    if grid is None:
        grid = default_rho_grid()

    rho = grid.values
    # Use distinct seeds per rho to reduce accidental correlation in the estimate series
    base_seed = None if seed is None else int(seed)

    p11_0 = np.empty_like(rho, dtype=float)
    p11_1 = np.empty_like(rho, dtype=float)
    for i, rr in enumerate(rho):
        sd = None if base_seed is None else (base_seed + 7919 * i)
        p11_0[i] = p11_gaussian_copula(
            w.pA0, w.pB0, float(rr), method=method, n_mc=n_mc, seed=sd, antithetic=antithetic
        )
        p11_1[i] = p11_gaussian_copula(
            w.pA1, w.pB1, float(rr), method=method, n_mc=n_mc, seed=sd, antithetic=antithetic
        )

    if r == "COND_OR":
        pC0 = np.full_like(rho, w.pA0 + (1.0 - w.pA0) * w.pB0, dtype=float)
        pC1 = np.full_like(rho, w.pA1 + (1.0 - w.pA1) * w.pB1, dtype=float)
    else:
        pC0 = np.array([composed_rate(r, w.pA0, w.pB0, float(x)) for x in p11_0], dtype=float)
        pC1 = np.array([composed_rate(r, w.pA1, w.pB1, float(x)) for x in p11_1], dtype=float)

    JC = np.abs(pC1 - pC0)
    CC = _compute_CC(JC, w)

    phi0, tau0 = _compute_optional_stats(w.pA0, w.pB0, p11_0, want_phi=want_phi, want_tau=want_tau)
    phi1, tau1 = _compute_optional_stats(w.pA1, w.pB1, p11_1, want_phi=want_phi, want_tau=want_tau)

    return CurveResult(
        family="gaussian",
        rule=r,
        grid=grid,
        p11_0=p11_0,
        p11_1=p11_1,
        pC0=pC0,
        pC1=pC1,
        JC=JC,
        CC=CC,
        phi0=phi0,
        phi1=phi1,
        tau0=tau0,
        tau1=tau1,
    )


# =============================================================================
# Bounds comparison and reporting
# =============================================================================


def bounds_summary(
    w: TwoWorldMarginals,
    rule: Any,
    curve: CurveResult,
) -> BoundsSummary:
    """
    Compare FH dependence-agnostic bounds to what a scan actually realizes.

    This prevents the classic error:
      "I scanned a dependence family and called it worst-case."
    when the family fails to span FH extremes.

    Returns a BoundsSummary with explicit undercoverage metrics.
    """
    r = _require_rule(rule)
    if curve.rule != r:
        raise InputValidationError("curve.rule must match the requested rule")
    if curve.family not in ("fh_path", "clayton", "gaussian"):
        raise InputValidationError(f"Unknown curve family {curve.family}")

    JC_FH_min, JC_FH_max = jc_bounds(w, r)
    CC_FH_min, CC_FH_max = cc_bounds(w, r)

    JC_scan_min = curve.JC_min
    JC_scan_max = curve.JC_max
    CC_scan_min = curve.CC_min
    CC_scan_max = curve.CC_max

    FH_range = JC_FH_max - JC_FH_min
    scan_range = JC_scan_max - JC_scan_min

    JC_under_low = max(0.0, JC_FH_min - JC_scan_min)
    # Typically scan_max <= FH_max, but keep both directions to catch bugs.
    JC_under_high = max(0.0, JC_scan_max - JC_FH_max)

    JC_missing_extremes = max(0.0, FH_range - scan_range)
    JC_range_ratio = (
        float("nan") if abs(FH_range) <= float(CONFIG.eps_prob) else (scan_range / FH_range)
    )

    FH_range_CC = CC_FH_max - CC_FH_min
    scan_range_CC = CC_scan_max - CC_scan_min
    CC_missing_extremes = max(0.0, FH_range_CC - scan_range_CC)
    CC_range_ratio = (
        float("nan")
        if abs(FH_range_CC) <= float(CONFIG.eps_prob)
        else (scan_range_CC / FH_range_CC)
    )

    return BoundsSummary(
        rule=r,
        JC_FH_min=float(JC_FH_min),
        JC_FH_max=float(JC_FH_max),
        CC_FH_min=float(CC_FH_min),
        CC_FH_max=float(CC_FH_max),
        JC_scan_min=float(JC_scan_min),
        JC_scan_max=float(JC_scan_max),
        CC_scan_min=float(CC_scan_min),
        CC_scan_max=float(CC_scan_max),
        JC_undercoverage_low=float(JC_under_low),
        JC_undercoverage_high=float(JC_under_high),
        JC_missing_extremes=float(JC_missing_extremes),
        JC_range_ratio=float(JC_range_ratio),
        CC_missing_extremes=float(CC_missing_extremes),
        CC_range_ratio=float(CC_range_ratio),
        family=curve.family,
        grid_name=curve.grid.name,
    )


def format_bounds_report(
    w: TwoWorldMarginals,
    summary: BoundsSummary,
    *,
    digits: int = 6,
) -> str:
    """
    Human-readable report suitable for logs / papers / your own sanity.

    It is intentionally explicit and slightly brutal.
    """
    d = int(digits)
    jA, jB, jbest = singleton_gaps(w)
    lines = []
    lines.append("=== Theory Analysis Report (Dependence Coverage) ===")
    lines.append(f"Rule: {summary.rule}")
    lines.append("")
    lines.append("Two-world marginals:")
    lines.append(f"  World0: pA0={w.pA0:.{d}f}, pB0={w.pB0:.{d}f}")
    lines.append(f"  World1: pA1={w.pA1:.{d}f}, pB1={w.pB1:.{d}f}")
    lines.append("")
    lines.append("Singleton gaps:")
    lines.append(f"  J_A   = |pA1-pA0| = {jA:.{d}f}")
    lines.append(f"  J_B   = |pB1-pB0| = {jB:.{d}f}")
    lines.append(f"  J_best= max(J_A,J_B) = {jbest:.{d}f}")
    lines.append("")
    lines.append("Dependence-agnostic (FH) bounds:")
    lines.append(f"  JC ∈ [{summary.JC_FH_min:.{d}f}, {summary.JC_FH_max:.{d}f}]")
    lines.append(f"  CC ∈ [{summary.CC_FH_min:.{d}f}, {summary.CC_FH_max:.{d}f}]")
    lines.append("")
    lines.append(f"Scan realized bounds (family={summary.family}, grid={summary.grid_name}):")
    lines.append(f"  JC ∈ [{summary.JC_scan_min:.{d}f}, {summary.JC_scan_max:.{d}f}]")
    lines.append(f"  CC ∈ [{summary.CC_scan_min:.{d}f}, {summary.CC_scan_max:.{d}f}]")
    lines.append("")
    # Coverage diagnostics
    lines.append("Coverage diagnostics (if nonzero, your scan is NOT worst-case):")
    lines.append(f"  JC undercoverage (low end):  {summary.JC_undercoverage_low:.{d}f}")
    lines.append(f"  JC undercoverage (high end): {summary.JC_undercoverage_high:.{d}f}")
    lines.append(f"  JC missing extremes (range): {summary.JC_missing_extremes:.{d}f}")
    lines.append(
        f"  JC range ratio (scan/FH):    {summary.JC_range_ratio:.{d}f}"
        if not math.isnan(summary.JC_range_ratio)
        else "  JC range ratio (scan/FH):    NaN (FH range ~0)"
    )
    lines.append(f"  CC missing extremes (range): {summary.CC_missing_extremes:.{d}f}")
    lines.append(
        f"  CC range ratio (scan/FH):    {summary.CC_range_ratio:.{d}f}"
        if not math.isnan(summary.CC_range_ratio)
        else "  CC range ratio (scan/FH):    NaN (FH range ~0)"
    )
    lines.append("")
    if summary.JC_missing_extremes > 0:
        lines.append("Verdict:")
        lines.append("  Your dependence family/path did NOT span the FH worst-case JC range.")
        lines.append(
            "  Any claim of robustness must be labeled as 'conditional on dependence assumption'."
        )
    else:
        lines.append("Verdict:")
        lines.append("  Your scan covered the FH JC range (at least at the grid resolution used).")
        lines.append("  Still: verify at higher resolution if this matters for a proof/claim.")
    return "\n".join(lines)


# =============================================================================
# Calibration / inversion helpers (FH paths)
# =============================================================================


def find_lambda_for_target_p11(
    path: str,
    target_p11: float,
    pA: float,
    pB: float,
    *,
    path_params: Optional[Mapping[str, Any]] = None,
) -> float:
    """
    Invert p11_from_lambda for FH paths to find λ producing a desired p11.

    This is only guaranteed well-behaved for monotone FH paths:
      - fh_linear: monotone
      - fh_power:  monotone
      - fh_scurve: monotone

    Returns λ in [0,1]. If target is outside FH, raises.
    If target equals a boundary, returns 0 or 1.
    """
    pA = _require_prob(pA, "pA")
    pB = _require_prob(pB, "pB")
    target_p11 = _require_prob(target_p11, "target_p11")
    target_p11 = validate_joint(pA, pB, target_p11)

    L, U = fh_bounds(pA, pB)
    W = U - L
    eps = float(CONFIG.eps_prob)
    if W <= eps:
        # degenerate: any lambda maps to same p11
        return 0.0

    # Normalize within envelope:
    t = (target_p11 - L) / W
    t = min(max(t, 0.0), 1.0)

    name = path.strip().lower()
    if name == "fh_linear":
        return float(t)

    if name == "fh_power":
        power = float((path_params or {}).get("power", 1.0))
        if not math.isfinite(power) or power <= 0.0:
            raise InputValidationError("fh_power inversion requires power>0")
        return float(t ** (1.0 / power))

    if name == "fh_scurve":
        # Use bisection on lambda since the normalization is not analytically invertible in a clean way
        def f(lam: float) -> float:
            return p11_from_lambda("fh_scurve", lam, pA, pB, path_params)

        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            val = f(mid)
            if val < target_p11:
                lo = mid
            else:
                hi = mid
        return float(0.5 * (lo + hi))

    raise InputValidationError(f"Unsupported path for inversion: {path!r}")


def find_lambda_for_target_phi(
    path: str,
    target_phi: float,
    pA: float,
    pB: float,
    *,
    path_params: Optional[Mapping[str, Any]] = None,
    grid_n: int = 2001,
) -> float:
    """
    Find λ that makes φ approximately match target_phi for a given world.

    WARNING:
      φ(λ) is often monotone for FH paths when margins are interior, but not a theorem
      you should bet your life on. So we do a dense grid search + local refine.

    Returns λ in [0,1] that minimizes |phi(λ)-target_phi|.
    """
    if not math.isfinite(float(target_phi)):
        raise InputValidationError("target_phi must be finite")

    pA = _require_prob(pA, "pA")
    pB = _require_prob(pB, "pB")
    lam = np.linspace(0.0, 1.0, int(grid_n), dtype=float)

    p11 = np.array([p11_from_lambda(path, float(l), pA, pB, path_params) for l in lam], dtype=float)
    phi = np.array([phi_from_joint(pA, pB, float(x)) for x in p11], dtype=float)

    idx = int(np.nanargmin(np.abs(phi - float(target_phi))))
    return float(lam[idx])


def find_lambda_for_extreme_JC(
    w: TwoWorldMarginals,
    rule: Any,
    *,
    path: str = "fh_linear",
    path_params: Optional[Mapping[str, Any]] = None,
    grid_n: int = 2001,
    extreme: Literal["min", "max"] = "max",
) -> Tuple[float, float]:
    """
    Find λ (tied across worlds) that yields extreme JC under an FH path.

    Uses a dense grid (grid_n) and returns:
      (lambda_star, JC(lambda_star))

    No promise of global optimality beyond grid resolution.
    If you need proof-level extremality, you must analyze the piecewise-linear
    structure for AND/OR explicitly and/or increase resolution.
    """
    r = _require_rule(rule)
    lam_grid = Grid1D(values=np.linspace(0.0, 1.0, int(grid_n), dtype=float), name="lambda")
    curve = scan_fh_path_tied_lambda(
        w, r, path=path, grid=lam_grid, path_params=path_params, want_phi=False, want_tau=False
    )

    if extreme == "max":
        i = curve.argmax_JC()
    elif extreme == "min":
        i = curve.argmin_JC()
    else:
        raise InputValidationError("extreme must be 'min' or 'max'")

    return float(curve.grid.values[i]), float(curve.JC[i])


# =============================================================================
# Optional: "single world" sanity utilities
# =============================================================================


def world_interval_summary(pA: float, pB: float, rule: Any) -> Dict[str, float]:
    """
    Provide a compact summary for one world:
      - FH bounds for p11
      - implied bounds for pC under rule
    """
    r = _require_rule(rule)
    pA = _require_prob(pA, "pA")
    pB = _require_prob(pB, "pB")
    lo11, hi11 = fh_bounds(pA, pB)
    loC, hiC = composed_rate_bounds(r, pA, pB)
    return {
        "pA": float(pA),
        "pB": float(pB),
        "p11_lo": float(lo11),
        "p11_hi": float(hi11),
        "pC_lo": float(loC),
        "pC_hi": float(hiC),
        "p11_width": float(hi11 - lo11),
        "pC_width": float(hiC - loC),
    }


# =============================================================================
# Export surface
# =============================================================================

__all__ = [
    # grids
    "Grid1D",
    "default_lambda_grid",
    "default_rho_grid",
    "default_theta_grid",
    # results
    "CurveResult",
    "SurfaceResult",
    "BoundsSummary",
    # scans
    "scan_fh_path_tied_lambda",
    "scan_fh_path_surface",
    "scan_clayton_theta_tied",
    "scan_gaussian_rho_tied",
    # bounds/report
    "bounds_summary",
    "format_bounds_report",
    # inversion/calibration
    "find_lambda_for_target_p11",
    "find_lambda_for_target_phi",
    "find_lambda_for_extreme_JC",
    # sanity
    "world_interval_summary",
]

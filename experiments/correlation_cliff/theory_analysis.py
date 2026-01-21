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
Interpretability note (non-negotiable)
-------------------------------------------------------------------------------
- FH bounds are *axiomatic* given only marginals.
- Any path/copula family is an *assumption* that chooses a subset of feasible joints.
- Analysis MUST report whether a claim depends on:
    (a) FH worst-case, or
    (b) assumed dependence family/path.

This module encodes that distinction in result structures and reporting.

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
    composed_rate,  # fallback for unknown future rules
    composed_rate_bounds,
    fh_bounds,
    jc_bounds,
    p11_clayton_copula,
    p11_from_lambda,
    p11_gaussian_copula,
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
    Comparison between FH worst-case bounds and realized scan extrema.

    Prevents the classic failure mode:
      "I scanned an assumption family and called it worst-case."
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
    JC_undercoverage_low: float
    JC_undercoverage_high: float
    JC_missing_extremes: float
    JC_range_ratio: float  # scan_range / FH_range (NaN if FH_range==0)

    CC_missing_extremes: float
    CC_range_ratio: float

    family: DependenceFamily
    grid_name: str


# =============================================================================
# Small utilities (strict + fast)
# =============================================================================


def default_lambda_grid(n: int = 201) -> Grid1D:
    """Default λ grid for FH scans."""
    if not isinstance(n, int) or n < 2:
        raise InputValidationError("n must be an int >= 2")
    return Grid1D(values=np.linspace(0.0, 1.0, n, dtype=float), name="lambda")


def default_rho_grid(n: int = 201) -> Grid1D:
    """Default ρ grid for Gaussian copula scans."""
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
    Large θ increases positive dependence in this family, but does NOT guarantee FH extremes.
    """
    theta_max_f = float(theta_max)
    if not math.isfinite(theta_max_f) or theta_max_f <= 0.0:
        raise InputValidationError("theta_max must be finite and > 0")
    if not isinstance(n, int) or n < 2:
        raise InputValidationError("n must be an int >= 2")

    if include_zero:
        vals = np.linspace(0.0, theta_max_f, n, dtype=float)
    else:
        vals = np.linspace(theta_max_f / (n + 1), theta_max_f, n, dtype=float)

    vals = np.unique(vals)  # enforces strictly increasing after floating artifacts
    if vals.size < 2:
        raise InputValidationError("theta grid collapsed; choose larger n or different theta_max")
    return Grid1D(values=vals, name="theta")


def _safe_divide(num: np.ndarray, den: float) -> np.ndarray:
    den_f = float(den)
    if abs(den_f) <= float(CONFIG.eps_prob):
        return np.zeros_like(num, dtype=float)
    return np.asarray(num, dtype=float) / den_f


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
    """
    Compute interpretability stats in a vectorized way.

    φ (phi coefficient):
      φ = (p11 - pA*pB) / sqrt(pA(1-pA)pB(1-pB))

    τ_a (Kendall tau-a for 2x2):
      τ_a = 2(p00*p11 - p01*p10)
    """
    p11 = np.asarray(p11_arr, dtype=float)

    phi_arr: Optional[np.ndarray] = None
    tau_arr: Optional[np.ndarray] = None

    if want_phi:
        denom = pA * (1.0 - pA) * pB * (1.0 - pB)
        if denom <= 0.0:
            phi_arr = np.full_like(p11, np.nan, dtype=float)
        else:
            phi_arr = (p11 - pA * pB) / math.sqrt(denom)

    if want_tau:
        p10 = pA - p11
        p01 = pB - p11
        p00 = 1.0 - pA - pB + p11
        tau_arr = 2.0 * (p00 * p11 - p01 * p10)

    return phi_arr, tau_arr


def _compose_rate_vec(rule: Rule, pA: float, pB: float, p11: np.ndarray) -> np.ndarray:
    """
    Fast vectorized composition for the known rule set.

    Falls back to scalar composed_rate if a new rule is introduced.
    """
    p11_arr = np.asarray(p11, dtype=float)

    if rule == "AND":
        return p11_arr
    if rule == "OR":
        return (pA + pB) - p11_arr
    if rule == "COND_OR":
        # independent conditional OR form used in theory_core
        return np.full_like(p11_arr, pA + (1.0 - pA) * pB, dtype=float)

    # Future-proof fallback (rare path)
    return np.fromiter(
        (composed_rate(rule, pA, pB, float(x)) for x in p11_arr), dtype=float, count=p11_arr.size
    )


def _p11_vec_fh_path(
    path: str,
    lam_grid: np.ndarray,
    pA: float,
    pB: float,
    path_params: Optional[Mapping[str, Any]],
) -> np.ndarray:
    """
    Fast scalar->vector wrapper around theory_core.p11_from_lambda.

    Uses np.fromiter to keep overhead low and determinism high.
    """
    params = {} if path_params is None else dict(path_params)
    lam_arr = np.asarray(lam_grid, dtype=float)
    return np.fromiter(
        (p11_from_lambda(path, float(lam_val), pA, pB, params) for lam_val in lam_arr),
        dtype=float,
        count=lam_arr.size,
    )


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
    """
    r = _require_rule(rule)
    if grid is None:
        grid = default_lambda_grid()

    lam_grid = grid.values

    # p11 arrays (robust > micro-optimized), but implemented with low-overhead fromiter
    p11_0 = _p11_vec_fh_path(path, lam_grid, w.pA0, w.pB0, path_params)
    p11_1 = _p11_vec_fh_path(path, lam_grid, w.pA1, w.pB1, path_params)

    # Compose (vectorized for known rules)
    pC0 = _compose_rate_vec(r, w.pA0, w.pB0, p11_0)
    pC1 = _compose_rate_vec(r, w.pA1, w.pB1, p11_1)

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

    Output:
      JC[i,j] = |pC(world1; λ1_j) - pC(world0; λ0_i)|
    """
    r = _require_rule(rule)
    if grid0 is None:
        grid0 = default_lambda_grid()
    if grid1 is None:
        grid1 = default_lambda_grid()

    lam0_grid = grid0.values
    lam1_grid = grid1.values

    p11_0 = _p11_vec_fh_path(path, lam0_grid, w.pA0, w.pB0, path_params)
    p11_1 = _p11_vec_fh_path(path, lam1_grid, w.pA1, w.pB1, path_params)

    pC0 = _compose_rate_vec(r, w.pA0, w.pB0, p11_0)
    pC1 = _compose_rate_vec(r, w.pA1, w.pB1, p11_1)

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

    This is *not* guaranteed to span FH extremes, but provides a smooth dependence curve.
    """
    r = _require_rule(rule)
    if grid is None:
        grid = default_theta_grid()

    theta_grid = grid.values

    # fromiter is faster than Python lists then np.array for large grids
    p11_0 = np.fromiter(
        (p11_clayton_copula(w.pA0, w.pB0, float(theta_val)) for theta_val in theta_grid),
        dtype=float,
        count=theta_grid.size,
    )
    p11_1 = np.fromiter(
        (p11_clayton_copula(w.pA1, w.pB1, float(theta_val)) for theta_val in theta_grid),
        dtype=float,
        count=theta_grid.size,
    )

    pC0 = _compose_rate_vec(r, w.pA0, w.pB0, p11_0)
    pC1 = _compose_rate_vec(r, w.pA1, w.pB1, p11_1)

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

    NOTE: MC scanning can be expensive. Reduce grid size, reduce n_mc, or use SciPy.
    """
    r = _require_rule(rule)
    if grid is None:
        grid = default_rho_grid()

    rho_grid = grid.values
    base_seed = None if seed is None else int(seed)

    p11_0 = np.empty_like(rho_grid, dtype=float)
    p11_1 = np.empty_like(rho_grid, dtype=float)

    # Use distinct seeds per rho to reduce accidental correlation across the estimate series
    for i, rho_val in enumerate(rho_grid):
        sd = None if base_seed is None else (base_seed + 7919 * i)
        p11_0[i] = p11_gaussian_copula(
            w.pA0,
            w.pB0,
            float(rho_val),
            method=method,
            n_mc=n_mc,
            seed=sd,
            antithetic=antithetic,
        )
        p11_1[i] = p11_gaussian_copula(
            w.pA1,
            w.pB1,
            float(rho_val),
            method=method,
            n_mc=n_mc,
            seed=sd,
            antithetic=antithetic,
        )

    pC0 = _compose_rate_vec(r, w.pA0, w.pB0, p11_0)
    pC1 = _compose_rate_vec(r, w.pA1, w.pB1, p11_1)

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
    Human-readable report suitable for logs / papers / sanity checks.

    It is intentionally explicit and slightly brutal.
    """
    d = int(digits)
    jA, jB, jbest = singleton_gaps(w)

    lines: list[str] = []
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

    if (
        summary.JC_missing_extremes > 0.0
        or summary.JC_undercoverage_low > 0.0
        or summary.JC_undercoverage_high > 0.0
    ):
        lines.append("Verdict:")
        lines.append("  Your dependence family/path did NOT span the FH worst-case JC range.")
        lines.append(
            "  Any claim of robustness must be labeled 'conditional on dependence assumption'."
        )
    else:
        lines.append("Verdict:")
        lines.append("  Your scan covered the FH JC range (at this grid resolution).")
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

    Guaranteed well-behaved for monotone FH paths:
      - fh_linear: monotone
      - fh_power:  monotone
      - fh_scurve: monotone

    Returns λ in [0,1]. If target is outside FH, raises.
    If target equals a boundary, returns 0 or 1.
    """
    pA_v = _require_prob(pA, "pA")
    pB_v = _require_prob(pB, "pB")
    target = _require_prob(target_p11, "target_p11")
    target = validate_joint(pA_v, pB_v, target)

    L, U = fh_bounds(pA_v, pB_v)
    width = U - L
    eps = float(CONFIG.eps_prob)
    if width <= eps:
        # degenerate: any lambda maps to same p11
        return 0.0

    # normalize within envelope:
    t = (target - L) / width
    t = min(max(t, 0.0), 1.0)

    name = path.strip().lower()
    params = {} if path_params is None else dict(path_params)

    if name == "fh_linear":
        return float(t)

    if name == "fh_power":
        power = float(params.get("power", 1.0))
        if not math.isfinite(power) or power <= 0.0:
            raise InputValidationError("fh_power inversion requires power>0")
        return float(t ** (1.0 / power))

    if name == "fh_scurve":
        # bisection on lambda (clean, robust, deterministic)
        def f(lam_val: float) -> float:
            return float(p11_from_lambda("fh_scurve", lam_val, pA_v, pB_v, params))

        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            val = f(mid)
            if val < target:
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
      φ(λ) is often monotone-ish for FH paths with interior margins,
      but this is not a theorem to rely on. We do dense grid search.

    Returns λ in [0,1] that minimizes |phi(λ)-target_phi|.
    """
    target_phi_f = float(target_phi)
    if not math.isfinite(target_phi_f):
        raise InputValidationError("target_phi must be finite")

    pA_v = _require_prob(pA, "pA")
    pB_v = _require_prob(pB, "pB")

    grid_n_i = int(grid_n)
    if grid_n_i < 2:
        raise InputValidationError("grid_n must be >= 2")

    lam_grid = np.linspace(0.0, 1.0, grid_n_i, dtype=float)
    p11 = _p11_vec_fh_path(path, lam_grid, pA_v, pB_v, path_params)

    phi_arr, _ = _compute_optional_stats(pA_v, pB_v, p11, want_phi=True, want_tau=False)
    assert phi_arr is not None  # by construction

    idx = int(np.nanargmin(np.abs(phi_arr - target_phi_f)))
    return float(lam_grid[idx])


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
    """
    r = _require_rule(rule)
    grid_n_i = int(grid_n)
    if grid_n_i < 2:
        raise InputValidationError("grid_n must be >= 2")

    lam_grid = Grid1D(values=np.linspace(0.0, 1.0, grid_n_i, dtype=float), name="lambda")
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
    pA_v = _require_prob(pA, "pA")
    pB_v = _require_prob(pB, "pB")
    lo11, hi11 = fh_bounds(pA_v, pB_v)
    loC, hiC = composed_rate_bounds(r, pA_v, pB_v)
    return {
        "pA": float(pA_v),
        "pB": float(pB_v),
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

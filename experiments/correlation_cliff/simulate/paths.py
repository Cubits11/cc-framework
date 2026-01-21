# experiments/correlation_cliff/simulate/paths.py
from __future__ import annotations

"""
simulate.paths
==============

PhD/enterprise-grade dependence-path primitives for Correlation Cliff experiments.

Goal
----
Construct a feasible joint probability p11 = P(A=1, B=1) given marginals pA, pB and a
dependence interpolation parameter lam ∈ [0,1], while guaranteeing Fréchet-Hoeffding
feasibility:

    L(pA,pB) <= p11 <= U(pA,pB)

This module provides a small family of paths that map lam to a point inside the FH
envelope, plus an optional Gaussian-copula path parameterized by Kendall's tau.

Design invariants (ResearchOS)
------------------------------
1) meta is NUMERIC-ONLY (all values are Python floats; keys are strings)
2) required meta keys ALWAYS present for ALL paths:
       L, U, FH_width, lam, lam_eff, raw_p11, clip_amt, clipped,
       fh_violation, fh_violation_amt
3) p11 returned ALWAYS satisfies L <= p11 <= U unless:
       clip_policy="raise" and a violation exceeds clip_tol -> FeasibilityError
4) bool-as-number is rejected (True/False are NOT accepted as numeric inputs)
5) deterministic behavior; no global state; no hidden RNG

Paths
-----
- fh_linear : linear interpolation inside FH envelope
- fh_power  : lam_eff = lam**gamma, then FH-linear in lam_eff
- fh_scurve : lam_eff = normalized logistic S-curve, then FH-linear in lam_eff
- gaussian_tau : lam -> tau in [-1,1] -> Gaussian copula joint; endpoints return FH bounds
                 WITHOUT SciPy; interior requires SciPy.

Notes
-----
- FH-family paths are routed through U.p11_fh_linear for testability and centralized theory.
- All numeric fields are float-cast at the boundary so meta remains strict.
"""

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from . import utils as U
from .config import Path

__all__ = [
    "FeasibilityError",
    "InputValidationError",
    "NumericalError",
    "P11FromPathError",
    "p11_from_path",
]


# =============================================================================
# Exceptions (structured + testable)
# =============================================================================


class P11FromPathError(ValueError):
    """Base error for p11 path construction."""


class InputValidationError(P11FromPathError):
    """Bad user inputs (pA/pB/lam/path/path_params)."""


class FeasibilityError(P11FromPathError):
    """Raw p11 violates FH by more than allowed tolerance (when clip_policy='raise')."""


class NumericalError(P11FromPathError):
    """Non-finite values, invalid FH bounds, or postcondition failure."""


# =============================================================================
# Strict parsing helpers (reject bool-as-number)
# =============================================================================


def _as_float(x: Any, name: str) -> float:
    """Coerce to Python float but reject bool explicitly (ResearchOS hygiene)."""
    if isinstance(x, bool):
        raise InputValidationError(f"{name} must be a float-like real number, got bool {x!r}")
    try:
        xf = float(x)
    except Exception as exc:  # pragma: no cover
        raise InputValidationError(f"{name} must be coercible to float, got {x!r}") from exc
    # Ensure Python float (not numpy scalar), and no -0.0 weirdness in audit
    xf = float(xf)
    if not math.isfinite(xf):
        raise InputValidationError(f"{name} must be finite, got {x!r}")
    return xf


def _finite_in_unit(x: Any, name: str) -> float:
    """Require finite float in [0,1]."""
    xf = _as_float(x, name)
    if not (0.0 <= xf <= 1.0):
        raise InputValidationError(f"{name} must be in [0,1], got {x!r}")
    return float(xf)


def _finite_pos(x: Any, name: str) -> float:
    """Require finite strictly-positive float."""
    xf = _as_float(x, name)
    if not (xf > 0.0):
        raise InputValidationError(f"{name} must be > 0, got {x!r}")
    return float(xf)


def _mapping(path_params: Any) -> Mapping[str, Any]:
    """Require mapping with string keys (OS hygiene)."""
    if isinstance(path_params, Mapping):
        for k in path_params:
            if not isinstance(k, str):
                raise InputValidationError(
                    f"path_params keys must be str; got key={k!r} ({type(k).__name__})"
                )
        return path_params
    raise InputValidationError(
        f"path_params must be a mapping/dict, got {type(path_params).__name__}"
    )


def _normalize_path(path: Path | str) -> str:
    """
    Normalize `path` into a canonical lowercase identifier.

    This fixes a common failure mode: Enum stringification can look like "Path.FH_LINEAR".
    We accept:
      - Enum Path values or names
      - strings like "fh_linear", "FH_LINEAR", "Path.FH_LINEAR", "path.fh_linear"
    """
    if isinstance(path, str):
        s = path
    else:
        # Prefer `.value` if it is a string-like; else fall back to `.name`.
        val = getattr(path, "value", None)
        s = val if isinstance(val, str) and val.strip() else getattr(path, "name", str(path))

    s = s.strip().lower()

    # Strip Enum-ish prefixes
    if s.startswith("path."):
        s = s.split(".", 1)[1].strip()
    # Some reprs can be "Path.FH_LINEAR"
    if s.startswith("path."):
        s = s.split(".", 1)[1].strip()

    # Accept "fh-linear" or "fh linear" style too
    s = s.replace("-", "_").replace(" ", "_")

    return s


# =============================================================================
# Lambda transforms (pure; return clipped to [0,1])
# =============================================================================


def _clip01(x: float) -> float:
    return float(min(max(float(x), 0.0), 1.0))


def _lam_power(lam: float, gamma: float) -> float:
    lam = _finite_in_unit(lam, "lam")
    gamma = _finite_pos(gamma, "gamma")
    if lam in (0.0, 1.0) or gamma == 1.0:
        return float(lam)
    return _clip01(lam**gamma)


def _lam_scurve(lam: float, k: float) -> float:
    lam = _finite_in_unit(lam, "lam")
    k = _finite_pos(k, "k")

    # Numerically stable sigmoid
    def _sigmoid(z: float) -> float:
        z = float(z)
        if z >= 0.0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    if lam <= 0.0:
        return 0.0
    if lam >= 1.0:
        return 1.0
    # If k is ~0, sigmoid degenerates to ~linear; keep identity for stability
    if k < 1e-8:
        return float(lam)

    # Normalize so lam=0 maps to 0 and lam=1 maps to 1 exactly
    a = _sigmoid(-0.5 * k)
    b = _sigmoid(+0.5 * k)
    denom = b - a
    if not (math.isfinite(denom) and denom > 0.0):
        return float(lam)

    s_raw = _sigmoid(k * (lam - 0.5))
    s = (s_raw - a) / denom
    return _clip01(s)


def _lam_to_tau(lam: float) -> float:
    """
    Canonical mapping for gaussian_tau path:
        lam ∈ [0,1] ↦ tau = 2*lam - 1 ∈ [-1, +1]
    """
    lam = _finite_in_unit(lam, "lam")
    return float(2.0 * lam - 1.0)


# =============================================================================
# Gaussian copula helpers (SciPy optional)
# =============================================================================


def _bvn_cdf_scipy(x: float, y: float, rho: float) -> float:
    """Bivariate Normal CDF via SciPy (optional dependency)."""
    x = _as_float(x, "x")
    y = _as_float(y, "y")
    rho = _as_float(rho, "rho")

    if not (-1.0 <= rho <= 1.0):
        raise InputValidationError(f"rho must be in [-1,1], got {rho!r}")

    try:
        from scipy.stats import multivariate_normal  # type: ignore
    except ImportError as exc:
        raise ImportError("SciPy not available for gaussian_tau path") from exc

    mean = np.array([0.0, 0.0], dtype=float)
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=float)

    val = float(multivariate_normal(mean=mean, cov=cov, allow_singular=True).cdf([x, y]))
    return _clip01(val)


def _p11_gaussian_tau(
    pA: float,
    pB: float,
    tau: float,
    *,
    ppf_clip_eps: float = 1e-12,
) -> tuple[float, dict[str, float]]:
    """
    Compute p11 under Gaussian copula parameterized by Kendall's tau.

    Implementation detail:
    - Convert tau -> rho via rho = sin(pi * tau / 2).
    - Use Normal PPF on clipped marginals to avoid ±inf.
    - Evaluate BVN CDF at (x,y).

    Returns:
      (p11, meta) where meta contains numeric parameters used.
    """
    pA = _finite_in_unit(pA, "pA")
    pB = _finite_in_unit(pB, "pB")
    tau = _as_float(tau, "tau")
    ppf_clip_eps = _as_float(ppf_clip_eps, "ppf_clip_eps")

    if not (-1.0 <= tau <= 1.0):
        raise InputValidationError(f"tau must be in [-1,1], got {tau!r}")
    if not (0.0 < ppf_clip_eps < 0.5):
        raise InputValidationError(f"ppf_clip_eps must be in (0, 0.5), got {ppf_clip_eps!r}")

    rho = float(math.sin(math.pi * tau / 2.0))

    try:
        from scipy.stats import norm  # type: ignore
    except ImportError as exc:
        raise ImportError("SciPy not available for gaussian_tau path") from exc

    pA_used = float(min(max(pA, ppf_clip_eps), 1.0 - ppf_clip_eps))
    pB_used = float(min(max(pB, ppf_clip_eps), 1.0 - ppf_clip_eps))

    x = float(norm.ppf(pA_used))
    y = float(norm.ppf(pB_used))

    p11 = _bvn_cdf_scipy(x, y, rho)

    meta = {
        "tau": float(tau),
        "rho": float(rho),
        "pA_used": float(pA_used),
        "pB_used": float(pB_used),
        "ppf_clip_eps": float(ppf_clip_eps),
    }
    return float(p11), meta


# =============================================================================
# Main API: p11_from_path
# =============================================================================


def p11_from_path(
    pA: float,
    pB: float,
    lam: float,
    *,
    path: Path | str,
    path_params: Mapping[str, Any],
) -> tuple[float, dict[str, float]]:
    """
    Enterprise-grade p11 constructor with explicit invariants and audit fields.

    Returns:
      (p11, meta) where meta is NUMERIC-ONLY by design.

    Required meta keys for ALL paths:
      - L, U, FH_width, lam, lam_eff
      - raw_p11, clip_amt, clipped
      - fh_violation, fh_violation_amt
    """
    pA_f = _finite_in_unit(pA, "pA")
    pB_f = _finite_in_unit(pB, "pB")
    lam_f = _finite_in_unit(lam, "lam")

    pp = _mapping(path_params)

    path_s = _normalize_path(path)
    if path_s not in ("fh_linear", "fh_power", "fh_scurve", "gaussian_tau"):
        raise InputValidationError(f"Unknown path: {path!r}")

    b = U.fh_bounds(pA_f, pB_f)
    L = float(b.L)
    Uu = float(b.U)
    if math.isfinite(L) and math.isfinite(Uu) and Uu < L and (L - Uu) <= 1e-12:
        L = Uu
    if not (math.isfinite(L) and math.isfinite(Uu) and 0.0 <= L <= Uu <= 1.0):
        raise NumericalError(f"FH bounds invalid for pA={pA_f}, pB={pB_f}: L={L}, U={Uu}")

    width = float(Uu - L)

    # Base meta: numeric-only, always present keys
    meta: dict[str, float] = {
        "L": float(L),
        "U": float(Uu),
        "FH_width": float(width),
        "lam": float(lam_f),
        "lam_eff": float("nan"),
        "raw_p11": float("nan"),
        "clip_amt": 0.0,
        "clipped": 0.0,
        "fh_violation": 0.0,
        "fh_violation_amt": 0.0,
    }

    clip_policy = str(pp.get("clip_policy", "clip")).lower().strip()
    if clip_policy not in ("clip", "raise"):
        raise InputValidationError(f"clip_policy must be 'clip' or 'raise', got {clip_policy!r}")

    clip_tol = _as_float(pp.get("clip_tol", 0.0), "clip_tol")
    # Tight by design: if you want bigger, do it explicitly and update tests.
    if not (0.0 <= clip_tol <= 1e-6):
        raise InputValidationError(
            f"clip_tol must be >=0 and small (<=1e-6) to preserve strict feasibility, got {clip_tol!r}"
        )

    def _enforce_numeric_only(d: dict[str, float]) -> None:
        for k, v in d.items():
            if not isinstance(k, str):
                raise NumericalError(f"meta keys must be str; got {k!r} ({type(k).__name__})")
            # Must be Python float, not numpy scalar
            if type(v) is not float:
                raise NumericalError(
                    f"meta values must be Python float; got {k!r}:{type(v).__name__}"
                )
            # Do not require finite for everything: but our required keys SHOULD be finite after finalize
            if not math.isfinite(v) and k not in ("lam_eff", "raw_p11"):
                raise NumericalError(f"meta contains non-finite value for {k!r}: {v!r}")

    def _finalize(raw: float, lam_eff: float) -> tuple[float, dict[str, float]]:
        raw_f = _as_float(raw, "raw_p11")
        lam_eff_f = _finite_in_unit(lam_eff, "lam_eff")  # stricter than before

        meta["lam_eff"] = float(lam_eff_f)
        meta["raw_p11"] = float(raw_f)

        # Violation magnitude beyond [L,U]
        below = max(0.0, (L - raw_f))
        above = max(0.0, (raw_f - Uu))
        viol_amt = float(max(below, above))

        meta["fh_violation_amt"] = float(viol_amt)
        meta["fh_violation"] = float(1.0 if viol_amt > 0.0 else 0.0)

        # Policy: raise if outside [L,U] beyond clip_tol
        if (raw_f < L - clip_tol or raw_f > Uu + clip_tol) and clip_policy == "raise":
            raise FeasibilityError(
                f"{path_s}: raw_p11 violates FH by more than clip_tol. "
                f"raw={raw_f}, L={L}, U={Uu}, clip_tol={clip_tol}"
            )

        clipped = raw_f
        if clipped < L:
            clipped = L
        elif clipped > Uu:
            clipped = Uu

        meta["clip_amt"] = float(
            raw_f - clipped
        )  # signed: positive means clipped downward; negative clipped upward
        meta["clipped"] = float(1.0 if clipped != raw_f else 0.0)

        if not (L <= clipped <= Uu):
            raise NumericalError(
                f"Postcondition failed: p11 not in [L,U]. p11={clipped}, L={L}, U={Uu}"
            )

        # Strong numeric-only enforcement
        _enforce_numeric_only(meta)

        return float(clipped), meta

    # ---- FH envelope paths (routed through U.p11_fh_linear for testability)
    if path_s == "fh_linear":
        lam_eff = float(lam_f)
        raw = float(U.p11_fh_linear(pA_f, pB_f, lam_eff))
        return _finalize(raw, lam_eff)

    if path_s == "fh_power":
        gamma = _finite_pos(pp.get("gamma", None), "gamma")
        lam_eff = _lam_power(lam_f, gamma)
        raw = float(U.p11_fh_linear(pA_f, pB_f, lam_eff))
        meta["gamma"] = float(gamma)
        return _finalize(raw, lam_eff)

    if path_s == "fh_scurve":
        k = _finite_pos(pp.get("k", None), "k")
        lam_eff = _lam_scurve(lam_f, k)
        raw = float(U.p11_fh_linear(pA_f, pB_f, lam_eff))
        meta["k"] = float(k)
        return _finalize(raw, lam_eff)

    # ---- Gaussian copula path (parameterized by Kendall's tau via lam)
    # Endpoints should not require SciPy: tau=-1 -> L, tau=+1 -> U.
    tau = _lam_to_tau(lam_f)
    meta["tau"] = float(tau)

    if tau <= -1.0:
        meta["rho"] = -1.0
        return _finalize(float(L), float(lam_f))

    if tau >= 1.0:
        meta["rho"] = 1.0
        return _finalize(float(Uu), float(lam_f))

    # Interior requires SciPy
    ppf_clip_eps = _as_float(pp.get("ppf_clip_eps", 1e-12), "ppf_clip_eps")
    p11_raw, gmeta = _p11_gaussian_tau(pA_f, pB_f, tau, ppf_clip_eps=ppf_clip_eps)

    for kk, vv in gmeta.items():
        meta[kk] = float(vv)

    return _finalize(float(p11_raw), float(lam_f))

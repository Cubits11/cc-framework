from __future__ import annotations

"""
simulate.paths
==============

Dependence-path primitives:
- fh_linear / fh_power / fh_scurve (FH envelope interpolation)
- gaussian_tau (optional SciPy)

Exports:
- p11_from_path(pA, pB, lam, path=..., path_params=...) -> (p11, meta)

Research-OS invariants:
- meta is NUMERIC-ONLY by design (floats)
- required meta keys ALWAYS present:
    L, U, FH_width, lam, lam_eff, raw_p11, clip_amt, clipped,
    fh_violation, fh_violation_amt
- p11 returned ALWAYS satisfies L <= p11 <= U
  (unless clip_policy='raise' and violation exceeds clip_tol, in which case raises)
"""

from typing import Any, Dict, Mapping, Tuple
import math
import numpy as np

from .config import Path
from . import utils as U


# -----------------------------------------------------------------------------
# Exceptions (structured + testable)
# -----------------------------------------------------------------------------
class P11FromPathError(ValueError):
    """Base error for p11 path construction."""


class InputValidationError(P11FromPathError):
    """Bad user inputs (pA/pB/lam/path/path_params)."""


class FeasibilityError(P11FromPathError):
    """Raw p11 violates FH by more than allowed tolerance (when clip_policy='raise')."""


class NumericalError(P11FromPathError):
    """Non-finite values, invalid FH bounds, or postcondition failure."""


# -----------------------------------------------------------------------------
# Strict parsing helpers (reject bool-as-number)
# -----------------------------------------------------------------------------
def _as_float(x: Any, name: str) -> float:
    if isinstance(x, bool):
        raise InputValidationError(f"{name} must be a float-like real number, got bool {x!r}")
    try:
        xf = float(x)
    except Exception as e:
        raise InputValidationError(f"{name} must be coercible to float, got {x!r}") from e
    return xf


def _finite_in_unit(x: Any, name: str) -> float:
    xf = _as_float(x, name)
    if not (math.isfinite(xf) and 0.0 <= xf <= 1.0):
        raise InputValidationError(f"{name} must be finite and in [0,1], got {x!r}")
    return xf


def _finite_pos(x: Any, name: str) -> float:
    xf = _as_float(x, name)
    if not (math.isfinite(xf) and xf > 0.0):
        raise InputValidationError(f"{name} must be finite and > 0, got {x!r}")
    return xf


def _mapping(path_params: Any) -> Mapping[str, Any]:
    if isinstance(path_params, Mapping):
        for k in path_params.keys():
            if not isinstance(k, str):
                raise InputValidationError(f"path_params keys must be str; got key={k!r} ({type(k).__name__})")
        return path_params
    raise InputValidationError(f"path_params must be a mapping/dict, got {type(path_params).__name__}")


# -----------------------------------------------------------------------------
# Helpers: lambda transforms (pure, clipped to [0,1])
# -----------------------------------------------------------------------------
def _lam_power(lam: float, gamma: float) -> float:
    lam = _finite_in_unit(lam, "lam")
    gamma = _finite_pos(gamma, "gamma")
    if lam == 0.0 or lam == 1.0 or gamma == 1.0:
        return float(lam)
    y = lam ** gamma
    return float(min(max(y, 0.0), 1.0))


def _lam_scurve(lam: float, k: float) -> float:
    lam = _finite_in_unit(lam, "lam")
    k = _finite_pos(k, "k")

    def _sigmoid(z: float) -> float:
        z = float(z)
        if z >= 0.0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    if lam == 0.0:
        return 0.0
    if lam == 1.0:
        return 1.0
    if k < 1e-8:
        return float(lam)

    a = _sigmoid(-0.5 * k)
    b = _sigmoid(+0.5 * k)
    denom = b - a
    if not (math.isfinite(denom) and denom > 0.0):
        return float(lam)

    s_raw = _sigmoid(k * (lam - 0.5))
    s = (s_raw - a) / denom
    return float(min(max(s, 0.0), 1.0))


def _lam_to_tau(lam: float) -> float:
    """
    Canonical mapping for gaussian_tau path.

    lam in [0,1] -> tau in [-1, +1] linearly:
        tau = 2*lam - 1
    Ensures endpoints map to tau=-1,+1 exactly.
    """
    lam = _finite_in_unit(lam, "lam")
    return float(2.0 * lam - 1.0)


# -----------------------------------------------------------------------------
# Gaussian copula helpers (SciPy optional)
# -----------------------------------------------------------------------------
def _bvn_cdf_scipy(x: float, y: float, rho: float) -> float:
    x = _as_float(x, "x")
    y = _as_float(y, "y")
    rho = _as_float(rho, "rho")

    if not (math.isfinite(x) and math.isfinite(y)):
        raise NumericalError(f"x and y must be finite, got x={x!r}, y={y!r}")
    if not (math.isfinite(rho) and -1.0 <= rho <= 1.0):
        raise InputValidationError(f"rho must be finite and in [-1,1], got {rho!r}")

    try:
        from scipy.stats import multivariate_normal  # type: ignore
    except ImportError as e:
        raise ImportError("SciPy not available for gaussian_tau path") from e

    mean = np.array([0.0, 0.0], dtype=float)
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=float)

    val = float(multivariate_normal(mean=mean, cov=cov, allow_singular=True).cdf([x, y]))
    return float(min(max(val, 0.0), 1.0))


def _p11_gaussian_tau(
    pA: float,
    pB: float,
    tau: float,
    *,
    ppf_clip_eps: float = 1e-12,
) -> Tuple[float, Dict[str, float]]:
    pA = _finite_in_unit(pA, "pA")
    pB = _finite_in_unit(pB, "pB")
    tau = _as_float(tau, "tau")
    ppf_clip_eps = _as_float(ppf_clip_eps, "ppf_clip_eps")

    if not (math.isfinite(tau) and -1.0 <= tau <= 1.0):
        raise InputValidationError(f"tau must be finite and in [-1,1], got {tau!r}")
    if not (math.isfinite(ppf_clip_eps) and 0.0 < ppf_clip_eps < 0.5):
        raise InputValidationError(f"ppf_clip_eps must be in (0, 0.5), got {ppf_clip_eps!r}")

    rho = math.sin(math.pi * tau / 2.0)

    try:
        from scipy.stats import norm  # type: ignore
    except ImportError as e:
        raise ImportError("SciPy not available for gaussian_tau path") from e

    pA_used = min(max(pA, ppf_clip_eps), 1.0 - ppf_clip_eps)
    pB_used = min(max(pB, ppf_clip_eps), 1.0 - ppf_clip_eps)

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


# -----------------------------------------------------------------------------
# Main API: p11_from_path
# -----------------------------------------------------------------------------
def p11_from_path(
    pA: float,
    pB: float,
    lam: float,
    *,
    path: Path | str,
    path_params: Mapping[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """
    Enterprise-grade p11 constructor with explicit invariants and audit fields.

    Returns (p11, meta) where meta is NUMERIC-ONLY by design.

    Required meta keys for ALL paths:
      - L, U, FH_width, lam, lam_eff
      - raw_p11, clip_amt, clipped
      - fh_violation, fh_violation_amt
    """
    pA_f = _finite_in_unit(pA, "pA")
    pB_f = _finite_in_unit(pB, "pB")
    lam_f = _finite_in_unit(lam, "lam")

    pp = _mapping(path_params)

    path_s = str(path).lower().strip()
    if path_s not in ("fh_linear", "fh_power", "fh_scurve", "gaussian_tau"):
        raise InputValidationError(f"Unknown path: {path!r}")

    b = U.fh_bounds(pA_f, pB_f)
    L = float(b.L)
    Uu = float(b.U)
    if not (math.isfinite(L) and math.isfinite(Uu) and 0.0 <= L <= Uu <= 1.0):
        raise NumericalError(f"FH bounds invalid for pA={pA_f}, pB={pB_f}: L={L}, U={Uu}")

    width = float(Uu - L)

    meta: Dict[str, float] = {
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
    if not (math.isfinite(clip_tol) and 0.0 <= clip_tol <= 1e-6):
        raise InputValidationError(f"clip_tol must be finite, >=0, and small (<=1e-6), got {clip_tol!r}")

    def _finalize(raw: float, lam_eff: float) -> Tuple[float, Dict[str, float]]:
        raw = _as_float(raw, "raw_p11")
        if not math.isfinite(raw):
            raise NumericalError(f"raw_p11 is non-finite: {raw!r}")

        meta["lam_eff"] = float(lam_eff)
        meta["raw_p11"] = float(raw)

        below = max(0.0, (L - raw))
        above = max(0.0, (raw - Uu))
        viol_amt = max(below, above)
        meta["fh_violation_amt"] = float(viol_amt)
        meta["fh_violation"] = float(1.0 if viol_amt > 0.0 else 0.0)

        if raw < L - clip_tol or raw > Uu + clip_tol:
            if clip_policy == "raise":
                raise FeasibilityError(
                    f"{path_s}: raw_p11 violates FH by more than clip_tol. "
                    f"raw={raw}, L={L}, U={Uu}, clip_tol={clip_tol}"
                )

        clipped = raw
        if clipped < L:
            clipped = L
        elif clipped > Uu:
            clipped = Uu

        meta["clip_amt"] = float(raw - clipped)  # signed
        meta["clipped"] = float(1.0 if clipped != raw else 0.0)

        if not (L <= clipped <= Uu):
            raise NumericalError(f"Postcondition failed: p11 not in [L,U]. p11={clipped}, L={L}, U={Uu}")

        return float(clipped), meta

    # ---- FH envelope paths (all routed through p11_fh_linear for testability)
    if path_s == "fh_linear":
        lam_eff = lam_f
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
        meta["rho"] = float(-1.0)
        return _finalize(float(L), float(lam_f))
    if tau >= 1.0:
        meta["rho"] = float(1.0)
        return _finalize(float(Uu), float(lam_f))

    # interior requires SciPy
    ppf_clip_eps = _as_float(pp.get("ppf_clip_eps", 1e-12), "ppf_clip_eps")
    p11_raw, gmeta = _p11_gaussian_tau(pA_f, pB_f, tau, ppf_clip_eps=ppf_clip_eps)
    for k, v in gmeta.items():
        meta[k] = float(v)
    meta["ppf_clip_eps"] = float(ppf_clip_eps)

    return _finalize(float(p11_raw), float(lam_f))
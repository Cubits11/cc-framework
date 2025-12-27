from __future__ import annotations

"""
simulate.paths
==============

Dependence-path primitives:
- FH linear/power/scurve
- gaussian_tau (optional SciPy)

Exports:
- p11_from_path(pA,pB,lam, path=..., path_params=...) -> (p11, meta)
"""

from typing import Any, Dict, Tuple

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
# Helpers: lambda transforms
# -----------------------------------------------------------------------------
def _lam_power(lam: float, gamma: float) -> float:
    lam = float(lam)
    gamma = float(gamma)

    if not math.isfinite(lam) or not (0.0 <= lam <= 1.0):
        raise ValueError(f"lam must be finite and in [0,1], got {lam!r}")
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(f"gamma must be finite and > 0, got {gamma!r}")

    if lam == 0.0 or lam == 1.0 or gamma == 1.0:
        return lam

    y = lam ** gamma
    if y < 0.0:
        y = 0.0
    elif y > 1.0:
        y = 1.0
    return float(y)


def _lam_scurve(lam: float, k: float) -> float:
    lam = float(lam)
    k = float(k)

    if not math.isfinite(lam) or not (0.0 <= lam <= 1.0):
        raise ValueError(f"lam must be finite and in [0,1], got {lam!r}")
    if not math.isfinite(k) or k <= 0.0:
        raise ValueError(f"k must be finite and > 0, got {k!r}")

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
        return lam

    a = _sigmoid(-0.5 * k)
    b = _sigmoid(+0.5 * k)
    denom = b - a
    if denom <= 0.0 or not math.isfinite(denom):
        return lam

    s_raw = _sigmoid(k * (lam - 0.5))
    s = (s_raw - a) / denom

    if s < 0.0:
        s = 0.0
    elif s > 1.0:
        s = 1.0

    return float(s)


# -----------------------------------------------------------------------------
# Gaussian copula helpers (SciPy optional)
# -----------------------------------------------------------------------------
def _bvn_cdf_scipy(x: float, y: float, rho: float) -> float:
    x = float(x)
    y = float(y)
    rho = float(rho)

    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"x and y must be finite, got x={x!r}, y={y!r}")
    if not math.isfinite(rho) or not (-1.0 <= rho <= 1.0):
        raise ValueError(f"rho must be finite and in [-1,1], got {rho!r}")

    try:
        from scipy.stats import multivariate_normal  # type: ignore
    except ImportError as e:
        raise ImportError("SciPy not available for gaussian_tau path") from e

    mean = np.array([0.0, 0.0], dtype=float)
    cov = np.array([[1.0, rho], [rho, 1.0]], dtype=float)
    val = float(multivariate_normal(mean=mean, cov=cov, allow_singular=True).cdf([x, y]))

    if val < 0.0:
        val = 0.0
    elif val > 1.0:
        val = 1.0
    return val


def _p11_gaussian_tau(
    pA: float,
    pB: float,
    tau: float,
    *,
    ppf_clip_eps: float = 1e-12,
) -> Tuple[float, Dict[str, float]]:
    pA = float(pA)
    pB = float(pB)
    tau = float(tau)
    ppf_clip_eps = float(ppf_clip_eps)

    if not (0.0 <= pA <= 1.0) or not math.isfinite(pA):
        raise ValueError(f"pA must be finite and in [0,1], got {pA!r}")
    if not (0.0 <= pB <= 1.0) or not math.isfinite(pB):
        raise ValueError(f"pB must be finite and in [0,1], got {pB!r}")
    if not math.isfinite(tau) or not (-1.0 <= tau <= 1.0):
        raise ValueError(f"tau must be finite and in [-1,1], got {tau!r}")
    if not math.isfinite(ppf_clip_eps) or not (0.0 < ppf_clip_eps < 0.5):
        raise ValueError(f"ppf_clip_eps must be in (0, 0.5), got {ppf_clip_eps!r}")

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
    path: Path,
    path_params: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
    """
    Research/enterprise-grade p11 constructor with explicit invariants and audit fields.

    Returns (p11, meta) where meta is NUMERIC-ONLY by design.
    Required meta keys for ALL paths:
      - L, U, FH_width, lam, lam_eff
      - raw_p11, clip_amt, clipped
    """
    try:
        pA = float(pA)
        pB = float(pB)
        lam = float(lam)
    except Exception as e:
        raise InputValidationError("pA, pB, lam must be coercible to float.") from e

    if not math.isfinite(pA) or not (0.0 <= pA <= 1.0):
        raise InputValidationError(f"pA must be finite and in [0,1], got {pA!r}")
    if not math.isfinite(pB) or not (0.0 <= pB <= 1.0):
        raise InputValidationError(f"pB must be finite and in [0,1], got {pB!r}")
    if not math.isfinite(lam) or not (0.0 <= lam <= 1.0):
        raise InputValidationError(f"lam must be finite and in [0,1], got {lam!r}")
    if not isinstance(path_params, dict):
        raise InputValidationError(f"path_params must be a dict, got {type(path_params).__name__}")

    b = U.fh_bounds(pA, pB)
    L = float(b.L)
    Uu = float(b.U)
    if not (math.isfinite(L) and math.isfinite(Uu) and 0.0 <= L <= Uu <= 1.0):
        raise NumericalError(f"FH bounds invalid for pA={pA}, pB={pB}: L={L}, U={Uu}")

    width = float(Uu - L)
    meta: Dict[str, float] = {
        "L": L,
        "U": Uu,
        "FH_width": width,
        "lam": float(lam),
        "lam_eff": float("nan"),
        "raw_p11": float("nan"),
        "clip_amt": 0.0,
        "clipped": 0.0,
    }

    clip_policy = str(path_params.get("clip_policy", "clip")).lower().strip()
    if clip_policy not in ("clip", "raise"):
        raise InputValidationError(f"clip_policy must be 'clip' or 'raise', got {clip_policy!r}")

    clip_tol = float(path_params.get("clip_tol", 0.0))
    if not math.isfinite(clip_tol) or clip_tol < 0.0 or clip_tol > 1e-6:
        raise InputValidationError(f"clip_tol must be finite, >=0, and small (<=1e-6). got {clip_tol!r}")

    def _finalize(raw: float, lam_eff: float) -> Tuple[float, Dict[str, float]]:
        if not math.isfinite(raw):
            raise NumericalError(f"raw_p11 is non-finite: {raw!r}")
        meta["lam_eff"] = float(lam_eff)
        meta["raw_p11"] = float(raw)

        if raw < L - clip_tol or raw > Uu + clip_tol:
            if clip_policy == "raise":
                raise FeasibilityError(
                    f"{path}: raw_p11 violates FH bounds by more than clip_tol. "
                    f"raw={raw}, L={L}, U={Uu}, clip_tol={clip_tol}"
                )

        clipped = raw
        if clipped < L:
            clipped = L
        elif clipped > Uu:
            clipped = Uu

        meta["clip_amt"] = float(raw - clipped)
        meta["clipped"] = float(1.0 if clipped != raw else 0.0)

        if not (L <= clipped <= Uu):
            raise NumericalError(f"Postcondition failed: p11 not in [L,U]. p11={clipped}, L={L}, U={Uu}")
        return float(clipped), meta

    if path == "fh_linear":
        raw = float(U.p11_fh_linear(pA, pB, lam))
        return _finalize(raw, lam_eff=lam)

    if path == "fh_power":
        gamma_raw = path_params.get("gamma", 1.0)
        gamma = float(gamma_raw)
        if not math.isfinite(gamma) or gamma <= 0.0:
            raise InputValidationError(f"fh_power requires gamma > 0, got {gamma_raw!r}")
        lam_eff = _lam_power(lam, gamma)
        raw = L + lam_eff * width
        meta["gamma"] = float(gamma)
        return _finalize(float(raw), lam_eff=float(lam_eff))

    if path == "fh_scurve":
        k_raw = path_params.get("k", 8.0)
        k = float(k_raw)
        if not math.isfinite(k) or k <= 0.0:
            raise InputValidationError(f"fh_scurve requires k > 0, got {k_raw!r}")
        lam_eff = _lam_scurve(lam, k)
        raw = L + lam_eff * width
        meta["k"] = float(k)
        return _finalize(float(raw), lam_eff=float(lam_eff))

    if path == "gaussian_tau":
        tau = 2.0 * lam - 1.0
        if not math.isfinite(tau) or not (-1.0 <= tau <= 1.0):
            raise InputValidationError(f"tau derived from lam must be in [-1,1], got {tau!r}")

        # degenerate marginals: overlap is determined
        if pA in (0.0, 1.0) or pB in (0.0, 1.0):
            raw = min(pA, pB)
            rho = math.sin(math.pi * tau / 2.0)
            meta.update({"tau": float(tau), "rho": float(rho)})
            meta.update(
                {"ppf_clip_eps": 0.0, "pA_used": float(pA), "pB_used": float(pB), "pA_clip": 0.0, "pB_clip": 0.0}
            )
            return _finalize(float(raw), lam_eff=lam)

        tau_ext_tol = float(path_params.get("tau_extreme_tol", 1e-12))
        if not math.isfinite(tau_ext_tol) or tau_ext_tol < 0.0 or tau_ext_tol > 1e-6:
            raise InputValidationError(f"tau_extreme_tol must be finite and small (<=1e-6), got {tau_ext_tol!r}")

        if tau >= 1.0 - tau_ext_tol:
            meta.update({"tau": float(tau), "rho": 1.0})
            meta.update(
                {"ppf_clip_eps": 0.0, "pA_used": float(pA), "pB_used": float(pB), "pA_clip": 0.0, "pB_clip": 0.0}
            )
            return _finalize(float(Uu), lam_eff=lam)

        if tau <= -1.0 + tau_ext_tol:
            meta.update({"tau": float(tau), "rho": -1.0})
            meta.update(
                {"ppf_clip_eps": 0.0, "pA_used": float(pA), "pB_used": float(pB), "pA_clip": 0.0, "pB_clip": 0.0}
            )
            return _finalize(float(L), lam_eff=lam)

        eps_raw = path_params.get("ppf_clip_eps", 1e-12)
        eps = float(eps_raw)
        if not math.isfinite(eps) or not (0.0 < eps < 0.5):
            raise InputValidationError(f"ppf_clip_eps must be in (0,0.5), got {eps_raw!r}")

        pA_eff = min(max(pA, eps), 1.0 - eps)
        pB_eff = min(max(pB, eps), 1.0 - eps)

        meta.update(
            {
                "ppf_clip_eps": float(eps),
                "pA_used": float(pA_eff),
                "pB_used": float(pB_eff),
                "pA_clip": float(pA - pA_eff),
                "pB_clip": float(pB - pB_eff),
            }
        )

        try:
            raw, m2 = _p11_gaussian_tau(pA_eff, pB_eff, tau)
        except ImportError as e:
            raise ImportError(
                "gaussian_tau requested but SciPy is unavailable. Install scipy or choose fh_* paths."
            ) from e

        for k, v in m2.items():
            meta[k] = float(v)

        return _finalize(float(raw), lam_eff=lam)

    raise InputValidationError(f"Unknown path: {path!r}")

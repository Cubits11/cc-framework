# src/cc/cartographer/bounds.py
"""
Module: bounds
Purpose:
  (A) Fréchet–Hoeffding ceilings for composed Youden's J over two ROC curves
  (B) Finite-sample FH–Bernstein utilities for CC at a fixed operating point θ
  (C) Seamless bridge to optional Enterprise add-ons if present

Design notes
------------
- Backward compatibility: the original public API is preserved.
- If `bounds_enterprise.py` is present, we:
    • expose its advanced classes in this module's namespace; and
    • allow select functions to delegate to the enterprise implementations
      when enterprise-only options are provided (e.g. use_gpu, uncertainty).
- If `bounds_enterprise.py` is absent, everything still works with the
  lean, dependency-light numpy core here.

Tip: For power users, you can import the advanced classes directly:
    from cc.cartographer.bounds import GPBounds, BanditThresholdSelector, ...
"""

from __future__ import annotations

from math import ceil, exp, log
from typing import Any, Dict, Iterable, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

# -----------------------------------------------------------------------------
# Optional enterprise bridge (best-effort import)
# -----------------------------------------------------------------------------
_HAS_ENTERPRISE = False
try:
    # Local sibling file expected at: src/cc/cartographer/bounds_enterprise.py
    from . import bounds_enterprise as _be  # type: ignore

    # Re-export enterprise classes & enums if available
    AdaptiveBounds = _be.AdaptiveBounds
    BayesianBounds = _be.BayesianBounds
    CausalBounds = _be.CausalBounds
    StreamingBounds = _be.StreamingBounds
    MultiObjectiveOptimizer = _be.MultiObjectiveOptimizer
    BanditThresholdSelector = _be.BanditThresholdSelector
    GPBounds = _be.GPBounds
    DistributedROCAnalyzer = _be.DistributedROCAnalyzer
    ConfidenceSequence = _be.ConfidenceSequence
    PredictionInterval = _be.PredictionInterval
    UncertaintyQuantifier = _be.UncertaintyQuantifier
    AdaptiveStrategy = _be.AdaptiveStrategy
    BoundType = _be.BoundType

    _HAS_ENTERPRISE = True
except Exception:  # noqa: BLE001
    # Enterprise is optional; quietly continue with the core-only implementation.
    pass

__all__ = [
    # Core ceilings over ROC curves
    "ROCArrayLike",
    "frechet_upper",
    "frechet_upper_with_argmax",
    "frechet_upper_with_argmax_points",
    "envelope_over_rocs",
    "ensure_anchors",
    # n-rail FH helpers
    "fh_and_bounds_n",
    "fh_or_bounds_n",
    # FH/Bernstein utilities at a fixed θ
    "fh_intervals",
    "fh_var_envelope",
    "bernstein_tail",
    "invert_bernstein_eps",
    "cc_two_sided_bound",
    "cc_confint",
    "needed_n_bernstein",
    "needed_n_bernstein_int",
]

# If enterprise is present, surface its extras from this module.
if _HAS_ENTERPRISE:
    __all__ += [
        "AdaptiveBounds",
        "BayesianBounds",
        "CausalBounds",
        "StreamingBounds",
        "MultiObjectiveOptimizer",
        "BanditThresholdSelector",
        "GPBounds",
        "DistributedROCAnalyzer",
        "ConfidenceSequence",
        "PredictionInterval",
        "UncertaintyQuantifier",
        "AdaptiveStrategy",
        "BoundType",
    ]

# ---- Types -----------------------------------------------------------------

ROCPoint: TypeAlias = Tuple[float, float]
ROCArrayLike: TypeAlias = Union[
    Sequence[ROCPoint],
    Iterable[ROCPoint],
    NDArray[np.float64],
]

# ---- Utilities (ROC arrays) ------------------------------------------------


def _to_array_roc(
    arr: ROCArrayLike, *, clip: Literal["silent", "warn", "error"] = "silent"
) -> NDArray[np.float64]:
    """
    Coerce to float array of shape (n, 2) with columns [FPR, TPR].

    clip:
        - "silent": silently clip out-of-range to [0,1] (default).
        - "warn":   clip and emit a RuntimeWarning.
        - "error":  raise on any out-of-range entry.
    """
    if isinstance(arr, np.ndarray):
        roc = arr.astype(float, copy=False)
    else:
        roc = np.asarray(list(arr), dtype=float)

    if roc.ndim != 2 or roc.shape[1] != 2:
        raise ValueError("ROC must be array-like of shape (n, 2) with columns [FPR, TPR].")

    oob_mask = (roc < 0.0) | (roc > 1.0)
    if oob_mask.any():
        if clip == "error":
            bad = roc[oob_mask.any(axis=1)]
            raise ValueError(f"ROC contains out-of-range values; first few: {bad[:4]!r}")
        elif clip == "warn":
            import warnings

            warnings.warn("Clipping ROC values to [0,1].", RuntimeWarning, stacklevel=2)
        roc = np.clip(roc, 0.0, 1.0)

    if not np.isfinite(roc).all():
        raise ValueError("ROC contains non-finite values.")
    return roc


def ensure_anchors(
    roc: ROCArrayLike,
    *,
    preserve_order: bool = True,
    clip: Literal["silent", "warn", "error"] = "silent",
) -> NDArray[np.float64]:
    """
    Ensure ROC contains (0,0) and (1,1).

    If preserve_order=True, append any missing anchors without global sorting
    (keeps caller indices stable). Otherwise, return unique, sorted rows
    (lexicographic by FPR asc, then TPR asc).
    """
    R = _to_array_roc(roc, clip=clip)
    has_00 = np.any((R[:, 0] == 0.0) & (R[:, 1] == 0.0))
    has_11 = np.any((R[:, 0] == 1.0) & (R[:, 1] == 1.0))
    if not has_00:
        R = np.vstack([R, [0.0, 0.0]])
    if not has_11:
        R = np.vstack([R, [1.0, 1.0]])
    if preserve_order:
        return R
    # unique + sorted (FPR asc, then TPR asc)
    R = np.unique(R, axis=0)
    order = np.lexsort((R[:, 1], R[:, 0]))
    return R[order]


def _frechet_grid_AND(
    FPR_a: NDArray[np.float64],
    TPR_a: NDArray[np.float64],
    FPR_b: NDArray[np.float64],
    TPR_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    tpr_and = np.minimum(TPR_a, TPR_b)
    fpr_and = np.maximum(0.0, FPR_a + FPR_b - 1.0)
    return tpr_and - fpr_and  # J grid


def _frechet_grid_OR(
    FPR_a: NDArray[np.float64],
    TPR_a: NDArray[np.float64],
    FPR_b: NDArray[np.float64],
    TPR_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    tpr_or = np.minimum(1.0, TPR_a + TPR_b)
    fpr_or = np.maximum(FPR_a, FPR_b)
    return tpr_or - fpr_or  # J grid


# ---- n-rail FH helpers -----------------------------------------------------


def fh_and_bounds_n(alphas: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    FH bounds for ∧_i A_i given per-rail trigger rates alphas = P(A_i) with shape (k, ...).
    Returns (lower, upper) where:
      lower = max(0, sum_i alpha_i - (k-1)),  upper = min_i alpha_i
    """
    if alphas.ndim < 1:
        raise ValueError("alphas must have rail dimension on axis 0.")
    k = alphas.shape[0]
    lower = np.maximum(0.0, np.sum(alphas, axis=0) - (k - 1))
    upper = np.min(alphas, axis=0)
    return lower, upper


def fh_or_bounds_n(alphas: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    FH bounds for ∨_i A_i given per-rail trigger rates alphas = P(A_i) with shape (k, ...).
    Returns (lower, upper) where:
      lower = max_i alpha_i,  upper = min(1, sum_i alpha_i)
    """
    if alphas.ndim < 1:
        raise ValueError("alphas must have rail dimension on axis 0.")
    lower = np.max(alphas, axis=0)
    upper = np.minimum(1.0, np.sum(alphas, axis=0))
    return lower, upper


# ---- Public API: FH ceilings over ROC curves -------------------------------


def _maybe_delegate_to_enterprise(fname: str, **kw: Any):
    """
    Internal: If enterprise has a same-named function and non-core options were
    provided, delegate to it. Otherwise return None to signal 'use core'.
    """
    if not _HAS_ENTERPRISE:
        return None
    f = getattr(_be, fname, None)
    if f is None:
        return None
    # If caller passed any enterprise-only options, prefer enterprise path.
    enterprise_flags = {"use_gpu", "uncertainty", "n_bootstrap"}
    if enterprise_flags & set(kw):
        return f
    # Even without flags, delegating is safe & feature-equivalent; but we keep
    # core-by-default for maximal determinism unless flags are present.
    return None


def frechet_upper(
    roc_a: ROCArrayLike,
    roc_b: ROCArrayLike,
    *,
    comp: Literal["AND", "OR", "and", "or"] = "AND",
    clip: Literal["silent", "warn", "error"] = "silent",
    add_anchors: bool = False,
    # Enterprise passthrough (ignored by core; used if bounds_enterprise is present):
    use_gpu: bool = False,
    uncertainty: bool = False,
    n_bootstrap: int = 1000,
) -> Union[float, Tuple[float, Dict[str, float]]]:
    """
    Fréchet-style *upper bound* on composed Youden's J over two ROC curves.

    Core path (always available): deterministic numpy implementation.
    Enterprise path (optional): honors `use_gpu`, `uncertainty`, `n_bootstrap`.
    """
    maybe = _maybe_delegate_to_enterprise(
        "frechet_upper",
        use_gpu=use_gpu,
        uncertainty=uncertainty,
        n_bootstrap=n_bootstrap,
    )
    if maybe is not None:
        # Delegate entirely; enterprise variant matches this signature.
        return maybe(
            roc_a,
            roc_b,
            comp=comp,
            clip=clip,
            add_anchors=add_anchors,
            use_gpu=use_gpu,
            uncertainty=uncertainty,
            n_bootstrap=n_bootstrap,
        )

    # ---- Core implementation (numpy-only) ---------------------------------
    if add_anchors:
        A = ensure_anchors(roc_a, clip=clip)
        B = ensure_anchors(roc_b, clip=clip)
    else:
        A = _to_array_roc(roc_a, clip=clip)
        B = _to_array_roc(roc_b, clip=clip)

    FPR_a = A[:, 0][:, None]  # (Na,1)
    TPR_a = A[:, 1][:, None]  # (Na,1)
    FPR_b = B[:, 0][None, :]  # (1,Nb)
    TPR_b = B[:, 1][None, :]  # (1,Nb)

    comp_u = comp.upper()
    if comp_u == "AND":
        Jgrid = _frechet_grid_AND(FPR_a, TPR_a, FPR_b, TPR_b)
    elif comp_u == "OR":
        Jgrid = _frechet_grid_OR(FPR_a, TPR_a, FPR_b, TPR_b)
    else:
        raise ValueError('comp must be "AND" or "OR".')

    if not Jgrid.size:
        return 0.0
    Jgrid = np.nan_to_num(Jgrid, nan=-1.0, posinf=1.0, neginf=-1.0)
    jmax = float(np.max(Jgrid))
    return float(np.clip(jmax, -1.0, 1.0))


def frechet_upper_with_argmax(
    roc_a: ROCArrayLike,
    roc_b: ROCArrayLike,
    *,
    comp: Literal["AND", "OR", "and", "or"] = "AND",
    clip: Literal["silent", "warn", "error"] = "silent",
    add_anchors: bool = False,
) -> Tuple[float, Optional[int], Optional[int]]:
    """As `frechet_upper`, but also return (ia, ib) of the maximizing cross-pair."""
    if add_anchors:
        A = ensure_anchors(roc_a, clip=clip)
        B = ensure_anchors(roc_b, clip=clip)
    else:
        A = _to_array_roc(roc_a, clip=clip)
        B = _to_array_roc(roc_b, clip=clip)

    if A.size == 0 or B.size == 0:
        return 0.0, None, None

    FPR_a = A[:, 0][:, None]
    TPR_a = A[:, 1][:, None]
    FPR_b = B[:, 0][None, :]
    TPR_b = B[:, 1][None, :]

    comp_u = comp.upper()
    if comp_u == "AND":
        Jgrid = _frechet_grid_AND(FPR_a, TPR_a, FPR_b, TPR_b)
    elif comp_u == "OR":
        Jgrid = _frechet_grid_OR(FPR_a, TPR_a, FPR_b, TPR_b)
    else:
        raise ValueError('comp must be "AND" or "OR".')

    Jgrid = np.nan_to_num(Jgrid, nan=-1.0, posinf=1.0, neginf=-1.0)
    flat_idx = int(np.nanargmax(Jgrid))
    ia, ib = np.unravel_index(flat_idx, Jgrid.shape)
    jmax = float(Jgrid[ia, ib])
    return float(np.clip(jmax, -1.0, 1.0)), int(ia), int(ib)


def frechet_upper_with_argmax_points(
    roc_a: ROCArrayLike,
    roc_b: ROCArrayLike,
    *,
    comp: Literal["AND", "OR", "and", "or"] = "AND",
    clip: Literal["silent", "warn", "error"] = "silent",
    add_anchors: bool = False,
) -> Tuple[float, ROCPoint, ROCPoint]:
    """
    Like `frechet_upper_with_argmax` but returns the maximizing ROC points themselves:
    (J_max, (fpr_a*, tpr_a*), (fpr_b*, tpr_b*))
    """
    if add_anchors:
        A = ensure_anchors(roc_a, clip=clip)
        B = ensure_anchors(roc_b, clip=clip)
    else:
        A = _to_array_roc(roc_a, clip=clip)
        B = _to_array_roc(roc_b, clip=clip)

    jmax, ia, ib = frechet_upper_with_argmax(A, B, comp=comp, clip=clip, add_anchors=False)
    if ia is None or ib is None:
        return 0.0, (0.0, 0.0), (0.0, 0.0)
    return jmax, (float(A[ia, 0]), float(A[ia, 1])), (float(B[ib, 0]), float(B[ib, 1]))


def envelope_over_rocs(
    roc_a: ROCArrayLike,
    roc_b: ROCArrayLike,
    *,
    comp: Literal["AND", "OR", "and", "or"] = "AND",
    clip: Literal["silent", "warn", "error"] = "silent",
    add_anchors: bool = False,
) -> Tuple[float, NDArray[np.float64]]:
    """
    Return (J_max, J_grid) where J_grid has shape (Na, Nb).
    Useful for plotting heatmaps and highlighting the argmax.
    """
    if add_anchors:
        A = ensure_anchors(roc_a, clip=clip)
        B = ensure_anchors(roc_b, clip=clip)
    else:
        A = _to_array_roc(roc_a, clip=clip)
        B = _to_array_roc(roc_b, clip=clip)

    FPR_a = A[:, 0][:, None]
    TPR_a = A[:, 1][:, None]
    FPR_b = B[:, 0][None, :]
    TPR_b = B[:, 1][None, :]

    comp_u = comp.upper()
    if comp_u == "AND":
        J = _frechet_grid_AND(FPR_a, TPR_a, FPR_b, TPR_b)
    elif comp_u == "OR":
        J = _frechet_grid_OR(FPR_a, TPR_a, FPR_b, TPR_b)
    else:
        raise ValueError('comp must be "AND" or "OR".')

    J = np.nan_to_num(J, nan=-1.0, posinf=1.0, neginf=-1.0)
    return float(np.clip(np.max(J), -1.0, 1.0)), J


# ---- FH/Bernstein utilities at a fixed θ -----------------------------------


def _validate_prob(name: str, x: float) -> None:
    if not (0.0 <= x <= 1.0) or not np.isfinite(x):
        raise ValueError(f"{name} must be a finite probability in [0,1]. Got {x}.")


def fh_intervals(
    tpr_a: float,
    tpr_b: float,
    fpr_a: float,
    fpr_b: float,
    *,
    alpha_cap: Optional[float] = None,
    cap_mode: Literal["error", "clip"] = "error",
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Fréchet–Hoeffding *sharp* intervals for:
      I1: p1 = P(A ∧ B = 1 | Y=1)   (AND on positives)
      I0: p0 = P(A ∨ B = 1 | Y=0)   (OR  on negatives)

    With an optional policy cap α for negatives:
      I0 = [max(fpr_a,fpr_b), min(α, fpr_a+fpr_b, 1)]  (raise or clip if empty).
    """
    for nm, x in [("tpr_a", tpr_a), ("tpr_b", tpr_b), ("fpr_a", fpr_a), ("fpr_b", fpr_b)]:
        _validate_prob(nm, x)

    L1 = max(0.0, tpr_a + tpr_b - 1.0)
    U1 = min(tpr_a, tpr_b)

    L0 = max(fpr_a, fpr_b)
    U0 = min(1.0, fpr_a + fpr_b)
    if alpha_cap is not None:
        _validate_prob("alpha_cap", alpha_cap)
        U0 = min(U0, alpha_cap)

    if U1 < L1:
        U1 = L1
    if U0 < L0:
        if cap_mode == "error":
            raise ValueError(
                f"Policy cap makes I0 empty: L0={L0:.6f}, U0={U0:.6f}. "
                "Relax α or adjust θ/marginals."
            )
        else:
            U0 = L0  # point interval under strict cap

    return (L1, U1), (L0, U0)


def fh_var_envelope(interval: Tuple[float, float]) -> float:
    """
    Return max_{p in [a,b]} p(1-p); 0.25 if 0.5∈[a,b], else max of endpoints.
    """
    a, b = interval
    if not (np.isfinite(a) and np.isfinite(b) and 0.0 <= a <= b <= 1.0):
        raise ValueError(f"Invalid interval {interval}; must satisfy 0<=a<=b<=1.")
    if a <= 0.5 <= b:
        return 0.25
    return max(a * (1.0 - a), b * (1.0 - b))


def bernstein_tail(
    *args,
    t: Optional[float] = None,
    eps: Optional[float] = None,
    n: Optional[int] = None,
    vbar: Optional[float] = None,
    D: float = 1.0,
    two_sided: bool = True,
) -> float:
    """
    Bernstein tail for a Bernoulli mean with variance envelope vbar.

    Two call styles are supported:
      • New (preferred): bernstein_tail(t=..., n=..., vbar=..., D=..., two_sided=True)
          uses ε = t·D (CC half-width t mapped to class-space).
      • Legacy       : bernstein_tail(n, eps, vbar) where eps=ε directly.

    Bound (two-sided):
      P(|p̂ - p| ≥ ε) ≤ 2 * exp( - n * ε^2 / (2 vbar + (2/3) ε) ).

    Monotonicity:
      increasing in n ↓ (prob decreases), increasing in ε (or t) ↓ (prob decreases).
    """
    # legacy positional (n, eps, vbar)
    if args:
        if len(args) == 3 and all(a is not None for a in args):
            n_pos, eps_pos, vbar_pos = args  # type: ignore
            n = int(n_pos)
            eps = float(eps_pos)
            vbar = float(vbar_pos)
        else:
            raise TypeError(
                "Legacy positional call must be bernstein_tail(n, eps, vbar). "
                "Prefer keyword style: bernstein_tail(t=..., n=..., vbar=..., D=...)."
            )

    if n is None or vbar is None:
        raise ValueError("Provide n and vbar (keywords or legacy positional).")
    if n <= 0:
        raise ValueError("n must be positive.")
    if not (0.0 <= vbar <= 0.25 + 1e-12):
        raise ValueError("vbar must lie in [0, 0.25].")

    if eps is None:
        if t is None:
            raise ValueError("Provide either t (with D) or eps.")
        if D <= 0:
            raise ValueError("D must be positive when using t.")
        if t < 0:
            raise ValueError("t must be nonnegative.")
        eps = t * D
    else:
        if eps < 0:
            raise ValueError("eps must be nonnegative.")

    if eps == 0.0:
        return 1.0

    denom = 2.0 * vbar + (2.0 / 3.0) * eps
    if denom <= 0.0:
        return 1.0

    exponent = -(n * eps * eps) / denom
    # Avoid under/overflow; map to [0,1].
    base = exp(exponent)
    prob = (2.0 * base) if two_sided else base
    return float(0.0 if prob < 0.0 else (1.0 if prob > 1.0 else prob))


def invert_bernstein_eps(n: int, vbar: float, delta: float) -> float:
    """
    Invert  2 * exp( - n * eps^2 / (2 vbar + (2/3) eps) ) ≤ (1 - delta)
    to solve for the smallest eps ≥ 0. Uses bracket + binary search.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if not (0.0 <= vbar <= 0.25):
        raise ValueError("vbar must be in [0, 0.25].")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1).")

    target = log(2.0 / (1.0 - delta))  # > 0
    lo, hi = 0.0, 1.0
    for _ in range(64):
        denom = 2.0 * vbar + (2.0 / 3.0) * hi
        val = 0.0 if denom <= 0.0 else n * (hi * hi) / denom
        if val >= target:
            break
        hi *= 2.0
        if hi > 1e6:
            break
    for _ in range(96):
        mid = 0.5 * (lo + hi)
        denom = 2.0 * vbar + (2.0 / 3.0) * mid
        val = 0.0 if denom <= 0.0 else n * (mid * mid) / denom
        if val >= target:
            hi = mid
        else:
            lo = mid
    return hi


def cc_two_sided_bound(
    n1: int,
    n0: int,
    t: float,
    D: float,
    I1: Tuple[float, float],
    I0: Tuple[float, float],
    *,
    cap_at_one: bool = False,
) -> float:
    """Union bound of two Bernstein tails on classes Y=1 and Y=0 for CC half-width t."""
    if D <= 0.0:
        return 1.0
    if t < 0.0:
        raise ValueError("t must be >= 0.")
    v1 = fh_var_envelope(I1)
    v0 = fh_var_envelope(I0)
    s = bernstein_tail(t=t, n=n1, vbar=v1, D=D, two_sided=True) + bernstein_tail(
        t=t, n=n0, vbar=v0, D=D, two_sided=True
    )
    return min(1.0, s) if cap_at_one else s


def cc_confint(
    n1: int,
    n0: int,
    p1_hat: float,
    p0_hat: float,
    D: float,
    I1: Tuple[float, float],
    I0: Tuple[float, float],
    *,
    delta: float = 0.05,
    split: Optional[Tuple[float, float]] = None,
    clamp01: bool = False,
) -> Tuple[float, float]:
    """
    Finite-sample two-sided CI for CC: invert classwise Bernstein tails, propagate through D.
    Returns (lo, hi) around CC_hat = (1 - (p1_hat - p0_hat)) / D.

    clamp01: if True, clamp (lo, hi) to [0,1] for display convenience.
    """
    if D <= 0.0:
        raise ValueError("D must be > 0.")
    for nm, x in [("p1_hat", p1_hat), ("p0_hat", p0_hat)]:
        _validate_prob(nm, x)
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1).")
    if split is None:
        d1 = d0 = 0.5 * delta
    else:
        d1, d0 = split
        if d1 <= 0.0 or d0 <= 0.0 or abs((d1 + d0) - delta) > 1e-12:
            raise ValueError("split must be positive and sum to delta.")

    v1 = fh_var_envelope(I1)
    v0 = fh_var_envelope(I0)
    eps1 = invert_bernstein_eps(n1, v1, 1.0 - d1)
    eps0 = invert_bernstein_eps(n0, v0, 1.0 - d0)
    cc_hat = (1.0 - (p1_hat - p0_hat)) / D
    t_half = (eps1 + eps0) / D
    lo, hi = cc_hat - t_half, cc_hat + t_half
    if clamp01:
        lo, hi = max(0.0, lo), min(1.0, hi)
    return lo, hi


def needed_n_bernstein(
    t: float,
    D: float,
    I1: Tuple[float, float],
    I0: Tuple[float, float],
    *,
    delta: float = 0.05,
    split: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """
    Closed-form per-class sample sizes (floats) so each Bernstein term ≤ δ_y.
    Enforce: 2 * exp( - n_y (tD)^2 / ( 2 v̄_y + (2/3) tD ) ) ≤ δ_y
      ⇒ n_y ≥ (( 2 v̄_y + (2/3) tD ) / (tD)^2) * log(2/δ_y).
    """
    if t <= 0.0 or D <= 0.0:
        raise ValueError("t and D must be > 0.")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1).")
    if split is None:
        d1 = d0 = 0.5 * delta
    else:
        d1, d0 = split
        if d1 <= 0.0 or d0 <= 0.0 or abs((d1 + d0) - delta) > 1e-12:
            raise ValueError("split must be positive and sum to delta.")

    v1 = fh_var_envelope(I1)
    v0 = fh_var_envelope(I0)

    def n_star(vbar: float, dely: float) -> float:
        num = 2.0 * vbar + (2.0 / 3.0) * t * D
        den = (t * D) ** 2
        return (num / den) * log(2.0 / dely)

    return n_star(v1, d1), n_star(v0, d0)


def needed_n_bernstein_int(
    t: float,
    D: float,
    I1: Tuple[float, float],
    I0: Tuple[float, float],
    *,
    delta: float = 0.05,
    split: Optional[Tuple[float, float]] = None,
) -> Tuple[int, int]:
    """
    Integer (ceil) per-class sample sizes for planner; wraps `needed_n_bernstein`.
    Returns (max(1, ceil(n1_float)), max(1, ceil(n0_float))).
    """
    n1f, n0f = needed_n_bernstein(t, D, I1, I0, delta=delta, split=split)
    return max(1, int(ceil(n1f))), max(1, int(ceil(n0f)))

# src/cc/cartographer/bounds.py
"""
Module: bounds
Purpose:
  (A) Fréchet–Hoeffding-style ceilings for composed Youden's J over two ROC curves
  (B) Finite-sample FH–Bernstein utilities for CC at a fixed operating point θ

Author: Pranav Bhave (upgraded with FH-variance + Bernstein/CC utilities)
Dates:
  - Original: 2025-08-31
  - Refined:  2025-09-05
  - Upgraded: 2025-09-11

Overview
--------
Given two ROC curves A and B as (FPR, TPR) point sets, (A) computes an *upper bound*
on the Youden's J statistic of their AND/OR composition under arbitrary dependence
(no independence assumed). Pointwise ceilings come from Fréchet–Hoeffding (FH).

Separately, for a *fixed* operating point θ = (τ_a, τ_b) with known class-conditional
marginals (TPR_a,TPR_b,FPR_a,FPR_b), (B) provides:
  • FH intervals I1 for p1 = P(A ∧ B = 1 | Y=1), and I0 for p0 = P(A ∨ B = 1 | Y=0);
  • A tight variance envelope  v̄ = max_{p∈I} p(1-p);
  • Two-sided Bernstein tails for Bernoulli means and a CC deviation bound;
  • Finite-sample CI for CC and a closed-form per-class sample-size planner.

Design notes
------------
- Robust to sparsity/ordering of ROC samples (no monotonicity assumed).
- Optional “anchor” augmentation adds (0,0) and (1,1) when desired.
- Type-annotated, pure NumPy + math; mypy-friendly.
- Bernstein utilities are distribution-free via FH variance envelopes.
- Clear exceptions on infeasible policy caps (I0 upper < lower).

"""

from __future__ import annotations

from typing import Iterable, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias
from math import exp, log

__all__ = [
    # ROC/FH ceilings on J over curves
    "ROCArrayLike",
    "frechet_upper",
    "frechet_upper_with_argmax",
    "ensure_anchors",
    # FH/Bernstein utilities at a fixed θ
    "fh_intervals",
    "fh_var_envelope",
    "bernstein_tail",
    "invert_bernstein_eps",
    "cc_two_sided_bound",
    "cc_confint",
    "needed_n_bernstein",
]

# ---- Types -----------------------------------------------------------------

ROCPoint: TypeAlias = Tuple[float, float]
ROCArrayLike: TypeAlias = Union[
    Sequence[ROCPoint],
    Iterable[ROCPoint],
    NDArray[np.float64],
]


# ---- Utilities (ROC arrays) ------------------------------------------------

def _to_array_roc(arr: ROCArrayLike) -> NDArray[np.float64]:
    """
    Coerce to float array of shape (n, 2) with columns [FPR, TPR], clipped into [0,1].

    Raises:
        ValueError: on bad shape or non-finite values.
    """
    if isinstance(arr, np.ndarray):
        roc = arr.astype(float, copy=False)
    else:
        roc = np.asarray(list(arr), dtype=float)

    if roc.ndim != 2 or roc.shape[1] != 2:
        raise ValueError("ROC must be array-like of shape (n, 2) with columns [FPR, TPR].")

    roc = np.clip(roc, 0.0, 1.0)
    if not np.isfinite(roc).all():
        raise ValueError("ROC contains non-finite values.")
    return roc


def ensure_anchors(roc: ROCArrayLike) -> NDArray[np.float64]:
    """
    Ensure ROC contains (0,0) and (1,1) anchor points. Returns a new array with uniques kept.
    Useful when curves are sampled sparsely and the envelope needs closure.
    """
    R = _to_array_roc(roc)
    anchors = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    stacked = np.vstack([R, anchors])
    uniq = np.unique(stacked, axis=0)
    return uniq


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


# ---- Public API: FH ceilings over ROC curves -------------------------------

def frechet_upper(
    roc_a: ROCArrayLike,
    roc_b: ROCArrayLike,
    *,
    comp: Literal["AND", "OR"] = "AND",
    add_anchors: bool = False,
) -> float:
    """
    Compute the Fréchet-style *upper bound* on composed Youden's J over two ROC curves.

    Args:
        roc_a: Sequence/array of (FPR, TPR) for guardrail A.
        roc_b: Sequence/array of (FPR, TPR) for guardrail B.
        comp:  Composition rule: "AND" or "OR".
        add_anchors: If True, add (0,0) and (1,1) anchors to each curve before bounding.

    Returns:
        Upper bound on composed J, clipped to [-1, 1].

    Notes:
        This is a ceiling under arbitrary dependence; it ignores any independence assumptions.
    """
    A = ensure_anchors(roc_a) if add_anchors else _to_array_roc(roc_a)
    B = ensure_anchors(roc_b) if add_anchors else _to_array_roc(roc_b)

    # Broadcast cross-pairs
    FPR_a = A[:, 0][:, None]  # (Na,1)
    TPR_a = A[:, 1][:, None]  # (Na,1)
    FPR_b = B[:, 0][None, :]  # (1,Nb)
    TPR_b = B[:, 1][None, :]  # (1,Nb)

    if comp == "AND":
        Jgrid = _frechet_grid_AND(FPR_a, TPR_a, FPR_b, TPR_b)
    elif comp == "OR":
        Jgrid = _frechet_grid_OR(FPR_a, TPR_a, FPR_b, TPR_b)
    else:
        raise ValueError('comp must be "AND" or "OR".')

    jmax = float(np.max(Jgrid)) if Jgrid.size else 0.0
    return float(np.clip(jmax, -1.0, 1.0))


def frechet_upper_with_argmax(
    roc_a: ROCArrayLike,
    roc_b: ROCArrayLike,
    *,
    comp: Literal["AND", "OR"] = "AND",
    add_anchors: bool = False,
) -> Tuple[float, Optional[int], Optional[int]]:
    """
    As `frechet_upper`, but also return indices (ia, ib) of the maximizing cross-pair.

    Returns:
        (jmax, ia, ib) where ia indexes into A, ib into B. If either curve is empty,
        returns (0.0, None, None).
    """
    A = ensure_anchors(roc_a) if add_anchors else _to_array_roc(roc_a)
    B = ensure_anchors(roc_b) if add_anchors else _to_array_roc(roc_b)

    if A.size == 0 or B.size == 0:
        return 0.0, None, None

    FPR_a = A[:, 0][:, None]
    TPR_a = A[:, 1][:, None]
    FPR_b = B[:, 0][None, :]
    TPR_b = B[:, 1][None, :]

    if comp == "AND":
        Jgrid = _frechet_grid_AND(FPR_a, TPR_a, FPR_b, TPR_b)
    elif comp == "OR":
        Jgrid = _frechet_grid_OR(FPR_a, TPR_a, FPR_b, TPR_b)
    else:
        raise ValueError('comp must be "AND" or "OR".')

    flat_idx = int(np.argmax(Jgrid))
    ia, ib = np.unravel_index(flat_idx, Jgrid.shape)
    jmax = float(Jgrid[ia, ib])
    return float(np.clip(jmax, -1.0, 1.0)), int(ia), int(ib)


# ---- FH/Bernstein utilities at a fixed θ -----------------------------------

def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


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
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Fréchet–Hoeffding *sharp* intervals for:
      I1: p1 = P(A ∧ B = 1 | Y=1)   (AND on positives)
      I0: p0 = P(A ∨ B = 1 | Y=0)   (OR  on negatives)

    Args:
        tpr_a, tpr_b: class-conditional TPRs at θ for rails A,B (Y=1).
        fpr_a, fpr_b: class-conditional FPRs at θ for rails A,B (Y=0).
        alpha_cap: optional policy cap for FPR at deploy-time (binds U0 = min(U0, alpha)).

    Returns:
        (I1, I0) where each is (L, U) with 0 <= L <= U <= 1.

    Raises:
        ValueError if the policy cap makes I0 empty (U0 < L0), signalling infeasible θ under policy.
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
        # Numerical or input inconsistency; clip to a degenerate but valid interval
        U1 = L1
    if U0 < L0:
        # Policy renders the target θ infeasible for OR-composite negatives.
        raise ValueError(
            f"Policy cap makes I0 empty: L0={L0:.6f}, U0={U0:.6f}. "
            "Relax α or adjust θ/marginals."
        )

    return (L1, U1), (L0, U0)


def fh_var_envelope(interval: Tuple[float, float]) -> float:
    """
    Max_{p in [a,b]} p(1-p). Concave with unique maximizer at 0.5.

    Returns:
        v̄ = 0.25 if the interval contains 0.5,
        else max{ a(1-a), b(1-b) }.

    Raises:
        ValueError on invalid interval.
    """
    a, b = interval
    if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0 and a <= b and np.isfinite(a) and np.isfinite(b)):
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
    Bernstein tail bound for a Bernoulli mean with known variance envelope vbar.

    Supports BOTH:
      - New style (keywords): bernstein_tail(t=..., n=..., vbar=..., D=..., two_sided=True)
                              OR bernstein_tail(eps=..., n=..., vbar=..., two_sided=True)
      - Legacy positional:   bernstein_tail(n, eps, vbar)  [two-sided implied; D=1.0]

    Bound:
      P(|p̂ - p| >= ε) ≤ 2 * exp( - n * ε^2 / (2 vbar + (2/3) ε) )

    If you are in CC-space with half-width t on CC, use t and D (ε = t * D).
    """
    # ---- Legacy positional parsing (n, eps, vbar) ----
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

    # ---- Validate / derive eps from t, D if needed ----
    if n is None or vbar is None:
        raise ValueError("Provide n and vbar (either via keywords or legacy positional args).")
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
        return 1.0  # degenerate safe bound

    denom = 2.0 * vbar + (2.0 / 3.0) * eps
    if denom <= 0.0:
        return 1.0

    exponent = - (n * eps * eps) / denom
    base = exp(exponent)
    prob = (2.0 * base) if two_sided else base
    # clip numeric
    if prob < 0.0:
        prob = 0.0
    elif prob > 1.0:
        prob = 1.0
    return float(prob)

def invert_bernstein_eps(n: int, vbar: float, delta: float) -> float:
    """
    Invert  2 * exp( - n * eps^2 / (2 vbar + (2/3) eps) ) <= delta
    to solve for the smallest eps >= 0 satisfying the inequality.

    Uses a safe bracket + binary search (monotone).

    Args:
        n: sample size (positive).
        vbar: variance envelope ∈ [0, 0.25].
        delta: tail budget ∈ (0, 2).

    Returns:
        eps >= 0 such that the two-sided Bernstein tail ≤ delta.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if not (0.0 <= vbar <= 0.25):
        raise ValueError("vbar must be in [0, 0.25].")
    if not (0.0 < delta < 2.0):
        raise ValueError("delta must be in (0, 2).")

    target = log(2.0 / delta)  # > 0
    # Expand upper bracket until f(hi) >= target
    lo, hi = 0.0, 1.0
    for _ in range(64):
        denom = 2.0 * vbar + (2.0 / 3.0) * hi
        val = 0.0 if denom <= 0.0 else n * (hi * hi) / denom
        if val >= target:
            break
        hi *= 2.0
        if hi > 1e6:
            break
    # Binary search
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
) -> float:
    """
    P(|CC_hat - CC| >= t) bound with FH variance envelopes and a union bound.

    Args:
        n1, n0: class-conditional sample sizes for Y=1, Y=0.
        t: CC deviation radius (> 0).
        D: denominator (> 0).
        I1, I0: FH intervals for p1, p0 respectively.

    Returns:
        Two-term Bernstein + union bound probability.

    Notes:
        This does not assume any dependence structure beyond the FH envelopes.
    """
    if D <= 0.0:
        return 1.0  # Fragile denominator; caller should flag elsewhere.
    if t < 0.0:
        raise ValueError("t must be >= 0.")
    v1 = fh_var_envelope(I1)
    v0 = fh_var_envelope(I0)
    return bernstein_tail(t=t, n=n1, vbar=v1, D=D, two_sided=True) \
     + bernstein_tail(t=t, n=n0, vbar=v0, D=D, two_sided=True)



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
) -> Tuple[float, float]:
    """
    Finite-sample two-sided confidence interval for CC at fixed θ.

    We invert Bernstein per class to obtain eps1, eps0 such that
    P(|p1_hat - p1| >= eps1) ≤ δ1 and P(|p0_hat - p0| ≥ eps0) ≤ δ0 with δ1+δ0=δ.
    Then |(p1_hat - p0_hat) - (p1 - p0)| ≤ eps1 + eps0 and
         |CC_hat - CC| ≤ (eps1 + eps0) / D.

    Args:
        n1, n0: class-conditional sample sizes.
        p1_hat, p0_hat: empirical composite rates at θ.
        D: denominator (> 0).
        I1, I0: FH intervals for p1 and p0.
        delta: total two-sided risk (default 0.05).
        split: optional (δ1, δ0). If None, uses symmetric (δ/2, δ/2).

    Returns:
        (lo, hi) CC interval centered at CC_hat = (1 - (p1_hat - p0_hat)) / D.
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
    eps1 = invert_bernstein_eps(n1, v1, d1)
    eps0 = invert_bernstein_eps(n0, v0, d0)
    cc_hat = (1.0 - (p1_hat - p0_hat)) / D
    t = (eps1 + eps0) / D
    return cc_hat - t, cc_hat + t


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
    Closed-form per-class sample sizes (n1*, n0*) to ensure each Bernstein term ≤ δ_y.

    We enforce:
      2 * exp( - n_y * (tD)^2 / ( 2 v̄_y + (2/3) tD ) ) ≤ δ_y
    ⇒  n_y ≥ ( 2 v̄_y + (2/3) tD ) / (tD)^2 * log(2/δ_y)

    Args:
        t: CC deviation target (> 0).
        D: denominator (> 0).
        I1, I0: FH intervals for p1, p0.
        delta: total two-sided risk (default 0.05).
        split: optional (δ1, δ0) with δ1+δ0=δ. If None, uses symmetric split.

    Returns:
        (n1_star, n0_star) as floats (caller may ceil to integers).
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

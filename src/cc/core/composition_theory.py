# src/cc/core/composition_theory.py
"""
Theoretical wrappers for guardrail composition.
Bridges IST 496 to PhD Year 1 (Fréchet–Hoeffding (FH) style ROC/J bounds).

Updated: 2025-09-28

What this module provides
-------------------------
1) Fast FH bounds on the *Youden J* statistic (J = TPR − FPR) for AND/OR composition
   of two guardrails represented as ROC point sets.
2) Robust, reproducible conversion from scored examples to a discrete ROC curve,
   with monotonic cleanup, optional convexification, and sample-weight support.
3) Utilities for ROC validation, convex hull, and J-statistic extraction.

Notes
-----
- FH bounds are computed *separately* on the positive and negative classes
  (TPR and FPR) and then combined to bound J. This is tight given only
  the marginals and no dependence information.
- All functions tolerate degenerate inputs (empty scores, single-class, NaN),
  returning safe defaults.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np

# Prefer project’s cartographer implementation if available.
try:
    # Expected to compute the supremum over all operating-point pairs.
    from cc.cartographer.bounds import frechet_upper as _carto_frechet_upper  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _carto_frechet_upper = None  # type: ignore

ROCPoint = Tuple[float, float]  # (FPR, TPR)
ROC = Sequence[ROCPoint]


# ---------------------------------------------------------------------------
# Public API — FH bounds on J for AND / OR (upper+lower)
# ---------------------------------------------------------------------------


def upper_bound_j_and(roc_a: ROC, roc_b: ROC) -> float:
    """FH-style upper bound on composed J for AND-composition."""
    if _carto_frechet_upper is not None:
        return float(_carto_frechet_upper(roc_a, roc_b, comp="AND"))
    return float(_frechet_bound_j(roc_a, roc_b, comp="AND", side="upper"))


def upper_bound_j_or(roc_a: ROC, roc_b: ROC) -> float:
    """FH-style upper bound on composed J for OR-composition."""
    if _carto_frechet_upper is not None:
        return float(_carto_frechet_upper(roc_a, roc_b, comp="OR"))
    return float(_frechet_bound_j(roc_a, roc_b, comp="OR", side="upper"))


def lower_bound_j_and(roc_a: ROC, roc_b: ROC) -> float:
    """FH-style lower bound on composed J for AND-composition."""
    return float(_frechet_bound_j(roc_a, roc_b, comp="AND", side="lower"))


def lower_bound_j_or(roc_a: ROC, roc_b: ROC) -> float:
    """FH-style lower bound on composed J for OR-composition."""
    return float(_frechet_bound_j(roc_a, roc_b, comp="OR", side="lower"))


# ---------------------------------------------------------------------------
# Scores → ROC (robust, optional convexification)
# ---------------------------------------------------------------------------


def discretize_scores_to_roc(
    scores: Iterable[Tuple[float, int]],
    *,
    sample_weight: Optional[Iterable[float]] = None,
    drop_intermediate: bool = False,
    convexify: bool = False,
) -> List[ROCPoint]:
    """
    Convert (score, label) pairs to a discrete ROC curve (FPR, TPR) by threshold sweep.
    label: 1 for attack/positive, 0 for benign/negative.

    Parameters
    ----------
    scores : iterable of (score, label)
        Scores higher → more likely positive.
    sample_weight : optional iterable of non-negative weights
    drop_intermediate : if True, removes points that are strictly dominated
        (monotonic cleanup keeps only Pareto-optimal points).
    convexify : if True, returns the *upper convex hull* of the ROC (optimal operating frontier).

    Returns
    -------
    List[(FPR, TPR)] with anchors (0,0) and (1,1) included and sorted by FPR.
    """
    scores_list = list(scores)
    n = len(scores_list)

    if n == 0:
        return [(0.0, 0.0), (1.0, 1.0)]

    if sample_weight is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(list(sample_weight), dtype=float)
        if w.size != n:
            raise ValueError("sample_weight length must match scores")
        w = np.clip(w, 0.0, np.inf)

    arr = np.asarray(scores_list, dtype=float)
    s = arr[:, 0].astype(float)
    y = arr[:, 1].astype(int)
    y = np.where(y >= 1, 1, 0)

    # Handle NaN scores robustly: treat as lowest score (never predicted positive)
    s = np.where(np.isnan(s), -np.inf, s)

    # Separate class weights
    w_pos = w * (y == 1)
    w_neg = w * (y == 0)
    P = float(w_pos.sum())
    N = float(w_neg.sum())

    # Degenerate: if no positives or no negatives
    if P <= 0.0 and N <= 0.0:
        return [(0.0, 0.0), (1.0, 1.0)]
    if P <= 0.0:
        # Only negatives: ROC is a vertical line at FPR in [0,1], TPR=0
        return [(0.0, 0.0), (1.0, 0.0)]
    if N <= 0.0:
        # Only positives: ROC is a horizontal line at TPR in [0,1], FPR=0
        return [(0.0, 1.0), (1.0, 1.0)]

    # Sort by score descending; group ties so threshold moves only at unique scores
    order = np.lexsort(
        (-y, s)
    )  # stable secondary key keeps positives before negatives at same score
    s_sorted = s[order][::-1]
    y_sorted = y[order][::-1]
    w_sorted = w[order][::-1]

    # Unique thresholds (at each unique score)
    thr, idx_start = np.unique(s_sorted, return_index=True)
    # Cumulative sums at *every* step
    tp_cum = np.cumsum(w_sorted * (y_sorted == 1))
    fp_cum = np.cumsum(w_sorted * (y_sorted == 0))

    # At each threshold index, examples >= threshold are predicted positive.
    tp_at = tp_cum[idx_start - 1] if idx_start.size else np.array([], dtype=float)
    fp_at = fp_cum[idx_start - 1] if idx_start.size else np.array([], dtype=float)
    # Fix off-by-one for idx_start==0
    tp_at = np.where(idx_start == 0, 0.0, tp_at)
    fp_at = np.where(idx_start == 0, 0.0, fp_at)

    # Build ROC points
    tpr = (P - (tp_at - 0.0)) / P  # this formulation ensures thresholds step correctly
    fpr = (N - (fp_at - 0.0)) / N
    # The above produces a non-increasing sweep because we reversed; sort by FPR
    pts = np.stack([fpr, tpr], axis=1)

    # Append anchors and clean up monotonicity
    pts = np.vstack([[0.0, 0.0], pts, [1.0, 1.0]])
    pts = _ensure_sorted_and_monotone(pts)

    if drop_intermediate:
        pts = _pareto_prune(pts)

    if convexify:
        pts = _roc_upper_convex_hull(pts)

    return [(float(x), float(y_)) for (x, y_) in pts]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def youden_j(roc: ROC) -> Tuple[float, ROCPoint]:
    """Return max J (TPR−FPR) over the ROC and the achieving point (FPR, TPR)."""
    if not roc:
        return 0.0, (0.0, 0.0)
    arr = _validate_and_array(roc)
    j = arr[:, 1] - arr[:, 0]
    k = int(j.argmax())
    return float(j[k]), (float(arr[k, 0]), float(arr[k, 1]))


def ensure_roc_monotone(roc: ROC) -> List[ROCPoint]:
    """Return a version of `roc` that is sorted by FPR and has non-decreasing TPR."""
    if not roc:
        return [(0.0, 0.0), (1.0, 1.0)]
    arr = _validate_and_array(roc)
    arr = _ensure_sorted_and_monotone(arr)
    return [(float(x), float(y)) for (x, y) in arr]


def roc_upper_convex_hull(roc: ROC) -> List[ROCPoint]:
    """Upper convex hull of an ROC (optimal frontier under threshold choice)."""
    if not roc:
        return [(0.0, 0.0), (1.0, 1.0)]
    arr = _validate_and_array(roc)
    arr = _ensure_sorted_and_monotone(arr)
    arr = _roc_upper_convex_hull(arr)
    return [(float(x), float(y)) for (x, y) in arr]


# ---------------------------------------------------------------------------
# Internal: FH bounds engine (vectorized over ROC×ROC)
# ---------------------------------------------------------------------------


def _frechet_bound_j(
    roc_a: ROC,
    roc_b: ROC,
    *,
    comp: Literal["AND", "OR"],
    side: Literal["upper", "lower"],
) -> float:
    """Compute FH bound on J for the composition over all operating-point pairs.

    We treat positives (TPR) and negatives (FPR) independently per FH bounds,
    then combine to bound J = TPR − FPR.

    AND-composition (both rails must trigger):
      Intersection bounds:
        upper: min(p1, p2)
        lower: max(0, p1 + p2 − 1)

    OR-composition (either rail triggers):
      Union bounds:
        upper: min(1, p1 + p2)
        lower: max(p1, p2)

    To maximize J (upper bound), use:
      TPR side = upper bound, FPR side = lower bound
    To minimize J (lower bound), use:
      TPR side = lower bound, FPR side = upper bound
    """
    A = _validate_and_array(roc_a)
    B = _validate_and_array(roc_b)
    if A.size == 0 or B.size == 0:
        return 0.0

    # Arrays: A_x = FPR_a, A_y = TPR_a, etc.
    Ax, Ay = A[:, 0], A[:, 1]
    Bx, By = B[:, 0], B[:, 1]

    # Broadcast (n_a, n_b)
    AyM = Ay[:, None]
    ByM = By[None, :]
    AxM = Ax[:, None]
    BxM = Bx[None, :]

    if comp == "AND":
        # Intersection for positives (TPR)
        tpr_upper = np.minimum(AyM, ByM)
        tpr_lower = np.maximum(0.0, AyM + ByM - 1.0)
        # Intersection for negatives (FPR)
        fpr_upper = np.minimum(AxM, BxM)
        fpr_lower = np.maximum(0.0, AxM + BxM - 1.0)
    elif comp == "OR":
        # Union for positives (TPR)
        tpr_upper = np.minimum(1.0, AyM + ByM)
        tpr_lower = np.maximum(AyM, ByM)
        # Union for negatives (FPR)
        fpr_upper = np.minimum(1.0, AxM + BxM)
        fpr_lower = np.maximum(AxM, BxM)
    else:
        raise ValueError("comp must be 'AND' or 'OR'")

    if side == "upper":
        J = tpr_upper - fpr_lower
    elif side == "lower":
        J = tpr_lower - fpr_upper
    else:
        raise ValueError("side must be 'upper' or 'lower'")

    # Clamp for numerical stability and take supremum/infimum.
    if side == "upper":
        return float(np.nanmax(np.clip(J, -1.0, 1.0)))
    return float(np.nanmin(np.clip(J, -1.0, 1.0)))


# ---------------------------------------------------------------------------
# Internal: ROC helpers
# ---------------------------------------------------------------------------


def _validate_and_array(roc: ROC) -> np.ndarray:
    """Return an (n,2) float array clipped to [0,1], deduped and sorted by FPR."""
    if roc is None:
        return np.zeros((0, 2), dtype=float)
    arr = np.asarray(list(roc), dtype=float).reshape(-1, 2)
    if arr.size == 0:
        return arr
    # Replace NaNs and clip
    arr = np.where(np.isnan(arr), 0.0, arr)
    arr[:, 0] = np.clip(arr[:, 0], 0.0, 1.0)  # FPR
    arr[:, 1] = np.clip(arr[:, 1], 0.0, 1.0)  # TPR
    # Sort by FPR, deduplicate
    idx = np.lexsort((arr[:, 1], arr[:, 0]))
    arr = arr[idx]
    # Ensure anchors present
    anchors = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    arr = _merge_and_unique(np.vstack([anchors, arr]))
    # Monotone cleanup
    arr = _ensure_sorted_and_monotone(arr)
    return arr


def _merge_and_unique(arr: np.ndarray) -> np.ndarray:
    arr = arr[np.lexsort((arr[:, 1], arr[:, 0]))]
    # unique rows
    keep = np.ones(len(arr), dtype=bool)
    keep[1:] = np.any(np.diff(arr, axis=0) != 0.0, axis=1)
    return arr[keep]


def _ensure_sorted_and_monotone(arr: np.ndarray) -> np.ndarray:
    """Sort by FPR and apply cumulative-max on TPR to enforce ROC monotonicity."""
    if arr.size == 0:
        return arr
    arr = arr[np.argsort(arr[:, 0], kind="mergesort")]
    # cumulative max on TPR
    arr[:, 1] = np.maximum.accumulate(arr[:, 1])
    # Ensure final point (1,1)
    if not (math.isclose(arr[-1, 0], 1.0) and math.isclose(arr[-1, 1], 1.0)):
        arr = np.vstack([arr, [1.0, 1.0]])
    # Ensure first point (0,0)
    if not (math.isclose(arr[0, 0], 0.0) and math.isclose(arr[0, 1], 0.0)):
        arr = np.vstack([[0.0, 0.0], arr])
    # Deduplicate again after adjustments
    arr = _merge_and_unique(arr)
    return arr


def _pareto_prune(arr: np.ndarray) -> np.ndarray:
    """Drop strictly dominated points (keep only Pareto frontier)."""
    keep = [0]
    for i in range(1, len(arr)):
        fpr, tpr = arr[i]
        if tpr > arr[keep[-1], 1] or fpr < arr[keep[-1], 0]:
            keep.append(i)
    return arr[keep]


def _roc_upper_convex_hull(arr: np.ndarray) -> np.ndarray:
    """Andrew’s monotone chain on (x=FPR, y=TPR) to get upper hull."""
    pts = _ensure_sorted_and_monotone(arr)
    # Remove duplicates after monotone cleanup
    pts = _merge_and_unique(pts)

    def cross(o, a, b):
        # cross product (a-o) x (b-o)
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # We need *upper* hull: keep points making a right turn (non-positive cross)
    hull: List[Tuple[float, float]] = []
    for p in pts:
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) >= 0.0:
            hull.pop()
        hull.append((float(p[0]), float(p[1])))

    # Ensure anchors present
    if hull[0] != (0.0, 0.0):
        hull.insert(0, (0.0, 0.0))
    if hull[-1] != (1.0, 1.0):
        hull.append((1.0, 1.0))

    return np.asarray(hull, dtype=float)


__all__ = [
    "ROCPoint",
    "ROC",
    # Bounds on J
    "upper_bound_j_and",
    "upper_bound_j_or",
    "lower_bound_j_and",
    "lower_bound_j_or",
    # Scores → ROC
    "discretize_scores_to_roc",
    # ROC utilities
    "youden_j",
    "ensure_roc_monotone",
    "roc_upper_convex_hull",
]

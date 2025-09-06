# src/cc/cartographer/bounds.py
"""
Module: bounds
Purpose: Fréchet–Hoeffding-style ceilings for composed Youden's J over two ROC curves
Author: Pranav Bhave
Date: 2025-08-31 (refined 2025-09-05)

Overview
--------
Given two ROC curves A and B as (FPR, TPR) point sets, this module computes an
*upper bound* on the Youden's J statistic of their AND/OR composition under
arbitrary dependence (no independence assumed). Pointwise bounds come from
Fréchet–Hoeffding inequalities:

  AND (conjunctive gate):
      TPR_and ≤ min(TPR_a, TPR_b)
      FPR_and ≥ max(0, FPR_a + FPR_b − 1)
      ⇒ J_and ≤ min(TPR_a, TPR_b) − max(0, FPR_a + FPR_b − 1)

  OR (disjunctive gate):
      TPR_or ≤ min(1, TPR_a + TPR_b)
      FPR_or ≥ max(FPR_a, FPR_b)
      ⇒ J_or ≤ min(1, TPR_a + TPR_b) − max(FPR_a, FPR_b)

We then take the maximum over all cross-pairs of ROC points.

Design notes
------------
- Robust to ordering / density of ROC samples (no monotonicity required).
- Optional “anchor” augmentation will add (0,0) and (1,1) if missing.
- Type-checked (mypy-clean) with explicit TypeAlias for ROCArrayLike.
- Provides both scalar max bound and argmax diagnostics (point indices).
- Pure NumPy; returns float in [-1, 1] (clipped for numeric safety).
"""

from __future__ import annotations

from typing import Iterable, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

__all__ = [
    "ROCArrayLike",
    "frechet_upper",
    "frechet_upper_with_argmax",
    "ensure_anchors",
]

# ---- Types -----------------------------------------------------------------

ROCPoint: TypeAlias = Tuple[float, float]
ROCArrayLike: TypeAlias = Union[
    Sequence[ROCPoint],
    Iterable[ROCPoint],
    NDArray[np.float64],
]


# ---- Utilities --------------------------------------------------------------

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


# ---- Public API -------------------------------------------------------------

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

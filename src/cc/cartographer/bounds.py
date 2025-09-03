"""
Module: bounds
Purpose: Fréchet-style upper bounds for composed J over two ROC curves
Dependencies: numpy, typing
Author: Pranav Bhave
Date: 2025-08-31
"""
from __future__ import annotations
from typing import Literal, Sequence, Tuple
import numpy as np

ArrayLike = Sequence[Tuple[float, float]]

def _as_roc(arr: ArrayLike) -> np.ndarray:
    roc = np.asarray(arr, dtype=float)
    if roc.ndim != 2 or roc.shape[1] != 2:
        raise ValueError("ROC must be array-like of shape (n, 2) with columns [FPR, TPR].")
    roc = np.clip(roc, 0.0, 1.0)
    if not np.isfinite(roc).all():
        raise ValueError("ROC contains non-finite values.")
    return roc

def frechet_upper(
    roc_a: ArrayLike,
    roc_b: ArrayLike,
    *,
    comp: Literal["AND", "OR"] = "AND",
) -> float:
    """
    Fréchet-style pointwise upper bound on composed J over two ROC curves.

    For any pair of ROC points (FPR_a, TPR_a) and (FPR_b, TPR_b):
      AND: j = min(TPR_a, TPR_b) - max(0, FPR_a + FPR_b - 1)
      OR : j = min(1, TPR_a + TPR_b) - max(FPR_a, FPR_b)

    Returns the maximum j across all cross-pairs. Clamped to [-1, 1].
    """
    A = _as_roc(roc_a)
    B = _as_roc(roc_b)

    FPR_a = A[:, 0][:, None]
    TPR_a = A[:, 1][:, None]
    FPR_b = B[:, 0][None, :]
    TPR_b = B[:, 1][None, :]

    if comp == "AND":
        tpr_and = np.minimum(TPR_a, TPR_b)
        fpr_and = np.maximum(0.0, FPR_a + FPR_b - 1.0)
        j = tpr_and - fpr_and
    elif comp == "OR":
        tpr_or = np.minimum(1.0, TPR_a + TPR_b)
        fpr_or = np.maximum(FPR_a, FPR_b)
        j = tpr_or - fpr_or
    else:
        raise ValueError('comp must be "AND" or "OR"')

    jmax = float(np.max(j))
    return float(np.clip(jmax, -1.0, 1.0))

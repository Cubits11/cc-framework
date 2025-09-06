# src/cc/core/composition_theory.py
"""
Theoretical wrappers for guardrail composition.
Bridges IST 496 to PhD Year 1 (FH-style ROC bounds).
"""

from __future__ import annotations
from typing import Iterable, Sequence, Tuple
import numpy as np
from cc.cartographer.bounds import frechet_upper

ROC = Sequence[Tuple[float, float]]

def upper_bound_j_and(roc_a: ROC, roc_b: ROC) -> float:
    """FH-style upper bound on composed J for AND-composition."""
    return frechet_upper(roc_a, roc_b, comp="AND")

def upper_bound_j_or(roc_a: ROC, roc_b: ROC) -> float:
    """FH-style upper bound on composed J for OR-composition."""
    return frechet_upper(roc_a, roc_b, comp="OR")

def discretize_scores_to_roc(scores: Iterable[Tuple[float, int]]) -> ROC:
    """
    Convert (score, label) pairs to a coarse ROC curve (FPR, TPR) by threshold sweep.
    label: 1 for attack/positive, 0 for benign/negative.
    """
    arr = np.asarray(list(scores), dtype=float)
    if arr.size == 0:
        return [(0.0, 0.0), (1.0, 1.0)]
    s = arr[:, 0]
    y = arr[:, 1].astype(int)

    # Unique thresholds (descending)
    thr = np.unique(s)[::-1]
    P = (y == 1).sum()
    N = (y == 0).sum()
    roc = []
    for t in thr:
        pred = s >= t
        tp = float((pred & (y == 1)).sum())
        fp = float((pred & (y == 0)).sum())
        tpr = tp / P if P > 0 else 0.0
        fpr = fp / N if N > 0 else 0.0
        roc.append((fpr, tpr))
    # ensure 0,0 and 1,1 anchors
    roc = [(0.0, 0.0)] + roc + [(1.0, 1.0)]
    # monotonic cleanup (optional): sort by FPR
    roc.sort(key=lambda z: z[0])
    return roc
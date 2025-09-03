# src/cc/core/metrics.py
from __future__ import annotations
from typing import Sequence, Tuple

def youden_j(tpr: float, fpr: float) -> float:
    return (tpr - fpr)

def delta_add(j_comp: float, j_a: float, j_b: float) -> float:
    return j_comp - max(j_a, j_b)

def cc_max(j_comp: float, j_a: float, j_b: float) -> float:
    denom = max(j_a, j_b)
    return (j_comp / denom) if denom > 0 else float("inf")

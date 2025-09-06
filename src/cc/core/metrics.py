# src/cc/core/metrics.py
"""
Module: metrics
Purpose: Core J-statistic and composability metrics
Author: Pranav Bhave
Date: 2025-08-31
"""

from __future__ import annotations

def youden_j(tpr: float, fpr: float) -> float:
    """Youden’s J = TPR - FPR, clipped to [-1,1]."""
    j = float(tpr) - float(fpr)
    return max(-1.0, min(1.0, j))

def delta_add(j_comp: float, j_a: float, j_b: float) -> float:
    """
    Additive deviation:
    Δ_add = J_comp - (J_A + J_B - J_A * J_B)
    """
    j_add = float(j_a) + float(j_b) - float(j_a) * float(j_b)
    return float(j_comp) - j_add

def cc_max(j_comp: float, j_a: float, j_b: float) -> float:
    """
    CC_max = J_comp / max(J_A, J_B); ∞ if both are 0.
    """
    denom = max(float(j_a), float(j_b))
    return (float(j_comp) / denom) if denom > 0 else float("inf")
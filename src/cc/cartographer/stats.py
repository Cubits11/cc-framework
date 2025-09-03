"""
Module: stats (cartographer shim)
Purpose: Back-compat wrappers and light helpers for Cartographer CLI
Dependencies: numpy, typing
Author: Pranav Bhave
Date: 2025-08-31
"""
from __future__ import annotations

from typing import NamedTuple, Protocol, Tuple, cast, Mapping, Any
import numpy as np


class JResult(NamedTuple):
    j: float
    ci: Tuple[float, float]


class JStatFn(Protocol):
    def __call__(
        self, s0: np.ndarray, s1: np.ndarray, *, n_boot: int, alpha: float
    ) -> JResult: ...


def _empirical_j(s0: np.ndarray, s1: np.ndarray) -> float:
    """Max over thresholds of TPR - FPR, using pooled unique thresholds."""
    s0 = np.asarray(s0, dtype=float)
    s1 = np.asarray(s1, dtype=float)
    if s0.size == 0 or s1.size == 0:
        return 0.0
    thr = np.unique(np.concatenate([s0, s1], axis=0))
    tpr = (s1[:, None] >= thr[None, :]).mean(axis=0)
    fpr = (s0[:, None] >= thr[None, :]).mean(axis=0)
    return float(np.max(tpr - fpr))


def compute_j_ci(
    s0: np.ndarray,
    s1: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, Tuple[float, float]]:
    """
    Back-compat API: Returns (J_hat, (ci_low, ci_high)).
    If a modern `j_statistic` is present, use it; otherwise bootstrap fallback.
    """
    func_obj = globals().get("j_statistic")
    if callable(func_obj):
        func = cast(JStatFn, func_obj)
        res = func(s0, s1, n_boot=n_boot, alpha=alpha)
        return float(res.j), (float(res.ci[0]), float(res.ci[1]))

    # Fallback bootstrap (seeded)
    rng = np.random.default_rng(17)
    j_hat = _empirical_j(s0, s1)
    if n_boot <= 0:
        return j_hat, (j_hat, j_hat)
    n0, n1 = len(s0), len(s1)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx0 = rng.integers(0, n0, size=n0)
        idx1 = rng.integers(0, n1, size=n1)
        boots[b] = _empirical_j(s0[idx0], s1[idx1])
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(j_hat), (lo, hi)


def compose_cc(JA: float, JB: float, Jc: float) -> Tuple[float, float]:
    """
    Minimal composition helpers for smoke:

    CC_max: normalized capacity vs a simple 'perfect stacking' upper bound.
      - Use J_max = max(JA, JB) to keep a conservative, monotone bound.
      - CC_max = clamp(Jc / max(J_max, eps), 0, 1.5)  (allow a bit >1 for detection)

    Δ_add: additivity delta vs an independence baseline (heuristic):
      - J_add = JA + JB - JA * JB
      - Δ_add = Jc - J_add
    """
    eps = 1e-12
    J_max = max(JA, JB)
    cc_max = 0.0 if J_max <= eps else float(Jc / max(J_max, eps))
    # allow some >1 to position in "Red Wedge" if destructive/over-additive appears
    cc_max = float(np.clip(cc_max, 0.0, 1.5))

    J_add = JA + JB - JA * JB
    delta_add = float(Jc - J_add)
    return cc_max, delta_add


def bootstrap_diagnostics(data: Mapping[str, Any], B: int = 10000) -> None:
    """
    Lightweight diagnostic to keep CLI verify-stats happy in smoke.
    Computes a couple of bootstrap J's to ensure pipeline is wired.
    """
    A0, A1 = data.get("A0"), data.get("A1")
    if isinstance(A0, np.ndarray) and isinstance(A1, np.ndarray):
        _ = compute_j_ci(A0, A1, n_boot=min(B, 200), alpha=0.05)
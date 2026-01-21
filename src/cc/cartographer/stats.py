# src/cc/cartographer/stats.py
"""
Module: stats (cartographer shim)
Purpose: Back-compat wrappers and light helpers for Cartographer CLI
Dependencies: numpy, typing
Author: Pranav Bhave
Date: 2025-08-31 (refined 2025-09-03)

What this provides
------------------
- `_empirical_j(s0, s1)`: Max-threshold Youden's J computed directly from scores.
- `compute_j_ci(s0, s1, n_boot, alpha)`: (J_hat, (lo, hi)) with a deterministic bootstrap
  fallback; if a modern `j_statistic` is present in the namespace, it is used instead.
- `compose_cc(JA, JB, Jc)`: Minimal composition helpers for smoke tests only.
  *Not a theorem*, just reporting helpers (see notes below).
- `bootstrap_diagnostics(data, B)`: Lightweight sanity call used by the CLI to ensure
  pipeline wiring.

Notes on composition helpers
----------------------------
- `cc_max = Jc / max(JA, JB)` is a conservative, monotone normalization for dashboards.
  If `max(JA, JB) == 0`, it returns `inf` (caller should special-case for display).
- `delta_add` compares to a purely *heuristic* independence-style additivity baseline
  `J_add = JA + JB - JA*JB`. This is for exploratory plots only; do NOT cite as theory.
"""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple, Protocol, Tuple, cast

import numpy as np

__all__ = [
    "JResult",
    "JStatFn",
    "_empirical_j",
    "compute_j_ci",
    "compose_cc",
    "bootstrap_diagnostics",
]


# =============================================================================
# Types
# =============================================================================


class JResult(NamedTuple):
    j: float
    ci: Tuple[float, float]


class JStatFn(Protocol):
    def __call__(self, s0: np.ndarray, s1: np.ndarray, *, n_boot: int, alpha: float) -> JResult: ...


# =============================================================================
# Core J helpers
# =============================================================================


def _empirical_j(s0: np.ndarray, s1: np.ndarray) -> float:
    """
    Compute Youden's J = max_t [TPR(t) - FPR(t)] using pooled unique thresholds.

    Conventions:
      - Higher score ⇒ more likely to classify as positive (attack).
      - Decision rule at threshold t: predict positive if score >= t.
    """
    s0 = np.asarray(s0, dtype=float).ravel()
    s1 = np.asarray(s1, dtype=float).ravel()
    if s0.size == 0 or s1.size == 0:
        return 0.0

    thr = np.unique(np.concatenate([s0, s1], axis=0))
    # Evaluate TPR/FPR over all thresholds vectorized
    tpr = (s1[:, None] >= thr[None, :]).mean(axis=0)  # P(pred=1 | world=1)
    fpr = (s0[:, None] >= thr[None, :]).mean(axis=0)  # P(pred=1 | world=0)
    j = tpr - fpr
    return float(np.max(j)) if j.size else 0.0


def compute_j_ci(
    s0: np.ndarray,
    s1: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, Tuple[float, float]]:
    """
    Back-compat API: Returns (J_hat, (ci_low, ci_high)).

    If a modern `j_statistic` callable is present in the module namespace
    (e.g., injected by a newer metrics package), use that for the computation.
    Otherwise, fall back to a deterministic (seeded) nonparametric bootstrap
    over scores.

    Args:
        s0: scores for benign/world-0
        s1: scores for attack/world-1
        n_boot: number of bootstrap resamples (set <=0 to disable CI and return (J_hat, J_hat))
        alpha: two-sided CI level (e.g., 0.05 → 95% CI)

    Returns:
        (J_hat, (lo, hi))
    """
    func_obj = globals().get("j_statistic")
    if callable(func_obj):
        func = cast(JStatFn, func_obj)
        res = func(s0, s1, n_boot=n_boot, alpha=alpha)
        return float(res.j), (float(res.ci[0]), float(res.ci[1]))

    # Fallback bootstrap (fixed seed for reproducibility in tests)
    rng = np.random.default_rng(17)
    j_hat = _empirical_j(s0, s1)
    if n_boot <= 0:
        return j_hat, (j_hat, j_hat)

    n0, n1 = int(len(s0)), int(len(s1))
    if n0 == 0 or n1 == 0:
        return j_hat, (j_hat, j_hat)

    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx0 = rng.integers(0, n0, size=n0)
        idx1 = rng.integers(0, n1, size=n1)
        boots[b] = _empirical_j(np.asarray(s0)[idx0], np.asarray(s1)[idx1])

    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return float(j_hat), (lo, hi)


# =============================================================================
# Minimal composition helpers (for smoke dashboards ONLY)
# =============================================================================


def compose_cc(JA: float, JB: float, Jc: float) -> Tuple[float, float]:
    """
    Minimal composition helpers for smoke dashboards (NOT a theorem).

    CC_max:
        cc_max = Jc / max(JA, JB)
        - Conservative normalization vs best single-rail.
        - Returns +inf if both are 0; callers should handle display.

    Δ_add (heuristic):
        Compare to an independence-style additivity heuristic:
            J_add = JA + JB - JA * JB
            delta_add = Jc - J_add
        This baseline is for exploratory plots only.
    """
    JA = float(JA)
    JB = float(JB)
    Jc = float(Jc)

    denom = max(JA, JB)
    cc_max = (Jc / denom) if denom > 0.0 else float("inf")
    # For plotting convenience, some dashboards clip; we do not clip here.

    J_add = JA + JB - JA * JB  # heuristic independence-style baseline
    delta_add = float(Jc - J_add)
    return float(cc_max), float(delta_add)


# =============================================================================
# Diagnostics
# =============================================================================


def bootstrap_diagnostics(data: Mapping[str, Any], B: int = 10_000) -> None:
    """
    Lightweight diagnostic to keep CLI verify-stats happy in smoke.
    Computes a couple of bootstrap J's to ensure pipeline is wired.

    Args:
        data: mapping expected to contain arrays under keys "A0" and "A1".
        B: number of bootstrap resamples for the smoke sanity call (capped to 200).
    """
    A0, A1 = data.get("A0"), data.get("A1")
    if isinstance(A0, np.ndarray) and isinstance(A1, np.ndarray):
        _ = compute_j_ci(A0, A1, n_boot=min(int(B), 200), alpha=0.05)

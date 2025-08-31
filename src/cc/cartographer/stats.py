from typing import Tuple
import numpy as np


def compute_j_ci(s0: np.ndarray, s1: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    J = _empirical_j(s0, s1)
    # TODO: replace with your bootstrap (B>=10k). Stub CI for smoke runs:
    ci_low = max(0.0, J - 0.04)
    ci_high = min(1.0, J + 0.04)
    return float(J), (float(ci_low), float(ci_high))


def _empirical_j(s0: np.ndarray, s1: np.ndarray) -> float:
    t = np.unique(np.concatenate([s0, s1]))
    t = np.concatenate([t, [np.inf]])
    F0 = (s0[:, None] >= t[None, :]).mean(0)
    F1 = (s1[:, None] >= t[None, :]).mean(0)
    return float((F1 - F0).max())


def compose_cc(JA: float, JB: float, Jc: float):
    base = max(JA, JB) if max(JA, JB) > 0 else 1e-9
    CCmax = Jc / base
    Dadd = Jc - base
    return float(CCmax), float(Dadd)


def bootstrap_diagnostics(_data, B: int = 10000):
    # Hook your real bootstrap coverage checks here.
    return True

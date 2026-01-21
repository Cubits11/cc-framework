from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from theory.fh_bounds import robust_inverse_normal


@dataclass
class SimulationResult:
    observed_j: float
    observed_tpr: float
    observed_fpr: float
    tp_counts: list[int]
    fn_counts: list[int]
    fp_counts: list[int]
    tn_counts: list[int]


def _sample_positive_stable(alpha: float, size: int, rng: np.random.Generator) -> np.ndarray:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    v = rng.uniform(0.0, math.pi, size=size)
    w = rng.exponential(scale=1.0, size=size)
    numerator = np.sin(alpha * v)
    denominator = np.sin(v) ** (1.0 / alpha)
    fraction = numerator / denominator
    exponent = (np.sin((1.0 - alpha) * v) / w) ** ((1.0 - alpha) / alpha)
    return fraction * exponent


def _sample_bivariate_copula(
    family: str,
    theta: float,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if family == "independence":
        u = rng.random(n)
        v = rng.random(n)
        return u, v
    if family == "comonotonic":
        u = rng.random(n)
        return u, u.copy()
    if family == "countermonotonic":
        u = rng.random(n)
        return u, 1.0 - u
    if family == "clayton":
        if theta <= 0:
            raise ValueError("Clayton copula requires theta > 0")
        s = rng.gamma(shape=1.0 / theta, scale=1.0, size=n)
        e1 = rng.exponential(scale=1.0, size=n)
        e2 = rng.exponential(scale=1.0, size=n)
        u = (1.0 + e1 / s) ** (-1.0 / theta)
        v = (1.0 + e2 / s) ** (-1.0 / theta)
        return u, v
    if family == "gumbel":
        if theta < 1.0:
            raise ValueError("Gumbel copula requires theta >= 1")
        if theta == 1.0:
            u = rng.random(n)
            v = rng.random(n)
            return u, v
        alpha = 1.0 / theta
        s = _sample_positive_stable(alpha, n, rng)
        e1 = rng.exponential(scale=1.0, size=n)
        e2 = rng.exponential(scale=1.0, size=n)
        u = np.exp(-((e1 / s) ** alpha))
        v = np.exp(-((e2 / s) ** alpha))
        return u, v
    if family == "frank":
        if abs(theta) < 1e-12:
            raise ValueError("Frank copula requires theta != 0")
        u = rng.random(n)
        w = rng.random(n)
        v = np.empty_like(u)
        d = math.expm1(-theta)
        for idx, (u_val, w_val) in enumerate(zip(u, w, strict=False)):
            lower, upper = 0.0, 1.0
            for _ in range(60):
                mid = 0.5 * (lower + upper)
                a = math.exp(-theta * u_val)
                b = math.exp(-theta * mid)
                k = 1.0 + (a - 1.0) * (b - 1.0) / d
                h = a * (b - 1.0) / (d * k)
                if h < w_val:
                    lower = mid
                else:
                    upper = mid
            v[idx] = 0.5 * (lower + upper)
        return u, v
    raise ValueError(f"Unknown copula family: {family}")


def _simulate_latent_factor(
    probs: list[float],
    rho: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if not (-0.99 <= rho <= 0.99):
        raise ValueError("rho must be in [-0.99, 0.99]")
    thresholds = np.array([robust_inverse_normal(p) for p in probs], dtype=float)
    z = rng.standard_normal(size=n)
    eps = rng.standard_normal(size=(n, len(probs)))
    latent = rho * z[:, None] + math.sqrt(1.0 - rho**2) * eps
    return latent < thresholds


def simulate_dependence(
    tprs: list[float],
    fprs: list[float],
    composition_type: str,
    family: str,
    theta: float,
    sample_size: int,
    rng: np.random.Generator,
) -> SimulationResult:
    if composition_type not in {"serial_or", "parallel_and"}:
        raise ValueError("composition_type must be 'serial_or' or 'parallel_and'")
    if len(tprs) != len(fprs):
        raise ValueError("tprs and fprs must be same length")
    k = len(tprs)

    if k == 2:
        u, v = _sample_bivariate_copula(family, theta, sample_size, rng)
        detections = np.column_stack([u < tprs[0], v < tprs[1]])
        u0, v0 = _sample_bivariate_copula(family, theta, sample_size, rng)
        false_alarms = np.column_stack([u0 < fprs[0], v0 < fprs[1]])
    else:
        rho = max(min(theta, 0.99), -0.99)
        detections = _simulate_latent_factor(tprs, rho, sample_size, rng)
        false_alarms = _simulate_latent_factor(fprs, rho, sample_size, rng)

    if composition_type == "serial_or":
        system_detect = detections.any(axis=1)
        system_false_alarm = false_alarms.any(axis=1)
    else:
        system_detect = detections.all(axis=1)
        system_false_alarm = false_alarms.all(axis=1)

    observed_tpr = float(system_detect.mean())
    observed_fpr = float(system_false_alarm.mean())
    observed_j = observed_tpr - observed_fpr

    tp_counts = detections.sum(axis=0).astype(int).tolist()
    fn_counts = (sample_size - detections.sum(axis=0)).astype(int).tolist()
    fp_counts = false_alarms.sum(axis=0).astype(int).tolist()
    tn_counts = (sample_size - false_alarms.sum(axis=0)).astype(int).tolist()

    return SimulationResult(
        observed_j=observed_j,
        observed_tpr=observed_tpr,
        observed_fpr=observed_fpr,
        tp_counts=tp_counts,
        fn_counts=fn_counts,
        fp_counts=fp_counts,
        tn_counts=tn_counts,
    )

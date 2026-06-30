"""Finite-sample helpers for Bernoulli guardrail failure rates.

These utilities are intentionally modest: they give distribution-free
Hoeffding-style radii and sample sizes for estimating binary singleton and
pairwise failure rates. They do not certify dataset representativeness or
deployment safety.
"""

from __future__ import annotations

import math

__all__ = [
    "bernoulli_rate_count",
    "hoeffding_radius",
    "pairwise_rate_count",
    "sample_size_for_radius",
    "simultaneous_bernoulli_radius",
    "simultaneous_sample_size",
]


def hoeffding_radius(n: int, delta: float) -> float:
    """Return a two-sided Hoeffding radius for one Bernoulli rate.

    With probability at least ``1 - delta``, ``|p_hat - p| <= radius``.
    """

    sample_count = _positive_int("n", n)
    delta_value = _probability_open("delta", delta)
    return math.sqrt(math.log(2.0 / delta_value) / (2.0 * sample_count))


def sample_size_for_radius(epsilon: float, delta: float) -> int:
    """Return the smallest Hoeffding sample size for one Bernoulli rate."""

    radius = _positive_float("epsilon", epsilon)
    delta_value = _probability_open("delta", delta)
    return math.ceil(math.log(2.0 / delta_value) / (2.0 * radius * radius))


def simultaneous_bernoulli_radius(n: int, num_rates: int, delta: float) -> float:
    """Return a union-bound Hoeffding radius for several Bernoulli rates."""

    sample_count = _positive_int("n", n)
    rate_count = _positive_int("num_rates", num_rates)
    delta_value = _probability_open("delta", delta)
    return math.sqrt(math.log((2.0 * rate_count) / delta_value) / (2.0 * sample_count))


def simultaneous_sample_size(epsilon: float, num_rates: int, delta: float) -> int:
    """Return the sample size for simultaneous Bernoulli rate estimation."""

    radius = _positive_float("epsilon", epsilon)
    rate_count = _positive_int("num_rates", num_rates)
    delta_value = _probability_open("delta", delta)
    return math.ceil(math.log((2.0 * rate_count) / delta_value) / (2.0 * radius * radius))


def pairwise_rate_count(num_guardrails: int) -> int:
    """Return the number of pairwise overlap rates among ``num_guardrails``."""

    guardrail_count = _nonnegative_int("num_guardrails", num_guardrails)
    return guardrail_count * (guardrail_count - 1) // 2


def bernoulli_rate_count(num_guardrails: int, *, include_pairwise: bool = True) -> int:
    """Return singleton plus optional pairwise Bernoulli rates for a stack."""

    guardrail_count = _positive_int("num_guardrails", num_guardrails)
    if not include_pairwise:
        return guardrail_count
    return guardrail_count + pairwise_rate_count(guardrail_count)


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _positive_float(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite positive number")
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return out


def _probability_open(name: str, value: float) -> float:
    out = _positive_float(name, value)
    if out >= 1.0:
        raise ValueError(f"{name} must lie in (0, 1)")
    return out

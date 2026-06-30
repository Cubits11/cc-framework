"""Finite-sample helpers for Bernoulli guardrail failure rates.

These utilities are intentionally modest: they give distribution-free
Hoeffding-style radii and sample sizes for estimating binary singleton and
pairwise failure rates. They do not certify dataset representativeness or
deployment safety.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from cc.kernel.sensitivity import AssumptionSet, IdentificationResult, LinearQuery

__all__ = [
    "BernoulliRateInterval",
    "FiniteSampleIdentificationResult",
    "PairwiseCountEvidence",
    "SingletonCountEvidence",
    "assumption_set_from_counts",
    "bernoulli_confidence_interval",
    "bernoulli_rate_count",
    "composition_bounds_from_counts",
    "hoeffding_radius",
    "pairwise_rate_count",
    "sample_size_for_radius",
    "simultaneous_bernoulli_radius",
    "simultaneous_sample_size",
]


@dataclass(frozen=True)
class SingletonCountEvidence:
    """Count evidence for one Bernoulli guardrail failure rate."""

    label: str
    failures: int
    n: int
    exact: bool = False


@dataclass(frozen=True)
class PairwiseCountEvidence:
    """Count evidence for one pairwise co-failure rate."""

    left: str
    right: str
    co_failures: int
    n: int
    exact: bool = False


@dataclass(frozen=True)
class BernoulliRateInterval:
    """A simultaneous confidence interval for a Bernoulli rate."""

    name: str
    estimate: float
    lower: float
    upper: float
    radius: float
    failures: int
    n: int
    exact: bool = False


@dataclass(frozen=True)
class FiniteSampleIdentificationResult:
    """Composition bounds produced after converting count evidence to intervals."""

    assumptions: AssumptionSet
    intervals: tuple[BernoulliRateInterval, ...]
    identification: IdentificationResult


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


def bernoulli_confidence_interval(
    *,
    name: str,
    failures: int,
    n: int,
    num_rates: int,
    delta: float,
    exact: bool = False,
) -> BernoulliRateInterval:
    """Return a clipped simultaneous Hoeffding interval for one Bernoulli rate.

    If ``exact`` is true, the empirical rate is treated as a declared exact
    constraint and the radius is zero.  This is useful for deterministic paper
    examples and for preserving already-identified constraints.
    """

    rate_name = _nonempty_string("name", name)
    failure_count = _count_in_trials("failures", failures, n)
    trial_count = _positive_int("n", n)
    rate_count = _positive_int("num_rates", num_rates)
    if not exact:
        _probability_open("delta", delta)
    estimate = failure_count / trial_count
    radius = 0.0 if exact else simultaneous_bernoulli_radius(trial_count, rate_count, delta)
    return BernoulliRateInterval(
        name=rate_name,
        estimate=estimate,
        lower=max(0.0, estimate - radius),
        upper=min(1.0, estimate + radius),
        radius=radius,
        failures=failure_count,
        n=trial_count,
        exact=exact,
    )


def assumption_set_from_counts(
    labels: Sequence[str],
    singleton_counts: Sequence[SingletonCountEvidence],
    *,
    pairwise_counts: Sequence[PairwiseCountEvidence] = (),
    delta: float = 0.05,
    metadata: Mapping[str, str | int | float | bool | None] | None = None,
) -> tuple[AssumptionSet, tuple[BernoulliRateInterval, ...]]:
    """Convert Bernoulli count evidence into interval constraints.

    The simultaneous coverage budget is shared across all supplied singleton
    and pairwise rates.  Pairwise evidence is interpreted as
    ``P(left failure and right failure)`` under the same ``Z_i=1`` convention.
    """

    guardrails = _labels_tuple(labels)
    singleton_by_label = _singleton_evidence_map(singleton_counts, guardrails)
    pairwise_items = tuple(pairwise_counts)
    _validate_pairwise_counts(pairwise_items, guardrails)
    num_rates = len(singleton_by_label) + len(pairwise_items)
    if num_rates == 0:
        raise ValueError("at least one singleton or pairwise count is required")

    assumptions = AssumptionSet.empty(
        guardrails,
        metadata={
            "finite_sample_method": "simultaneous_hoeffding_union_bound",
            "delta": float(_probability_open("delta", delta)),
            **dict(metadata or {}),
        },
    )
    intervals: list[BernoulliRateInterval] = []

    for label in guardrails:
        singleton_evidence = singleton_by_label[label]
        interval = bernoulli_confidence_interval(
            name=f"marginal:{label}",
            failures=singleton_evidence.failures,
            n=singleton_evidence.n,
            num_rates=num_rates,
            delta=delta,
            exact=singleton_evidence.exact,
        )
        assumptions = assumptions.with_marginal_interval(label, interval.lower, interval.upper)
        intervals.append(interval)

    for pairwise_evidence in pairwise_items:
        interval = bernoulli_confidence_interval(
            name=f"pairwise:{pairwise_evidence.left}&{pairwise_evidence.right}",
            failures=pairwise_evidence.co_failures,
            n=pairwise_evidence.n,
            num_rates=num_rates,
            delta=delta,
            exact=pairwise_evidence.exact,
        )
        assumptions = assumptions.with_pairwise_joint_interval(
            pairwise_evidence.left,
            pairwise_evidence.right,
            interval.lower,
            interval.upper,
        )
        intervals.append(interval)

    return assumptions, tuple(intervals)


def composition_bounds_from_counts(
    query: LinearQuery,
    labels: Sequence[str],
    singleton_counts: Sequence[SingletonCountEvidence],
    *,
    pairwise_counts: Sequence[PairwiseCountEvidence] = (),
    delta: float = 0.05,
    metadata: Mapping[str, str | int | float | bool | None] | None = None,
) -> FiniteSampleIdentificationResult:
    """Solve composition bounds after propagating count uncertainty."""

    assumptions, intervals = assumption_set_from_counts(
        labels,
        singleton_counts,
        pairwise_counts=pairwise_counts,
        delta=delta,
        metadata=metadata,
    )
    return FiniteSampleIdentificationResult(
        assumptions=assumptions,
        intervals=intervals,
        identification=assumptions.identify(query),
    )


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


def _count_in_trials(name: str, value: int, n: int) -> int:
    count = _nonnegative_int(name, value)
    trial_count = _positive_int("n", n)
    if count > trial_count:
        raise ValueError(f"{name} must not exceed n")
    return count


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


def _nonempty_string(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _labels_tuple(labels: Sequence[str]) -> tuple[str, ...]:
    if isinstance(labels, str):
        raise TypeError("labels must be a sequence of strings, not a string")
    out = tuple(_nonempty_string("label", label) for label in labels)
    if not out:
        raise ValueError("labels must be nonempty")
    if len(set(out)) != len(out):
        raise ValueError("labels must be unique")
    return out


def _singleton_evidence_map(
    singleton_counts: Sequence[SingletonCountEvidence],
    labels: tuple[str, ...],
) -> dict[str, SingletonCountEvidence]:
    by_label: dict[str, SingletonCountEvidence] = {}
    for evidence in singleton_counts:
        if not isinstance(evidence, SingletonCountEvidence):
            raise TypeError("singleton_counts must contain SingletonCountEvidence values")
        label = _nonempty_string("singleton label", evidence.label)
        if label not in labels:
            raise ValueError(f"singleton count label {label!r} is not in labels")
        if label in by_label:
            raise ValueError(f"duplicate singleton count for {label!r}")
        _count_in_trials("failures", evidence.failures, evidence.n)
        by_label[label] = evidence
    missing = set(labels) - set(by_label)
    if missing:
        raise ValueError(f"missing singleton counts for labels: {sorted(missing)}")
    return by_label


def _validate_pairwise_counts(
    pairwise_counts: tuple[PairwiseCountEvidence, ...],
    labels: tuple[str, ...],
) -> None:
    seen: set[tuple[str, str]] = set()
    for evidence in pairwise_counts:
        if not isinstance(evidence, PairwiseCountEvidence):
            raise TypeError("pairwise_counts must contain PairwiseCountEvidence values")
        left = _nonempty_string("pairwise left", evidence.left)
        right = _nonempty_string("pairwise right", evidence.right)
        if left == right:
            raise ValueError("pairwise count labels must be distinct")
        if left not in labels or right not in labels:
            raise ValueError(f"pairwise count ({left!r}, {right!r}) must use declared labels")
        key = (left, right) if left < right else (right, left)
        if key in seen:
            raise ValueError(f"duplicate pairwise count for {key[0]!r}, {key[1]!r}")
        _count_in_trials("co_failures", evidence.co_failures, evidence.n)
        seen.add(key)

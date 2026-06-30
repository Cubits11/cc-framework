"""Finite-sample helpers for Bernoulli guardrail failure rates.

These utilities are intentionally modest: they give distribution-free
Hoeffding-style radii and sample sizes for fixed, pre-declared binary singleton
and pairwise failure rates sampled iid from one target population. They convert
count-derived moments into interval constraints for the atom LP; declared exact
constraints and policy caps remain modeling assumptions, not empirical
estimates. The helpers do not model label noise, certify dataset
representativeness, or establish deployment safety.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from cc.kernel.sensitivity import AssumptionSet, IdentificationResult, LinearQuery

EvidenceRole = Literal["declared_exact", "empirical_estimate", "modeling_assumption"]
MomentKind = Literal["bernoulli_rate", "singleton_rate", "pairwise_rate", "policy_cap"]
ConstraintSense = Literal["==", "<=", ">="]
TargetSelection = Literal[
    "fixed_pre_sampling",
    "exploratory",
    "split_sample",
    "multiplicity_corrected",
    "post_selection_uncorrected",
]

__all__ = [
    "BernoulliRateInterval",
    "ConstraintSourceMetadata",
    "FiniteSampleIdentificationResult",
    "PairwiseCountEvidence",
    "PolicyCap",
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
class PolicyCap:
    """Declared non-empirical linear cap over the atom distribution.

    Policy caps are carried as modeling assumptions. They are not count-derived
    estimates, receive no confidence radius, and do not consume finite-sample
    alpha.
    """

    name: str
    coefficients: Sequence[float]
    sense: ConstraintSense
    rhs: float
    description: str | None = None

    def __post_init__(self) -> None:
        _nonempty_string("policy cap name", self.name)
        if self.sense not in {"==", "<=", ">="}:
            raise ValueError('policy cap sense must be one of "==", "<=", ">="')
        rhs = float(self.rhs)
        if not math.isfinite(rhs):
            raise ValueError("policy cap rhs must be finite")
        coefficients = tuple(float(value) for value in self.coefficients)
        if not coefficients:
            raise ValueError("policy cap coefficients must be nonempty")
        if not all(math.isfinite(value) for value in coefficients):
            raise ValueError("policy cap coefficients must be finite")
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "rhs", rhs)


@dataclass(frozen=True)
class ConstraintSourceMetadata:
    """Typed source metadata for a finite-sample or policy constraint."""

    name: str
    moment_kind: MomentKind
    role: EvidenceRole
    confidence_method: str | None = None
    simultaneous_correction: str | None = None
    alpha: float | None = None
    alpha_allocation: str | None = None
    allocated_alpha: float | None = None
    failures: int | None = None
    n: int | None = None
    target_selection: str = "fixed_pre_sampling"


@dataclass(frozen=True)
class BernoulliRateInterval:
    """A declared exact or simultaneous confidence interval for a Bernoulli rate."""

    name: str
    estimate: float
    lower: float
    upper: float
    radius: float
    failures: int
    n: int
    exact: bool = False
    moment_kind: MomentKind = "bernoulli_rate"
    role: EvidenceRole = "empirical_estimate"
    confidence_method: str | None = "hoeffding"
    simultaneous_correction: str | None = "union_bound"
    alpha: float | None = None
    alpha_allocation: str | None = "equal_per_reported_rate"
    allocated_alpha: float | None = None
    target_selection: str = "fixed_pre_sampling"


@dataclass(frozen=True)
class FiniteSampleIdentificationResult:
    """Composition bounds produced after converting count evidence to intervals."""

    assumptions: AssumptionSet
    intervals: tuple[BernoulliRateInterval, ...]
    identification: IdentificationResult
    constraint_metadata: tuple[ConstraintSourceMetadata, ...] = ()


def hoeffding_radius(n: int, delta: float) -> float:
    """Return a two-sided Hoeffding radius for one Bernoulli rate.

    Under fixed-before-sampling iid Bernoulli sampling from the target
    population, ``|p_hat - p| <= radius`` with probability at least
    ``1 - delta`` for this one pre-declared moment.
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
    """Return a union-bound Hoeffding radius for fixed Bernoulli rates.

    The claim assumes the reported labels, moments, and target query were fixed
    before interval construction and that each rate is estimated from iid
    Bernoulli samples from the same named target population.
    """

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
    moment_kind: MomentKind = "bernoulli_rate",
    target_selection: TargetSelection = "fixed_pre_sampling",
) -> BernoulliRateInterval:
    """Return a clipped simultaneous Hoeffding interval for one Bernoulli rate.

    If ``exact`` is true, the empirical rate is treated as a declared exact
    constraint and the radius is zero.  This is useful for deterministic paper
    examples and for preserving already-identified constraints. Estimated
    intervals assume fixed labels and target query before sampling, iid
    Bernoulli observations, and no uncorrected adaptive post-selection.
    """

    rate_name = _nonempty_string("name", name)
    failure_count = _count_in_trials("failures", failures, n)
    trial_count = _positive_int("n", n)
    rate_count = _positive_int("num_rates", num_rates)
    moment = _moment_kind(moment_kind)
    target_policy = _validate_target_selection(target_selection)
    if exact:
        delta_value = None
        radius = 0.0
        allocated_alpha = None
    else:
        delta_value = _probability_open("delta", delta)
        radius = simultaneous_bernoulli_radius(trial_count, rate_count, delta)
        allocated_alpha = delta_value / rate_count
    estimate = failure_count / trial_count
    role: EvidenceRole = "declared_exact" if exact else "empirical_estimate"
    return BernoulliRateInterval(
        name=rate_name,
        estimate=estimate,
        lower=max(0.0, estimate - radius),
        upper=min(1.0, estimate + radius),
        radius=radius,
        failures=failure_count,
        n=trial_count,
        exact=exact,
        moment_kind=moment,
        role=role,
        confidence_method=None if exact else "hoeffding",
        simultaneous_correction=None if exact else "union_bound",
        alpha=None if exact else delta_value,
        alpha_allocation=None if exact else "equal_per_reported_rate",
        allocated_alpha=allocated_alpha,
        target_selection=target_policy,
    )


def assumption_set_from_counts(
    labels: Sequence[str],
    singleton_counts: Sequence[SingletonCountEvidence],
    *,
    pairwise_counts: Sequence[PairwiseCountEvidence] = (),
    policy_caps: Sequence[PolicyCap] = (),
    delta: float = 0.05,
    metadata: Mapping[str, str | int | float | bool | None] | None = None,
    target_selection: TargetSelection = "fixed_pre_sampling",
) -> tuple[AssumptionSet, tuple[BernoulliRateInterval, ...]]:
    """Convert Bernoulli count evidence into interval constraints.

    The simultaneous coverage budget is shared across all supplied singleton
    and pairwise rates.  Pairwise evidence is interpreted as
    ``P(left failure and right failure)`` under the same ``Z_i=1`` convention.
    Policy caps, when supplied, are added as non-empirical modeling assumptions.
    """

    guardrails = _labels_tuple(labels)
    singleton_by_label = _singleton_evidence_map(singleton_counts, guardrails)
    pairwise_items = tuple(pairwise_counts)
    _validate_pairwise_counts(pairwise_items, guardrails)
    policy_cap_items = tuple(policy_caps)
    _validate_policy_caps(policy_cap_items)
    target_policy = _validate_target_selection(target_selection)
    num_rates = len(singleton_by_label) + len(pairwise_items)
    if num_rates == 0:
        raise ValueError("at least one singleton or pairwise count is required")
    exact_rate_count = sum(1 for item in singleton_by_label.values() if item.exact) + sum(
        1 for item in pairwise_items if item.exact
    )

    assumptions = AssumptionSet.empty(
        guardrails,
        metadata={
            "finite_sample_method": "simultaneous_hoeffding_union_bound",
            "confidence_method": "hoeffding",
            "simultaneous_correction": "union_bound",
            "delta": float(_probability_open("delta", delta)),
            "alpha": float(delta),
            "alpha_allocation": "equal_per_reported_rate",
            "reported_rate_count": num_rates,
            "estimated_rate_count": num_rates - exact_rate_count,
            "exact_rate_count": exact_rate_count,
            "policy_cap_count": len(policy_cap_items),
            "target_selection": target_policy,
            "sampling_assumptions": (
                "fixed labels and target query; iid Bernoulli samples; "
                "same target population; no uncorrected adaptive post-selection; "
                "label noise not modeled"
            ),
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
            moment_kind="singleton_rate",
            target_selection=target_policy,
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
            moment_kind="pairwise_rate",
            target_selection=target_policy,
        )
        assumptions = assumptions.with_pairwise_joint_interval(
            pairwise_evidence.left,
            pairwise_evidence.right,
            interval.lower,
            interval.upper,
        )
        intervals.append(interval)

    for cap in policy_cap_items:
        assumptions = assumptions.with_linear_constraint(
            cap.name,
            cap.coefficients,
            cap.sense,
            cap.rhs,
            description=cap.description,
        )

    return assumptions, tuple(intervals)


def composition_bounds_from_counts(
    query: LinearQuery,
    labels: Sequence[str],
    singleton_counts: Sequence[SingletonCountEvidence],
    *,
    pairwise_counts: Sequence[PairwiseCountEvidence] = (),
    policy_caps: Sequence[PolicyCap] = (),
    delta: float = 0.05,
    metadata: Mapping[str, str | int | float | bool | None] | None = None,
    target_selection: TargetSelection = "fixed_pre_sampling",
) -> FiniteSampleIdentificationResult:
    """Solve composition bounds after converting counts to interval constraints.

    The returned LP interval is an outer confidence interval for the target
    query only when the count intervals cover their true moments
    simultaneously, exact assumptions are true, and the sampling assumptions in
    the result metadata hold.
    """

    assumptions, intervals = assumption_set_from_counts(
        labels,
        singleton_counts,
        pairwise_counts=pairwise_counts,
        policy_caps=policy_caps,
        delta=delta,
        metadata=metadata,
        target_selection=target_selection,
    )
    return FiniteSampleIdentificationResult(
        assumptions=assumptions,
        intervals=intervals,
        identification=assumptions.identify(query),
        constraint_metadata=(
            *(_metadata_from_interval(interval) for interval in intervals),
            *(_metadata_from_policy_cap(cap) for cap in policy_caps),
        ),
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


def _moment_kind(value: str) -> MomentKind:
    if value not in {"bernoulli_rate", "singleton_rate", "pairwise_rate", "policy_cap"}:
        raise ValueError("moment_kind must name a supported finite-sample moment type")
    return cast(MomentKind, value)


def _validate_target_selection(value: str) -> TargetSelection:
    if value == "post_selection_uncorrected":
        raise ValueError(
            "post-sampling target selection must be marked exploratory or use "
            "split-sample/multiplicity-corrected construction"
        )
    if value not in {
        "fixed_pre_sampling",
        "exploratory",
        "split_sample",
        "multiplicity_corrected",
    }:
        raise ValueError(
            "target_selection must be fixed_pre_sampling, exploratory, split_sample, "
            "multiplicity_corrected, or post_selection_uncorrected"
        )
    return cast(TargetSelection, value)


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


def _validate_policy_caps(policy_caps: tuple[PolicyCap, ...]) -> None:
    seen: set[str] = set()
    for cap in policy_caps:
        if not isinstance(cap, PolicyCap):
            raise TypeError("policy_caps must contain PolicyCap values")
        if cap.name in seen:
            raise ValueError(f"duplicate policy cap name {cap.name!r}")
        seen.add(cap.name)


def _metadata_from_interval(interval: BernoulliRateInterval) -> ConstraintSourceMetadata:
    return ConstraintSourceMetadata(
        name=interval.name,
        moment_kind=interval.moment_kind,
        role=interval.role,
        confidence_method=interval.confidence_method,
        simultaneous_correction=interval.simultaneous_correction,
        alpha=interval.alpha,
        alpha_allocation=interval.alpha_allocation,
        allocated_alpha=interval.allocated_alpha,
        failures=interval.failures,
        n=interval.n,
        target_selection=interval.target_selection,
    )


def _metadata_from_policy_cap(cap: PolicyCap) -> ConstraintSourceMetadata:
    return ConstraintSourceMetadata(
        name=cap.name,
        moment_kind="policy_cap",
        role="modeling_assumption",
        target_selection="not_statistical",
    )

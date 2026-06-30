from __future__ import annotations

import numpy as np
import pytest

from cc.kernel.strict import (
    IdentificationInfeasibleError,
    LinearQuery,
    PairwiseCountEvidence,
    PolicyCap,
    SingletonCountEvidence,
    assumption_set_from_counts,
    bernoulli_confidence_interval,
    composition_bounds_from_counts,
)


def test_bernoulli_intervals_shrink_with_sample_size() -> None:
    small = bernoulli_confidence_interval(
        name="marginal:A",
        failures=50,
        n=100,
        num_rates=2,
        delta=0.05,
    )
    large = bernoulli_confidence_interval(
        name="marginal:A",
        failures=500,
        n=1000,
        num_rates=2,
        delta=0.05,
    )

    assert large.estimate == pytest.approx(small.estimate)
    assert large.radius < small.radius
    assert large.upper - large.lower < small.upper - small.lower


def test_singleton_count_evidence_solves_composition_bounds() -> None:
    labels = ("A", "B")
    query = LinearQuery.intersection(labels, labels, name="A_and_B")

    result = composition_bounds_from_counts(
        query,
        labels,
        [
            SingletonCountEvidence("A", failures=20, n=100),
            SingletonCountEvidence("B", failures=35, n=100),
        ],
        delta=0.05,
    )

    assert result.intervals[0].lower < 0.2 < result.intervals[0].upper
    assert result.intervals[1].lower < 0.35 < result.intervals[1].upper
    assert result.identification.lower_bound <= result.identification.upper_bound
    assert result.assumptions.metadata["finite_sample_method"] == (
        "simultaneous_hoeffding_union_bound"
    )
    assert result.assumptions.metadata["confidence_method"] == "hoeffding"
    assert result.assumptions.metadata["simultaneous_correction"] == "union_bound"
    assert result.assumptions.metadata["alpha_allocation"] == "equal_per_reported_rate"
    assert result.intervals[0].moment_kind == "singleton_rate"
    assert result.intervals[0].role == "empirical_estimate"
    assert result.intervals[0].confidence_method == "hoeffding"
    assert result.constraint_metadata[0].moment_kind == "singleton_rate"


def test_exact_pairwise_constraints_are_preserved() -> None:
    labels = ("A", "B")
    query = LinearQuery.joint(labels, "A", "B", name="A_and_B")

    result = composition_bounds_from_counts(
        query,
        labels,
        [
            SingletonCountEvidence("A", failures=20, n=100, exact=True),
            SingletonCountEvidence("B", failures=35, n=100, exact=True),
        ],
        pairwise_counts=[
            PairwiseCountEvidence("A", "B", co_failures=12, n=100, exact=True),
        ],
    )

    assert [interval.radius for interval in result.intervals] == [0.0, 0.0, 0.0]
    assert [interval.role for interval in result.intervals] == [
        "declared_exact",
        "declared_exact",
        "declared_exact",
    ]
    assert [interval.confidence_method for interval in result.intervals] == [None, None, None]
    assert result.identification.lower_bound == pytest.approx(0.12)
    assert result.identification.upper_bound == pytest.approx(0.12)

    assumptions, intervals = assumption_set_from_counts(
        labels,
        [
            SingletonCountEvidence("A", failures=20, n=100, exact=True),
            SingletonCountEvidence("B", failures=35, n=100, exact=True),
        ],
        pairwise_counts=[
            PairwiseCountEvidence("A", "B", co_failures=12, n=100, exact=True),
        ],
    )
    pairwise_interval = next(interval for interval in intervals if interval.name == "pairwise:A&B")
    assert pairwise_interval.lower == pytest.approx(pairwise_interval.upper)
    assert assumptions.identify(query).lower_bound == pytest.approx(0.12)


def test_estimated_pairwise_constraints_have_pairwise_metadata_and_own_n() -> None:
    labels = ("A", "B")
    assumptions, intervals = assumption_set_from_counts(
        labels,
        [
            SingletonCountEvidence("A", failures=20, n=100),
            SingletonCountEvidence("B", failures=35, n=100),
        ],
        pairwise_counts=[
            PairwiseCountEvidence("A", "B", co_failures=30, n=240),
        ],
    )

    pairwise_interval = next(interval for interval in intervals if interval.name == "pairwise:A&B")
    assert pairwise_interval.moment_kind == "pairwise_rate"
    assert pairwise_interval.role == "empirical_estimate"
    assert pairwise_interval.n == 240
    assert pairwise_interval.alpha_allocation == "equal_per_reported_rate"
    assert assumptions.metadata["reported_rate_count"] == 3


def test_incompatible_exact_constraints_and_estimated_intervals_raise() -> None:
    labels = ("A", "B")
    query = LinearQuery.joint(labels, "A", "B", name="A_and_B")

    with pytest.raises(IdentificationInfeasibleError):
        composition_bounds_from_counts(
            query,
            labels,
            [
                SingletonCountEvidence("A", failures=1, n=10, exact=True),
                SingletonCountEvidence("B", failures=1, n=10, exact=True),
            ],
            pairwise_counts=[
                PairwiseCountEvidence("A", "B", co_failures=100, n=100),
            ],
        )


def test_policy_caps_are_assumptions_not_empirical_estimates() -> None:
    labels = ("A", "B")
    query = LinearQuery.joint(labels, "A", "B", name="A_and_B")
    cap = PolicyCap(
        "policy_cap:A_and_B:upper",
        query.coefficients,
        "<=",
        0.2,
        description="Declared non-empirical policy cap for a worked example.",
    )

    result = composition_bounds_from_counts(
        query,
        labels,
        [
            SingletonCountEvidence("A", failures=50, n=100),
            SingletonCountEvidence("B", failures=50, n=100),
        ],
        policy_caps=[cap],
    )

    policy_metadata = result.constraint_metadata[-1]
    assert policy_metadata.name == "policy_cap:A_and_B:upper"
    assert policy_metadata.moment_kind == "policy_cap"
    assert policy_metadata.role == "modeling_assumption"
    assert policy_metadata.confidence_method is None
    assert result.assumptions.metadata["policy_cap_count"] == 1
    assert result.identification.upper_bound <= 0.2 + 1.0e-9


def test_adaptive_target_selection_must_be_marked_or_corrected() -> None:
    labels = ("A", "B")
    query = LinearQuery.joint(labels, "A", "B", name="A_and_B")
    counts = [
        SingletonCountEvidence("A", failures=20, n=100),
        SingletonCountEvidence("B", failures=35, n=100),
    ]

    with pytest.raises(ValueError, match="post-sampling target selection"):
        composition_bounds_from_counts(
            query,
            labels,
            counts,
            target_selection="post_selection_uncorrected",
        )

    exploratory = composition_bounds_from_counts(
        query,
        labels,
        counts,
        target_selection="exploratory",
    )
    assert exploratory.assumptions.metadata["target_selection"] == "exploratory"
    assert all(interval.target_selection == "exploratory" for interval in exploratory.intervals)


def test_deterministic_outer_interval_coverage_smoke() -> None:
    labels = ("A", "B")
    query = LinearQuery.joint(labels, "A", "B", name="A_and_B")
    true_atom_probabilities = np.asarray([0.55, 0.15, 0.20, 0.10], dtype=float)
    true_query = 0.10
    rng = np.random.default_rng(20260629)
    repetitions = 50
    covered = 0

    for _ in range(repetitions):
        n = 120
        atom_counts = rng.multinomial(n, true_atom_probabilities)
        result = composition_bounds_from_counts(
            query,
            labels,
            [
                SingletonCountEvidence("A", failures=int(atom_counts[1] + atom_counts[3]), n=n),
                SingletonCountEvidence("B", failures=int(atom_counts[2] + atom_counts[3]), n=n),
            ],
            pairwise_counts=[
                PairwiseCountEvidence("A", "B", co_failures=int(atom_counts[3]), n=n),
            ],
            delta=0.10,
        )
        if result.identification.lower_bound <= true_query <= result.identification.upper_bound:
            covered += 1

    assert covered / repetitions >= 0.90

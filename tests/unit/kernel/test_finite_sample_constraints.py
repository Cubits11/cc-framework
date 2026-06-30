from __future__ import annotations

import pytest

from cc.kernel.strict import (
    LinearQuery,
    PairwiseCountEvidence,
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

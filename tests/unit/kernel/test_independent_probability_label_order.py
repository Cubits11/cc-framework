from __future__ import annotations

import pytest

from cc.kernel.metrics import MetricDomainError, independent_event_probability
from cc.kernel.sensitivity import LinearQuery


def test_independent_probability_is_deterministic_under_mapping_order() -> None:
    labels = ("A", "B")
    marginal_a = LinearQuery.marginal(labels, "A")
    union = LinearQuery.union(labels, labels)

    first = {"A": 0.1, "B": 0.2}
    reordered = {"B": 0.2, "A": 0.1}

    assert independent_event_probability(first, marginal_a, labels=labels) == pytest.approx(0.1)
    assert independent_event_probability(reordered, marginal_a, labels=labels) == pytest.approx(0.1)
    assert independent_event_probability(first, union, labels=labels) == pytest.approx(0.28)
    assert independent_event_probability(reordered, union, labels=labels) == pytest.approx(0.28)


def test_independent_probability_validates_labels_and_query_dimension() -> None:
    labels = ("A", "B")
    query = LinearQuery.marginal(labels, "A")

    with pytest.raises(MetricDomainError, match="nonempty"):
        independent_event_probability({"A": 0.1}, query, labels=())
    with pytest.raises(MetricDomainError, match="unique"):
        independent_event_probability({"A": 0.1}, query, labels=("A", "A"))
    with pytest.raises(MetricDomainError, match="match labels"):
        independent_event_probability({"A": 0.1, "C": 0.2}, query, labels=labels)

    one_event_query = LinearQuery.marginal(("A",), "A")
    with pytest.raises(MetricDomainError, match="query dimension"):
        independent_event_probability({"A": 0.1, "B": 0.2}, one_event_query, labels=labels)


def test_independent_probability_rejects_non_event_coefficients() -> None:
    query = LinearQuery.from_coefficients(("A",), "bad", [-1.0, 2.0])
    with pytest.raises(MetricDomainError, match="event weights"):
        independent_event_probability({"A": 0.1}, query, labels=("A",))

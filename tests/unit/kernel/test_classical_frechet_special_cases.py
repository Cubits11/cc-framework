from __future__ import annotations

import numpy as np
import pytest

from cc.kernel.sensitivity import AssumptionSet, LinearQuery, identified_region


def _exact_marginal_assumptions(labels: tuple[str, ...], marginals: list[float]) -> AssumptionSet:
    assumptions = AssumptionSet.empty(labels)
    for label, marginal in zip(labels, marginals, strict=True):
        assumptions = assumptions.with_marginal_interval(label, marginal, marginal)
    return assumptions


def _closed_form(marginals: list[float], event: str) -> tuple[float, float]:
    if event == "and":
        return max(0.0, float(np.sum(marginals)) - (len(marginals) - 1)), min(marginals)
    return max(marginals), min(1.0, float(np.sum(marginals)))


@pytest.mark.parametrize(
    "marginals",
    [
        [0.3, 0.7],
        [0.2, 0.5, 0.8],
        [0.8, 0.9, 0.95],
    ],
)
@pytest.mark.parametrize("event", ["and", "or"])
def test_atom_lp_recovers_classical_frechet_closed_forms(
    marginals: list[float],
    event: str,
) -> None:
    labels = tuple(f"G{i}" for i in range(len(marginals)))
    assumptions = _exact_marginal_assumptions(labels, marginals)
    query = (
        LinearQuery.intersection(labels, labels, name="and_failure")
        if event == "and"
        else LinearQuery.union(labels, labels, name="or_failure")
    )

    result = identified_region(query, assumptions)
    expected_lower, expected_upper = _closed_form(marginals, event)

    assert result.lower_bound == pytest.approx(expected_lower)
    assert result.upper_bound == pytest.approx(expected_upper)

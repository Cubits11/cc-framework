from __future__ import annotations

import numpy as np

from cc.kernel.frechet_classes import distribution_moments
from cc.kernel.sensitivity import AssumptionSet, LinearQuery, identified_region


def _exact_marginal_assumptions(labels: tuple[str, ...], marginals: np.ndarray) -> AssumptionSet:
    assumptions = AssumptionSet.empty(labels)
    for label, marginal in zip(labels, marginals, strict=True):
        value = float(marginal)
        assumptions = assumptions.with_marginal_interval(label, value, value)
    return assumptions


def test_valid_feasible_refinement_can_only_tighten_identified_interval() -> None:
    labels = ("G0", "G1", "G2")
    distribution = np.asarray([0.12, 0.08, 0.10, 0.14, 0.09, 0.16, 0.11, 0.20])
    distribution = distribution / float(np.sum(distribution))
    marginals, pairwise = distribution_moments(distribution, len(labels))

    base = _exact_marginal_assumptions(labels, marginals)
    joint = float(pairwise[0, 1])
    refined = base.with_pairwise_joint_interval("G0", "G1", joint, joint)

    for query in (
        LinearQuery.intersection(labels, labels, name="all_fail"),
        LinearQuery.union(labels, labels, name="any_fail"),
    ):
        base_result = identified_region(query, base)
        refined_result = identified_region(query, refined)

        assert refined_result.lower_bound + 1.0e-8 >= base_result.lower_bound
        assert refined_result.upper_bound <= base_result.upper_bound + 1.0e-8
        assert refined_result.width <= base_result.width + 1.0e-8

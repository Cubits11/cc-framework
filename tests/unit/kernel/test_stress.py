from __future__ import annotations

import numpy as np
import pytest

from cc.kernel.frechet_classes import PairwiseDependence
from cc.kernel.stress import (
    BaselineDependence,
    StressBudget,
    composed_risk_given_guardrail_X_failure,
    stress_test,
)


def _two_guardrail_baseline() -> BaselineDependence:
    return BaselineDependence(
        marginals=[0.2, 0.2],
        pairwise=[PairwiseDependence(0, 1, "joint_probability", 0.08)],
    )


def test_wasserstein_stress_is_budget_constrained_not_fh_by_default() -> None:
    baseline = _two_guardrail_baseline()

    zero = stress_test(baseline, StressBudget(0.0, metric="wasserstein"))
    low = stress_test(baseline, StressBudget(0.02, metric="wasserstein"))
    medium = stress_test(baseline, StressBudget(0.10, metric="wasserstein"))
    unbounded = stress_test(baseline, StressBudget(float("inf"), metric="wasserstein"))

    assert zero.baseline_risk == pytest.approx(0.08)
    assert zero.stressed_risk == pytest.approx(0.08)
    assert low.stressed_risk == pytest.approx(0.10)
    assert low.realized_distance == pytest.approx(0.02)
    assert medium.stressed_risk == pytest.approx(0.18)
    assert medium.frechet_upper == pytest.approx(0.20)
    assert medium.gap_to_frechet_limit == pytest.approx(0.02)
    assert unbounded.stressed_risk == pytest.approx(0.20)
    assert unbounded.method == "wasserstein:fh_limit"


def test_samples_are_accepted_as_empirical_baseline_dependence() -> None:
    samples = np.asarray(
        [[0, 0]] * 68
        + [[1, 0]] * 12
        + [[0, 1]] * 12
        + [[1, 1]] * 8,
        dtype=int,
    )

    result = stress_test({"samples": samples}, {"budget": 0.02, "metric": "wasserstein"})

    assert result.marginals == pytest.approx([0.2, 0.2])
    assert result.baseline_risk == pytest.approx(0.08)
    assert result.stressed_risk == pytest.approx(0.10)


def test_kl_stress_respects_budget_and_moves_toward_fh_limit() -> None:
    baseline_distribution = [0.68, 0.12, 0.12, 0.08]

    result = stress_test(baseline_distribution, StressBudget(0.01, metric="kl"))

    assert result.realized_distance <= 0.010001
    assert result.baseline_risk == pytest.approx(0.08)
    assert 0.08 < result.stressed_risk < result.frechet_upper


def test_composed_risk_given_guardrail_failure_reports_protection_drop() -> None:
    result = composed_risk_given_guardrail_X_failure(_two_guardrail_baseline(), 0)

    assert result.unconditional_risk == pytest.approx(0.08)
    assert result.conditional_risk == pytest.approx(0.4)
    assert result.unconditional_effective_protection == pytest.approx(0.92)
    assert result.conditional_effective_protection == pytest.approx(0.6)
    assert result.protection_drop == pytest.approx(0.32)


def test_pairwise_only_baseline_is_rejected_when_it_does_not_identify_the_copula() -> None:
    baseline = BaselineDependence(
        marginals=[0.2, 0.2, 0.2],
        pairwise=[
            PairwiseDependence(0, 1, "joint_probability", 0.05),
            PairwiseDependence(0, 2, "joint_probability", 0.05),
            PairwiseDependence(1, 2, "joint_probability", 0.05),
        ],
    )

    with pytest.raises(ValueError, match="do not identify a unique dependence law"):
        stress_test(baseline, StressBudget(0.01))

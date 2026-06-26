from __future__ import annotations

import pytest

from cc.kernel.ccf_models import (
    DependenceEvidence,
    alpha_factor,
    alpha_factor_basic_event_probabilities,
    assert_within_fh_envelope,
    beta_factor,
    beta_factor_basic_event_probabilities,
    mgl_basic_event_probabilities,
    multiple_greek_letter,
    partition_failure_probability,
    recommend_model,
)


def test_beta_factor_basic_events_and_partition_estimate() -> None:
    q_values = beta_factor_basic_event_probabilities([0.1, 0.1, 0.1], beta=0.2)

    assert q_values == pytest.approx((0.08, 0.0, 0.02))
    assert beta_factor([0.1, 0.1, 0.1], beta=0.2) == pytest.approx(0.08**3 + 0.02)


def test_mgl_basic_events_match_classical_three_component_formula() -> None:
    q_values = mgl_basic_event_probabilities([0.1, 0.1, 0.1], [0.2, 0.3])

    assert q_values == pytest.approx((0.08, 0.007, 0.006))
    assert multiple_greek_letter([0.1, 0.1, 0.1], [0.2, 0.3]) == pytest.approx(
        0.08**3 + 3.0 * 0.08 * 0.007 + 0.006
    )


def test_alpha_factor_supports_staggered_and_non_staggered_forms() -> None:
    rates = [0.05, 0.05, 0.05]
    alphas = [0.980, 0.013, 0.007]

    staggered = alpha_factor_basic_event_probabilities(
        rates,
        alphas,
        testing_scheme="staggered",
    )
    non_staggered = alpha_factor_basic_event_probabilities(
        rates,
        alphas,
        testing_scheme="non_staggered",
    )

    assert staggered == pytest.approx((0.049, 0.000325, 0.00035))
    alpha_t = 1.0 * 0.980 + 2.0 * 0.013 + 3.0 * 0.007
    assert non_staggered == pytest.approx(
        (
            0.980 * 0.05 / alpha_t,
            2.0 * 0.013 * 0.05 / (2.0 * alpha_t),
            3.0 * 0.007 * 0.05 / alpha_t,
        )
    )


def test_reliasoft_alpha_factor_worked_example_is_reproduced() -> None:
    """Regression for ReliaSoft HotWire Issue 125, Tables 2-3.

    Source example: three identical components, ``Q_t = 0.05``, staggered
    alpha factors ``(0.980, 0.013, 0.007)``, converted to
    ``Q_1 = 0.049``, ``Q_2 = 0.000325``, ``Q_3 = 0.00035``.  NUREG/CR-5485
    Appendix E gives the corresponding three-component top event expression
    ``Q_s = Q_1^3 + 3 Q_1 Q_2 + Q_3``.
    """

    rates = [0.05, 0.05, 0.05]
    alphas = [0.980, 0.013, 0.007]

    q_values = alpha_factor_basic_event_probabilities(
        rates,
        alphas,
        testing_scheme="staggered",
    )
    failure_probability = alpha_factor(rates, alphas, testing_scheme="staggered")

    assert q_values == pytest.approx((0.049, 0.000325, 0.00035))
    assert failure_probability == pytest.approx(0.000515424)
    assert 1.0 - failure_probability == pytest.approx(0.9995, abs=5.0e-5)
    assert_within_fh_envelope(failure_probability, rates)


def test_partition_failure_probability_generalizes_three_component_formula() -> None:
    q1, q2, q3, q4 = 0.01, 0.002, 0.0003, 0.00004

    assert partition_failure_probability((q1, q2, q3)) == pytest.approx(
        q1**3 + 3.0 * q1 * q2 + q3
    )
    assert partition_failure_probability((q1, q2, q3, q4)) == pytest.approx(
        q1**4 + 6.0 * q1**2 * q2 + 3.0 * q2**2 + 4.0 * q1 * q3 + q4
    )


def test_fh_consistency_assertion_flags_invalid_high_probability_approximation() -> None:
    estimate = beta_factor([0.8, 0.8], beta=0.5)

    assert estimate == pytest.approx(0.56)
    with pytest.raises(AssertionError, match="outside the Frechet-Hoeffding envelope"):
        assert_within_fh_envelope(estimate, [0.8, 0.8])


def test_fh_consistency_returns_bounds_for_valid_ccf_estimate() -> None:
    estimate = beta_factor([0.2, 0.2, 0.2], beta=0.1)
    bounds = assert_within_fh_envelope(estimate, [0.2, 0.2, 0.2])

    assert bounds.lower == pytest.approx(0.0)
    assert bounds.upper == pytest.approx(0.2)


@pytest.mark.parametrize(
    "estimate",
    [
        beta_factor([0.05, 0.05, 0.05], beta=0.1),
        alpha_factor([0.05, 0.05, 0.05], [0.980, 0.013, 0.007]),
        multiple_greek_letter([0.05, 0.05, 0.05], [0.2, 0.3]),
    ],
)
def test_all_ccf_model_families_pass_fh_check_in_rare_event_regime(estimate: float) -> None:
    assert_within_fh_envelope(estimate, [0.05, 0.05, 0.05])


def test_classical_models_reject_heterogeneous_guardrail_rates() -> None:
    with pytest.raises(ValueError, match="homogeneous"):
        beta_factor([0.01, 0.02], beta=0.1)
    with pytest.raises(ValueError, match="homogeneous"):
        alpha_factor([0.01, 0.02], [0.9, 0.1])
    with pytest.raises(ValueError, match="homogeneous"):
        multiple_greek_letter([0.01, 0.02], [0.1])


@pytest.mark.parametrize(
    ("evidence", "expected"),
    [
        ("none", "FH-only"),
        ({"pairwise_correlation": True}, "FH+CCF-point-estimate"),
        (
            DependenceEvidence(has_historical_cofailure_counts=True),
            "full empirical estimation",
        ),
    ],
)
def test_recommend_model_maps_evidence_to_strategy(
    evidence: DependenceEvidence | dict[str, bool] | str,
    expected: str,
) -> None:
    recommendation = recommend_model(evidence)

    assert recommendation.recommendation == expected
    assert recommendation.justification
    assert recommendation.required_assumptions

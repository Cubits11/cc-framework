from __future__ import annotations

import math

import numpy as np
import pytest

from cc.kernel.cliff import (
    bootstrap_tail_dependence_ci,
    cliff_certificate,
    estimate_tail_dependence,
    fit_copula_family,
    sample_copula,
    theoretical_tail_dependence,
)


def test_theoretical_tail_dependence_closed_forms() -> None:
    gaussian = theoretical_tail_dependence("gaussian", {"rho": 0.9})
    clayton = theoretical_tail_dependence("clayton", {"theta": 2.0})
    gumbel = theoretical_tail_dependence("gumbel", {"theta": 2.0})
    student_t = theoretical_tail_dependence("student_t", {"rho": 0.5, "df": 4.0})

    assert gaussian.lambda_lower == pytest.approx(0.0)
    assert gaussian.lambda_upper == pytest.approx(0.0)

    assert clayton.lambda_lower == pytest.approx(2.0 ** (-1.0 / 2.0))
    assert clayton.lambda_upper == pytest.approx(0.0)

    assert gumbel.lambda_lower == pytest.approx(0.0)
    assert gumbel.lambda_upper == pytest.approx(2.0 - 2.0 ** (1.0 / 2.0))

    expected_t = 2.0 * pytest.importorskip("scipy.stats").t.cdf(
        -math.sqrt((4.0 + 1.0) * (1.0 - 0.5) / (1.0 + 0.5)),
        4.0 + 1.0,
    )
    assert student_t.lambda_lower == pytest.approx(expected_t)
    assert student_t.lambda_upper == pytest.approx(expected_t)


def test_fit_copula_family_selects_clayton_on_lower_tail_synthetic_data() -> None:
    samples = sample_copula("clayton", {"theta": 3.0}, 1_000, random_state=17)

    fit = fit_copula_family(None, samples, assume_uniform=True, criterion="bic")

    assert fit.family == "clayton"
    assert fit.parameters["theta"] == pytest.approx(3.0, rel=0.25)
    assert fit.tail_dependence.lambda_lower > 0.65
    assert fit.tail_dependence.lambda_upper == pytest.approx(0.0)
    assert [candidate.bic for candidate in fit.ranking] == sorted(
        candidate.bic for candidate in fit.ranking
    )


def test_cliff_certificate_regime_decisions_are_ci_based() -> None:
    est = estimate_tail_dependence(np.random.default_rng(0).random((2_000, 2)), assume_uniform=True)

    sub = cliff_certificate(est, (0.02, 0.10), critical_value=0.20)
    critical = cliff_certificate(est, (0.10, 0.30), critical_value=0.20)
    super_ = cliff_certificate(est, (0.25, 0.40), critical_value=0.20)

    assert sub.regime == "sub-critical"
    assert critical.regime == "critical"
    assert super_.regime == "super-critical"


def test_false_cliff_stress_controls_bootstrap_false_positive_rate() -> None:
    """No-tail independent copula should not be certified critical too often."""

    rng = np.random.default_rng(20260914)
    n_runs = 1_000
    n_samples = 1_024
    alpha = 0.05
    false_claims = 0

    for _ in range(n_runs):
        samples = rng.random((n_samples, 2))
        estimate = estimate_tail_dependence(samples, assume_uniform=True, tail_fraction=0.05)
        ci = bootstrap_tail_dependence_ci(
            samples,
            assume_uniform=True,
            tail_fraction=0.05,
            n_bootstrap=20,
            confidence_level=0.95,
            random_state=rng,
        )
        certificate = cliff_certificate(estimate, ci, critical_value=0.25)
        false_claims += int(certificate.regime != "sub-critical")

    false_positive_rate = false_claims / n_runs
    assert false_positive_rate <= alpha

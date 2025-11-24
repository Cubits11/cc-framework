# tests/unit/models/test_models_ccresult.py

"""
CCResult and bootstrap semantics.

Scope:
- Confidence interval shape and ci_level range
- Bootstrap sample normalization (list / numpy)
- Basic JSON-mode dumping of bootstrap_samples
- Monte Carlo sanity check of bootstrap CI coverage (slow)
"""

import math

import pytest
from pydantic import ValidationError

from cc.core.models import CCResult, NUMPY_AVAILABLE

if NUMPY_AVAILABLE:  # type: ignore[truthy-bool]
    import numpy as np  # type: ignore[import]


# ---------------------------------------------------------------------
# Basic validation
# ---------------------------------------------------------------------


def test_ccresult_ci_and_level_validation():
    """
    - confidence_interval must be (lo <= hi)
    - ci_level must live in (0.5, 1.0); e.g. 0.95 is typical
    """
    # Valid
    CCResult(
        j_empirical=0.1,
        cc_max=0.2,
        delta_add=0.1,
        confidence_interval=(0.0, 0.5),
    )

    # Lo > hi → ValueError
    with pytest.raises(ValueError):
        CCResult(
            j_empirical=0.1,
            cc_max=0.2,
            delta_add=0.1,
            confidence_interval=(0.6, 0.2),
        )

    # Out-of-range ci_level → ValidationError (pydantic field constraints)
    with pytest.raises(ValidationError):
        CCResult(
            j_empirical=0.1,
            cc_max=0.2,
            delta_add=0.1,
            ci_level=0.5,
        )
    with pytest.raises(ValidationError):
        CCResult(
            j_empirical=0.1,
            cc_max=0.2,
            delta_add=0.1,
            ci_level=1.0,
        )


def test_ccresult_bootstrap_normalization_list_and_string_error():
    """
    bootstrap_samples:

    - list input: NaN and ±inf removed
    - non-iterable / wrong-type input rejected
    """
    r = CCResult(
        j_empirical=0.1,
        cc_max=0.1,
        delta_add=0.1,
        bootstrap_samples=[0.1, float("nan"), 0.2, float("inf")],
    )
    assert r.bootstrap_samples == [0.1, 0.2]

    with pytest.raises(ValidationError):
        CCResult(
            j_empirical=0.1,
            cc_max=0.2,
            delta_add=0.1,
            bootstrap_samples="abc",  # type: ignore[arg-type]
        )


@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not available")
def test_ccresult_bootstrap_normalization_numpy_and_json_mode():
    """
    numpy arrays accepted for bootstrap_samples and normalized to clean lists
    in both model state and JSON dump.
    """
    arr = np.array([0.1, 0.2, float("nan")], dtype=float)  # type: ignore[attr-defined]

    r = CCResult(
        j_empirical=0.1,
        cc_max=0.2,
        delta_add=0.1,
        bootstrap_samples=arr,
    )
    assert r.bootstrap_samples == [0.1, 0.2]

    dumped = r.model_dump(mode="json")
    assert isinstance(dumped["bootstrap_samples"], list)
    assert dumped["bootstrap_samples"] == [0.1, 0.2]


# ---------------------------------------------------------------------
# Monte Carlo CI coverage sanity check (slow, stats focused)
# ---------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not available")
def test_ccresult_bootstrap_ci_coverage_monte_carlo():
    """
    Monte Carlo sanity: basic bootstrap percentile CI coverage.

    This does *not* depend on CCResult internals beyond accepting
    bootstrap_samples + confidence_interval; it's a guardrail on our
    mental model of how we should be computing CIs in analysis code.

    The experiment:
    - Draw 200 independent datasets of size 100 from N(0, 1).
    - For each, compute bootstrap means (1000 resamples) and the 95% CI
      using simple percentiles (2.5, 97.5).
    - Count how often the true mean (0.0) lies in that interval.

    We expect coverage in roughly [0.90, 0.98].
    """
    rng = np.random.default_rng(12345)

    coverage_count = 0
    n_trials = 200

    for _ in range(n_trials):
        true_mean = 0.0
        samples = rng.normal(0.0, 1.0, size=100)

        # Bootstrap means
        bootstrap_means = []
        for _ in range(1000):
            resample = rng.choice(samples, size=100, replace=True)
            bootstrap_means.append(float(resample.mean()))

        ci_lo, ci_hi = np.percentile(bootstrap_means, [2.5, 97.5])

        # Sanity: CCResult happily accepts this configuration.
        _ = CCResult(
            j_empirical=float(samples.mean()),
            cc_max=float(samples.mean()),
            delta_add=0.0,
            confidence_interval=(float(ci_lo), float(ci_hi)),
            bootstrap_samples=bootstrap_means,
        )

        if ci_lo <= true_mean <= ci_hi:
            coverage_count += 1

    coverage = coverage_count / n_trials
    assert 0.90 <= coverage <= 0.98

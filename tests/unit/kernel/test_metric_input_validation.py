from __future__ import annotations

import math

import pytest

from cc.kernel.metrics import (
    MetricDomainError,
    cc_gain,
    cc_shift,
    fh_position,
    fh_width,
    independence_regret,
)


@pytest.mark.parametrize("bad", [-0.01, 1.01, math.nan, math.inf, -math.inf])
def test_probability_inputs_must_be_finite_probabilities(bad: float) -> None:
    with pytest.raises(MetricDomainError):
        independence_regret(bad, 0.2)
    with pytest.raises(MetricDomainError):
        cc_gain(0.2, {"A": bad})
    with pytest.raises(MetricDomainError):
        cc_shift(0.1, 0.2, {"A": 0.1}, {"A": bad})


def test_invalid_bounds_raise() -> None:
    with pytest.raises(MetricDomainError, match="lower"):
        fh_width(0.8, 0.2)
    with pytest.raises(MetricDomainError, match="finite"):
        fh_position(math.nan, 0.2, 0.8)
    with pytest.raises(MetricDomainError, match="finite"):
        fh_position(0.5, 0.2, math.inf)


def test_empty_singleton_inputs_raise() -> None:
    with pytest.raises(MetricDomainError, match="nonempty"):
        cc_gain(0.1, [])
    with pytest.raises(MetricDomainError, match="nonempty"):
        cc_gain(0.1, {})
    with pytest.raises(MetricDomainError, match="nonempty"):
        cc_shift(0.1, 0.2, [], [])


def test_bad_eps_and_tol_raise() -> None:
    with pytest.raises(MetricDomainError, match="tol"):
        fh_position(0.5, 0.2, 0.8, tol=0.0)
    with pytest.raises(MetricDomainError, match="tol"):
        fh_width(0.2, 0.8, tol=-1.0)
    with pytest.raises(MetricDomainError, match="eps"):
        cc_gain(0.2, {"A": 0.1}, eps=0.0)
    with pytest.raises(MetricDomainError, match="eps"):
        cc_shift(0.1, 0.2, {"A": 0.1}, {"A": 0.2}, eps=-1.0)
